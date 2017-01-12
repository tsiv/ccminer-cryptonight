#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cryptonight.h"

#ifndef _WIN32
#include <unistd.h>
#endif

extern int device_arch[8][2];
extern int device_bfactor[8];
extern int device_bsleep[8];

#include "cuda_cryptonight_aes.cu"

__device__ __forceinline__ uint64_t cuda_mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi)
{
	*product_hi = __umul64hi(multiplier, multiplicand);
	return(multiplier * multiplicand);
}

__global__ void cryptonight_core_gpu_phase1(int threads, uint32_t * __restrict__ long_state, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	const int sub = (threadIdx.x & 7) << 2;

	if(thread < threads)
	{
		uint32_t key[40], text[4];

		MEMCPY8(key, ctx_key1 + thread * 40, 20);
		MEMCPY8(text, ctx_state + thread * 50 + sub + 16, 2);

		__syncthreads();
		for(int i = 0; i < 0x80000; i += 32)
		{
			cn_aes_pseudo_round_mut(sharedMemory, text, key);
			MEMCPY8(&long_state[(thread << 19) + sub + i], text, 2);
		}
	}
}

__device__ __forceinline__ void MUL_SUM_XOR_DST(const uint64_t *__restrict__ a, uint64_t *__restrict__ c, uint64_t *__restrict__ dst)
{
	uint64_t hi, lo = cuda_mul128(a[0], dst[0], &hi) + c[1];
	hi += c[0];
	c[0] = dst[0] ^ hi;
	c[1] = dst[1] ^ lo;
	dst[0] = hi;
	dst[1] = lo;
}

__global__ void cryptonight_core_gpu_phase2(uint32_t threads, int bfactor, int partidx, uint32_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);

	__syncthreads();

#if __CUDA_ARCH__ >= 300 && __CUDA_ARCH__ < 600

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	const int sub = threadIdx.x & 3;

	if(thread < threads)
	{
		int i, j, k;
		const int batchsize = ITER >> (2 + bfactor);
		const int start = partidx * batchsize;
		const int end = start + batchsize;
		uint32_t * __restrict__ long_state = &d_long_state[thread << 19];
		uint32_t * __restrict__ ctx_a = d_ctx_a + thread * 4;
		uint32_t * __restrict__ ctx_b = d_ctx_b + thread * 4;
		uint32_t a, b, c, x[4];
		uint32_t t1[4], t2[4], res;
		uint64_t reshi, reslo;

		a = ctx_a[sub];
		b = ctx_b[sub];

#pragma unroll 8
		for(i = start; i < end; ++i)
		{

			//j = ((uint32_t *)a)[0] & 0x1FFFF0;
			j = (__shfl((int)a, 0, 4) & 0x1FFFF0) >> 2;

			//cn_aes_single_round(sharedMemory, &long_state[j], c, a);
			x[0] = long_state[j + sub];
			x[1] = __shfl((int)x[0], sub + 1, 4);
			x[2] = __shfl((int)x[0], sub + 2, 4);
			x[3] = __shfl((int)x[0], sub + 3, 4);
			c = a ^
				t_fn0(x[0] & 0xff) ^
				t_fn1((x[1] >> 8) & 0xff) ^
				t_fn2((x[2] >> 16) & 0xff) ^
				t_fn3((x[3] >> 24) & 0xff);

			//XOR_BLOCKS_DST(c, b, &long_state[j]);
			long_state[j + sub] = c ^ b;

			//MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & 0x1FFFF0]);
			j = (__shfl((int)c, 0, 4) & 0x1FFFF0) >> 2;
#pragma unroll
			for(k = 0; k < 2; k++)
				t1[k] = __shfl((int)c, k, 4);
#pragma unroll
			for(k = 0; k < 4; k++)
				t2[k] = __shfl((int)a, k, 4);
			asm(
				"mad.lo.u64 %0, %2, %3, %4;\n\t"
				"mad.hi.u64 %1, %2, %3, %5;\n\t"
				: "=l"(reslo), "=l"(reshi)
				: "l"(((uint64_t *)t1)[0]), "l"(((uint64_t *)long_state)[j >> 1]), "l"(((uint64_t *)t2)[1]), "l"(((uint64_t *)t2)[0]));
			res = (sub & 2 ? reslo : reshi) >> (sub & 1 ? 32 : 0);
			a = long_state[j + sub] ^ res;
			long_state[j + sub] = res;

			//j = ((uint32_t *)a)[0] & 0x1FFFF0;
			j = (__shfl((int)a, 0, 4) & 0x1FFFF0) >> 2;

			//cn_aes_single_round(sharedMemory, &long_state[j], b, a);
			x[0] = long_state[j + sub];
			x[1] = __shfl((int)x[0], sub + 1, 4);
			x[2] = __shfl((int)x[0], sub + 2, 4);
			x[3] = __shfl((int)x[0], sub + 3, 4);
			b = a ^
				t_fn0(x[0] & 0xff) ^
				t_fn1((x[1] >> 8) & 0xff) ^
				t_fn2((x[2] >> 16) & 0xff) ^
				t_fn3((x[3] >> 24) & 0xff);

			//XOR_BLOCKS_DST(b, c, &long_state[j]);
			long_state[j + sub] = c ^ b;

			//MUL_SUM_XOR_DST(b, a, &long_state[((uint32_t *)b)[0] & 0x1FFFF0]);
			j = (__shfl((int)b, 0, 4) & 0x1FFFF0) >> 2;
#pragma unroll
			for(k = 0; k < 2; k++)
				t1[k] = __shfl((int)b, k, 4);
#pragma unroll
			for(k = 0; k < 4; k++)
				t2[k] = __shfl((int)a, k, 4);
			asm(
				"mad.lo.u64 %0, %2, %3, %4;\n\t"
				"mad.hi.u64 %1, %2, %3, %5;\n\t"
				: "=l"(reslo), "=l"(reshi)
				: "l"(((uint64_t *)t1)[0]), "l"(((uint64_t *)long_state)[j >> 1]), "l"(((uint64_t *)t2)[1]), "l"(((uint64_t *)t2)[0]));
			res = (sub & 2 ? reslo : reshi) >> (sub & 1 ? 32 : 0);
			a = long_state[j + sub] ^ res;
			long_state[j + sub] = res;
		}

		if(bfactor > 0)
		{
			ctx_a[sub] = a;
			ctx_b[sub] = b;
		}
	}

#else // __CUDA_ARCH__ < 300
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if(thread < threads)
	{
		int j;
		const int batchsize = ITER >> (2 + bfactor);
		const int start = partidx * batchsize;
		const int end = start + batchsize;
		uint32_t * __restrict__ long_state = &d_long_state[thread << 19];
		uint64_t * __restrict__ ctx_a = (uint64_t*)(d_ctx_a + thread * 4);
		uint64_t * __restrict__ ctx_b = (uint64_t*)(d_ctx_b + thread * 4);
		uint64_t a[2], b[2], c[2];
		uint32_t *a32 = (uint32_t*)a;
		uint32_t *b32 = (uint32_t*)b;
		uint32_t *c32 = (uint32_t*)c;

		a[0] = ctx_a[0];
		a[1] = ctx_a[1];
		b[0] = ctx_b[0];
		b[1] = ctx_b[1];

		for(int i = start; i < end; ++i)
		{
			j = (a32[0] & 0x001FFFF0) >> 2;
			cn_aes_single_round(sharedMemory, &long_state[j], c32, a32);
			XOR_BLOCKS_DST2(c, b, &long_state[j]);
			MUL_SUM_XOR_DST(c, a, (uint64_t*)&long_state[(c[0] & 0x001FFFF0) >> 2]);
			j = (((uint32_t*)a)[0] & 0x1FFFF0) >> 2;
			cn_aes_single_round(sharedMemory, &long_state[j], b32, a32);
			XOR_BLOCKS_DST2(b, c, &long_state[j]);
			MUL_SUM_XOR_DST(b, a, (uint64_t*)&long_state[(b[0] & 0x1FFFF0) >> 2]);
		}

		if(bfactor > 0)
		{

			ctx_a[0] = a[0];
			ctx_a[1] = a[1];
			ctx_b[0] = b[0];
			ctx_b[1] = b[1];
		}
	}
#endif // __CUDA_ARCH__ >= 300
}

__global__ void cryptonight_core_gpu_phase3(int threads, const uint32_t * __restrict__ long_state, uint32_t * __restrict__ d_ctx_state, uint32_t * __restrict__ d_ctx_key2)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);

	int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	int sub = (threadIdx.x & 7) << 2;

	if(thread < threads)
	{
		uint32_t key[40], text[4], i, j;
		MEMCPY8(key, d_ctx_key2 + thread * 40, 20);
		MEMCPY8(text, d_ctx_state + thread * 50 + sub + 16, 2);

		__syncthreads();
		for(i = 0; i < 0x80000; i += 32)
		{
#pragma unroll
			for(j = 0; j < 4; ++j)
				text[j] ^= long_state[(thread << 19) + sub + i + j];

			cn_aes_pseudo_round_mut(sharedMemory, text, key);
		}

		MEMCPY8(d_ctx_state + thread * 50 + sub + 16, text, 2);
	}
}

__host__ void cryptonight_core_cpu_hash(int thr_id, int blocks, int threads, uint32_t *d_long_state, uint32_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2)
{
	dim3 grid(blocks);
	dim3 block(threads);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	int i, partcount = 1 << device_bfactor[thr_id];

	cryptonight_core_gpu_phase1 << <grid, block8 >> >(blocks*threads, d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FILE__, __LINE__);
	if(partcount > 1) usleep(device_bsleep[thr_id]);

	for(i = 0; i < partcount; i++)
	{
		cryptonight_core_gpu_phase2 << <grid, ((device_arch[thr_id][0] == 3 || device_arch[thr_id][0] == 5) ? block4 : block)>> >(blocks*threads, device_bfactor[thr_id], i, d_long_state, d_ctx_a, d_ctx_b);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		if(partcount > 1) usleep(device_bsleep[thr_id]);
	}

	cryptonight_core_gpu_phase3 << <grid, block8 >> >(blocks*threads, d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FILE__, __LINE__);
}
