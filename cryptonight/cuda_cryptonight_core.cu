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

__global__ void cryptonight_core_gpu_phase1(int threads, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx)
{
	__shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init(sharedMemory);

	__syncthreads();

    int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
    int sub = threadIdx.x & 7;
   
    if (thread < threads)
    {
        int i, j;
        int start = 0, end = MEMORY;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        uint32_t *ls32;
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t key[40];
        uint32_t text[4];
        uint32_t *state = (uint32_t *)&ctx->state[16+(sub<<2)];

        MEMCPY8(key, ctx->key1, 20);
        for( i = 0; i < 4; i++ )
            text[i] = state[i];

        for (i = start; i < end; i += INIT_SIZE_BYTE) {

            ls32 = (uint32_t *)&long_state[i];

            cn_aes_pseudo_round_mut(sharedMemory, text, key);

            for( j = 0; j < 4; j++ )
                ls32[(sub<<2) + j] = text[j];
        }
    }
}

__global__ void cryptonight_core_gpu_phase2(int threads, int bfactor, int partidx, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx)
{
	__shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init(sharedMemory);

	__syncthreads();

    int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;

#if __CUDA_ARCH__ >= 300

    int sub = threadIdx.x & 3;

    if (thread < threads)
    {
        int i, j, k;
        int batchsize = ITER >> (2+bfactor);
        int start = partidx * batchsize;
        int end = start + batchsize;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t a, b, c, x[4];
        uint32_t *ls32;
        uint32_t t1[4], t2[4], res;
        uint64_t reshi, reslo;

        a = ctx->a[sub];
        b = ctx->b[sub];

        for (i = start; i < end; ++i) {

            //j = ((uint32_t *)a)[0] & 0x1FFFF0;
            j = __shfl((int)a, 0, 4) & 0x1FFFF0;
            
            //cn_aes_single_round(sharedMemory, &long_state[j], c, a);
            ls32 = (uint32_t *)&long_state[j];
            x[0] = ls32[sub];
            x[1] = __shfl((int)x[0], sub+1, 4);
            x[2] = __shfl((int)x[0], sub+2, 4);
            x[3] = __shfl((int)x[0], sub+3, 4);
            c = a ^
                t_fn0(x[0] & 0xff) ^
                t_fn1((x[1] >> 8) & 0xff) ^
                t_fn2((x[2] >> 16) & 0xff) ^
                t_fn3((x[3] >> 24) & 0xff);
            
            //XOR_BLOCKS_DST(c, b, &long_state[j]);
            ls32[sub] = c ^ b;

            //MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & 0x1FFFF0]);
            ls32 = (uint32_t *)&long_state[__shfl((int)c, 0, 4) & 0x1FFFF0];
            for( k = 0; k < 2; k++ ) t1[k] = __shfl((int)c, k, 4);
            for( k = 0; k < 4; k++ ) t2[k] = __shfl((int)a, k, 4);
            asm(
                "mad.lo.u64 %0, %2, %3, %4;\n\t"
                "mad.hi.u64 %1, %2, %3, %5;\n\t"
                : "=l"(reslo), "=l"(reshi)
                : "l"(((uint64_t *)t1)[0]), "l"(((uint64_t *)ls32)[0]), "l"(((uint64_t *)t2)[1]), "l"(((uint64_t *)t2)[0]));
            res = (sub & 2 ? reslo : reshi) >> (sub&1 ? 32 : 0);
            a = ls32[sub] ^ res;
            ls32[sub] = res;

            //j = ((uint32_t *)a)[0] & 0x1FFFF0;
            j = __shfl((int)a, 0, 4) & 0x1FFFF0;
            
            //cn_aes_single_round(sharedMemory, &long_state[j], b, a);
            ls32 = (uint32_t *)&long_state[j];
            x[0] = ls32[sub];
            x[1] = __shfl((int)x[0], sub+1, 4);
            x[2] = __shfl((int)x[0], sub+2, 4);
            x[3] = __shfl((int)x[0], sub+3, 4);
            b = a ^
                t_fn0(x[0] & 0xff) ^
                t_fn1((x[1] >> 8) & 0xff) ^
                t_fn2((x[2] >> 16) & 0xff) ^
                t_fn3((x[3] >> 24) & 0xff);

            //XOR_BLOCKS_DST(b, c, &long_state[j]);
            ls32[sub] = c ^ b;

            //MUL_SUM_XOR_DST(b, a, &long_state[((uint32_t *)b)[0] & 0x1FFFF0]);
            ls32 = (uint32_t *)&long_state[__shfl((int)b, 0, 4) & 0x1FFFF0];
            for( k = 0; k < 2; k++ ) t1[k] = __shfl((int)b, k, 4);
            for( k = 0; k < 4; k++ ) t2[k] = __shfl((int)a, k, 4);
            asm(
                "mad.lo.u64 %0, %2, %3, %4;\n\t"
                "mad.hi.u64 %1, %2, %3, %5;\n\t"
                : "=l"(reslo), "=l"(reshi)
                : "l"(((uint64_t *)t1)[0]), "l"(((uint64_t *)ls32)[0]), "l"(((uint64_t *)t2)[1]), "l"(((uint64_t *)t2)[0]));
            res = (sub & 2 ? reslo : reshi) >> (sub&1 ? 32 : 0);
            a = ls32[sub] ^ res;
            ls32[sub] = res;
        }

        if( bfactor > 0 ) {

            ctx->a[sub] = a;
            ctx->b[sub] = b;
        }
    }

#else // __CUDA_ARCH__ < 300

    if (thread < threads)
    {
        int i, j;
        int batchsize = ITER >> (2+bfactor);
        int start = partidx * batchsize;
        int end = start + batchsize;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t a[4], b[4], c[4];

        MEMCPY8(a, ctx->a, 2);
        MEMCPY8(b, ctx->b, 2);

        for (i = start; i < end; ++i) {

            j = ((uint32_t *)a)[0] & 0x1FFFF0;
            cn_aes_single_round(sharedMemory, &long_state[j], c, a);
            XOR_BLOCKS_DST(c, b, &long_state[j]);
            MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & 0x1FFFF0]);
            j = ((uint32_t *)a)[0] & 0x1FFFF0;
            cn_aes_single_round(sharedMemory, &long_state[j], b, a);
            XOR_BLOCKS_DST(b, c, &long_state[j]);
            MUL_SUM_XOR_DST(b, a, &long_state[((uint32_t *)b)[0] & 0x1FFFF0]);
        }
        
        if( bfactor > 0 ) {

            MEMCPY8(ctx->a, a, 2);
            MEMCPY8(ctx->b, b, 2);
        }
    }

#endif // __CUDA_ARCH__ >= 300
}

__global__ void cryptonight_core_gpu_phase3(int threads, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx)
{
	__shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init(sharedMemory);

	__syncthreads();

    int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
    int sub = threadIdx.x & 7;
   
    if (thread < threads)
    {
        int i, j;
        int start = 0, end = MEMORY;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        uint32_t *ls32;
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t key[40];
        uint32_t text[4];
        uint32_t *state = (uint32_t *)&ctx->state[16+(sub<<2)];

        MEMCPY8(key, ctx->key2, 20);
        for( i = 0; i < 4; i++ )
            text[i] = state[i];

        for (i = start; i < end; i += INIT_SIZE_BYTE) {

            ls32 = (uint32_t *)&long_state[i];

            for( j = 0; j < 4; j++ )
                text[j] ^= ls32[(sub<<2)+j];

            cn_aes_pseudo_round_mut(sharedMemory, text, key);
        }

        for( i = 0; i < 4; i++ )
            state[i] = text[i];
    }
}


__host__ void cryptonight_core_cpu_init(int thr_id, int threads)
{
	cn_aes_cpu_init();
}

__host__ void cryptonight_core_cpu_hash(int thr_id, int blocks, int threads, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx)
{
    dim3 grid(blocks);
    dim3 block(threads);
    dim3 block4(threads << 2);
    dim3 block8(threads << 3);

    size_t shared_size = 1024;
    int i, partcount = 1 << device_bfactor[thr_id];

    cryptonight_core_gpu_phase1<<<grid, block8, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();
    if( partcount > 1 ) usleep(device_bsleep[thr_id]);

    for( i = 0; i < partcount; i++ ) {
        cryptonight_core_gpu_phase2<<<grid, (device_arch[thr_id][0] >= 3 ? block4 : block), shared_size>>>(blocks*threads, device_bfactor[thr_id], i, d_long_state, d_ctx);
        cudaDeviceSynchronize();
        if( partcount > 1 ) usleep(device_bsleep[thr_id]);
    }

    cryptonight_core_gpu_phase3<<<grid, block8, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();
}
