#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cryptonight.h"

#ifndef _WIN32
#include <unistd.h>
#endif

#include "cuda_cryptonight_aes.cu"

#define hi_dword(x) (x >> 32)
#define lo_dword(x) (x & 0xFFFFFFFF)

__device__ __forceinline__ uint64_t cuda_mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi)
{
  uint64_t a = hi_dword(multiplier);
  uint64_t b = lo_dword(multiplier);
  uint64_t c = hi_dword(multiplicand);
  uint64_t d = lo_dword(multiplicand);

  uint64_t ac = a * c;
  uint64_t ad = a * d;
  uint64_t bc = b * c;
  uint64_t bd = b * d;

  uint64_t adbc = ad + bc;
  uint64_t adbc_carry = adbc < ad ? 1 : 0;

  uint64_t product_lo = bd + (adbc << 32);
  uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
  *product_hi = ac + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;

  return product_lo;
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
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        uint32_t *ls32;
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t key[40];
        uint32_t text[4];
        uint32_t *state = (uint32_t *)&ctx->state[16+(sub<<2)];

        MEMCPY8(key, ctx->key1, 20);
        for( i = 0; i < 4; i++ )
            text[i] = state[i];

        for (i = 0; i < MEMORY; i += INIT_SIZE_BYTE) {

            ls32 = (uint32_t *)&long_state[i];

            cn_aes_pseudo_round_mut(sharedMemory, text, key);

            for( j = 0; j < 4; j++ )
                ls32[(sub<<2) + j] = text[j];
        }
    }
}

__global__ void cryptonight_core_gpu_phase2(int threads, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx)
{
	__shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init(sharedMemory);

	__syncthreads();

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
   
    if (thread < threads)
    {
        int i, j;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t a[4], b[4], c[4];

        MEMCPY8(a, ctx->a, 2);
        MEMCPY8(b, ctx->b, 2);

        for (i = 0; i < ITER / 4; ++i) {

            j = E2I(a) * AES_BLOCK_SIZE;
            cn_aes_single_round(sharedMemory, &long_state[j], c, a);
            XOR_BLOCKS_DST(c, b, &long_state[j]);
            MUL_SUM_XOR_DST(c, a, &long_state[E2I(c) * AES_BLOCK_SIZE]);
            j = E2I(a) * AES_BLOCK_SIZE;
            cn_aes_single_round(sharedMemory, &long_state[j], b, a);
            XOR_BLOCKS_DST(b, c, &long_state[j]);
            MUL_SUM_XOR_DST(b, a, &long_state[E2I(b) * AES_BLOCK_SIZE]);
        }
    }
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
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        uint32_t *ls32;
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t key[40];
        uint32_t text[4];
        uint32_t *state = (uint32_t *)&ctx->state[16+(sub<<2)];

        MEMCPY8(key, ctx->key2, 20);
        for( i = 0; i < 4; i++ )
            text[i] = state[i];

        for (i = 0; i < MEMORY; i += INIT_SIZE_BYTE) {

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
    dim3 block8(threads << 3);

    size_t shared_size = 1024;

    cryptonight_core_gpu_phase1<<<grid, block8, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();

    cryptonight_core_gpu_phase2<<<grid, block, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();

    cryptonight_core_gpu_phase3<<<grid, block8, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();
}
