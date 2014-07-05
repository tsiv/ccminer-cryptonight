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

__global__ void cryptonight_core_gpu_phase2(int threads, int partcount, int partidx, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx)
{
	__shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init(sharedMemory);

	__syncthreads();

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
   
    if (thread < threads)
    {
        int i, j;
        int start = 0, end = ITER / 4;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        struct cryptonight_gpu_ctx *ctx = &d_ctx[thread];
        uint32_t a[4], b[4], c[4];

        MEMCPY8(a, ctx->a, 2);
        MEMCPY8(b, ctx->b, 2);

        if( partcount > 1 ) {
            
            int batchsize = (ITER / 4) / partcount;
            start = partidx * batchsize;
            end = start + batchsize;
        }

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
        
        if( partcount > 1 ) {

            MEMCPY8(ctx->a, a, 2);
            MEMCPY8(ctx->b, b, 2);
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
    dim3 block8(threads << 3);

    size_t shared_size = 1024;
    int i, partcount = 1 << device_bfactor[thr_id];

    cryptonight_core_gpu_phase1<<<grid, block8, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();
    if( partcount > 1 ) usleep(device_bsleep[thr_id]);

    for( i = 0; i < partcount; i++ ) {
        cryptonight_core_gpu_phase2<<<grid, block, shared_size>>>(blocks*threads, partcount, i, d_long_state, d_ctx);
        cudaDeviceSynchronize();
        if( partcount > 1 ) usleep(device_bsleep[thr_id]);
    }

    cryptonight_core_gpu_phase3<<<grid, block8, shared_size>>>(blocks*threads, d_long_state, d_ctx);
    cudaDeviceSynchronize();
}
