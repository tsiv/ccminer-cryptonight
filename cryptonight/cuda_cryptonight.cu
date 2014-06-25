#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cryptonight.h"

#ifndef _WIN32
#include <unistd.h>
#endif

typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

__constant__ uint32_t pTarget[8];
__constant__ uint32_t d_input[19];
extern uint32_t *d_resultNonce[8];

#define C32(x)    ((uint32_t)(x ## U))
#define T32(x) ((x) & C32(0xFFFFFFFF))

#if __CUDA_ARCH__ >= 350
__forceinline__ __device__ uint64_t ROTL64(const uint64_t value, const int offset) {
    uint2 result;
    if(offset >= 32) {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
    } else {
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
        asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
    }
    return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#else
#define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
#endif

#if __CUDA_ARCH__ < 350 
    #define ROTL32(x, n) T32(((x) << (n)) | ((x) >> (32 - (n))))
#else
    #define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#endif

#define ROTWL(x) { \
    uint8_t rotwl_tmp; \
    rotwl_tmp = x[0]; \
    x[0] = x[1]; \
    x[1] = x[2]; \
    x[2] = x[3]; \
    x[3] = rotwl_tmp; }

#define MEMSET8(dst,what,cnt) { \
    int i_memset8; \
    uint64_t *out_memset8 = (uint64_t *)(dst); \
    for( i_memset8 = 0; i_memset8 < cnt; i_memset8++ ) \
        out_memset8[i_memset8] = (what); }

#define MEMCPY8(dst,src,cnt) { \
    int i_memcpy8; \
    uint64_t *in_memcpy8 = (uint64_t *)(src); \
    uint64_t *out_memcpy8 = (uint64_t *)(dst); \
    for( i_memcpy8 = 0; i_memcpy8 < cnt; i_memcpy8++ ) \
        out_memcpy8[i_memcpy8] = in_memcpy8[i_memcpy8]; }

#define MEMCPY4(dst,src,cnt) { \
    int i_memcpy4; \
    uint32_t *in_memcpy4 = (uint32_t *)(src); \
    uint32_t *out_memcpy4 = (uint32_t *)(dst); \
    for( i_memcpy4 = 0; i_memcpy4 < cnt; i_memcpy4++ ) \
        out_memcpy4[i_memcpy4] = in_memcpy4[i_memcpy4]; }

#define XOR_BLOCKS(a,b) { \
    ((uint64_t *)a)[0] ^= ((uint64_t *)b)[0]; \
    ((uint64_t *)a)[1] ^= ((uint64_t *)b)[1]; }

#define XOR_BLOCKS_DST(x,y,z) { \
    ((uint64_t *)z)[0] = ((uint64_t *)x)[0] ^ ((uint64_t *)y)[0]; \
    ((uint64_t *)z)[1] = ((uint64_t *)x)[1] ^ ((uint64_t *)y)[1]; }

#define MUL_SUM_XOR_DST(a,c,dst) { \
    uint64_t hi, lo = cuda_mul128(((uint64_t *)a)[0], ((uint64_t *)dst)[0], &hi) + ((uint64_t *)c)[1]; \
    hi += ((uint64_t *)c)[0]; \
    ((uint64_t *)c)[0] = ((uint64_t *)dst)[0] ^ hi; \
    ((uint64_t *)c)[1] = ((uint64_t *)dst)[1] ^ lo; \
    ((uint64_t *)dst)[0] = hi; \
    ((uint64_t *)dst)[1] = lo; }

#define E2I(x) ((size_t)((*((uint64_t*)x) / AES_BLOCK_SIZE) & (MEMORY / AES_BLOCK_SIZE - 1)))

#include "cuda_cryptonight_aes.cu"
#include "cuda_cryptonight_keccak.cu"
#include "cuda_cryptonight_blake.cu"
#include "cuda_cryptonight_groestl.cu"
#include "cuda_cryptonight_jh.cu"
#include "cuda_cryptonight_skein.cu"

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

__global__ void cryptonight_gpu_hash(int threads, uint32_t startNonce, uint32_t *resNonce, uint8_t *d_long_state, union cn_gpu_hash_state *d_hash_state)
{
	__shared__ uint32_t sharedMemory[1024];

    cn_aes_gpu_init(sharedMemory);

	__syncthreads();

    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
   
    if (thread < threads)
    {
        int i, j;
        uint32_t nonce = startNonce + thread;
        uint8_t *long_state = &d_long_state[MEMORY * thread];
        union cn_gpu_hash_state *state = &d_hash_state[thread];
        uint32_t input[19];
        uint32_t hash[8];
        uint32_t key[40];
        uint32_t ctx_a[4], ctx_b[4], ctx_c[4], ctx_text[32];

        for( i = 0; i < 19; i++ ) input[i] = d_input[i];
        *((uint32_t *)(((char *)input) + 39)) = nonce;

        cryptonight_keccak((uint8_t*)input, 76, (uint8_t*)&state->hs, 200);
        MEMCPY8(ctx_text, &state->init, 16);

        cn_aes_set_key((uint8_t *)key, state->hs.b);

        for (i = 0; i < MEMORY; i += INIT_SIZE_BYTE) {

            for( j = 0; j < 8; j++ ) {

                cn_aes_pseudo_round_mut(sharedMemory, &ctx_text[(AES_BLOCK_SIZE >> 2) * j], key);
            }

            MEMCPY8(&long_state[i], ctx_text, 16);
        }

        XOR_BLOCKS_DST(&state->k[0], &state->k[32], ctx_a);
        XOR_BLOCKS_DST(&state->k[16], &state->k[48], ctx_b);

        for (i = 0; i < ITER / 4; ++i) {

            j = E2I(ctx_a) * AES_BLOCK_SIZE;
            cn_aes_single_round(sharedMemory, &long_state[j], ctx_c, ctx_a);
            XOR_BLOCKS_DST(ctx_c, ctx_b, &long_state[j]);
            MUL_SUM_XOR_DST(ctx_c, ctx_a, &long_state[E2I(ctx_c) * AES_BLOCK_SIZE]);
            j = E2I(ctx_a) * AES_BLOCK_SIZE;
            cn_aes_single_round(sharedMemory, &long_state[j], ctx_b, ctx_a);
            XOR_BLOCKS_DST(ctx_b, ctx_c, &long_state[j]);
            MUL_SUM_XOR_DST(ctx_b, ctx_a, &long_state[E2I(ctx_b) * AES_BLOCK_SIZE]);
        }

        MEMCPY8(ctx_text, state->init, 16);

        cn_aes_set_key((uint8_t *)key, &state->hs.b[32]);

        for (i = 0; i < MEMORY; i += INIT_SIZE_BYTE) {
            
            for( j = 0; j < 8; j++ ) {

                XOR_BLOCKS(&ctx_text[(AES_BLOCK_SIZE>>2) * j], &long_state[i + j * AES_BLOCK_SIZE]);
                cn_aes_pseudo_round_mut(sharedMemory, &ctx_text[j * (AES_BLOCK_SIZE>>2)], key);
            }
        }
        MEMCPY8(state->init, ctx_text, 16);
        cryptonight_keccakf((uint64_t*)&state->hs, 24);

        switch( state->hs.b[0] & 3 ) {
            case 0:
                cn_blake((const uint8_t *)state, 200, (uint8_t *)hash);
                break;
            case 1:
                cn_groestl((const BitSequence *)state, 200, (BitSequence *)hash);
                break;
            case 2:
                cn_jh((const BitSequence *)state, 200, (BitSequence *)hash);
                break;
            case 3:
                cn_skein((const BitSequence *)state, 200, (BitSequence *)hash);
                break;
            default:
                break;
        }

        int position = -1;
        bool rc = true;

#pragma unroll 8
        for (i = 7; i >= 0; i--) {
            if (hash[i] > pTarget[i]) {
                if(position < i) {
                    position = i;
                    rc = false;
                }
             }
             if (hash[i] < pTarget[i]) {
                if(position < i) {
                    position = i;
                    rc = true;
                }
             }
        }

        if(rc == true)
            if(resNonce[0] > nonce)
                resNonce[0] = nonce;
    }
}

__host__ void cryptonight_cpu_setInput(int thr_id, void *data, void *pTargetIn)
{
    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	cudaMemcpyToSymbol(d_input, data, 76, 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(pTarget, pTargetIn, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__ void cryptonight_cpu_init(int thr_id, int threads)
{
	cn_aes_cpu_init();
    cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)); 
    cudaMemcpyToSymbol( keccakf_rndc, h_keccakf_rndc, sizeof(h_keccakf_rndc), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( keccakf_rotc, h_keccakf_rotc, sizeof(h_keccakf_rotc), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( keccakf_piln, h_keccakf_piln, sizeof(h_keccakf_piln), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( d_blake_sigma, h_blake_sigma, sizeof(h_blake_sigma), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol( d_blake_cst, h_blake_cst, sizeof(h_blake_cst), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( d_groestl_T, h_groestl_T, sizeof(h_groestl_T), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( d_JH256_H0, h_JH256_H0, sizeof(h_JH256_H0), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( d_E8_rc, h_E8_rc, sizeof(h_E8_rc), 0, cudaMemcpyHostToDevice);
}

__host__ void cryptonight_cpu_hash(int thr_id, int blocks, int threads, uint32_t startNonce, uint32_t *nonce, uint8_t *d_long_state, union cn_gpu_hash_state *d_hash_state)
{
    dim3 grid(blocks);
    dim3 block(threads);

    size_t shared_size = 1024;

    cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
    cryptonight_gpu_hash<<<grid, block, shared_size>>>(blocks*threads, startNonce, d_resultNonce[thr_id], d_long_state, d_hash_state);

    cudaDeviceSynchronize();

    cudaMemcpy(nonce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

