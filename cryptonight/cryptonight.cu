#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"

extern "C"
{
#include "cpuminer-config.h"
#include "miner.h"
#include "cryptonight.h"
}

extern int device_map[8];
extern int device_config[8][2];

static uint8_t *d_long_state[8];
static struct cryptonight_gpu_ctx *d_ctx[8];

extern bool opt_benchmark;

extern void cryptonight_core_cpu_init(int thr_id, int threads);
extern void cryptonight_core_cpu_hash(int thr_id, int blocks, int threads, uint8_t *d_long_state, struct cryptonight_gpu_ctx *d_ctx);

extern void cryptonight_extra_cpu_setData(int thr_id, const void *data, const void *pTargetIn);
extern void cryptonight_extra_cpu_init(int thr_id);
extern void cryptonight_extra_cpu_prepare(int thr_id, int threads, uint32_t startNonce, struct cryptonight_gpu_ctx *d_ctx);
extern void cryptonight_extra_cpu_final(int thr_id, int threads, uint32_t startNonce, uint32_t *nonce, struct cryptonight_gpu_ctx *d_ctx);

extern "C" void cryptonight_hash(void* output, const void* input, size_t len);

extern "C" int scanhash_cryptonight(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
    uint32_t *nonceptr = (uint32_t*)(((char*)pdata) + 39);
    const uint32_t first_nonce = *nonceptr;
    uint32_t nonce = *nonceptr;
    int cn_blocks = device_config[thr_id][0], cn_threads = device_config[thr_id][1];

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x0000ff;
        pdata[17] = 0;
    }
	const uint32_t Htarg = ptarget[7];
	const int throughput = cn_threads * cn_blocks;
    const size_t alloc = MEMORY * throughput;

    static bool init[8] = { false, false, false, false, false, false, false, false };
	if (!init[thr_id])
	{
        cudaSetDevice(device_map[thr_id]);
        cudaDeviceReset();
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if( cudaMalloc(&d_long_state[thr_id], alloc) != cudaSuccess ) {
            applog(LOG_ERR, "GPU #%d: FATAL: failed to allocate device memory for long state", thr_id);
            exit(1);
        }
		if( cudaMalloc(&d_ctx[thr_id], sizeof(struct cryptonight_gpu_ctx) * throughput) != cudaSuccess ) {
            applog(LOG_ERR, "GPU #%d: FATAL: failed to allocate device memory for hash context", thr_id);
            exit(1);
        }
		cryptonight_core_cpu_init(thr_id, throughput);
        cryptonight_extra_cpu_init(thr_id);
		init[thr_id] = true;
	}

    cryptonight_extra_cpu_setData(thr_id, (const void *)pdata, (const void *)ptarget);

	do {
        uint32_t foundNonce = 0xFFFFFFFF;

        cryptonight_extra_cpu_prepare(thr_id, throughput, nonce, d_ctx[thr_id]);
        cryptonight_core_cpu_hash(thr_id, cn_blocks, cn_threads, d_long_state[thr_id], d_ctx[thr_id]);
        cryptonight_extra_cpu_final(thr_id, throughput, nonce, &foundNonce, d_ctx[thr_id]);

        if (foundNonce < 0xffffffff)
		{
			uint32_t vhash64[8];
            uint32_t tempdata[19];
            memcpy(tempdata, pdata, 76);
            uint32_t *tempnonceptr = (uint32_t*)(((char*)tempdata) + 39);
			*tempnonceptr = foundNonce;
			cryptonight_hash(vhash64, tempdata, 76);

            if( (vhash64[7] <= Htarg) && fulltest(vhash64, ptarget) ) {
                
			    *nonceptr = foundNonce;
                *hashes_done = foundNonce - first_nonce + 1;
                return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
			}
		
            foundNonce = 0xffffffff;
        }

		nonce += throughput;
	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = nonce - first_nonce + 1;
	return 0;
}
