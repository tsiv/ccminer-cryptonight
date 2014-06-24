#include <unistd.h>
#include <stdio.h>

extern "C"
{
#include "cpuminer-config.h"
#include "miner.h"
#include "cryptonight.h"
}

extern int device_map[8];

static uint8_t *d_long_state[8];
static union cn_gpu_hash_state *d_hash_state[8];

extern bool opt_benchmark;
extern int opt_cn_threads;
extern int opt_cn_blocks;

extern void cryptonight_cpu_init(int thr_id, int threads);
extern void cryptonight_cpu_setInput(int thr_id, void *data, void *pTargetIn);
extern void cryptonight_cpu_hash(int thr_id, int threads, uint32_t startNonce, uint32_t *nonce, uint8_t *d_long_state, union cn_gpu_hash_state *d_hash_state);

extern "C" void cryptonight_hash(void* output, const void* input, size_t len);

extern "C" int scanhash_cryptonight(int thr_id, uint32_t *pdata,
    const uint32_t *ptarget, uint32_t max_nonce,
    unsigned long *hashes_done)
{
    uint32_t *nonceptr = (uint32_t*)(((char*)pdata) + 39);
    const uint32_t first_nonce = *nonceptr;
    uint32_t nonce = *nonceptr;

	if (opt_benchmark) {
		((uint32_t*)ptarget)[7] = 0x0000ff;
        pdata[17] = 0;
    }
	const uint32_t Htarg = ptarget[7];
	const int throughput = opt_cn_threads * opt_cn_blocks;

    static bool init[8] = { false, false, false, false, false, false, false, false };
	if (!init[thr_id])
	{
        cudaSetDevice(device_map[thr_id]);
		cudaMalloc(&d_long_state[thr_id], MEMORY * throughput);
		cudaMalloc(&d_hash_state[thr_id], sizeof(union cn_gpu_hash_state) * throughput);
		cryptonight_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

    cryptonight_cpu_setInput(thr_id, (void *)pdata, (void*)ptarget);

	do {
        uint32_t foundNonce = 0xFFFFFFFF;

        cryptonight_cpu_hash(thr_id, throughput, nonce, &foundNonce, d_long_state[thr_id], d_hash_state[thr_id]);

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
