#include <ctype.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"

extern "C"
{
#include "cpuminer-config.h"
#include "miner.h"
}
#include "cryptonight.h"

extern char *device_name[MAX_GPU];
extern int device_arch[MAX_GPU][2];
extern int device_mpcount[MAX_GPU];
extern int device_map[MAX_GPU];
extern int device_config[MAX_GPU][2];

//Number of CUDA Devices on the system
extern "C" int cuda_num_devices()
{
	int version;
	cudaError_t err = cudaDriverGetVersion(&version);
	if(err != cudaSuccess)
	{
		applog(LOG_ERR, "Unable to query CUDA driver version! Is an nVidia driver installed?");
		exit(1);
	}

	if(version < CUDART_VERSION)
	{
		applog(LOG_ERR, "Driver does not support CUDA %d.%d API! Update your nVidia driver!", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
		exit(1);
	}

	int GPU_N;
	err = cudaGetDeviceCount(&GPU_N);
	if(err != cudaSuccess)
	{
		if(err != cudaErrorNoDevice)
			applog(LOG_ERR, "No CUDA device found!");
		else
			applog(LOG_ERR, "Unable to query number of CUDA devices!");
		exit(1);
	}
	return GPU_N;
}
extern "C" void cuda_set_device_config(int GPU_N)
{
	for(int i = 0; i < GPU_N; i++)
	{
		if(device_config[i][0] == 0)
		{
			device_config[i][0] = device_mpcount[i] * 4;
			device_config[i][1] = 64;

			/* sm_20 devices can only run 512 threads per cuda block
			* `cryptonight_core_gpu_phase1` and `cryptonight_core_gpu_phase3` starts
			* `8 * ctx->device_threads` threads per block
			*/
			if(device_arch[i][0] < 6)
			{
				//Try to stay under 950 threads ( 1900MiB memory per for hashes )
				while(device_config[i][0] * device_config[i][1] >= 950 && device_config[i][1] > 2)
				{
					device_config[i][1] /= 2;
				}
			}
			//Stay within 85% of the available RAM
			while(device_config[i][1] > 2)
			{
				size_t freeMemory = 0;
				size_t totalMemoery = 0;

				cudaError_t err = cudaSetDevice(device_map[i]);
				if(err != cudaSuccess)
				{
					applog(LOG_ERR, "GPU %d: %s", device_map[i], cudaGetErrorString(err));
					exit(EXIT_FAILURE);
				}
				err = cudaMemGetInfo(&freeMemory, &totalMemoery);
				if(err == cudaSuccess)
				{
					freeMemory = (freeMemory * size_t(85)) / 100;

					if(freeMemory > size_t(device_config[i][0]) * size_t(device_config[i][1]) * 2097832)
					{
						break;
					}
					else
					{
						device_config[i][1] /= 2;
					}
				}
				else
					applog(LOG_ERR, "GPU #%d: CUDA error: %s", device_map[i], cudaGetErrorString(err));
			}
		}
	}
}
extern "C" void cuda_deviceinfo(int GPU_N)
{
	cudaError_t err;
	for(int i = 0; i < GPU_N; i++)
	{
		cudaDeviceProp props;
		err = cudaGetDeviceProperties(&props, device_map[i]);
		if(err != cudaSuccess)
		{
			printf("\nGPU %d: %s\n%s line %d\n", device_map[i], cudaGetErrorString(err), __FILE__, __LINE__);
			exit(1);
		}

		device_name[i] = strdup(props.name);
		device_mpcount[i] = props.multiProcessorCount;
		device_arch[i][0] = props.major;
		device_arch[i][1] = props.minor;
	}
}

static bool substringsearch(const char *haystack, const char *needle, int &match)
{
	int hlen = (int)strlen(haystack);
	int nlen = (int)strlen(needle);
	for(int i = 0; i < hlen; ++i)
	{
		if(haystack[i] == ' ') continue;
		int j = 0, x = 0;
		while(j < nlen)
		{
			if(haystack[i + x] == ' ')
			{
				++x; continue;
			}
			if(needle[j] == ' ')
			{
				++j; continue;
			}
			if(needle[j] == '#') return ++match == needle[j + 1] - '0';
			if(tolower(haystack[i + x]) != tolower(needle[j])) break;
			++j; ++x;
		}
		if(j == nlen) return true;
	}
	return false;
}

extern "C" int cuda_finddevice(char *name)
{
	int num = cuda_num_devices();
	int match = 0;
	for(int i = 0; i < num; ++i)
	{
		cudaDeviceProp props;
		if(cudaGetDeviceProperties(&props, i) == cudaSuccess)
			if(substringsearch(props.name, name, match)) return i;
	}
	return -1;
}

static uint32_t *d_long_state[MAX_GPU];
static uint32_t *d_ctx_state[MAX_GPU];
static uint32_t *d_ctx_a[MAX_GPU];
static uint32_t *d_ctx_b[MAX_GPU];
static uint32_t *d_ctx_key1[MAX_GPU];
static uint32_t *d_ctx_key2[MAX_GPU];
static uint32_t *d_ctx_text[MAX_GPU];

extern bool opt_benchmark;
extern bool stop_mining;
extern volatile bool mining_has_stopped[MAX_GPU];


extern "C" void cryptonight_hash(void* output, const void* input, size_t len);

extern "C" int scanhash_cryptonight(int thr_id, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done, uint32_t *results)
{
	cudaError_t err;
	int res;
	uint32_t *nonceptr = (uint32_t*)(((char*)pdata) + 39);
	const uint32_t first_nonce = *nonceptr;
	uint32_t nonce = *nonceptr;
	int cn_blocks = device_config[thr_id][0];
	int cn_threads = device_config[thr_id][1];
	if(opt_benchmark)
	{
		((uint32_t*)ptarget)[7] = 0x0000ff;
		pdata[17] = 0;
	}
	const uint32_t Htarg = ptarget[7];
	const uint32_t throughput = cn_threads * cn_blocks;
	if(sizeof(size_t) == 4 && throughput > 0xffffffff / MEMORY)
	{
		applog(LOG_ERR, "GPU %d: THE 32bit VERSION CAN'T ALLOCATE MORE THAN 4GB OF MEMORY!", device_map[thr_id]);
		applog(LOG_ERR, "GPU %d: PLEASE REDUCE THE NUMBER OF THREADS OR BLOCKS", device_map[thr_id]);
		exit(1);
	}
	const size_t alloc = MEMORY * throughput;

	static bool init[MAX_GPU] = {false, false, false, false, false, false, false, false};
	if(!init[thr_id])
	{
		err = cudaSetDevice(device_map[thr_id]);
		if(err != cudaSuccess)
		{
			applog(LOG_ERR, "GPU %d: %s", device_map[thr_id], cudaGetErrorString(err));
		}
		cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		cudaMalloc(&d_long_state[thr_id], alloc);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		cudaMalloc(&d_ctx_state[thr_id], 50 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		cudaMalloc(&d_ctx_key1[thr_id], 40 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		cudaMalloc(&d_ctx_key2[thr_id], 40 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		cudaMalloc(&d_ctx_text[thr_id], 32 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		cudaMalloc(&d_ctx_a[thr_id], 4 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		cudaMalloc(&d_ctx_b[thr_id], 4 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);

		cryptonight_extra_cpu_init(thr_id);

		init[thr_id] = true;
	}

	cryptonight_extra_cpu_setData(thr_id, (const void *)pdata, (const void *)ptarget);

	do
	{
		uint32_t foundNonce[2];

		cryptonight_extra_cpu_prepare(thr_id, throughput, nonce, d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id], d_ctx_key1[thr_id], d_ctx_key2[thr_id]);
		cryptonight_core_cpu_hash(thr_id, cn_blocks, cn_threads, d_long_state[thr_id], d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id], d_ctx_key1[thr_id], d_ctx_key2[thr_id]);
		cryptonight_extra_cpu_final(thr_id, throughput, nonce, foundNonce, d_ctx_state[thr_id]);

		if(stop_mining)
		{
			mining_has_stopped[thr_id] = true;
			pthread_exit(NULL);
		}

		if(foundNonce[0] < 0xffffffff)
		{
			uint32_t vhash64[8] = {0, 0, 0, 0, 0, 0, 0, 0};
			uint32_t tempdata[19];
			memcpy(tempdata, pdata, 76);
			uint32_t *tempnonceptr = (uint32_t*)(((char*)tempdata) + 39);
			*tempnonceptr = foundNonce[0];
			cryptonight_hash(vhash64, tempdata, 76);
			if((vhash64[7] <= Htarg) && fulltest(vhash64, ptarget))
			{
				res = 1;
				if(opt_debug)
					applog(LOG_DEBUG, "GPU #%d: found nonce $%08X", device_map[thr_id], foundNonce[0]);
				results[0] = foundNonce[0];
				*hashes_done = nonce - first_nonce + throughput;
				if(foundNonce[1] < 0xffffffff)
				{
					*tempnonceptr = foundNonce[1];
					cryptonight_hash(vhash64, tempdata, 76);
					if((vhash64[7] <= Htarg) && fulltest(vhash64, ptarget))
					{
						res++;
						if(opt_debug)
							applog(LOG_DEBUG, "GPU #%d: found nonce $%08X", device_map[thr_id], foundNonce[1]);
						results[1] = foundNonce[1];
					}
					else
					{
						applog(LOG_WARNING, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce[1]);
					}
				}
				return res;
			}
			else
			{
				applog(LOG_WARNING, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce[0]);
			}
		}
		if((nonce & 0x00ffffff) > (0x00ffffff - throughput))
			nonce = max_nonce;
		else
			nonce += throughput;
	} while(nonce < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = nonce - first_nonce;
	return 0;
}
