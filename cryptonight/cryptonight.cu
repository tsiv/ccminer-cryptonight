#include <ctype.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include "cuda.h"
#include "cuda_runtime.h"

#ifdef WIN32
#include "cpuminer-config-win.h"
#else
#include "cpuminer-config.h"
#endif
#include "cryptonight.h"

extern uint64_t MEMORY;
extern uint32_t ITER;
extern void proper_exit(int);

void cryptonight_core_cpu_hash(int thr_id, int blocks, int threads, uint32_t *d_long_state, uint32_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint32_t *d_ctx_tweak1_2);

void cryptonight_extra_cpu_setData(int thr_id, const void *data, const void *pTargetIn);
void cryptonight_extra_cpu_init(int thr_id);
void cryptonight_extra_cpu_prepare(int thr_id, int threads, uint32_t startNonce, uint32_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint32_t *d_ctx_tweak1_2);
void cryptonight_extra_cpu_final(int thr_id, int threads, uint32_t startNonce, uint32_t *nonce, uint32_t *d_ctx_state);

extern char *device_name[MAX_GPU];
extern int device_arch[MAX_GPU][2];
extern int device_mpcount[MAX_GPU];
extern int device_map[MAX_GPU];
extern int device_config[MAX_GPU][2];
extern algo_t opt_algo;
extern int forkversion;

void exit_if_cudaerror(int thr_id, const char *file, int line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("\nGPU %d: %s\n%s line %d\n", device_map[thr_id], cudaGetErrorString(err), file, line);
		proper_exit(1);
	}
}

//Number of CUDA Devices on the system
int cuda_num_devices()
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
void cuda_set_device_config(int GPU_N)
{
	for (int i = 0; i < GPU_N; i++)
	{
		cudaError_t err;

		err = cudaSetDevice(device_map[i]);
		if (err != cudaSuccess)
		{
			applog(LOG_ERR, "GPU %d: %s", device_map[i], cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaDeviceReset();
		if (err != cudaSuccess)
		{
			applog(LOG_ERR, "GPU %d: %s", device_map[i], cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		if (device_config[i][0] == 0)
		{
			int len = strlen(device_name[i]);
			device_config[i][0] = device_mpcount[i] * 4;
			if(len > 0 && device_name[i][len - 1] == 'M') // mobile gpu
				device_config[i][1] = 8;
			else
				device_config[i][1] = 32;

			if (device_arch[i][0] < 6)
			{
				//Try to stay under 950 threads ( 1900MiB memory per for hashes )
				while (device_config[i][0] * device_config[i][1] >= 950 && device_config[i][0] > device_mpcount[i])
				{
					device_config[i][0] -= device_mpcount[i];
				}
			}
			//Stay within 85% of the available RAM
			size_t freeMemory = 0;
			size_t totalMemoery = 0;

			err = cudaMemGetInfo(&freeMemory, &totalMemoery);
			if (err == cudaSuccess)
			{
				freeMemory = (freeMemory * size_t(85)) / 100;
				while (device_config[i][0] > device_mpcount[i])
				{

					if (freeMemory > size_t(device_config[i][0]) * size_t(device_config[i][1]) * (MEMORY + 688))
					{
						break;
					}
					else
					{
						device_config[i][0] -= device_mpcount[i];
					}
				}
			}
			else
			{
				applog(LOG_ERR, "GPU #%d: CUDA error: %s", device_map[i], cudaGetErrorString(err));
				exit(1);
			}
		}
	}
}
void cuda_deviceinfo(int GPU_N)
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

int cuda_finddevice(char *name)
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
static uint32_t *d_ctx_tweak1_2[MAX_GPU];

extern bool opt_benchmark;
extern bool stop_mining;
extern volatile bool mining_has_stopped[MAX_GPU];


int cryptonight_hash(void* output, const void* input, size_t len, int variant);

int scanhash_cryptonight(int thr_id, uint32_t *pdata, const uint32_t *ptarget, uint32_t max_nonce, unsigned long *hashes_done, uint32_t *results)
{
	cudaError_t err;
	int res;
	uint32_t *nonceptr = (uint32_t*)(((char*)pdata) + 39);
	int variant = 0;
	const uint32_t first_nonce = *nonceptr;
	uint32_t nonce = *nonceptr;
	int cn_blocks = device_config[thr_id][0];
	int cn_threads = device_config[thr_id][1];

	bool heavy;
	if (opt_algo == algo_sumokoin)
		heavy = true;
	else
	{
		heavy = false;
		if (opt_algo != algo_old)
			variant = ((uint8_t*)pdata)[0] >= forkversion ? ((uint8_t*)pdata)[0] - forkversion + 1 : 0;
	}

	if(opt_benchmark)
	{
		((uint32_t*)ptarget)[7] = 0x0002ffff;
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
		cudaMalloc(&d_ctx_tweak1_2[thr_id], 2 * sizeof(uint32_t) * throughput);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);

		cryptonight_extra_cpu_init(thr_id);

		init[thr_id] = true;
	}

	cryptonight_extra_cpu_setData(thr_id, (const void *)pdata, (const void *)ptarget);

	do
	{
		uint32_t foundNonce[2];

		if (!heavy)
		{
			cryptonight_extra_cpu_prepare(thr_id, throughput, nonce, d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id], d_ctx_key1[thr_id], d_ctx_key2[thr_id], variant, d_ctx_tweak1_2[thr_id]);
			cryptonight_core_cpu_hash(thr_id, cn_blocks, cn_threads, d_long_state[thr_id], d_ctx_state[thr_id], d_ctx_a[thr_id], d_ctx_b[thr_id], d_ctx_key1[thr_id], d_ctx_key2[thr_id], variant, d_ctx_tweak1_2[thr_id]);
			cryptonight_extra_cpu_final(thr_id, throughput, nonce, foundNonce, d_ctx_state[thr_id]);
		}
		else
		{ } // not implemented yet

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
			const int rc = cryptonight_hash(vhash64, tempdata, 76, variant);
			if(rc && (vhash64[7] <= Htarg) && fulltest(vhash64, ptarget))
			{
				res = 1;
				if(opt_debug || opt_benchmark)
					applog(LOG_DEBUG, "GPU #%d: found nonce $%08X", device_map[thr_id], foundNonce[0]);
				results[0] = foundNonce[0];
				*hashes_done = nonce - first_nonce + throughput;
				if(foundNonce[1] < 0xffffffff)
				{
					*tempnonceptr = foundNonce[1];
					const int rc = cryptonight_hash(vhash64, tempdata, 76, variant);
					if(rc && (vhash64[7] <= Htarg) && fulltest(vhash64, ptarget))
					{
						res++;
						if(opt_debug || opt_benchmark)
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
