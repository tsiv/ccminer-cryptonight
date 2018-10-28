/*
* Copyright 2010 Jeff Garzik
* Copyright 2012-2014 pooler
*
* This program is free software; you can redistribute it and/or modify it
* under the terms of the GNU General Public License as published by the Free
* Software Foundation; either version 2 of the License, or (at your option)
* any later version.  See COPYING for more details.
*/
#ifdef WIN32
#include "cpuminer-config-win.h"
#else
#include "cpuminer-config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#ifdef WIN32
#include <windows.h>
#else
#include <errno.h>
#include <signal.h>
#include <sys/resource.h>
#if HAVE_SYS_SYSCTL_H
#include <sys/types.h>
#if HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
#include <sys/sysctl.h>
#endif
#endif
#include <jansson.h>
#include <curl/curl.h>
#include <openssl/sha.h>
#include <cuda_runtime.h>
#include "compat.h"
#include "cryptonight.h"

#define PROGRAM_NAME "ccminer-cryptonight"
#define LP_SCANTIME	60

int cuda_num_devices();
void cuda_deviceinfo(int);
void cuda_set_device_config(int);
int cuda_finddevice(char *name);

extern int cryptonight_hash(void* output, const void* input, size_t len, int variant, algo_t opt_algo);
void parse_device_config(int device, char *config, int *blocks, int *threads);

#ifdef __linux /* Linux specific policy and affinity management */
#include <sched.h>
static inline void drop_policy(void)
{
	struct sched_param param;
	param.sched_priority = 0;

#ifdef SCHED_IDLE
	if(unlikely(sched_setscheduler(0, SCHED_IDLE, &param) == -1))
#endif
#ifdef SCHED_BATCH
		sched_setscheduler(0, SCHED_BATCH, &param);
#endif
}

static inline void affine_to_cpu(int id, int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(&set), &set);
}
#elif defined(__FreeBSD__) /* FreeBSD specific policy and affinity management */
#include <sys/cpuset.h>
static inline void drop_policy(void)
{}

static inline void affine_to_cpu(int id, int cpu)
{
	cpuset_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_TID, -1, sizeof(cpuset_t), &set);
}
#else
static inline void drop_policy(void)
{}

static inline void affine_to_cpu(int id, int cpu)
{}
#endif

enum workio_commands
{
	WC_GET_WORK,
	WC_SUBMIT_WORK,
};

struct workio_cmd
{
	enum workio_commands cmd;
	struct thr_info	*thr;
	union
	{
		struct work	*work;
	} u;
};

bool stop_mining = false;
volatile bool mining_has_stopped[8] = {false};
bool opt_colors = false;
#ifdef WIN32
HANDLE handl;
#endif

extern uint64_t MEMORY;
extern uint32_t ITER;

algo_t opt_algo = algo_monero;
int forkversion = 7;
bool opt_debug = false;
bool opt_protocol = false;
bool opt_keepalive = false;
bool opt_benchmark = false;
bool want_longpoll = true;
bool have_longpoll = false;
bool want_stratum = true;
bool have_stratum = false;
static bool submit_old = false;
bool use_syslog = false;
static bool opt_background = false;
static bool opt_quiet = false;
static int opt_retries = -1;
static int opt_fail_pause = 30;
int opt_timeout = 270;
static int opt_scantime = 5;
static json_t *opt_config;
static const bool opt_time = true;
static int opt_n_threads = 0;
static double opt_difficulty = 1; // CH
static int num_processors;
int device_map[MAX_GPU]; // CB
char *device_name[MAX_GPU]; // CB
int device_arch[MAX_GPU][2];
int device_mpcount[MAX_GPU];
int device_bfactor[MAX_GPU];
int device_bsleep[MAX_GPU];
int device_config[MAX_GPU][2];
#ifdef WIN32
static int default_bfactor = 6;
static int default_bsleep = 100;
#else
static int default_bfactor = 0;
static int default_bsleep = 0;
#endif
static char *rpc_url;
static char *rpc_userpass;
static char *rpc_user, *rpc_pass;
char *opt_cert;
char *opt_proxy;
long opt_proxy_type;
struct thr_info *thr_info;
static int work_thr_id;
int longpoll_thr_id = -1;
int stratum_thr_id = -1;
struct work_restart *work_restart = NULL;
static struct stratum_ctx stratum;
int opt_cn_threads = 8;
int opt_cn_blocks = 0;

static char rpc2_id[64] = "";
static char *rpc2_blob = NULL;
static int rpc2_bloblen = 0;
static uint32_t rpc2_target = 0;
static char *rpc2_job_id = NULL;
static pthread_mutex_t rpc2_job_lock;
static pthread_mutex_t rpc2_login_lock;

pthread_mutex_t applog_lock;
static pthread_mutex_t stats_lock;

static unsigned long accepted_count = 0L;
static unsigned long rejected_count = 0L;
static double *thr_hashrates;
static uint64_t *thr_totalhashes;
static struct timeval stats_start;


#ifdef HAVE_GETOPT_LONG
#include <getopt.h>
#else
struct option
{
	const char *name;
	int has_arg;
	int *flag;
	int val;
};
#endif

const char *algo_names[] =
{
	"cryptonight",
	"monero",
	"graft",
	"stellite",
	"intense",
	"electroneum",
	"sumokoin"
};

static char const usage[] = "\
Usage: " PROGRAM_NAME " [OPTIONS]\n\
    Options:\n\
        -a  --algo              choose between the supported algos:\n\
                                  cryptonight\n\
                                  monero\n\
                                  electroneum\n\
                                  graft\n\
                                  stellite\n\
                                  intense\n\
                                  sumokoin\n\
        -d, --devices           takes a comma separated list of CUDA devices to use.\n\
                                Device IDs start counting from 0! Alternatively takes\n\
                                string names of your cards like gtx780ti or gt640#2\n\
                                (matching 2nd gt640 in the PC)\n\
        -f, --diff              Divide difficulty by this factor (std is 1) \n\
        -l, --launch=CONFIG     launch config for the Cryptonight kernel.\n\
                                a comma separated list of values in form of\n\
                                AxB where A is the number of threads to run in\n\
                                each thread block and B is the number of thread\n\
                                blocks to launch. If less values than devices in use\n\
                                are provided, the last value will be used for\n\
                                the remaining devices. If you don't need to vary the\n\
                                value between devices, you can just enter a single value\n\
                                and it will be used for all devices. (default: 8x40)\n\
            --bfactor=X         Enables running the Cryptonight kernel in smaller pieces.\n\
                                The kernel will be run in 2^X parts according to bfactor,\n\
                                with a small pause between parts, specified by --bsleep.\n\
                                This is a per-device setting like the launch config.\n\
                                (default: 0 (no splitting) on Linux, 6 (64 parts) on Windows)\n\
            --bsleep=X          Insert a delay of X microseconds between kernel launches.\n\
                                Use in combination with --bfactor to mitigate the lag\n\
                                when running on your primary GPU.\n\
                                This is a per-device setting like the launch config.\n\
        -o, --url=URL           URL of mining server\n\
        -O, --userpass=U:P      username:password pair for mining server\n\
        -u, --user=USERNAME     username for mining server\n\
        -p, --pass=PASSWORD     password for mining server\n\
            --cert=FILE         certificate for mining server using SSL\n\
        -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy\n\
        -k, --keepalive         send keepalive requests to avoid a stratum timeout\n\
        -t, --threads=N         number of miner threads (default: number of nVidia GPUs)\n\
        -r, --retries=N         number of times to retry if a network call fails\n\
                                (default: retry indefinitely)\n\
        -R, --retry-pause=N     time to pause between retries, in seconds (default: 30)\n\
            --timeout=N         network timeout, in seconds (default: 270)\n\
        -s, --scantime=N        upper bound on time spent scanning current work when\n\
                                long polling is unavailable, in seconds (default: 5)\n\
            --no-longpoll       disable X-Long-Polling support\n\
            --no-stratum        disable X-Stratum support\n\
        -q, --quiet             disable per-thread hashmeter output\n\
        -D, --debug             enable debug output\n\
            --color             enable output with colors\n\
        -P, --protocol-dump     verbose dump of protocol-level activities\n"
#ifdef HAVE_SYSLOG_H
"\
        -S, --syslog            use system log for output messages\n"
#endif
#ifndef WIN32
"\
        -B, --background        run the miner in the background\n"
#endif
"\
            --benchmark         run in offline benchmark mode\n\
        -c, --config=FILE       load a JSON-format configuration file\n\
        -V, --version           display version information and exit\n\
        -h, --help              display this help text and exit\n\
";

static char const short_options[] =
#ifdef HAVE_SYSLOG_H
"S"
#endif
"a:Bc:Dhp:Px:kqr:R:s:t:T:o:u:O:Vd:f:l:";

static struct option const options[] = {
	{"algo", 1, NULL, 'a' },
	{"background", 0, NULL, 'B'},
	{"benchmark", 0, NULL, 1005},
	{"cert", 1, NULL, 1001},
	{"config", 1, NULL, 'c'},
	{"debug", 0, NULL, 'D'},
	{"help", 0, NULL, 'h'},
	{"keepalive", 0, NULL, 'k'},
	{"no-longpoll", 0, NULL, 1003},
	{"no-stratum", 0, NULL, 1007},
	{"pass", 1, NULL, 'p'},
	{"protocol-dump", 0, NULL, 'P'},
	{"proxy", 1, NULL, 'x'},
	{"quiet", 0, NULL, 'q'},
	{"retries", 1, NULL, 'r'},
	{"retry-pause", 1, NULL, 'R'},
	{"scantime", 1, NULL, 's'},
#ifdef HAVE_SYSLOG_H
	{"syslog", 0, NULL, 'S'},
#endif
	{"threads", 1, NULL, 't'},
	{"timeout", 1, NULL, 'T'},
	{"url", 1, NULL, 'o'},
	{"user", 1, NULL, 'u'},
	{"userpass", 1, NULL, 'O'},
	{"version", 0, NULL, 'V'},
	{"devices", 1, NULL, 'd'},
	{"diff", 1, NULL, 'f'},
	{"launch", 1, NULL, 'l'},
	{"launch-config", 1, NULL, 'l'},
	{"bfactor", 1, NULL, 1008},
	{"bsleep", 1, NULL, 1009},
	{"color", 0, NULL, 1010},
	{0, 0, 0, 0}
};

static struct work g_work;
static time_t g_work_time;
static pthread_mutex_t g_work_lock;

static bool rpc2_login(CURL *curl);

void cuda_devicereset(int threads)
{
	for(int i = 0; i < threads; i++)
	{
		cudaError_t err;
		err = cudaSetDevice(device_map[i]);
		if(err == cudaSuccess)
		{
			cudaDeviceSynchronize();
			cudaDeviceReset();
		}
		else
			applog(LOG_WARNING, "can't reset GPU #%d", device_map[i]);
	}
}

void proper_exit(int exitcode)
{
	pthread_mutex_lock(&g_work_lock);	//freeze stratum
	stop_mining = true;
	applog(LOG_INFO, "stopping %d threads", opt_n_threads);
	bool everything_stopped;
	do
	{
		everything_stopped = true;
		for(int i = 0; i < opt_n_threads; i++)
		{
			if(!mining_has_stopped[i])
				everything_stopped = false;
		}
	} while(!everything_stopped);
	applog(LOG_INFO, "resetting GPUs");
	cuda_devicereset(opt_n_threads);

	curl_global_cleanup();

#ifdef WIN32
	timeEndPeriod(1); // else never executed
#endif
	exit(exitcode);
}

json_t *json_rpc2_call_recur(CURL *curl, const char *url,
							 const char *userpass, json_t *rpc_req,
							 int *curl_err, int flags, int recur)
{
	if(recur >= 5)
	{
		if(opt_debug)
			applog(LOG_DEBUG, "Failed to call rpc command after %i tries", recur);
		return NULL;
	}
	if(!strcmp(rpc2_id, ""))
	{
		if(opt_debug)
			applog(LOG_DEBUG, "Tried to call rpc2 command before authentication");
		return NULL;
	}
	json_t *params = json_object_get(rpc_req, "params");
	if(params)
	{
		json_t *auth_id = json_object_get(params, "id");
		if(auth_id)
		{
			json_string_set(auth_id, rpc2_id);
		}
	}
	json_t *res = json_rpc_call(curl, url, userpass, json_dumps(rpc_req, 0), false, false,
								curl_err);
	if(!res)
		return res;
	json_t *error = json_object_get(res, "error");
	if(!error)
		return res;
	json_t *message;
	if(json_is_string(error))
		message = error;
	else
		message = json_object_get(error, "message");
	if(!message || !json_is_string(message))
		return res;
	const char *mes = json_string_value(message);
	if(!strcmp(mes, "Unauthenticated"))
	{
		pthread_mutex_lock(&rpc2_login_lock);
		rpc2_login(curl);
		sleep(1);
		pthread_mutex_unlock(&rpc2_login_lock);
		return json_rpc2_call_recur(curl, url, userpass, rpc_req,
									curl_err, flags, recur + 1);
	}
	else if(!strcmp(mes, "Low difficulty share") || !strcmp(mes, "Block expired") || !strcmp(mes, "Invalid job id") || !strcmp(mes, "Duplicate share"))
	{
		json_t *result = json_object_get(res, "result");
		if(!result)
		{
			return res;
		}
		json_object_set(result, "reject-reason", json_string(mes));
	}
	else
	{
		applog(LOG_ERR, "json_rpc2.0 error: %s", mes);
		return NULL;
	}
	return res;
}

json_t *json_rpc2_call(CURL *curl, const char *url,
											 const char *userpass, const char *rpc_req,
											 int *curl_err, int flags)
{
	return json_rpc2_call_recur(curl, url, userpass, JSON_LOADS(rpc_req, NULL),
															curl_err, flags, 0);
}

static bool jobj_binary(const json_t *obj, const char *key,
												void *buf, size_t buflen)
{
	const char *hexstr;
	json_t *tmp;

	tmp = json_object_get(obj, key);
	if(unlikely(!tmp))
	{
		applog(LOG_ERR, "JSON key '%s' not found", key);
		return false;
	}
	hexstr = json_string_value(tmp);
	if(unlikely(!hexstr))
	{
		applog(LOG_ERR, "JSON key '%s' is not a string", key);
		return false;
	}
	if(!hex2bin((unsigned char*)buf, hexstr, buflen))
		return false;

	return true;
}

bool rpc2_job_decode(const json_t *job, struct work *work)
{
	json_t *tmp;
	tmp = json_object_get(job, "job_id");
	if(!tmp)
	{
		applog(LOG_ERR, "JSON inval job id");
		return false;
	}
	const char *job_id = json_string_value(tmp);
	tmp = json_object_get(job, "blob");
	if(!tmp)
	{
		applog(LOG_ERR, "JSON inval blob");
		return false;
	}
	const char *hexblob = json_string_value(tmp);
	int blobLen = (int)strlen(hexblob);
	if(blobLen % 2 != 0 || ((blobLen / 2) < 40 && blobLen != 0) || (blobLen / 2) > 128)
	{
		applog(LOG_ERR, "JSON invalid blob length");
		return false;
	}
	if(blobLen != 0)
	{
		pthread_mutex_lock(&rpc2_job_lock);
		char *blob = (char *)malloc(blobLen / 2);
		if (blob == NULL)
		{
			applog(LOG_ERR, "file %s line %d: Out of memory!", __FILE__, __LINE__);
			proper_exit(1);
		}
		if(!hex2bin((unsigned char *)blob, hexblob, blobLen / 2))
		{
			applog(LOG_ERR, "JSON inval blob");
			pthread_mutex_unlock(&rpc2_job_lock);
			return false;
		}
		if(rpc2_blob)
		{
			free(rpc2_blob);
		}
		rpc2_bloblen = blobLen / 2;
		rpc2_blob = (char *)malloc(rpc2_bloblen);
		if (rpc2_blob == NULL)
		{
			applog(LOG_ERR, "file %s line %d: Out of memory!", __FILE__, __LINE__);
			proper_exit(1);
		}
		memcpy(rpc2_blob, blob, blobLen / 2);

		free(blob);

		uint32_t target;
		jobj_binary(job, "target", &target, 4);
		if(rpc2_target != target)
		{
			double hashrate = 0.;
			pthread_mutex_lock(&stats_lock);
			for(int i = 0; i < opt_n_threads; i++)
				hashrate += thr_hashrates[i];
			pthread_mutex_unlock(&stats_lock);
			double difficulty = (((double)0xffffffff) / target);
			applog(LOG_INFO, "Pool set diff to %g", difficulty);
			rpc2_target = target;
		}

		if(rpc2_job_id)
		{
			free(rpc2_job_id);
		}
		rpc2_job_id = strdup(job_id);
		pthread_mutex_unlock(&rpc2_job_lock);
	}
	if(work)
	{
		if(!rpc2_blob)
		{
			applog(LOG_ERR, "Requested work before work was received");
			return false;
		}
		memcpy(work->data, rpc2_blob, rpc2_bloblen);
		memset(work->target, 0xff, sizeof(work->target));
		work->target[7] = rpc2_target;
		strncpy(work->job_id, rpc2_job_id, 128);
	}
	return true;
}

static bool work_decode(const json_t *val, struct work *work)
{
	return rpc2_job_decode(val, work);
}

bool rpc2_login_decode(const json_t *val)
{
	const char *id;
	const char *s;

	json_t *res = json_object_get(val, "result");
	if(!res)
	{
		applog(LOG_ERR, "JSON invalid result");
		return false;
	}

	json_t *tmp;
	tmp = json_object_get(res, "id");
	if(!tmp)
	{
		applog(LOG_ERR, "JSON inval id");
		return false;
	}
	id = json_string_value(tmp);
	if(!id)
	{
		applog(LOG_ERR, "JSON id is not a string");
		return false;
	}

	memcpy(&rpc2_id, id, 64);

	if(opt_debug)
		applog(LOG_DEBUG, "Auth id: %s", id);

	tmp = json_object_get(res, "status");
	if(!tmp)
	{
		applog(LOG_ERR, "JSON inval status");
		return false;
	}
	s = json_string_value(tmp);
	if(!s)
	{
		applog(LOG_ERR, "JSON status is not a string");
		return false;
	}
	if(strcmp(s, "OK"))
	{
		applog(LOG_ERR, "JSON returned status \"%s\"", s);
		return false;
	}

	return true;
}

static void share_result(int result, const char *reason)
{
	extern char *CL_GRN;
	extern char *CL_RED;
	extern char *CL_N;
	double hashrate;
	int i;

	hashrate = 0.;
	pthread_mutex_lock(&stats_lock);
	for(i = 0; i < opt_n_threads; i++)
		hashrate += thr_hashrates[i];
	result ? accepted_count++ : rejected_count++;
	pthread_mutex_unlock(&stats_lock);

	if(result)
		applog(LOG_INFO, "accepted: %lu/%lu (%.2f%%), %.2f H/s %s%s%s",
			   accepted_count, accepted_count + rejected_count,
			   100. * accepted_count / (accepted_count + rejected_count), hashrate,
			   CL_GRN, "(yay!!!)", CL_N);
	else
		applog(LOG_INFO, "accepted: %lu/%lu (%.2f%%), %.2f H/s %s%s%s",
			   accepted_count, accepted_count + rejected_count,
			   100. * accepted_count / (accepted_count + rejected_count), hashrate,
			   CL_RED, "(booooo)", CL_N);

	if(reason)
		applog(LOG_WARNING, "reject reason: %s", reason);
}

static bool submit_upstream_work(CURL *curl, struct work *work)
{
	char *str = NULL;
	json_t *val, *res, *reason;
	char s[345];
	bool rc = false;

	/* pass if the previous hash is not the current previous hash */
	if(memcmp(work->data + 1, g_work.data + 1, 32))
	{
		if(opt_debug)
			applog(LOG_DEBUG, "DEBUG: stale work detected, discarding");
		return true;
	}
	int variant = 0;
	if(have_stratum)
	{
		char *noncestr;

		noncestr = bin2hex(((const unsigned char*)work->data) + 39, 4);
		if(opt_algo != algo_old)
			variant = ((unsigned char*)work->data)[0] >= forkversion ? ((unsigned char*)work->data)[0] - forkversion + 1 : 0;
		char hash[32];
		if (!cryptonight_hash((void *)hash, (const void *)work->data, 76, variant, opt_algo)) {
			applog(LOG_ERR, "submit_upstream_work cryptonight_hash failed");
			free(str);
			return rc;
		}
		char *hashhex = bin2hex((const unsigned char *)hash, 32);
		snprintf(s, sizeof(s),
				 "{\"method\": \"submit\", \"params\": {\"id\": \"%s\", \"job_id\": \"%s\", \"nonce\": \"%s\", \"result\": \"%s\"}, \"id\":1}",
				 rpc2_id, work->job_id, noncestr, hashhex);
		free(hashhex);
		free(noncestr);

		if(unlikely(!stratum_send_line(&stratum, s)))
		{
			applog(LOG_ERR, "submit_upstream_work stratum_send_line failed");
			free(str);
			return rc;
		}
	}
	else
	{
		/* build JSON-RPC request */
		char *noncestr = bin2hex(((const unsigned char*)work->data) + 39, 4);
		if (opt_algo != algo_old)
			variant = ((unsigned char*)work->data)[0] >= forkversion ? ((unsigned char*)work->data)[0] - forkversion + 1 : 0;
		char hash[32];
		if (!cryptonight_hash((void *)hash, (const void *)work->data, 76, variant, opt_algo)) {
			applog(LOG_ERR, "submit_upstream_work cryptonight_hash failed");
			free(str);
			return rc;
		}
		char *hashhex = bin2hex((const unsigned char *)hash, 32);
		snprintf(s, sizeof(s),
				 "{\"method\": \"submit\", \"params\": {\"id\": \"%s\", \"job_id\": \"%s\", \"nonce\": \"%s\", \"result\": \"%s\"}, \"id\":1}",
				 rpc2_id, work->job_id, noncestr, hashhex);
		free(noncestr);
		free(hashhex);

		/* issue JSON-RPC request */
		val = json_rpc2_call(curl, rpc_url, rpc_userpass, s, NULL, 0);
		if(unlikely(!val))
		{
			applog(LOG_ERR, "submit_upstream_work json_rpc_call failed");
			free(str);
			return rc;
		}
		res = json_object_get(val, "result");
		json_t *status = json_object_get(res, "status");
		reason = json_object_get(res, "reject-reason");
		share_result(!strcmp(status ? json_string_value(status) : "", "OK"),
					 reason ? json_string_value(reason) : NULL);

		json_decref(val);
	}

	rc = true;
	free(str);
	return rc;
}

static const char *rpc_req =
"{\"method\": \"getwork\", \"params\": [], \"id\":0}\r\n";

static bool get_upstream_work(CURL *curl, struct work *work)
{
	json_t *val;
	bool rc;
	struct timeval tv_start, tv_end, diff;

	gettimeofday(&tv_start, NULL);

	char s[128];
	snprintf(s, 128, "{\"method\": \"getjob\", \"params\": {\"id\": \"%s\"}, \"id\":1}\r\n", rpc2_id);
	val = json_rpc2_call(curl, rpc_url, rpc_userpass, s, NULL, 0);

	gettimeofday(&tv_end, NULL);

	if(have_stratum)
	{
		if(val)
			json_decref(val);
		return true;
	}

	if(!val)
		return false;

	rc = work_decode(json_object_get(val, "result"), work);

	if(opt_debug && rc)
	{
		timeval_subtract(&diff, &tv_end, &tv_start);
		applog(LOG_DEBUG, "DEBUG: got new work in %d ms",
					 diff.tv_sec * 1000 + diff.tv_usec / 1000);
	}

	json_decref(val);

	return rc;
}

static bool rpc2_login(CURL *curl)
{
	json_t *val;
	bool rc;
	struct timeval tv_start, tv_end, diff;
	char s[345];

	snprintf(s, sizeof(s), "{\"method\": \"login\", \"params\": {\"login\": \"%s\", \"pass\": \"%s\", \"agent\": \"" USER_AGENT "\"}, \"id\": 1}", rpc_user, rpc_pass);

	gettimeofday(&tv_start, NULL);
	val = json_rpc_call(curl, rpc_url, rpc_userpass, s, false, false, NULL);
	gettimeofday(&tv_end, NULL);

	if(!val)
		return false;

	//    applog(LOG_DEBUG, "JSON value: %s", json_dumps(val, 0));

	rc = rpc2_login_decode(val);

	json_t *result = json_object_get(val, "result");

	if(!result) return rc;

	json_t *job = json_object_get(result, "job");

	if(!rpc2_job_decode(job, &g_work))
	{
		return rc;
	}

	if(opt_debug && rc)
	{
		timeval_subtract(&diff, &tv_end, &tv_start);
		applog(LOG_DEBUG, "DEBUG: authenticated in %d ms",
					 diff.tv_sec * 1000 + diff.tv_usec / 1000);
	}

	json_decref(val);

	return rc;
}

static void workio_cmd_free(struct workio_cmd *wc)
{
	if(!wc)
		return;

	switch(wc->cmd)
	{
		case WC_SUBMIT_WORK:
			free(wc->u.work);
			break;
		default: /* do nothing */
			break;
	}

	memset(wc, 0, sizeof(*wc));	/* poison */
	free(wc);
}

static bool workio_get_work(struct workio_cmd *wc, CURL *curl)
{
	struct work *ret_work;
	int failures = 0;

	ret_work = (struct work*)calloc(1, sizeof(*ret_work));
	if(!ret_work)
		return false;

	/* obtain new work from bitcoin via JSON-RPC */
	while(!get_upstream_work(curl, ret_work))
	{
		if(unlikely((opt_retries >= 0) && (++failures > opt_retries)))
		{
			applog(LOG_ERR, "json_rpc_call failed, terminating workio thread");
			free(ret_work);
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "json_rpc_call failed, retry after %d seconds",
					 opt_fail_pause);
		sleep(opt_fail_pause);
	}

	/* send work to requesting thread */
	if(!tq_push(wc->thr->q, ret_work))
		free(ret_work);

	return true;
}

static bool workio_submit_work(struct workio_cmd *wc, CURL *curl)
{
	int failures = 0;

	/* submit solution to bitcoin via JSON-RPC */
	while(!submit_upstream_work(curl, wc->u.work))
	{
		if(unlikely((opt_retries >= 0) && (++failures > opt_retries)))
		{
			applog(LOG_ERR, "...terminating workio thread");
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "...retry after %d seconds",
					 opt_fail_pause);
		sleep(opt_fail_pause);
	}

	return true;
}

static bool workio_login(CURL *curl)
{
	int failures = 0;

	/* submit solution to bitcoin via JSON-RPC */
	pthread_mutex_lock(&rpc2_login_lock);
	while(!rpc2_login(curl))
	{
		if(unlikely((opt_retries >= 0) && (++failures > opt_retries)))
		{
			applog(LOG_ERR, "...terminating workio thread");
			pthread_mutex_unlock(&rpc2_login_lock);
			return false;
		}

		/* pause, then restart work-request loop */
		applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
		sleep(opt_fail_pause);
		pthread_mutex_unlock(&rpc2_login_lock);
		pthread_mutex_lock(&rpc2_login_lock);
	}
	pthread_mutex_unlock(&rpc2_login_lock);

	return true;
}

static void *workio_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info*)userdata;
	CURL *curl;
	bool ok = true;

	curl = curl_easy_init();
	if(unlikely(!curl))
	{
		applog(LOG_ERR, "CURL initialization failed");
		return NULL;
	}

	if(!have_stratum && !opt_benchmark)
	{
		ok = workio_login(curl);
	}

	while(ok)
	{
		struct workio_cmd *wc;

		/* wait for workio_cmd sent to us, on our queue */
		wc = (struct workio_cmd *)tq_pop(mythr->q, NULL);
		if(!wc)
		{
			ok = false;
			break;
		}

		/* process workio_cmd */
		switch(wc->cmd)
		{
			case WC_GET_WORK:
				ok = workio_get_work(wc, curl);
				break;
			case WC_SUBMIT_WORK:
				ok = workio_submit_work(wc, curl);
				break;

			default:		/* should never happen */
				ok = false;
				break;
		}

		workio_cmd_free(wc);
	}

	tq_freeze(mythr->q);
	curl_easy_cleanup(curl);

	return NULL;
}

static bool get_work(struct thr_info *thr, struct work *work)
{
	struct workio_cmd *wc;
	struct work *work_heap;

	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if(!wc)
		return false;

	wc->cmd = WC_GET_WORK;
	wc->thr = thr;

	/* send work request to workio thread */
	if(!tq_push(thr_info[work_thr_id].q, wc))
	{
		workio_cmd_free(wc);
		return false;
	}

	/* wait for response, a unit of work */
	work_heap = (struct work *)tq_pop(thr->q, NULL);
	if(!work_heap)
		return false;

	/* copy returned work into storage provided by caller */
	memcpy(work, work_heap, sizeof(*work));
	free(work_heap);

	return true;
}

static bool submit_work(struct thr_info *thr, const struct work *work_in)
{
	struct workio_cmd *wc;
	/* fill out work request message */
	wc = (struct workio_cmd *)calloc(1, sizeof(*wc));
	if(!wc)
		return false;

	wc->u.work = (struct work *)malloc(sizeof(*work_in));
	if(wc->u.work == NULL)
	{
		applog(LOG_ERR, "file %s line %d: Out of memory!", __FILE__, __LINE__);
		proper_exit(1);
	}
	if(!wc->u.work) {
		workio_cmd_free(wc);
		return false;
	}

	wc->cmd = WC_SUBMIT_WORK;
	wc->thr = thr;
	memcpy(wc->u.work, work_in, sizeof(*work_in));

	/* send solution to workio thread */
	if(!tq_push(thr_info[work_thr_id].q, wc)) {
		workio_cmd_free(wc);
		return false;
	}

	return true;
}

static void stratum_gen_work(struct stratum_ctx *sctx, struct work *work)
{
	pthread_mutex_lock(&sctx->work_lock);

	memcpy(work, &sctx->work, sizeof(struct work));
	if(sctx->job.job_id) strncpy(work->job_id, sctx->job.job_id, 128);
	pthread_mutex_unlock(&sctx->work_lock);
}

static void *miner_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	int thr_id = mythr->id;
	struct work work;
	uint32_t max_nonce;
	uint32_t end_nonce;

	unsigned char *scratchbuf = NULL;
	int i;
	static int rounds = 0;
	end_nonce = 0x01000000U / opt_n_threads * (thr_id + 1) - 1;
	memset(&work, 0, sizeof(work)); // prevent work from being used uninitialized

	/* Set worker threads to nice 19 and then preferentially to SCHED_IDLE
	* and if that fails, then SCHED_BATCH. No need for this to be an
	* error if it fails */
	if(!opt_benchmark)
	{
		setpriority(PRIO_PROCESS, 0, 19);
		drop_policy();
	}

	/* Cpu affinity only makes sense if the number of threads is a multiple
	* of the number of CPUs */
	if(num_processors > 1 && opt_n_threads % num_processors == 0)
	{
		if(opt_debug)
			applog(LOG_INFO, "Binding thread %d to cpu %d",
				   thr_id, thr_id % num_processors);
		affine_to_cpu(thr_id, thr_id % num_processors);
	}

	applog(LOG_INFO, "GPU #%d: %s (%d SMX), using launch config %dx%d",
				 device_map[thr_id], device_name[thr_id], device_mpcount[thr_id], device_config[thr_id][1], device_config[thr_id][0]);

	if(device_config[thr_id][0] % device_mpcount[thr_id])
		applog(LOG_WARNING, "GPU #%d: Warning: block count %d is not a multiple of SMX count %d.",
		device_map[thr_id], device_config[thr_id][0], device_mpcount[thr_id]);

	uint32_t *const nonceptr = (uint32_t*)(((char*)work.data) + 39);

	thr_totalhashes[thr_id] = 0;

	while(1)
	{
		unsigned long hashes_done;
		struct timeval tv_start, tv_end, diff;
		double difftime;
		int64_t max64;
		int rc;

		if(have_stratum)
		{
			pthread_mutex_lock(&g_work_lock);
			if((*nonceptr) >= end_nonce &&
			   !(memcmp(work.data, g_work.data, 39) || memcmp(((uint8_t*)work.data) + 43, ((uint8_t*)g_work.data) + 43, 33)))
			{
				stratum_gen_work(&stratum, &g_work);
			}
		}
		else
		{
			/* obtain new work from internal workio thread */
			pthread_mutex_lock(&g_work_lock);
			if(!have_longpoll || time(NULL) >= g_work_time + LP_SCANTIME * 3 / 4 || *nonceptr >= end_nonce)
			{
				if(opt_benchmark)
				{
					g_work.data[0] = 8;
					memset(g_work.data + 1, 0x00, 76);
					g_work.data[20] = 0x80000000;
					memset(g_work.data + 21, 0x00, 22);
					g_work.data[31] = 0x00000280;
					memset(g_work.target, 0x00, sizeof(g_work.target));
				}
				else
					if(unlikely(!get_work(mythr, &g_work)))
					{
						applog(LOG_ERR, "work retrieval failed, exiting "
							   "mining thread %d", mythr->id);
						pthread_mutex_unlock(&g_work_lock);
						goto out;
					}
				g_work_time = time(NULL);
			}
		}
		if(memcmp(work.data, g_work.data, 39) || memcmp(((uint8_t*)work.data) + 43, ((uint8_t*)g_work.data) + 43, 33))
		{
			if(opt_debug)
				applog(LOG_DEBUG, "GPU #%d: %s, got new work", device_map[thr_id], device_name[thr_id]);
			memcpy(&work, &g_work, sizeof(struct work));
			end_nonce = 0x01000000U / opt_n_threads * (thr_id + 1) - 1 + *nonceptr;
			*nonceptr += 0x01000000U / opt_n_threads * thr_id;
		}
		else
		{
			if(opt_debug)
				applog(LOG_DEBUG, "GPU #%d: %s, continue with old work", device_map[thr_id], device_name[thr_id], *nonceptr, max_nonce);
			*nonceptr += hashes_done;
		}

		pthread_mutex_unlock(&g_work_lock);
		work_restart[thr_id].restart = 0;

		/* adjust max_nonce to meet target scan time */
		if(opt_benchmark)
			opt_scantime = 30;
		if(have_stratum)
			max64 = LP_SCANTIME;
		else
			max64 = g_work_time + (have_longpoll ? LP_SCANTIME : opt_scantime) - time(NULL);
		max64 *= (int64_t)thr_hashrates[thr_id];
		if(max64 <= 0)
			max64 = 0x200LL;
		if((int64_t)(*nonceptr) + max64 > end_nonce)
			max_nonce = end_nonce;
		else
			max_nonce = (uint32_t)(*nonceptr + max64);

		if(opt_debug)
			applog(LOG_DEBUG, "GPU #%d: %s, startnonce $%08X, endnonce $%08X", device_map[thr_id], device_name[thr_id], *nonceptr, max_nonce);

		hashes_done = 0;
		gettimeofday(&tv_start, NULL);

		uint32_t results[2];
		/* scan nonces for a proof-of-work hash */
		rc = scanhash_cryptonight(thr_id, work.data, work.target,	max_nonce, &hashes_done, results);

		thr_totalhashes[thr_id] += hashes_done;

		/* record scanhash elapsed time */
		gettimeofday(&tv_end, NULL);
		timeval_subtract(&diff, &tv_end, &tv_start);
		difftime = diff.tv_sec + 1e-6 * diff.tv_usec;
		if(difftime > 0)
		{
			pthread_mutex_lock(&stats_lock);
			thr_hashrates[thr_id] = hashes_done / difftime;
			pthread_mutex_unlock(&stats_lock);
		}

		timeval_subtract(&diff, &tv_end, &stats_start);
		difftime = diff.tv_sec + 1e-6 * diff.tv_usec;

		if(!opt_quiet)
			applog(LOG_INFO, "GPU #%d: %s, %.2f H/s (%.2f H/s avg)", device_map[thr_id], device_name[thr_id], thr_hashrates[thr_id], thr_totalhashes[thr_id] / difftime);

		if(opt_benchmark && thr_id == opt_n_threads - 1)
		{
			double hashrate = 0.;
			for(i = 0; i < opt_n_threads && thr_hashrates[i]; i++)
				hashrate += thr_hashrates[i];
			if(i == opt_n_threads)
			{
					applog(LOG_INFO, "Total: %.2f H/s", hashrate);
			}
		}

		/* if nonce found, submit work */
		if(rc && !opt_benchmark)
		{
			uint32_t backup = *nonceptr;
			*nonceptr = results[0];
			submit_work(mythr, &work);
			if(rc > 1)
			{
				*nonceptr = results[1];
				submit_work(mythr, &work);
			}
			*nonceptr = backup;
		}
	}

out:
	tq_freeze(mythr->q);

	return NULL;
}

static void restart_threads(void)
{
	int i;

	for(i = 0; i < opt_n_threads; i++)
		work_restart[i].restart = 1;
}

static void *longpoll_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	CURL *curl = NULL;
	char *copy_start = NULL, *hdr_path = NULL, *lp_url = NULL;
	bool need_slash = false;

	curl = curl_easy_init();
	if(unlikely(!curl))
	{
		applog(LOG_ERR, "CURL initialization failed");
		goto out;
	}

start:
	hdr_path = (char*)tq_pop(mythr->q, NULL);
	if(!hdr_path)
		goto out;

	/* full URL */
	if(strstr(hdr_path, "://"))
	{
		lp_url = hdr_path;
		hdr_path = NULL;
	}

	/* absolute path, on current server */
	else
	{
		copy_start = (*hdr_path == '/') ? (hdr_path + 1) : hdr_path;
		if(rpc_url[strlen(rpc_url) - 1] != '/')
			need_slash = true;

		lp_url = (char*)malloc(strlen(rpc_url) + strlen(copy_start) + 2);
		if(!lp_url)
			goto out;

		sprintf(lp_url, "%s%s%s", rpc_url, need_slash ? "/" : "", copy_start);
	}

	applog(LOG_INFO, "Long-polling activated for %s", lp_url);

	while(1)
	{
		json_t *val, *soval;
		int err;

		pthread_mutex_lock(&rpc2_login_lock);
		if(!strcmp(rpc2_id, ""))
		{
			sleep(1);
			continue;
		}
		char s[128];
		snprintf(s, 128, "{\"method\": \"getjob\", \"params\": {\"id\": \"%s\"}, \"id\":1}\r\n", rpc2_id);
		pthread_mutex_unlock(&rpc2_login_lock);
		val = json_rpc2_call(curl, rpc_url, rpc_userpass, s, &err, JSON_RPC_LONGPOLL);

		if(have_stratum)
		{
			if(val)
				json_decref(val);
			goto out;
		}
		if(likely(val))
		{

			soval = json_object_get(json_object_get(val, "result"), "submitold");
			submit_old = soval ? json_is_true(soval) : false;
			pthread_mutex_lock(&g_work_lock);
			char *start_job_id = strdup(g_work.job_id);
			if(work_decode(json_object_get(val, "result"), &g_work))
			{
				if(strcmp(start_job_id, g_work.job_id))
				{
					if(!opt_quiet) applog(LOG_INFO, "LONGPOLL detected new block");
					if(opt_debug)
						applog(LOG_DEBUG, "DEBUG: got new work");
					time(&g_work_time);
					restart_threads();
				}
			}
			pthread_mutex_unlock(&g_work_lock);
			json_decref(val);
		}
		else
		{
			pthread_mutex_lock(&g_work_lock);
			g_work_time -= LP_SCANTIME;
			pthread_mutex_unlock(&g_work_lock);
			if(err == CURLE_OPERATION_TIMEDOUT)
			{
				restart_threads();
			}
			else
			{
				have_longpoll = false;
				restart_threads();
				free(hdr_path);
				free(lp_url);
				lp_url = NULL;
				sleep(opt_fail_pause);
				goto start;
			}
		}
	}

out:
	free(hdr_path);
	free(lp_url);
	tq_freeze(mythr->q);
	if(curl)
		curl_easy_cleanup(curl);

	return NULL;
}

static bool stratum_handle_response(char *buf)
{
	json_t *val = NULL, *err_val = NULL, *res_val = NULL, *id_val = NULL;
	json_t *status = NULL;
	json_error_t err;
	char *s = NULL;
	bool ret = false;
	bool valid = false;

	val = JSON_LOADS(buf, &err);
	if(!val)
	{
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");
	id_val = json_object_get(val, "id");

	if(!id_val || json_is_null(id_val) || (!res_val && !err_val) )
		goto out;

	status = json_object_get(res_val, "status");
	if(status != NULL)
	{
		s = (char*)json_string_value(status);
		if(strcmp(s, "KEEPALIVED") == 0)
			goto out;
		valid = !strcmp(s, "OK") && json_is_null(err_val);
	}
	else
	{
		valid = json_is_null(err_val);
	}

	if(err_val)
	{
		share_result(valid, json_string_value(json_object_get(err_val, "message")));
	}
	else
		share_result(valid, NULL);

	ret = true;
out:
	if(val)
		json_decref(val);

	return ret;
}

static void *stratum_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;
	char *s;

	stratum.url = (char*)tq_pop(mythr->q, NULL);
	if(!stratum.url)
		return NULL;
	applog(LOG_INFO, "Starting Stratum on %s", stratum.url);

	while(1)
	{
		int failures = 0;

		while(!stratum.curl)
		{
			pthread_mutex_lock(&g_work_lock);
			g_work_time = 0;
			pthread_mutex_unlock(&g_work_lock);
			restart_threads();

			if(!stratum_connect(&stratum, stratum.url) ||
				 !stratum_authorize(&stratum, rpc_user, rpc_pass))
			{
				stratum_disconnect(&stratum);
				if(opt_retries >= 0 && ++failures > opt_retries)
				{
					applog(LOG_ERR, "...terminating workio thread");
					tq_push(thr_info[work_thr_id].q, NULL);
					return NULL;
				}
				applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
				sleep(opt_fail_pause);
			}
		}

		if(stratum.work.job_id
		   && (!g_work_time
			   || strcmp(stratum.work.job_id, g_work.job_id)))
		{
			pthread_mutex_lock(&g_work_lock);
			stratum_gen_work(&stratum, &g_work);
			time(&g_work_time);
			pthread_mutex_unlock(&g_work_lock);
			applog(LOG_INFO, "Stratum detected new block");
			restart_threads();
		}

		// Should we send a keepalive?
		if( opt_keepalive && !stratum_socket_full(&stratum, 90))
		{
			applog(LOG_INFO, "Keepalive send...");
			stratum_keepalived(&stratum,rpc2_id);
		}

		if(!stratum_socket_full(&stratum, 120))
		{
			applog(LOG_ERR, "Stratum connection timed out");
			s = NULL;
		}
		else
			s = stratum_recv_line(&stratum);
		if(!s)
		{
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "Stratum connection interrupted");
			continue;
		}
		if(!stratum_handle_method(&stratum, s))
			stratum_handle_response(s);
		free(s);
	}
	return NULL;
}

static void show_version_and_exit(void)
{
	printf("%s\n%s\n", PACKAGE_STRING, curl_version());
	exit(0);
}

static void show_usage_and_exit(int status)
{
	if(status)
		printf("Try `" PROGRAM_NAME " --help' for more information.\n");
	else
		printf(usage);
	exit(status);
}

void parse_device_config(int device, char *config, int *blocks, int *threads)
{
	char *p;
	int tmp_blocks, tmp_threads;

	if(config == NULL)
		return;

	p = strtok(config, "x");
	if(!p)
		return;

	tmp_threads = atoi(p);
	if(tmp_threads < 4 || tmp_threads > 1024)
		return;

	p = strtok(NULL, "x");
	if(!p)
		return;

	tmp_blocks = atoi(p);
	if(tmp_blocks < 1)
		return;

	*blocks = tmp_blocks;
	*threads = tmp_threads;
	return;
}

static void parse_arg(int key, char *arg)
{
	char *p;
	int v, i;
	double d;

	switch(key)
	{
		case 'a':
			for (i = 0; i < ARRAY_SIZE(algo_names); i++)
			{
				if (algo_names[i] && !strcasecmp(arg, algo_names[i]))
				{
					opt_algo = (algo_t)i;
					break;
				}
			}
			if (opt_algo == algo_monero)
				forkversion = 7;
			if (opt_algo == algo_graft)
				forkversion = 8;
			if (opt_algo == algo_stellite)
				forkversion = 3;
			if(opt_algo == algo_intense)
				forkversion = 4;
			if (opt_algo == algo_electroneum)
				forkversion = 6;
			if (opt_algo == algo_sumokoin)
			{
					MEMORY = 1 << 22;
					ITER = 1 << 19;
			}
			break;
		case 'B':
			opt_background = true;
			break;
		case 'c': {
			json_error_t err;
			if(opt_config)
				json_decref(opt_config);
#if JANSSON_VERSION_HEX >= 0x020000
			opt_config = json_load_file(arg, 0, &err);
#else
			opt_config = json_load_file(arg, &err);
#endif
			if(!json_is_object(opt_config))
			{
				applog(LOG_ERR, "JSON decode of %s failed", arg);
				exit(1);
			}
			break;
		}
		case 'k':
			opt_keepalive = true ;
			applog(LOG_INFO, "Keepalive activated");
			break;
		case 'q':
			opt_quiet = true;
			break;
		case 'D':
			opt_debug = true;
			break;
		case 'p':
			free(rpc_pass);
			rpc_pass = strdup(arg);
			break;
		case 'P':
			opt_protocol = true;
			break;
		case 'r':
			v = atoi(arg);
			if(v < -1 || v > 9999)	/* sanity check */
				show_usage_and_exit(1);
			opt_retries = v;
			break;
		case 'R':
			v = atoi(arg);
			if(v < 1 || v > 9999)	/* sanity check */
				show_usage_and_exit(1);
			opt_fail_pause = v;
			break;
		case 's':
			v = atoi(arg);
			if(v < 1 || v > 9999)	/* sanity check */
				show_usage_and_exit(1);
			opt_scantime = v;
			break;
		case 'T':
			v = atoi(arg);
			if(v < 1 || v > 99999)	/* sanity check */
				show_usage_and_exit(1);
			opt_timeout = v;
			break;
		case 't':
			v = atoi(arg);
			if(v < 1 || v > 9999)	/* sanity check */
				show_usage_and_exit(1);
			opt_n_threads = v;
			break;
		case 'v':
			break;
		case 'u':
			free(rpc_user);
			rpc_user = strdup(arg);
			break;
		case 'o':			/* --url */
			p = strstr(arg, "://");
			if(p)
			{
				if(strncasecmp(arg, "http://", 7)
					 && strncasecmp(arg, "https://", 8)
					 && strncasecmp(arg, "stratum+tcp://", 14))
					 show_usage_and_exit(1);
				free(rpc_url);
				rpc_url = strdup(arg);
			}
			else
			{
				if(!strlen(arg) || *arg == '/')
					show_usage_and_exit(1);
				free(rpc_url);
				rpc_url = (char*)malloc(strlen(arg) + 8);
				if (rpc_url == NULL)
				{
					applog(LOG_ERR, "file %s line %d: Out of memory!", __FILE__, __LINE__);
					exit(1);
				}
				sprintf(rpc_url, "http://%s", arg);
			}
			p = strrchr(rpc_url, '@');
			if(p)
			{
				char *sp, *ap;
				*p = '\0';
				ap = strstr(rpc_url, "://") + 3;
				sp = strchr(ap, ':');
				if(sp)
				{
					free(rpc_userpass);
					rpc_userpass = strdup(ap);
					free(rpc_user);
					rpc_user = (char*)calloc(sp - ap + 1, 1);
					strncpy(rpc_user, ap, sp - ap);
					free(rpc_pass);
					rpc_pass = strdup(sp + 1);
				}
				else
				{
					free(rpc_user);
					rpc_user = strdup(ap);
				}
				memmove(ap, p + 1, strlen(p + 1) + 1);
			}
			have_stratum = !opt_benchmark && !strncasecmp(rpc_url, "stratum", 7);
			break;
		case 'O':			/* --userpass */
			p = strchr(arg, ':');
			if(!p)
				show_usage_and_exit(1);
			free(rpc_userpass);
			rpc_userpass = strdup(arg);
			free(rpc_user);
			rpc_user = (char*)calloc(p - arg + 1, 1);
			strncpy(rpc_user, arg, p - arg);
			free(rpc_pass);
			rpc_pass = strdup(p + 1);
			break;
		case 'x':			/* --proxy */
			if(!strncasecmp(arg, "socks4://", 9))
				opt_proxy_type = CURLPROXY_SOCKS4;
			else if(!strncasecmp(arg, "socks5://", 9))
				opt_proxy_type = CURLPROXY_SOCKS5;
#if LIBCURL_VERSION_NUM >= 0x071200
			else if(!strncasecmp(arg, "socks4a://", 10))
				opt_proxy_type = CURLPROXY_SOCKS4A;
			else if(!strncasecmp(arg, "socks5h://", 10))
				opt_proxy_type = CURLPROXY_SOCKS5_HOSTNAME;
#endif
			else
				opt_proxy_type = CURLPROXY_HTTP;
			free(opt_proxy);
			opt_proxy = strdup(arg);
			break;
		case 1001:
			free(opt_cert);
			opt_cert = strdup(arg);
			break;
		case 1005:
			opt_benchmark = true;
			want_longpoll = false;
			want_stratum = false;
			have_stratum = false;
			break;
		case 1003:
			want_longpoll = false;
			break;
		case 1007:
			want_stratum = false;
			break;
		case 'S':
			use_syslog = true;
			break;
		case 'd': // CB
		{
			int i;
			bool gpu[32] = {false};
			char * pch = strtok(arg, ",");
			opt_n_threads = 0;
			while(pch != NULL)
			{
				if(pch[0] >= '0' && pch[0] <= '9')
				{
					i = atoi(pch);
					if(i < num_processors && gpu[i] == false && opt_n_threads < MAX_GPU)
					{
						gpu[i] = true;
						device_map[opt_n_threads++] = i;
					}
					else
					{
						if(opt_n_threads >= MAX_GPU)
						{
							applog(LOG_ERR, "Only %d gpus are supported in this ccminer build.", MAX_GPU);
							proper_exit(1);
						}
						if(gpu[i] == true)
						{
							applog(LOG_ERR, "Selected gpu #%d more than once in -d option. This is not supported.", i);
							proper_exit(1);
						}
						applog(LOG_ERR, "Non-existant CUDA device #%d specified in -d option", i);
						proper_exit(1);
					}
				}
				else
				{
					int device = cuda_finddevice(pch);
					if(device >= 0 && device < num_processors)
						device_map[opt_n_threads++] = device;
					else
					{
						applog(LOG_ERR, "Non-existant CUDA device '%s' specified in -d option", pch);
						exit(1);
					}
				}
				pch = strtok(NULL, ",");
			}
		}
		break;
		case 'f': // CH - Divisor for Difficulty
			d = atof(arg);
			if(d == 0)	/* sanity check */
				show_usage_and_exit(1);
			opt_difficulty = d;
			break;
		case 'l':			/* cryptonight launch config */
		{
			char *tmp_config[MAX_GPU];
			int tmp_blocks = opt_cn_blocks, tmp_threads = opt_cn_threads;
			for(i = 0; i < MAX_GPU; i++)
				tmp_config[i] = NULL;
			p = strtok(arg, ",");
			if(p == NULL) show_usage_and_exit(1);
			i = 0;
			while(p != NULL && i < MAX_GPU)
			{
				tmp_config[i++] = strdup(p);
				p = strtok(NULL, ",");
			}
			while(i < 8)
			{
				tmp_config[i] = strdup(tmp_config[i - 1]);
				i++;
			}

			for(i = 0; i < MAX_GPU; i++)
			{
				parse_device_config(i, tmp_config[i], &tmp_blocks, &tmp_threads);
				device_config[i][0] = tmp_blocks;
				device_config[i][1] = tmp_threads;
			}
		}
		break;
		case 1008:
		{
			p = strtok(arg, ",");
			if(p == NULL) show_usage_and_exit(1);
			int last;
			i = 0;
			while(p != NULL && i < MAX_GPU)
			{
				device_bfactor[i++] = last = atoi(p);
				if(last < 0 || last > 10)
				{
					applog(LOG_ERR, "Valid range for --bfactor is 0-10");
					exit(1);
				}
				p = strtok(NULL, ",");
			}
			while(i < MAX_GPU)
			{
				device_bfactor[i++] = last;
			}
		}
		break;
		case 1009:
			p = strtok(arg, ",");
			if(p == NULL) show_usage_and_exit(1);
			int last;
			i = 0;
			while(p != NULL && i < MAX_GPU)
			{
				device_bsleep[i++] = last = atoi(p);
				if(last < 0 || last > 1000000)
				{
					applog(LOG_ERR, "Valid range for --bsleep is 0-1000000");
					exit(1);
				}
				p = strtok(NULL, ",");
			}
			while(i < MAX_GPU)
			{
				device_bsleep[i++] = last;
			}
			break;
		case 1010:
			opt_colors = true;
#if defined WIN32 && defined ENABLE_VIRTUAL_TERMINAL_PROCESSING
			handl = GetStdHandle(STD_ERROR_HANDLE);
			SetConsoleMode(handl, ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT);
#endif
			break;

		case 'V':
			show_version_and_exit();
		case 'h':
			show_usage_and_exit(0);
		default:
			show_usage_and_exit(1);
	}
}

static void parse_config(void)
{
	int i;
	json_t *val;

	if(!json_is_object(opt_config))
		return;

	for(i = 0; i < ARRAY_SIZE(options); i++)
	{
		if(!options[i].name)
			break;
		if(!strcmp(options[i].name, "config"))
			continue;

		val = json_object_get(opt_config, options[i].name);
		if(!val)
			continue;

		if(options[i].has_arg && json_is_string(val))
		{
			char *s = strdup(json_string_value(val));
			if(!s)
				break;
			parse_arg(options[i].val, s);
			free(s);
		}
		else if(!options[i].has_arg && json_is_true(val))
			parse_arg(options[i].val, "");
		else
			applog(LOG_ERR, "JSON option %s invalid",
			options[i].name);
	}
}

static void parse_cmdline(int argc, char *argv[])
{
	int key;

	while(1)
	{
#if HAVE_GETOPT_LONG
		key = getopt_long(argc, argv, short_options, options, NULL);
#else
		key = getopt(argc, argv, short_options);
#endif
		if(key < 0)
			break;

		parse_arg(key, optarg);
	}
	if(optind < argc)
	{
		printf("%s: unsupported non-option argument '%s'\n",
						argv[0], argv[optind]);
		show_usage_and_exit(1);
	}
	parse_config();
}

#ifndef WIN32
static void signal_handler(int sig)
{
	switch(sig)
	{
		case SIGHUP:
			applog(LOG_INFO, "SIGHUP received");
			break;
		case SIGINT:
			applog(LOG_INFO, "SIGINT received, exiting");
			exit(0);
			break;
		case SIGTERM:
			applog(LOG_INFO, "SIGTERM received, exiting");
			exit(0);
			break;
	}
}
#else
BOOL WINAPI ConsoleHandler(DWORD dwType)
{
	switch(dwType)
	{
		case CTRL_C_EVENT:
			applog(LOG_INFO, "CTRL_C_EVENT received, exiting");
			proper_exit(EXIT_SUCCESS);
			break;
		case CTRL_BREAK_EVENT:
			applog(LOG_INFO, "CTRL_BREAK_EVENT received, exiting");
			proper_exit(EXIT_SUCCESS);
			break;
		default:
			return false;
	}
	return true;
}
#endif

static int msver(void)
{
	int version;
#ifdef _MSC_VER
	switch(_MSC_VER/100)
	{
		case 15: version = 2008; break;
		case 16: version = 2010; break;
		case 17: version = 2012; break;
		case 18: version = 2013; break;
		case 19: version = 2015; break;
		default: version = _MSC_VER / 100;
	}
	if(_MSC_VER > 1900)
		version = 2017;
#else
	version = 0;
#endif
	return version;
}

#define PROGRAM_VERSION "3.06"
int main(int argc, char *argv[])
{
	struct thr_info *thr;
	int i;
	/*
#ifdef WIN32
	SYSTEM_INFO sysinfo;
#endif
	*/
#if defined _WIN64 || defined _LP64
	int bits = 64;
#else
	int bits = 32;
#endif
	printf("    *** ccminer-cryptonight %s (%d bit) for nVidia GPUs by tsiv and KlausT \n", PROGRAM_VERSION, bits);
#ifdef _MSC_VER
	printf("    *** Built with Visual Studio %d ", msver());
#else
#ifdef __clang__
	printf("    *** Built with Clang %s ", __clang_version__);
#else
#ifdef __GNUC__
	printf("    *** Built with GCC %d.%d ", __GNUC__, __GNUC_MINOR__);
#else
	printf("    *** Built with an unusual compiler ");
#endif
#endif
#endif
	printf("using the Nvidia CUDA Toolkit %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
	printf(" tsiv's BTC donation address:   1JHDKp59t1RhHFXsTw2UQpR3F9BBz3R3cs\n");
	printf(" KlausT's BTC donation address: 1QHH2dibyYL5iyMDk3UN4PVvFVtrWD8QKp\n");
	printf(" for more donation addresses please read the README.txt\n");
	printf("-----------------------------------------------------------------\n");

	rpc_user = strdup("");
	rpc_pass = strdup("");

	pthread_mutex_init(&applog_lock, NULL);
	num_processors = cuda_num_devices();
	if(num_processors == 0)
	{
		applog(LOG_ERR, "No CUDA devices found! terminating.");
		exit(EXIT_FAILURE);
	}
	else
		applog(LOG_INFO, "%d CUDA devices detected", num_processors);

	if(!opt_n_threads)
		opt_n_threads = num_processors;

	for(i = 0; i < MAX_GPU; i++)
	{
		device_map[i] = i;
		device_bfactor[i] = default_bfactor;
		device_bsleep[i] = default_bsleep;
		device_config[i][0] = opt_cn_blocks;
	}

	/* parse command line */
	parse_cmdline(argc, argv);
	color_init();

	cuda_deviceinfo(opt_n_threads);
	cuda_set_device_config(opt_n_threads);

	if(!opt_benchmark && !rpc_url)
	{
		printf("%s: no URL supplied\n", argv[0]);
		show_usage_and_exit(1);
	}

	if(!rpc_userpass)
	{
		rpc_userpass = (char*)malloc(strlen(rpc_user) + strlen(rpc_pass) + 2);
		if(!rpc_userpass)
			exit(EXIT_FAILURE);
		sprintf(rpc_userpass, "%s:%s", rpc_user, rpc_pass);
	}

	pthread_mutex_init(&stats_lock, NULL);
	pthread_mutex_init(&g_work_lock, NULL);
	pthread_mutex_init(&rpc2_job_lock, NULL);
	pthread_mutex_init(&stratum.sock_lock, NULL);
	pthread_mutex_init(&stratum.work_lock, NULL);

	if(curl_global_init(CURL_GLOBAL_ALL))
	{
		applog(LOG_ERR, "CURL initialization failed");
		exit(EXIT_FAILURE);
	}

#ifndef WIN32
	if(opt_background)
	{
		i = fork();
		if(i < 0) exit(1);
		if(i > 0) exit(0);
		i = setsid();
		if(i < 0)
			applog(LOG_ERR, "setsid() failed (errno = %d)", errno);
		i = chdir("/");
		if(i < 0)
			applog(LOG_ERR, "chdir() failed (errno = %d)", errno);
		signal(SIGHUP, signal_handler);
		signal(SIGINT, signal_handler);
		signal(SIGTERM, signal_handler);
	}
	signal(SIGINT, signal_handler);
#else
	if(opt_background)
		applog(LOG_WARNING, "option -B is not supported under Windows");
	SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE);
	SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
#endif

#ifdef HAVE_SYSLOG_H
	if(use_syslog)
		openlog("cpuminer", LOG_PID, LOG_USER);
#endif

	work_restart = (struct work_restart *)calloc(opt_n_threads, sizeof(*work_restart));
	if(!work_restart)
		proper_exit(EXIT_FAILURE);

	thr_info = (struct thr_info *)calloc(opt_n_threads + 3, sizeof(*thr));
	if(!thr_info)
		proper_exit(EXIT_FAILURE);

	thr_hashrates = (double *)calloc(opt_n_threads, sizeof(double));
	if(!thr_hashrates)
		proper_exit(EXIT_FAILURE);

	thr_totalhashes = (uint64_t *)calloc(opt_n_threads, sizeof(uint64_t));
	if(!thr_hashrates)
		proper_exit(EXIT_FAILURE);

	/* init workio thread info */
	work_thr_id = opt_n_threads;
	thr = &thr_info[work_thr_id];
	thr->id = work_thr_id;
	thr->q = tq_new();
	if(!thr->q)
		proper_exit(EXIT_FAILURE);

	/* start work I/O thread */
	if(pthread_create(&thr->pth, NULL, workio_thread, thr))
	{
		applog(LOG_ERR, "workio thread create failed");
		proper_exit(EXIT_FAILURE);
	}

	if(want_longpoll && !have_stratum)
	{
		/* init longpoll thread info */
		longpoll_thr_id = opt_n_threads + 1;
		thr = &thr_info[longpoll_thr_id];
		thr->id = longpoll_thr_id;
		thr->q = tq_new();
		if(!thr->q)
			proper_exit(EXIT_FAILURE);

		/* start longpoll thread */
		if(unlikely(pthread_create(&thr->pth, NULL, longpoll_thread, thr)))
		{
			applog(LOG_ERR, "longpoll thread create failed");
			proper_exit(EXIT_FAILURE);
		}
	}
	if(want_stratum)
	{
		/* init stratum thread info */
		stratum_thr_id = opt_n_threads + 2;
		thr = &thr_info[stratum_thr_id];
		thr->id = stratum_thr_id;
		thr->q = tq_new();
		if(!thr->q)
			proper_exit(EXIT_FAILURE);

		/* start stratum thread */
		if(unlikely(pthread_create(&thr->pth, NULL, stratum_thread, thr)))
		{
			applog(LOG_ERR, "stratum thread create failed");
			proper_exit(EXIT_FAILURE);
		}

		if(have_stratum)
			tq_push(thr_info[stratum_thr_id].q, strdup(rpc_url));
	}
	gettimeofday(&stats_start, NULL);

	/* start mining threads */
	for(i = 0; i < opt_n_threads; i++)
	{
		thr = &thr_info[i];

		thr->id = i;
		thr->q = tq_new();
		if(!thr->q)
			proper_exit(EXIT_FAILURE);

		if(unlikely(pthread_create(&thr->pth, NULL, miner_thread, thr)))
		{
			applog(LOG_ERR, "thread %d create failed", i);
			proper_exit(EXIT_FAILURE);
		}
	}

	applog(LOG_INFO, "%d miner threads started", opt_n_threads);

#ifdef WIN32
	timeBeginPeriod(1); // enable high timer precision (similar to Google Chrome Trick)
#endif

	/* main loop - simply wait for workio thread to exit */
	pthread_join(thr_info[work_thr_id].pth, NULL);

#ifdef WIN32
	timeEndPeriod(1); // be nice and forego high timer precision
#endif

	applog(LOG_INFO, "workio thread dead, exiting.");

	proper_exit(EXIT_SUCCESS);
}

