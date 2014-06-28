
#define N_COLS          4
#define WPOLY           0x011b

static __constant__ uint32_t d_t_fn[4][256];
    
#define sb_data(w) {\
    w(0x63), w(0x7c), w(0x77), w(0x7b), w(0xf2), w(0x6b), w(0x6f), w(0xc5),\
    w(0x30), w(0x01), w(0x67), w(0x2b), w(0xfe), w(0xd7), w(0xab), w(0x76),\
    w(0xca), w(0x82), w(0xc9), w(0x7d), w(0xfa), w(0x59), w(0x47), w(0xf0),\
    w(0xad), w(0xd4), w(0xa2), w(0xaf), w(0x9c), w(0xa4), w(0x72), w(0xc0),\
    w(0xb7), w(0xfd), w(0x93), w(0x26), w(0x36), w(0x3f), w(0xf7), w(0xcc),\
    w(0x34), w(0xa5), w(0xe5), w(0xf1), w(0x71), w(0xd8), w(0x31), w(0x15),\
    w(0x04), w(0xc7), w(0x23), w(0xc3), w(0x18), w(0x96), w(0x05), w(0x9a),\
    w(0x07), w(0x12), w(0x80), w(0xe2), w(0xeb), w(0x27), w(0xb2), w(0x75),\
    w(0x09), w(0x83), w(0x2c), w(0x1a), w(0x1b), w(0x6e), w(0x5a), w(0xa0),\
    w(0x52), w(0x3b), w(0xd6), w(0xb3), w(0x29), w(0xe3), w(0x2f), w(0x84),\
    w(0x53), w(0xd1), w(0x00), w(0xed), w(0x20), w(0xfc), w(0xb1), w(0x5b),\
    w(0x6a), w(0xcb), w(0xbe), w(0x39), w(0x4a), w(0x4c), w(0x58), w(0xcf),\
    w(0xd0), w(0xef), w(0xaa), w(0xfb), w(0x43), w(0x4d), w(0x33), w(0x85),\
    w(0x45), w(0xf9), w(0x02), w(0x7f), w(0x50), w(0x3c), w(0x9f), w(0xa8),\
    w(0x51), w(0xa3), w(0x40), w(0x8f), w(0x92), w(0x9d), w(0x38), w(0xf5),\
    w(0xbc), w(0xb6), w(0xda), w(0x21), w(0x10), w(0xff), w(0xf3), w(0xd2),\
    w(0xcd), w(0x0c), w(0x13), w(0xec), w(0x5f), w(0x97), w(0x44), w(0x17),\
    w(0xc4), w(0xa7), w(0x7e), w(0x3d), w(0x64), w(0x5d), w(0x19), w(0x73),\
    w(0x60), w(0x81), w(0x4f), w(0xdc), w(0x22), w(0x2a), w(0x90), w(0x88),\
    w(0x46), w(0xee), w(0xb8), w(0x14), w(0xde), w(0x5e), w(0x0b), w(0xdb),\
    w(0xe0), w(0x32), w(0x3a), w(0x0a), w(0x49), w(0x06), w(0x24), w(0x5c),\
    w(0xc2), w(0xd3), w(0xac), w(0x62), w(0x91), w(0x95), w(0xe4), w(0x79),\
    w(0xe7), w(0xc8), w(0x37), w(0x6d), w(0x8d), w(0xd5), w(0x4e), w(0xa9),\
    w(0x6c), w(0x56), w(0xf4), w(0xea), w(0x65), w(0x7a), w(0xae), w(0x08),\
    w(0xba), w(0x78), w(0x25), w(0x2e), w(0x1c), w(0xa6), w(0xb4), w(0xc6),\
    w(0xe8), w(0xdd), w(0x74), w(0x1f), w(0x4b), w(0xbd), w(0x8b), w(0x8a),\
    w(0x70), w(0x3e), w(0xb5), w(0x66), w(0x48), w(0x03), w(0xf6), w(0x0e),\
    w(0x61), w(0x35), w(0x57), w(0xb9), w(0x86), w(0xc1), w(0x1d), w(0x9e),\
    w(0xe1), w(0xf8), w(0x98), w(0x11), w(0x69), w(0xd9), w(0x8e), w(0x94),\
    w(0x9b), w(0x1e), w(0x87), w(0xe9), w(0xce), w(0x55), w(0x28), w(0xdf),\
    w(0x8c), w(0xa1), w(0x89), w(0x0d), w(0xbf), w(0xe6), w(0x42), w(0x68),\
    w(0x41), w(0x99), w(0x2d), w(0x0f), w(0xb0), w(0x54), w(0xbb), w(0x16) }

#define bytes2word(b0, b1, b2, b3) (((uint32_t)(b3) << 24) | \
    ((uint32_t)(b2) << 16) | ((uint32_t)(b1) << 8) | (b0))

#define f2(x)   ((x<<1) ^ (((x>>7) & 1) * WPOLY))
#define f3(x)   (f2(x) ^ x)

#define u0(p)   bytes2word(f2(p), p, p, f3(p))
#define u1(p)   bytes2word(f3(p), f2(p), p, p)
#define u2(p)   bytes2word(p, f3(p), f2(p), p)
#define u3(p)   bytes2word(p, p, f3(p), f2(p))

#define t_dec(m,n) t_##m##n

#define d_4(t,n,b,e,f,g,h) const t n[4][256] = { b(e), b(f), b(g), b(h) }

d_4(uint32_t, t_dec(f,n), sb_data, u0, u1, u2, u3);

#define t_fn0(x) (*((uint32_t*)sharedMemory + (    (x))))
#define t_fn1(x) (*((uint32_t*)sharedMemory + (256+(x))))
#define t_fn2(x) (*((uint32_t*)sharedMemory + (512+(x))))
#define t_fn3(x) (*((uint32_t*)sharedMemory + (768+(x))))


#define round(dummy,y,x,k) \
    y[0] = (k)[0]  ^ (t_fn0(x[0] & 0xff) ^ t_fn1((x[1] >> 8) & 0xff) ^ t_fn2((x[2] >> 16) & 0xff) ^ t_fn3((x[3] >> 24) & 0xff)); \
    y[1] = (k)[1]  ^ (t_fn0(x[1] & 0xff) ^ t_fn1((x[2] >> 8) & 0xff) ^ t_fn2((x[3] >> 16) & 0xff) ^ t_fn3((x[0] >> 24) & 0xff)); \
    y[2] = (k)[2]  ^ (t_fn0(x[2] & 0xff) ^ t_fn1((x[3] >> 8) & 0xff) ^ t_fn2((x[0] >> 16) & 0xff) ^ t_fn3((x[1] >> 24) & 0xff)); \
    y[3] = (k)[3]  ^ (t_fn0(x[3] & 0xff) ^ t_fn1((x[0] >> 8) & 0xff) ^ t_fn2((x[1] >> 16) & 0xff) ^ t_fn3((x[2] >> 24) & 0xff));

__device__ __forceinline__ void cn_aes_single_round(uint32_t *sharedMemory, const uint8_t *in, uint32_t *out, uint32_t *expandedKey)
{
    round(sharedMemory, ((uint32_t*) out), ((uint32_t*) in), ((uint32_t*) expandedKey));
}

__device__ __forceinline__ void cn_aes_pseudo_round_mut(uint32_t *sharedMemory, uint32_t *val, uint32_t *expandedKey)
{
    uint32_t b1[4];
    round(sharedMemory, b1, ((uint32_t*) val), ((const uint32_t *) expandedKey));
    round(sharedMemory, ((uint32_t*) val), b1, ((const uint32_t *) expandedKey) + 1 * N_COLS);
    round(sharedMemory, b1, ((uint32_t*) val), ((const uint32_t *) expandedKey) + 2 * N_COLS);
    round(sharedMemory, ((uint32_t*) val), b1, ((const uint32_t *) expandedKey) + 3 * N_COLS);
    round(sharedMemory, b1, ((uint32_t*) val), ((const uint32_t *) expandedKey) + 4 * N_COLS);
    round(sharedMemory, ((uint32_t*) val), b1, ((const uint32_t *) expandedKey) + 5 * N_COLS);
    round(sharedMemory, b1, ((uint32_t*) val), ((const uint32_t *) expandedKey) + 6 * N_COLS);
    round(sharedMemory, ((uint32_t*) val), b1, ((const uint32_t *) expandedKey) + 7 * N_COLS);
    round(sharedMemory, b1, ((uint32_t*) val), ((const uint32_t *) expandedKey) + 8 * N_COLS);
    round(sharedMemory, ((uint32_t*) val), b1, ((const uint32_t *) expandedKey) + 9 * N_COLS);
}

static void cn_aes_cpu_init()
{
	cudaMemcpyToSymbol( d_t_fn, t_fn, sizeof(t_fn), 0, cudaMemcpyHostToDevice);
}

__device__ __forceinline__ void cn_aes_gpu_init(uint32_t *sharedMemory)
{
	int x;

    // TODO: actually spread the shared memory loading between more or less all threads
    // instead of just using the first four to do it...
    if(threadIdx.x < 4) {
		
        for( x = 0; x < 256; x++ ) {

            sharedMemory[threadIdx.x*256+x] = d_t_fn[threadIdx.x][x];
        }
	}
}
