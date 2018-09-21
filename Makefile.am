AUTOMAKE_OPTIONS = subdir-objects

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS		= compat

bin_PROGRAMS	= ccminer

ccminer_SOURCES		= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  cpu-miner.cpp util.cpp sha2.cpp \
			  cryptonight/cryptonight.cu cryptonight/cuda_cryptonight_core.cu cryptonight/cuda_cryptonight_extra.cu \
			  cryptonight-cpu.cpp \
			  crypto/oaes_lib.cpp \
			  crypto/c_keccak.cpp \
			  crypto/c_groestl.cpp \
			  crypto/c_blake256.cpp \
			  crypto/c_jh.cpp \
			  crypto/c_skein.cpp \
			  crypto/aesb.cpp


ccminer_LDFLAGS		= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD		= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
ccminer_CPPFLAGS	= $(CUDA_CFLAGS) -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

NVCC_GENCODE		= -D_FORCE_INLINES $(CUDA_CFLAGS) -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_50,code=sm_50  -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) @CFLAGS@ -I . $(NVCC_GENCODE) --maxrregcount=128 $(JANSSON_INCLUDES) -o $@ -c $<
