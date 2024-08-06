#ifndef INCLUDE_VECTOR_OPS_HPP
#define INCLUDE_VECTOR_OPS_HPP
// http://koturn.hatenablog.com/entry/2016/07/18/090000
// windows is not supported, but just in case (later someone might implement)
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=590,27,2
#ifdef _MSC_VER
#  include <immintrin.h> 
#else
#  include <x86intrin.h>
#endif

#include <string>

// These fast L2 squared distance codes (SSE and AVX) are from the Faiss library:
// https://github.com/facebookresearch/faiss/blob/master/utils.cpp
//
// Based on them, AVX512 implementation is also prepared.
// But it doesn't seem drastically fast. Only slightly faster than AVX:


// only use sse
#undef __AVX512F__
#undef __AVX__

#if defined(__AVX512F__)  
static const std::string g_simd_architecture = "avx512";
static const size_t bf_upbound_lim_float = 16;
static const size_t bf_upbound_lim_uint8 = 32;

// mm512 hadd function
static inline __m512 _mm512_hadd_ps(__m512 a);

#elif defined(__AVX__)  
static const std::string g_simd_architecture = "avx";
static const size_t bf_upbound_lim_float = 16;
static const size_t bf_upbound_lim_uint8 = 32;

#else 
static const std::string g_simd_architecture = "sse";
static const size_t bf_upbound_lim_float = 16;
static const size_t bf_upbound_lim_uint8 = 32;

#endif





// ========================= Distance functions ============================

float vec_L2sqr(const float *x, const float *y, size_t d);


float vec_L2sqr(const uint8_t *x, const uint8_t *y, size_t d);



#endif // DISTANCE_H
