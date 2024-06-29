# `.\numpy\numpy\distutils\checks\cpu_avx512_icl.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且__INTEL_COMPILER也被定义
     * 与GCC和CLANG不同，英特尔编译器会暴露所有支持的内联函数，
     * 无论是否指定了这些功能的构建选项。
     * 因此，当通过`--cpu-baseline`启用本机/主机选项或通过环境变量`CFLAGS`启用时，
     * 我们必须测试CPU功能的#定义，否则测试将失败并导致启用所有可能的功能。
     */
    #if !defined(__AVX512VPOPCNTDQ__) || !defined(__AVX512BITALG__) || !defined(__AVX512VPOPCNTDQ__)
        #error "HOST/ARCH不支持IceLake AVX512特性"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    /* VBMI2 */
    a = _mm512_shrdv_epi64(a, a, _mm512_setzero_si512());
    /* BITLAG */
    a = _mm512_popcnt_epi8(a);
    /* VPOPCNTDQ */
    a = _mm512_popcnt_epi64(a);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
```