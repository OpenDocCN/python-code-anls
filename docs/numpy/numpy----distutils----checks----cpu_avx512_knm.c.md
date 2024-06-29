# `.\numpy\numpy\distutils\checks\cpu_avx512_knm.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且定义了__INTEL_COMPILER，
     * 那么英特尔编译器会暴露所有支持的指令集，无论是否指定了这些特性的构建选项。
     * 因此，当通过`--cpu-baseline`启用选项本机/主机， 或通过环境变量`CFLAGS` 否则，
     * 我们必须测试CPU特性的#定义，否则测试将会无法正常工作，导致启用所有可能的特性。
     */
    #if !defined(__AVX5124FMAPS__) || !defined(__AVX5124VNNIW__) || !defined(__AVX512VPOPCNTDQ__)
        #error "HOST/ARCH不支持Knights Mill AVX512特性"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    __m512 b = _mm512_loadu_ps((const __m512*)argv[argc-2]);

    /* 4FMAPS */
    // 执行4FMAPS指令
    b = _mm512_4fmadd_ps(b, b, b, b, b, NULL);
    /* 4VNNIW */
    // 执行4VNNIW指令
    a = _mm512_4dpwssd_epi32(a, a, a, a, a, NULL);
    /* VPOPCNTDQ */
    // 执行VPOPCNTDQ指令
    a = _mm512_popcnt_epi64(a);

    // 执行矢量加法
    a = _mm512_add_epi32(a, _mm512_castps_si512(b));
    // 返回a的低128位整数
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
```