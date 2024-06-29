# `.\numpy\numpy\distutils\checks\cpu_avx512_clx.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES和__INTEL_COMPILER，则继续进行以下操作
     * 与GCC和CLANG不同，Intel Compiler会暴露所有支持的内部函数，无论是否指定了这些特性的构建选项。
     * 因此，当通过`--cpu-baseline`启用选项本机/主机，或通过环境变量`CFLAGS`启用选项时，必须测试CPU特性的#定义，否则测试将中断并导致启用所有可能的特性。
     */
    #ifndef __AVX512VNNI__
        #error "HOST/ARCH不支持CascadeLake AVX512特性"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    /* VNNI */
    // 通过argv[argc-1]加载512位未对齐整数到__m512i类型的寄存器a中
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    // 将a与512位零寄存器进行无符号32位整数累加，结果存储回a中
    a = _mm512_dpbusd_epi32(a, _mm512_setzero_si512(), a);
    // 将a的低128位转换为32位整数并返回
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
```