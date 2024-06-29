# `.\numpy\numpy\distutils\checks\cpu_avx512f.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES和__INTEL_COMPILER，则执行以下操作：
     * 与GCC和CLANG不同，Intel编译器会暴露所有支持的内联函数，
     * 无论是否指定了那些特性的构建选项。
     * 因此，当使用选项native/host启用`--cpu-baseline`或通过环境变量`CFLAGS`启用
     * 时，我们必须测试CPU特性的＃定义，否则测试将出错，并导致启用所有可能的特性。
     */
    #ifndef __AVX512F__
        #error "HOST/ARCH doesn't support AVX512F"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 从argv数组中加载数据到512位整型寄存器a，并求取绝对值
    __m512i a = _mm512_abs_epi32(_mm512_loadu_si512((const __m512i*)argv[argc-1]));
    // 将512位整型寄存器a转换为128位整型寄存器，并返回其低128位的整数值
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
```