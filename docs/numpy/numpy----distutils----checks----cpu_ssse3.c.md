# `.\numpy\numpy\distutils\checks\cpu_ssse3.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了 DETECT_FEATURES 和 __INTEL_COMPILER，
     * Intel 编译器与 GCC 和 CLANG 不同，会暴露所有支持的内置函数，
     * 不管是否通过 `--cpu-baseline` 或环境变量 `CFLAGS` 指定了构建选项。
     * 因此，我们必须测试 CPU 特性的 #定义，当启用了本地/主机选项时，
     * 通过 `--cpu-baseline` 或环境变量 `CFLAGS`，否则测试将会失败并导致启用所有可能的特性。
     */
    #ifndef __SSSE3__
        #error "主机/架构不支持 SSSE3"
    #endif
#endif

#include <tmmintrin.h>

int main(void)
{
    // 创建两个零值的 128 位整数向量，并对其进行 16 位整数的横向加法
    __m128i a = _mm_hadd_epi16(_mm_setzero_si128(), _mm_setzero_si128());
    // 将结果向量的低 32 位整数转换为整数并返回
    return (int)_mm_cvtsi128_si32(a);
}
```