# `.\numpy\numpy\distutils\checks\cpu_sse2.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用Intel编译器，
     * 与GCC和CLANG不同，Intel编译器会公开所有支持的内置函数，
     * 无论是否为这些特性指定了构建选项。
     * 因此，当通过`--cpu-baseline`启用了本地/主机选项或通过环境变量`CFLAGS`设置时，
     * 我们必须测试CPU特性的#定义，否则测试将失效，并导致启用所有可能的特性。
     */
    #ifndef __SSE2__
        #error "HOST/ARCH doesn't support SSE2"
    #endif
#endif

#include <emmintrin.h>

int main(void)
{
    // 创建一个全零的__m128i类型变量a
    __m128i a = _mm_add_epi16(_mm_setzero_si128(), _mm_setzero_si128());
    // 将__m128i类型变量a的低位128位整数转换为32位整数返回
    return _mm_cvtsi128_si32(a);
}
```