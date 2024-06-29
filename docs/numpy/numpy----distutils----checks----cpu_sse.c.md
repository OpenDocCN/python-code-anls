# `.\numpy\numpy\distutils\checks\cpu_sse.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用了Intel编译器
     * 不同于GCC和CLANG，Intel编译器会暴露所有支持的内联函数，
     * 无论是否为这些特性指定了构建选项。
     * 因此，当启用选项native/host通过`--cpu-baseline`或通过环境变量`CFLAGS`时，
     * 我们必须测试CPU特性的宏定义，否则测试将会失败，
     * 并导致启用所有可能的特性。
     */
    #ifndef __SSE__
        #error "HOST/ARCH不支持SSE"
    #endif
#endif

#include <xmmintrin.h>

int main(void)
{
    __m128 a = _mm_add_ps(_mm_setzero_ps(), _mm_setzero_ps());
    return (int)_mm_cvtss_f32(a);
}
```