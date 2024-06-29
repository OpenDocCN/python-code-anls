# `.\numpy\numpy\distutils\checks\cpu_sse3.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用Intel编译器，
     * Intel编译器与GCC和CLANG不同，它会暴露所有支持的内置函数，
     * 无论是否指定了这些特性的构建选项。
     * 因此，我们必须在启用`--cpu-baseline`或通过环境变量`CFLAGS`设置时测试CPU特性的#定义，
     * 否则测试将失效并导致启用所有可能的特性。
     */
    #ifndef __SSE3__
        #error "HOST/ARCH doesn't support SSE3"
    #endif
#endif

#include <pmmintrin.h>

int main(void)
{
    // 创建两个全零的单精度浮点数向量，并将它们按元素加和
    __m128 a = _mm_hadd_ps(_mm_setzero_ps(), _mm_setzero_ps());
    // 将结果向量的第一个单精度浮点数转换为整数并返回
    return (int)_mm_cvtss_f32(a);
}
```