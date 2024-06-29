# `.\numpy\numpy\distutils\checks\cpu_sse42.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES和__INTEL_COMPILER，表示需要检测CPU特性并且是使用Intel Compiler
     * 与GCC和CLANG不同，Intel编译器会暴露所有支持的内置函数，不管是否指定了这些特性的构建选项。
     * 因此，当通过`--cpu-baseline`启用了本地/主机选项或者通过环境变量`CFLAGS`启用了这些选项，我们必须测试CPU特性的#define
     * 否则，测试将失败并导致启用所有可能的特性。
     */
    #ifndef __SSE4_2__
        #error "HOST/ARCH doesn't support SSE42"
    #endif
#endif

#include <smmintrin.h>

int main(void)
{
    // 创建四个零的单精度浮点数的SSE寄存器
    __m128 a = _mm_hadd_ps(_mm_setzero_ps(), _mm_setzero_ps());
    // 将SSE寄存器的值转换为整数返回
    return (int)_mm_cvtss_f32(a);
}
```