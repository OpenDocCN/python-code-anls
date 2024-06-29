# `.\numpy\numpy\distutils\checks\cpu_fma3.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且定义了__INTEL_COMPILER，则编译器将暴露所有支持的内部函数，
     * 无论这些功能的构建选项是否已指定。因此，当通过`--cpu-baseline`启用选项本地/主机
     * 或通过环境变量`CFLAGS`启用选项时，我们必须测试CPU特性的#definitions，否则测试将被破坏，
     * 并导致启用所有可能的特性。
     */
    #if !defined(__FMA__) && !defined(__AVX2__)
        #error "HOST/ARCH doesn't support FMA3"
    #endif
#endif

#include <xmmintrin.h>
#include <immintrin.h>

int main(int argc, char **argv)
{
    __m256 a = _mm256_loadu_ps((const float*)argv[argc-1]); // 从内存中加载未对齐的256位单精度浮点数到__m256类型的变量a中
           a = _mm256_fmadd_ps(a, a, a);  // 对a中的每个元素执行乘法和加法
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));  // 将a的高128位数据转换为__m128类型，然后将其转换为float类型返回
}
```