# `.\numpy\numpy\distutils\checks\cpu_f16c.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果已定义了DETECT_FEATURES并且定义了__INTEL_COMPILER，
     * 那么英特尔编译器将暴露所有支持的指令集，
     * 无论是否指定了这些特性的构建选项。
     * 因此，当通过`--cpu-baseline`启用了native/host选项或者通过环境变量`CFLAGS`设置了它，
     * 我们必须在这些选项被启用时测试CPU特性的#定义，
     * 否则测试将会失效，并导致启用所有可能的特性。
     */
    #ifndef __F16C__
        #error "HOST/ARCH doesn't support F16C"
    #endif
#endif

#include <emmintrin.h>
#include <immintrin.h>

int main(int argc, char **argv)
{
    // 将argv[argc-1]的值加载为一个16位半精度浮点数，转换为4个单精度浮点数
    __m128 a  = _mm_cvtph_ps(_mm_loadu_si128((const __m128i*)argv[argc-1]));
    // 将argv[argc-2]的值加载为一个16位半精度浮点数，转换为8个单精度浮点数
    __m256 a8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)argv[argc-2]));
    // 返回a的单精度浮点数值与a8的单精度浮点数值之和的整数部分
    return (int)(_mm_cvtss_f32(a) + _mm_cvtss_f32(_mm256_castps256_ps128(a8)));
}
```