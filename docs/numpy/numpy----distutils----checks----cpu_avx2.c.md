# `.\numpy\numpy\distutils\checks\cpu_avx2.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用Intel编译器，
     * 那么与GCC和CLANG不同，Intel编译器公开所有支持的内部功能，
     * 无论是否指定了这些功能的构建选项。
     * 因此，当通过`--cpu-baseline`启用本机/主机选项时，
     * 我们必须测试CPU特性的＃定义，否则测试将被破坏，并导致启用所有可能的特性。
     */
    #ifndef __AVX2__
        #error "HOST/ARCH不支持AVX2"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 从argv[argc-1]加载16位整数，取绝对值并存储到__m256i类型的变量a中
    __m256i a = _mm256_abs_epi16(_mm256_loadu_si256((const __m256i*)argv[argc-1]));
    // 将__m256i类型的变量a转换为__m128i类型，再转换为32位整数并返回
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}
```