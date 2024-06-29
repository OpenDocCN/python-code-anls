# `.\numpy\numpy\distutils\checks\cpu_avx512_skx.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了 DETECT_FEATURES 和 __INTEL_COMPILER，
     * Intel 编译器会暴露所有支持的内置函数，不管是否已经通过编译选项指定了这些特性。
     * 因此，在启用了 `--cpu-baseline` 或者通过环境变量 `CFLAGS` 启用本地/主机选项时，
     * 我们必须测试 CPU 特性的 #definitions。
     * 否则，测试将无法正常运行，并导致启用所有可能的特性。
     */
    #if !defined(__AVX512VL__) || !defined(__AVX512BW__) || !defined(__AVX512DQ__)
        #error "HOST/ARCH 不支持 SkyLake AVX512 特性"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 从命令行参数中加载最后一个参数的值，并将其作为 __m512i 类型的变量 aa
    __m512i aa = _mm512_abs_epi32(_mm512_loadu_si512((const __m512i*)argv[argc-1]));
    /* VL */
    // 从 aa 的第二个 64 位整数中提取一个 __m256i 类型的变量 a，并对其进行绝对值操作
    __m256i a = _mm256_abs_epi64(_mm512_extracti64x4_epi64(aa, 1));
    /* DQ */
    // 将 a 的值广播到一个 __m512i 类型的变量 b 中
    __m512i b = _mm512_broadcast_i32x8(a);
    /* BW */
    // 对 b 中的每个元素执行绝对值操作
    b = _mm512_abs_epi16(b);
    // 将 b 转换为一个 __m128i 类型，并返回其低 128 位整数作为整数类型返回值
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(b));
}
```