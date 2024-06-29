# `.\numpy\numpy\distutils\checks\cpu_avx512_cnl.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了 DETECT_FEATURES 和 __INTEL_COMPILER
     * Intel 编译器与 GCC 和 CLANG 不同，会暴露所有支持的内置函数，
     * 无论是否为这些特性指定了构建选项。
     * 因此，我们必须测试 CPU 特性的 #定义，
     * 当启用 `--cpu-baseline` 或通过环境变量 `CFLAGS` 启用本地/主机选项时，
     * 否则测试将中断并导致启用所有可能的特性。
     */
    #if !defined(__AVX512VBMI__) || !defined(__AVX512IFMA__)
        #error "HOST/ARCH 不支持 CannonLake AVX512 特性"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 从 argv 中加载最后一个参数作为 __m512i 类型的变量 a
    __m512i a = _mm512_loadu_si512((const __m512i*)argv[argc-1]);
    /* IFMA */
    // 使用 _mm512_madd52hi_epu64 执行 IFMA 操作（Integer Fused Multiply-Add）
    a = _mm512_madd52hi_epu64(a, a, _mm512_setzero_si512());
    /* VMBI */
    // 使用 _mm512_permutex2var_epi8 执行 VMBI 操作（Vector Byte Permute）
    a = _mm512_permutex2var_epi8(a, _mm512_setzero_si512(), a);
    // 将 __m512i 类型的变量 a 转换为 __m128i 类型，并提取低 128 位返回一个整数
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
```