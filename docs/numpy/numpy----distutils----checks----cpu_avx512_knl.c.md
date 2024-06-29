# `.\numpy\numpy\distutils\checks\cpu_avx512_knl.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用Intel编译器，
     * Intel编译器与GCC和CLANG不同，会暴露所有支持的内置函数，
     * 无论是否指定了这些特性的构建选项。
     * 因此，我们必须在使用`--cpu-baseline`或通过环境变量`CFLAGS`启用本机/主机选项时测试CPU特性的#定义，
     * 否则测试将失效并导致启用所有可能的特性。
     */
    #if !defined(__AVX512ER__) || !defined(__AVX512PF__)
        #error "HOST/ARCH doesn't support Knights Landing AVX512 features"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 定义一个包含128个整数的数组base，初始化为零
    int base[128]={};
    // 从命令行参数中加载一个__m512d类型的向量ad
    __m512d ad = _mm512_loadu_pd((const __m512d*)argv[argc-1]);
    /* ER */
    // 将ad转换为__m512i类型的向量a，执行_exp2a23_pd()函数
    __m512i a = _mm512_castpd_si512(_mm512_exp2a23_pd(ad));
    /* PF */
    // 使用掩码_MM_HINT_T1，将a作为地址，预取64位整数散列存储到base中
    _mm512_mask_prefetch_i64scatter_pd(base, _mm512_cmpeq_epi64_mask(a, a), a, 1, _MM_HINT_T1);
    // 返回base数组的第一个元素
    return base[0];
}
```