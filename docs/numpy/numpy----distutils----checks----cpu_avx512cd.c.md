# `.\numpy\numpy\distutils\checks\cpu_avx512cd.c`

```py
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 如果定义了DETECT_FEATURES并且使用Intel编译器，
     * Intel编译器会公开所有支持的内置函数，无论是否指定了这些功能的构建选项。
     * 因此，我们必须在启用了`--cpu-baseline`或通过环境变量`CFLAGS`的本地/主机选项时测试CPU特性的#定义，
     * 否则测试将会失败，并导致启用所有可能的特性。
     */
    #ifndef __AVX512CD__
        #error "主机/架构不支持AVX512CD"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 从命令行参数中加载一个512位整数向量，并统计其前导零的数量
    __m512i a = _mm512_lzcnt_epi32(_mm512_loadu_si512((const __m512i*)argv[argc-1]));
    // 将512位整数向量转换为128位整数，并返回其最低位的32位整数
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
```