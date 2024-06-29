# `.\numpy\numpy\distutils\checks\cpu_avx.c`

```
#if defined(DETECT_FEATURES) && defined(__INTEL_COMPILER)
    /*
     * 当使用 Intel 编译器时，与 GCC 和 CLANG 不同，它会暴露所有支持的指令集，
     * 无论是否已经通过编译选项指定了这些特性。因此，我们必须测试 CPU 特性的 #definitions，
     * 当使用 `--cpu-baseline` 开启本地/主机优化或者通过环境变量 `CFLAGS` 设置时，
     * 否则测试将会失效并导致启用所有可能的特性。
     */
    #ifndef __AVX__
        #error "主机/架构不支持 AVX"
    #endif
#endif

#include <immintrin.h>

int main(int argc, char **argv)
{
    // 加载两个未对齐的 AVX 寄存器，分别从命令行参数的最后一个和第一个参数中加载
    __m256 a = _mm256_add_ps(_mm256_loadu_ps((const float*)argv[argc-1]), _mm256_loadu_ps((const float*)argv[1]));
    // 将结果向下转换为单精度浮点数，并将其强制转换为整数返回
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
```