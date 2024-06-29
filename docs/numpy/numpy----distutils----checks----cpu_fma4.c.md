# `.\numpy\numpy\distutils\checks\cpu_fma4.c`

```py
#include <immintrin.h>
#ifdef _MSC_VER
    #include <ammintrin.h>  // 包含适用于 Microsoft 编译器的特定头文件，用于 AVX 指令集
#else
    #include <x86intrin.h>  // 包含适用于通用 x86 架构的 AVX 指令集头文件
#endif

int main(int argc, char **argv)
{
    __m256 a = _mm256_loadu_ps((const float*)argv[argc-1]);
           // 使用 AVX 指令加载未对齐的单精度浮点数组，转换为 256 位 AVX 寄存器
    a = _mm256_macc_ps(a, a, a);
    // 使用 AVX 指令执行乘-累加操作：a = a * a + a
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
    // 将 AVX 256 位寄存器的高 128 位转换为单精度浮点数并返回其整数部分
}
```