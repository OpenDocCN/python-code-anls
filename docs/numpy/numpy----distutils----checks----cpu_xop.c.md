# `.\numpy\numpy\distutils\checks\cpu_xop.c`

```
# 包含 AVX2 指令集的头文件
#include <immintrin.h>
# 如果是 MSVC 编译器，则包含该头文件
#ifdef _MSC_VER
    #include <ammintrin.h>
# 否则，包含该头文件
#else
    #include <x86intrin.h>
#endif

# 主函数
int main(void)
{
    # 将两个 __m128i 类型的寄存器中每个 32 位无符号整数的每个位进行无符号比较，并返回结果
    __m128i a = _mm_comge_epu32(_mm_setzero_si128(), _mm_setzero_si128());
    # 将 __m128i 类型的寄存器的值转换为 32 位有符号整数
    return _mm_cvtsi128_si32(a);
}
```