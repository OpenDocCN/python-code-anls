# `.\numpy\numpy\distutils\checks\extra_avx512bw_mask.c`

```py
#include <immintrin.h>
/**
 * 测试 AVX-512 BW 掩码操作的效果，具体包括：
 *  - MSVC 自 vs2019 起支持此功能，详见
 *    https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html
 *  - Clang >= v8.0
 *  - GCC >= v7.1
 */
int main(void)
{
    // 创建一个所有位都置位的掩码 m64，用于比较两个 64 字节长的整数是否相等
    __mmask64 m64 = _mm512_cmpeq_epi8_mask(_mm512_set1_epi8((char)1), _mm512_set1_epi8((char)1));
    // 对 m64 进行按位或运算，结果仍存入 m64 中
    m64 = _kor_mask64(m64, m64);
    // 对 m64 进行按位异或运算，结果存入 m64 中
    m64 = _kxor_mask64(m64, m64);
    // 将 m64 转换为 u64 类型的掩码，再将其转换回掩码类型
    m64 = _cvtu64_mask64(_cvtmask64_u64(m64));
    // 对 m64 进行 unpack 操作，解压缩成更高位的掩码
    m64 = _mm512_kunpackd(m64, m64);
    // 将 m64 转换为更宽的掩码类型，并存回 m64
    m64 = (__mmask64)_mm512_kunpackw((__mmask32)m64, (__mmask32)m64);
    // 将最终的掩码 m64 转换为 u64 类型，并返回作为整数
    return (int)_cvtmask64_u64(m64);
}
```