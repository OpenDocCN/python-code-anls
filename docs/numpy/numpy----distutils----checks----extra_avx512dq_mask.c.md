# `.\numpy\numpy\distutils\checks\extra_avx512dq_mask.c`

```py
/**
 * Test DQ mask operations due to:
 *  - MSVC has supported it since vs2019 see,
 *    https://developercommunity.visualstudio.com/content/problem/518298/missing-avx512bw-mask-intrinsics.html
 *  - Clang >= v8.0
 *  - GCC >= v7.1
 */
int main(void)
{
    // 创建一个所有位都置为 1 的 8 位掩码 m8，用来表示相等比较结果
    __mmask8 m8 = _mm512_cmpeq_epi64_mask(_mm512_set1_epi64(1), _mm512_set1_epi64(1));
    
    // 对 m8 应用逻辑或运算，结果仍然为 m8，这一步没有变化
    m8 = _kor_mask8(m8, m8);
    
    // 对 m8 应用逻辑异或运算，结果为所有位都清零的掩码
    m8 = _kxor_mask8(m8, m8);
    
    // 将 m8 转换为一个 32 位无符号整数掩码，再转回 8 位掩码，但此时所有位都为 0
    m8 = _cvtu32_mask8(_cvtmask8_u32(m8));
    
    // 将 m8 转换为 32 位无符号整数掩码，再转换为整数返回
    return (int)_cvtmask8_u32(m8);
}
```