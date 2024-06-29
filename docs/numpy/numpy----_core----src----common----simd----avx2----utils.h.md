# `.\numpy\numpy\_core\src\common\simd\avx2\utils.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_AVX2_UTILS_H
#define _NPY_SIMD_AVX2_UTILS_H

// 定义一个宏，用于对256位整数型寄存器进行奇数位元素的重新排列
#define npyv256_shuffle_odd(A)    _mm256_permute4x64_epi64(A, _MM_SHUFFLE(3, 1, 2, 0))
// 定义一个宏，将256位浮点型寄存器转换为按奇数位元素排列的浮点型寄存器
#define npyv256_shuffle_odd_ps(A) _mm256_castsi256_ps(npyv256_shuffle_odd(_mm256_castps_si256(A)))
// 定义一个宏，用于对256位双精度浮点型寄存器进行奇数位元素的重新排列
#define npyv256_shuffle_odd_pd(A) _mm256_permute4x64_pd(A, _MM_SHUFFLE(3, 1, 2, 0))

// 定义一个内联函数，实现无符号8位整数型寄存器的乘法操作
NPY_FINLINE __m256i npyv256_mul_u8(__m256i a, __m256i b)
{
    // 创建一个掩码，用于选择奇数位元素
    const __m256i mask = _mm256_set1_epi32(0xFF00FF00);
    // 计算偶数位元素乘积
    __m256i even = _mm256_mullo_epi16(a, b);
    // 计算奇数位元素乘积
    __m256i odd  = _mm256_mullo_epi16(_mm256_srai_epi16(a, 8), _mm256_srai_epi16(b, 8));
    // 将奇数位元素乘积左移8位
    odd  = _mm256_slli_epi16(odd, 8);
    // 使用掩码选择奇数位元素乘积或偶数位元素乘积
    return _mm256_blendv_epi8(even, odd, mask);
}

#endif // _NPY_SIMD_AVX2_UTILS_H
```