# `.\numpy\numpy\distutils\checks\extra_avx512f_reduce.c`

```
#include <immintrin.h>
/**
 * The following intrinsics don't have direct native support but compilers
 * tend to emulate them.
 * They're usually supported by gcc >= 7.1, clang >= 4 and icc >= 19
 */

int main(void)
{
    // 创建一个包含所有元素为 1.0 的 __m512 类型的向量 one_ps
    __m512  one_ps = _mm512_set1_ps(1.0f);
    // 创建一个包含所有元素为 1.0 的 __m512d 类型的向量 one_pd
    __m512d one_pd = _mm512_set1_pd(1.0);
    // 创建一个包含所有元素为 1 的 __m512i 类型的整数向量 one_i64
    __m512i one_i64 = _mm512_set1_epi64(1);

    // 使用 _mm512_reduce_add_ps 计算 one_ps 向量中所有元素的和
    float sum_ps  = _mm512_reduce_add_ps(one_ps);
    // 使用 _mm512_reduce_add_pd 计算 one_pd 向量中所有元素的和
    double sum_pd = _mm512_reduce_add_pd(one_pd);
    // 将 one_i64 向量中所有元素视为 64 位有符号整数，计算其和
    int sum_int   = (int)_mm512_reduce_add_epi64(one_i64);
    // 将 one_i64 向量中所有元素视为 32 位有符号整数，计算其和
    sum_int  += (int)_mm512_reduce_add_epi32(one_i64);

    // 使用 _mm512_reduce_mul_ps 计算 one_ps 向量中所有元素的乘积，并加到 sum_ps
    sum_ps  += _mm512_reduce_mul_ps(one_ps);
    // 使用 _mm512_reduce_mul_pd 计算 one_pd 向量中所有元素的乘积，并加到 sum_pd
    sum_pd  += _mm512_reduce_mul_pd(one_pd);
    // 将 one_i64 向量中所有元素视为 64 位有符号整数，计算其乘积，并加到 sum_int
    sum_int += (int)_mm512_reduce_mul_epi64(one_i64);
    // 将 one_i64 向量中所有元素视为 32 位有符号整数，计算其乘积，并加到 sum_int
    sum_int += (int)_mm512_reduce_mul_epi32(one_i64);

    // 使用 _mm512_reduce_min_ps 找到 one_ps 向量中所有元素的最小值，并加到 sum_ps
    sum_ps  += _mm512_reduce_min_ps(one_ps);
    // 使用 _mm512_reduce_min_pd 找到 one_pd 向量中所有元素的最小值，并加到 sum_pd
    sum_pd  += _mm512_reduce_min_pd(one_pd);
    // 将 one_i64 向量中所有元素视为 32 位有符号整数，找到其最小值，并加到 sum_int
    sum_int += (int)_mm512_reduce_min_epi32(one_i64);
    // 将 one_i64 向量中所有元素视为 32 位无符号整数，找到其最小值，并加到 sum_int
    sum_int += (int)_mm512_reduce_min_epu32(one_i64);
    // 将 one_i64 向量中所有元素视为 64 位有符号整数，找到其最小值，并加到 sum_int
    sum_int += (int)_mm512_reduce_min_epi64(one_i64);

    // 使用 _mm512_reduce_max_ps 找到 one_ps 向量中所有元素的最大值，并加到 sum_ps
    sum_ps  += _mm512_reduce_max_ps(one_ps);
    // 使用 _mm512_reduce_max_pd 找到 one_pd 向量中所有元素的最大值，并加到 sum_pd
    sum_pd  += _mm512_reduce_max_pd(one_pd);
    // 将 one_i64 向量中所有元素视为 32 位有符号整数，找到其最大值，并加到 sum_int
    sum_int += (int)_mm512_reduce_max_epi32(one_i64);
    // 将 one_i64 向量中所有元素视为 32 位无符号整数，找到其最大值，并加到 sum_int
    sum_int += (int)_mm512_reduce_max_epu32(one_i64);
    // 将 one_i64 向量中所有元素视为 64 位有符号整数，找到其最大值，并加到 sum_int
    sum_int += (int)_mm512_reduce_max_epi64(one_i64);

    // 将 one_i64 向量中所有元素视为 32 位有符号整数，进行按位与操作，并加到 sum_int
    sum_int += (int)_mm512_reduce_and_epi32(one_i64);
    // 将 one_i64 向量中所有元素视为 64 位有符号整数，进行按位与操作，并加到 sum_int
    sum_int += (int)_mm512_reduce_and_epi64(one_i64);

    // 将 one_i64 向量中所有元素视为 32 位有符号整数，进行按位或操作，并加到 sum_int
    sum_int += (int)_mm512_reduce_or_epi32(one_i64);
    // 将 one_i64 向量中所有元素视为 64 位有符号整数，进行按位或操作，并加到 sum_int
    sum_int += (int)_mm512_reduce_or_epi64(one_i64);

    // 返回 sum_ps、sum_pd 和 sum_int 的整数和作为函数的返回值
    return (int)sum_ps + (int)sum_pd + sum_int;
}
```