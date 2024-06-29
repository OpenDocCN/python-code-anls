# `.\numpy\numpy\_core\src\common\simd\vec\reorder.h`

```
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_REORDER_H
#define _NPY_SIMD_VEC_REORDER_H

// combine lower part of two vectors
#define npyv__combinel(A, B) vec_mergeh((npyv_u64)(A), (npyv_u64)(B))
// Define macros to combine lower part of vectors for different types
#define npyv_combinel_u8(A, B)  ((npyv_u8) npyv__combinel(A, B))
#define npyv_combinel_s8(A, B)  ((npyv_s8) npyv__combinel(A, B))
#define npyv_combinel_u16(A, B) ((npyv_u16)npyv__combinel(A, B))
#define npyv_combinel_s16(A, B) ((npyv_s16)npyv__combinel(A, B))
#define npyv_combinel_u32(A, B) ((npyv_u32)npyv__combinel(A, B))
#define npyv_combinel_s32(A, B) ((npyv_s32)npyv__combinel(A, B))
#define npyv_combinel_u64       vec_mergeh
#define npyv_combinel_s64       vec_mergeh
#if NPY_SIMD_F32
    #define npyv_combinel_f32(A, B) ((npyv_f32)npyv__combinel(A, B))
#endif
#define npyv_combinel_f64       vec_mergeh

// combine higher part of two vectors
#define npyv__combineh(A, B) vec_mergel((npyv_u64)(A), (npyv_u64)(B))
// Define macros to combine higher part of vectors for different types
#define npyv_combineh_u8(A, B)  ((npyv_u8) npyv__combineh(A, B))
#define npyv_combineh_s8(A, B)  ((npyv_s8) npyv__combineh(A, B))
#define npyv_combineh_u16(A, B) ((npyv_u16)npyv__combineh(A, B))
#define npyv_combineh_s16(A, B) ((npyv_s16)npyv__combineh(A, B))
#define npyv_combineh_u32(A, B) ((npyv_u32)npyv__combineh(A, B))
#define npyv_combineh_s32(A, B) ((npyv_s32)npyv__combineh(A, B))
#define npyv_combineh_u64       vec_mergel
#define npyv_combineh_s64       vec_mergel
#if NPY_SIMD_F32
    #define npyv_combineh_f32(A, B) ((npyv_f32)npyv__combineh(A, B))
#endif
#define npyv_combineh_f64       vec_mergel

/*
 * combine: combine two vectors from lower and higher parts of two other vectors
 * zip: interleave two vectors
*/
#define NPYV_IMPL_VEC_COMBINE_ZIP(T_VEC, SFX)                  \
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = NPY_CAT(npyv_combinel_, SFX)(a, b);         \
        r.val[1] = NPY_CAT(npyv_combineh_, SFX)(a, b);         \
        return r;                                              \
    }                                                          \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)     \
    {                                                          \
        T_VEC##x2 r;                                           \
        r.val[0] = vec_mergeh(a, b);                           \
        r.val[1] = vec_mergel(a, b);                           \
        return r;                                              \
    }

// Define combine and zip operations for different vector types and sizes
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u8,  u8)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s8,  s8)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u16, u16)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s16, s16)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u32, u32)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s32, s32)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_u64, u64)
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_s64, s64)
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_COMBINE_ZIP(npyv_f32, f32)
#endif

#endif  // _NPY_SIMD_VEC_REORDER_H
// 合并两个名为 npyv_f64 的向量
NPYV_IMPL_VEC_COMBINE_ZIP(npyv_f64, f64)

// 解交错两个向量
NPY_FINLINE npyv_u8x2 npyv_unzip_u8(npyv_u8 ab0, npyv_u8 ab1)
{
    // 定义偶数索引顺序的向量
    const npyv_u8 idx_even = npyv_set_u8(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    );
    // 定义奇数索引顺序的向量
    const npyv_u8 idx_odd = npyv_set_u8(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
    );
    // 定义返回结果的变量
    npyv_u8x2 r;
    // 使用 vec_perm 函数根据偶数索引重新排列 ab0 和 ab1，并存储到 r 的第一个元素
    r.val[0] = vec_perm(ab0, ab1, idx_even);
    // 使用 vec_perm 函数根据奇数索引重新排列 ab0 和 ab1，并存储到 r 的第二个元素
    r.val[1] = vec_perm(ab0, ab1, idx_odd);
    return r;  // 返回重新排列后的向量
}

// 同样的功能，但用于有符号的 8 位整数向量
NPY_FINLINE npyv_s8x2 npyv_unzip_s8(npyv_s8 ab0, npyv_s8 ab1)
{
    // 调用无符号 8 位整数的解交错函数，并将结果转换为有符号 8 位整数向量
    npyv_u8x2 ru = npyv_unzip_u8((npyv_u8)ab0, (npyv_u8)ab1);
    // 定义返回结果的变量
    npyv_s8x2 r;
    // 将 ru 的第一个元素转换为有符号 8 位整数并存储到 r 的第一个元素
    r.val[0] = (npyv_s8)ru.val[0];
    // 将 ru 的第二个元素转换为有符号 8 位整数并存储到 r 的第二个元素
    r.val[1] = (npyv_s8)ru.val[1];
    return r;  // 返回重新排列后的向量
}

// 类似的函数，用于无符号 16 位整数向量
NPY_FINLINE npyv_u16x2 npyv_unzip_u16(npyv_u16 ab0, npyv_u16 ab1)
{
    // 定义偶数索引顺序的向量
    const npyv_u8 idx_even = npyv_set_u8(
        0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29
    );
    // 定义奇数索引顺序的向量
    const npyv_u8 idx_odd = npyv_set_u8(
        2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31
    );
    // 定义返回结果的变量
    npyv_u16x2 r;
    // 使用 vec_perm 函数根据偶数索引重新排列 ab0 和 ab1，并存储到 r 的第一个元素
    r.val[0] = vec_perm(ab0, ab1, idx_even);
    // 使用 vec_perm 函数根据奇数索引重新排列 ab0 和 ab1，并存储到 r 的第二个元素
    r.val[1] = vec_perm(ab0, ab1, idx_odd);
    return r;  // 返回重新排列后的向量
}

// 类似的函数，用于有符号 16 位整数向量
NPY_FINLINE npyv_s16x2 npyv_unzip_s16(npyv_s16 ab0, npyv_s16 ab1)
{
    // 调用无符号 16 位整数的解交错函数，并将结果转换为有符号 16 位整数向量
    npyv_u16x2 ru = npyv_unzip_u16((npyv_u16)ab0, (npyv_u16)ab1);
    // 定义返回结果的变量
    npyv_s16x2 r;
    // 将 ru 的第一个元素转换为有符号 16 位整数并存储到 r 的第一个元素
    r.val[0] = (npyv_s16)ru.val[0];
    // 将 ru 的第二个元素转换为有符号 16 位整数并存储到 r 的第二个元素
    r.val[1] = (npyv_s16)ru.val[1];
    return r;  // 返回重新排列后的向量
}

// 类似的函数，用于无符号 32 位整数向量
NPY_FINLINE npyv_u32x2 npyv_unzip_u32(npyv_u32 ab0, npyv_u32 ab1)
{
    // 合并 ab0 和 ab1 的高位和低位部分，得到两个新的 32 位整数向量
    npyv_u32 m0 = vec_mergeh(ab0, ab1);
    npyv_u32 m1 = vec_mergel(ab0, ab1);
    // 再次合并 m0 和 m1 的高位和低位部分，得到两个重新排列的 32 位整数向量
    npyv_u32 r0 = vec_mergeh(m0, m1);
    npyv_u32 r1 = vec_mergel(m0, m1);
    // 定义返回结果的变量
    npyv_u32x2 r;
    // 将重新排列后的向量存储到返回结果中
    r.val[0] = r0;
    r.val[1] = r1;
    return r;  // 返回重新排列后的向量
}

// 类似的函数，用于有符号 32 位整数向量
NPY_FINLINE npyv_s32x2 npyv_unzip_s32(npyv_s32 ab0, npyv_s32 ab1)
{
    // 调用无符号 32 位整数的解交错函数，并将结果转换为有符号 32 位整数向量
    npyv_u32x2 ru = npyv_unzip_u32((npyv_u32)ab0, (npyv_u32)ab1);
    // 定义返回结果的变量
    npyv_s32x2 r;
    // 将 ru 的第一个元素转换为有符号 32 位整数并存储到 r 的第一个元素
    r.val[0] = (npyv_s32)ru.val[0];
    // 将 ru 的第二个元素转换为有符号 32 位整数并存储到 r 的第二个元素
    r.val[1] = (npyv_s32)ru.val[1];
    return r;  // 返回重新排列后的向量
}

// 如果支持单精度浮点数 SIMD 计算，则定义解交错函数，用于单精度浮点数向量
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32x2 npyv_unzip_f32(npyv_f32 ab0, npyv_f32 ab1)
    {
        // 调用无符号 32 位整数的解交错函数，并将结果转换为单精度浮点数向量
        npyv_u32x2 ru = npyv_unzip_u32((npyv_u32)ab0, (npyv_u32)ab1);
        // 定义返回结果的变量
        npyv
    );
    // 使用给定的索引向量 idx 对向量 a 进行排列操作，并返回结果
    return vec_perm(a, a, idx);
#endif
}
```