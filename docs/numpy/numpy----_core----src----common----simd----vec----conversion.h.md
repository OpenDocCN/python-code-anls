# `.\numpy\numpy\_core\src\common\simd\vec\conversion.h`

```py
// 如果没有定义 NPY_SIMD，则输出错误提示信息
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

// 如果没有定义 _NPY_SIMD_VEC_CVT_H，则定义它
#ifndef _NPY_SIMD_VEC_CVT_H
#define _NPY_SIMD_VEC_CVT_H

// 将布尔矢量转换为整数矢量
#define npyv_cvt_u8_b8(BL)   ((npyv_u8)  BL)
#define npyv_cvt_s8_b8(BL)   ((npyv_s8)  BL)
#define npyv_cvt_u16_b16(BL) ((npyv_u16) BL)
#define npyv_cvt_s16_b16(BL) ((npyv_s16) BL)
#define npyv_cvt_u32_b32(BL) ((npyv_u32) BL)
#define npyv_cvt_s32_b32(BL) ((npyv_s32) BL)
#define npyv_cvt_u64_b64(BL) ((npyv_u64) BL)
#define npyv_cvt_s64_b64(BL) ((npyv_s64) BL)
#if NPY_SIMD_F32
    #define npyv_cvt_f32_b32(BL) ((npyv_f32) BL)
#endif
#define npyv_cvt_f64_b64(BL) ((npyv_f64) BL)

// 将整数矢量转换为布尔矢量
#define npyv_cvt_b8_u8(A)   ((npyv_b8)  A)
#define npyv_cvt_b8_s8(A)   ((npyv_b8)  A)
#define npyv_cvt_b16_u16(A) ((npyv_b16) A)
#define npyv_cvt_b16_s16(A) ((npyv_b16) A)
#define npyv_cvt_b32_u32(A) ((npyv_b32) A)
#define npyv_cvt_b32_s32(A) ((npyv_b32) A)
#define npyv_cvt_b64_u64(A) ((npyv_b64) A)
#define npyv_cvt_b64_s64(A) ((npyv_b64) A)
#if NPY_SIMD_F32
    #define npyv_cvt_b32_f32(A) ((npyv_b32) A)
#endif
#define npyv_cvt_b64_f64(A) ((npyv_b64) A)

// 扩展
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data)
{
    npyv_u16x2 r;
#ifdef NPY_HAVE_VX
    r.val[0] = vec_unpackh(data);
    r.val[1] = vec_unpackl(data);
#else
    npyv_u8 zero = npyv_zero_u8();
    r.val[0] = (npyv_u16)vec_mergeh(data, zero);
    r.val[1] = (npyv_u16)vec_mergel(data, zero);
#endif
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    npyv_u32x2 r;
#ifdef NPY_HAVE_VX
    r.val[0] = vec_unpackh(data);
    r.val[1] = vec_unpackl(data);
#else
    npyv_u16 zero = npyv_zero_u16();
    r.val[0] = (npyv_u32)vec_mergeh(data, zero);
    r.val[1] = (npyv_u32)vec_mergel(data, zero);
#endif
    return r;
}

// 将两个16位布尔值打包成一个8位布尔矢量
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    return vec_pack(a, b);
}

// 将四个32位布尔矢量打包成一个8位布尔矢量
NPY_FINLINE npyv_b8 npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    npyv_b16 ab = vec_pack(a, b);
    npyv_b16 cd = vec_pack(c, d);
    return npyv_pack_b8_b16(ab, cd);
}

// 将八个64位布尔矢量打包成一个8位布尔矢量
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    npyv_b32 ab = vec_pack(a, b);
    npyv_b32 cd = vec_pack(c, d);
    npyv_b32 ef = vec_pack(e, f);
    npyv_b32 gh = vec_pack(g, h);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// 将布尔矢量转换为整数位域
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX2)
    NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
    {
        const npyv_u8 qperm = npyv_set_u8(120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0);
        npyv_u16 r = (npyv_u16)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        // 如果定义了 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 3)
        return vec_extract(r, 3);
    #else
        // 如果未定义 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 4)
        return vec_extract(r, 4);
    #endif
        // 用于消除模糊警告：变量 `r` 被设置但未使用 [-Wunused-but-set-variable]
    (void)r;
    }
    
    // 将 16 位整数向量转换为位表示的64位整数
    NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
    {
        // 定义一个字节向量 qperm
        const npyv_u8 qperm = npyv_setf_u8(128, 112, 96, 80, 64, 48, 32, 16, 0);
        // 对输入向量 a 执行 qperm 排列并将结果转换为 8 位无符号整数向量 r
        npyv_u8 r = (npyv_u8)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        // 如果定义了 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 6)
        return vec_extract(r, 6);
    #else
        // 如果未定义 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 8)
        return vec_extract(r, 8);
    #endif
    // 用于消除模糊警告：变量 `r` 被设置但未使用 [-Wunused-but-set-variable]
    (void)r;
    }
    
    // 将 32 位整数向量转换为位表示的64位整数
    NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
    {
    #ifdef NPY_HAVE_VXE
        // 如果定义了 NPY_HAVE_VXE 宏，则定义一个特定的 qperm 字节向量
        const npyv_u8 qperm = npyv_setf_u8(128, 128, 128, 128, 128, 96, 64, 32, 0);
    #else
        // 如果未定义 NPY_HAVE_VXE 宏，则定义一个不同的 qperm 字节向量
        const npyv_u8 qperm = npyv_setf_u8(128, 96, 64, 32, 0);
    #endif
        // 对输入向量 a 执行 qperm 排列并将结果转换为 8 位无符号整数向量 r
        npyv_u8 r = (npyv_u8)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        // 如果定义了 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 6)
        return vec_extract(r, 6);
    #else
        // 如果未定义 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 8)
        return vec_extract(r, 8);
    #endif
    // 用于消除模糊警告：变量 `r` 被设置但未使用 [-Wunused-but-set-variable]
    (void)r;
    }
    
    // 将 64 位整数向量转换为位表示的64位整数
    NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
    {
    #ifdef NPY_HAVE_VXE
        // 如果定义了 NPY_HAVE_VXE 宏，则定义一个特定的 qperm 字节向量
        const npyv_u8 qperm = npyv_setf_u8(128, 128, 128, 128, 128, 128, 128, 64, 0);
    #else
        // 如果未定义 NPY_HAVE_VXE 宏，则定义一个不同的 qperm 字节向量
        const npyv_u8 qperm = npyv_setf_u8(128, 64, 0);
    #endif
        // 对输入向量 a 执行 qperm 排列并将结果转换为 8 位无符号整数向量 r
        npyv_u8 r = (npyv_u8)vec_vbpermq((npyv_u8)a, qperm);
    #ifdef NPY_HAVE_VXE
        // 如果定义了 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 6)
        return vec_extract(r, 6);
    #else
        // 如果未定义 NPY_HAVE_VXE 宏，则使用 vec_extract(r, 8)
        return vec_extract(r, 8);
    #endif
    // 用于消除模糊警告：变量 `r` 被设置但未使用 [-Wunused-but-set-variable]
    (void)r;
    }
NPY_FINLINE npyv_s32 npyv__trunc_s32_f64(npyv_f64 a, npyv_f64 b)
{
#ifdef NPY_HAVE_VX
    // 如果平台支持 VX 指令集，则使用向量打包将两个双精度向量 a 和 b 转换为带符号整数向量并返回
    return vec_packs(vec_signed(a), vec_signed(b));
// VSX
#elif defined(__IBMC__)
    // 如果使用 IBM XL C/C++ 编译器，则执行以下操作
    // 创建一个字节向量，指定低位和高位对应的位置
    const npyv_u8 seq_even = npyv_set_u8(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27);
    // 将双精度向量 a 和 b 分别转换为带符号整数向量 lo_even 和 hi_even
    npyv_s32 lo_even = vec_cts(a, 0);
    npyv_s32 hi_even = vec_cts(b, 0);
    // 使用 seq_even 的顺序重新排列 lo_even 和 hi_even 向量，返回结果向量
    return vec_perm(lo_even, hi_even, seq_even);
#else
    // 否则，执行以下操作（适用于大多数情况，如 GCC 编译器）
    // 创建一个字节向量，指定奇数位置的序列
    const npyv_u8 seq_odd = npyv_set_u8(4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31);
    #ifdef __clang__
        // 如果编译器是 Clang
    
        // __builtin_convertvector 在许多版本上不支持这种转换
        // 幸运的是，几乎所有版本都有直接的内建函数 'xvcvdpsxws'
        // 使用内建函数 'xvcvdpsxws' 将向量 a 转换为 s32 类型的 lo_odd
        npyv_s32 lo_odd = __builtin_vsx_xvcvdpsxws(a);
        // 使用内建函数 'xvcvdpsxws' 将向量 b 转换为 s32 类型的 hi_odd
        npyv_s32 hi_odd = __builtin_vsx_xvcvdpsxws(b);
    #else // gcc
        // 如果编译器是 GCC
    
        // 声明两个 s32 类型的变量 lo_odd 和 hi_odd，稍后将通过内联汇编进行赋值
        npyv_s32 lo_odd, hi_odd;
        // 使用内联汇编执行 'xvcvdpsxws' 指令，将向量 a 转换为 s32 类型的 lo_odd
        __asm__ ("xvcvdpsxws %x0,%x1" : "=wa" (lo_odd) : "wa" (a));
        // 使用内联汇编执行 'xvcvdpsxws' 指令，将向量 b 转换为 s32 类型的 hi_odd
        __asm__ ("xvcvdpsxws %x0,%x1" : "=wa" (hi_odd) : "wa" (b));
    #endif
    
    // 使用 vec_perm 函数对 lo_odd 和 hi_odd 进行向量的混合操作，使用 seq_odd 作为混合顺序
    return vec_perm(lo_odd, hi_odd, seq_odd);
#endif
}

// 结束一个条件编译块，对应于前面的 #if NPY_SIMD_F32
#if NPY_SIMD_F32
    // 定义一个内联函数，将 npyv_f32 类型的向量 a 四舍五入为 npyv_s32 类型的向量
    NPY_FINLINE npyv_s32 npyv_round_s32_f32(npyv_f32 a)
    { return npyv__trunc_s32_f32(vec_rint(a)); }
#endif

// 定义一个内联函数，将 npyv_f64 类型的向量 a 和 b 各自四舍五入为 npyv_s32 类型的向量，并返回 a 的结果
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__trunc_s32_f64(vec_rint(a), vec_rint(b)); }

// 结束当前头文件中的条件编译块，对应于前面的 #if NPY_SIMD_F32
#endif // _NPY_SIMD_VEC_CVT_H
```