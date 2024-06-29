# `.\numpy\numpy\_core\src\common\simd\vec\operators.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif
// 如果未定义 NPY_SIMD 宏，则产生错误消息 "Not a standalone header"

#ifndef _NPY_SIMD_VEC_OPERATORS_H
#define _NPY_SIMD_VEC_OPERATORS_H
// 如果未定义 _NPY_SIMD_VEC_OPERATORS_H 宏，则定义该宏，避免重复包含

/***************************
 * Shifting
 ***************************/

// Left
#define npyv_shl_u16(A, C) vec_sl(A, npyv_setall_u16(C))
// 定义无符号 16 位整数向左移位操作的宏，使用 vec_sl 函数

#define npyv_shl_s16(A, C) vec_sl_s16(A, npyv_setall_u16(C))
// 定义有符号 16 位整数向左移位操作的宏，使用 vec_sl_s16 函数

#define npyv_shl_u32(A, C) vec_sl(A, npyv_setall_u32(C))
// 定义无符号 32 位整数向左移位操作的宏，使用 vec_sl 函数

#define npyv_shl_s32(A, C) vec_sl_s32(A, npyv_setall_u32(C))
// 定义有符号 32 位整数向左移位操作的宏，使用 vec_sl_s32 函数

#define npyv_shl_u64(A, C) vec_sl(A, npyv_setall_u64(C))
// 定义无符号 64 位整数向左移位操作的宏，使用 vec_sl 函数

#define npyv_shl_s64(A, C) vec_sl_s64(A, npyv_setall_u64(C))
// 定义有符号 64 位整数向左移位操作的宏，使用 vec_sl_s64 函数

// Left by an immediate constant
#define npyv_shli_u16 npyv_shl_u16
// 定义无符号 16 位整数按立即常数向左移位操作的宏，与 npyv_shl_u16 相同

#define npyv_shli_s16 npyv_shl_s16
// 定义有符号 16 位整数按立即常数向左移位操作的宏，与 npyv_shl_s16 相同

#define npyv_shli_u32 npyv_shl_u32
// 定义无符号 32 位整数按立即常数向左移位操作的宏，与 npyv_shl_u32 相同

#define npyv_shli_s32 npyv_shl_s32
// 定义有符号 32 位整数按立即常数向左移位操作的宏，与 npyv_shl_s32 相同

#define npyv_shli_u64 npyv_shl_u64
// 定义无符号 64 位整数按立即常数向左移位操作的宏，与 npyv_shl_u64 相同

#define npyv_shli_s64 npyv_shl_s64
// 定义有符号 64 位整数按立即常数向左移位操作的宏，与 npyv_shl_s64 相同

// Right
#define npyv_shr_u16(A, C) vec_sr(A,  npyv_setall_u16(C))
// 定义无符号 16 位整数向右移位操作的宏，使用 vec_sr 函数

#define npyv_shr_s16(A, C) vec_sra_s16(A, npyv_setall_u16(C))
// 定义有符号 16 位整数向右移位操作的宏，使用 vec_sra_s16 函数

#define npyv_shr_u32(A, C) vec_sr(A,  npyv_setall_u32(C))
// 定义无符号 32 位整数向右移位操作的宏，使用 vec_sr 函数

#define npyv_shr_s32(A, C) vec_sra_s32(A, npyv_setall_u32(C))
// 定义有符号 32 位整数向右移位操作的宏，使用 vec_sra_s32 函数

#define npyv_shr_u64(A, C) vec_sr(A,  npyv_setall_u64(C))
// 定义无符号 64 位整数向右移位操作的宏，使用 vec_sr 函数

#define npyv_shr_s64(A, C) vec_sra_s64(A, npyv_setall_u64(C))
// 定义有符号 64 位整数向右移位操作的宏，使用 vec_sra_s64 函数

// Right by an immediate constant
#define npyv_shri_u16 npyv_shr_u16
// 定义无符号 16 位整数按立即常数向右移位操作的宏，与 npyv_shr_u16 相同

#define npyv_shri_s16 npyv_shr_s16
// 定义有符号 16 位整数按立即常数向右移位操作的宏，与 npyv_shr_s16 相同

#define npyv_shri_u32 npyv_shr_u32
// 定义无符号 32 位整数按立即常数向右移位操作的宏，与 npyv_shr_u32 相同

#define npyv_shri_s32 npyv_shr_s32
// 定义有符号 32 位整数按立即常数向右移位操作的宏，与 npyv_shr_s32 相同

#define npyv_shri_u64 npyv_shr_u64
// 定义无符号 64 位整数按立即常数向右移位操作的宏，与 npyv_shr_u64 相同

#define npyv_shri_s64 npyv_shr_s64
// 定义有符号 64 位整数按立即常数向右移位操作的宏，与 npyv_shr_s64 相同

/***************************
 * Logical
 ***************************/

#define NPYV_IMPL_VEC_BIN_CAST(INTRIN, SFX, CAST) \
    NPY_FINLINE npyv_##SFX npyv_##INTRIN##_##SFX(npyv_##SFX a, npyv_##SFX b) \
    { return (npyv_##SFX)vec_##INTRIN((CAST)a, (CAST)b); }
// 定义通用的二元逻辑运算的模板宏，使用 NPY_FINLINE 进行内联优化

// Up to GCC 6 logical intrinsics don't support bool long long
#if defined(__GNUC__) && __GNUC__ <= 6
    #define NPYV_IMPL_VEC_BIN_B64(INTRIN) NPYV_IMPL_VEC_BIN_CAST(INTRIN, b64, npyv_u64)
#else
    #define NPYV_IMPL_VEC_BIN_B64(INTRIN) NPYV_IMPL_VEC_BIN_CAST(INTRIN, b64, npyv_b64)
#endif
// 根据 GCC 编译器版本，选择适当的 64 位整数类型，避免布尔和长长整数支持问题

// AND
#define npyv_and_u8  vec_and
// 定义无符号 8 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_s8  vec_and
// 定义有符号 8 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_u16 vec_and
// 定义无符号 16 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_s16 vec_and
// 定义有符号 16 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_u32 vec_and
// 定义无符号 32 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_s32 vec_and
// 定义有符号 32 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_u64 vec_and
// 定义无符号 64 位整数按位与操作的宏，使用 vec_and 函数

#define npyv_and_s64 vec_and
// 定义有符号 64 位整数按位与操作的宏，使用 vec_and 函数

#if NPY_SIMD_F32
    #define npyv_and_f32 vec_and
#endif
// 如果支持单精度浮点
#define npyv_xor_s32 vec_xor
#define npyv_xor_u64 vec_xor
#define npyv_xor_s64 vec_xor
#if NPY_SIMD_F32
    #define npyv_xor_f32 vec_xor
#endif
#define npyv_xor_f64 vec_xor
#define npyv_xor_b8  vec_xor
#define npyv_xor_b16 vec_xor
#define npyv_xor_b32 vec_xor
NPYV_IMPL_VEC_BIN_B64(xor)


// 定义了多个宏，用于执行不同类型数据的按位异或操作，具体操作由 vec_xor 完成
#define npyv_xor_s32 vec_xor      // 32位有符号整数按位异或
#define npyv_xor_u64 vec_xor      // 64位无符号整数按位异或
#define npyv_xor_s64 vec_xor      // 64位有符号整数按位异或
#if NPY_SIMD_F32
    #define npyv_xor_f32 vec_xor  // 单精度浮点数按位异或（条件编译）
#endif
#define npyv_xor_f64 vec_xor      // 双精度浮点数按位异或
#define npyv_xor_b8  vec_xor      // 8位布尔型按位异或
#define npyv_xor_b16 vec_xor      // 16位布尔型按位异或
#define npyv_xor_b32 vec_xor      // 32位布尔型按位异或
NPYV_IMPL_VEC_BIN_B64(xor)      // 使用 vec_xor 实现64位布尔型按位异或


// NOT
// 注意：我们实现 npyv_not_b*(布尔类型) 供内部使用
#define NPYV_IMPL_VEC_NOT_INT(VEC_LEN)                                 \
    NPY_FINLINE npyv_u##VEC_LEN npyv_not_u##VEC_LEN(npyv_u##VEC_LEN a) \
    { return vec_nor(a, a); }                                          \
    NPY_FINLINE npyv_s##VEC_LEN npyv_not_s##VEC_LEN(npyv_s##VEC_LEN a) \
    { return vec_nor(a, a); }                                          \
    NPY_FINLINE npyv_b##VEC_LEN npyv_not_b##VEC_LEN(npyv_b##VEC_LEN a) \
    { return vec_nor(a, a); }


// 定义了一个宏 NPYV_IMPL_VEC_NOT_INT，用于实现多个整型和布尔型数据的按位非操作，具体操作由 vec_nor 完成
#define NPYV_IMPL_VEC_NOT_INT(VEC_LEN)                                 \
    NPY_FINLINE npyv_u##VEC_LEN npyv_not_u##VEC_LEN(npyv_u##VEC_LEN a) \
    { return vec_nor(a, a); }                                          \
    NPY_FINLINE npyv_s##VEC_LEN npyv_not_s##VEC_LEN(npyv_s##VEC_LEN a) \
    { return vec_nor(a, a); }                                          \
    NPY_FINLINE npyv_b##VEC_LEN npyv_not_b##VEC_LEN(npyv_b##VEC_LEN a) \
    { return vec_nor(a, a); }


NPYV_IMPL_VEC_NOT_INT(8)
NPYV_IMPL_VEC_NOT_INT(16)
NPYV_IMPL_VEC_NOT_INT(32)


// 使用宏 NPYV_IMPL_VEC_NOT_INT 分别实现8位、16位和32位整型和布尔型数据的按位非操作
NPYV_IMPL_VEC_NOT_INT(8)
NPYV_IMPL_VEC_NOT_INT(16)
NPYV_IMPL_VEC_NOT_INT(32)


// 在 ppc64 上，直到 gcc5，vec_nor 不支持布尔型长长整数
#if defined(NPY_HAVE_VSX) && defined(__GNUC__) && __GNUC__ > 5
    NPYV_IMPL_VEC_NOT_INT(64)
#else
    NPY_FINLINE npyv_u64 npyv_not_u64(npyv_u64 a)
    { return vec_nor(a, a); }
    NPY_FINLINE npyv_s64 npyv_not_s64(npyv_s64 a)
    { return vec_nor(a, a); }
    NPY_FINLINE npyv_b64 npyv_not_b64(npyv_b64 a)
    { return (npyv_b64)vec_nor((npyv_u64)a, (npyv_u64)a); }
#endif


// 根据条件，分别实现64位整型和布尔型数据的按位非操作
#if defined(NPY_HAVE_VSX) && defined(__GNUC__) && __GNUC__ > 5
    NPYV_IMPL_VEC_NOT_INT(64)  // 使用 NPYV_IMPL_VEC_NOT_INT 宏实现
#else
    NPY_FINLINE npyv_u64 npyv_not_u64(npyv_u64 a)
    { return vec_nor(a, a); }   // 64位无符号整数按位非
    NPY_FINLINE npyv_s64 npyv_not_s64(npyv_s64 a)
    { return vec_nor(a, a); }   // 64位有符号整数按位非
    NPY_FINLINE npyv_b64 npyv_not_b64(npyv_b64 a)
    { return (npyv_b64)vec_nor((npyv_u64)a, (npyv_u64)a); }  // 64位布尔型按位非
#endif


#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_not_f32(npyv_f32 a)
    { return vec_nor(a, a); }
#endif
NPY_FINLINE npyv_f64 npyv_not_f64(npyv_f64 a)
{ return vec_nor(a, a); }


// 如果支持单精度浮点数SIMD，实现单精度浮点数的按位非操作，否则忽略
#if NPY_SIMD_F32
    NPY_FINLINE npyv_f32 npyv_not_f32(npyv_f32 a)
    { return vec_nor(a, a); }  // 单精度浮点数按位非
#endif
NPY_FINLINE npyv_f64 npyv_not_f64(npyv_f64 a)
{ return vec_nor(a, a); }        // 双精度浮点数按位非


// ANDC, ORC and XNOR
#define npyv_andc_u8 vec_andc
#define npyv_andc_b8 vec_andc
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define npyv_orc_b8 vec_orc
    #define npyv_xnor_b8 vec_eqv
#else
    #define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)
    #define npyv_xnor_b8(A, B) npyv_not_b8(npyv_xor_b8(B, A))
#endif


// 定义了多个宏，用于执行不同类型数据的按位ANDC、ORC和XNOR操作，具体操作由 vec_andc、vec_orc 和 vec_eqv 完成
#define npyv_andc_u8 vec_andc      // 8位无符号整数按位ANDC
#define npyv_andc_b8 vec_andc      // 8位布尔型按位ANDC
#if defined(NPY_HAVE_VXE) || defined(NPY_HAVE_VSX)
    #define npyv_orc_b8 vec_orc    // 如果支持 VXE 或 VSX，8位布尔型按位ORC
    #define npyv_xnor_b8 vec_eqv   // 如果支持 VXE 或 VSX，8位布尔型按位XNOR
#else
    #define npyv_orc_b8(A, B) npyv_or_b8(npyv_not_b8(B), A)     // 否则，通过按位OR和按位NOT实现8位布尔型按位ORC
    #define npyv_xnor_b8(A, B) npyv_not_b8(npyv_xor_b8(B, A))   // 否则，通过按位XOR和按位NOT实现8位布尔型按位XNOR
#endif


/***************************
 * Comparison
 ***************************/

// Int Equal
#define npyv_cmpeq_u8  vec_cmpeq
#define npyv_cmpeq_s8  vec_cmpeq
#define npyv_cmpeq_u16 vec_cmpeq
#define npyv_cmpeq_s16 vec_cmpeq
#define npyv_cmpeq_u32 vec_cmpeq
#define npyv
    #define npyv_cmpneq_u16(A, B) npyv_not_b16(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个无符号16位整数向量A和B的元素是否不相等，返回结果取反
    
    #define npyv_cmpneq_s16(A, B) npyv_not_b16(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个有符号16位整数向量A和B的元素是否不相等，返回结果取反
    
    #define npyv_cmpneq_u32(A, B) npyv_not_b32(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个无符号32位整数向量A和B的元素是否不相等，返回结果取反
    
    #define npyv_cmpneq_s32(A, B) npyv_not_b32(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个有符号32位整数向量A和B的元素是否不相等，返回结果取反
    
    #define npyv_cmpneq_u64(A, B) npyv_not_b64(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个无符号64位整数向量A和B的元素是否不相等，返回结果取反
    
    #define npyv_cmpneq_s64(A, B) npyv_not_b64(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个有符号64位整数向量A和B的元素是否不相等，返回结果取反
    
    #if NPY_SIMD_F32
        #define npyv_cmpneq_f32(A, B) npyv_not_b32(vec_cmpeq(A, B))
    #endif
    // 如果支持32位单精度浮点数SIMD指令，则定义一个宏，用于比较两个单精度浮点数向量A和B的元素是否不相等，返回结果取反
    
    #define npyv_cmpneq_f64(A, B) npyv_not_b64(vec_cmpeq(A, B))
    // 定义一个宏，用于比较两个双精度浮点数向量A和B的元素是否不相等，返回结果取反
#endif

// Greater than
// 定义无符号8位整数向量的大于比较操作宏
#define npyv_cmpgt_u8  vec_cmpgt
// 定义有符号8位整数向量的大于比较操作宏
#define npyv_cmpgt_s8  vec_cmpgt
// 定义无符号16位整数向量的大于比较操作宏
#define npyv_cmpgt_u16 vec_cmpgt
// 定义有符号16位整数向量的大于比较操作宏
#define npyv_cmpgt_s16 vec_cmpgt
// 定义无符号32位整数向量的大于比较操作宏
#define npyv_cmpgt_u32 vec_cmpgt
// 定义有符号32位整数向量的大于比较操作宏
#define npyv_cmpgt_s32 vec_cmpgt
// 定义无符号64位整数向量的大于比较操作宏
#define npyv_cmpgt_u64 vec_cmpgt
// 定义有符号64位整数向量的大于比较操作宏
#define npyv_cmpgt_s64 vec_cmpgt
// 如果支持单精度浮点数向量，则定义单精度浮点数向量的大于比较操作宏
#if NPY_SIMD_F32
    #define npyv_cmpgt_f32 vec_cmpgt
#endif
// 定义双精度浮点数向量的大于比较操作宏
#define npyv_cmpgt_f64 vec_cmpgt

// Greater than or equal
// 如果支持向量扩展(VX)或者GCC版本大于5，则定义各类型向量的大于等于比较操作宏
#if defined(NPY_HAVE_VX) || (defined(__GNUC__) && __GNUC__ > 5)
    #define npyv_cmpge_u8  vec_cmpge
    #define npyv_cmpge_s8  vec_cmpge
    #define npyv_cmpge_u16 vec_cmpge
    #define npyv_cmpge_s16 vec_cmpge
    #define npyv_cmpge_u32 vec_cmpge
    #define npyv_cmpge_s32 vec_cmpge
    #define npyv_cmpge_u64 vec_cmpge
    #define npyv_cmpge_s64 vec_cmpge
// 否则，使用大于比较操作宏来实现各类型向量的大于等于比较操作宏
#else
    #define npyv_cmpge_u8(A, B)  npyv_not_b8(vec_cmpgt(B, A))
    #define npyv_cmpge_s8(A, B)  npyv_not_b8(vec_cmpgt(B, A))
    #define npyv_cmpge_u16(A, B) npyv_not_b16(vec_cmpgt(B, A))
    #define npyv_cmpge_s16(A, B) npyv_not_b16(vec_cmpgt(B, A))
    #define npyv_cmpge_u32(A, B) npyv_not_b32(vec_cmpgt(B, A))
    #define npyv_cmpge_s32(A, B) npyv_not_b32(vec_cmpgt(B, A))
    #define npyv_cmpge_u64(A, B) npyv_not_b64(vec_cmpgt(B, A))
    #define npyv_cmpge_s64(A, B) npyv_not_b64(vec_cmpgt(B, A))
#endif
// 如果支持单精度浮点数向量，则定义单精度浮点数向量的大于等于比较操作宏
#if NPY_SIMD_F32
    #define npyv_cmpge_f32 vec_cmpge
#endif
// 定义双精度浮点数向量的大于等于比较操作宏
#define npyv_cmpge_f64 vec_cmpge

// Less than
// 定义无符号8位整数向量的小于比较操作宏
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
// 定义有符号8位整数向量的小于比较操作宏
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
// 定义无符号16位整数向量的小于比较操作宏
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
// 定义有符号16位整数向量的小于比较操作宏
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
// 定义无符号32位整数向量的小于比较操作宏
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
// 定义有符号32位整数向量的小于比较操作宏
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
// 定义无符号64位整数向量的小于比较操作宏
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
// 定义有符号64位整数向量的小于比较操作宏
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)
// 如果支持单精度浮点数向量，则定义单精度浮点数向量的小于比较操作宏
#if NPY_SIMD_F32
    #define npyv_cmplt_f32(A, B) npyv_cmpgt_f32(B, A)
#endif
// 定义双精度浮点数向量的小于比较操作宏
#define npyv_cmplt_f64(A, B) npyv_cmpgt_f64(B, A)

// Less than or equal
// 定义无符号8位整数向量的小于等于比较操作宏
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
// 定义有符号8位整数向量的小于等于比较操作宏
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
// 定义无符号16位整数向量的小于等于比较操作宏
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
// 定义有符号16位整数向量的小于等于比较操作宏
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
// 定义无符号32位整数向量的小于等于比较操作宏
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
// 定义有符号32位整数向量的小于等于比较操作宏
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
// 定义无符号64位整数向量的小于等于比较操作宏
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
// 定义有符号64位整数向量的小于等于比较操作宏
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)
// 如果支持单精度浮点数向量，则定义单精度浮点数向量的小于等于比较操作宏
#if NPY_SIMD_F32
    #define npyv_cmple_f32(A, B) npyv_cmpge_f32(B, A)
#endif
// 定义双精度浮点数向量的小于等于比较操作宏
#define npyv_cmple_f64(A, B) npyv_cmpge_f64(B, A)

// check special cases
// 如果支持单精度浮点数向量，则定义检查单精度浮点数向量中非NaN值的宏
#if NPY_SIMD_F32
    NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
    { return vec_cmpeq(a, a); }
#endif
// 定义检查双精度浮点数向量中非NaN值的宏
NPY_FIN
    # 定义一个内联函数，检查向量中是否存在非零元素
    NPY_FINLINE bool npyv_any_##SFX(npyv_##SFX a)             \
    { return vec_any_ne(a, (npyv_##SFX)npyv_zero_##SFX2()); }
    # 定义一个内联函数，检查向量中是否所有元素都是非零的
    NPY_FINLINE bool npyv_all_##SFX(npyv_##SFX a)             \
    { return vec_all_ne(a, (npyv_##SFX)npyv_zero_##SFX2()); }
# 定义宏 NPYV_IMPL_VEC_ANYALL，用于生成各种类型的 SIMD 向量操作函数，处理布尔型（b8, b16, b32, b64）或整数型（u8, s8, u16, s16, u32, s32, u64, s64）数据
NPYV_IMPL_VEC_ANYALL(b8,  u8)    // 生成处理布尔型 b8 和无符号整数型 u8 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(b16, u16)   // 生成处理布尔型 b16 和无符号整数型 u16 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(b32, u32)   // 生成处理布尔型 b32 和无符号整数型 u32 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(b64, u64)   // 生成处理布尔型 b64 和无符号整数型 u64 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(u8,  u8)    // 生成处理无符号整数型 u8 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(s8,  s8)    // 生成处理有符号整数型 s8 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(u16, u16)   // 生成处理无符号整数型 u16 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(s16, s16)   // 生成处理有符号整数型 s16 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(u32, u32)   // 生成处理无符号整数型 u32 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(s32, s32)   // 生成处理有符号整数型 s32 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(u64, u64)   // 生成处理无符号整数型 u64 的 SIMD 向量操作函数
NPYV_IMPL_VEC_ANYALL(s64, s64)   // 生成处理有符号整数型 s64 的 SIMD 向量操作函数
#if NPY_SIMD_F32
    NPYV_IMPL_VEC_ANYALL(f32, f32)   // 生成处理单精度浮点数型 f32 的 SIMD 向量操作函数（条件编译）
#endif
NPYV_IMPL_VEC_ANYALL(f64, f64)   // 生成处理双精度浮点数型 f64 的 SIMD 向量操作函数
#undef NPYV_IMPL_VEC_ANYALL         // 取消定义宏 NPYV_IMPL_VEC_ANYALL，结束对 SIMD 向量操作函数的定义

#endif // _NPY_SIMD_VEC_OPERATORS_H   // 结束 SIMD 向量操作函数头文件的条件编译部分
```