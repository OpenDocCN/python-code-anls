# `.\numpy\numpy\_core\src\common\simd\vec\memory.h`

```py
#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_VEC_MEMORY_H
#define _NPY_SIMD_VEC_MEMORY_H

#include "misc.h"

/****************************
 * Private utilities
 ****************************/

// TODO: test load by cast
// 定义是否使用类型转换加载数据的宏，如果为真则使用类型转换加载
#define VSX__CAST_lOAD 0
#if VSX__CAST_lOAD
    // 使用类型转换加载数据
    #define npyv__load(T_VEC, PTR) (*((T_VEC*)(PTR)))
#else
    /**
     * CLANG fails to load unaligned addresses via vec_xl, vec_xst
     * so we failback to vec_vsx_ld, vec_vsx_st
     */
    // 在 CLANG 中无法通过 vec_xl、vec_xst 加载非对齐地址，因此回退到 vec_vsx_ld、vec_vsx_st
    #if defined (NPY_HAVE_VSX2) && ( \
        (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__)) \
    )
        #define npyv__load(T_VEC, PTR) vec_vsx_ld(0, PTR)
    #else // VX
        #define npyv__load(T_VEC, PTR) vec_xl(0, PTR)
    #endif
#endif

// unaligned store
// 非对齐存储宏定义
#if defined (NPY_HAVE_VSX2) && ( \
    (defined(__GNUC__) && !defined(vec_xl)) || (defined(__clang__) && !defined(__IBMC__)) \
)
    #define npyv__store(PTR, VEC) vec_vsx_st(VEC, 0, PTR)
#else // VX
    #define npyv__store(PTR, VEC) vec_xst(VEC, 0, PTR)
#endif

// aligned load/store
// 对齐加载/存储宏定义
#if defined (NPY_HAVE_VSX)
    #define npyv__loada(PTR) vec_ld(0, PTR)
    #define npyv__storea(PTR, VEC) vec_st(VEC, 0, PTR)
#else // VX
    #define npyv__loada(PTR) vec_xl(0, PTR)
    #define npyv__storea(PTR, VEC) vec_xst(VEC, 0, PTR)
#endif

// avoid aliasing rules
// 避免别名规则，将指针转换为 64 位无符号整数指针
NPY_FINLINE npy_uint64 *npyv__ptr2u64(const void *ptr)
{ npy_uint64 *ptr64 = (npy_uint64*)ptr; return ptr64; }

// load lower part
// 加载低部分宏定义
NPY_FINLINE npyv_u64 npyv__loadl(const void *ptr)
{
#ifdef NPY_HAVE_VSX
    #if defined(__clang__) && !defined(__IBMC__)
        // vec_promote doesn't support doubleword on clang
        // 在 clang 中，vec_promote 不支持双字
        return npyv_setall_u64(*npyv__ptr2u64(ptr));
    #else
        return vec_promote(*npyv__ptr2u64(ptr), 0);
    #endif
#else // VX
    return vec_load_len((const unsigned long long*)ptr, 7);
#endif
}

// store lower part
// 存储低部分宏定义
#define npyv__storel(PTR, VEC) \
    *npyv__ptr2u64(PTR) = vec_extract(((npyv_u64)VEC), 0)

#define npyv__storeh(PTR, VEC) \
    *npyv__ptr2u64(PTR) = vec_extract(((npyv_u64)VEC), 1)

/****************************
 * load/store
 ****************************/

// 实现向量内存加载/存储的宏定义，包含不同数据宽度的加载函数
#define NPYV_IMPL_VEC_MEM(SFX, DW_CAST)                                                 \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const npyv_lanetype_##SFX *ptr)              \
    { return (npyv_##SFX)npyv__load(npyv_##SFX, (const npyv_lanetype_##DW_CAST*)ptr); } \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const npyv_lanetype_##SFX *ptr)             \
    { return (npyv_##SFX)npyv__loada((const npyv_lanetype_u32*)ptr); }                  \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const npyv_lanetype_##SFX *ptr)             \
    { return npyv_loada_##SFX(ptr); }                                                   \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const npyv_lanetype_##SFX *ptr)             \
    { return (npyv_##SFX)npyv__loadl(ptr); }

#endif // _NPY_SIMD_VEC_MEMORY_H
    # 存储函数定义，将向量数据存储到指定地址
    
    NPY_FINLINE void npyv_store_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)
    { npyv__store((npyv_lanetype_##DW_CAST*)ptr, (npyv_##DW_CAST)vec); }
    
    # 存储对齐函数定义，将向量数据按对齐方式存储到指定地址
    
    NPY_FINLINE void npyv_storea_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)
    { npyv__storea((npyv_lanetype_u32*)ptr, (npyv_u32)vec); }
    
    # 存储对齐函数的简写定义，将向量数据按对齐方式存储到指定地址
    
    NPY_FINLINE void npyv_stores_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)
    { npyv_storea_##SFX(ptr, vec); }
    
    # 低位存储函数定义，将向量低位数据存储到指定地址
    
    NPY_FINLINE void npyv_storel_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)
    { npyv__storel(ptr, vec); }
    
    # 高位存储函数定义，将向量高位数据存储到指定地址
    
    NPY_FINLINE void npyv_storeh_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)
    { npyv__storeh(ptr, vec); }
//// 定义宏以实现不连续加载

NPYV_IMPL_VEC_MEM(u8,  u8)  // 宏定义，生成用于加载/存储无符号8位整数向量的函数
NPYV_IMPL_VEC_MEM(s8,  s8)  // 宏定义，生成用于加载/存储有符号8位整数向量的函数
NPYV_IMPL_VEC_MEM(u16, u16) // 宏定义，生成用于加载/存储无符号16位整数向量的函数
NPYV_IMPL_VEC_MEM(s16, s16) // 宏定义，生成用于加载/存储有符号16位整数向量的函数
NPYV_IMPL_VEC_MEM(u32, u32) // 宏定义，生成用于加载/存储无符号32位整数向量的函数
NPYV_IMPL_VEC_MEM(s32, s32) // 宏定义，生成用于加载/存储有符号32位整数向量的函数
NPYV_IMPL_VEC_MEM(u64, f64) // 宏定义，生成用于加载/存储无符号64位整数向量的函数（实际是使用双精度浮点向量）
NPYV_IMPL_VEC_MEM(s64, f64) // 宏定义，生成用于加载/存储有符号64位整数向量的函数（实际是使用双精度浮点向量）
#if NPY_SIMD_F32
NPYV_IMPL_VEC_MEM(f32, f32) // 如果支持单精度浮点数SIMD，则生成用于加载/存储单精度浮点数向量的函数
#endif
NPYV_IMPL_VEC_MEM(f64, f64) // 宏定义，生成用于加载/存储双精度浮点数向量的函数

/***************************
 * Non-contiguous Load
 ***************************/

//// 32
// 以32位整数步长从内存加载向量
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{
    return npyv_set_u32(
        ptr[stride * 0], ptr[stride * 1],
        ptr[stride * 2], ptr[stride * 3]
    );
}

// 以32位整数步长从内存加载有符号32位整数向量
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{ return (npyv_s32)npyv_loadn_u32((const npy_uint32*)ptr, stride); }

#if NPY_SIMD_F32
// 如果支持单精度浮点SIMD，则以32位整数步长从内存加载单精度浮点数向量
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return (npyv_f32)npyv_loadn_u32((const npy_uint32*)ptr, stride); }
#endif

//// 64
// 以64位整数步长从内存加载无符号64位整数向量
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_set_u64(ptr[0], ptr[stride]); }

// 以64位整数步长从内存加载有符号64位整数向量
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_set_s64(ptr[0], ptr[stride]); }

// 以64位整数步长从内存加载双精度浮点数向量
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return npyv_set_f64(ptr[0], ptr[stride]); }

//// 64-bit load over 32-bit stride
// 以64位整数步长（使用32位整数指针）从内存加载无符号32位整数向量
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return (npyv_u32)npyv_set_u64(*(npy_uint64*)ptr, *(npy_uint64*)(ptr + stride)); }

// 以64位整数步长（使用32位整数指针）从内存加载有符号32位整数向量
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return (npyv_s32)npyv_set_u64(*(npy_uint64*)ptr, *(npy_uint64*)(ptr + stride)); }

#if NPY_SIMD_F32
// 如果支持单精度浮点SIMD，则以64位整数步长（使用单精度浮点数指针）从内存加载单精度浮点数向量
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return (npyv_f32)npyv_set_u64(*(npy_uint64*)ptr, *(npy_uint64*)(ptr + stride)); }
#endif

//// 128-bit load over 64-bit stride
// 以64位整数步长从内存加载无符号64位整数向量
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_u64(ptr); }

// 以64位整数步长从内存加载有符号64位整数向量
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ (void)stride; return npyv_load_s64(ptr); }

// 以64位整数步长从内存加载双精度浮点数向量
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ (void)stride; return npyv_load_f64(ptr); }

/***************************
 * Non-contiguous Store
 ***************************/

//// 32
// 以32位整数步长将向量存储到内存
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    ptr[stride * 0] = vec_extract(a, 0);
    ptr[stride * 1] = vec_extract(a, 1);
    ptr[stride * 2] = vec_extract(a, 2);
    ptr[stride * 3] = vec_extract(a, 3);
}

// 以32位整数步长将有符号32位整数向量存储到内存
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }

#if NPY_SIMD_F32
// 如果支持单精度浮点SIMD，则以32位整数步长将单精度浮点数向量存储到内存
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }
#endif

//// 64
// 以64位整数步长将无符号64位整数向量存储到内存
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    ptr[stride * 0] = vec_extract(a, 0);
}
    # 将向量 a 中索引为 1 的元素提取出来，并存入 ptr 数组的第 stride 个位置
    ptr[stride * 1] = vec_extract(a, 1);
//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    // 将向量a中的第一个64位元素存储到ptr指向的位置
    *(npy_uint64*)ptr = vec_extract((npyv_u64)a, 0);
    // 将向量a中的第二个64位元素存储到ptr指向的位置加上stride的位置
    *(npy_uint64*)(ptr + stride) = vec_extract((npyv_u64)a, 1);
}

NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ 
    // 调用npyv_storen2_u32函数将向量a转换为无符号32位整数向量，并存储到ptr指向的位置
    npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); 
}

#if NPY_SIMD_F32
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ 
    // 调用npyv_storen2_u32函数将向量a转换为无符号32位整数向量，并存储到ptr指向的位置
    npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); 
}
#endif

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ 
    // 忽略stride，直接调用npyv_store_u64函数将向量a存储到ptr指向的位置
    (void)stride; npyv_store_u64(ptr, a); 
}

NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ 
    // 忽略stride，直接调用npyv_store_s64函数将向量a存储到ptr指向的位置
    (void)stride; npyv_store_s64(ptr, a); 
}

NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ 
    // 忽略stride，直接调用npyv_store_f64函数将向量a存储到ptr指向的位置
    (void)stride; npyv_store_f64(ptr, a); 
}

/*********************************
 * Partial Load
 *********************************/

//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    // 确保nlane大于0
    assert(nlane > 0);
    // 使用fill值创建32位整数向量vfill
    npyv_s32 vfill = npyv_setall_s32(fill);

#ifdef NPY_HAVE_VX
    // 如果支持VX指令集，根据nlane的大小加载数据
    const unsigned blane = (nlane > 4) ? 4 : nlane;
    const npyv_u32 steps = npyv_set_u32(0, 1, 2, 3);
    const npyv_u32 vlane = npyv_setall_u32(blane);
    const npyv_b32 mask  = vec_cmpgt(vlane, steps);
    npyv_s32 a = vec_load_len(ptr, blane*4-1);
    a = vec_sel(vfill, a, mask);
#else
    // 如果不支持VX指令集，根据nlane的大小选择不同的加载方式
    npyv_s32 a;
    switch(nlane) {
    case 1:
        a = vec_insert(ptr[0], vfill, 0);
        break;
    case 2:
        a = (npyv_s32)vec_insert(
            *npyv__ptr2u64(ptr), (npyv_u64)vfill, 0
        );
        break;
    case 3:
        vfill = vec_insert(ptr[2], vfill, 2);
        a = (npyv_s32)vec_insert(
            *npyv__ptr2u64(ptr), (npyv_u64)vfill, 0
        );
        break;
    default:
        return npyv_load_s32(ptr);
    }
#endif

#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 在支持SIMD保护部分加载时，执行一个volatile操作
    volatile npyv_s32 workaround = a;
    a = vec_or(workaround, a);
#endif

    return a;
}

// 填充剩余的通道为零
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
#ifdef NPY_HAVE_VX
    // 如果支持VX指令集，根据nlane的大小加载数据，剩余通道填充为零
    unsigned blane = (nlane > 4) ? 4 : nlane;
    return vec_load_len(ptr, blane*4-1);
#else
    // 如果不支持VX指令集，调用npv_load_till_s32函数加载数据，剩余通道填充为零
    return npyv_load_till_s32(ptr, nlane, 0);
#endif
}

//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    // 确保nlane大于0
    assert(nlane > 0);
    if (nlane == 1) {
        // 当nlane为1时，使用ptr[0]和fill值创建64位整数向量r
        npyv_s64 r = npyv_set_s64(ptr[0], fill);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        // 在支持SIMD保护部分加载时，执行一个volatile操作
        volatile npyv_s64 workaround = r;
        r = vec_or(workaround, r);
    #endif
        return r;
    }
}
    // 调用 npyv_load_s64 函数，传入指针 ptr，并返回其结果
    return npyv_load_s64(ptr);
// 以内联方式加载直到指定数量的64位整数，如果启用了VX指令集，则使用SIMD指令进行加载
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
#ifdef NPY_HAVE_VX
    // 如果请求的元素数量大于2，则限制为2个
    unsigned blane = (nlane > 2) ? 2 : nlane;
    // 使用SIMD指令加载长度为 blane*8-1 的数据
    return vec_load_len((const signed long long*)ptr, blane*8-1);
#else
    // 否则，调用非零填充版本的加载函数
    return npyv_load_till_s64(ptr, nlane, 0);
#endif
}

//// 64-bit nlane
// 以内联方式加载直到指定数量的32位整数，可以指定填充值来填充剩余的数据
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane == 1) {
        // 如果只有一个元素，使用指定的填充值创建32位整数向量
        npyv_s32 r = npyv_set_s32(ptr[0], ptr[1], fill_lo, fill_hi);
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        // 在需要保护部分加载的情况下，使用volatile变量执行一个操作以避免编译器优化
        volatile npyv_s32 workaround = r;
        r = vec_or(workaround, r);
    #endif
        return r;
    }
    // 否则，调用普通的加载函数来加载数据
    return npyv_load_s32(ptr);
}

// fill zero to rest lanes
// 以内联方式加载直到指定数量的32位整数，并填充剩余的数据为零
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return (npyv_s32)npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

//// 128-bit nlane
// 以内联方式加载直到指定数量的64位整数，无条件返回64位整数向量
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{ (void)nlane; (void)fill_lo; (void)fill_hi; return npyv_load_s64(ptr); }

// fill zero to rest lanes
// 以内联方式加载直到指定数量的64位整数，并填充剩余的数据为零
NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ (void)nlane; return npyv_load_s64(ptr); }

/*********************************
 * Non-contiguous partial load
 *********************************/

//// 32
// 以内联方式加载非连续的32位整数向量，可以指定填充值来填充剩余的数据
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    // 使用指定的填充值创建一个32位整数向量
    npyv_s32 vfill = npyv_setall_s32(fill);
    switch(nlane) {
    case 3:
        // 如果请求的元素数量为3，则插入第三个元素并更新vfill
        vfill = vec_insert(ptr[stride*2], vfill, 2);
    case 2:
        // 如果请求的元素数量为2，则插入第二个元素并更新vfill
        vfill = vec_insert(ptr[stride], vfill, 1);
    case 1:
        // 插入第一个元素并更新vfill
        vfill = vec_insert(*ptr, vfill, 0);
        break;
    default:
        // 否则，调用非连续加载函数来加载数据
        return npyv_loadn_s32(ptr, stride);
    } // switch
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 在需要保护部分加载的情况下，使用volatile变量执行一个操作以避免编译器优化
    volatile npyv_s32 workaround = vfill;
    vfill = vec_or(workaround, vfill);
#endif
    return vfill;
}

// fill zero to rest lanes
// 以内联方式加载非连续的32位整数向量，并填充剩余的数据为零
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

//// 64
// 以内联方式加载非连续的64位整数向量，可以指定填充值来填充剩余的数据
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        // 如果请求的元素数量为1，则调用带填充参数的加载函数
        return npyv_load_till_s64(ptr, nlane, fill);
    }
    // 否则，调用非连续加载函数来加载数据
    return npyv_loadn_s64(ptr, stride);
}

// fill zero to rest lanes
// 以内联方式加载非连续的64位整数向量，并填充剩余的数据为零
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s64(ptr, stride, nlane, 0); }

//// 64-bit load over 32-bit stride
// 以内联方式加载非连续的32位整数向量，可以指定填充值来填充剩余的数据
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    // 省略了注释的部分...
    # 检查是否只有一个通道
    if (nlane == 1) {
        # 如果只有一个通道，使用给定的指针和填充值创建一个 SIMD 向量
        npyv_s32 r = npyv_set_s32(ptr[0], ptr[1], fill_lo, fill_hi);
        
        # 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行以下代码段
        # 将结果存储在 volatile 变量 workaround 中，然后使用逻辑或操作符合并 r 和 workaround
        # 这段代码可能用于某些 SIMD 指令集下的部分加载情况的处理
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = r;
        r = vec_or(workaround, r);
    #endif
        
        # 返回创建的 SIMD 向量 r
        return r;
    }
    
    # 如果不止一个通道，使用给定的指针和步长来加载 SIMD 向量
    return npyv_loadn2_s32(ptr, stride);
/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    // 确保待存储的元素个数大于零
    assert(nlane > 0);
#ifdef NPY_HAVE_VX
    // 如果支持向量化指令，计算实际需要存储的元素个数
    unsigned blane = (nlane > 4) ? 4 : nlane;
    // 使用向量化指令部分存储数据到ptr中
    vec_store_len(a, ptr, blane*4-1);
#else
    // 如果不支持向量化指令，根据元素个数执行不同的存储方式
    switch(nlane) {
    case 1:
        // 存储第一个元素到ptr中
        *ptr = vec_extract(a, 0);
        break;
    case 2:
        // 存储前两个元素到ptr中
        npyv_storel_s32(ptr, a);
        break;
    case 3:
        // 存储前两个元素到ptr中，再存储第三个元素到ptr[2]中
        npyv_storel_s32(ptr, a);
        ptr[2] = vec_extract(a, 2);
        break;
    default:
        // 存储所有元素到ptr中
        npyv_store_s32(ptr, a);
    }
#endif
}

//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    // 确保待存储的元素个数大于零
    assert(nlane > 0);
#ifdef NPY_HAVE_VX
    // 如果支持向量化指令，计算实际需要存储的元素个数
    unsigned blane = (nlane > 2) ? 2 : nlane;
    // 使用向量化指令部分存储数据到ptr中
    vec_store_len(a, (signed long long*)ptr, blane*8-1);
#else
    if (nlane == 1) {
        // 如果只有一个元素，直接存储到ptr中
        npyv_storel_s64(ptr, a);
        return;
    }
    // 否则，存储所有元素到ptr中
    npyv_store_s64(ptr, a);
#endif
}

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ 
    // 调用64位存储函数，将32位向量a的数据存储到64位ptr中
    npyv_store_till_s64((npy_int64*)ptr, nlane, (npyv_s64)a); 
}

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    // 确保待存储的元素个数大于零
    assert(nlane > 0); 
    // 直接存储所有元素到ptr中
    npyv_store_s64(ptr, a);
}

/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 确保待存储的元素个数大于零
    assert(nlane > 0);
    // 将向量a中的第一个元素存储到ptr中的第一个位置
    ptr[stride*0] = vec_extract(a, 0);
    // 根据元素个数执行不同的存储方式
    switch(nlane) {
    case 1:
        // 如果只有一个元素，直接返回
        return;
    case 2:
        // 存储第二个元素到ptr中的第二个位置
        ptr[stride*1] = vec_extract(a, 1);
        return;
    case 3:
        // 存储第二个元素到ptr中的第二个位置，存储第三个元素到ptr中的第三个位置
        ptr[stride*1] = vec_extract(a, 1);
        ptr[stride*2] = vec_extract(a, 2);
        return;
    default:
         // 存储第二到第四个元素到ptr中对应位置
         ptr[stride*1] = vec_extract(a, 1);
         ptr[stride*2] = vec_extract(a, 2);
         ptr[stride*3] = vec_extract(a, 3);
    }
}
//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    # 断言确保 nlane 大于 0，如果不是则触发 AssertionError
    assert(nlane > 0);
    # 如果 nlane 等于 1，则使用 npyv_storel_s64 函数将 a 存储到 ptr 指向的内存中，并返回
    if (nlane == 1) {
        npyv_storel_s64(ptr, a);
        return;
    }
    # 如果 nlane 大于 1，则使用 npyv_storen_s64 函数将长度为 nlane 的向量 a 存储到 ptr 指向的内存中，
    # 并按照指定的步长 stride 进行存储
    npyv_storen_s64(ptr, stride, a);
// Store 64-bit integer vector `a` into memory `ptr` with a 32-bit stride for `nlane` elements
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    // 断言确保要存储的元素数量大于0
    assert(nlane > 0);
    // 将低32位元素存储到内存地址ptr处
    npyv_storel_s32(ptr, a);
    // 如果元素数量大于1，则将高32位元素存储到内存地址ptr + stride处
    if (nlane > 1) {
        npyv_storeh_s32(ptr + stride, a);
    }
}

// Store 128-bit integer vector `a` into memory `ptr` with a 64-bit stride for `nlane` elements
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{ 
    // 断言确保要存储的元素数量大于0
    assert(nlane > 0);
    // 调用存储128位整数向量的函数，存储到内存地址ptr处
    npyv_store_s64(ptr, a); 
}

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
// 宏定义，用于生成多种类型的部分加载/存储函数
#define NPYV_IMPL_VEC_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                      \
    // 部分加载函数，将类型为`F_SFX`的向量部分加载到`npyv_##F_SFX`向量中
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX                                         \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_lanetype_##F_SFX fill)         \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        // 利用联合体进行类型转换，填充为`T_SFX`类型，用于部分加载函数调用
        pun.from_##F_SFX = fill;                                                            \
        // 返回重新解释类型后的向量加载结果
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    // 部分加载函数，带有步长`stride`，将类型为`F_SFX`的向量部分加载到`npyv_##F_SFX`向量中
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        // 返回重新解释类型后的加载数据，将类型 F_SFX 转换为类型 T_SFX
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    // 加载直到末尾为空的类型 F_SFX 向量
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 返回重新解释类型后的加载直到末尾为空的类型 F_SFX 向量
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    // 加载带步长直到末尾为空的类型 F_SFX 向量
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        // 返回重新解释类型后的加载带步长直到末尾为空的类型 F_SFX 向量
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    // 存储直到末尾的类型 F_SFX 向量
    NPY_FINLINE void npyv_store_till_##F_SFX                                                \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        // 存储直到末尾的重新解释类型后的类型 F_SFX 向量
        npyv_store_till_##T_SFX(                                                            \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }



    }                                                                                       \

这是一个 C 语言中的预处理器宏定义的结尾部分。


    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \

定义了一个内联函数 `npyv_storen_till_##F_SFX`，其返回类型为 `void`，接受四个参数：指向 `npyv_lanetype_##F_SFX` 类型的指针 `ptr`，一个整数 `stride`，一个无符号整数 `nlane`，以及一个 `npyv_##F_SFX` 类型的参数 `a`。


    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \

这一行是函数参数列表的起始部分。


        npyv_storen_till_##T_SFX(                                                           \

调用了另一个内联函数 `npyv_storen_till_##T_SFX`，其返回类型为 `void`，并传入参数：将 `ptr` 强制类型转换为 `npyv_lanetype_##T_SFX *`，`stride`，`nlane`，以及将 `a` 强制类型转换为 `npyv_reinterpret_##T_SFX##_##F_SFX(a)`。


            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \

传递给 `npyv_storen_till_##T_SFX` 函数的参数列表的起始部分。


            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \

调用 `npyv_reinterpret_##T_SFX##_##F_SFX` 宏，用于重新解释参数 `a` 的类型为 `T_SFX` 对应的类型。


        );                                                                                  \

`npyv_storen_till_##T_SFX` 函数调用的结尾。


    }

函数 `npyv_storen_till_##F_SFX` 的定义结尾。
// 定义宏，用于生成一系列的向量加载函数，处理不完整类型对 (F_SFX, T_SFX)
#define NPYV_IMPL_VEC_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                 \
    // 定义内联函数 npyv_load2_till_##F_SFX，加载 F_SFX 类型数据直至指定数量
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                    \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                 \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                      \
    {                                                                                   \
        // 定义联合结构 pun，用于转换 F_SFX 类型数据到 T_SFX 类型
        union pun {                                                                     \
            npyv_lanetype_##F_SFX from_##F_SFX;                                         \
            npyv_lanetype_##T_SFX to_##T_SFX;                                           \
        };                                                                              \
        // 定义联合变量 pun_lo 和 pun_hi，分别存储 fill_lo 和 fill_hi 的值
        union pun pun_lo;                                                               \
        union pun pun_hi;                                                               \
        pun_lo.from_##F_SFX = fill_lo;                                                  \
        pun_hi.from_##F_SFX = fill_hi;                                                  \
        // 调用 npyv_load2_till_##T_SFX 函数加载 T_SFX 类型数据，利用 reinterpret 转换为 F_SFX 类型
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(              \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                             \
    }                                                                                   \
    // 定义内联函数 npyv_loadn2_till_##F_SFX，加载 F_SFX 类型数据，带有步长 stride，直至指定数量
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_till_##F_SFX                                   \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                      \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        // 使用联合体 pun 进行类型转换，将 fill_lo 赋值给 from_##F_SFX
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        // 使用联合体 pun 进行类型转换，将 fill_hi 赋值给 from_##F_SFX
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        // 调用 npyv_loadn2_till_##T_SFX 函数加载数据，以 pun_lo.to_##T_SFX 和 pun_hi.to_##T_SFX 为参数
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \
            pun_hi.to_##T_SFX                                                               \
        ));                                                                                 \
    }                                                                                       \
    // 定义 NPY_FINLINE 内联函数，加载指定长度的数据并返回转换后的 npyv_##F_SFX 结果
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 调用 npyv_load2_tillz_##T_SFX 函数加载数据，将 ptr 强制类型转换为 const npyv_lanetype_##T_SFX 指针
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    // 定义 NPY_FINLINE 内联函数，加载带步长的指定长度数据并返回转换后的 npyv_##F_SFX 结果
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        // 调用 npyv_loadn2_tillz_##T_SFX 函数加载数据，将 ptr 强制类型转换为 const npyv_lanetype_##T_SFX 指针，传递 stride 和 nlane 作为参数
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    // 定义 NPY_FINLINE 内联函数，存储指定长度的 npyv_##F_SFX 数据到 ptr
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store2_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen2_till_##T_SFX(                                                          \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }



    {                                                                                       \
        // 调用特定类型 T_SFX 的向量存储函数，将向量 a 重新解释为类型 T_SFX，存储到 ptr 指向的内存中，处理直到 nlane
        npyv_store2_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    // 定义内联函数 npyv_storen2_till_##F_SFX，存储 F_SFX 类型的向量 a 到 ptr 指向的内存中，处理直到 nlane
    NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        // 调用特定类型 T_SFX 的向量存储函数，将向量 a 重新解释为类型 T_SFX，存储到 ptr 指向的内存中，以 stride 为步幅，处理直到 nlane
        npyv_storen2_till_##T_SFX(                                                          \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }
// 宏定义，生成多种类型的向量操作函数
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(u32, s32)
#if NPY_SIMD_F32
// 如果支持单精度浮点型 SIMD，则生成相应类型的向量操作函数
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(f32, s32)
#endif
// 生成多种类型的向量操作函数
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_VEC_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interlave load / interleave contiguous store
 ************************************************************/

// 定义向量内存交织操作的宏
#define NPYV_IMPL_VEC_MEM_INTERLEAVE(SFX)                                \
    // 对两个同类型向量执行交织操作
    NPY_FINLINE npyv_##SFX##x2 npyv_zip_##SFX(npyv_##SFX, npyv_##SFX);   \
    // 对两个同类型向量执行反交织操作
    NPY_FINLINE npyv_##SFX##x2 npyv_unzip_##SFX(npyv_##SFX, npyv_##SFX); \
    // 加载两个连续内存位置的向量并进行反交织
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                      \
        const npyv_lanetype_##SFX *ptr                                   \
    ) {                                                                  \
        return npyv_unzip_##SFX(                                         \
            npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX) \
        );                                                               \
    }                                                                    \
    // 存储两个连续内存位置的向量并进行交织
    NPY_FINLINE void npyv_store_##SFX##x2(                               \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                       \
    ) {                                                                  \
        npyv_##SFX##x2 zip = npyv_zip_##SFX(v.val[0], v.val[1]);         \
        npyv_store_##SFX(ptr, zip.val[0]);                               \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);           \
    }

// 生成多种类型的向量内存交织操作函数
NPYV_IMPL_VEC_MEM_INTERLEAVE(u8)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s8)
NPYV_IMPL_VEC_MEM_INTERLEAVE(u16)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s16)
NPYV_IMPL_VEC_MEM_INTERLEAVE(u32)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s32)
NPYV_IMPL_VEC_MEM_INTERLEAVE(u64)
NPYV_IMPL_VEC_MEM_INTERLEAVE(s64)
#if NPY_SIMD_F32
// 如果支持单精度浮点型 SIMD，则生成单精度向量内存交织操作函数
NPYV_IMPL_VEC_MEM_INTERLEAVE(f32)
#endif
// 生成双精度向量内存交织操作函数
NPYV_IMPL_VEC_MEM_INTERLEAVE(f64)

/*********************************
 * Lookup table
 *********************************/

// 使用向量作为索引来查找包含32个float32元素的表
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{
    // 提取向量中的每个索引值
    const unsigned i0 = vec_extract(idx, 0);
    const unsigned i1 = vec_extract(idx, 1);
    const unsigned i2 = vec_extract(idx, 2);
    const unsigned i3 = vec_extract(idx, 3);
    // 使用表中的值来填充向量，并返回结果向量
    npyv_u32 r = vec_promote(table[i0], 0);
             r = vec_insert(table[i1], r, 1);
             r = vec_insert(table[i2], r, 2);
             r = vec_insert(table[i3], r, 3);
    return r;
}

// 使用向量作为索引来查找包含32个int32元素的表
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return (npyv_s32)npyv_lut32_u32((const npy_uint32*)table, idx); }

#if NPY_SIMD_F32
// 如果支持单精度浮点型 SIMD，则使用向量作为索引来查找包含32个float元素的表
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return (npyv_f32)npyv_lut32_u32((const npy_uint32*)table, idx); }
#endif
// 使用向量作为索引来查找包含16个float64元素的表
#ifdef NPY_HAVE_VX
    // 如果定义了 NPY_HAVE_VX 宏，则使用 vec_extract 提取 idx 的第二个和第四个元素作为索引
    const unsigned i0 = vec_extract((npyv_u32)idx, 1);
    const unsigned i1 = vec_extract((npyv_u32)idx, 3);
#else
    // 否则，使用 vec_extract 提取 idx 的第一个和第三个元素作为索引
    const unsigned i0 = vec_extract((npyv_u32)idx, 0);
    const unsigned i1 = vec_extract((npyv_u32)idx, 2);
#endif

    // 使用 i0 索引从 table 数组中读取一个 double 值，并将其扩展为 npyv_f64 类型的向量 r
    npyv_f64 r = vec_promote(table[i0], 0);
    // 使用 i1 索引从 table 数组中读取一个 double 值，并将其插入到向量 r 的第二个位置
    r = vec_insert(table[i1], r, 1);
    
    // 返回构造好的向量 r，作为 lut16_f64 函数的结果
    return r;
}

// npyv_lut16_u64 函数，将 npy_uint64 类型的表格和索引传递给 lut16_f64 函数，并将结果转换为 npyv_u64 类型
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{
    return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx));
}

// npyv_lut16_s64 函数，将 npy_int64 类型的表格和索引传递给 lut16_f64 函数，并将结果转换为 npyv_s64 类型
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{
    return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx));
}
```