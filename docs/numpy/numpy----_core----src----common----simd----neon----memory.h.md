# `.\numpy\numpy\_core\src\common\simd\neon\memory.h`

```py
#ifndef NPY_SIMD
    // 如果未定义 NPY_SIMD 宏，则报错，此头文件不是独立可用的
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_NEON_MEMORY_H
#define _NPY_SIMD_NEON_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/
// GCC 需要对指针类型进行明确的字面类型定义，否则会导致模棱两可的错误
#define NPYV_IMPL_NEON_MEM(SFX, CTYPE)                                           \
    // 加载操作，从 ptr 指向的地址加载数据到向量 npyv_##SFX 中
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const npyv_lanetype_##SFX *ptr)       \
    { return vld1q_##SFX((const CTYPE*)ptr); }                                   \
    // 加载操作，从 ptr 指向的地址加载数据到向量 npyv_##SFX 中
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const npyv_lanetype_##SFX *ptr)      \
    { return vld1q_##SFX((const CTYPE*)ptr); }                                   \
    // 加载操作，从 ptr 指向的地址加载数据到向量 npyv_##SFX 中
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const npyv_lanetype_##SFX *ptr)      \
    { return vld1q_##SFX((const CTYPE*)ptr); }                                   \
    // 加载操作，从 ptr 指向的地址加载数据到向量 npyv_##SFX 中，低位补零
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const npyv_lanetype_##SFX *ptr)      \
    {                                                                            \
        return vcombine_##SFX(                                                   \
            vld1_##SFX((const CTYPE*)ptr), vdup_n_##SFX(0)                       \
        );                                                                       \
    }                                                                            \
    // 存储操作，将向量 vec 的数据存储到 ptr 指向的地址
    NPY_FINLINE void npyv_store_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec)  \
    { vst1q_##SFX((CTYPE*)ptr, vec); }                                           \
    // 存储操作，将向量 vec 的数据存储到 ptr 指向的地址
    NPY_FINLINE void npyv_storea_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1q_##SFX((CTYPE*)ptr, vec); }                                           \
    // 存储操作，将向量 vec 的数据存储到 ptr 指向的地址
    NPY_FINLINE void npyv_stores_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1q_##SFX((CTYPE*)ptr, vec); }                                           \
    // 存储操作，将向量 vec 的低位数据存储到 ptr 指向的地址
    NPY_FINLINE void npyv_storel_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1_##SFX((CTYPE*)ptr, vget_low_##SFX(vec)); }                            \
    // 存储操作，将向量 vec 的高位数据存储到 ptr 指向的地址
    NPY_FINLINE void npyv_storeh_##SFX(npyv_lanetype_##SFX *ptr, npyv_##SFX vec) \
    { vst1_##SFX((CTYPE*)ptr, vget_high_##SFX(vec)); }

// 定义各种数据类型的 NEON 内存操作宏
NPYV_IMPL_NEON_MEM(u8,  uint8_t)
NPYV_IMPL_NEON_MEM(s8,  int8_t)
NPYV_IMPL_NEON_MEM(u16, uint16_t)
NPYV_IMPL_NEON_MEM(s16, int16_t)
NPYV_IMPL_NEON_MEM(u32, uint32_t)
NPYV_IMPL_NEON_MEM(s32, int32_t)
NPYV_IMPL_NEON_MEM(u64, uint64_t)
NPYV_IMPL_NEON_MEM(s64, int64_t)
NPYV_IMPL_NEON_MEM(f32, float)
#if NPY_SIMD_F64
NPYV_IMPL_NEON_MEM(f64, double)
#endif

/***************************
 * Non-contiguous Load
 ***************************/
// 非连续加载操作，从 ptr 指向的地址开始，按照给定的步长 stride 加载数据到向量 npyv_s32 中
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    // 创建一个全为 0 的 128 位整型 NEON 向量
    int32x4_t a = vdupq_n_s32(0);
    // 分别从不同地址加载数据到向量的不同通道
    a = vld1q_lane_s32((const int32_t*)ptr,            a, 0);
    a = vld1q_lane_s32((const int32_t*)ptr + stride,   a, 1);
    a = vld1q_lane_s32((const int32_t*)ptr + stride*2, a, 2);
    a = vld1q_lane_s32((const int32_t*)ptr + stride*3, a, 3);


这段代码主要是 NEON SIMD 操作的宏定义和函数实现，用于在 ARM 架构下进行内存的加载和存储操作，支持不同数据类型和非连续加载。
    # 返回变量 a 的值作为函数的输出结果
    return a;
//// 64
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{
    // 使用 Neon 指令加载两个 int64_t 元素到一个 128 位寄存器中
    return vcombine_s64(
        vld1_s64((const int64_t*)ptr), vld1_s64((const int64_t*)ptr + stride)
    );
}
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{
    // 将加载的 int64_t 数据重新解释为 uint64_t 数据
    return npyv_reinterpret_u64_s64(
        npyv_loadn_s64((const npy_int64*)ptr, stride)
    );
}
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{
    // 将加载的 int64_t 数据重新解释为 double 数据
    return npyv_reinterpret_f64_s64(
        npyv_loadn_s64((const npy_int64*)ptr, stride)
    );
}
#endif

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{
    // 使用 Neon 指令加载两个 uint32_t 元素到一个 64 位寄存器中
    return vcombine_u32(
        vld1_u32((const uint32_t*)ptr), vld1_u32((const uint32_t*)ptr + stride)
    );
}
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ 
    // 将加载的 uint32_t 数据重新解释为 int32_t 数据
    return npyv_reinterpret_s32_u32(npyv_loadn2_u32((const npy_uint32*)ptr, stride));
}
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ 
    // 将加载的 uint32_t 数据重新解释为 float 数据
    return npyv_reinterpret_f32_u32(npyv_loadn2_u32((const npy_uint32*)ptr, stride));
}

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ 
    // 无需考虑 stride，直接加载 uint64_t 数据
    (void)stride; return npyv_load_u64(ptr); 
}
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ 
    // 无需考虑 stride，直接加载 int64_t 数据
    (void)stride; return npyv_load_s64(ptr); 
}
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ 
    // 无需考虑 stride，直接加载 double 数据
    (void)stride; return npyv_load_f64(ptr); 
}
#endif

/***************************
 * Non-contiguous Store
 ***************************/
//// 32
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    // 分别存储 4 个 int32_t 元素到指定位置，考虑 stride
    vst1q_lane_s32((int32_t*)ptr, a, 0);
    vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
    vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
    vst1q_lane_s32((int32_t*)ptr + stride*3, a, 3);
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ 
    // 将存储的 int32_t 数据重新解释为 uint32_t 数据
    npyv_storen_s32((npy_int32*)ptr, stride, npyv_reinterpret_s32_u32(a)); 
}
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ 
    // 将存储的 int32_t 数据重新解释为 float 数据
    npyv_storen_s32((npy_int32*)ptr, stride, npyv_reinterpret_s32_f32(a)); 
}

//// 64
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{
    // 分别存储 2 个 int64_t 元素到指定位置，考虑 stride
    vst1q_lane_s64((int64_t*)ptr, a, 0);
    vst1q_lane_s64((int64_t*)ptr + stride, a, 1);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ 
    // 将存储的 int64_t 数据重新解释为 uint64_t 数据
    npyv_storen_s64((npy_int64*)ptr, stride, npyv_reinterpret_s64_u64(a)); 
}

#if NPY_SIMD_F64
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
#if NPY_SIMD_F64
    // 使用 SIMD 指令将第一个 64 位数据存储到 ptr，a 被重新解释为 64 位数据
    vst1q_lane_u64((uint64_t*)ptr, npyv_reinterpret_u64_u32(a), 0);
    // 使用 SIMD 指令将第二个 64 位数据存储到 ptr + stride，a 被重新解释为 64 位数据
    vst1q_lane_u64((uint64_t*)(ptr + stride), npyv_reinterpret_u64_u32(a), 1);
#else
    // 在 armhf 环境中，要求对齐存储，将 a 的低 32 位存储到 ptr
    vst1_u32((uint32_t*)ptr, vget_low_u32(a));
    // 在 armhf 环境中，将 a 的高 32 位存储到 ptr + stride
    vst1_u32((uint32_t*)ptr + stride, vget_high_u32(a));
#endif
}

NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    // 将 npyv_s32 类型数据 a 重新解释为 npyv_u32 并进行存储
    npyv_storen2_u32((npy_uint32*)ptr, stride, npyv_reinterpret_u32_s32(a));
}

NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{
    // 将 npyv_f32 类型数据 a 重新解释为 npyv_u32 并进行存储
    npyv_storen2_u32((npy_uint32*)ptr, stride, npyv_reinterpret_u32_f32(a));
}

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    // 忽略 stride 参数，直接存储 a 的数据到 ptr
    (void)stride;
    npyv_store_u64(ptr, a);
}

NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{
    // 忽略 stride 参数，直接存储 a 的数据到 ptr
    (void)stride;
    npyv_store_s64(ptr, a);
}

#if NPY_SIMD_F64
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{
    // 忽略 stride 参数，直接存储 a 的数据到 ptr
    (void)stride;
    npyv_store_f64(ptr, a);
}
#endif

/*********************************
 * Partial Load
 *********************************/

//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    npyv_s32 a;
    switch(nlane) {
    case 1:
        // 加载单个 32 位数据到 a，使用 fill 填充剩余的 SIMD 矢量
        a = vld1q_lane_s32((const int32_t*)ptr, vdupq_n_s32(fill), 0);
        break;
    case 2:
        // 组合两个 32 位数据到 a，并使用 fill 填充剩余的 SIMD 矢量
        a = vcombine_s32(vld1_s32((const int32_t*)ptr), vdup_n_s32(fill));
        break;
    case 3:
        // 组合前两个 32 位数据到 a，第三个数据使用 fill 填充剩余的 SIMD 矢量
        a = vcombine_s32(
            vld1_s32((const int32_t*)ptr),
            vld1_lane_s32((const int32_t*)ptr + 2, vdup_n_s32(fill), 0)
        );
        break;
    default:
        // 加载所有的 32 位数据到 a
        return npyv_load_s32(ptr);
    }
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行一个 workaround 操作
    volatile npyv_s32 workaround = a;
    a = vorrq_s32(workaround, a);
#endif
    return a;
}

// 使用 0 填充剩余的 SIMD 矢量
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    return npyv_load_till_s32(ptr, nlane, 0);
}

//// 64
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == 1) {
        // 加载单个 64 位数据到 a，并使用 fill 填充剩余的 SIMD 矢量
        npyv_s64 a = vcombine_s64(vld1_s64((const int64_t*)ptr), vdup_n_s64(fill));
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行一个 workaround 操作
        volatile npyv_s64 workaround = a;
        a = vorrq_s64(workaround, a);
    #endif
        return a;
    }
    // 加载所有的 64 位数据到 a
    return npyv_load_s64(ptr);
}

// 使用 0 填充剩余的 SIMD 矢量
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    return npyv_load_till_s64(ptr, nlane, 0);
}

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    // 加载 nlane 个 32 位数据到 a，同时使用 fill_lo 和 fill_hi 填充剩余的 SIMD 矢量

    // 确保 nlane 大于 0
    assert(nlane > 0);

    npyv_s32 a;
    switch(nlane) {
    case 1:
        // 加载一个 32 位数据到 a，并用 fill_lo 和 fill_hi 填充剩余的 SIMD 矢量
        a = vld1q_lane_s32((const int32_t*)ptr, vcombine_s32(vdup_n_s32(fill_lo), vdup_n_s32(fill_hi)), 0);
        break;
    case 2:
        // 加载两个 32 位数据到 a，并用 fill_hi 填充剩余的 SIMD 矢量
        a = vcombine_s32(vld1_s32((const int32_t*)ptr), vdup_n_s32(fill_hi));
        break;
    case 3:
        // 加载前两个 32 位数据到 a，第三个数据使用 fill_hi 填充剩余的 SIMD 矢量
        a = vcombine_s32(
            vld1_s32((const int32_t*)ptr),
            vld1_lane_s32((const int32_t*)ptr + 2, vdup_n_s32(fill_hi), 0)
        );
        break;
    default:
        // 加载所有的 32 位数据到 a
        return npyv_load_s32(ptr);
    }
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，执行一个 workaround 操作
    volatile npyv_s32 workaround = a;
    a = vorrq_s32(workaround, a);
#endif
    return a;
}
    # 如果 nlane 等于 1，执行以下操作
    if (nlane == 1) {
        # 声明并初始化一个 16 字节对齐的整型数组 fill，包含两个元素 fill_lo 和 fill_hi
        const int32_t NPY_DECL_ALIGNED(16) fill[2] = {fill_lo, fill_hi};
        # 从指针 ptr 处加载两个 int32_t 类型的值，并将它们组合成一个 npyv_s32 向量 a
        npyv_s32 a = vcombine_s32(vld1_s32((const int32_t*)ptr), vld1_s32(fill));
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        # 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD 宏，则使用 volatile 变量 workaround 来保证正确的部分加载
        volatile npyv_s32 workaround = a;
        # 使用或运算将 workaround 和 a 合并，以确保正确性
        a = vorrq_s32(workaround, a);
    #endif
        # 返回向量 a
        return a;
    }
    # 如果 nlane 不等于 1，加载 ptr 指针处的 int32_t 数据并返回
    return npyv_load_s32(ptr);
//// 128-bit nlane
// 加载指定长度的 int64 数据到 SIMD 寄存器中，并在剩余的位置填充零值
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                         npy_int64 fill_lo, npy_int64 fill_hi)
{ 
    // 忽略未使用的参数 fill_lo 和 fill_hi，直接加载所有数据到寄存器中
    (void)nlane; (void)fill_lo; (void)fill_hi; 
    return npyv_load_s64(ptr); // 调用加载 int64 数据的函数并返回结果
}

NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ 
    // 忽略未使用的参数 nlane，加载指定长度的 int64 数据到 SIMD 寄存器中，并在剩余的位置填充零值
    (void)nlane; 
    return npyv_load_s64(ptr); // 调用加载 int64 数据的函数并返回结果
}

/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
// 加载不连续的部分 int32 数据到 SIMD 寄存器中
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0); // 断言加载长度大于零
    int32x4_t vfill = vdupq_n_s32(fill); // 使用 fill 参数创建一个 int32x4_t 类型的常量向量
    switch(nlane) {
    case 3:
        vfill = vld1q_lane_s32((const int32_t*)ptr + stride*2, vfill, 2); // 加载第三个元素到向量中的第二个位置
    case 2:
        vfill = vld1q_lane_s32((const int32_t*)ptr + stride, vfill, 1); // 加载第二个元素到向量中的第一个位置
    case 1:
        vfill = vld1q_lane_s32((const int32_t*)ptr, vfill, 0); // 加载第一个元素到向量中的第零个位置
        break;
    default:
        return npyv_loadn_s32(ptr, stride); // 加载连续的 int32 数据到 SIMD 寄存器中
    }
#if NPY_SIMD_GUARD_PARTIAL_LOAD
    volatile npyv_s32 workaround = vfill; // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，则执行的虚拟访问以避免优化
    vfill = vorrq_s32(workaround, vfill); // 使用或运算以确保加载数据到 vfill 向量中
#endif
    return vfill; // 返回加载后的向量
}

// 加载指定长度的 int32 数据到 SIMD 寄存器中，并在剩余的位置填充零值
NPY_FINLINE npyv_s32
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ 
    return npyv_loadn_till_s32(ptr, stride, nlane, 0); // 调用加载不连续部分 int32 数据的函数，填充零值并返回结果
}

// 加载指定长度的 int64 数据到 SIMD 寄存器中，并在剩余的位置填充指定的值
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0); // 断言加载长度大于零
    if (nlane == 1) {
        return npyv_load_till_s64(ptr, 1, fill); // 如果加载长度为 1，则加载到 SIMD 寄存器中并填充指定的值
    }
    return npyv_loadn_s64(ptr, stride); // 否则加载连续的 int64 数据到 SIMD 寄存器中
}

// 加载指定长度的 int64 数据到 SIMD 寄存器中，并在剩余的位置填充零值
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ 
    return npyv_loadn_till_s64(ptr, stride, nlane, 0); // 调用加载不连续部分 int64 数据的函数，填充零值并返回结果
}

//// 64-bit load over 32-bit stride
// 使用 32 位步长加载指定长度的 int32 数据到 SIMD 寄存器中
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0); // 断言加载长度大于零
    if (nlane == 1) {
        const int32_t NPY_DECL_ALIGNED(16) fill[2] = {fill_lo, fill_hi}; // 声明一个填充数组，确保对齐到 16 字节边界
        npyv_s32 a = vcombine_s32(vld1_s32((const int32_t*)ptr), vld1_s32(fill)); // 使用填充值创建一个 int32x2_t 向量并与原始数据合并
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a; // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，则执行的虚拟访问以避免优化
        a = vorrq_s32(workaround, a); // 使用或运算以确保加载数据到 a 向量中
    #endif
        return a; // 返回加载后的向量
    }
    return npyv_loadn2_s32(ptr, stride); // 加载连续的 int32 数据到 SIMD 寄存器中
}

// 使用 32 位步长加载指定长度的 int32 数据到 SIMD 寄存器中，并在剩余的位置填充零值
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0); // 断言加载长度大于零
    if (nlane == 1) {
        npyv_s32 a = vcombine_s32(vld1_s32((const int32_t*)ptr), vdup_n_s32(0)); // 使用零值创建一个 int32x2_t 向量并与原始数据合并
    #if NPY_SIMD_GUARD_PARTIAL_LOAD
        volatile npyv_s32 workaround = a; // 如果定义了 NPY_SIMD_GUARD_PARTIAL_LOAD，则执行的虚拟访问以避免优化
        a = vorrq_s32(workaround, a); // 使用或运算以确保加载数据到 a 向量中
    #endif
        return a; // 返回加载后的向量
    }
    return npyv_loadn2_s32(ptr, stride); // 加载连续的 int32 数据到 SIMD 寄存器中
}
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);  // 断言确保 nlane 大于 0

    // 将第一个 lane 的值存储到 ptr 指向的地址
    vst1q_lane_s32((int32_t*)ptr, a, 0);

    switch(nlane) {
    case 1:
        return;  // 如果 nlane 为 1，直接返回，只存储了一个元素
    case 2:
        // 将第二个 lane 的值存储到 ptr + stride 指向的地址
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
        return;
    case 3:
        // 将第二个 lane 的值存储到 ptr + stride 指向的地址
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
        // 将第三个 lane 的值存储到 ptr + stride*2 指向的地址
        vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
        return;
    default:
        // 默认情况下，存储所有的 lanes 的值到 ptr + stride*i 指向的地址
        vst1q_lane_s32((int32_t*)ptr + stride, a, 1);
        vst1q_lane_s32((int32_t*)ptr + stride*2, a, 2);
        vst1q_lane_s32((int32_t*)ptr + stride*3, a, 3);
    }
}
    # 断言语句，用于确保条件 nlane > 0 成立，否则程序会抛出 AssertionError 异常。
    assert(nlane > 0);
#if NPY_SIMD_F64
    // 如果定义了 NPY_SIMD_F64，使用 Neon 指令将 s32 类型向量 a 转换为 s64 类型后存储到内存 ptr
    vst1q_lane_s64((int64_t*)ptr, npyv_reinterpret_s64_s32(a), 0);
    // 如果向量长度 nlane 大于 1，继续将向量 a 的第一个 64 位元素存储到 ptr + stride 处
    if (nlane > 1) {
        vst1q_lane_s64((int64_t*)(ptr + stride), npyv_reinterpret_s64_s32(a), 1);
    }
#else
    // 如果未定义 NPY_SIMD_F64，将 s32 类型向量 a 的低 32 位元素存储到 ptr 处
    npyv_storel_s32(ptr, a);
    // 如果向量长度 nlane 大于 1，将向量 a 的高 32 位元素存储到 ptr + stride 处
    if (nlane > 1) {
        npyv_storeh_s32(ptr + stride, a);
    }
#endif
}

//// 128-bit store over 64-bit stride
// 将长度为 nlane 的 s64 类型向量 a 存储到内存 ptr，步长为 stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{ assert(nlane > 0); (void)stride; (void)nlane; npyv_store_s64(ptr, a); }

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
// 定义宏 NPYV_IMPL_NEON_REST_PARTIAL_TYPES，用于实现通过类型转换实现部分加载/存储操作
#define NPYV_IMPL_NEON_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                     \
    // 部分加载函数：从 ptr 处加载长度为 nlane 的 F_SFX 类型向量，用 fill 填充空缺部分
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX                                         \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_lanetype_##F_SFX fill)         \
    {                                                                                       \
        // 联合体 pun，将 fill 转换为对应的 T_SFX 类型后再转换为 F_SFX 类型向量返回
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    // 部分加载函数，带步长版本：从 ptr 处以 stride 步长加载长度为 nlane 的 F_SFX 类型向量
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        // 使用联合体 pun 将 fill 转换为对应的类型 from_##F_SFX
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    // 以填充值 fill 加载数据直到最后的函数定义，将 F_SFX 类型指针 ptr 转换为 T_SFX 类型
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        // 调用 load_tillz_##T_SFX 将指针 ptr 解释为 T_SFX 类型，直到 nlane 长度
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    // 以填充值为零加载数据直到最后的函数定义，将 F_SFX 类型指针 ptr 转换为 T_SFX 类型
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        // 调用 loadn_tillz_##T_SFX 将指针 ptr 解释为 T_SFX 类型，步长为 stride，直到 nlane 长度
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    // 将数据存储直到最后的函数定义，将 F_SFX 类型指针 ptr 中的数据存储为 T_SFX 类型
    NPY_FINLINE void npyv_store_till_##F_SFX                                                \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        // 调用 store_till_##T_SFX 将类型为 F_SFX 的向量 a 存储到类型为 T_SFX 的指针 ptr 中，直到 nlane 长度
        npyv_store_till_##T_SFX(                                                            \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        // 定义一个内联函数，用于将长度为 nlane 的 npyv_##F_SFX 类型向量 a 存储到 ptr 指向的内存中，步长为 stride
        npyv_storen_till_##T_SFX(                                                           \
            // 将 npyv_##F_SFX 类型的向量 a 重新解释为 npyv_##T_SFX##_##F_SFX 类型的向量，并存储到 ptr 中
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }
// 定义宏 NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR，用于生成两个函数：
// - npyv_load2_till_##F_SFX：加载并转换两个连续的元素到指定类型 F_SFX 的向量，直到满足 nlane 的数量要求，
//   使用 fill_lo 和 fill_hi 分别填充未加载的元素
// - npyv_loadn2_till_##F_SFX：加载并转换两个连续的元素到指定类型 F_SFX 的向量，根据 stride 和 nlane 的需求，
//   使用 fill_lo 和 fill_hi 填充未加载的元素
#define NPYV_IMPL_NEON_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                \
    // 内联函数 npyv_load2_till_##F_SFX，加载指定类型 F_SFX 的向量，直到 nlane 个元素，
    // 使用 fill_lo 和 fill_hi 分别填充未加载的元素
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        // 联合 pun 用于类型转换
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        // 调用 npyv_load2_till_##T_SFX 函数加载类型为 T_SFX 的向量，转换为类型 F_SFX 的向量
        // 这里将指针 ptr 强制转换为指向类型为 T_SFX 的元素的指针，并传入填充值 pun_lo.to_##T_SFX 和 pun_hi.to_##T_SFX
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    // 内联函数 npyv_loadn2_till_##F_SFX，加载指定类型 F_SFX 的向量，根据 stride 和 nlane 的需求，
    // 使用 fill_lo 和 fill_hi 填充未加载的元素
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_till_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \  // 定义一个联合体 pun，用于类型 F_SFX 和 T_SFX 的转换
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \  // 联合体成员 from_F_SFX，用于类型 F_SFX
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \  // 联合体成员 to_T_SFX，用于类型 T_SFX
        };                                                                                  \
        union pun pun_lo;                                                                   \  // 声明 pun 结构体变量 pun_lo
        union pun pun_hi;                                                                   \  // 声明 pun 结构体变量 pun_hi
        pun_lo.from_##F_SFX = fill_lo;                                                      \  // 将 fill_lo 赋值给 pun_lo 的 from_F_SFX 成员
        pun_hi.from_##F_SFX = fill_hi;                                                      \  // 将 fill_hi 赋值给 pun_hi 的 from_F_SFX 成员
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \  // 调用类型 F_SFX 到类型 T_SFX 的重新解释加载函数
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \  // 使用 ptr 所指向的数据加载类型 T_SFX
            pun_hi.to_##T_SFX                                                               \  // 使用 pun_lo.to_T_SFX 和 pun_hi.to_T_SFX 进行加载
        ));                                                                                 \
    }                                                                                       \  // 函数结束
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \  // 内联函数定义，加载类型 F_SFX 数据，直到遇到零
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \  // 参数：指向类型 F_SFX 数据的指针 ptr，加载的数量 nlane
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \  // 调用类型 F_SFX 到类型 T_SFX 的重新解释加载函数，加载直到遇到零
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \  // 使用 ptr 所指向的数据加载类型 T_SFX
        ));                                                                                 \
    }                                                                                       \  // 函数结束
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \  // 内联函数定义，加载类型 F_SFX 数据，直到遇到零，支持步长
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \  // 参数：指向类型 F_SFX 数据的指针 ptr，步长 stride，加载的数量 nlane
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \  // 调用类型 F_SFX 到类型 T_SFX 的重新解释加载函数，加载直到遇到零，支持步长
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \  // 使用 ptr 所指向的数据加载类型 T_SFX
        ));                                                                                 \
    }                                                                                       \  // 函数结束
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \  // 内联函数定义，存储类型 F_SFX 数据，直到遇到零
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \  // 参数：指向类型 F_SFX 数据的指针 ptr，存储的数量 nlane，数据 a
    {
        # 将向量 `a` 重新解释为 `T_SFX` 类型，并将其存储到指针 `ptr` 所指向的内存中，
        # 存储的个数由 `nlane` 指定
        npyv_store2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
    NPY_FINLINE void npyv_storen2_till_##F_SFX
    (
        npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a
    )
    {
        # 将向量 `a` 重新解释为 `T_SFX` 类型，并将其存储到以 `ptr` 指向为起点、
        # 步长为 `stride` 的内存中，存储的个数由 `nlane` 指定
        npyv_storen2_till_##T_SFX(
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,
            npyv_reinterpret_##T_SFX##_##F_SFX(a)
        );
    }
// 定义一个宏，用于实现 NEON SIMD 操作的加载和存储，用于两个通道的情况

#define NPYV_IMPL_NEON_MEM_INTERLEAVE(SFX, T_PTR)                        \
    // 定义一个内联函数，用于加载两个通道的 NEON 数据
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                      \
        const npyv_lanetype_##SFX *ptr                                   \
    ) {                                                                  \
        // 使用 NEON 指令 vld2q_##SFX 加载指针 ptr 所指向的数据
        return vld2q_##SFX((const T_PTR*)ptr);                           \
    }                                                                    \
    // 定义一个内联函数，用于存储两个通道的 NEON 数据
    NPY_FINLINE void npyv_store_##SFX##x2(                               \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                       \
    ) {                                                                  \
        // 使用 NEON 指令 vst2q_##SFX 存储数据 v 到指针 ptr 所指向的位置
        vst2q_##SFX((T_PTR*)ptr, v);                                     \
    }
    #define NPYV_IMPL_NEON_MEM_INTERLEAVE_64(SFX)                               \
        // 定义 NEON SIMD 操作的函数模板，用于加载和存储 64 位数据的双向交错操作
        NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                         \
            const npyv_lanetype_##SFX *ptr)                                     \
        {                                                                       \
            // 加载指针指向的前两个数据作为第一组向量 a 和第二组向量 b
            npyv_##SFX a = npyv_load_##SFX(ptr);                                \
            npyv_##SFX b = npyv_load_##SFX(ptr + 2);                            \
            // 创建存储两个向量的结构体 r
            npyv_##SFX##x2 r;                                                   \
            // 将向量 a 和 b 的低位和高位组合成 r 的两个元素
            r.val[0] = vcombine_##SFX(vget_low_##SFX(a),  vget_low_##SFX(b));   \
            r.val[1] = vcombine_##SFX(vget_high_##SFX(a), vget_high_##SFX(b));  \
            // 返回结构体 r，包含了交错后的数据
            return r;                                                           \
        }                                                                       \
        // 定义存储函数，将交错后的数据存储回内存中
        NPY_FINLINE void npyv_store_##SFX##x2(                                  \
            npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v)                         \
        {                                                                       \
            // 将结构体 v 的第一组和第二组数据分别存储到指针指向的地址和下一个地址
            npyv_store_##SFX(ptr, vcombine_##SFX(                               \
                vget_low_##SFX(v.val[0]),  vget_low_##SFX(v.val[1])));          \
            npyv_store_##SFX(ptr + 2, vcombine_##SFX(                           \
                vget_high_##SFX(v.val[0]),  vget_high_##SFX(v.val[1])));        \
        }
        // 实例化宏，生成具体的函数定义和实现
        NPYV_IMPL_NEON_MEM_INTERLEAVE_64(u64)
        NPYV_IMPL_NEON_MEM_INTERLEAVE_64(s64)
#endif // 结束预处理指令

/*********************************
 * Lookup table
 *********************************/
// 使用矢量作为表中的索引
// 该表包含32个uint32元素。
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{
    // 提取索引中的第一个值
    const unsigned i0 = vgetq_lane_u32(idx, 0);
    // 提取索引中的第二个值
    const unsigned i1 = vgetq_lane_u32(idx, 1);
    // 提取索引中的第三个值
    const unsigned i2 = vgetq_lane_u32(idx, 2);
    // 提取索引中的第四个值
    const unsigned i3 = vgetq_lane_u32(idx, 3);

    // 创建一个包含table[i0]的低位uint32x2_t值
    uint32x2_t low = vcreate_u32(table[i0]);
               // 从table中以i1为索引读取一个uint32_t值并将它加载到low的第二个位置
               low = vld1_lane_u32((const uint32_t*)table + i1, low, 1);
    // 创建一个包含table[i2]的高位uint32x2_t值
    uint32x2_t high = vcreate_u32(table[i2]);
               // 从table中以i3为索引读取一个uint32_t值并将它加载到high的第二个位置
               high = vld1_lane_u32((const uint32_t*)table + i3, high, 1);
    return vcombine_u32(low, high); // 组合低位和高位，返回结果
}
// 对npy_int32类型的表进行32位查找
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }
// 对float类型的表进行32位查找
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return npyv_reinterpret_f32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }

// 使用矢量作为索引的表
// 该表包含16个uint64元素。
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{
    // 从索引中提取低位值
    const unsigned i0 = vgetq_lane_u32(vreinterpretq_u32_u64(idx), 0);
    // 从索引中提取高位值
    const unsigned i1 = vgetq_lane_u32(vreinterpretq_u32_u64(idx), 2);
    return vcombine_u64(
        vld1_u64((const uint64_t*)table + i0), // 从table中以i0为索引读取uint64_t值
        vld1_u64((const uint64_t*)table + i1)  // 从table中以i1为索引读取uint64_t值
    );
}
// 对npy_int64类型的表进行64位查找
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }
// 当NPY_SIMD_F64宏被定义时，对double类型的表进行64位查找
#if NPY_SIMD_F64
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{ return npyv_reinterpret_f64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }
#endif

#endif // 结束预处理指令
```