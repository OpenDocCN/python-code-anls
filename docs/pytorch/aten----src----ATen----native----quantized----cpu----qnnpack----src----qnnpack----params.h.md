# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\qnnpack\params.h`

```py
/*
 * 版权所有（c）Facebook公司及其关联公司。
 * 保留所有权利。
 *
 * 此源代码根据位于此源树根目录下的LICENSE文件中的BSD式许可证进行许可。
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <qnnpack/common.h>

#include <cpuinfo.h>

// 定义结构体，用于存储QNNPACK半精度浮点数的夹紧参数
struct pytorch_qnnp_fp16_clamping_params {
  uint16_t scale; // 缩放因子
  uint16_t max;   // 最大值
  uint16_t min;   // 最小值
};

// 定义结构体，用于存储QNNPACK单精度浮点数的夹紧参数
struct pytorch_qnnp_fp32_clamping_params {
  float max;  // 最大值
  float min;  // 最小值
};

// 定义联合体，包含QNNPACK单精度浮点数重新量化的参数
union pytorch_qnnp_fp32_requantization_params {
  struct {
    float* scales;               // 缩放数组
    uint8_t output_zero_point;   // 输出零点
    uint8_t output_max;          // 输出最大值
    uint8_t output_min;          // 输出最小值
    float min_less_zero_point;   // 最小值减去零点
    float max_less_zero_point;   // 最大值减去零点
    float magic;                 // 魔数
    int32_t magic_less_zero_point; // 魔数减去零点
  } scalar;                      // 标量结构体
  struct {
    float* scales;               // 缩放数组
    float max;                   // 最大值
    float min;                   // 最小值
    float magic;                 // 魔数
    int32_t magic_less_zero_point; // 魔数减去零点
  } neon;                        // NEON结构体
  struct {
    float* scales;               // 缩放数组
    int16_t zero_point;          // 零点
    uint8_t max;                 // 最大值
    uint8_t min;                 // 最小值
  } neonv8;                      // NEONv8结构体
  struct {
    PYTORCH_QNNP_ALIGN(16) float* scales;       // 对齐到16字节的缩放数组
    PYTORCH_QNNP_ALIGN(16) int16_t zero_point[8]; // 对齐到16字节的8个零点
    PYTORCH_QNNP_ALIGN(16) uint8_t max[16];      // 对齐到16字节的16个最大值
    PYTORCH_QNNP_ALIGN(16) uint8_t min[16];      // 对齐到16字节的16个最小值
  } sse2;                                      // SSE2结构体
  struct {
    PYTORCH_QNNP_ALIGN(16) float* scales;               // 对齐到16字节的缩放数组
    PYTORCH_QNNP_ALIGN(16) float min_less_zero_point[4]; // 对齐到16字节的4个最小值减零点
    PYTORCH_QNNP_ALIGN(16) float max_less_zero_point[4]; // 对齐到16字节的4个最大值减零点
    PYTORCH_QNNP_ALIGN(16) float magic[4];               // 对齐到16字节的4个魔数
    PYTORCH_QNNP_ALIGN(16) int32_t magic_less_zero_point[4]; // 对齐到16字节的4个魔数减零点
  } psimd;                                                // PSIMD结构体
};

// 定义联合体，包含QNNPACK精确重新量化的参数
union pytorch_qnnp_precise_requantization_params {
  struct {
    uint32_t multiplier;      // 乘数
    uint32_t rounding_lo;     // 低位舍入
    uint32_t rounding_hi;     // 高位舍入
    uint32_t shift_less_32;   // 左移32位
    int32_t min_less_zero_point; // 最小值减零点
    int32_t max_less_zero_point; // 最大值减零点
    int32_t zero_point;       // 零点
  } scalar;                    // 标量结构体
  struct {
    int32_t multiplier;       // 乘数
    int32_t right_shift;      // 右移位
    int16_t zero_point;       // 零点
    uint8_t max;              // 最大值
    uint8_t min;              // 最小值
  } neon;                      // NEON结构体
  struct {
    PYTORCH_QNNP_ALIGN(16) uint32_t multiplier[4];   // 对齐到16字节的4个乘数
    PYTORCH_QNNP_ALIGN(16) uint64_t rounding[2];     // 对齐到16字节的2个舍入
    PYTORCH_QNNP_ALIGN(16) uint32_t shift[4];        // 对齐到16字节的4个移位
    PYTORCH_QNNP_ALIGN(16) int16_t zero_point[8];    // 对齐到16字节的8个零点
    PYTORCH_QNNP_ALIGN(16) uint8_t max[16];          // 对齐到16字节的16个最大值
    PYTORCH_QNNP_ALIGN(16) uint8_t min[16];          // 对齐到16字节的16个最小值
  } sse2;                                            // SSE2结构体
};

// 定义联合体，包含QNNPACK Q31重新量化的参数
union pytorch_qnnp_q31_requantization_params {
  struct {
    int32_t multiplier;           // 乘数
    int32_t remainder_mask;       // 余数掩码
    int32_t remainder_threshold;  // 余数阈值
    uint32_t shift;               // 移位
    int32_t min_less_zero_point;  // 最小值减零点
    int32_t max_less_zero_point;  // 最大值减零点
    int32_t zero_point;           // 零点
  } scalar;                        // 标量结构体
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    int32_t multiplier;           // 乘数
    int32_t right_shift;          // 右移位
    int16_t zero_point;           // 零点
    uint8_t max;                  // 最大值
    uint8_t min;                  // 最小值
  } neon;                          // NEON结构体
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) uint32_t multiplier[4];       // 对齐到16字节的4个乘数
    PYTORCH_QNNP_ALIGN(16) uint64_t rounding[2];         // 对齐到16字节的2个舍入
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_mask[4];    // 对齐到16字节的4个余数掩码
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_threshold[4]; // 对齐到16字节的4个余数阈值

    PYTORCH_QNNP_ALIGN(16) int32_t min_less_zero_point[4];  // 对齐到16字节的4个最小值减零点
    PYTORCH_QNNP_ALIGN(16) int32_t max_less_zero_point[4];  // 对齐到16字节的4个最大值减零点
    int32_t zero_point;           // 零点
  } sse2;                        // SSE2结构体
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
    # 使用宏定义 PYTORCH_QNNP_ALIGN(16)，将 shift 数组对齐到 16 字节边界，并声明为 uint64_t 类型
    PYTORCH_QNNP_ALIGN(16) uint64_t shift[2];
    # 使用宏定义 PYTORCH_QNNP_ALIGN(16)，将 zero_point 数组对齐到 16 字节边界，并声明为 int16_t 类型
    PYTORCH_QNNP_ALIGN(16) int16_t zero_point[8];
    # 使用宏定义 PYTORCH_QNNP_ALIGN(16)，将 max 数组对齐到 16 字节边界，并声明为 uint8_t 类型
    PYTORCH_QNNP_ALIGN(16) uint8_t max[16];
    # 使用宏定义 PYTORCH_QNNP_ALIGN(16)，将 min 数组对齐到 16 字节边界，并声明为 uint8_t 类型
    PYTORCH_QNNP_ALIGN(16) uint8_t min[16];
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

// 联合体，包含量化卷积参数的不同结构定义
union pytorch_qnnp_conv_quantization_params {
  // 标量结构定义，用于通用的量化卷积参数
  struct {
    const uint8_t* kernel_zero_points; // 指向卷积核零点的指针
    int32_t input_zero_point; // 输入零点
    const float* requantization_scales; // 重量化比例因子的指针
    int32_t output_min_less_zero_point; // 输出最小值减去零点
    int32_t output_max_less_zero_point; // 输出最大值减去零点
    int32_t output_zero_point; // 输出零点
  } scalar;

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // ARM 架构特定的结构定义
  struct {
    const uint8_t* kernel_zero_points; // 指向卷积核零点的指针
    int16_t input_zero_point; // 输入零点（16位）
    const float* requantization_scales; // 重量化比例因子的指针
    int16_t output_zero_point; // 输出零点（16位）
    uint8_t output_max; // 输出最大值
    uint8_t output_min; // 输出最小值
    // 以下四个字段用于 ARM32 下的最近偶数舍入，节省指令
    float vfmax; // 最大浮点值
    float vfmin; // 最小浮点值
    float vfmagic; // 浮点数的魔数
    int32_t vimagic; // 整数的魔数
  } neon;
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // x86 架构特定的结构定义（使用 SSE2 指令集加速）
  struct {
    PYTORCH_QNNP_ALIGN(16) const uint8_t* kernel_zero_points; // 对齐到16字节边界的卷积核零点的指针
    PYTORCH_QNNP_ALIGN(16) int16_t input_zero_point[8]; // 对齐到16字节边界的输入零点数组（8个元素）
    const PYTORCH_QNNP_ALIGN(16) float* requantization_scales; // 对齐到16字节边界的重量化比例因子的指针
    PYTORCH_QNNP_ALIGN(16) int16_t output_zero_point[8]; // 对齐到16字节边界的输出零点数组（8个元素）
    PYTORCH_QNNP_ALIGN(16) uint8_t output_max[16]; // 对齐到16字节边界的输出最大值数组（16个元素）
    PYTORCH_QNNP_ALIGN(16) uint8_t output_min[16]; // 对齐到16字节边界的输出最小值数组（16个元素）
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

// 动态量化卷积参数的结构定义
struct pytorch_qnnp_conv_dynamic_quantization_params {
  int16_t input_zero_point; // 输入零点
  const uint8_t* kernel_zero_points; // 指向卷积核零点的指针
  const float* multipliers; // 多重量化因子的指针
};

// 联合体，包含重量化参数的不同结构定义
union pytorch_qnnp_requantization_params {
  union pytorch_qnnp_precise_requantization_params precise; // 精确重量化参数
  union pytorch_qnnp_fp32_requantization_params fp32; // FP32 重量化参数
  union pytorch_qnnp_q31_requantization_params q31; // Q31 重量化参数
};

// 联合体，包含加法量化参数的不同结构定义
union pytorch_qnnp_add_quantization_params {
  struct {
    int32_t zero_point_product; // 零点乘积
    uint32_t a_multiplier; // a 的乘法因子
    uint32_t b_multiplier; // b 的乘法因子
    uint32_t shift; // 右移位数
    int32_t remainder_mask; // 余数掩码
    int32_t remainder_threshold; // 余数阈值
    int32_t y_zero_point; // 输出零点
    int32_t y_max; // 输出最大值
    int32_t y_min; // 输出最小值
  } scalar;

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    uint8_t a_zero_point; // a 的零点
    uint8_t b_zero_point; // b 的零点
    int16_t y_zero_point; // 输出零点（16位）
    int32_t a_multiplier; // a 的乘法因子
    int32_t b_multiplier; // b 的乘法因子
    int32_t right_shift; // 右移位数
    uint8_t y_max; // 输出最大值
    uint8_t y_min; // 输出最小值
  } neon;
#endif

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) int32_t zero_point_product[4]; // 对齐到16字节边界的零点乘积数组（4个元素）
    PYTORCH_QNNP_ALIGN(16) uint16_t a_multiplier_lo[8]; // 对齐到16字节边界的 a 乘法因子低位数组（8个元素）
    PYTORCH_QNNP_ALIGN(16) uint16_t a_multiplier_hi[8]; // 对齐到16字节边界的 a 乘法因子高位数组（8个元素）
    PYTORCH_QNNP_ALIGN(16) uint16_t b_multiplier_lo[8]; // 对齐到16字节边界的 b 乘法因子低位数组（8个元素）
    PYTORCH_QNNP_ALIGN(16) uint16_t b_multiplier_hi[8]; // 对齐到16字节边界的 b 乘法因子高位数组（8个元素）
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_mask[4]; // 对齐到16字节边界的余数掩码数组（4个元素）
    PYTORCH_QNNP_ALIGN(16) int32_t remainder_threshold[4]; // 对齐到16字节边界的余数阈值数组（4个元素）
    PYTORCH_QNNP_ALIGN(16) int16_t y_zero_point[8]; // 对齐到16字节边界的输出零点数组（8个元素）
    PYTORCH_QNNP_ALIGN(16) uint8_t y_max[16]; // 对齐到16字节边界的输出最大值数组（16个元素）
    PYTORCH_QNNP_ALIGN(16) uint8_t y_min[16]; // 对齐到16字节边界的输出最小值数组（16个元素）
    uint32_t shift; // 右移位数
    uint32_t a_multiplier; // a 的乘法因子
    uint32_t b_multiplier; // b 的乘法因子
  } sse2;
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

// 联合体，包含平均池化量化参数的不同结构定义
union pytorch_qnnp_avgpool_quantization_params {
  struct {
    // 定义一个结构体 scalar，用于存储以下几个变量的值
    int32_t bias;                  // 32位有符号整数变量，用于存储偏置值
    float scale;                   // 单精度浮点数变量，用于存储比例因子
    int32_t output_zero_point;     // 32位有符号整数变量，用于存储输出零点
    uint8_t output_max;            // 8位无符号整数变量，用于存储输出的最大值
    uint8_t output_min;            // 8位无符号整数变量，用于存储输出的最小值
  } scalar;
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  // ARM 或 ARM64 架构下的特定参数结构体
  struct {
    int32_t bias;                          // 偏置值
    float scale;                           // 缩放因子
    int16_t output_zero_point;             // 输出零点
    uint8_t output_max;                    // 输出最大值
    uint8_t output_min;                    // 输出最小值
    // 以下四个字段用于 ARM32 下的最近偶数舍入，可以节省一些指令
    float vfmax;                           // 最大浮点数
    float vfmin;                           // 最小浮点数
    float vfmagic;                         // 浮点数魔数
    int32_t vimagic;                       // 整数魔数
  } neon;                                  // neon 结构体
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // X86 或 X86_64 架构下的特定参数结构体
  struct {
    PYTORCH_QNNP_ALIGN(16) int32_t bias[4];        // 偏置数组，16字节对齐
    PYTORCH_QNNP_ALIGN(16) float scale[4];         // 缩放因子数组，16字节对齐
    PYTORCH_QNNP_ALIGN(16) int16_t output_zero_point[8];   // 输出零点数组，16字节对齐
    PYTORCH_QNNP_ALIGN(16) uint8_t output_max[16];   // 输出最大值数组，16字节对齐
    PYTORCH_QNNP_ALIGN(16) uint8_t output_min[16];   // 输出最小值数组，16字节对齐
  } sse2;                                  // sse2 结构体
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */

};

union pytorch_qnnp_u8_clamping_params {
  struct {
    int32_t output_max;                    // 输出最大值（标量）
    int32_t output_min;                    // 输出最小值（标量）
  } scalar;                                // 标量结构体

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
  struct {
    uint8_t output_max;                    // 输出最大值（ARM 或 ARM64 架构）
    uint8_t output_min;                    // 输出最小值（ARM 或 ARM64 架构）
  } neon;                                  // neon 结构体
#endif /* CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64 */

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  struct {
    PYTORCH_QNNP_ALIGN(16) uint8_t output_max[16];   // 输出最大值数组，16字节对齐（X86 或 X86_64 架构）
    PYTORCH_QNNP_ALIGN(16) uint8_t output_min[16];   // 输出最小值数组，16字节对齐（X86 或 X86_64 架构）
  } sse2;                                  // sse2 结构体
#endif /* CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64 */
};

typedef void (*pytorch_q8gemm_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    uint8_t* c,
    size_t c_stride,
    size_t output_channel_index,
    const union pytorch_qnnp_conv_quantization_params* quantization_params);

/*
  动态量化支持的 Q8 GEMM 核心函数。

  参数 w 表示权重，需要像返回的打包函数一样传递给这个核心函数。
  缓冲区中的初始偏置部分将被忽略。

  参数 bias 预期包含最多 nr 或 8 个浮点偏置。
  技术上，核心函数只需要从此参数指向的缓冲区中读取 nr 个偏置，
  但最多读取 8 个以保持逻辑简单且快速。
  因此，请确保此参数有足够的存储空间以避免触发越界错误。
  如有剩余的 8 - nr 个偏置，将不会被使用。

  参数 quantization_params 包含量化参数，即输入和核心的零点，以及乘数。
  乘数预期等于 input_scale * kernel_scale。
*/

typedef void (*pytorch_q8gemm_dq_ukernel_function)(
    size_t mr,
    size_t nr,
    size_t k,
    const uint8_t* a,
    size_t a_stride,
    const void* w,
    const float* bias,
    float* c,
    size_t c_stride,
    size_t output_channel_index,
    const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);

typedef void (*pytorch_q8gemm_dq_sparse_ukernel_function)(
    size_t mr,
    size_t nr,
    const uint8_t* a,
    size_t a_stride,
    const uint8_t* packed_w,
    const uint32_t* w_row_ptr,
    const uint32_t* w_block_ids_ptr,
    const float* bias,
    float* c,
    # 参数c是一个指向float类型的指针，用于存储卷积操作的输出结果

    size_t c_stride,
    # c_stride表示输出数组c的跨度（stride），即连续存储空间中每个元素之间的偏移量

    size_t output_channel_index,
    # output_channel_index表示当前卷积操作的输出通道索引，用于确定输出结果在数组c中的存储位置

    const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params);
    # quantization_params是一个指向结构体pytorch_qnnp_conv_dynamic_quantization_params的常量指针，
    # 包含了卷积操作中的动态量化参数，例如量化的缩放因子和偏置等
# 定义了一系列函数指针类型，用于表示不同的函数签名和参数类型

typedef void (*pytorch_q8gemm_dq_sparse_packedA_w32_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    const uint8_t* a_packed,  // 压缩的矩阵 A 数据
    const uint8_t* packed_w,  // 压缩的权重数据
    const uint32_t* w_row_ptr,  // 权重数据的行指针
    const uint32_t* w_block_ids_ptr,  // 权重数据的块 ID 指针
    const float* bias,  // 偏置数组
    float* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    size_t output_channel_index,  // 输出通道的索引
    const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params  // 动态量化参数
);

typedef void (*pytorch_q8gemm_dq_sparse_packedA_w16_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    const uint8_t* a_packed,  // 压缩的矩阵 A 数据
    const uint8_t* packed_w,  // 压缩的权重数据
    const uint16_t* w_row_ptr,  // 权重数据的行指针
    const uint16_t* w_block_ids_ptr,  // 权重数据的块 ID 指针
    const float* bias,  // 偏置数组
    float* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    size_t output_channel_index,  // 输出通道的索引
    const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params  // 动态量化参数
);

typedef void (*pytorch_q8gemm_dq_sparse_packedA_w8_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    const uint8_t* a_packed,  // 压缩的矩阵 A 数据
    const uint8_t* packed_w,  // 压缩的权重数据
    const uint8_t* w_row_ptr,  // 权重数据的行指针
    const uint8_t* w_block_ids_ptr,  // 权重数据的块 ID 指针
    const float* bias,  // 偏置数组
    float* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    size_t output_channel_index,  // 输出通道的索引
    const struct pytorch_qnnp_conv_dynamic_quantization_params* quantization_params  // 动态量化参数
);

typedef void (*pytorch_q8gemm_sparse_packA_ukernel_function)(
    const size_t mr,  // 矩阵行数（输出通道）的大小
    const size_t K,  // 矩阵列数（输入通道）的大小
    const uint8_t* a,  // 输入矩阵 A 的数据
    const size_t a_stride,  // 输入矩阵 A 的行步幅
    uint8_t* a_packed  // 压缩后的输入矩阵 A 的数据
);

typedef void (*pytorch_q8conv_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    size_t kc,  // 卷积内核的列数
    size_t ks,  // 卷积内核的行数
    const uint8_t** a,  // 输入矩阵 A 的数据
    const void* w,  // 卷积内核的数据
    uint8_t* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    size_t output_channel_index,  // 输出通道的索引
    const union pytorch_qnnp_conv_quantization_params* quantization_params  // 卷积量化参数
);

typedef void (*pytorch_q8gemm_xzp_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    size_t k,  // 矩阵列数（输入通道）的大小
    const uint8_t* a,  // 输入矩阵 A 的数据
    size_t a_stride,  // 输入矩阵 A 的行步幅
    const int32_t* a_sum,  // 矩阵 A 求和的数据
    const void* w,  // 权重数据
    uint8_t* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    const union pytorch_qnnp_q31_requantization_params* requantization_params  // 重新量化参数
);

typedef void (*pytorch_q8sum_rows_ukernel_function)(
    const uint8_t* a,  // 输入矩阵 A 的数据
    size_t m,  // 矩阵行数
    size_t k,  // 矩阵列数
    size_t stride,  // 矩阵行步幅
    int32_t multiplier,  // 乘数
    int32_t* sums  // 结果数组
);

typedef void (*pytorch_xzipc_ukernel_function)(
    size_t n,  // 输入向量的大小
    const void* x,  // 输入向量 X 的数据
    void* y  // 输出向量 Y 的数据
);

typedef void (*pytorch_xzipv_ukernel_function)(
    size_t n,  // 输入向量 X 的大小
    size_t m,  // 输入向量 Y 的大小
    const void* x,  // 输入向量 X 的数据
    void* y  // 输出向量 Y 的数据
);

typedef void (*pytorch_x8lut_ukernel_function)(
    size_t n,  // 输入向量的大小
    const uint8_t* x,  // 输入向量 X 的数据
    const uint8_t* t,  // 查找表 T 的数据
    uint8_t* y  // 输出向量 Y 的数据
);

typedef void (*pytorch_sgemm_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    size_t k,  // 矩阵列数（输入通道）的大小
    const float* a,  // 输入矩阵 A 的数据
    size_t a_stride,  // 输入矩阵 A 的行步幅
    const float* w,  // 权重数据
    float* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    const struct pytorch_qnnp_fp32_clamping_params* clamping_params  // 浮点数限制参数
);

typedef void (*pytorch_sconv_ukernel_function)(
    size_t mr,  // 矩阵行数（输出通道）的大小
    size_t nr,  // 矩阵列数（输出通道）的大小
    size_t kc,  // 卷积内核的列数
    size_t ks,  // 卷积内核的行数
    const float** a,  // 输入矩阵 A 的数据
    const float* w,  // 卷积内核的数据
    float* c,  // 输出结果数组
    size_t c_stride,  // 输出结果数组的行步幅
    const struct pytorch_qnnp_fp32_clamping_params* clamping_params  // 浮点数限制参数
);
// 定义函数指针类型 pytorch_hgemm_ukernel_function，用于指向高效矩阵乘法内核函数
typedef void (*pytorch_hgemm_ukernel_function)(
    size_t mr,  // 矩阵行数
    size_t nr,  // 矩阵列数
    size_t k,   // 矩阵乘法的维度
    const void* a,  // 输入矩阵 A 的指针
    size_t a_stride,  // 矩阵 A 中的行跨度
    const void* w,    // 权重矩阵 W 的指针
    void* c,          // 输出矩阵 C 的指针
    size_t c_stride,  // 矩阵 C 中的行跨度
    const struct pytorch_qnnp_fp16_clamping_params* clamping_params  // 半精度浮点数的夹紧参数
);

// 定义函数指针类型 pytorch_q8dwconv2d_up_ukernel_function，用于指向深度卷积 2D 上采样操作的内核函数
typedef void (*pytorch_q8dwconv2d_up_ukernel_function)(
    size_t channels,  // 通道数
    size_t output_width,  // 输出图像的宽度
    const uint8_t** input,  // 输入图像的指针数组
    const void* weights,    // 卷积核权重的指针
    uint8_t* output,        // 输出图像的指针
    size_t input_stride,    // 输入图像的行跨度
    size_t output_increment,  // 输出图像每行的增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params  // 卷积操作的量化参数
);

// 定义函数指针类型 pytorch_q8dwconv2d_mp_ukernel_function，用于指向深度卷积 2D 多通道平均池化操作的内核函数
typedef void (*pytorch_q8dwconv2d_mp_ukernel_function)(
    size_t channels,  // 通道数
    size_t output_width,  // 输出图像的宽度
    const uint8_t** input,  // 输入图像的指针数组
    const void* weights,    // 卷积核权重的指针
    int32_t* buffer,        // 缓冲区的指针
    uint8_t* output,        // 输出图像的指针
    size_t input_stride,    // 输入图像的行跨度
    size_t output_increment,  // 输出图像每行的增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params  // 卷积操作的量化参数
);

// 定义函数指针类型 pytorch_q8dwconv3d_mp_ukernel_function，用于指向深度卷积 3D 多通道平均池化操作的内核函数
typedef void (*pytorch_q8dwconv3d_mp_ukernel_function)(
    size_t channels,  // 通道数
    size_t output_height,  // 输出图像的高度
    size_t output_width,   // 输出图像的宽度
    const uint8_t** input,  // 输入图像的指针数组
    const void* weights,    // 卷积核权重的指针
    int32_t* buffer,        // 缓冲区的指针
    uint8_t* output,        // 输出图像的指针
    size_t input_row_stride,    // 输入图像的行跨度
    size_t input_col_stride,    // 输入图像的列跨度
    size_t output_increment,    // 输出图像的增量
    const union pytorch_qnnp_conv_quantization_params* quantization_params  // 卷积操作的量化参数
);

// 定义函数指针类型 pytorch_q8gavgpool_up_ukernel_function，用于指向全局平均池化 2D 上采样操作的内核函数
typedef void (*pytorch_q8gavgpool_up_ukernel_function)(
    size_t m,  // 高度
    size_t n,  // 宽度
    const uint8_t* x,  // 输入图像的指针
    size_t x_stride,   // 输入图像的行跨度
    const uint8_t* zero,  // 常量零的指针
    uint8_t* y,  // 输出图像的指针
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params  // 池化操作的量化参数
);

// 定义函数指针类型 pytorch_q8gavgpool_mp_ukernel_function，用于指向全局平均池化 2D 多通道平均池化操作的内核函数
typedef void (*pytorch_q8gavgpool_mp_ukernel_function)(
    size_t m,  // 高度
    size_t n,  // 宽度
    const uint8_t* x,  // 输入图像的指针
    size_t x_stride,   // 输入图像的行跨度
    const uint8_t* zero,  // 常量零的指针
    int32_t* buffer,      // 缓冲区的指针
    uint8_t* y,           // 输出图像的指针
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params  // 池化操作的量化参数
);

// 定义函数指针类型 pytorch_q8avgpool_up_ukernel_function，用于指向平均池化 2D 上采样操作的内核函数
typedef void (*pytorch_q8avgpool_up_ukernel_function)(
    size_t n,  // 通道数
    size_t ks,  // 核大小
    size_t kc,  // 输入通道数
    const uint8_t** x,  // 输入图像的指针数组
    const uint8_t* zero,  // 常量零的指针
    uint8_t* y,  // 输出图像的指针
    size_t x_increment,  // 输入图像每行的增量
    size_t y_increment,  // 输出图像每行的增量
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params  // 池化操作的量化参数
);

// 定义函数指针类型 pytorch_q8avgpool_mp_ukernel_function，用于指向平均池化 2D 多通道平均池化操作的内核函数
typedef void (*pytorch_q8avgpool_mp_ukernel_function)(
    size_t n,  // 通道数
    size_t ks,  // 核大小
    size_t kc,  // 输入通道数
    const uint8_t** x,  // 输入图像的指针数组
    const uint8_t* zero,  // 常量零的指针
    int32_t* buffer,      // 缓冲区的指针
    uint8_t* y,           // 输出图像的指针
    size_t x_increment,   // 输入图像每行的增量
    size_t y_increment,   // 输出图像每行的增量
    const union pytorch_qnnp_avgpool_quantization_params* quantization_params  // 池化操作的量化参数
);

// 定义函数指针类型 pytorch_u8maxpool_ukernel_function，用于指向最大池化操作的内核函数
typedef void (*pytorch_u8maxpool_ukernel_function)(
    size_t n,  // 通道数
    size_t ks,  // 核大小
    size_t kc,  // 输入通道数
    const uint8_t** x,  // 输入图像的指针数组
    uint8_t* y,          // 输出图像的指针
    size_t x_increment,  // 输入图像每行的增量
    size_t y_increment,  // 输出图像每行的增量
    const union pytorch_qnnp_u8_clamping_params* params  // 无符号 8 位整数的夹紧参数
);

// 定义函数指针类型 pytorch_u8clamp_ukernel_function，用于指向无符号 8 位整数的夹紧操作的内核函数
typedef void
    // 声明一个指向 uint8_t 类型的指针 y，该指针作为函数的参数
    uint8_t* y);
// 定义一个函数指针类型 pytorch_q8vadd_ukernel_function，用于表示 Q8 矢量加法的内核函数类型
typedef void (*pytorch_q8vadd_ukernel_function)(
    size_t n, // n 表示操作数的数量
    const uint8_t* a, // 指向输入张量 A 数据的指针
    const uint8_t* b, // 指向输入张量 B 数据的指针
    uint8_t* y, // 指向输出张量 Y 数据的指针
    const union pytorch_qnnp_add_quantization_params* quantization_params); // 用于量化参数的联合体指针

// 定义 Q8 卷积参数结构体 pytorch_q8conv_parameters
struct pytorch_q8conv_parameters {
  pytorch_q8gemm_ukernel_function gemm; // Q8 矩阵乘法内核函数指针
  pytorch_q8conv_ukernel_function conv; // Q8 卷积内核函数指针
  pytorch_q8gemm_dq_ukernel_function gemm_dq; // Q8 矩阵乘法（数据量化）内核函数指针
  uint8_t mr; // 行寄存器数
  uint8_t nr; // 列寄存器数
  uint8_t kr; // 内核寄存器数
};

// 定义 Q8 稀疏矩阵乘法参数结构体 pytorch_q8gemm_sparse_parameters
struct pytorch_q8gemm_sparse_parameters {
  pytorch_q8gemm_dq_sparse_ukernel_function gemm_dq; // Q8 稀疏矩阵乘法（数据量化）内核函数指针
  // packedA_w32_gemm_dq, packedA_w16_gemm_dq, packedA_w8_gemm_dq 使用不同数据类型（uint32_t, uint16_t, uint8_t）对行值/列索引进行打包的内核函数指针
  pytorch_q8gemm_dq_sparse_packedA_w32_ukernel_function packedA_w32_gemm_dq;
  pytorch_q8gemm_dq_sparse_packedA_w16_ukernel_function packedA_w16_gemm_dq;
  pytorch_q8gemm_dq_sparse_packedA_w8_ukernel_function packedA_w8_gemm_dq;
  pytorch_q8gemm_sparse_packA_ukernel_function packA; // Q8 稀疏矩阵打包内核函数指针
  uint8_t mr; // 行寄存器数
  uint8_t nr; // 列寄存器数
  uint8_t kr; // 内核寄存器数
  uint8_t log2_mr; // 行寄存器数的对数
  uint8_t log2_row_block_size; // 行块大小的对数
  uint32_t row_block_size; // 行块大小
  uint32_t col_block_size; // 列块大小
};

// 定义 Q8 XZP 卷积参数结构体 pytorch_q8conv_xzp_parameters
struct pytorch_q8conv_xzp_parameters {
  pytorch_q8gemm_xzp_ukernel_function gemm; // Q8 XZP 矩阵乘法内核函数指针
  // 没有卷积内核函数
  uint8_t mr; // 行寄存器数
  uint8_t nr; // 列寄存器数
  uint8_t kr; // 内核寄存器数
  uint8_t kc; // 内核通道数
  size_t kthreshold; // 内核阈值
};

// 定义 Q8 深度可分离卷积 2D 上采样参数结构体 pytorch_q8dwconv2d_up_parameters
struct pytorch_q8dwconv2d_up_parameters {
  pytorch_q8dwconv2d_up_ukernel_function updw; // Q8 深度可分离卷积 2D 上采样内核函数指针
  pytorch_q8dwconv2d_up_ukernel_function updw_per_channel; // 每通道 Q8 深度可分离卷积 2D 上采样内核函数指针
  uint8_t cr; // 通道寄存器数
};

// 定义 Q8 深度可分离卷积 2D 多路参数结构体 pytorch_q8dwconv2d_mp_parameters
struct pytorch_q8dwconv2d_mp_parameters {
  pytorch_q8dwconv2d_mp_ukernel_function mpdw; // Q8 深度可分离卷积 2D 多路内核函数指针
  pytorch_q8dwconv2d_mp_ukernel_function mpdw_per_channel; // 每通道 Q8 深度可分离卷积 2D 多路内核函数指针
  uint8_t cr; // 通道寄存器数
};

// 定义 Q8 深度可分离卷积 3D 多路参数结构体 pytorch_q8dwconv3d_mp_parameters
struct pytorch_q8dwconv3d_mp_parameters {
  pytorch_q8dwconv3d_mp_ukernel_function mpdw; // Q8 深度可分离卷积 3D 多路内核函数指针
  uint8_t cr; // 通道寄存器数
};

// 定义 Q8 求行和参数结构体 pytorch_q8sum_rows_parameters
struct pytorch_q8sum_rows_parameters {
  pytorch_q8sum_rows_ukernel_function sum_rows; // Q8 求行和内核函数指针
  uint32_t m; // 矩阵行数
};

// 定义 Q8 全局平均池化参数结构体 pytorch_q8gavgpool_parameters
struct pytorch_q8gavgpool_parameters {
  pytorch_q8gavgpool_up_ukernel_function ltnr; // Q8 全局平均池化（低延迟、无残余）内核函数指针
  pytorch_q8gavgpool_up_ukernel_function genr_lemr; // Q8 全局平均池化（一般残余、低延迟、多残余）内核函数指针
  pytorch_q8gavgpool_mp_ukernel_function genr_gtmr; // Q8 全局平均池化（一般残余、较大残余）内核函数指针
  uint8_t mr; // 行寄存器数
  uint8_t nr; // 列寄存器数
};

// 定义 Q8 平均池化参数结构体 pytorch_q8avgpool_parameters
struct pytorch_q8avgpool_parameters {
  pytorch_q8avgpool_up_ukernel_function ltkr; // Q8 平均池化（低延迟、常规内核）内核函数指针
  pytorch_q8avgpool_up_ukernel_function gekr_lemr; // Q8 平均池化（一般残余、低延迟、常规内核）内核函数指针
  pytorch_q8avgpool_mp_ukernel_function gekr_gtmr; // Q8 平均池化（一般残余、较大残余、常规内核）内核函数指针
  uint8_t mr; // 行寄存器数
  uint8_t qr; // 列寄存器数
  uint8_t kr; // 内核寄存器数
};

// 定义 U8 最大池化参数结构体 pytorch_u8maxpool_parameters
struct pytorch_u8maxpool_parameters {
  pytorch_u8maxpool_ukernel_function ltkr; // U8 最大池化（低延迟、常规内核）内核函数指针
  pytorch_u8maxpool_ukernel_function gekr; // U8 最大池化（一般残余、常规内核）内核函数指针
  uint8_t mr; // 行寄存器数
  uint8_t qr; // 列寄存器数
  uint8_t kr; //
// 定义了一个结构体 pytorch_qnnp_parameters，用于存储 PyTorch QNNPACK 模块的参数
struct pytorch_qnnp_parameters {
  // 包含 QNNPACK Q8Conv 算子的参数结构体
  struct pytorch_q8conv_parameters q8conv;
  // 包含 QNNPACK Q8GEMM 稀疏算子（1x4）的参数结构体
  struct pytorch_q8gemm_sparse_parameters q8gemm_sparse_c1x4;
  // 包含 QNNPACK Q8GEMM 稀疏算子（8x1）的参数结构体
  struct pytorch_q8gemm_sparse_parameters q8gemm_sparse_c8x1;
  // 包含 QNNPACK Q8Conv_xzp 算子的参数结构体
  struct pytorch_q8conv_xzp_parameters q8conv_xzp;
  // 包含 QNNPACK Q8DWConv2D_Up 算子的参数结构体
  struct pytorch_q8dwconv2d_up_parameters q8dw9;
  // 包含 QNNPACK Q8DWConv2D_Mp 算子的参数结构体
  struct pytorch_q8dwconv2d_mp_parameters q8dw25;
  // 包含 QNNPACK Q8DWConv3D_Mp 算子的参数结构体
  struct pytorch_q8dwconv3d_mp_parameters q8dw27;
  // 包含 QNNPACK Q8SumRows 算子的参数结构体
  struct pytorch_q8sum_rows_parameters q8sum_rows;
  // 指向 QNNPACK Q8VAdd 算子函数的指针
  pytorch_q8vadd_ukernel_function q8vadd;
  // 包含 QNNPACK Q8GAvgPool 算子的参数结构体
  struct pytorch_q8gavgpool_parameters q8gavgpool;
  // 包含 QNNPACK Q8AvgPool 算子的参数结构体
  struct pytorch_q8avgpool_parameters q8avgpool;
  // 包含 QNNPACK U8MaxPool 算子的参数结构体
  struct pytorch_u8maxpool_parameters u8maxpool;
  // 指向 QNNPACK U8LUT32Norm 算子函数的指针
  pytorch_u8lut32norm_ukernel_function u8lut32norm;
  // 指向 QNNPACK U8Clamp 算子函数的指针
  pytorch_u8clamp_ukernel_function u8clamp;
  // 指向 QNNPACK U8RMax 算子函数的指针
  pytorch_u8rmax_ukernel_function u8rmax;
  // 包含 QNNPACK X8Zip 算子的参数结构体
  struct pytorch_x8zip_parameters x8zip;
  // 指向 QNNPACK X8LUT 算子函数的指针
  pytorch_x8lut_ukernel_function x8lut;
  // 表示结构体是否已初始化的布尔值
  bool initialized;
};

#ifdef __cplusplus
extern "C" {
#endif

// 声明一个全局的 pytorch_qnnp_parameters 结构体实例，用于外部访问
extern struct pytorch_qnnp_parameters pytorch_qnnp_params;

#ifdef __cplusplus
}
#endif
```