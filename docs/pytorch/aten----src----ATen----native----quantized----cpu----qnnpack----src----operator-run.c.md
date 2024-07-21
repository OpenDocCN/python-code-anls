# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\operator-run.c`

```py
/*
 * 该文件包含了 QNNPACK 库中 Q8GEMM 相关的实现，用于量化卷积操作。
 */

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <pytorch_qnnpack.h>
#include <qnnpack/common.h>
#include <qnnpack/log.h>
#include <qnnpack/math.h>
#include <qnnpack/operator.h>
#include <qnnpack/params.h>

#ifdef _MSC_VER
#include <malloc.h>
#endif

/*
 * 定义 Q8GEMM 运算的上下文结构体，包括了操作数、参数以及计算中使用的内存布局信息。
 */
struct q8gemm_context {
  size_t k;  // 矩阵乘法操作的维度大小
  size_t k_stride;  // 矩阵 A 的列步长
  size_t n;  // 输出矩阵 C 的列数
  size_t n_stride;  // 输出矩阵 C 的列步长
  const uint8_t* a;  // 输入矩阵 A 的数据指针
  size_t a_stride;  // 输入矩阵 A 的行步长
  const uint8_t* packed_w;  // 打包后的权重数据指针
  uint8_t* c;  // 输出矩阵 C 的数据指针
  size_t c_stride;  // 输出矩阵 C 的行步长
  union pytorch_qnnp_conv_quantization_params quantization_params;  // 量化参数结构体
  const pytorch_q8gemm_ukernel_function ukernel;  // Q8GEMM 的微内核函数指针
};

/*
 * 执行 Q8GEMM 计算的函数，计算量化矩阵乘法的一部分。
 */
static void compute_q8gemm(
    const struct q8gemm_context context[RESTRICT_STATIC 1],  // Q8GEMM 运算的上下文数组
    size_t group_index,  // 组索引，用于分组卷积
    size_t pixel_index,  // 像素索引，用于逐像素计算
    size_t mr_block_start,  // MR 块的起始索引
    size_t nr_block_start,  // NR 块的起始索引
    size_t group_range /* always 1 */,  // 组范围，通常为 1
    size_t pixel_range,  // 像素范围，用于逐像素计算
    size_t mr_block_size,  // MR 块的大小
    size_t nr_block_size) {  // NR 块的大小

  const size_t k = context->k;  // 获取上下文中的 k 大小
  const size_t k_stride = context->k_stride;  // 获取上下文中的 k 步长
  const size_t n = context->n;  // 获取上下文中的 n 大小
  const size_t n_stride = context->n_stride;  // 获取上下文中的 n 步长
  const uint8_t* restrict a = context->a;  // 获取上下文中的矩阵 A 数据指针
  const size_t a_stride = context->a_stride;  // 获取上下文中的矩阵 A 行步长
  const void* restrict packed_w = context->packed_w;  // 获取上下文中的打包后权重数据指针
  uint8_t* restrict c = context->c;  // 获取上下文中的输出矩阵 C 数据指针
  const size_t c_stride = context->c_stride;  // 获取上下文中的输出矩阵 C 行步长

  // 计算输出通道索引
  size_t output_channel_index = nr_block_start + group_index * n;

  // 调用 Q8GEMM 的微内核函数执行矩阵乘法计算
  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,  // 输入矩阵 A 的起始位置
      a_stride,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),  // 权重数据的起始位置
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
          group_index * n,  // 输出矩阵 C 的起始位置
      c_stride,
      output_channel_index,
      &context->quantization_params);  // 量化参数
}

/*
 * 在目前的实现中，我们选择移除不需要预打包的稀疏内核，因为它们的性能总是较差。
 */
#ifdef NO_PREPACK_SPARSE_KERNEL
/*
 * 定义 Q8GEMM 稀疏动态量化的上下文结构体，用于存储稀疏矩阵乘法的相关参数和数据。
 */
struct q8gemm_sparse_dq_context {
  const uint8_t* a;  // 输入矩阵 A 的数据指针
  size_t a_stride;  // 输入矩阵 A 的行步长
  const uint32_t* kernel_col_indices;  // 稀疏权重列索引
  const uint32_t* kernel_row_values;  // 稀疏权重行数值
  const uint8_t* kernel_values;  // 稀疏权重值
  const float* bias;  // 偏置值
  float* c;  // 输出矩阵 C 的数据指针，可以是 float 或 uint8_t 类型
  size_t c_stride;  // 输出矩阵 C 的行步长
  struct pytorch_qnnp_conv_dynamic_quantization_params quantization_params;  // 动态量化参数结构体
  const pytorch_q8gemm_dq_sparse_ukernel_function ukernel;  // Q8GEMM 稀疏动态量化的微内核函数指针
};

/*
 * 执行 Q8GEMM 稀疏动态量化计算的函数。
 */
static void compute_q8gemm_sparse_dq(
    const struct q8gemm_sparse_dq_context context[RESTRICT_STATIC 1],  // 稀疏动态量化上下文数组
    size_t group_index,  // 组索引，忽略
    size_t pixel_index,  // 像素索引，忽略
    size_t mr_block_start,  // MR 块的起始索引
    size_t nr_block_start,  // NR 块的起始索引
    size_t group_range /* always 1 */,  // 组范围，通常为 1
    size_t pixel_range,  // 像素范围，用于逐像素计算
    size_t mr_block_size,
    size_t nr_block_size) {

函数参数声明，`nr_block_size` 是一个 `size_t` 类型的参数，表示块的大小。


  const uint8_t* restrict a = context->a;

声明并初始化指向 `context` 结构体中 `a` 成员的指针 `a`，`a` 指向的数据类型是 `const uint8_t`，`restrict` 关键字表明该指针是访问其所指向的唯一方式。


  const size_t a_stride = context->a_stride;

从 `context` 结构体中获取 `a_stride` 成员的值，保存到 `a_stride` 变量中，表示 `a` 数组中每个元素之间的步长。


  float* restrict c = (float*)context->c;

声明并初始化指向 `context` 结构体中 `c` 成员的指针 `c`，强制类型转换为 `float*` 类型，表明 `c` 指向的是一个 `float` 类型的数据数组。


  const size_t c_stride = context->c_stride;

从 `context` 结构体中获取 `c_stride` 成员的值，保存到 `c_stride` 变量中，表示 `c` 数组中每个元素之间的步长。


  size_t output_channel_index = nr_block_start;

声明并初始化 `output_channel_index` 变量，将其赋值为 `nr_block_start`，表示输出通道的起始索引。


  context->ukernel(
      mr_block_size,
      nr_block_size,
      a + mr_block_start * a_stride,
      a_stride,
      context->kernel_values,
      context->kernel_row_values + nr_block_start,
      context->kernel_col_indices,
      context->bias + nr_block_start,
      c + mr_block_start * c_stride + nr_block_start,
      c_stride,
      output_channel_index,
      &context->quantization_params);

调用 `context` 结构体中的 `ukernel` 函数，传递多个参数给该函数，这些参数包括块的大小、输入数据数组 `a` 的起始位置、步长、核值、偏置值、输出数据数组 `c` 的起始位置和步长、输出通道索引以及量化参数的地址。
}
#endif

// 结构体定义，用于存储 Q8GEMM 稀疏矩阵预打包的相关上下文信息
struct q8gemm_prepackA_sparse_dq_context {
  size_t k;  // 矩阵 A 的列数
  const uint8_t* a;  // 稀疏矩阵 A 的指针
  size_t a_stride;  // 矩阵 A 的行步长
  uint8_t* a_packed;  // 预打包后的矩阵 A 的指针
  size_t a_packed_stride;  // 预打包后的矩阵 A 的行步长
  size_t log2_mr;  // log2(mr)，用于计算块的偏移
  size_t log2_row_block_size;  // log2(行块大小)，用于计算偏移
  union {
    const uint32_t* kernel_col_indices_w32;  // 列索引数组指针，不同位宽的联合体
    const uint16_t* kernel_col_indices_w16;
    const uint8_t* kernel_col_indices_w8;
  };
  union {
    const uint32_t* kernel_row_values_w32;  // 行值数组指针，不同位宽的联合体
    const uint16_t* kernel_row_values_w16;
    const uint8_t* kernel_row_values_w8;
  };
  enum pytorch_qnnp_sparse_matrix_indices_dtype kernel_indices_dtype;  // 稀疏矩阵索引数据类型
  const uint8_t* kernel_values;  // 稀疏矩阵的值数组指针
  const float* bias;  // 偏置数组指针
  float* c;  // 输出矩阵 C 的指针，可以是 float 或 uint8_t 类型
  size_t c_stride;  // 输出矩阵 C 的行步长
  struct pytorch_qnnp_conv_dynamic_quantization_params quantization_params;  // 动态量化参数结构体
  union {
    // 非 const，因为在上下文初始化后进行了赋值
    pytorch_q8gemm_dq_sparse_packedA_w32_ukernel_function ukernel_w32;  // 不同位宽的稀疏矩阵预打包函数指针联合体
    pytorch_q8gemm_dq_sparse_packedA_w16_ukernel_function ukernel_w16;
    pytorch_q8gemm_dq_sparse_packedA_w8_ukernel_function ukernel_w8;
  };
  const pytorch_q8gemm_sparse_packA_ukernel_function prepack_ukernel;  // 预打包函数指针
};

// 计算 Q8GEMM 稀疏矩阵预打包后的矩阵 A
static void compute_q8gemm_prepack_a_sparse(
    const struct q8gemm_prepackA_sparse_dq_context context[RESTRICT_STATIC 1],
    size_t group_index, /* ignored */
    size_t pixel_index, /* ignored */
    size_t mr_block_start,  // mr 块的起始索引
    size_t nr_block_start,  // nr 块的起始索引
    size_t group_range /* always 1 */,  // 组范围，总是 1
    size_t pixel_range,  // 像素范围
    size_t mr_block_size,  // mr 块的大小
    size_t nr_block_size) {  // nr 块的大小
  const uint8_t* restrict a = context->a;  // 稀疏矩阵 A 的指针
  const size_t a_stride = context->a_stride;  // 稀疏矩阵 A 的行步长
  const size_t mr_packed_block_start =
    ((mr_block_start >> context->log2_mr) * context->a_packed_stride);  // 计算预打包后矩阵 A 的块起始位置

  // 调用预打包函数，将稀疏矩阵 A 的块打包到预打包后的矩阵 A 中
  context->prepack_ukernel(
      mr_block_size,
      context->k,
      a + mr_block_start * a_stride,  // 稀疏矩阵 A 中块的起始位置
      a_stride,
      context->a_packed + mr_packed_block_start);  // 预打包后矩阵 A 中的起始位置
}

// 计算 Q8GEMM 预打包稀疏矩阵的乘积
static void compute_q8gemm_prepacked_sparse_dq(
    const struct q8gemm_prepackA_sparse_dq_context context[RESTRICT_STATIC 1],
    size_t group_index, /* ignored */
    size_t pixel_index, /* ignored */
    size_t mr_block_start,  // mr 块的起始索引
    size_t nr_block_start,  // nr 块的起始索引
    size_t group_range /* always 1 */,  // 组范围，总是 1
    size_t pixel_range,  // 像素范围
    size_t mr_block_size,  // mr 块的大小
    size_t nr_block_size) {  // nr 块的大小
  const size_t mr_packed_block_start =
    ((mr_block_start >> context->log2_mr) * context->a_packed_stride);  // 计算预打包后矩阵 A 的块起始位置
  const uint8_t* restrict a_packed = context->a_packed + mr_packed_block_start;  // 预打包后矩阵 A 中块的起始位置
  const size_t c_stride = context->c_stride;  // 输出矩阵 C 的行步长
  float* restrict c =
      ((float*)context->c) + mr_block_start * c_stride + nr_block_start;  // 输出矩阵 C 中块的起始位置
  const size_t kernel_row_values_shift =
      nr_block_start >> context->log2_row_block_size;  // 计算行值偏移
  const float* bias = context->bias + nr_block_start;  // 偏置数组中块的起始位置
  const size_t output_channel_index = nr_block_start;  // 输出通道的起始索引

  switch (context->kernel_indices_dtype) {
    // 当稀疏矩阵的索引数据类型为 uint32_t 时执行以下逻辑
    case pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t:
      // 调用相应的 32 位宽度的微核函数进行计算
      context->ukernel_w32(
          mr_block_size,                              // 输入矩阵行块大小
          nr_block_size,                              // 输出矩阵列块大小
          a_packed,                                   // 预打包的稀疏矩阵数据
          context->kernel_values,                     // 卷积核值
          context->kernel_row_values_w32 + kernel_row_values_shift,  // 卷积核行索引值
          context->kernel_col_indices_w32,            // 卷积核列索引值
          bias,                                       // 偏置值
          c,                                          // 输出矩阵
          c_stride,                                   // 输出矩阵步长
          output_channel_index,                       // 输出通道索引
          &context->quantization_params);             // 量化参数
      break;

    // 当稀疏矩阵的索引数据类型为 uint16_t 时执行以下逻辑
    case pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t:
      // 调用相应的 16 位宽度的微核函数进行计算
      context->ukernel_w16(
          mr_block_size,                              // 输入矩阵行块大小
          nr_block_size,                              // 输出矩阵列块大小
          a_packed,                                   // 预打包的稀疏矩阵数据
          context->kernel_values,                     // 卷积核值
          context->kernel_row_values_w16 + kernel_row_values_shift,  // 卷积核行索引值
          context->kernel_col_indices_w16,            // 卷积核列索引值
          bias,                                       // 偏置值
          c,                                          // 输出矩阵
          c_stride,                                   // 输出矩阵步长
          output_channel_index,                       // 输出通道索引
          &context->quantization_params);             // 量化参数
      break;

    // 当稀疏矩阵的索引数据类型为 uint8_t 时执行以下逻辑
    case pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t:
      // 调用相应的 8 位宽度的微核函数进行计算
      context->ukernel_w8(
          mr_block_size,                              // 输入矩阵行块大小
          nr_block_size,                              // 输出矩阵列块大小
          a_packed,                                   // 预打包的稀疏矩阵数据
          context->kernel_values,                     // 卷积核值
          context->kernel_row_values_w8 + kernel_row_values_shift,   // 卷积核行索引值
          context->kernel_col_indices_w8,             // 卷积核列索引值
          bias,                                       // 偏置值
          c,                                          // 输出矩阵
          c_stride,                                   // 输出矩阵步长
          output_channel_index,                       // 输出通道索引
          &context->quantization_params);             // 量化参数
      break;

    // 当稀疏矩阵的索引数据类型为无效时，发生以下情况
    case pytorch_qnnp_sparse_matrix_indices_dtype_invalid:
      // 输出错误信息，表明操作无法继续
      pytorch_qnnp_log_error(
          "Invalid indices dtype specified for "
          "operator-run compute_q8gemm_prepacked_sparse_dq");
      // 断言，用于标记不应该到达此处的代码分支
      assert(false);
  }
}

// 结构体定义，用于存储 q8sum_rows 函数的上下文信息
struct q8sum_rows_context {
  const uint8_t* a;             // 输入矩阵的指针
  size_t groups;                // 分组数
  size_t m;                     // 矩阵的行数
  size_t k;                     // 矩阵的列数
  size_t a_stride;              // 输入矩阵的行步长
  const int32_t multiplier;     // 乘法器，用于计算累加和
  int32_t* a_sum;               // 每行的累加和
  size_t a_sum_stride;          // 累加和数组的行步长
  const pytorch_q8sum_rows_ukernel_function ukernel;  // 指向用于计算每行累加和的函数指针
};

// 计算每行的累加和
static void compute_sum_rows(
    const struct q8sum_rows_context context[RESTRICT_STATIC 1],  // q8sum_rows 函数的上下文信息
    size_t group_index,         // 分组索引
    size_t batch_index,         // 批次索引
    size_t block_start,         // 块起始位置
    size_t group_range /* always 1 */,  // 分组范围（始终为1）
    size_t batch_range /* always 1 */,  // 批次范围（始终为1）
    size_t block_size) {        // 块大小
  const uint8_t* a = context->a;           // 输入矩阵的指针
  const size_t groups = context->groups;   // 分组数
  const size_t m = context->m;             // 矩阵的行数
  const size_t k = context->k;             // 矩阵的列数
  const size_t a_stride = context->a_stride;  // 输入矩阵的行步长
  const int32_t multiplier = context->multiplier;  // 乘法器
  int32_t* a_sum = context->a_sum;         // 每行的累加和
  const size_t a_sum_stride = context->a_sum_stride;  // 累加和数组的行步长

  // 调用指定的 ukernel 函数，计算每行的累加和
  context->ukernel(
      a + batch_index * m * a_stride + group_index * k + block_start * a_stride,
      min(block_size, m - block_start),
      k,
      a_stride,
      multiplier,
      a_sum + batch_index * groups * a_sum_stride + group_index * a_sum_stride +
          block_start);
}

// 结构体定义，用于存储 q8gemm_xzp 函数的上下文信息
struct q8gemm_xzp_context {
  size_t k;                       // 矩阵 W 的列数
  size_t k_stride;                // 矩阵 W 的列步长
  size_t n;                       // 矩阵 C 的列数
  size_t n_stride;                // 矩阵 C 的列步长
  const uint8_t* a;               // 输入矩阵 A 的指针
  size_t a_stride;                // 输入矩阵 A 的行步长
  const void* packed_w;           // 打包后的权重矩阵 W
  uint8_t* c;                     // 输出矩阵 C 的指针
  size_t c_stride;                // 输出矩阵 C 的行步长
  const int32_t* a_sum;           // 输入矩阵 A 的每行累加和
  size_t groups;                  // 分组数
  size_t batch_size;              // 批次大小
  size_t a_sum_stride;            // 输入矩阵 A 的累加和数组的行步长
  union pytorch_qnnp_q31_requantization_params requantization_params;  // 重新量化参数
  const pytorch_q8gemm_xzp_ukernel_function ukernel;  // 指向用于计算 Q8 GEMM 的函数指针
};

// 执行 Q8 GEMM 计算
static void compute_q8gemm_xzp(
    const struct q8gemm_xzp_context context[RESTRICT_STATIC 1],  // q8gemm_xzp 函数的上下文信息
    size_t group_index,           // 分组索引
    size_t pixel_index,           // 像素索引
    size_t mr_block_start,        // MR 块的起始位置
    size_t nr_block_start,        // NR 块的起始位置
    size_t group_range /* always 1 */,  // 分组范围（始终为1）
    size_t pixel_range,           // 像素范围
    size_t mr_block_size,         // MR 块的大小
    size_t nr_block_size) {       // NR 块的大小
  const size_t k = context->k;              // 矩阵 W 的列数
  const size_t k_stride = context->k_stride;  // 矩阵 W 的列步长
  const size_t n = context->n;              // 矩阵 C 的列数
  const size_t n_stride = context->n_stride;  // 矩阵 C 的列步长
  const uint8_t* restrict a = context->a;   // 输入矩阵 A 的指针
  const size_t a_stride = context->a_stride;  // 输入矩阵 A 的行步长
  const void* restrict packed_w = context->packed_w;  // 打包后的权重矩阵 W
  uint8_t* restrict c = context->c;         // 输出矩阵 C 的指针
  const size_t c_stride = context->c_stride;  // 输出矩阵 C 的行步长
  const int32_t* a_sum = context->a_sum;    // 输入矩阵 A 的每行累加和
  const size_t groups = context->groups;    // 分组数
  const size_t a_sum_stride = context->a_sum_stride;  // 输入矩阵 A 的累加和数组的行步长

  // 调用指定的 ukernel 函数，执行 Q8 GEMM 计算
  context->ukernel(
      mr_block_size,
      nr_block_size,
      k,
      a + (pixel_index + mr_block_start) * a_stride + group_index * k,
      a_stride,
      a_sum + pixel_index * groups + group_index * a_sum_stride +
          mr_block_start,
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (k_stride * sizeof(uint8_t) + sizeof(int32_t))),
      c + (pixel_index + mr_block_start) * c_stride + nr_block_start +
          group_index * n,
      c_stride,
      &context->requantization_params);
}
struct q8conv_context {
  size_t bs;                           // 输入块大小
  size_t ks;                           // 内核大小
  size_t kc;                           // 通道数
  size_t kc_stride;                    // 通道步幅
  size_t m;                            // 输出行数
  size_t m_stride;                     // 输出行步幅
  size_t n;                            // 输出列数
  size_t n_stride;                     // 输出列步幅
  const uint8_t** indirect_a;          // 输入间接索引数组
  const void* packed_w;                // 打包的权重数据
  uint8_t* c;                          // 输出数据
  size_t c_stride;                     // 输出数据步幅
  union pytorch_qnnp_conv_quantization_params quantization_params;  // 量化参数联合体
  const pytorch_q8conv_ukernel_function ukernel;   // 卷积内核函数
};

static void compute_q8conv(
    const struct q8conv_context context[RESTRICT_STATIC 1],   // q8conv上下文结构数组
    size_t group_index,                  // 组索引
    size_t image_index,                  // 图像索引
    size_t mr_block_start,               // mr块起始位置
    size_t nr_block_start,               // nr块起始位置
    size_t group_range /* always 1 */,   // 组范围（始终为1）
    size_t image_range /* always 1 */,   // 图像范围（始终为1）
    size_t mr_block_size,                // mr块大小
    size_t nr_block_size) {              // nr块大小
  const size_t bs = context->bs;         // 输入块大小
  const size_t ks = context->ks;         // 内核大小
  const size_t kc = context->kc;         // 通道数
  const size_t kc_stride = context->kc_stride;   // 通道步幅
  const size_t m = context->m;           // 输出行数
  const size_t m_stride = context->m_stride;     // 输出行步幅
  const size_t n = context->n;           // 输出列数
  const size_t n_stride = context->n_stride;     // 输出列步幅
  const uint8_t** restrict indirect_a = context->indirect_a;   // 输入间接索引数组（限制为只读）
  const void* restrict packed_w = context->packed_w;           // 打包的权重数据（限制为只读）
  uint8_t* restrict c = context->c;     // 输出数据（限制为只读）
  const size_t c_stride = context->c_stride;   // 输出数据步幅

  size_t output_channel_index = nr_block_start + group_index * n;   // 输出通道索引计算
  context->ukernel(
      mr_block_size,                   // mr块大小
      nr_block_size,                   // nr块大小
      kc,                              // 通道数
      ks,                              // 内核大小
      indirect_a +
          (mr_block_start + (image_index + group_index * bs) * m_stride) * ks,   // 计算间接索引数组偏移量
      (const void*)((uintptr_t)packed_w + (nr_block_start + group_index * n_stride) * (kc_stride * sizeof(uint8_t) + sizeof(int32_t))),   // 计算打包权重数据的偏移量
      c + (mr_block_start + image_index * m) * c_stride + group_index * n +
          nr_block_start,              // 计算输出数据偏移量
      c_stride,                        // 输出数据步幅
      output_channel_index,            // 输出通道索引
      &context->quantization_params);  // 量化参数
}

struct q8dwconv2d_context {
  size_t groups;                        // 组数
  size_t group_stride;                  // 组步幅
  const uint8_t** indirection_buffer;   // 输入间接缓冲区
  size_t indirection_buffer_row_stride; // 输入间接缓冲区行步幅
  size_t indirection_buffer_col_stride; // 输入间接缓冲区列步幅
  const void* packed_weights;           // 打包的权重数据
  uint8_t* output;                      // 输出数据
  size_t output_height;                 // 输出高度
  size_t output_width;                  // 输出宽度
  size_t output_row_stride;             // 输出行步幅
  size_t output_col_increment;          // 输出列增量
  union pytorch_qnnp_conv_quantization_params quantization_params;  // 量化参数联合体
  union {
    const pytorch_q8dwconv2d_up_ukernel_function unipass_ukernel;    // 双通道卷积单通道处理内核函数
    const pytorch_q8dwconv2d_mp_ukernel_function multipass_ukernel;  // 双通道卷积多通道处理内核函数
  };
};

struct q8dwconv3d_context {
  size_t groups;                               // 组数
  size_t group_stride;                         // 组步幅
  const uint8_t** indirection_buffer;          // 输入间接缓冲区
  size_t indirection_buffer_slice_stride;      // 输入间接缓冲区切片步幅
  size_t indirection_buffer_row_stride;        // 输入间接缓冲区行步幅
  size_t indirection_buffer_col_stride;        // 输入间接缓冲区列步幅
  const void* packed_weights;                  // 打包的权重数据
  uint8_t* output;                             // 输出数据
  size_t output_depth;                         // 输出深度
  size_t output_height;                        // 输出高度
  size_t output_width;                         // 输出宽度
  size_t output_slice_stride;                  // 输出切片步幅
  union pytorch_qnnp_conv_quantization_params quantization_params;  // 量化参数联合体
  const pytorch_q8dwconv3d_mp_ukernel_function multipass_ukernel;    // 三维双通道卷积多通道处理内核函数
};

static void compute_dwconv2d_unipass(
    const struct q8dwconv2d_context context[RESTRICT_STATIC 1],   // 双通道二维卷积上下文结构数组
    size_t image,
    size_t output_y) {

// 定义一个名为 output_y 的 size_t 类型参数，表示输出的垂直位置

  const size_t output_height = context->output_height;

// 声明并初始化一个常量 output_height，它存储在 context 结构体中的输出高度

  context->unipass_ukernel(

// 调用 context 结构体中的 unipass_ukernel 函数，该函数执行一次卷积操作的计算

      context->groups,

// 将 context 结构体中的 groups 参数传递给 unipass_ukernel 函数，表示卷积操作的组数

      context->output_width,

// 将 context 结构体中的 output_width 参数传递给 unipass_ukernel 函数，表示卷积操作的输出宽度

      context->indirection_buffer +
          (image * output_height + output_y) *
              context->indirection_buffer_row_stride,

// 将 context 结构体中的 indirection_buffer 参数传递给 unipass_ukernel 函数，该参数通过计算确定特定像素的输入数据缓冲区位置

      context->packed_weights,

// 将 context 结构体中的 packed_weights 参数传递给 unipass_ukernel 函数，表示卷积操作使用的压缩权重数据

      context->output +
          (image * output_height + output_y) * context->output_row_stride,

// 将 context 结构体中的 output 参数传递给 unipass_ukernel 函数，该参数通过计算确定特定像素的输出数据位置

      context->indirection_buffer_col_stride,

// 将 context 结构体中的 indirection_buffer_col_stride 参数传递给 unipass_ukernel 函数，表示输入数据缓冲区中列的跨步大小

      context->output_col_increment,

// 将 context 结构体中的 output_col_increment 参数传递给 unipass_ukernel 函数，表示输出数据的列增量

      &context->quantization_params);

// 将 context 结构体中的 quantization_params 参数的地址传递给 unipass_ukernel 函数，表示卷积操作的量化参数
}

static void compute_dwconv2d_multiipass(
    const struct q8dwconv2d_context context[RESTRICT_STATIC 1],
    size_t image,
    size_t output_y) {
  const size_t output_height = context->output_height;
  // 在栈上分配用于累加的缓冲区，用于多次计算的累积结果
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_acc = _malloca(sizeof(int32_t) * context->group_stride);
#else
  int32_t multipass_acc[context->group_stride];
#endif

  // 调用多次计算深度卷积的内核函数，累加结果存储在 multipass_acc 中
  context->multipass_ukernel(
      context->groups,
      context->output_width,
      context->indirection_buffer +
          (image * output_height + output_y) *
              context->indirection_buffer_row_stride,
      context->packed_weights,
      multipass_acc,
      context->output +
          (image * output_height + output_y) * context->output_row_stride,
      context->indirection_buffer_col_stride,
      context->output_col_increment,
      &context->quantization_params);

#ifdef _MSC_VER
  _freea(multipass_acc); // 释放 _malloca 分配的内存
#endif
}

static void compute_dwconv3d_multiipass(
    const struct q8dwconv3d_context context[1],
    size_t image,
    size_t output_z) {
  const size_t output_depth = context->output_depth;
  // 在栈上分配用于累加的缓冲区，用于多次计算的累积结果
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_acc =
      (int32_t*)_malloca(sizeof(int32_t) * context->group_stride);
#else
  int32_t multipass_acc[context->group_stride];
#endif

  // 调用多次计算三维深度卷积的内核函数，累加结果存储在 multipass_acc 中
  context->multipass_ukernel(
      context->groups,
      context->output_height,
      context->output_width,
      context->indirection_buffer +
          (image * output_depth + output_z) *
              context->indirection_buffer_slice_stride,
      context->packed_weights,
      multipass_acc,
      context->output +
          (image * output_depth + output_z) * context->output_slice_stride,
      context->indirection_buffer_row_stride,
      context->indirection_buffer_col_stride,
      0,
      &context->quantization_params);

#ifdef _MSC_VER
  _freea(multipass_acc); // 释放 _malloca 分配的内存
#endif
}

struct max_pooling_context {
  const void** indirect_input;  // 输入的间接指针数组
  size_t indirect_input_batch_stride;  // 批次步长
  size_t indirect_input_height_stride;  // 高度步长
  void* output;  // 输出数据
  size_t output_batch_stride;  // 输出批次步长
  size_t output_height_stride;  // 输出高度步长
  size_t output_width;  // 输出宽度
  size_t pooling_size;  // 池化尺寸
  size_t channels;  // 通道数
  size_t input_increment;  // 输入增量
  size_t output_increment;  // 输出增量
  union pytorch_qnnp_u8_clamping_params params;  // 量化参数
  pytorch_u8maxpool_ukernel_function ukernel;  // 最大池化内核函数
};

static void compute_max_pooling(
    const struct max_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index,
    size_t output_y) {
  // 计算间接输入指针数组的位置
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input +
      batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);
  void* output =
    // 计算输出数据的地址，根据批次索引、输出数据批次步长、输出数据高度步长来计算
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  // 调用汇聚函数进行池化操作
  context->ukernel(
      // 输出图像的宽度
      context->output_width,
      // 池化窗口的大小
      context->pooling_size,
      // 图像通道数
      context->channels,
      // 间接输入数据的指针数组
      (const uint8_t**)indirect_input,
      // 输出数据的地址
      output,
      // 输入数据增量
      context->input_increment,
      // 输出数据增量
      context->output_increment,
      // 汇聚操作的参数
      &context->params);
}

struct average_pooling_context {
  const void** indirect_input;  // 指向间接输入数据的指针数组
  size_t indirect_input_batch_stride;  // 批次步幅
  size_t indirect_input_height_stride;  // 高度步幅
  void* output;  // 指向输出数据的指针
  size_t output_batch_stride;  // 输出批次步幅
  size_t output_height_stride;  // 输出高度步幅
  size_t output_width;  // 输出宽度
  size_t pooling_size;  // 池化尺寸
  size_t channels;  // 通道数
  size_t packed_channels;  // 打包的通道数
  const void* zero;  // 零值
  size_t input_increment;  // 输入增量
  size_t output_increment;  // 输出增量
  union pytorch_qnnp_avgpool_quantization_params quantization_params;  // 平均池化量化参数的联合体
  union {
    pytorch_q8avgpool_up_ukernel_function unipass_ukernel;  // 单次通道深度量化平均池化内核函数
    pytorch_q8avgpool_mp_ukernel_function multipass_ukernel;  // 多次通道深度量化平均池化内核函数
  };
};

static void compute_average_pooling_unipass(
    const struct average_pooling_context context[RESTRICT_STATIC 1],  // 平均池化单次通道深度量化函数的上下文结构体数组
    size_t batch_index,  // 批次索引
    size_t output_y) {  // 输出 Y 轴索引
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input +
      batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);  // 计算间接输入的地址
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);  // 计算输出的地址

  context->unipass_ukernel(
      context->output_width,
      context->pooling_size,
      context->channels,
      (const uint8_t**)indirect_input,
      context->zero,
      output,
      context->input_increment,
      context->output_increment,
      &context->quantization_params);  // 调用单次通道深度量化平均池化内核函数
}

static void compute_average_pooling_multipass(
    const struct average_pooling_context context[RESTRICT_STATIC 1],  // 平均池化多次通道深度量化函数的上下文结构体数组
    size_t batch_index,  // 批次索引
    size_t output_y) {  // 输出 Y 轴索引
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input +
      batch_index * context->indirect_input_batch_stride + output_y * context->indirect_input_height_stride);  // 计算间接输入的地址
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride + output_y * context->output_height_stride);  // 计算输出的地址
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  int32_t* multipass_buffer =
      _malloca(sizeof(int32_t) * context->packed_channels);  // 在堆栈上分配多通道缓冲区
#else
  int32_t multipass_buffer[context->packed_channels];  // 多通道缓冲区
#endif

  context->multipass_ukernel(
      context->output_width,
      context->pooling_size,
      context->channels,
      (const uint8_t**)indirect_input,
      context->zero,
      multipass_buffer,
      output,
      context->input_increment,
      context->output_increment,
      &context->quantization_params);  // 调用多次通道深度量化平均池化内核函数

#ifdef _MSC_VER
  _freea(multipass_buffer);  // 释放在堆栈上分配的多通道缓冲区
#endif
}

struct global_average_pooling_context {
  const void* input;  // 输入数据的指针
  const void* zero;  // 零值
  size_t input_pixel_stride;  // 输入像素步幅
  size_t input_batch_stride;  // 输入批次步幅
  size_t input_elements;  // 输入元素数
  size_t channels;  // 通道数
  size_t packed_channels;  // 打包的通道数
  void* output;  // 输出数据的指针
  size_t output_batch_stride;  // 输出批次步幅
  union pytorch_qnnp_avgpool_quantization_params quantization_params;  // 平均池化量化参数的联合体
  union {
    pytorch_q8gavgpool_up_ukernel_function unipass_ukernel;  // 全局平均池化单次通道深度量化内核函数
    pytorch_q8gavgpool_mp_ukernel_function multipass_ukernel;  // 全局平均池化多次通道深度量化内核函数
  };
};

static void compute_global_average_pooling_unipass(
    const struct global_average_pooling_context context[RESTRICT_STATIC 1],
    // 声明一个指向常量结构体 global_average_pooling_context 数组的指针 context，具有 RESTRICT_STATIC 限定符
    size_t batch_index) {
    // 声明一个大小为 size_t 类型的变量 batch_index
    
    const void* input =
        (const void*)((uintptr_t)context->input + batch_index * context->input_batch_stride);
    // 声明一个指向常量 void 类型的指针 input，通过计算得到指向输入数据的偏移量
    
    void* output =
        (void*)((uintptr_t)context->output + batch_index * context->output_batch_stride);
    // 声明一个 void 类型的指针 output，通过计算得到指向输出数据的偏移量
    
    context->unipass_ukernel(
        context->input_elements,
        context->channels,
        input,
        context->input_pixel_stride,
        context->zero,
        output,
        &context->quantization_params);
    // 调用结构体 context 中的 unipass_ukernel 函数，传递输入元素数、通道数、输入数据指针、输入像素步幅、零值、输出数据指针及量化参数的地址作为参数
// 计算全局平均池化的多通道函数，使用给定上下文和批次索引
static void compute_global_average_pooling_multipass(
    const struct global_average_pooling_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  // 根据批次索引计算输入指针位置，并将其转换为常量指针
  const void* input =
      (const void*)((uintptr_t)context->input + batch_index * context->input_batch_stride);
  // 根据批次索引计算输出指针位置
  void* output =
      (void*)((uintptr_t)context->output + batch_index * context->output_batch_stride);
  // 使用16字节对齐分配多通道缓冲区，根据平台选择不同的内存分配方式
  PYTORCH_QNNP_ALIGN(16)
#ifdef _MSC_VER
  // 在 Windows 平台上使用 _malloca 分配多通道缓冲区
  int32_t* multipass_buffer =
      _malloca(sizeof(int32_t) * context->packed_channels);
#else
  // 在其他平台上使用栈上分配分配多通道缓冲区
  int32_t multipass_buffer[context->packed_channels];
#endif

  // 调用多通道计算核函数，计算全局平均池化操作
  context->multipass_ukernel(
      context->input_elements,
      context->channels,
      input,
      context->input_pixel_stride,
      context->zero,
      multipass_buffer,
      output,
      &context->quantization_params);

#ifdef _MSC_VER
  // 在 Windows 平台上释放 _malloca 分配的多通道缓冲区
  _freea(multipass_buffer);
#endif
}

// Q8 加法操作的步进结构上下文
struct q8add_strided_context {
  size_t n;  // 元素数量
  const uint8_t* a;  // 输入张量 A 的指针
  size_t a_stride;  // 输入张量 A 的步进
  const uint8_t* b;  // 输入张量 B 的指针
  size_t b_stride;  // 输入张量 B 的步进
  const uint8_t* y;  // 输出张量 Y 的指针
  size_t y_stride;  // 输出张量 Y 的步进
  union pytorch_qnnp_add_quantization_params quantization_params;  // 量化参数结构体
  pytorch_q8vadd_ukernel_function ukernel;  // Q8 加法计算核函数
};

// 执行 Q8 加法操作，使用给定上下文和批次偏移量
static void compute_q8add_strided(
    const struct q8add_strided_context context[RESTRICT_STATIC 1],
    size_t batch_offset,
    size_t batch_range /* always 1 */) {
  // 断言批次范围为 1
  assert(batch_range == 1);

  // 从上下文中提取所需的参数和数据指针
  const size_t n = context->n;  // 元素数量
  const size_t a_stride = context->a_stride;  // 张量 A 的步进
  const size_t b_stride = context->b_stride;  // 张量 B 的步进
  const size_t y_stride = context->y_stride;  // 张量 Y 的步进
  const void* a =
      (const void*)((uintptr_t)context->a + a_stride * batch_offset);  // 计算输入张量 A 的偏移指针
  const void* b =
      (const void*)((uintptr_t)context->b + b_stride * batch_offset);  // 计算输入张量 B 的偏移指针
  void* y = (void*)((uintptr_t)context->y + y_stride * batch_offset);  // 计算输出张量 Y 的偏移指针

  // 调用 Q8 加法计算核函数，执行相应的加法操作
  context->ukernel(n, a, b, y, &context->quantization_params);
}

// Q8 加法操作的连续内存结构上下文
struct q8add_contiguous_context {
  const uint8_t* a;  // 输入张量 A 的指针
  const uint8_t* b;  // 输入张量 B 的指针
  uint8_t* y;  // 输出张量 Y 的指针
  union pytorch_qnnp_add_quantization_params quantization_params;  // 量化参数结构体
  pytorch_q8vadd_ukernel_function ukernel;  // Q8 加法计算核函数
};

// 执行 Q8 连续内存加法操作，使用给定上下文、偏移量和大小
static void compute_q8add_contiguous(
    const struct q8add_contiguous_context context[RESTRICT_STATIC 1],
    size_t offset,
    size_t size) {
  // 从上下文中提取所需的参数和数据指针
  const void* a = (const void*)((uintptr_t)context->a + offset);  // 计算输入张量 A 的偏移指针
  const void* b = (const void*)((uintptr_t)context->b + offset);  // 计算输入张量 B 的偏移指针
  void* y = (void*)((uintptr_t)context->y + offset);  // 计算输出张量 Y 的偏移指针

  // 调用 Q8 加法计算核函数，执行相应的加法操作
  context->ukernel(size, a, b, y, &context->quantization_params);
}

// 通道重排列的上下文结构
struct channel_shuffle_context {
  const void* x;  // 输入张量 X 的指针
  size_t x_stride;  // 输入张量 X 的步进
  void* y;  // 输出张量 Y 的指针
  size_t y_stride;  // 输出张量 Y 的步进
  size_t n;  // 张量维度 N
  size_t m;  // 张量维度 M
  union {
    pytorch_xzipc_ukernel_function fixed_ukernel;  // 固定通道重排列计算核函数
    pytorch_xzipv_ukernel_function variable_ukernel;  // 可变通道重排列计算核函数
  };
};

// 执行固定通道重排列操作，使用给定上下文和索引
static void compute_channel_shuffle_fixed(
    const struct channel_shuffle_context context[RESTRICT_STATIC 1],
    size_t index) {
  // 从上下文中提取输入张量 X 和输出张量 Y 的指针，并根据索引计算偏移位置
  const void* x =
      (const void*)((uintptr_t)context->x + index * context->x_stride);
  void* y = (void*)((uintptr_t)context->y + index * context->y_stride);

  // 调用固定通道重排列计算核函数，执行相应的张量重排列操作
  context->fixed_ukernel(context->n, x, y);
}
static void compute_channel_shuffle_variable(
    const struct channel_shuffle_context context[RESTRICT_STATIC 1],
    size_t index) {
  // 根据索引计算输入数据 x 的指针位置
  const void* x =
      (const void*)((uintptr_t)context->x + index * context->x_stride);
  // 根据索引计算输出数据 y 的指针位置
  void* y = (void*)((uintptr_t)context->y + index * context->y_stride);

  // 调用变量核函数，处理输入 x 和输出 y
  context->variable_ukernel(context->n, context->m, x, y);
}

struct lut_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  pytorch_x8lut_ukernel_function ukernel;
};

static void compute_lut_strided(
    const struct lut_strided_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  // 根据批次索引计算输入数据 x 的指针位置
  const void* x =
      (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
  // 根据批次索引计算输出数据 y 的指针位置
  void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);

  // 调用 Look-Up Table（LUT）核函数，处理输入 x、转换表 t 和输出 y
  context->ukernel(context->n, x, context->t, y);
}

struct lut_contiguous_context {
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  pytorch_x8lut_ukernel_function ukernel;
};

static void compute_lut_contiguous(
    const struct lut_contiguous_context context[RESTRICT_STATIC 1],
    size_t offset,
    size_t size) {
  // 根据偏移量计算输入数据 x 的指针位置
  const void* x = (const void*)((uintptr_t)context->x + offset);
  // 根据偏移量计算输出数据 y 的指针位置
  void* y = (void*)((uintptr_t)context->y + offset);

  // 调用 LUT 核函数，处理指定大小的输入 x、转换表 t 和输出 y
  context->ukernel(size, x, context->t, y);
}

struct clamp_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  pytorch_u8clamp_ukernel_function ukernel;
  union pytorch_qnnp_u8_clamping_params params;
};

static void compute_clamp_strided(
    const struct clamp_strided_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  // 根据批次索引计算输入数据 x 的指针位置
  const void* x =
      (const void*)((uintptr_t)context->x + context->x_stride * batch_index);
  // 根据批次索引计算输出数据 y 的指针位置
  void* y = (void*)((uintptr_t)context->y + context->y_stride * batch_index);

  // 调用 Clamp 核函数，处理输入 x、输出 y，并传入参数 params 进行限幅处理
  context->ukernel(context->n, x, y, &context->params);
}

struct clamp_contiguous_context {
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  pytorch_u8clamp_ukernel_function ukernel;
  union pytorch_qnnp_u8_clamping_params params;
};

static void compute_clamp_contiguous(
    const struct clamp_contiguous_context context[RESTRICT_STATIC 1],
    size_t offset,
    size_t size) {
  // 根据偏移量计算输入数据 x 的指针位置
  const void* x = (const void*)((uintptr_t)context->x + offset);
  // 根据偏移量计算输出数据 y 的指针位置
  void* y = (void*)((uintptr_t)context->y + offset);

  // 调用 Clamp 核函数，处理指定大小的输入 x、输出 y，并传入参数 params 进行限幅处理
  context->ukernel(size, x, y, &context->params);
}

struct u8softargmax_context {
  size_t n;
  const uint8_t* x;
  size_t x_stride;
  const uint32_t* t;
  uint8_t* y;
  size_t y_stride;
  pytorch_u8rmax_ukernel_function rmax_ukernel;
  pytorch_u8lut32norm_ukernel_function lut_norm_ukernel;
};

static void compute_u8softargmax(
    const struct u8softargmax_context context[RESTRICT_STATIC 1],
    size_t batch_index) {
  // 根据批次索引计算输入数据 x 的指针位置
  const uint8_t* x =
      (const uint8_t*)((uintptr_t)context->x + context->x_stride * batch_index);
  // 根据批次索引计算输出数据 y 的指针位置
  uint8_t* y = (uint8_t*)((uintptr_t)context->y + context->y_stride * batch_index);

  // 调用 Softargmax 核函数，处理输入 x、转换表 t 和输出 y
  context->rmax_ukernel(context->n, x, context->t, y);
  // 调用 LUT 32位归一化核函数，对输出 y 进行归一化处理
  context->lut_norm_ukernel(context->n, y, y);
}
    size_t batch_index) {
```  
# 定义函数 `normalize`，接受一个 `batch_index` 参数，用于处理输入和输出的指针。  


  const uint8_t* x =
      (const uint8_t*)((uintptr_t)context->x + context->x_stride * batch_index);
```py  
# 从输入上下文中获取输入数据指针 `x`，通过对 `x_stride` 乘以 `batch_index` 计算出在批处理中的偏移位置，并将其转换为 `const uint8_t*` 类型。  


  uint8_t* y =
      (uint8_t*)((uintptr_t)context->y + context->y_stride * batch_index);
```  
# 从输出上下文中获取输出数据指针 `y`，通过对 `y_stride` 乘以 `batch_index` 计算出在批处理中的偏移位置，并将其转换为 `uint8_t*` 类型。  


  const size_t n = context->n;
```py  
# 获取上下文中的 `n` 值，表示处理的元素数量。  


  const uint8_t x_max = context->rmax_ukernel(n, x);
```  
# 调用上下文中的 `rmax_ukernel` 函数，计算输入数据 `x` 的最大值 `x_max`。  


  const size_t adjustment = x_max ^ 255;
```py  
# 计算一个调整值 `adjustment`，通过对 `x_max` 取异或 `255` 得到。  


  const uint32_t* t = (const uint32_t*)context->t + adjustment;
```  
# 根据计算出的 `adjustment` 值来调整上下文中的 `t` 指针，使其指向正确的位置，并将其转换为 `const uint32_t*` 类型。  


  context->lut_norm_ukernel(n, x, t, y);
```py  
# 调用上下文中的 `lut_norm_ukernel` 函数，执行归一化处理，将处理后的结果存储到输出数据 `y` 中。  
// 为了任何 ukernel 类型，在批处理大小为 0 时没有工作要做。
if (op->batch_size == 0) {
    返回成功状态，表示运算成功完成。
    return pytorch_qnnp_status_success;
}

switch (op->ukernel_type) {
    case pytorch_qnnp_ukernel_type_xzp_gemm: {
        // 获取操作的批处理大小、组数、组输入通道数和组输出通道数
        const size_t batch_size = op->batch_size;
        const size_t groups = op->groups;
        const size_t group_input_channels = op->group_input_channels;
        const size_t group_output_channels = op->group_output_channels;
        
        // 从参数中获取 q8conv_xzp 的块大小 mr、nr 和 kr
        const uint32_t mr = pytorch_qnnp_params.q8conv_xzp.mr;
        const uint32_t nr = pytorch_qnnp_params.q8conv_xzp.nr;
        const uint32_t kr = pytorch_qnnp_params.q8conv_xzp.kr;
        
        // 计算输入和输出通道的对齐增量
        const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
        const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;

        /* 计算输入行和 */
        // 计算输入图像的总大小
        const size_t input_size = op->input_height * op->input_width;
        // 将 op->a_sum 强制类型转换为 int32_t 指针，作为输入行和数组
        int32_t* a_sum = (int32_t*)op->a_sum;

        // 设置 q8sum_rows 函数的上下文
        struct q8sum_rows_context context = {
            .a = op->input,
            .groups = groups,
            .m = input_size,
            .k = group_input_channels,
            .a_stride = op->input_pixel_stride,
            .multiplier = (int32_t)-op->kernel_zero_point,
            .a_sum = a_sum,
            .a_sum_stride = input_size,
            .ukernel = pytorch_qnnp_params.q8sum_rows.sum_rows,
        };
        // 使用线程池计算 3D 块状数据的和行
        pthreadpool_compute_3d_tiled(
            threadpool,
            (pthreadpool_function_3d_tiled_t)compute_sum_rows,
            &context,
            groups,
            batch_size,
            input_size,
            1,
            1,
            pytorch_qnnp_params.q8sum_rows.m);

        // 设置 q8gemm_xzp 函数的上下文
        struct q8gemm_xzp_context q8gemm_xzp_context = {
            .k = group_input_channels,
            .k_stride = k_stride,
            .n = group_output_channels,
            .n_stride = n_stride,
            .a = op->input,
            .a_stride = op->input_pixel_stride,
            .packed_w = op->packed_weights,
            .c = op->output,
            .c_stride = op->output_pixel_stride,
            .a_sum = a_sum,
            .groups = op->groups,
            .batch_size = batch_size,
            .a_sum_stride = input_size,
            .requantization_params = op->requantization_params,
            .ukernel = pytorch_qnnp_params.q8conv_xzp.gemm,
        };
        // 使用线程池计算 4D 块状数据的 q8gemm_xzp 运算
        pthreadpool_compute_4d_tiled(
            threadpool,
            (pthreadpool_function_4d_tiled_t)compute_q8gemm_xzp,
            &q8gemm_xzp_context,
            groups,
            batch_size * input_size,
            input_size,
            group_output_channels,
            1,
            input_size,
            mr,
            nr);
        break;
    }
}
    // 对于 pytorch_qnnp_ukernel_type_gemm 类型的操作进行处理
    case pytorch_qnnp_ukernel_type_gemm: {
      // 提取操作中的批处理大小、组数、每组输入通道数和输出通道数等信息
      const size_t batch_size = op->batch_size;
      const size_t groups = op->groups;
      const size_t group_input_channels = op->group_input_channels;
      const size_t group_output_channels = op->group_output_channels;
      // 从 QNNPACK 参数中获取矩阵乘法的块大小 mr、nr 和 kr
      const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
      const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
      const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
      // 计算输入通道和输出通道的对齐长度
      const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
      const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;
      // 计算输出的深度，如果为 0 则设为 1，然后计算输出的总大小
      const size_t output_depth = op->output_depth;
      const size_t output_size = (output_depth != 0 ? output_depth : 1) *
          op->output_height * op->output_width;
    
      // 初始化 q8gemm_context 结构体，设置矩阵乘法的上下文信息
      struct q8gemm_context q8gemm_context = {
          .k = group_input_channels,
          .k_stride = k_stride,
          .n = group_output_channels,
          .n_stride = n_stride,
          .a = op->input,
          .a_stride = op->input_pixel_stride,
          .packed_w = op->packed_weights,
          .c = op->output,
          .c_stride = op->output_pixel_stride,
          .quantization_params = op->conv_quantization_params,
          .ukernel = pytorch_qnnp_params.q8conv.gemm,
      };
    
      // 使用线程池进行 4 维瓦片化计算，调用 compute_q8gemm 函数处理乘法
      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t)compute_q8gemm,
          &q8gemm_context,
          groups,
          batch_size * output_size,
          output_size,
          group_output_channels,
          1,
          output_size,
          mr,
          nr);
      // 结束当前 case 分支
      break;
    }
#ifdef NO_PREPACK_SPARSE_KERNEL
    // 如果未定义宏 NO_PREPACK_SPARSE_KERNEL，则不编译以下代码块
    case pytorch_qnnp_ukernel_type_gemm_sparse_dq: {
      // 当前操作类型为稀疏量化矩阵乘法
      const size_t batch_size = op->batch_size;  // 批处理大小
      const size_t groups = op->groups;  // 分组数量
      const size_t group_output_channels = op->group_output_channels;  // 每组输出通道数
      const uint32_t mr = pytorch_qnnp_params.q8gemm_sparse_c1x4.mr;  // MR 参数
      const uint32_t nr = pytorch_qnnp_params.q8gemm_sparse_c1x4.nr;  // NR 参数

      const size_t output_size = op->output_height * op->output_width;  // 输出大小
      // 初始化稀疏量化矩阵乘法上下文结构体
      struct q8gemm_sparse_dq_context q8gemm_sparse_dq_context = {
          .a = op->input,  // 输入数据
          .a_stride = op->input_pixel_stride,  // 输入像素步长
          .kernel_col_indices = op->sparse_matrix.col_indices,  // 稀疏矩阵列索引
          .kernel_row_values = op->sparse_matrix.row_values,  // 稀疏矩阵行值
          .kernel_values = op->sparse_matrix.values,  // 稀疏矩阵值
          .bias = (const float*)op->bias,  // 偏置值
          .c = (float*)op->output,  // 输出数据
          .c_stride = op->output_pixel_stride,  // 输出像素步长
          .quantization_params = op->dynamic_conv_quantization_params,  // 量化参数
          .ukernel = pytorch_qnnp_params.q8gemm_sparse_c1x4.gemm_dq,  // 稀疏矩阵乘法函数
      };

      // 使用线程池并行计算稀疏量化矩阵乘法
      pthreadpool_compute_4d_tiled(
          threadpool,
          (pthreadpool_function_4d_tiled_t)compute_q8gemm_sparse_dq,  // 计算函数
          &q8gemm_sparse_dq_context,  // 上下文结构体指针
          groups,  // 分组数
          batch_size * output_size,  // 总批处理大小
          output_size,  // 输出大小
          group_output_channels,  // 每组输出通道数
          1,  // 输入通道数（固定为1）
          output_size,  // 输出大小
          mr,  // MR 参数
          nr);  // NR 参数
      break;  // 结束当前 case 分支
    }
#endif
    }
    // 对于 pytorch_qnnp_ukernel_type_conv 类型的情况，执行卷积操作
    case pytorch_qnnp_ukernel_type_conv: {
        // 提取操作中的批处理大小
        const size_t batch_size = op->batch_size;
        // 提取操作中的分组数
        const size_t groups = op->groups;
        // 提取操作中的每组输入通道数
        const size_t group_input_channels = op->group_input_channels;
        // 提取操作中的每组输出通道数
        const size_t group_output_channels = op->group_output_channels;
        // 提取 QNNPACK 中 Q8Conv 的内部参数 mr, nr, kr
        const uint32_t mr = pytorch_qnnp_params.q8conv.mr;
        const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
        const uint32_t kr = pytorch_qnnp_params.q8conv.kr;
        // 计算输入通道数的步进值，以确保是 kr 的倍数
        const size_t k_stride = (group_input_channels + (kr - 1)) & -kr;
        // 计算输出通道数的步进值，以确保是 nr 的倍数
        const size_t n_stride = (group_output_channels + (nr - 1)) & -nr;
        // 提取操作中的输出深度
        const size_t output_depth = op->output_depth;
        // 计算输出大小，考虑到深度是否为零的情况
        const size_t output_size = (output_depth != 0 ? output_depth : 1) *
                                   op->output_height * op->output_width;
        // 提取操作中的卷积核深度
        const size_t kernel_depth = op->kernel_depth;
        // 计算卷积核大小，考虑到深度是否为零的情况
        const size_t kernel_size = (kernel_depth != 0 ? kernel_depth : 1) *
                                   op->kernel_height * op->kernel_width;
        // 计算输出大小的对齐值，以确保是 mr 的倍数
        const size_t m_stride = round_up(output_size, mr);
    
        // 创建 Q8Conv 上下文结构体，并初始化各个字段
        struct q8conv_context q8conv_context = {
            .bs = batch_size,
            .ks = kernel_size,
            .kc = group_input_channels,
            .kc_stride = k_stride * kernel_size,
            .m = output_size,
            .m_stride = m_stride,
            .n = group_output_channels,
            .n_stride = n_stride,
            .indirect_a = (const uint8_t**)op->indirection_buffer,
            .packed_w = op->packed_weights,
            .c = op->output,
            .c_stride = op->output_pixel_stride,
            .quantization_params = op->conv_quantization_params,
            .ukernel = pytorch_qnnp_params.q8conv.conv,
        };
    
        // 使用线程池执行四维分块计算，调用 compute_q8conv 函数
        pthreadpool_compute_4d_tiled(
            threadpool,
            (pthreadpool_function_4d_tiled_t)compute_q8conv,
            &q8conv_context,
            groups,
            batch_size,
            output_size,
            group_output_channels,
            1,
            1,
            mr,
            nr);
        // 结束当前 case 分支
        break;
    }
    // 如果是平均池化类型的运算
    case pytorch_qnnp_ukernel_type_average_pooling: {
      // 从参数结构中获取池化的各种尺寸和限制条件
      const uint32_t kr = pytorch_qnnp_params.q8avgpool.kr;  // kernel reduction size
      const uint32_t mr = pytorch_qnnp_params.q8avgpool.mr;  // min pooling size for multipass
      const uint32_t qr = pytorch_qnnp_params.q8avgpool.qr;  // quantum reduction size
      const size_t channels = op->channels;  // 输入通道数
      const size_t output_width = op->output_width;  // 输出宽度
      const size_t output_height = op->output_height;  // 输出高度
      const size_t pooling_height = op->kernel_height;  // 池化核高度
      const size_t pooling_width = op->kernel_width;  // 池化核宽度
      const size_t pooling_size = pooling_height * pooling_width;  // 池化核大小

      // 计算间接输入高度步幅和输出高度步幅
      const size_t indirect_input_height_stride =
          op->step_height * sizeof(void*);
      const size_t output_height_stride =
          output_width * op->output_pixel_stride;

      // 多通道处理调整量的计算
      size_t multipass_adjustment = 0;
      if (channels >= kr && pooling_size > mr) {
        multipass_adjustment = round_up(pooling_size - mr, qr) + mr - qr;
      }

      // 创建池化上下文结构体，包含所有池化操作的参数和缓冲区
      struct average_pooling_context context = {
          .indirect_input = op->indirection_buffer,
          .indirect_input_batch_stride =
              output_height * indirect_input_height_stride,
          .indirect_input_height_stride = indirect_input_height_stride,
          .output = op->output,
          .output_batch_stride = output_height * output_height_stride,
          .output_height_stride = output_height_stride,
          .output_width = output_width,
          .pooling_size = pooling_size,
          .channels = channels,
          .packed_channels = (channels + (kr - 1)) & -kr,
          .zero = op->zero_pointer,
          .input_increment =
              (pooling_height * op->step_width - multipass_adjustment) *
              sizeof(void*),
          .output_increment =
              (op->output_pixel_stride - channels) * sizeof(uint8_t),
          .quantization_params = op->avgpool_quantization_params,
      };

      // 根据通道数和池化核大小选择合适的计算函数和内核
      pthreadpool_function_2d_t compute_function = NULL;
      if (channels < kr) {
        compute_function =
            (pthreadpool_function_2d_t)compute_average_pooling_unipass;
        context.unipass_ukernel = pytorch_qnnp_params.q8avgpool.ltkr;
      } else {
        if (pooling_size <= mr) {
          compute_function =
              (pthreadpool_function_2d_t)compute_average_pooling_unipass;
          context.unipass_ukernel = pytorch_qnnp_params.q8avgpool.gekr_lemr;
        } else {
          compute_function =
              (pthreadpool_function_2d_t)compute_average_pooling_multipass;
          context.multipass_ukernel = pytorch_qnnp_params.q8avgpool.gekr_gtmr;
        }
      }

      // 使用线程池计算二维池化操作
      pthreadpool_compute_2d(
          threadpool,
          compute_function,
          &context,
          op->batch_size,
          output_height);
      break;
    }
    // 对于最大池化操作类型，设置相关参数和常量
    case pytorch_qnnp_ukernel_type_max_pooling: {
      // 从参数结构体中获取池化操作所需的 kr、mr、qr 值
      const uint32_t kr = pytorch_qnnp_params.u8maxpool.kr;
      const uint32_t mr = pytorch_qnnp_params.u8maxpool.mr;
      const uint32_t qr = pytorch_qnnp_params.u8maxpool.qr;
      // 获取输入数据的通道数、输出宽度和高度，以及池化核的尺寸
      const size_t channels = op->channels;
      const size_t output_width = op->output_width;
      const size_t output_height = op->output_height;
      const size_t pooling_height = op->kernel_height;
      const size_t pooling_width = op->kernel_width;
      const size_t pooling_size = pooling_height * pooling_width;

      // 计算间接输入数据在高度上的步长
      const size_t indirect_input_height_stride =
          op->step_height * sizeof(void*);
      // 计算输出数据在高度上的步长
      const size_t output_height_stride =
          output_width * op->output_pixel_stride;

      // 计算多通道池化时的调整值
      size_t multipass_adjustment = pooling_size;
      if (channels >= kr) {
        multipass_adjustment = round_up(doz(pooling_size, mr), qr) + mr;
      }
      
      // 创建最大池化操作的上下文结构体
      struct max_pooling_context context = {
          .indirect_input = op->indirection_buffer,
          .indirect_input_batch_stride =
              output_height * indirect_input_height_stride,
          .indirect_input_height_stride = indirect_input_height_stride,
          .output = op->output,
          .output_batch_stride = output_height * output_height_stride,
          .output_height_stride = output_height_stride,
          .output_width = output_width,
          .pooling_size = pooling_size,
          .channels = channels,
          .input_increment =
              (pooling_height * op->step_width - multipass_adjustment) *
              sizeof(void*),
          .output_increment =
              (op->output_pixel_stride - channels) * sizeof(uint8_t),
          .params = op->u8_clamping_params,
          .ukernel = channels < kr ? pytorch_qnnp_params.u8maxpool.ltkr
                                   : pytorch_qnnp_params.u8maxpool.gekr,
      };

      // 调用线程池进行二维最大池化计算
      pthreadpool_compute_2d(
          threadpool,
          (pthreadpool_function_2d_t)compute_max_pooling,
          &context,
          op->batch_size,
          output_height);
      // 结束 switch 语句块
      break;
    };
    # 当操作类型为 pytorch_qnnp_ukernel_type_add 时执行以下代码块
    case pytorch_qnnp_ukernel_type_add: {
      # 从操作结构体中获取批次大小、通道数以及输入和输出数据的步长
      const size_t batch_size = op->batch_size;
      const size_t channels = op->channels;
      const size_t a_stride = op->input_pixel_stride;
      const size_t b_stride = op->input2_pixel_stride;
      const size_t y_stride = op->output_pixel_stride;
      
      # 检查输入数据的步长是否与通道数一致，并且批次大小是否为1，如果是则执行以下逻辑
      if ((((a_stride ^ channels) | (b_stride ^ channels) |
            (y_stride ^ channels)) == 0) ||
          batch_size == 1) {
        # 定义块大小为4096
        const size_t block_size = 4096;
        # 定义包含连续数据加法运算所需的上下文结构体
        struct q8add_contiguous_context add_context = {
            .a = op->input,
            .b = op->input2,
            .y = op->output,
            .quantization_params = op->add_quantization_params,
            .ukernel = pytorch_qnnp_params.q8vadd,
        };
        # 使用线程池计算连续数据加法
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_q8add_contiguous,
            &add_context,
            batch_size * channels * sizeof(uint8_t),
            block_size);
      } else {
        # 定义包含分段数据加法运算所需的上下文结构体
        struct q8add_strided_context add_context = {
            .a = op->input,
            .a_stride = a_stride * sizeof(uint8_t),
            .b = op->input2,
            .b_stride = b_stride * sizeof(uint8_t),
            .y = op->output,
            .y_stride = y_stride * sizeof(uint8_t),
            .n = channels,
            .quantization_params = op->add_quantization_params,
            .ukernel = pytorch_qnnp_params.q8vadd,
        };
        # 使用线程池计算分段数据加法
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_q8add_strided,
            &add_context,
            batch_size,
            1);
      }
      break;
    }
    # 根据全局平均池化类型执行不同的操作
    case pytorch_qnnp_ukernel_type_global_average_pooling: {
      # 从参数中获取全局平均池化所需的一些常数
      const uint32_t nr = pytorch_qnnp_params.q8gavgpool.nr;  // 常数 nr
      const uint32_t mr = pytorch_qnnp_params.q8gavgpool.mr;  // 常数 mr
      # 计算输入数据的像素步幅，以字节为单位
      const size_t input_pixel_stride =
          op->input_pixel_stride * sizeof(uint8_t);
      # 输入数据的宽度
      const size_t input_width = op->input_width;
      # 输入数据的通道数
      const size_t channels = op->channels;
      
      # 定义全局平均池化的上下文结构体，包括输入、零指针、像素步幅等信息
      struct global_average_pooling_context context = {
          .input = op->input,                                      // 输入数据指针
          .zero = op->zero_pointer,                                // 零指针
          .input_pixel_stride = input_pixel_stride,                // 输入数据的像素步幅
          .input_batch_stride = input_pixel_stride * input_width,  // 输入数据的批次步幅
          .input_elements = input_width,                           // 输入数据的元素个数（宽度）
          .channels = channels,                                    // 输入数据的通道数
          .packed_channels = (channels + (nr - 1)) & -nr,           // 对齐后的通道数
          .output = op->output,                                    // 输出数据指针
          .output_batch_stride = op->output_pixel_stride * sizeof(uint8_t),  // 输出数据的批次步幅
          .quantization_params = op->avgpool_quantization_params,   // 平均池化的量化参数
      };
      
      # 根据通道数和输入宽度选择合适的计算函数
      pthreadpool_function_1d_t compute_function = NULL;
      if (channels < nr) {
        compute_function =
            (pthreadpool_function_1d_t)compute_global_average_pooling_unipass;
        context.unipass_ukernel = pytorch_qnnp_params.q8gavgpool.ltnr;  // 选择单通道计算内核
      } else {
        if (input_width <= mr) {
          compute_function =
              (pthreadpool_function_1d_t)compute_global_average_pooling_unipass;
          context.unipass_ukernel = pytorch_qnnp_params.q8gavgpool.genr_lemr;  // 选择小尺寸多通道计算内核
        } else {
          compute_function = (pthreadpool_function_1d_t)
              compute_global_average_pooling_multipass;  // 选择大尺寸多通道计算内核
          context.multipass_ukernel = pytorch_qnnp_params.q8gavgpool.genr_gtmr;
        }
      }
      
      # 使用线程池执行指定的计算函数
      pthreadpool_compute_1d(
          threadpool, compute_function, &context, op->batch_size);  // 在线程池中执行计算
      break;  // 结束当前的 case 块
    }
    case pytorch_qnnp_ukernel_type_lut: {
      // 获取操作的批处理大小、通道数、输入和输出像素步长
      const size_t batch_size = op->batch_size;
      const size_t channels = op->channels;
      const size_t x_stride = op->input_pixel_stride;
      const size_t y_stride = op->output_pixel_stride;
      // 如果输入和输出步长与通道数一致，或者批处理大小为1，则执行连续的查找表操作
      if ((((x_stride ^ channels) | (y_stride ^ channels)) == 0) ||
          batch_size == 1) {
        // 定义块大小为1024
        const size_t block_size = 1024;
        // 定义连续查找表上下文
        struct lut_contiguous_context context = {
            .x = op->input,                             // 输入数据指针
            .x_stride = x_stride * sizeof(uint8_t),     // 输入数据的字节步长
            .t = op->lookup_table,                      // 查找表数据指针
            .y = op->output,                            // 输出数据指针
            .y_stride = y_stride * sizeof(uint8_t),     // 输出数据的字节步长
            .ukernel = pytorch_qnnp_params.x8lut,       // 使用的内核函数
        };
        // 使用线程池执行一维分块计算，调用compute_lut_contiguous函数
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_lut_contiguous,
            &context,
            batch_size * channels * sizeof(uint8_t),    // 计算范围大小
            block_size);                                // 块大小
      } else {
        // 定义分步查找表上下文
        struct lut_strided_context context = {
            .n = channels,                              // 通道数
            .x = op->input,                             // 输入数据指针
            .x_stride = x_stride * sizeof(uint8_t),     // 输入数据的字节步长
            .t = op->lookup_table,                      // 查找表数据指针
            .y = op->output,                            // 输出数据指针
            .y_stride = y_stride * sizeof(uint8_t),     // 输出数据的字节步长
            .ukernel = pytorch_qnnp_params.x8lut,       // 使用的内核函数
        };
        // 使用线程池执行一维计算，调用compute_lut_strided函数
        pthreadpool_compute_1d(
            threadpool,
            (pthreadpool_function_1d_t)compute_lut_strided,
            &context,
            batch_size);                                // 批处理大小
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_clamp: {
      // 获取操作的批处理大小、通道数、输入和输出像素步长
      const size_t batch_size = op->batch_size;
      const size_t channels = op->channels;
      const size_t x_stride = op->input_pixel_stride;
      const size_t y_stride = op->output_pixel_stride;
      // 如果输入和输出步长与通道数一致，或者批处理大小为1，则执行连续的clamp操作
      if ((((x_stride ^ channels) | (y_stride ^ channels)) == 0) ||
          batch_size == 1) {
        // 定义块大小为4096
        const size_t block_size = 4096;
        // 定义连续clamp操作上下文
        struct clamp_contiguous_context context = {
            .x = op->input,                             // 输入数据指针
            .x_stride = x_stride * sizeof(uint8_t),     // 输入数据的字节步长
            .y = op->output,                            // 输出数据指针
            .y_stride = y_stride * sizeof(uint8_t),     // 输出数据的字节步长
            .ukernel = pytorch_qnnp_params.u8clamp,     // 使用的内核函数
            .params = op->u8_clamping_params,           // clamp参数
        };
        // 使用线程池执行一维分块计算，调用compute_clamp_contiguous函数
        pthreadpool_compute_1d_tiled(
            threadpool,
            (pthreadpool_function_1d_tiled_t)compute_clamp_contiguous,
            &context,
            batch_size * channels * sizeof(uint8_t),    // 计算范围大小
            block_size);                                // 块大小
      } else {
        // 定义分步clamp操作上下文
        struct clamp_strided_context context = {
            .n = channels,                              // 通道数
            .x = op->input,                             // 输入数据指针
            .x_stride = x_stride * sizeof(uint8_t),     // 输入数据的字节步长
            .y = op->output,                            // 输出数据指针
            .y_stride = y_stride * sizeof(uint8_t),     // 输出数据的字节步长
            .ukernel = pytorch_qnnp_params.u8clamp,     // 使用的内核函数
            .params = op->u8_clamping_params,           // clamp参数
        };
        // 使用线程池执行一维计算，调用compute_clamp_strided函数
        pthreadpool_compute_1d(
            threadpool,
            (pthreadpool_function_1d_t)compute_clamp_strided,
            &context,
            batch_size);                                // 批处理大小
      }
      break;
    }
    case pytorch_qnnp_ukernel_type_softargmax: {
      // 定义 softargmax 操作的上下文结构体
      struct u8softargmax_context context = {
          // 通道数
          .n = op->channels,
          // 输入数据指针
          .x = op->input,
          // 输入数据步长（每像素字节数）
          .x_stride = op->input_pixel_stride * sizeof(uint8_t),
          // 查找表指针
          .t = op->lookup_table,
          // 输出数据指针
          .y = op->output,
          // 输出数据步长（每像素字节数）
          .y_stride = op->output_pixel_stride * sizeof(uint8_t),
          // rmax 操作的内核函数
          .rmax_ukernel = pytorch_qnnp_params.u8rmax,
          // lut32norm 操作的内核函数
          .lut_norm_ukernel = pytorch_qnnp_params.u8lut32norm,
      };
      // 使用线程池并行计算 softargmax
      pthreadpool_compute_1d(
          threadpool,
          (pthreadpool_function_1d_t)compute_u8softargmax,
          &context,
          op->batch_size);
      break;
    }
    case pytorch_qnnp_ukernel_type_channel_shuffle: {
      // 获取通道重排操作的分组数
      const size_t groups = op->groups;
      // 定义 channel shuffle 操作的上下文结构体
      struct channel_shuffle_context channel_shuffle_context = {
          // 输入数据指针
          .x = op->input,
          // 输入数据步长（每像素字节数）
          .x_stride = op->input_pixel_stride * sizeof(uint8_t),
          // 输出数据指针
          .y = op->output,
          // 输出数据步长（每像素字节数）
          .y_stride = op->output_pixel_stride * sizeof(uint8_t),
          // 每组通道数
          .n = op->group_channels * sizeof(uint8_t),
          // 分组数
          .m = groups,
      };
      // 初始化计算函数指针为空
      pthreadpool_function_1d_t compute_function = NULL;
      // 根据分组数选择合适的计算函数和固定或可变内核
      switch (groups) {
        case 2:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
          channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x2;
          break;
        case 3:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
          channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x3;
          break;
        case 4:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_fixed;
          channel_shuffle_context.fixed_ukernel = pytorch_qnnp_params.x8zip.x4;
          break;
        default:
          compute_function =
              (pthreadpool_function_1d_t)compute_channel_shuffle_variable;
          channel_shuffle_context.variable_ukernel =
              pytorch_qnnp_params.x8zip.xm;
          break;
        case 0:
        case 1:
          // 不可达分支，用于标记错误状态
          PYTORCH_QNNP_UNREACHABLE;
      }
      // 使用线程池并行计算通道重排
      pthreadpool_compute_1d(
          threadpool,
          compute_function,
          &channel_shuffle_context,
          op->batch_size);
      break;
    }
    default:
      // 不可达分支，用于标记错误状态
      PYTORCH_QNNP_UNREACHABLE;
  }
  // 返回操作成功状态
  return pytorch_qnnp_status_success;
}


注释：

# 这行代码表示一个单独的右花括号 '}'，用于结束一个代码块或数据结构的定义。
# 在程序中，右花括号通常与左花括号 '{' 配对出现，用于界定代码块的开始和结束。
# 在此处，它可能用于结束一个函数、条件语句、循环或其他代码结构的定义。
```