# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\include\pytorch_qnnpack.h`

```
/*
 * 版权所有（c）Facebook，Inc.及其关联公司。
 * 保留所有权利。
 *
 * 此源代码在根目录中的LICENSE文件中以BSD样式许可证授权。
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <pthreadpool.h>    // 导入多线程池头文件
#include <qnnpack/log.h>    // 导入QNNPACK日志头文件

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief QNNPACK函数调用的状态码。
 */
enum pytorch_qnnp_status {
  /** 调用成功，并且所有输出参数现在包含有效数据。 */
  pytorch_qnnp_status_success = 0,                       // 成功状态码
  pytorch_qnnp_status_uninitialized = 1,                 // 未初始化状态码
  pytorch_qnnp_status_invalid_parameter = 2,             // 无效参数状态码
  pytorch_qnnp_status_unsupported_parameter = 3,         // 不支持的参数状态码
  pytorch_qnnp_status_unsupported_hardware = 4,          // 不支持的硬件状态码
  pytorch_qnnp_status_out_of_memory = 5,                 // 内存不足状态码
};

/**
 * @brief 稀疏矩阵索引的数据类型。
 */
enum pytorch_qnnp_sparse_matrix_indices_dtype {
  pytorch_qnnp_sparse_matrix_indices_dtype_invalid = 0,   // 无效的数据类型
  pytorch_qnnp_sparse_matrix_indices_dtype_uint8_t = 8,   // 8位无符号整数类型
  pytorch_qnnp_sparse_matrix_indices_dtype_uint16_t = 16, // 16位无符号整数类型
  pytorch_qnnp_sparse_matrix_indices_dtype_uint32_t = 32, // 32位无符号整数类型
};

/**
 * @brief 初始化QNNPACK库。
 */
enum pytorch_qnnp_status pytorch_qnnp_initialize(void);

/**
 * @brief 反初始化QNNPACK库。
 */
enum pytorch_qnnp_status pytorch_qnnp_deinitialize(void);

/**
 * @brief QNNPACK创建二维NHWC布局的量化卷积操作符。
 */
typedef struct pytorch_qnnp_operator* pytorch_qnnp_operator_t;

/**
 * @brief 创建二维NHWC布局的量化卷积操作符。
 */
enum pytorch_qnnp_status pytorch_qnnp_create_convolution2d_nhwc_q8(
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution);

/**
 * @brief 创建三维NDHWC布局的量化卷积操作符。
 */
enum pytorch_qnnp_status pytorch_qnnp_create_convolution3d_ndhwc_q8(
    uint32_t input_padding_depth,
    uint32_t input_padding_height,
    uint32_t input_padding_width,
    uint32_t kernel_depth,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_depth,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_depth,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    uint8_t input_zero_point,
    const uint8_t* kernel_zero_points,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    const float* requantization_scales,
    bool per_channel,
    pytorch_qnnp_operator_t* convolution);

/**
 * @brief 设置二维NHWC布局的量化卷积操作符。
 */
enum pytorch_qnnp_status pytorch_qnnp_setup_convolution2d_nhwc_q8(
    pytorch_qnnp_operator_t convolution,
    size_t batch_size,
    // 定义函数参数：输入图像的高度
    size_t input_height,
    // 定义函数参数：输入图像的宽度
    size_t input_width,
    // 定义函数参数：指向输入图像数据的指针，数据类型为 uint8_t（8位无符号整数）
    const uint8_t* input,
    // 定义函数参数：输入图像每行数据的跨度（即每行数据的字节数）
    size_t input_stride,
    // 定义函数参数：指向输出图像数据的指针，数据类型为 uint8_t
    uint8_t* output,
    // 定义函数参数：输出图像每行数据的跨度（即每行数据的字节数）
    size_t output_stride,
    // 定义函数参数：线程池对象，用于并行执行任务
    pthreadpool_t threadpool);
// 设置一个卷积操作的 QNNP 运算符，基于 N-D HWC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_convolution_ndhwc_q8(
    // 要设置的卷积运算符
    pytorch_qnnp_operator_t convolution,
    // 批量大小
    size_t batch_size,
    // 输入深度
    size_t input_depth,
    // 输入高度
    size_t input_height,
    // 输入宽度
    size_t input_width,
    // 输入数据的指针
    const uint8_t* input,
    // 输入数据的跨度
    size_t input_stride,
    // 输出数据的指针
    uint8_t* output,
    // 输出数据的跨度
    size_t output_stride,
    // 线程池用于并行计算
    pthreadpool_t threadpool);

// 创建一个 QNNP 的反卷积 2D 运算符，基于 NHWC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_create_deconvolution2d_nhwc_q8(
    // 输入填充高度
    uint32_t input_padding_height,
    // 输入填充宽度
    uint32_t input_padding_width,
    // 高度调整
    uint32_t adjustment_height,
    // 宽度调整
    uint32_t adjustment_width,
    // 卷积核高度
    uint32_t kernel_height,
    // 卷积核宽度
    uint32_t kernel_width,
    // 高度步长
    uint32_t stride_height,
    // 宽度步长
    uint32_t stride_width,
    // 高度膨胀率
    uint32_t dilation_height,
    // 宽度膨胀率
    uint32_t dilation_width,
    // 分组数
    uint32_t groups,
    // 分组输入通道数
    size_t group_input_channels,
    // 分组输出通道数
    size_t group_output_channels,
    // 输入的零点
    uint8_t input_zero_point,
    // 卷积核的零点数组
    const uint8_t* kernel_zero_points,
    // 卷积核数据
    const uint8_t* kernel,
    // 偏置数组
    const int32_t* bias,
    // 输出的零点
    uint8_t output_zero_point,
    // 输出的最小值
    uint8_t output_min,
    // 输出的最大值
    uint8_t output_max,
    // 标志位
    uint32_t flags,
    // 重新量化的比例数组
    const float* requantization_scales,
    // 输出的反卷积运算符
    pytorch_qnnp_operator_t* deconvolution);

// 设置一个 QNNP 的反卷积 2D 运算符，基于 NHWC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_deconvolution2d_nhwc_q8(
    // 要设置的反卷积运算符
    pytorch_qnnp_operator_t deconvolution,
    // 批量大小
    size_t batch_size,
    // 输入高度
    size_t input_height,
    // 输入宽度
    size_t input_width,
    // 输入数据的指针
    const uint8_t* input,
    // 输入数据的跨度
    size_t input_stride,
    // 输出数据的指针
    uint8_t* output,
    // 输出数据的跨度
    size_t output_stride,
    // 线程池用于并行计算
    pthreadpool_t threadpool);

// 创建一个 QNNP 的全连接（FC）运算符，基于 NC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_nc_q8(
    // 输入通道数
    size_t input_channels,
    // 输出通道数
    size_t output_channels,
    // 输入的零点
    uint8_t input_zero_point,
    // 卷积核的零点数组
    const uint8_t* kernel_zero_points,
    // 卷积核数据
    const uint8_t* kernel,
    // 偏置数组
    const int32_t* bias,
    // 输出的零点
    uint8_t output_zero_point,
    // 输出的最小值
    uint8_t output_min,
    // 输出的最大值
    uint8_t output_max,
    // 标志位
    uint32_t flags,
    // 重新量化的比例数组
    const float* requantization_scales,
    // 输出的全连接运算符
    pytorch_qnnp_operator_t* fully_connected);

// 创建一个 QNNP 的稀疏全连接（FC）运算符，基于 NC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_create_fully_connected_sparse_dq_nc_q8(
    // 输入通道数
    size_t input_channels,
    // 输出通道数
    size_t output_channels,
    // 输入的零点
    uint8_t input_zero_point,
    // 卷积核的零点数组
    const uint8_t* kernel_zero_points,
    // 卷积核列索引的指针
    const void* kernel_col_indices,
    // 卷积核行值的指针
    const void* kernel_row_values,
    // 卷积核值的指针
    const uint8_t* kernel_values,
    // 卷积核行块大小
    const uint32_t kernel_row_block_size,
    // 卷积核列块大小
    const uint32_t kernel_col_block_size,
    // 卷积核索引数据类型
    enum pytorch_qnnp_sparse_matrix_indices_dtype kernel_indices_dtype,
    // 输出的零点
    uint8_t output_zero_point,
    // 输出的最小值
    uint8_t output_min,
    // 输出的最大值
    uint8_t output_max,
    // 标志位
    uint32_t flags,
    // 重新量化的比例数组
    const float* requantization_scales,
    // 是否使用预打包卷积核的标志
    bool use_prepack_kernel,
    // 输出的稀疏全连接运算符
    pytorch_qnnp_operator_t* fully_connected);

// 设置一个 QNNP 的全连接（FC）运算符，基于 NC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_nc_q8(
    // 要设置的全连接运算符
    pytorch_qnnp_operator_t fully_connected,
    // 批量大小
    size_t batch_size,
    // 输入数据的指针
    const uint8_t* input,
    // 输入数据的跨度
    size_t input_stride,
    // 输出数据的指针
    uint8_t* output,
    // 输出数据的跨度
    size_t output_stride);

// 设置一个 QNNP 的稀疏全连接（FC）运算符，基于 NC 布局和量化 8 位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_fully_connected_sparse_dq_nc_q8(
    // 要设置的稀疏全连接运算符
    pytorch_qnnp_operator_t fully_connected,
    // 批量大小
    size_t batch_size,
    // 输入数据的指针
    const uint8_t* input,
    // 输入数据的跨度
    size_t input_stride,
    // 偏置数组的指针
    const float* bias,
    // 输出数据的指针
    float* output,
    // 输出数据的跨度
    size_t output_stride);
    size_t output_stride);



# 定义函数声明，指定函数名称为 output_stride，返回类型为 size_t（无符号整数）
// 创建一个全局平均池化操作符，适用于NWC格式的量化8位数据
enum pytorch_qnnp_status pytorch_qnnp_create_global_average_pooling_nwc_q8(
    size_t channels,                    // 输入张量的通道数
    uint8_t input_zero_point,           // 输入张量的零点
    float input_scale,                  // 输入张量的缩放因子
    uint8_t output_zero_point,          // 输出张量的零点
    float output_scale,                 // 输出张量的缩放因子
    uint8_t output_min,                 // 输出张量的最小值
    uint8_t output_max,                 // 输出张量的最大值
    uint32_t flags,                     // 操作标志位
    pytorch_qnnp_operator_t* global_average_pooling); // 指向全局平均池化操作符的指针

// 配置全局平均池化操作符，适用于NWC格式的量化8位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_global_average_pooling_nwc_q8(
    pytorch_qnnp_operator_t global_average_pooling, // 全局平均池化操作符
    size_t batch_size,                  // 批次大小
    size_t width,                       // 输入张量的宽度
    const uint8_t* input,               // 指向输入张量数据的指针
    size_t input_stride,                // 输入张量的行跨度
    uint8_t* output,                    // 指向输出张量数据的指针
    size_t output_stride);              // 输出张量的行跨度

// 创建一个二维平均池化操作符，适用于NHWC格式的量化8位数据
enum pytorch_qnnp_status pytorch_qnnp_create_average_pooling2d_nhwc_q8(
    uint32_t input_padding_height,      // 输入张量的垂直填充
    uint32_t input_padding_width,       // 输入张量的水平填充
    uint32_t pooling_height,            // 池化窗口的垂直大小
    uint32_t pooling_width,             // 池化窗口的水平大小
    uint32_t stride_height,             // 垂直方向的步幅
    uint32_t stride_width,              // 水平方向的步幅
    size_t channels,                    // 输入张量的通道数
    uint8_t input_zero_point,           // 输入张量的零点
    float input_scale,                  // 输入张量的缩放因子
    uint8_t output_zero_point,          // 输出张量的零点
    float output_scale,                 // 输出张量的缩放因子
    uint8_t output_min,                 // 输出张量的最小值
    uint8_t output_max,                 // 输出张量的最大值
    uint32_t flags,                     // 操作标志位
    pytorch_qnnp_operator_t* average_pooling); // 指向二维平均池化操作符的指针

// 配置二维平均池化操作符，适用于NHWC格式的量化8位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_average_pooling2d_nhwc_q8(
    pytorch_qnnp_operator_t average_pooling, // 二维平均池化操作符
    size_t batch_size,                  // 批次大小
    size_t input_height,                // 输入张量的高度
    size_t input_width,                 // 输入张量的宽度
    const uint8_t* input,               // 指向输入张量数据的指针
    size_t input_stride,                // 输入张量的行跨度
    uint8_t* output,                    // 指向输出张量数据的指针
    size_t output_stride,               // 输出张量的行跨度
    pthreadpool_t threadpool);          // 线程池对象

// 创建一个二维最大池化操作符，适用于NHWC格式的无符号8位数据
enum pytorch_qnnp_status pytorch_qnnp_create_max_pooling2d_nhwc_u8(
    uint32_t input_padding_height,      // 输入张量的垂直填充
    uint32_t input_padding_width,       // 输入张量的水平填充
    uint32_t pooling_height,            // 池化窗口的垂直大小
    uint32_t pooling_width,             // 池化窗口的水平大小
    uint32_t stride_height,             // 垂直方向的步幅
    uint32_t stride_width,              // 水平方向的步幅
    uint32_t dilation_height,           // 垂直方向的膨胀
    uint32_t dilation_width,            // 水平方向的膨胀
    size_t channels,                    // 输入张量的通道数
    uint8_t output_min,                 // 输出张量的最小值
    uint8_t output_max,                 // 输出张量的最大值
    uint32_t flags,                     // 操作标志位
    pytorch_qnnp_operator_t* max_pooling); // 指向二维最大池化操作符的指针

// 配置二维最大池化操作符，适用于NHWC格式的无符号8位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_max_pooling2d_nhwc_u8(
    pytorch_qnnp_operator_t max_pooling, // 二维最大池化操作符
    size_t batch_size,                  // 批次大小
    size_t input_height,                // 输入张量的高度
    size_t input_width,                 // 输入张量的宽度
    const uint8_t* input,               // 指向输入张量数据的指针
    size_t input_stride,                // 输入张量的行跨度
    uint8_t* output,                    // 指向输出张量数据的指针
    size_t output_stride,               // 输出张量的行跨度
    pthreadpool_t threadpool);          // 线程池对象

// 创建一个通道重排操作符，适用于NC_x8格式
enum pytorch_qnnp_status pytorch_qnnp_create_channel_shuffle_nc_x8(
    size_t groups,                      // 分组数
    size_t group_channels,              // 每组的通道数
    uint32_t flags,                     // 操作标志位
    pytorch_qnnp_operator_t* channel_shuffle); // 指向通道重排操作符的指针

// 配置通道重排操作符，适用于NC_x8格式
enum pytorch_qnnp_status pytorch_qnnp_setup_channel_shuffle_nc_x8(
    pytorch_qnnp_operator_t channel_shuffle, // 通道重排操作符
    size_t batch_size,                  // 批次大小
    const uint8_t* input,               // 指向输入数据的指针
    size_t input_stride,                // 输入数据的跨度
    uint8_t* output,                    // 指向输出数据的指针
    size_t output_stride);              // 输出数据的跨度

// 创建一个通道加法操作符，适用于NC格式的量化8位数据
enum pytorch_qnnp_status pytorch_qnnp_create_add_nc_q8(
    size_t channels,                    // 输入张量的通道数
    uint8_t a_zero_point,               // 输入张量A的零点
    float a_scale,                      // 输入张量A的缩放因子
    uint8_t b_zero_point,               // 输入张量B的零点
    float b_scale,                      // 输入张量B的缩放因子
    uint8_t sum_zero_point,             // 输出张量的零点
    float sum_scale,                    // 输出张量的缩放因子
    uint8_t sum_min,                    // 输出张量的最小值
    uint8_t sum_max,                    // 输出张量的最大值
    uint32_t flags,                     // 操作标志位
    pytorch_qnnp_operator_t* add);      // 指向通道加法操作符的指针

// 配置通道加法操作符，适用于NC格式的量化8位数据
enum pytorch_qnnp_status pytorch_qnnp_setup_add_nc_q8(
    pytorch_qnnp_operator_t add
    # 定义函数参数：批处理大小，指向数组 a 的指针，a 数组的步长，
    # 指向数组 b 的指针，b 数组的步长，存储结果的指针，结果数组的步长
    size_t batch_size,
    const uint8_t* a,
    size_t a_stride,
    const uint8_t* b,
    size_t b_stride,
    uint8_t* sum,
    size_t sum_stride);
// 创建一个 Clamp 操作符，用于限制输入在指定范围内的值，并将结果保存在 clamp 指针中
enum pytorch_qnnp_status pytorch_qnnp_create_clamp_nc_u8(
    size_t channels,                // 通道数
    uint8_t output_min,             // 输出的最小值
    uint8_t output_max,             // 输出的最大值
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* clamp  // 输出参数，指向创建的 Clamp 操作符的指针
);

// 配置一个已存在的 Clamp 操作符，以便使用在给定的输入和输出上执行操作
enum pytorch_qnnp_status pytorch_qnnp_setup_clamp_nc_u8(
    pytorch_qnnp_operator_t clamp,  // 要配置的 Clamp 操作符
    size_t batch_size,              // 批量大小
    const uint8_t* input,           // 输入数据的指针
    size_t input_stride,            // 输入步长（每个元素之间的距离）
    uint8_t* output,                // 输出数据的指针
    size_t output_stride            // 输出步长（每个元素之间的距离）
);

// 创建一个 Sigmoid 操作符，用于在量化输入上执行 Sigmoid 函数，并将结果保存在 sigmoid 指针中
enum pytorch_qnnp_status pytorch_qnnp_create_sigmoid_nc_q8(
    size_t channels,                // 通道数
    uint8_t input_zero_point,       // 输入的零点
    float input_scale,              // 输入的缩放因子
    uint8_t output_zero_point,      // 输出的零点
    float output_scale,             // 输出的缩放因子
    uint8_t output_min,             // 输出的最小值
    uint8_t output_max,             // 输出的最大值
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* sigmoid // 输出参数，指向创建的 Sigmoid 操作符的指针
);

// 配置一个已存在的 Sigmoid 操作符，以便使用在给定的输入和输出上执行操作
enum pytorch_qnnp_status pytorch_qnnp_setup_sigmoid_nc_q8(
    pytorch_qnnp_operator_t sigmoid,    // 要配置的 Sigmoid 操作符
    size_t batch_size,                  // 批量大小
    const uint8_t* input,               // 输入数据的指针
    size_t input_stride,                // 输入步长（每个元素之间的距离）
    uint8_t* output,                    // 输出数据的指针
    size_t output_stride                // 输出步长（每个元素之间的距离）
);

// 创建一个 Leaky ReLU 操作符，用于在量化输入上执行 Leaky ReLU 函数，并将结果保存在 leaky_relu 指针中
enum pytorch_qnnp_status pytorch_qnnp_create_leaky_relu_nc_q8(
    size_t channels,                // 通道数
    float negative_slope,           // 负斜率
    uint8_t input_zero_point,       // 输入的零点
    float input_scale,              // 输入的缩放因子
    uint8_t output_zero_point,      // 输出的零点
    float output_scale,             // 输出的缩放因子
    uint8_t output_min,             // 输出的最小值
    uint8_t output_max,             // 输出的最大值
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* leaky_relu  // 输出参数，指向创建的 Leaky ReLU 操作符的指针
);

// 配置一个已存在的 Leaky ReLU 操作符，以便使用在给定的输入和输出上执行操作
enum pytorch_qnnp_status pytorch_qnnp_setup_leaky_relu_nc_q8(
    pytorch_qnnp_operator_t leaky_relu, // 要配置的 Leaky ReLU 操作符
    size_t batch_size,                  // 批量大小
    const uint8_t* input,               // 输入数据的指针
    size_t input_stride,                // 输入步长（每个元素之间的距离）
    uint8_t* output,                    // 输出数据的指针
    size_t output_stride                // 输出步长（每个元素之间的距离）
);

// 创建一个 SoftArgMax 操作符，用于在量化输入上执行 SoftArgMax 函数，并将结果保存在 softargmax 指针中
enum pytorch_qnnp_status pytorch_qnnp_create_softargmax_nc_q8(
    size_t channels,                // 通道数
    float input_scale,              // 输入的缩放因子
    uint8_t output_zero_point,      // 输出的零点
    float output_scale,             // 输出的缩放因子
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* softargmax  // 输出参数，指向创建的 SoftArgMax 操作符的指针
);

// 配置一个已存在的 SoftArgMax 操作符，以便使用在给定的输入和输出上执行操作
enum pytorch_qnnp_status pytorch_qnnp_setup_softargmax_nc_q8(
    pytorch_qnnp_operator_t softargmax, // 要配置的 SoftArgMax 操作符
    size_t batch_size,                  // 批量大小
    const uint8_t* input,               // 输入数据的指针
    size_t input_stride,                // 输入步长（每个元素之间的距离）
    uint8_t* output,                    // 输出数据的指针
    size_t output_stride                // 输出步长（每个元素之间的距离）
);

// 创建一个 Tanh 操作符，用于在量化输入上执行 Tanh 函数，并将结果保存在 tanh 指针中
enum pytorch_qnnp_status pytorch_qnnp_create_tanh_nc_q8(
    size_t channels,                // 通道数
    uint8_t input_zero_point,       // 输入的零点
    float input_scale,              // 输入的缩放因子
    uint8_t output_zero_point,      // 输出的零点
    float output_scale,             // 输出的缩放因子
    uint8_t output_min,             // 输出的最小值
    uint8_t output_max,             // 输出的最大值
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* tanh   // 输出参数，指向创建的 Tanh 操作符的指针
);

// 配置一个已存在的 Tanh 操作符，以便使用在给定的输入和输出上执行操作
enum pytorch_qnnp_status pytorch_qnnp_setup_tanh_nc_q8(
    pytorch_qnnp_operator_t tanh,    // 要配置的 Tanh 操作符
    size_t batch_size,               // 批量大小
    const uint8_t* input,            // 输入数据的指针
    size_t input_stride,             // 输入步长（每个元素之间的距离）
    uint8_t* output,                 // 输出数据的指针
    size_t output_stride             // 输出步长（每个元素之间的距离）
);

// 创建一个 HardSigmoid 操作符，用于在量化输入上执行 HardSigmoid 函数，并将结果保存在 hardsigmoid 指针中
enum pytorch_qnnp_status pytorch_qnnp_create_hardsigmoid_nc_q8(
    size_t channels,                // 通道数
    uint8_t input_zero_point,       // 输入的零点
    float input_scale,              // 输入的缩放因子
    uint8_t output_zero_point,      // 输出的零点
    float output_scale,             // 输出的缩放因子
    uint8_t output_min,             // 输出的最小值
    uint8_t output_max,             // 输出的最大值
    uint32_t flags,                 // 标志位
    pytorch_qnnp_operator_t* hardsigmoid  // 输出参数，指向创建的 HardSigmoid 操作符的指针
);

// 配置一个已存在的 HardSigmoid 操作符，以便使用在给定的输入和输出上执行操作
enum pytorch_qnnp_status pytorch_qnnp_setup_hardsigmoid_nc_q8(
    pytorch_qnnp_operator_t hardsigmoid,  // 要配置的 HardSigmoid 操作符
# 定义一个函数 pytorch_qnnp_create_hardswish_nc_q8，用于创建 QNNPACK 中的硬切线激活函数运算符
enum pytorch_qnnp_status pytorch_qnnp_create_hardswish_nc_q8(
    size_t channels,                   // 输入通道数
    uint8_t input_zero_point,          // 输入数据的零点
    float input_scale,                 // 输入数据的缩放比例
    uint8_t output_zero_point,         // 输出数据的零点
    float output_scale,                // 输出数据的缩放比例
    uint8_t output_min,                // 输出数据的最小值
    uint8_t output_max,                // 输出数据的最大值
    uint32_t flags,                    // 标志位
    pytorch_qnnp_operator_t* hardswish // 指向硬切线激活函数运算符的指针
);

# 定义一个函数 pytorch_qnnp_setup_hardswish_nc_q8，用于配置 QNNPACK 中的硬切线激活函数运算符
enum pytorch_qnnp_status pytorch_qnnp_setup_hardswish_nc_q8(
    pytorch_qnnp_operator_t hardswish, // 硬切线激活函数运算符
    size_t batch_size,                 // 批处理大小
    const uint8_t* input,              // 输入数据指针
    size_t input_stride,               // 输入数据步长
    uint8_t* output,                   // 输出数据指针
    size_t output_stride               // 输出数据步长
);

# 定义一个函数 pytorch_qnnp_run_operator，用于在线程池中执行给定运算符
enum pytorch_qnnp_status pytorch_qnnp_run_operator(
    pytorch_qnnp_operator_t op,        // 待执行的运算符
    pthreadpool_t threadpool           // 线程池对象
);

# 定义一个函数 pytorch_qnnp_delete_operator，用于删除 QNNPACK 中的运算符
enum pytorch_qnnp_status pytorch_qnnp_delete_operator(
    pytorch_qnnp_operator_t op         // 待删除的运算符
);

#ifdef __cplusplus
} /* extern "C" */
#endif
```