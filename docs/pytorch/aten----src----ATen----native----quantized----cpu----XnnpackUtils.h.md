# `.\pytorch\aten\src\ATen\native\quantized\cpu\XnnpackUtils.h`

```
#pragma once

#ifdef USE_XNNPACK
#include <cstdint>

#include <ATen/core/Tensor.h>
#include <ATen/native/xnnpack/Common.h>

using xnnpack_operator = at::native::xnnpack::Operator;

namespace at {
namespace native {
namespace xnnp_utils {

/*
 * Return shape in the same order as the memory format
 * e.g. channels_last will return NHWC instead of NCHW
 */
// 根据内存格式返回形状，如 channels_last 返回 NHWC 而非 NCHW
std::vector<size_t> get_mem_format_aware_shape(const at::Tensor& in);

/*
 * Input is always int8_t, output can be [int8_t, uint8_t].
 * input  + offset = output
 * int8_t + 128    = uint8_t
 * int8_t + 0      = int8_t
 */
// 输入始终为 int8_t，输出可以是 [int8_t, uint8_t]。
// 输入 + 偏移量 = 输出
// int8_t + 128 = uint8_t
// int8_t + 0 = int8_t
template <typename PT>
void q8_copy_int8_weight_and_add_offset(const at::Tensor& in, at::Tensor& out);

template <int kSpatialDim>
Tensor convert_conv_weights_to_channel_last_tensor(
    const at::Tensor& src,
    int groups,
    bool transpose);

/*
 * Series of create wrapper functions to call xnn_create_[de]conv* functions.
 */
// 一系列的创建包装函数，用于调用 xnn_create_[de]conv* 函数。
C10_ALWAYS_INLINE
enum xnn_status xnnp_create_convolution2d_nhwc(
    uint32_t pad_top,
    uint32_t pad_right,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t kernel_h,
    uint32_t kernel_w,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t ip_chan_stride,
    size_t op_chan_stride,
    int8_t izp,
    float ip_scale,
    int8_t kzp,
    const float* k_scales,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t ozp,
    float op_scale,
    int8_t op_min,
    int8_t op_max,
    uint32_t flags,
    xnn_operator_t* op,
    bool per_channel,
    bool transpose) {
  /* Symmetric quantization forces kzp = 0 */
  // 对称量化要求 kzp = 0
  TORCH_CHECK(!kzp, "XNNPACK Q[SC]8 conv kernels expects kernel zero point to be zero."
                    "But got: ", kzp);

  if (transpose) {
    // 如果是转置操作
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");
    return xnn_create_deconvolution2d_nhwc_qs8(
        pad_top,        /* uint32_t output_padding_top          */  // 设置输出在顶部的填充像素数量
        pad_right,      /* uint32_t output_padding_right        */  // 设置输出在右侧的填充像素数量
        pad_bottom,     /* uint32_t output_padding_bottom       */  // 设置输出在底部的填充像素数量
        pad_left,       /* uint32_t output_padding_left         */  // 设置输出在左侧的填充像素数量
        kernel_h,       /* uint32_t kernel_height               */  // 设置卷积核的高度
        kernel_w,       /* uint32_t kernel_width                */  // 设置卷积核的宽度
        stride_h,       /* uint32_t stride_height               */  // 设置垂直方向的步幅
        stride_w,       /* uint32_t stride_width                */  // 设置水平方向的步幅
        dilation_h,     /* uint32_t dilation_height             */  // 设置垂直方向的膨胀率
        dilation_w,     /* uint32_t dilation_width              */  // 设置水平方向的膨胀率
        groups,         /* uint32_t groups                      */  // 设置分组数量
        group_input_channels,  /* size_t group_input_channels   */  // 设置每个分组的输入通道数
        group_output_channels, /* size_t group_output_channels  */  // 设置每个分组的输出通道数
        ip_chan_stride, /* size_t input_pixel_stride            */  // 设置输入通道之间的像素步幅
        op_chan_stride, /* size_t output_pixel_stride           */  // 设置输出通道之间的像素步幅
        izp,            /* int8_t input_zero_point              */  // 设置输入的零点偏移量
        ip_scale,       /* float input_scale                    */  // 设置输入的比例因子
        k_scales[0],    /* float kernel_scale                   */  // 设置卷积核的比例因子
        kernel,         /* const int8_t* kernel                 */  // 设置卷积核数据指针
        bias,           /* const int32_t* bias                  */  // 设置偏置数据指针
        ozp,            /* int8_t output_zero_point             */  // 设置输出的零点偏移量
        op_scale,       /* float output_scale                   */  // 设置输出的比例因子
        op_min,         /* int8_t output_min                    */  // 设置输出的最小值
        op_max,         /* int8_t output_max                    */  // 设置输出的最大值
        flags,          /* uint32_t flags                       */  // 设置特定标志位
        nullptr,        /* xnn_caches_t caches                  */  // 缓存相关信息，此处为 nullptr
        nullptr,        /* xnn_weights_cache_t weights_cache    */  // 权重缓存信息，此处为 nullptr
        op);            /* xnn_operator_t* deconvolution_op_out */  // 输出的反卷积操作对象指针
  }

  if (!per_channel) {
    // 调用 xnn_create_convolution2d_nhwc_qs8 函数，创建 NHWC 格式的量化 int8 卷积操作符
    return xnn_create_convolution2d_nhwc_qs8(
        pad_top,        /* uint32_t input_padding_top         */  // 输入顶部填充大小
        pad_right,      /* uint32_t input_padding_right       */  // 输入右侧填充大小
        pad_bottom,     /* uint32_t input_padding_bottom      */  // 输入底部填充大小
        pad_left,       /* uint32_t input_padding_left        */  // 输入左侧填充大小
        kernel_h,       /* uint32_t kernel_height             */  // 卷积核高度
        kernel_w,       /* uint32_t kernel_width              */  // 卷积核宽度
        stride_h,       /* uint32_t subsampling_height        */  // 下采样高度
        stride_w,       /* uint32_t subsampling_width         */  // 下采样宽度
        dilation_h,     /* uint32_t dilation_height           */  // 膨胀高度
        dilation_w,     /* uint32_t dilation_width            */  // 膨胀宽度
        groups,         /* uint32_t groups                    */  // 分组数
        group_input_channels,  /* size_t group_input_channels */  // 每组的输入通道数
        group_output_channels, /* size_t group_output_channels*/  // 每组的输出通道数
        ip_chan_stride, /* size_t input_channel_stride        */  // 输入通道步长
        op_chan_stride, /* size_t output_channel_stride       */  // 输出通道步长
        izp,            /* int8_t input_zero_point            */  // 输入零点
        ip_scale,       /* float input_scale                  */  // 输入缩放因子
        k_scales[0],    /* float kernel_scale                 */  // 卷积核缩放因子
        kernel,         /* const int8_t* kernel               */  // 卷积核数据
        bias,           /* const int32_t* bias                */  // 偏置数据
        ozp,            /* int8_t output_zero_point           */  // 输出零点
        op_scale,       /* float output_scale                 */  // 输出缩放因子
        op_min,         /* int8_t output_min                  */  // 输出最小值
        op_max,         /* int8_t output_max                  */  // 输出最大值
        flags,          /* uint32_t flags                     */  // 标志位
        nullptr,        /* xnn_caches_t caches                */  // 缓存
        nullptr,        /* xnn_weights_cache_t weights_cache */  // 权重缓存
        op);            /* xnn_operator_t* convolution_op_out */  // 输出卷积操作符指针
  } else { /* per_channel */
    return xnn_create_convolution2d_nhwc_qs8_qc8w(
        pad_top,        /* uint32_t input_padding_top         */  // 输入顶部填充大小
        pad_right,      /* uint32_t input_padding_right       */  // 输入右侧填充大小
        pad_bottom,     /* uint32_t input_padding_bottom      */  // 输入底部填充大小
        pad_left,       /* uint32_t input_padding_left        */  // 输入左侧填充大小
        kernel_h,       /* uint32_t kernel_height             */  // 卷积核高度
        kernel_w,       /* uint32_t kernel_width              */  // 卷积核宽度
        stride_h,       /* uint32_t subsampling_height        */  // 下采样高度（步幅）
        stride_w,       /* uint32_t subsampling_width         */  // 下采样宽度（步幅）
        dilation_h,     /* uint32_t dilation_height           */  // 膨胀高度
        dilation_w,     /* uint32_t dilation_width            */  // 膨胀宽度
        groups,         /* uint32_t groups                    */  // 分组数
        group_input_channels,  /* size_t group_input_channels */  // 每组的输入通道数
        group_output_channels, /* size_t group_output_channels*/  // 每组的输出通道数
        ip_chan_stride, /* size_t input_channel_stride        */  // 输入通道步长
        op_chan_stride, /* size_t output_channel_stride       */  // 输出通道步长
        izp,            /* int8_t input_zero_point            */  // 输入零点
        ip_scale,       /* float input_scale                  */  // 输入缩放因子
        k_scales,       /* const float* kernel_scale          */  // 卷积核缩放因子数组
        kernel,         /* const int8_t* kernel               */  // 卷积核数据
        bias,           /* const int32_t* bias                */  // 偏置数据
        ozp,            /* int8_t output_zero_point           */  // 输出零点
        op_scale,       /* float output_scale                 */  // 输出缩放因子
        op_min,         /* int8_t output_min                  */  // 输出最小值
        op_max,         /* int8_t output_max                  */  // 输出最大值
        flags,          /* uint32_t flags                     */  // 标志位
        nullptr,        /* xnn_caches_t caches                */  // 缓存数据结构
        nullptr,        /* xnn_weights_cache_t weights_cache  */  // 权重缓存数据结构
        op);            /* xnn_operator_t* convolution_op_out */  // 卷积操作器输出指针
  }
}

/*
 * Series of reshape wrapper functions to call xnn_reshape_[de]conv* functions.
 */
C10_ALWAYS_INLINE
// 定义了一个函数，用于在 NHWC 格式下重塑二维卷积运算符的形状
enum xnn_status xnnp_reshape_convolution2d_nhwc(
    xnn_operator_t op,      /* xnn_operator_t convolution_op */
    size_t batch,           /* size_t batch_size */
    size_t in_h,            /* size_t input_height */
    size_t in_w,            /* size_t input_width */
    pthreadpool_t pt_pool,  /* pthreadpool_t threadpool */
    bool per_channel = false, /* 是否按通道处理 */
    bool transpose = false, /* 是否转置操作 */
    uint32_t adj_h = 0,     /* uint32_t adjustment_height */
    uint32_t adj_w = 0) {   /* uint32_t adjustment_width */
  if(transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");
    // 如果进行转置操作，调用反卷积函数重塑形状并返回结果
    return xnn_reshape_deconvolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        batch,    /* size_t batch_size */
        in_h,     /* size_t input_height */
        in_w,     /* size_t input_width */
        adj_h,    /* uint32_t adjustment_height */
        adj_w,    /* uint32_t adjustment_width */
        nullptr,  /* size_t* output_height_out */
        nullptr,  /* size_t* output_width_out */
        pt_pool); /* pthreadpool_t threadpool */
  }

  // 初始化工作空间大小和对齐方式
  size_t workspace_size = SIZE_MAX;
  size_t workspace_alignment = SIZE_MAX;

  if (!per_channel) {
    // 如果不按通道处理，调用卷积函数重塑形状并返回结果
    return xnn_reshape_convolution2d_nhwc_qs8(
        op,                     /* xnn_operator_t convolution_op */
        batch,                  /* size_t batch_size */
        in_h,                   /* size_t input_height */
        in_w,                   /* size_t input_width */
        &workspace_size,        /* size_t* workspace_size */
        &workspace_alignment,   /* size_t* workspace_alignment */
        nullptr,                /* size_t* output_height_out */
        nullptr,                /* size_t* output_width_out */
        pt_pool);               /* pthreadpool_t threadpool */
  } else { // per_channel
    // 如果按通道处理，调用按量化标度因子重塑卷积函数形状并返回结果
    return xnn_reshape_convolution2d_nhwc_qs8_qc8w(
        op,                     /* xnn_operator_t convolution_op */
        batch,                  /* size_t batch_size */
        in_h,                   /* size_t input_height */
        in_w,                   /* size_t input_width */
        &workspace_size,        /* size_t* workspace_size */
        &workspace_alignment,   /* size_t* workspace_alignment */
        nullptr,                /* size_t* output_height_out */
        nullptr,                /* size_t* output_width_out */
        pt_pool);               /* pthreadpool_t threadpool */
  }
}


/*
 * Series of setup wrapper functions to call xnn_setup_[de]conv* functions.
 */
C10_ALWAYS_INLINE
// 定义了一个函数，用于在 NHWC 格式下设置二维卷积运算符的输入和输出
enum xnn_status xnnp_setup_convolution2d_nhwc(
    xnn_operator_t op,      /* xnn_operator_t convolution_op */
    const int8_t* inp,      /* const int8_t* input */
    int8_t* outp,           /* int8_t* output */
    bool per_channel = false, /* 是否按通道处理 */
    bool transpose = false) { /* 是否转置操作 */
  if(transpose) {
    TORCH_CHECK(!per_channel, "XNNPACK Q[SC]8 does not have a per channel deconvolution!");

    // 如果进行转置操作，调用设置反卷积函数并返回结果
    return xnn_setup_deconvolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        inp,      /* const int8_t* input */
        outp);    /* int8_t* output */
  }

  if (!per_channel) {
    // 如果不按通道处理，调用设置卷积函数并返回结果
    return xnn_setup_convolution2d_nhwc_qs8(
        op,       /* xnn_operator_t convolution_op */
        inp,      /* const int8_t* input */
        outp);    /* int8_t* output */
  } else {
    // 如果按通道处理，暂时没有相关函数调用，这里应当补充相关设置代码
  }
}
    # 如果不是按通道量化的情况下，使用 xnn_setup_convolution2d_nhwc_qs8 函数设置卷积操作
    return xnn_setup_convolution2d_nhwc_qs8(
        op,       /* xnn_operator_t deconvolution_op */
        nullptr,  /* void workspace                  */
        inp,      /* const int8_t* input             */
        outp);    /* int8_t* output                  */
  } else { /* per_channel */
    # 如果是按通道量化的情况下，使用 xnn_setup_convolution2d_nhwc_qs8_qc8w 函数设置卷积操作
    return xnn_setup_convolution2d_nhwc_qs8_qc8w(
        op,       /* xnn_operator_t deconvolution_op */
        nullptr,  /* void workspace                  */
        inp,      /* const int8_t* input             */
        outp);    /* int8_t* output                  */
  }
/*
 * Series of wrapper functions to call xnn_create* and xnn_setup*
 * functions for linear
 */
namespace xnnp_utils {
namespace native {

/*
 * Creates a fully connected operation using quantized symmetric 8-bit integer arithmetic.
 * This function initializes an operator for fully connected layer with quantized symmetric 8-bit weights.
 * It requires input and output scales, biases, and other parameters.
 * The kernel zero point must be zero for symmetric quantization.
 *
 * Parameters:
 * - input_channels: Number of input channels
 * - output_channels: Number of output channels
 * - input_stride: Stride of input tensor
 * - output_stride: Stride of output tensor
 * - input_zero_point: Zero point of input tensor
 * - input_scale: Scale of input tensor
 * - kernel_zero_point: Zero point of kernel tensor (must be 0 for symmetric quantization)
 * - kernel_scale: Scale of kernel tensor
 * - kernel: Pointer to the kernel tensor
 * - bias: Pointer to the bias tensor
 * - output_zero_point: Zero point of output tensor
 * - output_scale: Scale of output tensor
 * - output_min: Minimum output value
 * - output_max: Maximum output value
 * - flags: Additional flags for configuration
 * - fully_connected_op_out: Pointer to store the created fully connected operator
 *
 * Returns:
 * - Status code indicating success or failure
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_create_fully_connected_nc(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    int8_t input_zero_point,
    float input_scale,
    int8_t kernel_zero_point,
    float kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* fully_connected_op_out) {
  /* Symmetric quantization forces kzp = 0 */
  TORCH_CHECK(!kernel_zero_point, "XNNPACK QS8 linear kernel expects kernel zero point to be zero."
                    "But got: ", kernel_zero_point);
  return xnn_create_fully_connected_nc_qs8(
      input_channels,          /* size_t input_channels                  */
      output_channels,         /* size_t output_channels                 */
      input_stride,            /* size_t input_stride                    */
      output_stride,           /* size_t output_stride                   */
      input_zero_point,        /* int8_t input_zero_point                */
      input_scale,             /* float input_scale                      */
      kernel_scale,            /* float kernel_scale                     */
      kernel,                  /* const int8_t* kernel                   */
      bias,                    /* const int32_t* bias                    */
      output_zero_point,       /* int8_t output_zero_point               */
      output_scale,            /* float output_scale                     */
      output_min,              /* int8_t output_min                      */
      output_max,              /* int8_t output_max                      */
      flags,                   /* uint32_t flags                         */
      nullptr,                 /* xnn_caches_t caches                    */
      nullptr,                 /* xnn_weights_cache_t                    */
      fully_connected_op_out); /* xnn_operator_t* fully_connected_op_out */
}

/*
 * Reshapes a fully connected operator for a new batch size.
 *
 * Parameters:
 * - fully_connected_op: Fully connected operator to reshape
 * - batch_size: New batch size
 * - threadpool: Thread pool to execute reshaping operation
 *
 * Returns:
 * - Status code indicating success or failure
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_reshape_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    pthreadpool_t threadpool) {
  return xnn_reshape_fully_connected_nc_qs8(
      fully_connected_op, /* xnn_operator_t fully_connected_op */
      batch_size,         /* size_t batch_size                 */
      threadpool);        /* pthreadpool_t threadpool          */
}

/*
 * Sets up a fully connected operator for inference.
 * This function sets up a fully connected operator with quantized symmetric 8-bit weights
 * for inference, specifying input and output data pointers.
 *
 * Parameters:
 * - fully_connected_op: Fully connected operator to set up
 * - input: Pointer to input tensor data
 * - output: Pointer to output tensor data
 *
 * Returns:
 * - Status code indicating success or failure
 */
C10_ALWAYS_INLINE
enum xnn_status xnnp_setup_fully_connected_nc(
    xnn_operator_t fully_connected_op,
    const int8_t* input,
    int8_t* output) {
  return xnn_setup_fully_connected_nc_qs8(
      fully_connected_op, /* xnn_operator_t fully_connected_op */
      input,              /* const int8_t* input               */
      output              /* int8_t* output                    */
    );
}

} // namespace native
} // namespace xnnp_utils
} // 结束 at 命名空间

#endif // 如果定义了 USE_XNNPACK，则结束条件编译
```