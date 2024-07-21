# `.\pytorch\aten\src\ATen\native\quantized\cpu\QuantizedOps.h`

```py
#pragma once
// 在 C++ 中，#pragma once 指令确保头文件只被编译一次

#include <ATen/core/Tensor.h>
#include <ATen/core/IListRef.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

// 定义函数指针类型 qrelu_fn，用于指向整数 Relu 激活函数的函数指针
using qrelu_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);

// 定义函数指针类型 qrelu_leaky_fn，用于指向整数 Leaky Relu 激活函数的函数指针
using qrelu_leaky_fn = void (*)(Tensor& /*out*/, const Tensor& /*qx*/,
                                const Scalar& /*negval_*/);

// 定义函数指针类型 qgelu_fn，用于指向整数 GELU 激活函数的函数指针，带有近似值参数
using qgelu_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/, GeluType /* approximate */);

// 定义函数指针类型 qsigmoid_fn，用于指向整数 Sigmoid 激活函数的函数指针，带有输出缩放和零点参数
using qsigmoid_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/, double output_scale, int64_t output_zero_point);

// 定义函数指针类型 qhardsigmoid_fn，用于指向整数 Hard Sigmoid 激活函数的函数指针
using qhardsigmoid_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);

// 定义函数指针类型 qclamp_fn，用于指向整数 Clamp 操作的函数指针，限制输出在给定的最小值和最大值范围内
using qclamp_fn = void (*)(
    const at::Tensor& /*qx*/,
    const Scalar& min,
    const Scalar& max,
    at::Tensor& /*qy*/);

// 定义函数指针类型 qclamp_minmax_fn，用于指向整数 Clamp 操作的函数指针，限制输出在给定的最小值或最大值范围内
using qclamp_minmax_fn = void (*)(
    const at::Tensor& /*qx*/,
    const Scalar& /*min or max*/,
    at::Tensor& /*qy*/);

// 定义函数指针类型 qthreshold_fn，用于指向整数 Threshold 操作的函数指针，根据给定的阈值和值设置输出
using qthreshold_fn = void (*)(
    const at::Tensor& /*qx*/,
    const Scalar& threshold,
    const Scalar& value,
    at::Tensor& /*qy*/);

// 定义函数指针类型 qtanh_fn，用于指向整数 Tanh 激活函数的函数指针
using qtanh_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);

// 定义函数指针类型 qelu_fn，用于指向整数 ELU 激活函数的函数指针，带有 alpha、scale 和 input_scale 参数
using qelu_fn = void (*)(
    const at::Tensor& /*qx*/,
    const Scalar& /*alpha*/,
    const Scalar& /*scale*/,
    const Scalar& /*input_scale*/,
    at::Tensor& /*qy*/);

// 定义函数指针类型 qbinary_fn，用于指向整数二元操作（例如加法或乘法）的函数指针
using qbinary_fn =
    void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Tensor& /*other*/);

// 定义函数指针类型 qadd_scalar_fn，用于指向整数加法操作的函数指针，其中一个操作数是标量
using qadd_scalar_fn =
    void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Scalar& other /*other*/);

// 定义函数指针类型 qhardswish_fn，用于指向整数 Hard Swish 激活函数的函数指针
using qhardswish_fn = void (*)(const at::Tensor& /*qx*/, at::Tensor& /*qy*/);

// 定义函数指针类型 qdropout_fn，用于指向整数 Dropout 操作的函数指针，包括概率和训练标志
using qdropout_fn = void (*)(
    const at::Tensor& /*qx*/,
    const Scalar& /*p*/,
    bool training /*training*/,
    at::Tensor& /*qy*/);

// 定义函数指针类型 qmaxpool_2d_fn，用于指向整数 2D 最大池化操作的函数指针，包括输入输出尺寸、核大小、步幅、填充和扩展等参数
using qmaxpool_2d_fn = void (*)(
    const Tensor& qx,
    int64_t iC, // input/output channels
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sH,
    int64_t sW, // strides
    int64_t pH,
    int64_t pW, // padding
    int64_t dH,
    int64_t dW, // dilation
    Tensor& qy);

// 定义函数指针类型 qmaxpool_3d_fn，用于指向整数 3D 最大池化操作的函数指针，包括输入输出尺寸、核大小、步幅、填充和扩展等参数
using qmaxpool_3d_fn = void (*)(
    const Tensor& qx,
    int64_t iC, // input/output channels
    int64_t iT,
    int64_t iH,
    int64_t iW, // input sizes
    int64_t oT,
    int64_t oH,
    int64_t oW, // output sizes
    int64_t kT,
    int64_t kH,
    int64_t kW, // kernel size
    int64_t sT,
    int64_t sH,
    int64_t sW, // strides
    int64_t pT,
    int64_t pH,
    int64_t pW, // padding
    int64_t dT,
    int64_t dH,
    int64_t dW, // dilation
    Tensor& qy);

// 定义函数指针类型 qadaptive_avg_pool2d_fn，用于指向整数自适应平均池化操作的函数指针，包括输入输出尺寸和步幅等参数
using qadaptive_avg_pool2d_fn = void (*)(
    const Tensor& qx,
    Tensor& qy,
    int64_t sizeB,
    int64_t sizeC,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideC,
    int64_t istrideH,
    int64_t istrideW);
// 声明一个函数指针类型 qadaptive_avg_pool3d_fn，该函数用于执行3D自适应平均池化操作
using qadaptive_avg_pool3d_fn = void (*)(
    const Tensor& qx,                   // 输入张量
    Tensor& qy,                         // 输出张量
    int64_t sizeB,                      // 批量大小
    int64_t sizeC,                      // 输入通道数
    int64_t isizeD,                     // 输入深度
    int64_t isizeH,                     // 输入高度
    int64_t isizeW,                     // 输入宽度
    int64_t osizeD,                     // 输出深度
    int64_t osizeH,                     // 输出高度
    int64_t osizeW,                     // 输出宽度
    int64_t istrideB,                   // 批量步长
    int64_t istrideC,                   // 通道步长
    int64_t istrideD,                   // 深度步长
    int64_t istrideH,                   // 高度步长
    int64_t istrideW);                  // 宽度步长

// 声明一个函数指针类型 qavg_pool2d_fn，用于执行2D平均池化操作
using qavg_pool2d_fn = void (*)(
    const Tensor& qx,                   // 输入张量
    Tensor& qy,                         // 输出张量
    int64_t nBatch,                     // 批量大小
    int64_t nInputPlane,                // 输入通道数
    int64_t inputWidth,                 // 输入宽度
    int64_t inputHeight,                // 输入高度
    int64_t outputWidth,                // 输出宽度
    int64_t outputHeight,               // 输出高度
    int kW,                             // 内核宽度
    int kH,                             // 内核高度
    int dW,                             // 宽度步长
    int dH,                             // 高度步长
    int padW,                           // 宽度填充
    int padH,                           // 高度填充
    bool count_include_pad,             // 是否包含填充计数
    std::optional<int64_t> divisor_override);  // 可选的除数重写

// 声明一个函数指针类型 qavg_pool3d_fn，用于执行3D平均池化操作
using qavg_pool3d_fn = void (*)(
    const Tensor& qx,                   // 输入张量
    Tensor& qy,                         // 输出张量
    int64_t nBatch,                     // 批量大小
    int64_t nInputPlane,                // 输入通道数
    int64_t inputWidth,                 // 输入宽度
    int64_t inputHeight,                // 输入高度
    int64_t inputDepth,                 // 输入深度
    int64_t outputWidth,                // 输出宽度
    int64_t outputHeight,               // 输出高度
    int64_t outputDepth,                // 输出深度
    int kW,                             // 内核宽度
    int kH,                             // 内核高度
    int kD,                             // 内核深度
    int dW,                             // 宽度步长
    int dH,                             // 高度步长
    int dD,                             // 深度步长
    int padW,                           // 宽度填充
    int padH,                           // 高度填充
    int padD,                           // 深度填充
    bool count_include_pad,             // 是否包含填充计数
    std::optional<int64_t> divisor_override);  // 可选的除数重写

// 声明一个函数指针类型 qupsample_bilinear2d_fn，用于执行2D双线性上采样操作
using qupsample_bilinear2d_fn = void (*)(
    Tensor& output,                     // 输出张量
    const Tensor& input,                // 输入张量
    int64_t input_height,               // 输入高度
    int64_t input_width,                // 输入宽度
    int64_t output_height,              // 输出高度
    int64_t output_width,               // 输出宽度
    int64_t nbatch,                     // 批量大小
    int64_t channels,                   // 通道数
    bool align_corners,                 // 是否对齐角点
    std::optional<double> scales_h,     // 可选的高度缩放比例
    std::optional<double> scales_w);    // 可选的宽度缩放比例

// 声明一个函数指针类型 qcat_nhwc_fn，用于执行NHWC格式下的张量拼接操作
using qcat_nhwc_fn = Tensor (*)(
    const MaterializedITensorListRef& qxs,   // 输入张量列表
    int64_t dim,                            // 拼接维度
    double scale,                           // 缩放因子
    int64_t zero_point);                    // 零点

// 声明一个函数指针类型 qtopk_fn，用于执行Top-K操作
using qtopk_fn = void (*)(
    Tensor&,                                // 输出张量
    Tensor&,                                // 输出索引张量
    const Tensor&,                          // 输入张量
    int64_t,                                // K值
    int64_t,                                // dim维度
    bool,                                   // 是否对输入张量进行排序
    bool);                                  // 是否对K个最大的张量进行排序

// 声明一个函数指针类型 qbatch_norm_fn，用于执行批量归一化操作
using qbatch_norm_fn = void (*)(
    int64_t,                                // 批量大小
    int64_t,                                // 通道数
    int64_t,                                // 高度
    int64_t,                                // 宽度
    int64_t,                                // 深度
    const Tensor&,                          // 输入张量
    const Tensor&,                          // gamma张量
    const Tensor&,                          // beta张量
    Tensor&);                               // 输出张量

// 声明一个函数指针类型 qnormalize_fn，用于执行归一化操作
using qnormalize_fn = void (*)(
    const Tensor& /* X */,                 // 输入张量
    const Tensor& /* gamma */,             // gamma张量
    const Tensor& /* beta */,              // beta张量
    bool /* affine_per_channel */,         // 是否通道级别仿射变换
    int /* num_channels */,                // 通道数
    int /* num_groups */,                  // 分组数
    int64_t /* M */,                       // M值
    int64_t /* N */,                       // N值
    double /* eps */,                      // 用于数值稳定性的小值
    Tensor* /* Y */);                      // 输出张量指针

// 声明一个函数指针类型 qmean_inner_dim_fn，用于计算指定维度上的均值
using qmean_inner_dim_fn = void (*)(
    const Tensor& /* X */,                 // 输入张量
    OptionalIntArrayRef /* opt_dim */,     // 可选的维度参数
    bool /* keepdim */,                    // 是否保持维度
    std::optional<ScalarType> /* opt_dtype */,  // 可选的数据类型
    Tensor& /* Y */);                      // 输出张量

// 声明一个函数指针类型 qstd_inner_dim_fn，用于计算指定维度上的标准差
using qstd_inner_dim_fn = void (*)(
    const Tensor& /* X */,                 // 输入张量
    OptionalIntArrayRef /* dim */,         // 可选的维度参数
    const std::optional<Scalar>& /* correction */,  // 可选的修正值
    bool /* keepdim */,                    // 是否保持维度
    Tensor& /* Y */);                      // 输出张量

// 声明一个函数指针类型 qnormalize_nhwc_fn，用于执行NHWC格式下的归一化操作
using qnormalize_nhwc_fn = void (*)(
    const Tensor& /* X */,                 // 输入张量
    const Tensor& /* gamma */,             // gamma张量
    const Tensor& /* beta */,              // beta张量
    bool /* affine_per_channel */,         // 是否通道级别仿射变换
    int /* num_channels */,                // 通道数
    int /* num_groups */,                  // 分组数
    int64_t /* M */,                       // M值
    int64_t /* N */,                       // N值
    double /* eps */,                      // 用于数值稳定性的小值
    Tensor* /* Y */);                      // 输出张
# 声明并分发量化自适应平均池化函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qadaptive_avg_pool2d_fn, qadaptive_avg_pool2d_nhwc_stub);

# 声明并分发量化自适应平均池化函数的函数指针，采用 NDHWC 数据格式的存储
DECLARE_DISPATCH(qadaptive_avg_pool3d_fn, qadaptive_avg_pool3d_ndhwc_stub);

# 声明并分发量化加标量函数的函数指针，采用 ReLU 激活函数
DECLARE_DISPATCH(qadd_scalar_fn, qadd_scalar_relu_stub);

# 声明并分发量化加标量函数的函数指针，不使用激活函数
DECLARE_DISPATCH(qadd_scalar_fn, qadd_scalar_stub);

# 声明并分发量化平均池化函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qavg_pool2d_fn, qavg_pool2d_nhwc_stub);

# 声明并分发量化平均池化函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qavg_pool3d_fn, qavg_pool3d_nhwc_stub);

# 声明并分发量化批归一化函数的函数指针，采用 ReLU 激活函数
DECLARE_DISPATCH(qbatch_norm_fn, qbatch_norm_relu_stub);

# 声明并分发量化批归一化函数的函数指针，不使用激活函数
DECLARE_DISPATCH(qbatch_norm_fn, qbatch_norm_stub);

# 声明并分发量化二元运算函数的函数指针，采用加法和 ReLU 激活函数
DECLARE_DISPATCH(qbinary_fn, qadd_relu_stub);

# 声明并分发量化二元运算函数的函数指针，采用加法
DECLARE_DISPATCH(qbinary_fn, qadd_stub);

# 声明并分发量化二元运算函数的函数指针，采用乘法和 ReLU 激活函数
DECLARE_DISPATCH(qbinary_fn, qmul_relu_stub);

# 声明并分发量化二元运算函数的函数指针，采用乘法
DECLARE_DISPATCH(qbinary_fn, qmul_stub);

# 声明并分发量化按通道拼接函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qcat_nhwc_fn, qcat_nhwc_stub);

# 声明并分发量化按通道拼接函数的函数指针，采用 ReLU 激活函数的 NHWC 数据格式的存储
DECLARE_DISPATCH(qcat_nhwc_fn, qcat_relu_nhwc_stub);

# 声明并分发量化限幅函数的函数指针
DECLARE_DISPATCH(qclamp_fn, qclamp_stub);

# 声明并分发量化限幅函数的最小值限幅版本的函数指针
DECLARE_DISPATCH(qclamp_minmax_fn, qclamp_min_stub);

# 声明并分发量化限幅函数的最大值限幅版本的函数指针
DECLARE_DISPATCH(qclamp_minmax_fn, qclamp_max_stub);

# 声明并分发量化 ELU 函数的函数指针
DECLARE_DISPATCH(qelu_fn, qelu_stub);

# 声明并分发量化硬切比雪夫激活函数的函数指针
DECLARE_DISPATCH(qhardsigmoid_fn, qhardsigmoid_stub);

# 声明并分发量化硬 Swish 激活函数的函数指针
DECLARE_DISPATCH(qhardswish_fn, qhardswish_stub);

# 声明并分发量化 Dropout 函数的函数指针
DECLARE_DISPATCH(qdropout_fn, qdropout_stub);

# 声明并分发量化最大池化函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qmaxpool_2d_fn, qmaxpool_2d_nhwc_stub);

# 声明并分发量化最大池化函数的函数指针，采用 NTHWC 数据格式的存储
DECLARE_DISPATCH(qmaxpool_3d_fn, qmaxpool_3d_nthwc_stub);

# 声明并分发量化归一化函数的函数指针
DECLARE_DISPATCH(qnormalize_fn, quantized_normalize_stub);

# 声明并分发量化按通道归一化函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qnormalize_nhwc_fn, quantized_groupnorm_nhwc_stub);

# 声明并分发量化 ReLU 激活函数的函数指针
DECLARE_DISPATCH(qrelu_fn, qrelu_stub);

# 声明并分发量化带泄漏 ReLU 激活函数的函数指针
DECLARE_DISPATCH(qrelu_leaky_fn, qrelu_leaky_stub);

# 声明并分发量化 GELU 激活函数的函数指针
DECLARE_DISPATCH(qgelu_fn, qgelu_stub);

# 声明并分发量化 Sigmoid 激活函数的函数指针
DECLARE_DISPATCH(qsigmoid_fn, qsigmoid_stub);

# 声明并分发量化 Tanh 激活函数的函数指针
DECLARE_DISPATCH(qtanh_fn, qtanh_stub);

# 声明并分发量化阈值函数的函数指针
DECLARE_DISPATCH(qthreshold_fn, qthreshold_stub);

# 声明并分发量化 Top-K 函数的函数指针
DECLARE_DISPATCH(qtopk_fn, qtopk_stub);

# 声明并分发量化双线性上采样函数的函数指针，采用 NHWC 数据格式的存储
DECLARE_DISPATCH(qupsample_bilinear2d_fn, qupsample_bilinear2d_nhwc_stub);

# 声明并分发量化按内部维度计算均值函数的函数指针
DECLARE_DISPATCH(qmean_inner_dim_fn, qmean_inner_dim_stub);

# 声明并分发量化按内部维度计算标准差函数的函数指针
DECLARE_DISPATCH(qstd_inner_dim_fn, qstd_inner_dim_stub);

# 声明并分发量化 PReLU 激活函数的函数指针
DECLARE_DISPATCH(qprelu_fn, qprelu_stub);

} // namespace native
} // namespace at
```