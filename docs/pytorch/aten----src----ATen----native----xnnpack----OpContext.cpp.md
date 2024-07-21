# `.\pytorch\aten\src\ATen\native\xnnpack\OpContext.cpp`

```
#ifdef USE_XNNPACK
#include <ATen/native/xnnpack/Convolution.h>  // 导入 XNNPACK 卷积操作的头文件
#include <ATen/native/xnnpack/Linear.h>       // 导入 XNNPACK 线性操作的头文件
#include <ATen/native/xnnpack/OpContext.h>    // 导入 XNNPACK 操作上下文的头文件

#include <ATen/Context.h>                     // 导入 ATen 上下文的头文件

namespace at::native::xnnpack {

c10::intrusive_ptr<LinearOpContext>
XNNPackLinearOpContext::create_context(
    at::Tensor&& weight,                      // 移动语义的权重张量
    std::optional<at::Tensor>&& bias,         // 可选的移动语义偏置张量
    const std::optional<Scalar>& output_min,  // 可选的输出最小标量值
    const std::optional<Scalar>& output_max) { // 可选的输出最大标量值
  auto linear_op_context =
      c10::make_intrusive<XNNPackLinearOpContext>(  // 创建 XNNPack 线性操作上下文
          std::move(weight),
          std::move(bias),
          output_min,
          output_max,
          xnnpack::internal::linear::create(     // 使用 XNNPACK 内部函数创建线性操作
              weight,
              bias,
              output_min ? output_min->to<float>()
                         : xnnpack::ContextLinear::kMin,
              output_max ? output_max->to<float>()
                         : xnnpack::ContextLinear::kMax)
          );
  if (at::globalContext().releaseWeightsWhenPrepacking()) {
    linear_op_context->free_orig_weight_and_bias();  // 如果全局上下文允许在预打包时释放权重和偏置，则释放
  }

  return linear_op_context;  // 返回线性操作上下文指针
}

void XNNPackLinearOpContext::free_orig_weight_and_bias() {
  orig_weight_and_bias_freed_ = true;  // 标记原始权重和偏置已释放
  orig_weight_.reset();                // 重置原始权重
  orig_bias_.reset();                  // 重置原始偏置
}

Tensor XNNPackLinearOpContext::run(const Tensor& input) {
  return xnnpack::internal::linear::run(op_context_, input);  // 运行 XNNPack 线性操作
}

c10::intrusive_ptr<Conv2dOpContext>
XNNPackConv2dOpContext::create_context(at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  auto op_context =
      xnnpack::internal::convolution2d::create(    // 使用 XNNPACK 内部函数创建二维卷积操作上下文
          weight,
          bias,
          padding,
          {0, 0},  // 输出填充为零
          stride,
          dilation,
          groups,
          false,   // 非转置卷积
          output_min ? output_min->to<float>()
                     : xnnpack::ContextConv2D::kMin,
          output_max ? output_max->to<float>()
                     : xnnpack::ContextConv2D::kMax);

  auto conv2d_op_context =
      c10::make_intrusive<XNNPackConv2dOpContext>(  // 创建 XNNPack 二维卷积操作上下文
          std::move(weight),
          std::move(bias),
          std::move(padding),
          std::move(stride),
          std::move(dilation),
          groups,
          output_min,
          output_max,
          std::move(op_context));

  if (at::globalContext().releaseWeightsWhenPrepacking()) {
    conv2d_op_context->free_orig_weight_and_bias();  // 如果全局上下文允许在预打包时释放权重和偏置，则释放
  }

  return conv2d_op_context;  // 返回二维卷积操作上下文指针
}

c10::intrusive_ptr<TransposeConv2dOpContext>
XNNPackTransposeConv2dOpContext::create_context(at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    // 创建一个 XNNPack 内部的反卷积操作上下文，用于构建反卷积操作
    auto op_context =
        xnnpack::internal::convolution2d::create(
            weight,                     // 权重张量
            bias,                       // 偏置张量
            padding,                    // 填充大小
            output_padding,             // 输出填充大小
            stride,                     // 步幅大小
            dilation,                   // 膨胀大小
            groups,                     // 分组数
            true,                       // 是否是反卷积操作（转置）
            output_min ? output_min->to<float>() : xnnpack::ContextConv2D::kMin,  // 输出最小值，若未提供则使用默认最小值
            output_max ? output_max->to<float>() : xnnpack::ContextConv2D::kMax   // 输出最大值，若未提供则使用默认最大值
        );
    
    // 创建 XNNPack 反转置卷积操作上下文
    auto conv2d_op_context =
        c10::make_intrusive<XNNPackTransposeConv2dOpContext>(
            std::move(weight),          // 移动权重张量
            std::move(bias),            // 移动偏置张量
            std::move(padding),         // 移动填充大小
            std::move(output_padding),  // 移动输出填充大小
            std::move(stride),          // 移动步幅大小
            std::move(dilation),        // 移动膨胀大小
            groups,                     // 分组数
            output_min,                 // 输出最小值（可选）
            output_max,                 // 输出最大值（可选）
            std::move(op_context)       // 移动反卷积操作上下文
        );
    
    // 如果全局上下文要求在预打包时释放权重，则释放原始权重和偏置
    if (at::globalContext().releaseWeightsWhenPrepacking()) {
        conv2d_op_context->free_orig_weight_and_bias();
    }
    
    // 返回 XNNPack 反转置卷积操作上下文
    return conv2d_op_context;
}

// 定义命名空间结束标记，结束 at::native::xnnpack 命名空间的定义

#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则编译以下内容

Tensor XNNPackConv2dOpContext::run(const Tensor& input) {
  // 创建互斥锁保护
  std::lock_guard<std::mutex> lock(xnnp_mutex_);
  // 调用 XNNPack 内部的二维卷积运行函数，返回结果
  return xnnpack::internal::convolution2d::run(op_context_, input);
}

Tensor XNNPackTransposeConv2dOpContext::run(const Tensor& input) {
  // 创建互斥锁保护
  std::lock_guard<std::mutex> lock(xnnp_mutex_);
  // 调用 XNNPack 内部的转置二维卷积运行函数，返回结果
  return xnnpack::internal::convolution2d::run(op_context_, input);
}

void XNNPackConv2dOpContext::free_orig_weight_and_bias() {
  // 标记原始权重和偏置已释放
  orig_weight_and_bias_freed_ = true;
  // 重置原始权重指针
  orig_weight_.reset();
  // 重置原始偏置指针
  orig_bias_.reset();
}

void XNNPackTransposeConv2dOpContext::free_orig_weight_and_bias() {
  // 标记原始权重和偏置已释放
  orig_weight_and_bias_freed_ = true;
  // 重置原始权重指针
  orig_weight_.reset();
  // 重置原始偏置指针
  orig_bias_.reset();
}

} // namespace at::native::xnnpack
// 结束 at::native::xnnpack 命名空间的定义

#endif /* USE_XNNPACK */
// 结束 USE_XNNPACK 宏条件编译块
```