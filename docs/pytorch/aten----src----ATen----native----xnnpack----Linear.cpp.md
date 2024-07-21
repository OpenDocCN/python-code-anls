# `.\pytorch\aten\src\ATen\native\xnnpack\Linear.cpp`

```py
// 如果定义了预编译宏 USE_XNNPACK，则包含以下头文件和命名空间
#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>  // 包含 XNNPACK 的通用功能
#include <ATen/native/utils/Factory.h>   // 包含 ATen 工厂函数的实用工具
#include <ATen/native/xnnpack/Linear.h>  // 包含 XNNPACK 的线性运算函数

namespace at::native::xnnpack {  // 进入 ATen 的 XNNPACK 实现命名空间
namespace internal::linear {     // 进入线性运算的内部命名空间

namespace {

// 支持 NHWC 和 NCHW 格式的 FP32 线性运算操作。

// TODO: 解耦和改进错误处理和消息。
bool available(
    const Tensor& weight,                  // 权重张量
    const std::optional<Tensor>& bias,     // 可选的偏置张量
    const float output_min,                // 输出最小值
    const float output_max) {              // 输出最大值
  // 检查是否支持 XNNPACK，以及对权重、偏置、输出范围的一致性要求
  return xnnpack::available() &&                      // 检查 XNNPACK 是否可用
         (2 == weight.ndimension()) &&                // 权重张量必须是二维的
         (weight.device().is_cpu()) &&                // 权重张量必须在 CPU 上
         (kFloat == weight.scalar_type()) &&          // 权重张量的数据类型必须是 float32
         !weight.requires_grad() &&                   // 权重张量不能需要梯度计算
         // 如果有偏置，则对偏置进行一致性检查
         ((bias && bias->defined()) ? (
              (1 == bias->ndimension()) &&            // 偏置张量必须是一维的
              (bias->device().is_cpu()) &&            // 偏置张量必须在 CPU 上
              (kFloat == bias->scalar_type()) &&      // 偏置张量的数据类型必须是 float32
              (weight.size(Layout::Filter::output)) == bias->size(0) &&  // 偏置长度需与权重的输出通道数匹配
              !bias->requires_grad())                 // 偏置张量不能需要梯度计算
            : true) &&
         // 检查输出范围的有效性
         (output_max > output_min) &&                 // 输出最大值必须大于最小值
         true;                                        // 如果以上所有条件满足，则返回 true
}

// TODO: 解耦和改进错误处理和消息。
bool usable(const Tensor& input) {
  // 检查输入张量的一致性要求
  return (1 <= input.ndimension()) &&                // 输入张量至少是一维的
         (input.device().is_cpu()) &&                // 输入张量必须在 CPU 上
         (kFloat == input.scalar_type()) &&          // 输入张量的数据类型必须是 float32
         !input.requires_grad() &&                   // 输入张量不能需要梯度计算
         true;                                       // 如果所有条件满足，则返回 true
}

// 创建和运行 XNNPACK 的线性运算
Tensor create_and_run(
    const Tensor& input,            // 输入张量
    const Tensor& weight,           // 权重张量
    const Tensor& bias,             // 偏置张量
    const float output_min,         // 输出最小值
    const float output_max) {       // 输出最大值
  return run(                       // 运行线性运算
      create(                      // 创建线性运算的上下文
          weight,                  // 权重张量
          bias,                    // 偏置张量
          output_min,              // 输出最小值
          output_max),             // 输出最大值
      input);                      // 输入张量
}

} // 匿名命名空间结束

ContextLinear create(
    const Tensor& weight,                  // 权重张量
    const std::optional<Tensor>& bias,     // 可选的偏置张量
    const float output_min,
    // 从 weight 张量创建连续存储的新张量
    const Tensor weight_contig = weight.contiguous();
    
    // 使用 TORCH_CHECK 验证 XNNPACK 线性层是否可用，检查参数的有效性
    TORCH_CHECK(
        // 调用 available 函数检查权重张量、偏置、输出最小值和最大值的有效性
        available(
            weight_contig,
            bias,
            output_min,
            output_max),
        "XNNPACK Linear not available! "
        "Reason: The provided (weight, bias, output_min, output_max) parameters are "
        "either invalid individually or their combination is not supported by XNNPACK.");
    
    // 定义 XNNPACK 线性操作符对象
    xnn_operator_t linear_op{};
    
    // 调用 xnn_create_fully_connected_nc_f32 创建全连接层的 XNNPACK 线性操作符
    const xnn_status create_status = xnn_create_fully_connected_nc_f32(
        // 输入通道数，与权重张量的输入尺寸相关
        weight_contig.size(Layout::Filter::input),
        // 输出通道数，与权重张量的输出尺寸相关
        weight_contig.size(Layout::Filter::output),
        // 输入像素步长，与权重张量的输入尺寸相关
        weight_contig.size(Layout::Filter::input),
        // 输出像素步长，与权重张量的输出尺寸相关
        weight_contig.size(Layout::Filter::output),
        // 权重数据指针，指向权重张量的数据
        weight_contig.data_ptr<float>(),
        // 如果存在偏置且已定义，使用偏置张量的连续存储数据指针；否则为 nullptr
        (bias && bias->defined()) ?
            bias->contiguous().data_ptr<float>() :
            nullptr,
        // 输出最小值，用于量化操作
        output_min,
        // 输出最大值，用于量化操作
        output_max,
        // 标志位，目前设置为 0
        0u,
        // XNNPACK 缓存对象，暂未使用，设置为 nullptr
        nullptr,
        // XNNPACK 权重缓存对象，暂未使用，设置为 nullptr
        nullptr,
        // 输出参数：存储创建的线性操作符对象
        &linear_op);
    
    // 使用 TORCH_CHECK 验证 XNNPACK 线性操作符创建是否成功
    TORCH_CHECK(
        xnn_status_success == create_status,
        "xnn_create_fully_connected_nc_f32 failed!");
    
    // 返回 ContextLinear 对象，其中包含创建的操作符及输出通道数
    return ContextLinear(
        Operator(linear_op),
        weight_contig.size(Layout::Filter::output)
    );
}

// 函数 `run` 实现了线性运算，接受一个上下文和输入张量，返回输出张量
Tensor run(
    const ContextLinear& context,     // 上下文对象，包含线性层参数和配置信息
    const Tensor& input) {            // 输入张量
  using namespace internal;

  // 兼容 aten::linear 函数，对于维度为 1 的输入进行扩展
  auto ip = input;
  if (input.ndimension() == 1) {
    ip = input.unsqueeze(0);
  }

  // 如果需要，为输入张量分配连续内存并填充
  const Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
      ip, ip.suggest_memory_format());

  // 检查填充后的输入张量是否可用于 XNNPACK 线性操作
  TORCH_CHECK(
      usable(padded_input),
      "XNNPACK Linear not usable! "
      "Reason: The provided input tensor is either invalid or unsupported by XNNPACK.");

  // 获取填充后输入张量的大小
  const IntArrayRef input_size = padded_input.sizes();
  std::vector<int64_t> output_size(input_size.cbegin(), input_size.cend());
  output_size.back() = context.output_channels;

  // 创建输出张量，考虑尾部填充和内存格式
  Tensor output = mobile::empty_with_tail_padding(
      output_size,
      padded_input.options().dtype(),
      padded_input.suggest_memory_format(),
      padded_input.opt_names());

  // 对线性操作进行形状重塑
  const xnn_status reshape_status = xnn_reshape_fully_connected_nc_f32(
      context.op.get(),                                   // 操作符
      Layout::ActivationND::batch(padded_input.sizes()),  // 批处理形状
      caffe2::pthreadpool_());                            // 线程池

  TORCH_CHECK(
      xnn_status_success == reshape_status,
      "xnn_reshape_fully_connected_nc_f32 failed!");

  // 设置完全连接的线性操作
  const xnn_status setup_status = xnn_setup_fully_connected_nc_f32(
      context.op.get(),                                   // 操作符
      padded_input.data_ptr<float>(),                     // 输入数据指针
      output.data_ptr<float>());                          // 输出数据指针

  TORCH_CHECK(
      xnn_status_success == setup_status,
      "xnn_setup_fully_connected_nc_f32 failed!");

  // 运行操作符
  const xnn_status run_status = xnn_run_operator(
      context.op.get(),         // 操作符
      caffe2::pthreadpool_());  // 线程池

  TORCH_INTERNAL_ASSERT(
      xnn_status_success == run_status,
      "xnn_run_operator failed!");

  // 兼容 aten::linear 函数，对于维度为 1 的输入进行压缩
  if (input.ndimension() == 1) {
      output.squeeze_(0);
  }

  // 返回输出张量
  return output;
}

// 创建并返回 XNNPack 线性操作上下文对象
c10::intrusive_ptr<xnnpack::LinearOpContext> createLinearClampPrePackOpContext(
    Tensor weight,                                  // 权重张量
    std::optional<Tensor> bias,                     // 可选偏置张量
    const std::optional<Scalar>& output_min,        // 可选输出最小值
    const std::optional<Scalar>& output_max) {      // 可选输出最大值
  return xnnpack::XNNPackLinearOpContext::create_context(
      std::move(weight), std::move(bias), output_min, output_max);
}

// 运行经预打包的 XNNPack 线性操作
Tensor linear_clamp_run(
    const Tensor& input,                            // 输入张量
    const c10::intrusive_ptr<xnnpack::LinearOpContext>& op_context) {  // XNNPack 线性操作上下文
  return op_context->run(input);
}

// 解包预打包的 XNNPack 线性操作的尺寸信息
IValue
unpack_prepacked_sizes_linear(const IValue& ivalue) {  // 输入值
  auto op_context = ivalue.toCustomClass<xnnpack::LinearOpContext>();  // 获取线性操作上下文对象
  const auto tuple = op_context->unpack();  // 解包得到元组
  const auto& bias = std::get<1>(tuple);    // 获取偏置张量
  return IValue(std::make_tuple(
      std::get<0>(tuple).sizes(),           // 返回权重张量的尺寸
      (bias && bias->defined()) ? at::OptionalIntArrayRef(bias->sizes()) : c10::nullopt));  // 如果存在偏置张量，则返回其尺寸，否则返回空
}

} // namespace internal::linear
    // 检查线性函数可用性，需要权重和偏置作为参数
    const Tensor& weight,
    const Tensor& bias) {
  // 调用内部函数检查线性函数是否可用，使用给定的权重和偏置以及上下文范围
  return internal::linear::available(
            weight,
            bias,
            ContextLinear::kMin,
            ContextLinear::kMax) &&
         // 检查输入张量是否可用于线性操作
         internal::linear::usable(input);
      // 错误：未能正确关闭前一行的语句，这行代码不会被执行
      internal::linear::usable(input);
}

// 结束 at::native::xnnpack 命名空间定义

Tensor linear(
    const Tensor& input,  // 输入张量
    const Tensor& weight,  // 权重张量
    const Tensor& bias) {  // 偏置张量
  // 调用内部函数创建并运行线性运算，返回结果张量
  return internal::linear::create_and_run(
      input,
      weight,
      bias,
      ContextLinear::kMin,  // 上下文线性运算的最小值
      ContextLinear::kMax); // 上下文线性运算的最大值
}

} // 结束 at::native::xnnpack 命名空间定义

#endif /* USE_XNNPACK */
```