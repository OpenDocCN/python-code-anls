# `.\pytorch\aten\src\ATen\native\vulkan\ops\Batchnorm.cpp`

```
namespace at {
namespace native {
namespace vulkan {
namespace ops {

namespace batchnorm {

// 定义一个结构体，用于存储批归一化操作的参数
struct Params final {
  api::utils::ivec3 out_extents; // 输出张量的维度信息
  int32_t c4; // 通道数除以4后的整数部分，用于 Vulkan 着色器计算
  float eps; // 归一化过程中的 epsilon 参数
};

// 记录批归一化操作的函数
void record_op(
    api::Context* const context, // Vulkan API 上下文对象指针
    vTensor& v_output, // 输出 Vulkan 张量对象的引用
    const vTensor& v_input, // 输入 Vulkan 张量对象的常量引用
    const vTensor& v_weight, // 权重 Vulkan 张量对象的常量引用
    const vTensor& v_bias, // 偏置 Vulkan 张量对象的常量引用
    const vTensor& v_running_mean, // 运行时均值 Vulkan 张量对象的常量引用
    const vTensor& v_running_var, // 运行时方差 Vulkan 张量对象的常量引用
    const float eps) // 归一化过程中的 epsilon 参数
{
  api::PipelineBarrier pipeline_barrier{}; // 管道屏障对象的初始化

  api::utils::uvec3 global_size = v_output.extents(); // 获取全局工作组大小
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size); // 根据全局大小确定局部工作组大小

  uint32_t num_features = get_dim<Dim4D::Channel>(v_input.sizes()); // 获取输入张量的通道数
  uint32_t channels_ext = api::utils::div_up(num_features, 4u); // 计算扩展后的通道数，每组4个通道

  Params block{
      api::utils::make_ivec3(v_output.extents()), // 初始化 Params 结构体的输出维度
      api::utils::safe_downcast<int32_t>(channels_ext), // 初始化 Params 结构体的通道数
      eps, // 初始化 Params 结构体的 epsilon 参数
  };

  api::UniformParamsBuffer params(context, block); // 创建用于参数传递的统一缓冲区对象

  context->submit_compute_job(
      // Vulkan 着色器描述符
      VK_KERNEL(batchnorm),
      // 管道屏障
      pipeline_barrier,
      // 全局工作组大小
      global_size,
      // 局部工作组大小
      local_size,
      // 栅栏句柄
      VK_NULL_HANDLE,
      // 着色器参数
      v_output.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_input.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_weight.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_bias.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_running_mean.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_running_var.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      // 参数缓冲区
      params.buffer());
}

} // namespace batchnorm

namespace {

using namespace api::utils;

// 执行批归一化的函数，返回批归一化后的张量
Tensor batch_norm(
    const at::Tensor& input_arg, // 输入张量
    const std::optional<Tensor>& weight_opt /* optional */, // 可选的权重张量
    const std::optional<Tensor>& bias_opt /* optional */, // 可选的偏置张量
    const std::optional<Tensor>& running_mean_opt /* optional */, // 可选的运行时均值张量
    const std::optional<Tensor>& running_var_opt /* optional */, // 可选的运行时方差张量
    bool training, // 是否处于训练模式
    double /* momentum, not used in eval mode */, // 动量参数，在评估模式下不使用
    double eps, // 归一化过程中的 epsilon 参数
    bool /* cudnn_enable, deprecated */) // 是否启用 CUDNN，已弃用
{
  TORCH_CHECK(!training, "Only evaluation mode is supported!"); // 检查是否处于评估模式
  TORCH_CHECK(input_arg.dim() == 4, "Input must have dim == 4!"); // 检查输入张量的维度是否为4
  TORCH_CHECK(
      get_dim<Dim4D::Channel>(input_arg) % 4 == 0,
      "Input must have channels divisible by 4!"); // 检查输入张量的通道数是否可以被4整除

  return run_batchnorm_context(
      input_arg,
      c10::make_intrusive<BatchNormPackedContext>(BatchNormPackedContext(
          weight_opt, bias_opt, running_mean_opt, running_var_opt, eps))); // 运行批归一化的上下文对象
}

#ifdef USE_VULKAN_API

// Vulkan API 的 ATen 库实现
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::batch_norm"), TORCH_FN(batch_norm)); // 注册 Vulkan 实现的批归一化操作
}

#endif /* USE_VULKAN_API */

} // namespace
// 定义 BatchNormPackedContext 类的构造函数，接受多个可选的张量和一个双精度浮点数作为参数
BatchNormPackedContext::BatchNormPackedContext(
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    double eps)
    : unpacked_{c10::AnyType::get()} {
  // 初始化 packed_，用于存储压缩后的数据，预先分配空间以提高效率
  packed_.reserve(ListArgs::kNumArgs);

  // 每个可选张量参数，如果提供，则应为一维张量。为了更高效地作为纹理进行打包，它们首先被重塑为 {N, 1, 1} 的形状。
  // 最终这种重排应该在 vTensor 本身内部自动完成。

  // 权重
  TORCH_CHECK(weight_opt, "Weight must be provided!");
  TORCH_CHECK(weight_opt->dim() == 1, "Weight must have ndim == 1!");

  const int64_t num_features =
      api::utils::safe_downcast<int64_t>(weight_opt->numel());
  const Tensor weight_3d = weight_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(weight_3d.vulkan());

  // 偏置
  TORCH_CHECK(bias_opt, "Bias must be provided!");
  TORCH_CHECK(bias_opt->dim() == 1, "Bias must have ndim == 1!");
  TORCH_CHECK(
      bias_opt->numel() == num_features,
      "Bias must have the same numel as weight!");

  const Tensor bias_3d = bias_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(bias_3d.vulkan());

  // 运行时均值
  TORCH_CHECK(running_mean_opt, "Running mean must be provided!");
  TORCH_CHECK(running_mean_opt->dim() == 1, "Running mean must have ndim == 1");
  TORCH_CHECK(
      running_mean_opt->numel() == num_features,
      "Running mean must have the same numel as weight!");

  const Tensor running_mean_3d =
      running_mean_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(running_mean_3d.vulkan());

  // 运行时方差
  TORCH_CHECK(running_var_opt, "Running var must be provided!");
  TORCH_CHECK(running_var_opt->dim() == 1, "Running var must have ndim == 1");
  TORCH_CHECK(
      running_var_opt->numel() == num_features,
      "Running var must have the same numel as weight!");

  const Tensor running_var_3d = running_var_opt->reshape({num_features, 1, 1});
  packed_.emplace_back(running_var_3d.vulkan());

  // Epsilon 值
  packed_.emplace_back(eps);

  // 如果不需要在预打包时释放权重，则在 unpacked_ 中保留未打包的参数列表
  if (!at::globalContext().releaseWeightsWhenPrepacking()) {
    unpacked_.reserve(ListArgs::kNumArgs);
    unpacked_.emplace_back(weight_opt);
    unpacked_.emplace_back(bias_opt);
    unpacked_.emplace_back(running_mean_opt);
    unpacked_.emplace_back(running_var_opt);
    unpacked_.emplace_back(eps);
  }
}
    // 创建一个批归一化（BatchNorm）的打包上下文对象，用于批归一化操作
    return c10::make_intrusive<BatchNormPackedContext>(BatchNormPackedContext(
        // 将权重的 std::optional<Tensor>&& 参数传递给 BatchNormPackedContext 构造函数
        weight_opt,
        // 将偏置的 std::optional<Tensor>&& 参数传递给 BatchNormPackedContext 构造函数
        bias_opt,
        // 将运行均值的 std::optional<Tensor>&& 参数传递给 BatchNormPackedContext 构造函数
        running_mean_opt,
        // 将运行方差的 std::optional<Tensor>&& 参数传递给 BatchNormPackedContext 构造函数
        running_var_opt,
        // 传递是否处于训练状态的布尔值参数给 BatchNormPackedContext 构造函数
        eps
    ));
} // 结束 run_batchnorm_context 函数定义

namespace ops { // 进入 ops 命名空间

namespace vulkan { // 进入 vulkan 命名空间

namespace native { // 进入 native 命名空间

namespace at { // 进入 at 命名空间
```