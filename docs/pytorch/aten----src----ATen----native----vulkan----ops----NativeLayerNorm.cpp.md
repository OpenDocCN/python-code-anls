# `.\pytorch\aten\src\ATen\native\vulkan\ops\NativeLayerNorm.cpp`

```py
// 包含 Vulkan 操作的常用头文件
#include <ATen/native/vulkan/ops/Common.h>
// Torch 库的头文件
#include <torch/library.h>

// 命名空间 at::native::vulkan::ops::
namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

// 使用 api::utils 命名空间
using namespace api::utils;

// 检查 LayerNorm 的输入参数
void _check_layer_norm_inputs(
    const at::Tensor& input,                    // 输入张量
    IntArrayRef normalized_shape,                // 归一化形状
    const std::optional<Tensor>& weight /* optional */,  // 可选的权重张量
    const std::optional<Tensor>& bias /* optional */) {  // 可选的偏置张量

  // 归一化形状的维度数
  const auto normalized_ndim = normalized_shape.size();
  // 检查归一化形状至少是一维的
  TORCH_CHECK(
      normalized_ndim >= 1,
      "Expected normalized_shape to be at least 1-dimensional, i.e., ",
      "containing at least one element, but got normalized_shape = ",
      normalized_shape);
  
  // 检查权重与归一化形状的一致性
  TORCH_CHECK(
      !weight->defined() || weight->sizes().equals(normalized_shape),
      "Expected weight to be of same shape as normalized_shape, but got ",
      "weight of shape ",
      weight->sizes(),
      " and normalized_shape = ",
      normalized_shape);
  
  // 检查偏置与归一化形状的一致性
  TORCH_CHECK(
      !bias->defined() || bias->sizes().equals(normalized_shape),
      "Expected bias to be of same shape as normalized_shape, but got ",
      "bias of shape ",
      bias->sizes(),
      " and normalized_shape = ",
      normalized_shape);

  // 输入张量的形状和维度
  const auto input_shape = input.sizes();
  const auto input_ndim = input.sizes().size();

  // 检查输入张量的形状是否符合预期
  if (input_ndim < normalized_ndim ||
      !input_shape.slice(input_ndim - normalized_ndim)
           .equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape
       << ", expected input with shape [*";
    for (auto size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input_shape;
    AT_ERROR(ss.str());
  }
}

// Vulkan 实现的 LayerNorm 函数
std::tuple<Tensor, Tensor, Tensor> native_layer_norm(
    const at::Tensor& input_arg,                // 输入张量
    IntArrayRef normalized_shape,                // 归一化形状
    const std::optional<Tensor>& weight_opt /* optional */,  // 可选的权重张量
    const std::optional<Tensor>& bias_opt /* optional */,    // 可选的偏置张量
    double eps) {                                  // epsilon 值

  // 检查 LayerNorm 的输入参数
  _check_layer_norm_inputs(input_arg, normalized_shape, weight_opt, bias_opt);

  // 检查输入张量的维度
  TORCH_CHECK(
      input_arg.dim() >= 2 && input_arg.dim() <= 4,
      "Vulkan layernorm expects input of 2d, 3d or 4d!");

  // 如果输入张量是 Vulkan 张量，则直接使用；否则将其转换为 Vulkan 张量
  const Tensor input = input_arg.is_vulkan() ? input_arg : input_arg.vulkan();

  // 检查权重和偏置张量是否定义
  TORCH_CHECK(
      weight_opt->defined() && bias_opt->defined(),
      "Vulkan layernorm expects weight and bias arguments");

  // 如果权重张量是 Vulkan 张量，则直接使用；否则将其转换为 Vulkan 张量
  const Tensor weight =
      weight_opt->is_vulkan() ? *weight_opt : weight_opt->vulkan();

  // 如果偏置张量是 Vulkan 张量，则直接使用；否则将其转换为 Vulkan 张量
  const Tensor bias = bias_opt->is_vulkan() ? *bias_opt : bias_opt->vulkan();

  // 要进行归约的维度
  std::vector<int64_t> dims_to_reduce;
  // 遍历归一化形状的维度
  for (const auto i : c10::irange(normalized_shape.size())) {
    // 将要减少的维度依次加入 dims_to_reduce 向量
    dims_to_reduce.push_back(input_arg.dim() - i - 1);
  }
  // 创建 IntArrayRef 类型的引用 dims_to_reduce_ref
  IntArrayRef dims_to_reduce_ref = IntArrayRef(dims_to_reduce);

  // 设置是否保持维度
  bool mean_keep_dim = true;
  bool var_keep_dim = true;
  // 计算沿指定维度的均值 mean
  auto mean = input.mean(dims_to_reduce_ref, mean_keep_dim);

  // 为了避免重新计算均值，手动计算方差 var，而不是调用 var 运算符。
  auto input_minus_mean = input.sub(mean);
  auto var = input_minus_mean.mul(input_minus_mean)
                 .mean(dims_to_reduce_ref, var_keep_dim);
  // 计算标准差的倒数，加上 eps 以避免除以零
  auto std_inv = var.add(eps).pow(-0.5f);

  // 使用给定的公式计算 LayerNorm：
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  // 计算 LayerNorm，考虑到标准差的倒数、权重和偏置
  auto layernorm = input_minus_mean.mul(std_inv).mul(weight).add(bias);
  // 返回包含 layernorm、均值和标准差的元组
  std::tuple<Tensor, Tensor, Tensor> output =
      std::make_tuple(layernorm, mean, std_inv);
  // 返回最终的输出元组
  return output;
}

#ifdef USE_VULKAN_API

闭合了前面未显示的某个代码块。


TORCH_LIBRARY_IMPL(aten, Vulkan, m) {

定义了一个 Torch 库的实现，用于 Vulkan 后端。


  m.impl(
      TORCH_SELECTIVE_NAME("aten::native_layer_norm"),
      TORCH_FN(native_layer_norm));

注册了一个名为 "aten::native_layer_norm" 的操作，在 Vulkan 后端使用函数 `native_layer_norm` 来实现。


}
#endif /* USE_VULKAN_API */

结束了针对 Vulkan API 使用的条件编译区块。


} // namespace

结束了一个未显示的命名空间。


} // namespace ops

结束了 ops 命名空间。


} // namespace vulkan

结束了 vulkan 命名空间。


} // namespace native

结束了 native 命名空间。


} // namespace at

结束了 at 命名空间。
```