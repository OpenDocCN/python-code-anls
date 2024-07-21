# `.\pytorch\aten\src\ATen\native\Cross.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入 ATen 库中所需的头文件
#include <ATen/native/Cross.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorMeta.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 来选择引入不同的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cross_native.h>
#include <ATen/ops/linalg_cross.h>
#include <ATen/ops/linalg_cross_native.h>
#endif

// 定义在 at::meta 命名空间中
namespace at::meta {

// 实现 linalg_cross 函数的元信息函数
TORCH_META_FUNC(linalg_cross)
(const Tensor & input, const Tensor & other, int64_t dim) {
  // 获取输入张量的维度
  auto x_d = input.dim();
  auto y_d = other.dim();
  // 检查输入张量的维度是否一致
  TORCH_CHECK(x_d == y_d, "linalg.cross: inputs must have the same number of dimensions.");
  // 检查指定维度上输入张量的大小是否为 3
  TORCH_CHECK(input.size(dim) == 3 && other.size(dim) == 3, "linalg.cross: inputs dimension ", dim, " must have length 3. Got ", input.size(dim), " and ", other.size(dim));

  // 推断输出张量的大小，保持非批处理维度一致，对批处理维度进行广播
  auto out_size = infer_size(input.sizes(), other.sizes());

  // 设置输出张量的原始步幅信息，使用输入张量的选项
  set_output_raw_strided(0, out_size, {}, input.options());
}

} // namespace at::meta

// 实现在 at::native 命名空间中的函数
namespace at::native {

// 定义跨库调度函数 cross_stub
DEFINE_DISPATCH(cross_stub);

// 定义默认跨维度函数 _default_cross_dim
static int64_t _default_cross_dim(const std::optional<int64_t> &dimension, SymIntArrayRef sizes) {
  // 如果未指定维度，默认选择第一个大小为 3 的维度
  // 注意：此行为可能不符合预期
  // 在跨实现中内部调用 _default_cross_dim 来计算维度，并最终将跨操作委托给 linalg_cross 实现
  if(dimension.has_value()) {
    return *dimension;
  }

  // 遍历输入张量的尺寸，找到第一个大小为 3 的维度
  for(auto i : c10::irange(sizes.size())) {
    if(sizes[i] == 3) {
      return i;
    }
  }
  // 如果未找到大小为 3 的维度，抛出错误
  TORCH_CHECK(false, "no dimension of size 3 in input");
}

// 实现跨张量操作的函数 cross
Tensor cross(const Tensor & input, const Tensor & other, const std::optional<int64_t> dimension) {
  // 如果未指定维度，发出警告
  if (!dimension) {
    TORCH_WARN_ONCE(
      "Using torch.cross without specifying the dim arg is deprecated.\n",
      "Please either pass the dim explicitly or simply use torch.linalg.cross.\n",
      "The default value of dim will change to agree with that of linalg.cross in a future release."
    );
  }
  // 获取默认的跨维度并调用 linalg_cross 函数
  auto dim = _default_cross_dim(dimension, input.sym_sizes());
  return at::linalg_cross(input, other, dim);
}

// 实现带有输出参数的跨张量操作的函数 cross_out
Tensor & cross_out(const Tensor & input, const Tensor & other, const std::optional<int64_t> dimension, Tensor & out) {
  // 获取默认的跨维度并调用 linalg_cross_out 函数
  auto dim = _default_cross_dim(dimension, input.sym_sizes());
  return at::linalg_cross_out(out, input, other, dim);
}

// 实现在 at::native 命名空间中的函数
TORCH_IMPL_FUNC(linalg_cross_out)
(const Tensor & input, const Tensor & other, int64_t dim, const Tensor & out) {
  // 使用 maybe_wrap_dim 函数确保维度 dim 在有效范围内
  dim = maybe_wrap_dim(dim, input.dim());
  // 获取输出张量 out 的大小作为广播的目标大小
  auto out_size = out.sizes();
  // 将输入张量 input 广播至目标大小
  Tensor input_broadcasted = input.expand(out_size);
  // 将另一个张量 other 广播至目标大小
  Tensor other_broadcasted = other.expand(out_size);

  // 调用 cross_stub 函数执行特定设备上的交叉操作
  cross_stub(input.device().type(), out, input_broadcasted, other_broadcasted, dim);
}

} // namespace at::native
```