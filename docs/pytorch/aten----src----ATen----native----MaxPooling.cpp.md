# `.\pytorch\aten\src\ATen\native\MaxPooling.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/MaxPooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/max_pool1d_native.h>
#include <ATen/ops/max_pool1d_with_indices.h>
#include <ATen/ops/quantized_max_pool1d.h>
#endif

// 命名空间 at::native 开始
namespace at::native {

// 定义一个分发调度器，用于 max_pool1d 操作
DEFINE_DISPATCH(max_pool1d_stub);

// 匿名命名空间，实现 max_pool1d 操作的具体功能
namespace {

// 实现 max_pool1d 操作
Tensor max_pool1d_impl(
    const Tensor& self,               // 输入张量
    IntArrayRef kernel_size,          // 池化核大小
    IntArrayRef stride,               // 步幅
    IntArrayRef padding,              // 填充
    IntArrayRef dilation,             // 膨胀
    bool ceil_mode) {                 // 是否使用 ceil 模式

  NoNamesGuard guard;                 // 禁用名称传播

  // 如果步幅未指定，则设置为池化核大小
  if (stride.empty()) {
    stride = kernel_size;
  }

  // 计算输入张量的各维度大小
  const int64_t NB = self.dim() == 3 ? self.size(-3) : 1;
  const int64_t NC = self.size(-2);
  const int64_t IW = self.size(-1);
  const int64_t KW = kernel_size[0];
  const int64_t SJ = stride[0];
  const int64_t PJ = padding[0];
  const int64_t DJ = dilation[0];

  // 计算池化操作后输出的宽度 OW
  const int64_t OW = pooling_output_shape(IW, KW, PJ, SJ, DJ, ceil_mode);

  // 创建一个空张量作为输出，维度为 {NB, NC, OW}，使用输入张量的选项
  Tensor output = at::empty({NB, NC, OW}, self.options());

  // 定义池化参数
  PoolingParams1D params{NB, NC, IW, OW, KW, SJ, PJ, DJ};

  // 调用分发的 max_pool1d_stub 函数执行池化操作
  max_pool1d_stub(self.device().type(), output, self, params);

  // 如果输入张量是二维的，则压缩第一维度（批次维度）
  if (self.dim() == 2) {
    output.squeeze_(0);
  }

  guard.reset();  // 重置名称传播
  namedinference::propagate_names(output, self);  // 传播命名信息

  return output;  // 返回池化后的输出张量
}

} // namespace

// max_pool1d 函数定义，接受输入张量和池化参数
Tensor max_pool1d(
    const Tensor& self,               // 输入张量
    IntArrayRef kernel_size,          // 池化核大小
    IntArrayRef stride,               // 步幅
    IntArrayRef padding,              // 填充
    IntArrayRef dilation,             // 膨胀
    bool ceil_mode) {                 // 是否使用 ceil 模式

  auto ndim = self.ndimension();      // 计算输入张量的维度

  // 检查输入张量的维度是否符合要求
  TORCH_CHECK(
      (ndim == 2 && self.sym_size(0) != 0 && self.sym_size(1) != 0) ||
          (ndim == 3 && self.sym_size(1) != 0 && self.sym_size(2) != 0),
      "max_pool1d: Expected 2D or 3D (batch mode) tensor with optional 0 dim batch size for input, but got:",
      self.sym_sizes());

  // 如果输入张量是量化的，则调用量化的 max_pool1d 函数
  if (self.is_quantized()) {
    return at::quantized_max_pool1d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }

  // 检查 max_pool1d 操作的有效性
  check_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);

  // 如果需要计算梯度，并且梯度模式已启用，或者定义了前向梯度，并且输入张量不在 CPU 上，或者是张量子类
  if ((self.requires_grad() && at::GradMode::is_enabled()) ||
      self._fw_grad(/*level */ 0).defined() ||
      !self.device().is_cpu() ||
      isTensorSubclassLike(self)) {
    // 需要返回梯度索引，并且带有_indices 定义了 CUDA 分发
    return std::get<0>(at::max_pool1d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode));
  }

  // 否则，调用 max_pool1d_impl 执行池化操作
  return max_pool1d_impl(
      self, kernel_size, stride, padding, dilation, ceil_mode);
}

} // namespace at::native
```