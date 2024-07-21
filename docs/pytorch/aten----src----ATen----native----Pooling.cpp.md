# `.\pytorch\aten\src\ATen\native\Pooling.cpp`

```py
// 定义宏，仅包含方法操作符断言
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入张量核心功能库
#include <ATen/core/Tensor.h>
// 引入张量实用工具库
#include <ATen/TensorUtils.h>
// 引入命名张量实用工具库
#include <ATen/NamedTensorUtils.h>
// 引入XNNPACK引擎库
#include <ATen/native/xnnpack/Engine.h>
// 引入C10异常处理工具库
#include <c10/util/Exception.h>

// 如果未定义每个操作符的头文件，则引入张量函数库和原生函数库
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 如果定义了每个操作符的头文件，则引入适应性平均池化1D的原生操作
#else
#include <ATen/ops/adaptive_avg_pool1d_native.h>
// 引入适应性平均池化2D操作
#include <ATen/ops/adaptive_avg_pool2d.h>
// 引入适应性最大池化1D的原生操作
#include <ATen/ops/adaptive_max_pool1d_native.h>
// 引入适应性最大池化2D操作
#include <ATen/ops/adaptive_max_pool2d.h>
// 引入平均池化1D的原生操作
#include <ATen/ops/avg_pool1d_native.h>
// 引入平均池化2D操作
#include <ATen/ops/avg_pool2d.h>
// 引入带索引的最大池化1D原生操作
#include <ATen/ops/max_pool1d_with_indices_native.h>
// 引入最大池化2D原生操作
#include <ATen/ops/max_pool2d_native.h>
// 引入带索引的最大池化2D操作
#include <ATen/ops/max_pool2d_with_indices.h>
// 引入最大池化3D原生操作
#include <ATen/ops/max_pool3d_native.h>
// 引入带索引的最大池化3D操作
#include <ATen/ops/max_pool3d_with_indices.h>
// 引入MKLDNN引擎的最大池化2D操作
#include <ATen/ops/mkldnn_max_pool2d.h>
// 引入MKLDNN引擎的最大池化3D操作
#include <ATen/ops/mkldnn_max_pool3d.h>
// 引入量化的最大池化2D操作
#include <ATen/ops/quantized_max_pool2d.h>
// 引入量化的最大池化3D操作
#include <ATen/ops/quantized_max_pool3d.h>
#endif

// 引入元组类库
#include <tuple>

// 命名空间声明：at::native
namespace at::native {

// 静态方法：检查1D张量尺寸
static void check1d(
    const char* function_name,
    const char* argument_name,
    IntArrayRef x) {
  // 断言：张量尺寸应为1
  TORCH_CHECK(
      x.size() == 1,
      function_name, "() argument '", argument_name,
      "' should contain one int (got ", x.size(), ")");
}

// 函数：自适应平均池化1D操作
Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size) {
  // 检查维度范围
  checkDimRange("adaptive_avg_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  // 检查1D张量尺寸
  check1d("adaptive_avg_pool1d", "output_size", output_size);

  // 执行自适应平均池化2D操作
  auto output = at::adaptive_avg_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  // 压缩张量维度
  return output.squeeze(-2);
}

// 函数：自适应最大池化1D操作，返回元组
std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size) {
  // 检查维度范围
  checkDimRange("adaptive_max_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  // 检查1D张量尺寸
  check1d("adaptive_max_pool1d", "output_size", output_size);

  // 获取张量维度数
  int ndim = self.ndimension();
  // 循环遍历非批处理维度
  for (const auto i : c10::irange(1, ndim)) {
    // 断言：非批处理维度应为正数
    TORCH_CHECK(
        self.sym_size(i) > 0,
        "adaptive_max_pool1d(): ",
        "Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        self.sym_sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  // 执行自适应最大池化2D操作
  auto [output, indices] = at::adaptive_max_pool2d(
      self.unsqueeze(-2),
      {1, output_size[0]});

  // 压缩张量维度
  return std::make_tuple(output.squeeze(-2), indices.squeeze(-2));
}

// 函数：带索引的最大池化1D操作，返回元组
std::tuple<Tensor, Tensor> max_pool1d_with_indices(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 如果步长为空
  if (stride.empty()) {
    stride = kernel_size;

这行代码将变量 `stride` 设置为 `kernel_size` 的值，用于定义池化操作中的步幅。


  }
  checkDimRange("max_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);

这里结束了前面代码块的一个大括号，可能是一个条件或循环的结尾。


  check1d("max_pool1d", "kernel_size", kernel_size);

调用函数 `check1d` 来验证 `kernel_size` 的合法性，用于确保 `kernel_size` 是一个有效的一维尺寸。


  check1d("max_pool1d", "stride", stride);

验证 `stride` 的合法性，确保其为有效的一维步幅值。


  check1d("max_pool1d", "padding", padding);

验证 `padding` 的合法性，确保其为有效的一维填充值。


  check1d("max_pool1d", "dilation", dilation);

验证 `dilation` 的合法性，确保其为有效的一维膨胀值。


  NoNamesGuard guard;

创建 `NoNamesGuard` 对象 `guard`，用于在其生命周期内禁用命名推断功能。


  auto [output, indices] = at::max_pool2d_with_indices(
      self.unsqueeze(-2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      {1, dilation[0]},
      ceil_mode);

调用 PyTorch 的 `at::max_pool2d_with_indices` 函数进行最大池化操作，返回池化后的输出 `output` 和池化操作的索引 `indices`。


  output  = output.squeeze(-2);
  indices = indices.squeeze(-2);

对 `output` 和 `indices` 进行维度压缩，移除指定维度 `-2` 上的尺寸为1的维度。


  guard.reset();

重置 `NoNamesGuard` 对象 `guard`，重新启用命名推断功能。


  namedinference::propagate_names(output, self);
  namedinference::propagate_names(indices, self);

利用 `namedinference::propagate_names` 函数将 `output` 和 `indices` 的命名信息传播给 `self`，用于命名推断。


  return std::make_tuple(output, indices);

返回包含 `output` 和 `indices` 的元组作为函数的结果。
// 定义 avg_pool1d 函数，计算 1 维平均池化后的张量
Tensor avg_pool1d(
    // 输入张量 self，需要池化的核大小 kernel_size，步长 stride，填充 padding，是否向上取整 ceil_mode，是否包括填充 count_include_pad
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad) {
  // 如果步长为空，则将步长设置为核大小
  if (stride.empty()) {
    stride = kernel_size;
  }
  // 检查输入张量 self 的维度范围是否在 2 到 4 之间（不包括 4）
  checkDimRange("avg_pool1d", TensorArg(self, "self", 1), 2, 4 /* exclusive */);
  // 检查核大小 kernel_size 是否为 1 维
  check1d("avg_pool1d", "kernel_size", kernel_size);
  // 检查步长 stride 是否为 1 维
  check1d("avg_pool1d", "stride", stride);
  // 检查填充 padding 是否为 1 维
  check1d("avg_pool1d", "padding", padding);

  // 调用 at::avg_pool2d 执行 2 维平均池化，对输入张量在倒数第二维上进行扩展，然后收缩维度得到输出
  auto output = at::avg_pool2d(
      self.unsqueeze(-2),
      {1, kernel_size[0]},
      {1, stride[0]},
      {0, padding[0]},
      ceil_mode,
      count_include_pad);

  // 压缩维度，返回平均池化后的输出张量
  return output.squeeze(-2);
}

// 定义 max_pool2d 函数，计算 2 维最大池化后的张量
Tensor max_pool2d(
    // 输入张量 self，需要池化的核大小 kernel_size，步长 stride，填充 padding，扩张 dilation，是否向上取整 ceil_mode
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 如果输入张量被量化，则调用对应的量化最大池化函数
  if (self.is_quantized()) {
    return at::quantized_max_pool2d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
  // 如果输入张量使用 MKLDNN 引擎，则调用对应的 MKLDNN 最大池化函数
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool2d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  // 如果在移动设备上且使用 XNNPACK，则调用 XNNPACK 的最大池化函数
#if defined(C10_MOBILE)
  if(xnnpack::use_max_pool2d(self, kernel_size, padding, stride,
                             dilation, ceil_mode)) {
    return xnnpack::max_pool2d(
        self, kernel_size, padding, stride, dilation, ceil_mode);
  }
#endif
  // 否则，调用标准的带索引返回的最大池化函数
  auto output_and_indices = at::max_pool2d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  // 返回最大池化后的输出张量
  return std::get<0>(output_and_indices);
}

// 定义 max_pool3d 函数，计算 3 维最大池化后的张量
Tensor max_pool3d(
    // 输入张量 self，需要池化的核大小 kernel_size，步长 stride，填充 padding，扩张 dilation，是否向上取整 ceil_mode
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  // 如果输入张量被量化，则调用对应的量化最大池化函数
  if (self.is_quantized()) {
    return at::quantized_max_pool3d(self, kernel_size, stride, padding,
                                    dilation, ceil_mode);
  }
  // 如果输入张量使用 MKLDNN 引擎，则调用对应的 MKLDNN 最大池化函数
  if (self.is_mkldnn()) {
    return at::mkldnn_max_pool3d(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  // 否则，调用标准的带索引返回的 3 维最大池化函数
  auto output_and_indices = at::max_pool3d_with_indices(
      self, kernel_size, stride, padding, dilation, ceil_mode);
  // 返回最大池化后的输出张量
  return std::get<0>(output_and_indices);
}

// 结束 at::native 命名空间
} // namespace at::native
```