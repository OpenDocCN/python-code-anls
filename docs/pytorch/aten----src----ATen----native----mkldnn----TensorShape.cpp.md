# `.\pytorch\aten\src\ATen\native\mkldnn\TensorShape.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymIntArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_mkldnn_reshape_native.h>
#include <ATen/ops/_mkldnn_transpose_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/view_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

// 如果未启用 MKLDNN 支持，则定义以下函数，抛出错误信息
Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_clone(const Tensor& self, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "mkldnn_clone: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

// 如果启用了 MKLDNN 支持，则定义以下函数来处理 MKLDNN 张量操作

// 根据指定的尺寸重塑 MKLDNN 张量
Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false,
      "Currently Mkldnn tensor does not support view. Change to use reshape instead");
}

// 根据指定的尺寸重塑 MKLDNN 张量
Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  // 推断重塑后的尺寸
  auto inferred_size = at::infer_size(size, self.numel());
  // 如果当前张量尺寸与推断的尺寸相同，则直接返回当前张量
  if (self.sizes() == inferred_size) {
    return self;
  }
  // 将 MKLDNN 张量转换为 ideep::tensor 类型
  const ideep::tensor& x = itensor_from_mkldnn(self);
  // 创建一个新的 ideep::tensor 对象，并重塑为推断的尺寸
  ideep::tensor y{x};
  y.reshape(inferred_size);
  // 将 ideep::tensor 转换为 ATen 张量并返回
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

// 克隆 MKLDNN 张量
Tensor mkldnn_clone(const Tensor& self, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  // 获取 MKLDNN 张量的 ideep::tensor 引用
  ideep::tensor& src = itensor_from_mkldnn(self);
  // 创建一个新的 ideep::tensor 对象，并将数据从源张量复制过来
  ideep::tensor dst;
  ideep::direct_copy::compute(src, dst);
  // 将 ideep::tensor 转换为 ATen 张量并返回
  return new_with_itensor_mkldnn(std::move(dst), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}
// 定义一个函数 mkldnn_transpose，用于实现 MKL-DNN 张量的转置操作
Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  // 获取输入张量的维度数
  auto ndims = self.dim();
  // 根据维度数修正 dim0 和 dim1，确保它们在有效范围内
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  // 将 Torch 张量转换为 MKL-DNN 的 ideep::tensor
  const ideep::tensor& x = itensor_from_mkldnn(self);
  // 定义一个新的 ideep::tensor 对象 y
  ideep::tensor y;
  // 创建一个整数向量 axes，用来表示张量的维度索引
  std::vector<int> axes(x.ndims());
  // 初始化 axes，按顺序填充为 [0, 1, 2, ..., ndims-1]
  std::iota(axes.begin(), axes.end(), 0);
  // 交换 axes 中 dim0 和 dim1 位置的索引，实现转置操作
  std::swap(axes[dim0], axes[dim1]);
  // 使用转置后的 axes 对象，将 ideep::tensor x 的数据转置到 y 中
  y.transpose_from(x, axes);
  // 创建一个新的 Torch 张量，并使用转置后的 ideep::tensor y 初始化它
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

// 定义一个原地操作函数 mkldnn_transpose_，用于 MKL-DNN 张量的原地转置
Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  // 抛出错误，因为原地操作 mkldnn_transpose_ 目前不支持
  TORCH_CHECK(false, "mkldnn_transpose_: in-place mkldnn operations are not supported yet");
}

// 命名空间结束标记，native 命名空间
} // namespace native

// 命名空间结束标记，at 命名空间
} // namespace at

// 条件编译结束标记，AT_MKLDNN_ENABLED
#endif // AT_MKLDNN_ENABLED
```