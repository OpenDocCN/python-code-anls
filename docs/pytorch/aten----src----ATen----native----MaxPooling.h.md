# `.\pytorch\aten\src\ATen\native\MaxPooling.h`

```py
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pool.h>

namespace at::native {

// 检查 max_pool1d 函数的输入参数是否合法
inline void check_max_pool1d(
    const Tensor& self,               // 输入的张量 self
    IntArrayRef kernel_size,          // 池化核大小
    IntArrayRef stride,               // 步幅
    IntArrayRef padding,              // 填充
    IntArrayRef dilation,             // 空洞卷积的扩张率
    bool ceil_mode) {                 // 是否使用 ceil 模式进行池化计算

  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 3,
      "max_pool1d() Expected 2D or 3D input tensor, but got ", self.sizes());
  TORCH_CHECK(
      kernel_size.size() == 1,
      "max_pool1d() kernel_size must be an int, list of ints or tuple of ints of size 1 but got size ",
      kernel_size.size());
  TORCH_CHECK(
      stride.empty() || stride.size() == 1,
      "max_pool1d() stride must be None, an int, list of ints, or tuple of ints of size 1 but got size ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1,
      "max_pool1d() padding must be an int, list of ints, or tuple of ints of size 1 but got size ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1,
      "max_pool1d() dilation must be an int, list of ints or tuple of ints of size 1 but got size ",
      dilation.size());

  // 如果 stride 是空的，则将其设置为 kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }

  TORCH_CHECK(
      kernel_size[0] > 0,
      "max_pool1d() kernel_size must be greater than zero, but got ",
      kernel_size[0]);
  TORCH_CHECK(
      stride[0] > 0, "max_pool1d() stride must be greater than zero, but got ", stride[0]);
  TORCH_CHECK(
      padding[0] >= 0, "max_pool1d() padding must be non-negative, but got ", padding[0]);
  TORCH_CHECK(
      padding[0] <= kernel_size[0] / 2,
      "max_pool1d() padding should be at most half of kernel size, but got padding=",
      padding[0],
      " and kernel_size=",
      kernel_size[0]);
  TORCH_CHECK(
      dilation[0] > 0, "max_pool1d() dilation must be greater than zero, but got ", dilation[0]);

  // 计算池化操作后的输出宽度 OW
  const int64_t OW = pooling_output_shape(self.size(-1), kernel_size[0], padding[0], stride[0], dilation[0], ceil_mode);
  TORCH_CHECK(OW > 0, "max_pool1d() Invalid computed output size: ", OW);
}

// TODO(Heitor) Template by dimension
// 定义一维池化的参数结构体
struct PoolingParams1D {
  int64_t NB; // 批次数
  int64_t NC; // 通道数
  int64_t IW; // 输入宽度
  int64_t OW; // 输出宽度
  int64_t KW; // 核宽度
  int64_t SJ; // 列步幅
  int64_t PJ; // 列填充
  int64_t DJ; // 列空洞

  // 根据给定的核和输出索引返回输入元素的索引
  inline int64_t index(int64_t kj, int64_t oj) const {
    return oj * SJ + kj * DJ - PJ;
  }

  // 返回此核索引内有效输出的第一个索引
  inline int64_t valid_output_start(int64_t kj) const {
    int64_t ij = index(kj, 0);
    return ij < 0 ? at::divup(-ij, SJ) : 0;
  }

  // 返回当前核索引下的最后一个输出位置的下一个索引，确保在边界内
  inline int64_t valid_output_end(int64_t kj) const {
    // 计算核索引 kj 对应的输出索引 ij
    int64_t ij = index(kj, OW - 1);
    // 如果 ij 大于等于输入宽度 IW，则返回 OW 减去超出部分除以步长 SJ 后的结果
    return ij >= IW ? OW - at::divup(ij - (IW - 1), SJ) : OW;
  }
};

// 声明一个函数指针类型 pooling_fn，该函数接受一个 Tensor 引用、另一个 Tensor 以及 PoolingParams1D 对象作为参数，并返回 void
using pooling_fn = void (*)(Tensor&, const Tensor&, const PoolingParams1D&);

// 声明一个分发函数 max_pool1d_stub，其类型为 pooling_fn，用于执行一维最大池化操作
DECLARE_DISPATCH(pooling_fn, max_pool1d_stub);

// 结束命名空间 at::native
} // namespace at::native
```