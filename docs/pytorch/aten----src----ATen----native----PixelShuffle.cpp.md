# `.\pytorch\aten\src\ATen\native\PixelShuffle.cpp`

```
// 定义宏，用于仅在方法操作符中包含 ASSERT
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含张量转换相关头文件
#include <ATen/native/TensorTransformations.h>
// 包含 CPU 端像素重排的内核函数头文件
#include <ATen/native/cpu/PixelShuffleKernel.h>
// 包含像素重排操作的头文件
#include <ATen/native/PixelShuffle.h>

// 包含异常处理相关的实用函数
#include <c10/util/Exception.h>

// 根据 AT_PER_OPERATOR_HEADERS 宏条件包含不同的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
// 包含 ATen 库函数和原生函数的头文件
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
// 包含 ATen 中的 empty 操作头文件
#include <ATen/ops/empty.h>
// 包含 ATen 中的像素重排和逆重排操作头文件
#include <ATen/ops/pixel_shuffle_native.h>
#include <ATen/ops/pixel_unshuffle_native.h>
#endif

// 包含算法和容器相关的标准库头文件
#include <algorithm>
#include <numeric>
#include <vector>

// ATen 库的命名空间
namespace at::native {

// 实现 CPU 端的像素重排操作
Tensor pixel_shuffle_cpu(const Tensor& self, int64_t upscale_factor) {
  // 检查输入张量和放大因子的形状是否符合像素重排的要求
  check_pixel_shuffle_shapes(self, upscale_factor);

  // 构建输出张量的尺寸，格式为 (B1, ..., Bn), C, H, W
  std::vector<int64_t> output_sizes(self.sizes().begin(), self.sizes().end() - 3);
  output_sizes.insert(output_sizes.end(),
      {self.size(-3) / upscale_factor / upscale_factor,
       self.size(-2) * upscale_factor,
       self.size(-1) * upscale_factor});

  // 创建一个空的输出张量，使用与输入张量相同的选项
  auto output = at::empty({0}, self.options());
  // 建议的内存格式
  auto memory_format = self.suggest_memory_format();
  // 调整输出张量的大小和内存格式
  output.resize_(output_sizes, memory_format);

  // 如果输出张量元素数量为零，直接返回空张量
  if (output.numel() == 0) {
    return output;
  }

  // 使输入张量连续，并按建议的内存格式复制
  auto input = self.contiguous(memory_format);

  // 调用像素重排的内核函数进行计算
  pixel_shuffle_kernel(kCPU, output, input, upscale_factor);
  // 返回计算结果的输出张量
  return output;
}

// 实现 CPU 端的像素逆重排操作
Tensor pixel_unshuffle_cpu(const Tensor& self, int64_t downscale_factor) {
  // 检查输入张量和缩小因子的形状是否符合像素逆重排的要求
  check_pixel_unshuffle_shapes(self, downscale_factor);

  // 如果输入张量元素数量为零，直接返回其克隆
  if (self.numel() == 0) {
    return self.clone();
  }

  // 构建输出张量的尺寸，格式为 (B1, ..., Bn), C, H, W
  std::vector<int64_t> output_sizes(self.sizes().begin(), self.sizes().end() - 3);
  output_sizes.insert(output_sizes.end(),
      {self.size(-3) * downscale_factor * downscale_factor,
       self.size(-2) / downscale_factor,
       self.size(-1) / downscale_factor});

  // 创建一个空的输出张量，使用与输入张量相同的选项
  auto output = at::empty({0}, self.options());
  // 建议的内存格式
  auto memory_format = self.suggest_memory_format();
  // 调整输出张量的大小和内存格式
  output.resize_(output_sizes, memory_format);

  // 如果输出张量元素数量为零，直接返回空张量
  if (output.numel() == 0) {
    return output;
  }

  // 使输入张量连续，并按建议的内存格式复制
  auto input = self.contiguous(memory_format);

  // 调用像素逆重排的内核函数进行计算
  pixel_unshuffle_kernel(kCPU, output, input, downscale_factor);
  // 返回计算结果的输出张量
  return output;
}
// 实现像素混洗操作，将输入张量按指定的放大因子进行重组
Tensor math_pixel_shuffle(const Tensor& self, int64_t upscale_factor) {
  // 检查输入张量和放大因子的形状是否符合要求
  check_pixel_shuffle_shapes(self, upscale_factor);

  // 获取输入张量的通道数（C）、高度（H）、宽度（W）
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  const auto NUM_NON_BATCH_DIMS = 3;
  // self.sizes().end() - NUM_NON_BATCH_DIMS 表示除了批处理维度外的维度范围
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  // 计算放大因子的平方
  int64_t upscale_factor_squared = upscale_factor * upscale_factor;
  // 计算输出张量的通道数（oc）、高度（oh）、宽度（ow）
  int64_t oc = c / upscale_factor_squared;
  int64_t oh = h * upscale_factor;
  int64_t ow = w * upscale_factor;

  // 将输入张量重塑，将通道维度从 c 拆分为三个单独的维度：(oc, upscale_factor, upscale_factor, h, w)
  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {oc, upscale_factor, upscale_factor, h, w});
  const auto input_reshaped = self.reshape(added_dims_shape);

  // 通过重新排列维度，按照给定的顺序将新的 upscale_factor 维度与高度和宽度维度混洗
  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  // 使用 std::iota 保持批处理维度在排列中的顺序
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5 /* oc */, -2 /* h */, -4 /* 1st upscale_factor */, -1 /* w */,
                                         -3 /* 2nd upscale_factor */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // 最后，通过将 (h, upscale_factor) 合并为一个维度 (oh)，将 (w, upscale_factor) 合并为一个维度 (ow)，进行放大
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  // pixel_shuffle 期望 *永远* 不返回输入的别名
  // 使用 clone(at::MemoryFormat::Contiguous) 确保返回一个连续内存格式的副本，并按最终形状视图化
  return input_permuted.clone(at::MemoryFormat::Contiguous).view(final_shape);
}
// 定义函数 `math_pixel_unshuffle`，对输入张量进行像素反洗牌操作
Tensor math_pixel_unshuffle(const Tensor& self, int64_t downscale_factor) {
  // 检查输入张量和缩放因子的形状是否满足要求
  check_pixel_unshuffle_shapes(self, downscale_factor);

  // 获取输入张量的通道数（C）、高度（H）、宽度（W）
  int64_t c = self.size(-3);
  int64_t h = self.size(-2);
  int64_t w = self.size(-1);
  // 常量定义：非批量维度数为3
  constexpr auto NUM_NON_BATCH_DIMS = 3;
  // 计算出不包括批量维度在内的张量大小的末尾迭代器
  const auto self_sizes_batch_end = self.sizes().end() - NUM_NON_BATCH_DIMS;

  // 计算缩放因子的平方
  int64_t downscale_factor_squared = downscale_factor * downscale_factor;
  // 计算输出张量的通道数（oc）、高度（oh）、宽度（ow）
  int64_t oc = c * downscale_factor_squared;
  int64_t oh = h / downscale_factor;
  int64_t ow = w / downscale_factor;

  // 首先，重塑输入张量，将高度分解为（oh, downscale_factor）维度，宽度分解为（ow, downscale_factor）维度，
  // 为后续的反洗牌操作做准备
  std::vector<int64_t> added_dims_shape(
      self.sizes().begin(), self_sizes_batch_end);
  added_dims_shape.insert(
      added_dims_shape.end(), {c, oh, downscale_factor, ow, downscale_factor});
  const auto input_reshaped = self.reshape(added_dims_shape);

  // 接下来，通过重新排列维度来执行像素反洗牌操作，将缩放因子维度与通道维度一起排列
  std::vector<int64_t> permutation(self.sizes().begin(), self_sizes_batch_end);
  // 使用 std::iota 以保持批量维度在排列中的顺序
  std::iota(permutation.begin(), permutation.end(), 0);
  permutation.insert(permutation.end(), {-5 /* c */, -3 /* 1st downscale_factor */, -1 /* 2nd downscale_factor */,
                                         -4 /* oh */, -2 /* ow */});
  const auto input_permuted = input_reshaped.permute(permutation);

  // 最后，通过折叠（c, downscale_factor, downscale_factor）维度到单一维度（oc）来执行缩放操作，
  // 得到的输出张量高度为oh，宽度为ow
  std::vector<int64_t> final_shape(self.sizes().begin(), self_sizes_batch_end);
  final_shape.insert(final_shape.end(), {oc, oh, ow});

  // pixel_unshuffle 函数期望 *永远* 不返回输入的别名
  // 使用 contiguous 内存格式对结果进行克隆，并按最终形状进行视图变换后返回
  return input_permuted.clone(at::MemoryFormat::Contiguous).view(final_shape);
}

// 定义像素洗牌操作的分发函数和反洗牌操作的分发函数
DEFINE_DISPATCH(pixel_shuffle_kernel);
DEFINE_DISPATCH(pixel_unshuffle_kernel);

} // namespace at::native
```