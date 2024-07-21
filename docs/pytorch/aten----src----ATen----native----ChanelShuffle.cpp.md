# `.\pytorch\aten\src\ATen\native\ChanelShuffle.cpp`

```
// 定义宏 TORCH_ASSERT_ONLY_METHOD_OPERATORS
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 包含 ATen 库中的 NamedTensorUtils.h 头文件
#include <ATen/NamedTensorUtils.h>
// 如果在移动设备上并且使用了 XNNPACK，则包含 XNNPACK 引擎的头文件
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
// 包含 C++ 标准库中的异常处理工具 Exception.h
#include <c10/util/Exception.h>

// 包含 ATen 库中 CPU 相关的 ChannelShuffleKernel 头文件
#include <ATen/native/cpu/ChannelShuffleKernel.h>

// 如果未定义 AT_PER_OPERATOR_HEADERS，则包含以下标准 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
// 否则，包含特定的 channel_shuffle 相关头文件
#else
#include <ATen/ops/channel_shuffle_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/native_channel_shuffle.h>
#include <ATen/ops/native_channel_shuffle_native.h>
#endif

// 命名空间 at::native 中的实现
namespace at::native {

// CPU 版本的通道重排函数 channel_shuffle_cpu
Tensor channel_shuffle_cpu(const Tensor& self, int64_t groups) {
  // 定义输出 Tensor 变量 output
  Tensor output;
  // 如果输入 Tensor self 的元素数为 0，则创建别名 Tensor
  if (self.numel() == 0) {
    output = self.alias();
  } else {
    // 否则，根据建议的内存格式获取 memory_format
    auto memory_format = self.suggest_memory_format();
    // 创建一个空的 Tensor output，维度为 {0}，使用 self 的选项
    output = at::empty({0}, self.options());
    // 调整 output 的大小为 self 的大小，使用 memory_format
    output.resize_(self.sizes(), memory_format);
    // 获取 self 的连续版本 input，使用 memory_format
    auto input = self.contiguous(memory_format);
    // 调用 channel_shuffle_kernel 函数，对 input 进行通道重排，输出到 output
    channel_shuffle_kernel(kCPU, output, input, groups);
  }
  // 如果 self 具有命名，则根据非空输出的命名信息传播名称
  return namedinference::propagate_names_if_nonempty(
      output,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

// 通道重排函数 channel_shuffle
Tensor channel_shuffle(const Tensor& self, int64_t groups) {
  // 检查输入 Tensor self 的维度是否大于 2
  TORCH_CHECK(self.dim() > 2,
              "channel_shuffle expects input with > 2 dims, but got input with sizes ",
              self.sizes());
  // 获取输入 Tensor self 的通道数 c
  int64_t c = self.size(1);
  // 检查 groups 是否为正数
  TORCH_CHECK(groups > 0,
              "Number of groups to divide channels in must be positive.",
              " Value of groups:", groups);
  // 检查通道数 c 是否能被 groups 整除
  TORCH_CHECK((c % groups) == 0,
              "Number of channels must be divisible by groups. Got ",
              c, " channels and ", groups, " groups.");

  // 如果在移动设备上且 self 的内存格式为 ChannelsLast，且可以使用 xnnpack::channel_shuffle 函数
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
  if (self.is_contiguous(MemoryFormat::ChannelsLast) &&
      xnnpack::use_channel_shuffle(self, groups)) {
    // 如果 self 的元素数为 0，则创建别名 Tensor；否则使用 xnnpack::channel_shuffle 函数
    auto output = self.numel() == 0 ? self.alias() : xnnpack::channel_shuffle(self, groups);
    return output;
  }
#endif

  // 否则，如果 self 的元素数为 0，则创建别名 Tensor；否则使用 at::native_channel_shuffle 函数
  auto output = self.numel() == 0 ? self.alias() : at::native_channel_shuffle(self, groups);
  // 如果 self 具有命名，则根据非空输出的命名信息传播名称
  return namedinference::propagate_names_if_nonempty(
      output,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

} // namespace at::native
// 定义一个函数，用于在张量中进行通道重排
Tensor math_channel_shuffle(const Tensor& self, int64_t groups) {
  // 获取张量的批量大小
  int64_t b = self.size(0);
  // 获取张量的通道数
  int64_t c = self.size(1);
  // 计算每个输出通道的大小
  int64_t oc = c / groups;

  // 将输入张量重塑为四维张量，以便进行通道重排操作
  auto input_reshaped = self.view({b, groups, oc, -1});

  // TODO: contiguous 可以保留输入的内存格式。
  // 但由于上述重塑操作可能会破坏高度和宽度维度信息，
  // 在这种情况下，channels_last 的 contiguous 可能会认为 oc 和最后一个维度对应于高度和宽度？
  // 然而，从初步查看来看，这可能不正确。
  // 在这种情况下，如果我们想保留内存顺序，channels_last 可能需要自定义实现。
  // XNNPACK 提供了 NHWC 的通道重排操作，适用于移动端。
  // 对于服务器端，我们将需要进行自定义实现。
  // 对于 ChannelsFirst，即连续内存格式，可能也需要一个快速的自定义实现。

  // 执行张量的通道重排操作，将通道维度置换为指定顺序
  Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
      .contiguous() // 确保张量的内存是连续的
      .reshape(self.sizes()); // 将张量重塑为原始形状

  // 如果输入张量具有命名维度，则将命名信息传播到输出张量中
  return namedinference::propagate_names_if_nonempty(
      output_tensor,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

// 定义通道重排的调度器
DEFINE_DISPATCH(channel_shuffle_kernel);

} // namespace at::native


这段代码实现了一个通道重排的功能，将输入张量的通道重新组织为指定的顺序，并尝试在保持内存格式的同时进行操作。
```