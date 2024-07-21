# `.\pytorch\aten\src\ATen\native\xnnpack\ChannelShuffle.cpp`

```py
#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则包含以下头文件
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/utils/Factory.h>

// 进入 at::native::xnnpack 命名空间
namespace at::native::xnnpack {

// 判断是否可以使用通道重排功能
bool use_channel_shuffle(
    const Tensor& input,    // 输入张量
    const int64_t groups)   // 分组数
{
  using namespace internal;

  // 下面是需要满足的条件才能执行这段代码：
  // * 输入必须是 4 维 CPU 浮点张量，无梯度，并且所有维度必须是正数。
  // * 分组数必须大于 1，并且通道数必须能被分组数整除。
  return xnnpack::available() &&    // 检查 XNNPACK 是否可用
      // Input
      (4 == input.dim()) &&         // 输入张量必须是 4 维
      (input.device().is_cpu()) &&  // 输入张量在 CPU 上
      (kFloat == input.scalar_type()) &&  // 输入张量数据类型为浮点型
      (input.size(Layout::Activation4D::batch) >= 0) &&  // 批次维度大于等于 0
      (input.size(Layout::Activation4D::channels) > 0) &&  // 通道数大于 0
      (input.size(Layout::Activation4D::height) > 0) &&    // 高度大于 0
      (input.size(Layout::Activation4D::width) > 0) &&     // 宽度大于 0
      !input.requires_grad() &&   // 输入张量无需梯度
      // Groups
      groups > 1 &&                // 分组数大于 1
      (0 == input.size(Layout::Activation4D::channels) % groups) &&  // 通道数能被分组数整除
      true;                        // 总是返回 true，表示可以执行通道重排
}

// 通道重排操作的实现
Tensor channel_shuffle(
    const Tensor& input,    // 输入张量
}
// 结束 at::native::xnnpack 命名空间
} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
```