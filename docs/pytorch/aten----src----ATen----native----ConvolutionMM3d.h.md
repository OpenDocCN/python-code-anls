# `.\pytorch\aten\src\ATen\native\ConvolutionMM3d.h`

```
// 包含 ATen 库中的 Tensor 类的定义文件
#include <ATen/core/Tensor.h>

// 定义了 at::native 命名空间，包含了本地的实现
namespace at::native {

// 定义了一个函数 slow_conv3d_backward_cpu，用于计算 3D 卷积反向传播的梯度
std::tuple<Tensor, Tensor, Tensor> slow_conv3d_backward_cpu(
    // 输入参数 grad_output，表示输出梯度的张量
    const Tensor& grad_output,
    // 输入参数 self，表示输入张量
    const Tensor& self,
    // 输入参数 weight，表示卷积核张量
    const Tensor& weight,
    // 输入参数 kernel_size，表示卷积核大小的数组
    IntArrayRef kernel_size,
    // 输入参数 stride，表示卷积操作的步长的数组
    IntArrayRef stride,
    // 输入参数 padding，表示卷积操作的填充的数组
    IntArrayRef padding,
    // 输入参数 output_mask，表示掩码数组，用于指定输出的哪些部分需要计算
    std::array<bool, 3> output_mask);

} // namespace at::native
```