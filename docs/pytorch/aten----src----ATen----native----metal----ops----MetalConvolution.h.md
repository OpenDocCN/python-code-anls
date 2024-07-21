# `.\pytorch\aten\src\ATen\native\metal\ops\MetalConvolution.h`

```py
#import <ATen/native/metal/MetalConvParams.h>
#import <ATen/native/metal/MetalPrepackOpContext.h>
#include <c10/util/ArrayRef.h>

// 命名空间 at::native::metal 包含了使用 Metal 加速的相关函数和数据结构定义

namespace at::native::metal {

// 定义了一个名为 conv2d 的函数，用于执行 2D 卷积操作
Tensor conv2d(
    const Tensor& input,                    // 输入张量
    const Tensor& weight,                   // 卷积核张量
    const std::optional<at::Tensor>& bias,  // 可选的偏置张量
    IntArrayRef stride,                     // 步幅数组
    IntArrayRef padding,                    // 填充数组
    IntArrayRef dilation,                   // 膨胀数组
    int64_t groups);                        // 组数

// 命名空间 prepack 包含了预打包操作的相关函数和数据结构定义
namespace prepack {

// 定义了一个名为 conv2d 的函数，用于使用预打包上下文执行 2D 卷积操作
Tensor conv2d(const Tensor& input, Conv2dOpContext& context);

}

} // namespace at::native::metal
```