# `.\pytorch\aten\src\ATen\native\xnnpack\AveragePooling.cpp`

```
#ifdef USE_XNNPACK
// 如果定义了 USE_XNNPACK 宏，则包含以下头文件和命名空间声明

#include <ATen/native/utils/Factory.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/xnnpack/Pooling.h>

// 进入 ATen 库的 native::xnnpack 命名空间
namespace at::native::xnnpack {

// 定义一个函数 use_global_average_pool，用于检查是否可以使用 XNNPACK 进行全局平均池化操作
bool use_global_average_pool(const Tensor& input) {
  // 返回条件：XNNPACK 库可用、输入张量至少一维、在 CPU 设备上、数据类型为 float32、不需要梯度
  return xnnpack::available() && (1 <= input.ndimension()) &&
      (input.device().is_cpu()) && (kFloat == input.scalar_type()) &&
      !input.requires_grad() && true;
}

} // namespace at::native::xnnpack
// 结束 native::xnnpack 命名空间

#endif /* USE_XNNPACK */
// 结束 USE_XNNPACK 宏的条件编译块
```