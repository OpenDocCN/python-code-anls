# `.\pytorch\aten\src\ATen\core\adaption.cpp`

```py
#include <ATen/core/op_registration/adaption.h>

// 进入 c10::impl 命名空间
namespace c10::impl {

// 定义 common_device_check_failure 函数，用于检查设备一致性
void common_device_check_failure(Device common_device, const at::Tensor& tensor, at::CheckedFrom methodName, at::CheckedFrom argName) {
  // 使用 TORCH_CHECK 进行断言，若条件为 false，输出以下错误信息
  TORCH_CHECK(false,
    "Expected all tensors to be on the same device, but "
    "found at least two devices, ", common_device, " and ", tensor.device(), "! "
    "(when checking argument for argument ", argName, " in method ", methodName, ")");
}

} // namespace c10::impl
```