# `.\pytorch\aten\src\ATen\core\op_registration\adaption.h`

```
#pragma once
// 检查和更新共同设备的函数。该函数接收一个可选的共同设备引用、张量列表、调用方法和参数名称。
inline void check_and_update_common_device(optional<Device>& common_device, const List<optional<at::Tensor>>& tensors, at::CheckedFrom methodName, at::CheckedFrom argName) {
  // 遍历张量列表中的每一个张量
  for (const auto& tensor : tensors) {
    // 对每个张量调用检查和更新共同设备的函数，更新共同设备引用
    check_and_update_common_device(common_device, tensor, methodName, argName);
  }
}
// 命名空间结尾：impl
} // namespace impl
// 命名空间结尾：c10
} // namespace c10
```