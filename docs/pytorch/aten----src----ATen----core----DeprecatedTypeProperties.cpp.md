# `.\pytorch\aten\src\ATen\core\DeprecatedTypeProperties.cpp`

```
#include <ATen/core/DeprecatedTypeProperties.h>  // 包含 DeprecatedTypeProperties 类的头文件

#include <ATen/core/LegacyTypeDispatch.h>  // 包含 LegacyTypeDispatch 类的头文件
#include <ATen/core/Tensor.h>  // 包含 Tensor 类的头文件
#include <ATen/core/UnsafeFromTH.h>  // 包含 unsafeTensorFromTH 和 unsafeStorageFromTH 函数的头文件

namespace at {

Tensor DeprecatedTypeProperties::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  // 调用全局函数 unsafeTensorFromTH 处理传入的 TH 指针，返回一个 Tensor 对象
  return at::unsafeTensorFromTH(th_pointer, retain);
}

Storage DeprecatedTypeProperties::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  // 调用全局函数 unsafeStorageFromTH 处理传入的 TH 指针，返回一个 Storage 对象
  return at::unsafeStorageFromTH(th_pointer, retain);
}

Tensor DeprecatedTypeProperties::copy(const Tensor & src, bool non_blocking, std::optional<Device> to_device) const {
  if (to_device) {
    // 如果提供了目标设备 to_device，则将源张量 src 转换到指定设备，并复制数据
    return src.to(src.options().dtype(scalarType()).device(to_device), non_blocking, /*copy=*/true);
  }
  // 否则，将源张量 src 转换为与当前类型相同的张量，并复制数据
  return src.to(src.options().dtype(scalarType()), non_blocking, /*copy=*/true);
}

} // namespace at  // 命名空间结束注释
```