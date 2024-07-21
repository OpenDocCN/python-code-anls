# `.\pytorch\aten\src\ATen\detail\MetaGuardImpl.cpp`

```py
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace at::detail {

// 在 at::detail 命名空间内注册设备守卫实现，使用 NoOpDeviceGuardImpl 类处理 Meta 设备类型
C10_REGISTER_GUARD_IMPL(Meta, c10::impl::NoOpDeviceGuardImpl<DeviceType::Meta>);

} // namespace at::detail
```