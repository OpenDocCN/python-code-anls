# `.\pytorch\torch\csrc\distributed\c10d\GlooDeviceFactory.hpp`

```py
#pragma once

// 预处理指令：#pragma once，确保头文件只被包含一次，提高编译效率


#ifdef USE_C10D_GLOO

// 如果定义了宏 USE_C10D_GLOO，则编译以下内容；用于条件编译


#include <string>

// 包含标准库头文件 `<string>`，提供处理字符串的功能


#include <c10/util/Registry.h>
#include <gloo/config.h>
#include <gloo/transport/device.h>

// 包含 C10 库的注册工具 `<c10/util/Registry.h>`，以及 GLOO 库的配置 `<gloo/config.h>` 和传输设备 `<gloo/transport/device.h>` 的头文件


namespace c10d {

// 命名空间 c10d，用于包裹下面的类和函数，避免命名冲突


class TORCH_API GlooDeviceFactory {
 public:
  // Create new device instance for specific interface.
  static std::shared_ptr<::gloo::transport::Device> makeDeviceForInterface(
      const std::string& interface);

  // Create new device instance for specific hostname or address.
  static std::shared_ptr<::gloo::transport::Device> makeDeviceForHostname(
      const std::string& hostname);
};

// 定义类 GlooDeviceFactory，包含两个公共静态函数：
// - makeDeviceForInterface：根据指定接口创建新的设备实例
// - makeDeviceForHostname：根据指定主机名或地址创建新的设备实例


TORCH_DECLARE_SHARED_REGISTRY(
    GlooDeviceRegistry,
    ::gloo::transport::Device,
    const std::string&, /* interface */
    const std::string& /* hostname */);

// 使用宏 TORCH_DECLARE_SHARED_REGISTRY 定义 GlooDeviceRegistry 共享注册表，注册 ::gloo::transport::Device 类型对象，接受两个参数：
// - const std::string&：接口参数
// - const std::string&：主机名参数


} // namespace c10d

// 命名空间 c10d 的结束标记，结束命名空间定义


#endif // USE_C10D_GLOO

// 结束条件编译指令，关闭对 USE_C10D_GLOO 宏的条件判断
```