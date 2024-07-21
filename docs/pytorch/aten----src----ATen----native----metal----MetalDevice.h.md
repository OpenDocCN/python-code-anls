# `.\pytorch\aten\src\ATen\native\metal\MetalDevice.h`

```py
#ifndef PYTORCH_MOBILE_METAL_DEVICE_H_
#define PYTORCH_MOBILE_METAL_DEVICE_H_

#import <Metal/Metal.h>  // 导入 Metal 库

#include <string>  // 包含字符串库

namespace at::native::metal {

struct MetalDeviceInfo {  // 定义 Metal 设备信息结构体
  std::string name;  // 设备名称
  MTLLanguageVersion languageVersion;  // Metal 语言版本
};

static inline MetalDeviceInfo createDeviceInfo(id<MTLDevice> device) {  // 创建 Metal 设备信息的静态函数
  MetalDeviceInfo device_info;  // 创建 Metal 设备信息对象
  if (device.name != nil) {  // 如果设备名称不为空
    device_info.name = device.name.UTF8String;  // 获取设备名称并转换为 UTF-8 字符串
  }
  // 根据可用的平台版本设置 Metal 语言版本
  if (@available(macOS 11.0, iOS 14.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_3;
  } else if (@available(macOS 10.15, iOS 13.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_2;
  } else if (@available(macOS 10.14, iOS 12.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_1;
  } else if (@available(macOS 10.13, iOS 11.0, *)) {
    device_info.languageVersion = MTLLanguageVersion2_0;
  } else if (@available(macOS 10.12, iOS 10.0, *)) {
    device_info.languageVersion = MTLLanguageVersion1_2;
  } else if (@available(macOS 10.11, iOS 9.0, *)) {
    device_info.languageVersion = MTLLanguageVersion1_1;
  }
#if (                                                    \
    defined(__IPHONE_9_0) &&                             \
    __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_9_0) || \
    (defined(__MAC_10_11) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_11)
#else
#error "Metal is not available on the current platform."  // 如果 Metal 在当前平台不可用，则报错
#endif
  return device_info;  // 返回 Metal 设备信息对象
}

} // namespace at::native::metal

#endif  // 结束 Metal 设备头文件的条件编译指令
```