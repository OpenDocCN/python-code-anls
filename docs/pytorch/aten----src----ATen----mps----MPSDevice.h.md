# `.\pytorch\aten\src\ATen\mps\MPSDevice.h`

```
//  Copyright © 2022 Apple Inc.
// 版权声明，指明此代码的版权归属于苹果公司，年份为2022年

#pragma once
// 预处理命令，确保此头文件只被编译一次

#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
// 包含其他头文件，提供了所需的函数和类型声明

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
// 如果是 Objective-C 环境，包含相关 Objective-C 框架头文件
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLLibrary> MTLLibrary_t;
// 定义 Objective-C 对象类型别名
#else
typedef void* MTLDevice;
typedef void* MTLDevice_t;
typedef void* MTLLibrary_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLLibrary_t;
// 如果不是 Objective-C 环境，定义为空指针类型别名
#endif

namespace at::mps {

// Helper enum to check if a MPSGraph op is supported in a given macOS version
// 枚举类型，用于检查 MPSGraph 操作在特定 macOS 版本中的支持情况
enum class MacOSVersion : uint32_t {
  MACOS_VER_13_0_PLUS = 0,
  MACOS_VER_13_1_PLUS,
  MACOS_VER_13_2_PLUS,
  MACOS_VER_13_3_PLUS,
  MACOS_VER_14_0_PLUS,
};

//-----------------------------------------------------------------
//  MPSDevice
//
// MPSDevice is a singleton class that returns the default device
// MPSDevice 是返回默认设备的单例类
//-----------------------------------------------------------------

class TORCH_API MPSDevice {
 public:
  /**
   * MPSDevice should not be cloneable.
   */
  // MPSDevice 不可克隆的构造函数声明
  MPSDevice(MPSDevice& other) = delete;
  /**
   * MPSDevice should not be assignable.
   */
  // MPSDevice 不可赋值的成员函数声明
  void operator=(const MPSDevice&) = delete;
  /**
   * Gets single instance of the Device.
   */
  // 获取设备的单例实例函数声明
  static MPSDevice* getInstance();
  /**
   * Returns the single device.
   */
  // 返回单一设备的函数声明
  MTLDevice_t device() {
    return _mtl_device;
  }
  /**
   * Returns whether running on Ventura or newer
   */
  // 返回当前运行的 macOS 版本是否大于指定版本的函数声明
  bool isMacOS13Plus(MacOSVersion version) const;

  // 获取 Metal 计算管线状态对象的函数声明
  MTLComputePipelineState_t metalIndexingPSO(const std::string &kernel);
  // 获取 Metal 库对象的函数声明
  MTLLibrary_t getMetalIndexingLibrary();

  // MPSDevice 类的析构函数声明
  ~MPSDevice();

 private:
  // MPSDevice 的单例实例指针声明
  static MPSDevice* _device;
  // Metal 设备对象声明
  MTLDevice_t _mtl_device;
  // Metal 库对象声明
  MTLLibrary_t _mtl_indexing_library;

  // MPSDevice 的私有构造函数声明
  MPSDevice();
};

// 判断当前设备是否可用的函数声明
TORCH_API bool is_available();
// 判断当前 macOS 版本是否为 macOS 13 或更新的函数声明
TORCH_API bool is_macos_13_or_newer(MacOSVersion version = MacOSVersion::MACOS_VER_13_0_PLUS);
// 获取 MPS 分配器的函数声明
TORCH_API at::Allocator* GetMPSAllocator(bool useSharedAllocator = false);

} // namespace at::mps
// 命名空间结束声明
```