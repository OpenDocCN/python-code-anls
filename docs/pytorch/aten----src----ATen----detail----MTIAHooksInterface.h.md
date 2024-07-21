# `.\pytorch\aten\src\ATen\detail\MTIAHooksInterface.h`

```
#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <c10/core/Stream.h>
#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

#include <string>

// 声明名为 `at` 的命名空间，其中包含用于处理 PyTorch 上下文的类
namespace at {
class Context;
}

namespace at {

// 字符串常量，用于指示 MTIA 后端的错误信息
constexpr const char* MTIA_HELP =
    "The MTIA backend requires MTIA extension for PyTorch;"
    "this error has occurred because you are trying "
    "to use some MTIA's functionality without MTIA extension included.";

// MTIAHooksInterface 类，继承自 AcceleratorHooksInterface 接口
struct TORCH_API MTIAHooksInterface : AcceleratorHooksInterface {
// 如果调用 MTIAHooksInterface 的函数但没有加载 MTIA 后端，则会失败
#define FAIL_MTIAHOOKS_FUNC(func) \
  TORCH_CHECK(false, "Cannot execute ", func, "() without MTIA backend.");

  // 析构函数，使用默认行为
  ~MTIAHooksInterface() override = default;

  // 初始化 MTIA 后端，如果未动态加载 MTIA 扩展，则此函数不执行任何操作
  virtual void initMTIA() const {
    // 避免在此处记录日志，因为 MTIA 需要先初始化设备，然后它会知道有多少可用设备。如果未动态加载 mtia 扩展，则将其设置为无操作。
    return;
  }

  // 检查当前环境是否具有 MTIA 后端
  virtual bool hasMTIA() const {
    return false;
  }

  // 返回设备数量，始终返回 0
  DeviceIndex deviceCount() const override {
    return 0;
  }

  // 同步特定设备的操作，如果未加载 MTIA 后端，则调用此函数会导致失败
  virtual void deviceSynchronize(c10::DeviceIndex device_index) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  // 返回配置信息的字符串表示，如果未加载 MTIA 后端，则调用此函数会导致失败
  virtual std::string showConfig() const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  // 检查特定设备是否具有主上下文，始终返回 false
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    return false;
  }

  // 设置当前设备，如果未加载 MTIA 后端，则调用此函数会导致失败
  void setCurrentDevice(DeviceIndex device) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }

  // 返回当前设备的索引，如果未加载 MTIA 后端，则调用此函数会导致失败
  DeviceIndex getCurrentDevice() const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  // 交换设备，如果未加载 MTIA 后端，则调用此函数会导致失败
  DeviceIndex exchangeDevice(DeviceIndex device) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  // 可能交换设备，如果未加载 MTIA 后端，则调用此函数会导致失败
  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return -1;
  }

  // 返回当前设备的默认流，如果未加载 MTIA 后端，则调用此函数会导致失败
  virtual c10::Stream getCurrentStream(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

  // 返回默认流，如果未加载 MTIA 后端，则调用此函数会导致失败
  virtual c10::Stream getDefaultStream(DeviceIndex device) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
    return c10::Stream::unpack3(-1, 0, c10::DeviceType::MTIA);
  }

  // 设置当前流，如果未加载 MTIA 后端，则调用此函数会导致失败
  virtual void setCurrentStream(const c10::Stream& stream) const {
    FAIL_MTIAHOOKS_FUNC(__func__);
  }
};

// MTIAHooksArgs 结构体，目前为空
struct TORCH_API MTIAHooksArgs {};

// 定义 MTIAHooksRegistry，用于注册 MTIAHooksInterface 的具体实现
C10_DECLARE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs);

// 宏，用于注册 MTIAHooksInterface 的具体实现
#define REGISTER_MTIA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MTIAHooksRegistry, clsname, clsname)

namespace detail {
// 返回全局 MTIAHooksInterface 的引用
TORCH_API const MTIAHooksInterface& getMTIAHooks();
// 检查 MTIAHooks 是否已经构建
TORCH_API bool isMTIAHooksBuilt();
} // namespace detail
} // namespace at
```