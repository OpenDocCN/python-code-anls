# `.\pytorch\torch\csrc\profiler\stubs\itt.cpp`

```
// 包含头文件：用于字符串流操作
#include <sstream>

// 包含头文件：用于C++ 10范围操作的实用工具
#include <c10/util/irange.h>

// 包含头文件：用于性能分析器的ITT包装器
#include <torch/csrc/itt_wrapper.h>

// 包含头文件：用于性能分析器存根基类的定义
#include <torch/csrc/profiler/stubs/base.h>

// 命名空间开始：定义了名为torch的命名空间
namespace torch {

// 命名空间开始：定义了名为profiler的命名空间
namespace profiler {

// 命名空间开始：定义了名为impl的命名空间
namespace impl {

// 命名空间开始：定义了未命名的匿名命名空间
namespace {

// 结构体定义：继承自ProfilerStubs，实现了性能分析器的ITT方法
struct ITTMethods : public ProfilerStubs {

  // 方法定义：记录事件，但未实际实现功能
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {}

  // 方法定义：计算两个事件之间的时间间隔，始终返回0
  float elapsed(
      const ProfilerVoidEventStub* event,
      const ProfilerVoidEventStub* event2) const override {
    return 0;
  }

  // 方法定义：使用ITT标记当前事件
  void mark(const char* name) const override {
    torch::profiler::itt_mark(name);
  }

  // 方法定义：使用ITT推入新的时间范围
  void rangePush(const char* name) const override {
    torch::profiler::itt_range_push(name);
  }

  // 方法定义：使用ITT弹出当前时间范围
  void rangePop() const override {
    torch::profiler::itt_range_pop();
  }

  // 方法定义：对每个设备执行指定操作，但未实际实现功能
  void onEachDevice(std::function<void(int)> op) const override {}

  // 方法定义：同步操作，但未实际实现功能
  void synchronize() const override {}

  // 方法定义：返回性能分析器是否启用，始终返回true
  bool enabled() const override {
    return true;
  }
};

// 结构体实例化：静态注册ITTMethods对象
struct RegisterITTMethods {
  RegisterITTMethods() {
    static ITTMethods methods;
    registerITTMethods(&methods);
  }
};
// 实例化RegisterITTMethods对象，完成注册
RegisterITTMethods reg;

} // namespace
} // namespace impl
} // namespace profiler
} // namespace torch
// 命名空间结束
```