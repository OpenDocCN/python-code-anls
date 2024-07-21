# `.\pytorch\torch\csrc\api\src\xpu.cpp`

```
/// 包含 ATen 库的 Context 头文件
/// 包含 torch/xpu 头文件
#include <ATen/Context.h>
#include <torch/xpu.h>

/// 定义 torch::xpu 命名空间
namespace torch::xpu {

/// 返回当前系统中的 XPU 设备数量
size_t device_count() {
  return at::detail::getXPUHooks().getNumGPUs();
}

/// 检查系统中是否存在 XPU 设备
bool is_available() {
  return xpu::device_count() > 0;
}

/// 设置随机数生成器的种子
void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto index = at::detail::getXPUHooks().current_device();
    auto gen = at::detail::getXPUHooks().getDefaultXPUGenerator(index);
    {
      // See Note [Acquire lock when using random generators]
      // 使用随机生成器时需获取锁
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// 设置所有可用 XPU 设备的随机数生成器种子
void manual_seed_all(uint64_t seed) {
  auto num_gpu = device_count();
  for (const auto i : c10::irange(num_gpu)) {
    auto gen = at::detail::getXPUHooks().getDefaultXPUGenerator(i);
    {
      // See Note [Acquire lock when using random generators]
      // 使用随机生成器时需获取锁
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// 同步指定 XPU 设备上的操作
void synchronize(int64_t device_index) {
  // 检查是否有可用的 XPU 设备
  TORCH_CHECK(is_available(), "No XPU are available");
  // 执行 XPU 设备的同步操作
  at::detail::getXPUHooks().deviceSynchronize(
      static_cast<c10::DeviceIndex>(device_index));
}

} // namespace torch::xpu
```