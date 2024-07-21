# `.\pytorch\torch\csrc\mtia\Module.cpp`

```py
// 引入 Torch 的 ATen 库
#include <ATen/ATen.h>
// 引入 C10 库中的 CallOnce 工具
#include <c10/util/CallOnce.h>
// 引入 Torch 的 Generator 类
#include <torch/csrc/Generator.h>
// 引入 Torch 的 Stream 类
#include <torch/csrc/Stream.h>
// 引入 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>
// 引入 Torch 的设备延迟初始化工具
#include <torch/csrc/utils/device_lazy_init.h>
// 引入 Torch 的 Pybind 封装工具
#include <torch/csrc/utils/pybind.h>

// 引入 C10 核心中的 DeviceType 类
#include <c10/core/DeviceType.h>
// 引入 C10 核心中的 Stream 类
#include <c10/core/Stream.h>
// 在非 Windows 系统下引入 pthread.h 头文件
#ifndef WIN32
#include <pthread.h>
#endif

namespace torch {
namespace mtia {

// 标志在 mtia 初始化后 fork 出的子进程中为 true
static bool in_bad_fork = false;

#ifndef WIN32
// 如果 mtia 已经初始化，fork 出的子进程调用此函数
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_device_init(at::kMTIA, true);
}
#endif

// 在首次调用 mtia 之前应调用此函数
// 注意：这与 initExtension 不同，因为 stub mtia 实现有一些工作函数（如 device_count），但不能完全初始化
static void poison_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  // 在多线程环境中，只调用一次 pthread_atfork，注册 forked_child 函数
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

// 初始化模块，将函数注册到 Python 模块中
void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // 定义 _mtia_init 函数，初始化 MTIA，确保不在 bad fork 中
  m.def("_mtia_init", []() {
    TORCH_INTERNAL_ASSERT(!in_bad_fork); // 在 Python 层处理
    poison_fork();
    at::globalContext().lazyInitMTIA();
  });

  // 定义 _mtia_isBuilt 函数，检查是否已注册 MTIAHooks 类到注册表中
  m.def("_mtia_isBuilt", []() {
    return at::detail::isMTIAHooksBuilt();
  });

  // 定义 _mtia_isInBadFork 函数，返回 in_bad_fork 标志
  m.def("_mtia_isInBadFork", []() { return in_bad_fork; });

  // 定义 _mtia_getCurrentStream 函数，获取当前设备上的当前流
  m.def("_mtia_getCurrentStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentStream(device_index);
  });

  // 定义 _mtia_deviceSynchronize 函数，同步当前设备上的所有流
  m.def("_mtia_deviceSynchronize", []() {
    torch::utils::device_lazy_init(at::kMTIA);
    at::detail::getMTIAHooks().deviceSynchronize(
        at::detail::getMTIAHooks().getCurrentDevice());
  });

  // 定义 _mtia_getDefaultStream 函数，获取当前设备上的默认流
  m.def("_mtia_getDefaultStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getDefaultStream(device_index);
  });

  // 定义 _mtia_setCurrentStream 函数，设置当前设备上的当前流
  m.def("_mtia_setCurrentStream", [](const c10::Stream& stream) {
    torch::utils::device_lazy_init(at::kMTIA);
    auto device = at::detail::getMTIAHooks().getCurrentDevice();
    if (device != stream.device_index()) {
      at::detail::getMTIAHooks().setCurrentDevice(stream.device_index());
    }
    at::detail::getMTIAHooks().setCurrentStream(stream);
  });
}

} // namespace mtia
} // namespace torch
```