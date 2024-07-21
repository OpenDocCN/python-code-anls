# `.\pytorch\torch\csrc\utils\device_lazy_init.cpp`

```
// 包含 TorchDispatchModeTLS.h 头文件，提供 TorchDispatchModeTLS 类的实现
#include <c10/core/impl/TorchDispatchModeTLS.h>
// 包含 device_lazy_init.h 头文件，提供 device_lazy_init 函数的声明
#include <torch/csrc/utils/device_lazy_init.h>

// 包含 Exceptions.h、python_headers.h、object_ptr.h 头文件，用于异常处理、Python 头文件和智能指针管理
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
// 包含 iostream 头文件，提供标准输入输出流
#include <iostream>

// torch::utils 命名空间，用于包装实用函数和工具
namespace torch::utils {
// 匿名命名空间，定义内部静态数组 is_initialized，用于记录设备初始化状态
namespace {

// 定义静态数组 is_initialized，存储各设备类型的初始化状态，默认为 false
std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> is_initialized{};

} // anonymous namespace

// 函数：判断指定设备类型是否已初始化
bool is_device_initialized(at::DeviceType device_type) {
  // 获取全局解释器锁 GIL，确保线程安全
  pybind11::gil_scoped_acquire g;
  // 返回指定设备类型的初始化状态
  return is_initialized[static_cast<int>(device_type)];
}

// 函数：延迟初始化指定设备类型
void device_lazy_init(at::DeviceType device_type) {
  // 获取全局解释器锁 GIL，确保线程安全
  pybind11::gil_scoped_acquire g;
  // 如果设备已初始化，则直接返回
  if (is_device_initialized(device_type)) {
    return;
  }

  // 尝试获取 TorchDispatchModeTLS 的 FAKE 模式，若获取成功则返回
  auto maybe_mode = c10::impl::TorchDispatchModeTLS::get_mode(
      c10::impl::TorchDispatchModeKey::FAKE);
  if (maybe_mode) {
    return;
  }

  // 构建设备类型对应的模块名称，例如 "torch.CPU"
  std::string module_name = "torch." + at::DeviceTypeName(device_type, true);
  // 导入 Python 模块，若失败则抛出异常
  auto module = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
  if (!module) {
    throw python_error();
  }

  // 对于 PrivateUse1 设备类型，检查是否具有 "_lazy_init" 方法，若无则返回
  if (device_type == at::DeviceType::PrivateUse1) {
    auto has_lazy_init_method =
        PyObject_HasAttrString(module.get(), "_lazy_init") == 1;
    if (!has_lazy_init_method) {
      return;
    }
  }

  // 调用模块的 "_lazy_init" 方法，若调用失败则抛出异常
  auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
  if (!res) {
    throw python_error();
  }

  // 标记设备类型已初始化
  is_initialized[static_cast<int>(device_type)] = true;
}

// 函数：设置指定设备类型是否需要初始化
void set_requires_device_init(at::DeviceType device_type, bool value) {
  // 设置设备类型的初始化状态，取反值存储到 is_initialized 数组中
  is_initialized[static_cast<int>(device_type)] = !value;
}

} // namespace torch::utils
```