# `.\pytorch\torch\csrc\utils\tensor_memoryformats.cpp`

```py
#include <torch/csrc/utils/tensor_memoryformats.h>

#include <c10/core/MemoryFormat.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/MemoryFormat.h>

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::utils {

namespace {
// 存储各种内存格式对应的 Python 对象的注册表，使用静态大小的数组
std::array<PyObject*, static_cast<int>(at::MemoryFormat::NumOptions)>
    memory_format_registry = {};
} // anonymous namespace

// 根据内存格式枚举值获取对应的 Python 内存格式对象
PyObject* getTHPMemoryFormat(at::MemoryFormat memory_format) {
  auto py_memory_format =
      memory_format_registry[static_cast<int>(memory_format)];
  // 如果找不到对应的 Python 对象，抛出无效参数异常
  if (!py_memory_format) {
    throw std::invalid_argument("unsupported memory_format");
  }
  return py_memory_format;
}

// 初始化内存格式对象及其注册
void initializeMemoryFormats() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  // 导入 torch 模块失败，抛出 Python 错误
  if (!torch_module) {
    throw python_error();
  }

  // Lambda 函数，用于添加内存格式对象到 torch 模块
  auto add_memory_format = [&](at::MemoryFormat format, const char* name) {
    // 构建内存格式对象的名称
    std::string module_name = "torch.";
    // 创建新的 Python 内存格式对象
    PyObject* memory_format = THPMemoryFormat_New(format, module_name + name);
    // 增加 Python 对象的引用计数
    Py_INCREF(memory_format);
    // 将内存格式对象添加到 torch 模块中
    if (PyModule_AddObject(torch_module, name, memory_format) != 0) {
      Py_DECREF(memory_format);
      // 添加失败，抛出 Python 错误
      throw python_error();
    }
    // 成功添加后，再次增加 Python 对象的引用计数并存储到注册表中
    Py_INCREF(memory_format);
    memory_format_registry[static_cast<size_t>(format)] = memory_format;
  };

  // 添加常见的内存格式对象到 torch 模块中
  add_memory_format(at::MemoryFormat::Preserve, "preserve_format");
  add_memory_format(at::MemoryFormat::Contiguous, "contiguous_format");
  add_memory_format(at::MemoryFormat::ChannelsLast, "channels_last");
  add_memory_format(at::MemoryFormat::ChannelsLast3d, "channels_last_3d");
}

} // namespace torch::utils
```