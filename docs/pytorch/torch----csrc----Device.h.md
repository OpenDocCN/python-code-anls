# `.\pytorch\torch\csrc\Device.h`

```py
#pragma once
// 预处理命令，确保本头文件只被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 的导出宏定义

#include <torch/csrc/python_headers.h>
// 引入 Torch 使用的 Python 头文件

#include <ATen/Device.h>
// 引入 ATen 库中的 Device 类定义

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 禁止 lint 工具对下一行的类型成员初始化进行检查

struct TORCH_API THPDevice {
  // 定义 THPDevice 结构体，包含一个 at::Device 对象
  PyObject_HEAD 
  at::Device device;
};

// 声明 THPDeviceType 的外部 PyTypeObject
TORCH_API extern PyTypeObject THPDeviceType;

// 定义 THPDevice_Check 函数，用于检查对象是否为 THPDevice 类型
inline bool THPDevice_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPDeviceType;
}

// 声明 THPDevice_New 函数，用于创建 THPDevice 对象
TORCH_API PyObject* THPDevice_New(const at::Device& device);

// 声明 THPDevice_init 函数，用于初始化 THPDevice 模块
TORCH_API void THPDevice_init(PyObject* module);
```