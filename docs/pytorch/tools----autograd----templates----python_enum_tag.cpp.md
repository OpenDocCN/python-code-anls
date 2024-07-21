# `.\pytorch\tools\autograd\templates\python_enum_tag.cpp`

```py
#include <torch/csrc/autograd/python_enum_tag.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <ATen/core/enum_tag.h>

namespace py = pybind11;  // 导入 pybind11 库，并为其命名空间取别名为 py
namespace torch {         // 定义 torch 命名空间
    namespace autograd {  // 定义 autograd 命名空间，位于 torch 内部
        // 定义函数 initEnumTag，初始化给定的 Python 模块，使其包含 Enum 类型的 Tag
        void initEnumTag(PyObject* module) {
            auto m = py::handle(module).cast<py::module>();  // 将传入的 Python 对象转换为 py::module 对象
            // 创建名为 "Tag" 的枚举类型，并将其绑定到 Python 模块 m 上
            py::enum_<at::Tag>(m, "Tag")
            ${enum_of_valid_tags};  // 插入有效标签的枚举定义，可能是一个代码生成的部分
            // 设置 Python 模块的文档字符串
            m.doc() = "An Enum that contains tags that can be assigned to an operator registered in C++.";
        }
    }
}
```