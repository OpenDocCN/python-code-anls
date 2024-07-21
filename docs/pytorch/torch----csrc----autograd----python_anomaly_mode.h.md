# `.\pytorch\torch\csrc\autograd\python_anomaly_mode.h`

```
#pragma once

#include <pybind11/pybind11.h>  // 包含 pybind11 库的头文件
#include <torch/csrc/autograd/anomaly_mode.h>  // 包含异常模式相关的头文件
#include <torch/csrc/python_headers.h>  // 包含与 Python 交互所需的头文件
#include <torch/csrc/utils/pybind.h>  // 包含与 PyTorch Python 绑定相关的实用函数和类

namespace torch {
namespace autograd {

struct PyAnomalyMetadata : public AnomalyMetadata {
  static constexpr const char* ANOMALY_TRACE_KEY = "traceback_";  // 定义异常追踪信息的键名
  static constexpr const char* ANOMALY_PARENT_KEY = "parent_";  // 定义异常父节点信息的键名

  PyAnomalyMetadata() {
    pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁（GIL）
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    dict_ = PyDict_New();  // 创建一个新的 Python 字典对象，用于存储元数据
  }
  ~PyAnomalyMetadata() override {
    // 如果 Python 已经终止，释放包装的 Python 对象，否则不进行释放以避免潜在的崩溃
    if (Py_IsInitialized()) {
      pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁（GIL）
      Py_DECREF(dict_);  // 减少字典对象的引用计数，释放其内存
    }
  }
  void store_stack() override;  // 存储调用栈信息的虚函数声明
  void print_stack(const std::string& current_node_name) override;  // 打印调用栈信息的虚函数声明
  void assign_parent(const std::shared_ptr<Node>& parent_node) override;  // 分配父节点的虚函数声明

  PyObject* dict() {  // 返回包含元数据的 Python 字典对象的函数
    return dict_;
  }

 private:
  PyObject* dict_{nullptr};  // 私有成员变量，用于存储 Python 字典对象的指针，初始化为 nullptr
};

void _print_stack(
    PyObject* trace_stack,
    const std::string& current_node_name,
    bool is_parent);  // 打印调用栈信息的函数声明

} // namespace autograd
} // namespace torch
```