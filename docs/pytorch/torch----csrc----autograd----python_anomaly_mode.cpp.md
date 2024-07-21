# `.\pytorch\torch\csrc\autograd\python_anomaly_mode.cpp`

```
// 包含异常处理、Python绑定等必要头文件
#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_anomaly_mode.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

// 定义命名空间 torch::autograd
namespace torch {
namespace autograd {

// 存储异常追溯信息的类 PyAnomalyMetadata 的 store_stack 方法
void PyAnomalyMetadata::store_stack() {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 导入 torch.fx.traceback 模块
  THPObjectPtr mod(PyImport_ImportModule("torch.fx.traceback"));
  if (!mod) {
    throw python_error();
  }

  // 调用模块的 format_stack 方法，获取异常追踪堆栈信息
  THPObjectPtr list(PyObject_CallMethod(mod.get(), "format_stack", ""));
  if (!list) {
    throw python_error();
  }

  // 将异常追踪堆栈信息存储在 metadata 字典中的 ANOMALY_TRACE_KEY 键下
  if (PyDict_SetItemString(dict(), ANOMALY_TRACE_KEY, list.get())) {
    throw python_error();
  }
}

// 打印异常追踪堆栈信息的方法 print_stack
void PyAnomalyMetadata::print_stack(const std::string& current_node_name) {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 检查 metadata 是否为 Python 字典
  if (!PyDict_Check(dict())) {
    throw std::runtime_error("Anomaly metadata is not a python dictionary.");
  }
  // 获取 metadata 中 ANOMALY_TRACE_KEY 对应的异常追踪堆栈信息
  PyObject* trace_stack = PyDict_GetItemString(dict(), ANOMALY_TRACE_KEY);
  // 调用 _print_stack 方法打印堆栈信息
  _print_stack(trace_stack, current_node_name, false);
  // 获取 metadata 中 ANOMALY_PARENT_KEY 对应的父节点信息
  PyObject* pyparent(PyDict_GetItemString(dict(), ANOMALY_PARENT_KEY));

  // 若 metadata 中没有 "parent_" 键，即该节点为根节点，停止打印追溯信息
  while (pyparent) {
    // 获取父节点的 metadata
    THPObjectPtr parent_metadata(PyObject_GetAttrString(pyparent, "metadata"));
    if (!parent_metadata) {
      throw python_error();
    }
    // 获取父节点的名称
    THPObjectPtr parent_name_pyobj(PyObject_CallMethod(pyparent, "name", ""));
    if (!parent_name_pyobj) {
      throw python_error();
    }
    // 将父节点名称转换为 C++ 字符串
    const char* parent_name_char = PyUnicode_AsUTF8(parent_name_pyobj.get());
    if (!parent_name_char) {
      throw python_error();
    }
    const std::string parent_name(parent_name_char);
    // 获取父节点的异常追踪堆栈信息
    PyObject* parent_stack =
        PyDict_GetItemString(parent_metadata.get(), ANOMALY_TRACE_KEY);
    // 调用 _print_stack 方法打印父节点的堆栈信息
    _print_stack(parent_stack, parent_name, true);
    // 获取父节点的父节点，如果当前节点为根节点，则 pyparent 为 null
    pyparent = PyDict_GetItemString(parent_metadata.get(), ANOMALY_PARENT_KEY);
  }
}

// 将父节点信息分配给 metadata["parent_"] 的方法 assign_parent
void PyAnomalyMetadata::assign_parent(
    const std::shared_ptr<Node>& parent_node) {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 如果 parent_node 为空指针，则直接返回，不进行分配操作
  if (!parent_node)
    return;

  // 将 parent_node 转换为 Python 对象
  THPObjectPtr parent_node_(functionToPyObject(parent_node));
  if (!parent_node_) {
    throw python_error();
  }
  // 将 parent_node_ 分配给 metadata 中的 ANOMALY_PARENT_KEY 键
  if (PyDict_SetItemString(dict(), ANOMALY_PARENT_KEY, parent_node_.get())) {
    throw python_error();
  }
}

// 打印异常追踪堆栈信息的辅助函数 _print_stack
void _print_stack(
    PyObject* stack,
    const std::string& current_node_name,
    bool is_parent) {
  // 如果 stack 为空，直接返回
  if (!stack) {


以上是对 C++ 代码中每行的详细注释，说明了每行代码的作用和功能。
    // 输出警告信息，指示在当前节点名称中检测到错误
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "No forward pass information available. Enable detect anomaly "
        "during forward pass for more information.");
    // 直接返回，结束函数执行
    return;
  }

  // 创建一个空的 Python 字符串对象
  THPObjectPtr empty_string(PyUnicode_FromString(""));
  if (!empty_string) {
    // 如果创建失败，则抛出 Python 异常
    throw python_error();
  }

  // stack 是以 Python 字符串结尾的列表。使用 join 方法将它们连接成一个单独的字符串。
  // 创建包含整个堆栈信息的 Python 字符串对象
  THPObjectPtr msg(PyUnicode_Join(empty_string, stack));
  if (!msg) {
    // 如果创建失败，则抛出 Python 异常
    throw python_error();
  }

  // 如果当前节点不是父节点
  if (!is_parent) {
    // 输出警告信息，包含当前节点名称和前向调用错误的跟踪信息
    TORCH_WARN(
        "Error detected in ",
        current_node_name,
        ". ",
        "Traceback of forward call that caused the error:\n",
        THPUtils_unpackString(msg.get()));
  } else {
    // 如果当前节点是父节点
    // 输出警告信息，包含前一个计算是由当前节点引发的信息，以及相应的前向调用错误跟踪信息
    TORCH_WARN(
        "\n\n",
        "Previous calculation was induced by ",
        current_node_name,
        ". "
        "Traceback of forward call that induced the previous calculation:\n",
        THPUtils_unpackString(msg.get()));
  }
}

// 结束 autograd 命名空间
} // namespace autograd

// 结束 torch 命名空间
} // namespace torch
```