# `.\pytorch\torch\csrc\distributed\rpc\unpickled_python_call.cpp`

```
// 包含 Torch 分布式 RPC 模块中的 Python 反序列化调用的头文件
#include <torch/csrc/distributed/rpc/unpickled_python_call.h>

// 包含 Torch 分布式 RPC 模块中的 Python RPC 处理器的头文件
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

// Torch 的命名空间开始
namespace torch {
namespace distributed {
namespace rpc {

// UnpickledPythonCall 类的构造函数，接受序列化的 Python 对象和是否异步执行的标志
UnpickledPythonCall::UnpickledPythonCall(
    const SerializedPyObj& serializedPyObj,
    bool isAsyncExecution)
    : isAsyncExecution_(isAsyncExecution) {
  // 获取 PythonRpcHandler 的单例实例
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire ag;
  // 使用 PythonRpcHandler 对象对序列化的 Python 对象进行反序列化
  pythonUdf_ = pythonRpcHandler.deserialize(serializedPyObj);
}

// UnpickledPythonCall 类的析构函数
UnpickledPythonCall::~UnpickledPythonCall() {
  // 明确将 PyObject* 设置为 nullptr，防止 py::object 的析构函数再次对 PyObject 进行减引用
  // 参见 python_ivalue.h 中的注释 [Destructing py::object]
  py::gil_scoped_acquire acquire;
  pythonUdf_.dec_ref();
  pythonUdf_.ptr() = nullptr;
}

// 移动语义的 toMessageImpl 方法，由于 UnpickledPythonCall 不支持消息转换，引发断言错误
c10::intrusive_ptr<Message> UnpickledPythonCall::toMessageImpl() && {
  TORCH_INTERNAL_ASSERT(
      false, "UnpickledPythonCall does not support toMessage().");
}

// 返回成员变量 pythonUdf_ 的常引用，表示该对象是一个 Python 函数或方法的 py::object
const py::object& UnpickledPythonCall::pythonUdf() const {
  return pythonUdf_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```