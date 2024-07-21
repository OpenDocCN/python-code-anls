# `.\pytorch\torch\csrc\distributed\rpc\python_resp.cpp`

```
// 包含头文件 torch/csrc/distributed/rpc/python_resp.h
#include <torch/csrc/distributed/rpc/python_resp.h>

// 定义命名空间 torch::distributed::rpc
namespace torch {
namespace distributed {
namespace rpc {

// PythonResp 类的构造函数，接受一个右值引用的 SerializedPyObj 对象
PythonResp::PythonResp(SerializedPyObj&& serializedPyObj)
    : serializedPyObj_(std::move(serializedPyObj)) {}

// PythonResp 类的 toMessageImpl 方法的实现，右值引用版本
c10::intrusive_ptr<Message> PythonResp::toMessageImpl() && {
  // 创建一个包含 serializedPyObj_ payload 数据的 std::vector<char>
  auto payload = std::vector<char>(
      serializedPyObj_.payload_.begin(), serializedPyObj_.payload_.end());
  // 创建并返回一个 MessageType::PYTHON_RET 类型的 Message 对象
  return c10::make_intrusive<Message>(
      std::move(payload),
      std::move(serializedPyObj_.tensors_),
      MessageType::PYTHON_RET);
}

// PythonResp 类的静态方法 fromMessage，从 Message 对象创建 PythonResp 对象的实例
std::unique_ptr<PythonResp> PythonResp::fromMessage(const Message& message) {
  // 从 message 的 payload 创建一个 std::string 对象
  std::string payload(message.payload().begin(), message.payload().end());
  // 从 message 获取 tensors 列表
  std::vector<Tensor> tensors = message.tensors();
  // 使用 payload 和 tensors 创建 SerializedPyObj 对象
  SerializedPyObj serializedPyObj(std::move(payload), std::move(tensors));
  // 创建并返回一个 PythonResp 对象的 std::unique_ptr
  return std::make_unique<PythonResp>(std::move(serializedPyObj));
}

// 返回当前 PythonResp 对象的 serializedPyObj_ 成员变量的常量引用
const SerializedPyObj& PythonResp::serializedPyObj() const {
  return serializedPyObj_;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```