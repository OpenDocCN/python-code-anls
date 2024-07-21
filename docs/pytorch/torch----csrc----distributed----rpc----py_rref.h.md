# `.\pytorch\torch\csrc\distributed\rpc\py_rref.h`

```py
#pragma once
// 防止头文件被多次包含的预处理指令

#include <torch/csrc/distributed/rpc/rref_impl.h>
// 引入 RRef 实现相关的头文件

#include <torch/csrc/python_headers.h>
// 引入 Python 头文件

#include <torch/csrc/utils/pybind.h>
// 引入 PyTorch 的 Python 绑定工具

namespace torch {
namespace distributed {
namespace rpc {

enum RRefProxyType { RPC_SYNC, RPC_ASYNC, REMOTE };
// 枚举类型 RRefProxyType，定义了 RRef 的代理类型

// Python 封装的 RRef 共享指针，支持 Python 的 pickle 和 unpickle
class PYBIND11_EXPORT PyRRef {
 public:
  // 构造函数，使用 py::object 和 type_hint 创建 PyRRef 对象
  explicit PyRRef(const py::object& value, const py::object& type_hint);
  // 构造函数，使用 RRef 指针 rref 创建 PyRRef 对象
  explicit PyRRef(c10::intrusive_ptr<RRef> rref);
  // 默认拷贝构造函数
  PyRRef(const PyRRef&) = default;
  // 析构函数
  ~PyRRef();

  // 返回是否是所有者
  bool isOwner() const;
  // 返回是否被所有者确认
  bool confirmedByOwner() const;
  // 返回所有者的 WorkerInfo 对象
  WorkerInfo owner() const;
  // 返回所有者的名称
  std::string ownerName() const;
  // 将 RRef 数据传输到本地
  py::object toHere(const float timeoutSeconds = torch::distributed::rpc::kUnsetRpcTimeout) const;
  // 获取本地的值
  py::object localValue() const;
  // 返回字符串描述
  std::string str() const;
  // 返回 pickle 后的对象
  py::tuple pickle() const;
  // 静态方法，反序列化 PyRRef 对象
  static PyRRef unpickle(const py::tuple& t);
  // 转换为 IValue
  c10::IValue toIValue() const;
  // 获取与此 RRef 在远程端创建相关联的 Future
  c10::intrusive_ptr<JitFuture> getFuture() const;
  // 获取与性能分析相关的 Future
  c10::intrusive_ptr<JitFuture> getProfilingFuture() const;
  // 设置与性能分析相关的 Future
  void setProfilingFuture(c10::intrusive_ptr<JitFuture> profilingFuture);

  // 在此 RRef 上创建代理，用于在所有者节点上运行函数
  py::object createRRefProxy(const RRefProxyType& mode, float timeoutSeconds = rpc::kUnsetRpcTimeout) const;

  // 获取此 RRef 引用的数据对象的类型
  py::object getRRefType(float timeout = rpc::kUnsetRpcTimeout, bool blocking = true);

  // 在 RRef 作为根运行反向传播
  void backward(int64_t autogradContextId, bool retainGraph);

  // 静态方法，用于在给定的 RRef 上运行反向传播
  static void backward(int64_t autogradContextId, bool retainGraph, const c10::intrusive_ptr<RRef>& rref);

  // 如果 RRef 是 OwnerRRef，则进行特殊化的反向传播
  static void backwardOwnerRRef(int64_t autogradContextId, bool retainGraph, IValue value);

 private:
  c10::intrusive_ptr<RRef> rref_;  // RRef 的智能指针
  std::optional<c10::intrusive_ptr<JitFuture>> profilingFuture_;  // 与性能分析相关的 Future
  std::optional<py::object> type_;  // 类型的可选 Python 对象
};

} // namespace rpc
} // namespace distributed
} // namespace torch
```