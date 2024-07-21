# `.\pytorch\torch\csrc\distributed\rpc\python_rpc_handler.h`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <torch/csrc/distributed/rpc/message.h>
// 包含 Torch 分布式 RPC 消息定义的头文件

#include <torch/csrc/distributed/rpc/types.h>
// 包含 Torch 分布式 RPC 类型定义的头文件

#include <torch/csrc/jit/frontend/script_type_parser.h>
// 包含 Torch 脚本类型解析器的头文件

#include <torch/csrc/utils/pybind.h>
// 包含 Torch Python 绑定工具的头文件

namespace torch {
namespace distributed {
namespace rpc {

// 命名空间 torch::distributed::rpc，定义了 RPC 相关的类和函数

// Singleton class provides interface to execute python UDF remote call
// and deserialize the returned results by running python function
// in internal_rpc_utilities.
// The singleton object is constructed at first when RPC agent is
// constructed, where the python function in
// torch/distributed/internal_rpc_utils.py are imported only once.
// 单例类提供执行 Python UDF 远程调用的接口，
// 并通过运行 internal_rpc_utilities 中的 Python 函数反序列化返回的结果。
// 单例对象在 RPC agent 构造时首次构造，
// 其中 torch/distributed/internal_rpc_utils.py 中的 Python 函数仅导入一次。

class PYBIND11_EXPORT PythonRpcHandler {
 public:
  struct RRefProxyFunctions {
    py::object rrefProxyCtor_;  // RRefProxy 的构造函数对象
    py::object rpcSync_;        // 同步 RPC 调用的 Python 函数对象
    py::object rpcAsync_;       // 异步 RPC 调用的 Python 函数对象
    py::object remote_;         // 远程调用的 Python 函数对象
  };

  struct RRefTypeFunctions {
    py::object onOwner_;  // RRef 类型的 onOwner 方法的 Python 函数对象
  };

}; // namespace rpc
} // namespace distributed
} // namespace torch

// 结束命名空间 torch::distributed::rpc
```