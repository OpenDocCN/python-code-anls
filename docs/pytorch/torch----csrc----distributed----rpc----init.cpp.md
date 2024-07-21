# `.\pytorch\torch\csrc\distributed\rpc\init.cpp`

```
// 引入 Torch 的 Python 头文件
#include <torch/csrc/python_headers.h>

// 引入分布式 RPC 的远程分析管理器头文件
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>

// 引入分布式 RPC 的服务器进程全局分析器头文件
#include <torch/csrc/distributed/rpc/profiler/server_process_global_profiler.h>

// 引入分布式 RPC 的 Python 引用远程引用头文件
#include <torch/csrc/distributed/rpc/py_rref.h>

// 引入分布式 RPC 的 Python 函数头文件
#include <torch/csrc/distributed/rpc/python_functions.h>

// 引入分布式 RPC 的 Python RPC 处理器头文件
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

// 引入分布式 RPC 的请求回调实现头文件
#include <torch/csrc/distributed/rpc/request_callback_impl.h>

// 引入分布式 RPC 的 RPC 代理头文件
#include <torch/csrc/distributed/rpc/rpc_agent.h>

// 引入分布式 RPC 的远程引用上下文头文件
#include <torch/csrc/distributed/rpc/rref_context.h>

// 引入分布式 RPC 的 TensorPipe 代理头文件
#include <torch/csrc/distributed/rpc/tensorpipe_agent.h>

// 引入分布式 RPC 的 TorchScript 函数头文件
#include <torch/csrc/distributed/rpc/torchscript_functions.h>

// 引入分布式 RPC 的类型定义头文件
#include <torch/csrc/distributed/rpc/types.h>

// 引入 JIT Python 绑定工具头文件
#include <torch/csrc/jit/python/pybind_utils.h>

// 引入 Torch 的对象指针工具头文件
#include <torch/csrc/utils/object_ptr.h>

// 引入 Torch 的 Python 绑定头文件
#include <torch/csrc/utils/pybind.h>

// 引入 Torch 的类型头文件
#include <torch/types.h>

// 引入 pybind11 的时间相关头文件
#include <pybind11/chrono.h>

// 引入 pybind11 的操作符重载头文件
#include <pybind11/operators.h>

// Torch 分布式 RPC 的命名空间
namespace torch {
namespace distributed {
namespace rpc {

// 匿名命名空间用于定义局部常量
namespace {

// 定义删除所有用户超时时间为 100000 毫秒
constexpr std::chrono::milliseconds kDeleteAllUsersTimeout(100000);

// 定义一个模板，用于创建共享指针类型的 Python 类
template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// RPC 初始化函数的实现
PyObject* rpc_init(PyObject* _unused, PyObject* noargs) {
  // 导入 torch.distributed.rpc 模块
  auto rpc_module = THPObjectPtr(PyImport_ImportModule("torch.distributed.rpc"));
  if (!rpc_module) {
    throw python_error();  // 如果导入失败，则抛出 Python 错误
  }

  // 导入 torch._C 模块
  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    // 如果导入失败，则抛出 Python 错误
#ifdef USE_TENSORPIPE

#ifdef USE_TENSORPIPE 是一个条件编译指令，用于根据预定义的宏来决定编译是否包含这部分代码。


  module.def("_is_current_rpc_agent_set", &RpcAgent::isCurrentRpcAgentSet);

将 C++ 函数 RpcAgent::isCurrentRpcAgentSet() 暴露给 Python，使其可以通过 _is_current_rpc_agent_set() 调用。


  module.def("_get_current_rpc_agent", &RpcAgent::getCurrentRpcAgent);

将 C++ 函数 RpcAgent::getCurrentRpcAgent() 暴露给 Python，使其可以通过 _get_current_rpc_agent() 调用。


  module.def(
      "_set_and_start_rpc_agent",
      [](const std::shared_ptr<RpcAgent>& rpcAgent) {
        RpcAgent::setCurrentRpcAgent(rpcAgent);
        // Initializing typeResolver inside RpcAgent constructor will make
        // RpcAgent have python dependency. To avoid RpcAgent to have python
        // dependency, setTypeResolver() here.
        std::shared_ptr<TypeResolver> typeResolver =
            std::make_shared<TypeResolver>([&](const c10::QualifiedName& qn) {
              auto typePtr = PythonRpcHandler::getInstance().parseTypeFromStr(
                  qn.qualifiedName());
              return c10::StrongTypePtr(
                  PythonRpcHandler::getInstance().jitCompilationUnit(),
                  std::move(typePtr));
            });
        rpcAgent->setTypeResolver(typeResolver);
        rpcAgent->start();
      },
      py::call_guard<py::gil_scoped_release>());

定义一个 Python 函数 _set_and_start_rpc_agent，接受一个 std::shared_ptr<RpcAgent> 参数，并在其中设置当前的 RpcAgent，初始化类型解析器 typeResolver，并启动 RpcAgent 实例。


  module.def(
      "_reset_current_rpc_agent",
      []() { RpcAgent::setCurrentRpcAgent(nullptr); },
      py::call_guard<py::gil_scoped_release>());

定义一个 Python 函数 _reset_current_rpc_agent，用于将当前 RpcAgent 设置为 nullptr，即重置当前 RpcAgent。


  module.def(
      "_delete_all_user_and_unforked_owner_rrefs",
      [](std::chrono::milliseconds timeoutMillis) {
        RRefContext::getInstance().delAllUsersAndUnforkedOwners(timeoutMillis);
      },
      py::arg("timeout") = kDeleteAllUsersTimeout,
      py::call_guard<py::gil_scoped_release>());

定义一个 Python 函数 _delete_all_user_and_unforked_owner_rrefs，接受一个超时时间参数 timeoutMillis，用于删除所有用户和未分叉所有者的 RRef。


  module.def("_destroy_rref_context", [](bool ignoreRRefLeak) {
    // NB: do not release GIL in the function. The destroyInstance() method
    // returns a list of deleted OwnerRRefs that hold py::object instances.
    // Clearing those OwnerRRefs are likely to trigger Python deref, which
    // requires GIL.
    RRefContext::getInstance().destroyInstance(ignoreRRefLeak).clear();
  });

定义一个 Python 函数 _destroy_rref_context，接受一个布尔参数 ignoreRRefLeak，用于销毁 RRef 上下文实例。注意，此函数不释放 GIL，因为 destroyInstance() 方法返回的已删除的 OwnerRRefs 可能持有 py::object 实例，清理这些 OwnerRRefs 可能触发需要 GIL 的 Python 解引用操作。


  module.def("_rref_context_get_debug_info", []() {

定义一个 Python 函数 _rref_context_get_debug_info，无参数，用于获取 RRef 上下文的调试信息。
    return RRefContext::getInstance().getDebugInfo();
  });


  // 调用单例模式中的 RRefContext 对象，获取调试信息并返回
  module.def(
      "_cleanup_python_rpc_handler",
      []() { PythonRpcHandler::getInstance().cleanup(); },
      py::call_guard<py::gil_scoped_release>());


  // 定义 _cleanup_python_rpc_handler 函数，用于清理 Python RPC 处理器的状态
  module.def(
      "_invoke_rpc_builtin",
      [](const WorkerInfo& dst,
         const std::string& opName,
         const float rpcTimeoutSeconds,
         const py::args& args,
         const py::kwargs& kwargs) {
        // 调用 pyRpcBuiltin 函数发起 RPC 调用，并包装成 PythonFutureWrapper 对象返回
        return std::make_shared<jit::PythonFutureWrapper>(
            pyRpcBuiltin(dst, opName, args, kwargs, rpcTimeoutSeconds));
      },
      py::call_guard<py::gil_scoped_acquire>());


  // 定义 _invoke_rpc_builtin 函数，用于发起内置函数的 RPC 调用
  module.def(
      "_invoke_rpc_python_udf",
      [](const WorkerInfo& dst,
         std::string& pickledPythonUDF,
         std::vector<torch::Tensor>& tensors,
         const float rpcTimeoutSeconds,
         const bool isAsyncExecution) {
        // 调用 pyRpcPythonUdf 函数发起 Python UDF 的 RPC 调用，并返回包装的 PythonFutureWrapper 对象
        return std::make_shared<jit::PythonFutureWrapper>(pyRpcPythonUdf(
            dst,
            pickledPythonUDF,
            tensors,
            rpcTimeoutSeconds,
            isAsyncExecution));
      },
      py::call_guard<py::gil_scoped_release>());


  // 定义 _invoke_rpc_python_udf 函数，用于发起 Python 用户定义函数的 RPC 调用
  module.def(
      "_invoke_rpc_torchscript",
      [](const std::string& dstWorkerName,
         const std::string& qualifiedNameStr,
         const py::tuple& argsTuple,
         const py::dict& kwargsDict,
         const float rpcTimeoutSeconds,
         const bool isAsyncExecution) {
        // 调用 pyRpcTorchscript 函数发起 TorchScript 的 RPC 调用，并返回包装的 PythonFutureWrapper 对象
        return std::make_shared<jit::PythonFutureWrapper>(pyRpcTorchscript(
            dstWorkerName,
            qualifiedNameStr,
            argsTuple,
            kwargsDict,
            rpcTimeoutSeconds,
            isAsyncExecution));
      },
      py::call_guard<py::gil_scoped_release>());


  // 定义 _invoke_rpc_torchscript 函数，用于发起 TorchScript 的 RPC 调用
  module.def(
      "_invoke_remote_builtin",
      &pyRemoteBuiltin,
      py::call_guard<py::gil_scoped_acquire>());


  // 定义 _invoke_remote_builtin 函数，将其指定为 pyRemoteBuiltin 函数的 Python 绑定，加入 GIL 保护
  module.def(
      "_invoke_remote_python_udf",
      &pyRemotePythonUdf,
      py::call_guard<py::gil_scoped_release>());


  // 定义 _invoke_remote_python_udf 函数，将其指定为 pyRemotePythonUdf 函数的 Python 绑定，释放 GIL
  module.def(
      "_invoke_remote_torchscript",
      &pyRemoteTorchscript,
      py::call_guard<py::gil_scoped_release>());


  // 定义 _invoke_remote_torchscript 函数，将其指定为 pyRemoteTorchscript 函数的 Python 绑定，释放 GIL
  module.def(
      "get_rpc_timeout",
      []() {
        // 获取当前 RPC 代理的超时时间，以秒为单位返回
        return RpcAgent::getCurrentRpcAgent()->getRpcTimeout().count() /
            kSecToMsConversion;
      },
      R"(
          Retrieve the default timeout for all RPCs that was set during RPC initialization.
          The returned value will be in seconds.
          Returns:
            ``float`` indicating the RPC timeout in seconds.
      )");


  // 定义 get_rpc_timeout 函数，用于获取 RPC 的超时时间
  module.def(
      "enable_gil_profiling",
      [](bool flag) {
        // 根据 flag 设置是否启用 GIL 采样
        RpcAgent::getCurrentRpcAgent()->enableGILProfiling(flag);
      },
      R"(
    Set whether GIL wait times should be enabled or not. This incurs a slight
    overhead cost. Default is disabled for performance reasons.
      )");


  // 定义 enable_gil_profiling 函数，用于设置是否启用 GIL 采样
    Args:
        flag (bool): True to set GIL profiling, False to disable.
      )");

  module.def(
      "_set_rpc_timeout",
      [](const float rpcTimeoutSeconds) {
        // 将秒数转换为毫秒，并设置为 RPC 的默认超时时间
        auto rpcTimeout = std::chrono::milliseconds(
            static_cast<int>(rpcTimeoutSeconds * kSecToMsConversion));
        // 获取当前的 RPC 代理并设置 RPC 超时时间
        RpcAgent::getCurrentRpcAgent()->setRpcTimeout(rpcTimeout);
      },
      R"(
          设置所有 RPC 的默认超时时间。输入单位为秒。
          如果一个 RPC 在此时间内未完成，将会抛出超时异常。
          若要控制特定 RPC 的超时时间，可以在 :meth:`~torch.distributed.rpc.rpc_sync` 和
          :meth:`~torch.distributed.rpc.rpc_async` 中传入超时参数。

          Args:
            rpcTimeoutSeconds (float): 超时时间，单位为秒。
      )");

  module.def(
      "_enable_server_process_global_profiler",
      &profiler::processglobal::enableServer);
  module.def(
      "_disable_server_process_global_profiler",
      &profiler::processglobal::disableServer);

  module.def("_set_profiler_node_id", &at::RecordFunction::setDefaultNodeId);

  py::class_<
      RemoteProfilerManager,
      std::unique_ptr<RemoteProfilerManager, py::nodelete>>(
      module, "RemoteProfilerManager")
      .def("set_current_profiling_key", [](const std::string& key) {
        // 获取远程性能分析管理器的实例并设置当前的性能分析键
        auto& inst = RemoteProfilerManager::getInstance();
        inst.setCurrentKey(key);
      });

  module.def(
      "_enable_jit_rref_pickle",
      &enableJitRRefPickle,
      R"(
        允许 ``torch.jit.save`` 保存带有RPC上的pickled RRefs的 ``torch.jit.ScriptModule``。


        .. warning::
            这是危险的。如果模块包含RRefs，则pickled结果必须通过RPC发送，并在接收端解pickle以恢复模块。
            否则，将会出现RRef泄漏，可能导致程序挂起。
            当使用此API时，应用程序有责任确保上述假设始终成立。
      )");
  module.def("_disable_jit_rref_pickle", &disableJitRRefPickle);

  Py_RETURN_TRUE;
} // namespace



} // namespace

这行注释表示结束了一个命名空间的定义。


static PyMethodDef methods[] = { // NOLINT

这里定义了一个静态的PyMethodDef数组，用于存储Python扩展模块中的方法定义。


{"_rpc_init", rpc_init, METH_NOARGS, nullptr},

这是PyMethodDef数组中的一个元素，定义了一个名为"_rpc_init"的Python函数，其实现由rpc_init函数提供，且不接受任何参数（METH_NOARGS），并且没有文档字符串（nullptr）。


{nullptr, nullptr, 0, nullptr}};

PyMethodDef数组的最后一个元素，用于标记数组的结束，因为第一个元素为nullptr。


PyMethodDef* python_functions() {

这是一个名为python_functions的函数定义，返回类型为PyMethodDef*，即指向PyMethodDef数组的指针。


return methods;

该函数返回定义的PyMethodDef数组methods的指针。


} // namespace rpc
} // namespace distributed
} // namespace torch

这三行依次结束了嵌套的命名空间rpc、distributed和torch的定义。
```