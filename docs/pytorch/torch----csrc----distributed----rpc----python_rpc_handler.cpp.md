# `.\pytorch\torch\csrc\distributed\rpc\python_rpc_handler.cpp`

```
// 包含必要的头文件来支持 RPC 的 Python 处理程序和代理
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/python_compat.h>

// Torch 的命名空间开始
namespace torch {
namespace distributed {
namespace rpc {

// 匿名命名空间，定义常量内部模块的字符串
namespace {

// 定义一个宏，用于获取全局解释器锁 (GIL)，并在性能分析中记录 GIL 获取时间。
// 平均 GIL 获取时间将记录在 RpcAgent 的 getMetrics() 中。
#define PROFILE_GIL_SCOPED_ACQUIRE                                       \
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime; \
  auto shouldProfileGIL =                                                \
      RpcAgent::getCurrentRpcAgent() -> isGILProfilingEnabled();         \
  if (shouldProfileGIL) {                                                \
    startTime = std::chrono::high_resolution_clock::now();               \
  }                                                                      \
  pybind11::gil_scoped_acquire ag;                                       \
  if (shouldProfileGIL) {                                                \
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(    \
        std::chrono::high_resolution_clock::now() - startTime);          \
    RpcAgent::getCurrentRpcAgent()->addGilWaitTime(dur);                 \
  } // NOLINT

// 定义一个结构体 PythonTypeResolver，继承自 jit::Resolver，用于解析 Python 类型。
struct PythonTypeResolver : public jit::Resolver {
  // 解析值的函数，不需要实现
  std::shared_ptr<jit::SugaredValue> resolveValue(
      const std::string& /* unused */,
      torch::jit::GraphFunction& /* unused */,
      const jit::SourceRange& /* unused */) override {
    TORCH_INTERNAL_ASSERT(
        false, "RPC Type resolver does not need to resolve value");
  }

  // 解析类型的函数，根据名称返回对应的类型
  TypePtr resolveType(
      const std::string& name,
      const jit::SourceRange& /* unused */) override {
    // 如果名称为 "PyObject"，返回 PyObjectType::get()
    if (name == "PyObject") {
      return PyObjectType::get();
    }
    // 否则从 PythonRpcHandler 获取解释单元的类型
    return PythonRpcHandler::getInstance().jitCompilationUnit()->get_type(name);
  }
};

// 获取模块中指定名称的函数对象
py::object getFunction(const py::object& module, const char* name) {
  py::object fn = module.attr(name);
  // 检查获取的对象是否是函数类型
  TORCH_CHECK(
      py::isinstance<py::function>(fn),
      "attribute ",
      name,
      " is not a function");
  return fn;
}

// 清理 Python 对象的函数，减少引用计数并设置指针为 nullptr 避免二次减少引用
void cleanupPyObj(py::object& obj) {
  obj.dec_ref();
  // 明确地将 PyObject* 设置为 nullptr，以防止 py::object 的析构函数再次减少 PyObject 的引用计数。
  // 见 python_ivalue.h 中的注释 [Destructing py::object]
  obj.ptr() = nullptr;
}

} // namespace

// PythonRpcHandler 的初始化函数实现
void PythonRpcHandler::init() {
  // 使用互斥锁保护初始化过程
  std::lock_guard<std::mutex> guard(init_lock_);
  // 如果尚未初始化，则执行初始化
  if (!initialized_) {
    // 获取 GIL 并记录性能
    PROFILE_GIL_SCOPED_ACQUIRE;
    // 导入内部 RPC 模块
    py::object rpcInternal = py::module::import(kInternalModule);
    // 导入分布式 RPC API 模块
    py::object rpcApi = py::module::import("torch.distributed.rpc.api");


这样的注释应该涵盖了每行代码的功能和目的，而不会过多地总结代码的整体含义。
    # 导入 torch.distributed.rpc.rref_proxy 模块，并赋值给 rrefProxy 对象
    py::object rrefProxy =
        py::module::import("torch.distributed.rpc.rref_proxy");

    # 获取 rpcInternal 中的函数 "_run_function"，并赋值给 pyRunFunction_
    pyRunFunction_ = getFunction(rpcInternal, "_run_function");
    
    # 获取 rpcInternal 中的函数 "serialize"，并赋值给 pySerialize_
    pySerialize_ = getFunction(rpcInternal, "serialize");
    
    # 获取 rpcInternal 中的函数 "deserialize"，并赋值给 pyDeserialize_
    pyDeserialize_ = getFunction(rpcInternal, "deserialize");
    
    # 获取 rpcInternal 中的函数 "_handle_exception"，并赋值给 pyHandleException_
    pyHandleException_ = getFunction(rpcInternal, "_handle_exception");

    # 获取 rpcApi 中的函数 "_rref_typeof_on_owner"，并赋值给 rrefTypeFunctions_.onOwner_
    rrefTypeFunctions_.onOwner_ = getFunction(rpcApi, "_rref_typeof_on_owner");
    
    # 获取 rpcApi 中的函数 "_rref_typeof_on_user"，并赋值给 rrefTypeFunctions_.onUser_
    rrefTypeFunctions_.onUser_ = getFunction(rpcApi, "_rref_typeof_on_user");

    # 获取 rpcApi 中的函数 "rpc_sync"，并赋值给 rrefProxyFunctions_.rpcSync_
    rrefProxyFunctions_.rpcSync_ = getFunction(rpcApi, "rpc_sync");
    
    # 获取 rpcApi 中的函数 "rpc_async"，并赋值给 rrefProxyFunctions_.rpcAsync_
    rrefProxyFunctions_.rpcAsync_ = getFunction(rpcApi, "rpc_async");
    
    # 获取 rpcApi 中的函数 "remote"，并赋值给 rrefProxyFunctions_.remote_
    rrefProxyFunctions_.remote_ = getFunction(rpcApi, "remote");
    
    # 获取 torch.distributed.rpc.rref_proxy 中的函数 "RRefProxy"，并赋值给 rrefProxyFunctions_.rrefProxyCtor_
    rrefProxyFunctions_.rrefProxyCtor_ = getFunction(rrefProxy, "RRefProxy");

    # 使用 torch.jit 模块中的 get_python_cu 函数，获取 JIT 编译单元，赋值给 jitCompilationUnit_
    jitCompilationUnit_ = torch::jit::get_python_cu();
    
    # 创建一个共享的 ScriptTypeParser 对象，使用 PythonTypeResolver 初始化，赋值给 typeParser_
    typeParser_ = std::make_shared<jit::ScriptTypeParser>(
        std::make_shared<PythonTypeResolver>());
    
    # 将 initialized_ 标记为 true，表示初始化完成
    initialized_ = true;
  }
}

PythonRpcHandler::PythonRpcHandler() : initialized_(false) {}

void PythonRpcHandler::cleanup() {
  std::lock_guard<std::mutex> guard(init_lock_);
  // 用于安全地获取 GIL 并进行性能分析
  PROFILE_GIL_SCOPED_ACQUIRE;
  // 清理并释放 Python 函数对象
  cleanupPyObj(pyRunFunction_);
  cleanupPyObj(pySerialize_);
  cleanupPyObj(pyDeserialize_);
  cleanupPyObj(pyHandleException_);

  // 清理并释放远程引用代理函数对象
  cleanupPyObj(rrefProxyFunctions_.rpcSync_);
  cleanupPyObj(rrefProxyFunctions_.rpcAsync_);
  cleanupPyObj(rrefProxyFunctions_.remote_);
  cleanupPyObj(rrefProxyFunctions_.rrefProxyCtor_);

  // 重置 JIT 编译单元和类型解析器指针
  jitCompilationUnit_ = nullptr;
  typeParser_ = nullptr;
  // 标记实例未初始化
  initialized_ = false;
}

PythonRpcHandler& PythonRpcHandler::getInstance() {
  // 当调用 PythonRpcHandler::getInstance() 时，可能有线程持有 GIL，
  // 同时另一个线程可能正在调用 `new PythonRpcHandler()` 进行静态数据初始化，
  // 在静态数据初始化期间也需要 GIL。静态数据初始化是线程安全的，因此持有 GIL 的线程
  // 将等待另一个线程完成静态数据初始化后继续执行。由于初始化不能在没有 GIL 的情况下进行，
  // 这可能导致死锁。我们要求调用线程释放 GIL 以避免这种情况。
  TORCH_INTERNAL_ASSERT(!PyGILState_Check());
  // 为了避免模块析构的竞争条件，使用泄露单例模式。
  static PythonRpcHandler* handler = new PythonRpcHandler();
  handler->init();
  return *handler;
}

std::shared_ptr<torch::jit::CompilationUnit> PythonRpcHandler::
    jitCompilationUnit() {
  // 返回 JIT 编译单元的共享指针
  return jitCompilationUnit_;
}

py::object PythonRpcHandler::runPythonUdf(const py::object& pythonUdf) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // 如果 pyRunFunction_ 已清理，抛出描述性错误消息
  TORCH_INTERNAL_ASSERT(
      !pyRunFunction_.is_none(),
      "Cannot run python UDF since pyRunFunction_ is None. Check if python RPC "
      "handler is already cleaned up.");
  // 运行 Python 用户定义函数
  return pyRunFunction_(pythonUdf);
}

SerializedPyObj PythonRpcHandler::serialize(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // 调用 pySerialize_ 序列化 Python 对象，并构造 SerializedPyObj 返回
  py::tuple t = pySerialize_(obj);
  return SerializedPyObj(
      t[0].cast<std::string>(), t[1].cast<std::vector<torch::Tensor>>());
}

py::object PythonRpcHandler::deserialize(const SerializedPyObj& serializedObj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // 注意：pyDeserialize_ 在失败时可能会返回 AttributeError。使用结果的函数需要适当处理此类错误。
  // 反序列化 SerializedPyObj 对象
  return pyDeserialize_(
      py::bytes(serializedObj.payload_), serializedObj.tensors_);
}

void PythonRpcHandler::handleException(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;
  // 处理异常对象，调用 pyHandleException_
  pyHandleException_(obj);
}

void PythonRpcHandler::handleExceptionGILHeld(const py::object& obj) {
  // 检查 GIL 是否被持有
  TORCH_CHECK(PyGILState_Check(), "GIL should be held");
  // 在 GIL 被持有的情况下处理异常对象，调用 pyHandleException_
  pyHandleException_(obj);
}
// 检查给定的 Python 对象是否为远程异常对象
bool PythonRpcHandler::isRemoteException(const py::object& obj) {
  PROFILE_GIL_SCOPED_ACQUIRE;  // 获取全局解释器锁（GIL）的性能分析

  // 获取对象的类型
  auto type = obj.get_type();
  // 获取类型所在的模块名称作为字符串
  auto moduleName = type.attr("__module__").cast<std::string>();
  // 获取类型的限定名称（包括模块名）作为字符串
  auto qualName = type.attr("__qualname__").cast<std::string>();

  // 检查是否为内部模块定义的 Remote Exception
  return moduleName == kInternalModule && qualName == "RemoteException";
}

// 从字符串解析类型信息，并返回类型指针
TypePtr PythonRpcHandler::parseTypeFromStr(const std::string& type_str) {
  return typeParser_->parseType(type_str);  // 使用 typeParser_ 解析给定字符串表示的类型
}

// 返回 RRef 代理函数的常量引用
const PythonRpcHandler::RRefProxyFunctions& PythonRpcHandler::
    getRRefProxyFunctions() const {
  return rrefProxyFunctions_;  // 返回成员变量 rrefProxyFunctions_ 的常量引用
}

// 返回 RRef 类型函数的常量引用
const PythonRpcHandler::RRefTypeFunctions& PythonRpcHandler::
    getRRefTypeFunctions() const {
  return rrefTypeFunctions_;  // 返回成员变量 rrefTypeFunctions_ 的常量引用
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```