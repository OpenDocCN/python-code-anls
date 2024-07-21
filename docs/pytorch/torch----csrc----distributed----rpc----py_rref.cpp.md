# `.\pytorch\torch\csrc\distributed\rpc\py_rref.cpp`

```
// 包含 Torch 分布式 RPC 所需的头文件
#include <torch/csrc/distributed/rpc/py_rref.h>

// 包含 Torch 自动微分相关头文件
#include <torch/csrc/autograd/autograd.h>

// 包含 Torch 分布式自动微分相关头文件
#include <torch/csrc/distributed/autograd/autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>

// 包含 Torch 分布式 RPC 的 Python 函数头文件
#include <torch/csrc/distributed/rpc/python_functions.h>

// 包含 Torch 分布式 RPC 的 Python RPC 处理器头文件
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>

// 包含 Torch 分布式 RPC 的远程引用上下文头文件
#include <torch/csrc/distributed/rpc/rref_context.h>

// 包含 Torch JIT 的 Python 模块相关头文件
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>

// Torch 的命名空间
namespace torch {
// Torch 分布式命名空间
namespace distributed {
// Torch 分布式 RPC 命名空间
namespace rpc {

/////////////////////  Pickle/Unpickle Helplers ////////////////////////////

// 匿名命名空间，定义了将 RRefForkData 转换为 Python 元组的函数
namespace {

// 将 RRefForkData 转换为 Python 元组
py::tuple toPyTuple(const RRefForkData& rrefForkData) {
  // 获取全局解释器锁，因为要构造一个 py::object
  pybind11::gil_scoped_acquire ag;
  // 使用 RRefForkData 的各个字段构造 Python 元组
  return py::make_tuple(
      rrefForkData.ownerId_,
      rrefForkData.rrefId_.createdOn_,
      rrefForkData.rrefId_.localId_,
      rrefForkData.forkId_.createdOn_,
      rrefForkData.forkId_.localId_,
      rrefForkData.parent_,
      rrefForkData.typeStr_);
}

// 将 Python 元组转换为 RRefForkData
RRefForkData fromPyTuple(const py::tuple& pyTuple) {
  // 获取全局解释器锁，因为要访问一个 py::object
  pybind11::gil_scoped_acquire ag;
  // 断言 Python 元组的大小符合预期
  TORCH_INTERNAL_ASSERT(
      pyTuple.size() == RFD_TUPLE_SIZE,
      "Pickled RRefForkData must contain ",
      RFD_TUPLE_SIZE,
      " numbers.");
  // 从 Python 元组中提取各个字段并创建 RRefForkData 对象
  worker_id_t ownerId = pyTuple[OWNER_IDX].cast<worker_id_t>();
  const RRefId& rrefId = RRefId(
      pyTuple[RREFID_ON_IDX].cast<worker_id_t>(),
      pyTuple[RREFID_ID_IDX].cast<local_id_t>());
  const RRefId& forkId = RRefId(
      pyTuple[FORKID_ON_IDX].cast<worker_id_t>(),
      pyTuple[FORKID_ID_IDX].cast<local_id_t>());
  worker_id_t parent = pyTuple[PARENT_IDX].cast<worker_id_t>();
  const std::string& typeStr = pyTuple[TYPE_IDX].cast<std::string>();

  return RRefForkData(ownerId, rrefId, forkId, parent, typeStr);
}

// 尝试使用类型提示推断类型
TypePtr tryInferTypeWithTypeHint(
    const py::object& value,
    const py::object& type_hint) {
  // 如果要包含在 RRef 中的 py::object 是 ScriptModule，则强制用户提供其 ModuleInterface 类型提示
  if (auto module = jit::as_module(value)) {
    TORCH_CHECK(
        !type_hint.is_none(),
        "The RRef being created contains a ScriptModule, "
        "must provide its ModuleInterface type hint. ");
    // 获取类型的限定名
    c10::QualifiedName type_qualified_name = c10::QualifiedName(
        py::cast<std::string>(py::module::import("torch._jit_internal")
                                  .attr("_qualified_name")(type_hint)));
    // 根据限定名获取类型指针
    TypePtr type_hint_ptr =
        jit::get_python_cu()->get_interface(type_qualified_name);
    std::ostringstream subtype_check_msg;
    // 检查是否存在类型提示指针且该模块类型是类型提示的子类，否则抛出错误信息
    TORCH_CHECK(
        type_hint_ptr != nullptr &&
            module.value().type()->isSubtypeOfExt(
                *type_hint_ptr, &subtype_check_msg),
        module.value().type()->repr_str(),
        " is not a subtype of the type hint: ",
        type_qualified_name.qualifiedName(),
        ", did you pass a valid interface type?\n",
        subtype_check_msg.str());
    // 如果条件成立，返回类型提示指针
    return type_hint_ptr;
  } else {
    // 如果类型提示不为 nullptr，则抛出错误信息，表明在创建包含 ScriptModule 的 RRef 时不应该指定类型提示
    TORCH_CHECK(
        type_hint.is_none(),
        "type_hint should only be specified when the RRef being created contains a ScriptModule.");
  }

  // 检查 value 是否是 ScriptClass 的实例。如果不是，则跳过类型推断，因为这将尝试对 value 所属的类进行脚本化，应该避免这种情况。
  py::bool_ can_compile = py::module::import("torch._jit_internal")
                              .attr("can_compile_class")(value.get_type());

  // 如果 value 可以编译为 ScriptClass
  if (py::cast<bool>(can_compile)) {
    // 获取 value 所属的 ScriptClass
    py::object existing_ty = py::module::import("torch.jit._state")
                                 .attr("_get_script_class")(value.get_type());

    // 如果获取的 ScriptClass 为空，则返回 PyObjectType::get()
    if (existing_ty.is_none()) {
      return PyObjectType::get();
    }
  }

  // 注意：`jit::tryToInferType(..)` 用于推断类型，包括 ScriptClass，但不包括 ScriptModule。
  // 尝试从 pyobject 推断类型
  jit::InferredType type_inferred = jit::tryToInferType(value);
  
  // 如果成功推断类型
  if (type_inferred.success()) {
    // 返回推断的类型
    return type_inferred.type();
  }

  // 否则，返回一个包含 pyobject 类型的 RRef
  return PyObjectType::get();
} // namespace



///////////////////////////  PyRRef  //////////////////////////////////

PyRRef::PyRRef(c10::intrusive_ptr<RRef> rref)
    : rref_(std::move(rref)), profilingFuture_(c10::nullopt) {
  // 检查 rref_ 必须不为 nullptr
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
  // 记录 API 使用情况
  C10_LOG_API_USAGE_ONCE("torch.distributed.rref");
}

PyRRef::PyRRef(const py::object& value, const py::object& type_hint)
    : PyRRef([&value, &type_hint]() mutable {
        // 尝试根据类型提示推断元素类型
        TypePtr elem_type = tryInferTypeWithTypeHint(value, type_hint);
        // 创建一个拥有者 RRef 实例
        auto rref = RRefContext::getInstance().createOwnerRRef(elem_type);
        // 将 Python 对象转换为 IValue 类型
        IValue ivalue = jit::toIValue(value, elem_type);
        // 设置 RRef 的值
        rref->setValue(std::move(ivalue));
        return rref;
      }()) {}

PyRRef::~PyRRef() {
  if (type_.has_value()) {
    // 获取全局解释器锁
    pybind11::gil_scoped_acquire ag;
    // 减少类型引用计数
    (*type_).dec_ref();
    // 明确将 PyObject* 设置为 nullptr，以防止 py::object 的析构函数再次减少 PyObject 的引用计数
    (*type_).ptr() = nullptr;
  }
}

c10::intrusive_ptr<JitFuture> PyRRef::getFuture() const {
  // 将拥有者创建的未来状态转换为 PyJitFuture
  return toPyJitFuture(rref_->getOwnerCreationFuture(), false /* hasValue */);
}

c10::intrusive_ptr<JitFuture> PyRRef::getProfilingFuture() const {
  // 断言是否已设置 profilingFuture_
  TORCH_INTERNAL_ASSERT(profilingFuture_, "Profiling future has not been set!");
  return *profilingFuture_;
}

void PyRRef::setProfilingFuture(c10::intrusive_ptr<JitFuture> profilingFuture) {
  // 设置 profilingFuture_
  profilingFuture_ = std::move(profilingFuture);
}

bool PyRRef::isOwner() const {
  // 检查是否为拥有者
  return rref_->isOwner();
}

bool PyRRef::confirmedByOwner() const {
  // 检查是否被拥有者确认
  return rref_->confirmedByOwner();
}

WorkerInfo PyRRef::owner() const {
  // 获取拥有者的 WorkerInfo
  return RRefContext::getInstance().agent()->getWorkerInfo(rref_->owner());
}

std::string PyRRef::ownerName() const {
  // 获取拥有者的名称
  return rref_->ownerName();
}

py::object PyRRef::toHere(const float timeoutSeconds) const {
  // 记录 API 使用情况
  C10_LOG_API_USAGE_ONCE("torch.distributed.rref.to_here");
  if (rref_->isOwner()) {
    // 如果是拥有者，则返回本地值
    return localValue();
  } else {
    // 如果不是拥有者，则调用 toHere() 获取 Python 对象
    IValue value = c10::static_intrusive_pointer_cast<UserRRef>(rref_)->toHere(
        timeoutSeconds);


注释完整解释了每个语句的作用和目的，符合指定的格式和注意事项。
    // 如果 rref_ 是一个 Python 对象，则执行以下代码块
    if (rref_->isPyObj()) {
      // python_rpc_handler 反序列化将会获取 GIL（全局解释器锁）。
      auto rfr_values = value.toTupleRef().elements().vec();
      // 获取 PythonRpcHandler 的单例实例
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      // 调用 PythonRpcHandler 实例的 deserialize 方法，传入 SerializedPyObj::fromIValues 的结果
      auto ret = pythonRpcHandler.deserialize(
          SerializedPyObj::fromIValues(std::move(rfr_values)));
      // 处理可能发生的异常
      pythonRpcHandler.handleException(ret);
      // 返回 deserialize 的结果
      return ret;
    } else {
      // 如果 rref_ 不是 Python 对象，获取 GIL，因为 torch::jit::toPyObject 会创建新的 py::object 而不获取 GIL。
      pybind11::gil_scoped_acquire ag;
      // 调用 torch::jit::toPyObject 将 value 转换为 PyObject，并返回结果
      return torch::jit::toPyObject(std::move(value));
    }
  }
}

py::object PyRRef::localValue() const {
  // 检查当前 RRef 是否为所有者，如果不是，则抛出错误信息
  TORCH_CHECK(
      rref_->isOwner(),
      "For ",
      *rref_,
      ", can't call localValue() on user ",
      RRefContext::getInstance().agent()->getWorkerInfo(),
      ". Call it on owner ",
      owner());

  // 初始化结果对象
  py::object res;
  // 获取当前 RRef 对象的值
  auto value =
      c10::static_intrusive_pointer_cast<const OwnerRRef>(rref_)->getValue();
  // 获取 Python RPC 处理器的实例
  auto& rpcHandler = PythonRpcHandler::getInstance();
  {
    // 获取全局解释器锁（GIL），因为 torch::jit::toPyObject 可能创建新的 py::object 而不会获取 GIL
    pybind11::gil_scoped_acquire ag;
    // 将 RRef 的值转换为 Python 对象
    res = torch::jit::toPyObject(std::move(value));
    // 在 GIL 已获取的情况下处理异常
    rpcHandler.handleExceptionGILHeld(res);
  }
  // 返回转换后的 Python 对象
  return res;
}

std::string PyRRef::str() const {
  // 如果当前 RRef 是所有者，则返回相应的字符串表示
  if (rref_->isOwner()) {
    return c10::str("OwnerRRef(", rref_->rrefId(), ")");
  } else {
    // 如果当前 RRef 是用户端 RRef，则返回相应的字符串表示
    return c10::str(
        "UserRRef(RRefId = ",
        rref_->rrefId(),
        ", ForkId = ",
        c10::static_intrusive_pointer_cast<UserRRef>(rref_)->forkId(),
        ")");
  }
}

py::object PyRRef::createRRefProxy(
    const RRefProxyType& type,
    float timeoutSeconds) const {
  // 获取 Python RPC 处理器的实例
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  // 获取全局解释器锁（GIL）
  pybind11::gil_scoped_acquire ag;
  // 获取 RRef 代理函数集合
  auto& functions = pythonRpcHandler.getRRefProxyFunctions();
  // 获取 RRef 代理构造器函数
  auto& ctor = functions.rrefProxyCtor_;
  // 根据给定的类型选择相应的 RRef 代理类型进行创建
  switch (type) {
    case RRefProxyType::RPC_SYNC: {
      return ctor(*this, functions.rpcSync_, timeoutSeconds);
    }
    case RRefProxyType::RPC_ASYNC: {
      return ctor(*this, functions.rpcAsync_, timeoutSeconds);
    }
    case RRefProxyType::REMOTE: {
      return ctor(*this, functions.remote_, timeoutSeconds);
    }
    default: {
      // 如果出现未识别的 RRef 代理类型，则断言失败
      TORCH_INTERNAL_ASSERT(false, "Unrecognized RRefProxy type ", type);
    }
  }
}

py::object PyRRef::getRRefType(float timeout, bool blocking) {
  // 如果当前 RRef 的类型尚未确定
  if (!type_.has_value()) {
    // 释放 GIL，因为调用此函数时不会释放 GIL
    pybind11::gil_scoped_release release;
    // 获取 Python RPC 处理器的实例
    auto& pythonRpcHandler = PythonRpcHandler::getInstance();
    // 获取 RRef 类型相关的函数集合
    auto& typeFuncs = pythonRpcHandler.getRRefTypeFunctions();
    // 获取 GIL
    pybind11::gil_scoped_acquire acquire;
    // 根据当前 RRef 是所有者还是用户，调用相应的函数获取 RRef 的类型
    type_ = isOwner() ? typeFuncs.onOwner_(*this, blocking)
                      : typeFuncs.onUser_(*this, timeout, blocking);
  }
  // 返回 RRef 的类型，可以是 Python 类型或者 Future 对象
  return *type_;
}

py::tuple PyRRef::pickle() const {
  // 获取 RRef 上下文的实例
  auto& ctx = RRefContext::getInstance();
  // 准备 RRef 的 fork 数据
  auto rrefForkData = ctx.prepareChildFork(rref_);
  // 将 fork 数据转换为 Python 元组并返回
  return toPyTuple(rrefForkData);
}

PyRRef PyRRef::unpickle(const py::tuple& pyTuple) {
  // 获取 RRef 上下文的实例
  auto& ctx = RRefContext::getInstance();
  // 从 Python 元组中解析出 RRef 的 fork 数据
  auto rrefForkData = fromPyTuple(pyTuple);
  // 解析出 RRef 的类型
  TypePtr rrefType =
      PythonRpcHandler::getInstance().parseTypeFromStr(rrefForkData.typeStr_);
  // 根据 fork 数据和类型创建或获取 RRef 实例
  c10::intrusive_ptr<RRef> rref = ctx.getOrCreateRRef(rrefForkData, rrefType);
  // 通知所有者和父 RRef 关于 fork 的相关信息
  ctx.notifyOwnerAndParentOfFork(
      rrefForkData.forkId_, rrefForkData.parent_, rref);
  // 返回包装后的 PyRRef 对象
  return PyRRef(std::move(rref));
}
c10::IValue PyRRef::toIValue() const {
  // 将 rref_ 转换为 RRefInterface 类型的指针，以便放入 IValue 中返回
  auto rrefPtr = c10::static_intrusive_pointer_cast<c10::RRefInterface>(rref_);
  return IValue(rrefPtr);
}

void PyRRef::backward(int64_t autogradContextId, bool retainGraph) {
  // 调用带有 rref_ 参数的重载 backward 函数
  backward(autogradContextId, retainGraph, rref_);
}

void PyRRef::backwardOwnerRRef(
    int64_t autogradContextId,
    bool retainGraph,
    IValue value) {
  // 如果 value 是 PyObject，则从中提取出底层的 tensor
  if (value.isPyObject()) {
    py::gil_scoped_acquire gil;  // 获取 GIL（全局解释器锁）
    py::object obj = torch::jit::toPyObject(value);  // 将 IValue 转换为 Python 对象
    try {
      value = torch::jit::toIValue(obj, c10::TensorType::get());  // 尝试将 Python 对象转换为 IValue
    } catch (py::cast_error& e) {
      TORCH_CHECK(false, "RRef should contain a tensor for .backward()");
    }
  }

  TORCH_CHECK(value.isTensor(), "RRef should contain a tensor for .backward()");
  auto root = value.toTensor();  // 将 value 转换为 tensor

  if (autogradContextId == -1) {
    torch::autograd::backward({root});  // 执行单个 tensor 的自动求导
  } else {
    torch::distributed::autograd::backward(
        autogradContextId, {root}, retainGraph);  // 分布式环境下的自动求导
  }
}

void PyRRef::backward(
    int64_t autogradContextId,
    bool retainGraph,
    const c10::intrusive_ptr<RRef>& rref) {
  if (rref->isOwner()) {
    backwardOwnerRRef(
        autogradContextId,
        retainGraph,
        c10::static_intrusive_pointer_cast<const OwnerRRef>(rref)->getValue());
  } else {
    TORCH_CHECK(
        autogradContextId != -1,
        "User RRefs require 'dist_autograd_ctx_id' to be specified");

    autograd::RRefBackwardReq rrefBackwardReq(
        rref->rrefId(), autogradContextId, retainGraph);

    // 远程调用分布式环境下的自动求导
    auto rpcAgent = rpc::RpcAgent::getCurrentRpcAgent();
    rpcAgent
        ->send(
            rpcAgent->getWorkerInfo(rref->owner()),
            std::move(rrefBackwardReq).toMessage())
        ->waitAndThrow();
  }
}

} // namespace rpc
} // namespace distributed
} // namespace torch
```