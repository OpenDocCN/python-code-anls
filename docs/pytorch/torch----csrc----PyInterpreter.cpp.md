# `.\pytorch\torch\csrc\PyInterpreter.cpp`

```py
//cpp
// 包含所需的头文件，用于 Python 与 C++ 的交互
#include <ATen/core/PythonFallbackKernel.h>
#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_dispatch.h>

// 包含标准库中的字符串处理
#include <string>

// 使用 torch、at 和 c10 命名空间
using namespace torch;
using namespace at;
using namespace c10;

// 声明匿名命名空间，用于封装实现细节
namespace {

// 定义宏 CONCRETE_GPU_TRACE，用于执行 GPU 跟踪
#define CONCRETE_GPU_TRACE(device_type, func_name, ...)                       \
  // 在进入此宏时设置 TLS（线程局部存储）保护
  at::impl::MaybeSetTLSOnEntryGuard guard;                                    \
  // 如果 Python 已初始化
  if (Py_IsInitialized()) {                                                   \
    // 获取全局解释器锁（GIL），确保线程安全
    pybind11::gil_scoped_acquire gil;                                         \
    try {                                                                     \
      // 如果设备类型是 HIP，则将其伪装为 CUDA，因为 HIP 使用 `torch.cuda` 模块
      if (device_type == at::kHIP) {                                          \
        device_type = at::kCUDA;                                              \
      }                                                                       \
      // 构造模块名，例如 "torch.cuda"，根据设备类型
      std::string module_name = "torch." + DeviceTypeName(device_type, true); \
      // 导入对应的 Python 模块
      py::module mod = py::module::import(module_name.c_str());               \
      // 获取函数对象 hook，并调用其 fire_callbacks 方法
      py::object hook =                                                       \
          mod.attr("_gpu_trace").attr(func_name).attr("fire_callbacks");      \
      hook(__VA_ARGS__);                                                      \
    } catch (const std::exception& e) {                                       \
      // 捕获异常，记录错误日志，包括设备类型和异常信息
      LOG(ERROR) << device_type                                               \
                 << " trace hook execution failed: " << e.what();             \
    }                                                                         \
  }

// 定义结构体 ConcretePyInterpreterVTable，实现 Python 解释器的虚表
struct ConcretePyInterpreterVTable final
  : public c10::impl::PyInterpreterVTable {
  // 声明一个类，继承自 c10::impl::PyInterpreterVTable

  std::string name() const override;
  // 虚函数重写：返回一个字符串，表示名称

  void decref(PyObject* pyobj, bool has_pyobj_slot) const override;
  // 虚函数重写：减少引用计数，接受一个 PyObject 指针和一个 bool 值作为参数

  // TODO: Need to make this work for StorageImpl too. I imagine I'll want to
  // operate upon a PyObjectSlot rather than a TensorImpl
  // TODO 注释：需要使此函数也适用于 StorageImpl。我认为我希望操作一个 PyObjectSlot 而不是一个 TensorImpl

  c10::intrusive_ptr<c10::TensorImpl> detach(
      const c10::TensorImpl* self) const override;
  // 虚函数重写：将 TensorImpl 分离（detach），返回一个 intrusive_ptr<c10::TensorImpl>

  void dispatch(const c10::OperatorHandle& op, torch::jit::Stack* stack)
      const override;
  // 虚函数重写：分发操作，接受一个 OperatorHandle 和一个 torch::jit::Stack 指针

  void reportErrorCallback(PyObject* callback, DispatchKey key) const override;
  // 虚函数重写：报告错误回调，接受一个 PyObject 指针和一个 DispatchKey 作为参数

  void python_dispatcher(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet,
      torch::jit::Stack* stack) const override;
  // 虚函数重写：Python 分发器，接受一个 OperatorHandle、DispatchKeySet 和 torch::jit::Stack 指针

  // NB: this is defined in python_dispatch.cpp
  // 注意：这在 python_dispatch.cpp 中定义
  void python_op_registration_trampoline(
      const c10::OperatorHandle& op,
      c10::DispatchKey key,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack,
      bool with_keyset) const override {
    // 虚函数重写：Python 操作注册桥接函数，接受一个 OperatorHandle、DispatchKey、DispatchKeySet、torch::jit::Stack 指针和一个 bool 值

    torch::impl::dispatch::python_op_registration_trampoline_impl(
        op, key, keyset, stack, with_keyset);
    // 调用 torch::impl::dispatch::python_op_registration_trampoline_impl 函数
  }

  void throw_abstract_impl_not_imported_error(
      std::string opname,
      const char* pymodule,
      const char* context) const override {
    // 虚函数重写：抛出未导入抽象实现错误，接受一个操作名字符串、一个 Python 模块字符串和一个上下文字符串作为参数

    py::gil_scoped_acquire gil;
    // 获取 Python 全局锁

    pybind11::module::import("torch._utils_internal")
        .attr("throw_abstract_impl_not_imported_error")(
            opname, pymodule, context);
    // 导入 torch._utils_internal 模块，并调用其中的 throw_abstract_impl_not_imported_error 函数
  }

  bool is_contiguous(const c10::TensorImpl* self, at::MemoryFormat)
      const override;
  // 虚函数重写：检查张量是否连续，接受一个 TensorImpl 指针和一个 MemoryFormat 作为参数

  bool is_strides_like(const c10::TensorImpl* self, at::MemoryFormat)
      const override;
  // 虚函数重写：检查张量是否与给定内存格式的步长相似，接受一个 TensorImpl 指针和一个 MemoryFormat 作为参数

  bool is_non_overlapping_and_dense(const c10::TensorImpl* self) const override;
  // 虚函数重写：检查张量是否是非重叠和稠密的，接受一个 TensorImpl 指针作为参数

  c10::Device device(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的设备信息，接受一个 TensorImpl 指针作为参数

  int64_t dim(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的维度数，接受一个 TensorImpl 指针作为参数

  c10::IntArrayRef strides(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的步长数组引用，接受一个 TensorImpl 指针作为参数

  c10::IntArrayRef sizes(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的尺寸数组引用，接受一个 TensorImpl 指针作为参数

  c10::SymIntArrayRef sym_sizes(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的对称尺寸数组引用，接受一个 TensorImpl 指针作为参数

  c10::Layout layout(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的布局信息，接受一个 TensorImpl 指针作为参数

  int64_t numel(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的元素总数，接受一个 TensorImpl 指针作为参数

  c10::SymInt sym_numel(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的对称元素总数，接受一个 TensorImpl 指针作为参数

  c10::SymIntArrayRef sym_strides(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的对称步长数组引用，接受一个 TensorImpl 指针作为参数

  c10::SymInt sym_storage_offset(const c10::TensorImpl* self) const override;
  // 虚函数重写：返回张量的对称存储偏移量，接受一个 TensorImpl 指针作为参数

  void trace_gpu_event_creation(at::DeviceType device_type, uintptr_t event)
      const override {
    // 虚函数重写：跟踪 GPU 事件的创建，接受一个设备类型和一个事件指针作为参数

    CONCRETE_GPU_TRACE(device_type, "EventCreationCallbacks", event);
    // 调用 CONCRETE_GPU_TRACE 宏，记录 GPU 事件的创建
  }

  void trace_gpu_event_deletion(at::DeviceType device_type, uintptr_t event)
      const override {
    // 虚函数重写：跟踪 GPU 事件的删除，接受一个设备类型和一个事件指针作为参数

    CONCRETE_GPU_TRACE(device_type, "EventDeletionCallbacks", event);
    // 调用 CONCRETE_GPU_TRACE 宏，记录 GPU 事件的删除
  }

  void trace_gpu_event_record(
      at::DeviceType device_type,
      uintptr_t event,
      uintptr_t stream) const override {
    // 虚函数重写：跟踪 GPU 事件的记录，接受一个设备类型、事件指针和流指针作为参数
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 事件的回调函数。参数包括设备类型、事件、流。
    void trace_gpu_event_record(
        at::DeviceType device_type,
        uintptr_t event,
        uintptr_t stream) const override {
      CONCRETE_GPU_TRACE(device_type, "EventRecordCallbacks", event, stream);
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 事件等待的回调函数。参数包括设备类型、事件、流。
    void trace_gpu_event_wait(
        at::DeviceType device_type,
        uintptr_t event,
        uintptr_t stream) const override {
      CONCRETE_GPU_TRACE(device_type, "EventWaitCallbacks", event, stream);
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 内存分配的回调函数。参数包括设备类型和指针。
    void trace_gpu_memory_allocation(at::DeviceType device_type, uintptr_t ptr)
        const override {
      CONCRETE_GPU_TRACE(device_type, "MemoryAllocationCallbacks", ptr);
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 内存释放的回调函数。参数包括设备类型和指针。
    void trace_gpu_memory_deallocation(at::DeviceType device_type, uintptr_t ptr)
        const override {
      CONCRETE_GPU_TRACE(device_type, "MemoryDeallocationCallbacks", ptr);
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 流创建的回调函数。参数包括设备类型和流的指针。
    void trace_gpu_stream_creation(at::DeviceType device_type, uintptr_t stream)
        const override {
      CONCRETE_GPU_TRACE(device_type, "StreamCreationCallbacks", stream);
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 设备同步的回调函数。参数包括设备类型。
    void trace_gpu_device_synchronization(
        at::DeviceType device_type) const override {
      CONCRETE_GPU_TRACE(device_type, "DeviceSynchronizationCallbacks");
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 流同步的回调函数。参数包括设备类型和流的指针。
    void trace_gpu_stream_synchronization(
        at::DeviceType device_type,
        uintptr_t stream) const override {
      CONCRETE_GPU_TRACE(device_type, "StreamSynchronizationCallbacks", stream);
    }
    
    // 调用宏 CONCRETE_GPU_TRACE，记录 GPU 事件同步的回调函数。参数包括设备类型和事件。
    void trace_gpu_event_synchronization(
        at::DeviceType device_type,
        uintptr_t event) const override {
      CONCRETE_GPU_TRACE(device_type, "EventSynchronizationCallbacks", event);
    }
    
    // 重置反向钩子（backward hooks），这里未提供具体实现。
    
    // 静态函数，返回单例实例的指针。
    static ConcretePyInterpreterVTable* instance() {
      static ConcretePyInterpreterVTable s;
      return &s;
    }
};

// PyInterpreterHolder 类定义
class PyInterpreterHolder {
 public:
  // 构造函数，初始化 PyInterpreter 对象和是否为主解释器标志
  PyInterpreterHolder()
      : impl_(new c10::impl::PyInterpreter(
            ConcretePyInterpreterVTable::instance())),
        is_main_interpreter_(
            at::impl::PythonOpRegistrationTrampoline::registerInterpreter(
                impl_)) {}

  // 析构函数，释放 PyInterpreter 对象资源
  // 注意：PyInterpreter 对象故意泄漏，因为可能仍有对象引用它，这些对象在 Python 清理期间不会被销毁。
  ~PyInterpreterHolder() {
    impl_->disarm();
  }

  // 获取当前 PyInterpreter 对象指针
  c10::impl::PyInterpreter* get() const noexcept {
    return impl_;
  }

  // 检查当前是否为主解释器
  bool is_main_interpreter() const noexcept {
    return is_main_interpreter_;
  }

 private:
  c10::impl::PyInterpreter* impl_; // PyInterpreter 对象指针
  bool is_main_interpreter_;       // 是否为主解释器标志
};

// torchDispatchFromTensorImpl 函数定义
py::object torchDispatchFromTensorImpl(
    const c10::TensorImpl* self,
    const char* func_name,
    PyObject* torch_api_function,
    const char* module_name,
    // 警告：不应是张量参数
    c10::SmallVector<py::object, 1> extra_args = {}) {

  // 检查 torch_api_function 是否为空，若为空则抛出 python_error 异常
  if (torch_api_function == nullptr) {
    throw python_error();
  }

  // 检查是否已经获取了全局解释器锁（GIL）
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");

  // 创建一个空的 PyObject 指针向量
  std::vector<PyObject*> overloaded_args;

  // 创建一个 ATen 的 Tensor 对象，用于包装给定的 TensorImpl 对象
  at::Tensor self_t = at::Tensor(
      c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));

  // 将 self_t 转换为 Python 的 py::object 对象
  auto self_p =
      py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(self_t)));

  // 向过载的 tensor 参数列表中添加 self_p 指针
  append_overloaded_tensor(&overloaded_args, self_p.ptr());

  // 创建一个包含额外参数的 Python 元组对象 args
  auto args = py::reinterpret_steal<py::object>(
      PyTuple_New(static_cast<Py_ssize_t>(1 + extra_args.size())));
  PyTuple_SET_ITEM(args.ptr(), 0, self_p.release().ptr());
  int64_t i = 1;
  for (auto& a : extra_args) {
    if (a.ptr() == nullptr)
      throw python_error();
    PyTuple_SET_ITEM(args.ptr(), i, std::move(a).release().ptr());
    i++;
  }

  // 创建一个空的 Python 字典对象 kwargs
  py::dict kwargs;

  // 调用 handle_torch_function_no_python_arg_parser 函数处理 Torch 函数的调度
  return py::reinterpret_steal<py::object>(
      handle_torch_function_no_python_arg_parser(
          overloaded_args,
          args.ptr(),
          kwargs.ptr(),
          func_name,
          torch_api_function,
          module_name,
          TorchFunctionName::TorchDispatch));
}

// 注意：PyInterpreter::decref 函数接受一个 `has_pyobj_slot` 参数
// 在调用 PyInterpreter::decref 之前，我们必须静态地知道 pyobj 是否有 PyObjectSlot。
// - 如果有 PyObjectSlot，我们需要小心处理 PyObject 的复活问题。
// - 如果没有 PyObjectSlot，我们可以自由地 decref。
// 另一种方法是使用 PyObject_IsInstance 来获取这些信息。
// 然而，我们不想冒险出现不正确的情况。
// `__instancecheck__` changing the semantics here.
void ConcretePyInterpreterVTable::decref(PyObject* pyobj, bool has_pyobj_slot)
    const {
  // Leak the pyobj if not initialized.  This can happen if we are running
  // exit handlers that are destructing tensors with residual (owned)
  // PyObjects stored in them.
  // 如果 Python 没有初始化，就不释放 pyobj，可能发生在运行退出处理程序时，
  // 在其中销毁带有残留（拥有的）PyObject的张量。
  if (!Py_IsInitialized())
    return;

  pybind11::gil_scoped_acquire gil;
  // Two possibilities:
  // 1. We are decref-ing an object that has a PyObjectSlot, like a Tensor or
  // Storage. Then we must be careful about PyObject resurrection (see
  // THPVariable_clear).
  // 2. We are decref-ing some other Python object. We don't do
  // PyObject resurrection on non-Tensors, so we just carry on as usual
  // 有两种可能性：
  // 1. 我们正在减少具有 PyObjectSlot 的对象，比如张量或存储。那么我们必须小心处理 PyObject 的复活（参见 THPVariable_clear）。
  // 2. 我们正在减少其他 Python 对象。我们不会在非张量上进行 PyObject 的复活，所以我们只是像往常一样继续进行。
  if (has_pyobj_slot && Py_REFCNT(pyobj) > 1) {
    if (THPVariable_Check(pyobj)) {
      // It's still alive!  This can happen if a weak ref resurrected
      // the PyObject without flipping ownership.  At this point it is
      // too late to rescue the object, so just stub out the PyObject
      // so that it fails on subsequent uses.  Don't raise an error here;
      // you're probably in a destructor.
      // 对象仍然存活！这可能是因为弱引用在没有翻转所有权的情况下复活了 PyObject。
      // 在这一点上已经太晚拯救对象了，因此只需将 PyObject 设置为存根状态，
      // 以便在后续使用中失败。不要在这里引发错误；你可能在析构函数中。
      TORCH_WARN(
          "Deallocating Tensor that still has live PyObject references.  "
          "This probably happened because you took out a weak reference to "
          "Tensor and didn't call _fix_weakref() after dereferencing it.  "
          "Subsequent accesses to this tensor via the PyObject will now fail.");
      ((THPVariable*)pyobj)->cdata =
          c10::MaybeOwned<torch::autograd::Variable>();
    } else if (THPStorage_Check(pyobj)) {
      TORCH_WARN(
          "Deallocating UntypedStorage that still has live PyObject references.  "
          "This probably happened because you took out a weak reference to "
          "UntypedStorage and didn't call _fix_weakref() after dereferencing it.  "
          "Subsequent accesses to this storage via the PyObject will now fail.");
      ((THPStorage*)pyobj)->cdata = c10::MaybeOwned<c10::Storage>();
    }
  }
  // 减少 Python 对象的引用计数
  Py_DECREF(pyobj);
};

py::handle getTorchApiFunction(const c10::OperatorHandle& op) {
  return op.getPythonOp(getPyInterpreter(), [&]() -> PyObject* {
    // Parse the name into namespace and name (no overload_name)
    // TODO: put this into the library
    // 解析操作符名称为命名空间和名称（没有 overload_name）
    // TODO: 将此放入库中
    const auto& schema = op.schema();
    const auto& qualified_name = op.operator_name().name;
    const auto& overload_name = schema.overload_name();
    auto pos = qualified_name.find("::");
    TORCH_INTERNAL_ASSERT(pos != std::string::npos, qualified_name);
    // Make me some null terminated strings
    // 创建以空字符结尾的字符串
    std::string ns_str = qualified_name.substr(0, pos);
    const char* ns = ns_str.c_str();
    const char* func_name = qualified_name.c_str() + pos + strlen("::");

    // 获取 Torch API 函数的句柄
    py::handle torch_api_function =
        py::module::import("torch").attr("ops").attr(ns).attr(func_name);
    if (overload_name.empty()) {
      return torch_api_function.attr("default").ptr();
    } else {
      // 如果不是成员函数，则返回对应的 Torch API 函数对象指针
      return torch_api_function.attr(overload_name.c_str()).ptr();
    }
  });
}

bool isPythonTensor(const at::Tensor& tensor) {
  // 检查张量的实现是否具有 Python 的分发键
  return tensor.unsafeGetTensorImpl()->key_set().has(c10::DispatchKey::Python);
}

void ConcretePyInterpreterVTable::reportErrorCallback(
    PyObject* callback,
    DispatchKey key) const {
  // 获取全局解释器锁，以确保在 Python 中调用回调时的线程安全性
  py::gil_scoped_acquire g;
  // 将 C++ 的 PyObject* 转换为 pybind11 的 py::object
  auto func = py::reinterpret_borrow<py::object>(callback);
  // 向 Python 报告错误，传递 DispatchKey 对应的字符串描述
  func(c10::toString(key));
}

void ConcretePyInterpreterVTable::dispatch(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) const {
  // 获取操作的 schema
  const auto& schema = op.schema();
  // 获取参数的数量
  const auto num_arguments = schema.arguments().size();
  // 从堆栈中弹出所有参数
  auto arguments = torch::jit::pop(*stack, num_arguments);

  // 获取全局解释器锁，以确保在调用 Python API 时的线程安全性
  py::gil_scoped_acquire g;

  // 创建用于存储重载张量的 PyObject 指针向量
  std::vector<PyObject*> overloaded_args;
  // 获取与操作关联的 Torch API 函数的 pybind11 句柄
  py::handle torch_api_function_overload = getTorchApiFunction(op);

  // 遍历参数，寻找被 Python 重载的张量
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      const auto& tensor = ivalue.toTensor();
      if (isPythonTensor(tensor)) {
        // 将 Python 张量转换为 PyObject* 并添加到重载参数列表中
        append_overloaded_tensor(&overloaded_args, py::cast(tensor).ptr());
      }
    } else if (ivalue.isList()) {
      const auto& list = ivalue.toListRef();
      for (const auto jdx : c10::irange(list.size())) {
        const auto& nv = list[jdx];
        if (nv.isTensor()) {
          const auto& tensor = nv.toTensor();
          if (isPythonTensor(tensor)) {
            // 将 Python 张量转换为 PyObject* 并添加到重载参数列表中
            append_overloaded_tensor(&overloaded_args, py::cast(tensor).ptr());
          }
        }
      }
    }
  }

  // 将 IValue 参数解析为 Python 的 args 和 kwargs
  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  auto args = std::move(args_kwargs.first);
  auto kwargs = std::move(args_kwargs.second);

  // 调用 Torch 函数处理器，处理无 Python 参数解析的 Torch 函数调用
  PyObject* obj = handle_torch_function_no_python_arg_parser(
      overloaded_args,
      args.ptr(),
      kwargs.ptr(),
      nullptr,
      torch_api_function_overload.ptr(),
      nullptr,
      TorchFunctionName::TorchDispatch);
  // 将 Python 对象推送到堆栈中，用于 Torch 的分发
  pushPyOutToStack(
      op, stack, py::reinterpret_steal<py::object>(obj), "__torch_dispatch__");
}

void ConcretePyInterpreterVTable::python_dispatcher(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet ks,
    torch::jit::Stack* stack) const {
  // 获取全局解释器锁，以确保在 Python 中调用分派器时的线程安全性
  py::gil_scoped_acquire g;
  // 获取与操作关联的 Torch API 函数的 pybind11 句柄
  py::handle torch_api_function_overload = getTorchApiFunction(op);
  // TODO: 如果需要，可以优化以缓存分派缓存查找
  // TODO: 如果需要，可以优化 OpOverload 以具有插槽
  auto cache = py::dict(torch_api_function_overload.attr("_dispatch_cache"));
  if (cache.ptr() == nullptr) {
    throw python_error();
  }



  c10::DispatchKey k = ks.highestPriorityTypeId();
  // 获取最高优先级的调度键
  auto handler = py::reinterpret_borrow<py::object>(
      PyDict_GetItem(cache.ptr(), py::cast(k).ptr()));
  // 从缓存中获取处理函数对象，这里用到了解释Python对象为C++对象的方法
  if (handler.ptr() == nullptr) {
    // 如果未找到处理函数对象，则使用慢速路径获取
    handler = torch_api_function_overload.attr("_get_dispatch")(k);
  }
  // 如果处理函数对象是调度键类型
  if (py::isinstance<c10::DispatchKey>(handler)) {
    // 执行调度操作，但不进行重新调度，因为这会永久移除后续重新调度的Python分派器
    op.callBoxedForDispatchKey(py::cast<c10::DispatchKey>(handler), *stack);
    return;
  }



  const auto& schema = op.schema();
  // 获取操作的模式（schema）
  const auto num_arguments = schema.arguments().size();
  // 获取参数的数量
  auto arguments = torch::jit::pop(*stack, num_arguments);
  // 从堆栈中弹出指定数量的参数

  auto args_kwargs = parseIValuesToPyArgsKwargs(op, arguments);
  // 将IValues解析为Python函数的args和kwargs
  auto args = std::move(args_kwargs.first);
  auto kwargs = std::move(args_kwargs.second);

  py::object obj = py::reinterpret_steal<py::object>(
      PyObject_Call(handler.ptr(), args.ptr(), kwargs.ptr()));
  // 调用处理函数对象，传递args和kwargs作为参数，并获取返回的Python对象

  if (obj.ptr() == nullptr) {
    // 如果返回的Python对象为空指针，抛出Python错误
    throw python_error();
  }

  pushPyOutToStack(op, stack, std::move(obj), "Python dispatcher");
  // 将Python对象推送回堆栈，用于操作，标记为“Python dispatcher”
}

c10::intrusive_ptr<c10::TensorImpl> ConcretePyInterpreterVTable::detach(
    const c10::TensorImpl* self) const {
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁，确保线程安全
  at::impl::MaybeSetTLSOnEntryGuard guard;  // 设置TLS（线程局部存储）入口保护

  auto out = torchDispatchFromTensorImpl(
      self,
      "detach",
      py::module::import("torch")  // 导入torch模块
          .attr("ops")  // 获取ops命名空间
          .attr("aten")  // 获取aten子模块
          .attr("detach")  // 获取detach函数
          .attr("default")  // 获取默认版本的detach函数对象
          .ptr(),
      "torch.ops.aten");  // 指定函数在torch.ops.aten命名空间下的位置

  TORCH_CHECK(
      THPVariable_Check(out.ptr()),  // 检查返回的对象是否是THPVariable类型
      "detach returned invalid type ",  // 若不是THPVariable类型，抛出异常信息
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),  // 获取对象的完全限定类型名
      ", expected Tensor");
  const at::Tensor& res_t = THPVariable_Unpack(out.ptr());  // 解包THPVariable对象为at::Tensor类型
  return res_t.getIntrusivePtr();  // 返回at::Tensor对象的侵入式指针
}

bool ConcretePyInterpreterVTable::is_contiguous(
    const c10::TensorImpl* self,
    at::MemoryFormat memory_format) const {
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁，确保线程安全
  at::impl::MaybeSetTLSOnEntryGuard guard;  // 设置TLS（线程局部存储）入口保护

  py::object out;
  if (memory_format == at::MemoryFormat::Contiguous) {
    // 用于向后兼容
    out = torchDispatchFromTensorImpl(
        self,
        "is_contiguous",
        py::module::import("torch")  // 导入torch模块
            .attr("ops")  // 获取ops命名空间
            .attr("aten")  // 获取aten子模块
            .attr("is_contiguous")  // 获取is_contiguous函数
            .attr("default")  // 获取默认版本的is_contiguous函数对象
            .ptr(),
        "torch.ops.aten");  // 指定函数在torch.ops.aten命名空间下的位置
  } else {
    out = torchDispatchFromTensorImpl(
        self,
        "is_contiguous",
        py::module::import("torch")  // 导入torch模块
            .attr("ops")  // 获取ops命名空间
            .attr("aten")  // 获取aten子模块
            .attr("is_contiguous")  // 获取is_contiguous函数
            .attr("memory_format")  // 获取指定内存格式版本的is_contiguous函数对象
            .ptr(),
        "torch.ops.aten",
        {py::cast(memory_format)});  // 将内存格式参数转换为Python对象并传递给函数
  }

  if (out.is_none()) {
    return self->is_contiguous_default(memory_format);  // 如果返回值为空，则使用默认实现
  }

  TORCH_CHECK(
      PyBool_Check(out.ptr()),  // 检查返回的对象是否是Python中的bool类型
      "is_contiguous returned invalid type ",  // 若不是bool类型，抛出异常信息
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),  // 获取对象的完全限定类型名
      ", expected bool");

  return PyObject_IsTrue(out.ptr());  // 返回Python对象的真值表示
}

bool ConcretePyInterpreterVTable::is_strides_like(
    const c10::TensorImpl* self,
    at::MemoryFormat memory_format) const {
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁，确保线程安全
  at::impl::MaybeSetTLSOnEntryGuard guard;  // 设置TLS（线程局部存储）入口保护

  auto out = torchDispatchFromTensorImpl(
      self,
      "is_strides_like",
      py::module::import("torch")  // 导入torch模块
          .attr("ops")  // 获取ops命名空间
          .attr("aten")  // 获取aten子模块
          // 注意：故意加上_format后缀以避免与"_like"后缀匹配
          .attr("is_strides_like_format")  // 获取is_strides_like_format函数
          .attr("default")  // 获取默认版本的is_strides_like_format函数对象
          .ptr(),
      "torch.ops.aten",
      {py::cast(memory_format)});  // 将内存格式参数转换为Python对象并传递给函数

  if (out.is_none()) {
    return self->is_strides_like_default(memory_format);  // 如果返回值为空，则使用默认实现
  }

  TORCH_CHECK(
      PyBool_Check(out.ptr()),  // 检查返回的对象是否是Python中的bool类型
      "is_strides_like_format returned invalid type ",  // 若不是bool类型，抛出异常信息
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),  // 获取对象的完全限定类型名
      ", expected bool");

  return PyObject_IsTrue(out.ptr());  // 返回Python对象的真值表示
}

bool ConcretePyInterpreterVTable::is_non_overlapping_and_dense(
    // 获取全局解释器锁（Global Interpreter Lock，GIL），确保线程安全
    pybind11::gil_scoped_acquire gil;
    // 设置TLS（线程局部存储）的入口保护
    at::impl::MaybeSetTLSOnEntryGuard guard;

    // 调用 torchDispatchFromTensorImpl 函数，执行对 self 的 tensor dispatch
    auto out = torchDispatchFromTensorImpl(
        self,
        "is_non_overlapping_and_dense",
        // 获取 torch.ops.aten.is_non_overlapping_and_dense.default 对象
        py::module::import("torch")
            .attr("ops")
            .attr("aten")
            .attr("is_non_overlapping_and_dense")
            .attr("default")
            .ptr(),
        "torch.ops.aten");

    // 如果 out 为 None，则调用 self->is_non_overlapping_and_dense_default() 方法返回结果
    if (out.is_none()) {
        return self->is_non_overlapping_and_dense_default();
    }

    // 检查 out 的类型是否为 PyBoolObject，如果不是则抛出错误信息
    TORCH_CHECK(
        PyBool_Check(out.ptr()),
        "is_non_overlapping_and_dense returned invalid type ",
        py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
        ", expected bool");

    // 返回 out.ptr() 对象的布尔值表示
    return PyObject_IsTrue(out.ptr());
}

// ConcretePyInterpreterVTable::dim 方法实现
int64_t ConcretePyInterpreterVTable::dim(const c10::TensorImpl* self) const {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 设置 TLS 守卫
  at::impl::MaybeSetTLSOnEntryGuard guard;

  // 使用 torchDispatchFromTensorImpl 函数调用对应的 dim 方法
  auto out = torchDispatchFromTensorImpl(
      self,
      "dim",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("dim")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 检查返回值类型是否为 PyLong
  TORCH_CHECK(
      PyLong_Check(out.ptr()),
      "dim returned invalid type ",
      py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
      ", expected int");

  // 将 PyLong 转换为 int64_t 并返回
  return THPUtils_unpackLong(out.ptr());
}

// ConcretePyInterpreterVTable::device 方法实现
c10::Device ConcretePyInterpreterVTable::device(const c10::TensorImpl* self) const {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 设置 TLS 守卫
  at::impl::MaybeSetTLSOnEntryGuard guard;

  // 使用 torchDispatchFromTensorImpl 函数调用对应的 device 方法
  auto out = torchDispatchFromTensorImpl(
      self,
      "device",
      py::module::import("torch")
          .attr("ops")
          .attr("prim")
          .attr("device")
          .attr("default")
          .ptr(),
      "torch.ops.prim");

  // 将 Python 对象转换为 c10::Device 对象并返回
  return toDevice(out.ptr());
}

// set_tensor_attr_with_capsule 函数实现
static void set_tensor_attr_with_capsule(
    const c10::TensorImpl* tensor,
    py::capsule& capsule,
    const char* attr_name) {
  // 检查是否存在 Python 对象
  std::optional<PyObject*> mb_obj = tensor->pyobj_slot()->check_pyobj(
      getPyInterpreter(), /*ignore_hermetic_tls=*/false);
  // 如果不存在 Python 对象，抛出异常
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");
  auto obj = mb_obj.value();
  // 设置对象的属性为 capsule
  py::handle(obj).attr(attr_name) = capsule;
}

// get_set_cached_attr 模板函数实现
// 注释见下方
template <typename T>
static c10::ArrayRef<T> get_set_cached_attr(
    const c10::TensorImpl* tensor,
    const char* base_attr_name,


注释：


// Note [Tensor Subclass custom size/stride caching strategy]
// Tensor subclasses can use __torch_dispatch__ to override size/stride calls.
// However, this presents a problem:
// (1) When you return a custom (maybe symbolic) size/stride
//     from python, we need to stash this fresh vector of ints/symints
//     somewhere so that it has the same lifetime as the tensor.
// (2) If the subclass experiences a metadata mutation,
//     this stashed vector is no longer valid, so we need to allocate a fresh
//     buffer to store the new sizes the next time someone asks for them.
//
// We handle this in the same way that `TensorImpl::sizes_default()`
// handles its buffer: we simply reallocate the buffer whenever
// the number of dimensions changes due to a resize.
// Notable, we do *not* reallocate the buffer if the values changed,
// but the number of dimensions stayed the same (e.g. `.transpose_()`).
    const py::object& obj) {
  // 检查是否能从 tensor 的 pyobj_slot() 中获取到 PyObject 指针
  std::optional<PyObject*> mb_obj =
      tensor->pyobj_slot()->check_pyobj(getPyInterpreter());
  // 检查 mb_obj 是否有值，如果没有值则抛出异常
  TORCH_CHECK(
      mb_obj.has_value(), "Tensor subclass's PyInterpreter has no value");
  // 获取 tensor_obj，这是一个 PyObject 指针
  auto tensor_obj = mb_obj.value();
  // 构造新的属性名，形如 base_attr_name_len
  auto buffer_len_attr_name = std::string(base_attr_name) + std::string("_len");

  // 初始化标志和当前大小
  bool is_buffer_allocated = false;
  size_t curr_size = 0;
  // 检查 tensor_obj 是否有 buffer_len_attr_name 属性
  if (PyObject_HasAttrString(tensor_obj, buffer_len_attr_name.c_str())) {
    // 获取 len 属性的值，并转换为 size_t 类型
    auto len_pyobj = py::handle(tensor_obj).attr(buffer_len_attr_name.c_str());
    curr_size = py::cast<size_t>(len_pyobj);
    is_buffer_allocated = true;
  }

  // 获取新的大小
  size_t new_size = py::len(obj);

  // 进行小向量优化：当 new_size <= 5 时，始终分配大小为 5 的缓冲区，
  // 这样如果下一次调整大小仍然是 <= 5 个元素，就无需重新分配。
  // 注意：尝试移除此优化时触发 ASAN 错误
  bool needs_resize = false;
  // 需要调整大小的情况包括：
  // (1) 还未分配缓冲区
  // (2) 缓冲区大小与新大小不同
  //     （注意：我们使用小向量优化，即我们的缓冲区总是至少分配大小为 5，
  //     并且任何在 <= 5 范围内的调整大小不需要重新分配）
  auto is_smallvector = curr_size <= 5;
  needs_resize = !is_buffer_allocated || (is_smallvector && new_size > 5) ||
      (!is_smallvector && curr_size != new_size);
  if (needs_resize) {
    // 如果当前缓冲区大小不正确（因为尚未分配，或者有元数据变化改变了张量的维数），则分配新的缓冲区。
    // 注意，如果已经有缓冲区，这将覆盖先前的缓冲区，使来自旧的 .sym_size() 调用的任何现有 SymIntArrayRef 无效。
    auto new_buffer_size = new_size;
    if (new_size <= 5) {
      // 这是小向量优化
      new_buffer_size = 5;
    }
    // 分配新的缓冲区
    T* ptr = new T[new_buffer_size];
    // 创建一个 PyCapsule，负责管理指针数组的内存释放
    auto capsule =
        py::capsule(ptr, [](void* p) { delete[] reinterpret_cast<T*>(p); });
    int64_t idx = 0;
    // 将 obj 中的元素转换为 T 类型，并复制到 ptr 数组中
    for (auto it = obj.begin(); it != obj.end(); ++it, ++idx) {
      ptr[idx] = py::cast<T>(*it);
    }
    // 设置张量的属性，使用 PyCapsule 作为值
    set_tensor_attr_with_capsule(tensor, capsule, base_attr_name);
    // 设置 len 缓冲区的值
    py::handle(tensor_obj).attr(buffer_len_attr_name.c_str()) = new_size;
  } else {
    // 否则，当前缓冲区大小正确，直接使用它
    TORCH_INTERNAL_ASSERT(PyObject_HasAttrString(tensor_obj, base_attr_name));
    // 获取当前缓冲区的 Py 对象
    auto curr_buffer_pyobj = py::handle(tensor_obj).attr(base_attr_name);
    // 获取当前缓冲区对象的指针，使用 PyCapsule_GetPointer 函数
    void* buffer_pycapsule =
        PyCapsule_GetPointer(curr_buffer_pyobj.ptr(), nullptr);
    // 将指针重新解释为类型 T 的指针，即当前缓冲区的类型
    auto curr_buffer = reinterpret_cast<T*>(buffer_pycapsule);

    // 覆盖缓冲区的当前值为新的数值，但仅在任何值发生变化时进行
    // （由于元数据的变化）。
    // 这段代码在技术上不是线程安全的，因为更新是延迟执行的。
    // 张量上的原始元数据变异调用可能是线程安全的（例如 .resize_() 调用），
    // 但直到第一次调用 .sizes() 时才会实际变异大小缓冲区，用户可能以非线程安全的方式访问它。
    // 目前我们没有显式加锁，但也许我们应该加上。
    int64_t idx = 0;
    // 快速检查，确保我们的缓冲区大小足够大，能够与新缓冲区中的所有元素进行比较。
    size_t curr_buffer_size = 5;
    if (curr_buffer_size < curr_size) {
      curr_buffer_size = curr_size;
    }
    // 使用 TORCH_INTERNAL_ASSERT 断言，确保当前缓冲区大小大于等于新的大小。
    TORCH_INTERNAL_ASSERT(curr_buffer_size >= new_size);
    // 遍历对象 obj 的元素
    for (auto it = obj.begin(); it != obj.end(); ++it, ++idx) {
      // 将迭代器指向的元素转换为类型 T
      auto actual_val = py::cast<T>(*it);
      // 如果 T 是 c10::SymInt 类型，如果当前缓冲区的值不同于 actual_val，则更新当前缓冲区的值。
      if constexpr (std::is_same_v<T, c10::SymInt>) {
        if (!curr_buffer[idx].is_same(actual_val)) {
          curr_buffer[idx] = actual_val;
        }
      } else {  // 否则，如果当前缓冲区的值与 actual_val 不相等，则更新当前缓冲区的值。
        if (curr_buffer[idx] != actual_val) {
          curr_buffer[idx] = actual_val;
        }
      }
    }
  }

  // 现在缓冲区中存储了正确的数据 - 读取并返回它。
  // 获取当前缓冲区的 Python 对象，使用 py::handle(tensor_obj).attr(base_attr_name)
  auto curr_buffer_pyobj = py::handle(tensor_obj).attr(base_attr_name);
  // 获取缓冲区对象的指针，使用 PyCapsule_GetPointer 函数
  void* buffer_pycapsule =
      PyCapsule_GetPointer(curr_buffer_pyobj.ptr(), nullptr);
  // 将指针重新解释为类型 T 的指针，即当前缓冲区的类型
  auto curr_buffer = reinterpret_cast<T*>(buffer_pycapsule);
  // 返回类型为 c10::ArrayRef<T> 的对象，表示指向 curr_buffer 的数组引用，长度为 new_size
  return c10::ArrayRef<T>(curr_buffer, new_size);
}

c10::IntArrayRef ConcretePyInterpreterVTable::strides(
    const c10::TensorImpl* self) const {
  // 保证在调用Python解释器时全局解释器锁已获取
  pybind11::gil_scoped_acquire gil;
  // 确保进入函数时设置了TLS（线程局部存储）守卫
  at::impl::MaybeSetTLSOnEntryGuard guard;

  // 调用torchDispatchFromTensorImpl函数，从TensorImpl获取stride信息
  auto out = torchDispatchFromTensorImpl(
      self,
      "stride",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("stride")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果未找到对应操作，检查是否有符号化形状或步幅，若有则报错
  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call strides on a tensor with symbolic shapes/strides");
    // 返回默认的步幅信息
    return self->strides_default();
  }

  // 检查返回的out对象是否为列表或元组类型
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "strides must be a list or a tuple");

  // 更新步幅信息，并将结果缓存起来
  auto updated_strides =
      get_set_cached_attr<int64_t>(self, "_strides_capsule", out);
  return updated_strides;
}

c10::IntArrayRef ConcretePyInterpreterVTable::sizes(
    const c10::TensorImpl* self) const {
  // 保证在调用Python解释器时全局解释器锁已获取
  pybind11::gil_scoped_acquire gil;
  // 确保进入函数时设置了TLS（线程局部存储）守卫
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 处理Tensor错误的宏
  HANDLE_TH_ERRORS

  // 调用torchDispatchFromTensorImpl函数，从TensorImpl获取size信息
  auto out = torchDispatchFromTensorImpl(
      self,
      "size",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("size")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果未找到对应操作，检查是否有符号化形状或步幅，若有则报错
  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call sizes on a tensor with symbolic shapes/strides");
    // 返回默认的大小信息
    return self->sizes_default();
  }

  // 检查返回的out对象是否为列表或元组类型
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "sizes must be a list or a tuple");

  // 更新大小信息，并将结果缓存起来
  auto updated_sizes =
      get_set_cached_attr<int64_t>(self, "_sizes_capsule", out);
  return updated_sizes;

  // 结束处理Tensor错误的宏
  END_HANDLE_TH_ERRORS_PYBIND
}

c10::SymIntArrayRef ConcretePyInterpreterVTable::sym_sizes(
    const c10::TensorImpl* self) const {
  // 保证在调用Python解释器时全局解释器锁已获取
  pybind11::gil_scoped_acquire gil;
  // 确保进入函数时设置了TLS（线程局部存储）守卫
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 处理Tensor错误的宏
  HANDLE_TH_ERRORS

  // 调用torchDispatchFromTensorImpl函数，从TensorImpl获取sym_size信息
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_size",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_size")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果未找到对应操作，返回默认的符号化大小信息
  if (out.is_none()) {
    return self->sym_sizes_default();
  }

  // 检查返回的out对象是否为列表或元组类型
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "sym_size must be a list or a tuple");

  // 使用自定义的缓存策略更新符号化大小信息，并将结果缓存起来
  // 参见Note [Tensor Subclass custom size/stride caching strategy]
  auto updated_sym_sizes =
      get_set_cached_attr<c10::SymInt>(self, "_sym_sizes_capsule", out);
  return updated_sym_sizes;

  // 结束处理Tensor错误的宏
  END_HANDLE_TH_ERRORS_PYBIND
}

c10::Layout ConcretePyInterpreterVTable::layout(
    // 从给定的 TensorImpl 中获取布局信息
    const c10::TensorImpl* self) const {
      // 获取全局解释器锁，确保线程安全
      pybind11::gil_scoped_acquire gil;
      // 设置 TLS 入口守卫，处理 TLS 相关事务
      at::impl::MaybeSetTLSOnEntryGuard guard;
      // 调用 torchDispatchFromTensorImpl 函数，通过 TensorImpl 对象获取布局信息
      auto out = torchDispatchFromTensorImpl(
          self,
          "layout",
          // 导入 torch.ops.prim.layout.default 模块，并获取其指针
          py::module::import("torch")
              .attr("ops")
              .attr("prim")
              .attr("layout")
              .attr("default")
              .ptr(),
          "torch.ops.prim");
    
      // 检查返回值类型，确保是有效的布局类型或者整数类型
      TORCH_CHECK(
          THPLayout_Check(out.ptr()) || PyLong_Check(out.ptr()),
          "layout returned invalid type ",
          // 获取返回对象的类型名称
          py::detail::get_fully_qualified_tp_name(Py_TYPE(out.ptr())),
          ", expected Layout");
    
      // 如果返回的是 THPLayout 对象，则转换为 c10::Layout 类型并返回
      if (THPLayout_Check(out.ptr())) {
        return toLayout(out.ptr());
      } else {
        // 否则，将返回的 PyObject 转换为 int64_t 类型，然后构造成 c10::Layout 类型返回
        return c10::Layout(py::cast<int64_t>(out));
      }
    }
}

// 定义 ConcretePyInterpreterVTable 类的方法 numel，返回张量的元素数
int64_t ConcretePyInterpreterVTable::numel(const c10::TensorImpl* self) const {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 设置 TLS（线程本地存储）进入保护
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 调用 torchDispatchFromTensorImpl 函数，根据张量实现对象调度到对应的 torch 操作
  auto out = torchDispatchFromTensorImpl(
      self,
      "numel",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("numel")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果返回结果为 None，则检查张量是否具有符号大小和步幅
  if (out.is_none()) {
    TORCH_CHECK(
        !self->has_symbolic_sizes_strides(),
        "Cannot call sizes on a tensor with symbolic shapes/strides");
    // 返回张量默认的元素数
    return self->numel_default();
  }
  // 将输出结果转换为 int64_t 类型并返回
  return py::cast<int64_t>(out);
}

// 定义 ConcretePyInterpreterVTable 类的方法 sym_numel，返回张量的符号元素数
c10::SymInt ConcretePyInterpreterVTable::sym_numel(
    const c10::TensorImpl* self) const {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 设置 TLS（线程本地存储）进入保护
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 调用 torchDispatchFromTensorImpl 函数，根据张量实现对象调度到对应的 torch 操作
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_numel",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_numel")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果返回结果为 None，则返回张量的符号元素数的默认值
  if (out.is_none()) {
    return self->sym_numel_default();
  }
  // 如果输出结果是符号整数，则转换并返回；否则将其作为 int64_t 转换后返回
  return torch::is_symint(out) ? out.cast<c10::SymInt>()
                               : c10::SymInt{py::cast<int64_t>(out)};
}

// 定义 ConcretePyInterpreterVTable 类的方法 sym_storage_offset，返回张量的符号存储偏移量
c10::SymInt ConcretePyInterpreterVTable::sym_storage_offset(
    const c10::TensorImpl* self) const {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 设置 TLS（线程本地存储）进入保护
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 调用 torchDispatchFromTensorImpl 函数，根据张量实现对象调度到对应的 torch 操作
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_storage_offset",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_storage_offset")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果返回结果为 None，则返回张量的符号存储偏移量的默认值
  if (out.is_none()) {
    return self->sym_storage_offset_default();
  }
  // 如果输出结果是符号整数，则转换并返回；否则将其作为 int64_t 转换后返回
  return torch::is_symint(out) ? out.cast<c10::SymInt>()
                               : c10::SymInt{py::cast<int64_t>(out)};
}

// 定义 ConcretePyInterpreterVTable 类的方法 sym_strides，返回张量的符号步幅数组引用
c10::SymIntArrayRef ConcretePyInterpreterVTable::sym_strides(
    const c10::TensorImpl* self) const {
  // 获取全局解释器锁，确保线程安全
  pybind11::gil_scoped_acquire gil;
  // 设置 TLS（线程本地存储）进入保护
  at::impl::MaybeSetTLSOnEntryGuard guard;
  // 处理 Torch 异常错误
  HANDLE_TH_ERRORS
  // 调用 torchDispatchFromTensorImpl 函数，根据张量实现对象调度到对应的 torch 操作
  auto out = torchDispatchFromTensorImpl(
      self,
      "sym_stride",
      py::module::import("torch")
          .attr("ops")
          .attr("aten")
          .attr("sym_stride")
          .attr("default")
          .ptr(),
      "torch.ops.aten");

  // 如果返回结果为 None，则返回张量的符号步幅数组引用的默认值
  if (out.is_none()) {
    return self->sym_strides_default();
  }
  // 检查输出是否为元组或列表，因为 sym_strides 必须是列表或元组
  TORCH_CHECK(
      py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out),
      "sym_strides must be a list or a tuple");

  // 更新和缓存 _sym_strides_capsule 属性中的 c10::SymInt 元素
  auto updated_sym_strides =
      get_set_cached_attr<c10::SymInt>(self, "_sym_strides_capsule", out);
  // 返回更新后的符号步幅数组引用
  return updated_sym_strides;
  // 处理 Torch 异常错误的结束
  END_HANDLE_TH_ERRORS_PYBIND
}

// 定义 ConcretePyInterpreterVTable 类的方法 reset_backward_hooks，重置后向钩子
    // 以 const c10::TensorImpl* self 参数作为输入，函数返回 void
  pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁（GIL），确保线程安全
  at::impl::MaybeSetTLSOnEntryGuard guard;  // 设置 TLS（线程局部存储）进入守卫
  HANDLE_TH_ERRORS  // 开始处理 Torch 错误
  Tensor self_t =
      Tensor(c10::intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>::
                 // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
             unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(self)));
    // 通过 const_cast 去除 self 的 const 限定，将其转换为 Tensor 对象 self_t
  auto self_p =
      py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(self_t)));
    // 将 self_t 转换为 Python 对象 self_p，并且确保 C++ 与 Python 对象之间的所有权正确转移
  PyObject_SetAttrString(self_p.ptr(), "_backward_hooks", Py_None);
    // 设置 self_p 对象的 "_backward_hooks" 属性为 Python 中的 None
  END_HANDLE_TH_ERRORS_PYBIND
    // 结束处理 Torch 错误的块，使用 Pybind 包装
}

PyInterpreterHolder self_interpreter;


} // anonymous namespace


c10::impl::PyInterpreter* getPyInterpreter() {
  return self_interpreter.get();
}


bool isMainPyInterpreter() {
  return self_interpreter.is_main_interpreter();
}


std::string ConcretePyInterpreterVTable::name() const {
  // 创建一个字符串流对象
  std::stringstream ss;
  // 将当前 Python 解释器的地址输出到字符串流中
  ss << getPyInterpreter();
  // 将字符串流中的内容转换成字符串并返回
  return ss.str();
}
```