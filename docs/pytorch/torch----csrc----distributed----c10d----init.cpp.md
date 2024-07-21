# `.\pytorch\torch\csrc\distributed\c10d\init.cpp`

```
// 包含 Torch C++ 和 Python 之间交互的头文件
#include <torch/csrc/python_headers.h>

// 包含 C10 库中的便利工具，如 intrusive_ptr 和 string_view
#include <c10/util/intrusive_ptr.h>
#include <c10/util/string_view.h>

// 包含分布式相关的文件存储、组管理和 TCP 通信的头文件
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/ControlCollectives.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/StoreCollectives.hpp>
#include <torch/csrc/distributed/c10d/control_plane/WorkerServer.hpp>

// 包含标准库的 vector 头文件
#include <vector>

// 如果不是在 Windows 平台下编译，包含哈希存储和 RoundRobin 进程组的头文件
#ifndef _WIN32
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupRoundRobin.hpp>
#endif

// 包含虚拟进程组和基础进程组的头文件
#include <torch/csrc/distributed/c10d/FakeProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/PyProcessGroup.hpp>

// 根据预编译宏 USE_C10D_GLOO 的定义，包含 Gloo 进程组和进程组包装器的头文件
#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupWrapper.hpp>
#endif

// 根据预编译宏 USE_C10D_NCCL 的定义，包含 NCCL 工具和 NCCL 进程组的头文件
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>
#endif

// 根据预编译宏 USE_C10D_MPI 的定义，包含 MPI 进程组的头文件
#ifdef USE_C10D_MPI
#include <torch/csrc/distributed/c10d/ProcessGroupMPI.hpp>
#endif

// 根据预编译宏 USE_C10D_UCC 的定义，包含 UCC 进程组的头文件
#ifdef USE_C10D_UCC
#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>
#endif

// 包含 fmt 库提供的格式化输出的头文件和 pybind11 的时间处理头文件
#include <fmt/format.h>
#include <pybind11/chrono.h>

// 包含 Torch 分布式库中的 DMA 连接性、前缀存储和对称内存的头文件
#include <torch/csrc/distributed/c10d/DMAConnectivity.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/SymmetricMemory.hpp>

// 包含 Torch 分布式库中的通信、调试和日志头文件，以及分布式 reducer 头文件
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/debug.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/reducer.hpp>

// 包含 Torch 异常处理和 Python 通信钩子的头文件，以及 Torch Python 绑定工具头文件
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>
#include <torch/csrc/jit/python/pybind_utils.h>

// 包含 Torch 自定义类的头文件和 Torch 工具的对象指针头文件和 Python 绑定头文件
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/custom_class.h>

// 匿名命名空间用于定义内部链接的静态变量和函数

// 根据预编译宏 USE_C10D_NCCL 的定义，定义了一个函数用于获取全局解释器锁
namespace {
#ifdef USE_C10D_NCCL
bool acquire_gil() {
  // 如果 Python 解释器已初始化，尝试获取全局解释器锁，如果成功返回 true
  // 否则，直接返回 false，此函数用于在可能发生 GIL 竞争时的线程中进行检查
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    return true;
  }

  // 如果 Python 解释器未初始化，直接返回 false
  return false;
}
#endif // USE_C10D_NCCL
} // namespace
// 注册函数，用于设置全局的 GIL 检查器为 acquire_gil 函数，并返回 true 表示注册成功
bool registerGilChecker() {
  c10d::get_gil_checker() = &acquire_gil;
  return true;
}

// 静态变量，调用 registerGilChecker() 函数进行初始化
static bool registered = registerGilChecker();
#endif // USE_C10D_NCCL

// 包装类，确保在析构 ProcessGroupGloo 前释放 GIL
// TODO: 将此功能移至更通用的位置
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // 这个构造函数非常重要；参见 https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      // 如果当前线程拥有 GIL，则释放 GIL 后重置指针
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        // 否则直接重置指针
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // 匿名命名空间结束

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);

namespace torch::distributed::c10d {

namespace {

// 将 vector<uint8_t> 转换为 py::bytes
py::bytes toPyBytes(const std::vector<uint8_t>& data) {
  return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
}

// 将 vector<vector<uint8_t>> 转换为 vector<py::bytes>
std::vector<py::bytes> toPyBytes(
    const std::vector<std::vector<uint8_t>>& data) {
  std::vector<py::bytes> out;
  out.reserve(data.size());
  for (const std::vector<uint8_t>& data_ : data) {
    out.emplace_back(reinterpret_cast<const char*>(data_.data()), data_.size());
  }
  return out;
}

// 将 string 转换为 vector<uint8_t>
std::vector<uint8_t> toVec8(const std::string& data) {
  std::vector<uint8_t> out{data.begin(), data.end()};
  return out;
}

// 将 vector<string> 转换为 vector<vector<uint8_t>>
std::vector<std::vector<uint8_t>> toVec8(const std::vector<std::string>& data) {
  std::vector<std::vector<uint8_t>> out;
  out.reserve(data.size());
  for (auto& data_ : data) {
    out.emplace_back(toVec8(data_));
  }
  return out;
}

// 定义 shared_ptr_class_ 模板别名
template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

// 定义 intrusive_ptr_class_ 模板别名
template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

// 定义 intrusive_ptr_no_gil_destructor_class_ 模板别名
template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;

constexpr auto kDeprecationWarning =
    "{} API is being deprecated, please ping "
    "https://github.com/pytorch/pytorch/issues/46291 "
    "if you see this warning";
// PythonStore is a pybind11 trampoline class to allow a Python
// class to inherit from c10d.Store and implement its interface.
class PythonStore : public ::c10d::Store {
 public:
  using ::c10d::Store::Store;

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that we can call the Python-side
  // function with a std::string instead of a std::vector<uint8_t>.
  // 设置键值对应的数据，调用 Python 侧的重载函数，用 std::string 作为键值的类型
  void set(const std::string& key, const std::vector<uint8_t>& value) override {
    // 获取全局解释器锁，以便在多线程环境下安全地调用 Python
    pybind11::gil_scoped_acquire gil;
    // 获取与当前对象关联的 Python 侧的 set 函数重载
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "set");
    // 断言确保获取到了有效的函数对象
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // 调用函数，传递 py::bytes 对象作为值的表示
    fn(key, toPyBytes(value));
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  // 获取指定键对应的数据，调用 Python 侧的重载函数，返回值类型为 std::vector<uint8_t>
  std::vector<uint8_t> get(const std::string& key) override {
    // 获取全局解释器锁，以便在多线程环境下安全地调用 Python
    pybind11::gil_scoped_acquire gil;
    // 获取与当前对象关联的 Python 侧的 get 函数重载
    pybind11::function fn =
        pybind11::get_overload(static_cast<const ::c10d::Store*>(this), "get");
    // 断言确保获取到了有效的函数对象
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // 调用函数，将 Python 返回的 py::bytes 类型数据转换为 std::string，并构造成 std::vector<uint8_t> 返回
    std::string str = pybind11::cast<py::bytes>(fn(key));
    return toVec8(str);
  }

  // Note: this function manually calls the Python-side overload
  // for this function instead of using the PYBIND11_OVERLOAD_XYZ
  // macros. This is done so that the Python-side function can
  // return a py::bytes instead of a std::vector<uint8_t>.
  // 比较并设置指定键的值，调用 Python 侧的重载函数，返回值类型为 std::vector<uint8_t>
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override {
    // 获取全局解释器锁，以便在多线程环境下安全地调用 Python
    pybind11::gil_scoped_acquire gil;
    // 获取与当前对象关联的 Python 侧的 compare_set 函数重载
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "compare_set");
    // 断言确保获取到了有效的函数对象
    TORCH_INTERNAL_ASSERT(fn, "Not implemented.");
    // 调用函数，将 Python 返回的 py::bytes 类型数据转换为 std::string，并构造成 std::vector<uint8_t> 返回
    std::string str = pybind11::cast<py::bytes>(
        fn(key, toPyBytes(expectedValue), toPyBytes(desiredValue)));
    return toVec8(str);
  }

  // 使用 PYBIND11_OVERLOAD_PURE 宏调用基类 ::c10d::Store 中的纯虚函数 add
  int64_t add(const std::string& key, int64_t value) override {
    PYBIND11_OVERLOAD_PURE(int64_t, ::c10d::Store, add, key, value);
  }

  // 使用 PYBIND11_OVERLOAD 宏调用基类 ::c10d::Store 中的虚函数 getNumKeys
  int64_t getNumKeys() override {
    // 使用 PYBIND11_OVERLOAD_PURE 宏调用 Python 中的 getNumKeys 函数
    PYBIND11_OVERLOAD_PURE(int64_t, ::c10d::Store, getNumKeys);

  }

  // 覆盖基类的 deleteKey 函数，调用对应的 Python 重载版本
  bool deleteKey(const std::string& key) override {
    PYBIND11_OVERLOAD_PURE(bool, ::c10d::Store, deleteKey, key);
  }

  // 覆盖基类的 check 函数，调用对应的 Python 重载版本
  bool check(const std::vector<std::string>& keys) override {
    PYBIND11_OVERLOAD_PURE(bool, ::c10d::Store, check, keys);
  }

  // 覆盖基类的 wait 函数，调用对应的 Python 重载版本
  void wait(const std::vector<std::string>& keys) override {
    PYBIND11_OVERLOAD_PURE(void, ::c10d::Store, wait, keys);
  }

  // 覆盖带超时参数的 wait 函数，调用对应的 Python 重载版本
  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    PYBIND11_OVERLOAD_PURE(void, ::c10d::Store, wait, keys, timeout);
  }

  // 注意：此函数手动调用 Python 侧的重载版本，而非使用 PYBIND11_OVERLOAD_XYZ 宏
  // 目的是使用 std::string 而非 std::vector<uint8_t> 调用 Python 侧函数
  void append(const std::string& key, const std::vector<uint8_t>& value)
      override {
    pybind11::gil_scoped_acquire gil;
    // 获取 Python 侧的重载函数
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "append");
    // 如果没有找到重载版本，则调用基类的 append 函数
    if (!fn) {
      return Store::append(key, value);
    }
    // 使用 py::bytes 对象调用函数
    fn(key, toPyBytes(value));
  }

  // 覆盖 multiGet 函数，调用对应的 Python 重载版本
  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "multi_get");
    if (!fn) {
      return Store::multiGet(keys);
    }
    // 调用 Python 侧 multi_get 函数，并转换结果为 C++ 格式
    std::vector<std::string> py_list =
        pybind11::cast<std::vector<std::string>>(fn(keys));
    std::vector<std::vector<uint8_t>> res;
    res.reserve(py_list.size());

    for (auto& str : py_list) {
      res.emplace_back(str.begin(), str.end());
    }

    return res;
  }

  // 覆盖 multiSet 函数，调用对应的 Python 重载版本
  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    pybind11::gil_scoped_acquire gil;
    pybind11::function fn = pybind11::get_overload(
        static_cast<const ::c10d::Store*>(this), "multi_set");
    if (!fn) {
      return Store::multiSet(keys, values);
    }
    // 使用 py::bytes 对象调用函数
    fn(keys, toPyBytes(values));
  }

  // 判断是否存在扩展 API
  bool hasExtendedApi() const override {
    PYBIND11_OVERLOAD_NAME(
        bool, ::c10d::Store, "has_extended_api", hasExtendedApi);
  }
};

// 从DDP的Python API调用以创建一个c10d Python comm hook对象。
// 输入参数state和comm_hook都是Python对象。之后调用reducer的register_comm_hook函数注册这个钩子。
void _register_comm_hook(
    ::c10d::Reducer& reducer,  // reducer对象的引用
    py::object state,           // Python对象，用于设置comm hook
    py::object comm_hook) {     // Python对象，表示comm hook的可调用对象
  reducer.register_comm_hook(std::make_unique<::c10d::PythonCommHook>(
      std::move(state), std::move(comm_hook)));  // 使用state和comm_hook创建PythonCommHook对象并注册
}

// 从DDP的Python API调用以创建一个c10d C++ comm hook对象。
// 输入参数是一个enum hook type。之后调用reducer的register_builtin_comm_hook函数设置hook类型。
void _register_builtin_comm_hook(
    ::c10d::Reducer& reducer,                  // reducer对象的引用
    ::c10d::BuiltinCommHookType comm_hook_type) {  // 枚举类型，表示内置的comm hook类型
  reducer.register_builtin_comm_hook(comm_hook_type);  // 注册内置的comm hook类型
}

// 自定义::c10d::ReduceOp的元类，以保持向后兼容性。
// https://github.com/pytorch/pytorch/pull/84243将::c10d::ReduceOp从枚举改为了结构体，因此牺牲了一些Python内置函数支持，如`isinstance`和`copy`。
// 下面我们在CPython/pybind11中定义了自定义的`isinstance`（reduceopmeta___instancecheck__），并修改了pybind11的默认元类（GetReduceOpMetaclass），
// 使得`isinstance(torch.distributed.ReduceOp.SUM, torch.distributed.ReduceOp)`返回True，就像`ReduceOp`是一个枚举类型一样。
// 参考：
//   - https://docs.python.org/3/extending/newtypes_tutorial.html
//   - https://docs.python.org/3/c-api/typeobj.html?highlight=tp_methods
//   - https://github.com/pybind/pybind11/issues/2696
static PyObject* reduceopmeta___instancecheck__(
    PyObject* self,    // 对象自身
    PyObject* args) {  // 用于比较的参数对象
  if (Py_TYPE(self) == Py_TYPE(args)) {  // 如果self和args的类型相同
    Py_RETURN_TRUE;  // 返回True
  }
  if (c10::string_view(args->ob_type->tp_name).find("RedOpType") !=
      c10::string_view::npos) {  // 如果args的类型名称中包含"RedOpType"
    Py_RETURN_TRUE;  // 返回True
  }
  Py_RETURN_FALSE;  // 否则返回False
}

// NOLINTNEXTLINE(*c-arrays)
static PyMethodDef reduceopmeta_methods[] = {
    {"__instancecheck__",
     (PyCFunction)reduceopmeta___instancecheck__,  // 指向reduceopmeta___instancecheck__函数的指针
     METH_O,    // 方法接受单个参数
     "Custom `__instancecheck__` for ReduceOp"},  // 方法的文档字符串
    {nullptr, nullptr}  // 结束符
};

// 获取ReduceOp的元类对象。
PyTypeObject* GetReduceOpMetaclass() {
  static auto* metaclass = [] {  // 定义静态局部变量
    PyTypeObject* base_metaclass =  // 获取pybind11的默认元类
        pybind11::detail::get_internals().default_metaclass;
    // NOLINTNEXTLINE(*c-arrays)
    PyType_Slot slots[] = {
        {Py_tp_base, base_metaclass},      // 基类设置为默认元类
        {Py_tp_methods, reduceopmeta_methods},  // 设置自定义方法
        {0},                                // 结束符
    };
    PyType_Spec spec = {};
    spec.name = "torch._C._distributed_c10d._ReduceOpMeta";  // 类名
    // NOLINTNEXTLINE(*-narrowing-conversions)
    spec.basicsize = base_metaclass->tp_basicsize;  // 基本大小与默认元类相同
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;  // 标志
    spec.slots = slots;  // 设置插槽
    PyTypeObject* metaclass = (PyTypeObject*)PyType_FromSpec(&spec);  // 创建元类对象

    // 返回创建的元类对象
    return metaclass;
  }();  // 立即调用Lambda表达式创建元类对象并赋给静态局部变量metaclass

  return metaclass;  // 返回静态局部变量metaclass
}
    // 如果 metaclass 为空，则抛出当前 Python 异常状态
    if (!metaclass)
      throw py::error_already_set();
    // 返回 metaclass 变量的值，这是一个立即调用的匿名函数
    return metaclass;
  }();
  // 返回之前立即调用的匿名函数返回的 metaclass 值
  return metaclass;
}

PyObject* c10d_init(PyObject* _unused, PyObject* noargs) {
  // 记录 API 使用情况到日志，表示导入 c10d.python 模块
  C10_LOG_API_USAGE_ONCE("c10d.python.import");

  // 导入 torch.distributed 模块，如果失败则抛出异常
  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch.distributed"));
  if (!c10d_module) {
    throw python_error();
  }

  // 导入 torch._C 模块，如果失败则抛出异常
  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  // 将 torch._C 模块定义为子模块 "_distributed_c10d"，用于分布式 c10d 绑定
  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m =
      torch_C_m.def_submodule("_distributed_c10d", "distributed c10d bindings");

  auto module = py::handle(m).cast<py::module>();

  // 定义 _register_comm_hook 函数并注册到模块中，指定参数和线程保护机制
  module
      .def(
          "_register_comm_hook",
          &_register_comm_hook,
          py::arg("reducer"),
          py::arg("state"),
          py::arg("comm_hook"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_register_builtin_comm_hook",
          &_register_builtin_comm_hook,
          py::arg("reducer"),
          py::arg("comm_hook_type"));

  // 定义 GradBucket 类的 Python 绑定，提供说明文档和各成员函数的绑定
  shared_ptr_class_<::c10d::GradBucket>(
      module,
      "GradBucket",
      R"(
This class mainly passes a flattened gradient tensor
(returned by :meth:`~torch.distributed.GradBucket.buffer`)
to DDP communication hook.
This tensor can be further decomposed into a list of per-parameter tensors within this bucket
(returned by :meth:`~torch.distributed.GradBucket.get_per_parameter_tensors`)
to apply layer-wise operations.
)")
      .def(
          "index",
          &::c10d::GradBucket::getIndex,
          py::call_guard<py::gil_scoped_release>(),
          R"(
.. warning::
    Since the buckets are rebuilt after the first iteration, should not rely on the indices at the beginning of training.

Returns:
    The index of a bucket that stores gradients of a few contiguous layers.
    All the gradients are bucketized.
)")
      .def(
          "buffer",
          &::c10d::GradBucket::getBuffer,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A flattened 1D ``torch.Tensor`` buffer,
    which can be further decomposed into a list of per-parameter tensors within this bucket.
)")
      .def(
          "gradients",
          &::c10d::GradBucket::getGradients,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A list of ``torch.Tensor``. Each tensor in the list corresponds to a gradient.
)")
      .def(
          "parameters",
          &::c10d::GradBucket::getParameters,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    A list of ``torch.Tensor``. Each tensor in the list corresponds to a model
    parameter.
)")
      .def(
          "is_last",
          &::c10d::GradBucket::isLast,
          py::call_guard<py::gil_scoped_release>(),
          R"(
Returns:
    Whether this bucket is the last bucket to allreduce in an iteration.
    This also means that this bucket corresponds to the first few layers in the forward pass.
  ")
      .def(
          "set_buffer",
          &::c10d::GradBucket::setBuffer,
          py::arg("buffer"),
          py::call_guard<py::gil_scoped_release>(),
          R"(
Replaces the tensor in the bucket with the input tensor buffer.
)");

  py::enum_<::c10d::BuiltinCommHookType>(module, "BuiltinCommHookType", R"(
An enum-like class for available reduction operations: ``SUM``, ``PRODUCT``,
``MIN``, ``MAX``, ``BAND``, ``BOR``, ``BXOR``, and ``PREMUL_SUM``.

``BAND``, ``BOR``, and ``BXOR`` reductions are not available when
using the ``NCCL`` backend.

``AVG`` divides values by the world size before summing across ranks.
``AVG`` is only available with the ``NCCL`` backend,
and only for NCCL versions 2.10 or later.

``PREMUL_SUM`` multiplies inputs by a given scalar locally before reduction.
``PREMUL_SUM`` is only available with the ``NCCL`` backend,
and only available for NCCL versions 2.11 or later. Users are supposed to
use ``torch.distributed._make_nccl_premul_sum``.

Additionally, ``MAX``, ``MIN`` and ``PRODUCT`` are not supported for complex tensors.

The values of this class can be accessed as attributes, e.g., ``ReduceOp.SUM``.
They are used in specifying strategies for reduction collectives, e.g.,
:func:`reduce`.

Base class for all store implementations, such as the 3 provided by PyTorch
distributed: (:class:`~torch.distributed.TCPStore`, :class:`~torch.distributed.FileStore`,
and :class:`~torch.distributed.HashStore`).
)")
          // Default constructor.
          .def(py::init<>())
          // Convert from std::string to std::vector<uint8>.
          .def(
              "set",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) { store.set(key, toVec8(value)); },
              py::call_guard<py::gil_scoped_release>(),
              R"(
Inserts the key-value pair into the store based on the supplied ``key`` and
``value``. If ``key`` already exists in the store, it will overwrite the old
value with the new supplied ``value``.

Arguments:
    key (str): The key to be added to the store.
    value (str): The value associated with ``key`` to be added to the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # Should return "first_value"
    >>> store.get("first_key")
.def(
    "compare_set",
    // 定义 Python 绑定的函数 "compare_set"
    [](::c10d::Store& store,
       const std::string& key,
       const std::string& expected_value,
       const std::string& desired_value) -> py::bytes {
        // 使用 lambda 表达式定义内部函数 value，实现对 store 的操作
        auto value = [&]() {
            // 释放全局解释器锁，允许多线程执行
            py::gil_scoped_release guard;
            // 调用 C++ Store 类的 compareSet 方法进行比较和设置操作
            return store.compareSet(
                key, toVec8(expected_value), toVec8(desired_value));
        }();
        // 将返回的值转换为 Python 字节对象
        return toPyBytes(value);
    },
    R"(
Inserts the key-value pair into the store based on the supplied ``key`` and
performs comparison between ``expected_value`` and ``desired_value`` before inserting. ``desired_value``
will only be set if ``expected_value`` for the ``key`` already exists in the store or if ``expected_value``
is an empty string.

Arguments:
    key (str): The key to be checked in the store.
    expected_value (str): The value associated with ``key`` to be checked before insertion.
    desired_value (str): The value associated with ``key`` to be added to the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("key", "first_value")
    >>> store.compare_set("key", "first_value", "second_value")
    >>> # Should return "second_value"
    >>> store.get("key")
)")
// 在 Python 中定义 compare_set 方法的文档字符串和示例
.def(
    "get",
    // 定义 Python 绑定的函数 "get"
    [](::c10d::Store& store, const std::string& key) -> py::bytes {
        // 使用 lambda 表达式定义内部函数 value，实现对 store 的操作
        auto value = [&]() {
            // 释放全局解释器锁，允许多线程执行
            py::gil_scoped_release guard;
            // 调用 C++ Store 类的 get 方法获取 key 对应的值
            return store.get(key);
        }();
        // 将返回的值转换为 Python 字节对象
        return toPyBytes(value);
    },
    R"(
Retrieves the value associated with the given ``key`` in the store. If ``key`` is not
present in the store, the function will wait for ``timeout``, which is defined
when initializing the store, before throwing an exception.

Arguments:
    key (str): The function will return the value associated with this key.

Returns:
    Value associated with ``key`` if ``key`` is in the store.

Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # Should return "first_value"
    >>> store.get("first_key")
)")
// 在 Python 中定义 get 方法的文档字符串和示例
.def(
    "add",
    // 定义 Python 绑定的函数 "add"，直接调用 C++ Store 类的 add 方法
    &::c10d::Store::add,
    // 使用 gil_scoped_release 作为调用保护，释放全局解释器锁
    py::call_guard<py::gil_scoped_release>(),
    R"(
The first call to add for a given ``key`` creates a counter associated
with ``key`` in the store, initialized to ``amount``. Subsequent calls to add
with the same ``key`` increment the counter by the specified ``amount``.
# 调用 :meth:`~torch.distributed.store.add` 方法添加已经在存储中被 :meth:`~torch.distributed.store.set` 设置过的键会导致异常。
.def(
    "add",
    &::c10d::Store::add,
    py::call_guard<py::gil_scoped_release>(),
    R"(
    向存储中指定的键增加一个数量。
    
    Arguments:
        key (str): 要增加计数器的键。
        amount (int): 要增加的数量。
    
    Example::
        >>> import torch.distributed as dist
        >>> from datetime import timedelta
        >>> # 以 TCPStore 为例，其他存储类型也可以使用
        >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
        >>> store.add("first_key", 1)
        >>> store.add("first_key", 6)
        >>> # 应该返回 7
        >>> store.get("first_key")
    )"
)

.def(
    "check",
    &::c10d::Store::check,
    py::call_guard<py::gil_scoped_release>(),
    R"(
    调用以检查给定列表中的 ``keys`` 是否在存储中有值。该调用通常立即返回，但在某些边缘情况下可能会遇到死锁，例如在销毁 TCPStore 后调用 check。
    
    Arguments:
        keys (list[str]): 要查询是否在存储中的键列表。
    
    Example::
        >>> import torch.distributed as dist
        >>> from datetime import timedelta
        >>> # 以 TCPStore 为例，其他存储类型也可以使用
        >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
        >>> store.add("first_key", 1)
        >>> # 应该返回 True
        >>> store.check(["first_key"])
    )"
)

.def(
    "delete_key",
    &::c10d::Store::deleteKey,
    py::call_guard<py::gil_scoped_release>(),
    R"(
    从存储中删除与 ``key`` 关联的键值对。如果成功删除，则返回 `true`，否则返回 `false`。
    
    .. warning::
        ``delete_key`` API 仅由 :class:`~torch.distributed.TCPStore` 和 :class:`~torch.distributed.HashStore` 支持。在 :class:`~torch.distributed.FileStore` 上使用此 API 会导致异常。
    
    Arguments:
        key (str): 要从存储中删除的键。
    
    Returns:
        如果成功删除 ``key`` 则返回 `True`，否则返回 `False`。
    
    Example::
        >>> import torch.distributed as dist
        >>> from datetime import timedelta
        >>> # 以 TCPStore 为例，HashStore 也可以使用
        >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
        >>> store.set("first_key")
        >>> # 应该返回 True
        >>> store.delete_key("first_key")
        >>> # 应该返回 False
        >>> store.delete_key("bad_key")
    )"
)

.def(
    "num_keys",
    &::c10d::Store::getNumKeys,
    py::call_guard<py::gil_scoped_release>(),
    R"(
    返回存储中设置的键的数量。注意，这个数字通常会
          .def(
              "num_keys",
              &::c10d::Store::numKeys,
              py::call_guard<py::gil_scoped_release>(),
              R"(
返回存储中当前键的数量。

警告：
    当使用:class:`~torch.distributed.TCPStore`时，``num_keys`` 返回写入底层文件的键的数量。如果存储被销毁并且使用相同文件创建另一个存储，原始键将被保留。

返回：
    存储中键的数量。

示例：
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # 以TCPStore为例，其他存储类型也可以使用
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set("first_key", "first_value")
    >>> # 这应该返回2
    >>> store.num_keys()
)")
          .def(
              "set_timeout",
              &::c10d::Store::setTimeout,
              py::call_guard<py::gil_scoped_release>(),
              R"(
设置存储的默认超时时间。此超时时间在初始化期间以及在:meth:`~torch.distributed.store.wait` 和 :meth:`~torch.distributed.store.get` 中使用。

参数：
    timeout (timedelta): 要在存储中设置的超时时间。

示例：
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # 以TCPStore为例，其他存储类型也可以使用
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> store.set_timeout(timedelta(seconds=10))
    >>> # 在10秒后将抛出异常
    >>> store.wait(["bad_key"])
)")
          .def(
              "wait",
              [](::c10d::Store& store, const std::vector<std::string>& keys) {
                store.wait(keys);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
等待``keys``中的每个键被添加到存储中。如果在``timeout``（在存储初始化期间设置）之前没有设置所有键，则``wait``将抛出异常。

参数：
    keys (list): 要等待的键列表，直到它们在存储中设置。

示例：
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # 以TCPStore为例，其他存储类型也可以使用
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    >>> # 在30秒后将抛出异常
    >>> store.wait(["bad_key"])
)")
          .def(
              "wait",
              [](::c10d::Store& store,
                 const std::vector<std::string>& keys,
                 const std::chrono::milliseconds& timeout) {
                store.wait(keys, timeout);
              },
              py::call_guard<py::gil_scoped_release>(),
              R"(
等待``keys``中的每个键在存储中被添加，并在提供的``timeout``之前如果键没有被设置则抛出异常。

参数：
    keys (list): List of keys on which to wait until they are set in the store.
    # keys 是一个列表，包含了需要等待直到它们在存储中设置的键名列表。

    timeout (timedelta): Time to wait for the keys to be added before throwing an exception.
    # timeout 是一个 timedelta 对象，表示等待的最长时间，在此时间内如果键没有被添加，则抛出异常。
# 导入必要的库和模块
>>> import torch.distributed as dist
>>> from datetime import timedelta

# 使用 TCPStore 作为示例，也可以使用其他类型的存储
>>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))

# 在设定的超时时间内等待 ["bad_key"] 的出现，超时则抛出异常
>>> store.wait(["bad_key"], timedelta(seconds=10))
    >>> store = dist.TCPStore("127.0.0.1", 0, 1, True, timedelta(seconds=30))
    # 创建一个 TCPStore 对象，连接到本地 IP 地址 "127.0.0.1"，使用默认端口 0，设置超时时间为 30 秒
    
    >>> store.multi_set(["first_key", "second_key"], ["po", "tato"])
    # 在存储中设置多个键值对，将 "first_key" 对应的值设置为 "po"，"second_key" 对应的值设置为 "tato"
    
    >>> # Should return b"po"
    # 预期返回结果应为字节串 b"po"
    
    >>> store.get("first_key")
    # 获取存储中 "first_key" 对应的值
  ")
          .def(
              "has_extended_api",
              &::c10d::Store::hasExtendedApi,
              R"(Returns true if the store supports extended operations.)");



// 定义 has_extended_api 方法的 Python 绑定，返回存储是否支持扩展操作的布尔值
  intrusive_ptr_class_<::c10d::FileStore>(
      module,
      "FileStore",
      store,
      R"(
A store implementation that uses a file to store the underlying key-value pairs.

Arguments:
    file_name (str): path of the file in which to store the key-value pairs
    world_size (int, optional): The total number of processes using the store. Default is -1 (a negative value indicates a non-fixed number of store users).

Example::
    >>> import torch.distributed as dist
    >>> store1 = dist.FileStore("/tmp/filestore", 2)
    >>> store2 = dist.FileStore("/tmp/filestore", 2)
    >>> # Use any of the store methods from either the client or server after initialization
    >>> store1.set("first_key", "first_value")
    >>> store2.get("first_key")
      )")
      .def(
          py::init<const std::string&, int>(),
          py::arg("file_name"),
          py::arg("world_size") = -1)
      .def_property_readonly(
          "path",
          &::c10d::FileStore::getPath,
          R"(Gets the path of the file used by FileStore to store key-value pairs.)");



// 定义 FileStore 类的 Python 绑定，使用文件来存储键值对的存储实现
// 包含文件名和进程数作为参数，支持通过 Python 访问路径属性
#ifndef _WIN32
  intrusive_ptr_class_<::c10d::HashStore>(
      module,
      "HashStore",
      store,
      R"(
A thread-safe store implementation based on an underlying hashmap. This store can be used
within the same process (for example, by other threads), but cannot be used across processes.

Example::
    >>> import torch.distributed as dist
    >>> store = dist.HashStore()
    >>> # store can be used from other threads
    >>> # Use any of the store methods after initialization
    >>> store.set("first_key", "first_value")
      )")
      .def(py::init<>());
#endif



// 定义 HashStore 类的 Python 绑定，基于哈希表实现的线程安全存储
// 适用于同一进程内的多线程使用，不支持跨进程使用
// 提供初始化和设置方法的示例
  intrusive_ptr_class_<::c10d::TCPStore>(
      module,
      "TCPStore",
      store,
      R"(
A TCP-based distributed key-value store implementation. The server store holds
the data, while the client stores can connect to the server store over TCP and
perform actions such as :meth:`~torch.distributed.store.set` to insert a key-value
pair, :meth:`~torch.distributed.store.get` to retrieve a key-value pair, etc. There
should always be one server store initialized because the client store(s) will wait for
the server to establish a connection.

Arguments:
    host_name (str): The hostname or IP Address the server store should run on.
    port (int): The port on which the server store should listen for incoming requests.
    world_size (int, optional): The total number of store users (number of clients + 1 for the server). Default is None (None indicates a non-fixed number of store users).
    is_master (bool, optional): True when initializing the server store and False for client stores. Default is False.
      )");



// 定义 TCPStore 类的 Python 绑定，基于 TCP 的分布式键值存储实现
// 支持服务器端和客户端操作，提供初始化参数和使用示例
    # 设置存储的超时时间，用于存储初始化和方法（如 get 和 wait）的超时设置。默认为 timedelta(seconds=300)。
    timeout (timedelta, optional): Timeout used by the store during initialization and for methods such as :meth:`~torch.distributed.store.get` and :meth:`~torch.distributed.store.wait`. Default is timedelta(seconds=300)

    # 是否等待所有工作节点连接到服务器存储。仅在 world_size 是一个固定值时适用。默认为 True。
    wait_for_workers (bool, optional): Whether to wait for all the workers to connect with the server store. This is only applicable when world_size is a fixed value. Default is True.

    # 如果为 True，则当前进程中所有具有相同主机/端口的 TCPStore 实例将使用同一个底层 TCPServer。默认为 False。
    multi_tenant (bool, optional): If True, all ``TCPStore`` instances in the current process with the same host/port will use the same underlying ``TCPServer``. Default is False.

    # 如果指定了，底层的 TCPServer 将监听此文件描述符，该描述符必须是已绑定到 port 的套接字。在某些情况下，避免端口分配竞争很有用。默认为 None（表示服务器创建一个新套接字并尝试将其绑定到 port）。
    master_listen_fd (int, optional): If specified, the underlying ``TCPServer`` will listen on this file descriptor, which must be a socket already bound to ``port``. Useful to avoid port assignment races in some scenarios. Default is None (meaning the server creates a new socket and attempts to bind it to ``port``).

    # 如果为 True，则使用 libuv 作为 TCPServer 的后端。默认为 True。
    use_libuv (bool, optional): If True, use libuv for ``TCPServer`` backend. Default is True.
Example::
    >>> import torch.distributed as dist
    >>> from datetime import timedelta
    >>> # 在进程1上运行（服务器）
    >>> server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
    >>> # 在进程2上运行（客户端）
    >>> client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
    >>> # 初始化后，可以在客户端或服务器上使用存储的任意方法
    >>> server_store.set("first_key", "first_value")
    >>> client_store.get("first_key")
      )")
      .def(
          py::init([](const std::string& host,
                      uint16_t port,
                      std::optional<int> worldSize,
                      bool isServer,
                      std::chrono::milliseconds timeout,
                      bool waitWorkers,
                      bool multiTenant,
                      std::optional<int> masterListenFd,
                      bool useLibUV) {
            std::optional<std::size_t> numWorkers = c10::nullopt;
            if (worldSize.has_value() && worldSize.value() > -1) {
              numWorkers = static_cast<std::size_t>(worldSize.value());
            }

            ::c10d::TCPStoreOptions opts{
                port,
                isServer,
                numWorkers,
                waitWorkers,
                timeout,
                multiTenant,
                masterListenFd,
                useLibUV};

            return c10::make_intrusive<::c10d::TCPStore>(host, opts);
          }),
          py::arg("host_name"),
          py::arg("port"),
          py::arg("world_size") = py::none(),
          // 使用 noconvert() 要求此参数为 True 或 False
          // 防止意外的布尔类型隐式转换
          py::arg("is_master").noconvert() = false,
          py::arg("timeout") =
              std::chrono::milliseconds(::c10d::Store::kDefaultTimeout),
          py::arg("wait_for_workers") = true,
          py::arg("multi_tenant") = false,
          py::arg("master_listen_fd") = py::none(),
          py::arg("use_libuv") = true,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "collect_client_counters",
          &::c10d::TCPStore::collectClientCounters,
          "返回一个包含 TCP 存储客户端计数器的字典")
      .def_property_readonly(
          "host",
          &::c10d::TCPStore::getHost,
          R"(获取存储用于接收请求的主机名。)")

      .def_property_readonly(
          "port",
          &::c10d::TCPStore::getPort,
          R"(获取存储用于接收请求的端口号。)")
      .def_property_readonly(
          "libuvBackend",
          &::c10d::TCPStore::isLibUvBackend,
          R"(如果使用 libuv 后端，则返回 True。)");
  
  intrusive_ptr_class_<::c10d::PrefixStore>(
      module,
      "PrefixStore",
      store,
      R"(
A wrapper around any of the 3 key-value stores (:class:`~torch.distributed.TCPStore`,
# 导入所需的头文件或库
:class:`~torch.distributed.FileStore`, and :class:`~torch.distributed.HashStore`)
that adds a prefix to each key inserted to the store.

Arguments:
    prefix (str): The prefix string that is prepended to each key before being inserted into the store.
    store (torch.distributed.store): A store object that forms the underlying key-value store.
      )")
      .def(py::init<const std::string&, c10::intrusive_ptr<::c10d::Store>>())
      .def_property_readonly(
          "underlying_store",
          &::c10d::PrefixStore::getUnderlyingStore,
          R"(Gets the underlying store object that PrefixStore wraps around.)")
      .def_property_readonly(
          "_underlying_non_prefix_store",
          &::c10d::PrefixStore::getUnderlyingNonPrefixStore,
          R"(Recursively to get the store before layers of wrapping with PrefixStore.)");

  using namespace std::chrono_literals;

  # 定义控制集合（collectives）类，并为其设置 Python 绑定
  auto collectives =
      py::class_<
          ::c10d::ControlCollectives,
          c10::intrusive_ptr<::c10d::ControlCollectives>>(
          module,
          "_ControlCollectives",
          R"(
Base class for all ControlCollectives implementations.
)")
          .def(
              "barrier",
              &::c10d::ControlCollectives::barrier,
              py::arg("key"),
              py::arg("timeout") = 5min,
              py::arg("block") = true,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Blocks until all workers have entered this function.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
    block (bool): whether to block this working waiting on the results of the barrier.
)")
          .def(
              "all_sum",
              &::c10d::ControlCollectives::allSum,
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Computes a sum across all workers and returns the final value.

Arguments:
    key (str): The unique key used to identify this operation.
    data (int): The data to sum.
    timeout (duration): The timeout for this operation.
)")
          .def(
              "broadcast_send",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                collectives.broadcastSend(key, toVec8(data), timeout);
              },
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Sends data to all other workers. Must be only called from one worker.

Arguments:
    key (str): The unique key used to identify this operation.
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
          .def(
              "broadcast_recv",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 std::chrono::milliseconds timeout = 5min) {
                // 使用 lambda 函数来接收广播数据，释放全局解释器锁
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  // 调用 ControlCollectives 的 broadcastRecv 方法进行数据接收
                  return collectives.broadcastRecv(key, timeout);
                }();
                // 将结果转换为 Python 字节串并返回
                return toPyBytes(out);
              },
              // 定义 Python 绑定的参数 key 和 timeout，并提供默认值
              py::arg("key"),
              py::arg("timeout") = 5min,
              R"(
Receives data broadcasted from 1 worker.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
)")

          .def(
              "gather_send",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                // 调用 ControlCollectives 的 gatherSend 方法发送数据，使用 toVec8 转换数据为 vector<uint8_t>
                collectives.gatherSend(key, toVec8(data), timeout);
              },
              // 定义 Python 绑定的参数 key, data 和 timeout，并提供默认值
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              py::call_guard<py::gil_scoped_release>(),
              R"(
Sends data to one other worker.

Arguments:
    key (str): The unique key used to identify this operation.
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
)")

          .def(
              "gather_recv",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::string& data,
                 std::chrono::milliseconds timeout = 5min) {
                // 使用 lambda 函数来接收聚集数据，释放全局解释器锁
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  // 调用 ControlCollectives 的 gatherRecv 方法进行数据接收
                  return collectives.gatherRecv(key, toVec8(data), timeout);
                }();
                // 将结果转换为 Python 字节串并返回
                return toPyBytes(out);
              },
              // 定义 Python 绑定的参数 key, data 和 timeout，并提供默认值
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              R"(
Receives data broadcasted from all workers. Must only be called by one worker.

Arguments:
    key (str): The unique key used to identify this operation.
    timeout (duration): The timeout for this operation.
)")

          .def(
              "scatter_send",
              [](::c10d::ControlCollectives& collectives,
                 const std::string& key,
                 const std::vector<std::string>& data,
                 std::chrono::milliseconds timeout = 5min) {
                // 使用 lambda 函数来发送散列数据，释放全局解释器锁
                auto out = [&]() {
                  py::gil_scoped_release guard;
                  // 调用 ControlCollectives 的 scatterSend 方法发送数据，使用 toVec8 转换数据为 vector<uint8_t>
                  return collectives.scatterSend(key, toVec8(data), timeout);
                }();
                // 将结果转换为 Python 字节串并返回
                return toPyBytes(out);
              },
              // 定义 Python 绑定的参数 key, data 和 timeout，并提供默认值
              py::arg("key"),
              py::arg("data"),
              py::arg("timeout") = 5min,
              R"(
Sends rank specific data to all other workers.

Arguments:
    key (str): The unique key used to identify this operation.
    data (list[str]): The list of data strings to send.
    timeout (duration): The timeout for this operation.
)")
    data (str): The data to send.
    timeout (duration): The timeout for this operation.
intrusive_ptr_class_<::c10d::StoreCollectives>(
    module,
    "_StoreCollectives",
    collectives,
    R"(
An implementation of ControlCollectives that uses the provided store as the underlying
Register a hook function which is fired on every ``ProcessGroup::Work`` completion.
The hook must have the following signature:

>>> def hook(work_info: torch._C._distributed_c10d.WorkInfo) -> None:
>>>     # custom code
>>>     # work_info.op_type: type of collective of this work
>>>     # work_info.seq: sequence number of collective of this work
>>>     # work_info.time_started: system time when user code called this collective
>>>     # work_info.time_finished: system time when the watchdog thread detected
>>>     #     completion of this work. Note that, there can be delays between the
>>>     #     actual completion time and the detection time.
>>>     # work_info.active_duration: duration of this collective measured by CUDAEvents
>>>     #     which can accurately represent the duration between when the collective
>>>     #     is launched and when the collective completes.

.. warning ::
    This only works for NCCL backend for now. All hooks are fired on the cpp watch dog
    thread. Firing the Python hook and acquiring GIL requires Python interpreter to be
    alive. Therefore, users need to make sure calling ``destroy_process_group(pg)`` on
    every active ProcessGroup ``pg`` before exiting.

.. warning ::
    This is a class definition for _StoreCollectives, an implementation of ControlCollectives
    using the provided store as the underlying mechanism.
)"
);
    # 注意：传递给钩子函数的 `Work` 对象是部分复制的版本，不包含输出对象。
    # 因此，无法通过 `Work` 对象访问输出张量。
    Note that ``Work`` object passed to the hook is a partially copied version without
    the output objects. So accessing the output tensors from ``Work`` will not work.
# 定义一个函数，用于绑定 Python 调用接口到 C++ 函数：设置钩子函数
Arguments:
    hook (Callable): hook function.
          )")
      .def(
          "_wait_for_pending_works",
          &::c10d::ProcessGroup::waitForPendingWorks,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_has_hooks",
          &::c10d::ProcessGroup::hasHooks,
          py::call_guard<py::gil_scoped_acquire>())
      .def(
          "_enable_collectives_timing",
          &::c10d::ProcessGroup::enableCollectivesTiming,
          py::call_guard<py::gil_scoped_acquire>(),
          "Enable timing of collectives by all backends. This might incur in additional overhead.")
      .def(
          "_set_group_name",
          &::c10d::ProcessGroup::setGroupName,
          py::call_guard<py::gil_scoped_acquire>(),
          "Sets the process group name. This is an internal C10D method, do not use.")
      .def_property_readonly(
          "group_name",
          &::c10d::ProcessGroup::getGroupName,
          "(Gets this process group name. It's cluster unique)")
      .def(
          "_set_group_desc",
          &::c10d::ProcessGroup::setGroupDesc,
          py::call_guard<py::gil_scoped_acquire>(),
          "Sets the process group description. This is an internal C10D method, do not use.")
      .def_property_readonly(
          "group_desc",
          &::c10d::ProcessGroup::getGroupDesc,
          "Gets this process group description")
      .def_property(
          "bound_device_id",
          &::c10d::ProcessGroup::getBoundDeviceId,
          &::c10d::ProcessGroup::setBoundDeviceId)
      .def("boxed", [](c10::intrusive_ptr<::c10d::ProcessGroup> self) {
        return torch::jit::toPyObject(c10::IValue(std::move(self)));
      })
      .def_static("unbox", [](py::object obj) {
          auto typePtr = torch::getCustomClass("__torch__.torch.classes.c10d.ProcessGroup");
          auto ivalue = torch::jit::toIValue(std::move(obj), typePtr);
          return ivalue.toCustomClass<::c10d::ProcessGroup>();
      });

# 定义一个枚举类型，表示不同的后端类型，用于描述进程组的后端类型选择
py::enum_<::c10d::ProcessGroup::BackendType>(processGroup, "BackendType")
    .value("UNDEFINED", ::c10d::ProcessGroup::BackendType::UNDEFINED)
    .value("GLOO", ::c10d::ProcessGroup::BackendType::GLOO)
    .value("NCCL", ::c10d::ProcessGroup::BackendType::NCCL)
    .value("UCC", ::c10d::ProcessGroup::BackendType::UCC)
    .value("MPI", ::c10d::ProcessGroup::BackendType::MPI)
    .value("CUSTOM", ::c10d::ProcessGroup::BackendType::CUSTOM)
    .export_values();

# 基础的 ProcessGroup::Options 绑定
auto processGroupOptions =
    intrusive_ptr_class_<::c10d::ProcessGroup::Options>(
        processGroup,
        "Options",
        R"(
Base class for all processes group options implementations, such as the nccl
options :class:`~torch.distributed.ProcessGroupNCCL.Options`).
  ")
          .def(
              py::init([](const std::string& backend,
                          const std::chrono::milliseconds& timeout) {
                // 使用给定的后端和超时时间创建 ProcessGroup::Options 对象
                return c10::make_intrusive<::c10d::ProcessGroup::Options>(
                    backend, timeout);
              }),
              py::arg("backend"),  // Python 中的参数名称为 backend
              py::arg("timeout") = kProcessGroupDefaultTimeout,  // 超时时间，默认为 kProcessGroupDefaultTimeout
              py::call_guard<py::gil_scoped_release>())  // 释放全局解释器锁
          .def_readonly("backend", &::c10d::ProcessGroup::Options::backend)  // 只读属性 backend
          .def_readwrite("_timeout", &::c10d::ProcessGroup::Options::timeout);  // 可读写属性 _timeout

#ifndef _WIN32
  module.def(
      "_round_robin_process_groups",
      [](std::vector<c10::intrusive_ptr<::c10d::ProcessGroup>> processGroups)
          -> c10::intrusive_ptr<::c10d::ProcessGroup> {
        if (processGroups.empty()) {
          throw std::invalid_argument("Specify at least 1 process group");
        }
        const auto& first = processGroups.front();
        // 创建一个 RoundRobin 过程组，使用第一个过程组的信息
        return c10::make_intrusive<::c10d::ProcessGroupRoundRobin>(
            first->getRank(), first->getSize(), std::move(processGroups));
      },
      py::arg("process_groups"),  // Python 中的参数名称为 process_groups
      py::call_guard<py::gil_scoped_release>());  // 释放全局解释器锁
#endif

#ifdef NCCL_HAS_COMM_CTA_CGA
  py::class_<ncclConfig_t>(
      processGroupNCCL,
      "NCCLConfig",
      R"(
ncclConfig_t data type for configuring NCCL communicators.
See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
for details.
)")
      .def(py::init<>())  // NCCLConfig 类的构造函数
      .def_readwrite("blocking", &ncclConfig_t::blocking)  // 可读写属性 blocking
      .def_readwrite("cga_cluster_size", &ncclConfig_t::cgaClusterSize)  // 可读写属性 cga_cluster_size
      .def_readwrite("min_ctas", &ncclConfig_t::minCTAs)  // 可读写属性 min_ctas
      .def_readwrite("max_ctas", &ncclConfig_t::maxCTAs)  // 可读写属性 max_ctas
      .def_property(
          "net_name",
          [](const ncclConfig_t& self) { return self.netName; },  // 获取 net_name 属性的 lambda 函数
          // 注意: NCCL 在销毁通信器时会调用 free() 函数释放 netName 指针，
          // 因此 strdup 分配的内存不会因为泄漏而有问题。
          [](ncclConfig_t& self, const char* tmp) {
            self.netName = strdup(tmp);  // 设置 net_name 属性
          });
#endif

  intrusive_ptr_class_<::c10d::ProcessGroupNCCL::Options>(
      processGroupNCCL,
      "Options",
      processGroupOptions,
      R"(
ProcessGroup options for the NCCL backend

Arguments:
    is_high_priority_stream (bool, optional): flag to enable/disable process
            group to pick up high priority cuda streams. It lets CUDA driver
            to prioritize NCCL kernels when there are compute kernels waiting.
            Default is False.

Attributes:
    // 在此处可能有更多的属性描述
)");
    config (NCCLConfig): configures NCCL communicators (only avaiable for
            builds using NCCL 2.17+). This can be used to improve
            communication-computation overlap for NCCL kernels by tuning
            available parameters in the config. See
            https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
            for details.
Example::
    >>> import torch.distributed as dist
    >>>
    >>> # 创建一个 NCCL 进程组的选项对象，设置高优先级流
    >>> nccl_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
    >>> # 对于使用 NCCL 2.17+ 版本的构建，配置通信器
    >>> nccl_options.config.cga_cluster_size = 2
    >>> nccl_options.config.max_ctas = 4
    >>> nccl_options.config.min_ctas = 2
    >>> # 使用刚创建的选项初始化一个 NCCL 进程组
    >>> dist.init_process_group("nccl", pg_options=nccl_options)
#ifdef NCCL_HAS_COMM_CTA_CGA
      )")
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
#ifdef NCCL_HAS_COMM_CTA_CGA
      .def_readwrite("config", &::c10d::ProcessGroupNCCL::Options::config)
#endif
      .def_readwrite(
          "is_high_priority_stream",
          &::c10d::ProcessGroupNCCL::Options::is_high_priority_stream)
      .def_readwrite(
          "split_from", &::c10d::ProcessGroupNCCL::Options::split_from)
      .def_readwrite(
          "split_color", &::c10d::ProcessGroupNCCL::Options::split_color)
      .def_readwrite(
          "global_ranks_in_group",
          &::c10d::ProcessGroupNCCL::Options::global_ranks_in_group)
      .def_readwrite(
          "group_name", &::c10d::ProcessGroupNCCL::Options::group_name);
#endif

#ifdef USE_C10D_MPI
  auto processGroupMPI =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupMPI>(
          module, "ProcessGroupMPI", backend);

  // Define static create function instead of a constructor, because
  // this function may return null. This happens if this process is not
  // part of a sub group that is to be created.
  processGroupMPI.def_static(
      "create",
      [](std::vector<int> ranks) {
        return ::c10d::ProcessGroupMPI::createProcessGroupMPI(std::move(ranks));
      },
      py::call_guard<py::gil_scoped_release>());
#endif

#ifdef USE_C10D_UCC
  auto processGroupUCC =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupUCC>(
          module, "ProcessGroupUCC", backend)
          .def(
              py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                          int rank,
                          int size,
                          const std::chrono::milliseconds& timeout) {
                return c10::make_intrusive<::c10d::ProcessGroupUCC>(
                    store, rank, size, timeout);
              }),
              py::arg("store"),
              py::arg("rank"),
              py::arg("size"),
              py::arg("timeout") = kProcessGroupDefaultTimeout,
              py::call_guard<py::gil_scoped_release>());
#endif



// 结束宏定义区域，此处可能用于条件编译或包含的一部分
py::enum_<::c10d::OpType>(module, "OpType")
    .value("BROADCAST", ::c10d::OpType::BROADCAST)
    .value("ALLREDUCE", ::c10d::OpType::ALLREDUCE)
    .value("ALLREDUCE_COALESCED", ::c10d::OpType::ALLREDUCE_COALESCED)
    .value("REDUCE", ::c10d::OpType::REDUCE)
    .value("ALLGATHER", ::c10d::OpType::ALLGATHER)
    .value("_ALLGATHER_BASE", ::c10d::OpType::_ALLGATHER_BASE)
    .value("ALLGATHER_COALESCED", ::c10d::OpType::ALLGATHER_COALESCED)
    .value("GATHER", ::c10d::OpType::GATHER)
    .value("SCATTER", ::c10d::OpType::SCATTER)
    .value("REDUCE_SCATTER", ::c10d::OpType::REDUCE_SCATTER)
    .value("ALLTOALL_BASE", ::c10d::OpType::ALLTOALL_BASE)
    .value("ALLTOALL", ::c10d::OpType::ALLTOALL)
    .value("SEND", ::c10d::OpType::SEND)
    .value("RECV", ::c10d::OpType::RECV)
    .value("RECVANYSOURCE", ::c10d::OpType::RECVANYSOURCE)
    .value("BARRIER", ::c10d::OpType::BARRIER)
    .value("_REDUCE_SCATTER_BASE", ::c10d::OpType::_REDUCE_SCATTER_BASE)
    .value("COALESCED", ::c10d::OpType::COALESCED)
    .value("_ALLREDUCE_SPARSE", ::c10d::OpType::_ALLREDUCE_SPARSE)
    .value("UNKNOWN", ::c10d::OpType::UNKNOWN);



// 定义枚举类型 OpType，为 C++ 到 Python 的枚举映射提供支持
py::class_<::c10d::WorkInfo, std::shared_ptr<::c10d::WorkInfo>>(
    module, "WorkInfo")
    .def_readonly("op_type", &::c10d::WorkInfo::opType)
    .def_readonly("seq", &::c10d::WorkInfo::seq)
    .def_readonly("time_started", &::c10d::WorkInfo::timeStarted)
    .def_readonly("time_finished", &::c10d::WorkInfo::timeFinished)
    .def_readonly("active_duration", &::c10d::WorkInfo::activeDuration);



// 定义 WorkInfo 类，描述异步操作的信息，包括操作类型、序列号和时间信息等
py::class_<
    ::c10d::Work,
    c10::intrusive_ptr<::c10d::Work>,
    ::c10d::PyProcessGroup::PyWork>(module, "Work", R"(
A `Work` object represents the handle to a pending asynchronous operation in
PyTorch's distributed package. It is returned by non-blocking collective operations,
such as `dist.all_reduce(tensor, async_op=True)`.
#ifdef USE_C10D_NCCL
module.attr("_DEFAULT_PG_NCCL_TIMEOUT") =
    py::cast(::c10d::kProcessGroupNCCLDefaultTimeout);



// 定义 Work 类，表示 PyTorch 分布式包中待处理的异步操作句柄
// 此处包含条件编译指令，根据 USE_C10D_NCCL 宏设置默认的 NCCL 进程组超时时间
  module.attr("_DEFAULT_NO_TIMEOUT") = py::cast(kNoTimeout);


# 将 C++ 常量 kNoTimeout 转换为 Python 属性 _DEFAULT_NO_TIMEOUT



  module.def(
      "_set_global_rank",
      [](int64_t rank) { c10::SetGlobalRank(rank); },
      py::arg("rank"),
      R"(
        Arguments:
          rank(int): The rank of the default process group
        Informs the C++ runtime what the default process group (a strictly Python
        notion) is.  This mostly ensures that C++ log messages are prefixed with
        rank information.  This is not meant to be called manually; it is
        called by _update_default_pg.
      )");


# 定义名为 _set_global_rank 的 Python 绑定函数
# 函数接受一个参数 rank，用于设置全局默认进程组的等级
# 文档字符串解释了函数的参数和用途，描述了其如何通知 C++ 运行时关于默认进程组的概念



  module.def(
      "_create_work_from_future",
      [](const std::shared_ptr<jit::PythonFutureWrapper>& future) {
        return ::c10d::Work::create_from_future(future->fut);
      },
      py::arg("future"),
      R"(
        Arguments:
            future(str): The future to wrap.
        Returns:
            A ``Work`` object which is associated with the completion of
            the ``torch.futures.Future``.
        This is the preferred way of constructing Work objects when writing a custom ProcessGroup
        in python.
        Example::
            >>> class SingleRankProcessGroup(torch.distributed.ProcessGroup):
            >>>     def broadcast(self, tensor_list, opts):
            >>>         fut = torch.futures.Future()
            >>>         fut.set_result(tensor_list)
            >>>         return torch._C._distributed_c10d._create_work_from_future(fut)
        .. warning ::
            This API is experimental and subject to change.
            The returned Work object has multiple limitations:
            - synchronize() does nothing. Use ``torch.futures.Future`` based synchronization.
            - wait() ignored timeout argument.
            - sourceRank() raises.
            - abort() raises.
            The provided Future object result must be a Tensor or a list of Tensors.
       )");


# 定义名为 _create_work_from_future 的 Python 绑定函数
# 函数接受一个参数 future，是一个被包装的字符串表示的未来对象
# 返回一个与 torch.futures.Future 完成相关联的 Work 对象
# 文档字符串提供了参数的描述和返回值说明，展示了在编写自定义 ProcessGroup 时构建 Work 对象的首选方法
# 还包含了一个示例和一些警告信息，说明 API 是实验性的并且可能会发生变化
#ifdef USE_C10D_NCCL
  // 如果定义了 USE_C10D_NCCL 宏，则注册 _hash_tensors 方法
  module.def(
      "_hash_tensors",
      // Lambda 函数，接受一个 torch.Tensor 列表，返回其哈希值
      [](const std::vector<at::Tensor>& tensors) {
        return ::c10d::hashTensors(tensors);
      },
      // pybind11 参数注释，描述函数参数 tensors
      py::arg("tensors"),
      R"(
        Arguments:
          tensors(List[torch.Tensor]): List of tensors we want to hash.
      )");
  // 注册 _dump_nccl_trace 方法
  module.def(
      "_dump_nccl_trace",
      // Lambda 函数，接受三个可选参数，返回 NCCL 跟踪信息的字节串
      [](std::optional<bool> includeCollectives,
         std::optional<bool> includeStackTraces,
         std::optional<bool> onlyActive) {
        return py::bytes(::c10d::dump_nccl_trace(
            includeCollectives.value_or(true),
            includeStackTraces.value_or(true),
            onlyActive.value_or(false)));
      },
      // pybind11 参数注释，描述函数参数和返回值
      py::arg("includeCollectives") = std::optional<bool>(),
      py::arg("includeStackTraces") = std::optional<bool>(),
      py::arg("onlyActive") = std::optional<bool>(),
      R"(
        Arguments:
            includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
            includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
            onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
        Returns:
            Stringified pickle work traces.
            Default settings return everything - i.e. contains NCCL comm dumps and collective traces.
      )");
#endif

  // 注册 _WorkerServer 类
  intrusive_ptr_class_<::c10d::control_plane::WorkerServer>(
      module, "_WorkerServer", R"(
)")
      .def(
          // _WorkerServer 类构造函数，接受主机名或文件路径和端口号
          py::init([](const std::string& hostOrFile, int port) {
            return c10::make_intrusive<::c10d::control_plane::WorkerServer>(
                hostOrFile, port);
          }),
          // pybind11 参数注释，描述构造函数参数
          py::arg("host_or_file"),
          py::arg("port") = -1)
      // 注册 _WorkerServer 类方法 shutdown
      .def("shutdown", &::c10d::control_plane::WorkerServer::shutdown);
  
  // 返回 Py_TRUE，表示模块注册成功
  Py_RETURN_TRUE;
}

// 取消 PROCESS_GROUP_DEPRECATION_WARNING 宏定义

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = { // NOLINT
    // 注册 _c10d_init 方法
    {"_c10d_init", c10d_init, METH_NOARGS, nullptr},
    // Sentinel
    {nullptr, nullptr, 0, nullptr}};

// 返回注册的 Python 方法数组
PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch::distributed::c10d
```