# `.\pytorch\torch\csrc\autograd\profiler_python.cpp`

```
// 引入 Torch 库的 Python 接口，用于分析器模块
#include <torch/csrc/autograd/profiler_python.h>

// 引入 C++ 标准库头文件
#include <atomic>           // 原子操作库
#include <cstdint>          // 固定大小整数类型
#include <deque>            // 双端队列
#include <limits>           // 数值极限
#include <memory>           // 内存管理
#include <queue>            // 队列
#include <string>           // 字符串
#include <utility>          // 实用工具
#include <vector>           // 动态数组

// 引入 Python C API 头文件
#include <Python.h>             // Python 标准库
#include <frameobject.h>        // Python 帧对象

// 引入 PyTorch 核心头文件和宏定义
#include <ATen/core/TensorBase.h>       // TensorBase 类定义
#include <c10/macros/Macros.h>          // C10 宏定义
#include <c10/util/ApproximateClock.h>  // 近似时钟工具
#include <c10/util/Exception.h>         // 异常处理工具
#include <c10/util/Logging.h>           // 日志工具
#include <c10/util/Optional.h>          // 可选类型工具
#include <c10/util/flat_hash_map.h>     // 平面哈希映射工具
#include <c10/util/irange.h>            // 迭代范围工具

// 引入 PyTorch 分析器相关头文件
#include <torch/csrc/autograd/python_variable.h>    // PyTorch Python 变量接口
#include <torch/csrc/profiler/collection.h>         // 分析器数据收集
#include <torch/csrc/profiler/containers.h>         // 分析器数据容器
#include <torch/csrc/profiler/orchestration/python_tracer.h>  // Python 跟踪器
#include <torch/csrc/profiler/util.h>                // 分析器实用工具
#include <torch/csrc/utils/pybind.h>                 // PyTorch 绑定工具
#include <torch/csrc/utils/python_compat.h>          // Python 兼容性工具
#include <torch/csrc/utils/python_strings.h>         // Python 字符串工具

// 定义命名空间别名
namespace py = pybind11;

namespace torch {
namespace profiler {
namespace impl {

// 匿名命名空间，用于限定符号的作用域
namespace {

// 枚举类型，表示调用类型
enum CallType { PyCall = 0, PyModuleCall, PyCCall, PyOptimizerCall };

// 声明并初始化枚举类型大小
static constexpr size_t CallTypeSize = 4;

// 定义空类型别名
using no_ephemeral_t = std::tuple<>;

// 定义并初始化线程 ID 的常量
static constexpr uint64_t NoTID = std::numeric_limits<uint64_t>::max();

// ============================================================================
// == Miscellaneous structs and utils =========================================
// ============================================================================

// 代码位置结构体，用于表示代码在源文件中的位置
struct CodeLocation {
  CodeLocation() = default;
  explicit CodeLocation(PyFrameObject* frame)
      : line_number_{PyFrame_GetLineNumber(frame)} {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    filename_ = THPUtils_unpackStringView(code->co_filename).data();
    name_ = THPUtils_unpackStringView(code->co_name).data();
  }

  // 比较操作符重载，用于判断两个 CodeLocation 对象是否相等
  bool operator==(const CodeLocation& other) const {
    return filename_ == other.filename_ && name_ == other.name_ &&
        line_number_ == other.line_number_;
  }

  const char* filename_{nullptr};  // 源文件名指针
  const char* name_{nullptr};      // 函数名指针
  int line_number_{0};             // 行号
};

// 模板函数，根据不同的调用类型返回对应的 Python 代码对象
template <CallType C>
PyCodeObject* getCode();

// 特化模板函数，返回模块调用类型的 Python 代码对象
template <>
PyCodeObject* getCode<CallType::PyModuleCall>() {
  // Lambda 表达式，获取 torch.nn.Module.__call__.__code__ 对象
  static auto module_call_code = []() {
    pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
    auto res = py::module::import("torch.nn")  // 导入 torch.nn 模块
                   .attr("Module")             // 获取 Module 类
                   .attr("__call__")           // 获取 __call__ 方法
                   .attr("__code__")           // 获取代码对象
                   .ptr();                     // 转换为 Python 对象指针
    TORCH_INTERNAL_ASSERT(PyCode_Check(res)); // 断言对象是 PyCodeObject 类型
    return (PyCodeObject*)res;  // 返回 Python 代码对象指针
  }();
  return module_call_code;
};

// 特化模板函数，返回优化器调用类型的 Python 代码对象
template <>
PyCodeObject* getCode<CallType::PyOptimizerCall>() {
  // Lambda 表达式，获取 torch.optim.Optimizer._optimizer_step_code.__code__ 对象
  static auto optimizer_step_code = []() {
    pybind11::gil_scoped_acquire gil;  // 获取全局解释器锁
    auto res = py::module::import("torch.optim")  // 导入 torch.optim 模块
                   .attr("Optimizer")             // 获取 Optimizer 类
                   .attr("_optimizer_step_code")  // 获取 _optimizer_step_code 属性
                   .attr("__code__")              // 获取代码对象
                   .ptr();                        // 转换为 Python 对象指针
    TORCH_INTERNAL_ASSERT(PyCode_Check(res));  // 断言对象是 PyCodeObject 类型
    return (PyCodeObject*)res;  // 返回 Python 代码对象指针
  }();
  return optimizer_step_code;
};
// 结构体模板特化，用于计算 CodeLocation 对象的哈希值
template <>
struct std::hash<torch::profiler::impl::CodeLocation> {
  // 重载 () 操作符，接受一个 CodeLocation 对象并返回其哈希值
  size_t operator()(const torch::profiler::impl::CodeLocation& x) {
    // 调用 c10::get_hash 函数计算哈希值，传入 filename_, name_, line_number_ 作为参数
    return c10::get_hash(x.filename_, x.name_, x.line_number_);
  }
};

// 命名空间 torch::profiler::impl 匿名命名空间内的定义
namespace {
// ============================================================================
// == CallTypeHelper: 用于对特化进行泛型编程的工具类 ===================================
// ============================================================================

// CallTypeHelper 类模板
template <template <CallType> class ClassT>
class CallTypeHelper final {
 private:
  // 静态断言，确保 CallType::PyCall 的值为 0，以便进行整数计算
  static_assert(
      CallType::PyCall == 0,
      "CallTypeHelper uses integer math which depends on a zero start.");
  // End 常量设定为 CallTypeSize
  static constexpr size_t End = CallTypeSize;

  // make_tuple_impl 函数模板，返回一个包含 ClassT<(CallType)I>... 类型的元组
  template <size_t... I>
  static constexpr std::tuple<ClassT<(CallType)I>...> make_tuple_impl(
      std::index_sequence<I...>);

  // map 函数模板，遍历 tuple 类型 t 中的元素并应用函数对象 f
  template <size_t C, typename T, typename FunctorT, typename... Args>
  static void map(T& t, FunctorT& f, Args&&... args) {
    f(std::get<C>(t), args...);
    // 如果 C+1 小于 End，则递归调用 map 函数
    if constexpr (C + 1 < End) {
      map<C + 1>(t, f, std::forward<Args>(args)...);
    }
  }

 public:
  // tuple_type 类型别名，为 make_tuple_impl 的返回类型 decltype(make_tuple_impl(std::make_index_sequence<End>{}))
  using tuple_type = decltype(make_tuple_impl(std::make_index_sequence<End>{}));

  // map 函数模板，遍历 tuple 类型 t 中的元素并应用函数对象 f
  template <typename FunctorT, typename... Args>
  static void map(tuple_type& t, FunctorT& f, Args&&... args) {
    map<0>(t, f, std::forward<Args>(args)...);
  }
};
} // namespace

// ============================================================================
// == Event type definitions. =================================================
// ============================================================================

// 当跟踪 Python 程序时，记录进入或退出函数的每个事件的一般过程是，
// 在后处理期间重播这些事件。在剖析阶段，我们希望尽可能少地工作以捕获所有所需信息；
// 否则，会扭曲剖析数据。在后处理过程中，我们愿意做额外的修正工作以减少剖析阶段的开销。

// 当跟踪器首次进入帧时，它为该位置构造一个 CallKey。关键内容因上下文而异。
// 对于 Python 函数，关键是定义函数字节码的 (PyCodeObject*, int) 对。
// 对于 `nn.Module`，关键是指向 `self` 的非拥有指针。
// 对于绑定的 C 函数，它是指向绑定函数的非拥有指针。
// CallKey 应该是小型、廉价且 POD（平凡标量类型）。

// 我们然后收集调用帧的 Callsite 以进行更好的源跟踪。
// 此对是一个 `Callsite`，并在跟踪期间作为第一级键使用。
// 我们在线程本地缓存中查找 Callsite，该缓存映射为
// Callsite to a unique integer `TraceKey`. On a cache hit, we simply store the
// TraceKey and return. On a cache miss, we use a global value cache to store
// whatever fields we need from the two CallKeys, generate a new TraceKey, and
// update the local cache.
//
// During post processing we:
//   1) Determine the type represented by a TraceKey by checking which
//      sub-cache it appears in in the thread local cache.
//   2) Look up the pair of CallKeys from the thread local cache.
//   3) Look up the expanded values of each CallKey from the global value cache.
//
// To add a new event type to the cache:
//   1) Add an entry to the `CallType` enum.
//   2) Add a specialization of Config which defined key_t, ephemeral_t and
//      cache_t.
//   3) Add a specialization of ValueCache::store and ValueCache::load.
//
// -------------------------
// -- Ephemeral arguments --
// -------------------------
// The value cache mechanism assumes that `key_t` is enough to specify the
// correct value. However it may not be possible to materialize a value using
// only an instance of `key_t`. As a result, the cache also accepts "ephemeral"
// inputs which can be used to populate the value cache. Ephemeral inputs come
// with two caveats:
//  1) They are NOT safe to save, and cannot be used after `ValueCache::store`.
//  2) They should be used to access data that is not expect to change from
//     call to call, such as the name of a function.

template <CallType>
struct Config;

// Specialization of Config for CallType::PyCall, defining key_t as CodeLocation,
// ephemeral_t as no_ephemeral_t (indicating no ephemeral data is used), and
// cache_t as ska::flat_hash_map<CodeLocation, PyFrameState>.
template <>
struct Config<CallType::PyCall> {
  using key_t = CodeLocation;
  using ephemeral_t = no_ephemeral_t;
  using cache_t = ska::flat_hash_map<key_t, PyFrameState>;
  static constexpr EventType event_type = EventType::PyCall;
};

// Struct defining configuration for PyCall with extended parameters and caching.
template <typename Key, typename Cls, typename ParameterInfo>
struct ExtendedPyCallConfig {
  using key_t = Key;
  using cls_t = Cls;
  using ephemeral_t = PyFrameObject*;

  // Inner struct Cache holding specific data structures for caching.
  struct Cache {
    // Optional CodeLocation for specific location identification.
    std::optional<CodeLocation> location_;
    // Flat hash map associating key_t with ClsAndParameters.
    ska::flat_hash_map<key_t, ClsAndParameters> cls_and_parameters_;
    // Flat hash map associating cls_t with at::StringView names.
    ska::flat_hash_map<cls_t, at::StringView> cls_names_;
  };
  using cache_t = Cache;

  static constexpr EventType event_type = EventType::PyCall;
};

// Specialization of Config for CallType::PyModuleCall, inheriting from
// ExtendedPyCallConfig with specific types and parameters.
template <>
struct Config<CallType::PyModuleCall> : ExtendedPyCallConfig<
                                            PyModuleSelf,
                                            PyModuleCls,
                                            NNModuleInfo::ParameterInfo> {};

// Specialization of Config for CallType::PyOptimizerCall, inheriting from
// ExtendedPyCallConfig with specific types and parameters.
template <>
struct Config<CallType::PyOptimizerCall> : ExtendedPyCallConfig<
                                               PyOptimizerSelf,
                                               PyOptimizerCls,
                                               OptimizerInfo::ParameterInfo> {};
// 定义模板结构体 Config，用于不同的调用类型，这里是用于 PyCCall 类型的配置
struct Config<CallType::PyCCall> {
  // 使用 PyMethod 作为键类型
  using key_t = PyMethod;
  // 使用 PyObject* 作为临时对象类型
  using ephemeral_t = PyObject*;
  // 使用 ska::flat_hash_map 存储键值对，键是 key_t 类型，值是 at::StringView 类型
  using cache_t = ska::flat_hash_map<key_t, at::StringView>;
  // 定义事件类型为 PyCCall
  static constexpr EventType event_type = EventType::PyCCall;
};

// ============================================================================
// == Callsite & ValueCache: Storage during profiling =========================
// ============================================================================

// Callsite 模板类，用于存储不同调用类型的数据
template <CallType C>
class Callsite {
 public:
  // 静态成员常量，指明调用类型
  static constexpr CallType call_type = C;
  // 使用 Config<C>::key_t 作为键类型
  using key_t = typename Config<C>::key_t;

  // 静态断言，确保 key_t 是平凡的复制构造类型
  static_assert(
      std::is_trivially_copyable<key_t>::value,
      "Key should be trivial, as it is passed by value.");

  // 构造函数，初始化 Callsite 对象
  template <typename U>
  Callsite(U value, PyFrameObject* f_back) : value_(value), caller_(f_back) {}

  // 比较运算符重载，判断两个 Callsite 对象是否相等
  bool operator==(const Callsite<C>& other) const {
    return value_ == other.value_ && caller_ == other.caller_;
  }

  // 存储 Callsite 的值和调用者信息
  key_t value_;
  Config<CallType::PyCall>::key_t caller_;
};

// ============================================================================
// == Type specific store and load implementations. ===========================
// ============================================================================

// 使用不同调用类型的键类型进行别名定义
using PyCallKey = Config<CallType::PyCall>::key_t;
using PyModuleCallKey = Config<CallType::PyModuleCall>::key_t;
using PyCCallKey = Config<CallType::PyCCall>::key_t;
using PyOptimizerCallKey = Config<CallType::PyOptimizerCall>::key_t;

// ValueCache 类，用于存储不同调用类型的数据
class ValueCache {
 public:
  // 默认构造函数
  ValueCache() = default;
  // 禁用拷贝构造函数
  ValueCache(const ValueCache&) = delete;

  // 存储函数模板，存储特定调用类型的键和临时对象
  template <CallType C>
  void store(const typename Config<C>::key_t&, typename Config<C>::ephemeral_t);

  // 加载函数模板，根据 Callsite 和 Python 线程 ID 加载特定调用类型的数据
  template <CallType C>
  auto load(const Callsite<C>& callsite, size_t python_tid) const {
    auto caller = load<CallType::PyCall>(callsite.caller_);
    // 内部断言，确保 caller 没有 module_info_
    TORCH_INTERNAL_ASSERT(!caller.module_info_.has_value());
    // 返回 ExtraFields 结构体的实例，用于事件类型为 Config<C>::event_type 的附加字段
    return ExtraFields<Config<C>::event_type>{
        /*end_time_ns=*/std::numeric_limits<c10::time_t>::min(),
        python_tid,
        caller.frame_state_,
        load<C>(callsite.value_)};
  }

  // 如果是张量则记录其元数据
  std::optional<TensorMetadata> recordIfTensor(py::handle p);
  // 解包张量映射并返回解包后的结果
  std::vector<std::pair<std::string, TensorMetadata>> unpackTensorMap(
      const py::dict& tensor_map);
  // 修剪前缀信息
  void trimPrefixes();

 private:
  // 加载函数模板，根据调用类型 C 加载其数据
  template <CallType C>
  typename ExtraFields<Config<C>::event_type>::args_t load(
      const typename Config<C>::key_t&) const;

  // 使用调用类型 C 的状态别名定义
  template <CallType C>
  using State = typename Config<C>::cache_t;

  // 状态元组 state_，根据 CallTypeHelper 的 tuple_type 定义
  CallTypeHelper<State>::tuple_type state_;
};

// 设置类别函数模板，根据调用类型 C 设置类别
template <CallType C>
typename Config<C>::cls_t set_class(
    ValueCache* value_cache,
    typename Config<C>::cache_t& cache,
    const typename Config<C>::key_t& key,
    const typename Config<C>::ephemeral_t& frame) {
  // 如果 cache 的位置信息不存在，则执行以下操作
  if (C10_UNLIKELY(!cache.location_.has_value())) {
    // 获取代码对象的指针
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    // 内部断言，确保 code 等于 getCode<C>() 的返回值
    TORCH_INTERNAL_ASSERT(code.get() == getCode<C>());
    // 设置 cache 的位置信息为当前帧的 PyCallKey
    cache.location_ = PyCallKey(frame);
    # 使用 value_cache 对象调用 store 方法，存储 PyCall 类型的数据，使用 cache.location_ 作为位置参数，no_ephemeral_t() 作为额外参数
    value_cache->store<CallType::PyCall>(*cache.location_, no_ephemeral_t());
  }

  # 获取 key 对应的 Python 对象的类句柄
  auto cls_handle = py::handle((PyObject*)key).attr("__class__");
  # 从类句柄中获取类对象，并转换为对应的 C++ 类型
  auto cls = typename Config<C>::cls_t(cls_handle.ptr());

  # 检查 cache 中是否已经存在当前类对象 cls 的名称
  if (cache.cls_names_.find(cls) == cache.cls_names_.end()) {
    # 如果不存在，则将 cls 加入到 cache 的 cls_names_ 中，并关联其名称
    cache.cls_names_[cls] =
        at::StringView(py::str(cls_handle.attr("__name__")));
  }
  # 返回当前处理的类对象 cls
  return cls;
}

// 将 Python 对象转换为 TensorMetadata 结构
TensorMetadata toTensorMetadata(PyObject* self) {
  // 断言确保 self 是 THPVariable 类型的对象
  TORCH_INTERNAL_ASSERT(THPVariable_CheckExact(self));
  // 将 self 转换为 THPVariable 对象
  const auto& t = THPVariable_Unpack(self);
  // 使用 RawTensorMetadata 结构封装 t
  RawTensorMetadata m{t};
  // 返回包含 Tensor 元数据的 TensorMetadata 对象
  return TensorMetadata{
      m,
      t.sizes().vec(),
      // 如果布局是 Strided，获取 strides，否则返回空向量
      m.layout_ == at::kStrided ? t.strides().vec() : std::vector<int64_t>()};
}

// 如果是 Tensor 对象则记录其 TensorMetadata
std::optional<TensorMetadata> ValueCache::recordIfTensor(py::handle p) {
  // 检查 p 是否为 THPVariable 类型的对象
  return THPVariable_CheckExact(p.ptr())
      ? std::optional<TensorMetadata>{toTensorMetadata(p.ptr())}
      : c10::nullopt;
}

// 解析包含 TensorMetadata 的字典，并返回其向量化的形式
std::vector<std::pair<std::string, TensorMetadata>> ValueCache::unpackTensorMap(
    const py::dict& tensor_map) {
  // 准备用于输出的向量
  std::vector<std::pair<std::string, TensorMetadata>> out;
  // 遍历 Python 字典中的每一项
  for (auto& it : tensor_map) {
    auto* value = it.second.ptr();
    // 如果键是字符串且对应的值是 THPVariable 对象，则将其加入输出向量
    if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(value)) {
      out.emplace_back(
          py::cast<std::string>(it.first), toTensorMetadata(value));
    }
  }
  // 返回解析后的键-值对向量
  return out;
}

// 存储 PyCallKey 对应的信息，用于跟踪 Python 调用的位置等
template <>
void ValueCache::store<CallType::PyCall>(const PyCallKey& key, no_ephemeral_t) {
  auto& locations = std::get<CallType::PyCall>(state_);
  // 如果 key 尚未存储在 locations 中，则存储其位置信息
  if (C10_UNLIKELY(locations.find(key) == locations.end())) {
    locations[key] = {
        key.line_number_,
        at::StringView(key.filename_),
        at::StringView(key.name_)};
  }
}

// 加载 PyCallKey 对应的额外字段参数
template <>
ExtraFields<EventType::PyCall>::args_t ValueCache::load<CallType::PyCall>(
    const PyCallKey& key) const {
  // 返回 PyCallKey 对应的参数
  return {std::get<CallType::PyCall>(state_).at(key), c10::nullopt};
}

// 存储 PyModuleCallKey 对应的信息及其参数
template <>
void ValueCache::store<CallType::PyModuleCall>(
    const PyModuleCallKey& key,
    Config<CallType::PyModuleCall>::ephemeral_t frame) {
  auto& cache = std::get<CallType::PyModuleCall>(state_);
  // 如果 key 尚未存储在 cache.cls_and_parameters_ 中，则进行存储
  if (C10_UNLIKELY(
          cache.cls_and_parameters_.find(key) ==
          cache.cls_and_parameters_.end())) {
    // 设置 PyModuleCallKey 对应的类和参数信息
    auto cls = set_class<CallType::PyModuleCall>(this, cache, key, frame);

    // 获取 key 对象的 _parameters 属性，准备解析其参数信息
    py::dict params = py::handle((PyObject*)key).attr("_parameters");
    std::vector<NNModuleInfo::ParameterInfo> params_;
    // 遍历参数字典中的每一项
    for (auto& it : params) {
      auto* p = it.second.ptr();
      // 如果键是字符串且对应的值是 THPVariable 对象，则将其加入参数列表
      if (py::isinstance<py::str>(it.first) && THPVariable_CheckExact(p)) {
        params_.push_back(
            {it.first.cast<std::string>(),
             toTensorMetadata(p),
             recordIfTensor(py::getattr(it.second, "grad", py::none()))});
      }
    }
    // 存储类和参数信息到 cache.cls_and_parameters_ 中
    cache.cls_and_parameters_[key] = {cls, std::move(params_)};
  }
}

// 加载 PyModuleCallKey 对应的额外字段参数
template <>
ExtraFields<EventType::PyCall>::args_t ValueCache::load<CallType::PyModuleCall>(
    const PyModuleCallKey& key) const {
  // 返回 PyModuleCallKey 对应的参数
  return {std::get<CallType::PyModuleCall>(state_).at(key), c10::nullopt};
}
    // 引用状态中的 PyModuleCall 缓存
    const PyModuleCallKey& key) const {
  // 获取 PyModuleCall 缓存
  auto& cache = std::get<CallType::PyModuleCall>(state_);
  // 断言缓存中的位置信息已设置
  TORCH_INTERNAL_ASSERT(cache.location_.has_value());
  // 获取指定 key 对应的类和参数信息
  const auto& cls_and_parameters = cache.cls_and_parameters_.at(key);
  // 获取类对象
  const auto& cls = cls_and_parameters.cls_;
  // 构造 NNModuleInfo 结构体
  NNModuleInfo info{
      key, cls, cache.cls_names_.at(cls), cls_and_parameters.parameters_};
  // 返回结构体包含的内容
  return {
      /*frame_state_=*/std::get<CallType::PyCall>(state_).at(*cache.location_),
      /*module_info_=*/std::move(info),
      /*optimizer_info_=*/c10::nullopt};
}

template <>
void ValueCache::store<CallType::PyOptimizerCall>(
    const PyOptimizerCallKey& key,
    Config<CallType::PyOptimizerCall>::ephemeral_t frame) {
  auto& cache = std::get<CallType::PyOptimizerCall>(state_);
  // 检查缓存中是否已存在给定键的数据
  if (C10_UNLIKELY(
          cache.cls_and_parameters_.find(key) ==
          cache.cls_and_parameters_.end())) {
    // 如果不存在，则设置该键的类和参数信息
    auto cls = set_class<CallType::PyOptimizerCall>(this, cache, key, frame);
    const py::handle self{(PyObject*)key};
    std::vector<OptimizerInfo::ParameterInfo> params;

    // 遍历每个参数组的参数列表
    for (const auto& i : (py::list)self.attr("param_groups")) {
      for (auto& param : py::cast<py::dict>(i).attr("get")("params")) {
        // 检查参数是否为 THPVariable 类型
        if (THPVariable_CheckExact(param.ptr())) {
          // 将参数的元数据、梯度信息和状态信息记录到参数信息中
          params.push_back(
              {toTensorMetadata(param.ptr()),
               recordIfTensor(py::getattr(param, "grad", py::none())),
               unpackTensorMap(py::cast<py::dict>(self.attr("state"))
                                   .attr("get")(param, py::dict()))});
        }
      }
    }

    // 将类和参数信息存入缓存
    cache.cls_and_parameters_[key] = {cls, std::move(params)};
  }
}

template <>
ExtraFields<EventType::PyCall>::args_t ValueCache::load<
    CallType::PyOptimizerCall>(const PyOptimizerCallKey& key) const {
  auto& cache = std::get<CallType::PyOptimizerCall>(state_);
  // 获取缓存中指定键的类和参数信息
  const auto& cls_and_parameters = cache.cls_and_parameters_.at(key);
  auto cls = cls_and_parameters.cls_;
  // 构建优化器信息对象
  OptimizerInfo info{
      key, cls, cache.cls_names_.at(cls), cls_and_parameters.parameters_};
  return {
      // 返回额外字段的参数
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      /*frame_state_=*/std::get<CallType::PyCall>(state_).at(*cache.location_),
      /*module_info_=*/c10::nullopt,
      /*optimizer_info_=*/std::move(info)};
}

template <>
void ValueCache::store<CallType::PyCCall>(
    const PyCCallKey& key,
    Config<CallType::PyCCall>::ephemeral_t arg) {
  auto& names = std::get<CallType::PyCCall>(state_);
  // 如果给定键不存在于缓存中，则存储参数
  if (C10_UNLIKELY(names.find(key) == names.end())) {
    names[key] = at::StringView(py::repr(arg));
  }
}

template <>
ExtraFields<EventType::PyCCall>::args_t ValueCache::load<CallType::PyCCall>(
    const PyCCallKey& key) const {
  // 返回指定键的参数
  return std::get<CallType::PyCCall>(state_).at(key);
}

// TODO: Use re2.
void ValueCache::trimPrefixes() {
  // 获取 Python 模块中的前缀列表
  static const auto prefixes = []() {
    pybind11::gil_scoped_acquire gil;
    return py::module::import("torch.profiler.python_tracer")
        .attr("_prefix_regex")()
        .cast<std::vector<std::string>>();
  }();

  // 遍历状态中的每个调用信息
  for (auto& it : std::get<CallType::PyCall>(state_)) {
    // 获取文件名并进行处理
    std::string filename = it.second.filename_.str();
    // 对于每个前缀字符串 p，在 filename 中检查是否以 p 开头
    for (const auto& p : prefixes) {
      // 如果 filename 以当前前缀 p 开头
      if (filename.compare(0, p.size(), p) == 0) {
        // 从 filename 中删除前缀 p
        filename.erase(0, p.size());
        // 更新 it 对应的元素的文件名为剩余的 filename
        it.second.filename_ = at::StringView(filename);
        // 找到匹配的前缀后，退出循环
        break;
      }
    }
  }
    nullptr  // 没有特定的析构函数
    0,       // 基本大小为 0，因为不需要额外的数据
    nullptr, // 不需要获取属性
    nullptr, // 不需要设置属性
    nullptr, // 保留字段为空
    nullptr, // 字符串表示为空
    nullptr, // 数字操作为空
    nullptr, // 序列操作为空
    nullptr, // 映射操作为空
    nullptr, // 哈希操作为空
    nullptr, // 调用操作为空
    nullptr, // 字符串表示为空
    nullptr, // 获取属性为空
    nullptr, // 设置属性为空
    nullptr, // 缓冲区为空
    Py_TPFLAGS_DEFAULT, // 默认的类型标志
    "Python tracer TLS", // 类型文档字符串
    nullptr, // 不需要遍历
    nullptr, // 不需要清除
    nullptr, // 不需要富比较
    0,       // 弱引用偏移量为 0
    nullptr, // 迭代器为空
    nullptr, // 迭代器下一个为空
    nullptr, // 方法为空
    nullptr, // 成员为空
    nullptr, // 获取设置为空
    nullptr, // 基类为空
    nullptr, // 字典为空
    nullptr, // 获取描述符为空
    nullptr  // 设置描述符为空
};
    0, /* tp_dictoffset */
    // 在类型对象结构体中的字典偏移量，通常不需要修改，因此设为0
    nullptr, /* tp_init */
    // 初始化函数指针，指向对象初始化时调用的函数，这里设为nullptr表示不调用特定初始化函数
    nullptr, /* tp_alloc */
    // 分配函数指针，用于为对象分配内存空间的函数，这里设为nullptr表示使用默认的内存分配方式
    PyType_GenericNew, /* tp_new */
    // 新建对象函数指针，用于创建新的对象实例的函数
    nullptr /* tp_free */
    // 释放函数指针，用于释放对象实例占用的内存空间的函数，这里设为nullptr表示使用默认的释放方式
};

// ============================================================================
// == Thread local cache ======================================================
// ============================================================================

// 线程本地缓存，用于存储每个线程的跟踪结果和相关状态
class PythonTracer;

// 线程本地结果结构体，包含了跟踪器所需的各种状态和缓存
struct ThreadLocalResults {
  // 构造函数，初始化线程状态、跟踪上下文、值缓存和活跃的跟踪器
  ThreadLocalResults(
      PyThreadState* thread_state,
      ValueCache* value_cache,
      PythonTracer* active_tracer)
      : thread_state_{thread_state},
        ctx_{(TraceContext*)TraceContextType.tp_alloc(&TraceContextType, 0)},
        value_cache_{value_cache},
        active_tracer_{active_tracer} {
    ctx_->thread_local_results_ = this;  // 将当前结果与跟踪上下文关联起来
  }

  // 删除默认构造函数和拷贝赋值构造函数，确保不会被误用
  ThreadLocalResults() = delete;
  ThreadLocalResults(const ThreadLocalResults&) = delete;
  ThreadLocalResults(ThreadLocalResults&&) = delete;
  ThreadLocalResults& operator=(const ThreadLocalResults&) = delete;
  ThreadLocalResults& operator=(const ThreadLocalResults&&) = delete;

  // 析构函数，释放跟踪上下文的引用
  ~ThreadLocalResults() {
    Py_DECREF((PyObject*)ctx_);
  }

  // 模板方法，用于将跟踪信息注册到缓存中
  template <CallType C, EventType E, typename Ephemeral, typename... Args>
  TraceKey intern(Ephemeral ephemeral, Args... args) {
    static_assert(
        Config<C>::event_type == E,
        "ThreadLocalResults.intern called from the wrong typed context.");
    auto callsite = Callsite<C>(std::forward<Args>(args)...);
    return std::get<C>(trace_keys_).intern(callsite, ephemeral, *value_cache_);
  }

  // 块大小常量
  static constexpr size_t BLOCK_SIZE = 1024;

  // 成员变量：线程状态、跟踪上下文、值缓存、活跃跟踪器、跟踪键集合和退出时间列表
  PyThreadState* thread_state_;
  TraceContext* ctx_;
  ValueCache* value_cache_;
  PythonTracer* active_tracer_;
  CallTypeHelper<TraceKeyCacheState>::tuple_type trace_keys_;
  AppendOnlyList<c10::approx_time_t, BLOCK_SIZE> exit_times_;
  AppendOnlyList<c10::approx_time_t, BLOCK_SIZE> c_exit_times_;
};

// ============================================================================
// == Tracing implementation ==================================================
// ============================================================================
// PythonTracer 类的定义，继承自 PythonTracerBase 类
class PythonTracer final : public python_tracer::PythonTracerBase {
 public:
  // 构造函数，接受 torch::profiler::impl::RecordQueue* 类型参数
  PythonTracer(torch::profiler::impl::RecordQueue* queue);
  // 析构函数，覆盖基类虚函数
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~PythonTracer() override;

  // 静态成员函数，用于 Python 的 profile 回调
  static int pyProfileFn(
      PyObject* obj,
      PyFrameObject* frame,
      int what,
      PyObject* arg);

  // 停止追踪函数，覆盖基类虚函数
  void stop() override;

  // 获取事件的函数，返回 std::vector<std::shared_ptr<Result>> 类型结果
  std::vector<std::shared_ptr<Result>> getEvents(
      std::function<c10::time_t(c10::approx_time_t)> time_converter,
      std::vector<python_tracer::CompressedEvent>& enters,
      c10::time_t end_time_ns) override;

  // 内部结构体，用于表示开始帧的信息
  struct StartFrame {
    TraceKey trace_key_; // 跟踪键
    c10::approx_time_t start_time{}; // 开始时间
  };

 private:
  // 记录 Python 调用的函数
  void recordPyCall(
      ThreadLocalResults& tls,
      PyFrameObject* frame,
      bool is_startup_frame);

  // 记录 C 调用的函数
  void recordCCall(
      ThreadLocalResults& tls,
      PyFrameObject* frame,
      PyObject* arg);

  // 返回解释器中所有线程状态的函数
  const std::vector<PyThreadState*> interpreterThreads() const;

  // 原子布尔类型，用于标识当前是否处于活跃状态
  std::atomic<bool> active_lock_{false};
  bool active_{false};

  // 记录队列指针
  torch::profiler::impl::RecordQueue* queue_;

  // Python 解释器状态指针
  PyInterpreterState* interpreter_;

  // 模块调用代码对象指针
  PyCodeObject* module_call_code_;

  // 优化器钩子代码对象指针
  PyCodeObject* optimizer_hook_;

  // 存储开始帧信息的向量
  std::vector<StartFrame> start_frames_;

  // 线程本地结果的双端队列
  std::deque<ThreadLocalResults> thread_local_results_;

  // 值缓存
  ValueCache value_cache_;
};

// 返回当前解释器的所有线程状态
const std::vector<PyThreadState*> PythonTracer::interpreterThreads() const {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  std::vector<PyThreadState*> out;
  if (SOFT_ASSERT(interpreter_)) {
    // 获取解释器的第一个线程状态
    auto* thread_state = PyInterpreterState_ThreadHead(interpreter_);
    // 遍历所有线程状态，并加入输出向量
    while (thread_state != nullptr) {
      out.push_back(thread_state);
      thread_state = PyThreadState_Next(thread_state);
    }
  }
  return out; // 返回线程状态向量
}

// PythonTracer 类的构造函数定义
PythonTracer::PythonTracer(torch::profiler::impl::RecordQueue* queue)
    : queue_(queue),
      interpreter_(nullptr),
      module_call_code_(getCode<CallType::PyModuleCall>()),
      optimizer_hook_(getCode<CallType::PyOptimizerCall>()) {
  // 检查记录队列非空
  TORCH_CHECK(queue_ != nullptr);

  // 初始化活跃状态，只允许一个 Python 追踪器活跃
  bool expected{false};
  active_ = active_lock_.compare_exchange_strong(expected, true);
  if (!active_) {
    // 如果已有活跃追踪器，发出警告并返回
    TORCH_WARN(
        "There is already an active Python tracer. "
        "Refusing to register profile functions.");
    return;
  }

  // 获取全局解释器锁，并恢复线程状态
  gil_and_restore_thread gil;
  interpreter_ = PyInterpreterState_Get();

  if (!gil.initial_thread_state()) {
    // 如果初始线程状态为空，发出警告并返回
    TORCH_WARN("PyThreadState_Get returned NULL");
    return;
  }

  // 在每个线程中注册追踪器
  for (const auto thread_state : interpreterThreads()) {
    PyThreadState_Swap(thread_state);

    // 向线程本地结果队列中添加新的结果
    thread_local_results_.emplace_back(thread_state, &value_cache_, this);
    auto* ctx = thread_local_results_.back().ctx_;

    // 开始追踪时，Python 解释器栈上已有帧，需将所有先前帧的调用推入事件栈（深度限制为 128）
    std::vector<THPFrameObjectPtr> current_stack;
    auto frame = PyEval_GetFrame();
    Py_XINCREF(frame);
    size_t depth = 0; // 确保不会无限循环的深度计数器

    // 循环遍历 Python 帧栈直到为空
    while (frame != nullptr) {
      // 将当前帧添加到堆栈中
      current_stack.emplace_back(frame);
      // 如果深度达到 128 层，则中断循环
      if (++depth == 128) {
        break;
      }

      // 注意：`PyFrame_GetBack` 返回一个强引用
      frame = PyFrame_GetBack(frame);
    }

    // 逆序遍历当前堆栈中的帧
    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      // 记录 Python 函数调用，同时记录堆栈信息
      recordPyCall(thread_local_results_.back(), it->get(), true);
      // 获取当前帧对象的引用计数
      auto frame_refcount = Py_REFCNT(it->get());

      // 断言：我们在 `current_stack` 持有一个引用，解释器也持有另一个引用
      TORCH_INTERNAL_ASSERT(frame_refcount >= 2, frame_refcount);
    }

    // 注意：
    //   这个性能分析器与其他 CPython 分析器不兼容，
    //   也不能通过 `sys.settrace(sys.gettrace())` 实现往返操作
    PyEval_SetProfile(PythonTracer::pyProfileFn, (PyObject*)ctx);
};

// 停止 Python 追踪器，清除所有设置
void PythonTracer::stop() {
  // 获取全局解释器锁并恢复线程
  gil_and_restore_thread gil;
  // 如果追踪器当前处于活动状态
  if (active_) {
    // 遍历所有解释器线程状态
    for (const auto thread_state : interpreterThreads()) {
      // 如果线程的 c_profilefunc 是 PythonTracer::pyProfileFn 函数
      if (thread_state->c_profilefunc == &PythonTracer::pyProfileFn) {
        // 交换线程状态以设置新的 profile 函数为 nullptr
        PyThreadState_Swap(thread_state);
        // 设置 Python 运行时 profile 为 nullptr
        PyEval_SetProfile(nullptr, nullptr);
      }
    }

    // 尝试原子地将 active_ 标志设置为 false
    auto lock_returned = active_lock_.compare_exchange_strong(active_, false);
    // 断言锁设置成功，否则输出错误消息
    SOFT_ASSERT(lock_returned, "Failed to return python tracer lock.");
  }
}

// 析构函数，销毁 Python 追踪器对象
// NOLINTNEXTLINE(bugprone-exception-escape)
PythonTracer::~PythonTracer() {
  // 如果追踪器当前处于活动状态
  if (active_) {
    // 发出警告，提示未调用 PythonTracer::stop()
    TORCH_WARN("`PythonTracer::stop()` was not called.");
    // 调用 stop() 方法来确保清除所有设置
    stop();
  }
}

// 记录 Python 函数调用事件
void PythonTracer::recordPyCall(
    ThreadLocalResults& tls,
    PyFrameObject* frame,
    bool is_startup_frame) {
  // 定义事件类型为 PyCall
  static constexpr auto E = EventType::PyCall;
  // 根据代码对象获取跟踪键
  const auto key = [&]() -> TraceKey {
    auto code = THPCodeObjectPtr(PyFrame_GetCode(frame));
    // 如果是模块调用代码
    if (code.get() == module_call_code_) {
      // 获取本地变量字典，并从中获取名为 "self" 的项目
      auto locals = THPObjectPtr(PyFrame_GetLocals(frame));
      auto self = THPObjectPtr(PyDict_GetItemString(locals, "self"));
      Py_INCREF(self.get());
      // 获取当前帧的后续帧
      auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
      TORCH_INTERNAL_ASSERT(back != nullptr);
      // 创建 PyModuleCall 类型的事件，并返回跟踪键
      return tls.intern<CallType::PyModuleCall, E>(
          frame, self.get(), back.get());
    } else if (code.get() == optimizer_hook_) {
      // 如果是优化器钩子代码
      auto locals = THPObjectPtr(PyFrame_GetLocals(frame));
      auto self = THPObjectPtr(PyDict_GetItemString(locals, "self"));
      Py_INCREF(self.get());
      auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
      TORCH_INTERNAL_ASSERT(back != nullptr);
      // 创建 PyOptimizerCall 类型的事件，并返回跟踪键
      return tls.intern<CallType::PyOptimizerCall, E>(
          frame, self.get(), back.get());
    } else {
      // 否则，创建 PyCall 类型的事件，并返回跟踪键
      auto back = THPFrameObjectPtr(PyFrame_GetBack(frame));
      auto f_back = (back.get() != nullptr) ? back.get() : frame;
      return tls.intern<CallType::PyCall, E>(no_ephemeral_t(), frame, f_back);
    }
  }();
  // 获取近似时间戳
  const auto time = c10::getApproximateTime();
  // 如果是启动帧，则添加到启动帧列表中，否则添加到事件队列中
  is_startup_frame ? start_frames_.push_back({key, time})
                   : queue_->getSubqueue()->emplace_py_call(key, time);
}
    PyObject* arg) {
```  
# 接受一个 PyObject* 类型的参数 `arg`。


  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(PyCFunction_Check(arg));
```  
# 使用宏 `TORCH_INTERNAL_ASSERT_DEBUG_ONLY` 断言 `arg` 是一个 PyCFunction 对象。


  auto fn = reinterpret_cast<PyCFunctionObject*>(arg);
```  
# 将 `arg` 强制转换为 `PyCFunctionObject*` 类型的指针，并赋值给 `fn`。


  // NB: For C calls a new frame is not created, so we use `frame` rather than
  //     `frame->f_back`.
```  
# 注意：对于 C 调用，不会创建新的帧（frame），因此我们使用 `frame` 而不是 `frame->f_back`。


  auto key = tls.intern<CallType::PyCCall, EventType::PyCCall>(
      arg, (void*)(fn->m_ml), frame);
```  
# 使用 `tls.intern` 方法以 `CallType::PyCCall` 和 `EventType::PyCCall` 的类型创建一个键 `key`，传递参数 `arg`、函数指针 `(void*)(fn->m_ml)` 和 `frame`。


  queue_->getSubqueue()->emplace_py_call(key, c10::getApproximateTime());
```  
# 在队列 (`queue_`) 的子队列中插入一个 Python 调用，使用键 `key` 和当前的近似时间 (`c10::getApproximateTime()`)。
// ============================================================================
// == Post processing =========================================================
// ============================================================================

struct Exit {
  // 定义比较运算符，用于退出事件按时间戳 t_ 进行比较
  bool operator>(const Exit& other) const {
    return t_ > other.t_;
  }

  c10::time_t t_;        // 事件的时间戳
  size_t python_tid_;    // Python 线程 ID
};

class PostProcess {
 public:
  // 构造函数，接受时间转换器、线程局部结果队列、数值缓存和结束时间参数
  PostProcess(
      std::function<c10::time_t(c10::approx_time_t)> time_converter,
      std::deque<ThreadLocalResults>& tls,
      const ValueCache& value_cache,
      c10::time_t end_time_ns)
      : end_time_{end_time_ns}, time_converter_{std::move(time_converter)} {
    // 遍历每个 Python 线程的结果队列
    for (size_t python_tid : c10::irange(tls.size())) {
      // 处理 TraceKeyCacheState 的映射
      CallTypeHelper<TraceKeyCacheState>::map(
          tls[python_tid].trace_keys_, *this, value_cache, python_tid);

      // 添加 EventType::PyCall 类型的退出事件
      addExits<EventType::PyCall>(tls[python_tid].exit_times_, python_tid);
      // 添加 EventType::PyCCall 类型的 C 退出事件
      addExits<EventType::PyCCall>(tls[python_tid].c_exit_times_, python_tid);
    }
  }

  // 设置起始帧信息
  void set_start_frames(
      const std::vector<PythonTracer::StartFrame>& start_frames,
      std::vector<python_tracer::CompressedEvent>& enters) {
    // 遍历起始帧信息
    for (const auto& frame : start_frames) {
      // 将起始帧信息添加到进入事件列表中
      enters.push_back(
          {frame.trace_key_,
           NoTID, // 允许检测未处理的起始帧
           {},
           time_converter_(frame.start_time)});
    }
  }

  // 处理 TraceKeyCacheState 中不同 CallType 的模板函数
  template <CallType C>
  void operator()(
      const TraceKeyCacheState<C>& trace_cache,
      const ValueCache& value_cache,
      size_t python_tid) {
    // 遍历每个状态并将其加载到事件类型的字段中
    for (const auto& it : trace_cache.state_) {
      const auto inserted = get_state<Config<C>::event_type>().fields_.insert(
          {it.second, value_cache.load(it.first, python_tid)});
      TORCH_INTERNAL_ASSERT(inserted.second, "Duplicate key: ", it.second);
    }
  }

  // 添加退出事件到状态中
  template <EventType E, size_t N>
  void addExits(
      AppendOnlyList<c10::approx_time_t, N>& exits,
      size_t python_tid) {
    // 遍历退出事件列表并将其添加到相应事件类型的状态中
    for (const auto i : exits) {
      get_state<E>().exits_.push({time_converter_(i), python_tid});
    }
  }

  // 运行后处理逻辑并返回结果
  std::vector<std::shared_ptr<Result>> run(
      std::vector<python_tracer::CompressedEvent>& enters) {
    // 对进入事件按时间戳进行稳定排序
    std::stable_sort(
        enters.begin(), enters.end(), [](const auto a, const auto b) {
          return a.enter_t_ < b.enter_t_;
        });
    std::vector<std::shared_ptr<Result>> out;
    // 填充 PyCall 类型的结果
    populate<EventType::PyCall>(enters, out);
    // 填充 PyCCall 类型的结果
    populate<EventType::PyCCall>(enters, out);
    return out;
  }

 private:
  // 填充特定事件类型的结果
  template <EventType E>
  void populate(
      std::vector<python_tracer::CompressedEvent>& enters,
      std::vector<std::shared_ptr<Result>>& out) {
    using stack_t = std::vector<std::shared_ptr<Result>>;
    const auto initial_size = out.size();
    auto pop = [](stack_t& stack, c10::time_t t) {
      TORCH_INTERNAL_ASSERT(!stack.empty(), "Python replay stack is empty.");
      // 设置退出时间
      std::get<ExtraFields<E>>(stack.back()->extra_fields_).end_time_ns_ = t;
      stack.pop_back();
    };
    // 创建一个哈希映射，将 size_t 映射到 stack_t 类型的堆栈
    ska::flat_hash_map<size_t, stack_t> stacks;
    // 获取指定事件类型 E 对应的状态引用
    auto& state = get_state<E>();
    // 遍历 enters 容器中的每一个事件 enter
    for (const auto& enter : enters) {
      // 查找进入事件的 key_ 是否存在于 state 中的 fields_ 映射中
      auto fields_it = state.fields_.find(enter.key_);
      // 如果找到对应的字段信息
      if (fields_it != state.fields_.end()) {
        // 处理状态中已记录的退出事件，直到栈顶的退出事件的时间戳 t_ 不再小于当前进入事件的时间戳 enter_t_
        while (!state.exits_.empty() &&
               state.exits_.top().t_ < enter.enter_t_) {
          // 弹出栈顶的退出事件，并执行相应的操作
          auto& exit = state.exits_.top();
          pop(stacks[exit.python_tid_], exit.t_);
          state.exits_.pop();
        }
        // 将生成的结果添加到 out 容器中
        out.push_back(Result::create(
            enter.enter_t_,
            enter.system_tid_,
            enter.kineto_info_,
            fields_it->second));

        // 将结果压入对应 Python 线程 ID 的堆栈中
        stacks[fields_it->second.python_tid_].push_back(out.back());
      }
    }

    // 处理仍在运行中的事件，直到所有堆栈为空
    for (auto& i : stacks) {
      while (!i.second.empty()) {
        pop(i.second, end_time_);
      }
    }

    // 根据相同的 Python 线程 ID，为开始事件分配系统线程 ID
    ska::flat_hash_map<size_t, std::pair<size_t, kineto::DeviceAndResource>>
        tid_map;
    auto it = out.rbegin();
    // 对结果容器中的元素进行逆序迭代处理
    for (C10_UNUSED auto _ : c10::irange(initial_size, out.size())) {
      // 获取当前事件的 Python 线程 ID
      const auto python_tid =
          std::get<ExtraFields<E>>((*it)->extra_fields_).python_tid_;
      // 如果当前事件的开始线程 ID 为 NoTID，并且满足条件 E == EventType::PyCall
      if ((*it)->start_tid_ == NoTID && SOFT_ASSERT(E == EventType::PyCall)) {
        // 尝试插入 Python 线程 ID 到 tid_map 中，如果已存在则返回对应的信息
        const auto& tid_info =
            tid_map.insert({python_tid, {NoTID, kineto::DeviceAndResource()}})
                .first->second;
        // 更新当前事件的开始线程 ID 和 kineto 信息
        (*it)->start_tid_ = tid_info.first;
        (*it)->kineto_info_ = tid_info.second;
      }
      // 更新 tid_map 中的 Python 线程 ID 对应的开始线程 ID 和 kineto 信息
      tid_map[python_tid] = {(*it)->start_tid_, (*it)->kineto_info_};
      ++it;
    }
  }

  // 定义一个模板结构体 State，用于存储特定事件类型 E 的字段和退出事件队列
  template <EventType E>
  struct State {
    ska::flat_hash_map<TraceKey, ExtraFields<E>> fields_;
    std::priority_queue<Exit, std::vector<Exit>, std::greater<>> exits_;
  };

  // 获取特定事件类型 E 对应的状态对象的引用
  template <EventType E>
  auto& get_state() {
    // 根据事件类型 E 返回相应的状态对象
    return std::get < E == EventType::PyCall ? 0 : 1 > (state_);
  }

  // 记录 profiling 结束时的时间戳
  c10::time_t end_time_;
  // 定义一个函数对象，用于将 approx_time_t 类型转换为 c10::time_t 类型
  std::function<c10::time_t(c10::approx_time_t)> time_converter_;
  // 包含两个不同事件类型（PyCall 和 PyCCall）的状态对象的元组
  std::tuple<State<EventType::PyCall>, State<EventType::PyCCall>> state_;
};

// 定义一个结构体 PythonIDVisitor，用于访问并处理与 Python 事件相关的附加字段
struct PythonIDVisitor {
  // 处理 PyCall 类型事件的操作符重载函数
  void operator()(ExtraFields<EventType::PyCall>& py_call) {
    // 分配一个唯一的 Python ID 给当前的 PyCall 事件
    py_call.id_ = ++current_python_id_;
    // 如果模块存在，则为其分配一个唯一的模块 ID
    if (py_call.module_.has_value()) {
      auto& m = py_call.module_;
      auto& module_ids = module_ids_[m->cls_];
      m->id_ = module_ids.insert({m->self_, module_ids.size()}).first->second;
    }
  }

  // 处理 PyCCall 类型事件的操作符重载函数
  void operator()(ExtraFields<EventType::PyCCall>& py_call) {
    // 分配一个唯一的 Python ID 给当前的 PyCCall 事件
    py_call.id_ = ++current_python_id_;
  }

  // 对于除 PyCall 和 PyCCall 外的所有事件类型，不执行任何操作
  template <typename T>
  void operator()(T&) {}

  size_t current_python_id_{0};  // 当前 Python 事件的计数器
  ska::flat_hash_map<PyModuleCls, ska::flat_hash_map<PyModuleSelf, size_t>>
      module_ids_;  // 模块到其唯一 ID 的映射表
};

// PythonTracer 类的方法，获取事件列表并进行处理
std::vector<std::shared_ptr<Result>> PythonTracer::getEvents(
    std::function<c10::time_t(c10::approx_time_t)> time_converter,
    std::vector<python_tracer::CompressedEvent>& enters,
    c10::time_t end_time_ns) {
  // 清理值缓存的前缀
  value_cache_.trimPrefixes();
  // 创建后处理对象，并初始化其参数
  PostProcess post_process(
      std::move(time_converter),
      thread_local_results_,
      value_cache_,
      end_time_ns);
  // 设置起始帧和进入事件
  post_process.set_start_frames(start_frames_, enters);
  // 运行后处理过程，得到处理后的事件列表
  auto out = post_process.run(enters);

  // 对事件列表按开始时间进行稳定排序
  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a->start_time_ns_ < b->start_time_ns_;
  });

  // 创建并使用 PythonIDVisitor 对象，为每个事件的附加字段分配唯一的 Python ID
  PythonIDVisitor id_visitor;
  for (auto& i : out) {
    std::visit(id_visitor, i->extra_fields_);
  }

  // 返回处理后的事件列表
  return out;
}

// PythonTracer 类的方法，处理 Python 的追踪函数
int PythonTracer::pyProfileFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  auto& local_results =
      *reinterpret_cast<TraceContext*>(obj)->thread_local_results_;
  switch (what) {
    case PyTrace_CALL:
      // 记录 Python 调用事件
      local_results.active_tracer_->recordPyCall(local_results, frame, false);
      break;

    case PyTrace_C_CALL:
      // 记录 C 函数调用事件
      local_results.active_tracer_->recordCCall(local_results, frame, arg);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      // 记录 Python 异常或返回事件的退出时间
      local_results.exit_times_.emplace_back(c10::getApproximateTime());
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      // 记录 C 异常或返回事件的退出时间
      local_results.c_exit_times_.emplace_back(c10::getApproximateTime());
      break;
  }
  return 0;
}

// 获取 PythonTracer 的唯一指针，用于 Python 追踪
std::unique_ptr<python_tracer::PythonTracerBase> getTracer(
    torch::profiler::impl::RecordQueue* queue) {
  return std::make_unique<PythonTracer>(queue);
}

// 初始化函数，用于设置 Python 追踪器
namespace torch {
namespace autograd {
namespace profiler {
namespace python_tracer {

void init() {
  // 获取全局解释器锁
  pybind11::gil_scoped_acquire gil;
  // 检查并准备 Python 追踪上下文类型
  TORCH_CHECK(PyType_Ready(&torch::profiler::impl::TraceContextType) == 0);
  // 注册 Python 追踪器
  torch::profiler::impl::python_tracer::registerTracer(
      &torch::profiler::impl::getTracer);
}
} // namespace python_tracer
} // namespace profiler
} // namespace autograd
} // 结束 torch 的命名空间
```