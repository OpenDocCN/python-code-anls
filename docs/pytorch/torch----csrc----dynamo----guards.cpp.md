# `.\pytorch\torch\csrc\dynamo\guards.cpp`

```py
// 定义宏以确保使用清理的 ssize_t 值
#define PY_SSIZE_T_CLEAN
// 包含空张量相关的头文件
#include <ATen/EmptyTensor.h>
// 包含 flat_hash_map，一个高性能的哈希表实现
#include <c10/util/flat_hash_map.h>
// 包含自动微分模式相关的头文件
#include <torch/csrc/autograd/grad_mode.h>
// 包含包装输出的实用工具函数
#include <torch/csrc/autograd/utils/wrap_outputs.h>
// 包含 Dynamo 模块的保护头文件
#include <torch/csrc/dynamo/guards.h>
// 包含 Inductor 模块的操作头文件
#include <torch/csrc/inductor/inductor_ops.h>
// 包含禁用 Torch 函数相关的头文件
#include <torch/csrc/utils/disable_torch_function.h>
// 包含 Python 参数解析相关的头文件
#include <torch/csrc/utils/python_arg_parser.h>
// 包含 Python 兼容性相关的头文件
#include <torch/csrc/utils/python_compat.h>
// 包含 Python 数字处理相关的头文件
#include <torch/csrc/utils/python_numbers.h>
// 包含 Python 符号节点相关的头文件
#include <torch/csrc/utils/python_symnode.h>
// 包含 Python C API 兼容性相关的头文件
#include <torch/csrc/utils/pythoncapi_compat.h>
// 包含 Torch 扩展相关的头文件
#include <torch/extension.h>

#ifdef USE_CUDA
// 如果使用 CUDA，则包含 CUDA 空张量相关的头文件
#include <ATen/cuda/EmptyTensor.h>
#endif

// 包含标准字符串流相关的头文件
#include <sstream>
// 包含实用工具的头文件
#include <utility>

// 对于 TupleIteratorGetItemAccessor，我们需要一种快速检索底层元组并访问项目的方法。
// 在 Python 3.12 版本之前，数据结构位于 tupleobject.c 文件中。为了处理这个，
// 我们手动复制这个结构并手动将其强制转换为新的结构。
// 从 3.12 版本开始，这个结构已经包含在头文件中。
#if IS_PYTHON_3_12_PLUS

#define Py_BUILD_CORE
// 从头文件中引入 _PyTupleIterObject
#include <internal/pycore_tuple.h>
#undef Py_BUILD_CORE

#else

// 手动创建 _PyTupleIterObject 结构体
typedef struct {
  PyObject_HEAD Py_ssize_t it_index;
  PyTupleObject* it_seq; /* 当迭代器耗尽时设置为 NULL */
} _PyTupleIterObject;

#endif // IS_PYTHON_3_12_PLUS

namespace torch::dynamo {

// 定义一个宏来跳过重复添加的守卫，如 EQUALS_MATCH
#define SKIP_IF_GUARD_ALREADY_PRESENT(name) \
  if (self.is_leaf_guard_present(name)) {   \
    return;                                 \
  }                                         \
  self.insert_leaf_guard(name);

// TensorCheck 类的构造函数，用于检查张量相关属性
TensorCheck::TensorCheck(
    const LocalState& state,
    PyTypeObject* pt,
    const at::Tensor& v,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_strides)
    : pytype(pt),  // 设置 Python 类型对象
      dispatch_key_(state.apply(v.key_set()).raw_repr()),  // 设置调度键
      dtype_(v.dtype().toScalarType()),  // 设置数据类型
      device_index_(v.device().index()),  // 设置设备索引
      requires_grad_(v.requires_grad()),  // 设置是否需要梯度
      sizes_(std::move(dynamic_dims_sizes)),  // 设置尺寸信息
      strides_(std::move(dynamic_dims_strides)),  // 设置步幅信息
      dim_(static_cast<int64_t>(sizes_.size())) {  // 设置维度信息
  // TODO(voz): 在尺寸和步幅完全动态的情况下，是否应该将其视为可选？
}

// TensorCheck 类的另一个构造函数，用于检查张量相关属性
TensorCheck::TensorCheck(
    const LocalState& state,
    PyTypeObject* pt,
    uint64_t dispatch_key,
    at::ScalarType dtype,
    at::DeviceIndex device_index,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_sizes,
    std::vector<std::optional<c10::SymInt>> dynamic_dims_strides)
    : pytype(pt),  // 设置 Python 类型对象
      dispatch_key_(dispatch_key),  // 设置调度键
      dtype_(dtype),  // 设置数据类型
      device_index_(device_index),  // 设置设备索引
      requires_grad_(false),  // 初始化不需要梯度
      sizes_(std::move(dynamic_dims_sizes)),  // 设置尺寸信息
      strides_(std::move(dynamic_dims_strides)),  // 设置步幅信息
      dim_(static_cast<int64_t>(sizes_.size())) {  // 设置维度信息
}
    # 初始化函数，用于创建一个张量对象
    : pytype(pt),  # 指定张量的类型
    dispatch_key_(dispatch_key),  # 分发键，用于指示张量的分发策略
    dtype_(dtype),  # 张量的数据类型
    device_index_(device_index),  # 设备索引，指示张量存储的设备
    requires_grad_(false),  # 标志张量是否需要梯度，默认为false，表示不需要
    sizes_(std::move(dynamic_dims_sizes)),  # 张量的维度大小，移动语义传递动态维度的大小
    strides_(std::move(dynamic_dims_strides)),  # 张量的步长，移动语义传递动态维度的步长
    dim_(static_cast<int64_t>(sizes_.size())) {}  # 张量的维度数量，通过静态类型转换获取
// 检查给定的张量是否符合预期的条件
bool TensorCheck::check(const LocalState& state, const at::Tensor& v) {
  // 检查分派键是否与状态中的应用结果一致
  if (dispatch_key_ != state.apply(v.key_set()).raw_repr() ||
      // 检查张量的数据类型是否符合预期
      dtype_ != v.dtype().toScalarType() ||
      // 检查张量的设备索引是否符合预期
      device_index_ != v.device().index() ||
      // 检查张量的梯度属性是否符合预期
      requires_grad_ != v.requires_grad()) {
    return false;
  }
  auto ndim = v.ndimension();
  // 检查张量的维度是否与预期的维度一致
  if (ndim != dim_) {
    return false;
  }
  const auto& sizes = v.sym_sizes();
  const auto& strides = v.sym_strides();
  // 遍历张量的每个维度，检查尺寸和步长是否符合预期
  for (auto i : c10::irange(ndim)) {
    auto known_size = sizes_[i];
    auto known_stride = strides_[i];
    if (known_size.has_value()) {
      if (known_size.value() != sizes[i]) {
        return false;
      }
    }
    if (known_stride.has_value()) {
      if (known_stride.value() != strides[i]) {
        return false;
      }
    }
  }
  // 所有条件都符合预期，返回 true
  return true;
}

// 提供详细的失败原因描述，以帮助调试
std::string TensorCheck::check_verbose(
    const LocalState& state,
    const at::Tensor& v,
    const std::string& tensor_name) {
  std::stringstream fail_reason;
  fail_reason << "tensor '" << tensor_name << "' ";
  // 检查分派键是否与状态中的应用结果一致
  if (dispatch_key_ != state.apply(v.key_set()).raw_repr()) {
    fail_reason << "dispatch key set mismatch. expected "
                << c10::DispatchKeySet(c10::DispatchKeySet::RAW, dispatch_key_)
                << ", actual " << state.apply(v.key_set());
    return fail_reason.str();
  } else if (dtype_ != v.dtype().toScalarType()) {
    fail_reason << "dtype mismatch. expected " << dtype_ << ", actual "
                << v.dtype().toScalarType();
    return fail_reason.str();
  } else if (device_index_ != v.device().index()) {
    fail_reason << "Tensor device index mismatch. Expected device index to be "
                << device_index_ << ", actual " << v.device().index();
    return fail_reason.str();
  } else if (requires_grad_ != v.requires_grad()) {
    fail_reason << "requires_grad mismatch. expected requires_grad="
                << requires_grad_;
    return fail_reason.str();
  }
  auto ndim = v.ndimension();
  // 检查张量的维度是否与预期的维度一致
  if (ndim != dim_) {
    fail_reason << "rank mismatch. expected " << sizes_.size() << ", actual "
                << ndim;
    return fail_reason.str();
  }
  const auto& sizes = v.sym_sizes();
  const auto& strides = v.sym_strides();
  // 遍历张量的每个维度，检查尺寸和步长是否符合预期
  for (auto i : c10::irange(ndim)) {
    auto known_size = sizes_[i];
    auto known_stride = strides_[i];
    # 检查是否存在已知的大小，并且当前索引处的大小与已知大小不匹配
    if (known_size.has_value() && (known_size.value() != sizes[i])) {
        # 构建失败原因字符串，指明大小不匹配的索引及预期和实际大小
        fail_reason << "size mismatch at index " << i << ". expected "
                    << known_size.value() << ", actual " << sizes[i];
        # 返回失败原因的字符串表示
        return fail_reason.str();
    }
    # 检查是否存在已知的步幅，并且当前索引处的步幅与已知步幅不匹配
    if (known_stride.has_value() && known_stride.value() != strides[i]) {
        # 构建失败原因字符串，指明步幅不匹配的索引及预期和实际步幅
        fail_reason << "stride mismatch at index " << i << ". expected "
                    << known_stride.value() << ", actual " << strides[i];
        # 返回失败原因的字符串表示
        return fail_reason.str();
    }
}
# 如果以上条件都未触发返回，则表示所有检查通过，返回空字符串表示成功
return "";
}

namespace {

typedef std::vector<TensorCheck> ChecksList;  // 定义存储 TensorCheck 对象的向量类型 ChecksList

typedef struct {  // 定义一个结构体 TensorGuards，继承自 PyObject_HEAD，包含 ChecksList 指针
  PyObject_HEAD;
  ChecksList* checks;
} TensorGuards;

static void TensorGuards_dealloc(TensorGuards* self) {  // TensorGuards 对象的析构函数实现
  if (self->checks != nullptr) {  // 检查 checks 指针是否为空
    delete self->checks;  // 删除 checks 指针指向的对象
    self->checks = nullptr;  // 将 checks 指针置为空
  }
  Py_TYPE(self)->tp_free((PyObject*)self);  // 调用父类的释放函数释放对象内存
}

static PyObject* TensorGuards_new(  // TensorGuards 对象的创建函数实现
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
  TensorGuards* self = (TensorGuards*)type->tp_alloc(type, 0);  // 分配 TensorGuards 对象内存空间
  if (self != nullptr) {
    self->checks = new ChecksList();  // 创建新的 ChecksList 对象并赋值给 self->checks
  }
  return (PyObject*)self;  // 返回创建的 TensorGuards 对象
}

static std::vector<std::optional<c10::SymInt>> wrapIntegersInOptional(  // 将 c10::SymIntArrayRef 转换为 std::vector<std::optional<c10::SymInt>> 的辅助函数
    const c10::SymIntArrayRef& intArray) {
  std::vector<std::optional<c10::SymInt>> optVec(intArray.size());  // 创建大小与 intArray 相同的 std::vector<std::optional<c10::SymInt>>
  std::transform(  // 对 intArray 中的元素进行转换并存入 optVec 中
      intArray.begin(),
      intArray.end(),
      optVec.begin(),
      [](const c10::SymInt& value) { return std::make_optional(value); });
  return optVec;  // 返回包含转换后数据的 vector
}

static std::vector<std::optional<c10::SymInt>> pyListToVecOptInt(  // 将 Python 列表转换为 std::vector<std::optional<c10::SymInt>> 的辅助函数
    PyObject* pyList) {
  std::vector<std::optional<c10::SymInt>> vec;  // 创建一个空的 std::vector<std::optional<c10::SymInt>>
  Py_ssize_t size = PyList_Size(pyList);  // 获取 Python 列表的大小
  for (Py_ssize_t i = 0; i < size; i++) {  // 遍历 Python 列表中的每个元素
    PyObject* item = PyList_GetItem(pyList, i);  // 获取列表中的元素对象
    auto handle = py::handle(item);  // 使用 py::handle 封装 Python 对象
    if (item == Py_None) {  // 如果元素是 None
      vec.emplace_back(std::nullopt);  // 在 vec 中添加 std::nullopt 表示空值
    } else if (torch::is_symint(handle)) {  // 如果元素是 torch 的 SymInt 类型
      vec.emplace_back(py::cast<c10::SymInt>(handle));  // 将元素转换为 c10::SymInt 并添加到 vec 中
    } else {  // 其他情况
      int64_t value = PyLong_AsLongLong(item);  // 将 Python 对象转换为 int64_t 类型
      if (value == -1 && PyErr_Occurred()) {  // 如果转换出错
        PyErr_SetString(  // 设置 Python 异常信息
            PyExc_TypeError,
            "Size or stride list item is not a valid integer.");
        TORCH_CHECK(false, "Size or stride list item is not a valid integer.");  // 抛出 Torch 的检查错误
      }
      vec.emplace_back(c10::SymInt(value));  // 将转换后的值包装为 c10::SymInt 并添加到 vec 中
    }
  }
  return vec;  // 返回包含转换后数据的 vector
}

static std::vector<std::vector<std::optional<c10::SymInt>>> get_dynamic_dims(  // 获取动态维度的辅助函数
    PyObject* dynamic_dims_py) {
  std::vector<std::vector<std::optional<c10::SymInt>>> per_tensor_dynamic_dims;  // 创建用于存储每个张量动态维度的向量
  if (dynamic_dims_py != Py_None) {  // 检查 dynamic_dims_py 是否不为空
    Py_ssize_t size = PyList_Size(dynamic_dims_py);  // 获取 dynamic_dims_py 的大小
    for (Py_ssize_t i = 0; i < size; i++) {  // 遍历 dynamic_dims_py 中的每个元素
      PyObject* py_list = PyList_GetItem(dynamic_dims_py, i);  // 获取列表中的元素对象
      std::vector<std::optional<c10::SymInt>> vec = pyListToVecOptInt(py_list);  // 调用 pyListToVecOptInt 函数将 Python 列表转换为向量
      per_tensor_dynamic_dims.push_back(std::move(vec));  // 将转换后的向量添加到 per_tensor_dynamic_dims 中
    }
  }
  return per_tensor_dynamic_dims;  // 返回包含动态维度的向量的向量
}

static int TensorGuards_init(  // TensorGuards 对象的初始化函数实现
    TensorGuards* self,
    PyObject* args,
    PyObject* kwds) {
  if (!PyTuple_CheckExact(args)) {  // 检查传入的参数是否是元组
    PyErr_SetString(PyExc_TypeError, "expected tuple()");  // 设置 Python 异常信息
    return -1;  // 返回错误状态码
  }
  // Top level structure is List[List[Union[int, None]]]
  PyObject* dynamic_dims_sizes_py =  // 获取 dynamic_dims_sizes 参数
      PyDict_GetItemString(kwds, "dynamic_dims_sizes");
  if (dynamic_dims_sizes_py == nullptr) {  // 检查 dynamic_dims_sizes 参数是否为空
    PyErr_SetString(PyExc_TypeError, "missing dynamic_dims_sizes=...");  // 设置 Python 异常信息
    return -1;  // 返回错误状态码
  }
  PyObject* dynamic_dims_strides_py =  // 获取 dynamic_dims_strides 参数
      PyDict_GetItemString(kwds, "dynamic_dims_strides");
  if (dynamic_dims_strides_py == nullptr) {  // 检查 dynamic_dims_strides 参数是否为空
    // 设置类型错误异常，并指定错误消息为 "missing dynamic_dims_strides=..."
    PyErr_SetString(PyExc_TypeError, "missing dynamic_dims_strides=...");
    // 返回错误码 -1
    return -1;
  }

  // 当 dynamic_shapes=False 时，dynamic_dims_strides/sizes_py 为 None，这是一个优化，
  // 避免不必要地在 Python 中调用 .size()/.stride()
  std::vector<std::vector<std::optional<c10::SymInt>>>
      per_tensor_dynamic_dims_sizes = get_dynamic_dims(dynamic_dims_sizes_py);
  std::vector<std::vector<std::optional<c10::SymInt>>>
      per_tensor_dynamic_dims_strides =
          get_dynamic_dims(dynamic_dims_strides_py);

  // 获取 self 对象的 checks 成员
  auto& checks = *self->checks;
  // 获取参数 args 的长度
  auto len = PyTuple_GET_SIZE(args);
  // 预留足够的空间以容纳 len 个元素
  checks.reserve(len);
  // 创建本地状态对象 state
  LocalState state;

  // 遍历参数 args 中的每个元素
  for (auto i : c10::irange(len)) {
    // 获取 args 中的第 i 个元素
    PyObject* item = PyTuple_GET_ITEM(args, i);
    // 检查 item 是否为 THPVariable 类型的对象
    if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
      // 设置类型错误异常，并指定错误消息为 "expected Tensor()"
      PyErr_SetString(PyExc_TypeError, "expected Tensor()");
      // 返回错误码 -1
      return -1;
    }
    // 解包 item 成为 THPVariable 对象 tensor
    auto tensor = THPVariable_Unpack(item);
    // 获取当前 tensor 的动态维度尺寸
    std::vector<std::optional<c10::SymInt>> tensor_dims_size =
        per_tensor_dynamic_dims_sizes.empty()
        ? wrapIntegersInOptional(tensor.sym_sizes())
        : per_tensor_dynamic_dims_sizes[i];
    // 获取当前 tensor 的动态维度步长
    std::vector<std::optional<c10::SymInt>> tensor_dims_stride =
        per_tensor_dynamic_dims_strides.empty()
        ? wrapIntegersInOptional(tensor.sym_strides())
        : per_tensor_dynamic_dims_strides[i];

    // 向 checks 中添加一个新的检查项，包括 state 对象、item 的类型、tensor 对象、
    // tensor 的维度尺寸、tensor 的维度步长
    checks.emplace_back(
        state,
        Py_TYPE(item),
        std::move(tensor),
        std::move(tensor_dims_size),
        std::move(tensor_dims_stride));
  }
  // 返回正常状态码 0
  return 0;
  // 检查 args 是否为确切的 PyTuple 对象，如果不是则抛出类型错误异常
  if (!PyTuple_CheckExact(args)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }

  // 获取 self 对象中的 checks 引用
  auto& checks = *self->checks;
  // 获取 args 中元素的数量
  auto len = PyTuple_GET_SIZE(args);

  // kwargs 在此处被忽略

  // 检查 args 中元素的数量是否与 checks 中的要求一致，如果不一致则抛出类型错误异常
  if (static_cast<decltype(len)>(checks.size()) != len) {
    PyErr_SetString(PyExc_TypeError, "wrong length");
    return nullptr;
  }

  // 创建本地状态变量 state

  // Note - all the tensors that make it to guards must be unique. Dynamo
  // builder handles guarding for positive aliases (X is Y). However, we do not
  // create guards for negative alias (X is not Y) as that is an N^2
  // relationship. Instead, we rely on the uniqueness upstream to verify, at
  // check_fn time (this function).
  // 创建一个哈希映射 unique_tensors，用于确保 tensors 是唯一的
  ska::flat_hash_map<PyObject*, std::nullptr_t> unique_tensors;

  // 遍历 args 中的每个元素
  for (auto i : c10::irange(len)) {
    // 获取 args 中的第 i 个元素
    PyObject* item = PyTuple_GET_ITEM(args, i);

    // 检查 item 的类型是否符合 checks[i].pytype 的要求，如果不符合返回 False
    if (Py_TYPE(item) != checks[i].pytype) {
      Py_RETURN_FALSE;
    }

    // 将 item 插入到 unique_tensors 中，如果插入失败说明违反了唯一性要求，返回 False
    auto insertion = unique_tensors.insert({item, nullptr});
    if (!insertion.second) {
      // Violates uniqueness
      Py_RETURN_FALSE;
    }

    // 调用 checks[i].check 方法，检查 item 是否符合额外的条件
    if (!checks[i].check(state, THPVariable_Unpack(item))) {
      Py_RETURN_FALSE;
    }
  }

  // 如果所有检查都通过，则返回 True
  Py_RETURN_TRUE;
}
    // 检查 item 对象的类型是否与预期的 tensor 类型不同
    if (Py_TYPE(item) != checks[i].pytype) {
      // 创建一个字符串流对象，用于构建失败原因的详细描述
      std::stringstream fail_reason;
      // 获取 item 对象的类型描述，并添加到失败原因中
      PyObject* type_str = PyObject_Str(PyObject_Type(item));
      fail_reason << "expected type of '" << tensor_check_names[i]
                  << "' to be a tensor type, ";
      // 如果获取类型描述失败，则说明找到了不同的类型
      if (!type_str) {
        fail_reason << "but found a different type";
      } else {
        // 否则将实际找到的类型描述添加到失败原因中
        fail_reason << "' but found " << PyUnicode_AsUTF8(type_str);
      }
      // 使用失败原因构建 Python 字符串对象并返回
      return Py_BuildValue("s", fail_reason.str().c_str());
    }

    // 尝试将 item 插入到 unique_tensors 中，检查是否插入成功
    auto insertion = unique_tensors.insert({item, nullptr});
    if (!insertion.second) {
      // 如果插入失败，说明已经存在相同的 tensor，返回失败原因
      std::stringstream fail_reason;
      fail_reason << "Duplicate tensor found where not expected! ";
      fail_reason << tensor_check_names[i]
                  << "should not alias to anything, but is aliased";
      return Py_BuildValue("s", fail_reason.str().c_str());
    }

    // 执行详细的检查，检查 tensor 是否符合预期的条件
    std::string fail_reason = checks[i].check_verbose(
        state, THPVariable_Unpack(item), tensor_check_names[i]);
    // 如果有失败原因返回，说明检查未通过，返回失败原因
    if (fail_reason.length() > 0) {
      return Py_BuildValue("s", fail_reason.c_str());
    }
  }

  // 如果所有检查通过，返回一个 Python 的 True 值
  Py_RETURN_TRUE;
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
// 定义一个静态的 PyMethodDef 数组，用于描述 TensorGuards 对象的方法集合
static PyMethodDef TensorGuards_methods[] = {
    // 方法 "check" 的描述，指向 TensorGuards_check 函数，支持变长参数和关键字参数
    {"check",
     (PyCFunction)(void*)TensorGuards_check,
     METH_VARARGS | METH_KEYWORDS,
     ""},
    // 方法 "check_verbose" 的描述，指向 TensorGuards_check_verbose 函数，
    // 支持变长参数和关键字参数，并提供了失败检查详细原因的说明
    {"check_verbose",
     (PyCFunction)(void*)TensorGuards_check_verbose,
     METH_VARARGS | METH_KEYWORDS,
     "verbose fail reasons for failed checks"},
    // 数组结束的标志，用于指示 PyMethodDef 结束
    {nullptr} /* Sentinel */
};

// TensorGuardsType 的定义，表示一个 Python 类型对象，未初始化
static PyTypeObject TensorGuardsType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// TODO (janimesh) - Remove the PyObject_HEAD part when C++ guard manager is
// merged.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// GlobalStateGuard 结构体，带有 PyObject_HEAD，用于管理全局状态
struct GlobalStateGuard {
  PyObject_HEAD;

  // 初始化方法，初始化各个成员变量，关联到 ATen 全局上下文
  inline void init() {
    auto& ctx = at::globalContext();
    _grad_mode = at::GradMode::is_enabled();
    _torch_function = torch::torch_function_enabled();
    _deterministic_algorithms = ctx.deterministicAlgorithms();
    _deterministic_algorithms_warn_only = ctx.deterministicAlgorithmsWarnOnly();
    _allow_tf32 = ctx.allowTF32CuBLAS();
    _allow_fp16_reduce = ctx.allowFP16ReductionCuBLAS();
    _allow_bf16_reduce = ctx.allowBF16ReductionCuBLAS();
    _num_threads = at::get_num_threads();
    _default_dtype = at::get_default_dtype();
  }

  // 检查当前状态是否一致的方法，返回布尔值
  inline bool check() const {
    auto& ctx = at::globalContext();
    return (_grad_mode == at::GradMode::is_enabled() &&
            _torch_function == torch::torch_function_enabled() &&
            _deterministic_algorithms == ctx.deterministicAlgorithms() &&
            _deterministic_algorithms_warn_only ==
                ctx.deterministicAlgorithmsWarnOnly() &&
            _allow_tf32 == ctx.allowTF32CuBLAS() &&
            _allow_fp16_reduce == ctx.allowFP16ReductionCuBLAS() &&
            _allow_bf16_reduce == ctx.allowBF16ReductionCuBLAS() &&
            _num_threads == at::get_num_threads()) &&
        _default_dtype == at::get_default_dtype();
  }

  // 返回导致状态不一致的原因的方法，返回字符串
  inline std::string reason() const {
    std::ostringstream os;
    auto& ctx = at::globalContext();
    if (_grad_mode != at::GradMode::is_enabled())
      os << "grad_mode ";
    if (_torch_function != torch::torch_function_enabled())
      os << "torch_function ";
    if (_deterministic_algorithms != ctx.deterministicAlgorithms())
      os << "deterministic_algorithms ";
    if (_deterministic_algorithms_warn_only !=
        ctx.deterministicAlgorithmsWarnOnly())
      os << "deterministic_algorithms_warn_only ";
    if (_allow_tf32 != ctx.allowTF32CuBLAS())
      os << "allow_tf32 ";
    if (_allow_fp16_reduce != ctx.allowFP16ReductionCuBLAS())
      os << "allow_fp16_reduce ";
    if (_allow_bf16_reduce != ctx.allowBF16ReductionCuBLAS())
      os << "allow_bf16_reduce ";
    if (_num_threads != at::get_num_threads())
      os << "num_threads ";
    if (_default_dtype != at::get_default_dtype())
      os << "default_dtype ";
    return os.str();
  }

  bool _grad_mode;
  bool _torch_function;
  bool _deterministic_algorithms;
  bool _deterministic_algorithms_warn_only;
  bool _allow_tf32;
  bool _allow_fp16_reduce;
  bool _allow_bf16_reduce;
  int _num_threads;
  caffe2::TypeMeta _default_dtype;
  // TODO(jansel): we should guard on more state as inductor starts using it



// 返回字符串流 `os` 的字符串表示
return os.str();
// 下面是一系列的布尔值和整数变量声明，用于控制不同的运行时选项和算法：
bool _grad_mode;  // 控制梯度模式
bool _torch_function;  // 控制 Torch 函数
bool _deterministic_algorithms;  // 控制确定性算法
bool _deterministic_algorithms_warn_only;  // 控制确定性算法警告
bool _allow_tf32;  // 控制是否允许 TF32 类型
bool _allow_fp16_reduce;  // 控制是否允许 FP16 减少
bool _allow_bf16_reduce;  // 控制是否允许 BF16 减少
int _num_threads;  // 线程数量
caffe2::TypeMeta _default_dtype;  // 默认数据类型
// TODO(jansel): 当 Inductor 开始使用更多状态时，我们应该添加更多的保护措施
// 这是一个待办事项注释，提醒在未来代码中添加更多保护措施以支持 Inductor 使用的新功能
};

// 初始化 GlobalStateGuard 对象
int GlobalStateGuard_init(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  // 调用 self 对象的初始化方法
  self->init();
  // 返回状态码 0 表示成功
  return 0;
}

// 检查 GlobalStateGuard 对象的状态
PyObject* GlobalStateGuard_check(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  // 如果 self 对象的检查方法返回 true，则返回 Python 中的 True
  if (self->check()) {
    Py_RETURN_TRUE;
  } else {
    // 否则返回 Python 中的 False
    Py_RETURN_FALSE;
  }
}

// 获取 GlobalStateGuard 对象的失败原因
PyObject* GlobalStateGuard_reason(
    GlobalStateGuard* self,
    PyObject* args,
    PyObject* kwargs) {
  // 返回一个 Python 字符串对象，内容为 self 对象的失败原因
  return PyUnicode_FromString(self->reason().c_str());
}

// 定义 GlobalStateGuard 的方法集合
static PyMethodDef GlobalStateGuard_methods[] = {
    {"check",
     (PyCFunction)(void*)GlobalStateGuard_check,
     METH_NOARGS,
     "Return true if global state was the same as at creation time"},
    {"reason",
     (PyCFunction)(void*)GlobalStateGuard_reason,
     METH_NOARGS,
     "Return string reason for guard check failing"},
    {nullptr}};  // 方法集合结束标志

// 定义 GlobalStateGuardType 类型对象
static PyTypeObject GlobalStateGuardType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// 检查对象类型的 ID 是否匹配预期 ID
static PyObject* check_type_id(PyObject* dummy, PyObject* args) {
  // 定义 obj 和 expected 变量
  PyObject* obj = nullptr;
  unsigned long long expected = 0;
  // 解析 Python 参数，获取 obj 和 expected
  if (!PyArg_ParseTuple(args, "OK", &obj, &expected)) {
    return nullptr;
  }
  // 检查 obj 的类型 ID 是否与 expected 相同，返回对应的 Python 布尔值
  if (Py_TYPE(obj) == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// 检查对象 ID 是否匹配预期 ID
static PyObject* check_obj_id(PyObject* dummy, PyObject* args) {
  // 定义 obj 和 expected 变量
  PyObject* obj = nullptr;
  unsigned long long expected = 0;
  // 解析 Python 参数，获取 obj 和 expected
  if (!PyArg_ParseTuple(args, "OK", &obj, &expected)) {
    return nullptr;
  }
  // 检查 obj 的 ID 是否与 expected 相同，返回对应的 Python 布尔值
  if (obj == (void*)expected) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

// 如果 Python 版本大于等于 3.12，处理字典版本相关功能
#if IS_PYTHON_3_12_PLUS

// 存储字典对象与其版本号的映射关系
static std::unordered_map<PyObject*, uint64_t> dict_version_map;
// 字典版本监视器 ID
static int dict_version_watcher_id;
// 全局字典版本号
static uint64_t global_dict_version_id = 0;

// 处理字典版本变化的回调函数
static int dict_version_watch_callback(
    PyDict_WatchEvent event,
    PyObject* dict,
    PyObject* key,
    PyObject* new_value) noexcept {
  // 根据事件类型更新字典版本映射
  if (event == PyDict_EVENT_DEALLOCATED) {
    dict_version_map.erase(dict);
  } else if (event != PyDict_EVENT_CLONED) {
    dict_version_map[dict] = global_dict_version_id++;
  }
  return 0;
}

#endif  // 结束处理字典版本的条件编译块

// 获取字典对象的版本号（如果支持）
static uint64_t get_dict_version_unchecked(PyObject* dict) {
#if IS_PYTHON_3_12_PLUS

  // 添加字典版本监视器
  if (PyDict_Watch(dict_version_watcher_id, dict)) {
    throw std::runtime_error("failed to add version watcher to dict!");
  }
  // 如果映射中没有当前字典对象，则添加并分配一个全局唯一的版本号
  if (!dict_version_map.count(dict)) {
    dict_version_map[dict] = global_dict_version_id++;
  }
  // 返回字典对象的版本号
  return dict_version_map[dict];

#else

  // 返回字典对象的内部版本标签
  return ((PyDictObject*)dict)->ma_version_tag;

#endif
}

// 获取字典对象的版本号并返回给 Python 调用者
static PyObject* dict_version(PyObject* dummy, PyObject* args) {
  // 解析 Python 参数，获取对象 obj
  PyObject* obj = nullptr;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return nullptr;
  }
  // 检查 obj 是否为字典类型
  if (!PyDict_Check(obj)) {
    // 如果条件不满足，返回空指针
    return nullptr;
  }
  // 调用函数获取对象的版本号，并将其封装成无符号64位整数返回
  return THPUtils_packUInt64(get_dict_version_unchecked(obj));
static PyObject* assert_size_stride(PyObject* dummy, PyObject* args) {
  /*
   Assert that a given tensor has a given size/stride, but ignore strides
   of size==1 dimensions.  Implemented in C++ as this is on the hot path.
  */
  // 声明变量并初始化为nullptr
  PyObject* item = nullptr;
  PyObject* size = nullptr;
  PyObject* stride = nullptr;
  // 解析Python传入的参数为item, size, stride三个对象
  if (!PyArg_ParseTuple(args, "OOO", &item, &size, &stride)) {
    return nullptr;
  }
  // 检查item是否为THPVariable类型或者其子类
  if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
    PyErr_SetString(PyExc_TypeError, "expected Tensor()");
    return nullptr;
  }
  // 检查size和stride是否为元组类型
  if (!PyTuple_CheckExact(size) || !PyTuple_CheckExact(stride)) {
    PyErr_SetString(PyExc_TypeError, "expected tuple()");
    return nullptr;
  }
  // 将item解包为at::Tensor对象
  at::Tensor tensor = THPVariable_Unpack(item);
  // 获取tensor的维度数量
  int64_t ndim = tensor.ndimension();
  // 检查size和stride的长度是否与tensor的维度数量相同
  if (PyTuple_GET_SIZE(size) != ndim || PyTuple_GET_SIZE(stride) != ndim) {
    PyErr_SetString(PyExc_AssertionError, "wrong number of dimensions");
    return nullptr;
  }
  // 创建字符串流用于构建错误消息
  std::stringstream msg;
  // 记录错误数量
  int num_errors = 0;
  // 遍历tensor的每个维度
  for (auto i : c10::irange(ndim)) {
    // 获取期望的尺寸和步长
    int64_t want_size = THPUtils_unpackLong(PyTuple_GET_ITEM(size, i));
    int64_t want_stride = THPUtils_unpackLong(PyTuple_GET_ITEM(stride, i));
    // 获取实际的尺寸和步长
    int64_t actual_size = tensor.size(i);
    int64_t actual_stride = tensor.stride(i);
    // 检查尺寸和步长是否符合期望，当尺寸为1时忽略步长的差异
    if (want_size != actual_size ||
        (want_stride != actual_stride && actual_size > 1)) {
      if (num_errors > 0)
        msg << "; ";
      msg << "expected size " << actual_size << "==" << want_size << ", stride "
          << actual_stride << "==" << want_stride << " at dim=" << i;
      num_errors++;
    }
  }

  // 如果存在错误，设置错误消息并返回nullptr
  if (num_errors) {
    PyErr_SetString(PyExc_AssertionError, msg.str().c_str());
    return nullptr;
  }

  // 如果没有错误，返回True对象
  Py_RETURN_TRUE;
}

template <typename T>
inline static void unwrap_size_tuple(PyObject* obj, T& output) {
  // 检查obj是否为元组类型
  TORCH_CHECK(PyTuple_CheckExact(obj));
  // 获取元组的长度
  size_t len = PyTuple_GET_SIZE(obj);
  // 预留output的空间
  output.reserve(len);
  // 遍历元组的每个元素，将其转换为PyLong，并加入到output中
  for (size_t i = 0; i < len; ++i) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(obj, i));
    // 检查转换结果是否非负
    TORCH_CHECK(result >= 0);
    output.emplace_back(result);
  }
}

template <typename T>
inline static void _parse_empty_strided_args(
    PyObject* args,
    T& sizes,
    T& strides,
    at::ScalarType& dtype) {
  // 检查args是否为元组类型，并且长度为3
  TORCH_CHECK(PyTuple_CheckExact(args));
  TORCH_CHECK(PyTuple_GET_SIZE(args) == 3);
  // 解析args中的第一个和第二个元素为sizes和strides
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 0), sizes);
  unwrap_size_tuple(PyTuple_GET_ITEM(args, 1), strides);
  // 解析args中的第三个元素为dtype
  PyObject* py_dtype = PyTuple_GET_ITEM(args, 2);
  // 检查第三个元素是否为THPDtype类型
  TORCH_CHECK(THPDtype_Check(py_dtype));
  // 将THPDtype类型转换为at::ScalarType类型并赋值给dtype
  dtype = reinterpret_cast<THPDtype*>(py_dtype)->scalar_type;
}
static PyObject* _empty_strided_cpu(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is a lower-overhead
  // version that saves ~2us on every allocation.
  // 处理错误和异常
  HANDLE_TH_ERRORS;
  // 定义用于存储大小和步长的 SmallVector
  at::SmallVector<int64_t, 8> sizes;
  at::SmallVector<int64_t, 8> strides;
  // 定义数据类型，默认为 Undefined
  at::ScalarType dtype{at::ScalarType::Undefined};
  // 解析传入的参数并填充 sizes、strides 和 dtype
  _parse_empty_strided_args(args, sizes, strides, dtype);
  // 调用 empty_strided_cpu 函数创建新的 Tensor，并用 THPVariable_Wrap 封装返回的结果
  return THPVariable_Wrap(at::detail::empty_strided_cpu(sizes, strides, dtype));
  // 处理错误和异常的结束
  END_HANDLE_TH_ERRORS;
}

static PyObject* _empty_strided_cuda(PyObject* dummy, PyObject* args) {
  // at::empty_strided is surprising slow.  This is lower-overhead.
  // 处理错误和异常
  HANDLE_TH_ERRORS;
  // 如果使用 CUDA 编译，则执行以下代码块
#ifdef USE_CUDA
  // 定义用于存储大小和步长的 SmallVector
  at::SmallVector<int64_t, 8> sizes;
  at::SmallVector<int64_t, 8> strides;
  // 定义数据类型，默认为 Undefined
  at::ScalarType dtype{at::ScalarType::Undefined};
  // 解析传入的参数并填充 sizes、strides 和 dtype
  _parse_empty_strided_args(args, sizes, strides, dtype);
  // 调用 empty_strided_cuda 函数创建新的 Tensor，并用 THPVariable_Wrap 封装返回的结果
  return THPVariable_Wrap(at::detail::empty_strided_cuda(
      sizes, strides, dtype, c10::DeviceType::CUDA));
// 如果未使用 CUDA 编译，则执行以下代码块
#else
  // 抛出错误信息，表示 PyTorch 编译时未启用 CUDA
  TORCH_CHECK(false, "PyTorch compiled without USE_CUDA");
#endif
  // 处理错误和异常的结束
  END_HANDLE_TH_ERRORS;
}

static PyObject* _reinterpret_tensor(PyObject* dummy, PyObject* args) {
  // 处理错误和异常
  HANDLE_TH_ERRORS;
  // 静态定义 PythonArgParser 对象，用于解析传入的参数
  static PythonArgParser parser(
      {"_reinterpret_tensor(Tensor base, IntArrayRef sizes, IntArrayRef strides, int64_t offset_increment=0)"},
      /*traceable=*/true);

  // 解析参数并存储在 ParsedArgs 中
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, /*kwargs=*/nullptr, parsed_args);

  // 获取解析后的 Tensor 对象，大小列表，步长列表和偏移增量
  Tensor self = r.tensor(0);
  auto sizes = r.intlist(1);
  auto strides = r.intlist(2);
  auto offset_increment = r.toInt64(3);

  // 调用 _reinterpret_tensor 函数处理 Tensor，并返回处理后的结果
  auto res = torch::inductor::_reinterpret_tensor(
      self, sizes, strides, offset_increment);
  // 封装返回的结果为 PyObject*
  return torch::autograd::utils::wrap(res);

  // 处理错误和异常的结束
  END_HANDLE_TH_ERRORS;
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static PyMethodDef _methods[] = {
    {"check_type_id", check_type_id, METH_VARARGS, nullptr},
    {"check_obj_id", check_obj_id, METH_VARARGS, nullptr},
    {"assert_size_stride", assert_size_stride, METH_VARARGS, nullptr},
    {"dict_version", dict_version, METH_VARARGS, nullptr},
    {"_empty_strided_cpu", _empty_strided_cpu, METH_VARARGS, nullptr},
    {"_empty_strided_cuda", _empty_strided_cuda, METH_VARARGS, nullptr},
    {"_reinterpret_tensor", _reinterpret_tensor, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.guards",
    "Module containing checks on tensors",
    -1,
    _methods};
  
// 获取异常信息
std::string get_exception_message() {
  PyObject *ptype = nullptr, *pvalue = nullptr, *ptraceback = nullptr;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);

  // 获取异常消息的 Python 对象
  PyObject* exc_message_pyobj = PyObject_Str(pvalue);
  // 将 Python 对象转换为 UTF-8 格式的 C 字符串
  const char* exc_message = PyUnicode_AsUTF8(exc_message_pyobj);

  // 释放 Python 对象的引用
  Py_DECREF(exc_message_pyobj);
  Py_XDECREF(ptype);
  Py_XDECREF(pvalue);
  Py_XDECREF(ptraceback);

  // 返回异常消息的字符串形式
  return std::string(exc_message);
}
/**
 * 存储相关的守卫调试信息，例如 LeafGuard 失败的失败字符串。
 * 此数据结构在 Python 中也是可访问的。
 */
class GuardDebugInfo {
 public:
  // 构造函数，用于初始化守卫调试信息对象，接收详细的调试信息列表和执行的守卫数量
  GuardDebugInfo(
      bool result,
      py::list verbose_code_parts,
      int num_guards_executed)
      : result(result),
        verbose_code_parts(std::move(verbose_code_parts)),
        num_guards_executed(num_guards_executed) {}

  // 当守卫成功时使用的构造函数
  GuardDebugInfo(bool result, int num_guards_executed)
      : result(result), num_guards_executed(num_guards_executed) {}

  // 构造函数，用于守卫失败时的调试信息，接收失败原因字符串和执行的守卫数量
  GuardDebugInfo(
      bool result,
      const std::string& failed_reason,
      int num_guards_executed)
      : GuardDebugInfo(result, num_guards_executed) {
    verbose_code_parts.append(failed_reason);
  }

  // 将对象转换为字符串表示形式，用于调试目的
  std::string to_string() {
    std::stringstream ss;
    ss << "GuardDebugInfo(\n"
       << "result=" << result << ",\n"
       << "verbose_code_parts=" << verbose_code_parts << ",\n"
       << "num_guards_executed=" << num_guards_executed << ")\n";
    return ss.str();
  }

  // 守卫是否通过或失败的标志
  bool result;

  // 失败守卫的详细代码部分列表，当存在多个详细代码部分时，Python 端的重新编译推理基础设施可以迭代此列表，并评估每个字符串以确定失败的确切代码部分
  py::list verbose_code_parts;

  // 到目前为止执行的守卫总数，这在调试时很有帮助，用于验证混洗是否有效
  int num_guards_executed;
};

// 前置声明
class GuardManager;
class RootGuardManager;
class DictGuardManager;

/**
 * GuardManager 层次结构中叶守卫的基类。
 */
class LeafGuard {
 public:
  // 大多数守卫不需要根守卫管理器
  LeafGuard(py::object verbose_code_parts)
      : _verbose_code_parts(std::move(verbose_code_parts)) {}

  // 类似 TENSOR_MATCH 的守卫需要 RootGuardManager 来访问所有叶守卫共享的本地状态
  LeafGuard(RootGuardManager* root_guard_manager, py::object verbose_code_parts)
      : _root_guard_manager(root_guard_manager),
        _verbose_code_parts(std::move(verbose_code_parts)) {}

  // 可能被从 Python 调用的检查函数，用于调试目的
  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  // 返回详细的守卫检查调试信息
  GuardDebugInfo check_verbose(py::handle value) {
    return check_verbose_nopybind(value.ptr());
  }

  // 虚函数，返回详细的守卫检查调试信息，无 Python 绑定版本
  virtual GuardDebugInfo check_verbose_nopybind(
      PyObject* value) { // borrowed ref
    bool result = check_nopybind(value);
    if (!result) {
      return GuardDebugInfo(result, _verbose_code_parts, 0);
    }
    return GuardDebugInfo(true, 0);
  }

  // 返回详细代码部分列表
  py::list verbose_code_parts() {
    return _verbose_code_parts;
  }



  // 返回 LeafGuard 对象的 _verbose_code_parts 成员变量
  // 该方法位于热路径上，避免了来自 pybind 的任何引用计数代码。此方法不会暴露给 Python，只能从 C++ 中调用。
  virtual bool check_nopybind(PyObject* value) = 0;
  
  // LeafGuard 的析构函数，使用默认实现
  virtual ~LeafGuard() = default;

 protected:
  // RootGuardManager 具有 RootGuard 的公共状态，例如 LocalState。
  RootGuardManager* _root_guard_manager{nullptr};

 private:
  // 在构造 LeafGuard 时设置此变量，用于识别重新编译的原因。
  py::list _verbose_code_parts;



  // _verbose_code_parts 是 LeafGuard 的私有成员变量，用于保存详细的代码部分列表。
};

/**
 * Represents a leaf guard that accepts the Python guard check function. We
 * aim to transition most guards to C++ to avoid Python function calls, but this
 * transition will take time. Some guards may remain in Python if porting is 
 * difficult or tedious.
 *
 * LAMBDA_GUARD allows a gradual transition to C++. Initially, all guards are 
 * of type PythonLambaGuard, and expensive guards are incrementally moved to C++.
 */
class LAMBDA_GUARD : public LeafGuard {
 public:
  /**
   * Constructs a LAMBDA_GUARD instance with a Python guard function and verbose
   * code parts.
   * 
   * @param guard_check_fn A callable Python function used as the guard check.
   * @param verbose_code_parts Parts of verbose code for debugging purposes.
   * @throws py::type_error If guard_check_fn is not callable.
   */
  LAMBDA_GUARD(py::object guard_check_fn, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {
    if (py::isinstance<py::function>(guard_check_fn)) {
      _guard_check_fn = py::cast<py::function>(std::move(guard_check_fn));
    } else {
      throw py::type_error("LAMBDA_GUARD expects (callable, str)");
    }
  }

  /**
   * Executes the lambda function with the given value and checks its result.
   * 
   * @param value The value to pass to the lambda function.
   * @return true if the lambda function returns true, false otherwise.
   */
  bool check_nopybind(PyObject* value) override { // borrowed ref
    PyObject* x = PyObject_CallOneArg(_guard_check_fn.ptr(), value); // new ref
    if (x == nullptr) {
      // An exception occurred in the lambda function.
      PyErr_Clear();
      return false;
    }
    bool result = PyObject_IsTrue(x);
    Py_DECREF(x);
    return result;
  }

  /**
   * Executes the lambda function with the given value and checks its result, 
   * providing debug information in case of failure.
   * 
   * @param value The value to pass to the lambda function.
   * @return GuardDebugInfo object indicating the result and debug information.
   */
  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    PyObject* x = PyObject_CallOneArg(_guard_check_fn.ptr(), value); // new ref
    if (x == nullptr) {
      // An exception occurred in the lambda function.
      std::string exc_message = get_exception_message();
      PyErr_Clear();
      return GuardDebugInfo(false, exc_message, 0);
    }
    bool result = PyObject_IsTrue(x);
    Py_DECREF(x);
    if (result) {
      return GuardDebugInfo(true, 0);
    }
    return GuardDebugInfo(false, verbose_code_parts(), 0);
  }

 private:
  // The user-provided lambda function for check_fn.
  py::function _guard_check_fn;
};

/**
 * Represents a leaf guard that checks if the type of a value matches an expected
 * type ID.
 */
class TYPE_MATCH : public LeafGuard {
 public:
  /**
   * Constructs a TYPE_MATCH instance with an expected type ID and verbose code
   * parts.
   * 
   * @param type_id ID of the expected type.
   * @param verbose_code_parts Parts of verbose code for debugging purposes.
   */
  TYPE_MATCH(py::object type_id, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _expected(py::cast<intptr_t>(std::move(type_id))) {}

  /**
   * Checks if the type of the given value matches the expected type ID.
   * 
   * @param value The value to check against the expected type.
   * @return true if the types match, false otherwise.
   */
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return Py_TYPE(value) == (void*)_expected;
  }

 private:
  // ID of the type of the original object.
  intptr_t _expected;
};

/**
 * Represents a leaf guard that checks if the ID of a value matches an expected
 * object ID.
 */
class ID_MATCH : public LeafGuard {
 public:
  /**
   * Constructs an ID_MATCH instance with an expected object ID and verbose code
   * parts.
   * 
   * @param obj_id ID of the expected object.
   * @param verbose_code_parts Parts of verbose code for debugging purposes.
   */
  ID_MATCH(py::object obj_id, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _expected(py::cast<intptr_t>(std::move(obj_id))) {}

  /**
   * Checks if the ID of the given value matches the expected object ID.
   * 
   * @param value The value to check against the expected object ID.
   * @return true if the IDs match, false otherwise.
   */
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return value == (void*)_expected;
  }

 private:
  // ID of the original object.
  intptr_t _expected;
};
// EQUALS_MATCH 类继承自 LeafGuard 类，用于比较对象是否相等
class EQUALS_MATCH : public LeafGuard {
 public:
  // 构造函数，初始化 _value 和 _value_type 成员变量
  EQUALS_MATCH(py::object value, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _value(value),
        _value_type(Py_TYPE(value.ptr())) {}

  // 检查对象是否匹配的方法，覆盖自 LeafGuard 类
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 快速路径 - 指针相等性检查。由于 EQUALS_MATCH 中的对象是不可变的，因此指针相等性检查是安全的。
    if (value != _value.ptr() && value != _first_passing_value.ptr()) {
      // 检查对象类型是否匹配
      if (Py_TYPE(value) != _value_type) {
        return false;
      }
      // 使用 PyObject_RichCompareBool 函数进行对象值比较
      int result = PyObject_RichCompareBool(value, _value.ptr(), Py_EQ);
      // 检查是否发生异常
      if (result == -1) {
        PyErr_Clear();
        return false;
      }

      // 在此缓存匹配的值
      if (!_first_passing_value && result) {
        _first_passing_value = py::cast<py::object>(value);
      }
      return result;
    }
    return true;
  }

 private:
  // 用于比较的值，类型为 py::object，防止对象被垃圾回收
  py::object _value;

  // 缓存第一个与 _value.ptr() 不相等的值。在 nn 模块中的保护中很有用，其中 getattr 名称是字符串，与 __dict__ 中的键相同，但指针不同。
  py::object _first_passing_value;

  // 值的类型
  PyTypeObject* _value_type;
};

// TUPLE_ITERATOR_LEN 类继承自 LeafGuard 类，用于检查 tuple 迭代器的长度
class TUPLE_ITERATOR_LEN : public LeafGuard {
 public:
  // 构造函数，初始化 _length 和 _type_id 成员变量
  TUPLE_ITERATOR_LEN(
      py::object length,
      py::object type_id,
      py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _length(py::cast<Py_ssize_t>(std::move(length))),
        _type_id(py::cast<intptr_t>(std::move(type_id))) {}

  // 检查对象是否匹配的方法，覆盖自 LeafGuard 类
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 先进行类型匹配
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    if (Py_TYPE(value) != (void*)_type_id) {
      return false;
    }
    _PyTupleIterObject* it = (_PyTupleIterObject*)value;
    Py_ssize_t length = 0;
    if (it->it_seq)
      length = PyTuple_GET_SIZE(it->it_seq) - it->it_index;
    return length == _length;
  }

 private:
  // 受保护列表的长度
  Py_ssize_t _length;
  intptr_t _type_id;
};

// LENGTH_CHECK 类继承自 LeafGuard 类，用于检查序列的长度
class LENGTH_CHECK : public LeafGuard {
 public:
  // 构造函数，初始化 _length 成员变量
  LENGTH_CHECK(py::object value, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _length(py::cast<Py_ssize_t>(std::move(value))) {}

  // 检查对象是否匹配的方法，覆盖自 LeafGuard 类
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 如果对象不是序列，PySequence_Length 将返回 -1。因此，我们不需要测试 PySequence_Check。
    return PySequence_Length(value) == _length;
  }

 private:
  // 受保护列表的长度
  Py_ssize_t _length;
};
// 继承 LeafGuard 类并定义 DICT_LENGTH 类，用于检查字典的长度是否符合预期
class DICT_LENGTH : public LeafGuard {
 public:
  // 构造函数，接收 Python 对象 value 和 verbose_code_parts，并初始化 LeafGuard 基类
  DICT_LENGTH(py::object value, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _length(py::cast<Py_ssize_t>(std::move(value))) {}

  // 检查函数，用于检查传入的 PyObject 是否为字典且长度与 _length 相同
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyDict_Check(value) && PyDict_Size(value) == _length;
  }

 private:
  // 被保护字典的长度
  Py_ssize_t _length;
};

// 继承 LeafGuard 类并定义 NOT_NONE 类，用于检查对象是否不为 None
class NOT_NONE : public LeafGuard {
 public:
  // 构造函数，接收 verbose_code_parts 并初始化 LeafGuard 基类
  NOT_NONE(py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {}

  // 检查函数，用于检查传入的 PyObject 是否不为 None
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return value != Py_None;
  }
};

// 继承 LeafGuard 类并定义 DEFAULT_DEVICE 类，用于检查当前设备是否符合预期
class DEFAULT_DEVICE : public LeafGuard {
 public:
  // 构造函数，接收 verbose_code_parts，并在构造过程中获取当前设备信息
  DEFAULT_DEVICE(py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {
    // 导入 torch.utils._device 模块
    py::handle device_module = py::module::import("torch.utils._device");
    // 保存模块的 __dict__ 属性
    _utils_device_dict = device_module.attr("__dict__");
    // 获取 CURRENT_DEVICE 的值作为当前设备
    _device = _utils_device_dict["CURRENT_DEVICE"];
  }

  // 检查函数，用于检查传入的 PyObject 是否与保存的当前设备相同
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 创建一个静态的 interned 字符串来表示 CURRENT_DEVICE
    static PyObject* current_device_str =
        PyUnicode_InternFromString("CURRENT_DEVICE");
    // 从 _utils_device_dict 中获取当前设备信息
    PyObject* device = PyDict_GetItem(
        _utils_device_dict.ptr(), current_device_str); // borrowed ref
    // 检查当前设备是否与保存的设备相同
    if (device != _device.ptr()) {
      // 比较两个设备对象是否相等
      int result = PyObject_RichCompareBool(device, _device.ptr(), Py_EQ);
      if (result == -1) {
        PyErr_Clear();
        return false;
      }
      return result;
    }
    return true;
  }

 private:
  // 保存 utils._device 模块的 __dict__ 属性
  py::object _utils_device_dict;
  // 保存 CURRENT_DEVICE 的设备对象
  py::object _device;
};

// 继承 LeafGuard 类并定义 GLOBAL_STATE 类，用于检查全局状态的变化
class GLOBAL_STATE : public LeafGuard {
 public:
  // 构造函数，接收 verbose_code_parts，并在构造过程中初始化全局状态保护对象
  GLOBAL_STATE(py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {
    // 创建并初始化 GlobalStateGuard 对象
    _guard = std::make_unique<GlobalStateGuard>();
    _guard->init();
  }

  // 检查函数，用于检查全局状态是否符合预期
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 忽略传入的 value 参数，仅用于满足接口要求
    return _guard->check();
  }

  // 详细检查函数，用于在详细模式下检查全局状态是否符合预期
  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    // 如果全局状态不符合预期，返回详细的调试信息
    if (!_guard->check()) {
      return GuardDebugInfo(
          false, "GLOBAL_STATE changed: " + _guard->reason(), 0);
    }
    // 如果全局状态符合预期，返回成功的调试信息
    return GuardDebugInfo(true, 1);
  }

 private:
  // 保存全局状态保护对象的指针
  std::unique_ptr<GlobalStateGuard> _guard;
};

// 继承 LeafGuard 类并定义 DATA_PTR_MATCH 类，用于检查数据指针是否匹配的情况
class DATA_PTR_MATCH : public LeafGuard {
 public:
  // 构造函数，接收 tensor 和 verbose_code_parts，并在构造过程中进行初始化
  DATA_PTR_MATCH(py::object tensor, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {
    // 获取 tensor 对象的指针
    PyObject* value = tensor.ptr();
    // 如果 tensor 对象不是 THPVariable_CheckExact 或 THPVariable_Check 类型，则抛出异常
    if (!THPVariable_CheckExact(value) && !THPVariable_Check(value)) {
      throw std::runtime_error("DATA_PTR_MATCH guard requires a tensor");
  }
  // 获取THPVariable类型的对象的数据指针，并赋值给_data_ptr成员变量
  _data_ptr = THPVariable_Unpack(value).data_ptr();
}

bool check_nopybind(PyObject* value) override { // borrowed ref
  // 检查传入的PyObject对象是否是THPVariable类型的对象
  if (!THPVariable_CheckExact(value) && !THPVariable_Check(value)) {
    return false;
  }
  // 获取THPVariable类型的对象的数据指针
  void* data_ptr = THPVariable_Unpack(value).data_ptr();
  // 检查传入对象的数据指针是否与存储的_data_ptr相同
  return data_ptr == _data_ptr;
}

private:
// 原始张量数据指针
void* _data_ptr;
// 结束了前一个类定义的分号
};

// 检查对象中是否不存在某个属性。我们不需要相反的 HASATTR 保护，因为我们可以依赖 GetAttrGuardAccessor 作为 HASATTR 保护。
class NO_HASATTR : public LeafGuard {
 public:
  // 构造函数，接受属性名和详细代码部分作为参数
  NO_HASATTR(py::object attr_name, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _attr_name(std::move(attr_name)) {}

  // 检查函数，检查给定对象是否不含指定属性
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyObject_HasAttr(value, _attr_name.ptr()) == 0;
  }

 private:
  py::object _attr_name;  // 存储属性名的对象
};

// 检查字典中是否包含或不包含某个键。这在 PythonSysModulesVariable 跟踪器中发生。
// TODO(janimesh) - 检查是否可以使用 DictGuardManager。缺点可能是对 sys 模块的键数较多，所以 DICT_CONTAINS 可能仍然更快。
class DICT_CONTAINS : public LeafGuard {
 public:
  // 构造函数，接受是否包含、键和详细代码部分作为参数
  DICT_CONTAINS(bool contains, py::object key, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _contains(contains ? 1 : 0),
        _key(std::move(key)) {}

  // 检查函数，检查给定对象是否包含指定键
  bool check_nopybind(PyObject* value) override { // borrowed ref
    int result = PyDict_Contains(value, _key.ptr());
    if (result == -1) {
      PyErr_Clear();  // 清除异常状态
      return false;
    }
    return result == _contains;  // 检查是否符合预期的包含状态
  }

 private:
  int _contains;         // 保存包含状态的整数
  py::object _key;       // 存储键的对象
};

/**
 * 关系保护器比较多个值。我们通过在保护对象中捕获一些状态来实现关系保护器。例如，对张量别名保护器 - 张量 X 不是张量 Y - 我们构造一个叶子保护器，并将其安装为两个保护管理器的叶子（一个用于 X，另一个用于 Y）。因此，此保护器会运行两次。在第一次调用中，它保存第一个值（状态）并返回 True。在第二次调用中，它将保存的值与新值进行比较，并在它们不是别名时返回 True。
 *
 * 在其他保护器失败且关系保护器中有一些状态时，我们必须小心重置状态。这通过虚拟方法 reset_state() 完成。这是由 RootGuardManager 在退出之前调用的。
 */
class RelationalGuard : public LeafGuard {
 public:
  // 构造函数，接受详细代码部分作为参数
  RelationalGuard(py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {}

  // 重置关系保护器状态的虚拟方法，在保护管理器中调用
  virtual void reset_state() = 0;
};

/**
 * 检查张量 x 是否是张量 y。
 */
class TENSOR_ALIASING : public RelationalGuard {
 public:
  // 构造函数，接受详细代码部分作为参数
  TENSOR_ALIASING(py::object verbose_code_parts)
      : RelationalGuard(std::move(verbose_code_parts)) {}

  // 检查函数，检查给定对象是否与保存的第一个张量相同
  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (_is_first_call) {
      _first_tensor = value;  // 保存第一个张量
      _is_first_call = false; // 标记第一次调用已完成
      return true;
    }
    return _first_tensor == value;  // 检查是否与第一个张量相同
  }

  // 重置关系保护器状态的实现，标记第一次调用为真
  void reset_state() final {

    _is_first_call = true;  // 重置第一次调用标记
  }
};
    _is_first_call = true;


    // 将 _is_first_call 设置为 true，可能是在类的构造函数或者某个初始化方法中进行的初始设置
    _is_first_call = true;



  }

 private:
  bool _is_first_call{true};
  PyObject* _first_tensor{nullptr};


  // 私有成员变量，用于记录是否第一次调用某个函数或初始化类时的状态
  bool _is_first_call{true};

  // 私有成员变量，指向 PyObject 类型的指针，初始化为 nullptr
  PyObject* _first_tensor{nullptr};
};

/**
 * Checks that none of the tensors alias.
 */
class NO_TENSOR_ALIASING : public RelationalGuard {
 public:
  NO_TENSOR_ALIASING(
      const py::list& tensor_names,
      py::object verbose_code_parts)
      : RelationalGuard(std::move(verbose_code_parts)),
        _tensor_names(tensor_names) {
    _unique_tensors.reserve(tensor_names.size());
  }

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 增加对象的引用计数，以避免被垃圾回收
    Py_INCREF(value);
    // 将值插入到_unique_tensors中，检查是否已存在
    auto insertion = _unique_tensors.insert({value, nullptr});
    if (!insertion.second) {
      // 如果插入失败，表示已存在相同的tensor
      // 不需要清理_unique_tensors，reset_state方法会处理
      return false;
    }
    return true;
  }

  GuardDebugInfo check_verbose_nopybind(PyObject* value) override {
    // 检查是否有重复的tensor
    bool result = check_nopybind(value);

    if (!result) {
      // 返回详细的调试信息，说明找到了重复的tensor
      return GuardDebugInfo(
          false, "Duplicate tensor found where not expected!", 0);
    }
    // 返回正常的调试信息，表示未找到重复的tensor
    return GuardDebugInfo(true, 1);
  }

  void reset_state() final {
    // 重置状态时，减少_unique_tensors中所有tensor的引用计数
    for (auto item : _unique_tensors) {
      Py_DECREF(item.first);
    }
    // 清空_unique_tensors
    _unique_tensors.clear();
  }

 private:
  py::list _tensor_names; // 存储tensor名称的列表
  ska::flat_hash_map<PyObject*, std::nullptr_t> _unique_tensors; // 存储唯一tensor的哈希映射
};

class DYNAMIC_INDICES : public LeafGuard {
  // C++ equivalent of
  //  code.append(
  //      f"(({tensor_name}._dynamo_dynamic_indices.issubset({value._dynamo_dynamic_indices}))
  //      if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)"  #
  //      noqa: B950
  //  )
 public:
  DYNAMIC_INDICES(py::set dynamic_indices, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)),
        _dynamic_indices(std::move(dynamic_indices)) {}

  bool check_nopybind(PyObject* value) override { // borrowed ref
    // 创建interned字符串 "_dynamo_dynamic_indices"
    static PyObject* dynamic_indices_str =
        PyUnicode_InternFromString("_dynamo_dynamic_indices");
    // 获取value对象的"_dynamo_dynamic_indices"属性
    PyObject* indices = PyObject_GetAttr(value, dynamic_indices_str); // new ref
    if (indices == nullptr) {
      // 属性不存在时，清除异常并返回true
      PyErr_Clear();
      return true;
    }

    // 创建interned字符串 "issubset"
    static PyObject* issubset_str = PyUnicode_InternFromString("issubset");
    // 调用indices的issubset方法，检查_dynamic_indices是否是其子集
    PyObject* call_result = PyObject_CallMethodOneArg(
        indices, issubset_str, _dynamic_indices.ptr()); // new ref
    bool result = PyObject_IsTrue(call_result);
    Py_DECREF(call_result);
    Py_DECREF(indices);
    return result;
  }

 private:
  py::set _dynamic_indices; // 存储动态索引集合的Python集合对象
};
/**
 * DICT_VERSION 类继承自 LeafGuard 类。
 * LeafGuard 是一个基类，用于表示叶子节点的保护。
 * DICT_VERSION 类的主要作用是从给定的 Python 字典对象中提取并验证版本信息。
 */
class DICT_VERSION : public LeafGuard {
 public:
  /**
   * 构造函数，初始化 DICT_VERSION 对象。
   * @param value Python 字典对象，用于提取版本信息。
   * @param verbose_code_parts 详细的代码部分，传递给基类 LeafGuard。
   * @throws py::type_error 如果 value 不是一个字典对象，则抛出类型错误。
   */
  DICT_VERSION(py::object value, py::object verbose_code_parts)
      : LeafGuard(std::move(verbose_code_parts)) {
    if (!PyDict_Check(value.ptr())) {
      throw py::type_error("DICT_VERSION expects a dict");
    }
    // 从 value 中获取并保存字典版本信息。
    _tag = get_dict_version_unchecked(value.ptr());
  }

  /**
   * 虚函数，检查给定的 PyObject 对象是否符合条件。
   * @param value 要检查的 PyObject 对象。
   * @return 如果 value 是一个字典且其版本信息与 _tag 相符，则返回 true，否则返回 false。
   */
  bool check_nopybind(PyObject* value) override { // borrowed ref
    return PyDict_Check(value) && get_dict_version_unchecked(value) == _tag;
  }

  // 保存的字典版本信息。
  uint64_t _tag;
};

/**
 * GuardAccessor 类，表示访问器和关联的保护管理器的基类。
 * 访问器定义如何从父级检查函数给定的 py::object 中访问子值。
 *
 * GuardAccessors 可以类比于 guards.py 中 Source 对象的 name() 方法。
 * 在 Python 中，name() 方法返回一个字符串，我们可以在 f_locals 和 f_globals 中评估以检索实际的 py 对象。
 * GuardAccessor 执行的任务类似，不同之处在于 GuardManager 是一个树结构，因此 GuardAccessor 只需从树的下一级检索值并传递给子 GuardAccessor。
 *
 * GuardAccessor 还拥有与从 GuardAccessor 检索的值相关联的 GuardManager。
 */
class GuardAccessor {
 public:
  /**
   * 构造函数，初始化 GuardAccessor 对象。
   * @param root 根 GuardManager。
   * @param accessor_key 访问键，用于从父级检查函数的 py::object 中访问子值。
   * @param source 源字符串。
   * @param example_value 示例值，用于创建 GuardManager。
   * @param guard_manager_enum GuardManager 的枚举类型。
   */
  GuardAccessor(
      RootGuardManager* root,
      py::object accessor_key,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : _guard_manager(make_guard_manager(
            root,
            source,
            example_value,
            guard_manager_enum)),
        _accessor_key(std::move(accessor_key)),
        _source(std::move(source)) {}

  /**
   * 获取 GuardAccessor 拥有的 GuardManager 的引用。
   * @return GuardManager 的引用。
   */
  std::unique_ptr<GuardManager>& get_guard_manager() {
    return _guard_manager;
  }

  /**
   * 检查给定的 key 是否与访问键匹配。
   * @param key 要比较的 key。
   * @return 如果匹配则返回 true，否则返回 false。
   */
  bool matches_key(const py::handle& key) const {
    return _accessor_key.equal(key);
  }

  /**
   * 获取源字符串。
   * @return 源字符串。
   */
  std::string get_source() {
  // 返回 _source 变量
  return _source;
}

virtual bool check_nopybind(PyObject* obj) = 0;
// 检查是否可以绑定到 PyObject 对象的函数，纯虚函数

virtual GuardDebugInfo check_verbose_nopybind(PyObject* obj) = 0;
// 检查是否可以详细绑定到 PyObject 对象的函数，并返回调试信息，纯虚函数

virtual std::string repr() const = 0;
// 返回描述对象的字符串表示形式，纯虚函数

virtual ~GuardAccessor() = default;
// 虚析构函数

protected:
// 从 GuardAccessor 获取的守卫管理器
std::unique_ptr<GuardManager> _guard_manager;
// 访问器键可以是 py::str（用于 getattr、getitem）或 py::function（用于 lambda 访问器）。
// 是 py::object 类型，因为需要保持这些访问器键的生命周期。

// 一个可以在 f_locals 或 f_globals 上进行 eval 的字符串，用于访问变量值。
// 仅用于调试目的。
std::string _source;
};

/**
 * GuardManager 封装了与特定 py::object 相关的所有保护器。它是一个树形结构，
 * 包括：
 * 1）Leaf guards - 在用户给定对象上运行的保护器；
 * 2）Accessors - 用于访问树层次中下一个值的访问器（如 getattr、getitem）。
 * Accessor 对象还包含子 GuardManager。
 *
 * 让我们通过一个示例来理解其工作原理：
 * class Pair:
 *     int x = 1;
 *     int y = 2;
 *
 * 编译时：
 * >> GuardManager guard_manager = GuardManager()
 * >> guard_manager.x.add_lambda_guard(
 *        lambda x: isinstance(x, Pair),
 *        lambda x: f"expected Pair, found {type(x)}"
 *    )
 * >> guard_manager.x.add_lambda_guard(lambda x: x == 1, lambda x: f"found {x}, expected 1")
 * >> guard_manager.y.add_lambda_guard(lambda x: x == 2, lambda x: f"found {x}, expected 2")
 *
 * 运行时：
 * >> guard_manager.check(Pair())
 *
 * 在编译时建立了树形结构。当我们执行 `guard_manager.x` 时，它创建了一个 AttrGuardAccessorNode，
 * 初始化了一个带有此访问器节点的子 GuardManager，并将其添加为子节点。
 * 当我们执行 `guard_manager.x.add_lambda_guard` 时，我们在新创建的 GuardManager 上调用 add_lambda_guard，
 * 在其上注册一个新的 leaf guard。
 *
 * 在运行时，访问器节点的一个重要功能是提供一种访问子保护器值的方法。在上述示例中，guard_manager.x 添加了一个
 * 带有属性名 x 的 AttrGuardAccessorNode。当调用 check 函数时，父 GuardManager 在传递给 check 函数的值上调用
 * getattr(value, "x")，以调用子 GuardManager 的 check 函数。
 *
 * 效率优化 - 为了快速失败的性能优化，在运行时这里的一个优化是根据失败计数对子保护器进行排序执行。
 * 这确保我们首先运行统计上更容易失败的保护器。这可以提高在具有多个缓存条目时的缓存查找时间。
 */

class GuardManager {
 public:
  // 禁用默认构造函数
  GuardManager() = delete;
  
  // 构造函数，初始化 GuardManager
  GuardManager(RootGuardManager* root, std::string source)
      : _root(root), _source(std::move(source)) {}
  
  // 删除拷贝构造函数
  GuardManager(const GuardManager& m) = delete;
  
  // 删除赋值运算符重载
  GuardManager& operator=(const GuardManager&) = delete;
  
  // 虚析构函数，允许通过基类指针删除子类对象
  virtual ~GuardManager() = default;

  // 获取根 GuardManager 的指针
  RootGuardManager* get_root() {
    return _root;
  }

  // 获取源字符串
  std::string get_source() {
    return _source;
  }

  // 添加叶子保护器
  virtual void add_leaf_guard(std::shared_ptr<LeafGuard> leaf_guard) {
    _leaf_guards.emplace_back(std::move(leaf_guard));
  }

  /**
   * 添加具有适当 Accessor 的新保护器管理器。如果访问器已经存在，则直接返回保护器管理器。
   */
  template <typename GuardAccessorT>
  GuardManager* get_child_manager(
      py::object accessor_key,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum) {
    // accessor_key 的类型取决于 GuardAccessorT
    // 遍历所有的访问器，寻找匹配的访问器并返回其管理器
    for (const auto& accessor : _accessors) {
      if (accessor->matches_key(accessor_key)) {
        return accessor->get_guard_manager().get();
      }
    }

    // 如果没有找到匹配的访问器，则创建一个新的访问器并添加到列表中
    _accessors.emplace_back(std::make_unique<GuardAccessorT>(
        _root,
        std::move(accessor_key),
        source,
        example_value,
        guard_manager_enum));
    // 返回新添加访问器的管理器
    return _accessors.back()->get_guard_manager().get();
  }

  // 运行叶子守卫检查以及子管理器检查函数。
  //
  // 注意：此处与 check_verbose 函数存在代码重复。这是有意的。
  // check 函数位于热路径中，保持非常简单。
  // check_verbose 函数的目的是获取守卫失败的原因以了解重新编译。
  // check_verbose 函数不会改变守卫的状态，例如，不会重新排序守卫，也不会增加失败计数。
  // 为了简单起见，我们在这里复制了代码。
  virtual bool check_nopybind(PyObject* value) { // borrowed ref
    // 遍历叶子守卫
    for (const auto& guard : _leaf_guards) {
      // 如果叶子守卫检查失败，则立即退出
      if (!guard->check_nopybind(value)) { // 提前退出
        _fail_count += 1;
        // 不需要排序，直接返回失败
        return false;
      }
    }

    // 遍历访问器
    bool result = true;
    bool failed_on_first = true;
    for (const auto& accessor : _accessors) {
      // 如果访问器检查失败，则立即退出
      if (!accessor->check_nopybind(value)) { // 提前退出
        _fail_count += 1;
        result = false;
        // 需要排序，因此中断循环
        break;
      }
      failed_on_first = false;
    }

    // failed_on_first 变量仅仅是一种优化，用于避免在第一个访问器上失败时进行排序。
    // 这在我们已经对守卫进行了排序并且不需要再次排序时很有用。
    if (!result && !failed_on_first) {
      // 原地对子访问器按失败计数进行排序。这会将失败计数较高的访问器移动到队列的前面，
      // 并为下一个 check_verbose 启用快速失败。
      
      // 一个替代实现是直接在 _accessors 上使用优先队列，但由于每次运行 guards 都要弹出和创建新的优先队列，
      // 因此被拒绝。此外，这个排序发生在 check_verbose 守卫失败的不幸路径上，所以这样做应该是可以接受的。
      std::sort(
          _accessors.begin(),
          _accessors.end(),
          [](const std::unique_ptr<GuardAccessor>& a,
             const std::unique_ptr<GuardAccessor>& b) {
            return a->get_guard_manager()->fail_count() >
                b->get_guard_manager()->fail_count();
          });
    }
  // 返回一个结果变量
  return result;
}

// 这个函数与函数 check 有些重复代码。这是故意的，为了保持 check 函数简单快速。
virtual GuardDebugInfo check_verbose_nopybind(
    PyObject* value) { // borrowed ref
  int num_guards_executed = 0;
  // 遍历 leaf guards
  for (const auto& guard : _leaf_guards) {
    // 调用 leaf guard 的 check_verbose_nopybind 方法，获取调试信息
    const GuardDebugInfo& debug_info = guard->check_verbose_nopybind(value);
    num_guards_executed++;
    // 如果结果为 false，则返回调试信息，标记失败的位置
    if (!debug_info.result) {
      return GuardDebugInfo(
          false, debug_info.verbose_code_parts, num_guards_executed);
    }
  }

  // 遍历 accessors
  for (const auto& accessor : _accessors) {
    // 调用 accessor 的 check_verbose_nopybind 方法，获取调试信息
    const GuardDebugInfo& debug_info =
        accessor->check_verbose_nopybind(value);
    num_guards_executed += debug_info.num_guards_executed;
    // 如果结果为 false，则返回调试信息，标记失败的位置
    if (!debug_info.result) {
      return GuardDebugInfo(
          false, debug_info.verbose_code_parts, num_guards_executed);
    }
  }

  // 如果所有检查都通过，则返回成功的调试信息
  return GuardDebugInfo(true, num_guards_executed);
}

// 返回失败计数
int64_t fail_count() const {
  return _fail_count;
}

// DEBUG 函数 - 返回 GuardAccessor 指针的向量，因为无法返回 unique_ptr，且 pybind 不接受 unique_ptr 引用返回类型。
virtual std::vector<GuardAccessor*> get_accessors() const {
  std::vector<GuardAccessor*> ret;
  ret.reserve(_accessors.size());
  // 将 _accessors 中的每个 accessor 添加到返回向量中
  for (const auto& accessor : _accessors) {
    ret.emplace_back(accessor.get());
  }
  return ret;
}

// DEBUG 函数 - 返回 GuardManager 指针的向量，因为无法返回 unique_ptr，且 pybind 不接受 unique_ptr 引用返回类型。
virtual std::vector<GuardManager*> get_child_managers() {
  std::vector<GuardManager*> ret;
  ret.reserve(_accessors.size());
  // 将 _accessors 中每个 accessor 的 guard manager 添加到返回向量中
  for (const auto& accessor : _accessors) {
    ret.emplace_back(accessor->get_guard_manager().get());
  }
  return ret;
}

// DEBUG 函数 - 返回 LeafGuard 指针的向量，因为无法返回 unique_ptr，且 pybind 不接受 unique_ptr 引用返回类型。
std::vector<LeafGuard*> get_leaf_guards() const {
  std::vector<LeafGuard*> ret;
  ret.reserve(_leaf_guards.size());
  // 将 _leaf_guards 中的每个 leaf guard 添加到返回向量中
  for (const auto& guard : _leaf_guards) {
    ret.push_back(guard.get());
  }
  return ret;
}

// 检查 leaf guard 是否存在
bool is_leaf_guard_present(const std::string& guard_name) {
  return _inserted_leaf_guards.find(guard_name) !=
      _inserted_leaf_guards.end();
}

// 插入 leaf guard
void insert_leaf_guard(const std::string& guard_name) {
    // 将给定的 guard_name 插入到 _inserted_leaf_guards 中，用于防止重复插入
    _inserted_leaf_guards.insert(guard_name);
  }

 protected:
  // 记录该 guard 管理器的检查函数返回 False 的次数，用于排序优化
  int64_t _fail_count{0};

 private:
  // guard 管理器的根节点，用于安装关系型 guard resetters
  RootGuardManager* _root;

  // 用于在调试时在 f_locals 或 f_globals 中获取值的字符串，传递调试信息用
  std::string _source;

  // 已插入的 leaf guard 的集合，用于防止插入重复的 guards，如 TYPE_MATCH
  std::unordered_set<std::string> _inserted_leaf_guards;

  // leaf guards 是此对象的终端 guards，例如列表的类型检查。这些 guards 必须在任何子级运行之前运行。
  //
  // 这些 leaf guards 是不可洗牌的。在几乎所有情况下，这些 guards 将有一个顺序，例如 type(x) is int guard 和 x == 5 guard。
  // 我们也期望每个 GuardManager 节点的 leaf guards 非常少。
  //
  // 注意: 为什么 leaf guards 是 shared ptr？这主要是为了支持关系型 guards，如 `tensor X is not tensor Y`。
  // 这些 guards 需要多个值。我们通过创建一个持有状态的 guard 对象来处理，这个 guard 安装在多个 guard managers 中，因此是 shared ptr。
  std::vector<std::shared_ptr<LeafGuard>> _leaf_guards;

  // GuardAccessors 节点用于访问子 guards。这些 guards 是可洗牌的。在 guard 失败时，它们根据它们的失败计数进行排序，以便在下次检查时快速失败。
  std::vector<std::unique_ptr<GuardAccessor>> _accessors;
};

/**
 * RootGuardManager is the root of the guard tree. This is primarily
 * constructed to hold the relational guard pointers so that we can reset the
 * state of those guards on guard failure. All the other important
 * implementation is in GuardManager class.
 */
class RootGuardManager : public GuardManager {
 public:
  // This is the root node, set its _root member to nullptr
  RootGuardManager() : GuardManager(this, "L") {}

  // Adds the relational guard resetter
  void add_relational_guard_resetter(
      std::shared_ptr<RelationalGuard> relational_guard) {
    _relational_guard_resetters.emplace_back(std::move(relational_guard));
  }

  // Python visible API to check guard function.
  bool check(py::handle value) {
    return check_nopybind(value.ptr());
  }

  // Python visible API to check_verbose guard function.
  GuardDebugInfo check_verbose(py::handle value) {
    return check_verbose_nopybind(value.ptr());
  }

  // Fast check function.
  bool check_nopybind(PyObject* value) override { // borrowed ref
    // Check [Note on GIL interaction with mutex lock] for details on why we
    // need mutex and its interactions wth GIL.
    PyThreadState* _save = nullptr;
    Py_UNBLOCK_THREADS; // ; is added to avoid clang-formatting
    std::lock_guard<std::mutex> lock_guard(_lock);
    Py_BLOCK_THREADS; // ; is added to avoid clang-formatting

    // Get the local state. This will be used for TENSOR_MATCH guards.
    if (_init_local_state) {
      LocalState state;
      _local_state = state;
    }

    // Check the base class GuardManager's check_nopybind function
    if (!GuardManager::check_nopybind(value)) {
      _reset_relational_guard_state(); // Reset relational guard state on failure
      return false;
    }

    // Iterate over epilogue leaf guards.
    for (const auto& guard : _epilogue_lambda_guards) {
      if (!guard->check_nopybind(value)) { // early exit if any guard fails
        _reset_relational_guard_state(); // Reset relational guard state on failure
        return false;
      }
    }
    _reset_relational_guard_state(); // Reset relational guard state on success
    return true;
  }

  // Fast check_verbose function.
  GuardDebugInfo check_verbose_nopybind(
      PyObject* value) override { // borrowed ref
    // Check [Note on GIL interaction with mutex lock] for details on why we
    // need mutex and its interactions wth GIL.
    PyThreadState* _save = nullptr;
    Py_UNBLOCK_THREADS; // ; is added to avoid clang-formatting
    std::lock_guard<std::mutex> lock_guard(_lock);
    Py_BLOCK_THREADS; // ; is added to avoid clang-formatting

    // Get the local state. This will be used for TENSOR_MATCH guards.
    if (_init_local_state) {
      LocalState state;
      _local_state = state;
    }

    // Check the base class GuardManager's check_verbose_nopybind function
    GuardDebugInfo debug_info = GuardManager::check_verbose_nopybind(value);
    if (!debug_info.result) {
      _reset_relational_guard_state(); // Reset relational guard state on failure
      return debug_info;
    }

    int num_guards_executed = debug_info.num_guards_executed;

    // Iterate over epilogue leaf guards
    for (const auto& guard : _epilogue_lambda_guards) {
      GuardDebugInfo info = guard->check_verbose_nopybind(value);
      if (!info.result) { // early exit if any guard fails
        _reset_relational_guard_state(); // Reset relational guard state on failure
        return info;
      }
      num_guards_executed += info.num_guards_executed;
    }
    _reset_relational_guard_state(); // Reset relational guard state on success
    debug_info.num_guards_executed = num_guards_executed;
    return debug_info;
  }

 private:
  std::vector<std::shared_ptr<RelationalGuard>> _relational_guard_resetters;
};
    // 遍历_epilogue_lambda_guards中的每个元素，执行以下操作
    for (const auto& guard : _epilogue_lambda_guards) {
      // 调用LeafGuard对象的check_verbose_nopybind方法，传入参数value，返回GuardDebugInfo对象
      const GuardDebugInfo& tmp_debug_info =
          guard->check_verbose_nopybind(value);
      // 增加已执行的保护条件数量
      num_guards_executed++;
      // 如果tmp_debug_info.result为false，则执行以下操作
      if (!tmp_debug_info.result) {
        // 重置关系保护条件的状态
        _reset_relational_guard_state();
        // 返回一个包含错误信息的GuardDebugInfo对象，同时返回已执行的保护条件数量
        return GuardDebugInfo(
            false, tmp_debug_info.verbose_code_parts, num_guards_executed);
      }
    }
    // 在所有保护条件均通过时，重置关系保护条件的状态
    _reset_relational_guard_state();
    // 返回一个指示所有保护条件均通过的GuardDebugInfo对象，同时返回已执行的保护条件数量
    return GuardDebugInfo(true, num_guards_executed);
  }

  // 将一个唯一指针的LeafGuard对象添加到_epilogue_lambda_guards中
  void add_epilogue_lambda_guard(std::unique_ptr<LeafGuard> leaf_guard) {
    _epilogue_lambda_guards.emplace_back(std::move(leaf_guard));
  }

  // 设置_init_local_state标志为true
  void set_init_local_state_flag() {
    _init_local_state = true;
  }

  // DEBUG函数 - 返回原始指针，因为无法返回unique_ptr，而pybind不接受unique_ptr引用类型的返回值
  // 返回_epilogue_lambda_guards中所有LeafGuard对象的指针
  std::vector<LeafGuard*> get_epilogue_lambda_guards() const {
    std::vector<LeafGuard*> ret;
    ret.reserve(_epilogue_lambda_guards.size());
    // 遍历_epilogue_lambda_guards中的每个元素，将其指针添加到ret向量中
    for (const auto& guard : _epilogue_lambda_guards) {
      ret.push_back(guard.get());
    }
    // 返回包含所有LeafGuard对象指针的向量ret
    return ret;
  }

 private:
  // 重置所有关系保护条件的状态
  void _reset_relational_guard_state() {
    // 遍历_relational_guard_resetters中的每个元素，调用其reset_state方法
    for (auto& guard : _relational_guard_resetters) {
      guard->reset_state();
    }
  }
  }
  // 结束类定义的私有部分

 public:
  // TENSOR_MATCH guards 的本地状态
  LocalState _local_state;

 private:
  // 这个 guard 管理器下的所有关系型 guards。仅在 guard 评估为 False 时使用。
  // 这确保了在 guard 失败时重置 guard 状态，以便下一次调用是干净的。
  std::vector<std::shared_ptr<RelationalGuard>> _relational_guard_resetters;

  // 这些是 lambda guards，即缺乏 C++ 实现的 guards。
  // 为简单起见，我们将这些 guards 添加在根部。
  // 必须在所有其他 guard 管理器完成后运行这些 guards，以确保 epilogue guards 不会对某些不存在的 getattr 或 getitem 产生影响。
  std::vector<std::unique_ptr<LeafGuard>> _epilogue_lambda_guards;

  // [GIL 与互斥锁交互的注意事项]
  // 我们使用 std::mutex 来防止多个线程同时运行 check/check_verbose。
  // 这是为了防止由于 RelationalGuard 中状态变化而导致的竞态条件。
  //
  // 然而，我们还需要注意 GIL 与 mutex 的交互。存在死锁的可能性：
  //
  //    线程 1：持有 GIL，等待锁
  //    线程 2：持有锁，等待 GIL
  //
  // 当线程 2 较早地获取了 mutex 锁，开始运行 check 函数的关键部分，然后调用了某些 Python 函数（如 LAMBDA_GUARD）并进入了 CPython 代码库时，
  // CPython 代码库会检查是否应释放 GIL（通常在几条字节码指令之后发生）。这时线程 2 可以决定释放 GIL。
  // 线程 1 可以获取 GIL 并到达 mutex 处，那里它将永远等待下去。
  //
  // 为了避免这种情况，每个线程在获取 mutex 之前释放 GIL，并在获取 mutex 锁之后再次获取 GIL，使用 Py_BLOCK_THREADS 和 Py_UNBLOCK_THREADS 来实现。
  // 这样可以避免死锁。
  std::mutex _lock;

  // 仅在设置了该标志时初始化 LocalState。在 TENSOR_MATCH guard 初始化期间设置该标志。
  bool _init_local_state = false;
};

/*
 * Dicts are common in python code. Therefore, we handle guards for dicts
 * differently and use PyDict_* APIs which are faster than PyObject_* APIs
 * because of no ref count increments/decrements.
 *
 * DictGuardManager relies on the order of dict.keys(). It keeps track of the
 * indices of dict.keys() to access the key, value pair.
 */
typedef std::pair<std::unique_ptr<GuardManager>, std::unique_ptr<GuardManager>>
    KeyValueManager;
class DictGuardManager : public GuardManager {
 public:
  DictGuardManager(
      RootGuardManager* root,
      std::string source,
      py::handle example_value)
      : GuardManager(root, std::move(source)),
        _size(PyDict_Size(example_value.ptr())),  // 获取字典对象的大小
        _expected_type(Py_TYPE(example_value.ptr())),  // 获取字典对象的类型
        _is_exact_dict_type(PyDict_CheckExact(example_value.ptr())) {}  // 检查字典对象是否精确匹配

  GuardManager* get_key_manager(
      py::object key_index,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum) {
    KeyValueManager& key_value_manager =
        _get_index_manager(std::move(key_index));  // 获取键管理器
    if (!key_value_manager.first) {
      key_value_manager.first = make_guard_manager(
          this->get_root(),
          std::move(source),
          example_value,
          guard_manager_enum);  // 创建键的守护管理器
    };
    return key_value_manager.first.get();  // 返回键的守护管理器
  }

  GuardManager* get_value_manager(
      py::object key_index,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum) {
    KeyValueManager& key_value_manager =
        _get_index_manager(std::move(key_index));  // 获取值管理器
    if (!key_value_manager.second) {
      key_value_manager.second = make_guard_manager(
          this->get_root(),
          std::move(source),
          example_value,
          guard_manager_enum);  // 创建值的守护管理器
    };
    return key_value_manager.second.get();  // 返回值的守护管理器
  }

  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // TODO(janimesh) - Implement a fast-path using dict versions.

    if (Py_TYPE(obj) != _expected_type) {  // 检查对象类型是否符合预期
      _fail_count += 1;
      return false;
    }

    if (PyDict_Size(obj) != _size) {  // 检查字典对象大小是否与预期相同
      _fail_count += 1;
      return false;
    }

    // Early return
    if (_size == 0) {  // 如果字典为空，直接返回真
      return true;
    }

    // Invokes the base class's check_nopybind method. We permit a limited set
    // of leaf guards and accessors within the DictGuardManager framework.
    // Integrating certain guards or accessors directly within the
    // DictGuardManager can be challenging. For instance, `type(dict_object)` as
    // an accessor is permissible, which otherwise would be hard to integrate
    // directly into DictGuardManager.  Similarly, incorporating guards such as
    // DICT_CONTAINS and DICT_VERSION as leaf guards offers a simpler solution
    // than embedding these functionalities within the DictGuardManager itself.
    if (!GuardManager::check_nopybind(obj)) {  // 调用基类的方法检查对象
      _fail_count += 1;
      // No need to shuffle the child guards, just return.
      return false;
    }

    PyObject *key = nullptr, *value = nullptr;
    Py_ssize_t pos = 0;

    // Points to an element in the _indices vector.
    size_t index_pointer = 0;
    // Points to the current index in the Python dictionary
    Py_ssize_t dict_pointer = 0;

    // Iterate through _indices and Python dictionary obj simultaneously
    while (index_pointer < _indices.size() &&
           PyDict_Next(obj, &pos, &key, &value)) {
      // Skip if dict_pointer is not a saved index.
      if (dict_pointer == _indices[index_pointer]) {
        index_pointer += 1;
        // Access the key_value_manager corresponding to dict_pointer
        KeyValueManager& key_value_manager = _key_value_managers[dict_pointer];
        // Access the GuardManager for the key
        std::unique_ptr<GuardManager>& key_manager = key_value_manager.first;
        // Check if key_manager exists and if key passes the check_nopybind function
        if (key_manager && !key_manager->check_nopybind(key)) {
          return false;
        }
        // Access the GuardManager for the value
        std::unique_ptr<GuardManager>& value_manager = key_value_manager.second;
        // Check if value_manager exists and if value passes the check_nopybind function
        if (value_manager && !value_manager->check_nopybind(value)) {
          return false;
        }
      }
      dict_pointer += 1;
    }
    // All checks passed successfully
    return true;
  }

  // Verifies detailed debugging information for objects of _expected_type
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // Check if the object's type matches _expected_type
    if (Py_TYPE(obj) != _expected_type) {
      return GuardDebugInfo(false, "TYPE_MISMATCH(" + get_source() + ")", 0);
    }

    // Check if the size of the Python dictionary obj matches _size
    if (PyDict_Size(obj) != _size) {
      return GuardDebugInfo(
          false, "len(" + get_source() + ") != " + std::to_string(_size), 0);
    }

    // Early return if _size is 0, indicating success
    if (_size == 0) {
      return GuardDebugInfo(true, 0);
    }

    // Invoke the base class's check_nopybind method to perform basic checks
    GuardDebugInfo debug_info = GuardManager::check_verbose_nopybind(obj);
    // If basic checks failed, return the failure information
    if (!debug_info.result) {
      return debug_info;
    }

    PyObject *key = nullptr, *value = nullptr;
    Py_ssize_t pos = 0;

    // Points to an element in the _indices vector.
    size_t index_pointer = 0;
    // Points to the current index in the Python dictionary
    Py_ssize_t dict_pointer = 0;

    // Counter to track the number of guards executed
    int num_guards_executed = 0;
    while (index_pointer < _indices.size() &&
           PyDict_Next(obj, &pos, &key, &value)) {
      // 循环遍历直到索引指针超出范围或者无法从 Python 字典中取出下一个键值对
      // 检查当前 pos 是否为保存的索引，若不是则跳过
      if (dict_pointer == _indices[index_pointer]) {
        index_pointer += 1;
        // 获取当前索引对应的 KeyValueManager 对象引用
        KeyValueManager& key_value_manager = _key_value_managers[dict_pointer];
        // 获取键的 GuardManager 对象的唯一指针
        std::unique_ptr<GuardManager>& key_manager = key_value_manager.first;
        // 如果存在键的 GuardManager，则执行检查并获取调试信息
        if (key_manager) {
          GuardDebugInfo debug_info = key_manager->check_verbose_nopybind(key);
          num_guards_executed += debug_info.num_guards_executed;
          // 如果检查结果为假，则返回相应的 GuardDebugInfo 对象
          if (!debug_info.result) {
            return GuardDebugInfo(
                false, debug_info.verbose_code_parts, num_guards_executed);
          }
        }
        // 获取值的 GuardManager 对象的唯一指针
        std::unique_ptr<GuardManager>& value_manager = key_value_manager.second;
        // 如果存在值的 GuardManager，则执行检查并获取调试信息
        if (value_manager) {
          GuardDebugInfo debug_info =
              value_manager->check_verbose_nopybind(value);
          num_guards_executed += debug_info.num_guards_executed;
          // 如果检查结果为假，则返回相应的 GuardDebugInfo 对象
          if (!debug_info.result) {
            return GuardDebugInfo(
                false, debug_info.verbose_code_parts, num_guards_executed);
          }
        }
      }
      dict_pointer += 1;
    }
    // 循环结束后返回一个表示执行成功的 GuardDebugInfo 对象
    return GuardDebugInfo(true, num_guards_executed);
  }

  void skip_adding_guard(const py::object& a, const py::object& b) {
    // 重写了 `DictGuardManager` 中的 `add_leaf_guard` 方法，以阻止添加叶子 guard
    // 但是这种做法过于严格。Python 侧的 guard 管理频繁在 `DictGuardManager` 上添加 `TYPE_MATCH` 和 `DICT_LENGTH`
    // 我们可以重构 Python 侧代码，不再在 dict 对象上调用这些 guards，但会导致代码混乱。
    // 因此，我们选择重写这两个 guards，跳过 `add_leaf_guard` 的代码路径，从而跳过添加 guards 的步骤。这样做简化了 Python 侧的操作。
  }

  void fail_on_get_child_manager(
      const py::object& a,
      const std::string& source,
      const py::object& b) {
    // 抛出异常，指示不能向 `DictGuardManager` 添加访问器
    throw std::runtime_error("Can not add an accessor to DictGuardManager");
  }

  void add_leaf_guard(std::shared_ptr<LeafGuard> leaf_guard) override {
    // 如果调用了这个方法，你可能想要通过键值子管理器添加一个 leaf guard
    // `DictGuardManager` 已经内建了 `TYPE_MATCH` 和 `LENGTH_CHECK`
    throw std::runtime_error("DictGuardManager does not support a leaf_guard");
  }

  void add_permitted_leaf_guard(std::shared_ptr<LeafGuard> leaf_guard) {
    // 仅对允许的 guards 调用
    GuardManager::add_leaf_guard(std::move(leaf_guard));
  }

  // Debug helper - Returning raw pointers because we can't return unique_ptr
  // and pybind does not accept a unique_ptr reference return type.
  std::unordered_map<Py_ssize_t, std::pair<GuardManager*, GuardManager*>>
  get_key_value_managers() {
    std::unordered_map<Py_ssize_t, std::pair<GuardManager*, GuardManager*>> ret;
    // 返回包含索引到键值对 GuardManager 指针对的无序映射
  /**
   * 遍历 `_indices` 数组，根据每个索引获取 `_key_value_managers` 中对应的管理器，
   * 并将其作为键值对的形式存入 `ret` 数组中。
   * 返回 `ret` 数组，其中包含了索引到键值管理器的映射。
   */
  for (auto index : _indices) {
    ret[index] = std::make_pair(
        _key_value_managers[index].first.get(),
        _key_value_managers[index].second.get());
  }
  return ret;
}

/**
 * 返回 `_is_exact_dict_type` 的值，用于判断是否精确字典类型。
 */
bool is_exact_dict_type() {
  return _is_exact_dict_type;
}

private:
/**
 * 获取指定 `key_index` 的 `KeyValueManager` 对象。
 * 如果该索引的管理器已存在，则直接返回该管理器。
 * 如果不存在，则创建一个新的管理器，并将其与索引关联。
 * 同时，将索引添加到 `_indices` 数组，并保持数组按升序排序。
 * 最后返回与索引关联的 `KeyValueManager` 对象。
 */
KeyValueManager& _get_index_manager(py::object key_index) {
  // 检查索引是否已存在
  Py_ssize_t index = py::cast<Py_ssize_t>(std::move(key_index));
  auto it = _key_value_managers.find(index);
  if (it != _key_value_managers.end()) {
    return it->second;
  }
  _indices.push_back(index);
  // 总是保持 _indices 数组按升序排序
  std::sort(_indices.begin(), _indices.end());
  _key_value_managers[index] = std::make_pair(nullptr, nullptr);
  return _key_value_managers[index];
}

protected: // also used by DictSubclassGuardManager
Py_ssize_t _size;
// DictGuardManager supports both exact dict type and non-exact dict type.
// Therefore, we have to compare the type to early exit.
PyTypeObject* _expected_type;
bool _is_exact_dict_type; // Useful to check getattr_manager validity.
std::vector<Py_ssize_t> _indices;
std::unordered_map<Py_ssize_t, KeyValueManager> _key_value_managers;
};

/**
 * DictSubclassGuardManager 是为 dict 的子类设计的守卫管理器，
 * 特别是针对 OrderedDict。标准字典使用 PyDict_Next 函数迭代键、值和项。
 * OrderedDict 则依赖额外的链表结构来维护键的顺序。尽管 PyDict_Next 和
 * OrderedDict 通常产生相同的顺序，但在使用 OrderedDict 的 move_to_end 方法
 * （例如在 Pytorch 钩子中使用）时会出现差异。`move_to_end` 方法仅更新了链表，
 * 而不影响 PyDict_Next 的结果。因此，在这种情况下，DictSubclassGuardManager
 * 直接调用 .keys() 方法以准确捕捉键的顺序。
 */

class DictSubclassGuardManager : public DictGuardManager {
 public:
  DictSubclassGuardManager(
      RootGuardManager* root,
      std::string source,
      py::handle example_value)
      : DictGuardManager(root, std::move(source), example_value) {}

 public:
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // TODO(janimesh) - Implement a fast-path using dict versions.

    // 如果传入对象的类型与预期类型不符，则增加失败计数并返回 false
    if (Py_TYPE(obj) != _expected_type) {
      _fail_count += 1;
      return false;
    }

    // 如果传入对象的大小与预期大小不符，则增加失败计数并返回 false
    if (PyDict_Size(obj) != _size) {
      _fail_count += 1;
      return false;
    }

    // 如果预期大小为 0，则直接返回 true
    if (_size == 0) {
      return true;
    }

    // 如果基类 GuardManager 的检查方法返回 false，则增加失败计数并返回 false
    if (!GuardManager::check_nopybind(obj)) { // NOLINT
      _fail_count += 1;
      // 不需要重新排列子守卫，直接返回
      return false;
    }

    // 指向 _indices 向量中的一个元素
    size_t index_pointer = 0;
    // 指向字典中的键索引
    Py_ssize_t dict_pointer = 0;

    // 使用 iter(dict.keys()) 来迭代字典的键
    py::object keys =
        py::handle(obj).attr("keys")(); // py::object handles the references
    PyObject* iterator = PyObject_GetIter(keys.ptr()); // new reference
    PyObject* key = nullptr;

    // 当 index_pointer 小于 _indices 大小且仍有可迭代的键时执行循环
    while (index_pointer < _indices.size() &&
           (key = PyIter_Next(iterator))) { // new reference
      // 如果 dict_pointer 等于 _indices[index_pointer]，则执行以下操作
      if (dict_pointer == _indices[index_pointer]) {
        // 获取对应的 KeyValueManager 实例
        KeyValueManager& key_value_manager = _key_value_managers[dict_pointer];
        // 获取键管理器，并检查键是否符合预期
        std::unique_ptr<GuardManager>& key_manager = key_value_manager.first;
        if (key_manager && !key_manager->check_nopybind(key)) {
          Py_DECREF(key);
          Py_DECREF(iterator);
          return false;
        }

        // 获取字典中键对应的值
        PyObject* value = PyDict_GetItem(obj, key); // borrowed ref
        // 获取值管理器，并检查值是否符合预期
        std::unique_ptr<GuardManager>& value_manager = key_value_manager.second;
        if (value_manager && !value_manager->check_nopybind(value)) {
          Py_DECREF(key);
          Py_DECREF(iterator);
          return false;
        }

        // index_pointer 向后移动
        index_pointer++;
      }
      // dict_pointer 向后移动
      dict_pointer++;
      Py_DECREF(key);
    }

    // 释放迭代器对象并返回 true
    Py_DECREF(iterator);
    return true;
  }

  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // 检查对象的类型是否与预期类型不同
    if (Py_TYPE(obj) != _expected_type) {
      // 如果类型不匹配，则返回带有调试信息的 GuardDebugInfo 对象
      return GuardDebugInfo(false, "TYPE_MISMATCH(" + get_source() + ")", 0);
    }

    // 检查字典对象的大小是否与预期大小不同
    if (PyDict_Size(obj) != _size) {
      // 如果大小不匹配，则返回带有调试信息的 GuardDebugInfo 对象
      return GuardDebugInfo(
          false, "len(" + get_source() + ") != " + std::to_string(_size), 0);
    }

    // 若预期大小为 0，则直接返回执行成功的 GuardDebugInfo 对象
    if (_size == 0) {
      return GuardDebugInfo(true, 0);
    }

    // 调用 GuardManager::check_verbose_nopybind 检查对象，获取调试信息
    GuardDebugInfo debug_info =
        GuardManager::check_verbose_nopybind(obj); // NOLINT
    // 如果检查不通过，则直接返回调试信息
    if (!debug_info.result) {
      return debug_info;
    }

    // 初始化索引指针
    size_t index_pointer = 0;
    // 初始化字典键的指针
    Py_ssize_t dict_pointer = 0;

    // 初始化执行的守卫数量
    int num_guards_executed = 0;

    // 使用 iter(dict.keys()) 获取字典的键的迭代器
    py::object keys =
        py::handle(obj).attr("keys")(); // py::object handles the references
    // 获取键的迭代器对象
    PyObject* iterator = PyObject_GetIter(keys.ptr()); // new reference
    // 初始化键对象
    PyObject* key = nullptr;

    // 迭代键对象，同时检查索引指针是否小于 _indices 的大小
    while (index_pointer < _indices.size() &&
           (key = PyIter_Next(iterator))) { // new reference
      // 如果字典指针等于 _indices 中的索引值
      if (dict_pointer == _indices[index_pointer]) {
        // 获取键值管理器
        KeyValueManager& key_value_manager = _key_value_managers[dict_pointer];
        // 获取键管理器对象
        std::unique_ptr<GuardManager>& key_manager = key_value_manager.first;
        // 如果存在键管理器，则调用它的检查方法
        if (key_manager) {
          GuardDebugInfo debug_info = key_manager->check_verbose_nopybind(key);
          // 累加执行的守卫数量
          num_guards_executed += debug_info.num_guards_executed;
          // 如果检查不通过，则释放资源并返回错误调试信息
          if (!debug_info.result) {
            Py_DECREF(key);
            Py_DECREF(iterator);
            return GuardDebugInfo(
                false, debug_info.verbose_code_parts, num_guards_executed);
          }
        }

        // 获取键对应的值对象
        PyObject* value = PyDict_GetItem(obj, key); // borrowed ref
        // 获取值管理器对象
        std::unique_ptr<GuardManager>& value_manager = key_value_manager.second;
        // 如果存在值管理器，则调用它的检查方法
        if (value_manager) {
          GuardDebugInfo debug_info =
              value_manager->check_verbose_nopybind(value);
          // 累加执行的守卫数量
          num_guards_executed += debug_info.num_guards_executed;
          // 如果检查不通过，则释放资源并返回错误调试信息
          if (!debug_info.result) {
            Py_DECREF(key);
            Py_DECREF(iterator);
            return GuardDebugInfo(
                false, debug_info.verbose_code_parts, num_guards_executed);
          }
        }
        // 更新索引指针
        index_pointer++;
      }
      // 释放键对象
      Py_DECREF(key);
      // 更新字典指针
      dict_pointer++;
    }

    // 释放迭代器对象
    Py_DECREF(iterator);
    // 返回执行成功的 GuardDebugInfo 对象，包含执行的守卫数量
    return GuardDebugInfo(true, num_guards_executed);
  }
};

// 创建一个管理器的工厂函数，返回一个 unique_ptr，指向 GuardManager 的子类对象
std::unique_ptr<GuardManager> make_guard_manager(
    RootGuardManager* root,    // 根 GuardManager 指针
    std::string source,        // 源字符串
    py::handle example_value,  // Python 对象的句柄，用作示例值
    py::handle guard_manager_enum) {  // Python 对象的句柄，表示 GuardManager 的枚举

  // 导入 GuardManagerType 类
  static py::object guard_manager_enum_class =
      py::module_::import("torch._dynamo.guards").attr("GuardManagerType");

  // 获取 GuardManagerType 类中的三个枚举对象
  static py::object base_guard_manager_enum =
      guard_manager_enum_class.attr("GUARD_MANAGER");
  static py::object dict_guard_manager_enum =
      guard_manager_enum_class.attr("DICT_GUARD_MANAGER");
  static py::object dict_subclass_guard_manager_enum =
      guard_manager_enum_class.attr("DICT_SUBCLASS_GUARD_MANAGER");

  // 检查 example_value 是否为 Python 字典类型
  if (py::isinstance<py::dict>(example_value)) {
    // 对于需要处理字典及其子类中键顺序变化的情况，使用 DictGuardManager 和 DictSubclassGuardManager

    // 插入字典守卫时（参见 guards.py），我们依赖于 list(d.keys()) 的顺序。因此，cpp 守卫的等效版本必须具有相同的键顺序。
    // 对于标准字典，.keys() API 内部使用 PyDict_Next。因此，DictGuardManager 直接使用 PyDict_Next 加快键获取速度。

    // 但是 PyDict_Next 可能无法正确地处理字典的子类的键顺序。
    // 例如，OrderedDict 重写了 .keys() API 而没有更改底层数据结构。这导致其键的顺序与 PyDict_Next 给出的顺序不同。
    // 我们使用 DictSubclassGuardManager 来处理这种差异。DictSubclassGuardManager 直接调用 .keys() API 以准确捕获键顺序。
    // 这种方法不如使用 PyDict_Next 高效，但它确保正确性。

    // 由于常规字典比覆盖 keys 方法的字典子类更常见，我们仍然优化常见情况，使用 DictGuardManager 并依赖 PyDict_Next。

    if (guard_manager_enum.is(base_guard_manager_enum)) {
      // 对于不需要对键进行守卫的字典，我们可以简单地依赖于基础 GuardManager。
      return std::make_unique<GuardManager>(root, std::move(source));
    } else if (guard_manager_enum.is(dict_guard_manager_enum)) {
      // 使用 DictGuardManager 处理字典类型
      return std::make_unique<DictGuardManager>(
          root, std::move(source), example_value);
    } else if (guard_manager_enum.is(dict_subclass_guard_manager_enum)) {
      // 使用 DictSubclassGuardManager 处理字典子类类型
      return std::make_unique<DictSubclassGuardManager>(
          root, std::move(source), example_value);
    } else {
      // 抛出类型错误，表示无效的守卫管理器枚举
      throw py::type_error("Invalid guard manager enum");
    }
  }
  
  // 如果 example_value 不是 Python 字典类型，返回基础 GuardManager
  return std::make_unique<GuardManager>(root, std::move(source));
}
class TENSOR_MATCH : public LeafGuard {
 public:
  /**
   * Constructor for TENSOR_MATCH class.
   *
   * @param root_guard_manager Pointer to RootGuardManager instance.
   * @param value Python object representing the tensor.
   * @param dynamic_dims_sizes_py Python object holding dynamic dimensions sizes.
   * @param dynamic_dims_strides_py Python object holding dynamic dimensions strides.
   * @param tensor_name Python object representing the name of the tensor.
   * @param verbose_code_parts Python object containing verbose code parts.
   */
  TENSOR_MATCH(
      RootGuardManager* root_guard_manager,
      py::object value,
      py::object dynamic_dims_sizes_py,
      py::object dynamic_dims_strides_py,
      py::object tensor_name,
      py::object verbose_code_parts)
      : LeafGuard(root_guard_manager, std::move(verbose_code_parts)),
        _tensor_name(py::cast<py::str>(std::move(tensor_name))) {
    root_guard_manager->set_init_local_state_flag();
    PyObject* item = value.ptr();
    // Check if the provided Python object is of type Tensor
    if (!THPVariable_CheckExact(item) && !THPVariable_Check(item)) {
      PyErr_SetString(PyExc_TypeError, "expected Tensor()");
      return;
    }
    auto tensor = THPVariable_Unpack(item);

    // Convert dynamic dimensions sizes and strides from Python lists to C++ vectors of optional SymInt
    std::vector<std::optional<c10::SymInt>> tensor_dims_size =
        pyListToVecOptInt(dynamic_dims_sizes_py.ptr());
    std::vector<std::optional<c10::SymInt>> tensor_dims_stride =
        pyListToVecOptInt(dynamic_dims_strides_py.ptr());

    // If dynamic dimensions sizes or strides are empty, use symbolic sizes and strides from the tensor
    tensor_dims_size = tensor_dims_size.empty()
        ? wrapIntegersInOptional(tensor.sym_sizes())
        : tensor_dims_size;
    tensor_dims_stride = tensor_dims_stride.empty()
        ? wrapIntegersInOptional(tensor.sym_strides())
        : tensor_dims_stride;

    // Initialize local state and TensorCheck object
    LocalState state;
    _tensor_check = std::make_unique<TensorCheck>(
        state,
        Py_TYPE(item),
        std::move(tensor),
        std::move(tensor_dims_size),
        std::move(tensor_dims_stride));
  }

  /**
   * Checks if the provided Python object matches the tensor type.
   *
   * @param value Borrowed reference to the Python object to check.
   * @return True if the object matches the tensor type, otherwise false.
   */
  bool check_nopybind(PyObject* value) override { // borrowed ref
    if (Py_TYPE(value) != _tensor_check->pytype) {
      return false;
    }
    return _tensor_check->check(
        _root_guard_manager->_local_state, THPVariable_Unpack(value));
  }

  /**
   * Verbose check for tensor type compatibility.
   *
   * @param value Borrowed reference to the Python object to check.
   * @return GuardDebugInfo object indicating success or failure of the check.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* value) override { // borrowed ref

    if (Py_TYPE(value) != _tensor_check->pytype) {
      std::stringstream fail_reason;
      PyObject* type_str = PyObject_Str(PyObject_Type(value));
      fail_reason << "expected type of '" << _tensor_name
                  << "' to be a tensor type, ";
      if (!type_str) {
        fail_reason << "but found a different type";
      } else {
        fail_reason << "' but found " << PyUnicode_AsUTF8(type_str);
      }
      return GuardDebugInfo(false, fail_reason.str(), 0);
    }

    // Perform verbose tensor type check using TensorCheck object
    std::string fail_reason = _tensor_check->check_verbose(
        _root_guard_manager->_local_state,
        THPVariable_Unpack(value),
        _tensor_name);

    if (!fail_reason.empty()) {
      return GuardDebugInfo(false, fail_reason, 0);
    }
    return GuardDebugInfo(true, 1);
  }

 private:
  std::string _tensor_name;  // Name of the tensor
  std::unique_ptr<TensorCheck> _tensor_check;  // Unique pointer to TensorCheck object
};
/**
 * Represents __getattr__ acccessor.
 */
/**
 * Represents a guard accessor for accessing an attribute via PyObject_GetAttr.
 */
class GetAttrGuardAccessor : public GuardAccessor {
 public:
  GetAttrGuardAccessor(
      RootGuardManager* root,
      py::str name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            name,
            std::move(source),
            example_value,
            guard_manager_enum),
        _attr_name(name.ptr()) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  /**
   * Checks if the attribute '_attr_name' exists on the given PyObject 'obj'.
   * Returns true if the attribute exists and passes the guard check, false otherwise.
   * Clears any exceptions if the attribute is absent.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* x = PyObject_GetAttr(obj, _attr_name); // new ref
    if (x == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return false;
    }
    bool result = _guard_manager->check_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  /**
   * Verbose check if the attribute '_attr_name' exists on the given PyObject 'obj'.
   * Returns a GuardDebugInfo object indicating whether the check passed.
   * If the attribute is absent, returns a GuardDebugInfo with failure details.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* x = PyObject_GetAttr(obj, _attr_name); // new ref
    if (x == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return GuardDebugInfo(
          false, "getattr failed on source " + get_source(), 0);
    }
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  /**
   * Returns a string representation of this guard accessor.
   * Useful for debugging and printing the GuardManager tree structure.
   */
  std::string repr() const override {
    return "GetAttrGuardAccessor(" + py::str(_attr_name).cast<std::string>() +
        ")";
  }

 private:
  PyObject* _attr_name;  // Holds the PyObject representing the attribute name.
};

/**
 * Represents a guard accessor for accessing '__dict__' via PyObject_GenericGetDict.
 */
class GetGenericDictGuardAccessor : public GuardAccessor {
 public:
  GetGenericDictGuardAccessor(
      RootGuardManager* root,
      py::str name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            std::move(name),
            std::move(source),
            example_value,
            guard_manager_enum) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  /**
   * Checks if the '__dict__' of the given PyObject 'obj' exists and passes the guard check.
   * Returns true if '__dict__' exists and the guard check is successful, false otherwise.
   * Clears any exceptions if the '__dict__' is absent.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* x = PyObject_GenericGetDict(obj, nullptr); // new ref
    if (x == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return false;
    }
    bool result = _guard_manager->check_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  /**
   * Verbose check if the '__dict__' of the given PyObject 'obj' exists and passes the guard check.
   * Returns a GuardDebugInfo object indicating whether the check passed.
   * If '__dict__' is absent, returns a GuardDebugInfo with failure details.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* x = PyObject_GenericGetDict(obj, nullptr); // new ref
    if (x == nullptr) {
      // Attribute absent, clear the exception and return false.
      PyErr_Clear();
      return GuardDebugInfo(
          false, "getattr failed on source " + get_source(), 0);
    }
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    Py_DECREF(x);
    return result;
  }
};
    // 调用 _guard_manager 对象的 check_verbose_nopybind 方法，传入 x 作为参数，返回 GuardDebugInfo 结果
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    // 对传入的 x 执行 Py_DECREF，减少其引用计数，可能会释放 x 指向的对象
    Py_DECREF(x);
    // 返回 check_verbose_nopybind 方法的结果
    return result;
  }

  // 返回字符串 "GetGenericDictGuardAccessor"，用于表示此类的字符串表示形式
  std::string repr() const override {
    // 用于打印 GuardManager 树结构时的辅助信息
    return "GetGenericDictGuardAccessor";
  }
};

/**
 * Represents __getitem__ acccessor.
 */
class GetItemGuardAccessor : public GuardAccessor {
 public:
  GetItemGuardAccessor(
      RootGuardManager* root,
      py::object name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            name,
            std::move(source),
            example_value,
            guard_manager_enum),
        _attr_name(name.ptr()) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  // 检查是否可以获取对象的指定属性，无需绑定
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // 获取对象 obj 中的属性 _attr_name，返回一个新的引用 x
    PyObject* x = PyObject_GetItem(obj, _attr_name); // new ref
    if (x == nullptr) {
      PyErr_Clear();  // 清除 Python 错误状态
      return false;
    }
    // 使用 GuardManager 检查对象 x 是否符合条件
    bool result = _guard_manager->check_nopybind(x);
    Py_DECREF(x);  // 释放 x 的引用计数
    return result;
  }

  // 检查是否可以获取对象的指定属性，提供详细的错误信息，无需绑定
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // 获取对象 obj 中的属性 _attr_name，返回一个新的引用 x
    PyObject* x = PyObject_GetItem(obj, _attr_name); // new ref
    if (x == nullptr) {
      PyErr_Clear();  // 清除 Python 错误状态
      return GuardDebugInfo(
          false, std::string("KeyError on ") + get_source(), 0);
    }
    // 使用 GuardManager 检查对象 x 是否符合条件，提供详细的调试信息
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    Py_DECREF(x);  // 释放 x 的引用计数
    return result;
  }

  // 返回对象的字符串表示形式，包括属性 _attr_name 的名称
  std::string repr() const override {
    return "GetItemGuardAccessor(" + py::str(_attr_name).cast<std::string>() +
        ")";
  }

 private:
  // 由于属性名称已作为 accessor_key 传递给基类，这里不需要 py::object
  PyObject* _attr_name;  // 属性名称的 PyObject 指针
};

/**
 * Represents dict[name] acccessor. This is ONLY used for f_locals because its a
 * dict, and DictGuardManager does not support sorting. We differentiate it from
 * GetItemGuardAccessor because PyDict_GetItem should be fasten the
 * PyObject_GetItem.
 */
class DictGetItemGuardAccessor : public GuardAccessor {
 public:
  DictGetItemGuardAccessor(
      RootGuardManager* root,
      py::object key,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            key,
            std::move(source),
            example_value,
            guard_manager_enum),
        _key(key.ptr()) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  // 检查是否可以从字典对象获取指定键对应的值，无需绑定
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // 获取字典对象 obj 中键 _key 对应的值，返回一个 borrowed 引用 x
    PyObject* x = PyDict_GetItem(obj, _key); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();  // 清除 Python 错误状态
      return false;
    }
    // 使用 GuardManager 检查对象 x 是否符合条件
    bool result = _guard_manager->check_nopybind(x);
    return result;
  }

  // 检查是否可以从字典对象获取指定键对应的值，提供详细的错误信息，无需绑定
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // 获取字典对象 obj 中键 _key 对应的值，返回一个 borrowed 引用 x
    PyObject* x = PyDict_GetItem(obj, _key); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();  // 清除 Python 错误状态
      return GuardDebugInfo(
          false, std::string("KeyError on ") + get_source(), 0);
    }
    // 使用 GuardManager 检查对象 x 是否符合条件，提供详细的调试信息
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    return result;
  }

 private:
  PyObject* _key;  // 键的 PyObject 指针
};
    return result;
  }


// 返回当前对象的结果。通常用于返回函数的执行结果或状态。
    std::string repr() const override {


// 返回当前对象的字符串表示，格式为 "DictGetItemGuardAccessor(_key的字符串表示)"
    return "DictGetItemGuardAccessor(" + py::str(_key).cast<std::string>() +
        ")";
  }

 private:


// 私有成员变量，存储一个指向 PyObject 的指针 _key。
  PyObject* _key;
};

/**
 * Represents list[index] accessor. It is faster than generic
 * GetItemGuardAccessor.
 */
class ListGetItemGuardAccessor : public GuardAccessor {
 public:
  ListGetItemGuardAccessor(
      RootGuardManager* root,
      const py::object& index,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            index,
            std::move(source),
            example_value,
            guard_manager_enum),
        _index(py::cast<Py_ssize_t>(index)) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  /**
   * Check if the object at index in the list is valid and meets guard conditions.
   * This version is for quick checks without detailed debugging information.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* x = PyList_GetItem(obj, _index); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return false;
    }
    bool result = _guard_manager->check_nopybind(x);
    return result;
  }

  /**
   * Check if the object at index in the list is valid and meets guard conditions.
   * This version includes detailed debugging information on failure.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* x = PyList_GetItem(obj, _index); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return GuardDebugInfo(
          false, std::string("IndexError on ") + get_source(), 0);
    }
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    return result;
  }

  /**
   * Returns a string representation of the ListGetItemGuardAccessor.
   */
  std::string repr() const override {
    return "ListGetItemGuardAccessor(" + std::to_string(_index) + ")";
  }

 private:
  Py_ssize_t _index;
};

/**
 * Represents tuple[index] accessor. It is faster than generic
 * GetItemGuardAccessor.
 */
class TupleGetItemGuardAccessor : public GuardAccessor {
 public:
  TupleGetItemGuardAccessor(
      RootGuardManager* root,
      const py::object& index,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            index,
            std::move(source),
            example_value,
            guard_manager_enum),
        _index(py::cast<Py_ssize_t>(index)) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  /**
   * Check if the object at index in the tuple is valid and meets guard conditions.
   * This version is for quick checks without detailed debugging information.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* x = PyTuple_GetItem(obj, _index); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return false;
    }
    bool result = _guard_manager->check_nopybind(x);
    return result;
  }

  /**
   * Check if the object at index in the tuple is valid and meets guard conditions.
   * This version includes detailed debugging information on failure.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* x = PyTuple_GetItem(obj, _index); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return GuardDebugInfo(
          false, std::string("IndexError on ") + get_source(), 0);
    }
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    return result;
  }

  /**
   * Returns a string representation of the TupleGetItemGuardAccessor.
   */
  std::string repr() const override {
    return "TupleGetItemGuardAccessor(" + std::to_string(_index) + ")";
  }

 private:
  Py_ssize_t _index;
};

/**
 * Represents tensor.grad acccessor.
 */
/**
 * Represents a subclass of GuardAccessor specialized for accessing gradient-related properties.
 */
class GradGuardAccessor : public GuardAccessor {
 public:
  /**
   * Constructor for GradGuardAccessor.
   * @param root RootGuardManager instance for managing guards.
   * @param name Name of the guard accessor.
   * @param source Source description of the guard accessor.
   * @param example_value Example value handled by the guard accessor.
   * @param guard_manager_enum Enumeration handle for the guard manager.
   */
  GradGuardAccessor(
      RootGuardManager* root,
      py::str name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            std::move(name),
            std::move(source),
            example_value,
            guard_manager_enum) {}

  /**
   * Check method for verifying if the provided object is a tensor and guarded.
   * @param obj PyObject pointer to check (borrowed reference).
   * @return true if obj is a tensor and passes the guard check, false otherwise.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // check that its a tensor
    if (!THPVariable_CheckExact(obj) && !THPVariable_Check(obj)) {
      return false;
    }
    // wrap the gradient of the tensor into a new Python object
    PyObject* grad =
        THPVariable_Wrap(THPVariable_Unpack(obj).grad()); // New reference
    // perform guard check on the gradient
    bool result = _guard_manager->check_nopybind(grad);
    // For undefined tensor, THPVariable_Wrap returns Py_RETURN_NONE. So, no
    // need of Py_XDECREF.
    Py_DECREF(grad);
    return result;
  }

  /**
   * Verbose check method for detailed verification of tensor properties.
   * @param obj PyObject pointer to check (borrowed reference).
   * @return GuardDebugInfo object containing detailed guard verification results.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // check that its a tensor
    if (!THPVariable_CheckExact(obj) && !THPVariable_Check(obj)) {
      return GuardDebugInfo(
          false, "not a tensor - grad field is accessed " + get_source(), 0);
    }
    // wrap the gradient of the tensor into a new Python object
    PyObject* grad =
        THPVariable_Wrap(THPVariable_Unpack(obj).grad()); // New reference
    // perform detailed guard check on the gradient
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(grad);
    // For undefined tensor, THPVariable_Wrap returns Py_RETURN_NONE. So, no
    // need of Py_XDECREF.
    Py_DECREF(grad);
    return result;
  }

  /**
   * String representation method for displaying the structure of GuardManager.
   * @return String representation indicating this accessor deals with gradients.
   */
  std::string repr() const override {
    // Helpful when priting GuardManager tree structure.
    return "GradGuardAccessor(grad)";
  }
};

/**
 * Represents func.__defaults__ accessor for guarding.
 */
class FuncDefaultsGuardAccessor : public GuardAccessor {
 public:
  /**
   * Constructor for FuncDefaultsGuardAccessor.
   * @param root RootGuardManager instance for managing guards.
   * @param name Name of the guard accessor.
   * @param source Source description of the guard accessor.
   * @param example_value Example value handled by the guard accessor.
   * @param guard_manager_enum Enumeration handle for the guard manager.
   */
  FuncDefaultsGuardAccessor(
      RootGuardManager* root,
      py::object name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            std::move(name),
            std::move(source),
            example_value,
            guard_manager_enum) {}

  /**
   * Check method for verifying if the provided object's defaults are guarded.
   * @param obj PyObject pointer to check (borrowed reference).
   * @return true if obj's defaults pass the guard check, false otherwise.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* func = obj;
    if (PyMethod_Check(obj)) {
      func = PyMethod_GET_FUNCTION(obj); // borrowed ref
    } else if (PyInstanceMethod_Check(obj)) {
      func = PyInstanceMethod_GET_FUNCTION(obj); // borrowed ref
    }
    // retrieve function defaults and perform guard check
    PyObject* x = PyFunction_GetDefaults(func); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return false;
    }
    return _guard_manager->check_nopybind(x);
  }

  /**
   * Verbose check method for detailed verification of function defaults.
   * @param obj PyObject pointer to check (borrowed reference).
   * @return GuardDebugInfo object containing detailed guard verification results.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* func = obj;
    if (PyMethod_Check(obj)) {
      func = PyMethod_GET_FUNCTION(obj); // borrowed ref
    }
    // retrieve function defaults and perform detailed guard check
    PyObject* x = PyFunction_GetDefaults(func); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return GuardDebugInfo(false, "function has no defaults", 0);
    }
    return _guard_manager->check_verbose_nopybind(x);
  }
};
    } else if (PyInstanceMethod_Check(obj)) {
      func = PyInstanceMethod_GET_FUNCTION(obj); // 如果 obj 是 PyInstanceMethod，获取其关联的函数对象（借用引用）
    }
    PyObject* x = PyFunction_GetDefaults(func); // 获取函数对象 func 的默认参数
    if (x == nullptr) {
      PyErr_Clear(); // 清除当前的 Python 异常状态
      // 如果函数没有默认参数，返回一个包含调试信息的 GuardDebugInfo 对象
      return GuardDebugInfo(
          false,
          std::string(repr() + ": Not a function on ") + get_source(),
          0);
    }

    // 调用 _guard_manager 对象的方法，检查并输出详细信息，不使用 Pybind
    return _guard_manager->check_verbose_nopybind(x);
  }

  std::string repr() const override {
    return "FuncDefaultsGuardAccessor"; // 返回类的字符串表示，表示为 FuncDefaultsGuardAccessor
  }
};

/**
 * Represents func.__kwdefaults__ accessor.
 */
class FuncKwDefaultsGuardAccessor : public GuardAccessor {
 public:
  FuncKwDefaultsGuardAccessor(
      RootGuardManager* root,
      py::object name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            std::move(name),
            std::move(source),
            example_value,
            guard_manager_enum) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  
  /**
   * Check if the given Python object has keyword defaults (func.__kwdefaults__).
   * This method overrides the base class method.
   * @param obj The Python object to check (borrowed reference).
   * @return True if keyword defaults exist, false otherwise.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* func = obj;
    if (PyMethod_Check(obj)) {
      func = PyMethod_GET_FUNCTION(obj); // borrowed ref
    } else if (PyInstanceMethod_Check(obj)) {
      func = PyInstanceMethod_GET_FUNCTION(obj); // borrowed ref
    }
    PyObject* x = PyFunction_GetKwDefaults(func); // borrowed ref
    if (x == nullptr) {
      PyErr_Clear();
      return false;
    }
    return _guard_manager->check_nopybind(x);
  }

  /**
   * Verbose check for keyword defaults (func.__kwdefaults__) in the given Python object.
   * This method provides additional debug information and overrides the base class method.
   * @param obj The Python object to check (borrowed reference).
   * @return GuardDebugInfo containing the result of the verbose check.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* func = obj;
    if (PyMethod_Check(obj)) {
      func = PyMethod_GET_FUNCTION(obj); // borrowed ref
    } else if (PyInstanceMethod_Check(obj)) {
      func = PyInstanceMethod_GET_FUNCTION(obj); // borrowed ref
    }
    PyObject* x = PyFunction_GetKwDefaults(func);
    if (x == nullptr) {
      PyErr_Clear();
      return GuardDebugInfo(
          false,
          std::string(repr() + ": Not a function on ") + get_source(),
          0);
    }

    return _guard_manager->check_verbose_nopybind(x);
  }

  /**
   * Return a string representation of this accessor.
   * @return String representation.
   */
  std::string repr() const override {
    return "FuncKwDefaultsGuardAccessor";
  }
};

/**
 * Represents f_globals accessor. This sits as a child accessor of the
 * RootGuardManager.
 */
class GlobalsGuardAccessor : public GuardAccessor {
 public:
  GlobalsGuardAccessor(
      RootGuardManager* root,
      py::dict globals_dict,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            globals_dict,
            std::move(source),
            example_value,
            guard_manager_enum),
        _globals_dict(globals_dict.ptr()) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.

  /**
   * Check if the given Python object has global variables (f_globals).
   * This method overrides the base class method.
   * @param obj The Python object to check (borrowed reference).
   * @return True if globals exist, false otherwise.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // Ignore the obj arg. This is required to satisfy the function signature.
    // Just pass on the globals dict to the child manager.
    return _guard_manager->check_nopybind(_globals_dict);
  }

  /**
   * Verbose check for global variables (f_globals) in the given Python object.
   * This method provides additional debug information and overrides the base class method.
   * @param obj The Python object to check (borrowed reference).
   * @return GuardDebugInfo containing the result of the verbose check.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // Ignore the obj arg. This is required to satisfy the function signature.
    // Just pass on the globals dict to the child manager.
    return _guard_manager->check_verbose_nopybind(_globals_dict);
  }

  /**
   * Return a string representation of this accessor.
   * @return String representation.
   */
  std::string repr() const override {
    return "GlobalsGuardAccessor";
  }


// 返回字符串 "GlobalsGuardAccessor"，结束函数执行并返回该字符串作为结果



 private:
  // no need of py::object here because the globals_dict is already passed on to
  // the base class as accessor_key which is a py::object.
  PyObject* _globals_dict;


// 下面是类的私有部分

// 不需要在这里使用 py::object，因为 globals_dict 已经作为 accessor_key 传递给了基类，它是一个 py::object 类型。
// 定义一个名为 _globals_dict 的 PyObject 指针，用于存储 Python 对象的全局字典
};

/**
 * Represent type(...) accessor.
 */
class TypeGuardAccessor : public GuardAccessor {
 public:
  // name = __type_accessor__, a unique string used as attribute name.
  TypeGuardAccessor(
      RootGuardManager* root,
      py::str name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            std::move(name),
            std::move(source),
            example_value,
            guard_manager_enum) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  // 检查对象是否符合类型保护，不包含Pybind的版本（借用引用）
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    PyObject* x = (PyObject*)Py_TYPE(obj); // borrowed ref
    return _guard_manager->check_nopybind(x);
  }

  // 检查对象是否符合类型保护，详细版本不包含Pybind（借用引用）
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    PyObject* x = (PyObject*)Py_TYPE(obj); // borrowed ref
    return _guard_manager->check_verbose_nopybind(x);
  }

  // 返回此类型访问器的字符串表示
  std::string repr() const override {
    return "TypeGuardAccessor";
  }
};

/**
 * Getitem tuple_iterator accessor.
 */
class TupleIteratorGetItemAccessor : public GuardAccessor {
 public:
  TupleIteratorGetItemAccessor(
      RootGuardManager* root,
      py::object index,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            index,
            std::move(source),
            example_value,
            guard_manager_enum),
        _index(py::cast<Py_ssize_t>(std::move(index))) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  // 检查对象是否符合类型保护，不包含Pybind的版本（借用引用）
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    _PyTupleIterObject* it = (_PyTupleIterObject*)obj;
    PyObject* x =
        PyTuple_GET_ITEM(it->it_seq, it->it_index + _index); // borrowed ref
    if (x == nullptr) {
      // 超出范围。
      PyErr_Clear();
      return false;
    }
    bool result = _guard_manager->check_nopybind(x);
    return result;
  }

  // 检查对象是否符合类型保护，详细版本不包含Pybind的版本（借用引用）
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    _PyTupleIterObject* it = (_PyTupleIterObject*)obj;
    PyObject* x =
        PyTuple_GET_ITEM(it->it_seq, it->it_index + _index); // borrowed ref
    if (x == nullptr) {
      // 超出范围。
      PyErr_Clear();
      return GuardDebugInfo(false, std::string("IndexError ") + repr(), 0);
    }
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    return result;
  }

  // 返回此元组迭代器访问器的字符串表示，包含索引信息
  std::string repr() const override {
    return "TupleIteratorGetItemAccessor(" + std::to_string(_index) + ")";
  }

 private:
  Py_ssize_t _index;
};

/**
 * GlobalWeakRef accessor. Dynamo can insert a weakref object into the frame
 * globals. This accessor reads the globals and then calls the weakref object
 * to get the underlying object. This is a child of GlobalsGuardAccessor.
 * Therefore, we will get the globals dict while caling check_nopybind.
 */
/**
 * Represents an accessor for managing weak references in a global context.
 */
class GlobalWeakRefGuardAccessor : public GuardAccessor {
 public:
  /**
   * Constructor for GlobalWeakRefGuardAccessor.
   * Initializes with necessary parameters including the global name, source,
   * example value, and guard manager enumeration.
   */
  GlobalWeakRefGuardAccessor(
      RootGuardManager* root,
      py::object global_name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            global_name,
            std::move(source),
            example_value,
            guard_manager_enum),
        _global_name(global_name.ptr()) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.

  /**
   * Checks if the provided PyObject is a valid weak reference.
   * @param obj PyObject to check, expected to be a globals dictionary.
   * @return True if the weak reference is valid, otherwise false.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // obj is globals dict because GlobalWeakRefGuardAccessor has to be a
    // child of GlobalsGuardAccessor.
    PyObject* weakref = PyDict_GetItem(obj, _global_name); // borrowed ref
    if (weakref == nullptr) {
      // The weakref is not in the globals dict.
      PyErr_Clear();
      return false;
    }

    if (!PyWeakref_Check(weakref)) {
      return false;
    }

    PyObject* x = PyWeakref_GetObject(weakref); // borrowed ref
    return _guard_manager->check_nopybind(x);
  }

  /**
   * Checks if the provided PyObject is a valid weak reference with verbose debug info.
   * @param obj PyObject to check, expected to be a globals dictionary.
   * @return GuardDebugInfo object containing detailed debug information.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // obj is globals dict because GlobalWeakRefGuardAccessor has to be a
    // child of GlobalsGuardAccessor.
    PyObject* weakref = PyDict_GetItem(obj, _global_name); // borrowed ref
    if (weakref == nullptr) {
      // The weakref is not in the globals dict.
      PyErr_Clear();
      return GuardDebugInfo(
          false, std::string("KeyError on ") + get_source(), 0);
    }

    if (!PyWeakref_Check(weakref)) {
      return GuardDebugInfo(
          false, std::string("Not a weakref ") + get_source(), 0);
    }

    PyObject* x = PyWeakref_GetObject(weakref); // borrowed ref
    return _guard_manager->check_verbose_nopybind(x);
  }

  /**
   * Returns a string representation of GlobalWeakRefGuardAccessor.
   * @return String representation of the object.
   */
  std::string repr() const override {
    return "GlobalWeakRefGuardAccessor(" +
        py::str(_global_name).cast<std::string>() + ")";
  }

 private:
  PyObject* _global_name; /**< Stores the global name as a PyObject pointer. */
};

/**
 * Implements guard accessor for weak reference call - x_weak().
 */
class WeakRefCallGuardAccessor : public GuardAccessor {
 public:
  /**
   * Constructor for WeakRefCallGuardAccessor.
   * Initializes with necessary parameters including the name, source,
   * example value, and guard manager enumeration.
   */
  WeakRefCallGuardAccessor(
      RootGuardManager* root,
      py::str name,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            std::move(name),
            std::move(source),
            example_value,
            guard_manager_enum) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.

  /**
   * Checks if the provided PyObject is a valid weak reference.
   * @param obj PyObject to check, expected to be a weak reference.
   * @return True if the weak reference is valid, otherwise false.
   */
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    if (!PyWeakref_Check(obj)) {
      return false;
    }

    PyObject* x = PyWeakref_GetObject(obj); // borrowed ref
    return _guard_manager->check_nopybind(x);
  }

  /**
   * Checks if the provided PyObject is a valid weak reference with verbose debug info.
   * @param obj PyObject to check, expected to be a weak reference.
   * @return GuardDebugInfo object containing detailed debug information.
   */
  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    if (!PyWeakref_Check(obj)) {
      return GuardDebugInfo(
          false, std::string("Not a weakref ") + get_source(), 0);
    }

    PyObject* x = PyWeakref_GetObject(obj); // borrowed ref
    return _guard_manager->check_verbose_nopybind(x);
  }
};
    // 检查 obj 是否不是弱引用对象，如果不是，则返回调试信息
    if (!PyWeakref_Check(obj)) {
      return GuardDebugInfo(
          false, std::string("Not a weakref obj ") + get_source(), 0);
    }

    // 从弱引用对象获取被引用的对象 x，这里返回的是一个借用引用（borrowed reference）
    PyObject* x = PyWeakref_GetObject(obj); // borrowed ref
    
    // 使用 _guard_manager 调用 check_verbose_nopybind 函数，传入 x，并返回结果
    return _guard_manager->check_verbose_nopybind(x);
  }

  // 返回描述对象的字符串表示，此处返回固定的字符串 "WeakRefCallGuardAccessor()"
  std::string repr() const override {
    return "WeakRefCallGuardAccessor()";
  }
/**
 * Similar to PythonLambdaLeafGuard, this class is a way to allow developers to
 * supply accessor as a python function. This is useful for from_numpy source.
 */
class PythonLambdaGuardAccessor : public GuardAccessor {
 public:
  PythonLambdaGuardAccessor(
      RootGuardManager* root,
      py::function accessor_fn,
      std::string source,
      py::handle example_value,
      py::handle guard_manager_enum)
      : GuardAccessor(
            root,
            accessor_fn,
            std::move(source),
            example_value,
            guard_manager_enum),
        _accessor_fn(std::move(accessor_fn)) {}

  // NB: Intentional duplication between check_nopybind and
  // check_verbose_nopybind.
  bool check_nopybind(PyObject* obj) override { // borrowed ref
    // Call the Python accessor function with obj as an argument
    PyObject* x = PyObject_CallOneArg(_accessor_fn.ptr(), obj); // new ref
    if (x == nullptr) {
      // The accessor function failed. Clear the Python error state and return false
      PyErr_Clear();
      return false;
    }
    // Check the result using the GuardManager's check_nopybind method
    bool result = _guard_manager->check_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  GuardDebugInfo check_verbose_nopybind(
      PyObject* obj) override { // borrowed ref
    // Call the Python accessor function with obj as an argument
    PyObject* x = PyObject_CallOneArg(_accessor_fn.ptr(), obj); // new ref
    if (x == nullptr) {
      // The accessor function failed. Get the exception message, clear the Python error state, and return false
      std::string exc_message = get_exception_message();
      PyErr_Clear();
      return GuardDebugInfo(false, exc_message, 0);
    }
    // Check the result using the GuardManager's check_verbose_nopybind method
    GuardDebugInfo result = _guard_manager->check_verbose_nopybind(x);
    Py_DECREF(x);
    return result;
  }

  std::string repr() const override {
    return "PythonLambdaGuardAccessor";
  }

 private:
  py::object _accessor_fn; // The Python function object used as an accessor
};

/**
 * Adds a relational guard between two GuardManager instances for tensor aliasing.
 * The guard is registered with both GuardManagers.
 */
void install_tensor_aliasing_guard(
    GuardManager* x,
    GuardManager* y,
    py::object verbose_code_parts) {
  // Create a relational guard for tensor aliasing with verbose code parts
  std::shared_ptr<RelationalGuard> guard =
      std::make_shared<TENSOR_ALIASING>(std::move(verbose_code_parts));

  // Register the relational guard resetter with the root guard manager of GuardManager x
  x->get_root()->add_relational_guard_resetter(guard);

  // Add the relational guard to GuardManager x and GuardManager y
  x->add_leaf_guard(guard);
  y->add_leaf_guard(guard);
}

/**
 * Installs a guard to prevent tensor aliasing.
 * Multiple GuardManager instances and tensor names are passed as arguments.
 */
void install_no_tensor_aliasing_guard(
    const py::list& guard_managers,
    const py::list& tensor_names,
    ...
  // 创建一个关系型保护对象，用于检查张量是否有别名。这是一个关系型保护的示例。
  // 有一个保护对象被多个保护管理器共享。
  std::shared_ptr<RelationalGuard> guard = std::make_shared<NO_TENSOR_ALIASING>(
      tensor_names, std::move(verbose_code_parts));

  // 在根保护管理器上注册重置器，以便在保护评估失败时重置新添加的关系型保护。
  py::cast<GuardManager*>(guard_managers[0])
      ->get_root()
      ->add_relational_guard_resetter(guard);

  // 遍历所有的保护管理器，并将关系型保护对象添加为叶子保护。
  for (const auto& guard_manager : guard_managers) {
    py::cast<GuardManager*>(guard_manager)->add_leaf_guard(guard);
  }
`
// 结束 C++ 的命名空间
}

} // namespace

// 定义一个静态函数，用于从 PyObject 中获取 tensor 的数据指针
static void* _torchinductor_pyobject_tensor_data_ptr(PyObject* obj) {
  // 检查输入是否为 nullptr 或者不是 THPVariable 类型的对象，如果不是则抛出异常
  if (C10_UNLIKELY(
          obj == nullptr ||
          (!THPVariable_CheckExact(obj) && !THPVariable_Check(obj)))) {
    throw std::runtime_error(
        "_torchinductor_pyobject_tensor_data_ptr: non-tensor input");
  }
  // 获取 THPVariable 对象的数据指针并返回
  return THPVariable_Unpack(obj).data_ptr();
}

// 将 Python 对象转换为 RootGuardManager 指针
void* convert_to_root_guard_manager(py::object root) {
  // 使用 move 将 root 转移为 RootGuardManager 指针
  RootGuardManager* root_mgr = std::move(root).cast<RootGuardManager*>();
  return (void*)root_mgr;
}

// 运行 RootGuardManager 的检查函数，传入 f_locals 参数
bool run_root_guard_manager(void* root, PyObject* f_locals) {
  return ((RootGuardManager*)root)->check_nopybind(f_locals);
}

// 初始化 torch._C._dynamo.guards.TensorGuards 类型
PyObject* torch_c_dynamo_guards_init() {
  // 设置 TensorGuardsType 的基本信息和方法
  TensorGuardsType.tp_name = "torch._C._dynamo.guards.TensorGuards";
  TensorGuardsType.tp_basicsize = sizeof(TensorGuards);
  TensorGuardsType.tp_itemsize = 0;
  TensorGuardsType.tp_dealloc = (destructor)TensorGuards_dealloc;
  TensorGuardsType.tp_flags = Py_TPFLAGS_DEFAULT;
  TensorGuardsType.tp_doc = "Check properties of a torch.Tensor";
  TensorGuardsType.tp_methods = TensorGuards_methods;
  TensorGuardsType.tp_init = (initproc)TensorGuards_init;
  TensorGuardsType.tp_new = TensorGuards_new;

  // 检查类型是否准备好，如果不是则返回 nullptr
  if (PyType_Ready(&TensorGuardsType) < 0)
    return nullptr;

  // 初始化 GlobalStateGuardType 类型
  GlobalStateGuardType.tp_name = "torch._C._dynamo.guards.GlobalStateGuard";
  GlobalStateGuardType.tp_basicsize = sizeof(GlobalStateGuard);
  GlobalStateGuardType.tp_itemsize = 0;
  GlobalStateGuardType.tp_flags = Py_TPFLAGS_DEFAULT;
  GlobalStateGuardType.tp_doc = "Guard on PyTorch global flags such as no_grad";
  GlobalStateGuardType.tp_methods = GlobalStateGuard_methods;
  GlobalStateGuardType.tp_init = (initproc)GlobalStateGuard_init;
  GlobalStateGuardType.tp_new = PyType_GenericNew;

  // 检查类型是否准备好，如果不是则返回 nullptr
  if (PyType_Ready(&GlobalStateGuardType) < 0)
    return nullptr;

  // 创建一个 Python 模块对象
  auto m = PyModule_Create(&_module);
  if (m == nullptr)
    return nullptr;

  // 增加 TensorGuardsType 到模块中
  Py_INCREF(&TensorGuardsType);
  if (PyModule_AddObject(m, "TensorGuards", (PyObject*)&TensorGuardsType) < 0) {
    Py_DECREF(&TensorGuardsType);
    Py_DECREF(m);
    return nullptr;
  }

  // 增加 GlobalStateGuardType 到模块中
  Py_INCREF(&GlobalStateGuardType);
  if (PyModule_AddObject(
          m, "GlobalStateGuard", (PyObject*)&GlobalStateGuardType) < 0) {
    Py_DECREF(&GlobalStateGuardType);
    Py_DECREF(m);
    return nullptr;
  }

  // 将 _torchinductor_pyobject_tensor_data_ptr 函数的地址作为 PyLong 对象添加到模块中
  if (PyModule_AddObject(
          m,
          "_torchinductor_pyobject_tensor_data_ptr",
          PyLong_FromVoidPtr(reinterpret_cast<void*>(
              &_torchinductor_pyobject_tensor_data_ptr))) < 0) {
    return nullptr;
  }

  // 返回创建的模
#if IS_PYTHON_3_12_PLUS
    // 如果 Python 版本是 3.12 或更高
    // 添加字典版本监视器，用指定的回调函数 dict_version_watch_callback
    dict_version_watcher_id = PyDict_AddWatcher(dict_version_watch_callback);
    // 检查是否添加监视器失败
    if (dict_version_watcher_id == -1) {
        // 抛出运行时错误，提示安装 dict_version_watch_callback 失败
        throw std::runtime_error("Failed to install dict_version_watch_callback");
    }
#endif

// 返回 m 变量，完成函数执行
return m;
}

} // namespace torch::dynamo
```