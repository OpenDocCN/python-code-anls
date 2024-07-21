# `.\pytorch\torch\csrc\autograd\python_variable_indexing.cpp`

```py
// 引入 Torch 的头文件：变量索引的 Python 接口
#include <torch/csrc/autograd/python_variable_indexing.h>

// 引入 Torch 的其他头文件
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/tensor_types.h>

// 引入 ATen 库的头文件
#include <ATen/DeviceGuard.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TracerMode.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>

// 引入 ATen 库的布局头文件
#include <c10/core/Layout.h>

// 使用 at 命名空间
using namespace at;
// 使用 torch::autograd::utils 命名空间
using namespace torch::autograd::utils;

// 进入 torch::autograd 命名空间
namespace torch::autograd {

// 定义 THPVariable_length 函数，返回 PyObject 对象的长度
Py_ssize_t THPVariable_length(PyObject* self) {
  HANDLE_TH_ERRORS
  // 检查是否存在 Torch 函数 "__len__"
  if (check_has_torch_function(self)) {
    // 调用 Torch 函数 "__len__" 获取返回对象
    py::object ret = py::reinterpret_steal<py::object>(
        handle_torch_function(self, "__len__"));
    // 将返回对象转换为 Py_ssize_t 类型作为长度
    Py_ssize_t length = PyLong_AsSsize_t(ret.ptr());
    // 如果发生错误，抛出 Python 错误
    if (PyErr_Occurred()) {
      throw python_error();
    }
    // 返回长度
    return length;
  }
  // 解包 THPVariable 对象
  const auto& self_ = THPVariable_Unpack(self);
  // 如果 self_ 的维度为 0，则返回 0
  if (self_.dim() == 0) {
    return 0;
  }
  // 否则返回 self_ 的第一个维度的符号大小，使用 guard_int 进行保护
  // TODO: 或许应该直接返回 SymInt？
  return (Py_ssize_t)self_.sym_size(0).guard_int(__FILE__, __LINE__);
  END_HANDLE_TH_ERRORS_RET(-1)
}

// 允许使用整数、切片、省略号、None、Variables 和这些类型的元组进行索引
static inline int64_t count_specified_dimensions(PyObject* index) {
  // 统计指定的索引维度数（除了省略号和 None）
  // -1 用于 __torch_function__ 的标志
  int64_t count = 0;
  // 获取元组的大小
  auto size =
      PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  // 遍历元组中的每个索引项
  for (Py_ssize_t i = 0; i < size; i++) {
    // 获取索引项
    PyObject* obj = PyTuple_GET_ITEM(
        index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    // 检查是否存在 Torch 函数
    if (check_has_torch_function(obj))
      return -1;
    // 如果是 THPVariable 对象
    if (THPVariable_Check(obj)) {
      // 解包 THPVariable 对象
      const auto& var = THPVariable_Unpack(obj);
      // 获取变量的标量类型
      const auto& var_scalar_type = var.scalar_type();
      // 如果是 kByte 或 kBool 类型，增加维度数
      if (var_scalar_type == kByte || var_scalar_type == kBool) {
        count += var.dim();
      } else {
        count++;
      }
    } else if (
        obj != Py_None && obj != Py_Ellipsis && obj != Py_True &&
        obj != Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
      // 如果不是 None、省略号、True 或 False，增加维度数
      count++;
    }
  }
  // 返回统计的维度数
  return count;
}
// 当传入对象不合法时抛出异常，显示错误信息
[[noreturn]] static inline void invalid_index(PyObject* obj) {
  // 使用 TORCH_CHECK_INDEX 宏来验证条件，如果条件为假，则抛出异常
  TORCH_CHECK_INDEX(
      false,
      "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
      "Variables are valid indices (got ",
      // 获取传入对象的类型名称，并显示在错误信息中
      Py_TYPE(obj)->tp_name,
      ")");
}

// 将 Python 序列转换为 Torch 的 Variable 对象
static inline Variable sequenceToVariable(
    c10::TensorOptions options,
    PyObject* seq) {
  // 调用 torch::utils::indexing_tensor_from_data 函数，从 Python 序列中创建索引张量
  return torch::utils::indexing_tensor_from_data(
      options, kLong, c10::nullopt, seq);
}

// 将 Python 对象转换为 Torch 的 Tensor 对象
inline Variable valueToTensor(
    c10::TensorOptions options,
    PyObject* value,
    const at::Device& device) {
  // 如果传入对象是 THPVariable 类型，则解包并返回对应的 Torch 变量
  if (THPVariable_Check(value)) {
    return THPVariable_Unpack(value);
  }
  // 对 ATen 操作下的自动调度进行保护
  at::AutoDispatchBelowADInplaceOrView guard; // TODO: remove
  // 禁用 Torch 的追踪调度模式
  at::tracer::impl::NoTracerDispatchMode tracer_guard;
  // 根据传入对象的类型进行类型转换并返回相应的 Scalar 标量对象
  Scalar scalar;
  if (THPUtils_checkLong(value) || PyBool_Check(value)) {
    scalar = Scalar(THPUtils_unpackLong(value));
  } else if (PyFloat_Check(value)) {
    scalar = Scalar(THPUtils_unpackDouble(value));
  } else if (PyComplex_Check(value)) {
    scalar = Scalar(THPUtils_unpackComplexDouble(value));
  } else if (torch::is_symint(value)) {
    scalar = Scalar(py::cast<c10::SymInt>(py::handle(value)));
  } else if (torch::is_symfloat(value)) {
    scalar = Scalar(py::cast<c10::SymFloat>(py::handle(value)));
  } else if (torch::is_symbool(value)) {
    scalar = Scalar(py::cast<c10::SymBool>(py::handle(value)));
  } else {
    // 抛出类型错误异常，显示无法将给定类型分配给目标类型的消息
    throw TypeError(
        "can't assign a %s to a %s",
        Py_TYPE(value)->tp_name,
        torch::utils::options_to_string(options).c_str());
  }
  // 如果设备是 CPU 且标量不是符号的，则使用 at::lift_fresh 函数进行处理
  if (device == at::kCPU && !scalar.isSymbolic()) {
    // 对 CPU 设备上的标量进行提升处理，并返回索引化的标量对应的 Tensor
    return at::lift_fresh(
        at::indexing::scalarToTensor(scalar, options, device));
  } else {
    // 返回索引化的标量对应的 Tensor
    return at::indexing::scalarToTensor(scalar, options, device);
  }
}

// 记录切片操作的追踪信息
static inline void recordSliceTrace(PyObject* obj) {
  // 将传入的 Python 切片对象转换为 PySliceObject 类型
  PySliceObject* sliceobj = (PySliceObject*)obj;
  // 如果切片对象的起始值是 THPVariable 类型，则记录起始值的追踪信息
  if (THPVariable_Check(sliceobj->start)) {
    torch::jit::tracer::ArgumentStash::stashValue(
        std::string("start"),
        1,
        THPVariable_Unpack(sliceobj->start),
        torch::jit::IntType::get());
  }
  // 如果切片对象的停止值是 THPVariable 类型，则记录停止值的追踪信息
  if (THPVariable_Check(sliceobj->stop)) {
    torch::jit::tracer::ArgumentStash::stashValue(
        std::string("end"),
        1,
        THPVariable_Unpack(sliceobj->stop),
        torch::jit::IntType::get());
  }
  // 如果切片对象的步长值是 THPVariable 类型，则记录步长值的追踪信息
  if (THPVariable_Check(sliceobj->step)) {
    torch::jit::tracer::ArgumentStash::stashValue(
        std::string("step"),
        1,
        THPVariable_Unpack(sliceobj->step),
        torch::jit::IntType::get());
  }
}

// 记录选择操作的追踪信息
static inline void recordSelectTrace(const Tensor& index_tensor) {
  // 记录索引张量的追踪信息，用于 JIT 编译和追踪
  torch::jit::tracer::ArgumentStash::stashValue(
      std::string("index"), 1, index_tensor, torch::jit::IntType::get());
}

// 执行切片操作并返回结果
static inline Variable applySlicing(
    const Variable& self,
    PyObject* index,
    variable_list& outIndices,  // 引用类型的变量列表，用于输出索引
    bool is_tracing,  // 布尔类型，表示是否在追踪状态
    const at::Device& self_device,  // 引用类型的设备对象，表示当前张量的设备
    const std::optional<int64_t>& self_ndim,  // 可选的 int64_t 类型，表示当前张量的维度数
    int64_t specified_dims) {  // int64_t 类型，表示指定的索引维度数
  int64_t size =
      PyTuple_GET_SIZE(index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  int64_t dim = 0;

  // See NOTE [nested tensor size for indexing]
  // 如果 self_ndim 有值，则检查指定的维度数是否不超过 self_ndim，否则抛出异常
  if (self_ndim.has_value()) {
    TORCH_CHECK_INDEX(
        specified_dims <= self_ndim.value(),
        "too many indices for tensor of dimension ",
        self_ndim.value());
  }

  Variable result = self;
  // 遍历索引中的每个元素
  for (const auto i : c10::irange(size)) {
    PyObject* obj = PyTuple_GET_ITEM(
        index, i); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    // NOTE [nested tensor size for indexing]
    // 嵌套张量目前没有大小，因此我们暂时将其大小表示为 null
    // 在解决嵌套张量大小问题后，可能需要修改这里的逻辑
    // 如果 result 是嵌套张量，则 result_sizes 设为 null，否则设为 result 的符号化大小
    std::optional<SymIntArrayRef> result_sizes = result.is_nested()
        ? std::optional<SymIntArrayRef>(c10::nullopt)
        : std::optional<SymIntArrayRef>(result.sym_sizes());


这段代码是一个函数的参数列表和部分函数体，注释详细解释了每个参数的类型和含义，以及函数体中每一步骤的作用和相关注释说明。
    result = at::indexing::handleDimInMultiDimIndexing(
        /*prev_dim_result=*/result,
        /*original_tensor=*/self,
        /*index=*/([&]() {
          // 如果 obj 是长整型，则转换为 TensorIndex 对象
          if (THPUtils_checkLong(obj)) {
            // 如果在跟踪模式下且 obj 是 THPVariable 类型，则记录选择操作的跟踪信息
            if (is_tracing && THPVariable_Check(obj)) {
              recordSelectTrace(THPVariable_Unpack(obj));
            }
            return at::indexing::TensorIndex(THPUtils_unpackLong(obj));
          } 
          // 如果 obj 是 Python 切片对象
          else if (PySlice_Check(obj)) {
            // 解析 Python 切片对象
            auto val = __PySlice_Unpack(obj);
            // 如果在跟踪模式下，则记录切片操作的跟踪信息
            if (is_tracing) {
              recordSliceTrace(obj);
            }
            // 返回对应的 TensorIndex 对象，表示切片索引
            return at::indexing::TensorIndex(
                at::indexing::Slice(val.start, val.stop, val.step));
          } 
          // 如果 obj 是省略号（Ellipsis）
          else if (obj == Py_Ellipsis) {
            // 返回对应的 TensorIndex 对象，表示省略号索引
            return at::indexing::TensorIndex(at::indexing::Ellipsis);
          } 
          // 如果 obj 是 None
          else if (obj == Py_None) {
            // 返回对应的 TensorIndex 对象，表示 None 索引
            return at::indexing::TensorIndex(at::indexing::None);
          } 
          // 如果 obj 是布尔值
          else if (PyBool_Check(obj)) {
            // 返回对应的 TensorIndex 对象，表示布尔值索引
            return at::indexing::TensorIndex(obj == Py_True);
          } 
          // 如果 obj 是 THPVariable 类型
          else if (THPVariable_Check(obj)) {
            // 解包成 Tensor 对象
            Tensor tensor = THPVariable_Unpack(obj);
            // 如果在跟踪模式下，且 Tensor 是标量且满足条件，则记录选择操作的跟踪信息
            if (is_tracing) {
              auto scalar_type = tensor.scalar_type();
              if (tensor.dim() == 0 &&
                  at::isIntegralType(scalar_type, /*includeBool=*/false) &&
                  scalar_type != at::kByte) {
                recordSelectTrace(tensor);
              }
            }
            // 返回对应的 TensorIndex 对象，表示 Tensor 索引
            return at::indexing::TensorIndex(std::move(tensor));
          } 
          // 如果 obj 是 Python 序列对象
          else if (PySequence_Check(obj)) {
            // 将 Python 序列对象转换为 Variable 并返回对应的 TensorIndex 对象
            return at::indexing::TensorIndex(
                sequenceToVariable(self.options(), obj));
          } 
          // 如果 obj 不属于以上任何类型，则尝试将其解析为长整型索引
          else {
            auto idx = THPObjectPtr(PyNumber_Index(obj));
            // 如果无法解析为长整型索引，则清除错误并抛出无效索引异常
            if (!idx) {
              PyErr_Clear();
              invalid_index(obj);
            }
            // 如果在跟踪模式下且 idx 是 THPVariable 类型，则记录选择操作的跟踪信息
            if (is_tracing && THPVariable_Check(idx)) {
              recordSelectTrace(THPVariable_Unpack(idx));
            }
            // 返回对应的 TensorIndex 对象，表示长整型索引
            return at::indexing::TensorIndex(THPUtils_unpackLong(idx));
          }
        })(),
        /*dim_ptr=*/&dim,
        /*specified_dims_ptr=*/&specified_dims,
        /*real_dim=*/i,
        /*outIndices=*/outIndices,
        // 参见注释 [ 设置 `disable_slice_optimization` 在从 Python 调用 C++ 张量索引函数时 ]
        /*disable_slice_optimization=*/is_tracing,
        /*original_tensor_device=*/self_device,
        /*prev_dim_result_sizes=*/result_sizes);
  }
  // 返回处理后的结果
  return result;
// 如果输入的索引是元组，则返回 true
static inline bool treatSequenceAsTuple(PyObject* index) {
  // 检查索引是否为元组类型
  if (PyTuple_Check(index)) {
    return true;
  }
  // 如果索引是 THPVariable 类型，则返回 false
  if (THPVariable_Check(index)) {
    return false;
  }
  // 当使用 ndarray 作为索引时，如果 numpy 编译已启用，不应将其视为元组，因为其语法不同
#ifdef USE_NUMPY
  if (::torch::utils::is_numpy_available() && PyArray_CheckExact(index)) {
    return false;
  }
#endif
  // 如果索引不是序列类型，则返回 false
  if (!PySequence_Check(index)) {
    return false;
  }
  // 使用 NumPy 中的启发式方法判断是否将非元组序列视为元组
  // 根据 NumPy 代码注释，对于短序列，会采取与标量相同的处理方式，否则会视为索引列表
  auto n = PySequence_Size(index);
  // 如果获取序列大小失败，返回 false（通常是由于 Python 错误）
  if (n < 0) {
    PyErr_Clear();
    return false;
  }
  // 当序列长度大于等于 32 时，返回 false
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  if (n >= 32) {
    return false;
  }
  // 遍历序列中的每一项，判断是否应将其视为元组
  for (Py_ssize_t i = 0; i < n; i++) {
    auto obj = THPObjectPtr{PySequence_GetItem(index, i)};
    // 如果获取序列项失败，返回 false
    if (!obj.get()) {
      PyErr_Clear();
      return false;
    }
    // 如果序列项是 THPVariable、序列类型或者切片对象，则返回 true
    if (THPVariable_Check(obj.get()) || PySequence_Check(obj.get()) ||
        PySlice_Check(obj.get())) {
      return true;
    }
    // 如果序列项是省略号或者 None，则返回 true
    if (obj.get() == Py_Ellipsis || obj.get() == Py_None) {
      return true;
    }
  }
  // 默认返回 false
  return false;
}

// 将输入的索引对象包装成 Python 元组并返回
static inline THPObjectPtr wrapTuple(PyObject* index) {
  // 创建一个空的 THPObjectPtr 对象
  THPObjectPtr res;
  // 如果应将输入的索引对象视为元组，则使用 PySequence_Tuple 函数将其转换为元组
  if (treatSequenceAsTuple(index)) {
    res = PySequence_Tuple(index);
  } else {
    // 否则，使用 PyTuple_Pack 函数将其打包成元组（通常情况下为单个元素的元组）
    res = PyTuple_Pack(
        1, index); // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
  }
  // 如果转换或打包过程中出现错误，则抛出 python_error 异常
  if (!res)
    throw python_error();
  // 返回包装后的结果
  return res;
}

// 注意事项：这里是 `THPVariable_getitem` 的调度结构：

// 1. 对于 Python 1-D getter，将 Python 索引转换为 C++ 的 TensorIndex 后调用 `at::indexing::get_item`。
// 2. 对于 Python N-D getter，将 Python 索引转换为 C++ 的 TensorIndex 后，对每个维度调用 `at::indexing::handleDimInMultiDimIndexing`。
//    如果需要高级索引，则调用 `at::indexing::dispatch_index`。
PyObject* THPVariable_getitem(PyObject* self, PyObject* index) {
  HANDLE_TH_ERRORS
  // 如果 self 有 torch 函数定义，调用处理 torch 函数索引逻辑
  if (check_has_torch_function(self)) {
    return handle_torch_function_indexing(self, index);
  }
  // 解包 self_，获取对应的 THPVariable
  const auto& self_ = THPVariable_Unpack(self);
  // 根据 self_ 的设备类型设置设备保护
  OptionalDeviceGuard device_guard(device_of(self_));

  // 处理简单类型的索引：None 和 ellipsis
  if (index == Py_None) {
    return THPVariable_Wrap(at::indexing::get_item(
        self_, {at::indexing::TensorIndex(at::indexing::None)}));
  } else if (index == Py_Ellipsis) {
    // 返回包装后的 THPVariable，其中包含使用 get_item 函数获取的元素
    return THPVariable_Wrap(at::indexing::get_item(
        self_, {at::indexing::TensorIndex(at::indexing::Ellipsis)}));
  }

  // 检查是否正在进行追踪
  bool is_tracing = torch::jit::tracer::isTracing();

  // 处理简单类型：整数、切片、布尔值
  if (THPUtils_checkLong(index)) {
    // 如果正在追踪，并且 index 是 THPVariable 类型，则记录选择的追踪
    if (is_tracing && THPVariable_Check(index)) {
      recordSelectTrace(THPVariable_Unpack(index));
    }
    // 返回包装后的 THPVariable，其中包含使用 get_item 函数获取的元素
    return THPVariable_Wrap(at::indexing::get_item(
        self_, {at::indexing::TensorIndex(THPUtils_unpackLong(index))}));
  } else if (PySlice_Check(index)) {
    // 解包 PySlice 对象
    auto val = __PySlice_Unpack(index);
    // 如果正在追踪，则记录切片追踪
    if (is_tracing) {
      recordSliceTrace(index);
    }
    // 返回包装后的 THPVariable，其中包含使用 get_item 函数获取的元素
    return THPVariable_Wrap(at::indexing::get_item(
        self_,
        {at::indexing::TensorIndex(
            at::indexing::Slice(val.start, val.stop, val.step))}));
  } else if (index == Py_False || index == Py_True) {
    // 返回包装后的 THPVariable，其中包含使用 get_item 函数获取的元素
    return THPVariable_Wrap(([&]() {
      pybind11::gil_scoped_release no_gil;
      return at::indexing::get_item(
          self_, {at::indexing::TensorIndex(index == Py_True)});
    })());
  }

  // 如果 index 不是元组，则将其包装在一个元组中
  THPObjectPtr holder = wrapTuple(index);

  // 创建一个变量列表来保存索引
  variable_list variableIndices;
  // 计算指定维度的数量
  int64_t specified_dims = count_specified_dimensions(holder.get());
  // 如果没有指定维度，则调用处理 Torch 函数索引
  if (specified_dims == -1) {
    return handle_torch_function_indexing(self, holder.get());
  }
  // 应用切片操作，并返回切片后的变量
  Variable sliced = applySlicing(
      self_,
      holder.get(),
      variableIndices,
      /*is_tracing=*/is_tracing,
      self_.device(),
      self_.ndimension(),
      specified_dims);
  // 如果变量索引为空
  if (variableIndices.empty()) {
    // 如果切片和原始 self_ 相同，则返回切片的浅拷贝
    if (sliced.is_same(self_)) {
      sliced = at::alias(sliced);
    }
    // 返回包装后的 THPVariable，其中包含切片后的变量
    return THPVariable_Wrap(std::move(sliced));
  }

  // 对张量进行高级索引
  // 返回包装后的 THPVariable，其中包含使用 dispatch_index 函数获取的元素
  return THPVariable_Wrap(([&]() {
    pybind11::gil_scoped_release no_gil;
    return at::indexing::dispatch_index(sliced, std::move(variableIndices));
  })());

  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
// 结束函数 `THPVariable_setitem` 的定义
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  // 处理 Torch 错误，包装错误处理
  HANDLE_TH_ERRORS
  // 检查是否尝试删除项
  if (py_value == nullptr) {
    throw TypeError("Tensor does not support deleting items");
  }
  // 检查是否需要调用 Torch 函数处理索引或值
  if ((check_has_torch_function(self)) ||
      (check_has_torch_function(py_value))) {
    // 重新解释调用 Torch 函数处理索引和值
    py::object ret = py::reinterpret_steal<py::object>(
        handle_torch_function_indexing(self, index, py_value));
    return 0;  // 返回成功标志
  }

  // 解包 Tensor 对象
  const auto& self_ = THPVariable_Unpack(self);
  // 如果是稀疏张量，抛出类型错误
  if (self_.layout() == kSparse || self_.layout() == kSparseCsr ||
      self_.layout() == kSparseCsc || self_.layout() == kSparseBsr ||
      self_.layout() == kSparseBsc) {
    throw TypeError("Cannot assign to a sparse tensor");
  }
  // 设置设备保护，确保操作在正确设备上进行
  OptionalDeviceGuard device_guard(device_of(self_));
  at::Device self_device = self_.device();
  Variable value;
  // 处理 qint 类型的特殊情况
  if (isQIntType(self_.scalar_type())) {
    value =
        valueToTensor(device(kCPU).dtype(kFloat), py_value, at::Device(kCPU));
  } else if (self_device.is_cuda()) {
    value = valueToTensor(self_.options(), py_value, at::Device(kCPU));
  } else {
    value = valueToTensor(self_.options(), py_value, self_device);
  }

  // 处理简单类型：ellipsis, none, bool
  if (index == Py_False) { // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
    // 对于 false 不执行任何操作
    return 0;
  } else if (index == Py_Ellipsis) {
    // 处理省略号索引情况
    dispatch_set_item(
        self_, {at::indexing::TensorIndex(at::indexing::Ellipsis)}, value);
    return 0;
  } else if (index == Py_None) {
    // 处理 None 索引情况
    dispatch_set_item(
        self_, {at::indexing::TensorIndex(at::indexing::None)}, value);
    return 0;
  } else if (index == Py_True) {
    // 处理 true 索引情况
    dispatch_set_item(self_, {at::indexing::TensorIndex(true)}, value);
    return 0;
  }

  bool is_tracing = torch::jit::tracer::isTracing();

  // 处理简单类型：整数，切片
  if (THPUtils_checkLong(index) || torch::is_symint(index)) {
    if (is_tracing && THPVariable_Check(index)) {
      // 如果正在追踪，记录选择的追踪
      recordSelectTrace(THPVariable_Unpack(index));
    }
    auto symint = torch::is_symint(index) ? py::cast<SymInt>(index)
                                          : SymInt(THPUtils_unpackLong(index));
    // 使用 dispatch_set_item 方法设置 self_ 对象的元素，使用 TensorIndex(symint) 作为索引
    dispatch_set_item(self_, {at::indexing::TensorIndex(symint)}, value);
    // 返回操作成功的标志
    return 0;
  } else if (PySlice_Check(index)) {
    // 如果 index 是 PySlice 对象
    // 解包 PySlice 对象，获取其起始、终止和步长信息
    auto val = __PySlice_Unpack(index);
    // 如果处于追踪状态，记录切片追踪信息
    if (is_tracing) {
      recordSliceTrace(index);
    }
    // 设置索引，创建 Slice 对象，并设置是否禁用切片优化
    // 参见 NOTE [ Setting `disable_slice_optimization` when calling C++ tensor
    // indexing functions from Python ]
    dispatch_set_item(
        self_,
        {at::indexing::TensorIndex(
            at::indexing::Slice(val.start, val.stop, val.step))},
        value,
        /*disable_slice_optimization=*/is_tracing);
    // 返回操作成功的标志
    return 0;
  }

  // 如果 index 不是 PySlice 对象，则将其封装成元组
  THPObjectPtr holder = wrapTuple(index);

  // 创建变量列表来存储变量索引
  variable_list variableIndices;
  // 计算指定维度的数量
  int64_t specified_dims = count_specified_dimensions(holder.get());
  // 如果指定维度为 -1，调用 torch 函数处理索引操作，并返回操作成功的标志
  if (specified_dims == -1) {
    py::object val = py::reinterpret_steal<py::object>(
        handle_torch_function_indexing(self, index, py_value));
    return 0;
  }
  // 应用切片操作，返回切片后的 Variable 对象
  Variable sliced = applySlicing(
      self_,
      holder.get(),
      variableIndices,
      /*is_tracing=*/is_tracing,
      self_device,
      self_.ndimension(),
      specified_dims);
  // 如果变量索引列表为空
  if (variableIndices.empty()) {
    // 释放 GIL
    pybind11::gil_scoped_release no_gil;
    // 将值复制到切片对象中
    at::indexing::copy_to(sliced, value);
    // 返回操作成功的标志
    return 0;
  }

  // 如果变量索引列表不为空
  {
    // 释放 GIL
    pybind11::gil_scoped_release no_gil;
    // 获取值的符号大小信息
    SymIntArrayRef valueSizes = value.sym_sizes();
    // 计算切片后的值的符号大小信息
    SymIntArrayRef slicedValueSizes =
        at::indexing::slicePrefix1sSize(valueSizes);
    // 创建 Torch 自动求导变量 valuesSliced
    torch::autograd::Variable valuesSliced;
    // 如果值的大小信息不同，则重新视图为相应大小的值
    if (!valueSizes.equals(slicedValueSizes)) {
      valuesSliced = value.view_symint(slicedValueSizes);
    } else {
      valuesSliced = value;
    }
    // 调用索引设置函数，将切片后的值放入 sliced 中
    at::indexing::dispatch_index_put_(
        sliced, std::move(variableIndices), valuesSliced);
    // 返回操作成功的标志
    return 0;
  }
  // 处理 Torch 错误并返回错误标志
  END_HANDLE_TH_ERRORS_RET(-1)
}

} // namespace torch::autograd
```