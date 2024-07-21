# `.\pytorch\torch\csrc\autograd\python_torch_functions_manual.cpp`

```py
// 包含 Torch C++ API 中的头文件，用于定义数据类型、异常处理、自动微分等功能
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_torch_functions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/out_types.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/structseq.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>

// 包含 ATen 库的头文件，提供张量操作和功能性张量包装器
#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>

// 包含 Python C API 的头文件，支持 Python 交互和异常处理
#include <Python.h>

// 包含 fmt 库的头文件，用于格式化字符串输出
#include <fmt/format.h>

// 包含 pybind11 库的头文件，支持 C++ 和 Python 的无缝绑定
#include <pybind11/pybind11.h>

// 包含标准库的头文件，提供向量容器和实用工具
#include <utility>
#include <vector>

// 使用 ATen 命名空间中定义的符号，简化张量操作的代码书写
using at::DeviceGuard;
using at::DimnameList;
using at::IntArrayRef;
using at::OptionalDeviceGuard;
using at::Scalar;
using at::Tensor;
using at::TensorList;
using at::TensorOptions;

// 使用 torch::utils 命名空间中定义的符号，支持类型检查和工具函数调用
using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

// Torch 的 autograd 命名空间，提供自动微分相关的函数和类
namespace torch {
namespace autograd {

// 全局变量，指向 THPVariableFunctionsModule 对象，用于定义变量相关的 Python 函数
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyObject* THPVariableFunctionsModule = nullptr;

// 内联函数，执行 range 操作并分发到相应的实现
inline Tensor dispatch_range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor result) {
  // 释放 GIL（全局解释器锁），允许其他线程执行 Python 代码
  pybind11::gil_scoped_release no_gil;
  // 指定结果张量的设备，并保护其设备环境
  OptionalDeviceGuard device_guard(device_of(result));
  // 调用 ATen 库的 range_out 函数，生成指定范围内的张量
  return at::range_out(result, start, end, step);
}

// 内联函数，根据选项执行 range 操作并分发到相应的实现
inline Tensor dispatch_range(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    const TensorOptions& options) {
  // 初始化选项中指定的设备环境
  torch::utils::maybe_initialize_device(options);
  // 释放 GIL，允许其他线程执行 Python 代码
  pybind11::gil_scoped_release no_gil;
  // 指定设备，并保护其设备环境
  DeviceGuard device_guard(options.device());
  // 调用 Torch 库的 range 函数，生成指定范围内的张量
  return torch::range(start, end, step, options);
}

// 静态函数，提供 Python 绑定的 range 函数实现
static PyObject* THPVariable_range(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  // 处理 Torch 异常
  HANDLE_TH_ERRORS
  // 解析 Python 参数，支持多种参数配置
  static PythonArgParser parser({
      "range(Scalar start, Scalar end, Scalar step=1, *, Tensor out=None, ScalarType dtype=None, Layout layout=torch.strided, Device device=None, bool requires_grad=False)",
  });

  // 解析参数并返回解析结果
  ParsedArgs<8> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 根据解析结果执行相应的操作
  if (r.idx == 0) {
    // 发出 Python 用户警告，说明 torch.range 将来会被移除
    auto ret = PyErr_WarnEx(
        PyExc_UserWarning,
        "torch.range is deprecated and will be removed in a future release "
        "because its behavior is inconsistent with Python's range builtin. "
        "Instead, use torch.arange, which produces values in [start, end).",
        1);
    // 如果警告失败，抛出 Python 错误
    if (ret != 0)
      throw python_error();
    // 如果第四个参数为None，表示没有传入张量作为输出参数
    if (r.isNone(3)) {
      // 创建TensorOptions对象，设置张量的数据类型、设备、布局和是否需要梯度
      const auto options = TensorOptions()
                               .dtype(r.scalartype(4))   // 设置张量数据类型
                               .device(r.device(6))      // 设置张量所在设备
                               .layout(r.layout(5))      // 设置张量布局
                               .requires_grad(r.toBool(7));  // 设置是否需要梯度
      // 调用dispatch_range函数，使用指定的参数和选项创建张量，并将结果进行包装后返回
      return wrap(
          dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), options));
    } else {
      // 如果第四个参数不为None，需要检查输出张量的类型是否匹配预期
      check_out_type_matches(
          r.tensor(3),               // 获取传入的输出张量
          r.scalartype(4),           // 获取预期的数据类型
          r.isNone(4),               // 检查是否期望张量为None
          r.layout(5),               // 获取预期的布局
          r.device(6),               // 获取预期的设备
          r.isNone(6));              // 检查是否期望设备为None
      // 调用dispatch_range函数，使用传入的输出张量和其他参数创建张量，并设置是否需要梯度，然后将结果进行包装后返回
      return wrap(
          dispatch_range(r.scalar(0), r.scalar(1), r.scalar(2), r.tensor(3))
              .set_requires_grad(r.toBool(7)));
    }
  }
  // 返回Python中的None对象，表示函数执行完毕没有返回值
  Py_RETURN_NONE;
  // 在C++中处理异常并返回Python异常信息
  END_HANDLE_TH_ERRORS
// implemented on python object to allow torch.as_tensor to be constructed with
// arbitrarily nested python objects - list, tuple, np array, scalar, etc.
static PyObject* THPVariable_as_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，支持 as_tensor 函数的不同参数组合
  static PythonArgParser parser({
      "as_tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None)",
  });

  // 解析函数参数
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 检查是否存在 torch 函数重载
  if (r.has_torch_function()) {
    // 处理 torch 函数重载调用
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // 发出追踪警告
  jit::tracer::warn("torch.as_tensor", jit::tracer::WARN_CONSTRUCTOR);
  // 返回封装后的 THPVariable 对象，调用 torch::utils::as_tensor 函数
  return THPVariable_Wrap(torch::utils::as_tensor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// implemented on python object here because PyObject currently not natively
// declarable See: ATen/native/README.md for more context
static PyObject* THPVariable_from_numpy(PyObject* module, PyObject* arg) {
  HANDLE_TH_ERRORS
  // 发出追踪警告
  jit::tracer::warn("torch.from_numpy", jit::tracer::WARN_CONSTRUCTOR);
  // 返回封装后的 THPVariable 对象，调用 torch::utils::tensor_from_numpy 函数
  return THPVariable_Wrap(torch::utils::tensor_from_numpy(arg));
  END_HANDLE_TH_ERRORS
}

// 实现 dispatch_nonzero 函数，返回非零元素的索引张量
static Tensor dispatch_nonzero(const Tensor& self) {
  pybind11::gil_scoped_release no_gil;
  // 选择设备后，调用 self.nonzero() 函数
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero();
}

// 实现 dispatch_nonzero 函数的重载，将结果存储到给定的 out 张量中
static Tensor dispatch_nonzero(const Tensor& self, Tensor out) {
  pybind11::gil_scoped_release no_gil;
  // 选择设备后，调用 at::nonzero_out 函数
  OptionalDeviceGuard device_guard(device_of(self));
  return at::nonzero_out(out, self);
}

// 实现 dispatch_nonzero_numpy 函数，返回非零元素的索引张量数组
static std::vector<Tensor> dispatch_nonzero_numpy(const Tensor& self) {
  pybind11::gil_scoped_release no_gil;
  // 选择设备后，调用 self.nonzero_numpy() 函数
  OptionalDeviceGuard device_guard(device_of(self));
  return self.nonzero_numpy();
}

// 定义 THPVariable_nonzero 函数的声明
static PyObject* THPVariable_nonzero(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs);

// 定义宏 THPVARIABLE_SPARSE_COMPRESSED_CTOR，用于简化稀疏压缩构造函数的定义
#define THPVARIABLE_SPARSE_COMPRESSED_CTOR(NAME, NARGS, SIGNATURES)       \
  static PyObject* THPVariable_##NAME(                                    \
      PyObject* self, PyObject* args, PyObject* kwargs) {                 \
    HANDLE_TH_ERRORS                                                      \
    // 静态 Python 参数解析器，支持稀疏压缩构造函数的不同参数组合
    static PythonArgParser parser SIGNATURES;                             \
    // 解析函数参数
    ParsedArgs<NARGS> parsed_args;                                        \
    auto r = parser.parse(args, kwargs, parsed_args);                     \
    // 检查是否存在 torch 函数重载
    if (r.has_torch_function()) {                                         \
      // 处理 torch 函数重载调用
      return handle_torch_function(                                       \
          r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch"); \
    }                                                                     \
    // 发出追踪警告
    jit::tracer::warn("torch." #NAME, jit::tracer::WARN_CONSTRUCTOR);     \
    // 开始 TH 错误处理
    ```
    return THPVariable_Wrap(torch::utils::NAME##_ctor(                    \
        torch::tensors::get_default_dispatch_key(),                       \
        torch::tensors::get_default_scalar_type(),                        \
        r));                                                              \
    END_HANDLE_TH_ERRORS                                                  \
  }



    // 返回一个封装了 Torch 模块的变量，使用给定的构造函数和参数
    return THPVariable_Wrap(torch::utils::NAME##_ctor(                    \
        torch::tensors::get_default_dispatch_key(),                       \
        torch::tensors::get_default_scalar_type(),                        \
        r));                                                              \
    // 结束处理 Torch 错误的宏
    END_HANDLE_TH_ERRORS                                                  \
  }
// 定义 THPVARIABLE_SPARSE_COMPRESSED_CTOR 宏，用于定义稀疏压缩张量的构造函数
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    // 稀疏压缩张量类型名称为 sparse_compressed_tensor
    sparse_compressed_tensor,
    // 指定优先级为 10
    10,
    // 构造函数的重载，可以接受不同的参数组合
    ({
        // 构造函数声明及参数说明，包括压缩索引、普通索引、值、大小、数据类型、布局、设备等信息
        "sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
        // 简化参数组合，不指定大小参数
        "sparse_compressed_tensor(PyObject* compressed_indices, PyObject* plain_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"
    }))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_csr_tensor,
    10,
    ({
        "sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
        "sparse_csr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"
    }))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_csc_tensor,
    10,
    ({
        "sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
        "sparse_csc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"
    }))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_bsr_tensor,
    10,
    ({
        "sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
        "sparse_bsr_tensor(PyObject* crow_indices, PyObject* col_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"
    }))
THPVARIABLE_SPARSE_COMPRESSED_CTOR(
    sparse_bsc_tensor,
    10,
    ({
        "sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)",
        "sparse_bsc_tensor(PyObject* ccol_indices, PyObject* row_indices, PyObject* values, *, ScalarType dtype=None, Layout? layout=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, bool check_invariants=None)"
    }))

// 定义稀疏 COO 格式张量的构造函数，接受各种参数，包括 self 和 args
static PyObject* THPVariable_sparse_coo_tensor(
    PyObject* self,
    PyObject* args,
    // 下一行代码未显示完整，在此上下文中，通常是声明或定义的一部分
    // 定义函数 sparse_coo_tensor，接受 PyObject* 类型的参数 args 和 kwargs
    PyObject* sparse_coo_tensor(PyObject* args, PyObject* kwargs) {
      // 处理异常和错误
      HANDLE_TH_ERRORS
      // 定义静态 PythonArgParser 对象 parser，用于解析不同的函数签名
      static PythonArgParser parser({
          "sparse_coo_tensor(PyObject* indices, PyObject* values, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None)",
          "sparse_coo_tensor(PyObject* indices, PyObject* values, IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None, bool is_coalesced=None)",
          "sparse_coo_tensor(IntArrayRef size, *, ScalarType dtype=None, Device? device=None, bool requires_grad=False, bool check_invariants=None)",
      });
    
      // 创建 ParsedArgs 对象 parsed_args，最多包含 8 个参数
      ParsedArgs<8> parsed_args;
      // 使用 parser 解析参数 args 和 kwargs，并将结果存储在 r 中
      auto r = parser.parse(args, kwargs, parsed_args);
      // 如果解析结果需要调用 Torch 函数
      if (r.has_torch_function()) {
        // 调用 handle_torch_function 处理 Torch 函数调用
        return handle_torch_function(
            r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
      }
      // 发出警告，表示正在使用 torch.sparse_coo_tensor 构造函数
      jit::tracer::warn("torch.sparse_coo_tensor", jit::tracer::WARN_CONSTRUCTOR);
      // 调用 torch::utils::sparse_coo_tensor_ctor 创建稀疏 COO 张量，并用 THPVariable_Wrap 封装返回结果
      return THPVariable_Wrap(torch::utils::sparse_coo_tensor_ctor(
          torch::tensors::get_default_dispatch_key(),
          torch::tensors::get_default_scalar_type(),
          r));
      // 结束异常处理块
      END_HANDLE_TH_ERRORS
    }
}

// Python对象的实现，允许使用任意嵌套的Python对象（列表、元组、np数组、标量等）构造torch.tensor
static PyObject* THPVariable_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义Python参数解析器，支持以下构造函数
  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  // 解析参数对象
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 如果存在torch函数重载，调用处理函数
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // 发出追踪警告
  jit::tracer::warn("torch.tensor", jit::tracer::WARN_CONSTRUCTOR);
  // 返回torch::utils::tensor_ctor函数的包装结果
  return THPVariable_Wrap(torch::utils::tensor_ctor(
      torch::tensors::get_default_dispatch_key(),
      torch::tensors::get_default_scalar_type(),
      r));
  END_HANDLE_TH_ERRORS
}

// 获取Tensor输入的设备
static PyObject* THPVariable_get_device(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义Python参数解析器
  static PythonArgParser parser(
      {
          "get_device(Tensor input)",
      },
      /*traceable=*/false);

  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs);

  // 如果索引为0，返回输入Tensor的设备
  if (r.idx == 0) {
    return wrap(r.tensor(0).get_device());
  }
  // 返回None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 从缓冲区创建Tensor对象
static PyObject* THPVariable_frombuffer(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义Python参数解析器
  static PythonArgParser parser(
      {
          "frombuffer(PyObject* buffer, *, ScalarType dtype, int64_t count=-1, int64_t offset=0, bool requires_grad=False)",
      },
      /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs);

  // 如果索引为0，从缓冲区创建Tensor对象
  if (r.idx == 0) {
    auto buffer = r.pyobject(0);
    auto dtype = r.scalartype(1);
    auto count = r.toInt64(2);
    auto offset = r.toInt64(3);
    auto requires_grad = r.toBool(4);

    // 检查缓冲区对象是否实现了Python缓冲区协议
    TORCH_CHECK_VALUE(
        PyObject_CheckBuffer(buffer) != 0,
        "object does not implement Python buffer protocol.");
    // 返回torch::utils::tensor_frombuffer函数的包装结果
    return wrap(torch::utils::tensor_frombuffer(
        buffer, dtype, count, offset, requires_grad));
  }

  // 返回None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// 将Python对象转换为Tensor对象
static PyObject* THPVariable_asarray(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义Python参数解析器
  static PythonArgParser parser(
      {
          "asarray(PyObject* obj, *, ScalarType? dtype=None, Device? device=None, bool? copy=None, bool requires_grad=False)",
      },
      /*traceable=*/false);

  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs);

  // 如果存在torch函数重载，调用处理函数
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  // 如果索引为0，将Python对象转换为Tensor对象
  if (r.idx == 0) {
    auto obj = r.pyobject(0);
    auto dtype = r.scalartypeOptional(1);
    auto device = r.deviceOptional(2);
    auto copy = r.toBoolOptional(3);
    auto requires_grad = r.toBool(4);
    // 返回torch::utils::tensor_asarray函数的包装结果
    return wrap(torch::utils::tensor_asarray(
        obj, dtype, device, copy, requires_grad));
  }

  // 返回None
  Py_RETURN_NONE;
}
    # 调用对象 r 的方法 toBoolOptional，将第 3 个参数转换为可选的布尔值
    auto copy = r.toBoolOptional(3);
    # 调用对象 r 的方法 toBool，获取第 4 个参数作为布尔值
    auto requires_grad = r.toBool(4);
    # 调用 torch::utils::asarray 函数，将 obj 转换为张量数组
    # 使用指定的 dtype 和 device 参数，以及从 r 获取的 copy 和 requires_grad 参数
    return wrap(torch::utils::asarray(obj, dtype, device, copy, requires_grad));
  }

  # 返回 Python 中的 None 对象
  Py_RETURN_NONE;
  # 终止处理 Torch 异常
  END_HANDLE_TH_ERRORS
// THPVariable_numel 函数的声明，用于返回张量的元素数量
static PyObject* THPVariable_numel(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs);

// THPVariable__to_functional_tensor 函数的实现，将张量转换为功能张量
static PyObject* THPVariable__to_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，声明接受一个张量参数
  static PythonArgParser parser(
      {"_to_functional_tensor(Tensor t)"},
      /*traceable=*/true);

  // 解析函数参数
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  // 调用 Torch 实现的功能化转换函数
  auto wrapped = at::functionalization::impl::to_functional_tensor(self_);
  // 封装结果并返回
  return wrap(std::move(wrapped));
  END_HANDLE_TH_ERRORS
}

// _mirror_autograd_meta_to 函数实现，根据源张量设置部分自动求导元数据到目标张量
// 包括 requires_grad 和 grad_fn
static PyObject* THPVariable__mirror_autograd_meta_to(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，声明接受两个张量参数
  static PythonArgParser parser(
      {"_mirror_autograd_meta_to(Tensor source, Tensor dest)"},
      /*traceable=*/true);

  // 解析函数参数
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto src_ = r.tensor(0);
  auto dst_ = r.tensor(1);
  
  // 获取源张量的自动求导元数据，并根据需要设置到目标张量
  auto inner_autograd_meta = impl::get_autograd_meta(src_);
  if (inner_autograd_meta) {
    dst_.set_requires_grad(src_.requires_grad());
    if (dst_.requires_grad()) {
      // 创建一个新的错误 grad_fn，以确保包装的 is_leaf 元数据准确
      auto new_grad_fn = std::shared_ptr<torch::autograd::Error>(
          new torch::autograd::Error(
              "Cannot backprop through mirrored meta, file a bug in PyTorch"),
          torch::autograd::deleteNode);
      torch::autograd::set_history(dst_, new_grad_fn);
    }
  }
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// THPVariable__from_functional_tensor 函数实现，从功能张量中获取原始张量
static PyObject* THPVariable__from_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，声明接受一个张量参数
  static PythonArgParser parser(
      {"_from_functional_tensor(Tensor t)"}, /*traceable=*/true);

  // 解析函数参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs);
  auto self_ = r.tensor(0);
  // 调用 Torch 实现的功能化反转函数
  auto unwrapped = at::functionalization::impl::from_functional_tensor(self_);
  // 封装结果并返回
  return wrap(std::move(unwrapped));
  END_HANDLE_TH_ERRORS
}

// THPVariable__freeze_functional_tensor 函数实现，冻结功能张量，阻止其进一步修改
static PyObject* THPVariable__freeze_functional_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，声明接受一个张量参数
  static PythonArgParser parser(
      {"_freeze_functional_tensor(Tensor t)"}, /*traceable=*/true);

  // 解析函数参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs);
  auto self_ = r.tensor(0);
  // 调用 Torch 实现的功能化冻结函数
  at::functionalization::impl::freeze_functional_tensor(self_);
  // 返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
    PyObject* self,                          // 第一个参数：指向调用此函数的对象的指针
    PyObject* args,                          // 第二个参数：传递给函数的位置参数的元组
    PyObject* kwargs) {                      // 第三个参数：传递给函数的关键字参数的字典
  HANDLE_TH_ERRORS                          // 处理 Torch 错误的宏开始

  static PythonArgParser parser(             // 声明静态的 Python 参数解析器对象
      {"_is_functional_tensor(Tensor t)"},   // 解析器的参数信息，仅包含一个函数签名
      /*traceable=*/true);

  ParsedArgs<1> parsed_args;                 // 声明一个包含一个参数的解析结果对象
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析传入的参数并存储结果
  auto self_ = r.tensor(0);                  // 获取解析结果中的第一个参数作为 self_

  if (at::functionalization::impl::isFunctionalTensor(self_)) {  // 检查 self_ 是否为功能性张量
    Py_RETURN_TRUE;                          // 返回 Python 的 True
  } else {
    Py_RETURN_FALSE;                         // 返回 Python 的 False
  }

  END_HANDLE_TH_ERRORS                      // 处理 Torch 错误的宏结束
static PyObject* THPVariable__functionalize_was_storage_changed(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，用于解析输入参数和关键字参数
  static PythonArgParser parser(
      {"_functionalize_was_storage_changed(Tensor t)"}, /*traceable=*/true);

  // 解析函数参数，预期有一个张量类型的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);  // 获取解析结果中的第一个参数作为张量 self_
  // 内部断言，确保 self_ 是一个支持功能化的张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 获取张量的功能化包装器
  auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(self_);
  // 检查功能化包装器是否标记了存储变化
  if (wrapper->was_storage_changed()) {
    Py_RETURN_TRUE;  // 如果标记了存储变化，返回 Python 中的 True
  } else {
    Py_RETURN_FALSE;  // 如果没有标记存储变化，返回 Python 中的 False
  }
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常结束
}

static PyObject* THPVariable__functionalize_get_storage_size(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，用于解析输入参数和关键字参数
  static PythonArgParser parser(
      {"_functionalize_get_storage_size(Tensor t, bool before)"},
      /*traceable=*/true);

  // 解析函数参数，预期有两个参数：一个张量和一个布尔值
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);  // 获取解析结果中的第一个参数作为张量 self_
  auto before = r.toBool(1);  // 获取解析结果中的第二个参数作为布尔值 before
  // 内部断言，确保 self_ 是一个支持功能化的张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 获取张量的功能化包装器
  auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(self_);
  // 获取在指定时刻（之前或之后）的存储大小
  auto size = wrapper->get_storage_size(/*before=*/before);
  return toPyObject(size);  // 将存储大小转换为 Python 对象并返回
  Py_RETURN_NONE;  // 如果出现错误，返回 Python 中的 None 对象
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常结束
}

static PyObject* THPVariable__functionalize_has_data_mutation(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，用于解析输入参数和关键字参数
  static PythonArgParser parser(
      {"_functionalize_has_data_mutation(Tensor t)"}, /*traceable=*/true);

  // 解析函数参数，预期有一个张量类型的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);  // 获取解析结果中的第一个参数作为张量 self_
  // 内部断言，确保 self_ 是一个支持功能化的张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 获取张量的功能化包装器
  auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(self_);
  // 检查功能化包装器是否标记了数据变异
  if (wrapper->has_data_mutation()) {
    Py_RETURN_TRUE;  // 如果标记了数据变异，返回 Python 中的 True
  } else {
    Py_RETURN_FALSE;  // 如果没有标记数据变异，返回 Python 中的 False
  }
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常结束
}

static PyObject* THPVariable__functionalize_has_metadata_mutation(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，用于解析输入参数和关键字参数
  static PythonArgParser parser(
      {"_functionalize_has_metadata_mutation(Tensor t)"}, /*traceable=*/true);

  // 解析函数参数，预期有一个张量类型的参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);  // 获取解析结果中的第一个参数作为张量 self_
  // 内部断言，确保 self_ 是一个支持功能化的张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 获取张量的功能化包装器
  auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(self_);
  // 检查功能化包装器是否标记了元数据变异
  if (wrapper->has_metadata_mutation()) {
    Py_RETURN_TRUE;  // 如果标记了元数据变异，返回 Python 中的 True
  } else {
    Py_RETURN_FALSE;  // 如果没有标记元数据变异，返回 Python 中的 False
  }
  END_HANDLE_TH_ERRORS  // 处理 Torch 异常结束
}
    // 定义一个函数，接收一个 PyObject* 类型的参数列表和关键字参数字典
    PyObject* kwargs) {
      // 处理可能发生的 Torch 错误
      HANDLE_TH_ERRORS
      // 定义静态的 PythonArgParser 对象，用于解析参数
      static PythonArgParser parser(
          {"_enable_functionalization(*, bool reapply_views=False)"},
          /*traceable=*/true);
      // 定义 ParsedArgs 对象，用于存储解析后的参数
      ParsedArgs<1> parsed_args;
      // 调用 parser 对象的 parse 方法解析参数
      auto r = parser.parse(args, kwargs, parsed_args);
      // 从解析结果中获取 bool 类型的 reapply_views 参数值
      const auto reapply_views = r.toBool(0);
    
      // 检查是否当前线程局部存储中包含函数化分发键
      if (c10::impl::tls_is_dispatch_key_included(at::DispatchKey::Functionalize)) {
        // 如果是，则抛出内部断言错误，提示不支持多层模式样式函数化嵌套
        TORCH_INTERNAL_ASSERT(
            false,
            "multiple layers of mode-style functionalization nesting is not"
            " currently supported, outside of the functionalize() transform");
      }
      // 将函数化分发键包含在当前线程局部存储中
      c10::impl::tls_set_dispatch_key_included(
          at::DispatchKey::Functionalize, true);
      // 如果 reapply_views 参数为真，则设置 TLS 中的重新应用视图标志位为真
      if (reapply_views) {
        at::functionalization::impl::setFunctionalizationReapplyViewsTLS(true);
      }
      // 返回 Python 中的 None 对象
      Py_RETURN_NONE;
      // 处理 Torch 错误的结束标记
      END_HANDLE_TH_ERRORS
    }
static PyObject* THPVariable__functionalize_enable_reapply_views(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义静态 PythonArgParser 对象，用于解析函数参数
  static PythonArgParser parser(
      {"_functionalize_enable_reapply_views(bool reapply_views=False)"},
      /*traceable=*/true);
  // 解析函数参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 获取 reapply_views 参数的布尔值
  const auto reapply_views = r.toBool(0);
  // 获取当前的功能化重新应用视图标志
  auto old = at::functionalization::impl::getFunctionalizationReapplyViewsTLS();
  // 根据 reapply_views 的值设置功能化重新应用视图标志
  if (reapply_views) {
    at::functionalization::impl::setFunctionalizationReapplyViewsTLS(true);
  } else {
    at::functionalization::impl::setFunctionalizationReapplyViewsTLS(false);
  }
  // 根据旧标志返回相应的 Python 布尔值
  if (old) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__functionalize_is_multi_output_view(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义静态 PythonArgParser 对象，用于解析函数参数
  static PythonArgParser parser(
      {"_functionalize_is_multi_output_view(Tensor t)"},
      /*traceable=*/true);
  // 解析函数参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 获取参数中的 Tensor 对象
  auto t = r.tensor(0);
  // 检查是否为功能化张量
  TORCH_CHECK(at::functionalization::impl::isFunctionalTensor(t));
  // 获取功能化张量的实现包装器
  auto t_impl = at::functionalization::impl::unsafeGetFunctionalWrapper(t);
  // 返回功能化张量是否为多输出视图的 Python 布尔值
  if (t_impl->is_multi_output_view()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__disable_functionalization(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 禁用功能化分发键
  c10::impl::tls_set_dispatch_key_included(
      at::DispatchKey::Functionalize, false);
  // 设置功能化重新应用视图标志为 false
  at::functionalization::impl::setFunctionalizationReapplyViewsTLS(false);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__functionalize_replace(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义静态 PythonArgParser 对象，用于解析函数参数
  static PythonArgParser parser(
      {"_functionalize_replace(Tensor t, Tensor o)"}, /*traceable=*/true);

  // 解析函数参数
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 获取参数中的第一个和第二个 Tensor 对象
  auto self_ = r.tensor(0);
  auto other = r.tensor(1);
  // 内部断言：self_ 必须为功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 内部断言：other 不能为功能化张量
  TORCH_INTERNAL_ASSERT(
      !at::functionalization::impl::isFunctionalTensor(other));
  // 替换功能化张量 self_ 的实现为 other
  at::functionalization::impl::replace_(self_, other);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__functionalize_commit_update(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义静态 PythonArgParser 对象，用于解析函数参数
  static PythonArgParser parser(
      {"_functionalize_commit_update(Tensor t)"}, /*traceable=*/true);

  // 解析函数参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  // 获取参数中的 Tensor 对象
  auto self_ = r.tensor(0);
  // 内部断言：self_ 必须为功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 提交功能化张量的更新
  at::functionalization::impl::commit_update(self_);
  // 返回 Python 的 None 对象
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable__functionalize_sync(
    PyObject* self,                   // 第一个参数，表示指向当前对象的指针
    PyObject* args,                   // 第二个参数，表示传递给函数的位置参数元组
    PyObject* kwargs) {               // 第三个参数，表示传递给函数的关键字参数字典
  HANDLE_TH_ERRORS                    // 处理 Torch 异常的宏，进入异常处理逻辑

  static PythonArgParser parser(      // 创建静态的 PythonArgParser 对象
      {"_functionalize_sync(Tensor t)"}, /*traceable=*/true);  // 使用指定的函数签名初始化解析器

  ParsedArgs<1> parsed_args;          // 创建一个 ParsedArgs 实例，用于存储解析后的参数
  auto r = parser.parse(args, kwargs, parsed_args);  // 解析传入的参数，将结果存储在 r 中
  auto self_ = r.tensor(0);           // 从解析结果中获取第一个位置参数作为 Tensor 对象
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));  // 断言 Tensor 对象是函数化的
  at::functionalization::impl::sync(self_);  // 执行 Tensor 对象的同步操作
  Py_RETURN_NONE;                     // 返回 Python 中的 None 对象
  END_HANDLE_TH_ERRORS                // 结束 Torch 异常处理逻辑
static PyObject* THPVariable__functionalize_is_symbolic(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，指定函数签名并启用追踪
  static PythonArgParser parser(
      {"_functionalize_is_symbolic(Tensor tensor)"},
      /*traceable=*/true);

  // 解析输入参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto tensor = r.tensor(0);
  // 内部断言，确认输入的 tensor 是功能化张量
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(tensor));
  // 获取 tensor 的功能化包装器
  auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
  // 检查功能化包装器是否为符号化的
  if (impl->is_symbolic()) {
    Py_RETURN_TRUE;  // 返回 Python 的 True 对象
  } else {
    Py_RETURN_FALSE;  // 返回 Python 的 False 对象
  }
  END_HANDLE_TH_ERRORS  // 结束错误处理块
}

static PyObject* THPVariable__functionalize_apply_view_metas(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，指定函数签名并启用追踪
  static PythonArgParser parser(
      {"_functionalize_apply_view_metas(Tensor tensor, Tensor base)"},
      /*traceable=*/true);

  // 解析输入参数
  ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto tensor = r.tensor(0);
  // 内部断言，确认输入的 tensor 是功能化张量
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(tensor));
  // 获取 tensor 的功能化包装器
  auto impl = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
  // 应用视图元信息到 tensor 上并返回结果
  return wrap(impl->apply_view_metas(r.tensor(1)));
  END_HANDLE_TH_ERRORS  // 结束错误处理块
}

static PyObject* THPVariable__functionalize_mark_mutation_hidden_from_autograd(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，指定函数签名并启用追踪
  static PythonArgParser parser(
      {"_functionalize_mark_mutation_hidden_from_autograd(Tensor t)"},
      /*traceable=*/true);

  // 解析输入参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  // 内部断言，确认输入的 self_ 是功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 标记 self_ 的变异操作对自动求导不可见
  at::functionalization::impl::mark_mutation_hidden_from_autograd(self_);
  Py_RETURN_NONE;  // 返回 Python 的 None 对象
  END_HANDLE_TH_ERRORS  // 结束错误处理块
}

static PyObject*
THPVariable__functionalize_are_all_mutations_hidden_from_autograd(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，指定函数签名并启用追踪
  static PythonArgParser parser(
      {"_functionalize_are_all_mutations_hidden_from_autograd(Tensor t)"},
      /*traceable=*/true);

  // 解析输入参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  // 内部断言，确认输入的 self_ 是功能化张量
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 检查 self_ 的所有变异操作是否对自动求导都不可见
  if (at::functionalization::impl::are_all_mutations_hidden_from_autograd(
          self_)) {
    Py_RETURN_TRUE;  // 返回 Python 的 True 对象
  } else {
    Py_RETURN_FALSE;  // 返回 Python 的 False 对象
  }
  END_HANDLE_TH_ERRORS  // 结束错误处理块
}
    // 定义一个函数 `_functionalize_was_inductor_storage_resized`，接收一个参数 `t`，类型为 Tensor
    PyObject* kwargs) {
    // 处理异常
    HANDLE_TH_ERRORS
      // 定义静态的 PythonArgParser 对象 `parser`
      static PythonArgParser parser(
          // 指定支持的方法名和参数
          {"_functionalize_was_inductor_storage_resized(Tensor t)"},
          /*traceable=*/true);
    
      // 解析函数参数，预期有一个参数
      ParsedArgs<1> parsed_args;
      auto r = parser.parse(args, kwargs, parsed_args);
      // 获取第一个参数作为 self_
      auto self_ = r.tensor(0);
      // 在内部断言 self_ 是否为功能化张量
      TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
      // 获取功能化实现对象
      auto functional_impl =
          at::functionalization::impl::unsafeGetFunctionalWrapper(self_);
      // 如果功能化实现对象的 `was_inductor_storage_resized()` 返回 true，则返回 Python 的 True 对象
      if (functional_impl->was_inductor_storage_resized()) {
        Py_RETURN_TRUE;
      } else {
        // 否则返回 Python 的 False 对象
        Py_RETURN_FALSE;
      }
      // 结束异常处理
      END_HANDLE_TH_ERRORS
// 结束对异常的处理
END_HANDLE_TH_ERRORS
// 定义一个静态的Python方法，用于检查是否所有变异操作在没有梯度或推断模式下
static PyObject*
THPVariable__functionalize_are_all_mutations_under_no_grad_or_inference_mode(
    PyObject* self,  // Python 方法的自身对象
    PyObject* args,  // 方法的参数元组
    PyObject* kwargs) {  // 方法的关键字参数字典
  HANDLE_TH_ERRORS  // 开始处理异常
  // 定义Python参数解析器，该方法允许跟踪（traceable）
  static PythonArgParser parser(
      {"_functionalize_are_all_mutations_under_no_grad_or_inference_mode(Tensor t)"},
      /*traceable=*/true);
  // 解析参数
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);  // 获取第一个参数作为Tensor对象
  // 内部断言，确保self_是功能化张量（functional tensor）
  TORCH_INTERNAL_ASSERT(at::functionalization::impl::isFunctionalTensor(self_));
  // 如果所有变异操作都在没有梯度或推断模式下，则返回Python中的True
  if (at::functionalization::impl::
          are_all_mutations_under_no_grad_or_inference_mode(self_)) {
    Py_RETURN_TRUE;
  } else {  // 否则返回Python中的False
    Py_RETURN_FALSE;
  }
  // 结束对异常的处理
  END_HANDLE_TH_ERRORS
}
// XXX: ops that are bound here are not exposed to the C++ api nor the JIT.
// Any new ops added here should be accompanied with a comment why they are not
// being registered through native_functions.yaml, and be tagged cpp / JIT
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
// 定义Python方法数组，这些方法没有暴露给C++ API或JIT
static PyMethodDef torch_functions_manual[] = {
    // 将Python函数asarray注册到torch_functions_manual数组中
    {"asarray",
     castPyCFunctionWithKeywords(THPVariable_asarray),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数as_tensor注册到torch_functions_manual数组中
    {"as_tensor",
     castPyCFunctionWithKeywords(THPVariable_as_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数from_numpy注册到torch_functions_manual数组中
    {"from_numpy", THPVariable_from_numpy, METH_STATIC | METH_O, nullptr},
    // 将Python函数frombuffer注册到torch_functions_manual数组中
    {"frombuffer",
     castPyCFunctionWithKeywords(THPVariable_frombuffer),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_is_functional_tensor注册到torch_functions_manual数组中
    {"_is_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__is_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_to_functional_tensor注册到torch_functions_manual数组中
    {"_to_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__to_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_mirror_autograd_meta_to注册到torch_functions_manual数组中
    {"_mirror_autograd_meta_to",
     castPyCFunctionWithKeywords(THPVariable__mirror_autograd_meta_to),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_from_functional_tensor注册到torch_functions_manual数组中
    {"_from_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__from_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_freeze_functional_tensor注册到torch_functions_manual数组中
    {"_freeze_functional_tensor",
     castPyCFunctionWithKeywords(THPVariable__freeze_functional_tensor),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_functionalize_replace注册到torch_functions_manual数组中
    {"_functionalize_replace",
     castPyCFunctionWithKeywords(THPVariable__functionalize_replace),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_functionalize_commit_update注册到torch_functions_manual数组中
    {"_functionalize_commit_update",
     castPyCFunctionWithKeywords(THPVariable__functionalize_commit_update),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    // 将Python函数_functionalize_sync注册到torch_functions_manual数组中
    {"_functionalize_sync",
     castPyCFunctionWithKeywords(THPVariable__functionalize_sync),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    {"_functionalize_is_symbolic",
     castPyCFunctionWithKeywords(THPVariable__functionalize_is_symbolic),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_is_symbolic 的Python C函数，使用 THPVariable__functionalize_is_symbolic 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_apply_view_metas",
     castPyCFunctionWithKeywords(THPVariable__functionalize_apply_view_metas),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_apply_view_metas 的Python C函数，使用 THPVariable__functionalize_apply_view_metas 函数实现，接受变长参数和关键字参数，静态方法
    {"_enable_functionalization",
     castPyCFunctionWithKeywords(THPVariable__enable_functionalization),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _enable_functionalization 的Python C函数，使用 THPVariable__enable_functionalization 函数实现，接受变长参数和关键字参数，静态方法
    {"_disable_functionalization",
     castPyCFunctionWithKeywords(THPVariable__disable_functionalization),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _disable_functionalization 的Python C函数，使用 THPVariable__disable_functionalization 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_has_metadata_mutation",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_has_metadata_mutation),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_has_metadata_mutation 的Python C函数，使用 THPVariable__functionalize_has_metadata_mutation 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_mark_mutation_hidden_from_autograd",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_mark_mutation_hidden_from_autograd),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_mark_mutation_hidden_from_autograd 的Python C函数，使用 THPVariable__functionalize_mark_mutation_hidden_from_autograd 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_are_all_mutations_hidden_from_autograd",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_are_all_mutations_hidden_from_autograd),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_are_all_mutations_hidden_from_autograd 的Python C函数，使用 THPVariable__functionalize_are_all_mutations_hidden_from_autograd 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_was_inductor_storage_resized",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_was_inductor_storage_resized),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_was_inductor_storage_resized 的Python C函数，使用 THPVariable__functionalize_was_inductor_storage_resized 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_are_all_mutations_under_no_grad_or_inference_mode",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_are_all_mutations_under_no_grad_or_inference_mode),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_are_all_mutations_under_no_grad_or_inference_mode 的Python C函数，使用 THPVariable__functionalize_are_all_mutations_under_no_grad_or_inference_mode 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_is_multi_output_view",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_is_multi_output_view),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_is_multi_output_view 的Python C函数，使用 THPVariable__functionalize_is_multi_output_view 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_has_data_mutation",
     castPyCFunctionWithKeywords(THPVariable__functionalize_has_data_mutation),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_has_data_mutation 的Python C函数，使用 THPVariable__functionalize_has_data_mutation 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_was_storage_changed",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_was_storage_changed),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_was_storage_changed 的Python C函数，使用 THPVariable__functionalize_was_storage_changed 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_get_storage_size",
     castPyCFunctionWithKeywords(THPVariable__functionalize_get_storage_size),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_get_storage_size 的Python C函数，使用 THPVariable__functionalize_get_storage_size 函数实现，接受变长参数和关键字参数，静态方法
    {"_functionalize_enable_reapply_views",
     castPyCFunctionWithKeywords(
         THPVariable__functionalize_enable_reapply_views),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 _functionalize_enable_reapply_views 的Python C函数，使用 THPVariable__functionalize_enable_reapply_views 函数实现，接受变长参数和关键字参数，静态方法
    {"nonzero",
     castPyCFunctionWithKeywords(THPVariable_nonzero),
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,
     nullptr},
    # 定义一个名为 nonzero 的Python C函数，使用 THPVariable_nonzero 函数实现，接受变长参数和关键字参数，静态方法
    {"range",  // 定义名为 "range" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_range),  // 使用 THPVariable_range 函数创建一个 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"sparse_coo_tensor",  // 定义名为 "sparse_coo_tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_sparse_coo_tensor),  // 使用 THPVariable_sparse_coo_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"sparse_compressed_tensor",  // 定义名为 "sparse_compressed_tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_sparse_compressed_tensor),  // 使用 THPVariable_sparse_compressed_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"sparse_csr_tensor",  // 定义名为 "sparse_csr_tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_sparse_csr_tensor),  // 使用 THPVariable_sparse_csr_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"sparse_csc_tensor",  // 定义名为 "sparse_csc_tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_sparse_csc_tensor),  // 使用 THPVariable_sparse_csc_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"sparse_bsr_tensor",  // 定义名为 "sparse_bsr_tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_sparse_bsr_tensor),  // 使用 THPVariable_sparse_bsr_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"sparse_bsc_tensor",  // 定义名为 "sparse_bsc_tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_sparse_bsc_tensor),  // 使用 THPVariable_sparse_bsc_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"tensor",  // 定义名为 "tensor" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_tensor),  // 使用 THPVariable_tensor 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"get_device",  // 定义名为 "get_device" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_get_device),  // 使用 THPVariable_get_device 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串

    {"numel",  // 定义名为 "numel" 的函数映射
     castPyCFunctionWithKeywords(THPVariable_numel),  // 使用 THPVariable_numel 函数创建 Python C 函数对象
     METH_VARARGS | METH_KEYWORDS | METH_STATIC,  // 指定函数的调用约定：接受位置参数和关键字参数，并且是静态方法
     nullptr},  // 指定函数的文档字符串为 nullptr，即无文档字符串
};

static PyObject* THPVariable_nonzero(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，并指定唯一的参数签名
  static PythonArgParser parser({
      "nonzero(Tensor input, *, bool as_tuple=False, Tensor out=None)",
  });
  // 解析输入参数并保存结果
  ParsedArgs<3> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  // 如果解析结果表明存在 torch 函数的重载版本，则调用处理 torch 函数重载的函数
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  // 获取 as_tuple 参数的布尔值
  const auto as_tuple = r.toBool(1);
  // 检查是否传入了 out 参数
  const auto has_out = !r.isNone(2);

  // 如果 as_tuple 为真，则调用 numpy 风格的 nonzero 函数
  if (as_tuple) {
    TORCH_CHECK(
        !has_out,
        "nonzero does not support the out kwarg when as_tuple is True");
    return wrap(dispatch_nonzero_numpy(r.tensor(0)));
  }

  // 如果有传入 out 参数，则调用传入 out 参数的 nonzero 函数
  if (has_out) {
    return wrap(dispatch_nonzero(r.tensor(0), r.tensor(2)));
  }

  // 否则调用默认的 nonzero 函数
  return wrap(dispatch_nonzero(r.tensor(0)));

  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_numel(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // 定义 Python 参数解析器，此函数没有可选参数，因此 traceable 设为 false
  static PythonArgParser parser(
      {
          "numel(Tensor input)",
      },
      /*traceable=*/false);

  // 解析输入参数并保存结果
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs);

  // 如果解析结果表明存在 torch 函数的重载版本，则调用处理 torch 函数重载的函数
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, args, kwargs, THPVariableFunctionsModule, "torch");
  }

  // 直接返回输入 Tensor 的元素个数
  if (r.idx == 0) {
    return py::cast(r.tensor(0).sym_numel()).release().ptr();
  }
  // 如果解析失败，返回 None
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Sharded function definitions
// 定义三个不同的函数，用于收集 Torch 函数的方法定义
void gatherTorchFunctions_0(std::vector<PyMethodDef>& torch_functions);
void gatherTorchFunctions_1(std::vector<PyMethodDef>& torch_functions);
void gatherTorchFunctions_2(std::vector<PyMethodDef>& torch_functions);

// 收集所有 Torch 函数的方法定义
void gatherTorchFunctions(std::vector<PyMethodDef>& torch_functions) {
  // 计算手动定义 Torch 函数数量
  constexpr size_t num_functions =
      sizeof(torch_functions_manual) / sizeof(torch_functions_manual[0]);
  // 将手动定义的 Torch 函数添加到 torch_functions
  torch_functions.assign(
      torch_functions_manual, torch_functions_manual + num_functions);
  // 必须与 tools/autograd/gen_python_functions.py 中的 num_shards 同步
  gatherTorchFunctions_0(torch_functions);
  gatherTorchFunctions_1(torch_functions);
  gatherTorchFunctions_2(torch_functions);

  // 定义四组函数名与别名的对应关系
  static std::array<std::pair<const char*, const char*>, 4> aliases{
      {// 主函数名，别名
       {"sspaddmm", "saddmm"},
       {"mm", "spmm"},
       {"mm", "dsmm"},
       {"hspmm", "hsmm"}}};

  // 遍历别名列表，将主函数名替换为别名
  for (const auto& alias : aliases) {
    auto it = std::find_if(
        torch_functions.begin(),
        torch_functions.end(),
        [&](const PyMethodDef& def) {
          return strcmp(def.ml_name, alias.first) == 0;
        });
    // 断言是否成功找到要替换的函数名
    TORCH_INTERNAL_ASSERT(
        it != torch_functions.end(),
        "Failed to create function alias from ",
        alias.first,
        " to ",
        alias.second);
    // 复制函数定义，并将函数名修改为别名
    PyMethodDef alias_def = *it;
    alias_def.ml_name = alias.second;
    // 将 alias_def 添加到 torch_functions 的末尾
    torch_functions.push_back(alias_def);
  }

  // 在 torch_functions 的末尾添加一个空指针元素
  torch_functions.push_back({nullptr});

  // 优化内存，收缩 torch_functions 的容量以匹配其当前大小
  torch_functions.shrink_to_fit();
}

// 定义静态的 PyTypeObject 结构体 THPVariableFunctions
static PyTypeObject THPVariableFunctions = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C._VariableFunctionsClass", /* tp_name */
    0, /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr /* tp_new */
};

// 初始化 Torch 函数模块
void initTorchFunctions(PyObject* module) {
  // 静态的 PyMethodDef 向量，用于收集 Torch 函数
  static std::vector<PyMethodDef> torch_functions;
  // 收集 Torch 函数到 torch_functions 向量中
  gatherTorchFunctions(torch_functions);
  // 将收集到的 Torch 函数数据指针赋值给 THPVariableFunctions 的 tp_methods
  THPVariableFunctions.tp_methods = torch_functions.data();

  // 准备 PyTypeObject 结构 THPVariableFunctions，如果失败则抛出异常
  if (PyType_Ready(&THPVariableFunctions) < 0) {
    throw python_error();
  }
  // 增加对 THPVariableFunctions 的引用计数
  Py_INCREF(&THPVariableFunctions);

  // 增加对 THPVariableFunctions 的引用计数，并将其添加到 Python 模块中
  if (PyModule_AddObject(
          module,
          "_VariableFunctionsClass",
          reinterpret_cast<PyObject*>(&THPVariableFunctions)) < 0) {
    throw python_error();
  }

  // PyType_GenericNew 返回一个新的引用，将其赋给 THPVariableFunctionsModule
  THPVariableFunctionsModule =
      PyType_GenericNew(&THPVariableFunctions, Py_None, Py_None);
  // PyModule_AddObject 偷取了一个引用，将 THPVariableFunctionsModule 添加到 Python 模块中
  if (PyModule_AddObject(
          module, "_VariableFunctions", THPVariableFunctionsModule) < 0) {
    throw python_error();
  }
}

} // namespace autograd
} // namespace torch
```