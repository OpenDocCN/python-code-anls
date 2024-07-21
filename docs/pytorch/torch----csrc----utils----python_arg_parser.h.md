# `.\pytorch\torch\csrc\utils\python_arg_parser.h`

```
#pragma once

// 解析用于在C++中实现的Python函数的参数
// 这类似于PyArg_ParseTupleAndKeywords()，但专门处理PyTorch相关类型，并区分重载的函数签名。
//
// 示例:
//
//   static PythonArgParser parser({
//     "norm(Scalar p, int64_t dim, bool keepdim=False)",
//     "norm(Scalar p=2)",
//   });
//   ParsedArgs<3> parsed_args;
//   auto r = parser.parse(args, kwargs, parsed_args);
//   if (r.idx == 0) {
//     norm(r.scalar(0), r.int64(1), r.bool(0));
//   } else {
//     norm(r.scalar(0));
//   }
//
// 我们自动生成大多数PythonArgParser的用法；生成的文件位于torch/csrc/autograd/generated/python_*.cpp
//
// 一些需要注意的地方：
//
//    - 注意 [重载顺序的重要性]
//      重载的顺序很重要。一组输入参数可能与多个参数规范绑定；我们总是选择PythonArgParser中的第一个规范。
//      但是，在编写重载时（例如在native_functions.yaml中），您不必担心写入它们的顺序，因为代码生成逻辑始终为重载给出一个规范的顺序，
//      其中Tensor重载在Scalar重载之前。这一逻辑在tools/autograd/gen_python_functions.py的sort_declarations函数中实现。
//
//    - 零维张量（例如，torch.tensor(2)）同时与Scalar和Tensor绑定，除非它们需要梯度（在这种情况下，它们只绑定到Tensor）。
//

#include <pybind11/pytypes.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/python_dimname.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_symnode.h>
#include <torch/csrc/utils/six.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <c10/core/SymFloat.h>
#include <c10/core/SymNodeImpl.h>

#include <c10/core/DispatchKeySet.h>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

// 检查是否是标量对象的辅助函数
inline bool THPUtils_checkScalar(PyObject* obj) {
#ifdef USE_NUMPY
  // 如果是NumPy标量，则返回true
  if (torch::utils::is_numpy_scalar(obj)) {
    return true;
  }
#endif
  // 否则返回false
  return false;
}
#endif
  // 返回一个布尔值，指示对象是否为浮点数、长整数、复数、torch 的符号整数、符号浮点数或符号布尔数
  return PyFloat_Check(obj) || PyLong_Check(obj) || PyComplex_Check(obj) ||
      torch::is_symint(py::handle(obj)) ||
      torch::is_symfloat(py::handle(obj)) || torch::is_symbool(py::handle(obj));
}

namespace torch {

// 声明一个函数，用于确定是否应将数字视为张量的参数
bool should_allow_numbers_as_tensors(const std::string& name);

// 参数类型的枚举
enum class ParameterType {
  TENSOR,
  SCALAR,
  INT64,
  SYM_INT,
  DOUBLE,
  COMPLEX,
  TENSOR_LIST,
  INT_LIST,
  GENERATOR,
  BOOL,
  STORAGE,
  PYOBJECT,
  SCALARTYPE,
  LAYOUT,
  MEMORY_FORMAT,
  DEVICE,
  STREAM,
  STRING,
  DIMNAME,
  DIMNAME_LIST,
  QSCHEME,
  FLOAT_LIST,
  SCALAR_LIST,
  SYM_INT_LIST,
  DISPATCH_KEY_SET
};

// 函数参数和签名结构体的声明
struct FunctionParameter;
struct FunctionSignature;
struct PythonArgs;

// 包含按声明顺序绑定的 Python 参数
template <int N>
struct ParsedArgs {
  ParsedArgs() : args() {}
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  PyObject* args[N];
};

// PythonArgParser 包含一组有效的签名，通常作为全局变量且不可变
struct PYBIND11_EXPORT PythonArgParser {
  // 构造函数，接受格式字符串列表和可追踪性标志
  explicit PythonArgParser(
      const std::vector<std::string>& fmts,
      bool traceable = false);

  // 解析 Python 参数的模板方法，带有 self 参数
  template <int N>
  inline PythonArgs parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      ParsedArgs<N>& dst);

  // 解析 Python 参数的模板方法，不带 self 参数
  template <int N>
  inline PythonArgs parse(PyObject* args, PyObject* kwargs, ParsedArgs<N>& dst);

  // 解析 Python 参数的方法，只带 self 参数且无需解析参数
  inline PythonArgs parse(PyObject* self, ParsedArgs<0>& dst);

  // 获取非隐藏签名的格式化字符串列表
  std::vector<std::string> get_signatures() const;

 private:
  // 打印错误信息并终止程序的方法
  [[noreturn]] void print_error(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      PyObject* parsed_args[]);
  
  // 检查是否已弃用特定函数签名
  void check_deprecated(const FunctionSignature& signature);

  // 原始的 Python 参数解析方法
  PythonArgs raw_parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      PyObject* parsed_args[]);

  std::vector<FunctionSignature> signatures_;  // 存储函数签名列表
  std::string function_name;  // 函数名
  size_t max_args;  // 最大参数数目
  bool traceable;  // 是否可追踪
};

// FunctionSignature 表示 Python 函数的单个有效签名，一旦构造即为不可变
// 其包含的数据可以被多个调用并发访问
// FunctionSignature 结构体表示函数签名信息
struct FunctionSignature {
  // 构造函数，根据给定格式字符串和索引创建函数签名对象
  explicit FunctionSignature(const std::string& fmt, int index);

  // 解析函数参数，返回是否解析成功的布尔值
  bool parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      PyObject* dst[],
      // 重载参数的引用列表
      std::vector<PyObject*>& overloaded_args,
      // 是否抛出异常的标志
      bool raise_exception);

  // 将函数签名信息转换成字符串表示
  std::string toString() const;

  std::string name;                    // 函数名
  std::vector<FunctionParameter> params; // 函数参数列表
  size_t min_args;                     // 最小参数个数
  size_t max_args;                     // 最大参数个数
  size_t max_pos_args;                 // 最大位置参数个数
  int index;                           // 索引
  bool hidden;                         // 是否隐藏
  bool deprecated;                     // 是否已弃用
};

// PythonArgs 包含实际调用的绑定Python参数及其匹配的函数签名的引用
struct PythonArgs {
  // 返回是否存在torch函数重载
  bool has_torch_function();

  // 获取函数名
  std::string get_func_name();

  std::vector<PyObject*> overloaded_args; // 重载参数列表
  FunctionSignature signature;           // 函数签名对象
};

// FunctionParameter 是Python函数的单个形式参数
struct FunctionParameter {
  // 构造函数，根据给定格式字符串和是否只接受关键字创建函数参数对象
  FunctionParameter(const std::string& fmt, bool keyword_only);

  // 检查函数参数是否符合要求
  bool check(
      PyObject* obj,
      // 重载参数的引用列表
      std::vector<PyObject*>& overloaded_args,
      int argnum,
      int64_t* failed_idx = nullptr);

  // 设置默认字符串值
  void set_default_str(const std::string& str);
  
  // 返回参数类型名称
  std::string type_name() const;

  ParameterType type_;                    // 参数类型
  bool optional;                          // 是否可选
  bool allow_none;                        // 是否允许None值
  bool keyword_only;                      // 是否只接受关键字
  bool allow_numbers_as_tensors = false;  // 是否允许数字作为张量
  int size;                               // 大小
  std::string name;                       // 参数名
  // python_name作为原始PyObject*可能会泄漏，但这些通常仅由静态对象持有，而且当析构时已经可以调用Py_Finalize。
  PyObject* python_name;                  // Python参数名
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  at::SmallVector<PyObject*, 5> numpy_python_names; // numpy的Python参数名列表
  at::Scalar default_scalar;              // 默认标量值
  std::vector<int64_t> default_intlist;   // 默认整数列表
  std::string default_string;             // 默认字符串
  union {
    bool default_bool;                    // 默认布尔值
    int64_t default_int;                  // 默认整数值
    double default_double;                // 默认双精度浮点数值
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    double default_complex[2];            // 默认复数数组，参见Scalar
    at::ScalarType default_scalartype;    // 默认标量类型
    at::Layout default_layout;            // 默认布局
  };
  std::string default_value;              // 默认值字符串
};

// PythonArgParser 类用于解析Python参数
class PythonArgParser {
public:
  // 解析Python参数并返回解析结果
  template <int N>
  inline PythonArgs parse(
      PyObject* self,
      PyObject* args,
      PyObject* kwargs,
      ParsedArgs<N>& dst);

  // 解析Python参数并返回解析结果
  template <int N>
  inline PythonArgs parse(
      PyObject* args,
      PyObject* kwargs,
      ParsedArgs<N>& dst);

  // 解析Python参数并返回解析结果
  inline PythonArgs parse(PyObject* self, ParsedArgs<0>& dst);

  // 返回是否存在torch函数重载
  inline bool has_torch_function();

  // 获取函数名
  inline std::string get_func_name();
};

// TODO: 可能返回MaybeOwned
inline at::Tensor PythonArgs::tensor(int i) {
  // 检查参数列表中的第 i 个参数是否为 THPVariable 类型的对象
  if (args[i] && THPVariable_CheckExact(args[i])) {
    // 如果是，则解包该对象为 Tensor 类型并返回
    return THPVariable_Unpack(args[i]);
  }
  // 否则调用慢速路径函数 tensor_slow 处理
  return tensor_slow(i);
}

inline std::optional<at::Tensor> PythonArgs::optionalTensor(int i) {
  // 获取第 i 个参数作为 Tensor 对象
  at::Tensor t = tensor(i);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 检查 Tensor 是否已定义
  if (t.defined()) {
    // 如果已定义，则返回包含该 Tensor 的 std::optional 对象
    return t;
  } else {
    // 否则返回空的 std::optional 对象
    return c10::nullopt;
  }
}

inline at::Scalar PythonArgs::scalar(int i) {
  // 检查第 i 个参数是否为空
  if (!args[i])
    // 如果为空，则返回预定义的标量值
    return signature.params[i].default_scalar;
  // 否则调用慢速路径函数 scalar_slow 处理
  return scalar_slow(i);
}

inline std::vector<at::Scalar> PythonArgs::scalarlist(int i) {
  // 检查第 i 个参数是否为空
  if (!args[i])
    // 如果为空，则返回空的标量值向量
    return std::vector<at::Scalar>();
  // 检查参数是否为元组类型
  auto tuple = six::isTuple(args[i]);
  // 获取参数对象的引用
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 根据参数类型获取其大小
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  // 创建一个大小为 size 的标量值向量
  std::vector<at::Scalar> res(size);
  // 遍历参数对象的每个元素
  for (const auto idx : c10::irange(size)) {
    // 根据元组或列表类型获取对象
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // 调用慢速路径函数 scalar_slow 处理获取的对象，并存储到结果向量中
    res[idx] = scalar_slow(obj);
  }
  // 返回处理后的标量值向量
  return res;
}

inline at::Scalar PythonArgs::scalarWithDefault(
    int i,
    const at::Scalar& default_scalar) {
  // 检查第 i 个参数是否为空
  if (!args[i])
    // 如果为空，则返回默认标量值
    return default_scalar;
  // 否则调用慢速路径函数 scalar_slow 处理
  return scalar_slow(i);
}

inline std::optional<at::Scalar> PythonArgs::scalarOptional(int i) {
  // 检查第 i 个参数是否为空
  if (!args[i])
    // 如果为空，则返回空的 std::optional 对象
    return c10::nullopt;
  // 否则调用慢速路径函数 scalar_slow 处理
  return scalar_slow(i);
}

inline std::vector<at::Tensor> PythonArgs::tensorlist(int i) {
  // 检查第 i 个参数是否为空
  if (!args[i])
    // 如果为空，则返回空的 Tensor 对象向量
    return std::vector<at::Tensor>();
  // 检查参数是否为元组类型
  auto tuple = six::isTuple(args[i]);
  // 获取参数对象的引用
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 根据参数类型获取其大小
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  // 创建一个大小为 size 的 Tensor 对象向量
  std::vector<at::Tensor> res(size);
  // 遍历参数对象的每个元素
  for (const auto idx : c10::irange(size)) {
    // 根据元组或列表类型获取对象
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // 将获取的对象解包为 Tensor 对象，并存储到结果向量中
    res[idx] = THPVariable_Unpack(obj);
  }
  // 返回处理后的 Tensor 对象向量
  return res;
}

inline torch::List<std::optional<at::Tensor>> PythonArgs::
    list_of_optional_tensors(int i) {
  // 检查第 i 个参数是否为空
  if (!args[i])
    // 如果为空，则返回空的 torch::List 对象
    return torch::List<std::optional<at::Tensor>>();
  // 检查参数是否为元组类型
  auto tuple = six::isTuple(args[i]);
  // 获取参数对象的引用
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  // 根据参数类型获取其大小
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  // 创建一个 torch::List 对象用于存储 std::optional<at::Tensor> 对象
  torch::List<std::optional<at::Tensor>> res;
  res.reserve(size);
  // 遍历参数对象的每个元素
  for (const auto idx : c10::irange(size)) {
    // 根据元组或列表类型获取对象
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // 将获取的对象解包为 Tensor 对象，并存储到 torch::List 中
    res.push_back(THPVariable_Unpack(obj));
  }
  // 返回处理后的 torch::List 对象
  return res;
}
// 返回一个长度为N的Tensor数组，从Python参数中提取数据
inline std::array<at::Tensor, N> PythonArgs::tensorlist_n(int i) {
  // 创建一个空的Tensor数组
  auto res = std::array<at::Tensor, N>();
  // 如果参数为空指针，则直接返回空数组
  if (!args[i])
    return res;
  // 判断参数是否为元组
  auto tuple = six::isTuple(args[i]);
  // 尝试将参数转换为元组对象
  THPObjectPtr arg = six::maybeAsTuple(args[i]);
  // 根据参数类型确定其大小
  auto size = tuple ? PyTuple_GET_SIZE(arg.get()) : PyList_GET_SIZE(arg.get());
  // 如果大小与N不符，则抛出类型错误异常
  if (size != N) {
    throw TypeError("expected tuple of %d elements but got %d", N, (int)size);
  }
  // 遍历元组或列表，将每个元素解包为Tensor并存入结果数组
  for (const auto idx : c10::irange(size)) {
    PyObject* obj = tuple ? PyTuple_GET_ITEM(arg.get(), idx)
                          : PyList_GET_ITEM(arg.get(), idx);
    // 由于参数解析器已检查过类型，这里直接转换为Tensor，无需再次检查
    res[idx] = THPVariable_Unpack(obj);
  }
  // 返回包含Tensor数组的结果
  return res;
}

// 返回一个int64_t类型的向量，从Python参数中提取数据并带有默认值
inline std::vector<int64_t> PythonArgs::intlist(int i) {
  return intlistWithDefault(i, signature.params[i].default_intlist);
}

// 将c10::SymInt类型转换为PyObject指针
inline PyObject* toPyObject(const c10::SymInt& symint) {
  // 如果symint是符号型数据，转换为Python对象并返回
  if (symint.is_symbolic()) {
    auto r = py::cast(symint).release().ptr();
    TORCH_INTERNAL_ASSERT(r);
    return r;
  } else {
    // 否则尝试将symint作为int64_t转换为Python对象并返回
    auto m = symint.maybe_as_int();
    return THPUtils_packInt64(*m);
  }
}

// 抛出intlist异常，指示参数解析错误
inline void throw_intlist_exception(
    const torch::PythonArgs* args,
    size_t i,
    PyObject* obj,
    size_t idx,
    const std::exception& e = python_error()) {
  // 构造异常信息字符串，描述参数解析错误的详细信息
  std::string error = strlen(e.what())
      ? e.what()
      : std::string("type must be ") + args->signature.params[i].type_name() +
          ", but got " + Py_TYPE(obj)->tp_name;
  // 抛出类型错误异常，提醒用户出错的参数位置及具体错误信息
  throw TypeError(
      "%s(): argument '%s' failed to unpack the object at pos %zu with error \"%s\"",
      args->signature.name.c_str(),
      args->signature.params[i].name.c_str(),
      idx + 1,
      error.c_str());
}

// 返回一个c10::SymInt类型的向量，从Python参数中提取符号型整数数据
inline std::vector<c10::SymInt> PythonArgs::symintlist(int i) {
  // 如果参数为空，则返回带有默认值的符号整数向量
  if (!args[i]) {
    return c10::fmap(signature.params[i].default_intlist, [](int64_t di) {
      return c10::SymInt(di);
    });
  }

  // 获取预期大小
  const auto size1 = signature.params[i].size;
  // 如果大小大于0且参数可以解包为长整型，则直接返回包含单个SymInt的向量
  if (size1 > 0 && THPUtils_checkLong(args[i])) {
    return std::vector<c10::SymInt>(
        size1, c10::SymInt(THPUtils_unpackLong(args[i])));
  }

  // 如果大小大于0且参数是符号型整数，则直接返回包含单个SymInt的向量
  if (size1 > 0 && torch::is_symint(py::handle(args[i]))) {
    auto si = py::handle(args[i]).cast<c10::SymInt>();
    return std::vector<c10::SymInt>(size1, si);
  }

  // 其他情况，处理参数为元组或列表的情况
  PyObject* arg = args[i];
  auto tuple = PyTuple_Check(arg);
  // 根据参数类型确定其大小
  const auto size2 = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  // 预留足够的空间以容纳所有符号整数
  std::vector<c10::SymInt> res;
  res.reserve(size2);
  // 遍历元组或列表，将每个元素转换为符号型整数并存入结果向量
  for (const auto idx : c10::irange(size2)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);

    // torch.Size的元素在追踪过程中是Tensor，在转换为IntArrayRef之前需要记录额外信息
    // 如果需要进行追踪，并且正在进行跟踪，并且 obj 是 THPVariable 类型
    if (traceable && jit::tracer::isTracing() && THPVariable_Check(obj)) {
      // 将 obj 解包为 THPVariable 引用 var
      auto& var = THPVariable_Unpack(obj);
      // 将 var 中的数据作为整数数组元素暂存
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          signature.params[i].name, size2, idx, var);
      try {
        // 尝试将 var 转换为 int64_t 类型，并添加到结果列表中
        res.emplace_back(var.item<int64_t>());
        // 继续下一轮循环处理下一个参数
        continue;
      } catch (std::exception& e) {
        // 如果转换过程中出现异常，抛出整数列表异常
        throw_intlist_exception(this, i, obj, idx, e);
      }
      // 继续下一轮循环处理下一个参数
      continue;
    } else {
      // 在 try / catch 外部将张量转换为标量，
      // 这样张量子类的异常就不会被捕获。
      if (THPUtils_checkLongExact(obj)) {
        // 对于普通数字的快速路径
        try {
          // 将 obj 解包为 long 类型并添加到结果列表中
          res.emplace_back(THPUtils_unpackLong(obj));
        } catch (std::exception& e) {
          // 如果转换过程中出现异常，抛出整数列表异常
          throw_intlist_exception(this, i, obj, idx, e);
        }
      } else if (THPVariable_Check(obj)) {
        // 将 obj 解包为 THPVariable 引用 var
        auto& var = THPVariable_Unpack(obj);
        // 如果 var 不是标量或者不是整数类型
        if (var.numel() != 1 ||
            !at::isIntegralType(
                var.dtype().toScalarType(), /*include_bool*/ true)) {
          // 抛出整数列表异常
          throw_intlist_exception(this, i, obj, idx);
        }
        // 获取 var 的标量值
        auto scalar = var.item();
        // 检查 scalar 是否是整数（不包括 bool 类型）
        TORCH_CHECK(scalar.isIntegral(/*include bool*/ false));
        // 将 scalar 转换为符号整数并添加到结果列表中
        res.push_back(scalar.toSymInt());
      } else {
        // 尝试将 obj 解包为索引类型
        try {
          // 如果 obj 是符号整数类型，则添加到结果列表中
          if (is_symint(py::handle(obj))) {
            res.push_back(py::handle(obj).cast<c10::SymInt>());
          } else {
            // 否则将 obj 解包为索引类型并添加到结果列表中
            res.emplace_back(THPUtils_unpackIndex(obj));
          }
        } catch (std::exception& e) {
          // 如果转换过程中出现异常，抛出整数列表异常
          throw_intlist_exception(this, i, obj, idx, e);
        }
      }
    }
  }

  // 返回最终的结果列表
  return res;
// 如果参数列表中第 i 个参数为空指针，则返回默认的整数列表
inline std::vector<int64_t> PythonArgs::intlistWithDefault(
    int i,
    std::vector<int64_t> default_intlist) {
  if (!args[i]) // 检查参数列表中第 i 个位置是否为空
    return default_intlist; // 返回默认的整数列表

  PyObject* arg = args[i]; // 获取参数列表中第 i 个位置的 Python 对象
  const auto size1 = signature.params[i].size; // 获取参数签名中第 i 个参数的尺寸信息

  // 如果 size1 大于 0 并且 arg 是一个长整型，则创建一个包含 size1 个元素的整数列表
  if (size1 > 0 && THPUtils_checkLong(arg)) {
    return std::vector<int64_t>(size1, THPUtils_unpackLong(arg));
  }

  // 如果 size1 大于 0 并且 arg 是 torch::is_symint 标识的符号整数类型，则创建一个包含 size1 个元素的整数列表
  if (size1 > 0 && torch::is_symint(py::handle(arg))) {
    return std::vector<int64_t>(
        size1,
        py::handle(arg).cast<c10::SymInt>().guard_int(__FILE__, __LINE__));
  }

  auto tuple = PyTuple_Check(arg); // 检查 arg 是否为元组
  const auto size2 = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg); // 获取 arg 中元素的数量

  std::vector<int64_t> res(size2); // 创建一个大小为 size2 的整数列表

  // 遍历 arg 中的每个元素
  for (const auto idx : c10::irange(size2)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx); // 获取元组或列表中的具体元素

    // 如果跟踪开启且 obj 是 THPVariable 类型，并且在追踪期间，将 torch.Size 元素转换为标量
    if (traceable && jit::tracer::isTracing() && THPVariable_Check(obj)) {
      auto& var = THPVariable_Unpack(obj); // 解包 THPVariable 对象
      jit::tracer::ArgumentStash::stashIntArrayRefElem(
          signature.params[i].name, size2, idx, var); // 存储追踪数组引用元素的参数信息
      try {
        res[idx] = var.item<int64_t>(); // 尝试获取 var 的整数项并存储到结果列表中
        continue;
      } catch (std::exception& e) {
        throw_intlist_exception(this, i, obj, idx, e); // 捕获异常并抛出整数列表异常
      }
    } else {
      // 将 tensor 转换为标量，避免在 try / catch 外部转换时捕获 Tensor 子类异常
      if (THPUtils_checkLongExact(obj)) {
        // 对于普通数字的快速路径
        try {
          res[idx] = THPUtils_unpackLong(obj); // 尝试解包 obj 并将结果存储到结果列表中
        } catch (std::exception& e) {
          throw_intlist_exception(this, i, obj, idx, e); // 捕获异常并抛出整数列表异常
        }
      } else if (torch::is_symint(py::handle(obj))) {
        // 如果 obj 是 torch::is_symint 标识的符号整数类型，将其转换为整数并存储到结果列表中
        res[idx] = py::cast<c10::SymInt>(py::handle(obj))
                       .guard_int(__FILE__, __LINE__);
      } else if (THPVariable_Check(obj)) {
        auto& var = THPVariable_Unpack(obj); // 解包 THPVariable 对象
        // 如果 var 的元素数量不是 1 或者其数据类型不是整数类型（包括布尔类型），抛出整数列表异常
        if (var.numel() != 1 ||
            !at::isIntegralType(
                var.dtype().toScalarType(), /*include_bool*/ true)) {
          throw_intlist_exception(this, i, obj, idx);
        }
        res[idx] = var.item<int64_t>(); // 将 var 的整数项存储到结果列表中
      } else {
        try {
          res[idx] = THPUtils_unpackIndex(obj); // 尝试解包 obj 并将结果存储到结果列表中
        } catch (std::exception& e) {
          throw_intlist_exception(this, i, obj, idx, e); // 捕获异常并抛出整数列表异常
        }
      }
    }
  }
  return res; // 返回填充后的整数列表
}

// 如果参数列表中第 i 个参数为空指针，则返回空的可选整数数组
inline c10::OptionalArray<int64_t> PythonArgs::intlistOptional(int i) {
  if (!args[i]) { // 检查参数列表中第 i 个位置是否为空
    return {}; // 返回空的可选整数数组
  }
  return intlist(i); // 否则返回第 i 个参数的整数列表
}

// 如果参数列表中第 i 个参数为空指针，则返回空的可选符号整数数组
inline c10::OptionalArray<c10::SymInt> PythonArgs::symintlistOptional(int i) {
  if (!args[i]) { // 检查参数列表中第 i 个位置是否为空
    return {}; // 返回空的可选符号整数数组
  }
  return symintlist(i); // 否则返回第 i 个参数的符号整数列表
}
// 返回一个包含双精度浮点数列表的向量
inline std::vector<double> PythonArgs::getDoublelist(int i) {
  // 获取第 i 个参数
  PyObject* arg = args[i];
  // 检查参数是否为元组
  auto tuple = PyTuple_Check(arg);
  // 根据参数类型获取其大小
  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  // 创建一个大小为 size 的双精度浮点数向量
  std::vector<double> res(size);
  // 遍历元组或列表中的每个元素
  for (const auto idx : c10::irange(size)) {
    // 获取元组或列表中的第 idx 个对象
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    try {
      // 将对象解包成双精度浮点数并存储到结果向量中
      res[idx] = THPUtils_unpackDouble(obj);
    } catch (const std::exception&) {
      // 捕获异常并抛出类型错误，显示具体错误信息
      throw TypeError(
          "%s(): argument '%s' must be %s, but found element of type %s at pos %zu",
          signature.name.c_str(),
          signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(),
          Py_TYPE(obj)->tp_name,
          idx + 1);
    }
  }
  // 返回存储双精度浮点数的向量
  return res;
}

// 返回一个可选的双精度浮点数数组
inline c10::OptionalArray<double> PythonArgs::doublelistOptional(int i) {
  // 如果参数为空，则返回空数组
  if (!args[i]) {
    return {};
  }
  // 否则返回获取的双精度浮点数列表
  return this->getDoublelist(i);
}

// 返回一个双精度浮点数数组
inline std::vector<double> PythonArgs::doublelist(int i) {
  // 如果参数为空，则返回空数组
  if (!args[i]) {
    return {};
  }
  // 否则返回获取的双精度浮点数列表
  return this->getDoublelist(i);
}

// 返回一个可选的 DispatchKeySet
inline std::optional<c10::DispatchKeySet> PythonArgs::toDispatchKeySetOptional(
    int i) {
  // 如果参数为空，则返回空
  if (!args[i]) {
    return {};
  }
  // 将参数转换为 DispatchKeySet 并返回
  return py::cast<c10::DispatchKeySet>(py::handle(args[i]));
}

// 返回具有默认标量类型的标量类型
inline at::ScalarType PythonArgs::scalartypeWithDefault(
    int i,
    at::ScalarType default_scalartype) {
  // 如果参数为空，则返回默认的标量类型
  if (!args[i])
    return default_scalartype;
  // 否则返回获取的标量类型
  return scalartype(i);
}

// 根据对象返回对应的标量类型
inline at::ScalarType toScalarType(PyObject* obj) {
  // 根据不同的对象类型返回对应的标量类型
  if (obj == (PyObject*)&PyFloat_Type) {
    return at::ScalarType::Double;
  }
  if (obj == (PyObject*)&PyBool_Type) {
    return at::ScalarType::Bool;
  }
  if (obj == (PyObject*)&PyLong_Type) {
    return at::ScalarType::Long;
  }
  if (obj == (PyObject*)&PyComplex_Type) {
    return at::ScalarType::ComplexDouble;
  }
  // 否则返回对象对应的标量类型
  return reinterpret_cast<THPDtype*>(obj)->scalar_type;
}

// 返回指定索引处的标量类型
inline at::ScalarType PythonArgs::scalartype(int i) {
  // 如果参数为空，则返回默认的标量类型或者系统默认的标量类型
  if (!args[i]) {
    auto scalartype = signature.params[i].default_scalartype;
    return (scalartype == at::ScalarType::Undefined)
        ? torch::tensors::get_default_scalar_type()
        : scalartype;
  }
  // 否则返回参数对应的标量类型
  PyObject* obj = args[i];
  return toScalarType(obj);
}

// 返回一个可选的标量类型
inline std::optional<at::ScalarType> PythonArgs::scalartypeOptional(int i) {
  // 如果参数为空，则返回空
  if (!args[i])
    return c10::nullopt;
  // 否则返回获取的标量类型
  return scalartype(i);
}

// 根据对象返回布局类型
inline at::Layout toLayout(PyObject* obj) {
  // 将对象转换为布局类型并返回
  const auto layout = reinterpret_cast<THPLayout*>(obj);
  return layout->layout;
}

// 返回指定索引处的布局类型
inline at::Layout PythonArgs::layout(int i) {
  // 如果参数为空，则返回默认的布局类型
  if (!args[i])
    return signature.params[i].default_layout;
  // 否则返回参数对应的布局类型
  return toLayout(args[i]);
}

// 返回具有默认布局类型的布局类型
inline at::Layout PythonArgs::layoutWithDefault(
    int i,
    at::Layout default_layout) {
  // 如果参数为空，则返回默认的布局类型
  if (!args[i])
    return default_layout;
  // 否则返回获取的布局类型
  return layout(i);
}

// 返回一个可选的布局类型
inline std::optional<at::Layout> PythonArgs::layoutOptional(int i) {
  // 如果参数为空，则返回空
  if (!args[i])
    return c10::nullopt;
  // 否则返回获取的布局类型
  return layout(i);
}
inline at::Device toDevice(PyObject* obj) {
  // 检查参数是否为THPDevice类型
  if (THPDevice_Check(obj)) {
    // 将PyObject转换为THPDevice指针
    const auto device = reinterpret_cast<THPDevice*>(obj);
    // 返回THPDevice结构体中的device成员
    return device->device;
  }
  // 检查参数是否为整数类型
  if (THPUtils_checkLong(obj)) {
    // 解包获取整数值
    const auto device_index = THPUtils_unpackLong(obj);
    // 检查设备索引是否非负
    TORCH_CHECK(device_index >= 0, "Device index must not be negative");
    // 如果注册了PrivateUse1后端，则返回PrivateUse1设备
    if (c10::is_privateuse1_backend_registered()) {
      return at::Device(
          c10::DeviceType::PrivateUse1,
          static_cast<c10::DeviceIndex>(device_index));
    }
    // 否则返回CUDA设备
    return at::Device(
        c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device_index));
  }
  // 解包获取字符串，创建对应的设备对象
  const std::string& device_str = THPUtils_unpackString(obj);
  return at::Device(device_str);
}

inline at::Device PythonArgs::device(int i) {
  // 如果参数为空，则返回默认设备
  if (!args[i]) {
    return torch::tensors::get_default_device();
  }
  // 否则调用toDevice函数获取设备
  return toDevice(args[i]);
}

inline at::Device PythonArgs::deviceWithDefault(
    int i,
    const at::Device& default_device) {
  // 如果参数为空，则返回默认设备
  if (!args[i])
    return default_device;
  // 否则调用device函数获取设备
  return device(i);
}

inline std::optional<at::Device> PythonArgs::deviceOptional(int i) {
  // 如果参数为空，则返回空的optional
  if (!args[i])
    return c10::nullopt;
  // 否则调用device函数获取设备，返回optional
  return device(i);
}

inline at::Dimname PythonArgs::dimname(int i) {
  // 断言参数不为空
  TORCH_INTERNAL_ASSERT(args[i] != nullptr);
  // 解析参数为Dimname对象
  return THPDimname_parse(args[i]);
}

inline std::vector<at::Dimname> parseDimnameList(PyObject* arg) {
  // 检查参数是否为元组类型
  auto tuple = PyTuple_Check(arg);
  // 根据类型获取参数的长度
  auto size = tuple ? PyTuple_GET_SIZE(arg) : PyList_GET_SIZE(arg);
  std::vector<at::Dimname> res;
  res.reserve(size);
  // 遍历参数，解析每个元素为Dimname对象
  for (const auto idx : c10::irange(size)) {
    PyObject* obj =
        tuple ? PyTuple_GET_ITEM(arg, idx) : PyList_GET_ITEM(arg, idx);
    res.push_back(THPDimname_parse(obj));
  }
  return res;
}

inline std::optional<std::vector<at::Dimname>> PythonArgs::
    toDimnameListOptional(int i) {
  // 如果参数为空，则返回空的optional
  if (!args[i])
    return c10::nullopt;
  // 否则调用parseDimnameList函数获取Dimname对象列表，返回optional
  return parseDimnameList(args[i]);
}

inline std::vector<at::Dimname> PythonArgs::dimnamelist(int i) {
  // 断言参数不为空
  TORCH_INTERNAL_ASSERT(args[i]);
  // 获取参数对象
  PyObject* arg = args[i];
  // 获取参数的大小
  auto size = signature.params[i].size;
  // 断言参数大小为0或1
  TORCH_INTERNAL_ASSERT(size == 0 || size == 1);
  // 如果大小为1且参数为Dimname对象，则返回包含单个Dimname对象的向量
  if (size == 1 && THPUtils_checkDimname(arg)) {
    return {THPDimname_parse(arg)};
  }
  // 否则调用parseDimnameList函数获取Dimname对象列表
  return parseDimnameList(arg);
}

inline at::MemoryFormat PythonArgs::memoryformat(int i) {
  // 如果参数为空，则返回Contiguous内存格式
  if (!args[i])
    return at::MemoryFormat::Contiguous;
  // 检查参数是否为THPMemoryFormat类型
  TORCH_CHECK(
      THPMemoryFormat_Check(args[i]),
      "memory_format arg must be an instance of the torch.memory_format");
  // 将参数转换为THPMemoryFormat指针，获取内存格式
  const auto memory_format = reinterpret_cast<THPMemoryFormat*>(args[i]);
  return memory_format->memory_format;
}

inline std::optional<at::MemoryFormat> PythonArgs::memoryformatOptional(int i) {
  // 如果参数为空，则返回空的optional
  if (!args[i])
    return c10::nullopt;
  // 否则调用memoryformat函数获取内存格式，返回optional
  return memoryformat(i);
}

inline at::QScheme PythonArgs::toQScheme(int i) {
  // 如果参数为空，则
    // 返回一个常量值，表示每个张量都有单独的仿射量化参数
    return at::kPerTensorAffine;
  // 使用 TORCH_CHECK 宏检查参数是否符合要求，要求 args[i] 必须是 torch.qscheme 的实例
  TORCH_CHECK(
      THPQScheme_Check(args[i]),
      "qscheme arg must be an instance of the torch.qscheme");
  // 将 args[i] 转换为 THPQScheme 指针，并获取其 qscheme 属性
  const auto qscheme = reinterpret_cast<THPQScheme*>(args[i]);
  // 返回 THPQScheme 对象的 qscheme 属性值
  return qscheme->qscheme;
}

// 返回第 i 个参数作为字符串，如果参数为空则返回默认字符串
inline std::string PythonArgs::string(int i) {
  return stringWithDefault(i, signature.params[i].default_string);
}

// 返回第 i 个参数作为字符串，如果参数为空则返回默认字符串
inline std::string PythonArgs::stringWithDefault(
    int i,
    const std::string& default_str) {
  if (!args[i])
    return default_str;
  // 解包字符串参数并返回
  return THPUtils_unpackString(args[i]);
}

// 返回第 i 个参数作为可选的字符串，如果参数为空则返回空值
inline std::optional<std::string> PythonArgs::stringOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  // 解包字符串参数并返回
  return THPUtils_unpackString(args[i]);
}

// 返回第 i 个参数作为字符串视图，如果参数为空则返回默认字符串视图
inline c10::string_view PythonArgs::stringView(int i) {
  return stringViewWithDefault(i, signature.params[i].default_string);
}

// 返回第 i 个参数作为字符串视图，如果参数为空则返回默认字符串视图
inline c10::string_view PythonArgs::stringViewWithDefault(
    int i,
    const c10::string_view default_str) {
  if (!args[i])
    return default_str;
  // 解包字符串视图参数并返回
  return THPUtils_unpackStringView(args[i]);
}

// 返回第 i 个参数作为可选的字符串视图，如果参数为空则返回空值
inline std::optional<c10::string_view> PythonArgs::stringViewOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  // 解包字符串视图参数并返回
  return THPUtils_unpackStringView(args[i]);
}

// 返回第 i 个参数作为 int64_t 类型的整数，如果参数为空则返回默认整数
inline int64_t PythonArgs::toInt64(int i) {
  if (!args[i])
    return signature.params[i].default_int;
  
  // 在跟踪模式下，将参数值存储以供追踪
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::IntType::get());
  }
  
  // 如果参数是符号整数，返回符号整数值
  if (torch::is_symint(py::handle(args[i]))) {
    return py::cast<c10::SymInt>(py::handle(args[i]))
        .guard_int(__FILE__, __LINE__);
  }
  
  // 否则，解包长整数参数并返回
  return THPUtils_unpackLong(args[i]);
}

// 返回第 i 个参数作为符号整数，如果参数为空则返回默认符号整数
inline c10::SymInt PythonArgs::toSymInt(int i) {
  if (!args[i]) {
    return c10::SymInt(signature.params[i].default_int);
  }

  // 在跟踪模式下，将参数值存储以供追踪
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::IntType::get());
  }

  // 返回参数作为符号整数
  return py::cast<c10::SymInt>(py::handle(args[i]));
}

// 返回第 i 个参数作为符号布尔值，如果参数为空则返回默认符号布尔值
inline c10::SymBool PythonArgs::toSymBool(int i) {
  if (!args[i]) {
    return c10::SymBool(signature.params[i].default_bool);
  }
  
  // 在跟踪模式下，将参数值存储以供追踪
  if (traceable && jit::tracer::isTracing() && THPVariable_Check(args[i])) {
    auto& var = THPVariable_Unpack(args[i]);
    jit::tracer::ArgumentStash::stashValue(
        signature.params[i].name, idx, var, c10::BoolType::get());
  }

  // 返回参数作为符号布尔值
  return py::cast<c10::SymBool>(py::handle(args[i]));
}

// 返回第 i 个参数作为 int64_t 类型的整数，如果参数为空则返回默认整数
inline int64_t PythonArgs::toInt64WithDefault(int i, int64_t default_int) {
  if (!args[i])
    return default_int;
  // 返回第 i 个参数作为 int64_t 类型的整数
  return toInt64(i);
}

// 返回第 i 个参数作为可选的 int64_t 类型的整数，如果参数为空则返回空值
inline std::optional<int64_t> PythonArgs::toInt64Optional(int i) {
  if (!args[i])
    return c10::nullopt;
  // 返回第 i 个参数作为 int64_t 类型的整数
  return toInt64(i);
}

// 返回第 i 个参数作为可选的符号整数，如果参数为空则返回空值
inline std::optional<c10::SymInt> PythonArgs::toSymIntOptional(int i) {
  if (!args[i])
    return c10::nullopt;
  // 返回第 i 个参数作为符号整数
  return toSymInt(i);
}

// 返回第 i 个参数作为可选的布尔值，如果参数为空则返回空值
inline std::optional<bool> PythonArgs::toBoolOptional(int i) {
  if (!args[i]) {
    return c10::nullopt;
  }
  // 返回参数作为布尔值
  return toBool(i);
}

// 返回第 i 个参数作为可选的双精度浮点数，如果参数为空则返回空值
inline std::optional<double> PythonArgs::toDoubleOptional(int i) {
  if (!args[i]) {
    // 如果参数为空，返回空值
    return c10::nullopt;
  }
  // 否则，返回参数作为双精度浮点数
  return THPUtils_unpackDouble(args[i]);
}
    return c10::nullopt;
  }


// 如果条件不满足，返回一个空的 std::optional 对象
return c10::nullopt;



  return toDouble(i);


// 如果条件满足，调用 toDouble 函数并返回其结果
return toDouble(i);
// 返回第 i 个参数对应的双精度浮点数，若参数为空则返回默认值
inline double PythonArgs::toDouble(int i) {
  if (!args[i])
    return signature.params[i].default_double;
  // 如果参数是符号浮点数，则转换为 C++ SymFloat 类型，并进行浮点数保护
  if (torch::is_symfloat(py::handle(args[i]))) {
    return py::cast<c10::SymFloat>(py::handle(args[i]))
        .guard_float(__FILE__, __LINE__);
  }
  // 如果参数是符号整数，则转换为 double 类型，并进行整数保护
  if (torch::is_symint(py::handle(args[i]))) {
    return static_cast<double>(py::cast<c10::SymInt>(py::handle(args[i]))
                                   .guard_int(__FILE__, __LINE__));
  }
  // 否则，调用 THPUtils_unpackDouble 函数进行解包并返回结果
  return THPUtils_unpackDouble(args[i]);
}

// 返回第 i 个参数对应的布尔值，若参数为空则返回默认值
inline bool PythonArgs::toBool(int i) {
  if (!args[i])
    return signature.params[i].default_bool;
  // 如果参数是符号布尔值，则转换为 C++ SymBool 类型，并进行布尔值保护
  if (torch::is_symbool(py::handle(args[i]))) {
    return py::cast<c10::SymBool>(py::handle(args[i]))
        .guard_bool(__FILE__, __LINE__);
  }
  // 否则，直接比较参数是否为 Python 中的 True
  return args[i] == Py_True;
}

// 返回第 i 个参数对应的双精度浮点数，若参数为空则返回指定的默认值
inline double PythonArgs::toDoubleWithDefault(int i, double default_double) {
  if (!args[i])
    return default_double;
  // 调用 toDouble 函数获取参数对应的双精度浮点数
  return toDouble(i);
}

// 返回第 i 个参数对应的复数，若参数为空则返回默认复数值
inline c10::complex<double> PythonArgs::toComplex(int i) {
  if (!args[i])
    return *(reinterpret_cast<const c10::complex<double>*>(
        signature.params[i].default_complex));
  // 调用 THPUtils_unpackComplexDouble 函数解包复数参数并返回结果
  return THPUtils_unpackComplexDouble(args[i]);
}

// 返回第 i 个参数对应的复数，若参数为空则返回指定的默认复数值
inline c10::complex<double> PythonArgs::toComplexWithDefault(
    int i,
    c10::complex<double> default_value) {
  if (!args[i])
    return default_value;
  // 调用 toComplex 函数获取参数对应的复数
  return toComplex(i);
}

// 返回第 i 个参数对应的布尔值，若参数为空则返回指定的默认布尔值
inline bool PythonArgs::toBoolWithDefault(int i, bool default_bool) {
  if (!args[i])
    return default_bool;
  // 调用 toBool 函数获取参数对应的布尔值
  return toBool(i);
}

// 检查第 i 个参数是否为 None
inline bool PythonArgs::isNone(int i) {
  return args[i] == nullptr;
}

// 返回第 i 个参数对应的生成器对象，若参数为空则返回空的 std::optional 对象
inline std::optional<at::Generator> PythonArgs::generator(int i) {
  if (!args[i])
    return c10::nullopt;
  // 返回 args[i] 对应的生成器对象
  return reinterpret_cast<THPGenerator*>(args[i])->cdata;
}

// 返回第 i 个参数对应的 Storage 对象，若参数为空则返回默认的空 Storage 对象
inline at::Storage PythonArgs::storage(int i) {
  if (!args[i])
    return at::Storage();
  // 调用 createStorage 函数创建并返回对应的 Storage 对象
  return createStorage(args[i]);
}

// 返回第 i 个参数对应的 Storage 对象，并获取存储的标量类型和是否为类型化存储
inline at::Storage PythonArgs::storage(
    int i,
    at::ScalarType& storage_scalar_type,
    bool& is_typed_storage) {
  at::Storage storage;
  if (!args[i]) {
    // 若参数为空，则返回默认的空 Storage 对象，并设置标量类型和类型化存储的标志
    storage = at::Storage();
    is_typed_storage = false;
    storage_scalar_type = at::ScalarType::Undefined;
  } else {
    // 否则，调用 createStorageGetType 函数创建 Storage 对象，并获取相关信息
    std::tie(storage, storage_scalar_type, is_typed_storage) =
        createStorageGetType(args[i]);
  }
  return storage;
}

// 返回第 i 个参数对应的流对象，若参数为空则返回默认的 CPU 流对象
inline c10::Stream PythonArgs::stream(int i) {
  if (!args[i])
    return c10::Stream(
        c10::Stream::Default::DEFAULT, c10::Device(c10::DeviceType::CPU, -1));
  // 若参数不为空但不是预期的流对象类型，则抛出 TypeError 异常
  if (!THPStream_Check(args[i])) {
    throw TypeError(
        "expected Stream object. Got '%s'", Py_TYPE(args[i])->tp_name);
  }
  // 解包参数为流对象并返回对应的 C++ Stream 对象
  return c10::Stream::unpack3(
      ((THPStream*)args[i])->stream_id,
      static_cast<c10::DeviceIndex>(((THPStream*)args[i])->device_index),
      static_cast<c10::DeviceType>(((THPStream*)args[i])->device_type));
}

// 返回第 i 个参数对应的 Python 对象，若参数为空则返回 Py_None
inline PyObject* PythonArgs::pyobject(int i) {
  if (!args[i])
    return Py_None;
  // 否则，直接返回参数对应的 Python 对象
  return args[i];
}
/*
 * 处理 __torch_function__ 的重载，如果我们知道有重载的参数。
 * 存储在 r.overloaded_args 中的所有对象必须有一个 __torch_function__ 的实现，
 * 并且参数必须按优先顺序排列。优先顺序按照传递给函数签名的顺序从左到右排列，
 * 但子类始终优先于超类。
 *
 * 如果调用 __torch_function__ 的结果是 NotImplemented，
 * 则按优先顺序调用下一个实现。如果所有参数的 __torch_function__ 都返回
 * NotImplemented，则在 Python 中引发 TypeError。
 *
 * 假设 overloaded_args 至少有一个条目。所有条目必须有一个 __torch_function__
 * 属性，该属性解析为一个可调用对象，接受 torch API 函数、参数元组和关键字参数
 * 字典作为输入。
 *
 * 在调用此函数之前调用 PythonArgs::has_torch_function 可以验证是否存在有效参数。
 * 如果没有这样做，则必须特别注意确保有用于 __torch_function__ 的参数。
 *
 * 查看 torch._overrides.handle_torch_function 以获取纯 Python 实现中的等效代码。
 *
 * 'r' 是从 PythonArgParser::parse 返回的解析后的 PythonArgs 实例。
 * 'args' 是指向 torch API 函数的参数元组的引用。
 * 'kwargs' 是指向 torch API 函数的关键字参数字典的引用。
 * 'torch_api' 是指向 Python 中 torch API 命名空间的引用。
 * 'torch_api_function' 是指向原始 torch 方法的引用，通常我们可以使用 torch_api 和 func_name 获取 torch_api_function。
 * 'overloaded_args' 是具有重载 __torch_function__ 的参数。
 * 'func_name' 是原始 torch 方法的名称。
 * TODO: 我们可以使用不同的名称来代替 'handle_torch_function'，而不是重载。
 */
// 用于具有参数的 Tensor 方法。
auto handle_torch_function(
    PythonArgs& r,  // PythonArgs 实例，包含解析后的参数信息
    PyObject* self, // self 对象，通常是 Python 中的类实例
    PyObject* args, // 参数元组的引用
    PyObject* kwargs, // 关键字参数字典的引用
    PyObject* torch_api, // torch API 命名空间的引用
    const char* module_name, // 模块名称的 C 字符串
    const char* func_name_override = nullptr) -> PyObject*; // 可选的函数名覆盖

// 用于需要解析 Python 参数的函数。
auto handle_torch_function(
    PythonArgs& r, // PythonArgs 实例，包含解析后的参数信息
    PyObject* args, // 参数元组的引用
    PyObject* kwargs, // 关键字参数字典的引用
    PyObject* torch_api, // torch API 命名空间的引用
    const char* module_name, // 模块名称的 C 字符串
    const char* func_name_override = nullptr) -> PyObject*; // 可选的函数名覆盖

// 用于没有参数解析的函数。
auto handle_torch_function(
    PyObject* self, // self 对象，通常是 Python 中的类实例
    const std::string& func_name, // 函数名的 C++ 字符串
    PyObject* args = nullptr, // 参数元组的引用，默认为空
    PyObject* kwargs = nullptr, // 关键字参数字典的引用，默认为空
    // 定义一个名为 torch_api 的 PyObject 指针变量，初始化为 THPVariableClass
    PyObject* torch_api = THPVariableClass,
    // 定义一个名为 module_name 的常量引用，类型为 std::string，初始化为 "torch.Tensor"
    const std::string& module_name = "torch.Tensor") -> PyObject*;
    // 函数声明结束，该函数返回一个 PyObject 指针，并接受 THPVariableClass 和一个名为 module_name 的常量引用作为参数
// 用于在 C++ 中创建的函数，例如不使用 PythonArgParser 获取 overloaded_args 的 Torch 自定义操作，如 Torch custom op。
enum class TorchFunctionName { TorchFunction, TorchDispatch };

// 处理没有使用 PythonArgParser 获取 overloaded_args 的 Torch 函数。
auto TORCH_PYTHON_API handle_torch_function_no_python_arg_parser(
    at::ArrayRef<PyObject*> overloaded_args,  // 重载的参数列表
    PyObject* args,  // 参数元组
    PyObject* kwargs,  // 关键字参数字典
    const char* func_name,  // 函数名称
    PyObject* torch_api_function,  // Torch API 函数对象
    const char* module_name,  // 模块名称
    TorchFunctionName torch_function_name = TorchFunctionName::TorchFunction)  // Torch 函数名称，默认为 TorchFunction
    -> PyObject*;  // 返回 PyObject 指针

// 用于获取 Tensor 属性的 getter 函数
auto handle_torch_function_getter(
    THPVariable* self,  // THPVariable 类型的 self 对象
    const std::string& property_name)  // 属性名称
    -> PyObject*;  // 返回 PyObject 指针

// 用于设置 Tensor 属性的 setter 函数
auto handle_torch_function_setter(
    THPVariable* self,  // THPVariable 类型的 self 对象
    const std::string& property_name,  // 属性名称
    PyObject* value)  // 设置的值
    -> int;  // 返回整数

// 用于处理 __getitem__ 和 __setitem__ 的函数
auto handle_torch_function_indexing(
    PyObject* self,  // self 对象
    PyObject* index,  // 索引
    PyObject* val = nullptr)  // 可选的值，默认为空
    -> PyObject*;  // 返回 PyObject 指针

/*
 * 检查输入的 obj 是否为 Tensor 类型或其子类或重载类型。
 * 如果类型定义了 __torch_function__，也返回 true。
 * 否则返回 false。如果类不是 torch.Tensor，并且定义了 __torch_function__，
 * 则将 obj 添加到 overloaded_args 中。
 *
 * 'obj': 要检查的输入参数
 * 'overloaded_args': 要追加重载参数的向量
 */
bool is_tensor_and_append_overloaded(
    PyObject* obj,  // 要检查的输入对象
    std::vector<PyObject*>* overloaded_args);  // 要追加重载参数的向量

/*
 * 检查输入的 obj 是否为 Tensor 列表或 Tensor 元组类型。
 * 首先检查 obj 是否为元组或列表类型，如果是，则遍历每个元素并检查是否为 Tensor 类型或其子类或重载类型。
 * 同时，将重载参数追加到 overloaded_args 中。
 *
 * 'obj': 要检查的输入参数
 * 'overloaded_args': 要追加重载参数的向量
 * 'argnum': 要检查的函数的总参数数目
 * 'throw_error': 如果列表或元组中的任何元素不是 Tensor 类型或重载类型，是否抛出错误
 */
bool is_tensor_list_and_append_overloaded(
    PyObject* obj,  // 要检查的输入对象
    std::vector<PyObject*>* overloaded_args,  // 要追加重载参数的向量
    int argnum,  // 函数的总参数数目
    bool throw_error);  // 是否在出现错误时抛出异常

/* 给定一个肯定是 Tensor 并且肯定是重载的参数的参数，将其追加到重载参数列表中。
 * 在有 PyObject 对象并且你知道它肯定是 Tensor 并且是重载的情况下使用此函数，而不是 is_tensor_and_append_overloaded。
 *
 * 'overloaded_args': 要追加重载参数的向量
 * 'obj': 被重载的输入 Tensor
 */
void append_overloaded_tensor(
    std::vector<PyObject*>* overloaded_args,  // 要追加重载参数的向量
    PyObject* obj);  // 被重载的输入 Tensor
/* 给定一个类型参数，该参数肯定是重载的类型，并将其附加到重载参数列表中。
 * 仅在 __torch_dispatch__ 中使用，其中我们操作具有 __torch_dispatch__ 类方法的类。
 *
 * 'overloaded_args': 用于附加重载类型的向量
 * 'obj': 具有 __torch_dispatch__ 类方法的输入类
 */
void append_overloaded_type(
    std::vector<PyObject*>* overloaded_args,
    PyObject* obj);

} // namespace torch
```