# `.\pytorch\torch\csrc\jit\tensorexpr\tensorexpr_init.cpp`

```py
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/utils/pybind.h>
#ifdef USE_CUDA
#include <torch/csrc/jit/tensorexpr/cuda_codegen.h>
#endif
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/lowerings.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

// 定义特化的类型转换器，用于将 ArgValue 转换为 Python 对象
template <>
struct pybind11::detail::type_caster<torch::jit::tensorexpr::ArgValue>
    : public type_caster_base<torch::jit::tensorexpr::ArgValue> {};

// 命名空间别名，用于简化命名空间访问
namespace torch::jit {
using namespace torch::jit::tensorexpr;

// 将 Python 对象转换为 ArgValue 类型
ArgValue convertPyToArgValue(py::handle inp) {
  // 检查输入对象的类型，并根据类型进行转换
  if (py::isinstance<BufHandle>(inp)) {
    return py::cast<BufHandle>(inp);
  } else if (py::isinstance<VarHandle>(inp)) {
    return py::cast<VarHandle>(inp);
  } else if (py::isinstance<py::bool_>(inp)) {
    return py::cast<bool>(inp);
  } else if (py::isinstance<py::float_>(inp)) {
    return py::cast<double>(inp);
  } else if (py::isinstance<py::int_>(inp)) {
    return py::cast<int64_t>(inp);
  } else if (py::isinstance<py::none>(inp)) {
    return ArgNone();
  } else if (py::isinstance<py::list>(inp)) {
    auto l = py::cast<py::list>(inp);
    if (l.empty()) {
      return std::vector<BufHandle>();
    } else if (py::isinstance<py::int_>(l[0])) {
      return py::cast<IntList>(inp);
    } else if (py::isinstance<BufHandle>(l[0])) {
      return py::cast<BufList>(inp);
    } else {
      throw std::runtime_error("vector conversion failed");
    }
  } else {
    throw std::runtime_error("conversion not yet implemented");
  }
}

// 解析 Python 中的 Dtype 对象
Dtype parsePythonDtype(py::handle obj) {
  if (THPDtype_Check(obj.ptr())) {
    return Dtype(reinterpret_cast<THPDtype*>(obj.ptr())->scalar_type);
  } else {
    throw std::runtime_error("expected a torch.dtype instance");
  }
}

// 初始化 TensorExpr 的 Python 绑定
void initTensorExprBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // 在 Python 模块中定义名为 "_te" 的子模块
  auto te = m.def_submodule("_te");

  // 定义 Dtype 类的 Python 包装器，并使用 parsePythonDtype 函数初始化
  auto dtype_class =
      py::class_<Dtype>(te, "Dtype").def(py::init(&parsePythonDtype));
  py::implicitly_convertible<py::object, Dtype>();

  // 为每种标量类型定义静态只读属性
#define DTYPE_SINGLETON_ACCESSOR(ctype, name) \
  dtype_class.def_property_readonly_static(   \
      #name, [](py::object) { return k##name; }); // NOLINT
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DTYPE_SINGLETON_ACCESSOR)

  // 根据不同的标量类型定义初始化函数的 Python 包装器
#define EXPRHANDLE_INIT(ctype, name) \
  .def(py::init([](ctype val) { return name##Imm::make(val); }))
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, EXPRHANDLE_INIT)
#undef EXPRHANDLE_INIT
}
#define EXPRHANDLE_IMPL_CONV(ctype, name) \
  // 定义宏，将类型 ctype 隐式转换为 ExprHandle 类型
  py::implicitly_convertible<ctype, ExprHandle>();

// 取消先前定义的宏 EXPRHANDLE_IMPL_CONV

  te.def(
      "ifThenElse",
      // 定义 ifThenElse 函数，接受三个 ExprHandle 类型的参数，返回条件表达式的结果
      [](const ExprHandle& c, const ExprHandle& t, const ExprHandle& f) {
        return ifThenElse(c, t, f);
      });

  // 定义数学函数 sin，接受一个 ExprHandle 类型参数，返回其正弦值
  te.def("sin", [](const ExprHandle& v1) { return sin(v1); });

  // 定义数学函数 cos，接受一个 ExprHandle 类型参数，返回其余弦值
  te.def("cos", [](const ExprHandle& v1) { return cos(v1); });

  // 定义数学函数 tan，接受一个 ExprHandle 类型参数，返回其正切值
  te.def("tan", [](const ExprHandle& v1) { return tan(v1); });

  // 定义数学函数 asin，接受一个 ExprHandle 类型参数，返回其反正弦值
  te.def("asin", [](const ExprHandle& v1) { return asin(v1); });

  // 定义数学函数 acos，接受一个 ExprHandle 类型参数，返回其反余弦值
  te.def("acos", [](const ExprHandle& v1) { return acos(v1); });

  // 定义数学函数 atan，接受一个 ExprHandle 类型参数，返回其反正切值
  te.def("atan", [](const ExprHandle& v1) { return atan(v1); });

  // 定义数学函数 sinh，接受一个 ExprHandle 类型参数，返回其双曲正弦值
  te.def("sinh", [](const ExprHandle& v1) { return sinh(v1); });

  // 定义数学函数 cosh，接受一个 ExprHandle 类型参数，返回其双曲余弦值
  te.def("cosh", [](const ExprHandle& v1) { return cosh(v1); });

  // 定义数学函数 tanh，接受一个 ExprHandle 类型参数，返回其双曲正切值
  te.def("tanh", [](const ExprHandle& v1) { return tanh(v1); });

  // 定义数学函数 sigmoid，接受一个 ExprHandle 类型参数，返回其 sigmoid 函数值
  te.def("sigmoid", [](const ExprHandle& v1) { return sigmoid(v1); });

  // 定义数学函数 exp，接受一个 ExprHandle 类型参数，返回其指数函数值
  te.def("exp", [](const ExprHandle& v1) { return exp(v1); });

  // 定义数学函数 expm1，接受一个 ExprHandle 类型参数，返回 exp(v1)-1 的值
  te.def("expm1", [](const ExprHandle& v1) { return expm1(v1); });

  // 定义数学函数 abs，接受一个 ExprHandle 类型参数，返回其绝对值
  te.def("abs", [](const ExprHandle& v1) { return abs(v1); });

  // 定义数学函数 log，接受一个 ExprHandle 类型参数，返回其自然对数值
  te.def("log", [](const ExprHandle& v1) { return log(v1); });

  // 定义数学函数 log2，接受一个 ExprHandle 类型参数，返回其以2为底的对数值
  te.def("log2", [](const ExprHandle& v1) { return log2(v1); });

  // 定义数学函数 log10，接受一个 ExprHandle 类型参数，返回其以10为底的对数值
  te.def("log10", [](const ExprHandle& v1) { return log10(v1); });

  // 定义数学函数 log1p，接受一个 ExprHandle 类型参数，返回 log(1 + v1) 的值
  te.def("log1p", [](const ExprHandle& v1) { return log1p(v1); });

  // 定义数学函数 erf，接受一个 ExprHandle 类型参数，返回其误差函数值
  te.def("erf", [](const ExprHandle& v1) { return erf(v1); });

  // 定义数学函数 erfc，接受一个 ExprHandle 类型参数，返回其补误差函数值
  te.def("erfc", [](const ExprHandle& v1) { return erfc(v1); });

  // 定义数学函数 sqrt，接受一个 ExprHandle 类型参数，返回其平方根值
  te.def("sqrt", [](const ExprHandle& v1) { return sqrt(v1); });

  // 定义数学函数 rsqrt，接受一个 ExprHandle 类型参数，返回其倒数平方根值
  te.def("rsqrt", [](const ExprHandle& v1) { return rsqrt(v1); });

  // 定义数学函数 ceil，接受一个 ExprHandle 类型参数，返回其向上取整值
  te.def("ceil", [](const ExprHandle& v1) { return ceil(v1); });

  // 定义数学函数 floor，接受一个 ExprHandle 类型参数，返回其向下取整值
  te.def("floor", [](const ExprHandle& v1) { return floor(v1); });

  // 定义数学函数 round，接受一个 ExprHandle 类型参数，返回其四舍五入值
  te.def("round", [](const ExprHandle& v1) { return round(v1); });

  // 定义数学函数 trunc，接受一个 ExprHandle 类型参数，返回其截断整数值
  te.def("trunc", [](const ExprHandle& v1) { return trunc(v1); });

  // 定义数学函数 frac，接受一个 ExprHandle 类型参数，返回其小数部分值
  te.def("frac", [](const ExprHandle& v1) { return frac(v1); });

  // 定义数学函数 lgamma，接受一个 ExprHandle 类型参数，返回其对数伽玛函数值
  te.def("lgamma", [](const ExprHandle& v1) { return lgamma(v1); });

  // 定义数学函数 isnan，接受一个 ExprHandle 类型参数，返回其是否为 NaN 的布尔值
  te.def("isnan", [](const ExprHandle& v1) { return isnan(v1); });

  // 定义数学函数 atan2，接受两个 ExprHandle 类型参数，返回其反正切值
  te.def("atan2", [](const ExprHandle& v1, const ExprHandle& v2) {
    return atan2(v1, v2);
  });

  // 定义数学函数 pow，接受两个 ExprHandle 类型参数，返回其幂次方值
  te.def("pow", [](const ExprHandle& v1, const ExprHandle& v2) {
    return pow(v1, v2);
  });

  // 定义数学函数 fmod，接受两个 ExprHandle 类型参数，返回其浮点余数值
  te.def("fmod", [](const ExprHandle& v1, const ExprHandle& v2) {
    return fmod(v1, v2);
  });

  // 定义数学函数 remainder，接受两个 ExprHandle 类型参数，返回其余数值
  te.def("remainder", [](const ExprHandle& v1, const ExprHandle& v2) {
    return remainder(v1, v2);
  });

#define EXPRHANDLE_CTOR(ctype, name) \
  // 定义静态方法，将类型 ctype 转换为 ExprHandle 类型
  expr_handle_class.def_static(#ctype, [](ctype v) { return ExprHandle(v); });

// 使用宏 EXPRHANDLE_CTOR 分别对所有标量类型定义构造函数
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            // 如果系统是小端序，遍历传入的 Python 对象 values
            for (const auto& value : values) {
              // 如果值是整数类型
              if (py::isinstance<py::int_>(value)) {
                // 将整数值转换为 int64_t 类型，并添加到 value_ptrs 中
                value_ptrs.emplace_back(value.cast<int64_t>());
              } else {
                // 否则，假设值是 Tensor 类型，获取其数据指针并添加到 value_ptrs 中
                value_ptrs.emplace_back(value.cast<at::Tensor>().data_ptr());
              }
            }
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            // 如果系统是大端序，检查传入的 values 是否与缓冲区参数的大小相同
            if (py::len(values) != self.buffer_args().size()) {
              // 如果不相同，抛出异常，表示输入参数错误
              throw malformed_input("bad args in CodeGen.call function");
            }
            // 遍历传入的 Python 对象 values
            for (size_t i = 0; i < py::len(values); i++) {
              // 获取当前索引处的 value 和对应的缓冲区参数 bufArg
              const auto& value = values[i];
              const auto& bufArg = self.buffer_args()[i];
              // 如果值是整数类型
              if (py::isinstance<py::int_>(value)) {
                // 如果对应的 bufArg 不是变量，抛出异常
                if (!bufArg.isVar()) {
                  throw malformed_input(
                      "Integer variable expected in CodeGen.call function");
                }
                // 根据 bufArg 的数据类型进行类型匹配，并将值转换后添加到 value_ptrs 中
                switch (bufArg.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                    \
  case ScalarType::Name: {                       \
    value_ptrs.emplace_back(value.cast<Type>()); \
    break;                                       \
  }
                  AT_FORALL_INT_TYPES(TYPE_CASE);
                  default:
                    throw unsupported_dtype();
                }
              } else {
                // 否则，假设值是 Tensor 类型，获取其数据指针并添加到 value_ptrs 中
                value_ptrs.emplace_back(value.cast<at::Tensor>().data_ptr());
              }
            }
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
            // 调用 CodeGen 对象的 call 方法，传入解析后的参数值指针
            self.call(value_ptrs);
          })
      .def(
          "call_raw",
          [](CodeGen& self, const py::sequence& values) {
            std::vector<void*> value_ptrs;
            value_ptrs.reserve(py::len(values));
            // 遍历传入的 Python 对象 values
            for (const auto& value : values) {
              // 假设值是 Tensor 类型，将其数据指针重新解释为 void* 类型，并添加到 value_ptrs 中
              // 在 Python 中，Tensor.data_ptr() 返回的是一个整数
              value_ptrs.emplace_back(
                  reinterpret_cast<void*>(value.cast<intptr_t>()));
            }
            // 调用 CodeGen 对象的 call_raw 方法，传入解析后的参数值指针
            self.call_raw(value_ptrs);
          })
      .def(
          "get_code_text",
          [](CodeGen& self, const std::string& attr = "") {
            // 调用 CodeGen 对象的 getCodeText 方法，返回代码文本
            return self.getCodeText(attr);
          },
          py::arg("attr") = "");
  // 在 Python 中定义 SimpleIREvaluator 类，并继承 CodeGen
  py::class_<SimpleIREvaluator, CodeGen>(te, "SimpleIREvaluator"); // NOLINT
#ifdef TORCH_ENABLE_LLVM
  // 在 Python 中定义 LLVMCodeGen 类，并继承 CodeGen（仅在 LLVM 可用时）
  py::class_<LLVMCodeGen, CodeGen>(te, "LLVMCodeGen"); // NOLINT
#endif

  // 在 Python 中定义 BufferArg 类，并绑定构造函数，支持 Tensor、VarHandle、BufHandle 类型作为参数
  py::class_<CodeGen::BufferArg>(te, "BufferArg")
      .def(py::init<Tensor>())
      .def(py::init<const VarHandle&>())
      .def(py::init<const BufHandle&>());

  // 在 Python 中实现隐式类型转换，支持 Tensor、VarHandle、BufHandle 类型自动转换为 BufferArg 类型
  py::implicitly_convertible<Tensor, CodeGen::BufferArg>();
  py::implicitly_convertible<VarHandle, CodeGen::BufferArg>();
  py::implicitly_convertible<BufHandle, CodeGen::BufferArg>();

  // 在 Python 中定义 construct_codegen 函数，用于构建 CodeGen 对象，支持不同类型的 args 参数
  te.def(
      "construct_codegen",
      [](const std::string& name,
         StmtPtr stmt,
         const std::vector<CodeGen::BufferArg>& args) {
        // 根据 name 参数选择合适的 CodeGen 对象类型，构造并返回
        CodeGen* cg = nullptr;
        if (name == "llvm") {
#ifdef TORCH_ENABLE_LLVM
          // 如果编译时启用了 LLVM 支持，则创建 LLVMCodeGen 对象
          cg = new LLVMCodeGen(stmt, args);
#else
          // 如果未启用 LLVM 支持，则抛出运行时错误
          throw std::runtime_error("PyTorch not compiled with LLVM support!");
#endif
        } else if (name == "cuda") {
#ifdef USE_CUDA
          // 如果编译时启用了 CUDA 支持，则创建 CudaCodeGen 对象
          cg = new CudaCodeGen(stmt, args);
#else
          // 如果未启用 CUDA 支持，则抛出运行时错误
          throw std::runtime_error("PyTorch not compiled with CUDA support!");
#endif
        } else if (name == "ir_eval") {
          // 对于 'ir_eval'，创建 SimpleIREvaluator 对象
          cg = new SimpleIREvaluator(stmt, args);
        } else {
          // 对于其他未知的 name，抛出错误提示
          throw std::runtime_error(
              "construct_codegen() expects 'llvm', 'cuda', or 'ir_eval'");
        }
        // 返回相应的代码生成器对象
        return cg;
      });
  // 定义 Python 绑定的函数，用于各种 TensorExpr 的操作
  te.def("annotate_input_shapes", &tensorexpr::annotateInputShapes);
  te.def("remove_unused_self_argument", &tensorexpr::removeUnusedSelfArgument);
  te.def("make_shapes_symbolic", &tensorexpr::makeShapesSymbolic);
  te.def("is_graph_compilable", &tensorexpr::isGraphCompilable);
  te.def("fixup_missing_shape_info", &tensorexpr::fixupMissingShapeInfo);
  te.def("remove_graph_output", &tensorexpr::removeGraphOutput);
  te.def(
      "replace_list_output_with_tuple",
      &tensorexpr::replaceListOutputWithTuple);
  te.def("trim_graph", &tensorexpr::trimGraph);
#ifdef TORCH_ENABLE_LLVM
  // 如果编译时启用了 LLVM 支持，则设置 LLVM 相关的目标三元组
  te.def("set_llvm_target_triple", [](const std::optional<std::string>& val) {
    tensorexpr::LLVMTargetTriple() = val;
  });
  te.def("set_llvm_target_cpu", [](const std::optional<std::string>& val) {
    tensorexpr::LLVMTargetCPU() = val;
  });
  te.def("set_llvm_target_attrs", [](const std::optional<std::string>& val) {
    tensorexpr::LLVMTargetAttrs() = val;
  });
  te.def("set_llvm_aot_workflow", [](bool val) {
    tensorexpr::LLVMAOTWorkflow() = val;
  });
#endif
}

} // namespace torch::jit
```