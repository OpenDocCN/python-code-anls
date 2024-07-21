# `.\pytorch\torch\csrc\jit\tensorexpr\eval.h`

```
#pragma once


// 声明指令，确保头文件只被编译一次
#include <cmath>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/Logging.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/tensorexpr/types.h>
#include <torch/csrc/jit/tensorexpr/var_substitutor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class InterpValue {
 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 默认构造函数，初始化为整数类型
  InterpValue() : dtype_(kInt) {
    Intvalues.push_back(0);
  }

  // 根据指定的数据类型和值构造 InterpValue 对象
  template <typename T>
  InterpValue(Dtype dtype, T v) : dtype_(dtype) {
#define TYPE_CASE(Type, Name)  \
  if (dtype == k##Name) {      \
    Name##values.push_back(v); \
    return;                    \
  }
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    // 如果指定的数据类型不支持，则抛出异常
    throw unsupported_dtype();
  }

  // 根据数据类型 Name 和值 v 构造 InterpValue 对象
#define VALUE_CTOR(Type, Name)            \
  InterpValue(Type v) : dtype_(k##Name) { \
    Name##values.push_back(v);            \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_CTOR);
#undef VALUE_CTOR

  // 构造函数，用于特定的 c10::quint8 数据类型
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit InterpValue(c10::quint8 v) : dtype_(kQUInt8) {
    QUInt8values.emplace_back(v.val_);
  }

  // 构造函数，用于特定的 c10::qint8 数据类型
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  explicit InterpValue(c10::qint8 v) : dtype_(kQInt8) {
    QInt8values.emplace_back(v.val_);
  }

  // 根据值的向量 v 构造 InterpValue 对象
#define VALUE_VEC_CTOR(Type, Name)        \
  InterpValue(const std::vector<Type>& v) \
      : dtype_(Dtype(k##Name, v.size())), Name##values(v) {}
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_VEC_CTOR);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VALUE_VEC_CTOR(c10::quint8, QUInt8)
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  VALUE_VEC_CTOR(c10::qint8, QInt8)
#undef VALUE_VEC_CTOR

  // 返回当前值转换为指定类型 T 的值
  template <typename T>
  T as() const;

  // 返回当前值作为向量的形式
  template <typename T>
  const std::vector<T>& as_vec() const;

  // 返回整数值
  int64_t intValue() const;

  // 返回数据类型
  Dtype dtype() const {
    return dtype_;
  }

 private:
  Dtype dtype_;

  // 存储不同类型值的向量
#define VALUE_STORAGE(Type, Name) std::vector<Type> Name##values;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_STORAGE);
  VALUE_STORAGE(c10::qint8, QInt8);
  VALUE_STORAGE(c10::quint8, QUInt8);
#undef VALUE_STORAGE

  void* ptr;
};

// 返回特定类型的值的转换函数模板定义
#define VALUE_AS_DISPATCH(Type, Name)         \
  template <>                                 \
  inline Type InterpValue::as<Type>() const { \
    if (dtype_ != k##Name) {                  \
      throw unsupported_dtype();              \
    }                                         \
    return Name##values[0];                   \
  }
#define VALUE_AS_DISPATCH(Type, Name)                             \
  template <>                                                     \
  inline const Type& InterpValue::as<Type>() const {              \
    if (dtype_.scalar_type() != ScalarType::Name) {               \
      throw unsupported_dtype();                                  \
    }                                                             \
    return Name##val_;                                            \
  }

// 定义了一个模板函数，用于返回特定类型的值
// 根据模板参数 Type，返回相应的值，如果当前数据类型不匹配，则抛出异常
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_AS_DISPATCH);
VALUE_AS_DISPATCH(c10::quint8, QUInt8);
VALUE_AS_DISPATCH(c10::qint8, QInt8);
#undef VALUE_AS_DISPATCH

#define VALUE_AS_VEC_DISPATCH(Type, Name)                             \
  template <>                                                         \
  inline const std::vector<Type>& InterpValue::as_vec<Type>() const { \
    if (dtype_.scalar_type() != ScalarType::Name) {                   \
      throw unsupported_dtype();                                      \
    }                                                                 \
    return Name##values;                                              \
  }

// 定义了另一个模板函数，用于返回特定类型的向量
// 根据模板参数 Type，返回相应的向量，如果当前数据类型不匹配，则抛出异常
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, VALUE_AS_VEC_DISPATCH);
VALUE_AS_VEC_DISPATCH(c10::quint8, QUInt8);
VALUE_AS_VEC_DISPATCH(c10::qint8, QInt8);
#undef VALUE_AS_VEC_DISPATCH

template <typename Type>
auto underlyingValue(Type x) {
  return x;
}

// 特化模板函数 underlyingValue，用于获取 c10::quint8 类型的底层值
template <>
inline auto underlyingValue<c10::quint8>(c10::quint8 x) {
  return x.val_;
}

// 特化模板函数 underlyingValue，用于获取 c10::qint8 类型的底层值
template <>
inline auto underlyingValue<c10::qint8>(c10::qint8 x) {
  return x.val_;
}

// 实现了一个通用的原始比特转换函数
template <typename To, typename From>
To raw_bitcast(const From& src) {
  TORCH_CHECK(sizeof(To) == sizeof(From), "Invalid bitcast invocation");
  To storage;
  std::memcpy(&storage, &src, sizeof(To));
  return reinterpret_cast<To&>(storage);
}

// 声明一个简单的 IR 评估器实现类
class SimpleIREvaluatorImpl;
class TORCH_API SimpleIREvaluator : public CodeGen {
 public:
  SimpleIREvaluator(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func");

  ~SimpleIREvaluator() override;

  void call(const std::vector<CallArg>& args) override;
  void call_raw(const std::vector<void*>& args) override;

  // 模板函数，用于简化调用 IR 评估器对象的操作
  template <typename... Ts>
  void operator()(const Ts&... ts) {
    std::vector<CallArg> args({CallArg(ts)...});
    call(args);
  }

  // 将变量绑定到表达式的函数
  void bindVar(VarPtr v, ExprPtr e);
  // 获取当前评估器的值
  InterpValue value() const;

 private:
  // 绑定参数到指定的缓冲区
  void bindArg(const BufferArg& buf, void* data);
  // 展开内部函数调用
  void expand_intrinsics() {
    GenericIntrinsicsExpander intrinsics_expander;
    apply_mutator(&intrinsics_expander);
  }

  std::unique_ptr<SimpleIREvaluatorImpl> impl_;
};

// 表达式评估器模板类
template <class CodeGenType>
class ExprEval {
 public:
  using BufferArg = CodeGen::BufferArg;
  using CallArg = CodeGen::CallArg;

  // 构造函数，接受表达式和缓冲区参数，并初始化数据类型
  template <typename... Ts>
  ExprEval(const ExprHandle& expr, Ts... ts)
      : ExprEval(expr, {BufferArg(ts)...}) {}

  // 构造函数，接受表达式和完整的缓冲区参数列表，并初始化数据类型
  ExprEval(const ExprHandle& expr, const std::vector<BufferArg>& buffer_args)
      : dtype_(expr.dtype()) {
    std::vector<BufferArg> buffer_args_extended = buffer_args;
    BufHandle ret_buf("ret_val", {1}, dtype_);
    // 创建一个空的表达式向量
    std::vector<ExprHandle> indices;
    // 创建一个整数常量表达式，表示零
    ExprHandle zero = IntImm::make(0);
    // 遍历返回缓冲区的每个维度
    for (size_t i = 0; i < ret_buf.ndim(); i++) {
      // 将零添加到索引表达式向量中
      indices.push_back(zero);
    }
    // 创建一个存储语句，用于将表达式写入返回缓冲区的指定位置
    StmtPtr store_stmt = Store::make(ret_buf, indices, expr);
    // 将返回缓冲区添加到扩展的缓冲区参数列表中
    buffer_args_extended.emplace_back(ret_buf);
    // 创建一个新的代码生成器对象，并重置为存储语句和扩展缓冲区参数列表
    codegen_.reset(new CodeGenType(store_stmt, buffer_args_extended));
  }

  // 模板函数，用于调用operator()，支持任意参数数量
  template <typename... Ts>
  void operator()(Ts... ts) {
    call(ts...);
  }

  // 当调用以std::vector<CallArg>作为参数时，调用这个重载函数
  void operator()(const std::vector<CallArg>& call_args) {
    call(call_args);
  }

  // 将变量与表达式绑定，使用VarPtr和ExprPtr
  void bindVar(VarPtr v, ExprPtr e) {
    codegen_->bindVar(v, e);
  }

  // 将变量与表达式绑定，使用VarHandle和ExprHandle
  void bindVar(const VarHandle& v, const ExprHandle& e) {
    codegen_->bindVar(v.node(), e.node());
  }

  // 模板函数，用于调用call()，支持任意参数数量
  template <typename... Ts>
  void call(Ts... ts) {
    call({CallArg(ts)...});
  }

  // 调用函数，接受std::vector<CallArg>作为参数，处理dtype_.scalar_type()的不同情况
  void call(const std::vector<CallArg>& call_args) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 创建一个扩展的调用参数向量，从给定的调用参数向量初始化
    std::vector<CallArg> call_args_extended = call_args;
    // 使用dtype_.scalar_type()进行开关选择
    switch (dtype_.scalar_type()) {
// 定义宏 TYPE_CASE，用于处理不同的标量类型情况
#define TYPE_CASE(Type, Name)                    \
  case ScalarType::Name: {                       \
    // 创建长度为 1 的 Type 类型向量 ret_val_arg
    std::vector<Type> ret_val_arg(1);            \
    // 将 ret_val_arg 封装成 CallArg 对象并添加到 call_args_extended
    call_args_extended.push_back(CallArg(ret_val_arg)); \
    // 调用 codegen_ 对象的 call 方法
    codegen_->call(call_args_extended);           \
    // 将 ret_val_arg[0] 封装成 InterpValue 类型并赋给 ret_value_
    ret_value_ = InterpValue(ret_val_arg[0]);    \
  } break;
      // NOLINTNEXTLINE(modernize-use-emplace)
      // 对于所有标量类型，展开 TYPE_CASE 宏处理
      AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
      // NOLINTNEXTLINE(modernize-use-emplace)
      // 处理 c10::quint8 类型的情况
      TYPE_CASE(c10::quint8, QUInt8);
      // NOLINTNEXTLINE(modernize-use-emplace)
      // 处理 c10::qint8 类型的情况
      TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE

      // 处理布尔类型的情况
      case ScalarType::Bool: {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        // 创建长度为 1 的 unsigned char 类型向量 ret_val_arg
        std::vector<unsigned char> ret_val_arg(1);
        // 将 ret_val_arg.data() 封装成 CallArg 对象并添加到 call_args_extended
        call_args_extended.emplace_back(ret_val_arg.data());
        // 调用 codegen_ 对象的 call 方法
        codegen_->call(call_args_extended);
        // 将 ret_val_arg[0] 强制转换为 bool 类型后封装成 InterpValue 类型并赋给 ret_value_
        ret_value_ = InterpValue((bool)ret_val_arg[0]);
      } break;
      // 处理未支持的标量类型，抛出异常
      default:
        throw unsupported_dtype();
    }
  }

  // 使用原始参数列表调用 codegen_ 对象的 call_raw 方法
  void call_raw(const std::vector<void*>& args) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 创建 args 的扩展版本 args_extended
    std::vector<void*> args_extended = args;
    // 根据标量类型处理不同情况
    switch (dtype_.scalar_type()) {
#define TYPE_CASE(Type, Name)                    \
  case ScalarType::Name: {                       \
    // 创建长度为 1 的 Type 类型向量 ret_val_arg
    std::vector<Type> ret_val_arg(1);            \
    // 将 ret_val_arg.data() 封装成 void* 类型并添加到 args_extended
    args_extended.push_back(ret_val_arg.data()); \
    // 调用 codegen_ 对象的 call_raw 方法
    codegen_->call_raw(args_extended);           \
    // 将 ret_val_arg[0] 封装成 InterpValue 类型并赋给 ret_value_
    ret_value_ = InterpValue(ret_val_arg[0]);    \
  } break;
      // 对于所有标量类型，展开 TYPE_CASE 宏处理
      AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
      // 处理 c10::quint8 类型的情况
      TYPE_CASE(c10::quint8, QUInt8);
      // 处理 c10::qint8 类型的情况
      TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE

      // 处理布尔类型的情况
      case ScalarType::Bool: {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        // 创建长度为 1 的 unsigned char 类型向量 ret_val_arg
        std::vector<unsigned char> ret_val_arg(1);
        // 将 ret_val_arg.data() 封装成 void* 类型并添加到 args_extended
        args_extended.push_back(ret_val_arg.data());
        // 调用 codegen_ 对象的 call_raw 方法
        codegen_->call_raw(args_extended);
        // 将 ret_val_arg[0] 强制转换为 bool 类型后封装成 InterpValue 类型并赋给 ret_value_
        ret_value_ = InterpValue((bool)ret_val_arg[0]);
      } break;
      // 处理未支持的标量类型，抛出异常
      default:
        throw unsupported_dtype();
    }
  }

  // 使用原始参数列表调用 codegen_ 对象的 call_raw 方法，并返回结果类型为 T
  template <typename T>
  T value(const std::vector<void*>& args) {
    call_raw(args);
    // 将 ret_value_ 转换为 T 类型并返回
    return ret_value_.as<T>();
  }

  // 使用变参列表调用 codegen_ 对象的 call 方法，并返回结果类型为 T
  template <typename T, typename... Ts>
  T value(Ts... ts) {
    call(std::forward<Ts>(ts)...);
    // 将 ret_value_ 转换为 T 类型并返回
    return ret_value_.as<T>();
  }

  // 返回 dtype_ 成员变量
  Dtype dtype() {
    return dtype_;
  }

 private:
  Dtype dtype_; // 成员变量，表示数据类型
  std::unique_ptr<CodeGenType> codegen_; // 指向 CodeGenType 对象的独占指针
  InterpValue ret_value_; // 用于存储解释器返回值的对象
};

// 对给定表达式进行评估，如果结果为 int64_t 类型则返回其值的 std::optional 封装
std::optional<int64_t> evalInt(ExprPtr e);

// 使用变量映射 var_mapping 替换表达式 expr 中的变量，并返回替换后的表达式
inline ExprPtr Substitute(ExprPtr expr, const VarMapping& var_mapping) {
  VarSubMutator var_sub(var_mapping);
  return expr->accept_mutator(&var_sub);
}

// 使用变量映射 var_mapping 替换语句 expr 中的变量，并返回替换后的表达式
// 在给定的语句中进行变量替换操作，并返回替换后的新语句指针
inline StmtPtr Substitute(StmtPtr stmt, const VarMapping& var_mapping) {
  // 创建一个变量替换器对象，使用给定的变量映射进行初始化
  VarSubMutator var_sub(var_mapping);
  // 调用语句对象的 accept_mutator 方法，将变量替换器应用到语句上，并返回新的语句指针
  return stmt->accept_mutator(&var_sub);
}

// 创建输入表达式的克隆，并在克隆中使用给定的变量映射替换相应的变量表达式
// 注意：这种方法适用于克隆会重用变量而不会创建新变量的情况，VarMapping 输入将变量作为键
inline ExprPtr SubstituteInClone(ExprPtr expr, const VarMapping& var_mapping) {
  // 创建一个变量替换器对象，使用给定的变量映射进行初始化
  VarSubMutator var_sub(var_mapping);
  // 克隆输入的表达式，并将变量替换器应用到克隆后的表达式上，并返回新的表达式指针
  return Expr::clone(std::move(expr))->accept_mutator(&var_sub);
}

// 创建输入语句的克隆，并在克隆中使用给定的变量映射替换相应的变量表达式
// 注意：这种方法适用于克隆会重用变量而不会创建新变量的情况，VarMapping 输入将变量作为键
inline StmtPtr SubstituteInClone(StmtPtr stmt, const VarMapping& var_mapping) {
  // 创建一个变量替换器对象，使用给定的变量映射进行初始化
  VarSubMutator var_sub(var_mapping);
  // 克隆输入的语句，并将变量替换器应用到克隆后的语句上，并返回新的语句指针
  return Stmt::clone(std::move(stmt))->accept_mutator(&var_sub);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```