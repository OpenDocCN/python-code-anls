# `.\pytorch\torch\csrc\jit\tensorexpr\eval.cpp`

```py
// 包含 Torch 的 Tensor Expression 库中的评估器头文件
#include <torch/csrc/jit/tensorexpr/eval.h>

// 包含 Torch 的 JIT 日志头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch 的 Tensor Expression 核心外部函数头文件
#include <torch/csrc/jit/tensorexpr/external_functions_core.h>
// 包含 Torch 的 Tensor Expression 外部函数注册头文件
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

// 包含 C10 的工具类，用于整数范围
#include <c10/util/irange.h>

// Torch 的 JIT 和 Tensor Expression 命名空间
namespace torch::jit::tensorexpr {

// 注册简单 IR 评估器的代码生成器
RegisterCodeGen<SimpleIREvaluator> ir_eval_codegen_reg("simple_ir_eval");

// InterpValue 类的成员函数，返回整数值
int64_t InterpValue::intValue() const {
    // 定义宏 TYPE_CASE，用于处理各种整数类型的情况
#define TYPE_CASE(Type, Name)        \
  if (dtype_ == k##Name) {           \
    return int64_t{Name##values[0]}; \
  }
  AT_FORALL_INT_TYPES(TYPE_CASE);  // 遍历所有整数类型
#undef TYPE_CASE
  // 如果类型不支持，则抛出异常
  throw unsupported_dtype();
  return 0;  // 默认返回值
}

// 模运算的模板函数，适用于整数类型
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type mod_value(
    T lhs,
    T rhs) {
  return lhs % rhs;  // 返回 lhs 对 rhs 求模的结果
}

// 模板函数的特化版本，适用于浮点数类型
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
mod_value(T lhs, T rhs) {
  return std::fmod(lhs, rhs);  // 返回 lhs 对 rhs 求浮点数模的结果
}

// 布尔类型的模运算，不支持该操作，抛出异常
inline bool mod_value(bool lhs, bool rhs) {
  throw std::runtime_error("Attempted modulus of bool");
}

// 整数类型的除法模板函数
template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type div_value(
    T lhs,
    T rhs) {
  TORCH_CHECK(rhs != 0, "Division by zero");  // 检查除数是否为零
  return lhs / rhs;  // 返回 lhs 除以 rhs 的整数结果
}

// 浮点数类型的除法模板函数，忽略浮点数除以零的 UB
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::
    type __ubsan_ignore_float_divide_by_zero__
    div_value(T lhs, T rhs) {
  return lhs / rhs;  // 返回 lhs 除以 rhs 的浮点数结果
}

// 布尔类型的除法，不支持该操作，记录错误并返回 false
inline bool div_value(bool lhs, bool rhs) {
  LOG(FATAL) << "Attempted division of bool";
  return false;
}

// Half 类型的除法实现
inline c10::Half div_value(c10::Half lhs, c10::Half rhs) {
  return lhs / rhs;  // 返回 Half 类型的 lhs 除以 rhs 结果
}

// BFloat16 类型的除法实现
inline c10::BFloat16 div_value(c10::BFloat16 lhs, c10::BFloat16 rhs) {
  return lhs / rhs;  // 返回 BFloat16 类型的 lhs 除以 rhs 结果
}

// SimpleIREvaluatorImpl 类的实现，继承自 IRVisitor
class SimpleIREvaluatorImpl : public IRVisitor {
 public:
  // 默认构造函数，初始化对象
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  SimpleIREvaluatorImpl() = default;

  // 默认析构函数，释放对象
  ~SimpleIREvaluatorImpl() override = default;

  // 绑定缓冲区和指针，用于计算
  void bindBuf(BufPtr buf, void* ptr) {
    GRAPH_DEBUG("Binding ptr ", ptr, " with buf ", buf->name_hint());
    buffer_mapping_[buf] = ptr;  // 将缓冲区和指针映射存储起来
  }

  // 绑定变量和解释值，用于计算
  void bindVar(VarPtr var, const InterpValue& val) {
    eval_context_[var] = val;  // 将变量和解释值存储起来
    GRAPH_DEBUG(
        "Binding value ", val.intValue(), " with var ", var->name_hint());
  }

  // 计算表达式的解释值
  InterpValue evaluateExpr(ExprPtr e) {
    e->accept(this);  // 接受并访问表达式
    return value_;    // 返回计算得到的解释值
  }

  // 获取当前存储的解释值
  InterpValue value() const {
    return value_;  // 返回当前存储的解释值
  }

  // 清空评估器的上下文和缓冲区映射
  void clear() {
    eval_context_.clear();      // 清空变量的解释值
    buffer_mapping_.clear();    // 清空缓冲区映射
    internal_buffers_.clear();  // 清空内部缓冲区
  }

  // 访问 Add 表达式的虚函数实现
  TORCH_API void visit(AddPtr v) override {
    visit_binary_op(v);  // 访问二元加法操作
  }

  // 访问 Sub 表达式的虚函数实现
  TORCH_API void visit(SubPtr v) override {
    visit_binary_op(v);  // 访问二元减法操作
  }

  // 访问 Mul 表达式的虚函数实现
  TORCH_API void visit(MulPtr v) override {
    visit_binary_op(v);  // 访问二元乘法操作
  }

  // 访问 Div 表达式的虚函数实现
  TORCH_API void visit(DivPtr v) override {
    visit_binary_op(v);  // 访问二元除法操作
  }

  // 访问 Mod 表达式的虚函数实现
  TORCH_API void visit(ModPtr v) override {
    visit_binary_op(v);  // 访问二元模运算操作
  }

  // 访问 Max 表达式的虚函数实现，未完整列出
  TORCH_API void visit(MaxPtr v) override {
    // 实现未完整列出
  // 访问给定二元操作节点，处理它的传播 NaN 值的属性
  visit_binary_op(v, v->propagate_nans());
}
TORCH_API void visit(MinPtr v) override {
  // 访问给定最小值操作节点，处理它的传播 NaN 值的属性
  visit_binary_op(v, v->propagate_nans());
}

TORCH_API void visit(AndPtr v) override {
  // 访问给定按位与操作节点
  visit_binary_op(v);
}
TORCH_API void visit(OrPtr v) override {
  // 访问给定按位或操作节点
  visit_binary_op(v);
}
TORCH_API void visit(XorPtr v) override {
  // 访问给定按位异或操作节点
  visit_binary_op(v);
}
TORCH_API void visit(LshiftPtr v) override {
  // 访问给定左移操作节点
  visit_binary_op(v);
}
TORCH_API void visit(RshiftPtr v) override {
  // 访问给定右移操作节点
  visit_binary_op(v);
}

void visit(CompareSelectPtr v) override {
  // 访问给定比较选择操作节点，并处理其比较选择操作
  visit_compare_select_op(v, v->compare_select_op());
}

template <typename T>
typename std::enable_if_t<std::is_floating_point<T>::value, T> max_value(
    T a,
    T b) {
  // 返回两个浮点数中的较大值，处理 NaN 值
  return std::isnan(a) ? a : (std::isnan(b) ? b : (a < b ? b : a));
}

template <typename T>
typename std::enable_if_t<!std::is_floating_point<T>::value, T> max_value(
    T a,
    T b) {
  // 返回两个非浮点数中的较大值
  return a < b ? b : a;
}

template <typename T>
typename std::enable_if_t<std::is_floating_point<T>::value, T> min_value(
    T a,
    T b) {
  // 返回两个浮点数中的较小值，处理 NaN 值
  return std::isnan(a) ? a : (std::isnan(b) ? b : (a < b ? a : b));
}

template <typename T>
typename std::enable_if_t<!std::is_floating_point<T>::value, T> min_value(
    T a,
    T b) {
  // 返回两个非浮点数中的较小值
  return a < b ? a : b;
}

template <typename T>
InterpValue binary_op(
    const InterpValue& lhs,
    const InterpValue& rhs,
    IRNodeType op_type) {
  // 执行二元操作，根据操作类型对左右操作数进行操作，并返回结果
  std::vector<T> lhs_v = lhs.as_vec<T>();
  std::vector<T> rhs_v = rhs.as_vec<T>();
  std::vector<T> result_v(lhs_v.size());
  for (const auto i : c10::irange(lhs_v.size())) {
    switch (op_type) {
      case IRNodeType::kAdd:
        result_v[i] = lhs_v[i] + rhs_v[i];
        break;
      case IRNodeType::kSub:
        result_v[i] = lhs_v[i] - rhs_v[i];
        break;
      case IRNodeType::kMul:
        result_v[i] = lhs_v[i] * rhs_v[i];
        break;
      case IRNodeType::kDiv:
        result_v[i] = div_value(lhs_v[i], rhs_v[i]);
        break;
      case IRNodeType::kMod:
        result_v[i] = mod_value(lhs_v[i], rhs_v[i]);
        break;
      case IRNodeType::kMax:
        result_v[i] = max_value(lhs_v[i], rhs_v[i]);
        break;
      case IRNodeType::kMin:
        result_v[i] = min_value(lhs_v[i], rhs_v[i]);
        break;
      default:
        // 抛出运行时错误，如果操作类型无效
        throw std::runtime_error("invalid operator type");
    }
  }
  return InterpValue(result_v);
}

template <typename T>
InterpValue bitwise_binary_op(
    const InterpValue& lhs,
    const InterpValue& rhs,
    IRNodeType op_type) {
  // 执行按位二元操作，根据操作类型对左右操作数进行按位操作，并返回结果
  std::vector<T> lhs_v = lhs.as_vec<T>();
  std::vector<T> rhs_v = rhs.as_vec<T>();
  std::vector<T> result_v(lhs_v.size());
    for (const auto i : c10::irange(lhs_v.size())) {
      // 遍历左操作数向量的每个元素的索引
      switch (op_type) {
        case IRNodeType::kAnd:
          // 如果操作类型是按位与，计算左右操作数向量对应元素的按位与结果
          result_v[i] = lhs_v[i] & rhs_v[i];
          break;
        case IRNodeType::kOr:
          // 如果操作类型是按位或，计算左右操作数向量对应元素的按位或结果
          result_v[i] = lhs_v[i] | rhs_v[i];
          break;
        case IRNodeType::kXor:
          // 如果操作类型是按位异或，计算左右操作数向量对应元素的按位异或结果
          result_v[i] = lhs_v[i] ^ rhs_v[i];
          break;
        default:
          // 如果操作类型不在预期的枚举中，抛出运行时错误
          throw std::runtime_error("invalid operator type");
      }
    }
    // 返回操作结果的封装值对象
    return InterpValue(result_v);
  }

  template <typename T>
  InterpValue shift_binary_op(
      const InterpValue& lhs,
      const InterpValue& rhs,
      IRNodeType op_type) {
    // 将左右操作数转换为指定类型的向量
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    // 创建用于存储操作结果的向量
    std::vector<T> result_v(lhs_v.size());
    for (const auto i : c10::irange(lhs_v.size())) {
      switch (op_type) {
        case IRNodeType::kLshift: {
          // 如果操作类型是左移，将左操作数向量的元素左移右操作数向量的元素指定位数
          typename std::make_unsigned<T>::type a =
              static_cast<typename std::make_unsigned<T>::type>(lhs_v[i]);
          result_v[i] = a << rhs_v[i];
          break;
        }
        case IRNodeType::kRshift:
          // 如果操作类型是右移，将左操作数向量的元素右移右操作数向量的元素指定位数
          result_v[i] = lhs_v[i] >> rhs_v[i];
          break;
        default:
          // 如果操作类型不在预期的枚举中，抛出运行时错误
          throw std::runtime_error("invalid operator type");
      }
    }
    // 返回操作结果的封装值对象
    return InterpValue(result_v);
  }

  template <typename T, typename R>
  InterpValue compare_select_op(
      const InterpValue& lhs,
      const InterpValue& rhs,
      const InterpValue& retval1,
      const InterpValue& retval2,
      CompareSelectOperation cmp_op) {
    // 将左右操作数及返回值转换为指定类型的向量
    std::vector<T> lhs_v = lhs.as_vec<T>();
    std::vector<T> rhs_v = rhs.as_vec<T>();
    std::vector<R> ret_val1_v = retval1.as_vec<R>();
    std::vector<R> ret_val2_v = retval2.as_vec<R>();
    // 创建用于存储操作结果的返回值向量
    std::vector<R> result_v(lhs_v.size());
    for (const auto i : c10::irange(lhs_v.size())) {
      switch (cmp_op) {
        case CompareSelectOperation::kEQ:
          // 如果操作类型是等于比较，根据左右操作数向量的元素比较结果选择返回值1或返回值2
          result_v[i] = (lhs_v[i] == rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kNE:
          // 如果操作类型是不等于比较，根据左右操作数向量的元素比较结果选择返回值1或返回值2
          result_v[i] = (lhs_v[i] != rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kGT:
          // 如果操作类型是大于比较，根据左右操作数向量的元素比较结果选择返回值1或返回值2
          result_v[i] = (lhs_v[i] > rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kGE:
          // 如果操作类型是大于等于比较，根据左右操作数向量的元素比较结果选择返回值1或返回值2
          result_v[i] = (lhs_v[i] >= rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kLT:
          // 如果操作类型是小于比较，根据左右操作数向量的元素比较结果选择返回值1或返回值2
          result_v[i] = (lhs_v[i] < rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        case CompareSelectOperation::kLE:
          // 如果操作类型是小于等于比较，根据左右操作数向量的元素比较结果选择返回值1或返回值2
          result_v[i] = (lhs_v[i] <= rhs_v[i]) ? ret_val1_v[i] : ret_val2_v[i];
          break;
        default:
          // 如果操作类型不在预期的枚举中，抛出运行时错误
          throw std::runtime_error("invalid operator type");
      }
    }
    // 返回一个 InterpValue 对象，其中包含计算结果的值
    return InterpValue(result_v);
  }

  template <
      typename D,
      // 如果 detail::bin_op_deducer 返回的类型是 void，则启用该模板
      typename std::enable_if<std::is_same<
          decltype(detail::bin_op_deducer(std::declval<D>())),
          void>::value>::type* = nullptr>
  // 访问二元操作符节点的方法
  void visit_binary_op(NodePtr<D> v, bool option = false) {
    // 访问左操作数节点并执行其计算
    v->lhs()->accept(this);
    InterpValue lhs_v = value_;  // 获取左操作数节点的计算结果
    // 访问右操作数节点并执行其计算
    v->rhs()->accept(this);
    InterpValue rhs_v = value_;  // 获取右操作数节点的计算结果
    // 如果左右操作数的数据类型不一致，抛出异常
    if (lhs_v.dtype() != rhs_v.dtype()) {
      throw malformed_input("bad dtype in binary op", v);
    }

    IRNodeType expr_type = v->expr_type();  // 获取表达式的节点类型
    // 如果操作是逻辑与、逻辑或或者逻辑异或
    if (expr_type == IRNodeType::kAnd || expr_type == IRNodeType::kOr ||
        expr_type == IRNodeType::kXor) {
      // 根据左操作数的标量类型进行分类处理
      switch (lhs_v.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                  \
  case ScalarType::Name:                                       \  // 匹配标量类型为 Name 的情况
    value_ = bitwise_binary_op<Type>(lhs_v, rhs_v, expr_type); \  // 调用位运算二进制操作函数，并赋值给 value_
    break;                                                     \  // 结束当前 case
        AT_FORALL_INT_TYPES(TYPE_CASE);                        // 对所有整数类型应用 TYPE_CASE 宏
#undef TYPE_CASE                                                // 取消 TYPE_CASE 宏定义

        case ScalarType::Bool:                                  // 若标量类型为 Bool
          value_ = bitwise_binary_op<unsigned char>(lhs_v, rhs_v, expr_type);  // 使用无符号字符进行位运算二进制操作，并赋值给 value_
          break;                                                 // 结束当前 case
        default:                                                  // 若不匹配以上任何一种情况
          throw unsupported_dtype();                              // 抛出不支持的数据类型异常
      }                                                           // 结束 switch 语句
      return;                                                     // 返回

    }

    if (expr_type == IRNodeType::kLshift || expr_type == IRNodeType::kRshift) {  // 如果表达式类型是左移或右移
      switch (lhs_v.dtype().scalar_type()) {                   // 根据左操作数的标量类型进行 switch 分支
#define TYPE_CASE(Type, Name)                                \
  case ScalarType::Name:                                     \  // 匹配标量类型为 Name 的情况
    value_ = shift_binary_op<Type>(lhs_v, rhs_v, expr_type); \  // 调用位移二进制操作函数，并赋值给 value_
    break;                                                   \  // 结束当前 case
        AT_FORALL_INT_TYPES(TYPE_CASE);                      // 对所有整数类型应用 TYPE_CASE 宏
#undef TYPE_CASE                                              // 取消 TYPE_CASE 宏定义

        case ScalarType::Bool:                                // 若标量类型为 Bool
          value_ = shift_binary_op<unsigned char>(lhs_v, rhs_v, expr_type);  // 使用无符号字符进行位移二进制操作，并赋值给 value_
          break;                                               // 结束当前 case
        default:                                                // 若不匹配以上任何一种情况
          throw unsupported_dtype();                            // 抛出不支持的数据类型异常
      }                                                         // 结束 switch 语句
      return;                                                   // 返回
    }

    switch (lhs_v.dtype().scalar_type()) {                      // 根据左操作数的标量类型进行 switch 分支
#define TYPE_CASE(Type, Name)                          \
  case ScalarType::Name:                               \  // 匹配标量类型为 Name 的情况
    value_ = binary_op<Type>(lhs_v, rhs_v, expr_type); \  // 调用二进制操作函数，并赋值给 value_
    break;                                             \  // 结束当前 case
      AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);  // 对所有标量类型应用 TYPE_CASE 宏，除了 Half 和 BFloat16
#undef TYPE_CASE                                          // 取消 TYPE_CASE 宏定义

      case ScalarType::Bool:                              // 若标量类型为 Bool
        value_ = binary_op<unsigned char>(lhs_v, rhs_v, expr_type);  // 使用无符号字符进行二进制操作，并赋值给 value_
        break;                                             // 结束当前 case
      default:                                              // 若不匹配以上任何一种情况
        throw unsupported_dtype();                          // 抛出不支持的数据类型异常
    }                                                       // 结束 switch 语句
  }

  template <typename T>
  InterpValue compare_select_op_helper(
      const InterpValue& lhs,
      const InterpValue& rhs,
      const InterpValue& retval1,
      const InterpValue& retval2,
      CompareSelectOperation cmp_op) {
    InterpValue value;                                      // 声明一个 InterpValue 类型的变量 value
    switch (retval1.dtype().scalar_type()) {                // 根据返回值1的数据类型进行 switch 分支
#define TYPE_CASE(Type, Name)                                               \
  case ScalarType::Name:                                                    \  // 匹配标量类型为 Name 的情况
    value = compare_select_op<T, Type>(lhs, rhs, retval1, retval2, cmp_op); \  // 调用比较选择操作辅助函数，并赋值给 value
    break;                                                                 \  // 结束当前 case
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);         // 对所有标量类型应用 TYPE_CASE 宏，除了 Bool、Half 和 BFloat16
#undef TYPE_CASE                                                             // 取消 TYPE_CASE 宏定义
      default:                                                              // 若不匹配以上任何一种情况
        throw unsupported_dtype();                                          // 抛出不支持的数据类型异常
    }

    return value;  // 返回 value
  }

  void visit_compare_select_op(
      CompareSelectPtr v,
      CompareSelectOperation cmp_op) {
    v->lhs()->accept(this);                  // 访问比较选择操作的左操作数
    InterpValue lhs_v = value_;              // 将当前值赋给左操作数的 InterpValue
    v->rhs()->accept(this);                  // 访问比较选择操作的右操作数
    InterpValue rhs_v = value_;              // 将当前值赋给右操作数的 InterpValue
    v->ret_val1()->accept(this);             // 访问比较选择操作的第一个返回值
    InterpValue ret_val1_v = value_;         // 将当前值赋给第一个返回值的 InterpValue
    v->ret_val2()->accept(this);             // 访问比较选择操作的第二个返回值
    InterpValue ret_val2_v = value_;         // 将当前值赋给第二个返回值的 InterpValue

    if (lhs_v.dtype() != rhs_v.dtype() ||    // 如果左右操作数的数据类型不同
        ret_val1_v.dtype() != ret_val2_v.dtype()) {  // 或者第一个和第二个返回值的数据类型不同
      throw malformed_input("bad dtype in CompareSelect", v);  // 抛出异常，表明在比较选择操作中出现了错误的数据类型
    }

    switch (lhs_v.dtype().scalar_type()) {    // 根据左操作数的标量类型进行 switch 分支
#define TYPE_CASE(Type, Name)                          \
  case ScalarType::Name:                               \  // 匹配标量类型为 Name 的情况
    value_ = compare_select_op<T, Type>(lhs_v, rhs_v, ret_val1_v, ret_val2_v, cmp_op);  // 调用比较选择操作函数，并赋值给 value_
    break;                                             // 结束当前 case
    value_ = compare_select_op_helper<Type>(           \
        lhs_v, rhs_v, ret_val1_v, ret_val2_v, cmp_op); \
    // 调用一个模板函数 compare_select_op_helper，根据模板参数 Type 和给定的参数执行比较选择操作，
    // 将结果赋值给 value_

    break;
    // 跳出当前循环或者 switch 语句，结束循环执行

    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
    // 宏展开，遍历所有标量类型（包括 Bool、Half 和 BFloat16），执行 TYPE_CASE 宏
    // 这里假设宏展开后会生成对应的代码或者调用，用于处理不同类型的情况
// 定义宏，用于处理未定义的类型情况
#undef TYPE_CASE
default:
  // 抛出不支持的数据类型异常
  throw unsupported_dtype();
}

// 定义宏 IMM_VISIT，用于访问特定类型的立即数节点
#define IMM_VISIT(Type, Name)                     \
TORCH_API void visit(Name##ImmPtr v) override {    \
  // 将立即数节点的值转换为解释器值并保存
  value_ = InterpValue(v->value());                \
}
// 遍历所有标量类型，生成相应的访问函数
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_VISIT);
#undef IMM_VISIT

// 访问块节点的访问函数实现
TORCH_API void visit(BlockPtr v) override {
  // 保存当前作用域，并设置为新的作用域
  BlockPtr last = scope_;
  scope_ = v;
  // 遍历块中的语句并依次进行访问
  for (const StmtPtr& s : v->stmts()) {
    s->accept(this);
  }

  // 查找并清理作用域中的变量评估上下文
  auto it = var_by_scope_.find(v);
  if (it != var_by_scope_.end()) {
    for (const ExprPtr& v : it->second) {
      eval_context_.erase(v);
    }
    var_by_scope_.erase(it);
  }

  // 恢复上一个作用域
  scope_ = last;
}

// 访问变量节点的访问函数实现
TORCH_API void visit(VarPtr v) override {
  // 查找变量在评估上下文中的值
  auto iter = eval_context_.find(v);
  if (iter == eval_context_.end()) {
    // 如果找不到变量，抛出输入格式错误异常
    throw malformed_input("could not find Var in context", v);
  }

  // 将变量的值存储到解释器值中
  value_ = iter->second;
}

// 禁用 undefined 行为检测器以处理特定的数据类型转换问题
// 例如，将负浮点数转换为无符号字符时可能出现越界情况
template <typename SrcType, typename DstType>
std::vector<DstType> castValues(const Dtype& src_dtype, const InterpValue& v)
    __ubsan_ignore_undefined__ {
  // 将解释器值转换为源类型的向量
  const std::vector<SrcType>& src_values = v.as_vec<SrcType>();
  std::vector<DstType> dst_values(src_values.size());
  // 逐个元素进行类型转换
  for (int i = 0; i < src_dtype.lanes(); ++i) {
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    dst_values[i] = static_cast<DstType>(underlyingValue(src_values[i]));
  }
  return dst_values;
}

// 根据源类型执行类型转换的模板函数
template <typename SrcType>
void doCastFromSrc(
    const Dtype& src_dtype,
    const Dtype& dst_dtype,
    const InterpValue& v) {
  // 根据目标数据类型执行不同的转换操作
  switch (dst_dtype.scalar_type()) {
#define DST_TYPE_CASE(Type, Name)                                        \
case ScalarType::Name:                                                 \
  // 调用 castValues 进行类型转换并保存结果到解释器值
  this->value_ = InterpValue(castValues<SrcType, Type>(src_dtype, v)); \
  break;
  // 遍历所有标量类型并生成对应的类型转换处理分支
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DST_TYPE_CASE);
#undef DST_TYPE_CASE
#define DST_TYPE_CASE_QUANT(Type, Name, CppType)                           \
case ScalarType::Name: {                                                 \
  // 将 SrcType 转换为 CppType 类型的向量，再转换为 Type 类型的向量，并保存到解释器值
  std::vector<CppType> vec = castValues<SrcType, CppType>(dst_dtype, v); \
  std::vector<Type> qvec;                                                \
  qvec.reserve(vec.size());                                              \
  for (CppType u : vec) {                                                \
    qvec.emplace_back(u);                                                \
  }                                                                      \
  this->value_ = InterpValue(qvec);                                      \
} break;
  // 遍历量化标量类型并生成对应的类型转换处理分支
  DST_TYPE_CASE_QUANT(c10::quint8, QUInt8, uint8_t)
  DST_TYPE_CASE_QUANT(c10::qint8, QInt8, int8_t)
#undef DST_TYPE_CASE_QUANT
default:
  // 抛出不支持的数据类型异常
  throw unsupported_dtype();
    }
  }  // 结束 visit 方法的定义

  TORCH_API void visit(CastPtr v) override {
    // 获取 Cast 操作的源值表达式
    ExprPtr src_value = v->src_value();
    // 对源值表达式进行访问
    src_value->accept(this);
    // 获取 Cast 操作的目标数据类型
    Dtype dst_dtype = v->dtype();
    // 获取源值表达式的数据类型
    Dtype src_dtype = src_value->dtype();
    // 检查源数据类型和目标数据类型的通道数是否匹配
    if (src_dtype.lanes() != dst_dtype.lanes()) {
      // 如果通道数不匹配，则抛出异常
      throw malformed_input("lane mismatch in Cast", v);
    }

    // 如果源数据类型和目标数据类型不相同，则进行类型转换
    if (src_dtype != dst_dtype) {
      // 根据源数据类型的标量类型进行不同的处理
      switch (src_dtype.scalar_type()) {
#define SRC_TYPE_CASE(Type, Name)                      \
  case ScalarType::Name:                               \  // 匹配目标数据类型的情况
    doCastFromSrc<Type>(src_dtype, dst_dtype, value_); \  // 调用模板函数，进行从源类型到目标类型的转换
    break;
        AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, SRC_TYPE_CASE);  // 遍历所有标量类型，并应用上面的宏定义
        SRC_TYPE_CASE(c10::quint8, QUInt8);  // 特定情况下的源类型到目标类型的转换
        SRC_TYPE_CASE(c10::qint8, QInt8);    // 特定情况下的源类型到目标类型的转换
#undef SRC_TYPE_CASE                                // 取消前面的宏定义
        default:                                   // 默认情况，如果未匹配到目标数据类型
          throw unsupported_dtype();               // 抛出不支持的数据类型异常
      }
    }
  }

  template <typename SrcType, typename DstType>
  std::vector<DstType> bitcastValues(
      const Dtype& src_dtype,
      const InterpValue& v) {
    const std::vector<SrcType>& src_values = v.as_vec<SrcType>();  // 将解释值转换为源类型的向量
    std::vector<DstType> dst_values(src_values.size());             // 创建目标类型的向量，大小与源类型向量相同
    for (int i = 0; i < src_dtype.lanes(); ++i) {                   // 遍历源数据类型的通道数
      dst_values[i] = raw_bitcast<DstType>(src_values[i]);          // 使用原始位转换将源类型值转换为目标类型值
    }
    return dst_values;                                              // 返回转换后的目标类型值向量
  }

  template <typename SrcType>
  void doBitCastFromSrc(
      const Dtype& src_dtype,
      const Dtype& dst_dtype,
      const InterpValue& v) {
    switch (dst_dtype.scalar_type()) {                             // 根据目标数据类型执行不同操作
#define DST_TYPE_CASE(Type, Name)                                           \
  case ScalarType::Name:                                                    \  // 匹配目标数据类型的情况
    this->value_ = InterpValue(bitcastValues<SrcType, Type>(src_dtype, v)); \  // 使用模板函数进行位转换，并将结果存储在成员变量中
    break;
      // bool/half not supported
      AT_FORALL_SCALAR_TYPES(DST_TYPE_CASE);                         // 遍历所有标量类型，并应用上面的宏定义
#undef DST_TYPE_CASE                                                   // 取消前面的宏定义
      default:                                                         // 默认情况，如果未匹配到目标数据类型
        throw unsupported_dtype();                                     // 抛出不支持的数据类型异常
    }
  }

  TORCH_API void visit(BitCastPtr v) override {                        // 重写访问位转换指针的方法
    ExprPtr src_value = v->src_value();                                // 获取位转换指针的源值
    src_value->accept(this);                                           // 递归访问源值的子节点
    Dtype dst_dtype = v->dtype();                                       // 获取位转换指针的目标数据类型
    Dtype src_dtype = src_value->dtype();                               // 获取源值的数据类型
    if (src_dtype.byte_size() != dst_dtype.byte_size()) {               // 检查源数据类型与目标数据类型的字节大小是否匹配
      throw malformed_input("lane mismatch in Cast", v);                // 如果不匹配，抛出错误输入异常
    }
    if (src_dtype != dst_dtype) {                                       // 如果源数据类型不等于目标数据类型
      switch (src_dtype.scalar_type()) {                                // 根据源数据类型执行不同操作
#define SRC_TYPE_CASE(Type, Name)                         \
  case ScalarType::Name:                                  \  // 匹配源数据类型的情况
    doBitCastFromSrc<Type>(src_dtype, dst_dtype, value_); \  // 调用模板函数，进行位转换
    break;
        // bool/half not supported
        AT_FORALL_SCALAR_TYPES(SRC_TYPE_CASE);            // 遍历所有标量类型，并应用上面的宏定义
#undef SRC_TYPE_CASE                                       // 取消前面的宏定义
        default:                                           // 默认情况，如果未匹配到源数据类型
          throw unsupported_dtype();                       // 抛出不支持的数据类型异常
      }
    }
  }

  TORCH_API void visit(ForPtr v) override {                        // 重写访问For循环指针的方法
    ExprPtr var_node = v->var();                                   // 获取For循环指针的变量节点
    v->start()->accept(this);                                      // 访问For循环的起始值
    auto dtype = value_.dtype();                                   // 获取当前值的数据类型
    auto start = value_.intValue();                                // 获取起始值的整数值
    v->stop()->accept(this);                                       // 访问For循环的终止值
    auto stop = value_.intValue();                                 // 获取终止值的整数值
    if (eval_context_.count(var_node)) {                           // 检查评估上下文中是否包含变量节点
      throw malformed_input("could not find var_node in For context", v);  // 如果不包含，抛出错误输入异常
    }

    for (auto i = start; i < stop; i++) {                          // 循环遍历从起始值到终止值
      eval_context_[var_node] = InterpValue(dtype, i);             // 在评估上下文中存储变量节点和其相应的解释值
      if (v->body()) {                                             // 如果For循环有体
        v->body()->accept(this);                                   // 递归访问For循环的体
      }
    }
    eval_context_.erase(var_node);                                 // 从评估上下文中删除变量节点
  }

  TORCH_API void visit(RampPtr v) override {                        // 重写访问Ramp指针的方法
    v->base()->accept(this);                                        // 访问Ramp指针的基础
    auto base = value().intValue();                                 // 获取基础的整数值
    v->stride()->accept(this);                                      // 访问Ramp指针的步长
    auto stride = value().intValue();                               // 获取步长的整数值
    int lanes = v->lanes();                                         // 获取Ramp指针的通道数
    // 创建一个大小为lanes的整数向量values，用于存储基础值base以及根据步长stride生成的数列
    std::vector<int> values(lanes);
    // 遍历lanes的范围，计算每个索引位置对应的值，并存储在values向量中
    for (const auto i : c10::irange(lanes)) {
      values[i] = base + i * stride;
    }

    // 使用values向量创建一个InterpValue对象，并将其赋值给当前对象的成员变量value_
    value_ = InterpValue(values);
  }

  // 重写基类的虚函数visit，处理BroadcastPtr类型的指针v
  TORCH_API void visit(BroadcastPtr v) override {
    // 递归调用v指针的value()方法，并让其接受当前对象（即this）作为访问者
    v->value()->accept(this);
    // 获取当前对象（即this）的value()方法返回的InterpValue对象，并存储在局部变量value中
    InterpValue value = this->value();
    // 获取v指针的lanes()方法返回的广播数量，并存储在局部变量lanes中
    int lanes = v->lanes();
    // 根据value的数据类型的标量类型执行不同的操作
    switch (value.dtype().scalar_type()) {


这段代码是C++中的一部分，展示了对广播操作的访问处理，通过注释详细说明了每个语句的作用和意图。
  // 定义宏 TYPE_CASE，用于根据标量类型处理不同情况
  #define TYPE_CASE(Type, Name)                     \
    case ScalarType::Name: {                        \
      // 创建一个包含指定元素数量和值的向量 v
      std::vector<Type> v(lanes, value.as<Type>()); \
      // 使用向量 v 创建 InterpValue 对象，并赋值给成员变量 value_
      value_ = InterpValue(v);                      \
    } break;
  
  // 遍历所有标量类型，对每种类型执行 TYPE_CASE 宏
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
  
  // 取出标量类型，如果没有匹配的类型，抛出不支持的数据类型异常
  default:
    throw unsupported_dtype();
  }
}

// 重写 visit 方法以处理 IfThenElsePtr 类型的指针 v
TORCH_API void visit(IfThenElsePtr v) override {
  // 访问条件节点，并递归调用 accept 方法
  v->condition()->accept(this);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool cond_v;
  // 根据 value_ 的标量类型进行不同的处理
  switch (value_.dtype().scalar_type()) {
    // 定义宏 TYPE_CASE，处理不同的标量类型
    #define TYPE_CASE(Type, Name)   \
      case ScalarType::Name: {      \
        // 将 value_ 转换为对应的标量类型
        cond_v = value_.as<Type>(); \
      } break;
    // 遍历所有标量类型，对每种类型执行 TYPE_CASE 宏
    AT_FORALL_SCALAR_TYPES_AND(Bool, TYPE_CASE);
    
    // 如果标量类型为 Half，抛出不支持的数据类型异常
    case ScalarType::Half:
      throw unsupported_dtype("IfThenElse condition can't have Half dtype");
    // 如果标量类型为 BFloat16，抛出不支持的数据类型异常
    case ScalarType::BFloat16:
      throw unsupported_dtype("IfThenElse condition can't have BFloat16 dtype");
    // 如果标量类型未知，抛出不支持的数据类型异常
    default:
      throw unsupported_dtype();
  }

  // 根据条件 cond_v 执行相应分支的递归访问
  if (cond_v) {
    v->true_value()->accept(this);
  } else {
    v->false_value()->accept(this);
  }
}

// 定义模板函数 toLongVec，将任意类型 T 转换为 int64_t 类型的向量
template <typename T>
std::vector<int64_t> toLongVec(T&& t) {
  return std::vector<int64_t>{std::begin(t), std::end(t)};
}

// 定义函数 indexVec，返回 InterpValue 的标量类型对应的 int64_t 向量
std::vector<int64_t> indexVec(const InterpValue& v) {
  // 根据 InterpValue 的标量类型进行不同处理
  switch (v.dtype().scalar_type()) {
    // 定义宏 TYPE_CASE，处理不同的标量类型
    #define TYPE_CASE(Type, Name) \
      case ScalarType::Name:      \
        // 将 InterpValue 转换为对应的标量类型向量，并返回
        return toLongVec(v.as_vec<Type>());
    // 遍历所有整数类型，对每种类型执行 TYPE_CASE 宏
    AT_FORALL_INT_TYPES(TYPE_CASE);
    
    // 如果标量类型未知，抛出不支持的数据类型异常
    default:
      throw unsupported_dtype();
  }
  return {}; // 返回空向量
}

// 定义函数 check_bounds_throw，抛出索引超出范围异常
void check_bounds_throw(int64_t idx, int64_t bound, const BufPtr& buf) {
  // 创建异常描述信息的 stringstream 对象
  std::stringstream ss;
  // 构造异常信息，包含索引和边界信息
  ss << "Index out of bounds in check_bounds. Index: " << idx
     << "; bounds: [0, " << bound << ").";
  // 抛出索引超出范围的异常，包含异常描述信息和缓冲区指针
  throw malformed_input(ss.str(), buf);
}

// 定义函数 check_bounds，检查索引是否超出边界
void check_bounds(const BufPtr& buf, const std::vector<ExprPtr>& indices) {
  // 获取缓冲区的维度向量
  const std::vector<ExprPtr>& dims = buf->dims();
    // 检查维度大小与索引数量是否匹配
    if (dims.size() != indices.size()) {
      // 索引已经被展开，但未被缓存
      if (indices.size() == 1) {
        // 如果索引数量为1，检查缓存的步长数量是否与维度大小匹配
        if (dims.size() != buf->strides().size()) {
          // 抛出异常，指示维度数量与步长数量不匹配
          throw malformed_input(
              "Number of dimensions did not match number of strides", buf);
        }
        // 计算缓存大小
        int64_t buf_size = 1;
        if (!dims.empty()) {
          // 使用第一个维度大小创建表达式
          ExprHandle buf_size_expr = ExprHandle(immLike(dims[0], 1));
          ExprHandle negative_one = ExprHandle(immLike(dims[0], -1));
          for (const auto& i : c10::irange(dims.size())) {
            // 计算缓存大小表达式
            buf_size_expr = buf_size_expr +
                ((negative_one + ExprHandle(dims[i])) *
                 ExprHandle(buf->strides()[i]));
          }
          buf_size_expr.node()->accept(this);
          buf_size = value().intValue();
        }
        // 遍历索引值，检查是否超出缓存范围
        indices[0]->accept(this);
        const auto& index_values = indexVec(value());
        for (auto& j : index_values) {
          if (j < 0 || j >= buf_size) {
            // 如果索引值超出范围，抛出异常
            check_bounds_throw(j, buf_size, buf);
          }
        }
        return;
      }
      // 如果维度和索引数量不匹配且索引数量不为1，则抛出异常
      throw malformed_input(
          "dimensions and indices mismatch in check_bounds. Buf has " +
              std::to_string(dims.size()) + " dimensions and indices has " +
              std::to_string(indices.size()) + " dimensions.",
          buf);
    }
    // 遍历每个维度，检查索引是否超出维度边界
    for (const auto& i : c10::irange(dims.size())) {
      auto opt_dim = intValue(dims[i]);
      if (!opt_dim) {
        continue;
      }
      auto dim_bound = *opt_dim;
      // 接受索引节点
      indices[i]->accept(this);
      const auto& ithDimIndices = indexVec(value());
      for (auto& j : ithDimIndices) {
        // 检查当前维度的每个索引是否在边界内
        if (j < 0 || j >= dim_bound) {
          check_bounds_throw(j, dim_bound, buf);
        }
      }
    }
  }

  // 访问 LoadPtr 类型节点的处理函数
  TORCH_API void visit(LoadPtr v) override {
    // 查找 Load 对应的缓存节点
    auto iter = buffer_mapping_.find(v->buf());
    if (iter == buffer_mapping_.end()) {
      // 如果未找到对应的缓存节点，抛出异常
      throw malformed_input("could not find base node in Load", v);
    }
    // 获取缓存指针
    void* ptr = iter->second;

    // 检查边界，确保加载操作的索引合法
    check_bounds(v->buf(), v->indices());

    // 展平索引以获取扁平化的索引节点
    ExprPtr flat_idx =
        flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides());
    flat_idx->accept(this);
    auto index = indexVec(value());
    // 获取 Load 操作的标量类型
    ScalarType v_sdtype = v->dtype().scalar_type();
    switch (v_sdtype) {
#define TYPE_CASE(Type, Name)                        \
  case ScalarType::Name: {                           \
    Type* ptr##Name = static_cast<Type*>(ptr);       \
    // 创建一个存储类型为 Type 的向量 val，其大小与索引大小相同
    std::vector<Type> val(index.size());             \
    // 遍历索引范围
    for (const auto i : c10::irange(index.size())) { \
      // 从 ptr##Name 指针指向的数据中读取 index[i] 处的值到 val[i]
      val[i] = ptr##Name[index[i]];                  \
      // 输出调试信息，显示加载过程中的指针、缓冲区名、索引、以及值
      GRAPH_DEBUG(                                   \
          "LOAD: ptr=",                              \
          ptr##Name,                                 \
          ", buf=",                                  \
          v->buf()->name_hint(),                     \
          ", idx=",                                  \
          index[i],                                  \
          ", val=",                                  \
          (int)underlyingValue(val[i]));             \
    }                                                \
    // 将 val 转换为 InterpValue 并赋给 value_
    value_ = InterpValue(val);                       \
  } break;
// 对所有标量类型和额外的三种类型进行 TYPE_CASE 宏的调用
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
// NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
// 调用 TYPE_CASE 宏处理 c10::quint8 类型
TYPE_CASE(c10::quint8, QUInt8);
// NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
// 调用 TYPE_CASE 宏处理 c10::qint8 类型
TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE
// 默认情况下抛出异常，表示不支持的数据类型
default:
throw unsupported_dtype("scalar type:" + std::to_string(v_sdtype));
}

// 实现 visit 方法，用于处理 StorePtr 对象
TORCH_API void visit(StorePtr v) override {
// 在 buffer_mapping_ 中查找 v->buf() 对应的迭代器
auto iter = buffer_mapping_.find(v->buf());
// 如果未找到，则抛出异常
if (iter == buffer_mapping_.end()) {
throw malformed_input("could not find base node in Store", v);
}

// 获取 v->buf() 对应的指针
void* ptr = iter->second;

// 检查索引是否越界
check_bounds(v->buf(), v->indices());

// 将索引展开为一维索引
ExprPtr flat_idx =
flatten_index(v->buf()->dims(), v->indices(), v->buf()->strides());
flat_idx->accept(this);
// 获取展开后的索引向量
auto index = indexVec(value());
// 获取 v->value() 的标量类型
ScalarType v_sdtype = v->value()->dtype().scalar_type();

// 根据标量类型进行不同的处理
switch (v_sdtype) {
#define TYPE_CASE(Type, Name)                                   \
case ScalarType::Name: {                                      \
// 接受 v->value() 并将其转换为 Type 类型的向量 value
v->value()->accept(this);                                   \
std::vector<Type> value = this->value().as_vec<Type>();     \
// 检查 value 和 index 的大小是否一致
if (index.size() != value.size()) {                         \
throw malformed_input("value size mismatch in Store", v); \
}                                                           \
// 将 ptr##Name 强制转换为 Type* 类型
Type* ptr##Name = static_cast<Type*>(ptr);                  \
    // 使用范围循环遍历索引数组，遍历范围是从 0 到 index.size()-1
    for (const auto i : c10::irange(index.size())) {            \
      // 输出调试信息，显示存储操作的细节：指针、缓冲区名称、索引、值的底层整数表示
      GRAPH_DEBUG(                                              \
          "STORE: ptr=",                                        \
          ptr##Name,                                            \
          ", buf=",                                             \
          v->buf()->name_hint(),                                \
          ", idx=",                                             \
          index[i],                                             \
          ", val=",                                             \
          (int)underlyingValue(value[i]));                      \
      // 将 value[i] 存储到 ptr##Name 的第 index[i] 个位置
      ptr##Name[index[i]] = value[i];                           \
    }                                                           \
  } break;
      // 针对所有标量类型进行类型分支处理
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
      // 忽略 Facebook HTE 工具的本地未检查的数组边界警告
      // 处理 c10::quint8 类型的分支
      TYPE_CASE(c10::quint8, QUInt8);
      // 忽略 Facebook HTE 工具的本地未检查的数组边界警告
      // 处理 c10::qint8 类型的分支
      TYPE_CASE(c10::qint8, QInt8);
  void visit(ExternalCallPtr v) override {
    // 获取全局的 NNCF 函数注册表
    auto& func_registry = getNNCFunctionRegistry();
    // 检查是否注册了要调用的函数，如果未注册则抛出未实现的错误
    if (!func_registry.count(v->func_name())) {
      throw unimplemented_lowering(v);
    }
    // 打印调试信息，显示外部调用的函数名和缓冲区的名称提示
    GRAPH_DEBUG(
        "EXTERNAL CALL: func=",
        v->func_name(),
        ", buf=",
        v->buf()->name_hint());

    // 准备用于传递给外部函数的缓冲区列表，包括主缓冲区和缓冲区参数
    std::vector<BufPtr> bufs(v->buf_args());
    bufs.insert(bufs.begin(), v->buf());

    // 准备存储缓冲区指针、秩、维度、步长、数据类型以及额外参数的容器
    std::vector<void*> buf_ptrs;
    std::vector<int64_t> buf_ranks;
    std::vector<int64_t> buf_dims;
    std::vector<int64_t> buf_strides;
    std::vector<int8_t> buf_dtypes;
    std::vector<int64_t> extra_args;

    // 遍历缓冲区列表，处理每个缓冲区
    for (const BufPtr& b : bufs) {
      // 查找缓冲区在 buffer_mapping_ 中的映射
      auto iter = buffer_mapping_.find(b);
      // 如果找不到映射，则抛出格式错误异常
      if (iter == buffer_mapping_.end()) {
        throw malformed_input("could not find buf", v);
      }

      // 将缓冲区指针添加到 buf_ptrs
      buf_ptrs.push_back(iter->second);
      // 获取缓冲区的秩，并添加到 buf_ranks
      buf_ranks.push_back(b->dims().size());
      // 获取缓冲区的维度，并添加到 buf_dims
      for (const ExprPtr& dim_expr : b->dims()) {
        dim_expr->accept(this);
        buf_dims.push_back(value().intValue());
      }
      // 获取缓冲区的步长，并添加到 buf_strides
      for (const ExprPtr& stride_expr : b->strides()) {
        stride_expr->accept(this);
        buf_strides.push_back(value().intValue());
      }
    }

    // 处理外部函数的额外参数列表
    for (const ExprPtr& a : v->args()) {
      a->accept(this);
      // 根据数据类型处理值，转换为 int64_t 类型，并添加到 extra_args
      int64_t val;
      if (value().dtype() == kLong) {
        val = value().as<int64_t>();
      } else if (value().dtype() == kInt) {
        val = value().intValue();
      } else if (value().dtype() == kDouble) {
        auto x = value().as<double>();
        val = reinterpret_cast<int64_t*>(&x)[0];
      } else if (value().dtype() == kFloat) {
        auto x = value().as<float>();
        val = reinterpret_cast<int64_t*>(&x)[0];
      } else {
        // 如果数据类型不是预期的整数类型，则抛出格式错误异常
        throw malformed_input(
            "extra_args in ExternalCalls must have int64 dtype", v);
      }
      extra_args.push_back(val);
    }

    // 获取要调用的外部函数指针，并执行函数调用
    auto fn_ptr = func_registry.at(v->func_name());
    (*fn_ptr)(
        bufs.size(),
        buf_ptrs.data(),
        buf_ranks.data(),
        buf_dims.data(),
        buf_strides.data(),
        buf_dtypes.data(),
        extra_args.size(),
        extra_args.data());
  }
    // 创建一个空的 int64_t 类型的向量 extra_args
    std::vector<int64_t> extra_args;

    // 初始化计数器 i 为 0
    size_t i = 0;
    // 遍历输入的 bufs_in 向量
    for (const auto& b : bufs_in) {
      // 在 buffer_mapping_ 中查找 b 对应的迭代器
      auto iter = buffer_mapping_.find(b);
      // 如果找不到对应的迭代器，则抛出异常 malformed_input
      if (iter == buffer_mapping_.end()) {
        throw malformed_input("could not find buf", v);
      }
      // 将找到的 buf 对应的指针 iter->second 存入 buf_ptrs 数组
      buf_ptrs[bufs_out_size + i] = iter->second;
      // 将 b 的维度数量添加到 buf_ranks 向量
      buf_ranks.push_back(b->dims().size());
      // 将 b 的数据类型转换为 int8_t 后添加到 buf_dtypes 向量
      buf_dtypes.push_back((int8_t)b->dtype().scalar_type());
      // 遍历 b 的每个维度表达式 dim_expr
      for (const auto& dim_expr : b->dims()) {
        // 调用 dim_expr 的 accept 方法
        dim_expr->accept(this);
        // 将计算结果作为整数添加到 buf_dims 向量
        buf_dims.push_back(value().intValue());
      }
      // 遍历 b 的每个步长表达式 stride_expr
      for (const ExprPtr& stride_expr : b->strides()) {
        // 调用 stride_expr 的 accept 方法
        stride_expr->accept(this);
        // 将计算结果作为整数添加到 buf_strides 向量
        buf_strides.push_back(value().intValue());
      }
      // 计数器 i 自增
      i++;
    }
    // 遍历 v 的参数 args
    for (const auto& a : v->args()) {
      // 调用 a 的 accept 方法
      a->accept(this);
      // 声明 int64_t 类型的变量 val
      int64_t val;
      // 根据 value 的数据类型选择赋值方式
      if (value().dtype() == kLong) {
        val = value().as<int64_t>();
      } else if (value().dtype() == kInt) {
        val = value().intValue();
      } else if (value().dtype() == kDouble) {
        auto x = value().as<double>();
        val = reinterpret_cast<int64_t*>(&x)[0];
      } else if (value().dtype() == kFloat) {
        auto x = value().as<float>();
        val = reinterpret_cast<int64_t*>(&x)[0];
      } else {
        // 如果数据类型不是预期的 int64，则抛出异常 malformed_input
        throw malformed_input(
            "extra_args in ExternalCalls must have int64 dtype", v);
      }
      // 将计算得到的 val 添加到 extra_args 向量
      extra_args.push_back(val);
    }

    // 获取 v->func_name() 对应的函数指针 fn_ptr
    auto fn_ptr = func_registry.at(v->func_name());
    // 调用 fn_ptr 函数，并传递各个参数数组
    (*fn_ptr)(
        bufs_in_size,
        buf_ptrs.data(),
        buf_ranks.data(),
        buf_dims.data(),
        buf_strides.data(),
        buf_dtypes.data(),
        extra_args.size(),
        extra_args.data());

    // 遍历 bufs_out_size 的次数，更新 buffer_mapping_ 中的映射关系
    for (i = 0; i < bufs_out_size; ++i) {
      // 获取 bufs_out[i] 对应的 buf_out
      const auto& buf_out = bufs_out[i];
      // 将 buf_out 和 buf_ptrs[i] 的映射存入 buffer_mapping_
      buffer_mapping_[buf_out] = buf_ptrs[i];
      // 将 buf_ptrs[bufs_in_size + bufs_out_size + i] 存入 ext_bufs_free_ptr_ 中
      ext_bufs_free_ptr_[buf_out] = buf_ptrs[bufs_in_size + bufs_out_size + i];
    }
  }

  // 模板函数定义，接受 TReturn 和 TInput 类型的参数，并访问 IntrinsicsPtr 对象
  template <typename TReturn, typename TInput>
  void visit_intrinsics_helper(IntrinsicsPtr v) {
    // 创建一个 InterpValue 类型的向量 values，大小为 v->nparams()
    std::vector<InterpValue> values(v->nparams());
    // 遍历 v->nparams() 次数，访问并存储每个参数的值
    for (const auto i : c10::irange(v->nparams())) {
      // 调用 v 的 param(i) 的 accept 方法
      v->param(i)->accept(this);
      // 将计算得到的值存入 values 向量
      values[i] = this->value();
    }
    // 创建一个 TInput 类型的向量 v1
    std::vector<TInput> v1;
    // 如果 values 非空，则将 values[0] 转换为向量并赋给 v1
    if (!values.empty()) {
      v1 = values[0].as_vec<TInput>();
    }
    // 创建一个 TInput 类型的向量 v2
    std::vector<TInput> v2;
    // 如果 values 的大小大于等于 2，则将 values[1] 转换为向量并赋给 v2
    if (values.size() >= 2ULL) {
      v2 = values[1].as_vec<TInput>();
      // 如果 v1 和 v2 的大小不一致，则抛出异常 malformed_input
      if (v1.size() != v2.size()) {
        throw malformed_input("value size mismatch in Intrinsics", v);
      }
    }

    // 如果 values 的大小超过 2，则抛出 unimplemented_lowering 异常
    if (values.size() > 2) {
      throw unimplemented_lowering(v);
    }

    // 创建一个 TReturn 类型的向量 result，大小为 v1.size()，初始值为 -1
    std::vector<TReturn> result(v1.size(), -1);
    // 如果 values 的大小为 1，则遍历 v1.size() 次数，计算每个元素的内部函数并存储到 result 中
    if (values.size() == 1ULL) {
      for (const auto i : c10::irange(v1.size())) {
        result[i] = compute_intrinsics<TReturn>(v->op_type(), v1[i]);
      }
    }
  }

  // 访问 IntrinsicsPtr 类型的节点的方法
  TORCH_API void visit(IntrinsicsPtr v) override {
    // 获取节点的数据类型
    auto ty = v->dtype().scalar_type();
    // 如果操作类型是 kIsNan
    if (v->op_type() == kIsNan) {
      // 获取输入参数的数据类型
      auto inp_dtype = v->params().at(0)->dtype().scalar_type();
      // 根据输入参数的数据类型进行分支处理
      if (inp_dtype == ScalarType::Float) {
        // 调用帮助函数 visit_intrinsics_helper，返回 int 类型和 float 类型的计算结果
        visit_intrinsics_helper<int, float>(v);
      } else if (inp_dtype == ScalarType::Double) {
        // 调用帮助函数 visit_intrinsics_helper，返回 int 类型和 double 类型的计算结果
        visit_intrinsics_helper<int, double>(v);
      } else if (inp_dtype == ScalarType::Half) {
        // 抛出不支持的数据类型异常
        throw unsupported_dtype(); // TODO
      } else if (inp_dtype == ScalarType::BFloat16) {
        // 抛出不支持的数据类型异常
        throw unsupported_dtype(); // TODO
      }
    } else {
      // 对于其他操作类型，根据数据类型进行分支处理
      switch (ty) {
  #define TYPE_CASE(Type, Name)               \
  case ScalarType::Name:                    \  // 处理不同的标量类型的情况
    visit_intrinsics_helper<Type, Type>(v); \  // 调用辅助函数处理特定类型的内置函数
    break;
        AT_FORALL_SCALAR_TYPES(TYPE_CASE);  // 遍历所有标量类型并应用上述宏定义的处理逻辑
#undef TYPE_CASE  // 取消宏定义

        default:
          throw unsupported_dtype();  // 抛出不支持的数据类型异常
      }
    }
  }

  void visit(AllocatePtr v) override {
    BufPtr b = v->buf();  // 获取分配操作的缓冲区指针
    std::vector<ExprPtr> dims = b->dims();  // 获取缓冲区的维度
    int64_t total_byte_size = b->dtype().byte_size();  // 计算缓冲区总字节大小
    for (auto& dim : dims) {
      dim->accept(this);  // 访问并计算每个维度的值
      total_byte_size *= value_.intValue();  // 根据维度值更新总字节大小
    }
    auto int_count = (total_byte_size + sizeof(int) - 1) / sizeof(int);  // 计算需要的整数数目
    GRAPH_DEBUG(
        "ALLOCATE: buf=", v->buf()->name_hint(), ", size=", total_byte_size);  // 调试信息，记录分配操作的缓冲区名称和大小
    auto buffer = std::make_unique<std::vector<int>>(int_count);  // 创建一个整数向量缓冲区
    auto iter = buffer_mapping_.find(b);  // 查找缓冲区映射中是否已经存在该缓冲区
    if (iter != buffer_mapping_.end() && iter->second != nullptr) {
      throw std::runtime_error(
          "Allocate a buffer that has already been allocated: " +
          v->buffer_var()->name_hint());  // 如果已经分配过该缓冲区，则抛出运行时错误
    }
    buffer_mapping_[b] = buffer->data();  // 将新分配的缓冲区加入缓冲区映射
    internal_buffers_.insert(std::make_pair(b, std::move(buffer)));  // 将新分配的缓冲区添加到内部缓冲区集合
  }

  void visit(PlacementAllocatePtr v) override {
    buffer_mapping_[v->buf()] = buffer_mapping_.at(v->buf_to_reuse());  // 在缓冲区映射中进行缓冲区重用的处理
  }

  void visit(FreePtr v) override {
    BufPtr b = v->buf();  // 获取释放操作的缓冲区指针
    GRAPH_DEBUG("FREE: buf=", v->buf()->name_hint());  // 调试信息，记录释放操作的缓冲区名称
    int count = internal_buffers_.erase(b);  // 从内部缓冲区集合中删除缓冲区
    if (count == 0) {
      throw std::runtime_error(
          "Free a buffer that is not currently bound: " +
          v->buffer_var()->name_hint());  // 如果要释放的缓冲区未绑定，则抛出运行时错误
    }
    buffer_mapping_.erase(b);  // 从缓冲区映射中删除缓冲区
  }

  void visit(FreeExtPtr v) override {
    const auto& bufs = v->bufs();  // 获取要释放的外部缓冲区列表
    const auto bufs_num = bufs.size();  // 外部缓冲区数量
    std::vector<void*> buf_ptrs;  // 存储外部缓冲区的指针
    for (const auto& buf : bufs) {
      if (!ext_bufs_free_ptr_.count(buf)) {
        throw std::runtime_error(
            "Free an external allocated buffer that does not have corresponding pointer for freeing: " +
            buf->base_handle()->name_hint());  // 如果要释放的外部缓冲区没有对应的释放指针，则抛出运行时错误
      }
      buf_ptrs.push_back(ext_bufs_free_ptr_[buf]);  // 将外部缓冲区的释放指针添加到列表中
    }
    nnc_aten_free(bufs_num, buf_ptrs.data());  // 调用外部函数释放外部缓冲区
  }

  void visit(LetPtr v) override {
    var_by_scope_[scope_].push_back(v->var());  // 将变量添加到当前作用域的变量列表中
    bindVar(v->var(), evaluateExpr(v->value()));  // 绑定变量和对应的表达式值
  }

  void visit(CondPtr v) override {
    v->condition()->accept(this);  // 访问条件表达式并计算其值
    if (value().intValue()) {  // 如果条件值为真
      if (v->true_stmt()) {
        v->true_stmt()->accept(this);  // 访问并执行真实语句块
      }
    } else {  // 如果条件值为假
      if (v->false_stmt()) {
        v->false_stmt()->accept(this);  // 访问并执行假实语句块
      }
    }
  }

 private:
  template <
      typename TReturn,
      typename TInput,
      typename std::enable_if<std::is_floating_point<TInput>::value, int>::
          type = 0>
  static TReturn compute_intrinsics(IntrinsicsOp op_type, TInput v) {
    // 用于计算内置函数的模板函数，处理浮点数类型的输入
    switch (op_type) {
      case kSin:
        // 如果操作类型是求正弦值，则调用标准库的 sin 函数，并返回结果
        return std::sin(v);
      case kCos:
        // 如果操作类型是求余弦值，则调用标准库的 cos 函数，并返回结果
        return std::cos(v);
      case kTan:
        // 如果操作类型是求正切值，则调用标准库的 tan 函数，并返回结果
        return std::tan(v);
      case kAsin:
        // 如果操作类型是求反正弦值，则调用标准库的 asin 函数，并返回结果
        return std::asin(v);
      case kAcos:
        // 如果操作类型是求反余弦值，则调用标准库的 acos 函数，并返回结果
        return std::acos(v);
      case kAtan:
        // 如果操作类型是求反正切值，则调用标准库的 atan 函数，并返回结果
        return std::atan(v);
      case kSinh:
        // 如果操作类型是求双曲正弦值，则调用标准库的 sinh 函数，并返回结果
        return std::sinh(v);
      case kCosh:
        // 如果操作类型是求双曲余弦值，则调用标准库的 cosh 函数，并返回结果
        return std::cosh(v);
      case kTanh:
        // 如果操作类型是求双曲正切值，则调用标准库的 tanh 函数，并返回结果
        return std::tanh(v);
      case kExp:
        // 如果操作类型是求指数值（e 的幂），则调用标准库的 exp 函数，并返回结果
        return std::exp(v);
      case kAbs:
        // 如果操作类型是求绝对值，则调用标准库的 abs 函数，并返回结果
        return std::abs(v);
      case kExpm1:
        // 如果操作类型是求 exp(v) - 1，则调用标准库的 expm1 函数，并返回结果
        return std::expm1(v);
      case kLog:
        // 如果操作类型是求自然对数值，则调用标准库的 log 函数，并返回结果
        return std::log(v);
      case kLog2:
        // 如果操作类型是求以 2 为底的对数值，则调用标准库的 log2 函数，并返回结果
        return std::log2(v);
      case kLog10:
        // 如果操作类型是求以 10 为底的对数值，则调用标准库的 log10 函数，并返回结果
        return std::log10(v);
      case kLog1p:
        // 如果操作类型是求 log(v + 1)，则调用标准库的 log1p 函数，并返回结果
        return std::log1p(v);
      case kErf:
        // 如果操作类型是求误差函数值，则调用标准库的 erf 函数，并返回结果
        return std::erf(v);
      case kErfc:
        // 如果操作类型是求补误差函数值，则调用标准库的 erfc 函数，并返回结果
        return std::erfc(v);
      case kSqrt:
        // 如果操作类型是求平方根，则调用标准库的 sqrt 函数，并返回结果
        return std::sqrt(v);
      case kRsqrt: {
        // 如果操作类型是求倒数的平方根，则定义一个 lambda 函数 rsqrt
        auto rsqrt = [](TInput v) __ubsan_ignore_float_divide_by_zero__ {
          // lambda 函数内部返回 1.0f 除以标准库的 sqrt(v) 的结果
          return 1.0f / std::sqrt(v);
        };
        return rsqrt(v); // 调用 lambda 函数并返回结果
      }
      case kCeil:
        // 如果操作类型是向上取整，则调用标准库的 ceil 函数，并返回结果
        return std::ceil(v);
      case kFloor:
        // 如果操作类型是向下取整，则调用标准库的 floor 函数，并返回结果
        return std::floor(v);
      case kRound:
        // 如果操作类型是四舍五入，则调用标准库的 round 函数，并返回结果
        return std::round(v);
      case kTrunc:
        // 如果操作类型是截断小数部分，则调用标准库的 trunc 函数，并返回结果
        return std::trunc(v);
      case kLgamma:
        // 如果操作类型是求伽玛函数的自然对数值，则调用标准库的 lgamma 函数，并返回结果
        return std::lgamma(v);
      case kFrac:
        TInput intpart; // 声明一个变量 intpart，用于存放整数部分
        // 如果操作类型是求浮点数的小数部分，则调用标准库的 modf 函数，并返回结果
        return std::modf(v, &intpart);
      case kIsNan:
        // 如果操作类型是判断是否为 NaN，则调用标准库的 isnan 函数，并返回结果
        return std::isnan(v);
      default:
        // 如果操作类型不在上述枚举值内，则抛出运行时异常，附带错误信息
        throw std::runtime_error("Invalid op_type: " + std::to_string(op_type));
    }
  }

  template <
      typename TReturn,
      typename TInput,
      typename std::enable_if<std::is_integral<TInput>::value, int>::type = 0>
  static TReturn compute_intrinsics(IntrinsicsOp op_type, TInput v) {
    switch (op_type) {
      case kAbs: {
        // 如果操作类型是求绝对值，并且输入类型是无符号整数，则将输入类型转换为 int
        using X =
            std::conditional_t<std::is_unsigned<TInput>::value, int, TInput>;
        return std::is_unsigned<TInput>::value ? v
                                               : std::abs(static_cast<X>(v));
      }
      default:
        // 如果操作类型不是上述情况，则抛出运行时异常，附带错误信息
        throw std::runtime_error(
            "Invalid integral op_type: " + std::to_string(op_type));
    }
  }

  // specialization for float -> int ops (just kIsNan currently)
  int compute_intrinsics(IntrinsicsOp op_type, float v) {
    switch (op_type) {
      case kIsNan:
        // 如果操作类型是判断是否为 NaN，则调用标准库的 isnan 函数，并返回结果
        return std::isnan(v);
      default:
        // 如果操作类型不是上述情况，则抛出运行时异常，附带错误信息
        throw std::runtime_error("Invalid op_type: " + std::to_string(op_type));
    }
  }

  template <typename TReturn, typename TInput>
  TReturn compute_intrinsics(IntrinsicsOp op_type, TInput v1, TInput v2) {
    # 根据操作类型选择不同的数学运算并返回结果
    switch (op_type) {
      # 如果操作类型是求幂运算
      case kPow:
        # 使用标准库函数 std::pow 计算 v1 的 v2 次方并返回结果
        return std::pow(v1, v2);
      # 如果操作类型是求余数运算
      case kFmod:
        # 使用标准库函数 std::fmod 计算 v1 除以 v2 的余数并返回结果
        return std::fmod(v1, v2);
      # 如果操作类型是取余数运算
      case kRemainder:
        # 使用标准库函数 std::remainder 计算 v1 除以 v2 的余数并返回结果
        return std::remainder(v1, v2);
      # 如果操作类型是求反正切运算
      case kAtan2:
        # 使用标准库函数 std::atan2 计算以 v1 和 v2 为参数的反正切值并返回结果
        return std::atan2(v1, v2);
      # 如果操作类型不在上述列举中，抛出运行时错误
      default:
        throw std::runtime_error("Invalid op_type: " + std::to_string(op_type));
    }
  }

  # 存储表达式的计算结果
  InterpValue value_;
  # 存储块的作用域
  BlockPtr scope_;
  # 存储表达式和其对应的计算值的映射关系
  std::unordered_map<ExprPtr, InterpValue> eval_context_;
  # 存储每个块和其内部表达式的映射关系
  std::unordered_map<BlockPtr, std::vector<ExprPtr>> var_by_scope_;
  # 存储缓冲区指针和对应的数据缓冲区映射关系
  std::unordered_map<BufPtr, void*> buffer_mapping_;
  # 存储缓冲区指针和对应的整数向量缓冲区的唯一指针映射关系
  std::unordered_map<BufPtr, std::unique_ptr<std::vector<int>>>
      internal_buffers_;
  # 存储外部缓冲区指针和对应的空闲指针映射关系
  std::unordered_map<BufPtr, void*> ext_bufs_free_ptr_;
};

SimpleIREvaluator::SimpleIREvaluator(
    StmtPtr stmt,
    const std::vector<BufferArg>& buffer_args,
    at::Device device,
    const std::string& kernel_func_name)
    : CodeGen(stmt, buffer_args, device, kernel_func_name) {
  // 使用给定的语句、缓冲区参数、设备和内核函数名初始化 SimpleIREvaluator
  impl_ = std::make_unique<SimpleIREvaluatorImpl>();
  // 扩展内部指令集
  expand_intrinsics();
}

SimpleIREvaluator::~SimpleIREvaluator() = default;

void SimpleIREvaluator::call(const std::vector<CallArg>& args) {
  // 创建一个用于存储 void* 类型参数的向量
  std::vector<void*> raw_args(args.size());
  // 遍历每个调用参数
  for (size_t i = 0; i < args.size(); i++) {
    // 获取对应的 BufferArg 和 CallArg
    auto const& bufferArg = buffer_args()[i];
    auto const& callArg = args[i];
    // 将 CallArg 转换为 void* 类型并存储在 raw_args 中
    raw_args[i] = argToPtr(bufferArg, callArg);
  }
  // 调用 call_raw 方法
  call_raw(raw_args);
}

void SimpleIREvaluator::call_raw(const std::vector<void*>& args) {
  // 检查参数数量是否匹配
  if (args.size() != buffer_args().size()) {
    throw malformed_input("bad args in IREvaluator call");
  }
  // 逐个绑定参数
  for (const auto i : c10::irange(args.size())) {
    bindArg(buffer_args()[i], args[i]);
  }
  // 接受语句并调用 impl_ 的 clear 方法
  stmt()->accept(&*impl_);
  impl_->clear();
}

void SimpleIREvaluator::bindArg(const BufferArg& bufArg, void* data) {
  // 如果不是变量类型的 BufferArg，则绑定缓冲区和数据
  if (!bufArg.isVar()) {
    impl_->bindBuf(bufArg.buf(), data);
    return;
  }

  // 根据数据类型绑定变量
  switch (bufArg.dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                 \
  case ScalarType::Name: {                    \
    Type typed_data;                          \
    memcpy(&typed_data, data, sizeof(Type));  \
    impl_->bindVar(bufArg.var(), typed_data); \
    break;                                    \
  }
    // 对所有标量数据类型生成绑定的 switch 语句
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

void SimpleIREvaluator::bindVar(VarPtr v, ExprPtr e) {
  // 将变量绑定到求值后的表达式
  impl_->bindVar(v, impl_->evaluateExpr(e));
}

InterpValue SimpleIREvaluator::value() const {
  // 返回内部实现类的值
  return impl_->value();
}

std::optional<int64_t> evalInt(ExprPtr e) {
  try {
    // 评估表达式的整数值，并返回可选的 int64_t 结果
    return ExprEval<SimpleIREvaluator>(cast<int64_t>(ExprHandle(e)))
        .value<int64_t>();
  } catch (std::runtime_error& err) {
    // 捕获运行时错误，返回空的可选值
    return c10::nullopt;
  }
}

} // namespace torch::jit::tensorexpr
```