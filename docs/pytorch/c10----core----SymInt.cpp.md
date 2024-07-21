# `.\pytorch\c10\core\SymInt.cpp`

```py
// 包含C10库中所需的头文件
#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/safe_numerics.h>
#include <functional>

// 定义命名空间c10，包含了所有的符号化整数相关内容
namespace c10 {

// 前提条件：data_包含一个大的负数，应被视为一个常数。这不是一个有效的指针。换句话说，SymInt暂时违反了不变量。
// 后置条件：SymInt的不变量被修复
void SymInt::promote_to_negative() {
  // 创建一个新的SymInt对象s，使用ConstantSymNodeImpl<int64_t>封装data_，表示一个负数常量
  auto s = SymInt(SymNode(c10::make_intrusive<ConstantSymNodeImpl<int64_t>>(data_)));
  // 类似于移动赋值操作符，但不释放data_
  data_ = s.data_;
  s.data_ = 0;
}

// 返回SymInt对象的符号化节点SymNode
SymNode SymInt::toSymNode() const {
  // 使用TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE检查堆分配情况，抛出异常，显示C++调用堆栈
  TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
      is_heap_allocated(), "SymInt::toSymNode is_heap_allocated");
  // 使用toSymNodeImplUnowned()方法获取SymInt对象的非拥有符号化节点，并创建其副本进行返回
  return SymNode::reclaim_copy(toSymNodeImplUnowned());
}

// 构造函数：通过给定的SymNode对象sin_sp构造SymInt对象
SymInt::SymInt(SymNode sin_sp) {
  // 使用TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE检查sin_sp是否为整数类型的符号化节点，否则抛出异常
  TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
      sin_sp->is_int(), "SymInt::SymInt sin_sp->is_int()");
  // 将sin_sp释放并转换为uintptr_t，再进行位操作，设置标志位IS_SYM，将结果存储为int64_t类型的data_
  auto ptr = static_cast<uint64_t>(
      reinterpret_cast<uintptr_t>(static_cast<void*>(sin_sp.release())));
  auto rep = (ptr & ~MASK) | IS_SYM;
  data_ = static_cast<int64_t>(rep);
}

// 检查SymInt对象是否具有提示信息
bool SymInt::has_hint() const {
  // 如果不是堆分配的，则返回true
  if (!is_heap_allocated()) {
    return true;
  }
  // 否则，调用toSymNodeImplUnowned()->has_hint()方法检查非拥有的符号化节点是否具有提示信息
  return toSymNodeImplUnowned()->has_hint();
}

// 定义宏：定义二元操作符API（例如operator+），接受SymInt对象sci作为参数，返回指定的类型RET
#define DEFINE_BINARY(API, OP, METHOD, RET)                          \
  RET SymInt::API(const SymInt& sci) const {                         \
    // 如果当前对象和sci均可视为整数
    if (auto ma = maybe_as_int()) {                                  \
      if (auto mb = sci.maybe_as_int()) {                            \
        // 返回两个整数的运算结果
        return RET(OP(*ma, *mb));                                    \
      } else {                                                       \
        // 将sci转换为符号化节点b，调用METHOD方法，返回运算结果
        auto b = sci.toSymNode();                                    \
        return RET(b->wrap_int(*ma)->METHOD(b));                     \
      }                                                              \
    } else {                                                         \
      if (auto mb = sci.maybe_as_int()) {                            \
        // 将当前对象转换为符号化节点a，调用METHOD方法，返回运算结果
        auto a = toSymNodeImplUnowned();                             \
        return RET(a->METHOD(a->wrap_int(*mb)));                     \
      } else {                                                       \
        // 转换当前对象和sci为符号化节点，调用METHOD方法，返回运算结果
        return RET(toSymNodeImplUnowned()->METHOD(sci.toSymNode())); \
      }                                                              \
    }                                                                \
  }

// 定义加法运算符operator+的具体实现
DEFINE_BINARY(operator+, std::plus<>(), add, SymInt)
// 定义减法运算符operator-的具体实现
DEFINE_BINARY(operator-, std::minus<>(), sub, SymInt)
// 定义乘法运算符operator*的具体实现
DEFINE_BINARY(operator*, std::multiplies<>(), mul, SymInt)
// 定义除法运算符operator/的具体实现
DEFINE_BINARY(operator/, std::divides<>(), floordiv, SymInt)
// 定义取模运算符operator%的具体实现
DEFINE_BINARY(operator%, std::modulus<>(), mod, SymInt)
// 定义相等运算符sym_eq的具体实现
DEFINE_BINARY(sym_eq, std::equal_to<>(), eq, SymBool)
// 定义不等运算符sym_ne的具体实现
DEFINE_BINARY(sym_ne, std::not_equal_to<>(), ne, SymBool)
// 定义一个二进制操作符模板，使用默认的比较器 std::less<>，返回 SymBool 类型
DEFINE_BINARY(sym_lt, std::less<>(), lt, SymBool)

// 定义一个二进制操作符模板，使用 std::less_equal<> 比较器，返回 SymBool 类型
DEFINE_BINARY(sym_le, std::less_equal<>(), le, SymBool)

// 定义一个二进制操作符模板，使用 std::greater<> 比较器，返回 SymBool 类型
DEFINE_BINARY(sym_gt, std::greater<>(), gt, SymBool)

// 定义一个二进制操作符模板，使用 std::greater_equal<> 比较器，返回 SymBool 类型
DEFINE_BINARY(sym_ge, std::greater_equal<>(), ge, SymBool)

// 定义一个二进制操作符模板，使用 std::min 函数，返回 SymInt 类型
DEFINE_BINARY(min, std::min, sym_min, SymInt)

// 定义一个二进制操作符模板，使用 std::max 函数，返回 SymInt 类型
DEFINE_BINARY(max, std::max, sym_max, SymInt)
// clang-format on

// 将 SymInt 类型转换为 SymFloat 类型的转换操作符
SymInt::operator SymFloat() const {
  // 如果可以转换为整数，则返回转换后的 SymFloat 对象
  if (auto ma = maybe_as_int()) {
    return SymFloat(double(*ma));
  } else {
    // 否则返回基于符号节点的 SymFloat 对象
    return SymFloat(toSymNodeImplUnowned()->sym_float());
  }
}

// 比较两个 SymInt 对象是否相同的方法
bool SymInt::is_same(const SymInt& other) const {
  // 如果一个是堆分配而另一个不是，则返回 false
  if (is_heap_allocated() != other.is_heap_allocated()) {
    return false;
  }
  // 如果两者都不是堆分配且不相等，则返回 false
  if (!is_heap_allocated() && this->operator!=(other)) {
    return false;
  }
  // 如果两者都是堆分配且底层符号节点不相同，则返回 false
  if (is_heap_allocated() &&
      toSymNodeImplUnowned() != other.toSymNodeImplUnowned()) {
    return false;
  }
  // 否则返回 true
  return true;
}

// 根据 SymInt 对象包装给定符号节点的方法
SymNode SymInt::wrap_node(const SymNode& base) const {
  // 如果可以作为整数，则使用整数值来包装符号节点
  if (auto ma = maybe_as_int()) {
    return base->wrap_int(*ma);
  } else {
    // 否则直接返回 SymInt 对象的符号节点
    return toSymNode();
  }
}

// 克隆 SymInt 对象的方法
SymInt SymInt::clone() const {
  // 如果可以作为整数，则克隆整数值
  if (auto ma = maybe_as_int()) {
    return SymInt(*ma);
  } else {
    // 否则克隆底层符号节点的副本
    return SymInt(toSymNodeImplUnowned()->clone());
  }
}

// 保护整数值的方法，用于调试
int64_t SymInt::guard_int(const char* file, int64_t line) const {
  // 如果可以作为整数，则返回整数值
  if (auto ma = maybe_as_int()) {
    return *ma;
  } else {
    // 否则调用底层符号节点的保护整数方法
    return toSymNodeImplUnowned()->guard_int(file, line);
  }
}

// 期望 SymInt 对象表示的大小是否符合预期的方法
bool SymInt::expect_size(const char* file, int64_t line) const {
  // 如果可以作为整数，则判断整数值是否大于等于 0
  if (auto ma = maybe_as_int()) {
    return *ma >= 0;
  } else {
    // 否则调用底层符号节点的期望大小方法
    return toSymNodeImplUnowned()->expect_size(file, line);
  }
}

// 对 SymInt 对象取负操作符的重载
SymInt operator-(const SymInt& s) {
  // 如果可以作为整数，则返回其相反数的 SymInt 对象
  if (auto ma = s.maybe_as_int()) {
    const auto val = *ma;
    // 注意：对 `-std::numeric_limits<decltype(val)>::min()` 的结果是未定义的
    // 但在许多平台上，它等于 self + 设置 Carry/Overflow 标志
    // 在优化代码中，这可能会影响 `check_range` 条件的结果
    // 通过使用避免改变标志的三元运算符进行的 workaround
#if C10_HAS_BUILTIN_OVERFLOW()
    std::decay_t<decltype(val)> out = 0;
    if (C10_UNLIKELY(__builtin_sub_overflow(out, val, &out))) {
      return SymInt(val);
    }
    return SymInt(out);
#else
    // 如果不支持内建溢出检查，则使用条件运算符来避免出现问题
    constexpr auto val_min = std::numeric_limits<decltype(val)>::min();
    return SymInt(val != val_min ? -val : val_min);
#endif
  } else {
    // 否则返回基于符号节点的 SymInt 对象的负值
    return SymInt(s.toSymNodeImplUnowned()->neg());
  }
}

// SymInt 对象的乘法赋值操作符重载
void SymInt::operator*=(const SymInt& sci) {
  *this = *this * sci;
}

// SymInt 对象的除法赋值操作符重载
void SymInt::operator/=(const SymInt& sci) {
  *this = *this / sci;
}

// SymInt 对象的加法赋值操作符重载
void SymInt::operator+=(const SymInt& sci) {
  *this = *this + sci;
}

// SymInt 对象的输出流插入操作符重载
std::ostream& operator<<(std::ostream& os, const SymInt& s) {
  // 如果是堆分配的，则输出底层符号节点的字符串表示
  if (s.is_heap_allocated()) {
    os << s.toSymNodeImplUnowned()->str();
  } else {
    // 否则输出其作为整数的值
    os << s.as_int_unchecked();
  }
  return os;
}

// 用于实现身份转换时不增加引用计数的模板结构体
template <typename T>
struct Convert {};

// 空的特化声明，用于标志模板特化的结束
template <>
# 定义一个模板结构体 Convert，用于类型转换
struct Convert<SymInt> {
  # 重载函数调用运算符，返回参数本身，用于SymInt类型
  const SymInt& operator()(const SymInt& a) {
    return a;
  }
};

# 特化模板结构体 Convert 为 SymFloat 类型
template <>
struct Convert<SymFloat> {
  # 重载函数调用运算符，将 SymInt 类型转换为 SymFloat 类型并返回
  SymFloat operator()(const SymInt& a) {
    return a;
  }
};

# 宏定义，用于定义 SymInt 类型与标量类型之间的运算符重载
#define DEFINE_SYMINT_OP_INTONLY(scalar_t, RetTy) \
  # 定义 SymInt 类型与 scalar_t 类型取模运算符重载
  RetTy operator%(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) % RetTy(b);        \
  };                                              \
  # 定义 scalar_t 类型与 SymInt 类型取模运算符重载
  RetTy operator%(scalar_t a, const SymInt& b) {  \
    return RetTy(a) % Convert<RetTy>()(b);        \
  };

# 宏定义，用于定义 SymInt 类型与标量类型之间的多种运算符重载
#define DEFINE_SYMINT_OP(scalar_t, RetTy)        \
  # 定义 SymInt 类型与 scalar_t 类型加法运算符重载
  RetTy operator+(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) + RetTy(b);       \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型减法运算符重载
  RetTy operator-(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) - RetTy(b);       \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型乘法运算符重载
  RetTy operator*(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) * RetTy(b);       \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型除法运算符重载
  RetTy operator/(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) / RetTy(b);       \
  };                                             \
  # 定义 scalar_t 类型与 SymInt 类型加法运算符重载
  RetTy operator+(scalar_t a, const SymInt& b) { \
    return RetTy(a) + Convert<RetTy>()(b);       \
  };                                             \
  # 定义 scalar_t 类型与 SymInt 类型减法运算符重载
  RetTy operator-(scalar_t a, const SymInt& b) { \
    return RetTy(a) - Convert<RetTy>()(b);       \
  };                                             \
  # 定义 scalar_t 类型与 SymInt 类型乘法运算符重载
  RetTy operator*(scalar_t a, const SymInt& b) { \
    return RetTy(a) * Convert<RetTy>()(b);       \
  };                                             \
  # 定义 scalar_t 类型与 SymInt 类型除法运算符重载
  RetTy operator/(scalar_t a, const SymInt& b) { \
    return RetTy(a) / Convert<RetTy>()(b);       \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型相等比较运算符重载
  bool operator==(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) == RetTy(b);      \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型不等比较运算符重载
  bool operator!=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) != RetTy(b);      \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型小于比较运算符重载
  bool operator<(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) < RetTy(b);       \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型小于等于比较运算符重载
  bool operator<=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) <= RetTy(b);      \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型大于比较运算符重载
  bool operator>(const SymInt& a, scalar_t b) {  \
    return Convert<RetTy>()(a) > RetTy(b);       \
  };                                             \
  # 定义 SymInt 类型与 scalar_t 类型大于等于比较运算符重载
  bool operator>=(const SymInt& a, scalar_t b) { \
    return Convert<RetTy>()(a) >= RetTy(b);      \
  };                                             \
  # 定义 scalar_t 类型与 SymInt 类型相等比较运算符重载
  bool operator==(scalar_t a, const SymInt& b) { \
    return RetTy(a) == Convert<RetTy>()(b);      \
  };                                             \
  # 定义 scalar_t 类型与 SymInt 类型不等比较运算符重载
  bool operator!=(scalar_t a, const SymInt& b) {
  return RetTy(a) != Convert<RetTy>()(b);      \  
                                                # 返回 a 是否不等于 b 转换为 RetTy 类型后的结果
  };                                            \  
                                                # 结束 operator!= 的定义

  bool operator<(scalar_t a, const SymInt& b) {  \  
                                                # 定义一个 operator< 函数，比较 a 和 b 的大小
    return RetTy(a) < Convert<RetTy>()(b);       \  
                                                # 返回 a 是否小于 b 转换为 RetTy 类型后的结果
  };                                            \  
                                                # 结束 operator< 的定义

  bool operator<=(scalar_t a, const SymInt& b) { \  
                                                # 定义一个 operator<= 函数，比较 a 和 b 的大小
    return RetTy(a) <= Convert<RetTy>()(b);      \  
                                                # 返回 a 是否小于等于 b 转换为 RetTy 类型后的结果
  };                                            \  
                                                # 结束 operator<= 的定义

  bool operator>(scalar_t a, const SymInt& b) {  \  
                                                # 定义一个 operator> 函数，比较 a 和 b 的大小
    return RetTy(a) > Convert<RetTy>()(b);       \  
                                                # 返回 a 是否大于 b 转换为 RetTy 类型后的结果
  };                                            \  
                                                # 结束 operator> 的定义

  bool operator>=(scalar_t a, const SymInt& b) { \  
                                                # 定义一个 operator>= 函数，比较 a 和 b 的大小
    return RetTy(a) >= Convert<RetTy>()(b);      \  
                                                # 返回 a 是否大于等于 b 转换为 RetTy 类型后的结果
  };                                            \  
                                                # 结束 operator>= 的定义
DEFINE_SYMINT_OP_INTONLY(int64_t, SymInt)
// 定义一个模板宏，用于生成仅适用于 int64_t 类型的 SymInt 操作

DEFINE_SYMINT_OP_INTONLY(int32_t, SymInt)
// 定义一个模板宏，用于生成仅适用于 int32_t 类型的 SymInt 操作

DEFINE_SYMINT_OP_INTONLY(uint64_t, SymInt)
// 定义一个模板宏，用于生成仅适用于 uint64_t 类型的 SymInt 操作

DEFINE_SYMINT_OP_INTONLY(uint32_t, SymInt)
// 定义一个模板宏，用于生成仅适用于 uint32_t 类型的 SymInt 操作

DEFINE_SYMINT_OP(int64_t, SymInt)
// 定义一个模板宏，用于生成适用于 int64_t 类型的 SymInt 操作

DEFINE_SYMINT_OP(int32_t, SymInt)
// 定义一个模板宏，用于生成适用于 int32_t 类型的 SymInt 操作
// 确保常量的工作正常

DEFINE_SYMINT_OP(uint64_t, SymInt)
// 定义一个模板宏，用于生成适用于 uint64_t 类型的 SymInt 操作

DEFINE_SYMINT_OP(uint32_t, SymInt)
// 定义一个模板宏，用于生成适用于 uint32_t 类型的 SymInt 操作

DEFINE_SYMINT_OP(double, SymFloat)
// 定义一个模板宏，用于生成适用于 double 类型的 SymFloat 操作

DEFINE_SYMINT_OP(float, SymFloat)
// 定义一个模板宏，用于生成适用于 float 类型的 SymFloat 操作
// 仅出于完整性考虑

#if defined(__APPLE__)
DEFINE_SYMINT_OP_INTONLY(size_t, SymInt)
// 定义一个模板宏，用于生成仅适用于 size_t 类型的 SymInt 操作，这在 macOS 上是必需的

DEFINE_SYMINT_OP(size_t, SymInt)
// 定义一个模板宏，用于生成适用于 size_t 类型的 SymInt 操作，这在 macOS 上是必需的
#endif

} // namespace c10
```