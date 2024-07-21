# `.\pytorch\aten\src\ATen\core\NestedIntSymNodeImpl.h`

```py
#pragma once

#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>
#include <string>

namespace c10 {

// 用于表示分层整数符号节点的实现类，继承自 SymNodeImpl 类
class TORCH_API NestedIntSymNodeImpl : public SymNodeImpl {
 public:
  // 构造函数，初始化 NestedIntSymNodeImpl 对象
  // 参数:
  // - val: 整数值
  // - coeff: 正整数标量，用于计算步长
  explicit NestedIntSymNodeImpl(int64_t val, int64_t coeff)
      : val_(val), coeff_(coeff) {}

  // 返回 false，表示不是布尔值
  bool bool_() override {
    return false;
  }

  // 返回 true，表示是整数值
  bool is_int() override {
    return true;
  }

  // 返回 false，表示不是浮点数
  bool is_float() override {
    return false;
  }

  // 返回 false，表示不是布尔值
  bool is_bool() override {
    return false;
  }

  // 返回 true，表示是分层整数
  bool is_nested_int() const override {
    return true;
  }

  // 返回 true，表示有提示信息
  bool has_hint() override {
    return true;
  }

  // 根据给定的整数 num 包装成 SymNode 对象
  c10::SymNode wrap_int(int64_t num) override {
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<int64_t>>(num));
  };

  // 报告错误，中止程序执行，用于守卫整数类型的期望
  int64_t guard_int(const char* file, int64_t line) override {
    TORCH_CHECK(false);
  }

  // 报告错误，中止程序执行，用于守卫浮点数类型的期望
  double guard_float(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a float");
  }

  // 报告错误，中止程序执行，用于守卫布尔类型的期望
  bool guard_bool(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a bool");
  }

  // 报告错误，中止程序执行，获取整数值
  int64_t int_() override {
    TORCH_CHECK(false);
  }

  // 返回对象的字符串表示形式
  std::string str() override {
    // 如果 coeff_ 等于 1，则返回形如 "jval_" 的字符串
    if (coeff_ == 1) {
      return "j" + std::to_string(val_);
    }
  // 将系数和值转换为字符串，并以 "*j" 分隔连接，返回结果字符串
  return std::to_string(coeff_) + "*j" + std::to_string(val_);
}

// NOTE [ Inequalities with nested int ]
//
// 嵌套整数的关系语义是，其被视为已知在特定范围内的整数，
//
//     j0 \in [2, int64_t::max]
//
// 这允许我们回答像 j0 >= 1 (True) 和 j0 == 0 (False) 这样的查询。
// 这是用于描述不规则张量模式的有用默认范围，因为大小是非负的，并且我们需要通过 0/1 的特化检查。
//
// [ Indeterminate inequalities error out ]
//
// 根据上述定义的语义，诸如 j0 < 3 这样的关系因此是不确定的。在我们当前的实现中，评估这样的关系会导致错误。
//
// 看似方便的是将不确定的关系定义为返回 False，但我们在使用 sympy 并行维护的实现不允许这样做。
//
// Sympy 只允许重写 Ge。其他关系 (Lt, Gt, Le) 都是由 Ge 派生的，例如 Lt(a, b) := !Ge(a, b)。
// 这意味着如果我们将不确定的 j0 >= 3 定义为 False，则同样不确定的 j0 < 3 将被评估为 True！
//
// [ Coefficient are assumed positive ]
//
// 为了计算不等式，我们认为嵌套整数的系数是正整数。
//
// 因此，在逻辑上不需要对其进行修改，因为 j0 >= k 意味着 coeff * j0 >= k
//
c10::SymNode eq(const c10::SymNode& other) override;
c10::SymNode ne(const c10::SymNode& other) override;
c10::SymNode ge(const c10::SymNode& other) override;
c10::SymNode gt(const c10::SymNode& other) override;
c10::SymNode lt(const c10::SymNode& other) override;
c10::SymNode le(const c10::SymNode& other) override;
c10::SymNode mul(const c10::SymNode& other) override;

std::optional<int64_t> nested_int() override {
  return val_;
}

std::optional<int64_t> nested_int_coeff() override {
  return coeff_;
}

bool is_symbolic() override {
  return false;
}

c10::SymNode clone() override;
#define DEFINE_BINARY_NOT_SUPPORTED(name)                           \
  // 定义一个宏，用于生成一个函数，该函数抛出错误，指示不支持指定的二元操作符
  c10::SymNode name(const c10::SymNode& other) override {           \
    TORCH_CHECK(false, #name " not supported by NestedIntSymNode"); \
  }

DEFINE_BINARY_NOT_SUPPORTED(add)
DEFINE_BINARY_NOT_SUPPORTED(sub)
DEFINE_BINARY_NOT_SUPPORTED(truediv)
DEFINE_BINARY_NOT_SUPPORTED(pow)
DEFINE_BINARY_NOT_SUPPORTED(floordiv)
DEFINE_BINARY_NOT_SUPPORTED(mod)
DEFINE_BINARY_NOT_SUPPORTED(sym_min)
DEFINE_BINARY_NOT_SUPPORTED(sym_max)
DEFINE_BINARY_NOT_SUPPORTED(sym_and)
DEFINE_BINARY_NOT_SUPPORTED(sym_or)

#undef DEFINE_BINARY_NOT_SUPPORTED

#define DEFINE_NOT_SUPPORTED(name)                                     \
  // 定义一个宏，用于生成一个函数，该函数抛出错误，指示不支持指定的操作
  c10::SymNode name() override {                                       \
    TORCH_CHECK(false, #name " is not supported by NestedIntSymNode"); \
  }

DEFINE_NOT_SUPPORTED(sym_not)
DEFINE_NOT_SUPPORTED(ceil)
DEFINE_NOT_SUPPORTED(floor)
DEFINE_NOT_SUPPORTED(neg)
DEFINE_NOT_SUPPORTED(sym_float)

#undef DEFINE_NOT_SUPPORTED

private:
// 类的私有成员变量，存储整数值
int64_t val_;
// 类的私有成员变量，存储系数值
int64_t coeff_;
};

} // namespace c10
```