# `.\pytorch\aten\src\ATen\core\NestedIntSymNodeImpl.cpp`

```
#include <ATen/core/NestedIntSymNodeImpl.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/Exception.h>

namespace c10 {

namespace {
// 比较函数，检查两个 SymNodeImpl 对象是否相等
bool _eq(const char* op, c10::SymNodeImpl* lhs, c10::SymNodeImpl* rhs) {
  // 断言左操作数是一个嵌套整数节点
  TORCH_INTERNAL_ASSERT(lhs->is_nested_int());
  // 获取右操作数的嵌套整数值
  std::optional<int64_t> c = rhs->nested_int();
  // 比较两个嵌套整数节点的值和系数是否相等
  return (
      c.has_value() && lhs->nested_int() == *c &&
      lhs->nested_int_coeff() == rhs->nested_int_coeff());
}

// 比较函数，检查左操作数是否大于或等于右操作数
bool _ge(const char* op, c10::SymNodeImpl* lhs, c10::SymNodeImpl* rhs) {
  // 如果左操作数是一个嵌套整数节点
  if (auto mb_si = lhs->nested_int()) {
    // 如果右操作数也是一个嵌套整数节点
    if (auto mb_si2 = rhs->nested_int()) {
      // 如果两个嵌套整数节点的值相等，则比较它们的系数
      if (*mb_si == *mb_si2) {
        return lhs->nested_int_coeff() >= rhs->nested_int_coeff();
      }
      // 如果两个节点的值不相等，则抛出异常，表示关系不确定
      TORCH_CHECK(false, "nested int ", op, ": Relation is indeterminate");
    }
    // 如果右操作数是常数整数且值小于等于2，则返回 true
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (rhs->constant_int() && *rhs->constant_int() <= 2) {
      return true;
    }
    // 如果右操作数不是嵌套整数或常数整数满足条件，则抛出异常，表示关系不确定
    TORCH_CHECK(false, "nested int ", op, ": Relation is indeterminate");
  } else if (rhs->nested_int()) {
    // 如果右操作数是嵌套整数，且左操作数的常数整数值小于2，则返回 false
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (lhs->constant_int() && *lhs->constant_int() < 2) {
      return false;
    }
    // 如果左操作数不是嵌套整数或不满足条件，则抛出异常，表示关系不确定
    TORCH_CHECK(false, "nested int ", op, ": Relation is indeterminate");
  }
  // 断言失败，预期至少有一个嵌套整数节点
  TORCH_INTERNAL_ASSERT(false, "expect at least one nested int");
}
} // namespace

// 比较函数，返回嵌套整数节点与另一个节点的相等性
c10::SymNode NestedIntSymNodeImpl::eq(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      _eq("eq", this, other.get())));
}

// 比较函数，返回嵌套整数节点与另一个节点的不等性
c10::SymNode NestedIntSymNodeImpl::ne(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      !_eq("ne", this, other.get())));
}

// 比较函数，返回嵌套整数节点与另一个节点的大于或等于关系
c10::SymNode NestedIntSymNodeImpl::ge(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      _ge("ge", this, other.get())));
}

// 比较函数，返回嵌套整数节点与另一个节点的大于关系
c10::SymNode NestedIntSymNodeImpl::gt(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      !_ge("gt", other.get(), this)));
}

// 比较函数，返回嵌套整数节点与另一个节点的小于关系
c10::SymNode NestedIntSymNodeImpl::lt(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      !_ge("lt", this, other.get())));
}

// 比较函数，返回嵌套整数节点与另一个节点的小于或等于关系
c10::SymNode NestedIntSymNodeImpl::le(const c10::SymNode& other) {
  return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(
      _ge("le", other.get(), this)));
}

// 乘法操作，返回当前嵌套整数节点与另一个节点的乘积
c10::SymNode NestedIntSymNodeImpl::mul(const c10::SymNode& other) {
  // 检查右操作数不能是嵌套整数
  TORCH_CHECK(!other->nested_int(), "nested int cannot be multiplied by nested int");
  // 获取右操作数的常数整数值
  std::optional<int64_t> c = other->constant_int();
  // 断言右操作数必须是常数整数
  TORCH_CHECK(c.has_value());
  // 返回新的嵌套整数节点，其值为当前节点值乘以右操作数的常数整数值
  return SymNode(c10::make_intrusive<NestedIntSymNodeImpl>(val_, coeff_ * *c));
}

// 克隆函数，返回当前嵌套整数节点的一个副本
c10::SymNode NestedIntSymNodeImpl::clone() {
  // 返回新的嵌套整数节点，其值和系数与当前节点相同
  return SymNode(c10::make_intrusive<NestedIntSymNodeImpl>(val_, coeff_));
}

} // namespace c10
```