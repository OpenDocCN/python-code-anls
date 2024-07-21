# `.\pytorch\c10\core\SymNodeImpl.h`

```
#pragma once
// 一旦编译器到此处，就会防止头文件的多次包含

#include <c10/macros/Export.h>
// 导入 C10 库的导出宏

#include <c10/util/ArrayRef.h>
// 导入 C10 库的 ArrayRef 工具

#include <c10/util/Exception.h>
// 导入 C10 库的异常处理工具

#include <c10/util/Optional.h>
// 导入 C10 库的可选值工具

#include <c10/util/intrusive_ptr.h>
// 导入 C10 库的内部指针工具

#include <cstdint>
// 导入标准 C++ 的整数类型

#include <ostream>
// 导入输出流标准库

#include <string>
// 导入字符串标准库

namespace c10 {
// 进入 C10 命名空间

class SymNodeImpl;
// 声明 SymNodeImpl 类

using SymNode = c10::intrusive_ptr<SymNodeImpl>;
// 使用 C10 提供的内部指针工具定义 SymNode 类型

// 当你添加一个方法时，你需要编辑以下文件：
// torch/csrc/jit/python/init.cpp
// torch/csrc/utils/python_symnode.h
// c10/core/ConstantSymNodeImpl.h
class C10_API SymNodeImpl : public c10::intrusive_ptr_target {
// 声明 SymNodeImpl 类，继承自 c10 内部指针目标类
 public:
  ~SymNodeImpl() override = default;
  // 默认虚析构函数

  template <typename T>
  c10::intrusive_ptr<T> dyn_cast() const {
    // 动态类型转换模板方法，返回指定类型的内部指针
    return c10::intrusive_ptr<T>::reclaim_copy(dynamic_cast<T*>(this));
  }

  // 以下方法为虚方法，需要在实现 LTC 版本时实现具体逻辑

  virtual bool is_int() {
    // 检查是否为整数，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual bool is_bool() {
    // 检查是否为布尔值，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual bool is_float() {
    // 检查是否为浮点数，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual bool is_nested_int() const {
    // 检查是否为嵌套整数，目前始终返回 false
    return false;
  }

  virtual SymNode add(const SymNode& other) {
    // 实现加法操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode sub(const SymNode& other) {
    // 实现减法操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode mul(const SymNode& other) {
    // 实现乘法操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode truediv(const SymNode& other) {
    // 实现真除操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode float_truediv(const SymNode& other) {
    // 实现浮点数真除操作，委托给 truediv 方法
    return truediv(other);
  }

  virtual SymNode int_truediv(const SymNode& other) {
    // 实现整数真除操作，委托给 truediv 方法
    return truediv(other);
  }

  virtual SymNode pow(const SymNode& other) {
    // 实现指数操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode float_pow(const SymNode& other) {
    // 实现浮点数指数操作，委托给 pow 方法
    return pow(other);
  }

  virtual SymNode pow_by_natural(const SymNode& other) {
    // 实现自然数指数操作，委托给 pow 方法
    return pow(other);
  }

  virtual SymNode floordiv(const SymNode& other) {
    // 实现整数地板除操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode int_floordiv(const SymNode& other) {
    // 实现整数地板除操作，委托给 floordiv 方法
    return floordiv(other);
  }

  virtual SymNode mod(const SymNode& other) {
    // 实现取模操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode eq(const SymNode& other) {
    // 实现等于比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode ne(const SymNode& other) {
    // 实现不等于比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode gt(const SymNode& other) {
    // 实现大于比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode lt(const SymNode& other) {
    // 实现小于比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode le(const SymNode& other) {
    // 实现小于等于比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode ge(const SymNode& other) {
    // 实现大于等于比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode ceil() {
    // 实现向上取整操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode floor() {
    // 实现向下取整操作，目前未实现
    TORCH_CHECK(false, "NYI");
  }

  virtual SymNode neg() {
    // 实现取负操作，目前未实现
    TORCH_CHECK(false, "NYI");
  };

  virtual SymNode sym_min(const SymNode& other) {
    // 实现符号最小值比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  };

  virtual SymNode sym_max(const SymNode& other) {
    // 实现符号最大值比较操作，目前未实现
    TORCH_CHECK(false, "NYI");
  };
};

} // namespace c10
// 结束 C10 命名空间
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI"（Not Yet Implemented）信息
  TORCH_CHECK(false, "NYI");
};

// 返回与另一节点的符号逻辑或
virtual SymNode sym_or(const SymNode& other) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回与另一节点的符号逻辑与
virtual SymNode sym_and(const SymNode& other) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回符号逻辑非
virtual SymNode sym_not() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回符号条件表达式（如果-则-否）
virtual SymNode sym_ite(const SymNode& then_val, const SymNode& else_val) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回是否连续的符号节点（忽略 self，仅使用参数）
// NB: self 在此处被忽略，只使用了参数
virtual SymNode is_contiguous(
    ArrayRef<SymNode> sizes,
    ArrayRef<SymNode> strides) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回二维通道最后连续的符号节点
virtual SymNode is_channels_last_contiguous_2d(
    ArrayRef<SymNode> sizes,
    ArrayRef<SymNode> strides) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回三维通道最后连续的符号节点
virtual SymNode is_channels_last_contiguous_3d(
    ArrayRef<SymNode> sizes,
    ArrayRef<SymNode> strides) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回二维通道最后步幅的符号节点
virtual SymNode is_channels_last_strides_2d(
    ArrayRef<SymNode> sizes,
    ArrayRef<SymNode> strides) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回三维通道最后步幅的符号节点
virtual SymNode is_channels_last_strides_3d(
    ArrayRef<SymNode> sizes,
    ArrayRef<SymNode> strides) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回是否非重叠且密集的符号节点
virtual SymNode is_non_overlapping_and_dense(
    ArrayRef<SymNode> sizes,
    ArrayRef<SymNode> strides) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 克隆符号节点
virtual SymNode clone() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回符号浮点数
virtual SymNode sym_float() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
}

// 封装整数为符号节点
virtual SymNode wrap_int(int64_t num) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 封装浮点数为符号节点
virtual SymNode wrap_float(double num) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 封装布尔值为符号节点
virtual SymNode wrap_bool(bool num) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 保护整数值
virtual int64_t guard_int(const char* file, int64_t line) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 保护布尔值
virtual bool guard_bool(const char* file, int64_t line) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 保护浮点数值
virtual double guard_float(const char* file, int64_t line) {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 保护大小不可知
virtual bool guard_size_oblivious(const char* file, int64_t line) {
  // 默认情况下未支持的 SymBools 没有改进，用更好的实现替换！
  return guard_bool(file, line);
}

// 期望为真
virtual bool expect_true(const char* file, int64_t line) {
  // 默认情况下未支持的 SymBools 没有改进，用更好的实现替换！
  return guard_bool(file, line);
};

// 期望大小
virtual bool expect_size(const char* file, int64_t line) {
  // 默认情况下未支持的 SymInts 没有改进，用更好的实现替换！
  return ge(wrap_int(0))->guard_bool(file, line);
};

// 返回整数值
virtual int64_t int_() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回布尔值
virtual bool bool_() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 是否有提示
virtual bool has_hint() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回字符串表示
virtual std::string str() {
  // 使用 TORCH_CHECK 断言检查条件 false，并输出 "NYI" 信息
  TORCH_CHECK(false, "NYI");
};

// 返回嵌套整数（可选）
virtual std::optional<int64_t> nested_int() {
    // 返回空的std::optional对象，表示没有值
    return c10::nullopt;
  }
  // 返回空的std::optional对象，表示没有值
  virtual std::optional<int64_t> nested_int_coeff() {
    return c10::nullopt;
  }
  // 返回空的std::optional对象，表示没有值
  virtual std::optional<int64_t> constant_int() {
    return c10::nullopt;
  }
  // 返回空的std::optional对象，表示没有值
  virtual std::optional<bool> constant_bool() {
    return c10::nullopt;
  }
  // 返回空的std::optional对象，表示没有值
  virtual std::optional<int64_t> maybe_as_int() {
    return c10::nullopt;
  }
  // 始终返回false，表示不是常量
  virtual bool is_constant() {
    return false;
  }
  // 始终返回true，表示是符号化的
  virtual bool is_symbolic() {
    return true;
  }
  // 将对象转换为字符串并输出到给定的输出流中
  std::ostream& operator<<(std::ostream& os) {
    os << str();  // 调用当前对象的str()方法，将结果输出到os流
    return os;    // 返回修改后的流对象
  }
};

} // namespace c10
```