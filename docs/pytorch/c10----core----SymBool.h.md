# `.\pytorch\c10\core\SymBool.h`

```py
#pragma once
// 防止头文件被多次包含的预处理指令

#include <c10/core/SymNodeImpl.h>
// 包含SymNodeImpl类的头文件

#include <c10/macros/Export.h>
// 包含C10_API宏的头文件，用于导出符号

#include <c10/util/Exception.h>
// 包含异常处理相关的头文件

#include <c10/util/Optional.h>
// 包含可选值类型的头文件

#include <c10/util/intrusive_ptr.h>
// 包含intrusive_ptr模板的头文件

#include <cstdint>
// 包含整数类型的标准头文件

#include <ostream>
// 包含输出流相关的标准头文件

#include <utility>
// 包含实用工具相关的标准头文件

namespace c10 {

class C10_API SymBool {
// 声明SymBool类，使用C10_API导出符号

 public:
  /*implicit*/ SymBool(bool b) : data_(b){};
  // 隐式构造函数，根据布尔值b初始化SymBool对象的data_

  SymBool(SymNode ptr) : data_(false), ptr_(std::move(ptr)) {
    // 构造函数，根据SymNode对象ptr初始化SymBool对象，要求ptr必须是bool类型的SymNode
    TORCH_CHECK(ptr_->is_bool());
  };

  SymBool() : data_(false) {}
  // 默认构造函数，初始化SymBool对象的data_为false

  SymNodeImpl* toSymNodeImplUnowned() const {
    // 返回指向SymNodeImpl对象的指针，不拥有所有权
    return ptr_.get();
  }

  SymNodeImpl* release() && {
    // 移动构造函数，释放对SymNodeImpl对象的所有权
    return std::move(ptr_).release();
  }

  SymNode toSymNodeImpl() const;
  // 返回SymNode对象，确保返回一个SymNode对象，必要时使用base进行封装

  SymNode wrap_node(const SymNode& base) const;
  // 使用base封装SymNode对象，并返回SymNode对象

  bool expect_bool() const {
    // 检查是否可以转换为bool，如果可以则返回其值
    std::optional<bool> c = maybe_as_bool();
    TORCH_CHECK(c.has_value());
    return *c;
  }

  SymBool sym_and(const SymBool&) const;
  // 对两个SymBool对象进行逻辑与操作，并返回结果

  SymBool sym_or(const SymBool&) const;
  // 对两个SymBool对象进行逻辑或操作，并返回结果

  SymBool sym_not() const;
  // 对SymBool对象进行逻辑非操作，并返回结果

  SymBool operator&(const SymBool& other) const {
    // 重载位与运算符，返回逻辑与操作的结果
    return sym_and(other);
  }
  
  SymBool operator|(const SymBool& other) const {
    // 重载位或运算符，返回逻辑或操作的结果
    return sym_or(other);
  }
  
  SymBool operator~() const {
    // 重载位非运算符，返回逻辑非操作的结果
    return sym_not();
  }

  bool guard_bool(const char* file, int64_t line) const;
  // 插入一个保护来确保布尔值为其具体值，并返回该值。注意，C++比较操作默认返回bool，因此通常不必调用此函数

  bool expect_true(const char* file, int64_t line) const;
  // 检查布尔值为true，并返回该值

  bool guard_size_oblivious(const char* file, int64_t line) const;
  // 插入一个保护以确保大小无关的条件成立，并返回布尔值

  bool has_hint() const;
  // 检查是否具有提示信息

  bool as_bool_unchecked() const {
    // 返回未检查的布尔值
    return data_;
  }

  std::optional<bool> maybe_as_bool() const {
    // 返回可能的布尔值，如果未分配堆内存则返回data_，否则返回基于SymNodeImpl对象的常量布尔值
    if (!is_heap_allocated()) {
      return c10::make_optional(data_);
    }
    return toSymNodeImplUnowned()->constant_bool();
  }

  bool is_heap_allocated() const {
    // 检查SymBool对象是否分配了堆内存
    return ptr_;
  }

 private:
  // TODO: optimize to union
  bool data_;
  // 布尔数据成员，用于存储布尔值

  SymNode ptr_;
  // SymNode对象成员，用于存储SymNode对象
};

C10_API std::ostream& operator<<(std::ostream& os, const SymBool& s);
// 重载输出流操作符，以便将SymBool对象输出到ostream流中

#define TORCH_SYM_CHECK(cond, ...) \
  TORCH_CHECK((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)
// 宏定义，检查条件cond为true，否则报告错误，并允许附加信息(__VA_ARGS__)

#define TORCH_SYM_INTERNAL_ASSERT(cond, ...) \
  TORCH_INTERNAL_ASSERT((cond).expect_true(__FILE__, __LINE__), __VA_ARGS__)
// 宏定义，内部断言，检查条件cond为true，否则报告内部错误，并允许附加信息(__VA_ARGS__)

inline bool guard_size_oblivious(bool b, const char* file, int64_t line) {
  // 内联函数，无关大小的保护，返回布尔值b
  return b;
}

inline bool guard_size_oblivious(
    const c10::SymBool& b,
    const char* file,
    int64_t line) {
  // 内联函数，无关大小的保护，调用SymBool对象的guard_size_oblivious方法，并返回结果
  return b.guard_size_oblivious(file, line);
}

#define TORCH_GUARD_SIZE_OBLIVIOUS(cond) \
  c10::guard_size_oblivious((cond), __FILE__, __LINE__)
// 宏定义，无关大小的保护，调用guard_size_oblivious函数，并传递文件名和行号作为参数

} // namespace c10
// 结束c10命名空间
```