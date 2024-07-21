# `.\pytorch\c10\util\UniqueVoidPtr.h`

```py
#pragma once
// 防止头文件被多次包含

#include <cstddef>
// 包含标准库头文件，定义了 size_t 类型

#include <memory>
// 包含智能指针相关的头文件，用于管理内存资源

#include <utility>
// 包含实用工具头文件，定义了一些实用的模板函数

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
// 包含 C10 库的宏定义头文件

namespace c10 {

using DeleterFnPtr = void (*)(void*);
// 定义函数指针类型 DeleterFnPtr，指向形如 void(void*) 的函数

namespace detail {

// 不执行任何删除操作的函数
C10_API void deleteNothing(void*);

// UniqueVoidPtr 是一个像 unique_ptr 一样的拥有智能指针，但有三个主要区别：
//
//    1) 它是专门为 void 类型定制的
//
//    2) 它专为一个函数指针形式的删除器设计：void(void* ctx)；
//       即删除器不会获取数据的引用，而只获取一个上下文指针（被擦除成 void*）。
//       实际上，内部实现中，这个指针具有对上下文的拥有引用和对数据的非拥有引用；
//       这也是为什么你使用 release_context() 而不是 release() 的原因
//       （传统的 release() API 并不提供足够信息来正确处理对象的释放）。
//
//    3) 当唯一指针被析构且上下文非空时，保证删除器被调用；
//       这与 std::unique_ptr 不同，后者在数据指针为空时不会调用删除器。
//
// 一些方法的类型稍有不同于 std::unique_ptr，以反映这一点。
//
class UniqueVoidPtr {
 private:
  // 生命周期与 ctx_ 绑定
  void* data_;  // 数据指针
  std::unique_ptr<void, DeleterFnPtr> ctx_;  // 上下文的唯一指针

 public:
  UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, &deleteNothing) {}
  // 默认构造函数：数据指针为空，上下文指针为空，并使用 deleteNothing 函数作为删除器

  explicit UniqueVoidPtr(void* data)
      : data_(data), ctx_(nullptr, &deleteNothing) {}
  // 显式构造函数：初始化数据指针为给定值，上下文指针为空，并使用 deleteNothing 函数作为删除器

  UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)
      : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing) {}
  // 构造函数：初始化数据指针为给定值，上下文指针为给定值（使用给定的上下文删除器或默认的 deleteNothing 函数作为删除器）

  void* operator->() const {
    return data_;
  }
  // 指针成员访问操作符重载：返回数据指针

  void clear() {
    ctx_ = nullptr;
    data_ = nullptr;
  }
  // 清空数据和上下文指针：将上下文指针和数据指针都置为空

  void* get() const {
    return data_;
  }
  // 获取数据指针

  void* get_context() const {
    return ctx_.get();
  }
  // 获取上下文指针

  void* release_context() {
    return ctx_.release();
  }
  // 释放上下文指针的拥有权

  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return std::move(ctx_);
  }
  // 移动上下文指针的拥有权

  C10_NODISCARD bool compare_exchange_deleter(
      DeleterFnPtr expected_deleter,
      DeleterFnPtr new_deleter) {
    if (get_deleter() != expected_deleter)
      return false;
    ctx_ = std::unique_ptr<void, DeleterFnPtr>(ctx_.release(), new_deleter);
    return true;
  }
  // 比较并交换删除器函数：如果当前的删除器函数与期望的删除器函数相同，则替换为新的删除器函数

  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    if (get_deleter() != expected_deleter)
      return nullptr;
    return static_cast<T*>(get_context());
  }
  // 类型转换上下文指针：如果当前的删除器函数与期望的删除器函数相同，则将上下文指针转换为给定类型 T*

  operator bool() const {
    return data_ || ctx_;
  }
  // 类型转换操作符重载：判断数据指针或上下文指针是否非空

  DeleterFnPtr get_deleter() const {
    return ctx_.get_deleter();
  }
  // 获取删除器函数指针

};

// 注意 [UniqueVoidPtr 的实现方式]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// UniqueVoidPtr 解决了张量数据分配器常见的问题，
// 即你感兴趣的数据指针（例如 float*）并不是实际的唯一指针，
// 而是一个需要释放上下文信息的指针。
//
// 在这里添加更多的注释或细节信息以帮助理解该类的实现方式。
// 在这里定义了四个重载运算符，用于比较 UniqueVoidPtr 和 nullptr。
// 这些运算符使得 UniqueVoidPtr 类型的对象能够像普通指针一样与 nullptr 进行比较。
inline bool operator==(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return !sp;  // 检查 UniqueVoidPtr 对象是否为空指针
}
inline bool operator==(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return !sp;  // 检查 UniqueVoidPtr 对象是否为空指针
}
inline bool operator!=(const UniqueVoidPtr& sp, std::nullptr_t) noexcept {
  return sp;   // 检查 UniqueVoidPtr 对象是否不为空指针
}
inline bool operator!=(std::nullptr_t, const UniqueVoidPtr& sp) noexcept {
  return sp;   // 检查 UniqueVoidPtr 对象是否不为空指针
}

} // namespace detail
} // namespace c10
```