# `.\pytorch\c10\util\ThreadLocal.h`

```
cpp
#pragma once

#include <c10/macros/Macros.h>

/**
 * Android versions with libgnustl incorrectly handle thread_local C++
 * qualifier with composite types. NDK up to r17 version is affected.
 *
 * (A fix landed on Jun 4 2018:
 * https://android-review.googlesource.com/c/toolchain/gcc/+/683601)
 *
 * In such cases, use c10::ThreadLocal<T> wrapper
 * which is `pthread_*` based with smart pointer semantics.
 *
 * In addition, convenient macro C10_DEFINE_TLS_static is available.
 * To define static TLS variable of type std::string, do the following
 * ```
 *  C10_DEFINE_TLS_static(std::string, str_tls_);
 *  ///////
 *  {
 *    *str_tls_ = "abc";
 *    assert(str_tls_->length(), 3);
 *  }
 * ```
 *
 * (see c10/test/util/ThreadLocal_test.cpp for more examples)
 */
#if !defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(C10_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604
#define C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE
#endif // defined(C10_ANDROID) && defined(__GLIBCXX__) && __GLIBCXX__ < 20180604

#endif // !defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

#if defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
#include <c10/util/Exception.h>
#include <errno.h>
#include <pthread.h>
#include <memory>
namespace c10 {

/**
 * @brief Temporary thread_local C++ qualifier replacement for Android
 * based on `pthread_*`.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal {
 public:
  // Constructor: Creates a pthread_key for managing thread-local storage of Type.
  ThreadLocal() {
    pthread_key_create(
        &key_, [](void* buf) { delete static_cast<Type*>(buf); });
  }

  // Destructor: Cleans up pthread_key and deletes the stored Type object.
  ~ThreadLocal() {
    if (void* current = pthread_getspecific(key_)) {
      delete static_cast<Type*>(current);
    }

    pthread_key_delete(key_);
  }

  // Copy constructor and assignment operator are deleted.
  ThreadLocal(const ThreadLocal&) = delete;
  ThreadLocal& operator=(const ThreadLocal&) = delete;

  // Accessor function to retrieve the thread-local object of Type.
  Type& get() {
    if (void* current = pthread_getspecific(key_)) {
      return *static_cast<Type*>(current);
    }

    std::unique_ptr<Type> ptr = std::make_unique<Type>();
    if (0 == pthread_setspecific(key_, ptr.get())) {
      return *ptr.release();
    }

    int err = errno;
    TORCH_INTERNAL_ASSERT(false, "pthread_setspecific() failed, errno = ", err);
  }

  // Overloaded dereference operator to retrieve the thread-local object.
  Type& operator*() {
    return get();
  }

  // Overloaded arrow operator to access methods of the thread-local object.
  Type* operator->() {
    return &get();
  }

 private:
  pthread_key_t key_; // pthread_key_t variable for managing thread-local storage.
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name) static ::c10::ThreadLocal<Type> Name

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \
  static ::c10::ThreadLocal<Type> Name

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \
  ::c10::ThreadLocal<Type> Class::Name

#else // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)

namespace c10 {

/**
 * @brief Default thread_local implementation for non-Android cases.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>


In the provided C++ code snippet, the `ThreadLocal` class and related macros define thread-local storage mechanisms. Starting with conditional compilation directives, it first checks if custom thread-local storage is preferred based on Android and GCC compatibility issues. Inside the `c10` namespace, the `ThreadLocal` class template provides a pthread-based replacement for thread_local on Android, managing storage for types with default constructors. It includes a constructor for creating a pthread key, a destructor for cleanup, and methods (`get()`, `operator*`, `operator->`) for accessing thread-local instances. Macros like `C10_DEFINE_TLS_static` simplify defining static thread-local variables. This code facilitates safe and efficient thread-specific data management, crucial for multi-threaded applications.
class ThreadLocal {
 public:
  using Accessor = Type* (*)();  // 定义类型别名 Accessor 为指向函数的指针，该函数返回 Type* 类型
  explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}  // ThreadLocal 类的构造函数，接受一个 Accessor 参数并初始化成员变量 accessor_

  ThreadLocal(const ThreadLocal&) = delete;  // 删除拷贝构造函数，禁止对象的拷贝构造
  ThreadLocal& operator=(const ThreadLocal&) = delete;  // 删除赋值运算符重载，禁止对象的赋值操作

  Type& get() {  // 返回 Type 类型的引用，调用 accessor_ 函数获取
    return *accessor_();
  }

  Type& operator*() {  // 解引用运算符重载，返回调用 get() 函数的结果
    return get();
  }

  Type* operator->() {  // 箭头运算符重载，返回指向调用 get() 函数的结果的指针
    return &get();
  }

 private:
  Accessor accessor_;  // 成员变量，保存函数指针 accessor_
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name)     \  // 定义一个宏 C10_DEFINE_TLS_static，接受 Type 类型和 Name 变量名参数
  static ::c10::ThreadLocal<Type> Name([]() { \  // 在当前作用域中定义一个静态的 ThreadLocal 对象，用 lambda 表达式初始化，返回指向静态 thread_local 变量的指针
    static thread_local Type var;             \  // 定义静态的 thread_local 变量 var
    return &var;                              \  // 返回指向 var 的指针
  })

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \  // 定义一个宏 C10_DECLARE_TLS_class_static，接受 Class 类型、Type 类型和 Name 变量名参数
  static ::c10::ThreadLocal<Type> Name  // 声明一个静态的 ThreadLocal 对象 Name，属于 Class 类型的静态成员

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \  // 定义一个宏 C10_DEFINE_TLS_class_static，接受 Class 类型、Type 类型和 Name 变量名参数
  ::c10::ThreadLocal<Type> Class::Name([]() {          \  // 在 Class 类的作用域中定义一个静态的 ThreadLocal 对象 Name，用 lambda 表达式初始化，返回指向静态 thread_local 变量的指针
    static thread_local Type var;                      \  // 定义静态的 thread_local 变量 var
    return &var;                                       \  // 返回指向 var 的指针
  })

#endif // defined(C10_PREFER_CUSTOM_THREAD_LOCAL_STORAGE)
```