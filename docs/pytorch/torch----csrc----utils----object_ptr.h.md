# `.\pytorch\torch\csrc\utils\object_ptr.h`

```
#pragma once
// 只允许头文件被编译一次

#include <torch/csrc/Export.h>
// 引入 Torch 的导出头文件，用于生成符号导出/导入的宏定义

#include <torch/csrc/python_headers.h>
// 引入 Torch 使用的 Python 头文件

#include <utility>
// 引入 C++ 标准库中的实用工具模块

template <class T>
class TORCH_PYTHON_API THPPointer {
 public:
  THPPointer() : ptr(nullptr){};
  // 默认构造函数，初始化指针为空

  explicit THPPointer(T* ptr) noexcept : ptr(ptr){};
  // 显式构造函数，传入指针初始化，不抛出异常

  THPPointer(THPPointer&& p) noexcept : ptr(std::exchange(p.ptr, nullptr)) {}
  // 移动构造函数，转移资源所有权给新对象，不抛出异常

  ~THPPointer() {
    free();
  };
  // 析构函数，释放资源

  T* get() {
    return ptr;
  }
  // 获取指针

  const T* get() const {
    return ptr;
  }
  // 获取指针的常量版本

  T* release() {
    T* tmp = ptr;
    ptr = nullptr;
    return tmp;
  }
  // 释放指针，返回指针并置空成员变量

  operator T*() {
    return ptr;
  }
  // 类型转换运算符，将对象转换为指针类型

  THPPointer& operator=(T* new_ptr) noexcept {
    free();
    ptr = new_ptr;
    return *this;
  }
  // 赋值运算符重载，释放旧资源并接管新资源

  THPPointer& operator=(THPPointer&& p) noexcept {
    free();
    ptr = p.ptr;
    p.ptr = nullptr;
    return *this;
  }
  // 移动赋值运算符重载，接管新资源并释放旧资源

  T* operator->() {
    return ptr;
  }
  // 成员访问运算符重载，返回指针

  explicit operator bool() const {
    return ptr != nullptr;
  }
  // 显式转换运算符，检查指针是否有效

 private:
  void free();
  // 私有方法声明，用于释放资源

  T* ptr = nullptr;
};
// THPPointer 类模板定义结束

/**
 * An RAII-style, owning pointer to a PyObject.  You must protect
 * destruction of this object with the GIL.
 *
 * WARNING: Think twice before putting this as a field in a C++
 * struct.  This class does NOT take out the GIL on destruction,
 * so if you will need to ensure that the destructor of your struct
 * is either (a) always invoked when the GIL is taken or (b) takes
 * out the GIL itself.  Easiest way to avoid this problem is to
 * not use THPPointer in this situation.
 */
using THPObjectPtr = THPPointer<PyObject>;
// 定义 PyObject 的 RAII-style 持有指针类型 THPObjectPtr

using THPCodeObjectPtr = THPPointer<PyCodeObject>;
// 定义 PyCodeObject 的 RAII-style 持有指针类型 THPCodeObjectPtr

using THPFrameObjectPtr = THPPointer<PyFrameObject>;
// 定义 PyFrameObject 的 RAII-style 持有指针类型 THPFrameObjectPtr
```