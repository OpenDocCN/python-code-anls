# `D:\src\scipysrc\scipy\scipy\_lib\_uarray\small_dynamic_array.h`

```
// 预处理命令，确保头文件只被包含一次
#pragma once

// 包含断言、大小相关类型、内存分配函数、智能指针和类型特性的标准库头文件
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

/** Fixed size dynamic array with small buffer optimisation */
template <typename T, ptrdiff_t SmallCapacity = 1>
class SmallDynamicArray {
  ptrdiff_t size_; // 数组大小
  union {
    T elements[SmallCapacity]; // 小型缓冲区
    T * array; // 动态分配的数组内存
  } storage_;

  // 判断是否使用小型缓冲区
  bool is_small() const { return size_ <= SmallCapacity; }

  // 销毁缓冲区内的对象
  void destroy_buffer(T * first, T * last) noexcept {
    for (; first < last; ++first) {
      first->~T();
    }
  }

  // 默认构造缓冲区内的对象
  void default_construct_buffer(T * first, T * last) noexcept(
      std::is_nothrow_destructible<T>::value) {
    auto cur = first;
    try {
      for (; cur < last; ++cur) {
        new (cur) T(); // 使用放置 new 构造对象
      }
    } catch (...) {
      // 如果构造失败，销毁已经构造的对象
      destroy_buffer(first, cur);
      throw;
    }
  }

  // 移动构造缓冲区内的对象
  void move_construct_buffer(T * first, T * last, T * d_first) noexcept(
      std::is_nothrow_move_constructible<T>::value) {
    T * d_cur = d_first;

    try {
      for (; first < last; ++first, ++d_cur) {
        new (d_cur) T(std::move(*first)); // 使用放置 new 移动构造对象
      }
    } catch (...) {
      // 如果构造失败，销毁已经构造的对象
      destroy_buffer(d_first, d_cur);
      throw;
    }
  }

  // 分配内存
  void allocate() {
    assert(size_ >= 0); // 断言确保数组大小非负
    if (is_small())
      return;

    storage_.array = (T *)malloc(size_ * sizeof(T)); // 动态分配内存
    if (!storage_.array) {
      throw std::bad_alloc(); // 分配失败抛出异常
    }
  }

  // 释放内存
  void deallocate() noexcept {
    if (!is_small()) {
      free(storage_.array); // 释放动态分配的内存
    }
  }

public:
  // 类型定义
  using value_type = T;
  using iterator = value_type *;
  using const_iterator = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = ptrdiff_t;

  // 默认构造函数
  SmallDynamicArray(): size_(0) {}

  // 指定大小构造函数
  explicit SmallDynamicArray(size_t size): size_(size) {
    allocate(); // 分配内存
    auto first = begin();
    try {
      default_construct_buffer(first, first + size_); // 默认构造对象
    } catch (...) {
      deallocate(); // 构造失败，释放内存
      throw;
    }
  }

  // 指定大小和初始值构造函数
  SmallDynamicArray(size_type size, const T & fill_value): size_(size) {
    allocate(); // 分配内存
    try {
      std::uninitialized_fill_n(begin(), size_, fill_value); // 使用未初始化的 fill_n 填充对象
    } catch (...) {
      deallocate(); // 构造失败，释放内存
      throw;
    }
  }

  // 拷贝构造函数
  SmallDynamicArray(const SmallDynamicArray & copy): size_(copy.size_) {
    allocate(); // 分配内存
    try {
      std::uninitialized_copy_n(copy.begin(), size_, begin()); // 使用未初始化的 copy_n 拷贝对象
    } catch (...) {
      deallocate(); // 构造失败，释放内存
      throw;
    }
  }

  // 移动构造函数
  SmallDynamicArray(SmallDynamicArray && move) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : size_(move.size_) {
    if (!move.is_small()) {
      storage_.array = move.storage_.array; // 移动数组指针
      move.storage_.array = nullptr; // 原数组指针置空
      move.storage_.size = 0; // 原数组大小置零
      return;
    }

    auto m_first = move.begin();
    try {
      move_construct_buffer(m_first, m_first + size_, begin()); // 移动构造对象
    } catch (...) {
      // 捕获所有异常，执行清理工作
      destroy_buffer(m_first, m_first + size_);
      move.size_ = 0;
      size_ = 0;
      throw;  // 继续抛出异常
    }

    // 销毁移动对象的缓冲区
    destroy_buffer(&move.storage_.elements[0], &move.storage_.elements[size_]);
    move.size_ = 0;
  }

  // 析构函数，调用 clear() 方法进行清理
  ~SmallDynamicArray() { clear(); }

  // 赋值运算符重载，实现对象的深拷贝赋值
  SmallDynamicArray & operator=(const SmallDynamicArray & copy) {
    if (&copy == this)
      return *this;

    clear();  // 清理当前对象

    size_ = copy.size;  // 设置新大小
    try {
      allocate();  // 分配新的存储空间
    } catch (...) {
      size_ = 0;
      throw;
    }

    try {
      // 使用未初始化的拷贝构造复制元素
      std::uninitialized_copy_n(copy.begin(), size_, begin());
    } catch (...) {
      deallocate();  // 释放已分配的空间
      size_ = 0;
      throw;
    }
    return *this;
  }

  // 移动赋值运算符重载，实现对象的移动赋值
  SmallDynamicArray & operator=(SmallDynamicArray && move) noexcept(
      std::is_nothrow_move_constructible<T>::value) {
    if (&move == this)
      return *this;

    // 如果移动对象不是小对象，则直接转移其指针和大小
    if (!move.is_small()) {
      storage_.array = move.storage_.array;
      size_ = move.size_;
      move.storage_.array = nullptr;
      move.size_ = 0;
      return *this;
    }

    clear();  // 清理当前对象

    size_ = move.size_;
    auto m_first = move.begin();
    try {
      // 移动构造缓冲区中的元素
      move_construct_buffer(m_first, m_first + size_, begin());
    } catch (...) {
      destroy_buffer(m_first, m_first + size_);
      move.size_ = 0;
      size_ = 0;
      throw;
    }

    // 销毁移动对象的缓冲区
    destroy_buffer(m_first, m_first + size_);
    move.size_ = 0;
    return *this;
  }

  // 清空数组元素，使用析构函数 noexcept 保证不抛出异常
  void clear() noexcept(std::is_nothrow_destructible<T>::value) {
    auto first = begin();
    destroy_buffer(first, first + size_);
    deallocate();  // 释放内存
    size_ = 0;
  }

  // 返回迭代器指向数组的起始位置
  iterator begin() {
    return is_small() ? &storage_.elements[0] : storage_.array;
  }

  // 返回常量迭代器指向数组的起始位置
  const_iterator cbegin() const {
    return is_small() ? &storage_.elements[0] : storage_.array;
  }

  // 返回迭代器指向数组的末尾位置
  iterator end() { return begin() + size_; }

  // 返回常量迭代器指向数组的末尾位置
  const_iterator cend() const { return cbegin() + size_; }

  // 返回常量迭代器指向数组的起始位置
  const_iterator begin() const { return cbegin(); }

  // 返回常量迭代器指向数组的末尾位置
  const_iterator end() const { return cend(); }

  // 返回数组的大小
  size_type size() const { return size_; }

  // 访问数组元素的 const 引用版本
  const_reference operator[](size_type idx) const {
    assert(0 <= idx && idx < size_);
    return begin()[idx];
  }

  // 访问数组元素的引用版本
  reference operator[](size_type idx) {
    assert(0 <= idx && idx < size_);
    return begin()[idx];
  }
};



# 这是一个单独的分号，通常用于结束语句。在这个上下文中，它不会执行任何特定操作，因为它没有前置代码行。
```