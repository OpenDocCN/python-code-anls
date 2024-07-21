# `.\pytorch\c10\util\SmallBuffer.h`

```py
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Helper class for allocating temporary fixed size arrays with SBO.
 *
 * This is intentionally much simpler than SmallVector, to improve performance
 * at the expense of many features:
 * - No zero-initialization for numeric types
 * - No resizing after construction
 * - No copy/move
 * - No non-trivial types
 */

namespace c10 {

template <typename T, size_t N>
class SmallBuffer {
  static_assert(std::is_trivial_v<T>, "SmallBuffer is intended for POD types");

  std::array<T, N> storage_; // 固定大小的存储数组

  size_t size_{}; // 存储的大小
  T* data_{}; // 指向数据的指针

 public:
  // 构造函数，根据需求选择合适的存储方式
  SmallBuffer(size_t size) : size_(size) {
    if (size > N) { // 如果需要的大小超过预留的固定大小
      data_ = new T[size]; // 分配堆上的内存
    } else { // 否则使用栈上的固定大小数组
      data_ = &storage_[0];
    }
  }

  // 禁用拷贝构造函数和拷贝赋值操作符
  SmallBuffer(const SmallBuffer&) = delete;
  SmallBuffer& operator=(const SmallBuffer&) = delete;

  // 移动构造函数，用于支持函数返回时的移动语义
  SmallBuffer(SmallBuffer&& rhs) noexcept : size_{rhs.size_} {
    rhs.size_ = 0;
    if (size_ > N) {
      data_ = rhs.data_; // 如果超出固定大小，直接转移指针所有权
      rhs.data_ = nullptr;
    } else {
      storage_ = std::move(rhs.storage_); // 否则移动固定大小数组的内容
      data_ = &storage_[0];
    }
  }

  // 禁用移动赋值操作符
  SmallBuffer& operator=(SmallBuffer&&) = delete;

  // 析构函数，根据是否使用了堆上内存进行清理
  ~SmallBuffer() {
    if (size_ > N) {
      delete[] data_; // 如果使用了堆上内存，释放之
    }
  }

  // 下标操作符重载，返回指定索引处的元素引用
  T& operator[](size_t idx) {
    return data()[idx];
  }

  // const 重载的下标操作符，返回指定索引处的常量引用
  const T& operator[](size_t idx) const {
    return data()[idx];
  }

  // 返回数据的指针
  T* data() {
    return data_;
  }

  // 返回常量数据的指针
  const T* data() const {
    return data_;
  }

  // 返回存储的大小
  size_t size() const {
    return size_;
  }

  // 返回数据起始位置的指针
  T* begin() {
    return data_;
  }

  // 返回常量数据起始位置的指针
  const T* begin() const {
    return data_;
  }

  // 返回数据结束位置的指针
  T* end() {
    return data_ + size_;
  }

  // 返回常量数据结束位置的指针
  const T* end() const {
    return data_ + size_;
  }
};

} // namespace c10
```