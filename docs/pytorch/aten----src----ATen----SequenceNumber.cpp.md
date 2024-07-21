# `.\pytorch\aten\src\ATen\SequenceNumber.cpp`

```py
// 包含 ATen 库中的 SequenceNumber.h 头文件
#include <ATen/SequenceNumber.h>

// 定义 at 命名空间中的 sequence_number 命名空间
namespace at::sequence_number {

// 匿名命名空间内部的线程局部变量，用于存储序列号，初始值为 0
namespace {
thread_local uint64_t sequence_nr_ = 0;
} // namespace

// 返回当前存储的序列号
uint64_t peek() {
  return sequence_nr_;
}

// 返回当前存储的序列号并递增
uint64_t get_and_increment() {
  return sequence_nr_++;
}

} // namespace at::sequence_number
```