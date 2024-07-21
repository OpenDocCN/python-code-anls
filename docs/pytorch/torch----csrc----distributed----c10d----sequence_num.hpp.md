# `.\pytorch\torch\csrc\distributed\c10d\sequence_num.hpp`

```py
#pragma once

#include <c10/macros/Macros.h>   // 引入 c10 库的宏定义
#include <c10/util/Optional.h>   // 引入 c10 库的 Optional 实用工具
#include <c10/util/irange.h>     // 引入 c10 库的整数范围迭代器
#include <mutex>                 // 引入互斥锁
#include <vector>                // 引入向量容器

namespace c10d {

const int kUnsetSeqNum = 0;      // 定义未设置序列号的常量

namespace {
constexpr int kByteOffset = 8;   // 定义字节偏移量常量为 8
}

// 将 uint64_t 类型的数值转换为 T 类型的向量，以便写入存储
template <typename T>
inline std::vector<T> toVec(uint64_t num, int numBytes) {
  std::vector<T> values;         // 创建一个 T 类型的向量 values
  // 从右向左逐个读取字节，并推入 char 数组中
  for (const auto i : c10::irange(numBytes)) {
    uint8_t x = (num >> (kByteOffset * i)) & 0xff;  // 从 num 中读取第 i 个字节
    values.push_back(static_cast<T>(x));            // 将 x 转换为 T 类型并推入向量
  }
  return values;                  // 返回转换后的向量
}

// 将 char 向量 values（如从存储中读取的数据）转换为 uint64_t 类型
template <typename T>
inline uint64_t fromVec(const std::vector<T>& values) {
  uint64_t num = 0;               // 初始化 num 为 0
  // 在正确的位置设置每个字节到 num 中
  for (const auto i : c10::irange(values.size())) {
    uint8_t x = static_cast<uint8_t>(values[i]);    // 将 values 中的元素转换为 uint8_t
    num |= (static_cast<int64_t>(x) << (kByteOffset * i));  // 将 x 的值设置到 num 的对应位置
  }
  return num;                     // 返回转换后的 uint64_t 数值
}

class TORCH_API SequenceNum {
 public:
  SequenceNum();                  // 默认构造函数
  explicit SequenceNum(const uint64_t num);   // 显式构造函数，使用给定的 uint64_t 数值
  // 获取 num_ 值。如果未设置则抛出异常。
  uint64_t get() const;
  // 将 num_ 值增加。如果未设置则抛出异常。
  void increment();
  // 将 num_ 值增加并返回旧值。如果未设置则抛出异常。
  uint64_t getAndIncrement();
  // 设置 num_ 值
  void set(const uint64_t num);
  // 如果此 SequenceNum 已正确初始化为一个值，则返回 true，否则返回 false。
  bool isSet() const;

  SequenceNum& operator=(const SequenceNum& other);   // 赋值运算符重载
  SequenceNum(const SequenceNum& other);              // 拷贝构造函数

 private:
  std::optional<uint64_t> num_;   // 可选的 uint64_t 类型成员变量 num_
  mutable std::mutex lock_;       // 可变的互斥锁，用于保护数据访问
};

} // namespace c10d
```