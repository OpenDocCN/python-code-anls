# `.\pytorch\torch\csrc\distributed\c10d\sequence_num.cpp`

```
// 包含必要的头文件
#include <ATen/ThreadLocalState.h>
#include <torch/csrc/distributed/c10d/sequence_num.hpp>

// 包含日志记录工具的头文件
#include <c10/util/Logging.h>

// 定义命名空间 c10d
namespace c10d {

// 默认构造函数 SequenceNum，使用默认值初始化 num_
SequenceNum::SequenceNum() = default;

// 带参数的构造函数 SequenceNum，用给定的 num 初始化 num_
SequenceNum::SequenceNum(const uint64_t num) : num_(num) {}

// 拷贝构造函数 SequenceNum，从另一个 SequenceNum 对象拷贝数据
SequenceNum::SequenceNum(const SequenceNum& other) {
  // 如果另一个对象未设置数值，则将当前对象的 num_ 置为 c10::nullopt
  if (!other.isSet()) {
    num_ = c10::nullopt;
  } else {
    // 否则，将当前对象的 num_ 设置为另一个对象的数值
    num_ = other.get();
  }
}

// 获取当前 num_ 的值的方法
uint64_t SequenceNum::get() const {
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(lock_);
  // 返回 num_ 的值
  return *num_;
}

// 自增 num_ 的值的方法
void SequenceNum::increment() {
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(lock_);
  // 检查 num_ 不为空
  TORCH_CHECK(num_ != c10::nullopt);
  // 自增 num_ 的值
  num_ = ++(*num_);
}

// 获取当前 num_ 的值并自增的方法，避免重复加锁和解锁
uint64_t SequenceNum::getAndIncrement() {
  uint64_t curVal = 0;
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(lock_);
  // 检查 num_ 不为空
  TORCH_CHECK(num_ != c10::nullopt);
  // 获取当前 num_ 的值
  curVal = *num_;
  // 自增 num_ 的值
  num_ = ++(*num_);
  // 返回之前的 num_ 值
  return curVal;
}

// 设置 num_ 的值的方法
void SequenceNum::set(const uint64_t num) {
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(lock_);
  // 设置 num_ 的值为给定的 num
  num_ = num;
}

// 检查 num_ 是否已设置的方法
bool SequenceNum::isSet() const {
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(lock_);
  // 返回 num_ 是否不为空的状态
  return num_ != c10::nullopt;
}

// 赋值操作符重载，从另一个 SequenceNum 对象赋值
SequenceNum& SequenceNum::operator=(const SequenceNum& other) {
  // 使用互斥锁保护共享资源
  std::lock_guard<std::mutex> lock(lock_);
  // 如果另一个对象未设置数值，则将当前对象的 num_ 置为 c10::nullopt
  if (!other.isSet()) {
    num_ = c10::nullopt;
  } else {
    // 否则，将当前对象的 num_ 设置为另一个对象的数值
    num_ = other.get();
  }
  // 返回当前对象的引用
  return *this;
}

} // namespace c10d
```