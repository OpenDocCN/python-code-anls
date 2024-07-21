# `.\pytorch\test\cpp\tensorexpr\padded_buffer.h`

```
// 防止头文件被多次包含
#pragma once

// 引入必要的标准库头文件
#include <string>
#include <vector>

// 引入C10库中的irange.h头文件
#include <c10/util/irange.h>
// 引入TensorExpr库中的eval.h头文件
#include "torch/csrc/jit/tensorexpr/eval.h"

// 定义命名空间torch::jit::tensorexpr
namespace torch {
namespace jit {
namespace tensorexpr {

// 模板类，用于提供各种类型的默认填充值
template <typename T>
struct DefaultPaddedValue;

// 特化模板类DefaultPaddedValue，为int类型提供默认填充值0xDEADBEEF
template <>
struct DefaultPaddedValue<int> {
  static const int kValue = static_cast<int>(0xDEADBEEF);
};

// 特化模板类DefaultPaddedValue，为int8_t类型提供默认填充值0xBE
template <>
struct DefaultPaddedValue<int8_t> {
  static const int8_t kValue = static_cast<int8_t>(0xBE);
};

// 特化模板类DefaultPaddedValue，为uint8_t类型提供默认填充值0xBE
template <>
struct DefaultPaddedValue<uint8_t> {
  static const uint8_t kValue = static_cast<uint8_t>(0xBE);
};

// 特化模板类DefaultPaddedValue，为int16_t类型提供默认填充值0xBEEF
template <>
struct DefaultPaddedValue<int16_t> {
  static const int16_t kValue = static_cast<int16_t>(0xBEEF);
};

// 特化模板类DefaultPaddedValue，为int64_t类型提供默认填充值0xDEADBEEF
template <>
struct DefaultPaddedValue<int64_t> {
  static const int64_t kValue = static_cast<int64_t>(0xDEADBEEF);
};

// 特化模板类DefaultPaddedValue，为float类型提供默认填充值0.1357
template <>
struct DefaultPaddedValue<float> {
  static constexpr float kValue = 0.1357;
};

// 特化模板类DefaultPaddedValue，为at::Half类型提供默认填充值0x1357
template <>
struct DefaultPaddedValue<at::Half> {
  // at::Half的构造函数不是constexpr，因此使用其位表示作为默认填充值
  static constexpr uint16_t kValue = 1357;
};

// 特化模板类DefaultPaddedValue，为double类型提供默认填充值0.1357
template <>
struct DefaultPaddedValue<double> {
  static constexpr double kValue = 0.1357;
};

// PaddedBufferBase类的具体实现，用于表示带填充的缓冲区的基类
class PaddedBufferBase {
 public:
  // 返回缓冲区名称的引用
  const std::string& name() const {
    return name_;
  }

  // 返回缓冲区的大小（不包括填充部分）
  int size() const {
    return total_size_;
  }

  // 返回缓冲区的原始大小（包括填充部分）
  int raw_size() const {
    return total_size_ + 2 * kPaddingSize;
  }

  // 虚析构函数，用于派生类的销毁
  virtual ~PaddedBufferBase() {}

 protected:
  // 显式构造函数，初始化维度、名称等基本属性
  explicit PaddedBufferBase(
      const std::vector<int>& dims,
      const std::string& name);
  
  // 根据给定索引计算元素在缓冲区中的偏移量
  int Index(const std::vector<int>& indices) const;

  // 缓冲区的维度
  std::vector<int> dims_;
  // 缓冲区的名称
  std::string name_;
  // 缓冲区的步长（strides）
  std::vector<int> strides_;
  // 缓冲区的总大小（不包括填充部分）
  int total_size_;
  // 缓冲区的填充大小常量
  static constexpr int kPaddingSize = 64;
};

// PaddedBuffer类，继承自PaddedBufferBase，表示带有填充标记的缓冲区
// 用于测试目的，缓冲区两侧包含填充标记，以便捕获可能的越界写入
// 对于不应更改的只读数据，还可以创建备份并稍后进行比较
template <typename T>
class PaddedBuffer : public PaddedBufferBase {
 public:
  // 构造函数，根据维度和名称初始化PaddedBuffer
  PaddedBuffer(int d0, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0}), name) {}
  PaddedBuffer(int d0, int d1, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0, d1}), name) {}
  PaddedBuffer(int d0, int d1, int d2, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0, d1, d2}), name) {}
  PaddedBuffer(int d0, int d1, int d2, int d3, const std::string& name = "")
      : PaddedBuffer(std::vector<int>({d0, d1, d2, d3}), name) {}
  // 根据给定维度和名称构造PaddedBuffer
  PaddedBuffer(const std::vector<int>& dims, const std::string& name = "")
      : PaddedBufferBase(dims, name) {
    // 初始化数据，包括填充部分，使用默认填充值
    data_.resize(total_size_ + 2 * kPaddingSize, kPaddingValue);
  }
  // 拷贝构造函数，复制另一个PaddedBuffer，并设置新名称
  PaddedBuffer(const PaddedBuffer& other, const std::string& name)
      : PaddedBuffer(other) {
    this->name_ = name;
  }

  // 返回缓冲区的数据指针
  T* data() {
  // 返回当前数据的指针加上填充大小
  return data_.data() + kPaddingSize;
}

// 返回常量数据的指针
const T* data() const {
  return const_cast<PaddedBuffer*>(this)->data();
}

// 返回数据的原始指针
T* raw_data() {
  return data_.data();
}

// 返回常量数据的原始指针
const T* raw_data() const {
  return const_cast<PaddedBuffer*>(this)->raw_data();
}

// 重载函数，返回指定索引 i0 处的数据，支持单个索引
T& operator()(int i0) {
  // 这里形成一个向量会稍微影响性能。但这个数据结构仅用于测试，并非性能关键。
  return this->operator()(std::vector<int>({i0}));
}

// 重载函数，返回指定索引 i0 处的常量数据，支持单个索引
const T& operator()(int i0) const {
  return const_cast<PaddedBuffer*>(this)->operator()(i0);
}

// 重载函数，返回指定索引 i0, i1 处的数据
T& operator()(int i0, int i1) {
  return this->operator()(std::vector<int>({i0, i1}));
}

// 重载函数，返回指定索引 i0, i1 处的常量数据
const T& operator()(int i0, int i1) const {
  return const_cast<PaddedBuffer*>(this)->operator()(i0, i1);
}

// 重载函数，返回指定索引 i0, i1, i2 处的数据
T& operator()(int i0, int i1, int i2) {
  return this->operator()(std::vector<int>({i0, i1, i2}));
}

// 重载函数，返回指定索引 i0, i1, i2 处的常量数据
const T& operator()(int i0, int i1, int i2) const {
  return const_cast<PaddedBuffer*>(this)->operator()(i0, i1, i2);
}

// 重载函数，返回指定索引 i0, i1, i2, i3 处的数据
T& operator()(int i0, int i1, int i2, int i3) {
  return this->operator()(std::vector<int>({i0, i1, i2, i3}));
}

// 重载函数，返回指定索引 i0, i1, i2, i3 处的常量数据
const T& operator()(int i0, int i1, int i2, int i3) const {
  return const_cast<PaddedBuffer*>(this)->operator()(i0, i1, i2, i3);
}

// 重载函数，返回指定索引数组 indices 处的数据
T& operator()(const std::vector<int>& indices) {
  return data_[kPaddingSize + Index(indices)];
}

// 重载函数，返回指定索引数组 indices 处的常量数据
const T& operator()(const std::vector<int>& indices) const {
  return const_cast<PaddedBuffer*>(this)->operator()(indices);
}

// 友元函数声明，用于比较两个 PaddedBuffer 对象是否在一定误差范围内近似相等
template <typename U>
friend void ExpectAllNear(
    const PaddedBuffer<U>& v1,
    const PaddedBuffer<U>& v2,
    float abs_error);

// 友元函数声明，用于比较两个 PaddedBuffer 对象是否完全相等
template <typename U>
friend void ExpectAllEqual(
    const PaddedBuffer<U>& v1,
    const PaddedBuffer<U>& v2);

// 备份当前数据
void Backup() {
  backup_data_ = data_;
}

// 验证填充区域中的水印是否完好
void ValidateWatermark() const {
  for (const auto i : c10::irange(kPaddingSize)) {
    ASSERT_EQ(data_[i], kPaddingValue);
    ASSERT_EQ(data_[i + total_size_ + kPaddingSize], kPaddingValue);
  }
}

// 检查数据备份是否正确
void CheckBackup() const {
  ValidateWatermark();
  DCHECK(backup_data_.size() == data_.size())
      << "请确保在调用 CheckBackup() 前已调用 Backup()";
  for (const auto i : c10::irange(total_size_)) {
    ASSERT_EQ(data_[i + kPaddingSize], backup_data_[i + kPaddingSize]);
  }
}

private:
std::vector<T> data_;          // 主要数据存储
std::vector<T> backup_data_;   // 备份数据存储
T kPaddingValue = DefaultPaddedValue<T>::kValue;  // 填充值
};

// 定义模板函数 CallArg 的构造函数，接受 PaddedBuffer 类型的常引用 buffer
template <typename T>
inline CodeGen::CallArg::CallArg(const PaddedBuffer<T>& buffer)
    : data_(const_cast<T*>(buffer.data())) {}

// 比较两个 PaddedBuffer 对象 v1 和 v2 的指定索引处元素，生成错误信息字符串
template <typename T>
std::string CompareErrorMsg(
    const PaddedBuffer<T>& v1,
    const PaddedBuffer<T>& v2,
    int index) {
  std::ostringstream oss;
  oss << "index: " << index << ", v1: (" << v1.name() << ", " << v1(index)
      << ")"
      << ", v2: (" << v2.name() << ", " << v2(index) << ")";
  return oss.str();
}

// 检查两个 PaddedBuffer 对象 f1 和 f2 的所有元素是否相等
template <typename T>
void ExpectAllEqual(const PaddedBuffer<T>& f1, const PaddedBuffer<T>& f2) {
  const std::vector<T>& v1 = f1.data_;
  const std::vector<T>& v2 = f2.data_;
  const int kPaddingSize = f1.kPaddingSize;
  const int total_size = f1.total_size_;
  
  // 断言两个 vector 的大小相等
  ASSERT_EQ(v1.size(), v2.size());
  
  // 验证 f1 和 f2 的水印
  f1.ValidateWatermark();
  f2.ValidateWatermark();
  
  // 逐一比较有效数据区域的元素
  for (const auto i : c10::irange(total_size)) {
    ASSERT_EQ(v1[kPaddingSize + i], v2[kPaddingSize + i]);
  }
}

// 检查两个 PaddedBuffer 对象 f1 和 f2 的所有元素是否在给定的绝对误差范围内接近
template <typename T>
void ExpectAllNear(
    const PaddedBuffer<T>& f1,
    const PaddedBuffer<T>& f2,
    float abs_error) {
  const std::vector<T>& v1 = f1.data_;
  const std::vector<T>& v2 = f2.data_;
  const int kPaddingSize = f1.kPaddingSize;
  const int total_size = f1.total_size_;
  
  // 断言两个 vector 的大小相等
  ASSERT_EQ(v1.size(), v2.size());
  
  // 验证 f1 和 f2 的水印
  f1.ValidateWatermark();
  f2.ValidateWatermark();
  
  // 逐一比较有效数据区域的元素是否在给定的绝对误差范围内接近
  for (const auto i : c10::irange(total_size)) {
    ASSERT_NEAR(v1[kPaddingSize + i], v2[kPaddingSize + i], abs_error);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch


这些注释为给定的 C++ 代码块中的每行代码添加了解释和说明，使读者能够理解每个函数和操作的用途和目的。
```