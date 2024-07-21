# `.\pytorch\torch\csrc\api\include\torch\data\iterator.h`

```py
#pragma once

#include <torch/csrc/utils/variadic.h>  // 引入 Torch 的变长参数工具头文件
#include <torch/types.h>  // 引入 Torch 的类型定义头文件

#include <c10/util/Exception.h>  // 引入 C10 库的异常处理头文件

#include <functional>  // 引入函数对象的标准库支持
#include <iterator>    // 引入迭代器的标准库支持
#include <memory>      // 引入内存管理相关的标准库支持
#include <type_traits> // 引入类型特性判断的标准库支持
#include <utility>     // 引入实用工具函数的标准库支持

namespace torch {
namespace data {
namespace detail {

// For increased safety and more separated logic, this implementation of
// `Iterator` consists of a `ValidIterator` and a `SentinelIterator`. A
// `ValidIterator` yields new batches until the `DataLoader` is exhausted. While
// the `DataLoader` is not exhausted, `ValidIterator`s compare equal if they are
// the same object. When the `ValidIterator` becomes exhausted, it compares
// equal to the `SentinelIterator`, but not before. Half the code here is to
// implement double dispatch for the comparison. Got damnit, C++.

template <typename Batch>
struct ValidIterator;

template <typename Batch>
struct SentinelIterator;

/// Base class for the `ValidIterator` and `SentinelIterator`
template <typename Batch>
struct IteratorImpl {
  virtual ~IteratorImpl() = default;  // 虚析构函数，用于基类的多态析构
  virtual void next() = 0;  // 纯虚函数，子类需实现获取下一个元素的操作
  virtual Batch& get() = 0;  // 纯虚函数，子类需实现获取当前元素的操作
  virtual bool operator==(const IteratorImpl& other) const = 0;  // 纯虚函数，实现与另一迭代器的比较
  virtual bool operator==(const ValidIterator<Batch>& other) const = 0;  // 纯虚函数，实现与有效迭代器的比较
  virtual bool operator==(const SentinelIterator<Batch>& other) const = 0;  // 纯虚函数，实现与终止迭代器的比较
};

template <typename Batch>
struct ValidIterator : public IteratorImpl<Batch> {
  using BatchProducer = std::function<optional<Batch>()>;  // 使用函数对象类型定义批次生成器

  explicit ValidIterator(BatchProducer next_batch)
      : next_batch_(std::move(next_batch)) {}  // 显式构造函数，接受批次生成器并移动其所有权

  /// Fetches the next batch.
  void next() override {
    // If we didn't get the very first batch yet, get it now.
    lazy_initialize();  // 若尚未获取第一个批次，则进行懒初始化
    TORCH_CHECK(
        batch_.has_value(), "Attempted to increment iterator past the end");  // 检查是否尝试在迭代器结束后递增
    // Increment to the next batch.
    batch_ = next_batch_();  // 获取下一个批次
  }

  /// Returns the current batch. The precondition for this operation to not
  /// throw an exception is that it has been compared to the `SentinelIterator`
  /// and did not compare equal.
  Batch& get() override {
    // If we didn't get the very first batch yet, get it now.
    lazy_initialize();  // 若尚未获取第一个批次，则进行懒初始化
    TORCH_CHECK(
        batch_.has_value(),
        "Attempted to dereference iterator that was past the end");  // 检查是否尝试解引用超出迭代器末尾的位置
    return batch_.value();  // 返回当前批次的引用
  }

  /// Does double dispatch.
  bool operator==(const IteratorImpl<Batch>& other) const override {
    return other == *this;  // 执行双重分派比较操作
  }

  /// A `ValidIterator` is equal to the `SentinelIterator` iff. the
  /// `ValidIterator` has reached the end of the dataloader.
  bool operator==(const SentinelIterator<Batch>& /* unused */) const override {
    lazy_initialize();  // 若尚未获取第一个批次，则进行懒初始化
    return !batch_;  // 当批次为空时，有效迭代器等于终止迭代器
  }

  /// Returns true if the memory address of `other` equals that of `this`.
  bool operator==(const ValidIterator<Batch>& other) const override {
    return &other == this;  // 比较当前迭代器的内存地址与另一有效迭代器的内存地址
  }

  /// Gets the very first batch if it has not yet been fetched.
  void lazy_initialize() const {
    if (!batch_) {
      batch_ = next_batch_();  // 若尚未获取第一个批次，则从生成器中获取
    }
  }

  private:
    mutable optional<Batch> batch_;  // 可变的可选批次对象，用于延迟初始化
    BatchProducer next_batch_;  // 批次生成器，用于生成下一个批次
};
    # 如果初始化标志 initialized_ 为假（即未初始化状态）
    if (!initialized_) {
      # 调用 next_batch_() 方法生成下一个批次的数据，并存储在 batch_ 中
      batch_ = next_batch_();
      # 将 initialized_ 标志设为真，表示已经完成初始化
      initialized_ = true;
    }
  }

  # 声明一个函数对象 next_batch_，用于生成下一个数据批次
  BatchProducer next_batch_;
  # 可变的可选类型对象 batch_，用于存储当前批次的数据
  mutable optional<Batch> batch_;
  # 可变的布尔型变量 initialized_，表示对象是否已经完成初始化，默认为假
  mutable bool initialized_ = false;
};

/// `SentinelIterator` 结构体继承自 `IteratorImpl<Batch>`，用于实现迭代器的哨兵模式。
template <typename Batch>
struct SentinelIterator : public IteratorImpl<Batch> {
  /// 禁止对过结尾的迭代器进行递增操作，抛出错误信息。
  void next() override {
    AT_ERROR(
        "Incrementing the DataLoader's past-the-end iterator is not allowed");
  }

  /// 禁止对过结尾的迭代器进行解引用操作，抛出错误信息。
  Batch& get() override {
    AT_ERROR(
        "Dereferencing the DataLoader's past-the-end iterator is not allowed");
  }

  /// 双重分发，比较当前迭代器与另一个迭代器是否相等。
  bool operator==(const IteratorImpl<Batch>& other) const override {
    return other == *this;
  }

  /// 调用 `ValidIterator` 和 `SentinelIterator` 之间的比较运算符。
  bool operator==(const ValidIterator<Batch>& other) const override {
    return other == *this;
  }

  /// 哨兵迭代器始终相等。
  bool operator==(const SentinelIterator<Batch>& other) const override {
    return true;
  }
};
} // namespace detail

/// `Iterator` 类模板，用于定义迭代器的行为和特性。
template <typename Batch>
class Iterator {
 public:
  // 类型别名，使得该类被识别为合法的迭代器。
  using difference_type = std::ptrdiff_t;
  using value_type = Batch;
  using pointer = Batch*;
  using reference = Batch&;
  using iterator_category = std::input_iterator_tag;

  /// 构造函数，接受一个指向 `IteratorImpl<Batch>` 的智能指针。
  explicit Iterator(std::unique_ptr<detail::IteratorImpl<Batch>> impl)
      : impl_(std::move(impl)) {}

  /// 递增迭代器，仅适用于有效的迭代器（非结尾迭代器）。
  Iterator& operator++() {
    impl_->next();
    return *this;
  }

  /// 返回当前批次。
  /// 仅适用于有效的迭代器（非结尾迭代器）。
  Batch& operator*() {
    return impl_->get();
  }

  /// 返回指向当前批次的指针。
  /// 仅适用于有效的迭代器（非结尾迭代器）。
  Batch* operator->() {
    return &impl_->get();
  }

  /// 比较两个迭代器是否相等。
  bool operator==(const Iterator& other) const {
    return *impl_ == *other.impl_;
  }

  /// 比较两个迭代器是否不相等。
  bool operator!=(const Iterator& other) const {
    return !(*this == other);
  }

 private:
  /// 智能指针，指向 `IteratorImpl<Batch>` 的实例，可以是 `ValidIterator` 或 `SentinelIterator`。
  std::shared_ptr<detail::IteratorImpl<Batch>> impl_;
};
} // namespace data
} // namespace torch
```