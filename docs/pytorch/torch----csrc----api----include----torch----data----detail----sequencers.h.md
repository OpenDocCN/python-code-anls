# `.\pytorch\torch\csrc\api\include\torch\data\detail\sequencers.h`

```py
#pragma once

#include <torch/types.h>

#include <algorithm>  // 包含 STL 算法库，例如 std::any_of
#include <cstddef>    // 包含一些与大小和指针算术相关的常量
#include <vector>     // 包含 STL 向量容器类

namespace torch {
namespace data {
namespace detail {
namespace sequencers {
namespace detail {

// 检查缓冲区中是否包含有效结果的函数模板
template <typename Result>
bool buffer_contains_result(const std::vector<optional<Result>>& buffer) {
  // 使用 STL 算法 any_of 来遍历缓冲区，检查是否有有效的结果
  return std::any_of(
      buffer.begin(), buffer.end(), [](const optional<Result>& result) {
        return result.has_value();  // 检查 optional 对象是否有值
      });
}
} // namespace detail

/// A `Sequencer` accepts a function that yields the next result of a
/// `DataLoader` and then has the opportunity to influence the order in which
/// these results are returned. The `NoSequencer` does not enforce any
/// sequencing and returns any result directly. The `OrderedSequencer` instead
/// buffers results internally to return them in order of their sequence number.
template <typename Result>
struct Sequencer {
  using ResultProducer = std::function<optional<Result>()>;  // 结果生成器类型定义
  virtual ~Sequencer() = default;  // 虚析构函数

  // 纯虚函数，要求派生类实现的接口，获取下一个结果
  virtual optional<Result> next(ResultProducer next_result) = 0;
};

/// A `Sequencer` that does not enforce any ordering. It is effectively the
/// identity function.
template <typename Result>
struct NoSequencer final : public Sequencer<Result> {
  using typename Sequencer<Result>::ResultProducer;  // 使用基类中的类型定义
  optional<Result> next(ResultProducer next_result) override {
    return next_result();  // 直接返回下一个生成的结果
  }
};

/// A `Sequencer` that buffers results and returns them in order of their
/// sequence number. The `OrderedSequencer` maintains an internal, monotonically
/// incrementing counter for the next sequence number it expects. If it receives
/// a result with a higher sequence number, it will buffer it for later (when
/// the sequence number reaches that of this result). Otherwise, if the sequence
/// numbers match, the result is returned.
///
/// Implementation note: The `OrderedSequencer` is implemented with a fixed-size
/// buffer. Let `m` be the maximum number of jobs in the data loader's queue and
/// `s` be the current sequence number. Assume `m` jobs are scheduled in the
/// `DataLoader`. Any new result is stored at index `job.sqn mod m` in the
/// `OrderedSequencer`. Why are we sure sequence numbers of new jobs will not
/// collide with sequence numbers of buffered jobs? The `OrderedSequencer` will
/// not return from `next()` until it receives the result with sqn `s`. This
/// means no new jobs can be scheduled in the `DataLoader` in the meantime,
/// which enforces that as long as sqn `s` has not been received, `s + m` (which
/// would cause a collision in the fixed-size buffer) will not yet be scheduled.
template <typename Result>
struct OrderedSequencer final : public Sequencer<Result> {
  // 使用 typename 来引入基类的类型定义
  using typename Sequencer<Result>::ResultProducer;

  optional<Result> next(ResultProducer next_result) override {
    // 实现 OrderedSequencer 的 next 函数，处理结果的顺序

    // 实现细节参见上文注释
    // 省略具体实现细节，这里不需要提供具体代码内容，只需说明其作用和结构
  }
};

} // namespace detail
} // namespace sequencers
} // namespace detail
} // namespace data
} // namespace torch
struct OrderedSequencer : public Sequencer<Result> {
  using typename Sequencer<Result>::ResultProducer;

  /// Constructs the `OrderedSequencer` with the maximum number of results it
  /// will ever hold at one point in time.
  explicit OrderedSequencer(size_t max_jobs) : buffer_(max_jobs) {}

  /// Buffers results until the next one in the expected order is received.
  optional<Result> next(ResultProducer next_result) override {
    // 如果我们已经有下一个序列号的结果，直接返回它。
    if (auto& maybe_result = buffer(next_sequence_number_)) {
      auto result = std::move(*maybe_result);
      buffer(next_sequence_number_++).reset();
      return result;
    }
    // 否则等待下一个结果。
    while (true) {
      auto result = next_result();
      if (!result) {
        // 断言：缓冲区不包含任何结果。
        AT_ASSERT(!detail::buffer_contains_result(buffer_));
        break;
      }
      // 如果结果不是空且序列号匹配，则直接返回并增加序列号。
      if (result->sequence_number == next_sequence_number_) {
        ++next_sequence_number_;
        return result;
      }
      // 将结果暂存以备后用。
      // 断言：缓冲区中不存在特定序列号的结果。
      AT_ASSERT(!buffer(result->sequence_number).has_value());
      buffer(result->sequence_number) = std::move(result);
    }
    // 结果是空的可选值，表示这一轮的处理已完成。
    return nullopt;
  }

  /// Accesses the buffer at the `index` modulo the buffer size.
  optional<Result>& buffer(size_t index) {
    return buffer_.at(index % buffer_.size());
  }

  /// The monotonically increasing sequence number we expect.
  size_t next_sequence_number_ = 0;

  /// A fixed-size buffer (after construction).
  std::vector<optional<Result>> buffer_;
};
```