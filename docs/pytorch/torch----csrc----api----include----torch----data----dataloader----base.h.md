# `.\pytorch\torch\csrc\api\include\torch\data\dataloader\base.h`

```
#pragma once

#include <torch/data/dataloader_options.h>
#include <torch/data/detail/data_shuttle.h>
#include <torch/data/detail/sequencers.h>
#include <torch/data/iterator.h>
#include <torch/data/samplers/random.h>
#include <torch/data/worker_exception.h>
#include <torch/types.h>

#include <torch/csrc/utils/variadic.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace torch {
namespace data {

// DataLoaderBase 类定义
template <typename Dataset, typename Batch, typename BatchRequest>
class DataLoaderBase {
 public:
  // 类型别名定义
  using BatchType = Batch;
  using BatchRequestType = BatchRequest;

  /// 构造函数，从给定的 `dataset` 进行采样，使用 `options` 配置 DataLoader，`sampler` 指定采样策略。
  DataLoaderBase(
      DataLoaderOptions options,
      std::unique_ptr<Dataset> main_thread_dataset = nullptr)
      : options_(std::move(options)),
        main_thread_dataset_(std::move(main_thread_dataset)),
        sequencer_(new_sequencer()) {}

  // NOLINTNEXTLINE(bugprone-exception-escape)
  /// 析构函数，清理资源，包括终止工作线程。
  virtual ~DataLoaderBase() {
    join();
  }

  /// 返回一个迭代器，用于遍历 DataLoader。迭代器的生命周期与 DataLoader 绑定。
  /// 在 C++ 标准中，迭代器的类别为 `OutputIterator`。详情见：
  /// https://en.cppreference.com/w/cpp/named_req/OutputIterator
  /// 主要特点是可以递增和解引用，但不能回退或一次移动多个位置。
  /// 当 DataLoader 耗尽时，它将与特殊的“sentinel”迭代器相等，该迭代器由 `DataLoader::end()` 返回。
  /// 大部分情况下，推荐使用 range-for 循环遍历 DataLoader，也支持标准算法如 `std::copy(dataloader.begin(), dataloader.end(), output_iterator)`。
  Iterator<Batch> begin() {
    TORCH_CHECK(
        shuttle_.in_flight_jobs() == 0,
        "Attempted to get a new DataLoader iterator "
        "while another iterator is not yet exhausted");
    reset();
    return Iterator<Batch>(std::make_unique<detail::ValidIterator<Batch>>(
        [this] { return this->next(); }));
  }

  /// 返回一个特殊的“sentinel”迭代器，当 DataLoader 耗尽时，它将与非sentinel迭代器相等。
  Iterator<Batch> end() {
    return Iterator<Batch>(std::make_unique<detail::SentinelIterator<Batch>>());
  }

  /// 终止 DataLoader 的工作线程并清空内部队列。
  /// 只能在主线程中调用此函数（DataLoader 所在的线程）。
  void join() {
    if (joined_) {
      return;
    }
    shuttle_.drain();
    // 为每个工作线程发送一个 'quit' 消息。由于工作线程会终止（退出其
    //
  // 向每个工作线程推送一个 QuitWorker() 作业，确保每个工作线程都收到关闭消息
  for (const auto w : c10::irange(options_.workers)) {
    (void)w; // 抑制未使用变量警告
    push_job(QuitWorker());
  }

  // 等待所有工作线程完成
  for (auto& worker : workers_) {
    worker.join();
  }

  // 设置标志表示所有工作线程已经加入完成
  joined_ = true;
}

/// 返回 DataLoader 配置的选项
const FullDataLoaderOptions& options() const noexcept {
  return options_;
}

protected:
/// 简单的混入类，为对象提供一个序列号
struct Sequenced {
  Sequenced() = default;
  Sequenced(size_t sqn) : sequence_number(sqn) {}
  size_t sequence_number;
};

struct QuitWorker {};

/// Job 可以是一个 BatchRequest（获取数据的新索引）或 QuitWorker 对象（指示工作线程应关闭）
struct Job : Sequenced {
  Job() = default;
  Job(QuitWorker q, size_t sqn) : Sequenced(sqn), quit(q) {}
  Job(BatchRequest&& i, size_t sqn)
      : Sequenced(sqn), batch_request(std::move(i)) {}
  optional<QuitWorker> quit;
  optional<BatchRequest> batch_request;
};

/// Job 执行完成后的结果
struct Result : Sequenced {
  Result() = default;
  Result(optional<Batch>&& b, size_t sqn)
      : Sequenced(sqn), batch(std::move(b)) {}
  Result(std::exception_ptr exception, size_t sqn)
      : Sequenced(sqn), exception(std::move(exception)) {}
  optional<Batch> batch;
  std::exception_ptr exception;
};

/// 子类钩子，用于获取下一个批次请求
/// 无状态情况下会向采样器请求新的批次请求（例如索引向量）
/// 有状态情况下会直接返回批次大小
virtual optional<BatchRequestType> get_batch_request() = 0;

/// 重置 DataLoader 的内部状态，可选择预取新作业
virtual void reset() {
  shuttle_.drain(); // 清空队列
  sequence_number_ = 0; // 重置序列号
  sequencer_ = new_sequencer(); // 创建新的顺序器
  prefetch(); // 预取作业
}

/// 预取指定数量的作业
/// 实际调度的作业数量可能小于请求的数量，如果 DataLoader 耗尽作业
void prefetch(size_t requested_jobs) {
  for (const auto r : c10::irange(requested_jobs)) {
    (void)r; // 抑制未使用变量警告
    if (auto batch_request = get_batch_request()) {
      this->push_job(std::move(*batch_request));
    } else {
      break;
    }
  }
}

/// 根据 max_jobs 选项预取最大数量的作业
void prefetch() {
  prefetch(options_.max_jobs);
}

/// 返回下一个数据批次，如果 DataLoader 耗尽则返回空的 optional
/// 如果仍有批次可用，此操作会阻塞直到批次可用
optional<BatchType> next() {
    // 如果设置了多线程工作数量大于0，则进入循环
    if (options_.workers > 0) {
      // 反复尝试从结果队列中取出结果，直到队列为空
      while (optional<Result> result = this->pop_result()) {
        // 如果结果中包含异常信息，则抛出 WorkerException 异常
        if (result->exception) {
          throw WorkerException(result->exception);
        } else if (result->batch) {
          // 如果结果中包含批次数据，则预取一个新的批次并返回
          prefetch(1);
          return std::move(result->batch);
        }
      }
    } else if (auto batch_request = get_batch_request()) {
      // 如果未设置多线程工作或者结果队列为空，则获取下一个批次请求的数据
      return this->main_thread_dataset_->get_batch(std::move(*batch_request));
    }
    // 如果以上条件都不满足，则返回空值
    return nullopt;
  }

  /// 工作线程运行的函数
  void worker_thread(Dataset& dataset) {
    // 永久循环，直到收到退出信号
    while (true) {
      // 从任务队列中取出下一个作业
      auto job = shuttle_.pop_job();
      // 如果作业指示需要退出，则跳出循环
      if (job.quit) {
        break;
      }
      try {
        // 尝试获取批次数据并推送到结果队列中
        auto batch = dataset.get_batch(std::move(*job.batch_request));
        shuttle_.push_result({std::move(batch), job.sequence_number});
      } catch (...) {
        // 如果发生异常，则将异常信息推送到结果队列中
        shuttle_.push_result({std::current_exception(), job.sequence_number});
      }
    }
  }

  /// 方便的方法，使用给定的值调用 `shuttle_.push_job()` 并分配新的序列号
  template <typename T>
  void push_job(T value) {
    // 将给定的值和当前序列号打包成作业并推送到任务队列中
    shuttle_.push_job({std::move(value), sequence_number_++});
  }

  /// 方便的方法，从序列器中获取下一个结果
  optional<Result> pop_result() {
    // 调用序列器的 `next` 方法来获取下一个结果，传入获取结果的超时时间
    return sequencer_->next(
        [this] { return this->shuttle_.pop_result(this->options_.timeout); });
  }

  /// 方便的方法，基于 `enforce_ordering` 选项创建一个新的序列器
  std::unique_ptr<detail::sequencers::Sequencer<Result>> new_sequencer() {
    // 如果启用了严格顺序选项，则创建一个基于有序的序列器
    if (options_.enforce_ordering) {
      return std::make_unique<detail::sequencers::OrderedSequencer<Result>>(
          options_.max_jobs);
    }
    /// 返回一个独占指针，指向未配置顺序生成器的 `NoSequencer` 实例。
    return std::make_unique<detail::sequencers::NoSequencer<Result>>();
    
    
    
    /// DataLoader 配置的选项。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    const FullDataLoaderOptions options_;
    
    
    
    /// 主线程使用的数据集。当工作线程数配置为零时，主线程需要同步执行所有工作，因此唯一指针指向数据集，不是可选项。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::unique_ptr<Dataset> main_thread_dataset_;
    
    
    
    /// 下一个要从数据集中检索的批次的序列号。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    size_t sequence_number_ = 0;
    
    
    
    /// 工作线程，运行 `worker_thread()` 方法。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::vector<std::thread> workers_;
    
    
    
    /// `DataShuttle` 负责作业的生命周期。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    detail::DataShuttle<Job, Result> shuttle_;
    
    
    
    /// `Sequencer` 处理批次的可选排序。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    std::unique_ptr<detail::sequencers::Sequencer<Result>> sequencer_;
    
    
    
    /// 如果 DataLoader 已经加入了其工作线程，则为 true。
    // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
    bool joined_ = false;
};
} // namespace data
} // namespace torch
```