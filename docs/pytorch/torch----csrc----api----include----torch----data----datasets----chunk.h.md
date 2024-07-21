# `.\pytorch\torch\csrc\api\include\torch\data\datasets\chunk.h`

```
    // 等待直到队列中有可用数据或者所有数据块已加载（即当前轮次数据集已耗尽）
    cv_read_.wait(lock, [this] {
      return (
          this->total_example_count_in_queue_ >= batch_size_ || this->stop_);
    });

    // 如果批次队列为空
    if (batch_queue_.empty()) {
      AT_ASSERT(stop_);
      // 所有批次已获取完毕。返回一个空批次。
      return nullopt;
    }

    // 从队列中取出一个批次数据
    UnwrappedBatchData batch = std::move(batch_queue_.front());
    batch_queue_.pop();

    // 如果批次数据中有异常，抛出异常
    if (batch.exception) {
      throw WorkerException(batch.exception);
    }

    // 更新队列中的总样本计数
    total_example_count_in_queue_ -= batch.batch_data.size();
    lock.unlock();

    // 通知等待中的线程可以写入数据
    cv_write_.notify_all();
    return batch.batch_data;
  }

  /// Push preloaded chunks to batch queue. Called from the ChunkDataset worker
  /// threads.
  void add_chunk_data(UnwrappedBatchType data) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_write_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < this->queue_capacity_ ||
          this->stop_;
    });
    if (stop_) {
      // When stop_ is true, it means no further chunk loading is necessary.
      // Return without any further processing.
      return;
    }

    auto data_size = data.size();
    auto remaining_size = data_size;
    example_sampler_.reset(data_size);

    auto fill_batch = [&](size_t example_count, UnwrappedBatchType& batch) {
      auto batch_example_indices = this->example_sampler_.next(example_count);
      AT_ASSERT(
          batch_example_indices &&
          batch_example_indices.value().size() == example_count);
      BatchRequestType& indices = batch_example_indices.value();
      for (size_t i : indices) {
        TORCH_CHECK(i < data_size, "Index out of range");
        batch.emplace_back(std::move(data[i]));
      }
      remaining_size -= example_count;
    };

    if (!batch_queue_.empty()) {
      // if the queue has existing data, and the last batch doesn't have enough
      // examples to fill a batch_size batch, add more example to this batch
      // first.
      auto& batch = batch_queue_.back();
      size_t current_count = batch.batch_data.size();
      if (current_count < batch_size_) {
        auto example_count =
            std::min(remaining_size, batch_size_ - current_count);
        fill_batch(example_count, batch.batch_data);
      }
    }

    // If we still have data remaining after filling the last pushed batch, add
    // them to the queue too.
    // NOLINTNEXTLINE(bugprone-infinite-loop)
    while (remaining_size > 0) {
      UnwrappedBatchType current_batch;

      // Allocate the batch memory ahead of time.
      current_batch.reserve(batch_size_);

      auto example_count = std::min(remaining_size, batch_size_);
      fill_batch(example_count, current_batch);
      batch_queue_.emplace(std::move(current_batch));
    }
    total_example_count_in_queue_ += data_size;
    lock.unlock();
    cv_read_.notify_all();
  }

  /// Push exceptions thrown during preloading into batch queue. Called from
  /// the ChunkDataset worker threads.
  void add_chunk_data(std::exception_ptr e_ptr) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cv_write_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return (
          this->total_example_count_in_queue_ < this->queue_capacity_ ||
          this->stop_);
    });
    if (stop_) {
      // When stop_ is true, it means this current thread needs to be tore down,
      // the batch buffer will be discarded, so no need to enqueue any new
      // exceptions.
      return;
    }


注释：
    // 将传入的指针 e_ptr 添加到 batch_queue_ 中
    batch_queue_.emplace(e_ptr);
    // 解锁 mutex 对象 lock
    lock.unlock();
    // 通知所有等待在条件变量 cv_read_ 上的线程
    cv_read_.notify_all();
  }

  void stop() {
    {
      // 在修改 stop_ 前获取锁，以防止竞争条件，避免死锁的发生。
      // 具体来说，条件变量 cv_write_ 在 add_chunk_data() 中等待 stop_ 的谓词。
      // 等待分为两步骤：1) 在持有锁的同时，检查谓词是否为真；
      // 2) 如果为真，则继续，否则释放锁并等待通知。
      // 如果在不持有锁的情况下，cv_write_ 的通知可能发生在步骤 1) 和 2) 之间。
      // 在这种情况下，由于 cv_write_ 尚未处于等待状态，因此通知将丢失，
      // cv_write_ 将永远睡眠。通过在修改 stop_ 前获取锁，确保更新和评估 stop_ 始终同步进行。
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }

    // 通知所有等待在条件变量 cv_write_ 上的写线程，唤醒它们以退出当前方法。
    cv_write_.notify_all();
    // 通知所有等待在条件变量 cv_read_ 上的读线程。
    cv_read_.notify_all();
  }
  /// 批处理大小，用于从块数据中创建批次。类似于常规数据加载器，在提前获取的情况下创建批次。
  size_t batch_size_ = 0;

  /// 队列中存储的总示例数目
  size_t total_example_count_in_queue_ = 0;

  /// 包含未包装批次单元的结构体。未包装批次单元是没有 'optional' 包装的原始数据，
  /// 可以是图像、话语等的集合。
  struct UnwrappedBatchData {
    explicit UnwrappedBatchData(UnwrappedBatchType data)
        : batch_data(std::move(data)) {}

    // NOLINTNEXTLINE(modernize-pass-by-value)
    explicit UnwrappedBatchData(std::exception_ptr e) : exception(e) {}

    /// 要返回的批次数据
    UnwrappedBatchType batch_data;

    /// 异常指针，捕获在创建批次时发生的任何异常。
    std::exception_ptr exception;
  };
    // 异常指针，用于捕获并传播异常
    std::exception_ptr exception;
  };

  /// 本地缓存，存储从加载的块中提取的示例批次
  std::queue<UnwrappedBatchData> batch_queue_;

  // 同步更新 batch_queue_
  std::mutex queue_mutex_;

  // 条件变量，用于读操作的同步
  std::condition_variable cv_read_;
  
  // 条件变量，用于写操作的同步
  std::condition_variable cv_write_;

  // 示例采样器的引用
  ExampleSampler& example_sampler_;

  // 队列最大容量，可配置的元素数量上限
  size_t queue_capacity_;

  // 当设置为 true 时，唤醒写线程以退出当前函数调用。
  // 当在上一个 epoch 尚未耗尽时调用 ChunkDataSet.Reset 时，需要这样做。
  // 在 ChunkDataset 等待其预加载器完成前一个工作并拆除线程时，
  // 预加载器可能仍在等待条件变量，从而导致程序挂起。
  // 此布尔值用于打破此等待条件。
  bool stop_ = false;
};

} // namespace detail

/// Options to configure a `ChunkDataset`.
struct ChunkDatasetOptions {
  ChunkDatasetOptions() = delete;
  ChunkDatasetOptions(
      size_t preloader_count,                  // 构造函数，设置预加载器数量
      size_t batch_size,                       // 构造函数，设置批处理大小
      size_t cache_size = 2048,                // 构造函数，默认设置缓存大小为2048
      size_t cross_chunk_shuffle_count = 1)    // 构造函数，默认设置跨块洗牌次数为1
      : preloader_count_(preloader_count),     // 初始化预加载器数量成员变量
        batch_size_(batch_size),               // 初始化批处理大小成员变量
        cache_size_(cache_size),               // 初始化缓存大小成员变量
        cross_chunk_shuffle_count_(cross_chunk_shuffle_count) {  // 初始化跨块洗牌次数成员变量
    TORCH_CHECK(
        preloader_count_ > 0,                  // 检查预加载器数量是否大于0
        "Preloader count is 0. At least one preloader needs to be specified.");
    TORCH_CHECK(
        batch_size_ > 0,                       // 检查批处理大小是否大于0
        "Batch size is 0. A positive batch size needs to be specified.");
    TORCH_CHECK(
        cache_size_ > 0,                       // 检查缓存大小是否大于0
        "Cache size is 0. A positive cache size needs to be specified.");
    TORCH_CHECK(
        cache_size_ >= batch_size_,            // 检查缓存大小是否大于等于批处理大小
        "Cache size is less than batch size. Cache needs to be large enough to "
        "hold at least one batch.");
    TORCH_CHECK(
        cross_chunk_shuffle_count_ > 0,        // 检查跨块洗牌次数是否大于0
        "cross_chunk_shuffle_count needs to be greater than 0.");
  }

  /// The number of worker thread to preload chunk data.
  TORCH_ARG(size_t, preloader_count);          // 预加载器数量成员变量声明

  /// The size of each batch.
  TORCH_ARG(size_t, batch_size);               // 批处理大小成员变量声明

  /// The capacity of the queue for batch caching.
  TORCH_ARG(size_t, cache_size) = 2048;        // 缓存大小成员变量声明，默认值2048

  // The number of chunks to perfrom cross-chunk shuffling. Default to 1 meaning
  // no cross-chunk shuffling. When it is equal to n (n > 1), n random
  // chunks will be loaded at once and example shuffling will be performed
  // across all those n chunks.
  // Note: Usually the default config (1 chunk shuffle + example shuffle) is
  // good enough to generate random distributed data. Use this parameter only if
  // you know cross-shuffle is needed in your case. Also there is a performance
  // penalty when this value is greater than 1, as we need to do extra merge
  // between multiple chunks before performing example sampling.
  TORCH_ARG(size_t, cross_chunk_shuffle_count) = 1;  // 跨块洗牌次数成员变量声明，默认值1
};

/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks.
///
/// Unlike regular dataset, chunk dataset require two samplers to operate and
/// keeps an internal state. `ChunkSampler` selects, which chunk to load next,
/// while the `ExampleSampler` determins the order of Examples that are returned
/// in each `get_batch` call. The hierarchical sampling approach used here is
/// inspired by this paper http://martin.zinkevich.org/publications/nips2010.pdf
template <
    typename ChunkReader,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataset final
  /// public StatefulDataset<
  ///       ChunkDataset<ChunkReader, ChunkSampler, ExampleSampler>,
  ///       typename ChunkReader::BatchType,
  ///       size_t> {
  /// `ChunkDataset` 类继承自 `StatefulDataset`，使用了模板参数 `ChunkReader`、`ChunkSampler` 和 `ExampleSampler`，并指定了返回类型为 `typename ChunkReader::BatchType`，以及一个额外的 `size_t` 参数。
 public:
  /// 定义 `BatchType` 为 `torch::optional<typename ChunkReader::BatchType>`，表示可选类型的批次数据
  using BatchType = torch::optional<typename ChunkReader::BatchType>;
  /// 定义 `UnwrappedBatchType` 为 `typename ChunkReader::BatchType`，表示未包装的批次数据类型
  using UnwrappedBatchType = typename ChunkReader::BatchType;
  /// 定义 `BatchRequestType` 为 `size_t`，表示批次请求类型为大小
  using BatchRequestType = size_t;
  /// 定义 `ChunkSamplerType` 为 `ChunkSampler`，表示分块采样器的类型
  using ChunkSamplerType = ChunkSampler;
  /// 定义 `ExampleSamplerType` 为 `ExampleSampler`，表示示例采样器的类型

  /// 构造函数，初始化 `ChunkDataset` 对象
  ChunkDataset(
      ChunkReader chunk_reader,
      ChunkSampler chunk_sampler,
      ExampleSampler example_sampler,
      ChunkDatasetOptions options,
      std::function<void(UnwrappedBatchType&)> preprocessing_policy =
          std::function<void(UnwrappedBatchType&)>())
      : chunk_reader_(std::move(chunk_reader)),  // 初始化 chunk_reader_
        chunk_sampler_(std::move(chunk_sampler)),  // 初始化 chunk_sampler_
        example_sampler_(std::move(example_sampler)),  // 初始化 example_sampler_
        options_(std::move(options)),  // 初始化 options_
        preprocessing_policy_(std::move(preprocessing_policy)),  // 初始化 preprocessing_policy_
        quit_worker_(false),  // 标记是否退出工作线程为假
        running_preloaders_(0),  // 当前运行的预加载器数量为 0
        load_checkpoint_(false) {}  // 标记是否加载检查点为假

  /// 析构函数，释放 `ChunkDataset` 对象
  ~ChunkDataset() override {
    // 先停止批次缓冲区
    if (batch_buffer_) {
      batch_buffer_->stop();
    }
    // 释放工作线程资源
    free_workers();
  }

  /// 默认的 `get_batch` 方法，返回从预加载的块中创建的示例批次
  /// 方法不依赖具体数据集实现，不需要在不同的块数据集中重写
  BatchType get_batch(size_t batch_size) override {
    // 检查批次缓冲区是否已初始化
    TORCH_CHECK(
        batch_buffer_ != nullptr,
        "Dataset needs to call reset() before calling get_batch().");

    // 检查请求的批次大小是否与初始化的批次大小匹配
    TORCH_CHECK(
        batch_size == options_.batch_size(),
        "The requested batch size does not match with the initialized batch size.\n"
        " The requested batch size is ",
        batch_size,
        ", while the dataset is created with batch size equal to ",
        options_.batch_size());
    // 返回批次缓冲区中的批次数据
    return batch_buffer_->get_batch();
  }

  /// 简化的 `get_batch` 方法调用，默认使用 `options_` 中定义的批次大小
  BatchType get_batch() {
    return get_batch(options_.batch_size());
  }

  /// 清除任何内部状态并启动块数据集的内部预取机制
  void reset() override {
    // 用于支持通过数据加载器迭代器进行部分数据读取
    if (batch_buffer_) {
      batch_buffer_->stop();
    }
    // 释放之前重置的工作线程资源
    free_workers();
    // 清空预加载线程列表
    preload_threads_.clear();

    // 如果未加载检查点，则重置 `chunk_reader_` 和 `chunk_sampler_`
    if (!load_checkpoint_) {
      chunk_reader_.reset();
      chunk_sampler_.reset(chunk_reader_.chunk_count());
      load_checkpoint_ = false;
    }

    // 丢弃缓冲区中的任何现有批次，并重新创建一个新的块缓冲区
    batch_buffer_ = std::make_unique<
        detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>(
        options_.batch_size(), example_sampler_, options_.cache_size());

    // 为新的周期创建新的工作线程
    // 设置一个标志位，表示工作线程不应该退出
    quit_worker_ = false;

    // 断言运行中的预加载器数量为0，确保没有预加载器在运行
    AT_ASSERT(running_preloaders_ == 0);
    // 将预加载器的数量设置为选项中指定的数量
    running_preloaders_ = options_.preloader_count();
    // 根据预加载器数量循环，创建相应数量的线程并启动预加载器
    for (const auto i : c10::irange(options_.preloader_count())) {
      preload_threads_.emplace_back([this, i]() { this->preloader(i); });
    }
  }

  /// size is not used for chunk dataset.
  // 对于分块数据集，size 方法不被使用，始终返回空的 optional
  optional<size_t> size() const override {
    return torch::nullopt;
  }

  // 提供对分块采样器的引用。主要用于分布式数据加载，用于设置采样器的轮次编号。
  ChunkSamplerType& chunk_sampler() {
    return chunk_sampler_;
  }

  // 将当前对象保存到输出存档中
  void save(serialize::OutputArchive& archive) const override {
    std::lock_guard<std::mutex> lock(chunk_index_guard_);
    chunk_sampler_.save(archive);
  }

  // 从输入存档中加载对象
  void load(serialize::InputArchive& archive) override {
    std::lock_guard<std::mutex> lock(chunk_index_guard_);
    chunk_sampler_.load(archive);
    load_checkpoint_ = true;
  }

 private:
  /// 在工作线程上运行的预加载器，用于预加载块数据。
  void preloader(size_t id) {
    // 当 quit_worker_ 标志为 false 时循环执行预加载器任务
    while (!quit_worker_.load()) {
      try {
        std::vector<size_t> chunk_idx;
        {
          std::lock_guard<std::mutex> lock(chunk_index_guard_);
          // 获取下一个块采样器结果，根据跨块洗牌计数获取块索引
          if (auto chunk_sampler_result = chunk_sampler_.next(
                  this->options_.cross_chunk_shuffle_count())) {
            chunk_idx = chunk_sampler_result.value();
          } else {
            break; // 如果没有获取到有效的采样结果则退出循环
          }
        }
        // 读取第一个块的数据
        UnwrappedBatchType data = chunk_reader_.read_chunk(chunk_idx[0]);
        // 遍历读取剩余块的数据并添加到数据集中
        for (const auto i : c10::irange(1, chunk_idx.size())) {
          auto chunk_data = chunk_reader_.read_chunk(chunk_idx[i]);
          std::move(
              chunk_data.begin(), chunk_data.end(), std::back_inserter(data));
        }
        // 如果设置了预处理策略，则对数据进行预处理
        if (preprocessing_policy_) {
          preprocessing_policy_(data);
        }
        // 如果数据不为空，则将数据添加到批处理缓冲区中
        if (!data.empty()) { // 跳过空块数据
          batch_buffer_->add_chunk_data(std::move(data));
        }
      } catch (...) {
        // 在批处理缓冲区中添加当前异常
        batch_buffer_->add_chunk_data(std::current_exception());
      }
    }
    // 确保运行中的预加载器数量大于0
    AT_ASSERT(running_preloaders_.load() > 0);
    // 递减运行中的预加载器数量
    --running_preloaders_;
    // 如果运行中的预加载器数量为0，则通知批处理缓冲区停止操作
    if (running_preloaders_.load() == 0) {
      // 所有预加载器已完成，因此可以通知批处理缓冲区
      batch_buffer_->stop();
    }
  }

  /// 阻塞当前线程，直到所有工作线程执行完毕并退出。
  void free_workers() {
    if (!quit_worker_.load()) {
      // 设置退出工作线程的标志为真
      quit_worker_ = true;
      // 等待所有预加载线程结束
      for (auto& worker_thread : preload_threads_) {
        worker_thread.join();
      }
      //```
  }
}

private:
// 定义模板类，定义了什么是一个 chunk（数据块）以及如何读取 chunk 数据。
// 当 chunk_reader_ 返回一个 chunk，ChunkDataset 将其拆分为批次，并将它们缓存到 batch_buffer_ 中。
ChunkReader chunk_reader_;

// 用于对不同的 chunk 进行洗牌的 chunk sampler。
ChunkSamplerType chunk_sampler_;

// 用于对特定 chunk 中的示例进行洗牌的示例 sampler。
ExampleSamplerType example_sampler_;

// 批次数据缓冲区，保存来自预加载线程的 chunk 数据。
std::shared_ptr<detail::BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>> batch_buffer_;

// 工作线程池
std::vector<std::thread> preload_threads_;

/// Dataset 配置的选项。
const ChunkDatasetOptions options_;

// 函数指针包装器，用于在 chunk 数据上应用自定义处理。
// 这对于希望在示例抽样为小批次之前对 chunk 数据执行预处理（如分桶）的开发者来说是一个高级参数。
// 不同于 collate 函数，此策略应用于 chunk 级别，而不是小批次级别。
// 当加载数据块时（如果 cross_chunk_shuffle_count_ 大于 1，则是多个数据块），将此策略应用于完整加载的数据。
// 如果开发者希望在示例抽样器对数据进行抽样之前对 chunk 数据执行预处理（如分桶），这将非常有用。
// 默认情况下，它是一个空指针，不会执行任何操作。
std::function<void(UnwrappedBatchType&)> preprocessing_policy_;

// 指示工作线程是否可以停止的原子布尔值。
std::atomic<bool> quit_worker_;

// 跟踪正在运行的预加载器，以通知批次缓冲区。数值为 0 表示 chunk 加载已完成。
std::atomic<size_t> running_preloaders_;

// 用于同步 chunk sampler 的 next() 调用的互斥锁。
mutable std::mutex chunk_index_guard_;

// 布尔值，指示是否需要为 chunk_sampler_ 加载检查点。
bool load_checkpoint_;
};
} // 结束 torch 命名空间
} // 结束 data 命名空间
} // 结束 datasets 命名空间
} // 结束 namespace 声明
```