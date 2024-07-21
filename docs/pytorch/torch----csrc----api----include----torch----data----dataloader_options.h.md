# `.\pytorch\torch\csrc\api\include\torch\data\dataloader_options.h`

```
#pragma once
// 引入 Torch 库中的参数定义和类型定义
#include <torch/arg.h>
#include <torch/types.h>

// 引入标准库头文件
#include <chrono>
#include <cstddef>

// Torch 命名空间开始
namespace torch {
// 数据命名空间开始
namespace data {

/// Options to configure a `DataLoader`.
// 用于配置 `DataLoader` 的选项结构体
struct DataLoaderOptions {
  // 默认构造函数
  DataLoaderOptions() = default;
  // 显式构造函数，设置批处理大小
  /* implicit */ DataLoaderOptions(size_t batch_size)
      : batch_size_(batch_size) {}

  /// The size of each batch to fetch.
  // 每个批次的大小，默认为1
  TORCH_ARG(size_t, batch_size) = 1;

  /// The number of worker threads to launch. If zero, the main thread will
  /// synchronously perform the data loading.
  // 启动的工作线程数，如果为零，则主线程同步执行数据加载
  TORCH_ARG(size_t, workers) = 0;

  /// The maximum number of jobs to enqueue for fetching by worker threads.
  /// Defaults to two times the number of worker threads.
  // 由工作线程异步获取的最大作业数，默认为工作线程数的两倍
  TORCH_ARG(optional<size_t>, max_jobs);

  /// An optional limit on the time to wait for the next batch.
  // 等待下一个批次的可选时间限制
  TORCH_ARG(optional<std::chrono::milliseconds>, timeout);

  /// Whether to enforce ordering of batches when multiple are loaded
  /// asynchronously by worker threads. Set to `false` for better performance if
  /// you do not care about determinism.
  // 当多个批次被工作线程异步加载时，是否强制保持批次顺序，默认为 true
  TORCH_ARG(bool, enforce_ordering) = true;

  /// Whether to omit the last batch if it contains less than `batch_size`
  /// examples.
  // 是否省略包含少于 `batch_size` 个示例的最后一个批次，默认为 false
  TORCH_ARG(bool, drop_last) = false;
};

/// Like `DataLoaderOptions`, but without any unconfigured state.
/// `DataLoaderOptions` has some options that depend on other options
/// (`max_jobs` => `2 * workers`). In the spirit of properly using the C++ type
/// system, `DataLoaderOptions` allows only setting values. To access values,
/// you must create a `FullDataLoaderOptions` from a `DataLoaderOptions`
/// instance, which will do any necessary coalescing.
// 与 `DataLoaderOptions` 类似，但没有未配置的状态
// `DataLoaderOptions` 有些选项依赖于其他选项 (`max_jobs` => `2 * workers`)。
// 为了正确使用 C++ 类型系统，`DataLoaderOptions` 只允许设置值。
// 要访问值，必须从 `DataLoaderOptions` 实例创建 `FullDataLoaderOptions`，
// 这将执行任何必要的合并。
struct FullDataLoaderOptions {
  // 显式构造函数，从 `DataLoaderOptions` 实例初始化所有成员
  explicit FullDataLoaderOptions(DataLoaderOptions options)
      : batch_size(options.batch_size()),   // 批处理大小
        workers(options.workers()),         // 工作线程数
        max_jobs(options.max_jobs().value_or(2 * workers)),  // 最大作业数，默认为工作线程数的两倍
        timeout(options.timeout()),         // 等待时间限制
        enforce_ordering(options.enforce_ordering()),  // 是否强制批次顺序
        drop_last(options.drop_last()) {}   // 是否省略最后一个小批次

  size_t batch_size;  // 批处理大小
  size_t workers;     // 工作线程数
  size_t max_jobs;    // 最大作业数
  optional<std::chrono::milliseconds> timeout;  // 等待时间限制
  bool enforce_ordering;  // 是否强制批次顺序
  bool drop_last;         // 是否省略最后一个小批次
};

} // namespace data
} // namespace torch
```