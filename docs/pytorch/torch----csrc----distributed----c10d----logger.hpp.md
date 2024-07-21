# `.\pytorch\torch\csrc\distributed\c10d\logger.hpp`

```py
// 包含头文件 Logging.h 和 reducer.hpp，这些文件提供了日志和分布式减少器的定义
#include <c10/util/Logging.h>
#include <torch/csrc/distributed/c10d/reducer.hpp>

// 引入标准库中的实用工具
#include <utility>

// 声明 c10d 命名空间，用于封装相关类和函数
namespace c10d {

// 定义一个 TORCH_API 类 Logger，用于日志记录
class TORCH_API Logger {
 public:
  // 构造函数，接受一个 shared_ptr 指向 c10d::Reducer 对象
  explicit Logger(std::shared_ptr<c10d::Reducer> reducer);

  // 设置构造时可以获取的日志数据
  void set_construction_data_and_log(
      const std::string& module_name,  // 模块名
      const std::vector<int>& device_ids,  // 设备 IDs
      int output_device,  // 输出设备
      bool broadcast_buffers,  // 是否广播缓冲区
      bool has_sync_bn,  // 是否有同步 BN
      bool static_graph);  // 是否静态图

  // 设置静态图标志
  void set_static_graph();

  // 用户获取 DDPLoggingData 并在应用程序中记录的接口
  // DDPLoggingData 结构在 torch/c10/util/Logging.h 的 struct DDPLoggingData 中有详细说明
  at::DDPLoggingData get_ddp_logging_data();

  // 友元函数，重载流插入运算符 <<，用于将日志数据输出到流中，一般用于 TORCH_DISTRIBUTED_DEBUG
  friend std::ostream& operator<<(std::ostream& output, const Logger& logger);

  // 析构函数，抛出异常时不会调用 noexcept(false)
  ~Logger() noexcept(false) {
    // 在 Logger 的析构函数中记录 DDP 图是否是静态的，而不是在 Reducer 的析构函数中记录，因为 Logger 先于 Reducer 删除
  // 调用 log_if_graph_static 方法，传入 reducer_ 的 ddp_graph_static() 方法返回的结果作为参数
  log_if_graph_static(reducer_->ddp_graph_static());
}

// 设置环境变量。
void set_env_variables();

// 设置参数统计信息。
void set_parameter_stats();

// 获取每个桶的大小（字节）。
std::vector<int64_t> get_bucket_sizes();

// 获取每个桶中变量的索引。
std::vector<std::vector<size_t>> get_per_bucket_variable_indices();

// 设置通信钩子，如果有的话。
void set_comm_hook(const std::string& hook);

// 设置使用不均匀输入检测（在 model.join() 上下文管理器中）。
void set_uneven_input_join();

// 在当前迭代重置性能统计信息。
void reset_performance_stats();

// 使用记录在 reducer 中的 cpu 计时器和 gpu 计时器计算平均统计信息。
void calculate_avg_time(
    int64_t& avg_time,
    int64_t& time_duration,
    Timer& timer,
    Timer::Event start_event,
    Timer::Event end_event);

// 设置记录在 reducer 中的事件的绝对时间。
void set_event_time(int64_t& event_time, Timer& timer, Timer::Event event);

// 设置只能在训练循环期间收集的统计信息。
// 在 forward 调用的开始记录以前运行过的抽样迭代的运行时统计信息。
// GPU 性能统计信息仅适用于单进程单设备程序和单设备模块。
// TODO：支持单进程多设备和多设备模块时，需要创建并记录多设备上的事件。
void set_runtime_stats_and_log();

// 当 DDP/reducer 遇到错误时调用。
// 日志数据结构将填充两个字段：
// "has_error" 表示该迭代遇到错误，其他字段无效；
// "error" 是一个包含 DDP 失败错误消息的字符串。
template <typename... Args>
void set_error_and_log(const std::string& ddp_error, const Args&... args) {
  ddp_logging_data_->ints_map["has_error"] = 1;
  auto err = c10::str(ddp_error, args...);
  ddp_logging_data_->strs_map["error"] = err;
  // 报告错误发生时的迭代次数，以便用户知道在出现错误之前成功处理了多少个示例。
  ddp_logging_data_->ints_map["iteration"] = reducer_->num_iterations_;
  at::LogPyTorchDDPUsage(*ddp_logging_data_);
}

// 当运行时没有静态图时调用，用于在销毁 reducer 时记录日志，
// 如果图确实是静态的，并且是静态图优化的候选。
void log_if_graph_static(bool is_static);

private:
// ddp_logging_data_ 用于保存所有与 DDP 相关的日志数据字段。
std::unique_ptr<at::DDPLoggingData> ddp_logging_data_;
std::shared_ptr<c10d::Reducer> reducer_;
// 追踪到目前为止收集的运行时统计信息的迭代次数。
long num_iterations_stats_recorded_ = 0;
};

// 结构体定义：用于存储不同类型的日志数据，包括键值对字符串和整数
// 可以根据需要扩展到更多类型。
struct C10dLoggingData {
  // 字符串类型的日志字段
  std::map<std::string, std::string> strings;
  // int64_t 类型的日志字段
  std::map<std::string, int64_t> integers;
};

// TORCH_API 修饰的 C10dLogger 类
class TORCH_API C10dLogger {
 public:
  // 默认拷贝构造函数
  C10dLogger(const C10dLogger&) = default;
  // 删除移动构造函数
  C10dLogger(C10dLogger&&) = delete;
  // 默认拷贝赋值运算符
  C10dLogger& operator=(const C10dLogger&) = default;
  // 删除移动赋值运算符
  C10dLogger& operator=(C10dLogger&&) = delete;
  // 默认虚析构函数
  virtual ~C10dLogger() = default;
  
  // 纯虚函数，用于日志记录
  virtual void log(const C10dLoggingData& data);
  
  // 获取 logger 单例对象
  static C10dLogger* getLogger();
  
  // 注册 logger 实例
  static void registerLogger(std::unique_ptr<C10dLogger>);

 protected:
  // 构造函数被保护，单例模式
  // 参数为日志的目标位置
  C10dLogger(std::string logDestination)
      : logDestination_(std::move(logDestination)) {}

  // 日志记录的目标位置名称
  std::string logDestination_;

 private:
  // logger 单例指针
  static std::unique_ptr<C10dLogger> logger_;
  // 是否已注册的原子标志
  static std::atomic<bool> registered_;
};

// c10d 命名空间的结束
} // namespace c10d
```