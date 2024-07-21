# `.\pytorch\torch\csrc\distributed\c10d\logger.cpp`

```py
// 包含字符串处理工具库的头文件
#include <c10/util/StringUtil.h>
// 包含格式化输出库的头文件
#include <fmt/format.h>
// 包含分布式工具中的实用函数头文件
#include <torch/csrc/distributed/c10d/Utils.hpp>
// 包含分布式调试相关的头文件
#include <torch/csrc/distributed/c10d/debug.h>
// 包含分布式日志记录器的头文件
#include <torch/csrc/distributed/c10d/logger.hpp>
// 包含字符串操作工具库的头文件
#include <string>

// 包含线程安全调用一次的工具库的头文件
#include <c10/util/CallOnce.h>

#ifdef USE_C10D_GLOO
// 当启用 GLOO 后，包含 GLOO 进程组的头文件
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#endif

// 定义静态全局变量，用于记录 TORCH_NCCL_BLOCKING_WAIT 和 NCCL_BLOCKING_WAIT 的环境变量值
namespace c10d {
static std::vector<std::string> TORCH_NCCL_BLOCKING_WAIT = {
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_BLOCKING_WAIT"};
// 定义静态全局变量，用于记录 TORCH_NCCL_ASYNC_ERROR_HANDLING 和 NCCL_ASYNC_ERROR_HANDLING 的环境变量值
static std::vector<std::string> TORCH_NCCL_ASYNC_ERROR_HANDLING = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "NCCL_ASYNC_ERROR_HANDLING"};

// 定义用于记录运行时日志的迭代次数数组，这些迭代次数将被记录到日志中
const int LoggingIterations[] = {10, 20, 50, 100, 500, 800, 1000}; // NOLINT

// 重载运算符 <<，用于打印 Logger 对象的信息到输出流
std::ostream& operator<<(std::ostream& output, const Logger& logger) {
  // 获取日志数据
  auto& ddp_logging_data = (*logger.ddp_logging_data_);

  // 格式化日志信息
  std::string loggerInfo = fmt::format(
      "[Rank {} / {}] [before iteration {}] Training {} unused_parameter_size={} \n "
      "Avg forward compute time: {} \n Avg backward compute time: {} \n"
      "Avg backward comm. time: {} \n Avg backward comm/comp overlap time: {}",
      ddp_logging_data.ints_map["rank"],
      ddp_logging_data.ints_map["world_size"],
      ddp_logging_data.ints_map["iteration"],
      ddp_logging_data.strs_map["module_name"],
      ddp_logging_data.ints_map["unused_parameter_size"],
      ddp_logging_data.ints_map["avg_forward_compute_time"],
      ddp_logging_data.ints_map["avg_backward_compute_time"],
      ddp_logging_data.ints_map["avg_backward_comm_time"],
      ddp_logging_data.ints_map["avg_backward_compute_comm_overlap_time"]);

  // 如果存在通信钩子信息，则添加到日志中
  if (!ddp_logging_data.strs_map["comm_hook"].empty()) {
    loggerInfo += fmt::format(
        "\n Gradient comm. hook: {}", ddp_logging_data.strs_map["comm_hook"]);
  }

  // 如果设置了 uneven input 检测，则添加相应的信息到日志中
  if (ddp_logging_data.ints_map["join_uneven_inputs"]) {
    loggerInfo += "\n Uneven input detection with join() enabled.";
  }

  // 将格式化好的日志信息输出到流中
  return output << loggerInfo;
}

// Logger 类的构造函数，接受一个 Reducer 对象的智能指针作为参数
Logger::Logger(std::shared_ptr<c10d::Reducer> reducer)
    : reducer_(std::move(reducer)) {
  // 创建一个新的 DDPLoggingData 对象，用于存储日志数据
  ddp_logging_data_ = std::make_unique<at::DDPLoggingData>();
}

// 静态标志，用于标记是否已经记录了静态图的日志信息
c10::once_flag log_graph_static_flag;

// 如果静态图标志尚未设置，则记录静态图信息
void Logger::log_if_graph_static(bool is_static) {
  // 使用 c10::call_once 确保此操作仅执行一次
  c10::call_once(log_graph_static_flag, [this, is_static]() {
    // 设置静态图信息
    ddp_logging_data_->ints_map["can_set_static_graph"] = is_static;
    // 记录训练完成的迭代次数
    ddp_logging_data_->ints_map["iteration"] = reducer_->num_iterations_;
    // 调用 PyTorch DDP 使用日志记录器
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  });
}

// 环境变量相关的定义和处理
// 这部分略，未完整给出
void Logger::set_env_variables() {
  // 设置日志数据结构中的环境变量信息
  ddp_logging_data_->strs_map["master_port"] =
      getCvarString({"MASTER_PORT"}, "N/A");
  // 获取并设置 MASTER_ADDR 环境变量
  ddp_logging_data_->strs_map["master_addr"] =
      getCvarString({"MASTER_ADDR"}, "N/A");
  // 获取并设置 TORCH_DISTRIBUTED_DEBUG 环境变量
  ddp_logging_data_->strs_map["torch_distributed_debug"] =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, "N/A");
  // 获取并设置 CUDA_VISIBLE_DEVICES 环境变量
  ddp_logging_data_->strs_map["cuda_visible_devices"] =
      getCvarString({"CUDA_VISIBLE_DEVICES"}, "N/A");
  // 如果后端为 nccl，则获取并设置相关环境变量
  if (reducer_->process_group_->getBackendName() == "nccl") {
    // 获取并设置 NCCL_SOCKET_IFNAME 环境变量
    ddp_logging_data_->strs_map["nccl_socket_ifname"] =
        getCvarString({"NCCL_SOCKET_IFNAME"}, "N/A");
    // 获取并设置 TORCH_NCCL_BLOCKING_WAIT 环境变量
    ddp_logging_data_->strs_map["nccl_blocking_wait"] =
        getCvarString(TORCH_NCCL_BLOCKING_WAIT, "N/A");
    // 获取并设置 TORCH_NCCL_ASYNC_ERROR_HANDLING 环境变量
    ddp_logging_data_->strs_map["nccl_async_error_handling"] =
        getCvarString(TORCH_NCCL_ASYNC_ERROR_HANDLING, "N/A");
    // 获取并设置 NCCL_DEBUG 环境变量
    ddp_logging_data_->strs_map["nccl_debug"] =
        getCvarString({"NCCL_DEBUG"}, "N/A");
    // 获取并设置 NCCL_NTHREADS 环境变量
    ddp_logging_data_->strs_map["nccl_nthreads"] =
        getCvarString({"NCCL_NTHREADS"}, "N/A");
    // 获取并设置 NCCL_IB_TIMEOUT 环境变量
    ddp_logging_data_->strs_map["nccl_ib_timeout"] =
        getCvarString({"NCCL_IB_TIMEOUT"}, "N/A");
  }
  // 如果后端为 gloo，则获取并设置相关环境变量
  if (reducer_->process_group_->getBackendName() == "gloo") {
    // 获取并设置 GLOO_SOCKET_IFNAME 环境变量
    ddp_logging_data_->strs_map["gloo_socket_ifname"] =
        getCvarString({"GLOO_SOCKET_IFNAME"}, "N/A");
    // 获取并设置 GLOO_DEVICE_TRANSPORT 环境变量
    ddp_logging_data_->strs_map["gloo_device_transport"] =
        getCvarString({"GLOO_DEVICE_TRANSPORT"}, "N/A");

#ifdef USE_C10D_GLOO
    // 如果使用 C10D GLOO，则获取并设置 gloo_pg 的线程数
    auto gloo_pg = static_cast<c10d::ProcessGroupGloo*>(
        reducer_->process_group_
            ->getBackend(c10d::ProcessGroup::BackendType::GLOO)
            .get());
    auto n_threads = gloo_pg->getNumThreads();
    ddp_logging_data_->ints_map["gloo_num_threads"] = n_threads;
#endif
  }
}

void Logger::set_parameter_stats() {
  // 设置日志数据结构中的参数统计信息

  // 记录参数张量的数量
  ddp_logging_data_->ints_map["num_parameter_tensors"] =
      reducer_->params_.size();
  // 初始化总参数大小为 0 字节
  ddp_logging_data_->ints_map["total_parameter_size_bytes"] = 0;
  // 统计所有参数张量的总大小（字节）
  // 同时收集参数的数据类型，可能存在多种数据类型（用于混合精度训练）
  std::set<std::string> unique_dtypes;
  for (const auto& t : reducer_->params_) {
    // 计算每个张量的大小并累加到总大小
    ddp_logging_data_->ints_map["total_parameter_size_bytes"] +=
        t.numel() * t.element_size();
    // 收集张量的数据类型
    unique_dtypes.insert(std::string(t.dtype().name()));
  }
  // 将收集到的数据类型用逗号连接成字符串，并存入日志数据结构
  ddp_logging_data_->strs_map["dtypes"] = c10::Join(", ", unique_dtypes);
}

std::vector<std::vector<size_t>> Logger::get_per_bucket_variable_indices() {
  // 获取每个桶中变量的索引列表

  // 预留每个桶的变量索引列表的空间
  std::vector<std::vector<size_t>> per_bucket_variable_indices;
  per_bucket_variable_indices.reserve(reducer_->buckets_.size());
  // 遍历每个桶，获取其中的变量索引列表，并添加到结果集合中
  for (const auto& bucket : reducer_->buckets_) {
    const auto& indices = bucket.variable_indices;
    per_bucket_variable_indices.push_back(indices);
  }
  // 返回所有桶的变量索引列表
  return per_bucket_variable_indices;
}

std::vector<int64_t> Logger::get_bucket_sizes() {
  // 获取每个桶的大小信息

  std::vector<int64_t> bucket_sizes;
  // 遍历每个桶，获取其大小信息，并添加到结果集合中
  for (const auto& bucket : reducer_->buckets_) {
    # 获取 bucket 对象中的 variables 成员的引用
    const auto& variables = bucket.variables;
    
    # 初始化 bucket_size 变量为 0，用于累加 bucket 中所有变量的总大小
    int64_t bucket_size = 0;
    
    # 遍历 bucket 中的每个变量 v
    for (const auto& v : variables) {
      
      # 计算变量 v 的元素数量乘以每个元素的大小，并累加到 bucket_size
      bucket_size += v.numel() * v.element_size();
    }
    
    # 将计算得到的 bucket_size 添加到 bucket_sizes 后面
    bucket_sizes.push_back(bucket_size);
  }
  
  # 返回存储了所有 bucket 大小的列表 bucket_sizes
  return bucket_sizes;
}

// Communication hook. Empty string if not set, in which case it will not be
// logged.
void Logger::set_comm_hook(const std::string& hook) {
  // 将通讯钩子设置到字符串映射中，如果未设置，则为空字符串，不会记录日志
  ddp_logging_data_->strs_map["comm_hook"] = hook;
}

// Whether we are running under model.join() context manager for DDP uneven
// inputs.
void Logger::set_uneven_input_join() {
  // 设置标志以指示是否在 DDP 不均匀输入的情况下运行在 model.join() 上下文管理器中
  ddp_logging_data_->ints_map["join_uneven_inputs"] = true;
}

void Logger::set_static_graph() {
  // 将静态图设置到整数映射中，从 reducer 对象中获取 static_graph_ 属性
  ddp_logging_data_->ints_map["static_graph"] = reducer_->static_graph_;
}

// Data that can be got during DistributedDataParallel construction time
void Logger::set_construction_data_and_log(
    const std::string& module_name,
    const std::vector<int>& device_ids,
    int output_device,
    bool broadcast_buffers,
    bool has_sync_bn,
    bool static_graph) {
  // 在 DistributedDataParallel 构造函数中调用，无需锁
  if (static_graph) {
    set_static_graph();
  }
  // 设置模块名称到字符串映射中
  ddp_logging_data_->strs_map["module_name"] = module_name;
  // 设置世界大小到整数映射中，从 reducer 的 process_group_ 对象获取大小信息
  ddp_logging_data_->ints_map["world_size"] =
      reducer_->process_group_->getSize();
  // 设置当前进程的排名到整数映射中，从 reducer 的 process_group_ 对象获取排名信息
  ddp_logging_data_->ints_map["rank"] = reducer_->process_group_->getRank();
  // 设置迭代次数到整数映射中，默认为0，表示在训练循环之前获取数据
  ddp_logging_data_->ints_map["iteration"] = 0;
  // 设置是否为多设备模块到整数映射中，从 reducer 对象获取信息
  ddp_logging_data_->ints_map["is_multi_device_module"] =
      reducer_->is_multi_device_module_;

  // 设置参数统计信息
  set_parameter_stats();
  // 设置桶大小列表到字符串映射中，使用逗号分隔的格式输出
  ddp_logging_data_->strs_map["bucket_sizes"] =
      c10::Join(", ", get_bucket_sizes());
  // 设置环境变量
  set_env_variables();

  // 设置设备 IDs 列表到字符串映射中，使用逗号分隔的格式输出
  ddp_logging_data_->strs_map["device_ids"] = c10::Join(", ", device_ids);
  // 设置输出设备到整数映射中
  ddp_logging_data_->ints_map["output_device"] = output_device;
  // 设置是否广播缓冲区到整数映射中
  ddp_logging_data_->ints_map["broadcast_buffers"] = broadcast_buffers;
  // 设置是否有同步 BatchNorm 到整数映射中
  ddp_logging_data_->ints_map["has_sync_bn"] = has_sync_bn;
  // 设置桶容量字节数到整数映射中，从 reducer 对象获取信息
  ddp_logging_data_->ints_map["bucket_cap_bytes"] = reducer_->bucket_bytes_cap_;
  // 设置是否查找未使用的参数到整数映射中，从 reducer 对象获取信息
  ddp_logging_data_->ints_map["find_unused_parameters"] =
      reducer_->find_unused_parameters_;
  // 设置是否将梯度视图作为桶视图到整数映射中，从 reducer 对象获取信息
  ddp_logging_data_->ints_map["gradient_as_bucket_view"] =
      reducer_->gradient_as_bucket_view_;
  // 设置后端名称到字符串映射中，从 reducer 的 process_group_ 对象获取后端信息
  ddp_logging_data_->strs_map["backend_name"] =
      reducer_->process_group_->getBackendName();

  if (debug_level() != DebugLevel::Off) {
    // 如果调试级别不是关闭，则生成初始化信息字符串
    std::string initInfo = fmt::format(
        "[Rank {}]: DDP Initialized with: \n",
        ddp_logging_data_->ints_map["rank"]);
    // 生成整数映射信息的字符串
    std::stringstream ddpLoggingDataInfo;
    for (const auto& intItem : ddp_logging_data_->ints_map) {
      ddpLoggingDataInfo << intItem.first << ": " << intItem.second << "\n";
    }
    // 生成字符串映射信息的字符串
    for (const auto& strItem : ddp_logging_data_->strs_map) {
      ddpLoggingDataInfo << strItem.first << ": " << strItem.second << "\n";
    }
    LOG(INFO) << initInfo << ddpLoggingDataInfo.str();


    // 输出信息日志，包括初始化信息和数据信息的字符串表示
    LOG(INFO) << initInfo << ddpLoggingDataInfo.str();



  }

  at::LogPyTorchDDPUsage(*ddp_logging_data_);


  // 调用PyTorch的函数记录分布式数据并行（DDP）的使用情况
  at::LogPyTorchDDPUsage(*ddp_logging_data_);
}

// 设置事件时间的方法，将事件时间设置为指定事件的时间戳
void Logger::set_event_time(
    int64_t& event_time,
    Timer& timer,
    Timer::Event event) {
  // 获取指定事件的时间戳
  auto timestamp = timer.getTimestamp(event);
  // 如果时间戳有效，则将事件时间设置为该时间戳
  if (timestamp != c10::nullopt) {
    event_time = *timestamp;
  }
}

// 计算平均时间的方法，根据开始和结束事件计算时间差，并更新平均时间
void Logger::calculate_avg_time(
    int64_t& avg_time,
    int64_t& time_duration,
    Timer& timer,
    Timer::Event start_event,
    Timer::Event end_event) {
  // 检查记录的迭代次数是否大于0
  TORCH_CHECK(num_iterations_stats_recorded_ > 0);
  // 测量开始和结束事件之间的时间差
  std::optional<int64_t> maybe_time_duration =
      timer.measureDifference(start_event, end_event);
  // 如果时间差不可用，则返回
  if (!maybe_time_duration.has_value()) {
    return;
  }
  // 更新时间差
  time_duration = maybe_time_duration.value();
  // 计算新的平均时间
  avg_time = (time_duration + avg_time * (num_iterations_stats_recorded_ - 1)) /
      num_iterations_stats_recorded_;
}

// 重置性能统计数据的方法，将所有性能指标重置为零
void Logger::reset_performance_stats() {
  // 将所有性能指标重置为零
  ddp_logging_data_->ints_map["forward_compute_time"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time"] = 0;
  ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"] = 0;
  ddp_logging_data_->ints_map["forward_compute_time_start"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time_start"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time_start"] = 0;
  ddp_logging_data_->ints_map["backward_compute_time_end"] = 0;
  ddp_logging_data_->ints_map["backward_comm_time_end"] = 0;
}

// 设置运行时统计信息并记录日志的方法
void Logger::set_runtime_stats_and_log() {
  // 同步与reducer的数据，使用互斥锁确保线程安全
  std::lock_guard<std::mutex> lock(reducer_->mutex_);
  // 在采样迭代期间设置运行时统计信息
  if (!reducer_->should_collect_runtime_stats()) {
    return;
  }
  // 增加记录的迭代次数
  num_iterations_stats_recorded_++;
  // 设置当前迭代的序号
  ddp_logging_data_->ints_map["iteration"] = reducer_->num_iterations_;
  // 当调用get_ddp_logging_data()时，更新"unused_parameter_size"、
  // "has_rebuilt_buckets"和"rebuilt_bucket_sizes"在最新采样迭代中的值
  // 如果未使用的参数列表为空，并且需要查找未使用的参数，则设置未使用参数大小为零
  if (reducer_->unused_parameters_.empty() &&
      reducer_->find_unused_parameters_) {
    ddp_logging_data_->ints_map["unused_parameter_size"] = 0;
  }
  // 计算未使用参数的大小
  for (const auto& unused_index : reducer_->unused_parameters_) {
    const auto& v = reducer_->params_[unused_index];
    ddp_logging_data_->ints_map["unused_parameter_size"] +=
        v.numel() * v.element_size();
  }
  // 重建桶的统计信息只在第一次迭代后设置一次，因此只需在第一次检测到重建桶时设置
  if (ddp_logging_data_->ints_map["has_rebuilt_buckets"] !=
      reducer_->has_rebuilt_bucket_) {
    ddp_logging_data_->ints_map["has_rebuilt_buckets"] =
        reducer_->has_rebuilt_bucket_;
    // 将重新构建的桶大小信息记录到日志数据中
    ddp_logging_data_->strs_map["rebuilt_bucket_sizes"] =
        c10::Join(", ", get_bucket_sizes());

    // 记录每个桶的变量索引信息到字符串向量中
    std::vector<std::string> per_bucket_variable_indices;
    auto indices = get_per_bucket_variable_indices();
    per_bucket_variable_indices.reserve(indices.size());
    for (const auto& bucket_indices : indices) {
      per_bucket_variable_indices.push_back(c10::Join(" ", bucket_indices));
    }
    // 将重建的每个桶的参数索引信息记录到日志数据中
    ddp_logging_data_->strs_map["rebuilt_per_bucket_param_indices"] =
        c10::Join(", ", per_bucket_variable_indices);
  }

  // 记录梯度准备好的顺序索引到日志数据中
  if (!reducer_->grad_ready_order_indices_.empty()) {
    // 注意：这些索引是上一个迭代的，因为这个函数在前向传播中调用，
    // 我们上次计算梯度准备好的顺序是在上一个反向传播中。
    ddp_logging_data_->strs_map["prev_iteration_grad_ready_order_indices"] =
        c10::Join(", ", reducer_->grad_ready_order_indices_);
  }

  // 重置性能统计数据
  reset_performance_stats();

  // 仅针对单设备模块收集 CUDA 时间统计信息
  if (reducer_->params_[0].is_cuda() && reducer_->is_multi_device_module_) {
    // 如果是多设备模块，不收集 CUDA 时间统计信息，发出警告并返回
    TORCH_WARN_ONCE(
        "Cuda time stats are not collected for multi-device modules.");
    return;
  }

  // 如果计时器不存在且参数既不是 CUDA 也不是 CPU 设备，则发出警告
  if (!reducer_->timer_ &&
      (!reducer_->params_[0].is_cuda() && !reducer_->params_[0].is_cpu())) {
    TORCH_WARN_ONCE(
        "Time stats are currently only collected for CPU and CUDA devices. "
        "Please refer to CpuTimer or CudaTimer for how to register timer "
        "for other device type.");
  }
    return;
  }
  // 断言确保 reducer_ 的计时器已经初始化
  TORCH_INTERNAL_ASSERT(reducer_->timer_);
  // 计算前向传播平均时间
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_forward_compute_time"],  // 平均前向计算时间的存储位置
      ddp_logging_data_->ints_map["forward_compute_time"],      // 当前前向计算时间
      *reducer_->timer_,                                        // 计时器对象的引用
      Timer::Event::kForwardStart,                               // 前向传播开始事件
      Timer::Event::kBackwardComputeStart);                      // 后向计算开始事件
  // 计算后向传播平均时间
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_backward_compute_time"],  // 平均后向计算时间的存储位置
      ddp_logging_data_->ints_map["backward_compute_time"],      // 当前后向计算时间
      *reducer_->timer_,                                        // 计时器对象的引用
      Timer::Event::kBackwardComputeStart,                       // 后向计算开始事件
      Timer::Event::kBackwardComputeEnd);                        // 后向计算结束事件
  // 计算后向通信平均时间
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_backward_comm_time"],     // 平均后向通信时间的存储位置
      ddp_logging_data_->ints_map["backward_comm_time"],         // 当前后向通信时间
      *reducer_->timer_,                                        // 计时器对象的引用
      Timer::Event::kBackwardCommStart,                          // 后向通信开始事件
      Timer::Event::kBackwardCommEnd);                           // 后向通信结束事件
  // 计算后向计算与通信重叠时间平均值
  calculate_avg_time(
      ddp_logging_data_->ints_map["avg_backward_compute_comm_overlap_time"],  // 平均后向计算与通信重叠时间的存储位置
      ddp_logging_data_->ints_map["backward_compute_comm_overlap_time"],      // 当前后向计算与通信重叠时间
      *reducer_->timer_,                                                      // 计时器对象的引用
      Timer::Event::kBackwardCommStart,                                        // 后向通信开始事件
      Timer::Event::kBackwardComputeEnd);                                      // 后向计算结束事件

  // 设置前向计算时间起始点
  set_event_time(
      ddp_logging_data_->ints_map["forward_compute_time_start"],  // 前向计算时间起始点存储位置
      *reducer_->timer_,                                          // 计时器对象的引用
      Timer::Event::kForwardStart);                                // 前向计算开始事件
  // 设置后向计算时间起始点
  set_event_time(
      ddp_logging_data_->ints_map["backward_compute_time_start"],  // 后向计算时间起始点存储位置
      *reducer_->timer_,                                          // 计时器对象的引用
      Timer::Event::kBackwardComputeStart);                        // 后向计算开始事件
  // 设置后向通信时间起始点
  set_event_time(
      ddp_logging_data_->ints_map["backward_comm_time_start"],     // 后向通信时间起始点存储位置
      *reducer_->timer_,                                          // 计时器对象的引用
      Timer::Event::kBackwardCommStart);                           // 后向通信开始事件
  // 设置后向计算时间结束点
  set_event_time(
      ddp_logging_data_->ints_map["backward_compute_time_end"],    // 后向计算时间结束点存储位置
      *reducer_->timer_,                                          // 计时器对象的引用
      Timer::Event::kBackwardComputeEnd);                          // 后向计算结束事件
  // 设置后向通信时间结束点
  set_event_time(
      ddp_logging_data_->ints_map["backward_comm_time_end"],       // 后向通信时间结束点存储位置
      *reducer_->timer_,                                          // 计时器对象的引用
      Timer::Event::kBackwardCommEnd);                             // 后向通信结束事件

  // 如果启用了 TORCH_DISTRIBUTED_DEBUG=DETAIL，则将运行时统计信息记录到 stderr
  if (debug_level() == DebugLevel::Detail) {
    LOG(INFO) << *this;
  }

  // 在特定迭代次数记录运行时（例如平均性能）统计信息。
  // 这里选择的迭代次数（10/1000/10000）并不科学，假设大多数应用程序至少运行10次迭代。
  // 如果选择的 num_iterations_ 更大，统计数据的方差可能较小。
  if (std::find(
          std::begin(LoggingIterations),
          std::end(LoggingIterations),
          num_iterations_stats_recorded_) != std::end(LoggingIterations)) {
    // 记录 PyTorch DDP 使用情况日志
    at::LogPyTorchDDPUsage(*ddp_logging_data_);
  }
}

// 定义 Logger 类的成员函数 get_ddp_logging_data()
at::DDPLoggingData Logger::get_ddp_logging_data() {
  // 使用 mutex_ 对象进行互斥保护
  std::lock_guard<std::mutex> lock(reducer_->mutex_);
  // 返回 ddp_logging_data_ 指针指向的对象
  return *ddp_logging_data_;
}

// 初始化 C10dLogger 的静态变量
std::unique_ptr<C10dLogger> C10dLogger::logger_ = nullptr;
std::atomic<bool> C10dLogger::registered_(false);

// 获取 C10dLogger 单例的静态成员函数
C10dLogger* C10dLogger::getLogger() {
  // 若 registered_ 尚未加载，则返回空指针
  if (!registered_.load()) {
    return nullptr;
  }
  // 返回 logger_ 指针指向的对象
  return logger_.get();
}

// 注册 C10dLogger 单例的静态成员函数
void C10dLogger::registerLogger(std::unique_ptr<C10dLogger> logger) {
  // 若已注册，则记录警告信息并返回
  if (registered_.load()) {
    LOG(WARNING) << "C10dLogger has already been registered.";
    return;
  }
  // 将 registered_ 设置为 true，表示已注册
  registered_.store(true);
  // 移动传入的 logger 到 logger_ 静态变量中
  logger_ = std::move(logger);
}

// 记录日志的成员函数，接收 C10dLoggingData 对象作为参数
void C10dLogger::log(const C10dLoggingData& data) {
  // 遍历整数类型的数据并输出日志
  for (const auto& [key, value] : data.integers) {
    LOG(INFO) << key << ": " << value;
  }
  // 遍历字符串类型的数据并输出日志
  for (const auto& [key, value] : data.strings) {
    LOG(INFO) << key << ": " << value;
  }
  // 返回
  return;
}
} // namespace c10d
```