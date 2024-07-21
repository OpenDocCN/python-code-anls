# `.\pytorch\torch\csrc\profiler\kineto_shim.cpp`

```py
namespace torch {

namespace profiler::impl::kineto {

// Here lies pain and `#ifdef USE_KINETO`

#ifdef USE_KINETO
namespace {
// 定义包含不同类型的活动集合，用于标识 CPU 活动
const std::set<libkineto::ActivityType> kCpuTypes{
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::XPU_RUNTIME,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CUDA_DRIVER,
    libkineto::ActivityType::PYTHON_FUNCTION,
    libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
    libkineto::ActivityType::PRIVATEUSE1_DRIVER,
};

// 定义包含不同类型的活动集合，用于标识 CUDA 活动
const std::set<libkineto::ActivityType> kCudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::GPU_USER_ANNOTATION,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // CUDA_RUNTIME appears in both kCpuTypes and kCudaTypes.
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::CUDA_DRIVER,
};

// 定义包含不同类型的活动集合，用于标识 XPU 活动
const std::set<libkineto::ActivityType> kXpuTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // XPU_RUNTIME appears in both kCpuTypes and kXpuTypes.
    libkineto::ActivityType::XPU_RUNTIME,
};

// 定义包含不同类型的活动集合，用于标识 MTIA 活动
const std::set<libkineto::ActivityType> kMtiaTypes = {
    libkineto::ActivityType::MTIA_CCP_EVENTS,
    libkineto::ActivityType::MTIA_RUNTIME,
    libkineto::ActivityType::MTIA_WORKLOADD,
};

// 定义包含不同类型的活动集合，用于标识 PRIVATEUSE1 活动
const std::set<libkineto::ActivityType> kPrivateUse1Types = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::GPU_USER_ANNOTATION,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // PRIVATEUSE1_RUNTIME appears in both kCpuTypes and kPrivateUse1Types.
    libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
    libkineto::ActivityType::PRIVATEUSE1_DRIVER,
};
} // namespace
#endif // USE_KINETO

// 断言检查 DeviceAndResource 类型是否为 POD 类型
static_assert(
    c10::is_pod_v<DeviceAndResource>,
    "Kineto specific details should be in `kineto_ids`.");

// 获取 Kineto IDs，如果未定义 USE_KINETO 则返回空对象
const DeviceAndResource kineto_ids() {
#ifdef USE_KINETO
  return {
      /*device=*/libkineto::processId(),
      /*resource=*/libkineto::systemThreadId()};
#else
  return {};
#endif // USE_KINETO
}

// 为活动添加元数据
void addMetadata(
    activity_t* activity,
    const std::string& key,
    const std::string& value) {
#ifdef USE_KINETO
  activity->addMetadata(key, value);
#endif // USE_KINETO
}

// TraceWrapper 类的构造函数
TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
#ifdef USE_KINETO
    : cpu_trace_(std::make_unique<libkineto::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gpuOpCount = -1;
  cpu_trace_->span.name = name;
}
#else
{
}
#endif // USE_KINETO

// TraceWrapper 类的析构函数
TraceWrapper::~TraceWrapper() = default;

} // namespace profiler::impl::kineto

} // namespace torch
// 添加 CPU 活动到跟踪中，如果未启用 Kineto 则返回 nullptr
activity_t* TraceWrapper::addCPUActivity(
    const std::string& name,                           // 活动的名称
    const libkineto::ActivityType type,                // 活动的类型
    const DeviceAndResource device_and_resource,       // 设备和资源信息
    const uint64_t correlation_id,                     // 相关 ID
    const int64_t start_time,                          // 活动开始时间
    const int64_t end_time) {                          // 活动结束时间
#ifdef USE_KINETO
  TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");  // 检查是否存在有效的跟踪
  cpu_trace_->emplace_activity(cpu_trace_->span, type, name);              // 将活动添加到 CPU 跟踪中
  auto& act = libkineto::CpuTraceBuffer::toRef(cpu_trace_->activities.back());  // 获取最后添加的活动的引用
  act.device = device_and_resource.device;            // 设置活动所属的设备
  act.resource = device_and_resource.resource;        // 设置活动使用的资源
  act.id = static_cast<int32_t>(correlation_id);      // 设置活动的相关 ID
  act.startTime = start_time;                         // 设置活动的开始时间
  if (type != libkineto::ActivityType::CPU_INSTANT_EVENT) {  // 如果活动类型不是瞬时事件
    act.endTime = end_time;                           // 设置活动的结束时间
  }
  return cpu_trace_->activities.back().get();         // 返回最后添加的活动的指针
#else
  return nullptr;                                     // 如果未启用 Kineto，则返回空指针
#endif // USE_KINETO
}

// 转移 CPU 跟踪数据到活动分析器中
void TraceWrapper::transferCpuTrace(int64_t end_time) {
#ifdef USE_KINETO
  cpu_trace_->span.endTime = end_time;                // 设置 CPU 跟踪的结束时间
  libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace_));  // 转移 CPU 跟踪数据
#endif // USE_KINETO
}

// 判断跟踪对象是否有效
TraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return cpu_trace_ != nullptr;                       // 返回 CPU 跟踪对象是否不为空
#else
  return false;                                       // 如果未启用 Kineto，则返回 false
#endif // USE_KINETO
}

// 构造函数，初始化活动跟踪包装器
ActivityTraceWrapper::ActivityTraceWrapper(
    std::unique_ptr<interface_trace_t>&& trace)
    : trace_(std::move(trace)) {}

// 判断活动跟踪是否有效
ActivityTraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return trace_ != nullptr;                           // 返回活动跟踪对象是否不为空
#else
  return false;                                       // 如果未启用 Kineto，则返回 false
#endif // USE_KINETO
}

// 保存活动跟踪数据到指定路径
void ActivityTraceWrapper::save(const std::string& path) {
#ifdef USE_KINETO
  TORCH_CHECK(!saved_, "Trace is already saved.");    // 检查是否已经保存过跟踪数据
  TORCH_CHECK(trace_ != nullptr, "Missing trace.")   // 检查是否存在有效的跟踪数据
  trace_->save(path);                                 // 将跟踪数据保存到指定路径
  saved_ = true;                                      // 标记跟踪数据已保存
#else
  TORCH_CHECK(
      false,
      "Saving a trace requires using torch.profiler with Kineto support (USE_KINETO=1)");  // 如果未启用 Kineto，则抛出错误
#endif // USE_KINETO
}

// 处理 Kineto 的实验性配置选项
class ExperimentalConfigWrapper {
 public:
  explicit ExperimentalConfigWrapper(
      const torch::profiler::impl::ExperimentalConfig& config)
      : config_(config) {}

  // 断言实验配置是否有效
  bool assertValid() {
    return !config_.profiler_metrics.empty();         // 返回是否存在配置指标
  }

  // 使用实验性配置选项准备跟踪
  void prepareTraceWithExperimentalOptions(bool add_cpu_activity) {
#ifdef USE_KINETO
    std::set<libkineto::ActivityType> k_activities{
        libkineto::ActivityType::CUDA_PROFILER_RANGE};

    // 只有在测量每个内核范围时才添加 CPU 活动
    if (add_cpu_activity && config_.profiler_measure_per_kernel) {
      k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());  // 插入 CPU 活动类型集合
    }

    const size_t num_metrics = config_.profiler_metrics.size();  // 获取配置的指标数量
    std::stringstream configss;

    LOG(INFO) << "CUPTI profiler metrics size = " << num_metrics;  // 记录配置的指标数量日志

    // 构建配置字符串
    configss << "ACTIVITIES_WARMUP_PERIOD_SECS=0\n"
             << "CUPTI_PROFILER_METRICS=";

    // 将配置的指标加入到配置字符串中
    for (size_t i = 0; i < num_metrics; i++) {
      configss << config_.profiler_metrics[i];
      if (num_metrics > 1 && i < (num_metrics - 1)) {
        configss << ",";
      }
    }
#endif // USE_KINETO
}
    // 将 CUPTI_PROFILER_ENABLE_PER_KERNEL 的配置值（布尔类型）添加到 configss 字符串流中
    configss << "\nCUPTI_PROFILER_ENABLE_PER_KERNEL="
             << (config_.profiler_measure_per_kernel ? "true" : "false")
             << "\n";
    // 记录生成的配置信息到日志中
    LOG(INFO) << "Generated config = " << configss.str();

    // 准备追踪活动数据，使用配置字符串作为参数
    libkineto::api().activityProfiler().prepareTrace(
        k_activities, configss.str());
void prepareTrace(
    const bool cpuOnly,
    const ActivitySet& activities,
    const torch::profiler::impl::ExperimentalConfig& config) {
#ifdef USE_KINETO
  // 检查是否注册了 Kineto 分析器，若未注册则初始化
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  // 如果 Kineto 分析器未初始化，则初始化之
  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  // 创建一个用于存储 Kineto 活动类型的集合
  std::set<libkineto::ActivityType> k_activities;

  // 检查是否存在 CPU 活动
  bool has_cpu_activity =
      activities.count(torch::autograd::profiler::ActivityType::CPU);

  // 如果存在 CPU 活动，将其添加到活动类型集合中
  if (has_cpu_activity) {
    k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
  }

  // 如果存在 XPU 活动，将其添加到活动类型集合中
  if (activities.count(torch::autograd::profiler::ActivityType::XPU)) {
    k_activities.insert(kXpuTypes.begin(), kXpuTypes.end());
  }

  // 如果存在 MTIA 活动，将其添加到活动类型集合中
  if (activities.count(torch::autograd::profiler::ActivityType::MTIA)) {
    k_activities.insert(kMtiaTypes.begin(), kMtiaTypes.end());
  }

  // 如果存在 CUDA 活动，将其添加到活动类型集合中
  if (activities.count(torch::autograd::profiler::ActivityType::CUDA)) {
    k_activities.insert(kCudaTypes.begin(), kCudaTypes.end());

    // 如果配置允许 CUDA 同步事件或已启用 CUDA 同步事件，记录信息并添加 CUDA 同步活动类型
    if (config.enable_cuda_sync_events || get_cuda_sync_enabled()) {
      LOG(INFO) << "Enabling CUDA Sync Events";
      k_activities.insert(libkineto::ActivityType::CUDA_SYNC);
    }
  }

  // 如果存在集体通信分析器，添加集体通信活动类型
  if (collectivesProfilerExists()) {
    k_activities.insert(libkineto::ActivityType::COLLECTIVE_COMM);
  }

  // 如果存在 PrivateUse1 活动，将其添加到活动类型集合中
  if (activities.count(torch::autograd::profiler::ActivityType::PrivateUse1)) {
    k_activities.insert(kPrivateUse1Types.begin(), kPrivateUse1Types.end());
  }

  // 使用实验性配置包装器创建配置包装对象
  ExperimentalConfigWrapper configWrap(config);

  // 如果配置有效，根据实验性选项准备跟踪
  if (config && configWrap.assertValid()) {
    configWrap.prepareTraceWithExperimentalOptions(has_cpu_activity);
    return;
  }

  // 使用 Kineto API 准备跟踪活动
  libkineto::api().activityProfiler().prepareTrace(k_activities);
#endif // USE_KINETO
}

void startTrace() {
#ifdef USE_KINETO
  // 使用 Kineto API 开始跟踪活动
  libkineto::api().activityProfiler().startTrace();
#endif // USE_KINETO
}

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{
#ifdef USE_KINETO
      // 使用 Kineto API 停止跟踪活动
      libkineto::api().activityProfiler().stopTrace()
#else
      // 如果未使用 Kineto，返回空的接口跟踪对象
      std::make_unique<interface_trace_t>()
#endif // USE_KINETO
  };
}

void pushCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  // 使用 Kineto API 推送关联 ID
  libkineto::api().activityProfiler().pushCorrelationId(correlation_id);
#endif // USE_KINETO
}

void pushUserCorrelationId(uint64_t correlation_id) {
#ifdef USE_KINETO
  // 使用 Kineto API 推送用户关联 ID
  libkineto::api().activityProfiler().pushUserCorrelationId(correlation_id);
#endif // USE_KINETO
}

void popCorrelationId() {
#ifdef USE_KINETO
  // 使用 Kineto API 弹出关联 ID
  libkineto::api().activityProfiler().popCorrelationId();
#endif // USE_KINETO
}

void popUserCorrelationId() {
#ifdef USE_KINETO
  // 如果定义了 USE_KINETO 宏，则从活动分析器中弹出用户相关 ID
  libkineto::api().activityProfiler().popUserCorrelationId();
#endif // USE_KINETO
}

void recordThreadInfo() {
#ifdef USE_KINETO
  // 如果定义了 USE_KINETO 宏，则记录线程信息到活动分析器中
  libkineto::api().activityProfiler().recordThreadInfo();
#endif // USE_KINETO
}

void logInvariantViolation(
    const std::string& assertion,
    const std::string& error,
    const std::string& profile_id,
    const std::string& group_profile_id) {
#ifdef USE_KINETO
  // 如果定义了 USE_KINETO 宏，并且分析器已初始化，则记录不变性违规日志到活动分析器中
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().logInvariantViolation(
        profile_id, assertion, error, group_profile_id);
  }
#endif // USE_KINETO
}

} // namespace profiler::impl::kineto

namespace autograd::profiler {
c10::DeviceType deviceTypeFromActivity(libkineto::ActivityType activity_type) {
  // fallthrough
  switch (activity_type) {
    case libkineto::ActivityType::GPU_MEMCPY:
    case libkineto::ActivityType::GPU_MEMSET:
    case libkineto::ActivityType::CONCURRENT_KERNEL:
    case libkineto::ActivityType::CUDA_SYNC:
    case libkineto::ActivityType::GPU_USER_ANNOTATION:
    case libkineto::ActivityType::CUDA_PROFILER_RANGE: {
      // 私有使用1的 kineto 后端重用上述 ActivityTypes，
      // 如果启用了私有使用1后端，则应返回 c10::DeviceType::PrivateUse1。
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::CUDA;
      }();
      return device_type;
    }
    // TODO: T151322015
    case libkineto::ActivityType::MTIA_CCP_EVENTS:
    case libkineto::ActivityType::MTIA_WORKLOADD: {
      // 私有使用1的 kineto 后端重用上述 ActivityTypes，
      // 如果启用了私有使用1后端，则应返回 c10::DeviceType::PrivateUse1。
      c10::DeviceType device_type = []() {
        if (c10::get_privateuse1_backend() != "privateuseone") {
          return c10::DeviceType::PrivateUse1;
        }
        return c10::DeviceType::MTIA;
      }();
      return device_type;
    }
    case libkineto::ActivityType::CPU_OP:
    case libkineto::ActivityType::USER_ANNOTATION:
    case libkineto::ActivityType::EXTERNAL_CORRELATION:
    case libkineto::ActivityType::CUDA_RUNTIME:
    case libkineto::ActivityType::XPU_RUNTIME:
    case libkineto::ActivityType::CPU_INSTANT_EVENT:
    case libkineto::ActivityType::GLOW_RUNTIME:
    case libkineto::ActivityType::MTIA_RUNTIME:
    case libkineto::ActivityType::PYTHON_FUNCTION:
    case libkineto::ActivityType::CUDA_DRIVER:
    case libkineto::ActivityType::PRIVATEUSE1_RUNTIME:
    case libkineto::ActivityType::PRIVATEUSE1_DRIVER:
      // 对于其他未列出的 ActivityType，默认返回 CPU 设备类型
      return c10::DeviceType::CPU;
    default: {
      // 如果遇到未知的 ActivityType，记录警告并假设为 CPU 设备类型
      TORCH_WARN(
          "Unknown activity type (",
          (uint8_t)activity_type,
          "), assuming CPU device");
      return c10::DeviceType::CPU;
    }
  }
}

void addMetadataJson(const std::string& key, const std::string& value) {
#ifdef USE_KINETO
  // 检查是否已初始化分析器
  if (libkineto::api().isProfilerInitialized()) {
    // 若已初始化，则向活动分析器添加元数据
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    // 若未初始化，则记录警告信息并跳过添加元数据操作
    LOG(WARNING) << "Profiler is not initialized: skipping profiling metadata";
  }
#else
  // 如果未定义 USE_KINETO，则记录警告信息，指出需要启用 Kineto 支持才能添加分析元数据
  LOG(WARNING) << "Adding profiling metadata requires using "
               << "torch.profiler with Kineto support (USE_KINETO=1)";
#endif // USE_KINETO
}

void profilerStep() {
#ifdef USE_KINETO
  // 如果使用了 Kineto，初始化分析器（如果已注册）
  libkineto::api().initProfilerIfRegistered();

  // 检查是否已初始化分析器
  if (libkineto::api().isProfilerInitialized()) {
    // 若已初始化，则执行分析器的一步操作
    libkineto::api().activityProfiler().step();
  } else {
    // 若未初始化，则记录警告信息并跳过 step() 调用
    VLOG(1) << "Profiler is not initialized: skipping step() invocation";
  }
#endif // USE_KINETO
}

} // namespace autograd::profiler

} // namespace torch
```