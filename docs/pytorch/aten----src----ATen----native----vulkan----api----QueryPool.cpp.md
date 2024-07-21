# `.\pytorch\aten\src\ATen\native\vulkan\api\QueryPool.cpp`

```
// 引入Vulkan的查询池相关头文件
#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/Utils.h>

// 如果定义了USE_KINETO，则引入Kineto性能分析相关头文件
#ifdef USE_KINETO
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/orchestration/vulkan.h>
#endif // USE_KINETO

// 引入标准库头文件
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

// ATen命名空间下的Vulkan相关API
namespace at {
namespace native {
namespace vulkan {
namespace api {

// 匿名命名空间，定义默认的时间戳周期
namespace {
constexpr int64_t kDefaultNsPerTick = 52; // lround(52.08f);
} // namespace

// QueryPool类的构造函数定义
QueryPool::QueryPool(const QueryPoolConfig& config, const Adapter* adapter_p)
    : mutex_{}, // 初始化互斥锁
      device_(adapter_p->device_handle()), // 使用适配器的设备句柄初始化设备
      config_(config), // 使用传入的配置初始化config_
      querypool_(VK_NULL_HANDLE), // 初始化查询池句柄为VK_NULL_HANDLE
      shader_logs_(1), // 初始化着色器日志，初始大小为1
      in_use_(0), // 初始化查询池中正在使用的查询数为0
      previous_shader_count_(0u), // 初始化先前的着色器计数为0
      results_pending_(false) { // 初始化结果未决标志为false

  // 定义Vulkan的查询池创建信息
  const VkQueryPoolCreateInfo info{
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, // 结构类型
      nullptr, // 扩展信息
      0u, // 标志
      VK_QUERY_TYPE_TIMESTAMP, // 查询类型为时间戳
      config_.maxQueryCount, // 最大查询数
      0u, // 管道统计信息
  };

  // 创建Vulkan查询池，并进行错误检查
  VK_CHECK(vkCreateQueryPool(device_, &info, nullptr, &querypool_));

  // 预留初始大小的着色器日志空间
  shader_log().reserve(config_.initialReserveSize);

  // 如果adapter_p为空，输出错误信息
  VK_CHECK_COND(adapter_p, "Valid GPU device must be created for QueryPool");

  // 获取GPU的时间戳周期，如果为0则使用默认值kDefaultNsPerTick
  ns_per_tick_ = std::lround(adapter_p->timestamp_period());
  ns_per_tick_ = (ns_per_tick_ == 0) ? kDefaultNsPerTick : ns_per_tick_;

  // 如果定义了USE_KINETO，则注册获取着色器名称和执行时间的回调函数
#ifdef USE_KINETO
  torch::profiler::impl::vulkan::registerGetShaderNameAndDurationNs(
      [this](int64_t vulkan_id) {
        return get_shader_name_and_execution_duration_ns(vulkan_id);
      });
#endif // USE_KINETO
}

// QueryPool类的析构函数定义
QueryPool::~QueryPool() {
  // 如果查询池句柄为VK_NULL_HANDLE，直接返回
  if (VK_NULL_HANDLE == querypool_) {
    return;
  }
  
  // 销毁Vulkan查询池
  vkDestroyQueryPool(device_, querypool_, nullptr);

  // 如果定义了USE_KINETO，则取消注册获取着色器名称和执行时间的回调函数
#ifdef USE_KINETO
  torch::profiler::impl::vulkan::deregisterGetShaderNameAndDurationNs();
#endif // USE_KINETO
}

// 重置查询池状态的方法定义
void QueryPool::reset(const CommandBuffer& cmd) {
  // 使用互斥锁保护操作
  std::lock_guard<std::mutex> lock(mutex_);

  // 重置命令缓冲中的查询池
  cmd.reset_querypool(querypool_, 0u, in_use_);

  // 将先前的着色器日志大小添加到先前的着色器计数中
  previous_shader_count_ += shader_log().size();

  // 重置正在使用的查询数为0
  in_use_ = 0u;

  // 添加新的着色器日志条目
  shader_logs_.emplace_back();

  // 预留新的初始大小的着色器日志空间
  shader_log().reserve(config_.initialReserveSize);

  // 标记结果未决为false
  results_pending_ = false;
}

// 写入时间戳的方法定义
size_t QueryPool::write_timestamp(const CommandBuffer& cmd) {
  // 检查正在使用的查询数是否超过了最大允许的查询数
  VK_CHECK_COND(
      in_use_ < config_.maxQueryCount,
      "Vulkan QueryPool: Exceeded the maximum number of queries "
      "allowed by the queryPool (",
      config_.maxQueryCount,
      ")!");

  // 在命令缓冲中写入时间戳
  cmd.write_timestamp(querypool_, in_use_);

  // 返回当前写入的查询索引，并增加正在使用的查询数
  return in_use_++;
}

// 开始着色器性能分析的方法定义
uint32_t QueryPool::shader_profile_begin(
    const CommandBuffer& cmd,
    const std::string& kernel_name,
    const VkExtent3D global_workgroup_size,
    // 使用互斥锁保护临界区，确保线程安全操作
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 获取当前时间戳的查询索引
    uint32_t query_idx = write_timestamp(cmd);
    
    // 记录着色器日志的起始索引
    uint32_t log_idx = shader_log().size();
    
    // 创建着色器执行时长日志条目
    ShaderDuration log_entry{
        log_idx,
        // 执行属性
        kernel_name,             // 内核名称
        global_workgroup_size,   // 全局工作组大小
        local_workgroup_size,    // 局部工作组大小
        // 查询索引
        query_idx,               // 起始查询索引
        UINT32_MAX,              // 终止查询索引
        // 时间统计
        0u,                      // 开始时间
        0u,                      // 结束时间
        0u,                      // 持续时间
    };
    
    // 将日志条目添加到着色器日志中
    shader_log().emplace_back(log_entry);
    
    // 标记仍有结果待处理
    results_pending_ = true;
#ifdef USE_KINETO
  // 创建一个 Vulkan ID，用于表示当前的着色器日志索引
  torch::profiler::impl::vulkan_id_t vulkan_id =
      torch::profiler::impl::vulkan_id_t(previous_shader_count_ + log_idx);

  // 报告 Vulkan 事件给性能分析器
  torch::profiler::impl::_reportVulkanEventToProfiler(vulkan_id);
#endif // USE_KINETO

// 返回当前着色器日志索引
return log_idx;
}

void QueryPool::shader_profile_end(
    const CommandBuffer& cmd,
    const uint32_t log_idx) {
  // 加锁，确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);

  // 写入时间戳，并获取查询索引
  size_t query_idx = write_timestamp(cmd);

  // 设置着色器日志中当前索引的结束查询索引
  shader_log()[log_idx].end_query_idx = query_idx;
}

void QueryPool::extract_results() {
  // 加锁，确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);

  // 如果没有待处理的结果，直接返回
  if (!results_pending_) {
    return;
  }

  // 定义 Vulkan 查询结果标志
  const VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT;

  // 创建用于存储查询数据的向量，并设置大小
  std::vector<uint64_t> query_data;
  query_data.resize(in_use_);

  // 获取查询池的结果数据
  VK_CHECK(vkGetQueryPoolResults(
      device_,
      querypool_,
      0u, // firstQuery
      in_use_, // queryCount
      sizeof(uint64_t) * in_use_, // dataSize
      query_data.data(), // pData
      sizeof(uint64_t), // stride
      flags)); // flags

  // 遍历着色器执行日志，并根据查询数据设置时间戳
  for (ShaderDuration& entry : shader_log()) {
    entry.start_time_ns = query_data.at(entry.start_query_idx) * ns_per_tick_;
    entry.end_time_ns = query_data.at(entry.end_query_idx) * ns_per_tick_;
    entry.execution_duration_ns = entry.end_time_ns - entry.start_time_ns;
  }

  // 标记结果已处理完毕
  results_pending_ = false;
}

// 输出 VkExtent3D 结构体的流输出运算符重载
std::ostream& operator<<(std::ostream& os, const VkExtent3D& extents) {
  os << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return os;
}

// 将 VkExtent3D 结构体转换为字符串表示
std::string stringize(const VkExtent3D& extents) {
  std::stringstream ss;
  ss << "{" << extents.width << ", " << extents.height << ", " << extents.depth
     << "}";
  return ss.str();
}

// 生成着色器执行报告的字符串表示
std::string QueryPool::generate_string_report() {
  // 加锁，确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);

  // 创建字符串流
  std::stringstream ss;

  // 设置列的宽度
  int kernel_name_w = 40;
  int global_size_w = 15;
  int duration_w = 25;

  // 设置左对齐输出
  ss << std::left;
  ss << std::setw(kernel_name_w) << "Kernel Name";
  ss << std::setw(global_size_w) << "Workgroup Size";
  ss << std::right << std::setw(duration_w) << "Duration (ns)";
  ss << std::endl;

  // 输出列标题的分隔线
  ss << std::left;
  ss << std::setw(kernel_name_w) << "===========";
  ss << std::setw(global_size_w) << "==============";
  ss << std::right << std::setw(duration_w) << "===========";
  ss << std::endl;

  // 遍历着色器执行日志，生成每个条目的报告
  for (ShaderDuration& entry : shader_log()) {
    // 转换执行时间为纳秒
    std::chrono::duration<size_t, std::nano> exec_duration_ns(
        entry.execution_duration_ns);

    // 输出每个着色器执行条目的详细信息
    ss << std::left;
    ss << std::setw(kernel_name_w) << entry.kernel_name;
    ss << std::setw(global_size_w) << stringize(entry.global_workgroup_size);
    ss << std::right << std::setw(duration_w) << exec_duration_ns.count();
    ss << std::endl;
  }

  // 返回生成的报告字符串
  return ss.str();
}

// 打印着色器执行报告
void QueryPool::print_results() {
  std::cout << generate_string_report() << std::endl;
}

// 获取指定操作名称的总操作时间（纳秒）
uint64_t QueryPool::get_total_op_ns(const std::string& op_name) {
  // 加锁，确保线程安全
  std::lock_guard<std::mutex> lock(mutex_);
  uint64_t sum = 0;

  // 遍历着色器执行日志，累加指定操作的执行时间
  for (ShaderDuration& entry : shader_log()) {
    # 如果当前迭代的 entry 的 kernel_name 属性与 op_name 相等，则执行以下操作
    if (entry.kernel_name == op_name) {
      # 将 entry 的执行持续时间加到 sum 变量中
      sum += entry.execution_duration_ns;
    }
  }
  # 返回 sum 变量作为结果
  return sum;
}

// 定义了 QueryPool 类的成员函数 shader_log_for_each，用于对 shader_log 中的每个条目执行给定函数 fn
void QueryPool::shader_log_for_each(
    std::function<void(const ShaderDuration&)> fn) {
  // 使用 mutex 实现线程安全，锁定互斥量
  std::lock_guard<std::mutex> lock(mutex_);
  // 对 shader_log 中的每个条目应用函数 fn
  std::for_each(shader_log().begin(), shader_log().end(), std::move(fn));
}

// 定义了 QueryPool 类的成员函数 get_shader_name_and_execution_duration_ns，返回查询索引处的着色器名称和执行时长
std::tuple<std::string, uint64_t> QueryPool::
    get_shader_name_and_execution_duration_ns(size_t query_index) {
  // 提取结果（可能更新内部状态）
  extract_results();

  // 使用 mutex 实现线程安全，锁定互斥量
  std::lock_guard<std::mutex> lock(mutex_);

  // 获取线程不安全的 shader_logs 条目数
  const size_t entry_count = shader_logs_entry_count_thread_unsafe();
  
  // 检查 query_index 是否有效，若无效则输出错误信息并终止程序
  VK_CHECK_COND(
      (query_index >= 0 && query_index < entry_count),
      "query_index of ",
      query_index,
      " is out of bounds (",
      entry_count,
      ") in QueryPool::get_shader_name_and_duration_ns");

  // 确定 shader_logs 中的具体索引和条目位置
  size_t log_idx = 0;
  size_t entry_count_acc = 0;
  while (entry_count_acc + shader_logs_[log_idx].size() <= query_index) {
    entry_count_acc += shader_logs_[log_idx].size();
    log_idx += 1;
  }

  // 获取指定位置的 ShaderDuration 条目
  const ShaderDuration& entry =
      shader_logs_[log_idx][query_index - entry_count_acc];

  // 返回包含着色器名称和执行时长的元组
  return std::tuple<std::string, uint64_t>(
      entry.kernel_name, entry.execution_duration_ns);
}

// 定义了 QueryPool 类的成员函数 shader_logs_entry_count_thread_unsafe，返回 shader_logs 条目数（线程不安全）
size_t QueryPool::shader_logs_entry_count_thread_unsafe() {
  // 计算并返回前一次的 shader 数量加上当前 shader_logs 的条目数
  return previous_shader_count_ + shader_log().size();
}

// 定义了 QueryPool 类的成员函数 shader_logs_entry_count，返回 shader_logs 条目数（线程安全）
size_t QueryPool::shader_logs_entry_count() {
  // 使用 mutex 实现线程安全，锁定互斥量
  std::lock_guard<std::mutex> lock(mutex_);
  // 调用线程不安全的函数获取 shader_logs 条目数并返回
  return shader_logs_entry_count_thread_unsafe();
}

// 结束了命名空间声明，代码块结束
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
```