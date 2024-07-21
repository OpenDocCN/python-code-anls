# `.\pytorch\aten\src\ATen\native\vulkan\api\QueryPool.h`

```
#pragma once
// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <functional>
#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/vk_api.h>

#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Pipeline.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

// 结构体，定义了查询池的配置信息
struct QueryPoolConfig final {
  uint32_t maxQueryCount;         // 最大查询数量
  uint32_t initialReserveSize;    // 初始预留大小
};

// 结构体，表示着色器执行时间信息
struct ShaderDuration final {
  uint32_t idx;                   // 索引

  // 执行属性
  std::string kernel_name;        // 内核名称
  VkExtent3D global_workgroup_size;   // 全局工作组大小
  VkExtent3D local_workgroup_size;    // 局部工作组大小

  // 查询索引
  uint32_t start_query_idx;       // 起始查询索引
  uint32_t end_query_idx;         // 终止查询索引

  // 时间信息
  uint64_t start_time_ns;         // 开始时间（纳秒）
  uint64_t end_time_ns;           // 结束时间（纳秒）
  uint64_t execution_duration_ns; // 执行时长（纳秒）
};

// 类，表示查询池
class QueryPool final {
 public:
  explicit QueryPool(const QueryPoolConfig&, const Adapter* adapter_p);  // 构造函数，初始化查询池

  QueryPool(const QueryPool&) = delete;  // 删除复制构造函数
  QueryPool& operator=(const QueryPool&) = delete;  // 删除赋值运算符重载

  QueryPool(QueryPool&&) = delete;  // 删除移动构造函数
  QueryPool& operator=(QueryPool&&) = delete;  // 删除移动赋值运算符重载

  ~QueryPool();  // 析构函数，释放资源

 private:
  std::mutex mutex_;  // 互斥锁，用于保护数据访问

  VkDevice device_;   // Vulkan设备句柄
  QueryPoolConfig config_;  // 查询池配置

  VkQueryPool querypool_;  // Vulkan查询池句柄

  std::vector<std::vector<ShaderDuration>> shader_logs_;  // 所有着色器执行时间日志的二维向量
  size_t in_use_;  // 当前正在使用的日志数

  /** Total number of entries in shader logs from before most recent reset */
  size_t previous_shader_count_;  // 最近一次重置前的着色器日志总数

  /**
   * Indicates whether there are new log entries in the shader log since the
   * most recent call to extract_results()
   */
  bool results_pending_;  // 表示自上次提取结果以来是否有新的日志条目

 private:
  // 写入时间戳到查询池中
  size_t write_timestamp(const CommandBuffer&);

  // 生成字符串报告
  std::string generate_string_report();

  /** Most recent shader log since the last time the QueryPool was reset */
  inline std::vector<ShaderDuration>& shader_log() {
    return shader_logs_[shader_logs_.size() - 1];  // 返回最近一次重置后的最新着色器日志
  }

  /** Total number of entries in all shader logs, but without locking mutex */
  size_t shader_logs_entry_count_thread_unsafe();  // 获取所有着色器日志的总数，线程不安全版本

 public:
  // 返回查询池是否启用的状态
  inline bool is_enabled() const {
    return VK_NULL_HANDLE != querypool_;  // 查询池句柄非空表示启用
  }

  // 重置查询池
  void reset(const CommandBuffer&);

  // 开始着色器性能分析
  uint32_t shader_profile_begin(
      const CommandBuffer&,
      const std::string&,
      const VkExtent3D,
      const VkExtent3D);

  // 结束着色器性能分析
  void shader_profile_end(const CommandBuffer&, const uint32_t);

  // 提取查询结果
  void extract_results();

  // 打印查询结果
  void print_results();

  // 获取指定操作名称的总操作时长（纳秒）
  uint64_t get_total_op_ns(const std::string& op_name);

  uint64_t ns_per_tick_;  // 每个时钟周期的纳秒数

  // 遍历着色器日志，对每个条目执行指定操作
  void shader_log_for_each(std::function<void(const ShaderDuration&)> fn);

  /**
   * query_index is what number entry across all of the QueryPool's shader logs
   * is being queried, regardless of resets. This may be different than
   * ShaderDuration's idx field, which is what number entry it is since the last
   * reset before it was added to the shader logs.
   */
  // 根据查询索引获取着色器名称和执行时长（纳秒）
  std::tuple<std::string, uint64_t> get_shader_name_and_execution_duration_ns(
      size_t query_index);

  /** Total number of entries in all shader logs */
  size_t shader_logs_entry_count();  // 获取所有着色器日志的总数
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif // USE_VULKAN_API
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
```