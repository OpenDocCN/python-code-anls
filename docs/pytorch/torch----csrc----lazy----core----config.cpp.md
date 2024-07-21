# `.\pytorch\torch\csrc\lazy\core\config.cpp`

```
// 定义一个名为 torch_lazy_ir_debug 的布尔类型全局变量，默认为 false，用于控制懒惰张量 IR 的调试功能开关
C10_DEFINE_bool(torch_lazy_ir_debug, false, "Enable lazy tensor IR debugging");

// 定义一个名为 torch_lazy_param_aliasing 的布尔类型全局变量，默认为 true，用于控制参数别名支持的开关
C10_DEFINE_bool(
    torch_lazy_param_aliasing,
    true,
    "Enable parameter aliasing support");

// 定义一个名为 torch_lazy_handle_special_scalars 的布尔类型全局变量，默认为 false，用于控制特殊标量值 0 和 1 的特殊处理开关
C10_DEFINE_bool(
    torch_lazy_handle_special_scalars,
    false,
    "Handle special scalars 0 and 1 differently");

// 定义一个名为 torch_lazy_all_numbers_special_scalars 的布尔类型全局变量，默认为 false，用于控制将所有数字视为特殊标量的开关
C10_DEFINE_bool(
    torch_lazy_all_numbers_special_scalars,
    false,
    "Handle all numbers as special scalars");

// 定义一个名为 torch_lazy_reuse_ir 的布尔类型全局变量，默认为 false，用于控制是否尝试重用先前追踪的 IR 节点
C10_DEFINE_bool(
    torch_lazy_reuse_ir,
    false,
    "Reuse IR nodes from previous tracing when possible");

// 定义一个名为 torch_lazy_use_thread_pool 的布尔类型全局变量，默认为 false，用于控制是否使用线程池调度后端执行
C10_DEFINE_bool(
    torch_lazy_use_thread_pool,
    false,
    "Use thread pool to schedule backend execution");

// 定义一个名为 torch_lazy_enable_device_data_cache 的布尔类型全局变量，默认为 true，用于控制设备数据缓存的启用或禁用
C10_DEFINE_bool(
    torch_lazy_enable_device_data_cache,
    true,
    "Enable or disable device data cache (turns cache on or off), does not change cache state");

// 定义一个名为 torch_lazy_compilation_cache_size 的整数类型全局变量，默认为 1024，用于设置编译缓存的大小
C10_DEFINE_int(
    torch_lazy_compilation_cache_size,
    1024,
    "Size of the compilation cache");

// 定义一个名为 torch_lazy_device_data_cache_size 的整数类型全局变量，默认为 128，用于设置设备数据缓存的大小
C10_DEFINE_int(
    torch_lazy_device_data_cache_size,
    128,
    "Size of the DeviceData cache");

// 定义一个名为 torch_lazy_io_thread_pool_size 的整数类型全局变量，默认为 1，用于设置执行线程池的大小
C10_DEFINE_int(
    torch_lazy_io_thread_pool_size,
    // TODO: measure which default value will give better
    // performance, std::thread::hardware_concurrency()?
    1,
    "Size of the execution thread pool");

// 定义一个名为 torch_lazy_metrics_samples 的整数类型全局变量，默认为 1024，用于设置最大度量样本大小
C10_DEFINE_int(torch_lazy_metrics_samples, 1024, "Max metrics sample size");

// 定义一个名为 torch_lazy_trim_graph_check_frequency 的整数类型全局变量，默认为 5000，用于设置检查图是否需要拆分的频率
C10_DEFINE_int(
    torch_lazy_trim_graph_check_frequency,
    5000,
    "How often to check for whether a graph needs to be split");

// 定义一个名为 torch_lazy_trim_graph_size 的整数类型全局变量，默认为 100000，用于设置图拆分的节点数阈值
C10_DEFINE_int(
    torch_lazy_trim_graph_size,
    100000,
    "The threshold (in terms of the number of nodes) for splitting a graph");

// 定义一个名为 torch_lazy_metrics_percentiles 的字符串类型全局变量，默认为 "0.01:0.05:0.1:0.2:0.5:0.8:0.9:0.95:0.99"，用于设置要收集的度量百分位数
C10_DEFINE_string(
    torch_lazy_metrics_percentiles,
    "0.01:0.05:0.1:0.2:0.5:0.8:0.9:0.95:0.99",
    "Metrics percentiles to be collected, using : as the delimiter");

// 定义一个名为 torch_lazy_shape_cache_size 的整数类型全局变量，默认为 4096，用于设置形状推断中使用的形状缓存大小
C10_DEFINE_int(
    torch_lazy_shape_cache_size,
    4096,
    "Set the size for the shape cache used for shape inference");

// torch 命名空间下的 lazy 子命名空间，包含一个函数 getLTCForceFallback，用于获取强制回退的配置值
namespace torch {
namespace lazy {

// 定义一个静态局部变量 config，存储通过环境变量 LTC_FORCE_FALLBACK 获取的值
std::string& getLTCForceFallback() {
  static std::string config;
  // 静态局部变量 _ignore，用于确保只在第一次调用时执行环境变量的读取操作
  static bool _ignore = [&]() {
    char* envptr = std::getenv("LTC_FORCE_FALLBACK");
    if (envptr) {
      config = std::string(envptr);
    }
    return true;
  }();
  // 避免未使用变量警告
  (void)_ignore; // avoid unused variables warning
  // 返回 config 变量的引用
  return config;
}

} // namespace lazy
} // namespace torch
```