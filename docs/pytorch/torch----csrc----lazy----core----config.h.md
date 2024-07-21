# `.\pytorch\torch\csrc\lazy\core\config.h`

```py
#pragma once
// C10 库中的宏定义，用于导出符号
#include <c10/macros/Export.h>
// C10 库中的 Flags.h，提供了声明和定义 C10 库中使用的标志的工具

// 声明一系列的布尔类型的标志，用于配置懒执行模式下的调试和优化行为
C10_DECLARE_bool(torch_lazy_ir_debug);
C10_DECLARE_bool(torch_lazy_handle_special_scalars);
C10_DECLARE_bool(torch_lazy_all_numbers_special_scalars);
C10_DECLARE_bool(torch_lazy_param_aliasing);
C10_DECLARE_bool(torch_lazy_reuse_ir);
C10_DECLARE_bool(torch_lazy_use_thread_pool);
C10_DECLARE_bool(torch_lazy_enable_device_data_cache);

// 声明一系列的整数类型的标志，用于配置懒执行模式下的各种缓存和度量参数
C10_DECLARE_int(torch_lazy_compilation_cache_size);
C10_DECLARE_int(torch_lazy_device_data_cache_size);
C10_DECLARE_int(torch_lazy_io_thread_pool_size);
C10_DECLARE_int(torch_lazy_metrics_samples);
C10_DECLARE_int(torch_lazy_trim_graph_check_frequency);
C10_DECLARE_int(torch_lazy_trim_graph_size);

// 声明一个字符串类型的标志，用于配置懒执行模式下的度量百分位数
C10_DECLARE_string(torch_lazy_metrics_percentiles);

// 声明一个整数类型的标志，用于配置懒执行模式下的形状缓存大小
C10_DECLARE_int(torch_lazy_shape_cache_size);

// 声明一个函数，用于获取强制回退的 LTC（Long-Term Compilation）字符串
namespace torch {
namespace lazy {
// 在 torch::lazy 命名空间下，导出 API，提供获取 LTC 强制回退字符串的功能
TORCH_API std::string& getLTCForceFallback();
}
} // namespace torch
```