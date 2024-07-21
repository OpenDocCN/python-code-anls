# `.\pytorch\torch\csrc\jit\mobile\compatibility\runtime_compatibility.h`

```py
#pragma once

#include <c10/macros/Export.h>  // 导入 C10 库的导出宏定义
#include <c10/util/Optional.h>  // 导入 C10 库的可选类型工具

#include <memory>               // 导入标准库的内存管理工具
#include <unordered_map>        // 导入标准库的无序映射容器
#include <unordered_set>        // 导入标准库的无序集合容器

namespace torch {
namespace jit {

// 存储运算符元数据的结构体，可用于版本控制
struct OperatorInfo {
  // 运算符 schema 中的参数数量，使用可选类型来表示
  std::optional<int> num_schema_args;
};

// 运行时兼容性信息结构体
struct RuntimeCompatibilityInfo {
  // 支持的最小和最大字节码版本号的配对
  std::pair<uint64_t, uint64_t> min_max_supported_bytecode_version;
  // 运算符信息的无序映射，映射字符串到 OperatorInfo 结构体
  std::unordered_map<std::string, OperatorInfo> operator_info;
  // 支持的类型集合，使用无序集合存储字符串类型
  std::unordered_set<std::string> supported_types;
  // 支持的最小和最大运算符版本号的配对
  std::pair<uint64_t, uint64_t> min_max_supported_opperator_versions;

  // 工厂方法，用于获取 RuntimeCompatibilityInfo 实例
  static TORCH_API RuntimeCompatibilityInfo get();
};

// 获取运行时字节码版本号的 API 函数声明
TORCH_API uint64_t _get_runtime_bytecode_version();

// 获取运行时字节码最小和最大版本号的 API 函数声明
TORCH_API std::pair<uint64_t, uint64_t> _get_runtime_bytecode_min_max_versions();

// 获取运行时运算符最小和最大版本号的 API 函数声明
TORCH_API std::pair<uint64_t, uint64_t> _get_runtime_operators_min_max_versions();

// 获取移动端支持的类型集合的 API 函数声明
TORCH_API std::unordered_set<std::string> _get_mobile_supported_types();

// 获取已加载的自定义类名称集合的 API 函数声明
TORCH_API std::unordered_set<std::string> _get_loaded_custom_classes();

} // namespace jit
} // namespace torch
```