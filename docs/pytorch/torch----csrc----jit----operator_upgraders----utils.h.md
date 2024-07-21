# `.\pytorch\torch\csrc\jit\operator_upgraders\utils.h`

```
// 预处理指令，用于确保头文件只被包含一次
#pragma once

// 包含导出宏、可选类型等 C++ 头文件
#include <c10/macros/Export.h>
#include <c10/util/Optional.h>

// 包含 Torch JIT 中的运算符升级相关头文件
#include <torch/csrc/jit/operator_upgraders/version_map.h>

// 包含标准整数类型、字符串和向量容器的头文件
#include <cstdint>
#include <string>
#include <vector>

// 定义 Torch JIT 命名空间
namespace torch::jit {

// 定义结构体 UpgraderRange，表示操作符升级器支持的版本范围
struct UpgraderRange {
  int min_version;  // 最小版本号
  int max_version;  // 最大版本号
};

// 给定操作符的升级器条目列表和当前模型版本号，查找有效的升级器条目
TORCH_API std::optional<UpgraderEntry> findUpgrader(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

// 根据操作符的所有注册升级器判断操作符是否是最新的
TORCH_API bool isOpCurrentBasedOnUpgraderEntries(
    const std::vector<UpgraderEntry>& upgraders_for_schema,
    size_t current_version);

// 根据操作符名称和当前版本号判断操作符的符号是否是最新的
TORCH_API bool isOpSymbolCurrent(
    const std::string& name,
    size_t current_version);

// 返回操作符不存在时可能的旧版架构的列表，适用于被废弃的操作符
TORCH_API std::vector<std::string> loadPossibleHistoricOps(
    const std::string& name,
    std::optional<size_t> version);

// 返回操作符的最大版本号
TORCH_API uint64_t getMaxOperatorVersion();

// 返回操作符的所有升级器支持的最小和最大版本号范围列表
TORCH_API std::vector<UpgraderRange> getUpgradersRangeForOp(
    const std::string& name);

} // namespace torch::jit
```