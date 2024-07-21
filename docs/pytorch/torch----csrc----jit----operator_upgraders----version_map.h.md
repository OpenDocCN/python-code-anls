# `.\pytorch\torch\csrc\jit\operator_upgraders\version_map.h`

```py
// 预处理指令，确保头文件只被包含一次
#pragma once

// 包含导出宏定义头文件，用于跨平台导出符号
#include <c10/macros/Export.h>

// 包含字符串类的头文件
#include <string>

// 包含无序映射容器的头文件
#include <unordered_map>

// 包含向量容器的头文件
#include <vector>

// 命名空间 torch::jit，包含了所有的声明
namespace torch::jit {

// 结构体 UpgraderEntry，用于描述模型升级条目的信息
struct UpgraderEntry {
  int bumped_at_version;  // 整数字段，表示在哪个版本升级
  std::string upgrader_name;  // 字符串字段，升级器的名称
  std::string old_schema;  // 字符串字段，旧的模式描述
};

// 设置函数，用于切换基于升级器计算模块版本的行为
TORCH_API void calculate_package_version_based_on_upgraders(bool val);

// 获取函数，返回当前模块版本计算的标志状态
TORCH_API bool get_version_calculator_flag();

// 获取函数，返回操作符版本映射的引用
TORCH_API const std::unordered_map<std::string, std::vector<UpgraderEntry>>&
get_operator_version_map();

// 用于测试的函数，添加指定操作名的升级条目
TORCH_API void test_only_add_entry(
    const std::string& op_name,
    UpgraderEntry entry);

// 用于测试的函数，移除指定操作名的升级条目
TORCH_API void test_only_remove_entry(const std::string& op_name);

// 用于测试的函数，重置测试标志位
TORCH_API void test_only_reset_flag();

} // 命名空间 torch::jit 结束
```