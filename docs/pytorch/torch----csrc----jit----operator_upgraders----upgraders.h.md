# `.\pytorch\torch\csrc\jit\operator_upgraders\upgraders.h`

```py
#pragma once
#include <c10/macros/Export.h>  // 包含导出宏定义
#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的 IR 类定义
#include <mutex>  // 包含互斥量的头文件
#include <string>  // 包含字符串处理的头文件
#include <unordered_map>  // 包含无序映射的头文件

namespace torch::jit {

class UpgradersMap {
 public:
  void set_content(
      std::unordered_map<std::string, std::shared_ptr<Graph>>&& content);  // 设置内容函数声明，接受右值引用的无序映射
  int count();  // 返回内容数量函数声明
  const std::unordered_map<std::string, std::shared_ptr<Graph>>& get_content();  // 获取内容函数声明，返回常引用的无序映射
  bool is_populated();  // 是否已填充函数声明
  // THESE METHODS ARE ONLY USED FOR TESTING PURPOSES
  void test_only_set_content(
      const std::unordered_map<std::string, std::string>& content);  // 仅用于测试目的的设置内容函数声明，接受常引用的字符串到字符串的无序映射
  void test_only_remove_content(
      const std::unordered_map<std::string, std::string>& content);  // 仅用于测试目的的移除内容函数声明，接受常引用的字符串到字符串的无序映射

 private:
  std::unordered_map<std::string, std::shared_ptr<Graph>> content_;  // 内容的无序映射，映射键为字符串，值为共享指针指向 Graph 对象
  std::mutex lock;  // 互斥量对象，用于保护内容的并发访问
  bool isPopulated = false;  // 布尔值，标志内容是否已填充
};

TORCH_API void populate_upgraders_map(
    std::unordered_map<std::string, std::shared_ptr<Graph>>&& content);  // 填充升级映射函数声明，接受右值引用的无序映射

TORCH_API int get_upgraders_map_size();  // 获取升级映射大小函数声明，返回整数

TORCH_API bool is_upgraders_map_populated();  // 升级映射是否已填充函数声明，返回布尔值

TORCH_API const std::unordered_map<std::string, std::shared_ptr<Graph>>&
dump_upgraders_map();  // 转储升级映射函数声明，返回常引用的无序映射

// THESE TWO METHODS BELOW ARE ONLY USED FOR TESTING
TORCH_API void test_only_populate_upgraders(
    const std::unordered_map<std::string, std::string>& content);  // 仅用于测试目的的填充升级函数声明，接受常引用的字符串到字符串的无序映射

TORCH_API void test_only_remove_upgraders(
    const std::unordered_map<std::string, std::string>& content);  // 仅用于测试目的的移除升级函数声明，接受常引用的字符串到字符串的无序映射

} // namespace torch::jit
```