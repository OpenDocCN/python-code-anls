# `.\pytorch\torch\csrc\jit\operator_upgraders\upgraders.cpp`

```py
#include <torch/csrc/jit/operator_upgraders/upgraders.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace torch::jit {

// 静态成员变量，用于存储操作升级器的映射关系
static UpgradersMap upgradersMap;

// 设置操作升级器映射的内容
void UpgradersMap::set_content(
    std::unordered_map<std::string, std::shared_ptr<Graph>>&& content) {
  // 确保只在首次填充映射时进行操作
  std::lock_guard<std::mutex> _(lock);
  if (isPopulated) {
    return;
  }

  // 移动传入的内容到成员变量
  content_ = std::move(content);
  // 标记映射已经填充
  isPopulated = true;
}

// 返回操作升级器映射的大小
int UpgradersMap::count() {
  std::lock_guard<std::mutex> _(lock);
  return content_.size();
}

// 检查操作升级器映射是否已经填充
bool UpgradersMap::is_populated() {
  std::lock_guard<std::mutex> _(lock);
  return isPopulated;
}

// 获取操作升级器映射的内容
const std::unordered_map<std::string, std::shared_ptr<Graph>>& UpgradersMap::
    get_content() {
  std::lock_guard<std::mutex> _(lock);
  return content_;
}

// 仅用于测试，设置操作升级器映射的内容
void UpgradersMap::test_only_set_content(
    const std::unordered_map<std::string, std::string>& content) {
  std::lock_guard<std::mutex> _(lock);
  for (const auto& entry : content) {
    auto graph = std::make_shared<Graph>();
    // 解析输入的 IR 字符串，并将结果图形对象插入映射
    torch::jit::parseIR(entry.second, graph.get());
    content_.insert(std::make_pair(entry.first, graph));
  }
}

// 仅用于测试，移除操作升级器映射的内容
void UpgradersMap::test_only_remove_content(
    const std::unordered_map<std::string, std::string>& content) {
  std::lock_guard<std::mutex> _(lock);
  for (const auto& entry : content) {
    // 移除指定键的内容
    content_.erase(entry.first);
  }
}

// 填充全局的操作升级器映射
void populate_upgraders_map(
    std::unordered_map<std::string, std::shared_ptr<Graph>>&& content) {
  upgradersMap.set_content(std::move(content));
}

// 获取全局操作升级器映射的大小
int get_upgraders_map_size() {
  return upgradersMap.count();
}

// 检查全局操作升级器映射是否已经填充
bool is_upgraders_map_populated() {
  return upgradersMap.is_populated();
}

// 导出全局操作升级器映射的内容
const std::unordered_map<std::string, std::shared_ptr<Graph>>&
dump_upgraders_map() {
  return upgradersMap.get_content();
}

// 仅用于测试，填充全局操作升级器映射的内容
void test_only_populate_upgraders(
    const std::unordered_map<std::string, std::string>& content) {
  upgradersMap.test_only_set_content(content);
}

// 仅用于测试，移除全局操作升级器映射的内容
void test_only_remove_upgraders(
    const std::unordered_map<std::string, std::string>& content) {
  upgradersMap.test_only_remove_content(content);
}

} // namespace torch::jit
```