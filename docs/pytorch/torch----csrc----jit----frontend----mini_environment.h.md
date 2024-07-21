# `.\pytorch\torch\csrc\jit\frontend\mini_environment.h`

```py
#pragma once

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// 简单的数据结构，用于在嵌套控制块中包含类型 T
// 只能在初始编译后使用，用于类型检查和加载/存储的发射

template <typename T>
struct MiniEnvironment {
  MiniEnvironment(Block* b, std::shared_ptr<MiniEnvironment> next = nullptr)
      : next(std::move(next)) {}

  // 下一个环境的指针，允许链式查找
  std::shared_ptr<MiniEnvironment<T>> next;

  // 在当前环境中查找给定名称的变量
  T findInThisFrame(const std::string& name) {
    auto it = table.find(name);
    if (it != table.end()) {
      return it->second;
    }
    return nullptr;
  }

  // 在所有环境中查找给定名称的变量，包括当前及其父环境
  T findInAnyFrame(const std::string& name) {
    for (auto runner = this; runner; runner = runner->next.get()) {
      if (auto r = runner->findInThisFrame(name)) {
        return r;
      }
    }
    return nullptr;
  }

  // 设置变量名和其对应的值到当前环境
  void setVar(const std::string& name, T value) {
    table[name] = value;
  }

  // 返回当前环境中定义的所有变量名，按字母顺序排列
  std::vector<std::string> definedVariables() {
    std::vector<std::string> result;
    result.reserve(table.size());
    for (auto& kv : table) {
      result.push_back(kv.first);
    }
    std::sort(result.begin(), result.end());
    return result;
  }

 private:
  // 存储变量名到值的映射表
  std::unordered_map<std::string, T> table;
};

} // namespace jit
} // namespace torch
```