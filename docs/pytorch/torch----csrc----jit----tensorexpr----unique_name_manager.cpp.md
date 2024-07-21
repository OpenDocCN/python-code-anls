# `.\pytorch\torch\csrc\jit\tensorexpr\unique_name_manager.cpp`

```
// 包含TensorExpr库中所需的头文件

#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

// 包含TensorExpr库中定义的IR和相关库的头文件
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <cctype>

// 定义torch::jit::tensorexpr命名空间
namespace torch::jit::tensorexpr {

// 返回与给定变量关联的唯一名称
const std::string& UniqueNameManager::get_unique_name(VarPtr v) {
  // 查找是否已经遇到过这个变量
  auto iter = unique_name_mapping_.find(v);
  // 如果找到，直接返回已经存储的唯一名称
  if (iter != unique_name_mapping_.end()) {
    return iter->second;
  }

  // 使用变量的名称提示作为前缀来检查是否存在相同前缀的其他名称
  std::string name_hint = v->name_hint();
  if (name_hint.empty()) {
    name_hint = "v";
  } else if (std::isdigit(name_hint[0])) {
    name_hint = "v" + name_hint;
  }
  // 获取当前前缀的计数器
  int& count = unique_name_count_[name_hint];
  while (true) {
    // 即使使用新计数，这个名称可能已经被使用。例如("x", 1) 可能与 ("x_1", 0) 冲突
    int count_v = count++;
    std::string unique_name = name_hint;
    if (count_v > 0) {
      unique_name += "_" + std::to_string(count_v);
    }
    // 如果生成的唯一名称未被使用，则将其加入到已使用名称集合中，并将其与变量关联
    if (all_unique_names_.count(unique_name) == 0) {
      all_unique_names_.insert(unique_name);
      auto result = unique_name_mapping_.insert(std::make_pair(v, unique_name));
      return result.first->second;
    }
  }
}

// 获取与给定变量句柄关联的唯一名称
const std::string& UniqueNameManager::get_unique_name(const VarHandle& v) {
  return get_unique_name(v.node());
}

} // namespace torch::jit::tensorexpr
```