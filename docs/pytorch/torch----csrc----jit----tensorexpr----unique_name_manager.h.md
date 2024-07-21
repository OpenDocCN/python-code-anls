# `.\pytorch\torch\csrc\jit\tensorexpr\unique_name_manager.h`

```py
#pragma once
// 预处理指令，确保此头文件只被包含一次

#include <string>
// 包含标准字符串库

#include <unordered_map>
// 包含无序映射容器库

#include <unordered_set>
// 包含无序集合容器库

#include <torch/csrc/Export.h>
// 包含 Torch 导出相关的头文件

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
// 包含 Torch 的 JIT（即时编译）库中的前向声明头文件

namespace torch {
namespace jit {
namespace tensorexpr {

class VarHandle;
class Var;
// 声明 VarHandle 和 Var 类

using VarNameMap = std::unordered_map<VarPtr, std::string>;
// 使用无序映射，将 VarPtr 映射到字符串的映射类型定义

// 管理器，用于从变量中获取唯一名称
// 它从变量的名称提示开始，并附加 "_" + $counter，直到找到唯一名称为止。
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API UniqueNameManager {
 public:
  // 获取变量句柄对应的唯一名称
  const std::string& get_unique_name(const VarHandle& v);

  // 获取变量指针对应的唯一名称
  const std::string& get_unique_name(VarPtr v);

 private:
  friend class ScopedVarName;
  VarNameMap unique_name_mapping_; // 记录每个变量指针对应的唯一名称
  std::unordered_map<std::string, int> unique_name_count_; // 记录每种名称的计数
  std::unordered_set<std::string> all_unique_names_; // 记录所有已使用的唯一名称集合
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```