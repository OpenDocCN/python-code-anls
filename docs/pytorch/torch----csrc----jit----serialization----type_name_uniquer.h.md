# `.\pytorch\torch\csrc\jit\serialization\type_name_uniquer.h`

```
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <torch/csrc/jit/frontend/name_mangler.h>
// 包含 Torch 的名称混淆器头文件

#include <torch/csrc/jit/ir/type_hashing.h>
// 包含 Torch 的类型哈希头文件

namespace torch::jit {

/**
 * class TypeNameUniquer
 *
 * Generates a unique name for every type `t` passed in. Types that compare
 * equal with EqualType will receive the same unique name.
 *
 * This is used during Module::save(), to resolve type name collisions during
 * serialization.
 */
class TORCH_API TypeNameUniquer {
 public:
  // 返回一个唯一的限定名，用于给定的类型 `t`
  c10::QualifiedName getUniqueName(c10::ConstNamedTypePtr t);

 private:
  NameMangler mangler_;
  // 名称混淆器对象，用于生成唯一的名称

  std::unordered_set<c10::QualifiedName> used_names_;
  // 存储已使用的限定名的集合，用于确保生成的唯一名称不会重复

  std::unordered_map<
      c10::ConstNamedTypePtr,
      c10::QualifiedName,
      HashType,
      EqualType>
      name_map_;
  // 映射表，将每个类型映射到其唯一名称，使用自定义的类型哈希和相等性判断
};

} // namespace torch::jit
// 结束 torch::jit 命名空间
```