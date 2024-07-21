# `.\pytorch\torch\csrc\jit\ir\type_hashing.cpp`

```py
#include <torch/c`
# 包含 Torch JIT 类型哈希算法的头文件
#include <torch/csrc/jit/ir/type_hashing.h>

# 包含 ATen 核心功能的头文件
#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/hash.h>

# 包含 Torch JIT IR 相关的头文件
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

namespace {
# 定义用于计算类型哈希值的函数
size_t hashType(const Type& type) {
  # 如果类型为命名类型（ClassType），则计算其哈希值
  if (auto named_type = type.castRaw<ClassType>()) {
    return c10::get_hash(
        named_type->name().value(), named_type->compilation_unit());
  }
  # 初始化哈希值为0
  size_t hash = 0;
  # 遍历类型中包含的所有子类型，并计算它们的哈希值
  for (const auto& containedType : type.containedTypes()) {
    hash = at::hash_combine(hash, hashType(*containedType));
  }
  # 将类型的种类（kind）加入到哈希值中
  hash = at::hash_combine(hash, get_hash(type.kind()));
  # 返回最终计算得到的哈希值
  return hash;
}
} // namespace

# 实现类型指针（TypePtr）的哈希函数
size_t HashType::operator()(const TypePtr& type) const {
  return hashType(*type);
}

# 实现常量类型指针（ConstTypePtr）的哈希函数
size_t HashType::operator()(const c10::ConstTypePtr& type) const {
  return hashType(*type);
}

# 实现类型指针（TypePtr）的相等比较运算符
bool EqualType::operator()(const TypePtr& a, const TypePtr& b) const {
  return *a == *b;
}

# 实现常量类型指针（ConstTypePtr）的相等比较运算符
bool EqualType::operator()(
    const c10::ConstTypePtr& a,
    const c10::ConstTypePtr& b) const {
  return *a == *b;
}

} // namespace torch::jit
```