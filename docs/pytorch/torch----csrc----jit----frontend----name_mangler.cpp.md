# `.\pytorch\torch\csrc\jit\frontend\name_mangler.cpp`

```py
// 包含 Torch 的名称管理器头文件
#include <torch/csrc/jit/frontend/name_mangler.h>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 实现 NameMangler 类的 mangle 方法，用于名称混淆
c10::QualifiedName NameMangler::mangle(const c10::QualifiedName& name) {
  // 定义名称混淆的前缀
  static const std::string manglePrefix = "___torch_mangle_";
  // 获取名称的原子列表
  std::vector<std::string> atoms = name.atoms();

  // 搜索已存在的名称混淆命名空间
  // 如果名称已经被混淆过，则只需增加整数部分
  for (auto& atom : atoms) {
    // 检查是否包含名称混淆的前缀
    auto pos = atom.find(manglePrefix);
    if (pos != std::string::npos) {
      // 提取出当前混淆索引部分
      auto num = atom.substr(pos + manglePrefix.size());
      // 将字符串索引转换为整数
      size_t num_i = std::stoi(num);
      // 更新 mangleIndex_ 为当前索引值加一的最大值
      mangleIndex_ = std::max(mangleIndex_, num_i + 1);
      // 构建新的原子前缀
      std::string newAtomPrefix;
      newAtomPrefix.reserve(atom.size());
      // 将名称的一部分追加到前缀的末尾
      newAtomPrefix.append(atom, 0, pos);
      newAtomPrefix.append(manglePrefix);
      // 更新原子为新的混淆名称
      atom = newAtomPrefix + std::to_string(mangleIndex_++);
      // 返回更新后的完整限定名称
      return c10::QualifiedName(atoms);
    }
  }

  // 否则，在基本名称之前添加一个名称混淆命名空间
  TORCH_INTERNAL_ASSERT(!atoms.empty());
  atoms.insert(atoms.end() - 1, manglePrefix + std::to_string(mangleIndex_++));
  // 返回更新后的完整限定名称
  return c10::QualifiedName(atoms);
}

} // namespace torch::jit
```