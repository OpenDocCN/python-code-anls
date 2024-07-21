# `.\pytorch\torch\csrc\jit\serialization\type_name_uniquer.cpp`

```
// 引入 Torch 库中的类型名称唯一化工具头文件
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 获取唯一类型名称的方法
c10::QualifiedName TypeNameUniquer::getUniqueName(c10::ConstNamedTypePtr t) {
    // 查找是否已经存在该类型的唯一名称
    auto it = name_map_.find(t);
    if (it != name_map_.cend()) {
        // 已经有该类型的唯一名称，直接返回
        return it->second;
    }

    // 获取类型的完全限定名称
    auto qualifiedName = t->name().value();
    if (!used_names_.count(qualifiedName)) {
        // 如果该限定名称还未被使用，则将其分配给该类型
        used_names_.insert(qualifiedName);
        name_map_.emplace(std::move(t), qualifiedName);
        return qualifiedName;
    }

    // 如果该限定名称已经被使用，需对其进行处理以获取唯一名称
    // 使用名称转换器进行名称编码以确保唯一性
    auto mangled = mangler_.mangle(qualifiedName);
    while (used_names_.count(mangled)) {
        // 如果生成的名称仍然被使用，继续进行名称转换直到找到唯一的名称
        mangled = mangler_.mangle(qualifiedName);
    }

    // 将生成的唯一名称映射到该类型，并标记为已使用
    name_map_.emplace(std::move(t), mangled);
    used_names_.insert(mangled);
    return mangled;
}

} // namespace torch::jit
```