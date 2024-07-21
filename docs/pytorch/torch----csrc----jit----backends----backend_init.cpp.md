# `.\pytorch\torch\csrc\jit\backends\backend_init.cpp`

```
#include <torch/csrc/jit/backends/backend_init.h>

#include <pybind11/iostream.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_resolver.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

// 获取根据模块 \p mod 的层次结构中共享的所有类型
std::unordered_set<TypePtr> getSharedModuleTypes(Module& mod) {
  // 维护一个所有 TypePtr 的集合
  std::unordered_set<TypePtr> types;
  // 维护另一个集合，其中包含遇到多次的 TypePtr
  std::unordered_set<TypePtr> duplicate_types;

  // 遍历层次结构中的所有模块，包括根模块
  for (auto module : mod.modules()) {
    auto module_type = module.type();
    if (types.count(module_type) > 0) {
      duplicate_types.insert(module_type);
    }

    types.insert(module_type);
  }

  return duplicate_types;
}

// 有选择地将 \p mod 降级到后端。调用 \p to_backend
// 以降级模块。\p modules_to_lower 包含应该被降级的 \p mod 的子模块的限定名称。
void toBackendSelectiveImpl(
    Module& mod,
    const py::function& to_backend,
    const std::vector<std::string>& modules_to_lower,
    const std::unordered_set<TypePtr>& duplicate_types) {
  // 此映射将在稍后用于重新映射所有降级子模块的祖先模块图中的类型
  std::unordered_map<TypePtr, TypePtr> type_remap;

  // 对于每个应该被降级的模块：
  for (const auto& module_to_lower : modules_to_lower) {
    // 使用 QualifiedName 解析限定模块名称
    c10::QualifiedName qual_module_name(module_to_lower);
    auto& atoms = qual_module_name.atoms();

    // 使用 qual_module_name 的原子搜索模块层次结构，直到 current 指向要降级的模块，parent 指向其父级
    Module current = mod;
    Module parent;

    for (size_t i = 0, e = atoms.size(); i < e; ++i) {
      IValue submodule = current.attr(atoms[i]);
      if (submodule.isModule()) {
        if (i == e - 1) {
          parent = current;
        }
        current = submodule.toModule();
      } else {
        std::stringstream err;
        err << "Attribute named " << atoms[i] << " is not a Module";
        throw std::runtime_error(err.str());
      }
    }

    // 检查父类型是否共享，因此可以进行编辑
    if (duplicate_types.count(parent.type()) > 0) {
      throw py::cast_error(c10::str(
          "Selective lowering is only supported for module hierarchies with unique types for selected modules; ",
          parent.type()->repr_str(),
          " is shared"));
    }

    // 在需要降级的模块上调用 to_backend。在这样做之前，需要对其进行包装，因为 _to_jit_backend 接受包装的模块
    // 将Python对象转换为C++ Module类型，以便访问其属性和方法
    auto lowered_submodule =
        py::cast<Module>(to_backend(py::module::import("torch.jit._recursive")
                                        .attr("wrap_cpp_module")(current))
                             .attr("_c"));
    
    // 调整父模块的类型，使子模块的类型与lowered_submodule的类型匹配
    auto parent_type = parent.type();
    
    parent_type->unsafeChangeAttributeType(
        atoms.back(), lowered_submodule.type());
    parent.setattr(atoms.back(), lowered_submodule._ivalue());
    
    // 记录从旧类型到lowered类型的类型映射关系
    type_remap[current.type()] = lowered_submodule.type();
    }
    
    // 将需要降级的所有模块降级后，重映射层次结构中所有图中的类型，确保所有图使用新的降级类型
    auto type_remap_fn = [&type_remap](TypePtr in) {
    auto it = type_remap.find(in);
    if (it == type_remap.end())
      return in;
    return it->second;
    };
    
    // modules()遍历层次结构中的所有模块，包括根模块
    for (auto module : mod.modules()) {
    auto module_type = module.type();
    for (auto& fn : module_type->methods()) {
      // 获取模块的方法和其对应的图
      auto method = module.get_method(fn->name());
      auto graph = method.graph();
      
      // 重映射图中的类型，使其使用新的类型映射
      graph->remapTypes(type_remap_fn);
      
      // 使用新的类型映射克隆并设置方法的新Schema
      auto new_schema = fn->getSchema().cloneWithRemappedTypes(type_remap_fn);
      fn->setSchema(new_schema);
    }
    }
} // namespace jit
} // namespace torch
```