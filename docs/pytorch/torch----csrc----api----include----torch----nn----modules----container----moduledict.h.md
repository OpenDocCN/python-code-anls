# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\moduledict.h`

```
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/ordered_dict.h>
#include <vector>

namespace torch {
namespace nn {

/// An OrderedDict of `Module`s that registers its elements by their `key`s.
///
/// \rst
/// .. code-block:: cpp
///
///   torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
///     {"linear", Linear(10, 3).ptr()},  // 创建名为 "linear" 的线性模块指针，参数为 (10, 3)
///     {"conv", Conv2d(1, 2, 3).ptr()},  // 创建名为 "conv" 的二维卷积模块指针，参数为 (1, 2, 3)
///     {"dropout", Dropout(0.5).ptr()},  // 创建名为 "dropout" 的随机丢弃模块指针，参数为 0.5
///   };
///   torch::nn::ModuleDict dict1(ordereddict);  // 使用有序字典初始化 ModuleDict 对象 dict1
///
///   for (const auto &module : *dict1) {  // 遍历 dict1 中的每个模块
///     module->pretty_print(std::cout);  // 打印模块的信息到控制台
///   }
///
///   std::vector<std::pair<std::string, std::shared_ptr<Module>>> list = {
///     {"linear", Linear(10, 3).ptr()},  // 创建另一个名为 "linear" 的线性模块指针
///     {"conv", Conv2d(1, 2, 3).ptr()},  // 创建另一个名为 "conv" 的二维卷积模块指针
///     {"dropout", Dropout(0.5).ptr()},  // 创建另一个名为 "dropout" 的随机丢弃模块指针
///   };
///   torch::nn::ModuleDict dict2(list);  // 使用列表初始化 ModuleDict 对象 dict2
///
///   for (const auto &module : *dict2) {  // 遍历 dict2 中的每个模块
///     module->pretty_print(std::cout);  // 打印模块的信息到控制台
///   }
///
/// \endrst
///
/// Why should you use `ModuleDict` instead of a simple `map` or `OrderedDict`?
/// The value a `ModuleDict` provides over manually calling an ordered map of
/// modules is that it allows treating the whole container *as a single module*,
/// such that performing a transformation on the `ModuleDict` applies to each of
/// the modules it stores (which are each a registered submodule of the
/// `ModuleDict`). For example, calling `.to(torch::kCUDA)` on a `ModuleDict`
/// will move each module in the map to CUDA memory. For example:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::OrderedDict<std::string, std::shared_ptr<Module>> ordereddict = {
///     {"linear", Linear(10, 3).ptr()},
///     {"conv", Conv2d(1, 2, 3).ptr()},
///     {"dropout", Dropout(0.5).ptr()},
///   };
///   torch::nn::ModuleDict dict(ordereddict);
///
///   // Convert all modules to CUDA.
///   dict->to(torch::kCUDA);
///
/// \endrst
///
/// Finally, `ModuleDict` provides a lightweight container API, such as allowing
/// iteration over submodules, positional access, adding new modules from a
/// vector of key-module pairs or an `OrderedDict` or another `ModuleDict` after
/// construction via `update`.
class ModuleDictImpl : public Cloneable<ModuleDictImpl> {
 public:
  using Iterator =
      torch::OrderedDict<std::string, std::shared_ptr<Module>>::Iterator;
  using ConstIterator =
      torch::OrderedDict<std::string, std::shared_ptr<Module>>::ConstIterator;

  ModuleDictImpl() = default;

  /// Constructs the `ModuleDict` from a list of string-Module pairs.
  explicit ModuleDictImpl(
      const std::vector<std::pair<std::string, std::shared_ptr<Module>>>&
          modules) {
    update(modules);  // 使用给定的模块列表更新 ModuleDict
  }

  /// Constructs the `ModuleDict` from an `OrderedDict`.
  explicit ModuleDictImpl(
      const torch::OrderedDict<std::string, std::shared_ptr<Module>>& modules) {
    update(modules);  // 使用给定的有序字典更新 ModuleDict
  }
  /// 更新模块列表。
  update(modules);
}

/// 返回 `ModuleDict` 中的所有项。
std::vector<std::pair<std::string, std::shared_ptr<Module>>> items() const {
  return modules_.pairs();
}

/// 返回 `ModuleDict` 中所有的键。
std::vector<std::string> keys() const {
  return modules_.keys();
}

/// 返回 `ModuleDict` 中所有的值。
std::vector<std::shared_ptr<Module>> values() const {
  return modules_.values();
}

/// 返回指向 `ModuleDict` 开始处的迭代器。
Iterator begin() {
  return modules_.begin();
}

/// 返回指向 `ModuleDict` 开始处的常量迭代器。
ConstIterator begin() const {
  return modules_.begin();
}

/// 返回指向 `ModuleDict` 结尾处的迭代器。
Iterator end() {
  return modules_.end();
}

/// 返回指向 `ModuleDict` 结尾处的常量迭代器。
ConstIterator end() const {
  return modules_.end();
}

/// 返回当前存储在 `ModuleDict` 中的项数。
size_t size() const noexcept {
  return modules_.size();
}

/// 如果 `ModuleDict` 为空则返回 true，否则返回 false。
bool empty() const noexcept {
  return modules_.is_empty();
}

/// 检查 `ModuleDict` 中是否包含指定键。
bool contains(const std::string& key) const noexcept {
  return modules_.contains(key);
}

/// 清空 `ModuleDict` 中的所有项。
void clear() {
  // 不删除模块的注册，以保持与 Python 版本的一致性。
  modules_.clear();
}

/// `ModuleDict` 的特殊克隆函数，因为它不使用 `reset()`。
std::shared_ptr<Module> clone(
    const optional<Device>& device = nullopt) const override {
  auto clone = std::make_shared<ModuleDictImpl>();
  for (const auto& module : modules_) {
    clone->insert(module.key(), module.value()->clone(device));
  }
  return clone;
}

/// 对于 `ModuleDict`，`reset()` 是空的，因为它没有自己的参数。
void reset() override {}

/// 将 `ModuleDict` 漂亮地打印到给定的 `stream` 中。
void pretty_print(std::ostream& stream) const override {
  stream << "torch::nn::ModuleDict";
}

/// 尝试返回与给定键关联的 `Module`。如果 `ModuleDict` 中没有这样的键，则抛出异常。
/// 可以先使用 contains(key) 进行非抛出式访问检查。
std::shared_ptr<Module> operator[](const std::string& key) const {
  return modules_[key];
}

/// 尝试以请求的类型返回给定键处的模块。如果 `ModuleDict` 中没有这样的键，则抛出异常。
/// 可以先使用 contains(key) 进行非抛出式访问检查。
template <typename T>
T& at(const std::string& key) {
  static_assert(
      torch::detail::is_module<T>::value,
      "Can only call ModuleList::at with an nn::Module type");
  auto module = modules_[key]->as<T>();
  /// Checks if the `ModuleDict` contains the specified key.
  /// Returns true if the key is found, false otherwise.
  bool contains(const std::string& key) const {
    return modules_.contains(key);
  }

  /// Returns a reference to the `Module` associated with the given `key`.
  /// Throws an exception if no such `key` is stored in the `ModuleDict`.
  /// Use contains(key) before to ensure the key exists if a non-throwing access is preferred.
  std::shared_ptr<Module>& operator[](const std::string& key) {
    return modules_[key];
  }

  /// Const version of operator[], returns a const reference to the `Module`
  /// associated with the given `key`.
  /// Throws an exception if no such `key` is stored in the `ModuleDict`.
  /// Use contains(key) before to ensure the key exists if a non-throwing access is preferred.
  const std::shared_ptr<Module>& operator[](const std::string& key) const {
    return modules_[key];
  }

private:
  /// Private `OrderedDict` holding the key-Module pairs.
  torch::OrderedDict<std::string, std::shared_ptr<Module>> modules_;

  /// Insert a key-module pair by overwriting existing keys,
  /// and register or replace the `Module`.
  void insert(const std::string& key, std::shared_ptr<Module> module) {
    if (contains(key)) {
      modules_[key] = std::move(module);
      replace_module(key, modules_[key]);
    } else {
      modules_.insert(key, std::move(module));
      register_module(key, modules_.back().value());
    }
  }
};

/// 这是一个命名空间尾部的闭合标记，结束了 `nn` 命名空间的定义。

/// `ModuleHolder` 是 `ModuleDictImpl` 的子类。
/// 查看 `ModuleDictImpl` 类的文档以了解它提供了哪些方法，
/// 或者查看 `ModuleHolder` 的文档以了解 PyTorch 模块存储语义的细节。
TORCH_MODULE(ModuleDict);

} // namespace nn
} // namespace torch
```