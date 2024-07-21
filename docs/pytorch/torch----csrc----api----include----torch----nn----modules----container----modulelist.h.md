# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\modulelist.h`

```
  // 使用#pragma once确保头文件只被编译一次
#pragma once

  // 包含所需的头文件
#include <c10/util/irange.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

#include <utility>
#include <vector>

  // 声明torch命名空间和nn命名空间
namespace torch {
namespace nn {

  // `ModuleListImpl`类继承自`Cloneable`模板类，用于保存注册的`Module`列表
class ModuleListImpl : public Cloneable<ModuleListImpl> {
 public:
  // 定义迭代器类型
  using Iterator = std::vector<std::shared_ptr<Module>>::iterator;
  using ConstIterator = std::vector<std::shared_ptr<Module>>::const_iterator;

  // 默认构造函数
  ModuleListImpl() = default;

  // 构造函数，接受可变数量的模块作为参数
  template <typename... Modules>
  explicit ModuleListImpl(Modules&&... modules) {
    // 预留空间以容纳传入的模块数量
    modules_.reserve(sizeof...(Modules));
    // 将传入的模块添加到列表中
    push_back_var(std::forward<Modules>(modules)...);
  }

  // 克隆函数，复制整个`ModuleList`，并可选择指定设备
  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
    auto clone = std::make_shared<ModuleListImpl>();
    // 克隆每个模块，并添加到克隆的`ModuleList`中
    for (const auto& module : modules_) {
      clone->push_back(module->clone(device));
    }
    return clone;
  }

  // 重置函数，对于`ModuleList`为空操作，因为它没有自己的参数
  void reset() override {}

  // 打印函数，将`ModuleList`模块的信息输出到给定的流中
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ModuleList";
  }

  // 将模块添加到列表末尾
  void push_back(std::shared_ptr<Module> module) {
    modules_.push_back(std::move(module));
    const auto index = modules_.size() - 1;
  /// 将给定的索引和模块注册到模块列表中。
  register_module(std::to_string(index), modules_[index]);
}

/// 向 `ModuleList` 容器添加一个新的 `Module`，通过移动或复制到内部的 `shared_ptr`。
/// 此方法允许传递值类型，并且让容器处理封箱。
template <typename M, typename = torch::detail::enable_if_module_t<M>>
void push_back(M&& module) {
  using Type = typename std::remove_reference<M>::type;
  push_back(std::make_shared<Type>(std::forward<M>(module)));
}

/// 解封 `ModuleHolder` 的包含模块，并将其添加到 `ModuleList` 中。
template <typename M>
void push_back(const ModuleHolder<M>& module_holder) {
  push_back(module_holder.ptr());
}

/// 迭代容器并对每个值调用 `push_back()`。
template <typename Container>
void extend(const Container& container) {
  for (const auto& module : container) {
    push_back(module);
  }
}

/// 返回指向 `ModuleList` 开始的迭代器。
Iterator begin() {
  return modules_.begin();
}

/// 返回指向 `ModuleList` 开始的常量迭代器。
ConstIterator begin() const {
  return modules_.begin();
}

/// 返回指向 `ModuleList` 结尾的迭代器。
Iterator end() {
  return modules_.end();
}

/// 返回指向 `ModuleList` 结尾的常量迭代器。
ConstIterator end() const {
  return modules_.end();
}

/// 尝试以请求的类型返回给定索引处的模块。
/// 如果索引超出范围或类型不匹配，则抛出异常。
template <typename T>
T& at(size_t index) {
  static_assert(
      torch::detail::is_module<T>::value,
      "Can only call ModuleList::at with an nn::Module type");
  TORCH_CHECK(index < size(), "Index out of range");
  auto module = modules_[index]->as<T>();
  TORCH_CHECK(
      module,
      "Unable to cast module[",
      index,
      "] to ",
      c10::demangle(typeid(T).name()));
  return *module;
}

/// 尝试以请求的类型返回给定索引处的模块（常量版本）。
/// 如果索引超出范围或类型不匹配，则抛出异常。
template <typename T>
const T& at(size_t index) const {
  static_assert(
      torch::detail::is_module<T>::value,
      "Can only call ModuleList::at with an nn::Module type");
  TORCH_CHECK(index < size(), "Index out of range");
  const auto module = modules_[index]->as<T>();
  TORCH_CHECK(
      module,
      "Unable to cast module[",
      index,
      "] to ",
      c10::demangle(typeid(T).name()));
  return *module;
}

/// 尝试返回一个 `std::shared_ptr`，其动态类型为给定索引处的底层模块类型。
/// 如果索引超出范围，则抛出异常。
std::shared_ptr<Module> ptr(size_t index) const {
  TORCH_CHECK(index < size(), "Index out of range");
  // 返回模块列表中索引为 `index` 的模块指针
  std::shared_ptr<Module> operator[](size_t index) const {
    // 调用 `ptr(index)` 方法返回模块指针
    return ptr(index);
  }

  // 返回模块列表的当前大小
  size_t size() const noexcept {
    // 返回内部存储的模块指针的数量
    return modules_.size();
  }

  // 检查模块列表是否为空
  bool is_empty() const noexcept {
    // 如果模块列表的大小为零，则返回 true
    return size() == 0;
  }

  // 在指定索引处插入一个模块指针
  void insert(size_t index, std::shared_ptr<Module> module) {
    // 检查索引是否在有效范围内
    TORCH_CHECK(index <= size(), "Index out of range");

    // 如果索引等于当前大小，将模块指针移动到末尾
    if (index == size())
      push_back(std::move(module));
    else {
      // 否则，在指定索引处插入模块指针
      modules_.insert(
          modules_.begin() + Iterator::difference_type(index),
          std::move(module));

      // 更新索引之后的模块名称和模块的注册
      for (const auto i : c10::irange(index, size() - 1)) {
        (void)i; // 抑制未使用变量警告
        replace_module(std::to_string(index), modules_[index]);
      }
      register_module(std::to_string(size() - 1), modules_.back());
    }
  }

  // 将一个 `ModuleHolder` 中包含的模块插入到指定索引处
  template <typename M>
  void insert(size_t index, const ModuleHolder<M>& module_holder) {
    // 调用重载的 `insert` 方法插入模块指针
    insert(index, module_holder.ptr());
  }

  // 将一个新的模块插入到指定索引处，通过移动或复制它到内部的 `shared_ptr`
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  void insert(size_t index, M&& module) {
    // 调用重载的 `insert` 方法插入模块指针
    using Type = typename std::remove_reference<M>::type;
    insert(index, std::make_shared<Type>(std::forward<M>(module)));
  }
};

/// 结束了在命名空间 `nn` 中的声明和定义

/// `ModuleListImpl` 的 `ModuleHolder` 子类。
/// 参见 `ModuleListImpl` 类的文档，了解它提供了哪些方法；或者参见 `ModuleHolder` 的文档，
/// 了解 PyTorch 的模块存储语义。

/// 使用 `TORCH_MODULE(ModuleList)` 宏定义了一个 `ModuleList` 类，但具体实现细节未在此处展示。
/// 需要查看宏的定义以了解其具体实现。

} // namespace nn
} // namespace torch
```