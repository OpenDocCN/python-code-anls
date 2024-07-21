# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\parameterlist.h`

```py
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

#include <vector>

namespace torch {
namespace nn {

// ParameterListImpl 类继承自 Cloneable<ParameterListImpl>，表示参数列表实现
class ParameterListImpl : public Cloneable<ParameterListImpl> {
 public:
  // 定义迭代器类型，用于访问参数列表中的项目
  using Iterator = typename std::vector<OrderedDict<std::string, torch::Tensor>::Item>::iterator;
  using ConstIterator = typename std::vector<OrderedDict<std::string, torch::Tensor>::Item>::const_iterator;

  // 默认构造函数
  ParameterListImpl() = default;

  /// Constructs the `ParameterList` from a variadic list of ParameterList.
  // 从参数列表构造 ParameterListImpl 对象，支持变长参数
  template <typename... Tensors>
  explicit ParameterListImpl(Tensors&&... params) {
    parameters_.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  // 从常量参数列表构造 ParameterListImpl 对象，支持变长参数
  template <typename... Tensors>
  explicit ParameterListImpl(const Tensors&... params) {
    parameters_.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  /// `reset()` is empty for `ParameterList`, since it does not have parameters
  // reset() 函数为空实现，因为 ParameterList 不包含自己的参数
  void reset() override {}

  /// Pretty prints the `ParameterList` module into the given `stream`.
  // 将 ParameterList 模块美观地打印到流中
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ParameterList(" << std::endl;
    for (const auto& pair : parameters_) {
      stream << "(" << pair.key() << ")"
             << ": Parameter containing: [" << pair.value().scalar_type()
             << " of size " << pair.value().sizes() << "]";
      ;
      stream << std::endl;
    }
    stream << ")";
  }

  /// push the a given parameter at the end of the list
  // 在参数列表末尾添加给定的参数
  void append(torch::Tensor&& param) {
    bool requires_grad = param.requires_grad();
    register_parameter(std::to_string(parameters_.size()), std::move(param), requires_grad);
  }

  /// push the a given parameter at the end of the list
  // 在参数列表末尾添加给定的参数
  void append(const torch::Tensor& param) {
    bool requires_grad = param.requires_grad();
    register_parameter(std::to_string(parameters_.size()), param, requires_grad);
  }

  /// push the a given parameter at the end of the list
  /// And the key of the pair will be discarded, only the value
  /// will be added into the `ParameterList`
  // 在参数列表末尾添加给定的参数，只使用键值对中的值，丢弃键
  void append(const OrderedDict<std::string, torch::Tensor>::Item& pair) {
    register_parameter(std::to_string(parameters_.size()), pair.value(), pair.value().requires_grad());
  }

  /// extend parameters from a container to the end of the list
  // 将容器中的参数扩展到参数列表的末尾
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& param : container) {
      append(param);
    }
  }

  /// Returns an iterator to the start of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string, torch::Tensor>::Item`
  // 返回参数列表的起始迭代器，迭代器类型为 OrderedDict<std::string, torch::Tensor>::Item
  Iterator begin() {
    /// Returns an iterator pointing to the beginning of the ParameterList.
    /// The returned iterator is of type `OrderedDict<std::string, torch::Tensor>::Item`.
    Iterator begin() {
      return parameters_.begin();
    }
    
    /// Returns a const iterator pointing to the beginning of the ParameterList.
    /// The returned iterator is of type `OrderedDict<std::string, torch::Tensor>::Item`.
    ConstIterator begin() const {
      return parameters_.begin();
    }
    
    /// Returns an iterator pointing to the end of the ParameterList.
    /// The returned iterator is of type `OrderedDict<std::string, torch::Tensor>::Item`.
    Iterator end() {
      return parameters_.end();
    }
    
    /// Returns a const iterator pointing to the end of the ParameterList.
    /// The returned iterator is of type `OrderedDict<std::string, torch::Tensor>::Item`.
    ConstIterator end() const {
      return parameters_.end();
    }
    
    /// Returns the tensor associated with the given index `idx`.
    /// Throws an exception if `idx` is out of range.
    /// Use `contains(idx)` to check for existence before accessing.
    at::Tensor& at(size_t idx) {
      TORCH_CHECK(idx < size(), "Index out of range");
      return parameters_[std::to_string(idx)];
    }
    
    /// Returns the tensor associated with the given index `idx`.
    /// Throws an exception if `idx` is out of range.
    /// Use `contains(idx)` to check for existence before accessing.
    const at::Tensor& at(size_t idx) const {
      TORCH_CHECK(idx < size(), "Index out of range");
      return parameters_[std::to_string(idx)];
    }
    
    /// Provides access to the tensor at the given index `idx` using operator[].
    /// Throws an exception if `idx` is out of range.
    /// Use `contains(idx)` to check for existence before accessing.
    at::Tensor& operator[](size_t idx) {
      return at(idx);
    }
    
    /// Provides access to the tensor at the given index `idx` using operator[].
    /// Throws an exception if `idx` is out of range.
    /// Use `contains(idx)` to check for existence before accessing.
    const at::Tensor& operator[](size_t idx) const {
      return at(idx);
    }
    
    /// Returns the number of elements in the ParameterList.
    size_t size() const noexcept {
      return parameters_.size();
    }
    
    /// Returns true if the ParameterList is empty.
    bool is_empty() const noexcept {
      return parameters_.is_empty();
    }
    
    /// Overloads the += operator to allow incremental addition of another Container to this ParameterList.
    template <typename Container>
    Container& operator+=(const Container& other) {
      extend(other);
      return *this;
    }
    
    private:
    /// Helper function to recursively append variables to the ParameterList.
    template <typename Head, typename... Tail>
    void push_back_var(Head&& head, Tail&&... tail) {
      append(std::forward<Head>(head));
      // Recursively calls this method until only one entry is left in the parameter pack.
      // Then calls `push_back()` a final time (above).
      push_back_var(std::forward<Tail>(tail)...);
    }
    
    /// Base case of `push_back_var` when the list of modules is empty.
    void push_back_var() {}
};
TORCH_MODULE(ParameterList);
} // namespace nn
} // namespace torch


// 结束 nn 命名空间定义
};
// 声明 TORCH_MODULE 宏，用于定义 Torch 模块
TORCH_MODULE(ParameterList);
// 结束 nn 命名空间定义
} // namespace nn
// 结束 torch 命名空间定义
} // namespace torch
```