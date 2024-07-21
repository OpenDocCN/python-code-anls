# `.\pytorch\torch\csrc\api\include\torch\nn\modules\container\parameterdict.h`

```
#pragma once

#include <torch/nn/cloneable.h>  // 包含克隆接口的头文件
#include <torch/nn/pimpl.h>     // 包含私有实现的头文件
#include <torch/ordered_dict.h> // 包含有序字典的头文件
#include <utility>              // 包含标准工具库
#include <vector>               // 包含向量容器的头文件

namespace torch {
namespace nn {

class ParameterDictImpl : public Cloneable<ParameterDictImpl> {
 public:
  using Iterator = OrderedDict<std::string, Tensor>::Iterator;         // 使用有序字典的迭代器类型
  using ConstIterator = OrderedDict<std::string, Tensor>::ConstIterator; // 使用有序字典的常量迭代器类型

  ParameterDictImpl() = default;  // 默认构造函数

  explicit ParameterDictImpl(
      const torch::OrderedDict<std::string, torch::Tensor>& params) {  // 显式构造函数，接受有序字典作为参数
    parameters_ = params;  // 将参数复制给成员变量
  }

  /// `reset()` is empty for `ParameterDict`, since it does not have
  /// parameters of its own.
  void reset() override {}  // 重置函数，对于ParameterDict为空实现，因为它本身没有参数

  /// Pretty prints the `ParameterDict` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {  // 美化打印函数，输出ParameterDict模块的详细信息到流中
    stream << "torch::nn::ParameterDict(" << std::endl;
    for (const auto& pair : parameters_) {  // 遍历参数字典中的每一对键值对
      stream << "(" << pair.key() << ")"
             << ": Parameter containing: [" << pair.value().scalar_type() // 输出参数的数据类型
             << " of size " << pair.value().sizes() << "]";  // 输出参数的尺寸信息
      ;
      stream << std::endl;
    }
    stream << ")";  // 输出结束标记
  }

  /// Insert the parameter along with the key into ParameterDict
  /// The parameter is set to be require grad by default
  Tensor& insert(std::string key, Tensor param) {  // 插入函数，将参数和键插入ParameterDict
    bool requires_grad = param.requires_grad();   // 获取参数是否需要梯度
    return register_parameter(std::move(key), std::move(param), requires_grad); // 调用注册参数的函数
  }

  /// Remove key from the ParameterDict and return its value, throw exception
  /// if the key is not contained. Please check contains(key) before for a
  /// non-throwing access.
  Tensor pop(const std::string& key) {  // 弹出函数，移除键对应的值并返回
    torch::Tensor v = parameters_[key];  // 获取键对应的值
    parameters_.erase(key);              // 从字典中移除该键值对
    return v;                            // 返回移除的值
  }

  /// Return the keys in the dict
  ::std::vector<std::string> keys() const {  // 返回字典中所有键的向量
    return parameters_.keys();  // 调用有序字典的键获取函数
  }

  /// Return the Values in the dict
  ::std::vector<torch::Tensor> values() const {  // 返回字典中所有值的向量
    return parameters_.values();  // 调用有序字典的值获取函数
  }

  /// Return an iterator to the start of ParameterDict
  Iterator begin() {  // 返回开始位置的迭代器
    return parameters_.begin();  // 调用有序字典的开始迭代器
  }

  /// Return a const iterator to the start of ParameterDict
  ConstIterator begin() const {  // 返回开始位置的常量迭代器
    return parameters_.begin();  // 调用有序字典的常量开始迭代器
  }

  /// Return an iterator to the end of ParameterDict
  Iterator end() {  // 返回结束位置的迭代器
    return parameters_.end();  // 调用有序字典的结束迭代器
  }

  /// Return a const iterator to the end of ParameterDict
  ConstIterator end() const {  // 返回结束位置的常量迭代器
    return parameters_.end();  // 调用有序字典的常量结束迭代器
  }

  /// Return the number of items currently stored in the ParameterDict
  size_t size() const noexcept {  // 返回当前存储在ParameterDict中的条目数量
    return parameters_.size();  // 调用有序字典的大小获取函数
  }

  /// Return true if the ParameterDict is empty, otherwise return false
  bool empty() const noexcept {  // 如果ParameterDict为空则返回true，否则返回false
    return parameters_.is_empty();  // 调用有序字典的空判断函数
  }

  /// Update the ParameterDict with the key-value pairs from
  /// another ParameterDict, overwriting existing key
  template <typename Container>
  void update(const Container& container) {  // 更新函数，使用另一个ParameterDict的键值对更新当前字典，覆盖现有键
    /// Iterate over each item in the container and update the corresponding key-value pair
    /// in the internal `parameters_` dictionary.
    for (auto& item : container) {
      parameters_[item.key()] = item.value();
    }
    
    /// Remove all key-value pairs from the `parameters_` dictionary.
    void clear() {
      parameters_.clear();
    }
    
    /// Check if the `parameters_` dictionary contains a specific key.
    /// Returns true if the key is found, otherwise returns false.
    bool contains(const std::string& key) const noexcept {
      return parameters_.contains(key);
    }
    
    /// Retrieve the value associated with the given `key` from the `parameters_` dictionary.
    /// Throws an exception if the key does not exist in the dictionary.
    /// Use `contains(key)` beforehand for a non-throwing alternative.
    const Tensor& get(const std::string& key) const {
      return parameters_[key];
    }
    
    /// Retrieve the value associated with the given `key` from the `parameters_` dictionary.
    /// Throws an exception if the key does not exist in the dictionary.
    /// Use `contains(key)` beforehand for a non-throwing alternative.
    Tensor& get(const std::string& key) {
      return parameters_[key];
    }
    
    /// Retrieve or update the value associated with the given `key` in the `parameters_` dictionary.
    /// Throws an exception if the key does not exist in the dictionary.
    /// Use `contains(key)` beforehand for a non-throwing alternative.
    Tensor& operator[](const std::string& key) {
      return parameters_[key];
    }
    
    /// Retrieve the value associated with the given `key` from the `parameters_` dictionary.
    /// Throws an exception if the key does not exist in the dictionary.
    /// Use `contains(key)` beforehand for a non-throwing alternative.
    const Tensor& operator[](const std::string& key) const {
      return parameters_[key];
    }
};

TORCH_MODULE(ParameterDict);

在 C++ 代码中声明了一个名为 `TORCH_MODULE` 的宏，并传入了 `ParameterDict` 作为参数。


} // namespace nn
} // namespace torch

结束了 `nn` 和 `torch` 命名空间的声明，将当前作用域从这两个命名空间中移出。
```