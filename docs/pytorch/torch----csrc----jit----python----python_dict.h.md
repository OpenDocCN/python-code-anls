# `.\pytorch\torch\csrc\jit\python\python_dict.h`

```py
#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::jit {

// 初始化脚本字典绑定到 Python 对象的函数
void initScriptDictBindings(PyObject* module);

/// An iterator over the keys of ScriptDict. This is used to support
/// .keys() and iteration.
class ScriptDictKeyIterator final {
 public:
  // 构造函数，初始化迭代器的起始和结束位置
  ScriptDictKeyIterator(
      c10::impl::GenericDict::iterator iter,
      c10::impl::GenericDict::iterator end)
      : iter_(std::move(iter)), end_(std::move(end)) {}
  // 返回下一个键的 IValue
  IValue next();

 private:
  c10::impl::GenericDict::iterator iter_;  // 当前迭代器位置
  c10::impl::GenericDict::iterator end_;   // 迭代器的结束位置
};

/// An iterator over the key-value pairs of ScriptDict. This is used to support
/// .items().
class ScriptDictIterator final {
 public:
  // 构造函数，初始化迭代器的起始和结束位置
  ScriptDictIterator(
      c10::impl::GenericDict::iterator iter,
      c10::impl::GenericDict::iterator end)
      : iter_(std::move(iter)), end_(std::move(end)) {}
  // 返回下一个键值对的 IValue
  IValue next();

 private:
  c10::impl::GenericDict::iterator iter_;  // 当前迭代器位置
  c10::impl::GenericDict::iterator end_;   // 迭代器的结束位置
};

/// A wrapper around c10::Dict that can be exposed in Python via pybind
/// with an API identical to the Python dictionary class. This allows
/// dictionaries to have reference semantics across the Python/TorchScript
/// boundary.
class ScriptDict final {
 public:
  // Constructor.
  // 使用给定的 IValue 数据构造 ScriptDict 对象
  ScriptDict(IValue data) : dict_(AnyType::get(), AnyType::get()) {
    TORCH_INTERNAL_ASSERT(data.isGenericDict());
    dict_ = data.toGenericDict();
  }

  // Get the type of the dictionary.
  // 返回字典的类型信息
  DictTypePtr type() const {
    return DictType::create(dict_.keyType(), dict_.valueType());
  }

  // Return a string representation that can be used
  // to reconstruct the instance.
  // 返回用于重建对象的字符串表示形式
  std::string repr() const {
    std::ostringstream s;
    s << '{';
    bool f = false;
    for (auto const& kv : dict_) {
      if (f) {
        s << ", ";
      }
      s << kv.key() << ": " << kv.value();
      f = true;
    }
    s << '}';
    return s.str();
  }

  // Return an iterator over the keys of the dictionary.
  // 返回字典键的迭代器
  ScriptDictKeyIterator iter() const {
    auto begin = dict_.begin();
    auto end = dict_.end();
    return ScriptDictKeyIterator(begin, end);
  }

  // Return an iterator over the key-value pairs of the dictionary.
  // 返回字典键值对的迭代器
  ScriptDictIterator items() const {
    auto begin = dict_.begin();
    auto end = dict_.end();
    return ScriptDictIterator(begin, end);
  }

  // Interpret the dictionary as a boolean; empty means false, non-empty means
  // true.
  // 将字典解释为布尔值，空为 false，非空为 true
  bool toBool() const {
    return !(dict_.empty());
  }

  // Get the value for the given key. Throws std::out_of_range if the key does
  // not exist.
  // 根据给定的键获取对应的值，如果键不存在则抛出 std::out_of_range 异常
  IValue getItem(const IValue& key) {
    return dict_.at(key);
  };

  // Set the value for the given key.
  // 设置给定键的值
  void setItem(const IValue& key, const IValue& value) {
    dict_.insert_or_assign(key, value);
  };

  // Check whether the dictionary contains the given key.
  // 检查字典是否包含给定的键
  bool contains(const IValue& key) {
    // Implementation not provided here
    // 检查字典中是否包含指定的键，返回布尔值
    return dict_.contains(key);
  }

  // 从字典中删除给定的键
  bool delItem(const IValue& key) {
    // 调用 erase 方法从字典中删除指定的键，并返回删除操作的结果
    return dict_.erase(key);
  }

  // 获取字典的大小
  int64_t len() const {
    // 返回字典中键值对的数量
    return dict_.size();
  }

  // 存储实际数据的 c10::Dict 实例
  c10::impl::GenericDict dict_;
};

// 结束 torch::jit 命名空间的定义
} // namespace torch::jit
```