# `.\pytorch\torch\csrc\jit\python\python_list.h`

```
#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <c10/util/Optional.h>
#include <pybind11/detail/common.h>
#include <torch/csrc/utils/pybind.h>
#include <cstddef>
#include <stdexcept>

namespace torch::jit {

// 初始化脚本列表绑定到 Python 对象
void initScriptListBindings(PyObject* module);

/// An iterator over the elements of ScriptList. This is used to support
/// __iter__(), .
class ScriptListIterator final {
 public:
  // 构造函数，初始化迭代器和结束迭代器
  ScriptListIterator(
      c10::impl::GenericList::iterator iter,
      c10::impl::GenericList::iterator end)
      : iter_(iter), end_(end) {}

  // 返回迭代器当前位置的值
  IValue next();

  // 检查迭代是否完成
  bool done() const;

 private:
  c10::impl::GenericList::iterator iter_; // 当前迭代器位置
  c10::impl::GenericList::iterator end_;  // 结束迭代器位置
};

/// A wrapper around c10::List that can be exposed in Python via pybind
/// with an API identical to the Python list class. This allows
/// lists to have reference semantics across the Python/TorchScript
/// boundary.
class ScriptList final {
 public:
  // TODO: Do these make sense?
  using size_type = size_t;      // 定义大小类型
  using diff_type = ptrdiff_t;   // 定义差异类型
  using ssize_t = Py_ssize_t;    // 定义 ssize_t 类型

  // Constructor for empty lists created during slicing, extending, etc.
  // 构造函数，用于创建空列表，例如在切片、扩展等操作时使用
  ScriptList(const TypePtr& type) : list_(AnyType::get()) {
    auto list_type = type->expect<ListType>();
    list_ = c10::impl::GenericList(list_type);  // 使用列表类型创建通用列表
  }

  // Constructor for instances based on existing lists (e.g. a
  // Python instance or a list nested inside another).
  // 构造函数，用于基于现有列表实例化对象（例如 Python 实例或嵌套在其他列表中的列表）
  ScriptList(IValue data) : list_(AnyType::get()) {
    TORCH_INTERNAL_ASSERT(data.isList());
    list_ = data.toList();  // 将 IValue 转换为列表类型
  }

  // 返回列表的类型信息
  ListTypePtr type() const {
    return ListType::create(list_.elementType());
  }

  // 返回可用于重建对象的字符串表示形式
  std::string repr() const {
    std::ostringstream s;
    s << '[';
    bool f = false;
    for (auto const& elem : list_) {
      if (f) {
        s << ", ";
      }
      s << IValue(elem);
      f = true;
    }
    s << ']';
    return s.str();
  }

  // 返回列表元素的迭代器
  ScriptListIterator iter() const {
    auto begin = list_.begin();
    auto end = list_.end();
    return ScriptListIterator(begin, end);
  }

  // 将列表解释为布尔值；空列表为假，非空列表为真
  bool toBool() const {
    return !(list_.empty());
  }

  // 获取给定索引处的值
  IValue getItem(diff_type idx) {
    idx = wrap_index(idx);
    return list_.get(idx);
  };

  // 设置给定索引处的值
  void setItem(diff_type idx, const IValue& value) {
    idx = wrap_index(idx);
    return list_.set(idx, value);
  }

  // 检查列表是否包含给定值
  bool contains(const IValue& value) {
    for (const auto& elem : list_) {
      if (elem == value) {
        return true;
      }
    }
    return false;
  }

  // 删除列表中给定索引处的项目
  void delItem(diff_type idx) {
  // Wrap an index to ensure it is within valid bounds for list access.
  // This function adjusts negative indices and checks for out-of-range errors.
  diff_type wrap_index(diff_type idx) {
    auto sz = len();  // 获取列表的当前长度
    if (idx < 0) {    // 如果索引是负数，则将其转换为正数
      idx += sz;
    }

    if (idx < 0 || idx >= sz) {  // 检查索引是否超出有效范围
      throw std::out_of_range("list index out of range");
    }

    return idx;  // 返回处理后的索引
  }

  // Remove an element at the specified index from the list.
  // This function adjusts the index using wrap_index to ensure validity,
  // then erases the corresponding element from the list.
  void erase(size_type idx) {
    idx = wrap_index(idx);         // 使用 wrap_index 处理索引
    auto iter = list_.begin() + idx;  // 获取指向要删除元素的迭代器
    list_.erase(iter);             // 从列表中删除该元素
  }

  // Get the size of the list.
  // Returns the number of elements currently stored in the list.
  ssize_t len() const {
    return list_.size();  // 返回列表的当前长度
  }

  // Count the number of times a value appears in the list.
  // Returns the count of occurrences of the specified value in the list.
  ssize_t count(const IValue& value) const {
    ssize_t total = 0;  // 初始化计数器

    for (const auto& elem : list_) {  // 遍历列表中的每个元素
      if (elem == value) {           // 如果元素等于指定值
        ++total;                     // 计数器加一
      }
    }

    return total;  // 返回指定值在列表中出现的次数
  }

  // Remove the first occurrence of a value from the list.
  // Removes the first element equal to the specified value from the list.
  void remove(const IValue& value) {
    auto list = list_;  // 复制列表以进行操作

    int64_t idx = -1, i = 0;  // 初始化索引和计数器

    for (const auto& elem : list) {  // 遍历复制的列表
      if (elem == value) {          // 如果找到与指定值相等的元素
        idx = i;                    // 记录元素的索引
        break;                      // 停止查找
      }

      ++i;  // 更新计数器
    }

    if (idx == -1) {            // 如果未找到指定值的元素
      throw py::value_error();  // 抛出值错误异常
    }

    list.erase(list.begin() + idx);  // 从复制的列表中删除找到的元素
  }

  // Append a value to the end of the list.
  // Adds the specified value to the end of the list.
  void append(const IValue& value) {
    list_.emplace_back(value);  // 在列表末尾添加指定的值
  }

  // Clear the contents of the list.
  // Removes all elements from the list, leaving it empty.
  void clear() {
    list_.clear();  // 清空列表内容
  }

  // Append the contents of an iterable to the list.
  // Adds elements from the iterable to the end of the list.
  void extend(const IValue& iterable) {
    list_.append(iterable.toList());  // 将可迭代对象的元素添加到列表末尾
  }

  // Remove and return the element at the specified index from the list.
  // If no index is passed, remove and return the last element.
  // Returns the removed element.
  IValue pop(std::optional<size_type> idx = c10::nullopt) {
    IValue ret;  // 初始化返回值

    if (idx) {  // 如果传入了索引值
      idx = wrap_index(*idx);      // 使用 wrap_index 处理索引
      ret = list_.get(*idx);       // 获取索引处的元素并存储在返回值中
      list_.erase(list_.begin() + *idx);  // 从列表中删除该元素
    } else {    // 如果未传入索引值
      ret = list_.get(list_.size() - 1);  // 获取并存储最后一个元素的值
      list_.pop_back();                  // 移除列表中的最后一个元素
    }

    return ret;  // 返回删除的元素
  }

  // Insert a value before the given index.
  // Inserts the specified value into the list at the specified index.
  void insert(const IValue& value, diff_type idx) {
    // wrap_index cannot be used; idx == len() is allowed
    if (idx < 0) {          // 如果索引是负数
      idx += len();         // 调整索引为正数
    }

    if (idx < 0 || idx > len()) {  // 检查索引是否超出有效范围
      throw std::out_of_range("list index out of range");  // 抛出越界异常
    }

    list_.insert(list_.begin() + idx, value);  // 在指定索引处插入值
  }

  // A c10::List instance that holds the actual data.
  // This member variable stores the actual elements of the list.
  c10::impl::GenericList list_;
};

} // namespace torch::jit
```