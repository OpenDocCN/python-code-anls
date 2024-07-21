# `.\pytorch\aten\src\ATen\core\Dict.h`

```
#pragma once
// 防止头文件被多次包含

#include <c10/macros/Macros.h>
// 引入 C10 库中的宏定义

#include <c10/macros/Export.h>
// 引入 C10 库中的导出宏定义

#include <c10/util/TypeTraits.h>
// 引入 C10 库中的类型特性工具

#include <c10/util/TypeList.h>
// 引入 C10 库中的类型列表工具

#include <c10/util/intrusive_ptr.h>
// 引入 C10 库中的侵入式指针工具

#include <c10/util/order_preserving_flat_hash_map.h>
// 引入 C10 库中的保序扁平哈希映射工具

#include <c10/util/Optional.h>
// 引入 C10 库中的可选类型工具

#include <ATen/core/TensorBody.h>
// 引入 ATen 库中的张量体定义

#include <ATen/core/jit_type_base.h>
// 引入 ATen 库中的 JIT 类型基础定义

namespace c10 {
// 命名空间 c10，包含 C10 库的组件

struct IValue;
// 声明 IValue 结构体，未提供具体实现

template<class Key, class Value> class Dict;
// 声明模板类 Dict，用于关联键值对

struct Type;
// 声明 Type 结构体，未提供具体实现

namespace impl {
// 命名空间 impl，包含 C10 库的实现细节

using valid_dict_key_types = guts::typelist::typelist<
  int64_t,
  std::string,
  double,
  c10::complex<double>,
  bool,
  at::Tensor
>;
// 定义有效的字典键类型列表，包括整数、字符串、浮点数、复数、布尔值和张量

}

namespace detail {
// 命名空间 detail，包含 C10 库的详细实现

struct DictKeyHash {
  size_t operator()(const IValue& ivalue) const;
  // 声明操作符重载函数，用于计算 IValue 类型对象的哈希值
};

struct DictKeyEqualTo {
  bool operator()(const IValue& lhs, const IValue& rhs) const;
  // 声明操作符重载函数，用于比较两个 IValue 类型对象是否相等
};

struct DictImpl final : public c10::intrusive_ptr_target {
  // 定义 DictImpl 结构体，继承自侵入式指针的目标类

  using dict_map_type = ska_ordered::order_preserving_flat_hash_map<IValue, IValue, DictKeyHash, DictKeyEqualTo>;
  // 使用保序扁平哈希映射作为字典映射类型

  struct DictElementTypes final {
    TypePtr keyType;
    TypePtr valueType;
    // 定义字典元素类型结构体，包含键和值的类型指针
  };

  explicit DictImpl(dict_map_type dict_, DictElementTypes elementTypes_)
  : dict(std::move(dict_))
  , elementTypes(std::move(elementTypes_)) {}
  // 定义 DictImpl 结构体的构造函数，初始化字典和元素类型

  dict_map_type dict;
  // 定义字典映射对象

  DictElementTypes elementTypes;
  // 定义字典元素类型对象

  intrusive_ptr<DictImpl> copy() const;
  // 声明复制字典对象的函数

  friend TORCH_API bool operator==(const DictImpl& lhs, const DictImpl& rhs);
  // 声明比较两个 DictImpl 对象是否相等的友元函数
};

}

namespace impl {
// 再次声明命名空间 impl，用于定义 C10 库的实现细节

template<class Key, class Value, class Iterator> class DictIterator;
// 声明模板类 DictIterator，用于迭代 Dict 的键值对

/**
 * A reference to an entry in the Dict.
 * Use the `key()` and `value()` methods to read the element.
 */
template<class Key, class Value, class Iterator>
class DictEntryRef final {
// 定义模板类 DictEntryRef，表示对字典中条目的引用

public:
  explicit DictEntryRef(Iterator iterator)
  : iterator_(std::move(iterator)) {}
  // 定义构造函数，初始化迭代器

  decltype(auto) key() const {
    return iterator_->first.template to<Key>();
  }
  // 定义返回键的方法，将迭代器指向的第一个元素转换为指定类型的键

  decltype(auto) value() const {
    return iterator_->second.template to<Value>();
  }
  // 定义返回值的方法，将迭代器指向的第二个元素转换为指定类型的值

  template<class Value_>
  void setValue(Value_&& value) const {
    static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of setValue()");
    iterator_->second = Value(std::forward<Value_>(value));
  }
  // 定义设置值的方法，将指定类型的值设置到迭代器指向的第二个元素中

private:
  // allow copying and moving, but only our friends (i.e. the Dict class) can do
  // it. Copying/moving this reference wrapper would be too ambiguous to allow it
  // in the public API.
  DictEntryRef(const DictEntryRef&) = default;
  DictEntryRef& operator=(const DictEntryRef&) = default;
  DictEntryRef(DictEntryRef&&) noexcept = default;
  DictEntryRef& operator=(DictEntryRef&& rhs) & noexcept = default;
  // 禁止公共 API 中的拷贝和移动，只允许字典类使用这些操作

  Iterator iterator_;
  // 定义迭代器对象

  friend class DictIterator<Key, Value, Iterator>;
  friend class Dict<Key, Value>;
  // 声明 DictIterator 和 Dict 类为友元类
};

// this wraps map_type::iterator to make sure user code can't rely
// on it being the type of the underlying map.
template<class Key, class Value, class Iterator>
class DictIterator final {
// 定义模板类 DictIterator，包装字典的迭代器，确保用户代码不能依赖于底层映射的类型。
// 声明一个公共的类定义，该类实现了一个符合 C++17 标准的迭代器
public:
   // 定义迭代器的类型为前向迭代器
  using iterator_category = std::forward_iterator_tag;
  // 定义迭代器指向的值类型为 DictEntryRef<Key, Value, Iterator>
  using value_type = DictEntryRef<Key, Value, Iterator>;
  // 定义迭代器之间的距离类型为 ptrdiff_t
  using difference_type = std::ptrdiff_t;
  // 定义迭代器的指针类型为指向 value_type 的指针
  using pointer = value_type*;
  // 定义迭代器的引用类型为对 value_type 的引用
  using reference = value_type&;

  // 默认构造函数，使用默认方式构造迭代器
  explicit DictIterator() = default;
  // 默认析构函数，使用默认方式销毁迭代器
  ~DictIterator() = default;

  // 复制构造函数，根据另一个迭代器构造当前迭代器
  DictIterator(const DictIterator& rhs): entryRef_(rhs.entryRef_) {}
  // 移动构造函数，移动另一个迭代器的资源到当前迭代器
  DictIterator(DictIterator&& rhs) noexcept: entryRef_(std::move(rhs.entryRef_)) {}
  // 复制赋值运算符重载，将另一个迭代器的资源复制给当前迭代器
  DictIterator& operator=(const DictIterator& rhs) {
    entryRef_ = rhs.entryRef_;
    return *this;
  }
  // 移动赋值运算符重载，将另一个迭代器的资源移动给当前迭代器
  DictIterator& operator=(DictIterator&& rhs) noexcept {
    entryRef_ = std::move(rhs.entryRef_);
    return *this;
  }

  // 前缀递增运算符重载，使迭代器向前移动一位
  DictIterator& operator++() {
      ++entryRef_.iterator_;
      return *this;
  }

  // 后缀递增运算符重载，使迭代器向前移动一位，返回移动前的副本
  DictIterator operator++(int) {
      DictIterator copy(*this);
      ++*this;
      return copy;
  }

  // 解引用运算符重载，返回当前迭代器所指的引用
  const DictEntryRef<Key, Value, Iterator>& operator*() const {
      return entryRef_;
  }

  // 成员访问运算符重载，返回当前迭代器所指对象的指针
  const DictEntryRef<Key, Value, Iterator>* operator->() const {
    return &entryRef_;
  }

  // 友元函数，计算两个迭代器之间的距离
  friend difference_type operator-(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.entryRef_.iterator_ - rhs.entryRef_.iterator_;
  }

private:
  // 私有构造函数，根据指定的迭代器构造当前迭代器
  explicit DictIterator(Iterator iterator): entryRef_(std::move(iterator)) {}

  // 私有成员函数，返回当前迭代器内部的迭代器
  const Iterator& get_iterator_() const {
    return entryRef_.iterator_;
  }

  // 友元函数，判断两个迭代器是否相等
  friend bool operator==(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() == rhs.get_iterator_();
  }

  // 友元函数，判断两个迭代器是否不等
  friend bool operator!=(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() != rhs.get_iterator_();
  }

  // 友元函数，判断一个迭代器是否小于另一个迭代器
  friend bool operator<(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() < rhs.get_iterator_();
  }

  // 友元函数，判断一个迭代器是否小于等于另一个迭代器
  friend bool operator<=(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() <= rhs.get_iterator_();
  }

  // 友元函数，判断一个迭代器是否大于另一个迭代器
  friend bool operator>(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() > rhs.get_iterator_();
  }

  // 友元函数，判断一个迭代器是否大于等于另一个迭代器
  friend bool operator>=(const DictIterator& lhs, const DictIterator& rhs) {
    return lhs.get_iterator_() >= rhs.get_iterator_();
  }

  // 成员变量，保存迭代器的引用
  DictEntryRef<Key, Value, Iterator> entryRef_;

  // 友元类声明，声明另一个模板类是当前类的友元
  friend class DictIterator<Key, Value, typename c10::detail::DictImpl::dict_map_type::iterator>;
  // 友元类声明，声明一个模板类是当前类的友元
  friend class Dict<Key, Value>;
};
/**
 * An object of this class stores a map from Key to Value.
 *
 * This is a pointer type. After a copy, both Dicts
 * will share the same storage:
 *
 * > Dict<int, string> a;
 * > Dict<int, string> b = a;
 * > b.insert(3, "three");
 * > ASSERT("three" == a.at(3));
 *
 * We use this class in the PyTorch kernel API because that
 * allows us to do optimizations and switch out the underlying
 * map implementation without breaking backwards compatibility
 * for the kernel API.
 */
template<class Key, class Value>
class Dict final {
private:
  // Ensure Key and Value are compatible with IValue or listed valid types
  static_assert((std::is_same_v<IValue, Key> && std::is_same_v<IValue, Value>) || guts::typelist::contains<impl::valid_dict_key_types, Key>::value, "Invalid Key type for Dict. We only support int64_t, double, bool, and string.");

  // Holds the underlying map as a ska_ordered::order_preserving_flat_hash_map
  // This is an intrusive_ptr because Dict is a pointer type.
  // Invariant: Never null; always a valid DictImpl.
  c10::intrusive_ptr<detail::DictImpl> impl_;

  // Constructor taking ownership of DictImpl
  explicit Dict(c10::intrusive_ptr<detail::DictImpl>&& impl);

  // Allow access to private members by IValue
  friend struct IValue;
  // Allow conversion to typed Dict
  template<class K, class V> friend Dict<K, V> impl::toTypedDict(Dict<IValue, IValue>);
  // Allow conversion from generic Dict
  template<class K, class V> friend Dict<IValue, IValue> impl::toGenericDict(Dict<K, V>);

};

namespace impl {
// GenericDict used internally by IValue, not part of public API
// Kernels should use Dicts with concrete types for Key and Value
using GenericDict = Dict<IValue, IValue>;

}
}

namespace torch {
  // Alias for Dict template in c10 namespace
  template<class Key, class Value> using Dict = c10::Dict<Key, Value>;
}

// Include inline definitions for Dict
#include <ATen/core/Dict_inl.h>  // IWYU pragma: keep
```