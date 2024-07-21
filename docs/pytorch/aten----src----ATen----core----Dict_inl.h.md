# `.\pytorch\aten\src\ATen\core\Dict_inl.h`

```
#pragma once
// 使用 #pragma once 防止头文件被多次包含

#include <ATen/core/ivalue.h>
// 包含 ATen 库中的 IValue 头文件
#include <c10/util/hash.h>
// 包含 c10 库中的 hash 头文件

namespace c10 {
namespace detail {
// 定义在 c10 命名空间中的 detail 命名空间

inline bool DictKeyEqualTo::operator()(const IValue& lhs, const IValue& rhs) const {
  // 实现 DictKeyEqualTo 的成员函数 operator()，比较两个 IValue 是否相等
  if (lhs.isTensor() && rhs.isTensor()) {
    // 如果 lhs 和 rhs 都是 Tensor，只比较它们的标识（按照 Python 的方式）
    return lhs.is(rhs);
  }
  // 否则，首先通过标识比较以提高效率，然后再比较值（参见 [container equality]）
  return _fastEqualsForContainer(lhs, rhs);
}
}

template<class T> decltype(auto) getTypePtr();
// 声明一个模板函数 getTypePtr，返回类型的指针或引用

std::string toString(const Type& type);
// 声明一个函数 toString，接受 Type 类型的参数并返回其字符串表示

namespace impl {
// 定义在 impl 命名空间中

template<class Key, class Value>
Dict<Key, Value> toTypedDict(GenericDict dict) {
  // 将 GenericDict 转换为特定类型的 Dict<Key, Value>，并进行类型断言
  TORCH_INTERNAL_ASSERT(*getTypePtr<Key>() == *dict.impl_->elementTypes.keyType, "Tried to cast a Dict<", toString(*dict.impl_->elementTypes.keyType), ", ", toString(*dict.impl_->elementTypes.valueType) ,"> to a Dict<", toString(*getTypePtr<Key>()), ", ", toString(*getTypePtr<Value>()), ">. Key types mismatch.");
  TORCH_INTERNAL_ASSERT(*getTypePtr<Value>() == *dict.impl_->elementTypes.valueType, "Tried to cast a Dict<", toString(*dict.impl_->elementTypes.keyType), ", ", toString(*dict.impl_->elementTypes.valueType) ,"> to a Dict<", toString(*getTypePtr<Key>()), ", ", toString(*getTypePtr<Value>()), ">. Value types mismatch.");

  return Dict<Key, Value>(std::move(dict.impl_));
}

template<class Key, class Value>
GenericDict toGenericDict(Dict<Key, Value> dict) {
  // 将特定类型的 Dict<Key, Value> 转换为 GenericDict
  return GenericDict(std::move(dict.impl_));
}
}

namespace detail {
// 定义在 detail 命名空间中

inline size_t DictKeyHash::operator()(const IValue& ivalue) const {
  // 实现 DictKeyHash 的成员函数 operator()，计算给定 IValue 的哈希值
  if (ivalue.isInt()) {
    return std::hash<int64_t>()(ivalue.toInt());
  } else if (ivalue.isString()) {
    return std::hash<c10::string_view>()(ivalue.toStringView());
  } else if (ivalue.isDouble()) {
    return std::hash<double>()(ivalue.toDouble());
  } else if (ivalue.isComplexDouble()) {
    return c10::hash<c10::complex<double>>()(ivalue.toComplexDouble());
  } else if (ivalue.isBool()) {
    return std::hash<bool>()(ivalue.toBool());
  } else if (ivalue.isTensor()) {
    return std::hash<TensorImpl*>()(ivalue.toTensor().unsafeGetTensorImpl());
  } else if (ivalue.isDevice()) {
    return std::hash<Device>()(ivalue.toDevice());
  } else {
    // 如果无法处理特定类型的 IValue，抛出运行时异常
    throw std::runtime_error(
        "Can't hash IValues with tag '" + ivalue.tagKind() + "'");
  }
}

inline intrusive_ptr<DictImpl> DictImpl::copy() const {
  // 实现 DictImpl 的成员函数 copy()，返回一个 DictImpl 的复制品
  return make_intrusive<DictImpl>(dict, elementTypes);
}

}

template<class Key, class Value>
// 模板声明，定义一个模板类型的函数或类
// 默认构造函数，初始化一个空的字典
template<class Key, class Value>
Dict<Key, Value>::Dict()
  :Dict(make_intrusive<detail::DictImpl>(
      detail::DictImpl::dict_map_type(),
      detail::DictImpl::DictElementTypes{getTypePtr<Key>(), getTypePtr<Value>()})) {
  // 确保 Key 和 Value 不是 IValue 类型，否则抛出静态断言异常
  static_assert(!std::is_same<Key, IValue>::value, "This constructor is not valid for Dict<IValue, _>. Please use c10::impl::GenericDict(keyType, valueType) instead.");
  static_assert(!std::is_same<Value, IValue>::value, "This constructor is not valid for Dict<_, IValue>. Please use c10::impl::GenericDict(keyType, valueType) instead.");
}

// 带类型指针参数的构造函数，初始化字典
template<class Key, class Value>
Dict<Key, Value>::Dict(TypePtr keyType, TypePtr valueType)
: Dict(make_intrusive<detail::DictImpl>(
    detail::DictImpl::dict_map_type(),
    detail::DictImpl::DictElementTypes {std::move(keyType), std::move(valueType)})) {
  // 确保 Key 和 Value 类型为 IValue，否则抛出静态断言异常
  static_assert(std::is_same<Key, IValue>::value, "This constructor is only valid for c10::impl::GenericDict.");
  static_assert(std::is_same<Value, IValue>::value, "This constructor is only valid for c10::impl::GenericDict.");
}

// 构造函数，接受指向 DictImpl 的智能指针
template<class Key, class Value>
Dict<Key, Value>::Dict(c10::intrusive_ptr<detail::DictImpl>&& impl): impl_(std::move(impl)) {}

// 复制函数，返回当前字典的副本
template<class Key, class Value>
Dict<Key, Value> Dict<Key, Value>::copy() const {
  return Dict<Key, Value>(impl_->copy());
}

// 返回字典的起始迭代器
template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::begin() const {
  return iterator{impl_->dict.begin()};
}

// 返回字典的结束迭代器
template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::end() const {
  return iterator{impl_->dict.end()};
}

// 检查字典是否为空
template<class Key, class Value>
bool Dict<Key, Value>::empty() const {
  return impl_->dict.empty();
}

// 返回字典中元素的数量
template<class Key, class Value>
typename Dict<Key, Value>::size_type Dict<Key, Value>::size() const {
  return impl_->dict.size();
}

// 清空字典中的所有元素
template<class Key, class Value>
void Dict<Key, Value>::clear() const {
  impl_->dict.clear();
}

// 插入或更新元素到字典中
template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert(Key_&& key, Value_&& value) const {
  // 确保 key 和 value 的类型能够构造成 Key 和 Value 类型，否则抛出静态断言异常
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert");
  // 在字典中插入元素并返回插入的迭代器和是否成功的标志
  auto inserted = impl_->dict.emplace(
      Key(std::forward<Key_>(key)),
      Value(std::forward<Value_>(value)));
  return {iterator{inserted.first}, inserted.second};
}

// 插入或更新元素到字典中，如果已存在则更新其值
template<class Key, class Value>
template<class Key_, class Value_>
std::pair<typename Dict<Key, Value>::iterator, bool> Dict<Key, Value>::insert_or_assign(Key_&& key, Value_&& value) const {
  // 确保 key 和 value 的类型能够构造成 Key 和 Value 类型，否则抛出静态断言异常
  static_assert(std::is_constructible<Key, Key_>::value, "Wrong type for the key argument of Dict::insert_or_assign");
  static_assert(std::is_constructible<Value, Value_>::value, "Wrong type for the value argument of Dict::insert_or_assign");
  // 插入或更新元素到字典中并返回插入的迭代器和是否成功的标志
  auto inserted = impl_->dict.insert_or_assign(
      //
    Key(std::forward<Key_>(key)),
    Value(std::forward<Value_>(value)));

初始化 `Key` 和 `Value` 对象，使用了完美转发（perfect forwarding），将参数 `key` 和 `value` 转发给构造函数。


  return {iterator{inserted.first}, inserted.second};

返回一个包含两个元素的 `std::pair` 对象，其中第一个元素是一个迭代器，指向刚插入的元素；第二个元素是一个布尔值，表示插入是否成功（true 表示成功，false 表示已存在相同的键）。
}

template<class Key, class Value>
void Dict<Key, Value>::erase(iterator iter) const {
    // 使用实现类的字典对象，通过迭代器删除条目
    impl_->dict.erase(iter.entryRef_.iterator_);
}

template<class Key, class Value>
C10_NODISCARD size_t Dict<Key, Value>::erase(const Key& key) const {
    // 使用实现类的字典对象，通过键删除条目，并返回删除的条目数
    return impl_->dict.erase(key);
}

template<class Key, class Value>
Value Dict<Key, Value>::at(const Key& key) const {
    // 返回实现类的字典对象中特定键对应的值，并将其转换为指定类型的值
    return impl_->dict.at(key).template to<Value>();
}

template<class Key, class Value>
typename Dict<Key, Value>::iterator Dict<Key, Value>::find(const Key& key) const {
    // 在实现类的字典对象中查找特定键，并返回对应的迭代器
    return iterator{impl_->dict.find(key)};
}

template<class Key, class Value>
bool Dict<Key, Value>::contains(const Key& key) const {
    // 检查实现类的字典对象中是否包含特定键
    return end() != find(key);
}

template<class Key, class Value>
void Dict<Key, Value>::reserve(size_type count) const {
    // 在实现类的字典对象中预留足够的空间来存储指定数量的元素
    impl_->dict.reserve(count);
}

template<class Key, class Value>
TypePtr Dict<Key, Value>::keyType() const {
    // 返回实现类的元素类型中键的类型
    return impl_->elementTypes.keyType;
}

template<class Key, class Value>
TypePtr Dict<Key, Value>::valueType() const {
    // 返回实现类的元素类型中值的类型
    return impl_->elementTypes.valueType;
}

template <class Key, class Value>
void Dict<Key, Value>::unsafeSetKeyType(TypePtr t) {
    // 设置实现类的元素类型中键的类型，不安全地移动类型指针
    impl_->elementTypes.keyType = std::move(t);
}

template <class Key, class Value>
void Dict<Key, Value>::unsafeSetValueType(TypePtr t) {
    // 设置实现类的元素类型中值的类型，不安全地移动类型指针
    impl_->elementTypes.valueType = std::move(t);
}

template <class Key_, class Value_>
bool operator==(const Dict<Key_, Value_>& lhs, const Dict<Key_, Value_>& rhs) {
    // 比较两个字典是否相等，如果实现类相同则直接相等，否则比较它们的实际内容
    if (lhs.impl_ == rhs.impl_) {
        return true;
    }
    return *lhs.impl_ == *rhs.impl_;
}

template <class Key_, class Value_>
bool operator!=(const Dict<Key_, Value_>& lhs, const Dict<Key_, Value_>& rhs) {
    // 检查两个字典是否不相等，通过调用相等运算符实现
    return !(lhs == rhs);
}

template <class Key, class Value>
bool Dict<Key, Value>::is(const Dict& rhs) const {
    // 检查当前字典是否与另一个字典是同一个实例
    return this->impl_ == rhs.impl_;
}
```