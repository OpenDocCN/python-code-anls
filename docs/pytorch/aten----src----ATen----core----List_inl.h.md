# `.\pytorch\aten\src\ATen\core\List_inl.h`

```
#pragma once

#include <ATen/core/jit_type_base.h>
#include <ATen/core/ivalue.h>

namespace c10 {

// 声明一个模板函数，返回类型的指针
template<class T> decltype(auto) getTypePtr();
// 声明一个函数，将类型转换为字符串表示
std::string toString(const Type& type);

// List<T> 类的构造函数，接受移动语义的元素列表
template<class T>
List<T>::List(c10::intrusive_ptr<c10::detail::ListImpl>&& elements)
: impl_(std::move(elements)) {}

// List<T> 类的构造函数，接受常量引用的元素列表
template<class T>
List<T>::List(const c10::intrusive_ptr<c10::detail::ListImpl>& elements)
: impl_(elements) {}

// List<T> 类的默认构造函数
template<class T>
List<T>::List()
: List(make_intrusive<c10::detail::ListImpl>(
  typename c10::detail::ListImpl::list_type(),
  getTypePtr<T>())) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType) instead.");
}

// List<T> 类的构造函数，接受数组引用的值列表
template<class T>
List<T>::List(ArrayRef<T> values)
: List(make_intrusive<c10::detail::ListImpl>(
    typename c10::detail::ListImpl::list_type(),
    getTypePtr<T>())) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType).");
  impl_->list.reserve(values.size());
  for (const T& element : values) {
    impl_->list.push_back(element);
  }
}

// List<T> 类的构造函数，接受初始化列表的初始值
template<class T>
List<T>::List(std::initializer_list<T> initial_values)
: List(ArrayRef<T>(initial_values)) {
  static_assert(!std::is_same<T, IValue>::value, "This constructor is not valid for List<IValue>. Please use c10::impl::GenericList(elementType).");
}

// List<T> 类的构造函数，接受类型指针的元素类型
template<class T>
List<T>::List(TypePtr elementType)
: List(make_intrusive<c10::detail::ListImpl>(
    typename c10::detail::ListImpl::list_type(),
    std::move(elementType))) {
  static_assert(std::is_same<T, IValue>::value || std::is_same<T, c10::intrusive_ptr<ivalue::Future>>::value,
                "This constructor is only valid for c10::impl::GenericList or List<Future>.");
}

namespace impl {

// 将通用列表 impl::GenericList 转换为具体类型 List<T>
template<class T>
List<T> toTypedList(impl::GenericList list) {
  // 如果列表有其他实例（即 list.use_count() > 1），则必须保持不变，
  // 因为向上转型会允许向新列表添加可能破坏旧列表的类型。
  // 但是，如果此列表没有其他实例（即 list.use_count() == 1），则可以允许向上转型。
  // 这可以提升性能，因为可以直接将 List<T> 转换为 List<optional<T>>，而无需复制。
  // 这也用于提供与旧模型的向后兼容性，
  // 在旧模型中，序列化为 aten::index、aten::index_put、aten::index_put_ 和 aten::index_put_impl_ 的索引参数为 List<Tensor>，
  // 然后我们将该参数更改为 List<optional<Tensor>>。在反序列化时，list.use_count() == 1，
  // 可以直接将 List<Tensor> 反序列化为 List<optional<Tensor>>。
  TORCH_CHECK(*list.impl_->elementType == *getTypePtr<T>()
    || (list.use_count() == 1 && list.impl_->elementType->isSubtypeOf(*getTypePtr<T>()))
    // 报告类型不匹配错误，试图将类型为 List<elementType> 的列表转换为 List<T>。类型不匹配。
    , "Tried to cast a List<", toString(*list.impl_->elementType), "> to a List<", toString(*getTypePtr<T>()), ">. Types mismatch.");
  
  // 返回一个新的 List<T> 对象，使用 std::move 将原始列表的实现指针转移给新列表
  return List<T>(std::move(list.impl_));
}

// 定义模板函数 toList，将 List<T>&& 转换为 impl::GenericList
template<class T>
impl::GenericList toList(List<T>&& list) {
  return GenericList(std::move(list.impl_));
}

// 定义模板函数 toList，将 const List<T>& 转换为 impl::GenericList
template<class T>
impl::GenericList toList(const List<T>& list) {
  return GenericList(list.impl_);
}

}

// 实现 List<T>::copy() 函数，返回当前列表的副本
template<class T>
List<T> List<T>::copy() const {
  return List<T>(impl_->copy());
}

namespace detail {

  // 定义模板函数 list_element_to，用于从 T 转换为 T
  template<class T>
  T list_element_to(T element) {
    return element;
  }

  // 定义模板函数 list_element_to，用于从 IValue 转换为 T
  template<class T>
  T list_element_to(const IValue& element) {
    return element.template to<T>();
  }

  // 定义模板函数 list_element_to，用于从 IValue&& 转换为 T
  template<class T>
  T list_element_to(IValue&& element) {
    return std::move(element).template to<T>();
  }

  // 定义结构体 ListElementFrom<T>，提供从 T 到 IValue 的转换
  template<class T>
  struct ListElementFrom {
    // 从 const T& 转换为 IValue
    static IValue from(const T& element) {
      return element;
    }
    // 从 T&& 转换为 IValue
    static IValue from(T&& element) {
      return std::move(element);
    }
  };

  // 部分特化 ListElementFrom<IValue>，提供从 IValue 到 IValue 的转换
  template<>
  struct ListElementFrom<IValue> {
    // 从 const IValue& 转换为 const IValue&
    static const IValue& from(const IValue& element) {
      return element;
    }
    // 从 IValue&& 转换为 IValue&&
    static IValue&& from(IValue&& element) {
      return std::move(element);
    }
  };

} // namespace detail

// 实现 ListElementReference<T, Iterator> 的成员函数

template <class T, class Iterator>
ListElementReference<T, Iterator>::operator std::conditional_t<
    std::is_reference_v<typename c10::detail::ivalue_to_const_ref_overload_return<
        T>::type>,
    const T&,
    T>() const {
  return iterator_->template to<T>();
}

template<class T, class Iterator>
ListElementReference<T, Iterator>& ListElementReference<T, Iterator>::operator=(T&& new_value) && {
  *iterator_ = c10::detail::ListElementFrom<T>::from(std::move(new_value));
  return *this;
}

template<class T, class Iterator>
ListElementReference<T, Iterator>& ListElementReference<T, Iterator>::operator=(const T& new_value) && {
  *iterator_ = c10::detail::ListElementFrom<T>::from(new_value);
  return *this;
}

template<class T, class Iterator>
ListElementReference<T, Iterator>& ListElementReference<T, Iterator>::operator=(ListElementReference<T, Iterator>&& rhs) && noexcept {
  *iterator_ = *rhs.iterator_;
  return *this;
}

template<class T, class Iterator>
void swap(ListElementReference<T, Iterator>&& lhs, ListElementReference<T, Iterator>&& rhs)  noexcept {
  std::swap(*lhs.iterator_, *rhs.iterator_);
}

template<class T, class Iterator>
bool operator==(const ListElementReference<T, Iterator>& lhs, const T& rhs) {
  const T& lhs_tmp = lhs;
  return lhs_tmp == rhs;
}

template<class T, class Iterator>
inline bool operator==(const T& lhs, const ListElementReference<T, Iterator>& rhs) {
  return rhs == lhs;
}

// 实现 list_element_to_const_ref 函数模板

template<class T>
inline typename ListElementConstReferenceTraits<T>::const_reference
list_element_to_const_ref(const IValue& element) {
  return element.template to<T>();
}

// 部分特化 list_element_to_const_ref，处理 std::optional<std::string> 类型
template<>
inline typename ListElementConstReferenceTraits<std::optional<std::string>>::const_reference
list_element_to_const_ref<std::optional<std::string>>(const IValue& element) {
  return element.toOptionalStringRef();
}

} // namespace impl
// 设置列表中指定位置的元素为给定值，通过调用ListElementFrom的静态方法进行类型转换
template<class T>
void List<T>::set(size_type pos, const value_type& value) const {
  impl_->list.at(pos) = c10::detail::ListElementFrom<T>::from(value);
}

// 设置列表中指定位置的元素为移动构造的值，通过调用ListElementFrom的静态方法进行类型转换
template<class T>
void List<T>::set(size_type pos, value_type&& value) const {
  impl_->list.at(pos) = c10::detail::ListElementFrom<T>::from(std::move(value));
}

// 获取列表中指定位置的元素的常量引用，并调用operator[]实现
template<class T>
typename List<T>::internal_const_reference_type List<T>::get(size_type pos) const {
  return operator[](pos);
}

// 获取列表中指定位置的元素的常量引用
template<class T>
typename List<T>::internal_const_reference_type List<T>::operator[](size_type pos) const {
  return c10::impl::list_element_to_const_ref<T>(impl_->list.at(pos));
}

// 获取列表中指定位置的元素的引用，如果超出范围则抛出异常
template<class T>
typename List<T>::internal_reference_type List<T>::operator[](size_type pos) {
  static_cast<void>(impl_->list.at(pos)); // Throw the exception if it is out of range.
  return {impl_->list.begin() + static_cast<typename decltype(impl_->list)::difference_type>(pos)};
}

// 提取列表中指定位置的元素并返回，同时将该位置的元素重置为T()，保持正确的类型
template<class T>
typename List<T>::value_type List<T>::extract(size_type pos) const {
  auto& elem = impl_->list.at(pos);
  auto result = c10::detail::list_element_to<T>(std::move(elem));
  elem = c10::detail::ListElementFrom<T>::from(T{}); // Reset the list element to a T() instead of None to keep it correctly typed
  return result;
}

// 返回指向列表开头的迭代器
template<class T>
typename List<T>::iterator List<T>::begin() const {
  return iterator(impl_->list.begin());
}

// 返回指向列表末尾的迭代器
template<class T>
typename List<T>::iterator List<T>::end() const {
  return iterator(impl_->list.end());
}

// 检查列表是否为空
template<class T>
bool List<T>::empty() const {
  return impl_->list.empty();
}

// 返回列表中元素的数量
template<class T>
typename List<T>::size_type List<T>::size() const {
  return impl_->list.size();
}

// 预留列表中的空间以容纳指定数量的元素
template<class T>
void List<T>::reserve(size_type new_cap) const {
  impl_->list.reserve(new_cap);
}

// 清空列表中的所有元素
template<class T>
void List<T>::clear() const {
  impl_->list.clear();
}

// 在指定位置插入元素的常规引用版本，并返回插入后的迭代器
template<class T>
typename List<T>::iterator List<T>::insert(iterator pos, const T& value) const {
  return iterator { impl_->list.insert(pos.iterator_, c10::detail::ListElementFrom<T>::from(value)) };
}

// 在指定位置插入元素的移动构造版本，并返回插入后的迭代器
template<class T>
typename List<T>::iterator List<T>::insert(iterator pos, T&& value) const {
  return iterator { impl_->list.insert(pos.iterator_, c10::detail::ListElementFrom<T>::from(std::move(value))) };
}

// 在指定位置以完美转发的方式插入元素，并返回插入后的迭代器
template<class T>
template<class... Args>
typename List<T>::iterator List<T>::emplace(iterator pos, Args&&... value) const {
  // TODO Use list_element_from?
  return iterator { impl_->list.emplace(pos.iterator_, std::forward<Args>(value)...) };
}

// 将元素添加到列表末尾的常规引用版本
template<class T>
void List<T>::push_back(const T& value) const {
  impl_->list.push_back(c10::detail::ListElementFrom<T>::from(value));
}

// 将元素添加到列表末尾的移动构造版本
template<class T>
void List<T>::push_back(T&& value) const {
  impl_->list.push_back(c10::detail::ListElementFrom<T>::from(std::move(value)));
}

// 将另一个列表的所有元素追加到当前列表末尾，如果引用计数为1，则执行
template<class T>
void List<T>::append(List<T> b) const {
  if (b.use_count() == 1) {
    # 如果当前对象的实现指针指向的列表为空，则执行以下操作：
    impl_->list.insert(impl_->list.end(), make_move_iterator(b.impl_->list.begin()), make_move_iterator(b.impl_->list.end()));
    # 使用移动迭代器，将对象 b 的实现指针指向列表的元素从开头到结尾移动插入到当前对象的实现指针指向的列表末尾。
    # 否则，如果当前对象的实现指针指向的列表非空，则执行以下操作：
    impl_->list.insert(impl_->list.end(), b.impl_->list.begin(), b.impl_->list.end());
    # 将对象 b 的实现指针指向列表的元素从开头到结尾插入到当前对象的实现指针指向的列表末尾。
// 结束模板类 List<T> 的命名空间
}

template<class T>
template<class... Args>
void List<T>::emplace_back(Args&&... args) const {
  // TODO Use list_element_from?
  // 使用 list_element_from 吗？待办事项，暂时未实现
  impl_->list.push_back(T(std::forward<Args>(args)...));
  // 在列表的末尾添加一个元素，使用完美转发构造 T 类型的对象并加入列表
}

template<class T>
typename List<T>::iterator List<T>::erase(iterator pos) const {
  // 从列表中删除指定位置的元素，并返回指向下一个元素的迭代器
  return iterator { impl_->list.erase(pos.iterator_) };
}

template<class T>
typename List<T>::iterator List<T>::erase(iterator first, iterator last) const {
  // 从列表中删除指定范围的元素 [first, last)，并返回指向 last 之后元素的迭代器
  return iterator { impl_->list.erase(first.iterator_, last.iterator_) };
}

template<class T>
void List<T>::pop_back() const {
  // 移除列表末尾的元素
  impl_->list.pop_back();
}

template<class T>
void List<T>::resize(size_type count) const {
  // 调整列表的大小为 count，如果当前大小比 count 小，则填充默认构造的 T 类型对象
  impl_->list.resize(count, T{});
}

template<class T>
void List<T>::resize(size_type count, const T& value) const {
  // 调整列表的大小为 count，如果当前大小比 count 小，则填充指定值 value
  impl_->list.resize(count, value);
}

template<class T>
bool operator==(const List<T>& lhs, const List<T>& rhs) {
  // 判断两个 List<T> 是否相等
  // 如果两者的实现指针相同，则认为相等
  if (lhs.impl_ == rhs.impl_) {
    return true;
  }

  // 否则，直接比较值
  return *lhs.impl_ == *rhs.impl_;
}

template<class T>
bool operator!=(const List<T>& lhs, const List<T>& rhs) {
  // 判断两个 List<T> 是否不相等
  return !(lhs == rhs);
}

template<class T>
bool List<T>::is(const List<T>& rhs) const {
  // 检查当前 List<T> 的实现指针是否与 rhs 相同
  return this->impl_ == rhs.impl_;
}

template<class T>
std::vector<T> List<T>::vec() const {
  // 将列表中的元素转换为 std::vector<T> 并返回
  std::vector<T> result(begin(), end());
  return result;
}

template<class T>
size_t List<T>::use_count() const {
  // 返回当前 List<T> 的实现指针的引用计数
  return impl_.use_count();
}

template <class T>
TypePtr List<T>::elementType() const {
  // 返回当前 List<T> 的元素类型指针
  return impl_->elementType;
}

template <class T>
void List<T>::unsafeSetElementType(TypePtr t) {
  // 不安全地设置当前 List<T> 的元素类型指针
  impl_->elementType = std::move(t);
}

// 结束模板类 List<T> 的实现
```