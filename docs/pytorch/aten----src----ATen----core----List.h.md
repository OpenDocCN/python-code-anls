# `.\pytorch\aten\src\ATen\core\List.h`

```
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <ATen/core/ivalue_to.h>
// 包含 ATen 库的 ivalue_to.h 头文件，用于 IValue 类型转换

#include <ATen/core/jit_type_base.h>
// 包含 ATen 库的 jit_type_base.h 头文件，定义了 JIT 类型基础

#include <c10/macros/Macros.h>
// 包含 c10 库的 Macros.h 头文件，提供了一些宏定义

#include <c10/macros/Export.h>
// 包含 c10 库的 Export.h 头文件，定义了导出宏

#include <c10/util/TypeTraits.h>
// 包含 c10 库的 TypeTraits.h 头文件，提供了类型特性工具

#include <c10/util/TypeList.h>
// 包含 c10 库的 TypeList.h 头文件，提供了类型列表工具

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库的 intrusive_ptr.h 头文件，定义了侵入式指针

#include <c10/util/ArrayRef.h>
// 包含 c10 库的 ArrayRef.h 头文件，提供了数组引用类

#include <c10/util/Optional.h>
// 包含 c10 库的 Optional.h 头文件，提供了可选值类型

#include <vector>
// 包含标准库的 vector 头文件，提供了动态数组容器

namespace at {
class Tensor;
}
// 命名空间 at：定义了 Tensor 类

namespace c10 {
struct IValue;
// 声明了 IValue 结构体

template<class T> class List;
// 声明了 List 模板类

struct Type;
// 声明了 Type 结构体

namespace detail {

struct ListImpl final : public c10::intrusive_ptr_target {
// 定义了 ListImpl 结构体，继承自 c10::intrusive_ptr_target 类

  using list_type = std::vector<IValue>;
  // 定义了 list_type 别名，表示 std::vector<IValue> 类型的动态数组

  explicit TORCH_API ListImpl(list_type list_, TypePtr elementType_);
  // ListImpl 类的构造函数声明，使用 list_ 和 elementType_ 初始化

  list_type list;
  // list 成员变量：存储 ListImpl 对象的动态数组

  TypePtr elementType;
  // elementType 成员变量：存储 ListImpl 对象的元素类型指针

  intrusive_ptr<ListImpl> copy() const {
    return make_intrusive<ListImpl>(list, elementType);
  }
  // copy 方法：创建并返回当前 ListImpl 对象的浅拷贝

  friend TORCH_API bool operator==(const ListImpl& lhs, const ListImpl& rhs);
  // 声明了 operator== 运算符重载方法，用于比较两个 ListImpl 对象是否相等
};

}

namespace impl {

template<class T, class Iterator> class ListIterator;
// 声明了 ListIterator 模板类

template<class T, class Iterator> class ListElementReference;
// 声明了 ListElementReference 模板类

template<class T, class Iterator>
void swap(ListElementReference<T, Iterator>&& lhs, ListElementReference<T, Iterator>&& rhs) noexcept;
// 声明了 swap 函数模板，用于交换两个 ListElementReference 对象

template<class T, class Iterator>
bool operator==(const ListElementReference<T, Iterator>& lhs, const T& rhs);
// 声明了 operator== 运算符重载方法，用于比较 ListElementReference 对象和 T 类型对象是否相等

template<class T, class Iterator>
bool operator==(const T& lhs, const ListElementReference<T, Iterator>& rhs);
// 声明了 operator== 运算符重载方法，用于比较 T 类型对象和 ListElementReference 对象是否相等

template<class T>
struct ListElementConstReferenceTraits {
// 定义了 ListElementConstReferenceTraits 结构体模板

  // In the general case, we use IValue::to().
  using const_reference = typename c10::detail::ivalue_to_const_ref_overload_return<T>::type;
  // const_reference 别名：根据类型 T 确定常量引用类型

};

// There is no to() overload for std::optional<std::string>.
template<>
struct ListElementConstReferenceTraits<std::optional<std::string>> {
// 定义了特化版本的 ListElementConstReferenceTraits 结构体模板

  using const_reference = std::optional<std::reference_wrapper<const std::string>>;
  // const_reference 别名：为 std::optional<std::string> 指定特定的常量引用类型

};

template<class T, class Iterator>
class ListElementReference final {
// 定义了 ListElementReference 模板类

public:
  operator std::conditional_t<
      std::is_reference_v<typename c10::detail::
                            ivalue_to_const_ref_overload_return<T>::type>,
      const T&,
      T>() const;
  // 类型转换运算符重载：将 ListElementReference 转换为 T 类型对象的引用或值

  ListElementReference& operator=(T&& new_value) &&;
  // 移动赋值运算符重载：将新值移动赋给 ListElementReference 对象

  ListElementReference& operator=(const T& new_value) &&;
  // 赋值运算符重载：将新值赋给 ListElementReference 对象

  // assigning another ref to this assigns the underlying value
  ListElementReference& operator=(ListElementReference&& rhs) && noexcept;
  // 移动赋值运算符重载：将另一个 ListElementReference 对象的值移动给当前对象，无异常抛出保证

  const IValue& get() const& {
    return *iterator_;
  }
  // get 方法：返回当前 ListElementReference 对象引用的 IValue 对象的常量引用

  friend void swap<T, Iterator>(ListElementReference&& lhs, ListElementReference&& rhs) noexcept;
  // 声明了友元函数 swap，用于交换两个 ListElementReference 对象

  ListElementReference(const ListElementReference&) = delete;
  // 删除拷贝构造函数

  ListElementReference& operator=(const ListElementReference&) = delete;
  // 删除拷贝赋值运算符

private:
  ListElementReference(Iterator iter)
  : iterator_(iter) {}
  // 私有构造函数：通过迭代器 iter 初始化 ListElementReference 对象

  // allow moving, but only our friends (i.e. the List class) can move us
  ListElementReference(ListElementReference&&) noexcept = default;
  // 默认移动构造函数：允许移动构造 ListElementReference 对象

  ListElementReference& operator=(ListElementReference&& rhs) & noexcept {
    iterator_ = std::move(rhs.iterator_);
    return *this;
  }
  // 移动赋值运算符重载：移动赋值 ListElementReference 对象的值，返回当前对象的引用

  Iterator iterator_;
  // iterator_ 成员变量：存储 ListElementReference 对象的迭代器

};
    // 返回当前对象的引用
    return *this;
  }

  // 声明 List<T> 为友元类，允许其访问本类的私有成员
  friend class List<T>;
  // 声明 ListIterator<T, Iterator> 为友元类，允许其访问本类的私有成员
  friend class ListIterator<T, Iterator>;

  // 声明一个成员变量 iterator_，类型为 Iterator
  Iterator iterator_;
};

// vector::iterator 包装器，确保用户代码不能依赖于其底层 vector 的具体类型。
template <class T, class Iterator>
class ListIterator final {
 public:
   // C++17 友好的 std::iterator 实现
  using iterator_category = std::random_access_iterator_tag;  // 迭代器类别为随机访问迭代器
  using value_type = T;  // 值类型为 T
  using difference_type = std::ptrdiff_t;  // 差值类型为 std::ptrdiff_t
  using pointer = T*;  // 指针类型为 T*
  using reference = ListElementReference<T, Iterator>;  // 引用类型为 ListElementReference<T, Iterator>

  explicit ListIterator() = default;  // 默认构造函数
  ~ListIterator() = default;  // 默认析构函数

  ListIterator(const ListIterator&) = default;  // 拷贝构造函数
  ListIterator(ListIterator&&) noexcept = default;  // 移动构造函数
  ListIterator& operator=(const ListIterator&) = default;  // 拷贝赋值运算符
  ListIterator& operator=(ListIterator&&) noexcept = default;  // 移动赋值运算符

  ListIterator& operator++() {  // 前置自增运算符重载
      ++iterator_;
      return *this;
  }

  ListIterator operator++(int) {  // 后置自增运算符重载
      ListIterator copy(*this);
      ++*this;
      return copy;
  }

  ListIterator& operator--() {  // 前置自减运算符重载
      --iterator_;
      return *this;
  }

  ListIterator operator--(int) {  // 后置自减运算符重载
      ListIterator copy(*this);
      --*this;
      return copy;
  }

  ListIterator& operator+=(typename List<T>::size_type offset) {  // 复合赋值加法运算符重载
      iterator_ += offset;
      return *this;
  }

  ListIterator& operator-=(typename List<T>::size_type offset) {  // 复合赋值减法运算符重载
      iterator_ -= offset;
      return *this;
  }

  ListIterator operator+(typename List<T>::size_type offset) const {  // 加法运算符重载
    return ListIterator{iterator_ + offset};
  }

  ListIterator operator-(typename List<T>::size_type offset) const {  // 减法运算符重载
    return ListIterator{iterator_ - offset};
  }

  friend difference_type operator-(const ListIterator& lhs, const ListIterator& rhs) {  // 差值运算符重载为友元函数
    return lhs.iterator_ - rhs.iterator_;
  }

  ListElementReference<T, Iterator> operator*() const {  // 解引用运算符重载
    return {iterator_};
  }

  ListElementReference<T, Iterator> operator[](typename List<T>::size_type offset) const {  // 下标运算符重载
    return {iterator_ + offset};
  }

private:
  explicit ListIterator(Iterator iterator): iterator_(std::move(iterator)) {}  // 显式构造函数

  Iterator iterator_;  // 迭代器成员变量

  friend bool operator==(const ListIterator& lhs, const ListIterator& rhs) {  // 相等运算符重载为友元函数
    return lhs.iterator_ == rhs.iterator_;
  }

  friend bool operator!=(const ListIterator& lhs, const ListIterator& rhs) {  // 不等运算符重载为友元函数
    return !(lhs == rhs);
  }

  friend bool operator<(const ListIterator& lhs, const ListIterator& rhs) {  // 小于运算符重载为友元函数
    return lhs.iterator_ < rhs.iterator_;
  }

  friend bool operator<=(const ListIterator& lhs, const ListIterator& rhs) {  // 小于等于运算符重载为友元函数
    return lhs.iterator_ <= rhs.iterator_;
  }

  friend bool operator>(const ListIterator& lhs, const ListIterator& rhs) {  // 大于运算符重载为友元函数
    return lhs.iterator_ > rhs.iterator_;
  }

  friend bool operator>=(const ListIterator& lhs, const ListIterator& rhs) {  // 大于等于运算符重载为友元函数
    return lhs.iterator_ >= rhs.iterator_;
  }

  friend class ListIterator<T, typename c10::detail::ListImpl::list_type::iterator>;  // 友元类声明
  friend class List<T>;  // 友元类声明
};

template<class T> List<T> toTypedList(List<IValue> list);  // 函数模板声明
template<class T> List<IValue> toList(List<T>&& list);  // 函数模板声明
/**
 * Convert a List<T> to a List<IValue>.
 *
 * This function takes a reference to a List<T> and returns a new List<IValue>
 * containing the elements converted from type T to IValue.
 *
 * @tparam T The type of elements in the original List<T>.
 * @param list The reference to the original List<T> to convert.
 * @return List<IValue> A new List<IValue> containing elements converted from type T.
 */
template<class T>
List<IValue> toList(const List<T>& list);

/**
 * Get a pointer to the first element of a List<IValue>.
 *
 * This function returns a pointer to the first element of the given List<IValue>.
 *
 * @param list The List<IValue> from which to retrieve the pointer to the first element.
 * @return const IValue* A pointer to the first element of the List<IValue>.
 */
const IValue* ptr_to_first_element(const List<IValue>& list);
/**
 * An object of this class stores a list of values of type T.
 *
 * This is a pointer type. After a copy, both Lists
 * will share the same storage:
 *
 * > List<int> a;
 * > List<int> b = a;
 * > b.push_back("three");
 * > ASSERT("three" == a.get(0));
 *
 * We use this class in the PyTorch kernel API instead of
 * std::vector<T>, because that allows us to do optimizations
 * and switch out the underlying list implementation without
 * breaking backwards compatibility for the kernel API.
 */
template<class T>
class List final {
private:
  // This is an intrusive_ptr because List is a pointer type.
  // Invariant: This will never be a nullptr, there will always be a valid
  // ListImpl.
  c10::intrusive_ptr<c10::detail::ListImpl> impl_;

  using internal_reference_type = impl::ListElementReference<T, typename c10::detail::ListImpl::list_type::iterator>;
  using internal_const_reference_type = typename impl::ListElementConstReferenceTraits<T>::const_reference;

private:
  explicit List(c10::intrusive_ptr<c10::detail::ListImpl>&& elements);
  explicit List(const c10::intrusive_ptr<c10::detail::ListImpl>& elements);
  friend struct IValue;
  template<class T_> friend List<T_> impl::toTypedList(List<IValue>);
  template<class T_> friend List<IValue> impl::toList(List<T_>&&);
  template<class T_> friend List<IValue> impl::toList(const List<T_>&);
  friend const IValue* impl::ptr_to_first_element(const List<IValue>& list);
};

namespace impl {
// GenericList is how IValue stores lists. It is, however, not part of the
// public API. Kernels should use Lists with concrete types instead
// (maybe except for some internal prim ops).
using GenericList = List<IValue>;

}
}

namespace torch {
  template<class T> using List = c10::List<T>;
}

#include <ATen/core/List_inl.h>  // IWYU pragma: keep
/**
 * This pragma ensures that the List_inl.h file is kept in the include chain
 * regardless of whether it's directly used by this header file or not.
 */
```