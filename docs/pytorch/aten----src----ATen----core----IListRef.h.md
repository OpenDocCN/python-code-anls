# `.\pytorch\aten\src\ATen\core\IListRef.h`

```py
#pragma once
// 保证头文件只被编译一次

#include <ATen/core/ivalue_to.h>
// 导入 ATen 库的 ivalue_to 头文件
#include <c10/util/ArrayRef.h>
// 导入 c10 库的 ArrayRef 头文件
#include <c10/util/Exception.h>
// 导入 c10 库的 Exception 头文件

#include <functional>
// 导入 C++ 标准库的 functional 头文件
#include <initializer_list>
// 导入 C++ 标准库的 initializer_list 头文件
#include <iterator>
// 导入 C++ 标准库的 iterator 头文件
#include <type_traits>
// 导入 C++ 标准库的 type_traits 头文件

namespace c10 {
// 进入 c10 命名空间

template <typename T>
// 模板声明，T 为模板参数
class IListRef;
// 声明模板类 IListRef

/*
 * Applies arbitrary macros to each `IListRefTag`.
 */
#define TORCH_ILISTREF_FORALL_TAGS(_, ...) \
  _(Unboxed, ##__VA_ARGS__)                \
  _(Boxed, ##__VA_ARGS__)                  \
  _(Materialized, ##__VA_ARGS__)
// 宏定义，对每个 IListRefTag 应用任意的宏

/*
 * Defines a "switch-case" for `TAG`. Inside, it executes `BODY`,
 * while bringing to scope:
 *
 * - `ImplT`: the implementation class for `TAG`
 * - `this_`: the result of unwrapping `this`
 */
#define TORCH_ILISTREF_UNWRAP_CASE(TAG, BODY)                        \
  case c10::IListRefTag::TAG: {                                      \
    using ImplT = c10::detail::IListRefTagImpl<IListRefTag::TAG, T>; \
    auto& this_ = ImplT::unwrap(*this);                              \
    BODY                                                             \
  } break;
// 宏定义，为 TAG 定义一个 switch-case，执行 BODY，在作用域内引入 ImplT 和 this_

/*
 * Dispatches the unwrap call, depending on `TAG`, followed by
 * the execution of `BODY`. It aborts if `TAG` is not a `IListRefTag`.
 *
 * This macro is useful because it allows us to handle different
 * types (that correspond to different tags) to be implemented
 * only once. We can do it even when the implementation of the
 * different tags aren't syntatically the same, by dispatching
 * it to a function (e.g. `ImplT::<dispatch-function>(this_)`).
 */
#define TORCH_ILISTREF_UNWRAP(TAG, BODY)                         \
  switch (TAG) {                                                 \
    TORCH_ILISTREF_FORALL_TAGS(TORCH_ILISTREF_UNWRAP_CASE, BODY) \
    break;                                                       \
    default:                                                     \
      TORCH_INTERNAL_ASSERT(false, "invalid IListRef tag.");     \
  }
// 宏定义，根据 TAG 分派 unwrap 调用，然后执行 BODY。如果 TAG 不是 IListRefTag，则中止。

enum class IListRefTag {
#define DEFINE_TAG(tag, ...) tag,
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
      None
};
// 枚举类 IListRefTag，列举所有可能的列表引用标签

namespace detail {
// 进入 detail 命名空间

/*
 * Type alias that specifies whether we return a reference or a copy of `T`.
 *
 * What is this for?
 * =================
 * Since values in the boxed world are represented by an `IValue`, we also
 * depend on whether it can be converted to a const-reference (`Tensor`) or
 * has to create a new copy of `T` (`OptionalTensorRef`).
 */
template <typename T>
// 模板声明，T 为模板参数
using IListRefConstRef = typename ivalue_to_const_ref_overload_return<T>::type;
// 定义类型别名，指定我们是返回 T 的引用还是副本
/*
 * Interface that implements key functions for each `IListRefTag` type.
 *
 * What is this for?
 * =================
 * Given an `IListRef(Iterator)<T>`, some methods have to be implemented
 * differently for each `TAG`. Therefore, the methods inside this class
 * are used as dispatch targets for the different `IListRefTag` values.
 *
 * You should create an specialization of this class for each possible
 * combination of `IListRefTag` type (except `None`) and element types
 * (e.g. `Tensor`).
 *
 * What does it do?
 * ================
 * 1. defines static methods to be used as dispatch targets by both
 *    `IListRef<T>` and `IListRefIterator<T>` (see the implementation of
 *    `IListRefTagImplBase`).
 *
 * 2. defines the `elem_type` and `list_type` aliases that will be
 *    used in the definition of `IListRef<T>`. In general, we should do
 *    so by inheriting from `IListRefTagImplBase<TAG, T, ListElemT>`.
 *
 * [Note: IListRefTagImpl Specialization]
 * ======================================
 * For `IListRef(Iterator)<at::Tensor>`:
 * - <IListRefTag::Unboxed, at::Tensor>
 * - <IListRefTag::Boxed, at::Tensor>
 * - <IListRefTag::Materialized, at::Tensor>
 *
 * For `IListRef(Iterator)<at::OptionalTensorRef>`:
 * - <IListRefTag::Unboxed, at::OptionalTensorRef>
 * - <IListRefTag::Boxed, at::OptionalTensorRef>
 * - <IListRefTag::Materialized, at::OptionalTensorRef>
 */
template <IListRefTag TAG, typename T>
class IListRefTagImpl {};

/*
 * Base implementation of `IListRefTagImpl<TAG, T>` methods.
 *
 * What is this for?
 * =================
 * This should make adding specializations for new types easier. For
 * example, one should be able to add a new type just by making its
 * `IListRefTagImpl` specialization inherit from `IListRefTagImplBase`.
 *
 * You should create a partial specialization for this class only if
 * you introduce a new `IListRefTag`. The idea being that there is one
 * default implementation for each possible value of `IListRefTag`.
 *
 * What does it do?
 * ================
 * 1. defines `elem_type` as an alias to `ListElemT`.
 *
 * 2. defines `list_type` as an alias to the default container type
 *    that will hold a collection of `elem_type`. The idea being that
 *    all types tagged as `TAG` will have `list_type` as its container,
 *    with different `elem_type`.
 *
 * 3. defines the default implementation for each of the methods that
 *    are supposed to be defined on `IListRefTagImpl` specializations.
 *
 * 4. inheriting from `IListRefTagImplBase<TAG, T, ListElemT>` also means
 *    that the payload of the type `IListRef<T>` will be of type `list_type`
 *    when it is tagged as `TAG`.
 */
template <IListRefTag TAG, typename T, typename ListElemT = T>
class IListRefTagImplBase {};
/*
 * Materialized container for `IListRef<T>`.
 *
 * What is this for?
 * =================
 * Container that groups `T` references together. This exchanges the
 * overhead of every method call from `IListRef<T>` for a dynamic allocation.
 *
 * You should use this container instead of `IListRef<T>` if:
 *
 *   - You are going to iterate the list more than once
 *   - You need to repeatedly access arbitrary elements (using `operator[]`)
 */

template <typename T>
using _MaterializedIListRefElem = std::conditional_t<
    std::is_reference_v<T>,
    typename std::reference_wrapper<std::remove_reference_t<T>>,
    T>;

/*
 * `MaterializedIListRefElem<T>` is an alias for `_MaterializedIListRefElem<IListRefConstRef<T>>`.
 *
 * What does it do?
 * ================
 * Removes the reference (&) from the type, and wraps it into a
 * `std::reference_wrapper`. If `IListRefConstRef<T>` is not a
 * reference type, then it's left unchanged.
 */
template <typename T>
using MaterializedIListRefElem = _MaterializedIListRefElem<IListRefConstRef<T>>;

/*
 * `MaterializedIListRef<T>` is a vector of `MaterializedIListRefElem<T>`.
 *
 * What does it do?
 * ================
 * It provides a vector container that stores elements of type `MaterializedIListRefElem<T>`.
 * This allows efficient storage and retrieval of references or non-references depending
 * on the type `T`.
 */
template <typename T>
using MaterializedIListRef = std::vector<MaterializedIListRefElem<T>>;

} // namespace detail

/*
 * Iterator for `IListRef<T>`.
 *
 * What is it?
 * ===========
 * Currently, a `std::bidirectional_iterator` that wraps the iterator
 * types defined for each of the `IListRefTag`.
 *
 * One should be able to use it, as if it were the unwrapped
 * iterators themselves.
 *
 * What does it do?
 * ================
 * Similarly to `IListRef<T>`, this is a wrapper class. Specifically, it
 * wraps each container's `const_iterator` type alias. So, for example,
 * given that the container for `IListRefTag::Boxed` is `c10::List`, this
 * iterator will wrap a `c10::List::const_iterator`.
 *
 * [Note: MSVC Iterator Debug]
 * ===========================
 * MSVC `vector<T>::iterator` implementation (used in the boxed variant)
 * makes it so this union's destructor, copy-constructor (assignment), and
 * move-constructor (assignment) are implicitly deleted.
 *
 * Therefore, we need to explicitly define them as needed. Follows a list
 * of places where these are needed and their reason:
 *
 *   - `Payload` destructor:
 *     it is deleted only if the macro `_ITERATOR_DEBUG_LEVEL` is set to 2.
 *
 *   - `IListRefIterator` destructor:
 *     same as above. However, we need to explicitly call the variant
 *     destructor explicitly.
 *
 *   - `IListRefIterator` copy-constructor:
 *     it is deleted only if the macro `_ITERATOR_DEBUG_LEVEL` is different
 *     than 0.
 */
template <typename T>
class IListRefIterator {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...)                        \
  friend class detail::IListRefTagImpl<IListRefTag::TAG, T>; \
  friend class detail::IListRefTagImplBase<                  \
      IListRefTag::TAG,                                      \
      T,                                                     \
      typename detail::IListRefTagImpl<IListRefTag::TAG, T>::elem_type>;
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS
// 取消宏定义 DEFINE_FRIEND_CLASS

 public:
  // C++17 友好的 std::iterator 实现
  using iterator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  // 定义未装箱迭代器类型
  using unboxed_iterator_type = typename detail::
      IListRefTagImpl<IListRefTag::Unboxed, T>::list_type::const_iterator;
  // 定义装箱迭代器类型
  using boxed_iterator_type = typename detail::
      IListRefTagImpl<IListRefTag::Boxed, T>::list_type::const_iterator;
  // 定义材料化迭代器类型
  using materialized_iterator_type =
      typename detail::MaterializedIListRef<T>::const_iterator;

  // 默认构造函数，初始化 tag_ 为 IListRefTag::None
  IListRefIterator() : tag_(IListRefTag::None) {}

#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL != 0
  // See [Note: MSVC Iterator Debug]
  // 复制构造函数，用于 MSVC 迭代器调试级别不为 0 的情况
  IListRefIterator(const IListRefIterator& iterator)
      : tag_(iterator.tag_) {
    switch (tag_) {
      case IListRefTag::Boxed:
        payload_.boxed_iterator = iterator.payload_.boxed_iterator;
        break;
      case IListRefTag::Unboxed:
        payload_.unboxed_iterator = iterator.payload_.unboxed_iterator;
        break;
      case IListRefTag::Materialized:
        payload_.materialized_iterator = iterator.payload_.materialized_iterator;
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "invalid IListRef tag.");
    }
  }
#endif

#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL == 2
  // See [Note: MSVC Iterator Debug]
  // 析构函数，用于 MSVC 迭代器调试级别为 2 的情况
  ~IListRefIterator() noexcept(false) {
    switch (tag_) {
      case IListRefTag::Boxed:
        payload_.boxed_iterator.~boxed_iterator_type();
        break;
      case IListRefTag::Unboxed:
        payload_.unboxed_iterator.~unboxed_iterator_type();
        break;
      case IListRefTag::Materialized:
        payload_.materialized_iterator.~materialized_iterator_type();
        break;
      default:
        TORCH_INTERNAL_ASSERT(false, "invalid IListRef tag.");
    }
  }
#endif

  // 装箱迭代器构造函数，初始化 tag_ 为 IListRefTag::Boxed
  IListRefIterator(boxed_iterator_type boxed) : tag_(IListRefTag::Boxed) {
    payload_.boxed_iterator = boxed;
  }

  // 未装箱迭代器构造函数，初始化 tag_ 为 IListRefTag::Unboxed
  IListRefIterator(unboxed_iterator_type unboxed) : tag_(IListRefTag::Unboxed) {
    payload_.unboxed_iterator = unboxed;
  }

  // 材料化迭代器构造函数，初始化 tag_ 为 IListRefTag::Materialized
  IListRefIterator(materialized_iterator_type materialized) : tag_(IListRefTag::Materialized) {
    payload_.materialized_iterator = materialized;
  }

  // 解引用操作符重载，返回 detail::IListRefConstRef<T> 对象
  detail::IListRefConstRef<T> operator*() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return ImplT::iterator_get(this_); });
  }

  // 前置递增操作符重载
  IListRefIterator& operator++() {
    TORCH_ILISTREF_UNWRAP(tag_, { ++this_; });
    return *this;
  }

  // 后置递增操作符重载
  IListRefIterator operator++(int) {
    auto old = *this;
    TORCH_ILISTREF_UNWRAP(tag_, { ++this_; });
    return old;
  }

  // 前置递减操作符重载
  IListRefIterator& operator--() {
    TORCH_ILISTREF_UNWRAP(tag_, { --this_; });
    return *this;
  }

  // 后置递减操作符重载
  IListRefIterator operator--(int) {
    auto old = *this;
    TORCH_ILISTREF_UNWRAP(tag_, { --this_; });
    return old;
  }

  // 等于操作符重载，比较两个迭代器是否相等
  bool operator==(const IListRefIterator& rhs) const {
    if (tag_ != rhs.tag_) {
      return false;
    }
    TORCH_ILISTREF_UNWRAP(tag_, {
      auto& rhs_it = ImplT::unwrap(rhs);
      return this_ == rhs_it;
    });


    // 使用 TORCH_ILISTREF_UNWRAP 宏，展开迭代器标签和代码块
    TORCH_ILISTREF_UNWRAP(tag_, {
      // 解引用 rhs 参数，获取其实际迭代器对象的引用
      auto& rhs_it = ImplT::unwrap(rhs);
      // 比较当前迭代器对象 this_ 和 rhs_it 是否相等，并返回比较结果
      return this_ == rhs_it;
    });



  }


  // IListRefIterator 类的成员函数定义结束
  }



  bool operator!=(const IListRefIterator& rhs) const {
    return !(*this == rhs);
  }


  // 重载不等于运算符，检查当前迭代器对象是否不等于 rhs 参数指定的迭代器对象
  bool operator!=(const IListRefIterator& rhs) const {
    // 使用已定义的相等运算符进行比较，然后对结果取反
    return !(*this == rhs);
  }



 private:
  union Payload {
    boxed_iterator_type boxed_iterator;
    unboxed_iterator_type unboxed_iterator;
    materialized_iterator_type materialized_iterator;
    void* _init_ptr;
    Payload() : _init_ptr(nullptr) {}


 private:
  // 定义一个联合体 Payload，用于在同一内存位置存储不同类型的数据
  union Payload {
    boxed_iterator_type boxed_iterator;                   // 包装过的迭代器类型
    unboxed_iterator_type unboxed_iterator;               // 非包装过的迭代器类型
    materialized_iterator_type materialized_iterator;     // 材料化迭代器类型
    void* _init_ptr;                                      // 初始化指针
    // 默认构造函数将 _init_ptr 初始化为 nullptr
    Payload() : _init_ptr(nullptr) {}
#if defined(_MSC_VER)
    // 如果定义了 _MSC_VER，编译器为 MSVC，则定义空析构函数
    // See [Note: MSVC Iterator Debug]
    ~Payload() {}
#endif
  };

  Payload payload_;
  IListRefTag tag_;
};

/*
 * IListRef 模板类定义，支持不同的列表引用标签
 *
 * See [Note: IListRef]
 */
template <typename T>
class IListRef {
 private:
#define DEFINE_FRIEND_CLASS(TAG, ...)                        \
  friend class detail::IListRefTagImpl<IListRefTag::TAG, T>; \
  friend class detail::IListRefTagImplBase<                  \
      IListRefTag::TAG,                                      \
      T,                                                     \
      typename detail::IListRefTagImpl<IListRefTag::TAG, T>::elem_type>;
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_FRIEND_CLASS)
#undef DEFINE_FRIEND_CLASS

 public:
  using unboxed_type =
      typename detail::IListRefTagImpl<IListRefTag::Unboxed, T>::list_type;
  using boxed_type =
      typename detail::IListRefTagImpl<IListRefTag::Boxed, T>::list_type;
  using materialized_type =
      typename detail::MaterializedIListRef<T>;

  using iterator = IListRefIterator<T>;
  using const_iterator = IListRefIterator<T>;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using value_type = typename iterator::value_type;

  IListRef() : tag_(IListRefTag::None) {}

  IListRef(const boxed_type& boxed) : tag_(IListRefTag::Boxed) {
    payload_.boxed = &boxed;
  }

  IListRef(const unboxed_type& unboxed) : tag_(IListRefTag::Unboxed) {
    payload_.unboxed = unboxed;
  }

  IListRef(const std::initializer_list<T>& list) : tag_(IListRefTag::Unboxed) {
    // 使用初始化列表构造未装箱类型引用
    payload_.unboxed = at::ArrayRef<T>(list);
  }

  template <
      typename... UnboxedConstructorArgs,
      typename = std::enable_if_t<
          std::is_constructible_v<unboxed_type, UnboxedConstructorArgs...>>>
  IListRef(UnboxedConstructorArgs&&... args) : tag_(IListRefTag::Unboxed) {
    // 使用变长模板构造未装箱类型引用
    payload_.unboxed = unboxed_type(std::forward<UnboxedConstructorArgs>(args)...);
  }

  IListRef(const materialized_type& materialized) : tag_(IListRefTag::Materialized) {
    // 使用已材料化类型构造引用
    payload_.materialized = &materialized;
  }

  // 返回列表大小
  size_t size() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return this_.size(); });
  }

  // 检查列表是否为空
  bool empty() const {
    return size() == 0;
  }

  // 返回列表的起始迭代器
  iterator begin() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return this_.begin(); });
  }

  // 返回列表的结束迭代器
  iterator end() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return this_.end(); });
  }

  // 返回列表的第一个元素的常量引用
  detail::IListRefConstRef<T> front() const {
    TORCH_ILISTREF_UNWRAP(tag_, { return ImplT::front(this_); });
  }

  /*
   * 将 IListRef 材料化为 std::vector。
   *
   * 当希望：
   *   - 多次迭代列表时，每次 IListRefIterator 成员函数调用都要经过开关，引入非常量级的开销
   *   - 使用 operator[] 随机访问元素时，同上述原因
   *   时应使用此方法。
   */
  detail::MaterializedIListRef<T> materialize() const {
    if (isMaterialized()) {
      return toMaterialized();
    }
    # 创建一个名为 materialized 的对象，类型为 MaterializedIListRef<T>，用于存储当前对象的元素
    detail::MaterializedIListRef<T> materialized;
    
    # 预留足够的空间，以容纳当前对象的元素数量
    materialized.reserve(size());
    
    # 遍历当前对象 (*this) 的每个元素，并将其添加到 materialized 中
    for (const auto& t : *this) {
      materialized.emplace_back(t);
    }
    
    # 返回已经填充好的 materialized 对象，其中包含了当前对象的所有元素
    return materialized;
#define DEFINE_CHECK(TAG, ...)    \
  bool is##TAG() const {          \  # 定义一个以 TAG 为后缀的成员函数 isTAG，用于检查当前对象的标签是否为 TAG
    return tag_ == IListRefTag::TAG; \  # 返回当前对象的标签是否与 TAG 相同
  }
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_CHECK);  \  # 对所有的 IListRefTag 标签循环展开 DEFINE_CHECK 宏

#undef DEFINE_CHECK  \  # 取消定义之前定义的 DEFINE_CHECK 宏

bool isNone() const {  \  # 定义一个成员函数 isNone，用于检查当前对象的标签是否为 None
  return tag_ == IListRefTag::None;  \  # 返回当前对象的标签是否为 None
}

#define DEFINE_CASTING(TAG, ...)                                          \
  const typename detail::IListRefTagImpl<IListRefTag::TAG, T>::list_type& \
      to##TAG() const {                                                   \
    TORCH_INTERNAL_ASSERT(is##TAG());                                     \  # 断言当前对象的标签是否为 TAG，以确保类型转换的有效性
    return detail::IListRefTagImpl<IListRefTag::TAG, T>::unwrap(*this);   \  # 调用具体的类型转换实现，将当前对象转换为 TAG 类型
  }
  TORCH_ILISTREF_FORALL_TAGS(DEFINE_CASTING);  \  # 对所有的 IListRefTag 标签循环展开 DEFINE_CASTING 宏

#undef DEFINE_CASTING  \  # 取消定义之前定义的 DEFINE_CASTING 宏

private:
union Payload {  \  # 定义一个联合体 Payload，用于存储不同类型的数据
  const boxed_type* boxed;  \  # 指向 boxed_type 类型的常量指针
  unboxed_type unboxed;  \  # unboxed_type 类型的对象
  const materialized_type* materialized;  \  # 指向 materialized_type 类型的常量指针
  Payload() : boxed(nullptr) {}  \  # 构造函数，默认初始化为 nullptr
};

Payload payload_;  \  # 声明一个 Payload 类型的成员变量 payload_
IListRefTag tag_;  \  # 声明一个 IListRefTag 类型的成员变量 tag_
};

} // namespace c10  \  # 结束 c10 命名空间

#include <ATen/core/IListRef_inl.h>  \  # 包含 IListRef 类的内联实现文件
```