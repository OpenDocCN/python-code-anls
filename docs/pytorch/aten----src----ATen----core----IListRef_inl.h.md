# `.\pytorch\aten\src\ATen\core\IListRef_inl.h`

```py
/*
 * Specializations of `IListRefTagImplBase` that implement the default
 * implementation for `IListRefTag::Unboxed`.
 */
template <typename T, typename ListElemT>
class IListRefTagImplBase<IListRefTag::Unboxed, T, ListElemT> {
 public:
  using elem_type = ListElemT;
  using list_type = ArrayRef<elem_type>;

  /*
   * These `unwrap` static methods unwraps the inner containers out
   * of `IListRef<T>` (and `IListRefIterator<T>`). They are required when
   * the macro `TORCH_ILISTREF_UNWRAP` is called.
   */
  static const list_type& unwrap(const IListRef<T>& ilist) {
    return ilist.payload_.unboxed;
  }

  static typename list_type::const_iterator& unwrap(IListRefIterator<T>& it) {
    return it.payload_.unboxed_iterator;
  }

  static const typename list_type::const_iterator& unwrap(
      const IListRefIterator<T>& it) {
    return it.payload_.unboxed_iterator;
  }

  /*
   * We have these function (besides the `unwrap`s above) because the
   * implementation for both `IListRef::operator[]` and `IListRefIterator::operator*`
   * weren't syntatically equal for the existing tags at the time
   * (`Unboxed` and `Boxed`).
   */
  static IListRefConstRef<T> front(const list_type& lst) {
    return lst.front();
  }

  static IListRefConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return *it;
  }
};
    # 返回给定迭代器中的实际数据对象
    return it.payload_.materialized_iterator;
  }

  # 解封装给定的 IListRefIterator 对象，返回其内部实际迭代器的引用
  static const typename list_type::const_iterator& unwrap(
      const IListRefIterator<T>& it) {
    return it.payload_.materialized_iterator;
  }

  # 返回给定列表的第一个元素作为 IListRefConstRef<T> 类型的常量引用
  static IListRefConstRef<T> front(const list_type& lst) {
    return lst[0];
  }

  # 从给定的 const_iterator 中获取对应元素的 IListRefConstRef<T> 类型的常量引用
  static IListRefConstRef<T> iterator_get(
      const typename list_type::const_iterator& it) {
    return *it;
  }
};

/*
 * [Note: ITensorListRef]
 * Specializations necessary for `IListRef<at::Tensor>` type.
 *
 * Since the default implementations are usually done with supporting
 * `Tensor` in mind, we only have to inherit from the base implementations.
 */
template <>
class IListRefTagImpl<IListRefTag::Unboxed, at::Tensor>
    : public IListRefTagImplBase<IListRefTag::Unboxed, at::Tensor> {};

template <>
class IListRefTagImpl<IListRefTag::Boxed, at::Tensor>
    : public IListRefTagImplBase<IListRefTag::Boxed, at::Tensor> {};

template <>
class IListRefTagImpl<IListRefTag::Materialized, at::Tensor>
    : public IListRefTagImplBase<
          IListRefTag::Materialized,
          at::Tensor,
          MaterializedIListRefElem<at::Tensor>> {};

/*
 * [Note: IOptTensorListRef]
 * Specializations necessary for `IListRef<at::OptionalTensorRef>` type.
 *
 * We can't get an `at::OptionalTensorRef` directly from an instance of
 * `List<optional<Tensor>>` (the type that corresponds to the boxed world).
 *
 * So, the default implementation won't help us. Thus, we have to implement
 * this method ourselves.
 */
template <>
class IListRefTagImpl<IListRefTag::Unboxed, at::OptionalTensorRef>
    : public IListRefTagImplBase<IListRefTag::Unboxed, at::OptionalTensorRef> {};

template <>
class IListRefTagImpl<IListRefTag::Boxed, at::OptionalTensorRef>
    : public IListRefTagImplBase<IListRefTag::Boxed, at::OptionalTensorRef, optional<at::Tensor>> {

 public:
  /*
   * Given an instance of the types corresponding to the `Boxed` tag, we override
   * the default implementation, so that we can return a `at::OptionalTensorRef`.
   */
  static IListRefConstRef<at::OptionalTensorRef> iterator_get(
      const typename list_type::const_iterator& it) {
    const auto& ivalue = (*it).get();
    if (!ivalue.isNone()) {
        const auto& tensor = ivalue.toTensor();
        return (tensor.defined()) ? tensor : at::OptionalTensorRef{};
    }
    return {};
  }
};

template <>
class IListRefTagImpl<IListRefTag::Materialized, at::OptionalTensorRef>
    : public IListRefTagImplBase<
          IListRefTag::Materialized,
          at::OptionalTensorRef,
          MaterializedIListRefElem<at::OptionalTensorRef>> {};

} // namespace c10::detail


namespace at {

// [Note: ITensorListRef]
using ITensorListRef = c10::IListRef<at::Tensor>;
using ITensorListRefIterator = c10::IListRefIterator<at::Tensor>;
using MaterializedITensorListRef = c10::detail::MaterializedIListRef<at::Tensor>;
// [Note: IOptTensorListRef]
using IOptTensorListRef = c10::IListRef<at::OptionalTensorRef>;
using IOptTensorListRefIterator = c10::IListRefIterator<at::OptionalTensorRef>;
using MaterializedIOptTensorListRef = c10::detail::MaterializedIListRef<at::OptionalTensorRef>;

} // namespace at
```