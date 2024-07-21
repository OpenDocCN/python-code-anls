# `.\pytorch\torch\csrc\api\include\torch\nn\pimpl-inl.h`

```py
// This class serves as a marker type for template specialization and SFINAE purposes
// to identify types that are ModuleHolders.
struct ModuleHolderIndicator {};

// Type trait that checks if a type T is derived from ModuleHolderIndicator.
template <typename T>
using is_module_holder =
    std::is_base_of<ModuleHolderIndicator, std::decay_t<T>>;

// Type trait that disables a template if the type T is a ModuleHolder.
template <typename T>
using disable_if_module_holder_t =
    std::enable_if_t<!is_module_holder<T>::value>;

// Template struct that determines whether a type T is a ModuleHolder containing
// a specific contained type C, using template specialization.
template <bool is_module_holder_value, typename T, typename C>
struct is_module_holder_of_impl;

// Specialization for when T is not a ModuleHolder.
template <typename T, typename C>
struct is_module_holder_of_impl<false, T, C> : std::false_type {};

// Specialization for when T is a ModuleHolder, allowing access to its ContainedType
// and checking if it matches type C.
template <typename T, typename C>
struct is_module_holder_of_impl<true, T, C>
    : std::is_same<typename T::ContainedType, C> {};

// Helper template that simplifies the usage of is_module_holder_of.
template <typename T, typename C>
struct is_module_holder_of : is_module_holder_of_impl<
                                 is_module_holder<T>::value,
                                 std::decay_t<T>,
                                 std::decay_t<C>> {};

// Template struct that determines the return type of a forward() method for a class C
// based on whether such a method exists, using SFINAE.
template <bool has_forward_value, typename C, typename... Args>
struct return_type_of_forward_impl;

// Specialization for when C has a valid forward() method with Args... arguments.
template <typename C, typename... Args>
struct return_type_of_forward_impl<true, C, Args...> {
  using type = decltype(::std::declval<C>().forward(::std::declval<Args>()...));
};

// Specialization for when C does not have a forward() method.
template <typename C, typename... Args>
struct return_type_of_forward_impl<false, C, Args...> {
  using type = void;
};
// 定义类型别名 return_type_of_forward，使用 return_type_of_forward_impl 模板进行类型推断
using return_type_of_forward = return_type_of_forward_impl<
    // 检查类型 C 是否有 forward 方法，并获取其返回类型
    torch::detail::has_forward<C>::value,
    // C 类型
    C,
    // Args... 变参列表
    Args...
>;

// 定义模板类型别名 return_type_of_forward_t，为上述类型别名的类型成员
template <typename C, typename... Args>
using return_type_of_forward_t =
    typename return_type_of_forward<C, Args...>::type;
```