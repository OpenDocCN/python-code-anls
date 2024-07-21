# `.\pytorch\c10\util\TypeList.h`

```
#pragma once

#include <c10/util/TypeTraits.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace c10::guts {

template <class... T>
struct false_t : std::false_type {}; // 结构体模板 false_t，继承自 std::false_type，表示值为 false
template <template <class> class... T>
struct false_higher_t : std::false_type {}; // 结构体模板 false_higher_t，继承自 std::false_type，表示值为 false

namespace typelist {

/**
 * Type holding a list of types for compile time type computations
 */
template <class... Items>
struct typelist final {
 public:
  typelist() = delete; // 删除默认构造函数，防止实例化
};

/**
 * Returns the number of types in a typelist
 * Example:
 *   3  ==  size<typelist<int, int, double>>::value
 */
template <class TypeList>
struct size final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::size<T>, T must be typelist<...>."); // 静态断言，要求 TypeList 必须是 typelist<...> 类型
};
template <class... Types>
struct size<typelist<Types...>> final {
  static constexpr size_t value = sizeof...(Types); // 编译时常量，表示 typelist 中包含的类型数量
};

/**
 * Transforms a list of types into a tuple holding these types.
 * Example:
 *   std::tuple<int, string>  ==  to_tuple_t<typelist<int, string>>
 */
template <class TypeList>
struct to_tuple final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::to_tuple<T>, T must be typelist<...>."); // 静态断言，要求 TypeList 必须是 typelist<...> 类型
};
template <class... Types>
struct to_tuple<typelist<Types...>> final {
  using type = std::tuple<Types...>; // 类型转换，将 typelist 转换为 std::tuple
};
template <class TypeList>
using to_tuple_t = typename to_tuple<TypeList>::type; // 类型别名模板，用于获取 typelist 转换后的 std::tuple 类型

/**
 * Creates a typelist containing the types of a given tuple.
 * Example:
 *   typelist<int, string>  ==  from_tuple_t<std::tuple<int, string>>
 */
template <class Tuple>
struct from_tuple final {
  static_assert(
      false_t<Tuple>::value,
      "In typelist::from_tuple<T>, T must be std::tuple<...>."); // 静态断言，要求 Tuple 必须是 std::tuple<...> 类型
};
template <class... Types>
struct from_tuple<std::tuple<Types...>> final {
  using type = typelist<Types...>; // 类型转换，将 std::tuple 转换为 typelist
};
template <class Tuple>
using from_tuple_t = typename from_tuple<Tuple>::type; // 类型别名模板，用于获取 std::tuple 转换后的 typelist 类型

/**
 * Concatenates multiple type lists.
 * Example:
 *   typelist<int, string, int>  ==  concat_t<typelist<int, string>,
 * typelist<int>>
 */
template <class... TypeLists>
struct concat final {
  static_assert(
      false_t<TypeLists...>::value,
      "In typelist::concat<T1, ...>, the T arguments each must be typelist<...>."); // 静态断言，要求每个 T 参数必须是 typelist<...> 类型
};
template <class... Head1Types, class... Head2Types, class... TailLists>
struct concat<typelist<Head1Types...>, typelist<Head2Types...>, TailLists...>
    final {
  using type =
      typename concat<typelist<Head1Types..., Head2Types...>, TailLists...>::
          type; // 类型拼接，将多个 typelist 拼接为一个 typelist
};
template <class... HeadTypes>
struct concat<typelist<HeadTypes...>> final {
  using type = typelist<HeadTypes...>; // 单个 typelist 的情况，返回原始 typelist
};
template <>
struct concat<> final {
  using type = typelist<>; // 没有参数时的特化，返回空 typelist
};
template <class... TypeLists>
using concat_t = typename concat<TypeLists...>::type; // 类型别名模板，用于获取多个 typelist 拼接后的 typelist 类型

} // namespace typelist

} // namespace c10::guts
/**
 * Filters the types in a type list by a type trait.
 * Examples:
 *   typelist<int&, const string&&>  ==  filter_t<std::is_reference,
 * typelist<void, string, int&, bool, const string&&, int>>
 */
template <template <class> class Condition, class TypeList>
struct filter final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::filter<Condition, TypeList>, the TypeList argument must be typelist<...>.");
};

/**
 * Partial specialization of filter for non-empty typelist.
 * Filters types in TypeList based on the Condition trait.
 */
template <template <class> class Condition, class Head, class... Tail>
struct filter<Condition, typelist<Head, Tail...>> final {
  static_assert(
      is_type_condition<Condition>::value,
      "In typelist::filter<Condition, TypeList>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");

  // Defines the filtered typelist type based on Condition evaluation.
  using type = std::conditional_t<
      Condition<Head>::value,
      concat_t<
          typelist<Head>,
          typename filter<Condition, typelist<Tail...>>::type>,
      typename filter<Condition, typelist<Tail...>>::type>;
};

/**
 * Specialization of filter for an empty typelist.
 * Returns an empty typelist when filtering an empty input.
 */
template <template <class> class Condition>
struct filter<Condition, typelist<>> final {
  static_assert(
      is_type_condition<Condition>::value,
      "In typelist::filter<Condition, TypeList>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");

  using type = typelist<>;  // Resulting typelist is empty.
};

/**
 * Alias template for simplifying access to the filtered typelist type.
 */
template <template <class> class Condition, class TypeList>
using filter_t = typename filter<Condition, TypeList>::type;

/**
 * Counts how many types in the list fulfill a type trait.
 * Examples:
 *   2  ==  count_if<std::is_reference, typelist<void, string, int&, bool, const
 * string&&, int>>
 */
template <template <class> class Condition, class TypeList>
struct count_if final {
  static_assert(
      is_type_condition<Condition>::value,
      "In typelist::count_if<Condition, TypeList>, the Condition argument must be a condition type trait, i.e. have a static constexpr bool ::value member.");
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::count_if<Condition, TypeList>, the TypeList argument must be typelist<...>.");

  // TODO Direct implementation might be faster
  // Calculates the number of types in TypeList fulfilling Condition.
  static constexpr size_t value = size<filter_t<Condition, TypeList>>::value;
};

/**
 * Checks if a typelist contains a certain type.
 * Examples:
 *  contains<typelist<int, string>, string> == true_type
 *  contains<typelist<int, string>, double> == false_type
 */
namespace detail {
template <class TypeList, class Type, class Enable = void>
struct contains {};

// Base case: typelist is empty, thus doesn't contain Type.
template <class Type>
struct contains<typelist<>, Type, void> : std::false_type {};

// Case where the head of TypeList matches Type.
template <class Type, class Head, class... Tail>
struct contains<
    typelist<Head, Tail...>,
    Type,
    std::enable_if_t<std::is_same_v<Head, Type>>> : std::true_type {};

// Recursive case: TypeList not empty, check the tail for Type.
template <class Type, class Head, class... Tail>
struct contains<
    typelist<Head, Tail...>,
    Type,
    std::enable_if_t<!std::is_same_v<Head, Type>>>
    : contains<typelist<Tail...>, Type> {};
/**
 * 结束 detail 命名空间
 */
} // namespace detail

/**
 * 使用别名模板定义，判断 TypeList 是否包含 Type 类型的成员
 */
template <class TypeList, class Type>
using contains = typename detail::contains<TypeList, Type>::type;

/**
 * 检查类型特征 Condition 是否对 TypeList 中所有类型都为真
 * 示例：
 *   true   ==  all<std::is_reference, typelist<int&, const float&&, const MyClass&>>::value
 *   false  ==  all<std::is_reference, typelist<int&, const float&&, MyClass>>::value
 */
template <template <class> class Condition, class TypeList>
struct all {
  /**
   * 在 typelist::all<Condition, TypeList> 中，TypeList 参数必须是 typelist<...>
   */
  static_assert(
      false_t<TypeList>::value,
      "In typelist::all<Condition, TypeList>, the TypeList argument must be typelist<...>.");
};

/**
 * 检查类型特征 Condition 是否对 TypeList 中任何类型为真
 * 示例：
 *   true   ==  true_for_any_type<std::is_reference, typelist<int, const float&&, const MyClass>>::value
 *   false  ==  true_for_any_type<std::is_reference, typelist<int, const float, MyClass>>::value
 */
template <template <class> class Condition, class TypeList>
struct true_for_any_type final {
  /**
   * 在 typelist::true_for_any_type<Condition, TypeList> 中，TypeList 参数必须是 typelist<...>
   */
  static_assert(
      false_t<TypeList>::value,
      "In typelist::true_for_any_type<Condition, TypeList>, the TypeList argument must be typelist<...>.");
};

/**
 * 使用类型特征 Mapper 映射 TypeList 中的类型
 * 示例：
 *   typelist<int&, double&, string&>  ==  map_t<std::add_lvalue_reference_t, typelist<int, double, string>>
 */
template <template <class> class Mapper, class TypeList>
struct map final {
  /**
   * 在 typelist::map<Mapper, TypeList> 中，TypeList 参数必须是 typelist<...>
   */
  static_assert(
      false_t<TypeList>::value,
      "In typelist::map<Mapper, TypeList>, the TypeList argument must be typelist<...>.");
};

/**
 * 返回 TypeList 中的第一个元素类型
 * 示例：
 *   int  ==  head_t<typelist<int, string>>
 */
template <class TypeList>
struct head final {
  /**
   * 在 typelist::head<T> 中，T 参数必须是 typelist<...>
   */
  static_assert(
      false_t<TypeList>::value,
      "In typelist::head<T>, the T argument must be typelist<...>.");
};
/**
 * Defines an alias template `head_t` that extracts the first element type from
 * a type list.
 * Example:
 *   int  ==  head_t<typelist<int, string>>
 */
template <class TypeList>
using head_t = typename head<TypeList>::type;

/**
 * Defines a struct `head_with_default` that provides the first element type
 * from a type list or a default type if the list is empty.
 * Example:
 *   int  ==  head_with_default_t<int, typelist<>>
 */
template <class Default, class TypeList>
struct head_with_default final {
  using type = Default;
};
template <class Default, class Head, class... Tail>
struct head_with_default<Default, typelist<Head, Tail...>> final {
  using type = Head;
};
template <class Default, class TypeList>
using head_with_default_t = typename head_with_default<Default, TypeList>::type;

/**
 * Defines a struct `element` that retrieves the N-th element type from a type
 * list.
 * Example:
 *   int  ==  element_t<1, typelist<float, int, char>>
 */
template <size_t Index, class TypeList>
struct element final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::element<T>, the T argument must be typelist<...>.");
};

/// Successful case for retrieving the head type.
template <class Head, class... Tail>
struct element<0, typelist<Head, Tail...>> {
  using type = Head;
};

/// Error case if index is out of bounds for the typelist.
template <size_t Index, class... Ts>
struct element<Index, typelist<Ts...>> {
  static_assert(
      Index < sizeof...(Ts),
      "Index is out of bounds in typelist::element");
};

/// Recursive case to move through the typelist until reaching the desired index.
template <size_t Index, class Head, class... Tail>
struct element<Index, typelist<Head, Tail...>>
    : element<Index - 1, typelist<Tail...>> {};

/// Alias template for convenient use.
template <size_t Index, class TypeList>
using element_t = typename element<Index, TypeList>::type;

/**
 * Defines a struct `last` that retrieves the last element type from a type list.
 * Example:
 *   int  ==  last_t<typelist<int, string>>
 */
template <class TypeList>
struct last final {
  static_assert(
      false_t<TypeList>::value,
      "In typelist::last<T>, the T argument must be typelist<...>.");
};
template <class Head, class... Tail>
struct last<typelist<Head, Tail...>> final {
  using type = typename last<typelist<Tail...>>::type;
};
template <class Head>
struct last<typelist<Head>> final {
  using type = Head;
};
template <class TypeList>
using last_t = typename last<TypeList>::type;
static_assert(std::is_same_v<int, last_t<typelist<double, float, int>>>);

/**
 * Defines utility structs and alias templates within the `detail` namespace
 * for manipulating type lists, specifically for taking or dropping a number
 * of elements.
 * Example:
 *   typelist<int, string> == take_t<typelist<int, string, bool>, 2>
 *   typelist<bool> == drop_t<typelist<int, string, bool>, 2>
 */
namespace detail {
template <class TypeList, size_t offset, class IndexSequence>
struct take_elements final {};

template <class TypeList, size_t offset, size_t... Indices>
struct take_elements<TypeList, offset, std::index_sequence<Indices...>> final {
  using type = typelist<typename element<offset + Indices, TypeList>::type...>;
};
} // namespace detail
// 定义模板结构体 take，用于从 TypeList 中取出前 num 个元素
template <class TypeList, size_t num>
struct take final {
  // 确保 TypeList 是 typelist 的实例化
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::take<T, num>, the T argument must be typelist<...>.");
  // 确保 num 不超过 TypeList 的大小
  static_assert(
      num <= size<TypeList>::value,
      "Tried to typelist::take more elements than there are in the list");
  // 使用 take_elements 辅助模板从 TypeList 中提取指定数量的元素
  using type = typename detail::
      take_elements<TypeList, 0, std::make_index_sequence<num>>::type;
};

// take_t 是 take 结构体的类型别名，用于简化类型调用
template <class TypeList, size_t num>
using take_t = typename take<TypeList, num>::type;

// 定义模板结构体 drop，用于从 TypeList 中删除前 num 个元素
template <class TypeList, size_t num>
struct drop final {
  // 确保 TypeList 是 typelist 的实例化
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::drop<T, num>, the T argument must be typelist<...>.");
  // 确保 num 不超过 TypeList 的大小
  static_assert(
      num <= size<TypeList>::value,
      "Tried to typelist::drop more elements than there are in the list");
  // 使用 take_elements 辅助模板从 TypeList 中删除前 num 个元素
  using type = typename detail::take_elements<
      TypeList,
      num,
      std::make_index_sequence<size<TypeList>::value - num>>::type;
};

// drop_t 是 drop 结构体的类型别名，用于简化类型调用
template <class TypeList, size_t num>
using drop_t = typename drop<TypeList, num>::type;

/**
 * Like drop, but returns an empty list rather than an assertion error if `num`
 * is larger than the size of the TypeList.
 * Example:
 *   typelist<> == drop_if_nonempty_t<typelist<string, bool>, 2>
 *   typelist<> == drop_if_nonempty_t<typelist<int, string, bool>, 3>
 */
// 定义模板结构体 drop_if_nonempty，与 drop 类似，但允许 num 大于 TypeList 大小时返回空列表
template <class TypeList, size_t num>
struct drop_if_nonempty final {
  // 确保 TypeList 是 typelist 的实例化
  static_assert(
      is_instantiation_of<typelist, TypeList>::value,
      "In typelist::drop<T, num>, the T argument must be typelist<...>.");
  // 使用 take_elements 辅助模板从 TypeList 中删除前指定数量的元素
  using type = typename detail::take_elements<
      TypeList,
      std::min(num, size<TypeList>::value),
      std::make_index_sequence<
          size<TypeList>::value - std::min(num, size<TypeList>::value)>>::type;
};

// drop_if_nonempty_t 是 drop_if_nonempty 结构体的类型别名，用于简化类型调用
template <class TypeList, size_t num>
using drop_if_nonempty_t = typename drop_if_nonempty<TypeList, num>::type;

/**
 * Reverses a typelist.
 * Example:
 *   typelist<int, string>  == reverse_t<typelist<string, int>>
 */
// 定义模板结构体 reverse，用于反转 TypeList 中的元素顺序
template <class TypeList>
struct reverse final {
  // 确保 TypeList 是 typelist 的实例化
  static_assert(
      false_t<TypeList>::value,
      "In typelist::reverse<T>, the T argument must be typelist<...>.");
};

// 特化 reverse 结构体，对非空 TypeList 进行反转操作
template <class Head, class... Tail>
struct reverse<typelist<Head, Tail...>> final {
  // 使用 concat_t 辅助模板将 Head 放在尾部并递归反转 Tail
  using type =
      concat_t<typename reverse<typelist<Tail...>>::type, typelist<Head>>;
};

// 特化 reverse 结构体，处理空 TypeList 的情况
template <>
struct reverse<typelist<>> final {
  // 空 TypeList 的反转结果仍为 typelist<>
  using type = typelist<>;
};

// reverse_t 是 reverse 结构体的类型别名，用于简化类型调用
template <class TypeList>
using reverse_t = typename reverse<TypeList>::type;

/**
 * Find the index of the first type in a typelist fulfilling a type trait
 * condition. Example:
 *
 * 2 == find_if<typelist<char, int, char&, int&>, std::is_reference>::value
 */
// 定义模板结构体 find_if，用于在 TypeList 中查找符合类型特征条件的第一个类型的索引
template <class TypeList, template <class> class Condition, class Enable = void>
/**
 * find_if 结构体用于在 typelist 中查找满足条件的类型位置索引。
 * 它包括多个模板特化版本，每个版本对应不同的条件和类型列表情况。
 */

struct find_if final {
  // 强制断言，如果触发说明此处的 static_assert 没有通过，打印错误消息
  static_assert(
      false_t<TypeList>::value,
      "In typelist::find_if<TypeList, Condition>, the TypeList argument must be typelist<...>.");
};

template <template <class> class Condition>
struct find_if<typelist<>, Condition, void> final {
  // 强制断言，如果触发说明没有找到满足条件的类型，打印错误消息
  static_assert(
      false_higher_t<Condition>::value,
      "In typelist::find_if<Type/List, Condition>, didn't find any type fulfilling the Condition.");
};

template <class Head, class... Tail, template <class> class Condition>
struct find_if<
    typelist<Head, Tail...>,
    Condition,
    std::enable_if_t<Condition<Head>::value>>
    final {
  // 如果 Head 满足条件，返回索引值 0
  static constexpr size_t value = 0;
};

template <class Head, class... Tail, template <class> class Condition>
struct find_if<
    typelist<Head, Tail...>,
    Condition,
    std::enable_if_t<!Condition<Head>::value>>
    final {
  // 如果 Head 不满足条件，递归查找 Tail 部分，索引值加 1
  static constexpr size_t value =
      1 + find_if<typelist<Tail...>, Condition>::value;
};

/**
 * namespace detail 内部命名空间定义了 map_types_to_values 结构体，用于将类型列表映射为值列表。
 */

namespace detail {
  
template <class T>
struct type_ final {
  using type = T;
};

template <class TypeList>
struct map_types_to_values final {
  // 强制断言，如果触发说明 TypeList 不是 typelist 类型，打印错误消息
  static_assert(
      false_t<TypeList>::value,
      "In typelist::map_types_to_values<T>, the T argument must be typelist<...>.");
};

template <class... Types>
struct map_types_to_values<typelist<Types...>> final {
  /**
   * 调用模板函数 call，接受一个 Func 参数，并返回一个元组，
   * 其中每个元素由 func(type_<Types>())... 函数调用产生。
   */
  template <class Func>
  static auto call(Func&& func) {
    return std::tuple{std::forward<Func>(func)(type_<Types>())...};
  }
};

} // namespace detail

/**
 * map_types_to_values 函数模板用于外部调用，将类型列表映射为对应的值列表。
 * 参数 TypeList 表示类型列表，Func 表示处理每个类型的函数对象。
 */

template <class TypeList, class Func>
decltype(auto) map_types_to_values(Func&& func) {
  return detail::map_types_to_values<TypeList>::call(std::forward<Func>(func));
}

} // namespace typelist
} // namespace c10::guts
```