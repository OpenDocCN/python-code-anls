# `.\pytorch\c10\test\util\TypeList_test.cpp`

```
#include <c10/util/TypeList.h>  // 包含 C10 库中的 TypeList 头文件
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件
#include <memory>  // 包含标准库中的 memory 头文件

using namespace c10::guts::typelist;  // 使用 C10 库中 typelist 的命名空间

// NOLINTBEGIN(modernize-unary-static-assert)
// 声明一个名为 test_size 的命名空间，用于测试 typelist 的大小计算
namespace test_size {
    class MyClass {};  // 声明一个空的 MyClass 类

    // 静态断言：typelist<>::value 的大小应为 0
    static_assert(0 == size<typelist<>>::value, "");

    // 静态断言：typelist<int>::value 的大小应为 1
    static_assert(1 == size<typelist<int>>::value, "");

    // 静态断言：typelist<int, float&, const MyClass&&>::value 的大小应为 3
    static_assert(3 == size<typelist<int, float&, const MyClass&&>>::value, "");
} // namespace test_size

// 声明一个名为 test_from_tuple 的命名空间，用于测试从 std::tuple 构建 typelist
namespace test_from_tuple {
    class MyClass {};  // 声明一个空的 MyClass 类

    // 静态断言：从 std::tuple<int, float&, const MyClass&&> 构建的 typelist 应该与目标类型相同
    static_assert(
        std::is_same<
            typelist<int, float&, const MyClass&&>,
            from_tuple_t<std::tuple<int, float&, const MyClass&&>>>::value,
        "");
    
    // 静态断言：从空的 std::tuple<> 构建的 typelist 应该是一个空的 typelist
    static_assert(std::is_same<typelist<>, from_tuple_t<std::tuple<>>>::value, "");
} // namespace test_from_tuple

// 声明一个名为 test_to_tuple 的命名空间，用于测试将 typelist 转换为 std::tuple
namespace test_to_tuple {
    class MyClass {};  // 声明一个空的 MyClass 类

    // 静态断言：将 typelist<int, float&, const MyClass&&> 转换为 std::tuple 应该与目标类型相同
    static_assert(
        std::is_same<
            std::tuple<int, float&, const MyClass&&>,
            to_tuple_t<typelist<int, float&, const MyClass&&>>>::value,
        "");
    
    // 静态断言：将空的 typelist<> 转换为 std::tuple 应该是一个空的 std::tuple
    static_assert(std::is_same<std::tuple<>, to_tuple_t<typelist<>>>::value, "");
} // namespace test_to_tuple

// 声明一个名为 test_concat 的命名空间，用于测试 typelist 的连接操作
namespace test_concat {
    class MyClass {};  // 声明一个空的 MyClass 类

    // 静态断言：连接空的 typelist<> 应该得到一个空的 typelist
    static_assert(std::is_same<typelist<>, concat_t<>>::value, "");

    // 静态断言：连接 typelist<> 和 typelist<> 应该得到一个空的 typelist
    static_assert(std::is_same<typelist<>, concat_t<typelist<>>>::value, "");

    // 静态断言：连接两个空的 typelist<> 应该得到一个空的 typelist
    static_assert(
        std::is_same<typelist<>, concat_t<typelist<>, typelist<>>>::value,
        "");

    // 静态断言：连接 typelist<int> 应该得到 typelist<int>
    static_assert(std::is_same<typelist<int>, concat_t<typelist<int>>>::value, "");

    // 静态断言：连接 typelist<int> 和空的 typelist<> 应该得到 typelist<int>
    static_assert(
        std::is_same<typelist<int>, concat_t<typelist<int>, typelist<>>>::value,
        "");

    // 静态断言：连接空的 typelist<> 和 typelist<int> 应该得到 typelist<int>
    static_assert(
        std::is_same<typelist<int>, concat_t<typelist<>, typelist<int>>>::value,
        "");

    // 静态断言：连接 typelist<>、typelist<int> 和 typelist<> 应该得到 typelist<int>
    static_assert(
        std::is_same<
            typelist<int>,
            concat_t<typelist<>, typelist<int>, typelist<>>>::value,
        "");

    // 静态断言：连接 typelist<int> 和 typelist<float&> 应该得到 typelist<int, float&>
    static_assert(
        std::is_same<
            typelist<int, float&>,
            concat_t<typelist<int>, typelist<float&>>>::value,
        "");

    // 静态断言：连接空的 typelist<>、typelist<int, float&> 和 typelist<> 应该得到 typelist<int, float&>
    static_assert(
        std::is_same<
            typelist<int, float&>,
            concat_t<typelist<>, typelist<int, float&>, typelist<>>>::value,
        "");

    // 静态断言：连接 typelist<>、typelist<int, float&> 和 typelist<const MyClass&&> 应该得到 typelist<int, float&, const MyClass&&>
    static_assert(
        std::is_same<
            typelist<int, float&, const MyClass&&>,
            concat_t<
                typelist<>,
                typelist<int, float&>,
                typelist<const MyClass&&>>>::value,
        "");
} // namespace test_concat

// 声明一个名为 test_filter 的命名空间，用于测试 typelist 的筛选操作
namespace test_filter {
    class MyClass {};  // 声明一个空的 MyClass 类

    // 静态断言：筛选空的 typelist<> 中的引用类型应得到一个空的 typelist
    static_assert(
        std::is_same<typelist<>, filter_t<std::is_reference, typelist<>>>::value,
        "");

    // 静态断言：筛选 typelist<int, float, double, MyClass> 中的引用类型应得到 typelist<float&, const MyClass&&>
    static_assert(
        std::is_same<
            typelist<float&, const MyClass&&>,
            filter_t<std::is_reference, typelist<int, float, double, MyClass>>>::
            value,
        "");
} // namespace test_filter

// 声明一个名为 test_count_if 的命名空间，用于测试 typelist 的条件计数操作
namespace test_count_if {
    class MyClass final {};  // 声明一个最终类的 MyClass

    // 静态断言：计算 typelist<int, bool&, const MyClass&&, float, double> 中引用类型的数量应为 2
    static_assert(
        count_if<
            std::is_reference,
            typelist<int, bool&, const MyClass&&, float, double>>::value == 2,
        "");
} // namespace test_count_if
// 静态断言，检查 typelist<int, bool> 中满足 std::is_reference 条件的元素数量为 0
static_assert(count_if<std::is_reference, typelist<int, bool>>::value == 0, "");

// 静态断言，检查 typelist<> 中满足 std::is_reference 条件的元素数量为 0
static_assert(count_if<std::is_reference, typelist<>>::value == 0, "");
} // namespace test_count_if

namespace test_true_for_each_type {
// 定义 Test 模板类的前向声明
template <class>
class Test;
// MyClass 类的定义
class MyClass {};
// 静态断言，检查 typelist<int&, const float&&, const MyClass&> 中所有元素都满足 std::is_reference 条件
static_assert(
    all<std::is_reference,
        typelist<int&, const float&&, const MyClass&>>::value,
    "");
// 静态断言，检查 typelist<int&, const float, const MyClass&> 中是否有任何元素不满足 std::is_reference 条件
static_assert(
    !all<std::is_reference, typelist<int&, const float, const MyClass&>>::value,
    "");
// 静态断言，检查 typelist<> 中所有元素都满足 std::is_reference 条件
static_assert(all<std::is_reference, typelist<>>::value, "");
} // namespace test_true_for_each_type

namespace test_true_for_any_type {
// 定义 Test 模板类的前向声明
template <class>
class Test;
// MyClass 类的定义
class MyClass {};
// 静态断言，检查 typelist<int&, const float&&, const MyClass&> 中是否有任何元素满足 std::is_reference 条件
static_assert(
    true_for_any_type<
        std::is_reference,
        typelist<int&, const float&&, const MyClass&>>::value,
    "");
// 静态断言，检查 typelist<int&, const float, const MyClass&> 中是否有任何元素满足 std::is_reference 条件
static_assert(
    true_for_any_type<
        std::is_reference,
        typelist<int&, const float, const MyClass&>>::value,
    "");
// 静态断言，检查 typelist<int, const float, const MyClass> 中是否有任何元素满足 std::is_reference 条件
static_assert(
    !true_for_any_type<
        std::is_reference,
        typelist<int, const float, const MyClass>>::value,
    "");
// 静态断言，检查 typelist<> 中是否有任何元素满足 std::is_reference 条件
static_assert(!true_for_any_type<std::is_reference, typelist<>>::value, "");
} // namespace test_true_for_any_type

namespace test_map {
// MyClass 类的定义
class MyClass {};
// 静态断言，检查 map_t<std::add_lvalue_reference_t, typelist<>> 结果是否与 typelist<> 相同
static_assert(
    std::is_same<typelist<>, map_t<std::add_lvalue_reference_t, typelist<>>>::
        value,
    "");
// 静态断言，检查 map_t<std::add_lvalue_reference_t, typelist<int>> 结果是否与 typelist<int&> 相同
static_assert(
    std::is_same<
        typelist<int&>,
        map_t<std::add_lvalue_reference_t, typelist<int>>>::value,
    "");
// 静态断言，检查 map_t<std::add_lvalue_reference_t, typelist<int, double, const MyClass>> 结果是否与 typelist<int&, double&, const MyClass&> 相同
static_assert(
    std::is_same<
        typelist<int&, double&, const MyClass&>,
        map_t<
            std::add_lvalue_reference_t,
            typelist<int, double, const MyClass>>>::value,
    "");
} // namespace test_map

namespace test_head {
// MyClass 类的定义
class MyClass {};
// 静态断言，检查 head_t<typelist<int, double>> 结果是否与 int 相同
static_assert(std::is_same<int, head_t<typelist<int, double>>>::value, "");
// 静态断言，检查 head_t<typelist<const MyClass&, double>> 结果是否与 const MyClass& 相同
static_assert(
    std::is_same<const MyClass&, head_t<typelist<const MyClass&, double>>>::
        value,
    "");
// 静态断言，检查 head_t<typelist<MyClass&&, MyClass>> 结果是否与 MyClass&& 相同
static_assert(
    std::is_same<MyClass&&, head_t<typelist<MyClass&&, MyClass>>>::value,
    "");
// 静态断言，检查 head_t<typelist<bool>> 结果是否与 bool 相同
static_assert(std::is_same<bool, head_t<typelist<bool>>>::value, "");
} // namespace test_head

namespace test_head_with_default {
// MyClass 类的定义
class MyClass {};
// 静态断言，检查 head_with_default_t<bool, typelist<int, double>> 结果是否与 int 相同
static_assert(
    std::is_same<int, head_with_default_t<bool, typelist<int, double>>>::value,
    "");
// 静态断言，检查 head_with_default_t<bool, typelist<const MyClass&, double>> 结果是否与 const MyClass& 相同
static_assert(
    std::is_same<
        const MyClass&,
        head_with_default_t<bool, typelist<const MyClass&, double>>>::value,
    "");
// 静态断言，检查 head_with_default_t<bool, typelist<MyClass&&, MyClass>> 结果是否与 MyClass&& 相同
static_assert(
    std::is_same<
        MyClass&&,
        head_with_default_t<bool, typelist<MyClass&&, MyClass>>>::value,
    "");
// 静态断言，检查 head_with_default_t<bool, typelist<int>> 结果是否与 int 相同
static_assert(
    std::is_same<int, head_with_default_t<bool, typelist<int>>>::value,
    "");
// 静态断言，检查 head_with_default_t<bool, typelist<>> 结果是否与 bool 相同
static_assert(
    std::is_same<bool, head_with_default_t<bool, typelist<>>>::value,
    "");
} // namespace test_head_with_default

namespace test_reverse {
// MyClass 类的定义
class MyClass {};
// 静态断言，检查 reverse_t<typelist<>> 结果是否与 typelist<> 相同
static_assert(
    // 检查 std::is_same 是否能够判断两个类型列表是否相同，
    // 分别是 typelist<int, double, MyClass*, const MyClass&&> 和
    // reverse_t<typelist<const MyClass&&, MyClass*, double, int>>。
    // 这里使用了 std::is_same::value 来获取结果，期望是 true，即这两个类型列表相同。
    std::is_same<
        typelist<int, double, MyClass*, const MyClass&&>,
        reverse_t<typelist<const MyClass&&, MyClass*, double, int>>>::value,
    "");
// 结束 namespace test_reverse

namespace test_map_types_to_values {

struct map_to_size {
  // 模板函数，返回类型 T 的大小
  template <class T>
  constexpr size_t operator()(T) const {
    return sizeof(typename T::type);
  }
};

// 测试函数 TypeListTest.MapTypesToValues_sametype
TEST(TypeListTest, MapTypesToValues_sametype) {
  // 调用 map_types_to_values 函数，使用 map_to_size 结构作为转换器
  auto sizes = map_types_to_values<typelist<int64_t, bool, uint32_t>>(map_to_size());
  // 期望的大小 tuple
  std::tuple<size_t, size_t, size_t> expected(8, 1, 4);
  // 断言 sizes 与 expected 类型相同
  static_assert(std::is_same<decltype(expected), decltype(sizes)>::value, "");
  // 断言 sizes 等于 expected
  EXPECT_EQ(expected, sizes);
}

struct map_make_shared {
  // 模板函数，返回类型 T 的 shared_ptr
  template <class T>
  std::shared_ptr<typename T::type> operator()(T) {
    return std::make_shared<typename T::type>();
  }
};

// 测试函数 TypeListTest.MapTypesToValues_differenttypes
TEST(TypeListTest, MapTypesToValues_differenttypes) {
  // 调用 map_types_to_values 函数，使用 map_make_shared 结构作为转换器
  auto shared_ptrs = map_types_to_values<typelist<int, double>>(map_make_shared());
  // 断言 shared_ptrs 与 std::tuple<std::shared_ptr<int>, std::shared_ptr<double>> 类型相同
  static_assert(
      std::is_same<
          std::tuple<std::shared_ptr<int>, std::shared_ptr<double>>,
          decltype(shared_ptrs)>::value,
      "");
}

struct Class1 {
  static int func() {
    return 3;
  }
};
struct Class2 {
  static double func() {
    return 2.0;
  }
};

struct mapper_call_func {
  // 模板函数，调用类型 T 的静态成员函数 func()
  template <class T>
  decltype(auto) operator()(T) {
    return T::type::func();
  }
};

// 测试函数 TypeListTest.MapTypesToValues_members
TEST(TypeListTest, MapTypesToValues_members) {
  // 调用 map_types_to_values 函数，使用 mapper_call_func 结构作为转换器
  auto result = map_types_to_values<typelist<Class1, Class2>>(mapper_call_func());
  // 期望的 tuple
  std::tuple<int, double> expected(3, 2.0);
  // 断言 result 与 expected 类型相同
  static_assert(std::is_same<decltype(expected), decltype(result)>::value, "");
  // 断言 result 等于 expected
  EXPECT_EQ(expected, result);
}

struct mapper_call_nonexistent_function {
  // 模板函数，调用类型 T 的不存在的成员函数 this_doesnt_exist()
  template <class T>
  decltype(auto) operator()(T) {
    return T::type::this_doesnt_exist();
  }
};

// 测试函数 TypeListTest.MapTypesToValues_empty
TEST(TypeListTest, MapTypesToValues_empty) {
  // 调用 map_types_to_values 函数，使用 mapper_call_nonexistent_function 结构作为转换器
  auto result = map_types_to_values<typelist<>>(mapper_call_nonexistent_function());
  // 期望的空 tuple
  std::tuple<> expected;
  // 断言 result 与 expected 类型相同
  static_assert(std::is_same<decltype(expected), decltype(result)>::value, "");
  // 断言 result 等于 expected
  EXPECT_EQ(expected, result);
}

} // namespace test_map_types_to_values

// 结束 namespace test_map_types_to_values

namespace test_find_if {
// 断言在 typelist<char&> 中找到 std::is_reference 返回值为 0
static_assert(0 == find_if<typelist<char&>, std::is_reference>::value, "");

// 断言在 typelist<char&, int, char&, int&> 中找到 std::is_reference 返回值为 0
static_assert(
    0 == find_if<typelist<char&, int, char&, int&>, std::is_reference>::value,
    "");

// 断言在 typelist<char, int, char&, int&> 中找到 std::is_reference 返回值为 2
static_assert(
    2 == find_if<typelist<char, int, char&, int&>, std::is_reference>::value,
    "");

// 断言在 typelist<char, int, char, int&> 中找到 std::is_reference 返回值为 3
static_assert(
    3 == find_if<typelist<char, int, char, int&>, std::is_reference>::value,
    "");
} // namespace test_find_if

// 结束 namespace test_find_if

namespace test_contains {
// 断言 typelist<double> 中包含类型 double
static_assert(contains<typelist<double>, double>::value, "");

// 断言 typelist<int, double> 中包含类型 double
static_assert(contains<typelist<int, double>, double>::value, "");

// 断言 typelist<int, double> 中不包含类型 float
static_assert(!contains<typelist<int, double>, float>::value, "");

// 断言 typelist<> 中不包含类型 double
static_assert(!contains<typelist<>, double>::value, "");
} // namespace test_contains

// 结束 namespace test_contains

namespace test_take {
// 断言 take_t<typelist<>, 0> 返回 typelist<>
static_assert(std::is_same<typelist<>, take_t<typelist<>, 0>>::value, "");

// 断言 take_t<typelist<int64_t>, 0> 返回 typelist<>
static_assert(
    std::is_same<typelist<>, take_t<typelist<int64_t>, 0>>::value,
    "");

// ```
    // 检查是否 typelist<int64_t> 和 take_t<typelist<int64_t>, 1> 是同一类型，结果会存储在 std::is_same<>::value 中
    std::is_same<typelist<int64_t>, take_t<typelist<int64_t>, 1>>::value,
    // 如果上述类型相同，返回空字符串；否则返回空字符串，可能用于静态断言或编译时检查
    "");
// 静态断言，验证 typelist<int64_t, int32_t> 的第 0 个元素的类型为空 typelist<>
static_assert(
    std::is_same<typelist<>, take_t<typelist<int64_t, int32_t>, 0>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的前 1 个元素为 typelist<int64_t>
static_assert(
    std::is_same<typelist<int64_t>, take_t<typelist<int64_t, int32_t>, 1>>::
        value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的前 2 个元素为 typelist<int64_t, int32_t>
static_assert(
    std::is_same<
        typelist<int64_t, int32_t>,
        take_t<typelist<int64_t, int32_t>, 2>>::value,
    "");
} // namespace test_take

namespace test_drop {
// 静态断言，验证空 typelist<> 的第 0 个元素为空 typelist<>
static_assert(std::is_same<typelist<>, drop_t<typelist<>, 0>>::value, "");

// 静态断言，验证 typelist<int64_t> 的第 0 个元素为 typelist<int64_t>
static_assert(
    std::is_same<typelist<int64_t>, drop_t<typelist<int64_t>, 0>>::value,
    "");

// 静态断言，验证 typelist<int64_t> 的第 1 个元素为空 typelist<>
static_assert(
    std::is_same<typelist<>, drop_t<typelist<int64_t>, 1>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 0 个元素为 typelist<int64_t, int32_t>
static_assert(
    std::is_same<
        typelist<int64_t, int32_t>,
        drop_t<typelist<int64_t, int32_t>, 0>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 1 个元素为 typelist<int32_t>
static_assert(
    std::is_same<typelist<int32_t>, drop_t<typelist<int64_t, int32_t>, 1>>::
        value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 2 个元素为空 typelist<>
static_assert(
    std::is_same<typelist<>, drop_t<typelist<int64_t, int32_t>, 2>>::value,
    "");
} // namespace test_drop

namespace test_drop_if_nonempty {
// 静态断言，验证空 typelist<> 的第 0 个元素为空 typelist<>
static_assert(
    std::is_same<typelist<>, drop_if_nonempty_t<typelist<>, 0>>::value,
    "");

// 静态断言，验证 typelist<int64_t> 的第 0 个元素为 typelist<int64_t>
static_assert(
    std::is_same<typelist<int64_t>, drop_if_nonempty_t<typelist<int64_t>, 0>>::
        value,
    "");

// 静态断言，验证 typelist<int64_t> 的第 1 个元素为空 typelist<>
static_assert(
    std::is_same<typelist<>, drop_if_nonempty_t<typelist<int64_t>, 1>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 0 个元素为 typelist<int64_t, int32_t>
static_assert(
    std::is_same<
        typelist<int64_t, int32_t>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 0>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 1 个元素为 typelist<int32_t>
static_assert(
    std::is_same<
        typelist<int32_t>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 1>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 2 个元素为空 typelist<>
static_assert(
    std::is_same<
        typelist<>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 2>>::value,
    "");

// 静态断言，验证空 typelist<> 的第 1 个元素为空 typelist<>
static_assert(
    std::is_same<typelist<>, drop_if_nonempty_t<typelist<>, 1>>::value,
    "");

// 静态断言，验证 typelist<int64_t, int32_t> 的第 3 个元素为空 typelist<>
static_assert(
    std::is_same<
        typelist<>,
        drop_if_nonempty_t<typelist<int64_t, int32_t>, 3>>::value,
    "");
} // namespace test_drop_if_nonempty
// NOLINTEND(modernize-unary-static-assert)
```