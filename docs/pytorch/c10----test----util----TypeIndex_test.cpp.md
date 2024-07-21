# `.\pytorch\c10\test\util\TypeIndex_test.cpp`

```
// 包含必要的头文件：c10/util/Metaprogramming.h，用于元编程；c10/util/TypeIndex.h，用于类型索引；gtest/gtest.h，用于 Google 测试框架
#include <c10/util/Metaprogramming.h>
#include <c10/util/TypeIndex.h>
#include <gtest/gtest.h>

// 使用 c10 命名空间中的 string_view 和 util 中的两个函数：get_fully_qualified_type_name 和 get_type_index
using c10::string_view;
using c10::util::get_fully_qualified_type_name;
using c10::util::get_type_index;

// NOLINTBEGIN(modernize-unary-static-assert)
// 匿名命名空间，用于定义静态断言
namespace {

// 检查同一类型的静态断言
static_assert(get_type_index<int>() == get_type_index<int>(), "");

// 检查同一类型的静态断言（float）
static_assert(get_type_index<float>() == get_type_index<float>(), "");

// 检查不同类型的静态断言（int 和 float）
static_assert(get_type_index<int>() != get_type_index<float>(), "");

// 检查相同参数类型的函数类型的静态断言
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int(double, double)>(),
    "");

// 检查不同参数类型的函数类型的静态断言
static_assert(
    get_type_index<int(double, double)>() != get_type_index<int(double)>(),
    "");

// 检查函数指针类型与函数类型的静态断言
static_assert(
    get_type_index<int(double, double)>() ==
        get_type_index<int (*)(double, double)>(),
    "");

// 检查 std::function 类型相同参数的静态断言
static_assert(
    get_type_index<std::function<int(double, double)>>() ==
        get_type_index<std::function<int(double, double)>>(),
    "");

// 检查 std::function 类型不同参数的静态断言
static_assert(
    get_type_index<std::function<int(double, double)>>() !=
        get_type_index<std::function<int(double)>>(),
    "");

// 检查 int 和其引用类型的静态断言
static_assert(get_type_index<int>() == get_type_index<int&>(), "");

// 检查 int 和其右值引用类型的静态断言
static_assert(get_type_index<int>() == get_type_index<int&&>(), "");

// 检查 int 和 const int& 类型的静态断言
static_assert(get_type_index<int>() == get_type_index<const int&>(), "");

// 检查 int 和 const int 类型的静态断言
static_assert(get_type_index<int>() == get_type_index<const int>(), "");

// 检查 const int 和 int& 类型的静态断言
static_assert(get_type_index<const int>() == get_type_index<int&>(), "");

// 检查 int 和 int* 类型的静态断言
static_assert(get_type_index<int>() != get_type_index<int*>(), "");

// 检查 int* 和 int** 类型的静态断言
static_assert(get_type_index<int*>() != get_type_index<int**>(), "");

// 检查不同参数类型的函数类型的静态断言
static_assert(
    get_type_index<int(double&, double)>() !=
        get_type_index<int(double, double)>(),
    "");

// 结构体 Dummy 的定义
struct Dummy final {};

// 结构体 Functor 的定义，包含一个带有特定参数的操作符()
struct Functor final {
  int64_t operator()(uint32_t, Dummy&&, const Dummy&) const;
};

// 检查 Functor 类型与函数类型推断的静态断言
static_assert(
    get_type_index<int64_t(uint32_t, Dummy&&, const Dummy&)>() ==
        get_type_index<
            c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");

// 嵌套命名空间 test_top_level_name，测试类型名称的顶层名称
namespace test_top_level_name {
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
// 检查 Dummy 类型的全限定类型名包含 "Dummy" 的静态断言
static_assert(
    string_view::npos != get_fully_qualified_type_name<Dummy>().find("Dummy"),
    "");
#endif

// 测试函数：TypeIndex.TopLevelName，检查 Dummy 类型全限定类型名包含 "Dummy"
TEST(TypeIndex, TopLevelName) {
  EXPECT_NE(
      string_view::npos, get_fully_qualified_type_name<Dummy>().find("Dummy"));
}
} // namespace test_top_level_name

// 嵌套命名空间 test_nested_name，测试类型名称的嵌套名称
namespace test_nested_name {
// 再次定义结构体 Dummy，用于测试
struct Dummy final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
// 检查 Dummy 类型的全限定类型名包含 "test_nested_name::Dummy" 的静态断言
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy"),
    "");
#endif

// 测试函数：TypeIndex.NestedName，检查 Dummy 类型全限定类型名包含 "test_nested_name::Dummy"
TEST(TypeIndex, NestedName) {
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Dummy>().find("test_nested_name::Dummy"));
}
} // namespace test_nested_name

// 嵌套命名空间 test_type_template_parameter，测试模板参数的类型名称
template <class T>
struct Outer final {};

// 结构体 Inner 的定义
struct Inner final {};

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
// 检查 Outer<T> 类型的全限定类型名包含 "Outer" 的静态断言
static_assert(

// 检查 Outer<T> 类型的全限定类型名包含 "Outer" 的静态断言
    string_view::npos != get_fully_qualified_type_name<Outer<int>>().find("Outer"),
    "");
#endif

// 结构体 Inner 类型的全限定类型名包含 "Inner" 的静态断言
static_assert(
    string_view::npos != get_fully_qualified_type_name<Inner>().find("Inner"),
    "");

} // namespace test_type_template_parameter

} // anonymous namespace
    # 使用 std::string_view 的静态成员 npos 来判断 find 方法的返回值是否不等于 npos
    string_view::npos !=
        # 调用模板函数 get_fully_qualified_type_name，获取 Outer<Inner> 类型的完全限定名
        get_fully_qualified_type_name<Outer<Inner>>().find(
            # 在获取到的完全限定名中查找是否包含字符串 "test_type_template_parameter::Outer"
            "test_type_template_parameter::Outer"),
        "");
    # 注释：这行代码可能用于检查某个类型名称是否包含特定的字符串 "test_type_template_parameter::Outer"
// 静态断言，确保在字符串视图中能找到完全限定类型名
static_assert(
    string_view::npos !=
        get_fully_qualified_type_name<Outer<Inner>>().find(
            "test_type_template_parameter::Inner"),
    "");

// TypeIndex 测试用例，检验模板类型参数
TEST(TypeIndex, TypeTemplateParameter) {
  // 断言：完全限定类型名中包含 "test_type_template_parameter::Outer"
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Outer<Inner>>().find(
          "test_type_template_parameter::Outer"));
  // 断言：完全限定类型名中包含 "test_type_template_parameter::Inner"
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Outer<Inner>>().find(
          "test_type_template_parameter::Inner"));
}

} // namespace test_type_template_parameter

namespace test_nontype_template_parameter {
// 模板类，具有非类型参数 N
template <size_t N>
struct Class final {};

// 如果支持 constexpr，则进行静态断言
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    // 确保完全限定类型名中能找到特定非类型参数 N
    string_view::npos !=
        get_fully_qualified_type_name<Class<38474355>>().find("38474355"),
    "");
#endif

// TypeIndex 测试用例，检验非类型模板参数
TEST(TypeIndex, NonTypeTemplateParameter) {
  // 断言：完全限定类型名中包含 "38474355"
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<Class<38474355>>().find("38474355"));
}
} // namespace test_nontype_template_parameter

namespace test_type_computations_are_resolved {
// 模板类 Type，对模板参数进行类型计算
template <class T>
struct Type final {
  using type = const T*;
};

// 如果支持 constexpr，则进行静态断言
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    // 确保完全限定类型名中能找到 "int"
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("int"),
    "");
static_assert(
    // 确保完全限定类型名中能找到 "*"
    string_view::npos !=
        get_fully_qualified_type_name<typename Type<int>::type>().find("*"),
    "");

// 移除指针后，确保完全限定类型名中不能再找到 "*"
static_assert(
    string_view::npos ==
        get_fully_qualified_type_name<
            typename std::remove_pointer<typename Type<int>::type>::type>()
            .find("*"),
    "");
#endif

// TypeIndex 测试用例，检验类型计算是否解析正确
TEST(TypeIndex, TypeComputationsAreResolved) {
  // 断言：完全限定类型名中包含 "int"
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<typename Type<int>::type>().find("int"));
  // 断言：完全限定类型名中包含 "*"
  EXPECT_NE(
      string_view::npos,
      get_fully_qualified_type_name<typename Type<int>::type>().find("*"));
  // 断言：移除指针后，完全限定类型名中不应包含 "*"
  EXPECT_EQ(
      string_view::npos,
      get_fully_qualified_type_name<
          typename std::remove_pointer<typename Type<int>::type>::type>()
          .find("*"));
}

// 结构体 Functor，包含函数调用运算符
struct Functor final {
  std::string operator()(int64_t a, const Type<int>& b) const;
};

// 如果支持 constexpr，则进行静态断言
#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    // 确保函数类型的完全限定类型名相同
    get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>() ==
        get_fully_qualified_type_name<
            typename c10::guts::infer_function_traits_t<Functor>::func_type>(),
    "");
#endif

// TypeIndex 测试用例，检验函数参数和返回值类型的计算是否解析正确
TEST(TypeIndex, FunctionTypeComputationsAreResolved) {
  // 断言：函数类型的完全限定类型名相同
  EXPECT_EQ(
      get_fully_qualified_type_name<std::string(int64_t, const Type<int>&)>(),
      get_fully_qualified_type_name<
          typename c10::guts::infer_function_traits_t<Functor>::func_type>());
}
} // namespace test_type_computations_are_resolved

namespace test_function_arguments_and_returns {
class Dummy final {};  // 定义一个最终类 Dummy

#if C10_TYPENAME_SUPPORTS_CONSTEXPR
static_assert(
    string_view::npos !=  // 检查条件：string_view::npos 不等于
        get_fully_qualified_type_name<Dummy(int)>().find(  // 获取 Dummy(int) 的全限定类型名并查找
            "test_function_arguments_and_returns::Dummy"),  // 查找包含指定命名空间的类型名
    "");  // 静态断言的错误信息（空字符串表示无错误信息）

static_assert(
    string_view::npos !=  // 检查条件：string_view::npos 不等于
        get_fully_qualified_type_name<void(Dummy)>().find(  // 获取 void(Dummy) 的全限定类型名并查找
            "test_function_arguments_and_returns::Dummy"),  // 查找包含指定命名空间的类型名
    "");  // 静态断言的错误信息（空字符串表示无错误信息）
#endif

TEST(TypeIndex, FunctionArgumentsAndReturns) {  // 测试函数：类型索引，测试函数参数和返回值
  EXPECT_NE(  // 断言：期望不相等
      string_view::npos,  // 第一个参数：字符串视图的无效位置（未找到）
      get_fully_qualified_type_name<Dummy(int)>().find(  // 获取 Dummy(int) 的全限定类型名并查找
          "test_function_arguments_and_returns::Dummy"));  // 查找包含指定命名空间的类型名

  EXPECT_NE(  // 断言：期望不相等
      string_view::npos,  // 第一个参数：字符串视图的无效位置（未找到）
      get_fully_qualified_type_name<void(Dummy)>().find(  // 获取 void(Dummy) 的全限定类型名并查找
          "test_function_arguments_and_returns::Dummy"));  // 查找包含指定命名空间的类型名
}

}  // namespace test_function_arguments_and_returns
}  // namespace
// NOLINTEND(modernize-unary-static-assert)
```