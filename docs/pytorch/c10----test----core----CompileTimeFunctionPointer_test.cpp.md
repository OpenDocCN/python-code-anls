# `.\pytorch\c10\test\core\CompileTimeFunctionPointer_test.cpp`

```py
// 引入 Torch 的编译时函数指针相关头文件
#include <c10/core/CompileTimeFunctionPointer.h>
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>

// 定义测试命名空间 test_is_compile_time_function_pointer
namespace test_is_compile_time_function_pointer {
// 断言：void() 不是编译时函数指针
static_assert(!c10::is_compile_time_function_pointer<void()>::value);

// 定义一个空函数 dummy
void dummy() {}
// 断言：TORCH_FN_TYPE(dummy) 是编译时函数指针
static_assert(
    c10::is_compile_time_function_pointer<TORCH_FN_TYPE(dummy)>::value);
} // namespace test_is_compile_time_function_pointer

// 定义测试命名空间 test_access_through_type
namespace test_access_through_type {
// 定义一个空函数 dummy
void dummy() {}
// 使用 TORCH_FN_TYPE 定义一个函数指针类型 dummy_ptr
using dummy_ptr = TORCH_FN_TYPE(dummy);
// 断言：dummy_ptr 是编译时函数指针
static_assert(c10::is_compile_time_function_pointer<dummy_ptr>::value);
// 断言：dummy_ptr::func_ptr() 等于 dummy 函数的地址
static_assert(dummy_ptr::func_ptr() == &dummy);
// 断言：dummy_ptr::FuncType 类型是 void()
static_assert(std::is_same<void(), dummy_ptr::FuncType>::value);
} // namespace test_access_through_type

// 定义测试命名空间 test_access_through_value
namespace test_access_through_value {
// 定义一个空函数 dummy
void dummy() {}
// 使用 TORCH_FN 定义一个 constexpr 的函数指针 dummy_ptr
constexpr auto dummy_ptr = TORCH_FN(dummy);
// 断言：dummy_ptr.func_ptr() 等于 dummy 函数的地址
static_assert(dummy_ptr.func_ptr() == &dummy);
// 断言：decltype(dummy_ptr)::FuncType 类型是 void()
static_assert(std::is_same<void(), decltype(dummy_ptr)::FuncType>::value);
} // namespace test_access_through_value

// 定义测试命名空间 test_access_through_type_also_works_if_specified_as_pointer
namespace test_access_through_type_also_works_if_specified_as_pointer {
// 定义一个空函数 dummy
void dummy() {}
// 使用 TORCH_FN_TYPE(&dummy) 定义一个函数指针类型 dummy_ptr
using dummy_ptr = TORCH_FN_TYPE(&dummy);
// 断言：dummy_ptr 是编译时函数指针
static_assert(c10::is_compile_time_function_pointer<dummy_ptr>::value);
// 断言：dummy_ptr::func_ptr() 等于 dummy 函数的地址
static_assert(dummy_ptr::func_ptr() == &dummy);
// 断言：dummy_ptr::FuncType 类型是 void()
static_assert(std::is_same<void(), dummy_ptr::FuncType>::value);
} // namespace test_access_through_type_also_works_if_specified_as_pointer

// 定义测试命名空间 test_access_through_value_also_works_if_specified_as_pointer
namespace test_access_through_value_also_works_if_specified_as_pointer {
// 定义一个空函数 dummy
void dummy() {}
// 使用 TORCH_FN(&dummy) 定义一个 constexpr 的函数指针 dummy_ptr
constexpr auto dummy_ptr = TORCH_FN(&dummy);
// 断言：dummy_ptr.func_ptr() 等于 dummy 函数的地址
static_assert(dummy_ptr.func_ptr() == &dummy);
// 断言：decltype(dummy_ptr)::FuncType 类型是 void()
static_assert(std::is_same<void(), decltype(dummy_ptr)::FuncType>::value);
} // namespace test_access_through_value_also_works_if_specified_as_pointer

// 定义测试命名空间 test_run_through_type
namespace test_run_through_type {
// 定义一个加法函数 add
int add(int a, int b) {
  return a + b;
}
// 使用 TORCH_FN_TYPE(add) 定义一个函数指针类型 Add
using Add = TORCH_FN_TYPE(add);
// 定义一个模板结构体 Executor，用于执行函数
template <class Func>
struct Executor {
  // execute 方法执行 Func 类型的函数指针，传入参数 a 和 b
  int execute(int a, int b) {
    return Func::func_ptr()(a, b);
  }
};

// 定义 Google Test 的测试用例 CompileTimeFunctionPointerTest.runFunctionThroughType
TEST(CompileTimeFunctionPointerTest, runFunctionThroughType) {
  // 实例化 Executor 结构体，使用 Add 类型作为模板参数
  Executor<Add> executor;
  // 断言：executor 执行 add 函数，传入参数 1 和 2 结果为 3
  EXPECT_EQ(3, executor.execute(1, 2));
}
} // namespace test_run_through_type

// 定义测试命名空间 test_run_through_value
namespace test_run_through_value {
// 定义一个加法函数 add
int add(int a, int b) {
  return a + b;
}
// execute 函数接受一个函数指针 Func 和两个整数参数 a 和 b
template <class Func>
int execute(Func, int a, int b) {
  return Func::func_ptr()(a, b);
}

// 定义 Google Test 的测试用例 CompileTimeFunctionPointerTest.runFunctionThroughValue
TEST(CompileTimeFunctionPointerTest, runFunctionThroughValue) {
  // 断言：执行 execute 函数，传入 TORCH_FN(add) 作为函数指针，参数为 1 和 2 结果为 3
  EXPECT_EQ(3, execute(TORCH_FN(add), 1, 2));
}
} // namespace test_run_through_value
```