# `.\pytorch\aten\src\ATen\core\dispatch\CppSignature_test.cpp`

```py
// 包含 ATen 库的 CppSignature 头文件
#include <ATen/core/dispatch/CppSignature.h>
// 包含 Google 测试框架的头文件
#include <gtest/gtest.h>
// 包含字符串操作的标准库
#include <string>

// 使用 c10 命名空间中的 CppSignature 类
using c10::impl::CppSignature;

// 匿名命名空间，用于封装单元测试
namespace {

// 测试用例：当签名相等时，期望它们相等
TEST(CppSignatureTest, given_equalSignature_then_areEqual) {
    // 检查两个空参数无返回值的函数签名是否相等
    EXPECT_EQ(CppSignature::make<void()>(), CppSignature::make<void()>());
    // 检查带有一个字符串参数和一个整数参数的函数签名是否相等
    EXPECT_EQ(CppSignature::make<int64_t(std::string, int64_t)>(), CppSignature::make<int64_t(std::string, int64_t)>());
}

// 测试用例：当签名不同时，期望它们不相等
TEST(CppSignatureTest, given_differentSignature_then_areDifferent) {
    // 检查空参数无返回值与空参数返回整数签名是否不相等
    EXPECT_NE(CppSignature::make<void()>(), CppSignature::make<int64_t()>());
    // 检查一个字符串参数的签名与一个字符串和一个整数参数的签名是否不相等
    EXPECT_NE(CppSignature::make<int64_t(std::string)>(), CppSignature::make<int64_t(std::string, int64_t)>());
    // 检查一个字符串参数返回字符串与一个字符串参数返回整数的签名是否不相等
    EXPECT_NE(CppSignature::make<std::string(std::string)>(), CppSignature::make<int64_t(std::string)>());
}

// 测试用例：当给定相等的函数对象和函数签名时，期望它们相等
TEST(CppSignatureTest, given_equalFunctorAndFunction_then_areEqual) {
    // 定义一个函数对象 Functor，实现返回整数的操作符
    struct Functor final {
        int64_t operator()(std::string) {return 0;}
    };
    // 检查 Functor 函数对象与接受一个字符串参数返回整数的函数签名是否相等
    EXPECT_EQ(CppSignature::make<Functor>(), CppSignature::make<int64_t(std::string)>());
}

// 测试用例：当给定不同的函数对象和函数签名时，期望它们不相等
TEST(CppSignatureTest, given_differentFunctorAndFunction_then_areDifferent) {
    // 定义一个函数对象 Functor，实现返回整数的操作符
    struct Functor final {
        int64_t operator()(std::string) {return 0;}
    };
    // 检查 Functor 函数对象与接受一个字符串和一个整数参数返回整数的函数签名是否不相等
    EXPECT_NE(CppSignature::make<Functor>(), CppSignature::make<int64_t(std::string, int64_t)>());
}

// 测试用例：当给定 CppSignature 时，能够查询其名称而不崩溃
TEST(CppSignatureTest, given_cppSignature_then_canQueryNameWithoutCrashing) {
    // 调用 CppSignature 对象的 name() 方法，验证能否正常获取名称
    CppSignature::make<void(int64_t, const int64_t&)>().name();
}

} // end namespace
```