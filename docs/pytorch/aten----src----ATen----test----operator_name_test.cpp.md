# `.\pytorch\aten\src\ATen\test\operator_name_test.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/core/operator_name.h>  // 引入 ATen 库中的操作符命名相关头文件

TEST(OperatorNameTest, SetNamespaceIfNotSetWithoutExistingNamespace) {
  c10::OperatorName testName("operator", "operator.overload");  // 创建一个 OperatorName 对象 testName，指定基本名称和重载名称

  const auto result = testName.setNamespaceIfNotSet("ns");  // 调用方法设置命名空间，返回操作是否成功的结果
  EXPECT_TRUE(result);  // 断言操作成功
  EXPECT_EQ(testName.name, "ns::operator");  // 断言设置后的名称符合预期
  EXPECT_EQ(testName.overload_name, "operator.overload");  // 断言重载名称不变
  EXPECT_EQ(testName.getNamespace(), std::optional<c10::string_view>("ns"));  // 断言获取的命名空间符合预期
}

TEST(OperatorNameTest, SetNamespaceIfNotSetWithExistingNamespace) {
  c10::OperatorName namespacedName("already_namespaced::operator", "operator.overload");  // 创建一个已有命名空间的 OperatorName 对象

  const auto result = namespacedName.setNamespaceIfNotSet("namespace");  // 尝试设置命名空间，返回操作是否成功的结果
  EXPECT_FALSE(result);  // 断言操作未成功（因为命名空间已存在）
  EXPECT_EQ(namespacedName.name, "already_namespaced::operator");  // 断言名称不变
  EXPECT_EQ(namespacedName.overload_name, "operator.overload");  // 断言重载名称不变
  EXPECT_EQ(namespacedName.getNamespace(), std::optional<c10::string_view>("already_namespaced"));  // 断言获取的命名空间符合预期
}
```