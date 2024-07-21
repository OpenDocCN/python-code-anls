# `.\pytorch\test\cpp\jit\test_upgrader_utils.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>
// 包含 Torch JIT 中升级操作符的实用工具函数的头文件
#include <torch/csrc/jit/operator_upgraders/utils.h>
// 包含 Torch JIT 中升级操作符的版本映射的头文件
#include <torch/csrc/jit/operator_upgraders/version_map.h>

// 包含 JIT 测试工具函数的头文件
#include <test/cpp/jit/test_utils.h>

// 包含向量容器的头文件
#include <vector>

// Torch 命名空间
namespace torch {
// JIT 命名空间
namespace jit {

// 测试用例：查找正确的升级器
TEST(UpgraderUtils, FindCorrectUpgrader) {
  // 创建一个包含两个 UpgraderEntry 的向量
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.bar()"},
  };

  // 查找版本号为 6 的升级器
  auto upgrader_at_6 = findUpgrader(dummy_entry, 6);
  EXPECT_TRUE(upgrader_at_6.has_value());
  EXPECT_EQ(upgrader_at_6.value().upgrader_name, "foo__4_7");

  // 查找版本号为 1 的升级器
  auto upgrader_at_1 = findUpgrader(dummy_entry, 1);
  EXPECT_TRUE(upgrader_at_1.has_value());
  EXPECT_EQ(upgrader_at_1.value().upgrader_name, "foo__0_3");

  // 查找版本号为 10 的升级器
  auto upgrader_at_10 = findUpgrader(dummy_entry, 10);
  EXPECT_TRUE(upgrader_at_1.has_value()); // 该行应为 EXPECT_TRUE(upgrader_at_10.has_value());
  EXPECT_EQ(upgrader_at_1.value().upgrader_name, "foo__0_3");
}

// 测试用例：检查版本映射是否按升级版本排序
TEST(UpgraderUtils, IsVersionMapSorted) {
  // 获取操作符版本映射
  auto map = get_operator_version_map();
  // 遍历映射中的每个条目
  for (const auto& entry : map) {
    // 创建一个整数向量存储每个 UpgraderEntry 的 bumped_at_version 字段
    std::vector<int> versions;
    for (const auto& el : entry.second) {
      versions.push_back(el.bumped_at_version);
    }
    // 检查版本号是否按升序排列
    EXPECT_TRUE(std::is_sorted(versions.begin(), versions.end()));
  }
}

// 测试用例：检查基于升级器条目判断操作符是否为当前版本
TEST(UpgraderUtils, FindIfOpIsCurrent) {
  // 创建一个包含两个 UpgraderEntry 的向量
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.bar()"},
  };

  // 检查版本号为 6 的操作符是否为当前版本
  auto isCurrent = isOpCurrentBasedOnUpgraderEntries(dummy_entry, 6);
  auto isCurrentV2 = isOpCurrentBasedOnUpgraderEntries(dummy_entry, 8);
  EXPECT_FALSE(isCurrent);
  EXPECT_TRUE(isCurrentV2);

  // 基于符号查找
  test_only_add_entry("foo", dummy_entry[0]);
  test_only_add_entry("foo", dummy_entry[1]);
  EXPECT_FALSE(isOpSymbolCurrent("foo", 6));
  EXPECT_TRUE(isOpSymbolCurrent("foo", 8));
  test_only_remove_entry("foo");
}

// 测试用例：检查是否能够加载历史操作符
TEST(UpgraderUtils, CanLoadHistoricOp) {
  // 创建一个包含两个 UpgraderEntry 的向量
  std::vector<UpgraderEntry> dummy_entry = {
      {4, "foo__0_3", "foo.bar()"},
      {8, "foo__4_7", "foo.foo()"},
  };

  // 创建包含两个历史模式的字符串向量
  std::vector<std::string> schemas = {"foo.bar()", "foo.foo()"};

  // 基于符号查找
  test_only_add_entry("old_op_not_exist.first", dummy_entry[0]);
  test_only_add_entry("old_op_not_exist.second", dummy_entry[1]);

  // 加载可能的历史操作符
  auto oldSchemas = loadPossibleHistoricOps("old_op_not_exist", 2);
  EXPECT_EQ(oldSchemas.size(), 2);
  for (const auto& entry : oldSchemas) {
    # 断言：检查是否能在schemas列表中找到entry项
    EXPECT_TRUE(
        std::find(schemas.begin(), schemas.end(), entry) != schemas.end());
  }

  # 载入指定版本的历史操作，这里版本号为9，返回结果为空列表
  auto oldSchemasWithCurrentVersion =
      loadPossibleHistoricOps("old_op_not_exist", 9);
  # 断言：验证载入的旧操作列表长度为0
  EXPECT_EQ(oldSchemasWithCurrentVersion.size(), 0);

  # 测试函数：仅移除指定条目"old_op_not_exist.first"
  test_only_remove_entry("old_op_not_exist.first");
  # 再次测试函数：仅移除指定条目"old_op_not_exist.first"
  test_only_remove_entry("old_op_not_exist.first");

  # 注释：旧的schema没有过载也是可以的
  # 测试函数：仅添加指定条目"old_op_not_exist_no_overload"，使用dummy_entry[0]作为值
  test_only_add_entry("old_op_not_exist_no_overload", dummy_entry[0]);
  # 载入指定版本的历史操作，这里版本号为2，返回结果包含一个元素
  auto oldSchemasNoOverload =
      loadPossibleHistoricOps("old_op_not_exist_no_overload", 2);
  # 断言：验证载入的旧操作列表长度为1
  EXPECT_EQ(oldSchemasNoOverload.size(), 1);
  # 断言：验证载入的旧操作列表第一个元素为"foo.bar()"
  EXPECT_EQ(oldSchemasNoOverload[0], "foo.bar()");
  # 测试函数：仅移除指定条目"old_op_not_exist_no_overload"
  test_only_remove_entry("old_op_not_exist_no_overload");
}

} // namespace jit
} // namespace torch
```