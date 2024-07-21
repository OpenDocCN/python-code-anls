# `.\pytorch\test\cpp\c10d\StoreTestCommon.hpp`

```py
#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp> // 包含分布式存储的头文件
#include "TestUtils.hpp" // 包含测试工具的头文件

#include <gtest/gtest.h> // 包含 Google 测试框架的头文件

namespace c10d {
namespace test {

// 设置键值对到存储中
inline void set(
    Store& store, // 分布式存储对象的引用
    const std::string& key, // 键的字符串表示
    const std::string& value) { // 值的字符串表示
  // 将字符串值转换为字节向量
  std::vector<uint8_t> data(value.begin(), value.end());
  // 调用存储对象的设置方法
  store.set(key, data);
}

// 比较并设置存储中的值
inline std::vector<uint8_t> compareSet(
    Store& store, // 分布式存储对象的引用
    const std::string& key, // 键的字符串表示
    const std::string& expectedValue, // 期望的字符串值
    const std::string& desiredValue) { // 想要设置的字符串值
  // 将期望的字符串值和想要设置的字符串值转换为字节向量
  std::vector<uint8_t> expectedData(expectedValue.begin(), expectedValue.end());
  std::vector<uint8_t> desiredData(desiredValue.begin(), desiredValue.end());
  // 调用存储对象的比较并设置方法
  return store.compareSet(key, expectedData, desiredData);
}

// 检查存储中的键对应的值是否与预期相同
inline void check(
    Store& store, // 分布式存储对象的引用
    const std::string& key, // 键的字符串表示
    const std::string& expected) { // 期望的字符串值
  // 调用存储对象的获取方法获取键对应的值
  auto tmp = store.get(key);
  // 将获取到的值转换为字符串
  auto actual = std::string((const char*)tmp.data(), tmp.size());
  // 使用 Google 测试框架断言检查实际值与期望值是否相等
  EXPECT_EQ(actual, expected);
}

// 删除存储中指定的键值对
inline void deleteKey(
    Store& store, // 分布式存储对象的引用
    const std::string& key) { // 要删除的键的字符串表示
  // 调用存储对象的删除方法
  store.deleteKey(key);
}

} // namespace test
} // namespace c10d
```