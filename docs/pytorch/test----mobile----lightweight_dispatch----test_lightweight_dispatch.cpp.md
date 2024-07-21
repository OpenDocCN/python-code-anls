# `.\pytorch\test\mobile\lightweight_dispatch\test_lightweight_dispatch.cpp`

```
#include <gtest/gtest.h>

// 添加负号标志到过滤器字符串
std::string add_negative_flag(const std::string& flag) {
  // 获取当前的测试过滤器字符串
  std::string filter = ::testing::GTEST_FLAG(filter);
  // 如果过滤器字符串中不存在 '-'，则添加 '-' 到末尾
  if (filter.find('-') == std::string::npos) {
    filter.push_back('-');
  } else {
    // 否则添加 ':' 到末尾
    filter.push_back(':');
  }
  // 添加新的标志到过滤器字符串
  filter += flag;
  // 返回更新后的过滤器字符串
  return filter;
}

// 主函数，初始化 Google Test 框架并设置测试过滤器，运行所有测试用例
int main(int argc, char* argv[]) {
    // 初始化 Google Test 框架
    ::testing::InitGoogleTest(&argc, argv);
    // 设置测试过滤器为包含特定 CUDA 相关标志的测试用例，并加入负号标志
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_CUDA:*_MultiCUDA");

    // 运行所有的测试用例，并返回结果
    return RUN_ALL_TESTS();
}
```