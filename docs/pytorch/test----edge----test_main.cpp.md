# `.\pytorch\test\edge\test_main.cpp`

```
#include <gtest/gtest.h>

// 添加负号标志到过滤器字符串中，并返回修改后的过滤器字符串
std::string add_negative_flag(const std::string& flag) {
  // 获取当前的测试过滤器字符串
  std::string filter = ::testing::GTEST_FLAG(filter);

  // 检查过滤器字符串中是否已经包含负号
  if (filter.find('-') == std::string::npos) {
    // 如果没有负号，则在末尾添加一个负号
    filter.push_back('-');
  } else {
    // 如果已经有负号，则在末尾添加一个冒号
    filter.push_back(':');
  }

  // 添加要传入的标志到过滤器字符串的末尾
  filter += flag;

  // 返回修改后的过滤器字符串
  return filter;
}

int main(int argc, char* argv[]) {
    // 初始化 Google Test 框架
    ::testing::InitGoogleTest(&argc, argv);
    
    // 将修改后的过滤器字符串应用到 GTEST_FLAG(filter) 中
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_CUDA:*_MultiCUDA");

    // 运行所有的测试，并返回测试结果
    return RUN_ALL_TESTS();
}
```