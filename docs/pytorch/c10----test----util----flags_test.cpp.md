# `.\pytorch\c10\test\util\flags_test.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <iostream>  // 包含标准输入输出流的头文件

#include <c10/util/Flags.h>  // 包含 Caffe2 中用于处理命令行标志的头文件

C10_DEFINE_bool(c10_flags_test_only_flag, true, "Only used in test.");  // 定义一个布尔型命令行标志 c10_flags_test_only_flag，默认为 true，仅用于测试

namespace c10_test {

TEST(FlagsTest, TestGflagsCorrectness) {  // 定义一个名为 FlagsTest 的测试类，测试命令行标志的正确性
#ifdef C10_USE_GFLAGS
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);  // 使用 Google Test 的断言检查 FLAGS_c10_flags_test_only_flag 是否为 true
  FLAGS_c10_flags_test_only_flag = false;  // 修改 FLAGS_c10_flags_test_only_flag 为 false
  FLAGS_c10_flags_test_only_flag = true;  // 将 FLAGS_c10_flags_test_only_flag 重新设为 true
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);  // 再次使用断言检查 FLAGS_c10_flags_test_only_flag 是否为 true
#else // C10_USE_GFLAGS
  std::cout << "Caffe2 is not built with gflags. Nothing to test here."  // 如果未使用 gflags 构建 Caffe2，则输出此消息
            << std::endl;
#endif
}

} // namespace c10_test
```