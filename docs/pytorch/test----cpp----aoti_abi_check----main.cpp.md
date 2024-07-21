# `.\pytorch\test\cpp\aoti_abi_check\main.cpp`

```py
# 包含 Google Test 框架的头文件，用于单元测试
#include <gtest/gtest.h>

# 主函数，程序入口
int main(int argc, char* argv[]) {
  # 初始化 Google Test 框架，传入命令行参数 argc 和 argv
  ::testing::InitGoogleTest(&argc, argv);
  # 运行所有的测试用例，并返回测试结果
  return RUN_ALL_TESTS();
}
```