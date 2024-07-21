# `.\pytorch\aten\src\ATen\test\vitals.cpp`

```py
#include <gmock/gmock.h>  // 引入 Google Mock 测试框架的头文件
#include <gtest/gtest.h>  // 引入 Google Test 测试框架的头文件

#include <ATen/ATen.h>  // 引入 PyTorch 的 ATen 头文件
#include <ATen/core/Vitals.h>  // 引入 PyTorch 的关键信息记录头文件
#include <c10/util/irange.h>  // 引入 C10 库中的循环范围头文件
#include <cstdlib>  // 引入 C 标准库中的通用函数头文件

using namespace at::vitals;  // 使用 PyTorch 的关键信息命名空间
using ::testing::HasSubstr;  // 使用 Google Test 中的 HasSubstr 断言

TEST(Vitals, Basic) {  // 测试用例：基本测试
  std::stringstream buffer;  // 创建一个字符串流对象，用于捕获输出

  std::streambuf* sbuf = std::cout.rdbuf();  // 获取标准输出的流缓冲区
  std::cout.rdbuf(buffer.rdbuf());  // 重定向标准输出到字符串流
  {
#ifdef _WIN32
    _putenv("TORCH_VITAL=1");  // 在 Windows 下设置环境变量 TORCH_VITAL 为 1
#else
    setenv("TORCH_VITAL", "1", 1);  // 在 POSIX 系统下设置环境变量 TORCH_VITAL 为 "1"
#endif
    TORCH_VITAL_DEFINE(Testing);  // 定义一个名为 Testing 的关键信息记录器
    TORCH_VITAL(Testing, Attribute0) << 1;  // 记录整数属性 Attribute0 的值为 1
    TORCH_VITAL(Testing, Attribute1) << "1";  // 记录字符串属性 Attribute1 的值为 "1"
    TORCH_VITAL(Testing, Attribute2) << 1.0f;  // 记录浮点数属性 Attribute2 的值为 1.0f
    TORCH_VITAL(Testing, Attribute3) << 1.0;  // 记录双精度浮点数属性 Attribute3 的值为 1.0
    auto t = at::ones({1, 1});  // 创建一个大小为 1x1 的张量 t，所有元素为 1
    TORCH_VITAL(Testing, Attribute4) << t;  // 记录张量属性 Attribute4 的值为张量 t 的内容
  }
  std::cout.rdbuf(sbuf);  // 恢复标准输出流缓冲区

  auto s = buffer.str();  // 将捕获的输出转换为字符串
  ASSERT_THAT(s, HasSubstr("Testing.Attribute0\t\t 1"));  // 断言捕获的输出包含指定的字符串
  ASSERT_THAT(s, HasSubstr("Testing.Attribute1\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute2\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute3\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute4\t\t  1"));
}

TEST(Vitals, MultiString) {  // 测试用例：多个字符串测试
  std::stringstream buffer;  // 创建一个字符串流对象，用于捕获输出

  std::streambuf* sbuf = std::cout.rdbuf();  // 获取标准输出的流缓冲区
  std::cout.rdbuf(buffer.rdbuf());  // 重定向标准输出到字符串流
  {
#ifdef _WIN32
    _putenv("TORCH_VITAL=1");  // 在 Windows 下设置环境变量 TORCH_VITAL 为 1
#else
    setenv("TORCH_VITAL", "1", 1);  // 在 POSIX 系统下设置环境变量 TORCH_VITAL 为 "1"
#endif
    TORCH_VITAL_DEFINE(Testing);  // 定义一个名为 Testing 的关键信息记录器
    TORCH_VITAL(Testing, Attribute0) << 1 << " of " << 2;  // 记录多个值到 Attribute0
    TORCH_VITAL(Testing, Attribute1) << 1;  // 记录整数属性 Attribute1 的值为 1
    TORCH_VITAL(Testing, Attribute1) << " of ";  // 继续记录字符串到 Attribute1
    TORCH_VITAL(Testing, Attribute1) << 2;  // 继续记录整数到 Attribute1
  }
  std::cout.rdbuf(sbuf);  // 恢复标准输出流缓冲区

  auto s = buffer.str();  // 将捕获的输出转换为字符串
  ASSERT_THAT(s, HasSubstr("Testing.Attribute0\t\t 1 of 2"));  // 断言捕获的输出包含指定的字符串
  ASSERT_THAT(s, HasSubstr("Testing.Attribute1\t\t 1 of 2"));
}

TEST(Vitals, OnAndOff) {  // 测试用例：开启和关闭测试
  for (const auto i : c10::irange(2)) {  // 遍历值为 0 和 1 的范围
    std::stringstream buffer;  // 创建一个字符串流对象，用于捕获输出

    std::streambuf* sbuf = std::cout.rdbuf();  // 获取标准输出的流缓冲区
    std::cout.rdbuf(buffer.rdbuf());  // 重定向标准输出到字符串流
    {
#ifdef _WIN32
      if (i) {
        _putenv("TORCH_VITAL=1");  // 在 Windows 下设置环境变量 TORCH_VITAL 为 1 或 0
      } else {
        _putenv("TORCH_VITAL=0");
      }
#else
      setenv("TORCH_VITAL", i ? "1" : "", 1);  // 在 POSIX 系统下设置环境变量 TORCH_VITAL 为 "1" 或 ""
#endif
      TORCH_VITAL_DEFINE(Testing);  // 定义一个名为 Testing 的关键信息记录器
      TORCH_VITAL(Testing, Attribute0) << 1;  // 记录整数属性 Attribute0 的值为 1
    }
    std::cout.rdbuf(sbuf);  // 恢复标准输出流缓冲区

    auto s = buffer.str();  // 将捕获的输出转换为字符串
    auto f = s.find("Testing.Attribute0\t\t 1");  // 查找是否包含特定的输出
    if (i) {
      ASSERT_TRUE(f != std::string::npos);  // 断言输出包含指定的字符串
    } else {
      ASSERT_TRUE(f == std::string::npos);  // 断言输出不包含指定的字符串
    }
  }
}

TEST(Vitals, APIVitals) {  // 测试用例：API 关键信息记录测试
  std::stringstream buffer;  // 创建一个字符串流对象，用于捕获输出
  bool rvalue;  // 声明一个布尔值变量

  std::streambuf* sbuf = std::cout.rdbuf();  // 获取标准输出的流缓冲区
  std::cout.rdbuf(buffer.rdbuf());  // 重定向标准输出到字符串流
  {
#ifdef _WIN32
    _putenv("TORCH_VITAL=1");  // 在 Windows 下设置环境变量 TORCH_VITAL 为 1
#else
    setenv("TORCH_VITAL", "1", 1);  // 在 POSIX 系统下设置环境变量 TORCH_VITAL 为 "1"
#endif
    APIVitals api_vitals;  // 创建 APIVitals 对象
    rvalue = api_vitals.setVital("TestingSetVital", "TestAttr", "TestValue");  // 设置关键信息记录的值
  }
  std::cout.rdbuf(sbuf);  // 恢复标准输出流缓冲区

  auto s = buffer.str();  // 将捕获的输出转换为字符串
  ASSERT_TRUE(rvalue);  // 断言 API 设置操作成功
  ASSERT_THAT(s, HasSubstr("TestingSetVital.TestAttr\t\t TestValue"));  // 断言捕获的输出包含指定的字符串
}
```