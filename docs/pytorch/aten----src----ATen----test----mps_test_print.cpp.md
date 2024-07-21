# `.\pytorch\aten\src\ATen\test\mps_test_print.cpp`

```
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <limits>
#include <sstream>

// 判断字符串 str 是否以字符串 suffix 结尾的函数
bool ends_with(const std::string& str, const std::string& suffix) {
  // 获取字符串 str 和 suffix 的长度
  const auto str_len = str.length();
  const auto suffix_len = suffix.length();
  // 如果 str 的长度小于 suffix 的长度，直接返回 false
  return str_len < suffix_len ? false : suffix == str.substr(str_len - suffix_len, suffix_len);
}

// 测试用例，打印随机生成的浮点数矩阵
TEST(MPSPrintTest, PrintFloatMatrix) {
  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 向 ss 中插入随机生成的浮点数矩阵的字符串表示
  ss << torch::randn({3, 3}, at::device(at::kMPS));
  // 断言 ss 的内容以指定格式结尾，如果不是则输出错误信息
  ASSERT_TRUE (ends_with(ss.str(), "[ MPSFloatType{3,3} ]")) << " got " << ss.str();
}

// 测试用例，打印随机生成的半精度四维张量
TEST(MPSPrintTest, PrintHalf4DTensor) {
  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 向 ss 中插入随机生成的半精度四维张量的字符串表示
  ss << torch::randn({2, 2, 2, 2}, at::device(at::kMPS).dtype(at::kHalf));
  // 断言 ss 的内容以指定格式结尾，如果不是则输出错误信息
  ASSERT_TRUE (ends_with(ss.str(), "[ MPSHalfType{2,2,2,2} ]")) << " got " << ss.str();
}

// 测试用例，打印填充为指定最大整数值的长整型矩阵
TEST(MPSPrintTest, PrintLongMatrix) {
  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 向 ss 中插入填充为指定最大整数值的长整型矩阵的字符串表示
  ss << torch::full({2, 2}, std::numeric_limits<int>::max(), at::device(at::kMPS));
  // 断言 ss 的内容以指定格式结尾，如果不是则输出错误信息
  ASSERT_TRUE (ends_with(ss.str(), "[ MPSLongType{2,2} ]")) << " got " << ss.str();
}

// 测试用例，打印随机生成的浮点数标量
TEST(MPSPrintTest, PrintFloatScalar) {
  // 创建一个字符串流对象 ss
  std::stringstream ss;
  // 向 ss 中插入随机生成的浮点数标量的字符串表示
  ss << torch::ones({}, at::device(at::kMPS));
  // 断言 ss 的内容等于指定的字符串，如果不是则输出错误信息
  ASSERT_TRUE(ss.str() == "1\n[ MPSFloatType{} ]") << " got " << ss.str();
}
```