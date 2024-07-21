# `.\pytorch\test\cpp\aoti_abi_check\test_vec.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/cpu/vec/vec.h>  // 包含 ATen 的 CPU 向量化功能的头文件

#include <iostream>  // 输入输出流的头文件
namespace torch {
namespace aot_inductor {

TEST(TestVec, TestAdd) {  // 定义测试用例 TestAdd
  using Vec = at::vec::Vectorized<int>;  // 使用 ATen 的整数向量化类型 Vec
  std::vector<int> a(1024, 1);  // 创建长度为 1024 的整数向量 a，初始值为 1
  std::vector<int> b(1024, 2);  // 创建长度为 1024 的整数向量 b，初始值为 2
  Vec a_vec = Vec::loadu(a.data());  // 将向量 a 载入 Vec 类型的向量 a_vec
  Vec b_vec = Vec::loadu(b.data());  // 将向量 b 载入 Vec 类型的向量 b_vec
  Vec actual_vec = a_vec + b_vec;  // 计算向量 a_vec 和 b_vec 的逐元素相加结果
  std::vector<int> expected(1024, 3);  // 创建长度为 1024 的整数向量 expected，每个元素为 3
  Vec expected_vec = Vec::loadu(expected.data());  // 将向量 expected 载入 Vec 类型的向量 expected_vec

  for (int i = 0; i < Vec::size(); i++) {  // 遍历向量中的每个元素
    EXPECT_EQ(expected_vec[i], actual_vec[i]);  // 断言实际结果与期望结果相等
  }
}

TEST(TestVec, TestMax) {  // 定义测试用例 TestMax
  using Vec = at::vec::Vectorized<int>;  // 使用 ATen 的整数向量化类型 Vec
  std::vector<int> a(1024, -1);  // 创建长度为 1024 的整数向量 a，初始值为 -1
  std::vector<int> b(1024, 2);  // 创建长度为 1024 的整数向量 b，初始值为 2
  Vec a_vec = Vec::loadu(a.data());  // 将向量 a 载入 Vec 类型的向量 a_vec
  Vec b_vec = Vec::loadu(b.data());  // 将向量 b 载入 Vec 类型的向量 b_vec
  Vec actual_vec = at::vec::maximum(a_vec, b_vec);  // 计算向量 a_vec 和 b_vec 的逐元素最大值
  Vec expected_vec = b_vec;  // 将向量 b_vec 赋值给期望向量 expected_vec

  for (int i = 0; i < Vec::size(); i++) {  // 遍历向量中的每个元素
    EXPECT_EQ(expected_vec[i], actual_vec[i]);  // 断言实际结果与期望结果相等
  }
}

TEST(TestVec, TestMin) {  // 定义测试用例 TestMin
  using Vec = at::vec::Vectorized<int>;  // 使用 ATen 的整数向量化类型 Vec
  std::vector<int> a(1024, -1);  // 创建长度为 1024 的整数向量 a，初始值为 -1
  std::vector<int> b(1024, 2);  // 创建长度为 1024 的整数向量 b，初始值为 2
  Vec a_vec = Vec::loadu(a.data());  // 将向量 a 载入 Vec 类型的向量 a_vec
  Vec b_vec = Vec::loadu(b.data());  // 将向量 b 载入 Vec 类型的向量 b_vec
  Vec actual_vec = at::vec::minimum(a_vec, b_vec);  // 计算向量 a_vec 和 b_vec 的逐元素最小值
  Vec expected_vec = a_vec;  // 将向量 a_vec 赋值给期望向量 expected_vec

  for (int i = 0; i < Vec::size(); i++) {  // 遍历向量中的每个元素
    EXPECT_EQ(expected_vec[i], actual_vec[i]);  // 断言实际结果与期望结果相等
  }
}

TEST(TestVec, TestConvert) {  // 定义测试用例 TestConvert
  std::vector<int> a(1024, -1);  // 创建长度为 1024 的整数向量 a，初始值为 -1
  std::vector<float> b(1024, -1.0);  // 创建长度为 1024 的浮点数向量 b，初始值为 -1.0
  at::vec::Vectorized<int> a_vec = at::vec::Vectorized<int>::loadu(a.data());  // 将向量 a 载入整数向量化类型 a_vec
  at::vec::Vectorized<float> b_vec = at::vec::Vectorized<float>::loadu(b.data());  // 将向量 b 载入浮点数向量化类型 b_vec
  auto actual_vec = at::vec::convert<float>(a_vec);  // 将整数向量化类型 a_vec 转换为浮点数向量化类型
  auto expected_vec = b_vec;  // 将浮点数向量化类型 b_vec 赋值给期望向量 expected_vec

  for (int i = 0; i < at::vec::Vectorized<int>::size(); i++) {  // 遍历向量中的每个元素
    EXPECT_EQ(expected_vec[i], actual_vec[i]);  // 断言实际结果与期望结果相等
  }
}

TEST(TestVec, TestClampMin) {  // 定义测试用例 TestClampMin
  using Vec = at::vec::Vectorized<float>;  // 使用 ATen 的浮点数向量化类型 Vec
  std::vector<float> a(1024, -2.0);  // 创建长度为 1024 的浮点数向量 a，初始值为 -2.0
  std::vector<float> min(1024, -1.0);  // 创建长度为 1024 的浮点数向量 min，初始值为 -1.0
  Vec a_vec = Vec::loadu(a.data());  // 将向量 a 载入 Vec 类型的向量 a_vec
  Vec min_vec = Vec::loadu(min.data());  // 将向量 min 载入 Vec 类型的向量 min_vec
  Vec actual_vec = at::vec::clamp_min(a_vec, min_vec);  // 将向量 a_vec 的每个元素与 min_vec 的对应元素进行下限截断
  Vec expected_vec = min_vec;  // 将向量 min_vec 赋值给期望向量 expected_vec

  for (int i = 0; i < Vec::size(); i++) {  // 遍历向量中的每个元素
    EXPECT_EQ(expected_vec[i], actual_vec[i]);  // 断言实际结果与期望结果相等
  }
}

} // namespace aot_inductor
} // namespace torch
```