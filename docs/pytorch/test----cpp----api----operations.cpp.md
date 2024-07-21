# `.\pytorch\test\cpp\api\operations.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <c10/util/irange.h>  // 包含 C10 库中的 irange 函数头文件
#include <torch/torch.h>  // 包含 PyTorch 库的头文件

#include <test/cpp/api/support.h>  // 包含测试辅助函数的头文件

struct OperationTest : torch::test::SeedingFixture {  // 定义测试结构体 OperationTest，继承自 SeedingFixture
 protected:
  void SetUp() override {}  // 设置测试环境的初始化函数

  const int TEST_AMOUNT = 10;  // 定义常量 TEST_AMOUNT，用于测试次数
};

TEST_F(OperationTest, Lerp) {  // 定义测试用例 Lerp
  for (const auto i : c10::irange(TEST_AMOUNT)) {  // 循环执行 TEST_AMOUNT 次
    (void)i; // Suppress unused variable warning (抑制未使用变量的警告)
    // test lerp_kernel_scalar (测试标量插值)
    auto start = torch::rand({3, 5});  // 随机生成一个形状为 [3, 5] 的张量 start
    auto end = torch::rand({3, 5});  // 随机生成一个形状为 [3, 5] 的张量 end
    auto scalar = 0.5;  // 定义标量 scalar
    // expected and actual (期望值和实际值)
    auto scalar_expected = start + scalar * (end - start);  // 计算标量插值的期望结果
    auto out = torch::lerp(start, end, scalar);  // 使用 torch::lerp 计算标量插值的实际结果
    // compare (比较结果)
    ASSERT_EQ(out.dtype(), scalar_expected.dtype());  // 断言实际输出的数据类型与期望输出一致
    ASSERT_TRUE(out.allclose(scalar_expected));  // 断言实际输出与期望输出在数值上近似

    // test lerp_kernel_tensor (测试张量插值)
    auto weight = torch::rand({3, 5});  // 随机生成一个形状为 [3, 5] 的权重张量 weight
    // expected and actual (期望值和实际值)
    auto tensor_expected = start + weight * (end - start);  // 计算张量插值的期望结果
    out = torch::lerp(start, end, weight);  // 使用 torch::lerp 计算张量插值的实际结果
    // compare (比较结果)
    ASSERT_EQ(out.dtype(), tensor_expected.dtype());  // 断言实际输出的数据类型与期望输出一致
    ASSERT_TRUE(out.allclose(tensor_expected));  // 断言实际输出与期望输出在数值上近似
  }
}

TEST_F(OperationTest, Cross) {  // 定义测试用例 Cross
  for (const auto i : c10::irange(TEST_AMOUNT)) {  // 循环执行 TEST_AMOUNT 次
    (void)i; // Suppress unused variable warning (抑制未使用变量的警告)
    // input (输入)
    auto a = torch::rand({10, 3});  // 随机生成一个形状为 [10, 3] 的张量 a
    auto b = torch::rand({10, 3});  // 随机生成一个形状为 [10, 3] 的张量 b
    // expected (期望输出)
    auto exp = torch::empty({10, 3});  // 创建一个空的形状为 [10, 3] 的张量 exp
    for (const auto j : c10::irange(10)) {  // 遍历 0 到 9 的循环
      auto u1 = a[j][0], u2 = a[j][1], u3 = a[j][2];  // 提取张量 a 第 j 行的三个元素
      auto v1 = b[j][0], v2 = b[j][1], v3 = b[j][2];  // 提取张量 b 第 j 行的三个元素
      exp[j][0] = u2 * v3 - v2 * u3;  // 计算交叉乘积的 x 分量
      exp[j][1] = v1 * u3 - u1 * v3;  // 计算交叉乘积的 y 分量
      exp[j][2] = u1 * v2 - v1 * u2;  // 计算交叉乘积的 z 分量
    }
    // actual (实际输出)
    auto out = torch::cross(a, b);  // 使用 torch::cross 计算张量 a 和 b 的交叉乘积
    // compare (比较结果)
    ASSERT_EQ(out.dtype(), exp.dtype());  // 断言实际输出的数据类型与期望输出一致
    ASSERT_TRUE(out.allclose(exp));  // 断言实际输出与期望输出在数值上近似
  }
}

TEST_F(OperationTest, Linear_out) {  // 定义测试用例 Linear_out
  {
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});  // 创建一个形状为 [3, 3, 2] 的张量 x
    const auto w = torch::arange(200., 206).resize_({3, 2});  // 创建一个形状为 [3, 2] 的权重张量 w
    const auto b = torch::arange(300., 303);  // 创建一个偏置张量 b
    auto y = torch::empty({3, 3, 3});  // 创建一个空的形状为 [3, 3, 3] 的张量 y
    at::linear_out(y, x, w, b);  // 使用 at::linear_out 函数进行线性计算，结果写入 y
    const auto y_exp = torch::tensor(  // 创建期望结果张量 y_exp
        {{{40601, 41004, 41407}, {41403, 41814, 42225}, {42205, 42624, 43043}},
         {{43007, 43434, 43861}, {43809, 44244, 44679}, {44611, 45054, 45497}},
         {{45413, 45864, 46315}, {46215, 46674, 47133}, {47017, 47484, 47951}}},
        torch::kFloat);  // 指定期望结果张量的数据类型为浮点型
    ASSERT_TRUE(torch::allclose(y, y_exp));  // 断言实际输出 y 与期望输出 y_exp 在数值上近似
  }
  {
    const auto x = torch::arange(100., 118).resize_({3, 3, 2});  // 创建一个形状为 [3, 3, 2] 的张量 x
    const auto w = torch::arange(200., 206).resize_({3, 2});  // 创建一个形状为 [3, 2] 的权重张量 w
    auto y = torch::empty({3, 3, 3});  // 创建一个空的形状为 [3, 3, 3] 的张量 y
    at::linear_out(y, x, w);  // 使用 at::linear_out 函数进行线性计算，结果写入 y
    ASSERT_EQ(y.ndimension(), 3);  // 断言输出张量 y 的维度为 3
    ASSERT_EQ(y.sizes(), torch::IntArrayRef({3, 3, 3}));  // 断言输出张量 y 的形状为 [3, 3, 3]
    const auto y_exp = torch::tensor(  // 创建期望结果张量 y_exp
        {{{40301, 40703, 41105}, {41103, 41513, 41923}, {41905, 42323, 42741}},
         {{42707, 43133, 43559}, {43509, 43943, 44377}, {44311, 44753, 45195}},
         {{45113, 45563, 46013}, {45915, 46373, 46831}, {46717, 47183, 47649}}},
        torch::kFloat);
    # 使用断言进行测试，验证张量 y 是否与预期张量 y_exp 在数值上全部接近
    ASSERT_TRUE(torch::allclose(y, y_exp));
}


注释：


# 这行代码关闭了一个代码块。通常情况下，这种结构会与一对大括号（{}）的开头对应，表示一个函数或控制流结构的结束。
```