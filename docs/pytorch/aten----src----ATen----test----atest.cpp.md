# `.\pytorch\aten\src\ATen\test\atest.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <iostream>
using namespace std;
using namespace at;

class atest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 初始化 x_tensor 和 y_tensor 张量
    x_tensor = tensor({10, -1, 0, 1, -10});
    y_tensor = tensor({-10, 1, 0, -1, 10});
    // 初始化 x_logical 和 y_logical 张量
    x_logical = tensor({1, 1, 0, 1, 0});
    y_logical = tensor({0, 1, 0, 1, 1});
    // 初始化 x_float 和 y_float 张量
    x_float = tensor({2.0, 2.4, 5.6, 7.0, 36.0});
    y_float = tensor({1.0, 1.1, 8.7, 10.0, 24.0});
  }

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor x_tensor, y_tensor;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor x_logical, y_logical;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Tensor x_float, y_float;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INT = 1;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int FLOAT = 2;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INTFLOAT = 3;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INTBOOL = 5;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  const int INTBOOLFLOAT = 7;
};

namespace BinaryOpsKernel {
const int IntMask = 1; // test dtype = kInt
const int FloatMask = 2; // test dtype = kFloat
const int BoolMask = 4; // test dtype = kBool
} // namespace BinaryOpsKernel

template <typename T, typename... Args>
void unit_binary_ops_test(
    T func,
    const Tensor& x_tensor,
    const Tensor& y_tensor,
    const Tensor& exp,
    ScalarType dtype,
    Args... args) {
  // 创建空的输出张量
  auto out_tensor = empty({5}, dtype);
  // 调用函数 func 进行运算，并将结果存储在 out_tensor 中
  func(out_tensor, x_tensor.to(dtype), y_tensor.to(dtype), args...);
  // 断言输出张量的数据类型为 dtype
  ASSERT_EQ(out_tensor.dtype(), dtype);
  // 如果 dtype 是浮点类型，使用 allclose 进行近似相等性检查；否则使用 equal 进行精确相等性检查
  if (dtype == kFloat) {
    ASSERT_TRUE(exp.to(dtype).allclose(out_tensor));
  } else {
    ASSERT_TRUE(exp.to(dtype).equal(out_tensor));
  }
}

/*
  用于运行二元运算测试的模板函数
  - exp: 期望的输出张量
  - func: 待测试的函数
  - option: 3 位掩码，
    - 第 1 位: 测试整数张量的运算
    - 第 2 位: 测试浮点张量的运算
    - 第 3 位: 测试布尔张量的运算
    例如，如果函数应该在整数/布尔张量上进行测试但不涉及浮点张量，则 option 应为 1 * 1 + 0 * 2 + 1 * 4 = 5。
    如果在所有类型上进行测试，option 应为 7。
*/
template <typename T, typename... Args>
void run_binary_ops_test(
    T func,
    const Tensor& x_tensor,
    const Tensor& y_tensor,
    const Tensor& exp,
    int option,
    Args... args) {
  // 测试整数张量的运算
  if (option & BinaryOpsKernel::IntMask) {
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kInt, args...);
  }

  // 测试浮点张量的运算
  if (option & BinaryOpsKernel::FloatMask) {
    // （未完成）：在此添加浮点张量运算的测试
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kFloat, args...);
  }

  // 如果选项中包含浮点数掩码，则对浮点数张量进行二元操作测试
  if (option & BinaryOpsKernel::FloatMask) {
    // 调用单元测试函数，测试给定函数对浮点数张量的操作
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kFloat, args...);
  }

  // 如果选项中包含布尔值掩码，则对布尔值张量进行二元操作测试
  if (option & BinaryOpsKernel::BoolMask) {
    // 调用单元测试函数，测试给定函数对布尔值张量的操作
    unit_binary_ops_test(func, x_tensor, y_tensor, exp, kBool, args...);
  }
}

// 定义一个名为 trace 的函数，没有参数和返回值
void trace() {
  // 创建一个大小为 12x12 的随机张量 foo
  Tensor foo = rand({12, 12});

  // 将 foo 转换为二维张量，元素类型为 float
  auto foo_a = foo.accessor<float, 2>();
  // 初始化 trace 变量为 0
  float trace = 0;

  // 遍历 foo_a 的第一维，计算对角线元素之和，即迹(trace)
  for (const auto i : c10::irange(foo_a.size(0))) {
    trace += foo_a[i][i];
  }

  // 使用 ASSERT_FLOAT_EQ 断言 foo 的迹与计算得到的 trace 相等
  ASSERT_FLOAT_EQ(foo.trace().item<float>(), trace);
}

// 定义一个名为 operators 的测试用例
TEST_F(atest, operators) {
  // 初始化整数 a 和 b 的值
  int a = 0b10101011;
  int b = 0b01111011;

  // 创建包含整数 a 和 b 的张量 a_tensor 和 b_tensor
  auto a_tensor = tensor({a});
  auto b_tensor = tensor({b});

  // 使用 ASSERT_TRUE 断言位取反操作的结果
  ASSERT_TRUE(tensor({~a}).equal(~a_tensor));
  // 使用 ASSERT_TRUE 断言按位或操作的结果
  ASSERT_TRUE(tensor({a | b}).equal(a_tensor | b_tensor));
  // 使用 ASSERT_TRUE 断言按位与操作的结果
  ASSERT_TRUE(tensor({a & b}).equal(a_tensor & b_tensor));
  // 使用 ASSERT_TRUE 断言按位异或操作的结果
  ASSERT_TRUE(tensor({a ^ b}).equal(a_tensor ^ b_tensor));
}

// 定义名为 logical_and_operators 的测试用例
TEST_F(atest, logical_and_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({0, 1, 0, 1, 0});
  // 调用 run_binary_ops_test 函数，测试逻辑与操作
  run_binary_ops_test(
      logical_and_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 logical_or_operators 的测试用例
TEST_F(atest, logical_or_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({1, 1, 0, 1, 1});
  // 调用 run_binary_ops_test 函数，测试逻辑或操作
  run_binary_ops_test(
      logical_or_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 logical_xor_operators 的测试用例
TEST_F(atest, logical_xor_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({1, 0, 0, 0, 1});
  // 调用 run_binary_ops_test 函数，测试逻辑异或操作
  run_binary_ops_test(
      logical_xor_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 lt_operators 的测试用例
TEST_F(atest, lt_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({0, 0, 0, 0, 1});
  // 调用 run_binary_ops_test 函数，测试小于操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      lt_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 le_operators 的测试用例
TEST_F(atest, le_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({0, 1, 1, 1, 1});
  // 调用 run_binary_ops_test 函数，测试小于等于操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      le_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 gt_operators 的测试用例
TEST_F(atest, gt_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({1, 0, 0, 0, 0});
  // 调用 run_binary_ops_test 函数，测试大于操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      gt_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 ge_operators 的测试用例
TEST_F(atest, ge_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({1, 1, 1, 1, 0});
  // 调用 run_binary_ops_test 函数，测试大于等于操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      ge_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 eq_operators 的测试用例
TEST_F(atest, eq_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({0, 1, 1, 1, 0});
  // 调用 run_binary_ops_test 函数，测试等于操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      eq_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 ne_operators 的测试用例
TEST_F(atest, ne_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({1, 0, 0, 0, 1});
  // 调用 run_binary_ops_test 函数，测试不等于操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(
      ne_out, x_logical, y_logical, exp_tensor, INTBOOL);
}

// 定义名为 add_operators 的测试用例
TEST_F(atest, add_operators) {
  // 创建期望的结果张量 exp_tensor
  auto exp_tensor = tensor({-10, 1, 0, -1, 10});
  // 调用 run_binary_ops_test 函数，测试加法操作
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Scalar&)>(
      add_out, x_tensor, y_tensor, exp_tensor, INTBOOL, 2);
}
TEST_F(atest, max_operators) {
  // 定义期望输出的张量，包含数值 {10, 1, 0, 1, 10}
  auto exp_tensor = tensor({10, 1, 0, 1, 10});
  // 运行二元操作的测试，使用 max_out 函数，传入 x_tensor, y_tensor, exp_tensor 和 INTBOOLFLOAT
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(max_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);
}

TEST_F(atest, min_operators) {
  // 定义期望输出的张量，包含数值 {-10, -1, 0, -1, -10}
  auto exp_tensor = tensor({-10, -1, 0, -1, -10});
  // 运行二元操作的测试，使用 min_out 函数，传入 x_tensor, y_tensor, exp_tensor 和 INTBOOLFLOAT
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(min_out, x_tensor, y_tensor, exp_tensor, INTBOOLFLOAT);
}

TEST_F(atest, sigmoid_backward_operator) {
  // 定义期望输出的张量，包含数值 {-1100, 0, 0, -2, 900}
  auto exp_tensor = tensor({-1100, 0, 0, -2, 900});
  // 只测试类型为 Float 的情况
  // 运行二元操作的测试，使用 sigmoid_backward_out 函数，传入 x_tensor, y_tensor, exp_tensor 和 FLOAT
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(sigmoid_backward_out, x_tensor, y_tensor, exp_tensor, FLOAT);
}

TEST_F(atest, fmod_tensor_operators) {
  // 定义期望输出的张量，包含数值 {0.0, 0.2, 5.6, 7.0, 12.0}
  auto exp_tensor = tensor({0.0, 0.2, 5.6, 7.0, 12.0});
  // 运行二元操作的测试，使用 fmod_out 函数，传入 x_float, y_float, exp_tensor 和 INTFLOAT
  run_binary_ops_test<
      at::Tensor& (*)(at::Tensor&, const at::Tensor&, const at::Tensor&)>(fmod_out, x_float, y_float, exp_tensor, INTFLOAT);
}

// TEST_CASE( "atest", "[]" ) {
TEST_F(atest, atest) {
  // 设置随机种子为 123
  manual_seed(123);

  // 生成一个大小为 {12, 6} 的随机张量
  auto foo = rand({12, 6});

  // 断言张量的大小为 12x6
  ASSERT_EQ(foo.size(0), 12);
  ASSERT_EQ(foo.size(1), 6);

  // 对张量进行运算： foo = foo + foo * 3
  foo = foo + foo * 3;
  // 对张量进行减法操作： foo -= 4
  foo -= 4;

  // 定义标量 a，并将其转换为 float 类型
  Scalar a = 4;
  float b = a.to<float>();
  // 断言转换后的值为 4
  ASSERT_EQ(b, 4);

  // 将张量转换为 kByte 类型
  foo = ((foo * foo) == (foo.pow(3))).to(kByte);
  // 对张量进行数值运算： foo = 2 + (foo + 1)
  foo = 2 + (foo + 1);

  // 获取张量的访问器，类型为 uint8_t，维度为 2
  auto foo_v = foo.accessor<uint8_t, 2>();

  // 使用循环对 foo_v 中的每个元素加一
  for (const auto i : c10::irange(foo_v.size(0))) {
    for (const auto j : c10::irange(foo_v.size(1))) {
      foo_v[i][j]++;
    }
  }

  // 断言 foo 是否等于一个全部为 4 的张量，类型为 kByte
  ASSERT_TRUE(foo.equal(4 * ones({12, 6}, kByte)));

  // 执行跟踪函数 trace()
  trace();

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  // 定义一个 float 类型的数组 data
  float data[] = {1, 2, 3, 4, 5, 6};

  // 从数据中创建张量 f，形状为 {1, 2, 3}
  auto f = from_blob(data, {1, 2, 3});
  // 获取张量 f 的访问器，类型为 float，维度为 3
  auto f_a = f.accessor<float, 3>();

  // 断言张量 f_a 中的特定元素值
  ASSERT_EQ(f_a[0][0][0], 1.0);
  ASSERT_EQ(f_a[0][1][1], 5.0);

  // 断言张量 f 的步长和大小
  ASSERT_EQ(f.strides()[0], 6);
  ASSERT_EQ(f.strides()[1], 3);
  ASSERT_EQ(f.strides()[2], 1);
  ASSERT_EQ(f.sizes()[0], 1);
  ASSERT_EQ(f.sizes()[1], 2);
  ASSERT_EQ(f.sizes()[2], 3);

  // 尝试对张量 f 执行大小变换，并断言抛出异常
  // TODO(ezyang): maybe do a more precise exception type.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(f.resize_({3, 4, 5}), std::exception);
  
  {
    int isgone = 0;
    {
      // 创建一个新的张量 f2，从 data 创建，形状为 {1, 2, 3}，并在析构时增加 isgone 计数
      auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
    }
    // 断言 isgone 的值为 1
    ASSERT_EQ(isgone, 1);
  }
  
  {
    int isgone = 0;
    Tensor a_view;
    {
      // 创建一个新的张量 f2，从 data 创建，形状为 {1, 2, 3}，并在析构时增加 isgone 计数
      auto f2 = from_blob(data, {1, 2, 3}, [&](void*) { isgone++; });
      // 创建一个 f2 的视图，形状为 {3, 2, 1}，并赋值给 a_view
      a_view = f2.view({3, 2, 1});
    }
    // 断言 isgone 的值为 0
    ASSERT_EQ(isgone, 0);
    // 重置 a_view
    a_view.reset();
    // 断言 isgone 的值为 1
    ASSERT_EQ(isgone, 1);
  }

  // 如果当前环境支持 CUDA
  if (at::hasCUDA()) {
    int isgone = 0;
    {
      // 创建一个新的空张量 base，形状为 {1, 2, 3}，类型为 kCUDA，并在析构时增加 isgone 计数
      auto base = at::empty({1, 2, 3}, TensorOptions(kCUDA));
      auto f2 = from_blob(base.mutable_data_ptr(), {1, 2, 3}, [&](void*) { isgone++; });
    }
    // 断言 isgone 的值为 1

    // 尝试在 from_blob 中指定错误的设备
    // Attempt to specify the wrong device in from_blob
    // 暂未完全注释完，可能需要继续添加
    # 创建一个新的张量 `t`，形状为 [1, 2, 3]，在 CUDA 设备上，使用设备索引 0
    auto t = at::empty({1, 2, 3}, TensorOptions(kCUDA, 0));
    
    # 使用 NOLINTNEXTLINE 禁止特定的静态分析规则，这里忽略了 cppcoreguidelines-avoid-goto 和 hicpp-avoid-goto
    # 期望在执行以下代码时抛出异常：从张量 `t` 的数据指针创建 blob，并指定形状为 [1, 2, 3]，目标设备为 CUDA 设备索引 1
    EXPECT_ANY_THROW(from_blob(t.data_ptr(), {1, 2, 3}, at::Device(kCUDA, 1)));

    # 推断正确的设备
    # 从张量 `t` 的数据指针创建 blob，并指定形状为 [1, 2, 3]，自动推断使用 CUDA 设备
    auto t_ = from_blob(t.data_ptr(), {1, 2, 3}, kCUDA);
    
    # 使用断言验证 `t_` 的设备是否为 CUDA 设备索引 0
    ASSERT_EQ(t_.device(), at::Device(kCUDA, 0));
}



# 这行代码关闭了一个代码块，结束了某个代码结构（如函数、循环、条件语句等）的定义。
```