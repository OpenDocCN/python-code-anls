# `.\pytorch\aten\src\ATen\test\pow_test.cpp`

```
#include <gtest/gtest.h>

#include <ATen/native/Pow.h>
#include <c10/util/irange.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <iostream>
#include <vector>
#include <type_traits>

using namespace at;

namespace {

// 定义各种数值类型的最小值和最大值
const auto int_min = std::numeric_limits<int>::min();
const auto int_max = std::numeric_limits<int>::max();
const auto long_min = std::numeric_limits<int64_t>::min();
const auto long_max = std::numeric_limits<int64_t>::max();
const auto float_lowest = std::numeric_limits<float>::lowest();
const auto float_min = std::numeric_limits<float>::min();
const auto float_max = std::numeric_limits<float>::max();
const auto double_lowest = std::numeric_limits<double>::lowest();
const auto double_min = std::numeric_limits<double>::min();
const auto double_max = std::numeric_limits<double>::max();

// 定义包含不同整数值的向量
const std::vector<int> ints {
  int_min,
  int_min + 1,
  int_min + 2,
  static_cast<int>(-sqrt(static_cast<double>(int_max))),
  -3, -2, -1, 0, 1, 2, 3,
  static_cast<int>(sqrt(static_cast<double>(int_max))),
  int_max - 2,
  int_max - 1,
  int_max
};

// 定义包含非负整数值的向量
const std::vector<int> non_neg_ints {
  0, 1, 2, 3,
  static_cast<int>(sqrt(static_cast<double>(int_max))),
  int_max - 2,
  int_max - 1,
  int_max
};

// 定义包含不同长整数值的向量
const std::vector<int64_t> longs {
  long_min,
  long_min + 1,
  long_min + 2,
  static_cast<int64_t>(-sqrt(static_cast<double>(long_max))),
  -3, -2, -1, 0, 1, 2, 3,
  static_cast<int64_t>(sqrt(static_cast<double>(long_max))),
  long_max - 2,
  long_max - 1,
  long_max
};

// 定义包含非负长整数值的向量
const std::vector<int64_t> non_neg_longs {
  0, 1, 2, 3,
  static_cast<int64_t>(sqrt(static_cast<double>(long_max))),
  long_max - 2,
  long_max - 1,
  long_max
};

// 定义包含不同浮点数值的向量
const std::vector<float> floats {
  float_lowest,
  -3.0f, -2.0f, -1.0f, -1.0f/2.0f, -1.0f/3.0f,
  -float_min,
  0.0,
  float_min,
  1.0f/3.0f, 1.0f/2.0f, 1.0f, 2.0f, 3.0f,
  float_max,
};

// 定义包含不同双精度浮点数值的向量
const std::vector<double> doubles {
  double_lowest,
  -3.0, -2.0, -1.0, -1.0/2.0, -1.0/3.0,
  -double_min,
  0.0,
  double_min,
  1.0/3.0, 1.0/2.0, 1.0, 2.0, 3.0,
  double_max,
};

// 断言函数模板，用于比较浮点数值是否相等
template <class T,
  typename std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
void assert_eq(T val, T act, T exp) {
  // 如果其中一个值是 NaN，则直接返回
  if (std::isnan(act) || std::isnan(exp)) {
    return;
  }
  // 使用 GTest 的断言函数 ASSERT_FLOAT_EQ 检查浮点数值是否相等
  ASSERT_FLOAT_EQ(act, exp);
}

// 断言函数模板，用于比较整数值是否相等
template <class T,
  typename std::enable_if_t<std::is_integral_v<T>, T>* = nullptr>
void assert_eq(T val, T act, T exp) {
  // 如果预期值和实际值都为 0，或者其中一个值为 0 而另一个不为 0，则直接返回
  if (val != 0 && act == 0) {
    return;
  }
  if (val != 0 && exp == 0) {
    return;
  }
  // 获取数值类型 T 的最小值
  const auto min = std::numeric_limits<T>::min();
  // 如果预期值为最小值且实际值不是，则直接返回
  if (exp == min && val != min) {
    return;
  }
  // 使用 GTest 的断言函数 ASSERT_EQ 检查整数值是否相等
  ASSERT_EQ(act, exp);
}

// 数值类型为浮点数的幂函数模板
template <class T,
  typename std::enable_if_t<std::is_floating_point_v<T>, T>* = nullptr>
T typed_pow(T base, T exp) {
  // 使用 std::pow 计算浮点数的幂
  return std::pow(base, exp);
}

// 数值类型为整数的幂函数模板
template <class T,
  typename std::enable_if_t<std::is_integral_v<T>, T>* = nullptr>
T typed_pow(T base, T exp) {
  // 使用 ATen 库中的 native::powi 计算整数的幂
  return native::powi(base, exp);
}

// 模板函数，接受两个向量作为参数
template<typename Vals, typename Pows>
// 对输入的值向量 vals 和指数向量 pows 进行张量的指数运算
void tensor_pow_scalar(const Vals vals, const Pows pows, const torch::ScalarType valsDtype, const torch::ScalarType dtype) {
  // 根据给定的值和数据类型创建张量 tensor
  const auto tensor = torch::tensor(vals, valsDtype);

  // 遍历指数向量 pows 中的每个指数值 pow
  for (const auto pow : pows) {
    // 检查 dtype 是否为 kInt，且 pow 是否大于 int 类型的最大值，如果是则抛出异常
    // NOLINTNEXTLINE(clang-diagnostic-implicit-const-int-float-conversion)
    if ( dtype == kInt && pow > static_cast<float>(std::numeric_limits<int>::max())) {
      // 值无法转换为 int 类型而导致溢出
      // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
      EXPECT_THROW(tensor.pow(pow), std::runtime_error);
      continue;
    }
    // 使用张量 tensor 对 pow 进行指数运算，返回结果 actual_pow
    auto actual_pow = tensor.pow(pow);

    // 创建一个与 actual_pow 具有相同形状的空张量 actual_pow_
    auto actual_pow_ = torch::empty_like(actual_pow);
    // 将 tensor 的值复制到 actual_pow_
    actual_pow_.copy_(tensor);
    // 对 actual_pow_ 进行 pow 次指数运算
    actual_pow_.pow_(pow);

    // 创建一个与 actual_pow 具有相同形状的空张量 actual_pow_out
    auto actual_pow_out = torch::empty_like(actual_pow);
    // 使用 torch::pow_out 函数将 tensor 的 pow 次幂赋值给 actual_pow_out
    torch::pow_out(actual_pow_out, tensor, pow);

    // 使用 torch::pow 函数计算 tensor 的 pow 次幂，返回 actual_torch_pow
    auto actual_torch_pow = torch::pow(tensor, pow);

    // 初始化索引变量 i
    int i = 0;
    // 遍历输入值向量 vals 中的每个值 val
    for (const auto val : vals) {
      // 计算期望的指数运算结果 exp
      const auto exp = torch::pow(torch::tensor({val}, dtype), torch::tensor(pow, dtype)).template item<double>();

      // 获取 actual_pow 的第 i 个元素，转换为 double 类型，与 exp 进行比较断言
      const auto act_pow = actual_pow[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_pow, exp);

      // 获取 actual_pow_ 的第 i 个元素，转换为 double 类型，与 exp 进行比较断言
      const auto act_pow_ = actual_pow_[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_pow_, exp);

      // 获取 actual_pow_out 的第 i 个元素，转换为 double 类型，与 exp 进行比较断言
      const auto act_pow_out = actual_pow_out[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_pow_out, exp);

      // 获取 actual_torch_pow 的第 i 个元素，转换为 double 类型，与 exp 进行比较断言
      const auto act_torch_pow = actual_torch_pow[i].to(at::kDouble).template item<double>();
      assert_eq<long double>(val, act_torch_pow, exp);

      // 更新索引 i
      i++;
    }
  }
}

// 对输入的值向量 vals 和指数向量 pows 进行标量的张量指数运算
template<typename Vals, typename Pows>
void scalar_pow_tensor(const Vals vals, c10::ScalarType vals_dtype, const Pows pows, c10::ScalarType pows_dtype) {
  // 定义类型 T 为指数向量 Pows 中的值类型
  using T = typename Pows::value_type;

  // 根据给定的指数向量 pows 和数据类型 pows_dtype 创建张量 pow_tensor
  const auto pow_tensor = torch::tensor(pows, pows_dtype);

  // 遍历输入值向量 vals 中的每个值 val
  for (const auto val : vals) {
    // 使用 torch::pow 函数对 val 和 pow_tensor 进行张量的指数运算，返回结果 actual_pow
    const auto actual_pow = torch::pow(val, pow_tensor);

    // 创建一个与 actual_pow 具有相同形状的空张量 actual_pow_out1
    auto actual_pow_out1 = torch::empty_like(actual_pow);
    // 使用 torch::pow_out 函数将 val 和 pow_tensor 的指数运算结果赋值给 actual_pow_out1
    const auto actual_pow_out2 =
      torch::pow_out(actual_pow_out1, val, pow_tensor);

    // 初始化索引变量 i
    int i = 0;
    // 遍历指数向量 pows 中的每个指数值 pow
    for (const auto pow : pows) {
      // 计算期望的指数运算结果 exp
      const auto exp = typed_pow(static_cast<T>(val), T(pow));

      // 获取 actual_pow 的第 i 个元素，转换为类型 T，与 exp 进行比较断言
      const auto act_pow = actual_pow[i].template item<T>();
      assert_eq<T>(val, act_pow, exp);

      // 获取 actual_pow_out1 的第 i 个元素，转换为类型 T，与 exp 进行比较断言
      const auto act_pow_out1 = actual_pow_out1[i].template item<T>();
      assert_eq<T>(val, act_pow_out1, exp);

      // 获取 actual_pow_out2 的第 i 个元素，转换为类型 T，与 exp 进行比较断言
      const auto act_pow_out2 = actual_pow_out2[i].template item<T>();
      assert_eq<T>(val, act_pow_out2, exp);

      // 更新索引 i
      i++;
    }
  }
}
void tensor_pow_tensor(const Vals vals, c10::ScalarType vals_dtype, Pows pows, c10::ScalarType pows_dtype) {
  // 使用 Vals 的 value_type 定义类型 T
  using T = typename Vals::value_type;

  // 设置输出浮点数精度为双精度最大精度
  typedef std::numeric_limits< double > dbl;
  std::cout.precision(dbl::max_digits10);

  // 将 vals 转换为 Torch 张量，使用指定的数据类型 vals_dtype
  const auto vals_tensor = torch::tensor(vals, vals_dtype);

  // 遍历 pows 的大小
  for ([[maybe_unused]] const auto shirt : c10::irange(pows.size())) {
    // 将 pows 转换为 Torch 张量，使用指定的数据类型 pows_dtype
    const auto pows_tensor = torch::tensor(pows, pows_dtype);

    // 计算 vals_tensor 的 pows_tensor 次幂，存储在 actual_pow 中
    const auto actual_pow = vals_tensor.pow(pows_tensor);

    // 克隆 vals_tensor 并就地执行 pows_tensor 次幂运算，结果存储在 actual_pow_ 中
    auto actual_pow_ = vals_tensor.clone();
    actual_pow_.pow_(pows_tensor);

    // 创建与 vals_tensor 相同形状的空张量 actual_pow_out，并将 vals_tensor 的 pows_tensor 次幂存储在其中
    auto actual_pow_out = torch::empty_like(vals_tensor);
    torch::pow_out(actual_pow_out, vals_tensor, pows_tensor);

    // 使用 torch::pow 函数计算 vals_tensor 的 pows_tensor 次幂，存储在 actual_torch_pow 中
    auto actual_torch_pow = torch::pow(vals_tensor, pows_tensor);

    // 初始化循环计数器 i
    int i = 0;
    // 遍历 vals 中的每个值
    for (const auto val : vals) {
      // 获取当前值的幂次方
      const auto pow = pows[i];
      // 使用 typed_pow 函数计算 val 的 pow 次幂，存储在 exp 中

      // 获取 actual_pow 中第 i 个元素的值，转换为类型 T，与 exp 断言相等性
      const auto act_pow = actual_pow[i].template item<T>();
      assert_eq(val, act_pow, exp);

      // 获取 actual_pow_ 中第 i 个元素的值，转换为类型 T，与 exp 断言相等性
      const auto act_pow_ = actual_pow_[i].template item<T>();
      assert_eq(val, act_pow_, exp);

      // 获取 actual_pow_out 中第 i 个元素的值，转换为类型 T，与 exp 断言相等性
      const auto act_pow_out = actual_pow_out[i].template item<T>();
      assert_eq(val, act_pow_out, exp);

      // 获取 actual_torch_pow 中第 i 个元素的值，转换为类型 T，与 exp 断言相等性
      const auto act_torch_pow = actual_torch_pow[i].template item<T>();
      assert_eq(val, act_torch_pow, exp);

      // 更新循环计数器 i
      i++;
    }

    // 将 pows 的元素旋转，以便下一次迭代使用不同的幂次
    std::rotate(pows.begin(), pows.begin() + 1, pows.end());
  }
}

template<typename T>
void test_pow_one(const std::vector<T> vals) {
  // 遍历 vals 中的每个值
  for (const auto val : vals) {
    // 断言 native::powi(val, T(1)) 的结果等于 val
    ASSERT_EQ(native::powi(val, T(1)), val);
  }
}

template<typename T>
void test_squared(const std::vector<T> vals) {
  // 遍历 vals 中的每个值
  for (const auto val : vals) {
    // 断言 native::powi(val, T(2)) 的结果等于 val 的平方
    ASSERT_EQ(native::powi(val, T(2)), val * val);
  }
}

template<typename T>
void test_cubed(const std::vector<T> vals) {
  // 遍历 vals 中的每个值
  for (const auto val : vals) {
    // 断言 native::powi(val, T(3)) 的结果等于 val 的立方
    ASSERT_EQ(native::powi(val, T(3)), val * val * val);
  }
}

template<typename T>
void test_inverse(const std::vector<T> vals) {
  // 遍历 vals 中的每个值
  for (const auto val : vals) {
    // 如果 val 不等于 1 且不等于 -1，则执行以下断言
    if ( val != 1 && val != -1) {
      // 断言 native::powi(val, T(-4)) 的结果等于 0
      ASSERT_EQ(native::powi(val, T(-4)), 0);
      // 断言 native::powi(val, T(-1)) 的结果等于 (val == 1)
      ASSERT_EQ(native::powi(val, T(-1)), val==1);
    }
  }
  // 初始化变量 neg1 为 -1
  T neg1 = -1;
  // 一系列断言，验证负一的不同次幂的结果
  ASSERT_EQ(native::powi(neg1, T(0)), 1);
  ASSERT_EQ(native::powi(neg1, T(-1)), -1);
  ASSERT_EQ(native::powi(neg1, T(-2)), 1);
  ASSERT_EQ(native::powi(neg1, T(-3)), -1);
  ASSERT_EQ(native::powi(neg1, T(-4)), 1);

  // 初始化变量 one 为 1
  T one = 1;
  // 一系列断言，验证一的不同次幂的结果
  ASSERT_EQ(native::powi(one, T(0)), 1);
  ASSERT_EQ(native::powi(one, T(-1)), 1);
  ASSERT_EQ(native::powi(one, T(-2)), 1);
  ASSERT_EQ(native::powi(one, T(-3)), 1);
  ASSERT_EQ(native::powi(one, T(-4)), 1);

}

}

// 测试例子，测试整数张量的不同幂次计算
TEST(PowTest, IntTensorPowAllScalars) {
  tensor_pow_scalar(ints, non_neg_ints, kInt, kInt);
  tensor_pow_scalar(ints, non_neg_longs, kInt, kInt);
  tensor_pow_scalar(ints, floats, kInt, kFloat);
  tensor_pow_scalar(ints, doubles, kInt, kDouble);
}
// 在单元测试中，测试长整型张量和标量的幂运算
TEST(PowTest, LongTensorPowAllScalars) {
  // 对长整型张量和非负整数进行幂运算，输出为长整型
  tensor_pow_scalar(longs, non_neg_ints, kLong, kLong);
  // 对长整型张量和非负长整数进行幂运算，输出为长整型
  tensor_pow_scalar(longs, non_neg_longs, kLong, kLong);
  // 对长整型张量和浮点数进行幂运算，输出为浮点数
  tensor_pow_scalar(longs, floats, kLong, kFloat);
  // 对长整型张量和双精度浮点数进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(longs, doubles, kLong, kDouble);
}

// 在单元测试中，测试浮点型张量和标量的幂运算
TEST(PowTest, FloatTensorPowAllScalars) {
  // 对浮点型张量和整数进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(floats, ints, kFloat, kDouble);
  // 对浮点型张量和长整型进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(floats, longs, kFloat, kDouble);
  // 对浮点型张量和浮点数进行幂运算，输出为浮点数
  tensor_pow_scalar(floats, floats, kFloat, kFloat);
  // 对浮点型张量和双精度浮点数进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(floats, doubles, kFloat, kDouble);
}

// 在单元测试中，测试双精度浮点型张量和标量的幂运算
TEST(PowTest, DoubleTensorPowAllScalars) {
  // 对双精度浮点型张量和整数进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(doubles, ints, kDouble, kDouble);
  // 对双精度浮点型张量和长整型进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(doubles, longs, kDouble, kDouble);
  // 对双精度浮点型张量和浮点数进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(doubles, floats, kDouble, kDouble);
  // 对双精度浮点型张量和双精度浮点数进行幂运算，输出为双精度浮点数
  tensor_pow_scalar(doubles, doubles, kDouble, kDouble);
}

// 在单元测试中，测试整型标量和各类型张量的幂运算
TEST(PowTest, IntScalarPowAllTensors) {
  // 对整型标量和整型张量进行幂运算，输出为整数
  scalar_pow_tensor(ints, c10::kInt, ints, c10::kInt);
  // 对整型标量和长整型张量进行幂运算，输出为长整型
  scalar_pow_tensor(ints, c10::kInt, longs, c10::kLong);
  // 对整型标量和浮点型张量进行幂运算，输出为浮点数
  scalar_pow_tensor(ints, c10::kInt, floats, c10::kFloat);
  // 对整型标量和双精度浮点型张量进行幂运算，输出为双精度浮点数
  scalar_pow_tensor(ints, c10::kInt, doubles, c10::kDouble);
}

// 在单元测试中，测试长整型标量和各类型张量的幂运算
TEST(PowTest, LongScalarPowAllTensors) {
  // 对长整型标量和长整型张量进行幂运算，输出为长整型
  scalar_pow_tensor(longs, c10::kLong, longs, c10::kLong);
  // 对长整型标量和浮点型张量进行幂运算，输出为浮点数
  scalar_pow_tensor(longs, c10::kLong, floats, c10::kFloat);
  // 对长整型标量和双精度浮点型张量进行幂运算，输出为双精度浮点数
  scalar_pow_tensor(longs, c10::kLong, doubles, c10::kDouble);
}

// 在单元测试中，测试浮点型标量和浮点型张量的幂运算
TEST(PowTest, FloatScalarPowAllTensors) {
  // 对浮点型标量和浮点型张量进行幂运算，输出为浮点数
  scalar_pow_tensor(floats, c10::kFloat, floats, c10::kFloat);
  // 对浮点型标量和双精度浮点型张量进行幂运算，输出为双精度浮点数
  scalar_pow_tensor(floats, c10::kFloat, doubles, c10::kDouble);
}

// 在单元测试中，测试双精度浮点型标量和双精度浮点型张量的幂运算
TEST(PowTest, DoubleScalarPowAllTensors) {
  // 对双精度浮点型标量和双精度浮点型张量进行幂运算，输出为双精度浮点数
  scalar_pow_tensor(doubles, c10::kDouble, doubles, c10::kDouble);
}

// 在单元测试中，测试整型张量和整型张量的幂运算
TEST(PowTest, IntTensorPowIntTensor) {
  // 对整型张量和整型张量进行幂运算，输出为整数
  tensor_pow_tensor(ints, c10::kInt, ints, c10::kInt);
}

// 在单元测试中，测试长整型张量和长整型张量的幂运算
TEST(PowTest, LongTensorPowLongTensor) {
  // 对长整型张量和长整型张量进行幂运算，输出为长整型
  tensor_pow_tensor(longs, c10::kLong, longs, c10::kLong);
}

// 在单元测试中，测试浮点型张量和浮点型张量的幂运算
TEST(PowTest, FloatTensorPowFloatTensor) {
  // 对浮点型张量和浮点型张量进行幂运算，输出为浮点数
  tensor_pow_tensor(floats, c10::kFloat, floats, c10::kFloat);
}

// 在单元测试中，测试双精度浮点型张量和双精度浮点型张量的幂运算
TEST(PowTest, DoubleTensorPowDoubleTensor) {
  // 对双精度浮点型张量和双精度浮点型张量进行幂运算，输出为双精度浮点数
  tensor_pow_tensor(doubles, c10::kDouble, doubles, c10::kDouble);
}

// 在单元测试中，测试长整型和整型张量的幂运算
TEST(PowTest, TestIntegralPow) {
  // 测试长整型张量的幂运算结果
  test_pow_one(longs);
  // 测试整型张量的幂运算结果
  test_pow_one(ints);

  // 测试长整型张量的平方运算结果
  test_squared(longs);
  // 测试整型张量的平方运算结果
  test_squared(ints);

  // 测试长整型张量的立方运算结果
  test_cubed(longs);
  // 测试整型张量的立方运算结果
  test_cubed(ints);

  // 测试长整型张量的倒数运算结果
  test_inverse(longs);
  // 测试整型张量的倒数运算结果
  test_inverse(ints);
}
```