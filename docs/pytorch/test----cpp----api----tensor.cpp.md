# `.\pytorch\test\cpp\api\tensor.cpp`

```
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <test/cpp/common/support.h>

using namespace torch::test;

// 模板函数，用于检查两个值是否精确相等
template <typename T>
bool exactly_equal(at::Tensor left, T right) {
  return left.item<T>() == right;
}

// 模板函数，用于检查两个值是否在给定的容差范围内几乎相等
template <typename T>
bool almost_equal(at::Tensor left, T right, double tolerance = 1e-4) {
  return std::abs(left.item<T>() - right) < tolerance;
}

// 宏定义，用于验证张量的设备、索引、数据类型和布局是否满足指定条件
#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_TRUE(                                                             \
      tensor.device().type() == at::Device((device_), (index_)).type());   \
  ASSERT_TRUE(                                                             \
      tensor.device().index() == at::Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.dtype(), (type_));                                      \
  ASSERT_TRUE(tensor.layout() == (layout_))

// 测试用例，验证张量的数据类型转换功能
TEST(TensorTest, ToDtype) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::kChar);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::kDouble);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kInt));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kChar));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kDouble));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
}

// 测试用例，验证张量的数据类型转换和属性设置功能
TEST(TensorTest, ToTensorAndTensorAttributes) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  auto other = at::empty({3, 4}, at::kInt);
  tensor = tensor.to(other);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  other = at::empty({3, 4}, at::kDouble);
  tensor = tensor.to(other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
  tensor = tensor.to(other.device());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);

  other = at::empty({3, 4}, at::kLong);
  tensor = tensor.to(other.device(), other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kLong, at::kStrided);

  other = at::empty({3, 4}, at::kInt);
  tensor = tensor.to(other.options());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);
}

// 以下测试用例因为不受支持而被注释掉
// TEST(TensorTest, ToLayout) {
//   auto tensor = at::empty({3, 4});
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
//
//   tensor = tensor.to(at::kSparse);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kSparse);
//
//   tensor = tensor.to(at::kStrided);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
// }
TEST(TensorTest, ToOptionsWithRequiresGrad) {
  {
    // 创建一个空的张量，形状为{3, 4}，并设置 requires_grad 为 true
    auto tensor = torch::empty({3, 4}, at::requires_grad());
    ASSERT_TRUE(tensor.requires_grad());

    // 将张量转换为 double 类型，保持 requires_grad 的设置不变
    tensor = tensor.to(at::kDouble);
    ASSERT_TRUE(tensor.requires_grad());

    // 如果在 TensorOptions 中设置了 requires_grad 为 true，则抛出异常
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(true)), c10::Error);

    // 如果在 TensorOptions 中未设置 requires_grad，则不抛出异常
    tensor.to(at::TensorOptions());
    tensor.to(at::TensorOptions().requires_grad(false));
  }
  {
    // 创建一个空的张量，形状为{3, 4}，默认 requires_grad 为 false
    auto tensor = torch::empty({3, 4});
    ASSERT_FALSE(tensor.requires_grad());

    // 将张量转换为 double 类型，保持 requires_grad 的设置不变
    tensor = tensor.to(at::kDouble);
    ASSERT_FALSE(tensor.requires_grad());

    // 如果在 TensorOptions 中设置了 requires_grad 为 true，则抛出异常
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(true)), c10::Error);

    // 如果在 TensorOptions 中未设置 requires_grad，则不抛出异常
    tensor.to(at::TensorOptions());
    tensor.to(at::TensorOptions().requires_grad(false));
  }
}

TEST(TensorTest, ToDoesNotCopyWhenOptionsAreAllTheSame) {
  {
    // 创建一个形状为{3, 4}、数据类型为 float 的空张量
    auto tensor = at::empty({3, 4}, at::kFloat);
    // 将张量转换为相同数据类型的张量，希望不会复制数据
    auto hopefully_not_copy = tensor.to(at::kFloat);
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    // 创建一个形状为{3, 4}、数据类型为 float 的空张量
    auto tensor = at::empty({3, 4}, at::kFloat);
    // 将张量转换为与自身选项相同的张量，希望不会复制数据
    auto hopefully_not_copy = tensor.to(tensor.options());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    // 创建一个形状为{3, 4}、数据类型为 float 的空张量
    auto tensor = at::empty({3, 4}, at::kFloat);
    // 将张量转换为与自身数据类型相同的张量，希望不会复制数据
    auto hopefully_not_copy = tensor.to(tensor.dtype());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    // 创建一个形状为{3, 4}、数据类型为 float 的空张量
    auto tensor = at::empty({3, 4}, at::kFloat);
    // 将张量转换为与自身设备相同的张量，希望不会复制数据
    auto hopefully_not_copy = tensor.to(tensor.device());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    // 创建一个形状为{3, 4}、数据类型为 float 的空张量
    auto tensor = at::empty({3, 4}, at::kFloat);
    // 将张量转换为自身，希望不会复制数据
    auto hopefully_not_copy = tensor.to(tensor);
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
}
# 定义测试用例函数，测试张量的构造函数，针对不同类型的输入进行测试
TEST(TensorTest, AtTensorCtorScalar) {
  # 创建一个整数张量，值为123
  auto tensor = at::tensor(123);
  # 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  # 断言张量数据类型为整型
  ASSERT_EQ(tensor.dtype(), at::kInt);
  # 断言张量第一个元素的值为123
  ASSERT_EQ(tensor[0].item<int32_t>(), 123);

  # 创建一个单精度浮点数张量，值为123.456
  tensor = at::tensor(123.456f);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  # 断言张量第一个元素值接近123.456
  ASSERT_TRUE(almost_equal(tensor[0], 123.456f));

  # 创建一个双精度浮点数张量，值为123.456
  tensor = at::tensor(123.456);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  # 断言张量第一个元素值接近123.456
  ASSERT_TRUE(almost_equal(tensor[0], 123.456));

  # 创建一个浮点数张量，值为123，指定数据类型为单精度浮点数，并加上0.5
  tensor = at::tensor(123, at::dtype(at::kFloat)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  # 断言张量第一个元素值接近123.5
  ASSERT_TRUE(almost_equal(tensor[0], 123.5));

  # 创建一个复数张量，值为复数(1.0, 2.0)，并加上0.5
  tensor = at::tensor(c10::complex<float>(1.0, 2.0)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexFloat);
  # 断言张量第一个元素值接近复数(1.5, 2.0)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<float>(1.5, 2.0)));

  # 创建一个复数张量，值为复数(1.0, 2.0)，指定数据类型为复数单精度浮点数，并加上0.5
  tensor =
      at::tensor(c10::complex<float>(1.0, 2.0), at::dtype(at::kComplexFloat)) +
      0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexFloat);
  # 断言张量第一个元素值接近复数(1.5, 2.0)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<float>(1.5, 2.0)));

  # 创建一个复数张量，值为复数(1.0, 2.0)，并加上0.5
  tensor = at::tensor(c10::complex<double>(1.0, 2.0)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  # 断言张量第一个元素值接近复数(1.5, 2.0)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5, 2.0)));

  # 创建一个复数张量，值为复数(1.0, 2.0)，指定数据类型为复数双精度浮点数，并加上0.5
  tensor =
      at::tensor(c10::complex<float>(1.0, 2.0), at::dtype(at::kComplexDouble)) +
      0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  # 断言张量第一个元素值接近复数(1.5, 2.0)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5, 2.0)));

  # 使用给定的向量w创建张量
  std::vector<double> w = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
  tensor = at::tensor(w);
  ASSERT_EQ(tensor.numel(), w.size());
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  # 遍历张量的每个元素，断言它们与向量w中对应元素接近
  for (const auto i : c10::irange(w.size())) {
    ASSERT_TRUE(almost_equal(tensor[i], w.at(i)));
  }

  # 使用给定的复数向量x创建张量
  std::vector<c10::complex<double>> x = {
      {1.1, -1.1},
      {2.2, -2.2},
      {3.3, -3.3},
      {4.4, -4.4},
      {5.5, -5.5},
      {6.6, -6.6},
      {7.7, -7.7},
      {8.8, -8.8},
      {9.9, -9.9},
      {10.0, -10.0}};
  tensor = at::tensor(x);
  ASSERT_EQ(tensor.numel(), x.size());
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  # 遍历张量的每个元素，断言它们与复数向量x中对应元素接近
  for (const auto i : c10::irange(x.size())) {
    ASSERT_TRUE(almost_equal(tensor[i], x.at(i)));
  }
}
// 测试用例：AtTensorCastRealToComplex
TEST(TensorTest, AtTensorCastRealToComplex) {
  // 创建复数张量，包含双精度浮点数向量，数据类型为复数双精度浮点数
  auto tensor =
      at::tensor(std::vector<double>({1.5, 2.5, 3.5}), at::kComplexDouble);
  // 断言张量元素数量为3
  ASSERT_EQ(tensor.numel(), 3);
  // 断言张量数据类型为复数双精度浮点数
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  // 断言第一个张量元素几乎等于复数<double>(1.5)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  // 断言第二个张量元素几乎等于复数<double>(2.5)
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  // 断言第三个张量元素几乎等于复数<double>(3.5)
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  // 使用初始化列表创建复数张量，数据类型为复数双精度浮点数
  tensor = at::tensor({1.5, 2.5, 3.5}, at::kComplexDouble);
  // 断言张量元素数量为3
  ASSERT_EQ(tensor.numel(), 3);
  // 断言张量数据类型为复数双精度浮点数
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  // 断言第一个张量元素几乎等于复数<double>(1.5)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  // 断言第二个张量元素几乎等于复数<double>(2.5)
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  // 断言第三个张量元素几乎等于复数<double>(3.5)
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  // 使用标量创建复数张量，数据类型为复数双精度浮点数
  tensor = at::tensor(1.5, at::kComplexDouble);
  // 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  // 断言张量数据类型为复数双精度浮点数
  ASSERT_EQ(tensor.dtype(), at::kComplexDouble);
  // 断言张量唯一元素几乎等于复数<double>(1.5)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
}

// 测试用例：AtTensorCastComplexToRealErrorChecks
TEST(TensorTest, AtTensorCastComplexToRealErrorChecks) {
  {
    // 断言抛出异常，因为在'Float'上未实现 'tensor_cpu'
    ASSERT_THROWS_WITH(
        at::tensor(c10::complex<float>(0.1, 0.2), at::kFloat),
        "\"tensor_cpu\" not implemented for 'Float'");
  }
  {
    // 断言抛出异常，因为在'Float'上未实现 'tensor_cpu'
    ASSERT_THROWS_WITH(
        at::tensor({c10::complex<float>(0.1, 0.2)}, at::kFloat),
        "\"tensor_cpu\" not implemented for 'Float'");
  }
  {
    // 断言抛出异常，因为在'Float'上未实现 'tensor_cpu'
    ASSERT_THROWS_WITH(
        at::tensor(
            std::vector<c10::complex<float>>{c10::complex<float>(0.1, 0.2)},
            at::kFloat),
        "\"tensor_cpu\" not implemented for 'Float'");
  }
}

// 测试用例：TorchTensorCtorScalarIntegralType
TEST(TensorTest, TorchTensorCtorScalarIntegralType) {
  // 创建整数张量，数据为123
  auto tensor = torch::tensor(123);
  // 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  // 断言张量形状为空向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  // 断言张量数据类型为长整型
  ASSERT_EQ(tensor.dtype(), at::kLong);
  // 断言张量元素值为123
  ASSERT_EQ(tensor.item<int64_t>(), 123);
}

// 函数：test_TorchTensorCtorScalarFloatingType_expected_dtype
void test_TorchTensorCtorScalarFloatingType_expected_dtype(
    c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  // 创建浮点数张量，数据为123.456f，数据类型由参数指定
  auto tensor = torch::tensor(123.456f);
  // 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  // 断言张量形状为空向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  // 断言张量数据类型与参数指定的默认数据类型相同
  ASSERT_EQ(tensor.dtype(), default_dtype);
  // 断言张量元素几乎等于123.456f
  ASSERT_TRUE(almost_equal(tensor, 123.456f));

  // 创建浮点数张量，数据为123.456，数据类型由参数指定
  tensor = torch::tensor(123.456);
  // 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  // 断言张量形状为空向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  // 断言张量数据类型与参数指定的默认数据类型相同
  ASSERT_EQ(tensor.dtype(), default_dtype);
  // 断言张量元素几乎等于123.456
  ASSERT_TRUE(almost_equal(tensor, 123.456));

  // 创建浮点数张量，数据为{123.456}，数据类型由参数指定
  tensor = torch::tensor({123.456});
  // 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  // 断言张量形状为包含一个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1}));
  // 断言张量数据类型与参数指定的默认数据类型相同
  ASSERT_EQ(tensor.dtype(), default_dtype);
  // 断言张量第一个元素几乎等于123.456
  ASSERT_TRUE(almost_equal(tensor[0], 123.456));
}

// 测试用例：TorchTensorCtorScalarFloatingType
TEST(TensorTest, TorchTensorCtorScalarFloatingType) {
  // 测试浮点数张量构造函数，验证不同默认数据类型下的行为
  test_TorchTensorCtorScalarFloatingType_expected_dtype(
      /*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorScalarFloatingType_expected_dtype(
      /*default_dtype=*/torch::kDouble);
}
# 测试用例：TensorTest，测试 TorchTensorCtorScalarBoolType 函数
TEST(TensorTest, TorchTensorCtorScalarBoolType) {
  # 创建一个标量布尔类型的张量
  auto tensor = torch::tensor(true);
  # 断言张量的元素数为 1
  ASSERT_EQ(tensor.numel(), 1);
  # 断言张量的大小为空向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  # 断言张量的数据类型为布尔类型
  ASSERT_EQ(tensor.dtype(), at::kBool);
  # 断言张量的值确实等于 true
  ASSERT_TRUE(exactly_equal(tensor, true));

  # 重新赋值：创建一个包含单个布尔值的张量
  tensor = torch::tensor({true});
  # 断言张量的元素数为 1
  ASSERT_EQ(tensor.numel(), 1);
  # 断言张量的大小为包含一个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1}));
  # 断言张量的数据类型为布尔类型
  ASSERT_EQ(tensor.dtype(), at::kBool);
  # 断言张量的第一个元素确实等于 true
  ASSERT_TRUE(exactly_equal(tensor[0], true));
}

# 测试用例：TensorTest，测试 TorchTensorCtorSingleDimIntegralType 函数
TEST(TensorTest, TorchTensorCtorSingleDimIntegralType) {
  # 创建一个包含整数的一维张量
  auto tensor = torch::tensor({1, 2, 3});
  # 断言张量的元素数为 3
  ASSERT_EQ(tensor.numel(), 3);
  # 断言张量的大小为包含三个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  # 断言张量的数据类型为长整型
  ASSERT_EQ(tensor.dtype(), at::kLong);
  # 断言张量的各个元素分别等于 1, 2, 3
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  # 重新赋值：使用 ArrayRef<int> 创建张量
  tensor = torch::tensor(at::ArrayRef<int>({1, 2, 3}));
  # 断言张量的元素数为 3
  ASSERT_EQ(tensor.numel(), 3);
  # 断言张量的大小为包含三个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  # 断言张量的数据类型为长整型
  ASSERT_EQ(tensor.dtype(), at::kLong);
  # 断言张量的各个元素分别等于 1, 2, 3
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  # 重新赋值：使用 std::vector<int> 创建张量
  tensor = torch::tensor(std::vector<int>({1, 2, 3}));
  # 断言张量的元素数为 3
  ASSERT_EQ(tensor.numel(), 3);
  # 断言张量的大小为包含三个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  # 断言张量的数据类型为长整型
  ASSERT_EQ(tensor.dtype(), at::kLong);
  # 断言张量的各个元素分别等于 1, 2, 3
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  # 重新赋值：使用 ArrayRef<int64_t> 创建张量
  tensor = torch::tensor(at::ArrayRef<int64_t>({1, 2, 3}));
  # 断言张量的元素数为 3
  ASSERT_EQ(tensor.numel(), 3);
  # 断言张量的大小为包含三个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  # 断言张量的数据类型为长整型
  ASSERT_EQ(tensor.dtype(), at::kLong);
  # 断言张量的各个元素分别等于 1, 2, 3
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  # 重新赋值：使用 std::vector<int64_t> 创建张量
  tensor = torch::tensor(std::vector<int64_t>({1, 2, 3}));
  # 断言张量的元素数为 3
  ASSERT_EQ(tensor.numel(), 3);
  # 断言张量的大小为包含三个元素的向量
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  # 断言张量的数据类型为长整型
  ASSERT_EQ(tensor.dtype(), at::kLong);
  # 断言张量的各个元素分别等于 1, 2, 3
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));
}
    # 定义一个函数，创建一个新的张量，并设置默认的数据类型
    void create_and_check_tensor(
        c10::ScalarType default_dtype) {
      # 自动管理默认数据类型的模式
      AutoDefaultDtypeMode dtype_mode(default_dtype);
    
      # 创建一个张量，包含指定的浮点数值
      auto tensor = torch::tensor({1.5, 2.25, 3.125});
      # 断言张量中元素的个数为3
      ASSERT_EQ(tensor.numel(), 3);
      # 断言张量的尺寸为[3]
      ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
      # 断言张量的数据类型为默认设置的数据类型
      ASSERT_EQ(tensor.dtype(), default_dtype);
      # 断言张量的第一个元素接近于1.5
      ASSERT_TRUE(almost_equal(tensor[0], 1.5));
      # 断言张量的第二个元素接近于2.25
      ASSERT_TRUE(almost_equal(tensor[1], 2.25));
      # 断言张量的第三个元素接近于3.125
      ASSERT_TRUE(almost_equal(tensor[2], 3.125));
    
      # 创建一个新的张量，包含指定的浮点数值，使用单精度浮点数
      tensor = torch::tensor({1.5f, 2.25f, 3.125f});
      # 断言张量中元素的个数为3
      ASSERT_EQ(tensor.numel(), 3);
      # 断言张量的尺寸为[3]
      ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
      # 断言张量的数据类型为默认设置的数据类型
      ASSERT_EQ(tensor.dtype(), default_dtype);
      # 断言张量的第一个元素接近于1.5
      ASSERT_TRUE(almost_equal(tensor[0], 1.5f));
      # 断言张量的第二个元素接近于2.25
      ASSERT_TRUE(almost_equal(tensor[1], 2.25f));
      # 断言张量的第三个元素接近于3.125
      ASSERT_TRUE(almost_equal(tensor[2], 3.125f));
    
      # 创建一个新的张量，使用给定的单精度浮点数数组
      tensor = torch::tensor(at::ArrayRef<float>({1.5f, 2.25f, 3.125f}));
      # 断言张量中元素的个数为3
      ASSERT_EQ(tensor.numel(), 3);
      # 断言张量的数据类型为默认设置的数据类型
      ASSERT_EQ(tensor.dtype(), default_dtype);
      # 断言张量的第一个元素接近于1.5
      ASSERT_TRUE(almost_equal(tensor[0], 1.5));
      # 断言张量的第二个元素接近于2.25
      ASSERT_TRUE(almost_equal(tensor[1], 2.25));
      # 断言张量的第三个元素接近于3.125
      ASSERT_TRUE(almost_equal(tensor[2], 3.125));
    
      # 创建一个新的张量，使用给定的双精度浮点数向量
      tensor = torch::tensor(std::vector<float>({1.5f, 2.25f, 3.125f}));
      # 断言张量中元素的个数为3
      ASSERT_EQ(tensor.numel(), 3);
      # 断言张量的尺寸为[3]
      ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
      # 断言张量的数据类型为默认设置的数据类型
      ASSERT_EQ(tensor.dtype(), default_dtype);
      # 断言张量的第一个元素接近于1.5
      ASSERT_TRUE(almost_equal(tensor[0], 1.5));
      # 断言张量的第二个元素接近于2.25
      ASSERT_TRUE(almost_equal(tensor[1], 2.25));
      # 断言张量的第三个元素接近于3.125
      ASSERT_TRUE(almost_equal(tensor[2], 3.125));
    
      # 创建一个新的张量，使用给定的双精度浮点数数组
      tensor = torch::tensor(at::ArrayRef<double>({1.5, 2.25, 3.125}));
      # 断言张量中元素的个数为3
      ASSERT_EQ(tensor.numel(), 3);
      # 断言张量的数据类型为默认设置的数据类型
      ASSERT_EQ(tensor.dtype(), default_dtype);
      # 断言张量的第一个元素接近于1.5
      ASSERT_TRUE(almost_equal(tensor[0], 1.5));
      # 断言张量的第二个元素接近于2.25
      ASSERT_TRUE(almost_equal(tensor[1], 2.25));
      # 断言张量的第三个元素接近于3.125
      ASSERT_TRUE(almost_equal(tensor[2], 3.125));
    
      # 创建一个新的张量，使用给定的双精度浮点数向量
      tensor = torch::tensor(std::vector<double>({1.5, 2.25, 3.125}));
      # 断言张量中元素的个数为3
      ASSERT_EQ(tensor.numel(), 3);
      # 断言张量的尺寸为[3]
      ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
      # 断言张量的数据类型为默认设置的数据类型
      ASSERT_EQ(tensor.dtype(), default_dtype);
      # 断言张量的第一个元素接近于1.5
      ASSERT_TRUE(almost_equal(tensor[0], 1.5));
      # 断言张量的第二个元素接近于2.25
      ASSERT_TRUE(almost_equal(tensor[1], 2.25));
      # 断言张量的第三个元素接近于3.125
      ASSERT_TRUE(almost_equal(tensor[2], 3.125));
    }
TEST(TensorTest, TorchTensorCtorMultiDimIntegralType) {
  {
    // 创建一个包含单个元素的长整型张量 [{1, 2}]
    auto tensor = torch::tensor({{1, 2}});
    // 断言张量的数据类型为长整型
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    // 断言张量的尺寸为 {1, 2}
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    // 断言张量与从 1 到 2 的长整型序列全等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    // 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含两个元素的长整型张量 [{1}, {2}]
    auto tensor = torch::tensor({{1}, {2}});
    // 断言张量的数据类型为长整型
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    // 断言张量的尺寸为 {2, 1}
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 1}));
    // 断言张量与从 1 到 2 的长整型序列全等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    // 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含单个元素的长整型张量 [[{1, 2}]]
    auto tensor = torch::tensor({{{1, 2}}});
    // 断言张量的数据类型为长整型
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    // 断言张量的尺寸为 {1, 1, 2}
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 2}));
    // 断言张量与从 1 到 2 的长整型序列全等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    // 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含两个元素的长整型张量 [[[{1}, {2}]]]
    auto tensor = torch::tensor({{{1}, {2}}});
    // 断言张量的数据类型为长整型
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    // 断言张量的尺寸为 {1, 2, 1}
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2, 1}));
    // 断言张量与从 1 到 2 的长整型序列全等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    // 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含四个元素的长整型张量 [{1, 2}, {3, 4}]
    auto tensor = torch::tensor({{1, 2}, {3, 4}});
    // 断言张量的数据类型为长整型
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    // 断言张量的尺寸为 {2, 2}
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 2}));
    // 断言张量与从 1 到 4 的长整型序列全等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 5, torch::kLong).view(tensor.sizes())));
    // 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含十个元素的长整型张量 [[[[[[[[[1]]]]]]]]]
    auto tensor = torch::tensor({{{{{{{{{{1}}}}}}}}}});
    // 断言张量的数据类型为长整型
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    // 断言张量的尺寸为 {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    ASSERT_EQ(
        tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    // 断言张量与全为 1 的长整型张量全等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::full({1}, 1, torch::kLong).view(tensor.sizes())));
    // 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
  }
}
    # 创建一个张量，包含元素值为 1 和 2，张量自动推断类型
    auto tensor = torch::tensor({{{{{{{{{{1, 2}}}}}}}}}});
    
    # 断言张量的数据类型为 long
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    
    # 断言张量的维度大小为 [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    ASSERT_EQ(
        tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 2}));
    
    # 断言张量的值与 torch 库生成的从 1 到 2（包括 1 不包括 3）的序列视图相等
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    
    # 断言张量不需要梯度计算
    ASSERT_FALSE(tensor.requires_grad());
}

void test_TorchTensorCtorMultiDimFloatingType_expected_dtype(
    c10::ScalarType default_dtype) {
  // 进入自动默认数据类型模式，使用给定的默认数据类型
  AutoDefaultDtypeMode dtype_mode(default_dtype);
  {
    // 创建一个包含两个元素的二维张量，数据类型与默认数据类型相同
    auto tensor = torch::tensor({{1.0, 2.0}});
    // 断言张量的数据类型与默认数据类型相同
    ASSERT_EQ(tensor.dtype(), default_dtype);
    // 断言张量的形状为 [1, 2]
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    // 断言张量的值与从1到2（不包括2）的数值相等，视图形状与张量相同
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, default_dtype).view(tensor.sizes())));
    // 断言张量不需要梯度
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含一个元素的八维张量，元素值为1.0到9.0的浮点数
    auto tensor = torch::tensor(
        {{{{{{{{1.0, 2.0, 3.0}}}}},
           {{{{{4.0, 5.0, 6.0}}}}},
           {{{{{7.0, 8.0, 9.0}}}}}}}});
    // 断言张量的数据类型与默认数据类型相同
    ASSERT_EQ(tensor.dtype(), default_dtype);
    // 断言张量的形状为 [1, 1, 3, 1, 1, 1, 1, 3]
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 3, 1, 1, 1, 1, 3}));
    // 断言张量的值与从1到9（不包括9）的数值相等，视图形状与张量相同
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 10, default_dtype).view(tensor.sizes())));
    // 断言张量不需要梯度
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorMultiDimFloatingType) {
  // 测试用例：使用浮点类型作为默认数据类型进行测试
  test_TorchTensorCtorMultiDimFloatingType_expected_dtype(
      /*default_dtype=*/torch::kFloat);
  // 测试用例：使用双精度浮点类型作为默认数据类型进行测试
  test_TorchTensorCtorMultiDimFloatingType_expected_dtype(
      /*default_dtype=*/torch::kDouble);
}

TEST(TensorTest, TorchTensorCtorMultiDimBoolType) {
  {
    // 创建一个包含两个元素的二维张量，数据类型为布尔类型
    auto tensor = torch::tensor({{true, false}});
    // 断言张量的数据类型为布尔类型
    ASSERT_EQ(tensor.dtype(), torch::kBool);
    // 断言张量的形状为 [1, 2]
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    // 创建一个预期的张量，值分别为true和false，形状与张量相同
    auto expected = torch::empty(tensor.sizes(), torch::kBool);
    expected[0][0] = true;
    expected[0][1] = false;
    // 断言张量与预期张量相等
    ASSERT_TRUE(torch::equal(tensor, expected));
    // 断言张量不需要梯度
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含两个元素的二维张量，数据类型为布尔类型
    auto tensor = torch::tensor({{true}, {false}});
    // 断言张量的数据类型为布尔类型
    ASSERT_EQ(tensor.dtype(), torch::kBool);
    // 断言张量的形状为 [2, 1]
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 1}));
    // 创建一个预期的张量，值分别为true和false，形状与张量相同
    auto expected = torch::empty(tensor.sizes(), torch::kBool);
    expected[0][0] = true;
    expected[1][0] = false;
    // 断言张量与预期张量相等
    ASSERT_TRUE(torch::equal(tensor, expected));
    // 断言张量不需要梯度
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorMultiDimWithOptions) {
  {
    // 创建一个包含两个元素的二维张量，数据类型为整数类型
    auto tensor = torch::tensor({{1, 2}}, torch::dtype(torch::kInt));
    // 断言张量的数据类型为整数类型
    ASSERT_EQ(tensor.dtype(), torch::kInt);
    // 断言张量的形状为 [1, 2]
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    // 断言张量的值与从1到2（不包括2）的整数值相等，视图形状与张量相同
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 3, torch::kInt).view(tensor.sizes())));
    // 断言张量不需要梯度
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    // 创建一个包含两个元素的二维张量，数据类型为浮点类型，需要梯度
    auto tensor = torch::tensor(
        {{1, 2}, {3, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
    // 断言张量的数据类型为浮点类型
    ASSERT_EQ(tensor.dtype(), torch::kFloat);
    // 断言张量的形状为 [2, 2]
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 2}));
    // 断言张量的值与从1到4（不包括4）的浮点数值相等，视图形状与张量相同
    ASSERT_TRUE(torch::allclose(
        tensor, torch::arange(1, 5, torch::kFloat).view(tensor.sizes())));
    // 断言张量需要梯度
    ASSERT_TRUE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorMultiDimErrorChecks) {
  {
    // 断言创建张量时出现错误，期望所有子列表具有大小为2，但得到的子列表 {7} 的大小为1
    ASSERT_THROWS_WITH(
        torch::tensor({{{2, 3, 4}, {{5, 6}, {7}}}}),
        "Expected all sub-lists to have sizes: 2 (e.g. {5, 6}), but got sub-list {7} with sizes: 1");
  }
  {
    // 断言抛出异常，当张量中存在不同标量类型的元素时，应抛出异常并指定错误信息
    ASSERT_THROWS_WITH(
        torch::tensor({{{1, 2.0}, {1, 2.0}}}),
        "Expected all elements of the tensor to have the same scalar type: Int, but got element of scalar type: Double");
    }
    
    // 断言抛出异常，当张量中存在不同标量类型的元素时，应抛出异常并指定错误信息
    {
        ASSERT_THROWS_WITH(
            torch::tensor({{{true, 2.0, 3}, {true, 2.0, 3}}}),
            "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Double");
    }
    
    // 断言抛出异常，当张量中存在不同标量类型的元素时，应抛出异常并指定错误信息
    {
        ASSERT_THROWS_WITH(
            torch::tensor({{{true}, {2}}}),
            "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Int");
    }
    
    // 断言抛出异常，当张量中存在不同标量类型的元素时，应抛出异常并指定错误信息
    {
        ASSERT_THROWS_WITH(
            torch::tensor({{{true, 2}}}),
            "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Int");
    }
}

TEST(TensorTest, TorchTensorCastRealToComplex) {
  // 创建一个包含实数数据的复数张量
  auto tensor = torch::tensor(
      std::vector<double>({1.5, 2.5, 3.5}), torch::kComplexDouble);
  // 断言张量元素数量为3
  ASSERT_EQ(tensor.numel(), 3);
  // 断言张量数据类型为复数双精度
  ASSERT_EQ(tensor.dtype(), torch::kComplexDouble);
  // 断言张量第一个元素几乎等于复数值(1.5)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  // 断言张量第二个元素几乎等于复数值(2.5)
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  // 断言张量第三个元素几乎等于复数值(3.5)
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  // 使用初始化列表创建复数张量
  tensor = torch::tensor({1.5, 2.5, 3.5}, torch::kComplexDouble);
  // 断言张量元素数量为3
  ASSERT_EQ(tensor.numel(), 3);
  // 断言张量数据类型为复数双精度
  ASSERT_EQ(tensor.dtype(), torch::kComplexDouble);
  // 断言张量第一个元素几乎等于复数值(1.5)
  ASSERT_TRUE(almost_equal(tensor[0], c10::complex<double>(1.5)));
  // 断言张量第二个元素几乎等于复数值(2.5)
  ASSERT_TRUE(almost_equal(tensor[1], c10::complex<double>(2.5)));
  // 断言张量第三个元素几乎等于复数值(3.5)
  ASSERT_TRUE(almost_equal(tensor[2], c10::complex<double>(3.5)));

  // 使用单个实数创建复数张量
  tensor = torch::tensor(1.5, torch::kComplexDouble);
  // 断言张量元素数量为1
  ASSERT_EQ(tensor.numel(), 1);
  // 断言张量数据类型为复数双精度
  ASSERT_EQ(tensor.dtype(), torch::kComplexDouble);
  // 断言张量元素几乎等于复数值(1.5)
  ASSERT_TRUE(almost_equal(tensor, c10::complex<double>(1.5)));
}

TEST(TensorTest, TorchTensorCastComplexToRealErrorChecks) {
  {
    // 断言创建复数张量时出现的异常消息
    ASSERT_THROWS_WITH(
        torch::tensor(c10::complex<float>(0.1, 0.2), torch::kFloat),
        "value cannot be converted to type float without overflow");
  }
  {
    // 断言创建复数张量列表时出现的异常消息
    ASSERT_THROWS_WITH(
        torch::tensor(
            {c10::complex<float>(0.1, 0.2), c10::complex<float>(0.3, 0.4)},
            torch::kFloat),
        "value cannot be converted to type float without overflow");
  }
  {
    // 断言创建复数张量向量时出现的异常消息
    ASSERT_THROWS_WITH(
        torch::tensor(
            std::vector<c10::complex<float>>{
                c10::complex<float>(0.1, 0.2), c10::complex<float>(0.3, 0.4)},
            torch::kFloat),
        "can not do torch::tensor(complex, dtype=non-complex) because complex can not be casted to real number without loss of information");
  }
}

void test_TorchTensorCtorMultiDim_CUDA_expected_dtype(
    c10::ScalarType default_dtype) {
  // 设置默认数据类型
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  // 创建多维张量并指定CUDA设备
  auto tensor = torch::tensor(
      {{{{{{{{1.0, 2.0, 3.0}}}}},
         {{{{{4.0, 5.0, 6.0}}}}},
         {{{{{7.0, 8.0, 9.0}}}}}}}},
      torch::dtype(default_dtype).device(torch::kCUDA));
  // 断言张量在CUDA设备上
  ASSERT_TRUE(tensor.device().is_cuda());
  // 断言张量数据类型与预期一致
  ASSERT_EQ(tensor.dtype(), default_dtype);
  // 断言张量维度大小符合预期
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 3, 1, 1, 1, 1, 3}));
  // 断言张量数据几乎等于指定范围的数值张量，并且在CUDA设备上
  ASSERT_TRUE(torch::allclose(
      tensor,
      torch::arange(1, 10, default_dtype)
          .view(tensor.sizes())
          .to(torch::kCUDA)));
  // 断言张量不需要梯度计算
  ASSERT_FALSE(tensor.requires_grad());
}

TEST(TensorTest, TorchTensorCtorMultiDim_CUDA) {
  // 测试指定默认数据类型的多维张量在CUDA上的行为
  test_TorchTensorCtorMultiDim_CUDA_expected_dtype(
      /*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorMultiDim_CUDA_expected_dtype(
      /*default_dtype=*/torch::kDouble);
}

void test_TorchTensorCtorZeroSizedDim_expected_dtype(
    c10::ScalarType default_dtype) {
  // 设置默认数据类型
  AutoDefaultDtypeMode dtype_mode(default_dtype);
  {
    // 创建空张量，并断言元素数量为0
    auto tensor = torch::tensor({});
    ASSERT_EQ(tensor.numel(), 0);
    {
        // 创建一个空张量，并验证其大小是否为 {0}
        ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({0}));
        // 验证张量的数据类型是否为默认数据类型
        ASSERT_EQ(tensor.dtype(), default_dtype);
        // 验证张量是否不需要梯度计算
        ASSERT_FALSE(tensor.requires_grad());
    }
    {
        // 创建一个包含两个空张量的张量，并验证其元素个数是否为 0
        auto tensor = torch::tensor({{}, {}});
        ASSERT_EQ(tensor.numel(), 0);
        // 验证张量的大小是否为 {2, 0}
        ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 0}));
        // 验证张量的数据类型是否为默认数据类型
        ASSERT_EQ(tensor.dtype(), default_dtype);
        // 验证张量是否不需要梯度计算
        ASSERT_FALSE(tensor.requires_grad());
    }
    {
        // 创建一个包含一个空张量的张量，并验证其元素个数是否为 0
        auto tensor = torch::tensor({{{}}});
        ASSERT_EQ(tensor.numel(), 0);
        // 验证张量的大小是否为 {1, 1, 0}
        ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 0}));
        // 验证张量的数据类型是否为默认数据类型
        ASSERT_EQ(tensor.dtype(), default_dtype);
        // 验证张量是否不需要梯度计算
        ASSERT_FALSE(tensor.requires_grad());
    }
    {
        // 创建一个包含多层嵌套的空张量的张量，并验证其元素个数是否为 0
        auto tensor = torch::tensor({{{{{{{{}}}}}}}});
        ASSERT_EQ(tensor.numel(), 0);
        // 验证张量的大小是否为 {1, 1, 1, 1, 1, 1, 1, 0}
        ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 0}));
        // 验证张量的数据类型是否为默认数据类型
        ASSERT_EQ(tensor.dtype(), default_dtype);
        // 验证张量是否不需要梯度计算
        ASSERT_FALSE(tensor.requires_grad());
    }
    {
        // 创建一个更深层次的包含多层嵌套的空张量的张量，并验证其元素个数是否为 0
        auto tensor = torch::tensor({{{{{{{{}}}}, {{{{}}}}}}}});
        ASSERT_EQ(tensor.numel(), 0);
        // 验证张量的大小是否为 {1, 1, 1, 2, 1, 1, 1, 0}
        ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 2, 1, 1, 1, 0}));
        // 验证张量的数据类型是否为默认数据类型
        ASSERT_EQ(tensor.dtype(), default_dtype);
        // 验证张量是否不需要梯度计算
        ASSERT_FALSE(tensor.requires_grad());
    }
    {
        // 创建一个极其深层次的包含多层嵌套的空张量的张量，并验证其元素个数是否为 0
        auto tensor = torch::tensor({{{{{{{{{{}}}}}}}}}});
        ASSERT_EQ(tensor.numel(), 0);
        // 验证张量的大小是否为 {1, 1, 1, 1, 1, 1, 1, 1, 1, 0}
        ASSERT_EQ(
            tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 0}));
        // 验证张量的数据类型是否为默认数据类型
        ASSERT_EQ(tensor.dtype(), default_dtype);
        // 验证张量是否不需要梯度计算
        ASSERT_FALSE(tensor.requires_grad());
    }
}

# 定义了一个测试用例 TEST(TensorTest, TorchTensorCtorZeroSizedDim)，测试 Torch 张量在零维情况下的构造函数行为
TEST(TensorTest, TorchTensorCtorZeroSizedDim) {
  # 调用 test_TorchTensorCtorZeroSizedDim_expected_dtype 函数，预期默认数据类型为 torch::kFloat
  test_TorchTensorCtorZeroSizedDim_expected_dtype(
      /*default_dtype=*/torch::kFloat);
  # 再次调用 test_TorchTensorCtorZeroSizedDim_expected_dtype 函数，预期默认数据类型为 torch::kDouble
  test_TorchTensorCtorZeroSizedDim_expected_dtype(
      /*default_dtype=*/torch::kDouble);
}

# 定义了一个函数 test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype，用于测试在不指定数据类型的情况下 Torch 张量构造函数的行为
void test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype(
    c10::ScalarType default_dtype) {
  # 设置默认数据类型为 default_dtype
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  # 断言 Torch 张量的数据类型为 default_dtype
  ASSERT_EQ(torch::tensor({1., 2., 3.}).dtype(), default_dtype);
  ASSERT_EQ(torch::tensor({{1., 2., 3.}}).dtype(), default_dtype);
  ASSERT_EQ(
      torch::tensor({1., 2., 3.}, torch::TensorOptions()).dtype(),
      default_dtype);
  ASSERT_EQ(
      torch::tensor({{1., 2., 3.}}, torch::TensorOptions()).dtype(),
      default_dtype);
}

# 定义了一个测试用例 TEST(TensorTest, TorchTensorCtorWithoutSpecifyingDtype)，测试 Torch 张量在不指定数据类型的情况下的构造函数行为
TEST(TensorTest, TorchTensorCtorWithoutSpecifyingDtype) {
  # 断言 Torch 张量的默认数据类型为 torch::kLong
  ASSERT_EQ(torch::tensor({1, 2, 3}).dtype(), torch::kLong);
  ASSERT_EQ(torch::tensor({{1, 2, 3}}).dtype(), torch::kLong);
  ASSERT_EQ(
      torch::tensor({1, 2, 3}, torch::TensorOptions()).dtype(), torch::kLong);
  ASSERT_EQ(
      torch::tensor({{1, 2, 3}}, torch::TensorOptions()).dtype(), torch::kLong);

  # 调用 test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype 函数，预期默认数据类型为 torch::kFloat
  test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype(
      /*default_dtype=*/torch::kFloat);
  # 再次调用 test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype 函数，预期默认数据类型为 torch::kDouble
  test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype(
      /*default_dtype=*/torch::kDouble);
}

# 定义了一个函数 test_TorchTensorCtorWithNonDtypeOptions_expected_dtype，用于测试在指定非数据类型选项的情况下 Torch 张量构造函数的行为
void test_TorchTensorCtorWithNonDtypeOptions_expected_dtype(
    c10::ScalarType default_dtype) {
  # 设置默认数据类型为 default_dtype
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  # 断言 Torch 张量的数据类型为 torch::kLong
  ASSERT_EQ(
      torch::tensor({1, 2, 3}, torch::TensorOptions()).dtype(), torch::kLong);
  ASSERT_EQ(
      torch::tensor(at::ArrayRef<int>({1, 2, 3}), torch::TensorOptions())
          .dtype(),
      torch::kLong);
  ASSERT_EQ(
      torch::tensor(std::vector<int>({1, 2, 3}), torch::TensorOptions())
          .dtype(),
      torch::kLong);

  # 断言 Torch 张量的数据类型为 default_dtype
  ASSERT_EQ(
      torch::tensor({1., 2., 3.}, torch::TensorOptions()).dtype(),
      default_dtype);
  ASSERT_EQ(
      torch::tensor(at::ArrayRef<double>({1., 2., 3.}), torch::TensorOptions())
          .dtype(),
      default_dtype);
  ASSERT_EQ(
      torch::tensor(std::vector<double>({1., 2., 3.}), torch::TensorOptions())
          .dtype(),
      default_dtype);

  # 断言 Torch 张量的数据类型为 default_dtype
  ASSERT_EQ(
      torch::tensor({1.f, 2.f, 3.f}, torch::TensorOptions()).dtype(),
      default_dtype);
  ASSERT_EQ(
      torch::tensor(
          at::ArrayRef<float>({1.f, 2.f, 3.f}), torch::TensorOptions())
          .dtype(),
      default_dtype);
  ASSERT_EQ(
      torch::tensor(std::vector<float>({1.f, 2.f, 3.f}), torch::TensorOptions())
          .dtype(),
      default_dtype);
}

# 定义了一个测试用例 TEST(TensorTest, TorchTensorCtorWithNonDtypeOptions)，测试在指定非数据类型选项的情况下 Torch 张量构造函数的行为
TEST(TensorTest, TorchTensorCtorWithNonDtypeOptions) {
  # 调用 test_TorchTensorCtorWithNonDtypeOptions_expected_dtype 函数，预期默认数据类型为 torch::kFloat
  test_TorchTensorCtorWithNonDtypeOptions_expected_dtype(
      /*default_dtype=*/torch::kFloat);
  # 再次调用 test_TorchTensorCtorWithNonDtypeOptions_expected_dtype 函数，预期默认数据类型为 torch::kDouble
  test_TorchTensorCtorWithNonDtypeOptions_expected_dtype(
      /*default_dtype=*/torch::kDouble);
}

# 定义了一个函数 test_Arange_expected_dtype，用于测试 torch::arange 函数的行为
void test_Arange_expected_dtype(c10::ScalarType default_dtype) {
  # 设置默认数据类型为 default_dtype
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  # 断言 torch::arange 函数返回张量的数据类型为 default_dtype
  ASSERT_EQ(torch::arange(0., 5).dtype(), default_dtype);
}
TEST(TensorTest, Arange) {
  {
    // 调用 torch 库中的 arange 函数生成从 0 到 4 的整数张量
    auto x = torch::arange(0, 5);
    // 断言张量 x 的数据类型为长整型
    ASSERT_EQ(x.dtype(), torch::kLong);
  }
  // 调用测试函数，验证 arange 函数生成的张量数据类型为指定类型（torch::kFloat）
  test_Arange_expected_dtype(torch::kFloat);
  // 调用测试函数，验证 arange 函数生成的张量数据类型为指定类型（torch::kDouble）
  test_Arange_expected_dtype(torch::kDouble);
}

TEST(TensorTest, PrettyPrintTensorDataContainer) {
  // 断言转换 TensorDataContainer 对象为字符串后的结果为 "1.1"
  { ASSERT_EQ(c10::str(torch::detail::TensorDataContainer(1.1)), "1.1"); }
  // 断言转换包含双精度浮点数的 TensorDataContainer 对象为字符串后的结果为 "{1.1, 2.2}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer({1.1, 2.2})), "{1.1, 2.2}");
  }
  // 断言转换包含双精度浮点数矩阵的 TensorDataContainer 对象为字符串后的结果为 "{{1, 2}, {3, 4}}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer({{1, 2}, {3, 4}})),
        "{{1, 2}, {3, 4}}");
  }
  // 断言转换包含深层嵌套的双精度浮点数张量的 TensorDataContainer 对象为字符串后的结果
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer(
            {{{{{{{{1.1, 2.2, 3.3}}}}},
               {{{{{4.4, 5.5, 6.6}}}}},
               {{{{{7.7, 8.8, 9.9}}}}}}}})),
        "{{{{{{{{1.1, 2.2, 3.3}}}}}, {{{{{4.4, 5.5, 6.6}}}}}, {{{{{7.7, 8.8, 9.9}}}}}}}}");
  }
  // 断言转换包含深层嵌套的整数张量的 TensorDataContainer 对象为字符串后的结果为 "{{{{{{{{{{1}}}}}}}}}}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer({{{{{{{{{{1}}}}}}}}}})),
        "{{{{{{{{{{1}}}}}}}}}}");
  }
  // 断言转换包含空内容的 TensorDataContainer 对象为字符串后的结果为 "{{{{{{{{{{}}}}}}}}}}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer({{{{{{{{{{}}}}}}}}}})),
        "{{{{{{{{{{}}}}}}}}}}");
  }
  // 断言转换包含深层嵌套的整数张量的 TensorDataContainer 对象为字符串后的结果为 "{{{{{{{{{{1, 2}}}}}}}}}}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer({{{{{{{{{{1, 2}}}}}}}}}})),
        "{{{{{{{{{{1, 2}}}}}}}}}}");
  }
  // 断言转换包含双精度浮点数数组的 TensorDataContainer 对象为字符串后的结果为 "{1.1, 2.2}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer(
            at::ArrayRef<double>({1.1, 2.2}))),
        "{1.1, 2.2}");
  }
  // 断言转换包含双精度浮点数向量的 TensorDataContainer 对象为字符串后的结果为 "{1.1, 2.2}"
  {
    ASSERT_EQ(
        c10::str(torch::detail::TensorDataContainer(
            std::vector<double>({1.1, 2.2}))),
        "{1.1, 2.2}");
  }
}

TEST(TensorTest, TensorDataContainerCallingAccessorOfWrongType) {
  // 断言在 TensorDataContainer 对象为浮点数时调用 init_list() 方法会抛出特定异常
  {
    ASSERT_THROWS_WITH(
        torch::detail::TensorDataContainer(1.1).init_list(),
        "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
    // 断言在 TensorDataContainer 对象为浮点数时调用 tensor() 方法会抛出特定异常
    ASSERT_THROWS_WITH(
        torch::detail::TensorDataContainer(1.1).tensor(),
        "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
  }
  // 断言在 TensorDataContainer 对象为双精度浮点数向量时调用 scalar() 方法会抛出特定异常
  {
    ASSERT_THROWS_WITH(
        torch::detail::TensorDataContainer({1.1, 2.2}).scalar(),
        "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    // 断言在 TensorDataContainer 对象为双精度浮点数向量时调用 tensor() 方法会抛出特定异常
    ASSERT_THROWS_WITH(
        torch::detail::TensorDataContainer({1.1, 2.2}).tensor(),
        "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
  }
  // 断言在 TensorDataContainer 对象为双精度浮点数数组时调用 scalar() 方法会抛出特定异常
  {
    ASSERT_THROWS_WITH(
        torch::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2}))
            .scalar(),
        "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    // 断言在 TensorDataContainer 对象为双精度浮点数数组时调用 init_list() 方法会抛出特定异常
    ASSERT_THROWS_WITH(
        torch::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2}))
            .init_list(),
        "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
  }
}
TEST(TensorTest, FromBlob) {
  // 创建包含双精度浮点数的向量
  std::vector<double> v = {1.0, 2.0, 3.0};
  // 从给定数据创建张量，指定数据类型为双精度浮点数，并标记为需要梯度计算
  auto tensor = torch::from_blob(
      v.data(), v.size(), torch::dtype(torch::kFloat64).requires_grad(true));
  // 断言张量需要进行梯度计算
  ASSERT_TRUE(tensor.requires_grad());
  // 断言张量的数据类型为双精度浮点数
  ASSERT_EQ(tensor.dtype(), torch::kFloat64);
  // 断言张量元素个数为3
  ASSERT_EQ(tensor.numel(), 3);
  // 断言张量的第一个元素值为1
  ASSERT_EQ(tensor[0].item<double>(), 1);
  // 断言张量的第二个元素值为2
  ASSERT_EQ(tensor[1].item<double>(), 2);
  // 断言张量的第三个元素值为3
  ASSERT_EQ(tensor[2].item<double>(), 3);
  // 上述语法未复制数据，并且删除器上下文为nullptr
  ASSERT_EQ(tensor.storage().data_ptr().get_context(), nullptr);
}

TEST(TensorTest, FromBlobUsesDeleter) {
  // 布尔变量，用于检测删除器是否被调用
  bool called = false;
  {
    // 创建包含整数的向量
    std::vector<int32_t> v = {1, 2, 3};
    // 从给定数据创建张量，指定数据类型为32位整数，并传入删除器函数
    auto tensor = torch::from_blob(
        v.data(),
        v.size(),
        /*deleter=*/[&called](void* data) { called = true; },
        torch::kInt32);
  }
  // 断言删除器被调用
  ASSERT_TRUE(called);
}

TEST(TensorTest, FromBlobWithStrides) {
  // clang-format off
  // 创建包含整数的向量，按照列主序组织
  std::vector<int32_t> v = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  // clang-format on
  // 从给定数据创建张量，指定尺寸为3x3，指定步幅为{1, 3}，数据类型为32位整数
  auto tensor = torch::from_blob(
      v.data(),
      /*sizes=*/{3, 3},
      /*strides=*/{1, 3},
      torch::kInt32);
  // 断言张量的数据类型为32位整数
  ASSERT_EQ(tensor.dtype(), torch::kInt32);
  // 断言张量元素个数为9
  ASSERT_EQ(tensor.numel(), 9);
  // 预期的步幅
  const std::vector<int64_t> expected_strides = {1, 3};
  // 断言张量的步幅符合预期
  ASSERT_EQ(tensor.strides(), expected_strides);
  // 遍历张量，检查每个元素的值是否正确
  for (const auto i : c10::irange(tensor.size(0))) {
    for (const auto j : c10::irange(tensor.size(1))) {
      // 注意：这是列主序因为步幅被交换了
      EXPECT_EQ(tensor[i][j].item<int32_t>(), 1 + (j * tensor.size(1)) + i);
    }
  }
}

TEST(TensorTest, Item) {
  {
    // 创建一个包含单个浮点数的张量
    torch::Tensor tensor = torch::tensor(3.14);
    // 获取张量的标量值
    torch::Scalar scalar = tensor.item();
    // 断言标量值接近于3.14
    ASSERT_NEAR(scalar.to<float>(), 3.14, 1e-5);
  }
  {
    // 创建一个包含单个整数的张量
    torch::Tensor tensor = torch::tensor(123);
    // 获取张量的标量值
    torch::Scalar scalar = tensor.item();
    // 断言标量值为123
    ASSERT_EQ(scalar.to<int>(), 123);
  }
}

TEST(TensorTest, Item_CUDA) {
  {
    // 在CUDA设备上创建一个包含单个浮点数的张量
    torch::Tensor tensor = torch::tensor(3.14, torch::kCUDA);
    // 获取张量的标量值
    torch::Scalar scalar = tensor.item();
    // 断言标量值接近于3.14
    ASSERT_NEAR(scalar.to<float>(), 3.14, 1e-5);
  }
  {
    // 在CUDA设备上创建一个包含单个整数的张量
    torch::Tensor tensor = torch::tensor(123, torch::kCUDA);
    // 获取张量的标量值
    torch::Scalar scalar = tensor.item();
    // 断言标量值为123
    ASSERT_EQ(scalar.to<int>(), 123);
  }
}

TEST(TensorTest, DataPtr) {
  // 创建一个空的张量，尺寸为{3, 4}，数据类型为32位浮点数
  auto tensor = at::empty({3, 4}, at::kFloat);
  // 使用相同的选项创建张量的副本
  auto tensor_not_copy = tensor.to(tensor.options());
  // 断言两个张量共享相同的数据指针
  ASSERT_EQ(tensor_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  // 断言两个张量的数据指针相等
  ASSERT_EQ(tensor_not_copy.data_ptr(), tensor.data_ptr());
}

TEST(TensorTest, Data) {
  // 创建一个3x3随机张量
  const auto tensor = torch::rand({3, 3});
  // 断言张量与其数据相等
  ASSERT_TRUE(torch::equal(tensor, tensor.data()));
}

TEST(TensorTest, BackwardAndGrad) {
  // 创建一个包含单个浮点数的张量，标记为需要梯度计算
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 计算张量的平方
  auto y = x * x;
  // 对张量进行反向传播
  y.backward();
  // 断言张量x的梯度为10.0
  ASSERT_EQ(x.grad().item<float>(), 10.0);
}
TEST(TensorTest, BackwardCreatesOnesGrad) {
  // 创建一个张量 x，包含一个元素值为 5，类型为浮点数，需要计算梯度
  const auto x =
      torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 对张量 x 进行反向传播计算梯度
  x.backward();
  // 断言：x 的梯度应该等于一个和 x 形状相同的全为 1 的张量
  ASSERT_TRUE(torch::equal(x.grad(), torch::ones_like(x)));
}

TEST(TensorTest, BackwardNonScalarOutputs) {
  // 创建一个 5x5 大小的张量 x，并且需要计算梯度
  auto x = torch::randn({5, 5}, torch::requires_grad());
  // 计算张量 x 的平方，得到张量 y
  auto y = x * x;
  // 断言：对非标量输出调用 backward() 应该抛出异常，异常信息为指定的字符串
  ASSERT_THROWS_WITH(
      y.backward(), "grad can be implicitly created only for scalar outputs");
}

TEST(TensorTest, BackwardComplexScalarOutput) {
  // 创建一个 5x5 大小的张量 x，并且需要计算梯度
  auto x = torch::randn({5, 5}, torch::requires_grad());
  // 创建一个复数标量常量，将其与张量 x 的元素相乘并求和得到标量 y
  auto y = (x * c10::Scalar(c10::complex<float>(0, 0.5))).sum();
  // 断言：对复数标量输出调用 backward() 应该抛出异常，异常信息为指定的字符串
  ASSERT_THROWS_WITH(
      y.backward(), "grad can be computed only for real scalar outputs");
}

TEST(TensorTest, IsLeaf) {
  // 创建一个包含单个元素值为 5 的张量 x，并且需要计算梯度
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 计算张量 x 的平方，得到张量 y
  auto y = x * x;
  // 断言：张量 x 应该是叶子节点
  ASSERT_TRUE(x.is_leaf());
  // 断言：张量 y 不应该是叶子节点
  ASSERT_FALSE(y.is_leaf());
}

TEST(TensorTest, OutputNr) {
  // 创建一个包含单个元素值为 5 的张量 x，并且需要计算梯度
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 计算张量 x 的平方，得到张量 y
  auto y = x * x;
  // 断言：张量 x 的输出编号应该为 0
  ASSERT_EQ(x.output_nr(), 0);
  // 断言：张量 y 的输出编号应该为 0
  ASSERT_EQ(y.output_nr(), 0);
}

TEST(TensorTest, Version) {
  // 创建一个元素全为 1 的长度为 3 的张量 x
  auto x = torch::ones(3);
  // 断言：张量 x 的版本号应该为 0
  ASSERT_EQ(x._version(), 0);
  // 张量 x 执行原地乘以 2 的操作
  x.mul_(2);
  // 断言：张量 x 的版本号应该为 1
  ASSERT_EQ(x._version(), 1);
  // 张量 x 执行原地加 1 的操作
  x.add_(1);
  // 断言：张量 x 的版本号应该为 2
  ASSERT_EQ(x._version(), 2);
}

TEST(TensorTest, Detach) {
  // 创建一个包含单个元素值为 5 的张量 x，并且需要计算梯度
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 计算张量 x 的平方，得到张量 y
  auto y = x * x;
  // 对张量 y 进行 detach 操作，得到 detached 张量 y_detached
  const auto y_detached = y.detach();
  // 断言：张量 y 不应该是叶子节点
  ASSERT_FALSE(y.is_leaf());
  // 断言：detached 张量 y_detached 应该是叶子节点
  ASSERT_TRUE(y_detached.is_leaf());
  // 断言：detached 张量 y_detached 不需要计算梯度
  ASSERT_FALSE(y_detached.requires_grad());
}

TEST(TensorTest, DetachInplace) {
  // 创建一个包含单个元素值为 5 的张量 x，并且需要计算梯度
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 计算张量 x 的平方，得到张量 y
  auto y = x * x;
  // 对张量 y 执行原地 detach 操作，得到 detached 张量 y_detached
  auto y_detached = y.detach_();
  // 断言：张量 y 应该是叶子节点
  ASSERT_TRUE(y.is_leaf());
  // 断言：张量 y 不需要计算梯度
  ASSERT_FALSE(y.requires_grad());
  // 断言：detached 张量 y_detached 应该是叶子节点
  ASSERT_TRUE(y_detached.is_leaf());
  // 断言：detached 张量 y_detached 不需要计算梯度
  ASSERT_FALSE(y_detached.requires_grad());
}

TEST(TensorTest, SetData) {
  // 创建一个包含随机数的大小为 5 的张量 x
  auto x = torch::randn({5});
  // 创建一个包含随机数的大小为 5 的张量 y
  auto y = torch::randn({5});
  // 断言：张量 x 和 y 应该不相等
  ASSERT_FALSE(torch::equal(x, y));
  // 断言：张量 x 和 y 应该拥有不同的数据指针
  ASSERT_NE(x.data_ptr<float>(), y.data_ptr<float>());

  // 将张量 x 的数据替换为张量 y 的数据
  x.set_data(y);
  // 断言：张量 x 和 y 应该相等
  ASSERT_TRUE(torch::equal(x, y));
  // 断言：张量 x 和 y 应该共享相同的数据指针
  ASSERT_EQ(x.data_ptr<float>(), y.data_ptr<float>());
}

TEST(TensorTest, RequiresGradInplace) {
  // 创建一个包含单个元素值为 5.0 的张量 x
  auto x = torch::tensor({5.0});
  // 将张量 x 设置为需要计算梯度
  x.requires_grad_(true);
  // 断言：张量 x 应该需要计算梯度
  ASSERT_TRUE(x.requires_grad());

  // 计算张量 x 的平方，得到张量 y
  auto y = x * x;
  // 断言：尝试对非叶子节点 y 执行 requires_grad_(false) 应该抛出异常
  ASSERT_THROWS_WITH(
      y.requires_grad_(false),
      "you can only change requires_grad flags of leaf variables.");

  // 将张量 x 设置为不需要计算梯度
  x.requires_grad_(false);
  // 断言：张量 x 不应该需要计算梯度
  ASSERT_FALSE(x.requires_grad());

  // 创建一个包含整数 5 的张量 int_tensor
  const auto int_tensor =
      torch::tensor({5}, at::TensorOptions().dtype(torch::kInt));
  // 断言：尝试对整数张量 int_tensor 执行 requires_grad_(true) 应该抛出异常
  ASSERT_THROWS_WITH(
      int_tensor.requires_grad_(true),
      "Only Tensors of floating point and complex dtype can require gradients");
}
TEST(TensorTest, StdDimension) {
  // Test that std(0) doesn't select the std(unbiased=False) overload (gh-40287)
  // 生成一个大小为[4, 3]的随机张量
  auto x = torch::randn({4, 3});
  // 计算张量在维度0上的标准差
  auto std = x.std(0);

  // 断言：计算方差在维度0上的元素数量应为3
  ASSERT_EQ(x.var(0).numel(), 3);
  // 断言：计算标准差在维度0上的元素数量应为3
  ASSERT_EQ(x.std(0).numel(), 3);

  // 断言：使用无偏估计计算方差在维度0上的元素数量应为3
  ASSERT_EQ(x.var(0, /*unbiased=*/true).numel(), 3);
  // 断言：使用无偏估计计算标准差在维度0上的元素数量应为3
  ASSERT_EQ(x.std(0, /*unbiased=*/true).numel(), 3);

  // 断言：使用静态方法计算方差在维度0上的元素数量应为3
  ASSERT_EQ(torch::var(x, 0).numel(), 3);
  // 断言：使用静态方法计算方差和均值在维度0上的元素数量应为3
  ASSERT_EQ(std::get<0>(torch::var_mean(x, 0)).numel(), 3);
  // 断言：使用静态方法计算标准差在维度0上的元素数量应为3
  ASSERT_EQ(torch::std(x, 0).numel(), 3);
  // 断言：使用静态方法计算标准差和均值在维度0上的元素数量应为3
  ASSERT_EQ(std::get<0>(torch::std_mean(x, 0)).numel(), 3);

  // 断言：使用无偏估计计算方差在维度0上的元素数量应为3
  ASSERT_EQ(torch::var(x, 0, /*unbiased=*/true).numel(), 3);
  // 断言：使用无偏估计计算方差和均值在维度0上的元素数量应为3
  ASSERT_EQ(std::get<0>(torch::var_mean(x, 0, /*unbiased=*/true)).numel(), 3);
  // 断言：使用无偏估计计算标准差在维度0上的元素数量应为3
  ASSERT_EQ(torch::std(x, 0, /*unbiased=*/true).numel(), 3);
  // 断言：使用无偏估计计算标准差和均值在维度0上的元素数量应为3
  ASSERT_EQ(std::get<0>(torch::std_mean(x, 0, /*unbiased=*/true)).numel(), 3);
}

TEST(TensorTest, ReshapeAlias) {
  // Tests the behavior of the _reshape_alias private operator so
  // that it matches the behavior of as_strided and view.
  // 生成一个大小为[3, 3]的随机张量
  auto x = torch::randn({3, 3});
  // 断言：比较_reshape_alias操作符和as_strided的行为是否相等
  ASSERT_TRUE(torch::equal(
      torch::_reshape_alias(x, {2, 2}, {1, 2}),
      torch::as_strided(x, {2, 2}, {1, 2})));
  // 断言：比较_reshape_alias操作符和view的行为是否相等
  ASSERT_TRUE(torch::equal(torch::_reshape_alias(x, {9}, {1}), x.view({-1})));

  // 测试反向传播是否正常工作
  auto y = torch::randn({3, 3}, torch::requires_grad(true));
  auto z = torch::clone(y).detach().requires_grad_(true);
  (y * y).view({-1}).mean().backward();
  torch::_reshape_alias((z * z), {9}, {1}).mean().backward();
  // 断言：比较y和z的梯度是否相等
  ASSERT_TRUE(torch::equal(y.grad(), z.grad()));
}

TEST(TensorTest, BackendMetadata) {
  // Tests ability to assign custom backend metadata to tensor.
  // 定义一个自定义的后端元数据结构
  struct CustomBackendMetadata : public c10::BackendMeta {
    mutable bool cloned_{false}; // for testing this field will mutate when
                                 // clone() is called by shallow_copy_from.
    // 覆盖基类方法，标记为已克隆
    c10::intrusive_ptr<c10::BackendMeta> clone(
        const c10::intrusive_ptr<c10::BackendMeta>& ptr) const override {
      cloned_ = true;
      return c10::BackendMeta::clone(ptr);
    }
  };

  at::Tensor y;
  c10::intrusive_ptr<c10::BackendMeta> tmeta{};
  CustomBackendMetadata* custom_tmeta{nullptr};

  {
    // 生成一个大小为[3, 3]的张量
    auto x = torch::ones({3, 3});
    auto impl{x.unsafeGetTensorImpl()};
    ASSERT_TRUE(impl != nullptr);

    // 获取张量的后端元数据指针，并进行断言
    tmeta = impl->get_backend_meta_intrusive_ptr();
    ASSERT_TRUE(tmeta == nullptr);
    // 创建一个新的自定义后端元数据对象并设置给张量
    c10::intrusive_ptr<c10::BackendMeta> new_tmeta{
        std::unique_ptr<c10::BackendMeta>(new CustomBackendMetadata())};
    impl->set_backend_meta(new_tmeta);
    // 再次获取后端元数据指针，并进行断言
    tmeta = impl->get_backend_meta_intrusive_ptr();
    ASSERT_TRUE(tmeta == new_tmeta);
    // 将tmeta转换为CustomBackendMetadata类型，并进行进一步断言
    custom_tmeta = dynamic_cast<CustomBackendMetadata*>(tmeta.get());
    ASSERT_TRUE(custom_tmeta != nullptr);
    ASSERT_TRUE(custom_tmeta->cloned_ == false);
    // 使用 unsafeGetTensorImpl() 获取 y 引用的 TensorImpl 对象，并调用 shallow_copy_from() 方法将 x 的指针浅复制到该对象
    y.unsafeGetTensorImpl()->shallow_copy_from(x.getIntrusivePtr());
  }

  // 断言：验证 y 引用的 TensorImpl 对象的后端元数据指针与 tmeta 相等
  ASSERT_TRUE(
      tmeta == y.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr());
  // 断言：验证 y 引用的 TensorImpl 对象的后端元数据指针与 tmeta 的原生指针相等
  ASSERT_TRUE(tmeta.get() == y.unsafeGetTensorImpl()->get_backend_meta());
  // 断言：验证自定义的后端元数据对象 custom_tmeta 的 cloned_ 标志为 true
  ASSERT_TRUE(custom_tmeta->cloned_ == true);
}


注释：

# 这行代码表示一个代码块的结束，关闭了一个代码段或函数的定义。
```