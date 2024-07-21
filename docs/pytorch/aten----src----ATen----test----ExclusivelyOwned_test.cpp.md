# `.\pytorch\aten\src\ATen\test\ExclusivelyOwned_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <caffe2/core/tensor.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/intrusive_ptr.h>

#include <string>

// 匿名命名空间，限定作用域在当前文件
namespace {

// 模板类，用于测试 ExclusivelyOwned<T> 的行为
template <typename T>
class ExclusivelyOwnedTest : public ::testing::Test {
 public:
  // 默认构造的 ExclusivelyOwned 对象
  c10::ExclusivelyOwned<T> defaultConstructed;
  // 带有样本数据的 ExclusivelyOwned 对象
  c10::ExclusivelyOwned<T> sample;

 protected:
  // 设置测试环境，在每个测试用例执行前调用
  void SetUp() override;
  
  // 拆卸测试环境，在每个测试用例执行后调用
  void TearDown() override {
    // 重新构造 defaultConstructed
    defaultConstructed = c10::ExclusivelyOwned<T>();
    // 重新构造 sample
    sample = c10::ExclusivelyOwned<T>();
  }
};

// 获取样本值的模板函数声明
template <typename T>
T getSampleValue();

// 特化模板函数，返回一个 ATen Tensor 的样本值
template <>
at::Tensor getSampleValue() {
  return at::zeros({2, 2}).to(at::kCPU);
}

// 特化模板函数，返回一个 Caffe2 Tensor 的样本值
template <>
caffe2::Tensor getSampleValue() {
  return caffe2::Tensor(getSampleValue<at::Tensor>());
}

// 断言函数模板声明，用于验证是否为样本对象
template <typename T>
void assertIsSampleObject(const T& eo);

// 特化模板函数，用于验证 ATen Tensor 是否为样本对象
template <>
void assertIsSampleObject<at::Tensor>(const at::Tensor& t) {
  // 断言 ATen Tensor 的形状为 {2, 2}
  EXPECT_EQ(t.sizes(), (c10::IntArrayRef{2, 2}));
  // 断言 ATen Tensor 的步长为 {2, 1}
  EXPECT_EQ(t.strides(), (c10::IntArrayRef{2, 1}));
  // 断言 ATen Tensor 的数据类型为 Float
  ASSERT_EQ(t.scalar_type(), at::ScalarType::Float);
  // 预定义的全零数组，长度为 4
  static const float zeros[4] = {0};
  // 检查 ATen Tensor 的数据指针是否为全零数组
  EXPECT_EQ(memcmp(zeros, t.data_ptr(), 4 * sizeof(float)), 0);
}

// 特化模板函数，用于验证 Caffe2 Tensor 是否为样本对象
template <>
void assertIsSampleObject<caffe2::Tensor>(const caffe2::Tensor& t) {
  // 验证 Caffe2 Tensor 是否为样本 ATen Tensor 对象
  assertIsSampleObject<at::Tensor>(at::Tensor(t));
}

// 设置测试环境的具体实现
template <typename T>
void ExclusivelyOwnedTest<T>::SetUp() {
  // 默认构造一个 ExclusivelyOwned<T> 对象
  defaultConstructed = c10::ExclusivelyOwned<T>();
  // 使用样本值构造一个 ExclusivelyOwned<T> 对象
  sample = c10::ExclusivelyOwned<T>(getSampleValue<T>());
}

// 定义类型别名，测试类型为 ATen Tensor 和 Caffe2 Tensor
using ExclusivelyOwnedTypes = ::testing::Types<
  at::Tensor,
  caffe2::Tensor
>;

// 模板化测试套件，使用 ExclusivelyOwnedTest 模板类，测试指定的类型
TYPED_TEST_SUITE(ExclusivelyOwnedTest, ExclusivelyOwnedTypes);

// 测试默认构造函数的行为
TYPED_TEST(ExclusivelyOwnedTest, DefaultConstructor) {
  c10::ExclusivelyOwned<TypeParam> defaultConstructed;
}

// 测试移动构造函数的行为
TYPED_TEST(ExclusivelyOwnedTest, MoveConstructor) {
  // 移动构造 defaultConstructed
  auto movedDefault = std::move(this->defaultConstructed);
  // 移动构造 sample
  auto movedSample = std::move(this->sample);

  // 验证 movedSample 是否为样本对象
  assertIsSampleObject(*movedSample);
}

// 测试移动赋值操作的行为
TYPED_TEST(ExclusivelyOwnedTest, MoveAssignment) {
  // TearDown 函数会处理从默认构造的 ExclusivelyOwned 移动赋值操作
  c10::ExclusivelyOwned<TypeParam> anotherSample = c10::ExclusivelyOwned<TypeParam>(getSampleValue<TypeParam>());
  // 将 this->sample 的内容移动赋值给 anotherSample
  anotherSample = std::move(this->sample);
  // 验证 anotherSample 是否为样本对象
  assertIsSampleObject(*anotherSample);
}

// 测试从包含类型进行的移动赋值操作的行为
TYPED_TEST(ExclusivelyOwnedTest, MoveAssignmentFromContainedType) {
  c10::ExclusivelyOwned<TypeParam> anotherSample = c10::ExclusivelyOwned<TypeParam>(getSampleValue<TypeParam>());
  // 将 getSampleValue<TypeParam>() 的内容移动赋值给 anotherSample
  anotherSample = getSampleValue<TypeParam>();
  // 验证 anotherSample 是否为样本对象
  assertIsSampleObject(*anotherSample);
}

// 测试 take 函数的行为
TYPED_TEST(ExclusivelyOwnedTest, Take) {
  // 移动 this->sample 的内容到 x
  auto x = std::move(this->sample).take();
  // 验证 x 是否为样本对象
  assertIsSampleObject(x);
}

} // namespace

// C 风格的函数声明，用于检查 ATen Tensor 的样本值
extern "C" void inspectTensor() {
  auto t = getSampleValue<at::Tensor>();
}

// C 风格的函数声明，用于检查 ExclusivelyOwned 的 ATen Tensor 的样本值
extern "C" void inspectExclusivelyOwnedTensor() {
  c10::ExclusivelyOwned<at::Tensor> t(getSampleValue<at::Tensor>());
}
```