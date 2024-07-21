# `.\pytorch\test\mobile\lightweight_dispatch\test_codegen_unboxing.cpp`

```
#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/torch.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

// Cover codegen'd unboxing logic for these types:
//'Device',
//'Device?',
//'Dimname',
//'Dimname[1]',
//'Dimname[]',
//'Dimname[]?',
//'Generator?',
//'Layout?',
//'MemoryFormat',
//'MemoryFormat?',
//'Scalar',
//'Scalar?',
//'ScalarType',
//'ScalarType?',
//'Scalar[]',
//'Storage',
//'Stream',
//'Tensor',
//'Tensor(a!)',
//'Tensor(a!)[]',
//'Tensor(a)',
//'Tensor(b!)',
//'Tensor(c!)',
//'Tensor(d!)',
//'Tensor?',
//'Tensor?[]',
//'Tensor[]',
//'bool',
//'bool?',
//'bool[2]',
//'bool[3]',
//'bool[4]',
//'float',
//'float?',
//'float[]?',
//'int',
//'int?',
//'int[1]',
//'int[1]?',
//'int[2]',
//'int[2]?',
//'int[3]',
//'int[4]',
//'int[5]',
//'int[6]',
//'int[]',
//'int[]?',
//'str',
//'str?'

namespace torch {
namespace jit {
namespace mobile {

// 测试函数，验证 LiteInterpreter 对 int[], ScalarType?, Layout?, Device?, bool? 的解释器逻辑
TEST(LiteInterpreterTest, Ones) {
  // 加载待测试的模型文件名：ModelWithDTypeDeviceLayoutPinMemory.ptl
  auto testModelFile = "ModelWithDTypeDeviceLayoutPinMemory.ptl";

  // 模型 ModelWithDTypeDeviceLayoutPinMemory(torch.nn.Module):
  // 在 forward 函数中，创建一个大小为 [3, x] 的 torch.Tensor，全为 1，指定数据类型为 torch.int64，
  // 布局为 torch.strided，设备为 "cpu"
  Module bc = _load_for_mobile(testModelFile);

  // 设置输入为整数 4
  std::vector<c10::IValue> input{c10::IValue(4)};

  // 执行模型的 forward 方法，并获取结果
  const auto result = bc.forward(input);

  // 断言结果的第一个维度为 3
  ASSERT_EQ(result.toTensor().size(0), 3);
  // 断言结果的第二个维度为 4
  ASSERT_EQ(result.toTensor().size(1), 4);
}

// 测试函数，验证 LiteInterpreter 对索引操作的解释器逻辑
TEST(LiteInterpreterTest, Index) {
  // 加载待测试的模型文件名：ModelWithTensorOptional.ptl
  auto testModelFile = "ModelWithTensorOptional.ptl";

  // 模型 ModelWithTensorOptional(torch.nn.Module):
  // 在 forward 函数中，创建一个 2x2 的全零 torch.Tensor，并进行部分元素赋值
  Module bc = _load_for_mobile(testModelFile);

  // 设置索引为整数 0
  int64_t ind_1 = 0;

  // 执行模型的 forward 方法，传入索引值
  const auto result_1 = bc.forward({at::tensor(ind_1)});

  // 创建期望的 tensor，与模型输出结果进行比较
  at::Tensor expected = at::empty({1, 2}, c10::TensorOptions(c10::ScalarType::Float));
  expected[0][0] = 0;
  expected[0][1] = 1;

  // 断言模型输出结果与期望结果相等
  AT_ASSERT(result_1.toTensor().equal(expected));
}
TEST(LiteInterpreterTest, Gradient) {
  // 加载模型检查点：ModelWithScalarList.ptl
  auto testModelFile = "ModelWithScalarList.ptl";

  // 创建模块对象，加载移动设备上的模型
  Module bc = _load_for_mobile(testModelFile);

  // 对模型进行前向传播，参数为{0}
  const auto result_1 = bc.forward({0});
  
  // 期望的输出张量
  at::Tensor expected_1 = at::tensor({-1.5, -0.75, 3.75, 7.5}, c10::TensorOptions(c10::ScalarType::Float));
  
  // 断言前向传播的结果是否与期望的输出张量相等
  AT_ASSERT(result_1.toList().get(0).toTensor().equal(expected_1));

  // 对模型进行前向传播，参数为{1}
  const auto result_2 = bc.forward({1});
  
  // 期望的输出张量
  at::Tensor expected_2 = at::tensor({-3.0, -1.5, 7.5, 15.0}, c10::TensorOptions(c10::ScalarType::Float));
  
  // 断言前向传播的结果是否与期望的输出张量相等
  AT_ASSERT(result_2.toList().get(0).toTensor().equal(expected_2));
}

TEST(LiteInterpreterTest, Upsample) {
  // 加载模型检查点：ModelWithFloatList.ptl
  auto testModelFile = "ModelWithFloatList.ptl";

  // 创建模块对象，加载移动设备上的模型
  Module bc = _load_for_mobile(testModelFile);

  // 对模型进行前向传播，参数为1x2x3的全1张量
  const auto result_1 = bc.forward({at::ones({1, 2, 3})});
  
  // 期望的输出张量
  at::Tensor expected_1 = at::ones({1, 2, 6}, c10::TensorOptions(c10::ScalarType::Float));
  
  // 断言前向传播的结果是否与期望的输出张量相等
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, IndexTensor) {
  // 加载模型检查点：ModelWithListOfOptionalTensors.ptl
  auto testModelFile = "ModelWithListOfOptionalTensors.ptl";

  // 创建模块对象，加载移动设备上的模型
  Module bc = _load_for_mobile(testModelFile);

  // 对模型进行前向传播，参数为包含索引1的长整型张量
  const auto result_1 = bc.forward({at::tensor({1}, c10::TensorOptions(c10::ScalarType::Long))});
  
  // 期望的输出张量
  at::Tensor expected_1 = at::tensor({1.}, c10::TensorOptions(c10::ScalarType::Float));
  
  // 断言前向传播的结果是否与期望的输出张量相等
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, Conv2d) {
  // 加载模型检查点：ModelWithArrayOfInt.ptl
  auto testModelFile = "ModelWithArrayOfInt.ptl";

  // 创建模块对象，加载移动设备上的模型
  Module bc = _load_for_mobile(testModelFile);

  // 对模型进行前向传播，参数为1x1x1x1的全1张量
  const auto result_1 = bc.forward({at::ones({1, 1, 1, 1})});
  
  // 断言前向传播的结果的张量尺寸是否符合预期
  ASSERT_EQ(result_1.toTensor().sizes(), c10::IntArrayRef ({1,2,2,2}));
}
TEST(LiteInterpreterTest, AddTensor) {
  // 加载检查模型：ModelWithTensors.ptl
  auto testModelFile = "ModelWithTensors.ptl";

  // 创建模块对象并加载移动端模型文件
  Module bc = _load_for_mobile(testModelFile);

  // 调用模型的前向传播方法，传入参数并获取结果
  const auto result_1 = bc.forward({at::tensor({1, 2, 3}, c10::TensorOptions(c10::ScalarType::Long))});

  // 定义预期的张量结果
  at::Tensor expected_1 = at::tensor({2, 3, 4}, c10::TensorOptions(c10::ScalarType::Long));

  // 断言模型输出与预期结果相等
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, DivideTensor) {
  // 加载检查模型：ModelWithStringOptional.ptl
  auto testModelFile = "ModelWithStringOptional.ptl";

  // 创建模块对象并加载移动端模型文件
  Module bc = _load_for_mobile(testModelFile);

  // 调用模型的前向传播方法，传入参数并获取结果
  const auto result_1 = bc.forward({at::tensor({-12}, c10::TensorOptions(c10::ScalarType::Long))});

  // 定义预期的张量结果
  at::Tensor expected_1 = at::tensor({-4}, c10::TensorOptions(c10::ScalarType::Long));
  at::Tensor expected_2 = at::tensor({-4.}, c10::TensorOptions(c10::ScalarType::Float));

  // 断言模型输出中的第一个结果与预期结果相等
  AT_ASSERT(result_1.toList().get(0).toTensor().equal(expected_1));

  // 断言模型输出中的第二个结果与预期结果相等
  AT_ASSERT(result_1.toList().get(1).toTensor().equal(expected_2));
}

TEST(LiteInterpreterTest, MultipleOps) {
  // 加载检查模型：ModelWithMultipleOps.ptl
  auto testModelFile = "ModelWithMultipleOps.ptl";

  // 创建模块对象并加载移动端模型文件
  Module bc = _load_for_mobile(testModelFile);

  // 创建输入张量 b
  auto b = at::ones({2, 2, 2, 2});

  // 调用模型的前向传播方法，传入参数并获取结果
  const auto result = bc.forward({b});

  // 定义预期的张量结果
  at::Tensor expected = torch::tensor({{1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0}}, c10::TensorOptions(c10::ScalarType::Float));

  // 断言模型输出与预期结果相等
  AT_ASSERT(result.toTensor().equal(expected));
}
} // namespace mobile
} // namespace jit
} // namespace torch
```