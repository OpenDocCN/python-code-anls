# `.\pytorch\test\cpp\api\tensor_flatten.cpp`

```
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include <test/cpp/api/support.h> // 包含测试支持函数的头文件

#include <torch/csrc/autograd/variable.h> // 包含 PyTorch 自动求导变量的头文件
#include <torch/csrc/utils/tensor_flatten.h> // 包含 PyTorch 张量展平工具的头文件
#include <torch/torch.h> // 包含 PyTorch 核心头文件

using namespace torch::test; // 使用 torch::test 命名空间

TEST(UnflattenDenseTensorTest, TestEmptyTensor) { // 定义名为 UnflattenDenseTensorTest 的测试用例类，测试空张量情况
  auto emptyTensor1 = at::tensor(std::vector<int>()); // 创建一个空张量 emptyTensor1
  auto emptyTensor2 = at::tensor(std::vector<int>()); // 创建另一个空张量 emptyTensor2
  auto tensor1 = at::tensor({1, 2, 3}); // 创建张量 tensor1，并初始化为 {1, 2, 3}
  auto tensor2 = at::tensor({4, 5}); // 创建张量 tensor2，并初始化为 {4, 5}
  auto tensorList = // 创建张量列表 tensorList，包含 tensor1, emptyTensor1, emptyTensor2, tensor2
      std::vector<at::Tensor>({tensor1, emptyTensor1, emptyTensor2, tensor2});
  auto flatTensor = at::tensor({1, 2, 3, 4, 5}); // 创建扁平化张量 flatTensor，并初始化为 {1, 2, 3, 4, 5}
  auto unflatten_results = // 调用 unflatten_dense_tensors 函数将 flatTensor 恢复为原始张量列表
      torch::utils::unflatten_dense_tensors(flatTensor, tensorList);
  ASSERT_EQ(unflatten_results.size(), 4); // 断言恢复的结果列表大小为 4
  ASSERT_EQ(unflatten_results.at(0).numel(), 3); // 断言第一个张量的元素数量为 3
  ASSERT_EQ(unflatten_results.at(1).numel(), 0); // 断言第二个张量的元素数量为 0
  ASSERT_EQ(unflatten_results.at(2).numel(), 0); // 断言第三个张量的元素数量为 0
  ASSERT_EQ(unflatten_results.at(3).numel(), 2); // 断言第四个张量的元素数量为 2

  // 空张量的地址为 nullptr，因为内存尚未分配
  ASSERT_EQ(unflatten_results.at(1).data_ptr(), nullptr);
  ASSERT_EQ(unflatten_results.at(2).data_ptr(), nullptr);

  // 在 unflatten_dense_tensors() 函数未修复空张量问题时，
  // 恢复的空张量 unflatten_results.at(1) 可能与非空张量 unflatten_results.at(3) 共享相同的存储空间。
  // 经修复后，空张量和非空张量不再共享存储空间。
  ASSERT_NE(
      unflatten_results.at(1).data_ptr(), unflatten_results.at(3).data_ptr());

  unflatten_results.at(1).resize_(1); // 调整第一个空张量的大小为 1
  unflatten_results.at(2).resize_(1); // 调整第二个空张量的大小为 1

  // 调整大小后的两个空张量不再共享相同的存储空间。在 unflatten_dense_tensors() 函数未修复空张量问题时，
  // 调整大小后的张量可能共享相同的存储空间。
  ASSERT_NE(
      unflatten_results.at(1).data_ptr(), unflatten_results.at(2).data_ptr());
}
```