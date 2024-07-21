# `.\pytorch\test\cpp\api\ivalue.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/core/ivalue.h>  // 包含 PyTorch ATen 库的 IValue 头文件

#include <c10/util/flat_hash_map.h>  // 包含 C10 库的 flat_hash_map 头文件
#include <c10/util/irange.h>  // 包含 C10 库的 irange 头文件
#include <c10/util/tempfile.h>  // 包含 C10 库的临时文件操作头文件

#include <torch/torch.h>  // 包含 PyTorch 主头文件

#include <test/cpp/api/support.h>  // 包含 PyTorch 测试支持库的头文件

#include <cstdio>  // 包含 C 标准输入输出库的头文件
#include <memory>  // 包含 C++ 标准库的内存管理头文件
#include <sstream>  // 包含 C++ 标准库的字符串流头文件
#include <string>  // 包含 C++ 标准库的字符串处理头文件
#include <vector>  // 包含 C++ 标准库的向量容器头文件

using namespace torch::test;  // 使用 torch::test 命名空间
using namespace torch::nn;  // 使用 torch::nn 命名空间
using namespace torch::optim;  // 使用 torch::optim 命名空间

TEST(IValueTest, DeepcopyTensors) {  // 定义测试用例 IValueTest.DeepcopyTensors
  torch::Tensor t0 = torch::randn({2, 3});  // 创建形状为 (2, 3) 的随机张量 t0
  torch::Tensor t1 = torch::randn({3, 4});  // 创建形状为 (3, 4) 的随机张量 t1
  torch::Tensor t2 = t0.detach();  // 对 t0 进行分离操作，得到张量 t2
  torch::Tensor t3 = t0;  // 将 t0 赋值给张量 t3
  torch::Tensor t4 = t1.as_strided({2, 3}, {3, 1}, 2);  // 使用 as_strided 创建形状为 (2, 3) 的张量 t4
  std::vector<torch::Tensor> tensor_vector = {t0, t1, t2, t3, t4};  // 创建张量向量 tensor_vector 包含 t0, t1, t2, t3, t4
  c10::List<torch::Tensor> tensor_list(tensor_vector);  // 使用 tensor_vector 创建 C10 的 Tensor 列表 tensor_list
  torch::IValue tensor_list_ivalue(tensor_list);  // 使用 tensor_list 创建 PyTorch 的 IValue 对象 tensor_list_ivalue

  c10::IValue::CompIdentityIValues ivalue_compare;  // 创建 CompIdentityIValues 对象 ivalue_compare，用于比较 IValue 的内容是否相同

  // 确保设置的配置正确
  ASSERT_TRUE(ivalue_compare(tensor_list[0].get(), tensor_list[3].get()));  // 断言 tensor_list 中第0个和第3个元素相等
  ASSERT_FALSE(ivalue_compare(tensor_list[0].get(), tensor_list[1].get()));  // 断言 tensor_list 中第0个和第1个元素不相等
  ASSERT_FALSE(ivalue_compare(tensor_list[0].get(), tensor_list[2].get()));  // 断言 tensor_list 中第0个和第2个元素不相等
  ASSERT_FALSE(ivalue_compare(tensor_list[1].get(), tensor_list[4].get()));  // 断言 tensor_list 中第1个和第4个元素不相等
  ASSERT_TRUE(tensor_list[0].get().isAliasOf(tensor_list[2].get()));  // 断言 tensor_list 中第0个和第2个元素是别名关系

  c10::IValue copied_ivalue = tensor_list_ivalue.deepcopy();  // 深拷贝 tensor_list_ivalue 得到 copied_ivalue
  c10::List<torch::IValue> copied_list = copied_ivalue.toList();  // 将 copied_ivalue 转换为列表 copied_list

  // 确保设置的配置正确
  ASSERT_TRUE(ivalue_compare(copied_list[0].get(), copied_list[3].get()));  // 断言 copied_list 中第0个和第3个元素相等
  ASSERT_FALSE(ivalue_compare(copied_list[0].get(), copied_list[1].get()));  // 断言 copied_list 中第0个和第1个元素不相等
  ASSERT_FALSE(ivalue_compare(copied_list[0].get(), copied_list[2].get()));  // 断言 copied_list 中第0个和第2个元素不相等
  ASSERT_FALSE(ivalue_compare(copied_list[1].get(), copied_list[4].get()));  // 断言 copied_list 中第1个和第4个元素不相等
  // 注意：实际上，这是错误的。理想情况下，这些应该是别名关系。
  ASSERT_FALSE(copied_list[0].get().isAliasOf(copied_list[2].get()));  // 断言 copied_list 中第0个和第2个元素不是别名关系

  ASSERT_TRUE(copied_list[0].get().toTensor().allclose(
      tensor_list[0].get().toTensor()));  // 断言 copied_list 中第0个元素的张量与 tensor_list 中第0个元素的张量在数值上接近
  ASSERT_TRUE(copied_list[1].get().toTensor().allclose(
      tensor_list[1].get().toTensor()));  // 断言 copied_list 中第1个元素的张量与 tensor_list 中第1个元素的张量在数值上接近
  ASSERT_TRUE(copied_list[2].get().toTensor().allclose(
      tensor_list[2].get().toTensor()));  // 断言 copied_list 中第2个元素的张量与 tensor_list 中第2个元素的张量在数值上接近
  ASSERT_TRUE(copied_list[3].get().toTensor().allclose(
      tensor_list[3].get().toTensor()));  // 断言 copied_list 中第3个元素的张量与 tensor_list 中第3个元素的张量在数值上接近
  ASSERT_TRUE(copied_list[4].get().toTensor().allclose(
      tensor_list[4].get().toTensor()));  // 断言 copied_list 中第4个元素的张量与 tensor_list 中第4个元素的张量在数值上接近
}
```