# `.\pytorch\test\cpp\api\parameterlist.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <c10/util/irange.h>  // 引入用于迭代范围的头文件
#include <torch/torch.h>  // 引入 PyTorch 的头文件

#include <algorithm>  // 引入算法头文件
#include <memory>  // 引入内存管理头文件
#include <vector>  // 引入向量容器头文件

#include <test/cpp/api/support.h>  // 引入测试支持相关的头文件

using namespace torch::nn;  // 使用 PyTorch 的神经网络命名空间
using namespace torch::test;  // 使用 PyTorch 测试相关命名空间

struct ParameterListTest : torch::test::SeedingFixture {};  // 定义测试结构体，继承自 SeedingFixture

TEST_F(ParameterListTest, ConstructsFromSharedPointer) {  // 定义测试用例
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));  // 创建一个需要梯度的随机张量
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));  // 创建一个不需要梯度的随机张量
  torch::Tensor tc = torch::randn({1, 2});  // 创建一个随机张量
  ASSERT_TRUE(ta.requires_grad());  // 断言张量 ta 需要梯度
  ASSERT_FALSE(tb.requires_grad());  // 断言张量 tb 不需要梯度
  ParameterList list(ta, tb, tc);  // 创建参数列表对象，并初始化为 ta、tb、tc
  ASSERT_EQ(list->size(), 3);  // 断言参数列表中元素个数为 3
}

TEST_F(ParameterListTest, isEmpty) {  // 定义测试用例
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));  // 创建一个需要梯度的随机张量
  ParameterList list;  // 创建空的参数列表对象
  ASSERT_TRUE(list->is_empty());  // 断言参数列表为空
  list->append(ta);  // 向参数列表中添加张量 ta
  ASSERT_FALSE(list->is_empty());  // 断言参数列表不为空
  ASSERT_EQ(list->size(), 1);  // 断言参数列表中元素个数为 1
}

TEST_F(ParameterListTest, PushBackAddsAnElement) {  // 定义测试用例
  ParameterList list;  // 创建空的参数列表对象
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));  // 创建一个需要梯度的随机张量
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));  // 创建一个不需要梯度的随机张量
  torch::Tensor tc = torch::randn({1, 2});  // 创建一个随机张量
  torch::Tensor td = torch::randn({1, 2, 3});  // 创建一个三维随机张量
  ASSERT_EQ(list->size(), 0);  // 断言参数列表中元素个数为 0
  ASSERT_TRUE(list->is_empty());  // 断言参数列表为空
  list->append(ta);  // 向参数列表中添加张量 ta
  ASSERT_EQ(list->size(), 1);  // 断言参数列表中元素个数为 1
  list->append(tb);  // 向参数列表中添加张量 tb
  ASSERT_EQ(list->size(), 2);  // 断言参数列表中元素个数为 2
  list->append(tc);  // 向参数列表中添加张量 tc
  ASSERT_EQ(list->size(), 3);  // 断言参数列表中元素个数为 3
  list->append(td);  // 向参数列表中添加张量 td
  ASSERT_EQ(list->size(), 4);  // 断言参数列表中元素个数为 4
}

TEST_F(ParameterListTest, ForEachLoop) {  // 定义测试用例
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));  // 创建一个需要梯度的随机张量
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));  // 创建一个不需要梯度的随机张量
  torch::Tensor tc = torch::randn({1, 2});  // 创建一个随机张量
  torch::Tensor td = torch::randn({1, 2, 3});  // 创建一个三维随机张量
  ParameterList list(ta, tb, tc, td);  // 创建参数列表对象，并初始化为 ta、tb、tc、td
  std::vector<torch::Tensor> params = {ta, tb, tc, td};  // 创建张量向量 params，包含 ta、tb、tc、td
  ASSERT_EQ(list->size(), 4);  // 断言参数列表中元素个数为 4
  int idx = 0;  // 初始化索引为 0
  for (const auto& pair : *list) {  // 遍历参数列表中的每个元素
    ASSERT_TRUE(  // 断言以下条件为真
        torch::all(torch::eq(pair.value(), params[idx++])).item<bool>());  // 检查 pair 的值是否与 params 向量中的对应张量相等
  }
}

TEST_F(ParameterListTest, AccessWithAt) {  // 定义测试用例
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));  // 创建一个需要梯度的随机张量
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));  // 创建一个不需要梯度的随机张量
  torch::Tensor tc = torch::randn({1, 2});  // 创建一个随机张量
  torch::Tensor td = torch::randn({1, 2, 3});  // 创建一个三维随机张量
  std::vector<torch::Tensor> params = {ta, tb, tc, td};  // 创建张量向量 params，包含 ta、tb、tc、td

  ParameterList list;  // 创建空的参数列表对象
  for (auto& param : params) {  // 遍历张量向量 params
    list->append(param);  // 向参数列表中添加张量 param
  }
  ASSERT_EQ(list->size(), 4);  // 断言参数列表中元素个数为 4

  // returns the correct module for a given index
  for (const auto i : c10::irange(params.size())) {  // 使用 c10::irange 迭代 params.size() 次数
    ASSERT_TRUE(torch::all(torch::eq(list->at(i), params[i])).item<bool>());  // 断言 list 中第 i 个元素是否与 params 中的第 i 个元素相等
  }

  for (const auto i : c10::irange(params.size())) {  // 使用 c10::irange 迭代 params.size() 次数
    ASSERT_TRUE(torch::all(torch::eq(list[i], params[i])).item<bool>());

# 断言：验证 `list[i]` 和 `params[i]` 中的所有元素是否相等，并返回布尔值
# 如果不相等，将引发断言错误。

  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->at(params.size() + 100), "Index out of range");

# 断言：验证对于超出范围的索引，使用 `list` 指针的 `at` 方法引发异常
# 错误消息会包含 "Index out of range"。

  ASSERT_THROWS_WITH(list->at(params.size() + 1), "Index out of range");

# 断言：验证对于超出范围的索引，使用 `list` 指针的 `at` 方法引发异常
# 错误消息会包含 "Index out of range"。

  ASSERT_THROWS_WITH(list[params.size() + 1], "Index out of range");

# 断言：验证对于超出范围的索引，直接对 `list` 进行索引引发异常
# 错误消息会包含 "Index out of range"。
TEST_F(ParameterListTest, ExtendPushesParametersFromOtherParameterList) {
  // 创建张量 ta, tb, tc, td, te, tf，并设置是否需要梯度
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  torch::Tensor te = torch::randn({1, 2});
  torch::Tensor tf = torch::randn({1, 2, 3});

  // 创建 ParameterList a 和 b，分别包含 ta, tb 和 tc, td
  ParameterList a(ta, tb);
  ParameterList b(tc, td);

  // 将 b 中的参数扩展到 a 中
  a->extend(*b);

  // 断言扩展后 a 的大小为 4
  ASSERT_EQ(a->size(), 4);
  // 断言 a 的第一个参数等于 ta
  ASSERT_TRUE(torch::all(torch::eq(a[0], ta)).item<bool>());
  // 断言 a 的第二个参数等于 tb
  ASSERT_TRUE(torch::all(torch::eq(a[1], tb)).item<bool>());
  // 断言 a 的第三个参数等于 tc
  ASSERT_TRUE(torch::all(torch::eq(a[2], tc)).item<bool>());
  // 断言 a 的第四个参数等于 td
  ASSERT_TRUE(torch::all(torch::eq(a[3], td)).item<bool>());

  // 断言 b 的大小未改变，仍为 2
  ASSERT_EQ(b->size(), 2);
  // 断言 b 的第一个参数等于 tc
  ASSERT_TRUE(torch::all(torch::eq(b[0], tc)).item<bool>());
  // 断言 b 的第二个参数等于 td
  ASSERT_TRUE(torch::all(torch::eq(b[1], td)).item<bool>());

  // 创建张量数组 c，包含 te 和 tf
  std::vector<torch::Tensor> c = {te, tf};
  // 将 c 中的参数扩展到 b 中
  b->extend(c);

  // 断言扩展后 b 的大小为 4
  ASSERT_EQ(b->size(), 4);
  // 断言 b 的第一个参数等于 tc
  ASSERT_TRUE(torch::all(torch::eq(b[0], tc)).item<bool>());
  // 断言 b 的第二个参数等于 td
  ASSERT_TRUE(torch::all(torch::eq(b[1], td)).item<bool>());
  // 断言 b 的第三个参数等于 te
  ASSERT_TRUE(torch::all(torch::eq(b[2], te)).item<bool>());
  // 断言 b 的第四个参数等于 tf
  ASSERT_TRUE(torch::all(torch::eq(b[3], tf)).item<bool>());
}

TEST_F(ParameterListTest, PrettyPrintParameterList) {
  // 创建张量 ta, tb, tc，并设置是否需要梯度
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  // 创建 ParameterList list 包含 ta, tb, tc
  ParameterList list(ta, tb, tc);
  // 断言 ParameterList list 的字符串表示
  ASSERT_EQ(
      c10::str(list),
      "torch::nn::ParameterList(\n"
      "(0): Parameter containing: [Float of size [1, 2]]\n"
      "(1): Parameter containing: [Float of size [1, 2]]\n"
      "(2): Parameter containing: [Float of size [1, 2]]\n"
      ")");
}

TEST_F(ParameterListTest, IncrementAdd) {
  // 创建张量 ta, tb, tc, td, te, tf，并设置是否需要梯度
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  torch::Tensor td = torch::randn({1, 2, 3});
  torch::Tensor te = torch::randn({1, 2});
  torch::Tensor tf = torch::randn({1, 2, 3});

  // 创建两个 ParameterList listA 和 listB，分别包含 ta, tb, tc 和 td, te, tf
  ParameterList listA(ta, tb, tc);
  ParameterList listB(td, te, tf);

  // 创建张量数组 tensors，包含 ta, tb, tc, td, te, tf
  std::vector<torch::Tensor> tensors{ta, tb, tc, td, te, tf};
  int idx = 0;

  // 将 listB 中的参数增加到 listA 中
  *listA += *listB;

  // 断言 listA 中的每个参数与对应的张量相等
  ASSERT_TRUE(torch::all(torch::eq(listA[0], ta)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[1], tb)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[2], tc)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[3], td)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[4], te)).item<bool>());
  ASSERT_TRUE(torch::all(torch::eq(listA[5], tf)).item<bool>());

  // 遍历 listA 的所有命名参数，并断言它们与 tensors 中的张量相等
  for (const auto& P : listA->named_parameters(false))
    ASSERT_TRUE(torch::all(torch::eq(P.value(), tensors[idx++])).item<bool>());

  // 断言 idx 的值为 6，即 tensors 的大小
  ASSERT_EQ(idx, 6);
}
```