# `.\pytorch\test\cpp\api\parameterdict.cpp`

```
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

// 参数字典测试结构体，继承自 SeedingFixture，用于测试参数字典功能
struct ParameterDictTest : torch::test::SeedingFixture {};

// 测试从张量构造参数字典的情况
TEST_F(ParameterDictTest, ConstructFromTensor) {
  // 创建空的参数字典
  ParameterDict dict;
  // 创建三个张量，其中 ta 和 tb 需要梯度，tc 不需要
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  // 断言 ta 需要梯度，tb 不需要梯度
  ASSERT_TRUE(ta.requires_grad());
  ASSERT_FALSE(tb.requires_grad());
  // 将张量插入参数字典中
  dict->insert("A", ta);
  dict->insert("B", tb);
  dict->insert("C", tc);
  // 断言参数字典的大小为 3
  ASSERT_EQ(dict->size(), 3);
  // 断言参数字典中的张量与原始张量相等，并且保持了梯度要求
  ASSERT_TRUE(torch::all(torch::eq(dict["A"], ta)).item<bool>());
  ASSERT_TRUE(dict["A"].requires_grad());
  ASSERT_TRUE(torch::all(torch::eq(dict["B"], tb)).item<bool>());
  ASSERT_FALSE(dict["B"].requires_grad());
}

// 测试从有序字典构造参数字典的情况
TEST_F(ParameterDictTest, ConstructFromOrderedDict) {
  // 创建三个张量，ta 和 tb 需要梯度，tc 不需要
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  // 使用有序字典创建参数字典
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"A", ta}, {"B", tb}, {"C", tc}};
  auto dict = torch::nn::ParameterDict(params);
  // 断言参数字典的大小为 3
  ASSERT_EQ(dict->size(), 3);
  // 断言参数字典中的张量与原始张量相等，并且保持了梯度要求
  ASSERT_TRUE(torch::all(torch::eq(dict["A"], ta)).item<bool>());
  ASSERT_TRUE(dict["A"].requires_grad());
  ASSERT_TRUE(torch::all(torch::eq(dict["B"], tb)).item<bool>());
  ASSERT_FALSE(dict["B"].requires_grad());
}

// 测试插入和包含操作
TEST_F(ParameterDictTest, InsertAndContains) {
  // 创建空的参数字典
  ParameterDict dict;
  // 向参数字典中插入张量
  dict->insert("A", torch::tensor({1.0}));
  // 断言参数字典的大小为 1，并且包含键 "A"，不包含键 "C"
  ASSERT_EQ(dict->size(), 1);
  ASSERT_TRUE(dict->contains("A"));
  ASSERT_FALSE(dict->contains("C"));
}

// 测试插入和清空操作
TEST_F(ParameterDictTest, InsertAndClear) {
  // 创建空的参数字典
  ParameterDict dict;
  // 向参数字典中插入张量
  dict->insert("A", torch::tensor({1.0}));
  // 断言参数字典的大小为 1
  ASSERT_EQ(dict->size(), 1);
  // 清空参数字典后，断言大小为 0
  dict->clear();
  ASSERT_EQ(dict->size(), 0);
}

// 测试插入和弹出操作
TEST_F(ParameterDictTest, InsertAndPop) {
  // 创建空的参数字典
  ParameterDict dict;
  // 向参数字典中插入张量
  dict->insert("A", torch::tensor({1.0}));
  // 断言参数字典的大小为 1
  ASSERT_EQ(dict->size(), 1);
  // 尝试弹出不存在的键 "B"，预期抛出异常
  ASSERT_THROWS_WITH(dict->pop("B"), "Parameter 'B' is not defined");
  // 弹出键 "A" 对应的张量，并断言其值与预期相等
  torch::Tensor p = dict->pop("A");
  ASSERT_EQ(dict->size(), 0);
  ASSERT_TRUE(torch::eq(p, torch::tensor({1.0})).item<bool>());
}

// 测试简单更新操作
TEST_F(ParameterDictTest, SimpleUpdate) {
  // 创建空的参数字典和两个其他参数字典
  ParameterDict dict;
  ParameterDict wrongDict;
  ParameterDict rightDict;
  // 向 dict 中插入三个张量
  dict->insert("A", torch::tensor({1.0}));
  dict->insert("B", torch::tensor({2.0}));
  dict->insert("C", torch::tensor({3.0}));
  // 向 wrongDict 中插入一个不在 dict 中的张量
  wrongDict->insert("A", torch::tensor({5.0}));
  wrongDict->insert("D", torch::tensor({5.0}));
  // 尝试使用 wrongDict 更新 dict，预期抛出异常
  ASSERT_THROWS_WITH(dict->update(*wrongDict), "Parameter 'D' is not defined");
  // 向 rightDict 中插入正确的更新项，并使用其更新 dict
  rightDict->insert("A", torch::tensor({5.0}));
  dict->update(*rightDict);
  // 断言更新后参数字典的大小为 3，并且更新正确
  ASSERT_EQ(dict->size(), 3);
  ASSERT_TRUE(torch::eq(dict["A"], torch::tensor({5.0})).item<bool>());
}


这段代码是用于测试参数字典类 `ParameterDict` 的各种功能，包括从张量或有序字典构造、插入、包含、清空、弹出和更新操作。
TEST_F(ParameterDictTest, Keys) {
  // 创建包含三个张量的有序字典，每个张量用字符串作为键
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0})},
      {"c", torch::tensor({1.0, 2.0})}};
  // 使用有序字典初始化 ParameterDict 对象
  auto dict = torch::nn::ParameterDict(params);
  // 获取字典中的所有键
  std::vector<std::string> keys = dict->keys();
  // 预期的键列表
  std::vector<std::string> true_keys{"a", "b", "c"};
  // 断言实际获取的键列表与预期相等
  ASSERT_EQ(keys, true_keys);
}

TEST_F(ParameterDictTest, Values) {
  // 创建三个张量
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  // 使用张量初始化有序字典
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", ta}, {"b", tb}, {"c", tc}};
  // 使用有序字典初始化 ParameterDict 对象
  auto dict = torch::nn::ParameterDict(params);
  // 获取字典中的所有值
  std::vector<torch::Tensor> values = dict->values();
  // 预期的值列表
  std::vector<torch::Tensor> true_values{ta, tb, tc};
  // 遍历比较每个值是否与预期相等
  for (auto i = 0U; i < values.size(); i += 1) {
    ASSERT_TRUE(torch::all(torch::eq(values[i], true_values[i])).item<bool>());
  }
}

TEST_F(ParameterDictTest, Get) {
  // 创建空的 ParameterDict 对象
  ParameterDict dict;
  // 创建三个张量
  torch::Tensor ta = torch::randn({1, 2}, torch::requires_grad(true));
  torch::Tensor tb = torch::randn({1, 2}, torch::requires_grad(false));
  torch::Tensor tc = torch::randn({1, 2});
  // 断言张量 ta 是需要梯度的
  ASSERT_TRUE(ta.requires_grad());
  // 断言张量 tb 不需要梯度
  ASSERT_FALSE(tb.requires_grad());
  // 将张量插入到 ParameterDict 中
  dict->insert("A", ta);
  dict->insert("B", tb);
  dict->insert("C", tc);
  // 断言 ParameterDict 的大小为 3
  ASSERT_EQ(dict->size(), 3);
  // 断言获取的张量与预期相等，并且需要梯度
  ASSERT_TRUE(torch::all(torch::eq(dict->get("A"), ta)).item<bool>());
  ASSERT_TRUE(dict->get("A").requires_grad());
  // 断言获取的张量与预期相等，并且不需要梯度
  ASSERT_TRUE(torch::all(torch::eq(dict->get("B"), tb)).item<bool>());
  ASSERT_FALSE(dict->get("B").requires_grad());
}

TEST_F(ParameterDictTest, PrettyPrintParameterDict) {
  // 创建包含四个张量的有序字典，每个张量用字符串作为键
  torch::OrderedDict<std::string, torch::Tensor> params = {
      {"a", torch::tensor({1.0})},
      {"b", torch::tensor({2.0, 1.0})},
      {"c", torch::tensor({{3.0}, {2.1}})},
      {"d", torch::tensor({{3.0, 1.3}, {1.2, 2.1}})}};
  // 使用有序字典初始化 ParameterDict 对象
  auto dict = torch::nn::ParameterDict(params);
  // 断言 ParameterDict 对象的字符串表示与预期相等
  ASSERT_EQ(
      c10::str(dict),
      "torch::nn::ParameterDict(\n"
      "(a): Parameter containing: [Float of size [1]]\n"
      "(b): Parameter containing: [Float of size [2]]\n"
      "(c): Parameter containing: [Float of size [2, 1]]\n"
      "(d): Parameter containing: [Float of size [2, 2]]\n"
      ")");
}
```