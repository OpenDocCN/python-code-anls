# `.\pytorch\test\cpp\api\grad_mode.cpp`

```
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件

#include <test/cpp/api/support.h> // 包含测试支持函数的头文件

#include <torch/script.h> // 包含 PyTorch 的脚本模块头文件

using namespace torch::autograd; // 使用 PyTorch 的自动求导命名空间
using namespace torch::test; // 使用 PyTorch 测试相关的命名空间

// 定义测试用例 GradModeTest，测试函数 TestRequiresGradFunctionalOp
TEST(GradModeTest, TestRequiresGradFunctionalOp) {
  torch::AutoGradMode mode(false); // 关闭自动求导模式

  for (bool requires_grad : {true, false}) { // 遍历是否需要梯度的布尔值列表
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad); // 创建一个张量 c，设置是否需要梯度

    torch::Tensor func_out = c * c; // 执行张量 c 的乘法运算，得到 func_out
    ASSERT_FALSE(func_out.requires_grad()); // 断言 func_out 不需要梯度
    ASSERT_TRUE(func_out.is_leaf()); // 断言 func_out 是叶子节点
  }
}

// 定义测试用例 GradModeTest，测试函数 TestRequiresGradInplaceOp
TEST(GradModeTest, TestRequiresGradInplaceOp) {
  torch::AutoGradMode mode(false); // 关闭自动求导模式

  for (bool requires_grad : {true, false}) { // 遍历是否需要梯度的布尔值列表
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad); // 创建一个张量 c，设置是否需要梯度

    c.mul_(2); // 原地乘法操作，将张量 c 中的每个元素乘以 2
    ASSERT_EQ(c.requires_grad(), requires_grad); // 断言张量 c 是否需要梯度与 requires_grad 相符
  }
}

// 定义测试用例 GradModeTest，测试函数 TestRequiresGradViewOp
TEST(GradModeTest, TestRequiresGradViewOp) {
  torch::AutoGradMode mode(false); // 关闭自动求导模式

  for (bool requires_grad : {true, false}) { // 遍历是否需要梯度的布尔值列表
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad); // 创建一个张量 c，设置是否需要梯度

    torch::Tensor view_out = c.view({2, 3}); // 创建一个视图张量 view_out，改变形状为 {2, 3}
    ASSERT_EQ(view_out.requires_grad(), requires_grad); // 断言 view_out 是否需要梯度与 requires_grad 相符
    ASSERT_TRUE(view_out.is_leaf()); // 断言 view_out 是叶子节点
  }
}

// 定义测试用例 GradModeTest，测试函数 TestRequiresGradViewOpExiting
TEST(GradModeTest, TestRequiresGradViewOpExiting) {
  for (bool requires_grad : {true, false}) { // 遍历是否需要梯度的布尔值列表
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad); // 创建一个张量 s，设置是否需要梯度

    torch::Tensor a = s.clone(); // 克隆张量 s 到 a
    torch::Tensor view_out, tmp; // 定义张量 view_out 和 tmp

    {
      torch::AutoGradMode mode(false); // 在作用域内关闭自动求导模式

      view_out = a.view({2, 3}); // 创建一个视图张量 view_out，改变形状为 {2, 3}
      assert_tensor_creation_meta(
          view_out, torch::autograd::CreationMeta::NO_GRAD_MODE); // 断言 view_out 的创建元信息为 NO_GRAD_MODE
      ASSERT_EQ(view_out.requires_grad(), requires_grad); // 断言 view_out 是否需要梯度与 requires_grad 相符
      ASSERT_TRUE(view_out.is_leaf()); // 断言 view_out 是叶子节点
    }

    tmp = view_out * view_out; // 对 view_out 执行乘法操作，结果赋给 tmp
    ASSERT_EQ(tmp.requires_grad(), requires_grad); // 断言 tmp 是否需要梯度与 requires_grad 相符

    if (requires_grad) {
      tmp.backward(torch::ones_like(tmp)); // 如果 requires_grad 为 true，则对 tmp 执行反向传播
      // TODO: this behavior is a side effect of issue #11390.
      ASSERT_FALSE(view_out.grad().defined()); // 断言 view_out 的梯度不存在
    }

    if (requires_grad) {
      ASSERT_THROWS_WITH(
          view_out.mul_(
              2), // 原地乘法操作，可能会抛出异常
          "A view was created in no_grad mode and is being modified inplace"); // 断言异常信息
    } else {
      view_out.mul_(2); // 原地乘法操作，修改 view_out
    }

    tmp = view_out.view({2, 3}); // 创建一个视图张量 tmp，改变形状为 {2, 3}
    ASSERT_EQ(tmp.requires_grad(), requires_grad); // 断言 tmp 是否需要梯度与 requires_grad 相符
    assert_tensor_creation_meta(
        tmp, torch::autograd::CreationMeta::NO_GRAD_MODE); // 断言 tmp 的创建元信息为 NO_GRAD_MODE
  }
}
```