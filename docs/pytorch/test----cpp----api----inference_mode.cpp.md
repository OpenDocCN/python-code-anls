# `.\pytorch\test\cpp\api\inference_mode.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件
#include <test/cpp/api/support.h>  // 包含测试支持函数的头文件
#include <torch/script.h>  // 包含 PyTorch 的脚本模式头文件

using namespace torch::autograd;  // 使用 PyTorch 的自动求导命名空间
using namespace torch::test;  // 使用 PyTorch 测试命名空间

namespace {
torch::Tensor functional_op(torch::Tensor& x) {  // 定义一个函数 functional_op，接受一个张量并返回其平方的张量
  return x * x;
}

void inplace_op(torch::Tensor& x) {  // 定义一个函数 inplace_op，对输入张量执行就地乘1的操作
  x.mul_(1);
}

torch::Tensor view_op(torch::Tensor& x) {  // 定义一个函数 view_op，返回输入张量的形状为{2, 3}的视图张量
  return x.view({2, 3});
}

/*
  只有以下 Autograd & ADInplaceOrView 键的张量组合是有效的：
    - Autograd=true, ADInplaceOrView=true (普通张量)
    - Autograd=false, ADInplaceOrView=false (推断张量)
  在推断模式下创建的张量大多数是推断张量。唯一的例外是在推断模式下创建的普通张量的视图仍然产生普通张量。
*/
void assert_TLS_states(bool inference_mode) {
  ASSERT_EQ(InferenceMode::is_enabled(), inference_mode);  // 断言推断模式是否启用
  ASSERT_FALSE(c10::impl::tls_is_dispatch_key_excluded(
      c10::DispatchKey::ADInplaceOrView));  // 断言 ADInplaceOrView 分发键是否排除
  ASSERT_FALSE(c10::impl::tls_is_dispatch_keyset_included(
      c10::autograd_dispatch_keyset));  // 断言是否包含 autograd_dispatch_keyset
  ASSERT_EQ(
      c10::impl::tls_is_dispatch_keyset_excluded(c10::autograd_dispatch_keyset),
      inference_mode);  // 断言是否排除 autograd_dispatch_keyset
  ASSERT_EQ(
      c10::impl::tls_is_dispatch_key_included(
          c10::DispatchKey::ADInplaceOrView),
      !inference_mode);  // 断言是否包含 ADInplaceOrView 分发键
  ASSERT_EQ(GradMode::is_enabled(), !inference_mode);  // 断言梯度模式是否启用
}
} // namespace

TEST(InferenceModeTest, TestTLSState) {
  assert_TLS_states(false);  // 测试非推断模式下的 TLS 状态
  {
    InferenceMode guard;  // 进入推断模式的保护区块
    assert_TLS_states(true);  // 测试推断模式下的 TLS 状态
    {
      InferenceMode guard(false);  // 退出推断模式的保护区块
      assert_TLS_states(false);  // 再次测试非推断模式下的 TLS 状态
    }
    assert_TLS_states(true);  // 再次测试推断模式下的 TLS 状态
  }
  assert_TLS_states(false);  // 最终确认退出推断模式后的 TLS 状态
}

TEST(InferenceModeTest, TestInferenceTensorCreation) {
  {
    InferenceMode guard;  // 进入推断模式的保护区块
    // 通过构造函数创建的新张量是推断张量。
    torch::Tensor c = torch::ones({1, 2, 3});
    ASSERT_FALSE(c.requires_grad());  // 断言张量不需要梯度
    ASSERT_TRUE(c.is_inference());  // 断言张量是推断张量

    // 在推断模式下，requires_grad 不会改变推断张量的行为。
    torch::Tensor tmp = torch::ones({1, 2, 3}).set_requires_grad(true);
    ASSERT_TRUE(tmp.requires_grad());  // 断言张量需要梯度
    ASSERT_TRUE(tmp.is_inference());  // 断言张量是推断张量

    tmp = torch::ones({1, 2, 3}).set_requires_grad(false);
    ASSERT_FALSE(tmp.requires_grad());  // 断言张量不需要梯度
    ASSERT_TRUE(tmp.is_inference());  // 断言张量是推断张量
  }
}

TEST(InferenceModeTest, TestExistingAutogradSession) {
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor a = s.clone();

  // 将 `a` 保存在现有的自动求导会话中
  torch::Tensor out = a * a;
  {
    InferenceMode guard;  // 进入推断模式的保护区块
    inplace_op(a);  // 对 `a` 执行就地操作
  }
  // 执行反向传播应该触发错误，因为 `a` 的版本已被修改。
  ASSERT_THROWS_WITH(
      out.backward(torch::ones_like(out)),
      "one of the variables needed for gradient computation has been modified by an inplace operation");
}

TEST(InferenceModeTest, TestInferenceTensorInInferenceModeFunctionalOp) {
  c10::InferenceMode guard;  // 进入推断模式的保护区块
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全 1 张量，并设置是否需要梯度根据 requires_grad 参数
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    // 将张量 c 作为参数传递给 functional_op 函数，执行功能操作，并返回结果张量 func_out
    torch::Tensor func_out = functional_op(c); // 经过内核处理：CPU

    // 断言 func_out 是推断模式（不需要梯度）
    ASSERT_TRUE(func_out.is_inference());

    // 断言 func_out 不需要梯度
    ASSERT_FALSE(func_out.requires_grad());
}
}

// 定义测试案例：在推断模式下测试推断模式中的原位操作
TEST(InferenceModeTest, TestInferenceTensorInInferenceModeInplaceOp) {
  // 进入推断模式
  c10::InferenceMode guard;
  // 遍历是否需要梯度的布尔值
  for (bool requires_grad : {true, false}) {
    // 创建张量 c，并设置是否需要梯度
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    // 执行原位操作 inplace_op(c)
    inplace_op(c); // 通过内核处理：CPU
    // 断言 c 处于推断模式
    ASSERT_TRUE(c.is_inference());
    // 断言 c 的梯度需求与 requires_grad 一致
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

// 定义测试案例：在推断模式下测试推断模式中的视图操作
TEST(InferenceModeTest, TestInferenceTensorInInferenceModeViewOp) {
  // 进入推断模式
  c10::InferenceMode guard;
  // 遍历是否需要梯度的布尔值
  for (bool requires_grad : {true, false}) {
    // 创建张量 c，并设置是否需要梯度
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    // 执行视图操作 view_op(c)，并将结果保存在 view_out 中
    torch::Tensor view_out = view_op(c); // 通过内核处理：CPU
    // 断言 view_out 处于推断模式
    ASSERT_TRUE(view_out.is_inference());
    // 断言 view_out 不需要梯度
    // 注意这与 NoGradMode 不同，但是在这种情况下是合理的
    ASSERT_FALSE(view_out.requires_grad());
    // 断言 view_out 不是视图
    ASSERT_FALSE(view_out.is_view());
  }
}

// 定义测试案例：在正常模式下测试推断张量的功能操作
TEST(InferenceModeTest, TestInferenceTensorInNormalModeFunctionalOp) {
  // 声明推断张量
  torch::Tensor inference_tensor;
  // 遍历是否需要梯度的布尔值
  for (bool requires_grad : {true, false}) {
    {
      // 进入推断模式
      InferenceMode guard;
      // 创建张量 inference_tensor，并设置是否需要梯度
      inference_tensor =
          torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }

    // 执行功能操作 functional_op(inference_tensor)
    torch::Tensor tmp =
        functional_op(inference_tensor); // 通过内核处理：ADInplaceOrView(fallthrough), CPU
    // 断言 tmp 不处于推断模式
    ASSERT_FALSE(tmp.is_inference());
    // 断言 tmp 不需要梯度
    ASSERT_FALSE(tmp.requires_grad());
  }
}

// 定义测试案例：在正常模式下测试推断张量的原位操作
TEST(InferenceModeTest, TestInferenceTensorInNormalModeInplaceOp) {
  // 声明推断张量
  torch::Tensor inference_tensor;
  // 遍历是否需要梯度的布尔值
  for (bool requires_grad : {true, false}) {
    {
      // 进入推断模式
      InferenceMode guard;
      // 创建张量 inference_tensor，并设置是否需要梯度
      inference_tensor =
          torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }
    // 断言调用原位操作 inplace_op(inference_tensor) 会抛出异常
    ASSERT_THROWS_WITH(
        inplace_op(
            inference_tensor), // 通过内核处理：ADInplaceOrView, CPU
        "Inplace update to inference tensor outside InferenceMode is not allowed");
  }
}

// 定义测试案例：在正常模式下测试推断张量的视图操作
TEST(InferenceModeTest, TestInferenceTensorInNormalModeViewOp) {
  // 声明推断张量
  torch::Tensor inference_tensor;
  // 遍历是否需要梯度的布尔值
  for (bool requires_grad : {true, false}) {
    {
      // 进入推断模式
      InferenceMode guard;
      // 创建张量 inference_tensor，并设置是否需要梯度
      inference_tensor =
          torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }
    // 执行视图操作 view_op(inference_tensor)
    torch::Tensor out =
        view_op(inference_tensor); // 通过内核处理：ADInplaceOrView, CPU
    // 断言 out 处于推断模式
    ASSERT_TRUE(out.is_inference());
    // 断言 out 不需要梯度
    ASSERT_FALSE(out.requires_grad());
    // 断言 out 不是视图
    ASSERT_FALSE(out.is_view());
    // 断言 out 是叶子张量
    ASSERT_TRUE(out.is_leaf());
  }
}

// 定义测试案例：在推断模式下测试正常张量的原位输出
TEST(InferenceModeTest, TestNormalTensorInplaceOutputInInferenceMode) {
  // 遍历是否需要梯度的布尔值
  for (bool requires_grad : {true, false}) {
    // 创建张量 s，并设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 克隆张量 s 到 a
    torch::Tensor a = s.clone();


这些注释根据代码的每行功能添加了适当的解释，保留了原有的缩进和结构。
    {
      // 进入 C10 引擎的推断模式保护区域
      c10::InferenceMode guard;
    
      // 对张量 a 执行原地操作，通过 ADInplaceOrView 和 CPU 内核处理
      inplace_op(a);
      // 断言：a 不处于推断模式
      ASSERT_FALSE(a.is_inference());
      // 断言：a 的梯度需求与指定的需求一致
      ASSERT_EQ(a.requires_grad(), requires_grad);
    
      // 再次对张量 a 执行原地操作，通过 ADInplaceOrView 和 CPU 内核处理
      inplace_op(a);
      // 断言：a 不处于推断模式
      ASSERT_FALSE(a.is_inference());
      // 断言：a 的梯度需求与指定的需求一致
      ASSERT_EQ(a.requires_grad(), requires_grad);
    
      // 将张量 a 视图化，返回视图的输出张量
      torch::Tensor view_out = view_op(a);
      // 断言：视图输出不处于推断模式
      ASSERT_FALSE(view_out.is_inference());
      // 断言：视图输出的梯度需求与指定的需求一致
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
    }
}

TEST(InferenceModeTest, TestNormalTensorInplaceOutputInNormalMode) {
  // 循环测试是否需要梯度
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的张量，并设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 克隆张量 s 到张量 a
    torch::Tensor a = s.clone();

    {
      // 进入推理模式保护区域
      c10::InferenceMode guard;

      // 对张量 a 进行原地操作
      inplace_op(a); // 经过内核：ADInplaceOrView, CPU
      // 断言张量 a 不处于推理模式
      ASSERT_FALSE(a.is_inference());
      // 断言张量 a 的梯度要求与原始设置一致
      ASSERT_EQ(a.requires_grad(), requires_grad);
    }

    // 对张量 a 进行函数式操作
    torch::Tensor tmp = functional_op(a); // 经过内核：VariableType, ADInplaceOrView(fallthrough), CPU
    // 断言临时张量 tmp 不处于推理模式
    ASSERT_FALSE(tmp.is_inference());
    // 断言临时张量 tmp 的梯度要求与原始设置一致
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    // 再次对张量 a 进行原地操作
    inplace_op(a); // 经过内核：VariableType, ADInplaceOrView, CPU
    // 断言张量 a 不处于推理模式
    ASSERT_FALSE(a.is_inference());
    // 断言张量 a 的梯度要求与原始设置一致
    ASSERT_EQ(a.requires_grad(), requires_grad);

    // 对张量 a 进行视图操作
    tmp = view_op(a); // 经过内核：VariableType, ADInplaceOrView, CPU
    // 断言临时张量 tmp 不处于推理模式
    ASSERT_FALSE(tmp.is_inference());
    // 断言临时张量 tmp 的梯度要求与原始设置一致
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestNormalTensorViewOutputInInferenceMode) {
  // 循环测试是否需要梯度
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的张量，并设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 克隆张量 s 到张量 a
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      // 进入推理模式保护区域
      c10::InferenceMode guard;

      // 视图操作在正常张量上产生正常张量作为输出
      // - 对于视图操作，由于我们在 alias_with_sizes_and_strides 中创建视图张量的方式，
      //   它们既有 Autograd 也有 ADInplaceOrView 键。但它们仍然是特殊的，
      //   因为它们将具有 CreationMeta::INFERENCE_MODE。换句话说，它们的行为与在
      //   no_grad 模式下创建的视图张量完全相同。

      // 对张量 a 进行视图操作
      view_out = view_op(a); // 经过内核：ADInplaceOrView, CPU
      // 断言视图输出张量 view_out 不处于推理模式
      ASSERT_FALSE(view_out.is_inference());
      // 检查张量创建元数据，应为 CreationMeta::INFERENCE_MODE
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      // 断言视图输出张量 view_out 的梯度要求与原始设置一致
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      // 断言视图输出张量 view_out 是叶节点
      ASSERT_TRUE(view_out.is_leaf());

      // 视图 -> 视图
      // 对视图输出张量 view_out 进行视图操作
      tmp = view_op(view_out); // 经过内核：ADInplaceOrView, CPU
      // 断言临时张量 tmp 不处于推理模式
      ASSERT_FALSE(tmp.is_inference());
      // 检查张量创建元数据，应为 CreationMeta::INFERENCE_MODE
      assert_tensor_creation_meta(tmp, CreationMeta::INFERENCE_MODE);
      // 断言临时张量 tmp 的梯度要求与原始设置一致
      ASSERT_EQ(tmp.requires_grad(), requires_grad);
      // 断言临时张量 tmp 是叶节点
      ASSERT_TRUE(tmp.is_leaf());

      // 视图 -> 视图 -> 原地操作
      // 对临时张量 tmp 进行原地操作
      inplace_op(tmp); // 经过内核：ADInplaceOrView, CPU
      // 检查张量创建元数据，应为 CreationMeta::INFERENCE_MODE
      assert_tensor_creation_meta(tmp, CreationMeta::INFERENCE_MODE);
      // 断言临时张量 tmp 不处于推理模式
      ASSERT_FALSE(tmp.is_inference());
      // 断言临时张量 tmp 的梯度要求与原始设置一致
      ASSERT_EQ(tmp.requires_grad(), requires_grad);
      // 断言临时张量 tmp 是叶节点
      ASSERT_TRUE(tmp.is_leaf());
      // 断言张量 a 和 tmp 的版本号相同
      ASSERT_EQ(a._version(), tmp._version());
    }
  }
}
TEST(InferenceModeTest, TestNormalTensorViewOutputInNormalMode) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全为1的张量，并根据 requires_grad 参数设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 克隆张量 s，生成张量 a
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      // 进入推理模式的上下文
      c10::InferenceMode guard;
      // 对张量 a 执行视图操作，生成 view_out，通过 ADInplaceOrView 和 CPU 内核
      view_out = view_op(a);
      // 断言 view_out 不处于推理模式
      ASSERT_FALSE(view_out.is_inference());
      // 断言 view_out 的创建元信息为 INFERENCE_MODE
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      // 断言 view_out 的梯度属性与 requires_grad 参数一致
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      // 断言 view_out 是叶子节点
      ASSERT_TRUE(view_out.is_leaf());
    }

    // 对 view_out 执行功能性操作，结果存储在 tmp 中
    tmp = functional_op(view_out);
    // 断言 view_out 不处于推理模式
    ASSERT_FALSE(view_out.is_inference());
    // 断言 tmp 的梯度属性与 requires_grad 参数一致
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    if (requires_grad) {
      // 对 view_out 执行原地操作 inplace_op，通过 VariableType、ADInplaceOrView 和 CPU 内核
      ASSERT_THROWS_WITH(
          inplace_op(view_out),
          "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out);
    }

    // 对 view_out 执行视图操作，结果存储在 tmp 中
    tmp = view_op(view_out);
    // 断言 view_out 不处于推理模式
    ASSERT_FALSE(view_out.is_inference());
    // 断言 tmp 的梯度属性与 requires_grad 参数一致
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorFunctionalOp) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全为1的张量，并根据 requires_grad 参数设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      // 进入推理模式的上下文
      InferenceMode guard;
      // 创建一个形状为 {1, 2, 3} 的全为1的张量 c，并根据 requires_grad 参数设置是否需要梯度
      c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }

    // 使用张量 c 和 s 执行加法操作，结果存储在 out 中，通过 VariableType、ADInplaceOrView(fallthrough) 和 CPU 内核
    torch::Tensor out = c.add(s);
    // 断言 out 不处于推理模式
    ASSERT_FALSE(out.is_inference());
    // 断言 out 的梯度属性与 requires_grad 参数一致
    ASSERT_EQ(out.requires_grad(), requires_grad);

    if (requires_grad) {
      // 对 out 执行反向传播操作，使用与 out 同形状的全为1的张量
      out.backward(torch::ones_like(out));
      // 断言 c 的梯度等于与 c 同形状的全为1的张量
      assert_tensor_equal(c.grad(), torch::ones_like(c));
    }

    if (requires_grad) {
      // 对推理张量 c 和 s 执行乘法操作，抛出异常 "Inference tensors cannot be saved for backward."
      ASSERT_THROWS_WITH(
          c.mul(s),
          "Inference tensors cannot be saved for backward.");

      // 推理张量作为 TensorList 输入，执行堆叠操作，抛出异常 "Inference tensors cannot be saved for backward."
      /*
      std::vector<torch::Tensor> inputs = {s, c};
      ASSERT_THROWS_WITH(
          torch::stack(inputs),
          "Inference tensors cannot be saved for backward.")
      */
    }
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorInplaceOp) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全为1的张量，并根据 requires_grad 参数设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

TEST(InferenceModeTest, TestNormalTensorViewOutputInNormalMode) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全为1的张量，并根据 requires_grad 参数设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 克隆张量 s，生成张量 a
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      // 进入推理模式的上下文
      c10::InferenceMode guard;
      // 对张量 a 执行视图操作，生成 view_out，通过 ADInplaceOrView 和 CPU 内核
      view_out = view_op(a);
      // 断言 view_out 不处于推理模式
      ASSERT_FALSE(view_out.is_inference());
      // 断言 view_out 的创建元信息为 INFERENCE_MODE
      assert_tensor_creation_meta(view_out, CreationMeta::INFERENCE_MODE);
      // 断言 view_out 的梯度属性与 requires_grad 参数一致
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      // 断言 view_out 是叶子节点
      ASSERT_TRUE(view_out.is_leaf());
    }

    // 对 view_out 执行功能性操作，结果存储在 tmp 中
    tmp = functional_op(view_out);
    // 断言 view_out 不处于推理模式
    ASSERT_FALSE(view_out.is_inference());
    // 断言 tmp 的梯度属性与 requires_grad 参数一致
    ASSERT_EQ(tmp.requires_grad(), requires_grad);

    if (requires_grad) {
      // 对 view_out 执行原地操作 inplace_op，通过 VariableType、ADInplaceOrView 和 CPU 内核
      ASSERT_THROWS_WITH(
          inplace_op(view_out),
          "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out);
    }

    // 对 view_out 执行视图操作，结果存储在 tmp 中
    tmp = view_op(view_out);
    // 断言 view_out 不处于推理模式
    ASSERT_FALSE(view_out.is_inference());
    // 断言 tmp 的梯度属性与 requires_grad 参数一致
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorFunctionalOp) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全为1的张量，并根据 requires_grad 参数设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      // 进入推理模式的上下文
      InferenceMode guard;
      // 创建一个形状为 {1, 2, 3} 的全为1的张量 c，并根据 requires_grad 参数设置是否需要梯度
      c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    }

    // 使用张量 c 和 s 执行加法操作，结果存储在 out 中，通过 VariableType、ADInplaceOrView(fallthrough) 和 CPU 内核
    torch::Tensor out = c.add(s);
    // 断言 out 不处于推理模式
    ASSERT_FALSE(out.is_inference());
    // 断言 out 的梯度属性与 requires_grad 参数一致
    ASSERT_EQ(out.requires_grad(), requires_grad);

    if (requires_grad) {
      // 对 out 执行反向传播操作，使用与 out 同形状的全为1的张量
      out.backward(torch::ones_like(out));
      // 断言 c 的梯度等于与 c 同形状的全为1的张量
      assert_tensor_equal(c.grad(), torch::ones_like(c));
    }

    if (requires_grad) {
      // 对推理张量 c 和 s 执行乘法操作，抛出异常 "Inference tensors cannot be saved for backward."
      ASSERT_THROWS_WITH(
          c.mul(s),
          "Inference tensors cannot be saved for backward.");

      // 推理张量作为 TensorList 输入，执行堆叠操作，抛出异常 "Inference tensors cannot be saved for backward."
      /*
      std::vector<torch::Tensor> inputs = {s, c};
      ASSERT_THROWS_WITH(
          torch::stack(inputs),
          "Inference tensors cannot be saved for backward.")
      */
    }
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorInplaceOp) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全为1的张量，并根据 requires_grad 参数设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 使用张量 s 创建张量 a 的副本
    torch::Tensor a = s.clone();
    // 声明张量 c
    torch::Tensor c;
    {
      // 进入推断模式的作用域
      InferenceMode guard;
      // 初始化张量 c 为全1的张量，形状为 {1, 2, 3}
      c = torch::ones({1, 2, 3});
    }

    // 如果需要梯度计算
    if (requires_grad) {
      // 断言操作，验证在推断模式下无法保存反向传播所需的张量
      ASSERT_THROWS_WITH(
          a.mul_(c), // 经过核心处理: VariableType(ERROR!), InferenceMode, CPU
          "Inference tensors cannot be saved for backward.");

      // 断言操作，验证 torch::mul_out 函数不支持自动微分的输出参数
      ASSERT_THROWS_WITH(
          torch::mul_out(
              /*out=*/c, s, s), // 经过核心处理: VariableType(ERROR!), ADInplaceOrView, CPU
          "out=... arguments don't support automatic differentiation, but one of the arguments requires grad")
    } else {
      // 在不需要梯度计算时，对张量 a 执行就地乘法操作
      a.mul_(c);

      // 断言操作，验证在推断模式外部不允许对推断张量进行就地更新
      ASSERT_THROWS_WITH(
          torch::mul_out(/*out=*/c, s, s), // 经过核心处理: VariableType, ADInplaceOrView(ERROR!), CPU
          "Inplace update to inference tensor outside InferenceMode is not allowed");
    }
  }
}

TEST(InferenceModeTest, TestMixInferenceAndNormalTensorViewOp) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全一张量，并根据 requires_grad 设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor c;
    {
      // 进入推断模式
      InferenceMode guard;
      // 创建一个形状为 {1, 2, 3} 的全一张量 c
      c = torch::ones({1, 2, 3});
    }

    // 使用 view_as 方法进行视图操作，其内部调用了 view() 方法，只接受一个张量作为参数
    // 因此不会出现混合推断张量和普通张量输入的情况
    torch::Tensor tmp1 =
        c.view_as(s); // 经过的内核: ADInplaceOrView, CPU
    ASSERT_TRUE(tmp1.is_inference()); // 断言 tmp1 是推断模式张量
    ASSERT_FALSE(tmp1.requires_grad()); // 断言 tmp1 不需要梯度

    // 这里的操作相当于 s.view(c.sizes())，不涉及混合输入场景
    torch::Tensor tmp2 =
        s.view_as(c); // 经过的内核: VariableType, ADInplaceOrView, CPU
    ASSERT_FALSE(tmp2.is_inference()); // 断言 tmp2 不是推断模式张量
    ASSERT_EQ(tmp2.requires_grad(), requires_grad); // 断言 tmp2 的梯度需求与 requires_grad 相符
  }
}

TEST(InferenceModeTest, TestHandleDirectViewOnRebase) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全一张量，并根据 requires_grad 设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out;
    {
      // 进入推断模式
      InferenceMode guard;
      view_out = view_op(a); // 经过的内核: ADInplaceOrView, CPU
    }
    if (requires_grad) {
      // 如果需要梯度，则断言在推断模式下创建视图并直接就地修改，会抛出异常
      ASSERT_THROWS_WITH(
          inplace_op(view_out),
          "A view was created in inference mode and is being modified inplace")
    } else {
      inplace_op(view_out); // 否则直接就地修改视图
    }
  }
}

TEST(InferenceModeTest, TestHandleInDirectViewOnRebase) {
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的全一张量，并根据 requires_grad 设置是否需要梯度
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out;
    {
      // 进入推断模式
      InferenceMode guard;
      view_out = view_op(a); // 经过的内核: ADInplaceOrView, CPU
    }
    inplace_op(a); // 在推断模式下直接就地修改 a
    if (requires_grad) {
      // 如果需要梯度，则断言在推断模式下创建视图并且其基础张量或其视图的基础张量已被就地修改，会抛出异常
      ASSERT_THROWS_WITH(
          view_out.grad_fn(),
          "A view was created in inference mode and its base or another view of its base has been modified inplace");
    } else {
      view_out.grad_fn(); // 否则访问视图的梯度函数
    }
  }
}

TEST(InferenceModeTest, TestCreationMetaPropagation) {
  // 创建一个形状为 {1, 2, 3} 的全一张量，并设置需要梯度
  torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(true);
  torch::Tensor b, c;
  {
    // 进入推断模式
    InferenceMode guard;
    b = s.view_as(s); // 创建一个 s 的视图 b
  }
  // 断言在推断模式下创建视图并就地修改，会抛出异常
  ASSERT_THROWS_WITH(
      b.add_(1),
      "A view was created in inference mode and is being modified inplace");
  {
    // 离开推断模式，进入自动梯度模式
    AutoGradMode mode(false);
    c = b.view_as(b); // 创建一个 b 的视图 c
  }
  // 断言在推断模式下创建视图并就地修改，会抛出异常
  ASSERT_THROWS_WITH(
      c.add_(1),
      "A view was created in inference mode and is being modified inplace");
}

TEST(InferenceModeTest, TestCreationMetaPropagationInput) {
  // 创建一个形状为 {2, 2, 3} 的全一张量，并设置需要梯度
  torch::Tensor s = torch::ones({2, 2, 3}).set_requires_grad(true);
  auto s_view = s.view_as(s); // 创建 s 的视图 s_view
  std::vector<at::Tensor> b, c;
  {
    // 进入推断模式
    InferenceMode guard;
    b = s_view.split_with_sizes({1, 1}); // 将 s_view 按指定大小拆分成张量向量 b

    s = s.view_as(s); // 将 s 重新作为视图重新赋值给 s
    c = s.split_with_sizes({1, 1}); // 将 s 按指定大小拆分成张量向量 c
  }
  for (auto& b_el : b) {
    # 对张量的创建元信息进行断言，确保处于推断模式
    assert_tensor_creation_meta(b_el, CreationMeta::INFERENCE_MODE);
    # 断言在推断模式下创建的视图不应该进行原地修改，否则抛出异常并包含特定错误信息
    ASSERT_THROWS_WITH(
        b_el.add_(1),
        "A view was created in inference mode and is being modified inplace");
  }
  # 对张量集合 c 中的每个张量执行以下操作
  for (auto& c_el : c) {
    # 断言张量的创建元信息，确保处于推断模式
    assert_tensor_creation_meta(c_el, CreationMeta::INFERENCE_MODE);
    # 断言在推断模式下创建的视图不应该进行原地修改，否则抛出异常并包含特定错误信息
    ASSERT_THROWS_WITH(
        c_el.add_(1),
        "A view was created in inference mode and is being modified inplace");
  }
TEST(InferenceModeTest, TestInplaceCopyOnInferenceTensor) {
  // 对于每个 requires_grad 值进行测试
  for (bool requires_grad : {true, false}) {
    // 创建一个形状为 {1, 2, 3} 的张量 s，设置 requires_grad 属性
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    // 声明一个空张量 t
    torch::Tensor t;
    {
      // 进入推断模式
      InferenceMode guard;
      // 创建一个形状为 {1, 2, 3} 的张量 t
      t = torch::ones({1, 2, 3});
      // 将张量 s 的值复制给张量 t
      t.copy_(s);
      // 断言 t 已处于推断模式
      ASSERT_TRUE(t.is_inference());
      // 断言 t 不需要梯度计算
      ASSERT_FALSE(t.requires_grad());
    }

    // 断言尝试在推断模式外部对 t 进行 inplace 复制会抛出异常
    ASSERT_THROWS_WITH(
        t.copy_(s),
        "Inplace update to inference tensor outside InferenceMode is not allowed");
  }
}

TEST(InferenceModeTest, TestSetRequiresGradInNormalMode) {
  // 声明一个张量 t
  torch::Tensor t;
  {
    // 进入推断模式
    InferenceMode guard;
    // 创建一个形状为 {1, 2, 3} 的张量 t
    t = torch::ones({1, 2, 3});
  }
  // 在推断模式外部设置 t 的 requires_grad 属性为 false，并断言会抛出异常
  t.set_requires_grad(false);
  ASSERT_THROWS_WITH(
      t.set_requires_grad(true),
      "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
}

TEST(InferenceModeTest, TestAccessVersionCounter) {
  // 声明一个张量 t
  torch::Tensor t;
  {
    // 进入推断模式
    InferenceMode guard;
    // 创建一个形状为 {1, 2, 3} 的张量 t
    t = torch::ones({1, 2, 3});
    // 断言尝试访问推断张量的 version_counter 会抛出异常
    ASSERT_THROWS_WITH(
        t.unsafeGetTensorImpl()->version_counter().current_version(),
        "Inference tensors do not track version counter.");
    // 增加推断张量的版本号
    t.unsafeGetTensorImpl()->bump_version();
  }
  // 在推断模式外部再次断言尝试访问推断张量的 version_counter 会抛出异常
  ASSERT_THROWS_WITH(
      t.unsafeGetTensorImpl()->version_counter().current_version(),
      "Inference tensors do not track version counter.");
  // 在推断模式外部断言尝试 inplace 更新推断张量会抛出异常
  ASSERT_THROWS_WITH(
      t.unsafeGetTensorImpl()->bump_version(),
      "Inplace update to inference tensor outside InferenceMode is not allowed.");
  // 建议的解决方案：克隆张量 t 并测试版本号是否增加
  torch::Tensor c = t.clone();
  uint32_t v = c.unsafeGetTensorImpl()->version_counter().current_version();
  c.unsafeGetTensorImpl()->bump_version();
  ASSERT_EQ(
      c.unsafeGetTensorImpl()->version_counter().current_version(), v + 1);
}

TEST(InferenceModeTest, TestInplaceUpdateInferenceTensorWithNormalTensor) {
  // 创建一个形状为 {1, 2, 3} 的张量 s
  torch::Tensor s = torch::ones({1, 2, 3});
  // 声明一个张量 t
  torch::Tensor t;
  {
    // 进入推断模式
    InferenceMode guard;
    // 创建一个形状为 {1, 2, 3} 的张量 t
    t = torch::ones({1, 2, 3});
    // 将 t 的值 inplace 复制给 s
    s.copy_(t);
    // 将 t 的值 inplace 加到 s 上
    s.add_(t);
    // 将 s 的值 inplace 加到 t 上
    t.add_(s);
    // 将 s 的值 inplace 复制给 t
    t.copy_(s);
  }
  // 尝试在推断模式外部对 t 进行 inplace 复制，断言会抛出异常
  s.copy_(t);
  // 尝试在推断模式外部对 t 进行 inplace 加法，断言会抛出异常
  s.add_(t);
  ASSERT_THROWS_WITH(
      t.copy_(s),
      "Inplace update to inference tensor outside InferenceMode is not allowed");

  ASSERT_THROWS_WITH(
      t.add_(s),
      "Inplace update to inference tensor outside InferenceMode is not allowed");
}

TEST(InferenceModeTest, TestComplexViewInInferenceMode) {
  // 创建一个形状为 {3, 3, 2} 的张量 s
  torch::Tensor s = torch::ones({3, 3, 2});
  // 创建一个以复数形式视图的张量 t
  torch::Tensor t = torch::view_as_complex(s);
  {
    // 进入推断模式
    InferenceMode guard;
    // 声明一个临时张量 tmp
    torch::Tensor tmp;

    // 将 t 视图转换为实数形式，断言不处于推断模式
    tmp = torch::view_as_real(t);
    ASSERT_FALSE(tmp.is_inference());
    // 将 s 视图转换为复数形式，断言不处于推断模式
    tmp = torch::view_as_complex(s);
    ASSERT_FALSE(tmp.is_inference());

    // 创建一个形状为 {3, 3, 2} 的张量 e
    torch::Tensor e = torch::ones({3, 3, 2});
    // 将 e 视图转换为复数形式，断言处于推断模式
    tmp = torch::view_as_complex(e);
    ASSERT_TRUE(tmp.is_inference());
    // 将 tmp 视图转换为实数形式，断言处于推断模式
    tmp = torch::view_as_real(tmp);
    ASSERT_TRUE(tmp.is_inference());
  }
}

TEST(InferenceModeTest, TestComplexViewInNormalMode) {
  // 声明一个张量 s
  torch::Tensor s;
  {
    // 进入推断模式
    InferenceMode guard;
    // 创建一个形状为 {3, 3, 2} 的张量 s
    s = torch::ones({3, 3, 2});
  }
  // 在推断模式外部尝试设置 s 的 requires_grad 属性，断言会抛出异常
  s.set_requires_grad(false);
  ASSERT_THROWS_WITH(
      s.set_requires_grad(true),
      "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
}
    // 声明 InferenceMode 变量 guard，进入推断模式
    InferenceMode guard;
    // 创建一个形状为 {3, 3, 2} 的全一张量 s
    s = torch::ones({3, 3, 2});
  }
  // 将张量 s 转换为复数类型的张量 tmp
  torch::Tensor tmp = torch::view_as_complex(s);
  // 断言 tmp 是否处于推断模式
  ASSERT_TRUE(tmp.is_inference());
  // 将 tmp 转换为实数类型的张量
  tmp = torch::view_as_real(tmp);
  // 再次断言 tmp 是否处于推断模式
  ASSERT_TRUE(tmp.is_inference());
}

TEST(InferenceModeTest, TestCustomFunction) {
  // 定义自定义函数对象 MyFunction，继承自模板类 Function
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，计算并返回结果张量
    static Variable forward(
        AutogradContext* ctx,
        Variable var1,
        int mul,
        Variable var2) {
      // 在自动求导上下文中保存乘数 mul
      ctx->saved_data["mul"] = mul;
      // 保存需要在反向传播时使用的变量列表
      ctx->save_for_backward({var1, var2});
      // 计算并返回 var1 + mul * var2 + var1 * var2
      return var1 + mul * var2 + var1 * var2;
    }

    // 反向传播函数，计算并返回梯度张量列表
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 从保存的数据中获取乘数 mul
      int mul = ctx->saved_data["mul"].toInt();
      // 获取保存的变量列表
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];
      // 计算并返回梯度张量列表
      variable_list output = {
          grad_output[0] + grad_output[0] * var2,
          Variable(),  // 空变量
          grad_output[0] * mul + grad_output[0] * var1};
      return output;
    }
  };

  {
    // 进入推断模式保护区域
    InferenceMode guard;
    // 创建张量 var1，全为1，需要计算梯度
    torch::Tensor var1 = torch::ones({3, 3}).set_requires_grad(true);
    // 克隆 var1 得到 var2
    auto var2 = var1.clone();
    int mul = 2;
    // 如果推断模式没有自动设置 NoGradGuard，此处会在尝试保存 var1 和 var2 时报错
    auto y = MyFunction::apply(var1, mul, var2);
    // 预期结果张量
    torch::Tensor expected = var1 + mul * var2 + var1 * var2;
    // 断言 y 与预期结果 expected 相等
    assert_tensor_equal(y, expected);
  }
}

TEST(InferenceModeTest, TestLegacyAutoNonVariableTypeModeWarning) {
  // 总是发出警告
  c10::WarningUtils::WarnAlways warn_always(true);
  // 捕获警告消息
  WarningCapture warnings;
  // 进入自动非变量类型模式保护区域
  at::AutoNonVariableTypeMode guard;
  // 断言是否包含特定警告消息
  ASSERT_TRUE(
      warnings.str().find("AutoNonVariableTypeMode is deprecated") !=
      std::string::npos);
}
```