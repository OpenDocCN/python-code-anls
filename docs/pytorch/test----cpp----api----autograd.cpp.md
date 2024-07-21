# `.\pytorch\test\cpp\api\autograd.cpp`

```py
// 引入 ATen 核心库的测试辅助工具头文件
#include <ATen/core/boxing/impl/test_helpers.h>
// 引入 Google 测试框架头文件
#include <gtest/gtest.h>

// 引入 ATen 操作注册相关头文件
#include <ATen/core/op_registration/op_registration.h>
// 引入 PyTorch 核心头文件
#include <torch/torch.h>

// 引入 PyTorch 自动求导函数手动定义头文件
#include <torch/csrc/autograd/FunctionsManual.h>
// 引入 PyTorch 自动求导基础操作头文件
#include <torch/csrc/autograd/functions/basic_ops.h>

// 引入 C++ API 支持测试辅助函数头文件
#include <test/cpp/api/support.h>

// 使用 torch::autograd 命名空间
using namespace torch::autograd;
// 使用 torch::test 命名空间
using namespace torch::test;

// 定义宏，用于比较两个 Variable 是否接近相等
#define ASSERT_VARIABLE_EQ(a, b) ASSERT_TRUE(torch::allclose((a), (b)))
#define EXPECT_VARIABLE_EQ(a, b) EXPECT_TRUE(torch::allclose((a), (b)))

// 定义函数，返回节点的图描述字符串
std::string graph_desc(std::shared_ptr<Node> node) {
  // 若节点为空，则返回 "None"
  if (!node) {
    return "None";
  }
  // 从节点获取名称
  auto result = node->name() + "(";
  // 获取节点的所有后续边
  auto next_edges = node->next_edges();
  // 遍历所有后续边
  for (auto& edge : next_edges) {
    // 递归获取下一个节点的图描述，并添加到结果中
    result += graph_desc(edge.function);
  }
  // 返回结果字符串，包含节点名称及其后续边的描述
  return result + ")";
}

// 定义简单函数，进行 Variable 的简单数学运算
Variable simple_fn(const Variable& x, const Variable& y) {
  return x + 2 * y + x * y;
}

// 定义测试用例，测试注册 Hook 函数，返回类型为 void 的情况
TEST(AutogradAPITests, RegisterHookVoidReturnAcceptsUndefinedTensor) {
  // 创建大小为 {} 的 CPU 上全零 Tensor
  auto x = at::zeros({}, at::kCPU);
  // 设置 x 为需要梯度计算的 Tensor
  x.requires_grad_();
  // 注册 Hook 函数，接受 at::TensorBase 类型参数并返回空值
  x.register_hook([](at::TensorBase x) { return; });
  // 创建 UndefinedGrad 对象，对 x 执行操作并得到结果 y
  auto y = torch::autograd::UndefinedGrad().apply({x});
  // 对 y[0] 进行反向传播
  y[0].backward();
}

// 定义测试用例，测试注册 Hook 函数，返回类型为 Tensor 的情况
TEST(AutogradAPITests, RegisterHookTensorReturnAcceptsUndefinedTensor) {
  // 创建大小为 {} 的 CPU 上全零 Tensor
  auto x = at::zeros({}, at::kCPU);
  // 设置 x 为需要梯度计算的 Tensor
  x.requires_grad_();
  // 注册 Hook 函数，接受 at::Tensor 类型参数并返回该参数
  x.register_hook([](at::Tensor x) -> at::Tensor { return x; });
  // 创建 UndefinedGrad 对象，对 x 执行操作并得到结果 y
  auto y = torch::autograd::UndefinedGrad().apply({x});
  // 对 y[0] 进行反向传播
  y[0].backward();
}

// 定义测试用例，测试简单的反向传播功能
TEST(AutogradAPITests, BackwardSimpleTest) {
  // 创建大小为 {2, 2} 的需要梯度计算的随机 Tensor x 和 y
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  // 对 x 和 y 进行简单函数计算得到 res
  auto res = simple_fn(x, y);
  // 对 res.sum() 进行反向传播
  backward({res.sum()}, {});

  // 断言 x 的梯度应接近于 y + 全 1 矩阵
  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({2, 2}));
  // 断言 y 的梯度应接近于 x + 全 1 矩阵乘以 2
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({2, 2}) * 2);
}

// 定义测试用例，测试带参数的反向传播功能
TEST(AutogradAPITests, BackwardTest) {
  // 创建大小为 {2, 2} 的需要梯度计算的随机 Tensor x 和 y
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  // 对 x 和 y 进行简单函数计算得到 res
  auto res = simple_fn(x, y);
  // 使用权重为全 1 矩阵进行反向传播
  backward({res}, {torch::ones({2, 2})}, {}, true);

  // 断言 x 的梯度应为 2 倍的 y + 全 1 矩阵
  ASSERT_VARIABLE_EQ(x.grad(), 2 * (y + torch::ones({2, 2})));
  // 断言 y 的梯度应为 2 倍的 x + 全 1 矩阵乘以 2
  ASSERT_VARIABLE_EQ(y.grad(), 2 * (x + torch::ones({2, 2}) * 2));
}

// 定义测试用例，测试简单的梯度计算功能
TEST(AutogradAPITests, GradSimpleTest) {
  // 创建大小为 {2, 2} 的需要梯度计算的随机 Tensor x 和 y
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  // 对 x 和 y 进行简单函数计算得到 res
  auto res = simple_fn(x, y);
  // 计算 res 对 x 和 y 的梯度，并使用全 1 矩阵作为权重
  auto grad_res = grad({res}, {x, y}, {torch::ones({2, 2})});

  // 断言计算得到的 x 的梯度应接近于 y + 全 1 矩阵
  ASSERT_VARIABLE_EQ(grad_res[0], y + torch::ones({2, 2}));
  // 断言计算得到的 y 的梯度应接近于 x + 全 1 矩阵乘以 2
  ASSERT_VARIABLE_EQ(grad_res[1], x + torch::ones({2, 2}) * 2);
}
TEST(AutogradAPITests, GradTest) {
  // 创建具有梯度的随机变量 x 和 y
  Variable x = torch::randn({2, 2}, torch::requires_grad());
  Variable y = torch::randn({2, 2}, torch::requires_grad());

  // 调用 simple_fn 函数计算结果 res
  auto res = simple_fn(x, y);

  // 对 res 进行反向传播，使用全为 1 的梯度，计算 x 和 y 的梯度
  res.backward(torch::ones({2, 2}), false, true);

  // 根据计算得到的梯度，计算 x 和 y 的预期梯度
  Variable x_grad = y + torch::ones({2, 2});
  Variable y_grad = x + torch::ones({2, 2}) * 2;

  // 断言 x 和 y 的梯度与预期值相等
  ASSERT_VARIABLE_EQ(x.grad(), x_grad);
  ASSERT_VARIABLE_EQ(y.grad(), y_grad);

  // 计算 grad_sum，并进行高阶梯度计算，对 x 进行 Hessian 向量积
  Variable grad_sum = 2 * x.grad() + y.grad();
  auto x_hv = grad({grad_sum}, {x}, {torch::ones({2, 2})}, {}, true);

  // 断言 x 的 Hessian 向量积与预期值相等
  ASSERT_VARIABLE_EQ(x_hv[0], torch::ones({2, 2}));
  // 再次验证 x 和 y 的梯度与预期值相等
  ASSERT_VARIABLE_EQ(x.grad(), x_grad);
  ASSERT_VARIABLE_EQ(y.grad(), y_grad);
}

TEST(AutogradAPITests, GradNonLeafTest) {
  // 创建具有梯度的随机变量 x_init, x 和 y
  Variable x_init = torch::randn({2, 2}, torch::requires_grad());
  Variable x = x_init;
  Variable y = torch::randn({2, 2}, torch::requires_grad());
  Variable grad_output = torch::ones({2, 2});

  // 循环执行简单函数 simple_fn，计算梯度
  for (int i = 0; i < 5; ++i) {
    auto res = simple_fn(x, y);
    auto input_grads = grad({res}, {x}, {grad_output}, {}, true);

    // 计算预期的 x 的梯度
    Variable grad_x_expected = y + torch::ones({2, 2});
    ASSERT_VARIABLE_EQ(input_grads[0], grad_x_expected);

    // 验证 x 和 y 的梯度未定义
    ASSERT_FALSE(x.grad().defined());
    ASSERT_FALSE(y.grad().defined());

    // 更新 x 的值
    x = x + 0.05 * input_grads[0];
  }

  // 计算简单函数在初始和最终状态下的值，验证最终状态的值大于初始状态
  float val_init = simple_fn(x_init, y).sum().item().toFloat();
  float val_final = simple_fn(x, y).sum().item().toFloat();
  ASSERT_TRUE(val_final > val_init);

  // 对 x 执行反向传播，计算梯度
  x.backward(grad_output, false, true);

  // 验证初始变量 x_init 和 y 的梯度已定义
  ASSERT_TRUE(x_init.grad().defined());
  ASSERT_TRUE(y.grad().defined());
}

TEST(AutogradAPITests, GradUnreachableTest) {
  // 创建具有梯度的变量 x 和 y
  Variable x = torch::ones({1}, torch::requires_grad());
  Variable y = torch::ones({1}, torch::requires_grad());

  // 创建变量 z 和 w，分别为 x 和 y 的两倍
  Variable z = x * 2;
  Variable w = y * 2;

  // 对 x 计算梯度，y 不计算梯度，预期结果为 x * 2
  auto grad_res = grad({x * 2}, {x, y}, {}, {}, false, true);
  ASSERT_VARIABLE_EQ(grad_res[0], x * 2);
  ASSERT_FALSE(grad_res[1].defined());

  // 变量 z 覆盖为具有梯度的新变量
  z = torch::ones({1}, torch::requires_grad());

  // 再次计算梯度，y 不计算梯度，预期结果为 x * 2
  grad_res = grad({x * 2}, {x, z}, {}, {}, false, true);
  ASSERT_VARIABLE_EQ(grad_res[0], x * 2);
  ASSERT_FALSE(grad_res[1].defined());

  // allow_unused=False，但梯度包含 None，应抛出异常
  ASSERT_THROWS_WITH(
      grad({x * 2}, {x, y}, {}, {}, false, false), "Set allow_unused=True");
}

TEST(CustomAutogradTest, GradUnreachableDiscoveryTest) {
  // 测试在某些节点不可达时，不会误执行特定节点
  // 参见 issue #39784
  struct MyFunction : public Function<MyFunction> {
    static Variable forward(AutogradContext* ctx, Variable var) {
      return var;
    }

    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      ADD_FAILURE() << "This node should not be executed!";
      return grad_output;
    }
  };
}
    }
  };



// 定义一个匿名的自动求导函数对象，将 x 和 x1 作为输入
auto x = torch::randn(1, torch::requires_grad());
// 创建一个不需要求导的随机张量 x1
auto x1 = torch::randn(1);
// 调用自定义函数 MyFunction 的静态方法 apply，传入 x + x1 作为参数
auto x2 = MyFunction::apply(x + x1);

// 创建一个需要求导的随机张量 y
auto y = torch::randn(1, torch::requires_grad());
// 计算 x2 对 y 的梯度
auto grad_res = torch::autograd::grad({x2}, {y}, {}, {}, false, true);
// 断言梯度结果 grad_res[0] 未被定义（应为未指定 y 对 x2 的求导路径）
ASSERT_FALSE(grad_res[0].defined());
}

// 定义一个测试用例，测试在空输入情况下的自动求导API行为
TEST(AutogradAPITests, EmptyInput) {
  // 创建一个张量 x，其值为1，需要梯度
  Variable x = torch::ones({1}, torch::requires_grad());
  // 断言在调用 grad 函数时传入空输入会抛出异常，并验证异常信息
  ASSERT_THROWS_WITH(
      grad({x * 2}, /*inputs=*/{}, {x}), "grad requires non-empty inputs.");
}

// 定义一个测试用例，测试在保留梯度的情况下的自动求导API行为
TEST(AutogradAPITests, RetainGrad) {
  // 创建一个随机初始化的张量 input，需要梯度
  auto input = torch::rand({1, 3}, torch::requires_grad());
  // 计算 h1，并保留其梯度
  auto h1 = input * 3;
  // 计算 out，并对其进行求和
  auto out = (h1 * h1).sum();

  {
    // 在访问非叶子张量的梯度时，捕获警告信息
    WarningCapture warnings;
    ASSERT_FALSE(h1.grad().defined());
    ASSERT_TRUE(warnings.str().find("is not a leaf") != std::string::npos);
  }

  // 可以多次调用 retain_grad()，确认其效果
  h1.retain_grad();
  h1.retain_grad();
  {
    // 如果对非叶子张量启用了 retain_grad，
    // 当访问梯度时不应有警告信息
    WarningCapture warnings;
    ASSERT_FALSE(h1.grad().defined());
    ASSERT_FALSE(warnings.str().find("is not a leaf") != std::string::npos);
  }

  // 梯度应该被累积
  // NOLINTNEXTLINE(bugprone-argument-comment)
  out.backward({}, /*keep_graph=*/true);
  ASSERT_VARIABLE_EQ(h1 * 2, h1.grad());
  // NOLINTNEXTLINE(bugprone-argument-comment)
  out.backward({}, /*keep_graph=*/true);
  ASSERT_VARIABLE_EQ(h1 * 4, h1.grad());

  {
    // 进入无梯度计算区域
    torch::NoGradGuard no_grad;
    input.grad().zero_();
  }

  // 对于叶子节点应该是无操作的
  input.retain_grad();
  input.retain_grad();
  out.backward();
  ASSERT_VARIABLE_EQ(input * 18, input.grad());
}

// 定义一个测试用例，测试异常模式下的自动求导API行为
TEST(AutogradAPITests, AnomalyMode) {
  // 开启异常检测
  torch::autograd::DetectAnomalyGuard detect_anomaly;
  {
    // 捕获警告信息
    WarningCapture warnings;
    auto x = torch::tensor({5.0}, torch::requires_grad());
    auto y = x * x;
    auto z = y * y;
    y += 1;
    // 断言在 inplace 操作时会抛出异常，并验证异常信息
    ASSERT_THROWS_WITH(z.backward(), "inplace");
    ASSERT_TRUE(
        warnings.str().find("Traceback of forward") != std::string::npos);
  }

  // 定义一个函数，用于测试双重反向传播是否产生 NaN
  auto double_backward_produce_nan = [](bool should_throw) {
    auto x = torch::tensor({0.0}, torch::requires_grad());
    auto y = x.pow(1.5);
    auto gr =
        // NOLINTNEXTLINE(bugprone-argument-comment)
        grad({y}, {x}, {}, /*retain_graph=*/true, /*create_backward=*/true);
    if (should_throw) {
      WarningCapture warnings;
      // 断言在计算梯度时返回 NaN 会抛出异常，并验证警告信息
      ASSERT_THROWS_WITH(grad({gr[0]}, {x}, {torch::tensor({0.0})}),
                         "returned nan");
      auto msgs = warnings.messages();
      ASSERT_EQ(msgs.size(), 2);
      ASSERT_TRUE(
          msgs[0].find("Traceback of forward call that caused the error") !=
          std::string::npos);
      ASSERT_TRUE(
          msgs[1].find(
              "Traceback of forward call that induced the previous calculation") !=
          std::string::npos);
    } else {
      grad({gr[0]}, {x}, {torch::tensor({0.0})});
    }
  };

  // 在异常模式下测试双重反向传播是否产生 NaN
  double_backward_produce_nan(true);

  {
    // 禁止检测 NaN 的异常模式
    torch::autograd::DetectAnomalyGuard detect_anomaly(/*check_nan=*/false);
    double_backward_produce_nan(false);
    {
      // 创建一个 DetectAnomalyGuard 对象，用于检测自动求导过程中的异常，设置检查NaN值为true
      torch::autograd::DetectAnomalyGuard detect_anomaly(/*check_nan=*/true);
      // 调用函数设置双向梯度传播时产生NaN的标志为true，开启双向梯度传播时产生NaN值的检测
      double_backward_produce_nan(true);
    }
    // 上述代码块结束
    
    // 设置全局双向梯度传播时产生NaN的标志为true，可能会影响后续的计算
    double_backward_produce_nan(true);
}

TEST(CustomAutogradTest, CustomFunctionReturnInputAsIsAndSavesIt) {
  // 定义自定义的函数对象 MyFunction，继承自 torch::autograd::Function
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数的实现
    static Variable forward(
        AutogradContext* ctx,
        Variable var1,
        Variable var2) {
      // 在 AutogradContext 中保存 var1 和 var2 的引用
      ctx->save_for_backward({var1, var2});
      // 返回 var1 * var2 和 var1 的乘积作为结果
      return var1 * var2, var1;
    }

    // 反向传播函数的实现
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 返回空的梯度列表，因为不需要计算梯度
      return {};
    }
  };

  // 创建两个需要梯度的随机张量变量 x 和 y
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  // 调用 MyFunction 的前向传播方法 apply
  MyFunction::apply(x, y);
}

TEST(CustomAutogradTest, CustomFunction) {
  // 定义自定义的函数对象 MyFunction，继承自 torch::autograd::Function
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数的实现
    static Variable forward(
        AutogradContext* ctx,
        Variable var1,
        int mul,
        Variable var2) {
      // 在 AutogradContext 中保存 mul 的值
      ctx->saved_data["mul"] = mul;
      // 在 AutogradContext 中保存 var1 和 var2 的引用
      ctx->save_for_backward({var1, var2});
      // 返回 var1 + mul * var2 + var1 * var2 的结果张量
      return var1 + mul * var2 + var1 * var2;
    }

    // 反向传播函数的实现
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 从 AutogradContext 中获取保存的变量
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];
      // 计算并返回梯度列表
      variable_list output = {
          grad_output[0] + grad_output[0] * var2,
          Variable(),
          grad_output[0] * mul + grad_output[0] * var1};  // 错误：应该是 mul.toInt()
      return output;
    }
  };

  // 创建两个需要梯度的随机张量变量 x 和 y
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());
  // 调用 MyFunction 的前向传播方法 apply，传入参数 x, 2, y
  auto res = MyFunction::apply(x, 2, y);
  // 创建一个需要梯度的全一张量 go
  auto go = torch::ones({}, torch::requires_grad());
  // 对 res 的所有元素求和，并进行反向传播计算梯度
  res.sum().backward(go, false, true);

  // 断言 x 的梯度应为 y + 全一张量
  ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({5, 5}));
  // 断言 y 的梯度应为 x + 全一张量乘以 2
  ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({5, 5}) * 2);
}

TEST(CustomAutogradTest, CustomFunctionWithTensorList) {
  // 定义自定义的函数对象 MyFunction，继承自 torch::autograd::Function
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数的实现
    static Variable forward(AutogradContext* ctx, at::TensorList tensors) {
      // 创建 torch::autograd::variable_list 对象 vars
      torch::autograd::variable_list vars;
      // 遍历 tensors 中的每个张量 tensor，并添加到 vars 中
      for (const at::Tensor& tensor : tensors) {
        vars.push_back(tensor);
      }
      // 在 AutogradContext 中保存 vars
      ctx->save_for_backward(vars);
      // 返回 tensors[0] + tensors[1] + tensors[0] * tensors[1] 的结果张量
      return tensors[0] + tensors[1] + tensors[0] * tensors[1];
    }

    // 反向传播函数的实现
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 从 AutogradContext 中获取保存的变量
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];
      // 计算并返回梯度列表
      variable_list output = {
          grad_output[0] + grad_output[0] * var2,
          grad_output[0] + grad_output[0] * var1};
      return output;
  }
};

// 创建一个 5x5 的张量 x，并指定需要计算梯度
at::Tensor x = torch::randn({5, 5}, torch::requires_grad());
// 创建一个 5x5 的张量 y，并指定需要计算梯度
at::Tensor y = torch::randn({5, 5}, torch::requires_grad());
// 将 x 和 y 加入到变量列表中
torch::autograd::variable_list variables = {x, y};
// 将变量列表转换为张量列表
at::TensorList tensors = variables;
// 调用自定义函数 MyFunction::apply，传入张量列表 tensors，并获得结果
auto res = MyFunction::apply(tensors);
// 创建一个标量张量 go，其值为 1，并指定需要计算梯度
auto go = torch::ones({}, torch::requires_grad());
// 对结果张量 res 所有元素求和并反向传播梯度
res.sum().backward(go, false, true);

// 使用断言检查张量 x 的梯度是否与 y 加上一个全为1的张量相等
ASSERT_VARIABLE_EQ(x.grad(), y + torch::ones({5, 5}));
// 使用断言检查张量 y 的梯度是否与 x 加上一个全为1的张量相等
ASSERT_VARIABLE_EQ(y.grad(), x + torch::ones({5, 5}));
// 定义一个测试用例，名为 CustomAutogradTest，测试 GraphTaskTrimEdges 函数
TEST(CustomAutogradTest, GraphTaskTrimEdges) {
  // 定义一个结构 MyFunction，继承自 Function<MyFunction>
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数的实现
    static Variable forward(
        AutogradContext* ctx,  // 自动微分上下文对象指针
        Variable var1,         // 输入变量 var1
        Variable var2,         // 输入变量 var2
        int mul,               // 整数乘数
        bool needs_input1_grad,  // 是否需要 var1 的梯度
        bool needs_input2_grad) {  // 是否需要 var2 的梯度
      // 设置需要保存的数据：是否需要 var1 和 var2 的梯度
      ctx->saved_data["needs_input1_grad"] = needs_input1_grad;
      ctx->saved_data["needs_input2_grad"] = needs_input2_grad;

      // 保存数据：乘数 mul
      ctx->saved_data["mul"] = mul;
      // 保存需要反向传播的变量列表：var1 和 var2
      ctx->save_for_backward({var1, var2});
      // 返回计算结果：var1 + mul * var2 + var1 * var2
      return var1 + mul * var2 + var1 * var2;
    }

    // 反向传播函数的实现
    static variable_list backward(
        AutogradContext* ctx,        // 自动微分上下文对象指针
        variable_list grad_output) { // 梯度输出变量列表
      // 在反向传播函数内部测试 needs_input_grad 方法是否正确工作
      auto needs_input1_grad = ctx->saved_data["needs_input1_grad"].toBool();
      auto needs_input2_grad = ctx->saved_data["needs_input2_grad"].toBool();
      
      // 定义索引范围：var1_idx 为 [0, 1]，var2_idx 为 [1, 2]
      IndexRange var1_idx = {0, 1};
      IndexRange var2_idx = {1, 2};
      
      // 检查需要输入梯度的情况是否与预期一致
      EXPECT_EQ(ctx->needs_input_grad(0), needs_input1_grad);
      EXPECT_EQ(ctx->needs_input_grad(1), needs_input2_grad);
      EXPECT_EQ(ctx->needs_input_grad({var1_idx}), needs_input1_grad);
      EXPECT_EQ(ctx->needs_input_grad({var2_idx}), needs_input2_grad);
      EXPECT_EQ(
          ctx->needs_input_grad({var1_idx, var2_idx}),
          needs_input1_grad || needs_input2_grad);

      // 计算梯度
      int mul = ctx->saved_data["mul"].toInt();
      auto saved = ctx->get_saved_variables();
      auto var1 = saved[0];
      auto var2 = saved[1];

      Variable grad_var1, grad_var2;
      // 计算 var1 的梯度
      if (ctx->needs_input_grad(0)) {
        grad_var1 = grad_output[0] + grad_output[0] * var2;
      }
      // 计算 var2 的梯度
      if (ctx->needs_input_grad(1)) {
        grad_var2 = grad_output[0] * mul + grad_output[0] * var1;
      }

      // 返回梯度变量列表
      variable_list output = {
          grad_var1,
          grad_var2,
          Variable(),  // 空变量
          Variable(),  // 空变量
          Variable(),  // 空变量
      };
      return output;
    }
  };
  }
};

// 创建随机张量 x 和 y，使其需要梯度计算
Variable x = torch::randn({5, 5}, torch::requires_grad());
Variable y = torch::randn({5, 5}, torch::requires_grad());

// 创建全为1的张量 go，用于梯度计算
auto go = torch::ones_like(x);

// 定义变量 out
Variable out;

// 计算 x 的梯度
out = MyFunction::apply(
    x,
    y,
    2,
    /* needs_input1_grad= */ true,
    /* needs_input2_grad= */ false);
auto grad_x = torch::autograd::grad({out}, {x}, {go})[0];
ASSERT_VARIABLE_EQ(grad_x, y + torch::ones({5, 5}));

// 计算 y 的梯度
out = MyFunction::apply(
    x,
    y,
    2,
    /* needs_input1_grad= */ false,
    /* needs_input2_grad= */ true);
auto grad_y = torch::autograd::grad({out}, {y}, {go})[0];
ASSERT_VARIABLE_EQ(grad_y, x + torch::ones({5, 5}) * 2);

// 同时计算 x 和 y 的梯度
out = MyFunction::apply(
    x,
    y,
    2,
    /* needs_input1_grad= */ true,
    /* needs_input2_grad= */ true);
auto grads = torch::autograd::grad({out}, {x, y}, {go});
ASSERT_VARIABLE_EQ(grads[0], y + torch::ones({5, 5}));
ASSERT_VARIABLE_EQ(grads[1], x + torch::ones({5, 5}) * 2);
}

TEST(CustomAutogradTest, FunctionReturnsInput) {
  // 定义自定义函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接收 AutogradContext 和 Variable 参数，直接返回输入变量 var1
    static Variable forward(AutogradContext* ctx, Variable var1) {
      return var1;
    }

    // 反向传播函数，接收 AutogradContext 和 variable_list 参数 grad_output，
    // 返回 grad_output[0] 的两倍作为梯度列表
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      return {grad_output[0] * 2};
    }
  };

  // 创建一个需要梯度的张量 x，其值为 1
  Variable x(torch::ones(1, torch::requires_grad()));
  // 调用 MyFunction 的静态方法 apply 对 x 进行前向传播和反向传播
  MyFunction::apply(x).backward(torch::ones(1), true, true);
  // 断言 x 的梯度为全为 2 的张量
  ASSERT_VARIABLE_EQ(x.grad(), torch::full(1, 2.));
}

TEST(CustomAutogradTest, FunctionReturnsUndefined) {
  // 定义自定义函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接收 AutogradContext 和 Variable 参数 var，
    // 返回 var 的两倍作为输出
    static Variable forward(AutogradContext* ctx, Variable var) {
      return var * 2;
    }

    // 反向传播函数，接收 AutogradContext 和 variable_list 参数 grad_output，
    // 创建一个未定义的张量 undefined_tensor，并将其作为梯度输出
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      at::Tensor undefined_tensor;
      return {undefined_tensor};
    }
  };

  // 创建一个需要梯度的张量 x，其值为 1
  auto x = torch::ones(1, torch::requires_grad());

  // 调用 MyFunction 的 apply 方法进行前向传播和反向传播，并断言 x 的梯度未定义
  MyFunction::apply(x).backward();
  ASSERT_FALSE(x.grad().defined());

  // 对 x 的平方应用 MyFunction，再次断言 x 的梯度未定义
  MyFunction::apply(x.pow(2)).backward();
  ASSERT_FALSE(x.grad().defined());

  // 对 x 应用 MyFunction，并对结果求和进行反向传播，再次断言 x 的梯度未定义
  MyFunction::apply(x).sum().backward();
  ASSERT_FALSE(x.grad().defined());

  // 使用 torch::autograd::grad 对 MyFunction 应用到 x，不计算梯度，断言 x 的梯度未定义
  ASSERT_FALSE(torch::autograd::grad(
                   {MyFunction::apply(x)}, {x}, {}, false, false, true)[0]
                   .defined());
}

TEST(CustomAutogradTest, MaterializeGrads) {
  // 定义自定义函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接收 AutogradContext 和 Variable 参数 var，
    // 直接返回输入变量 var
    static Variable forward(AutogradContext* ctx, Variable var) {
      return var;
    }

    // 反向传播函数，接收 AutogradContext 和 variable_list 参数 grad_output，
    // 断言 grad_output[0] 为全零，并将其作为梯度输出
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      EXPECT_VARIABLE_EQ(grad_output[0], torch::zeros(1));
      return grad_output;
    }
  };

  // 创建一个需要梯度的张量 x，其值为 1
  auto x = torch::ones(1, torch::requires_grad());
  // 调用 UndefinedGrad().apply 方法应用 MyFunction 到 x，得到结果并进行反向传播
  UndefinedGrad().apply({MyFunction::apply(x)})[0].backward();
}

TEST(CustomAutogradTest, DontMaterializeGrads) {
  // 定义自定义函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接收 AutogradContext 和 Variable 参数 var，
    // 设置 AutogradContext 的梯度材料化为 false，并返回输入变量 var
    static Variable forward(AutogradContext* ctx, Variable var) {
      ctx->set_materialize_grads(false);
      return var;
    }

    // 反向传播函数，接收 AutogradContext 和 variable_list 参数 grad_output，
    // 断言 grad_output[0] 未定义，并将其作为梯度输出
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      EXPECT_FALSE(grad_output[0].defined());
      return grad_output;
    }
  };

  // 创建一个需要梯度的张量 x，其值为 1
  auto x = torch::ones(1, torch::requires_grad());
  // 调用 UndefinedGrad().apply 方法应用 MyFunction 到 x，得到结果并进行反向传播
  UndefinedGrad().apply({MyFunction::apply(x)})[0].backward();
}

TEST(CustomAutogradTest, NoGradCustomFunction) {
  // Custom Function 应该尊重无梯度模式
  struct MyOp : public Function<MyOp> {
    // 前向传播函数，接收 AutogradContext 和 Variable 参数 x，
    // 返回 x + 1
    static Variable forward(AutogradContext* ctx, Variable x) {
      return x + 1;
    }

    // 反向传播函数，接收 AutogradContext 和 variable_list 参数 dy，
    // 直接返回 dy
    static variable_list backward(AutogradContext* ctx, variable_list dy) {
      return dy;
    }
  };

  // 创建一个需要梯度的 5x5 张量 x，其值为 1
  auto x = torch::ones({5, 5}, torch::requires_grad());
  {
    // 使用 NoGradGuard 禁用梯度计算
    at::NoGradGuard no_grad;
    // 调用 MyOp::apply 方法应用 MyOp 到 x，并断言 y 不需要梯度
    auto y = MyOp::apply(x);
    ASSERT_FALSE(y.requires_grad());
  }
}

TEST(CustomAutogradTest, MarkDirty) {
  // 定义自定义函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
  static Variable forward(AutogradContext* ctx, Variable v) {
    // 在原地修改变量的值
    auto v_data = v.data_ptr<float>();
    v_data[0] = 2;
    // 标记变量为已修改状态，以便后续计算梯度
    ctx->mark_dirty({v});
    // 返回修改后的变量
    return v;
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // 返回梯度的计算结果，此处简单地将输入梯度乘以2.0
    return {(grad_output[0] * 2.0)};
  }
};

// 在这里进行克隆操作，因为不允许原地修改叶节点的值
auto x = torch::randn({5, 5}, torch::requires_grad()).clone();
auto version_before = x._version();
// 调用自定义的前向传播函数进行计算
auto out = MyFunction::apply(x);
auto version_after = x._version();
// 确保经过前向传播后版本号至少增加了1
ASSERT_TRUE(version_after >= (version_before + 1));
// 对输出进行求和，并自动计算反向传播
out.sum().backward();
TEST(CustomAutogradTest, MarkNonDifferentiable) {
  // 定义一个结构体 MyFunction，继承自 Function<MyFunction>
  struct MyFunction : public Function<MyFunction> {
    // 静态方法：前向传播
    static Variable forward(AutogradContext* ctx, Variable v) {
      // 计算 v 是否大于 0，并生成对应的布尔类型 Variable
      Variable output = v > 0;
      // 标记 output 为非可微分的
      ctx->mark_non_differentiable({output});
      // 返回 output
      return output;
    }

    // 静态方法：反向传播
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 返回一个列表，其中仅包含 grad_output[0] 乘以 0.0 的结果
      return {(grad_output[0] * 0.0)};
    }
  };

  // 生成一个 5x5 的随机张量 x，并要求其可导
  auto x = torch::randn({5, 5}, torch::requires_grad());
  // 调用 MyFunction 的 apply 方法，生成 mask
  auto mask = MyFunction::apply(x);
  // 断言 mask 不要求梯度跟踪
  ASSERT_FALSE(mask.requires_grad());
  // 使用 mask 对 x 进行填充操作，将 mask 为真的位置填充为 0
  auto y = x.masked_fill(mask, 0);
  // 对 y 的所有元素求和并进行反向传播
  y.sum().backward();
}

TEST(CustomAutogradTest, MarkNonDifferentiableMixed) {
  // 定义一个结构体 MyFunction，继承自 Function<MyFunction>
  struct MyFunction : public Function<MyFunction> {
    // 静态方法：前向传播
    static variable_list forward(AutogradContext* ctx, Variable input) {
      // 计算 input + 1，并赋值给变量 a
      Variable a = input + 1;
      // 计算 input + 2，并赋值给变量 b
      Variable b = input + 2;
      // 标记 a 为非可微分的
      ctx->mark_non_differentiable({a});
      // 返回列表 [a, b]
      return {a, b};
    }

    // 静态方法：反向传播
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 获取 grad_output 中的第一个元素，并命名为 grad_a，第二个元素命名为 grad_b
      const Variable &grad_a = grad_output[0], &grad_b = grad_output[1];
      // 断言 grad_a 是一个全为 0 的 5x5 张量
      EXPECT_VARIABLE_EQ(grad_a, torch::zeros({5, 5}));
      // 断言 grad_b 是一个全为 1 的 5x5 张量
      EXPECT_VARIABLE_EQ(grad_b, torch::ones({5, 5}));
      // 返回列表 [grad_b]
      return {grad_b};
    }
  };

  // 生成一个 5x5 的随机张量 x，并要求其可导
  auto x = torch::randn({5, 5}, torch::requires_grad());
  // 调用 MyFunction 的 apply 方法，生成 out 列表
  auto out = MyFunction::apply(x);

  // 断言 out 中的第一个元素不要求梯度跟踪
  ASSERT_FALSE(out[0].requires_grad());
  // 断言 out 中的第二个元素要求梯度跟踪
  ASSERT_TRUE(out[1].requires_grad());
  // 对 out 中的第二个元素求和并进行反向传播
  out[1].sum().backward();
  // 断言 x 的梯度是一个全为 1 的 5x5 张量
  ASSERT_VARIABLE_EQ(x.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, MarkNonDifferentiableNone) {
  // 定义一个结构体 MyFunction，继承自 Function<MyFunction>
  struct MyFunction : public Function<MyFunction> {
    // 静态方法：前向传播
    static Variable forward(AutogradContext* ctx, Variable input) {
      // 克隆 input，并赋值给 output
      auto output = input.clone();
      // 标记 output 为非可微分的
      ctx->mark_non_differentiable({output});
      // 返回 output
      return output;
    }

    // 静态方法：反向传播
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_outputs) {
      // 返回一个空列表
      return {};
    }
  };

  // 生成一个 5x5 的随机张量 x，并要求其可导
  auto x = torch::randn({5, 5}, torch::requires_grad());
  // 调用 MyFunction 的 apply 方法，生成 r
  auto r = MyFunction::apply(x * x);
  // 对 r 乘以 x，并对结果求和进行反向传播
  (r * x).sum().backward();
}

TEST(CustomAutogradTest, ReturnLeafInplace) {
  // 定义一个结构体 Inplace，继承自 Function<Inplace>
  struct Inplace : public Function<Inplace> {
    // 静态方法：前向传播
    static variable_list forward(AutogradContext* ctx, Variable a, Variable b) {
      // 标记 a 为脏（dirty）
      ctx->mark_dirty({a});
      // 返回列表 [a.add_(b), b + 2]
      return {a.add_(b), b + 2};
    }

    // 静态方法：反向传播
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 返回列表 [grad_output[0], grad_output[0] + grad_output[1]]
      return {grad_output[0], grad_output[0] + grad_output[1]};
    }
  };

  // 生成一个 5x5 的随机张量 x
  Variable x = torch::randn({5, 5});
  // 生成一个 5x5 的随机张量 y，并要求其可导
  Variable y = torch::randn({5, 5}, torch::requires_grad());

  // 调用 Inplace 的 apply 方法，生成 out 列表
  auto out = Inplace::apply(x, y);
  // 获取 out 中的第一个元素，并赋值给 q
  auto& q = out[0];
  // 断言 q 与 x 相等
  ASSERT_TRUE(torch::equal(q, x));
  // 断言 q 要求梯度跟踪
  ASSERT_TRUE(q.requires_grad());
  // 对 q 的所有元素求和并进行反向传播
  q.sum().backward();
  // 断言 y 的梯度是一个全为 1 的 5x5 张量
  ASSERT_VARIABLE_EQ(y.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, ReturnDuplicateInplace) {
  // 定义一个结构体 DoubleInplace，继承自 Function<DoubleInplace>
    // 定义一个静态函数 `forward`，用于执行前向传播
    static variable_list forward(AutogradContext* ctx, Variable x) {
      // 将输入变量 x 原地乘以 2
      x.mul_(2);
      // 标记变量 x 为已修改状态，以便后续反向传播使用
      ctx->mark_dirty({x});
      // 返回两次变换后的变量 x 作为输出
      return {x, x};
    }

    // 定义一个静态函数 `backward`，用于执行反向传播
    static variable_list backward(
        AutogradContext* ctsx,
        variable_list grad_outputs) {
      // 返回两个梯度输出的加权和，每个输出乘以 2
      return {grad_outputs[0] * 2 + grad_outputs[1] * 2};
    }
  };

  // 创建一个大小为 5x5 的张量 x，要求计算其梯度
  auto x = torch::randn({5, 5}, torch::requires_grad());

  // 断言：调用 DoubleInplace 类的 apply 函数会抛出异常 "leaf Variable that requires grad"
  ASSERT_THROWS_WITH(
      DoubleInplace::apply(x), "leaf Variable that requires grad");
  
  // TODO ASSERT_THROWS_WITH(DoubleInplace::apply(x.clone()[0]), "only one
  // output");

  // 复制张量 x，并将其传递给 DoubleInplace 类的 apply 函数
  auto out = DoubleInplace::apply(x.clone());
  
  // 断言：验证复制后的两个输出张量是否相等
  ASSERT_TRUE(torch::equal(out[0], out[1]));
TEST(CustomAutogradTest, ReturnDuplicate) {
  // 定义一个自定义的自动求导函数 DoubleDuplicate，继承自 Function 类
  struct DoubleDuplicate : public Function<DoubleDuplicate> {
    // 前向传播函数，接受 AutogradContext 上下文和变量 x
    static variable_list forward(AutogradContext* ctx, Variable x) {
      // 计算输出为输入 x 的两倍
      auto output = x * 2;
      // 返回两次相同的输出作为结果
      return {output, output};
    }

    // 反向传播函数，接受 AutogradContext 上下文和梯度输出 grad_outputs
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_outputs) {
      // 返回输入梯度的两倍
      return {grad_outputs[0] * 2 + grad_outputs[1] * 2};
    }
  };

  // 生成一个随机张量 x，形状为 (5, 5)，需要计算梯度
  auto x = torch::randn({5, 5}, torch::requires_grad());
  // 调用 DoubleDuplicate 的前向传播方法
  auto out = DoubleDuplicate::apply(x);
  // 断言两个输出张量相等
  ASSERT_TRUE(torch::equal(out[0], out[1]));
}

TEST(CustomAutogradTest, SaveEmptyForBackward) {
  // 定义一个自定义的自动求导函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接受 AutogradContext 上下文和输入变量 input
    static Variable forward(AutogradContext* ctx, Variable input) {
      // 保存一个空变量、输入变量和另一个空变量供反向传播使用
      ctx->save_for_backward({Variable(), input, Variable()});
      // 返回输入变量的平方作为输出
      return input * input;
    }

    // 反向传播函数，接受 AutogradContext 上下文和梯度输出 grad_output
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 获取保存的变量
      auto saved = ctx->get_saved_variables();
      // 断言第一个和第三个保存的变量未定义
      EXPECT_FALSE(saved[0].defined());
      EXPECT_FALSE(saved[2].defined());
      // 返回输入变量的梯度乘以 2
      return {saved[1] * 2 * grad_output[0]};
    }
  };

  // 生成一个随机张量 x，形状为 (5, 5)，需要计算梯度
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  // 调用 MyFunction 的前向传播方法
  auto y = MyFunction::apply(x);
  // 对 y 的和进行反向传播
  y.sum().backward();
  // 断言 x 的梯度是其本身的两倍
  ASSERT_VARIABLE_EQ(x.grad(), 2 * x);
}

TEST(CustomAutogradTest, InvalidGradients) {
  // 定义一个自定义的自动求导函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接受 AutogradContext 上下文和变量 x
    static Variable forward(AutogradContext* ctx, Variable x) {
      // 返回输入变量的两倍作为输出
      return x * 2;
    }

    // 反向传播函数，接受 AutogradContext 上下文和梯度输出 grad_outputs
    static variable_list backward(
        AutogradContext* ctsx,
        variable_list grad_outputs) {
      // 返回一个形状为 (10,) 的随机张量作为梯度输出
      return {
          torch::randn(10, torch::dtype(torch::kFloat).requires_grad(true))};
    }
  };

  // 生成一个形状为 (5, 5) 的随机浮点数张量，需要计算梯度
  auto input1 =
      torch::randn({5, 5}, torch::dtype(torch::kFloat).requires_grad(true));
  // 断言 MyFunction 对 input1 的应用抛出异常并包含 "expected shape"
  ASSERT_THROWS_WITH(
      MyFunction::apply(input1).sum().backward(), "expected shape");
  
  // 生成一个形状为 (10,) 的随机双精度张量，需要计算梯度
  auto input2 =
      torch::randn(10, torch::dtype(torch::kDouble).requires_grad(true));
}

TEST(CustomAutogradTest, NoGradInput) {
  // 定义一个自定义的自动求导函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接受 AutogradContext 上下文和变量 x
    static Variable forward(AutogradContext*, Variable x) {
      // 直接返回输入变量 x
      return x;
    }

    // 反向传播函数，接受 AutogradContext 上下文和梯度输出 grad_outputs
    static variable_list backward(
        AutogradContext*,
        variable_list grad_outputs) {
      // 返回梯度输出本身
      return grad_outputs;
    }
  };

  // 生成一个形状为 (5, 5) 的随机张量 x，需要计算梯度
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  // 进入无梯度环境，对 x 应用 MyFunction
  Variable y;
  {
    at::NoGradGuard no_grad;
    y = MyFunction::apply(x);
  }

  // 断言 x 仍然需要计算梯度
  ASSERT_TRUE(x.requires_grad());
  // 断言 y 没有梯度函数
  ASSERT_FALSE(y.grad_fn());
}

TEST(CustomAutogradTest, TooManyGrads) {
  // 定义一个自定义的自动求导函数 MyFunction，继承自 Function 类
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接受 AutogradContext 上下文和输入变量 input
    static Variable forward(AutogradContext*, Variable input) {
      // 直接返回输入变量 input
      return input;
    }

    // 反向传播函数，接受 AutogradContext 上下文和梯度输出 grad_output
    static variable_list backward(AutogradContext*, variable_list grad_output) {
      // 在 grad_output 后面插入两个空变量
      grad_output.insert(grad_output.end(), {Variable(), Variable()});
      // 返回修改后的 grad_output
      return grad_output;
    }
  };
}

TEST(CustomAutogradTest, DepNoGrad) {
  // 定义一个自定义的自动求导函数 F1，继承自 Function 类
  struct F1 : public Function<F1> {
    // ...
  // 定义一个结构体 F1，继承自 Function<F1>
  struct F1 : public Function<F1> {
    // 前向传播函数，接收 AutogradContext 指针和 Variable 输入，返回 Variable 列表
    static Variable forward(AutogradContext* ctx, Variable input) {
      // 生成一个与 input 相同尺寸的随机数 Tensor
      auto out = torch::randn(input.sizes());
      // 标记 out 为不可微分
      ctx->mark_non_differentiable({out});
      // 返回 input 和 out 的 Variable 列表
      return {input, out};
    }

    // 反向传播函数，接收 AutogradContext 指针和 Variable 列表 grad_output，返回 Variable 列表
    static variable_list backward(AutogradContext* ctx, variable_list grad_output) {
      // 返回 grad_output 的第一个元素
      return {grad_output[0]};
    }
  };

  // 定义一个结构体 F2，继承自 Function<F2>
  struct F2 : public Function<F2> {
    // 前向传播函数，接收 AutogradContext 指针和两个 Variable 输入，返回一个 Variable
    static Variable forward(AutogradContext* ctx, Variable input, Variable ignore) {
      // 返回第一个输入 input
      return input;
    }

    // 反向传播函数，接收 AutogradContext 指针和 Variable 列表 grad_output，返回 Variable 列表
    static variable_list backward(AutogradContext* ctx, variable_list grad_output) {
      // 返回 grad_output 的第一个元素和一个空 Variable
      return {grad_output[0], Variable()};
    }
  };

  // 生成一个尺寸为 (5,) 的随机数 Tensor x，并要求计算梯度
  auto x = torch::randn(5, torch::requires_grad());
  // 使用 F1::apply 对 x 进行前向传播，得到结果 out
  auto out = F1::apply(x);
  // 将 out 的第一个元素赋给变量 a，第二个元素赋给变量 b
  Variable &a = out[0], &b = out[1];
  // 对 b 执行加法操作（与 F1 和 F2 分隔开）
  b = b + 1;
  // 断言 a 需要计算梯度
  ASSERT_TRUE(a.requires_grad());
  // 断言 b 不需要计算梯度
  ASSERT_FALSE(b.requires_grad());

  // 使用 F2::apply 对 a 和 b 进行前向传播，得到结果 c
  auto c = F2::apply(a, b);
  // 对 c 执行反向传播，使用全为 1 的梯度，不保留计算图，不执行梯度优化
  c.backward(torch::ones(c.sizes()), false, false);
  // 断言 x 的梯度与全为 1 的 Tensor 相等
  ASSERT_VARIABLE_EQ(x.grad(), torch::ones(x.sizes()));
TEST(CustomAutogradTest, ReentrantPriority) {
  // 静态变量，用于记录 backward 调用的顺序
  static std::vector<int> order;

  // 自定义函数对象 MyFunction，继承自 Function 类模板
  struct MyFunction : public Function<MyFunction> {
    // 前向传播函数，接收 AutogradContext 和输入变量 x
    static Variable forward(AutogradContext*, Variable x) {
      // 前向传播直接返回输入变量 x
      return x;
    }

    // 反向传播函数，接收 AutogradContext 和梯度 grad
    static variable_list backward(AutogradContext*, variable_list grad) {
      // 将序号 0 添加到 order 向量中
      order.push_back(0);
      // 返回梯度 grad
      return grad;
    }
  };

  // 自定义函数对象 Reenter，继承自 Function 类模板
  struct Reenter : public Function<Reenter> {
    // 前向传播函数，接收 AutogradContext 和输入变量 x
    static Variable forward(AutogradContext* ctx, Variable x) {
      {
        // 开启自动求导模式
        at::AutoGradMode enable_grad(true);
        // 使用输入变量 x 的数据创建可变变量，并减去 1，存储在 AutogradContext 中
        ctx->saved_data["x"] = make_variable(x.tensor_data(), true) - 1;
      }
      // 返回 AutogradContext 中存储的变量 x 的张量数据，并且分离出来（不再跟踪梯度）
      return ctx->saved_data["x"].toTensor().detach();
    }

    // 反向传播函数，接收 AutogradContext 和梯度 grad_output
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 如果存储在 AutogradContext 中的张量 x 不为零
      if (!at::native::is_nonzero(ctx->saved_data["x"].toTensor())) {
        // 直接返回梯度 grad_output
        return grad_output;
      }
      {
        // 开启自动求导模式
        at::AutoGradMode enable_grad(true);
        // 对存储在 AutogradContext 中的张量 x 的函数应用，并计算其第一个元素的和的梯度
        apply(ctx->saved_data["x"].toTensor())[0].sum().backward();
        // 返回梯度 grad_output
        return grad_output;
      }
    }
  };

  // 创建一个张量 v，包含单个元素 8193，数据类型为浮点型，并且要求跟踪其梯度
  auto v =
      torch::tensor({8193}, torch::dtype(torch::kFloat).requires_grad(true));
  // 调用 DeepReenter 的 apply 方法，对张量 v 进行前向传播和反向传播
  DeepReenter::apply(v).sum().backward();
}
    // 定义静态方法 `backward`，接收自动微分上下文 `ctx` 和梯度输出变量列表 `grad_output`
    static variable_list backward(
        AutogradContext* ctx,
        variable_list grad_output) {
      // 将数值 1 添加到全局顺序记录列表 `order` 中
      order.push_back(1);
      // 检查保存在上下文中的张量数据 `x` 是否为非零
      if (!at::native::is_nonzero(ctx->saved_data["x"].toTensor())) {
        // 如果 `x` 是零，则直接返回梯度输出变量列表 `grad_output`
        return grad_output;
      }
      {
        // 开启自动梯度计算模式
        at::AutoGradMode enable_grad(true);
        // 应用保存在上下文中的张量 `x`，取第一个元素并求和，然后进行反向传播
        apply(ctx->saved_data["x"].toTensor())[0].sum().backward();
        // 返回梯度输出变量列表 `grad_output`
        return grad_output;
      }
    }
  };

  // 调用 MyFunction 的静态方法 `apply`，传入一个包含数值 6 的张量，并设置需要梯度计算
  auto a = MyFunction::apply(
      torch::tensor({6}, torch::dtype(torch::kFloat).requires_grad(true)));
  // 调用 Reenter 的静态方法 `apply`，传入一个包含数值 9 的张量，并设置需要梯度计算
  auto b = Reenter::apply(
      torch::tensor({9}, torch::dtype(torch::kFloat).requires_grad(true)));
  // 计算张量 `a` 和 `b` 的乘积 `v`
  auto v = a * b;
  // 对乘积 `v` 进行反向传播计算梯度
  v.backward();

  // 断言：所有的重新进入任务应优先于 MyFunction 的反向传播任务
  ASSERT_EQ(order.size(), 10);
  ASSERT_EQ(std::count(order.begin(), order.end(), 1), 9);
  ASSERT_EQ(order.back(), 0);
  // 清空静态全局变量 `order`，以防测试在循环中执行时受到影响
  order.clear();
}

TEST(CustomAutogradTest, Hooks) {
  // 创建一个 5x5 的张量 x，所有元素为1，并需要计算梯度
  Variable x = torch::ones({5, 5}, torch::requires_grad());
  // 创建一个 5x5 的张量 y，所有元素为4，并标记需要计算梯度
  Variable y = torch::ones({5, 5}) * 4;
  y.set_requires_grad(true);

  // 计数器，用于记录 backward hook 被调用的次数
  int counter = 0;

  // 定义一个 backward hook 函数 bw_hook，它会增加计数器的值
  std::function<void(int, Variable)> bw_hook(
      [&counter](int inc, Variable grad) { counter += inc; });

  // 创建张量 z，通过复杂的运算得到结果
  Variable z = x * x + x * 2 + x * y + y;
  // 将 bw_hook 注册为 x 的 backward hook 函数
  x.register_hook([&bw_hook](Variable grad) { bw_hook(0, grad); });
  // 将 bw_hook 注册为 z 的 backward hook 函数，保存 hook 对象到 hook_1
  auto hook_1 =
      z.register_hook([&bw_hook](Variable grad) { bw_hook(1, grad); });
  // 执行反向传播，计算梯度，同时保留计算图并执行所有 backward hooks
  z.backward(torch::ones({5, 5}), true, true);
  // 断言计数器的值为1
  ASSERT_EQ(counter, 1);

  // 将 bw_hook 注册为 z 的 backward hook 函数，保存 hook 对象到 hook_2
  auto hook_2 =
      z.register_hook([&bw_hook](Variable grad) { bw_hook(2, grad); });
  // 再次执行反向传播，计算梯度，同时保留计算图并执行所有 backward hooks
  z.backward(torch::ones({5, 5}), true, true);
  // 断言计数器的值为4
  ASSERT_EQ(counter, 4);

  // 移除 hook_2
  z.remove_hook(hook_2);
  // 再次执行反向传播，计算梯度，同时保留计算图并执行所有 backward hooks
  z.backward(torch::ones({5, 5}), true, true);
  // 断言计数器的值为5
  ASSERT_EQ(counter, 5);

  // 定义一个修改梯度的 backward hook 函数 bw_hook_modify
  std::function<Variable(Variable)> bw_hook_modify(
      [](Variable grad) { return grad.mul(2); });

  // 移除 hook_1
  z.remove_hook(hook_1);
  // 注册 bw_hook_modify 为 z 的 backward hook 函数
  z.register_hook(bw_hook_modify);
  // 清零 y 的梯度
  y.grad().zero_();
  // 再次执行反向传播，计算梯度，不保留计算图，并执行所有 backward hooks
  z.backward(torch::ones({5, 5}), true, false);
  // 断言 y 的梯度是否与预期相等
  ASSERT_VARIABLE_EQ(y.grad(), (x + 1) * 2);

  // 注册 bw_hook_modify 为 y 的 backward hook 函数
  y.register_hook(bw_hook_modify);
  // 清零 y 的梯度
  y.grad().zero_();
  // 再次执行反向传播，不保留计算图，并执行所有 backward hooks
  z.backward(torch::ones({5, 5}), false, false);
  // 断言 y 的梯度是否与预期相等
  ASSERT_VARIABLE_EQ(y.grad(), (x + 1) * 4);

  // 断言移除 hook 的操作是否抛出预期的异常信息
  ASSERT_THROWS_WITH(y.remove_hook(3), "Invalid index");
}

TEST(CustomAutogradTest, HooksInplace) {
  // 克隆一个张量 a，并标记需要计算梯度
  auto a = torch::ones({5, 5}, torch::requires_grad()).clone();

  // 定义计数器 hook1_count，用于记录 hook1 的调用次数
  int hook1_count = 0;
  // 定义 hook1，用于检查梯度是否为 torch::ones({5, 5}) * 2
  auto hook1 = ([&hook1_count](Variable grad) {
    hook1_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 2);
  });

  // 定义计数器 hook2_count，用于记录 hook2 的调用次数
  int hook2_count = 0;
  // 定义 hook2，用于检查梯度是否为 torch::ones({5, 5})
  auto hook2 = ([&hook2_count](Variable grad) {
    hook2_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}));
  });

  // 将 hook1 注册为 a 的 backward hook 函数
  a.register_hook(hook1);
  // 对 a 执行原地乘法操作，即将所有元素乘以2
  a.mul_(2);
  // 将 hook2 注册为 a 的 backward hook 函数
  a.register_hook(hook2);

  // 计算 (a + 1) 的所有元素之和，并进行反向传播
  auto out = (a + 1).sum();
  out.backward();

  // 断言 hook1 的调用次数为1
  ASSERT_EQ(hook1_count, 1);
  // 断言 hook2 的调用次数为1
  ASSERT_EQ(hook2_count, 1);
}

TEST(CustomAutogradTest, HooksInplaceWithRetainsGrad) {
  // 克隆一个张量 a，并标记需要计算梯度
  auto a = torch::ones({5, 5}, torch::requires_grad()).clone();

  // 定义计数器 hook1_count，用于记录 hook1 的调用次数
  int hook1_count = 0;
  // 定义 hook1，用于检查梯度是否为 torch::ones({5, 5}) * 2
  auto hook1 = ([&hook1_count](Variable grad) {
    hook1_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 2);
  });

  // 定义计数器 hook2_count，用于记录 hook2 的调用次数
  int hook2_count = 0;
  // 定义 hook2，用于检查梯度是否为 torch::ones({5, 5}) * 2
  auto hook2 = ([&hook2_count](Variable grad) {
    hook2_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 2);
  });

  // 定义计数器 hook3_count，用于记录 hook3 的调用次数
  int hook3_count = 0;
  // 定义 hook3，用于检查梯度是否为 torch::ones({5, 5})
  auto hook3 = ([&hook3_count](Variable grad) {
    hook3_count++;
    ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}));
  });

  // 将 hook1 注册为 a 的 backward hook 函数
  a.register_hook(hook1);
  // 保留 a 的梯度
  a.retain_grad();
  // 将 hook2 注册为 a 的 backward hook 函数
  a.register_hook(hook2);

  // 对 a 执行原地乘法操作，即将所有元素乘以2
  a.mul_(2);
  // 将 hook3 注册为 a 的 backward hook 函数
  a.register_hook(hook3);

  // 计算 (a + 1) 的所有元素之和，并进行反向传播
  auto out = (a + 1).sum();
  out.backward();

  // 断言 hook1 的调用次数为1
  ASSERT_EQ(hook1_count, 1);
  // 断言 hook2 的调用次数为1
  ASSERT_EQ(hook2_count, 1);
  // 断言 hook3 的调用次数为1
  ASSERT_EQ(hook3_count, 1);

  // 断言 a 是否保留了梯度信息
  ASSERT_TRUE(a.retains_grad());
  // 断言 a 的梯度是否为 torch::ones({5, 5})
  ASSERT_VARIABLE_EQ(a.grad(), torch::ones({5, 5}));
}

TEST(CustomAutogradTest, HooksInplaceTwiceWithRetainsGrad) {
  // 克隆一个张量 a，并
  // 声明并初始化 hook1_count 变量，用于计数 hook1 函数被调用的次数
  hook1_count++;

  // 断言 grad 变量的值等于一个 5x5 的张量，每个元素为 4
  ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 4);
});

// 声明并初始化 hook2_count 变量，用于计数 hook2 函数被调用的次数
int hook2_count = 0;

// 创建一个 lambda 函数 hook2，捕获 hook2_count 变量的引用，实现 hook2 的功能
auto hook2 = ([&hook2_count](Variable grad) {
  // hook2 被调用时，增加 hook2_count 计数
  hook2_count++;
  // 断言 grad 变量的值等于一个 5x5 的张量，每个元素为 4
  ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}) * 4);
});

// 声明并初始化 hook3_count 变量，用于计数 hook3 函数被调用的次数
int hook3_count = 0;

// 创建一个 lambda 函数 hook3，捕获 hook3_count 变量的引用，实现 hook3 的功能
auto hook3 = ([&hook3_count](Variable grad) {
  // hook3 被调用时，增加 hook3_count 计数
  hook3_count++;
  // 断言 grad 变量的值等于一个 5x5 的张量，每个元素为 1
  ASSERT_VARIABLE_EQ(grad, torch::ones({5, 5}));
});

// 将 hook1 注册为 a 的钩子函数
a.register_hook(hook1);
// 保留 a 的梯度信息
a.retain_grad();
// 将 hook2 注册为 a 的钩子函数
a.register_hook(hook2);

// 对 a 进行两次乘法操作
a.mul_(2);
a.mul_(2);
// 将 hook3 注册为 a 的钩子函数
a.register_hook(hook3);

// 计算 (a + 1) 的和，并执行反向传播
auto out = (a + 1).sum();
out.backward();

// 断言 hook1_count 变量的值为 1
ASSERT_EQ(hook1_count, 1);
// 断言 hook2_count 变量的值为 1
ASSERT_EQ(hook2_count, 1);
// 断言 hook3_count 变量的值为 1
ASSERT_EQ(hook3_count, 1);

// 断言 a 保留了梯度信息
ASSERT_TRUE(a.retains_grad());
// 断言 a 的梯度值为一个 5x5 的张量，每个元素为 1
ASSERT_VARIABLE_EQ(a.grad(), torch::ones({5, 5}));
}

// 定义测试用例 CustomAutogradTest 中的 HookNone 测试
TEST(CustomAutogradTest, HookNone) {
  // 定义 NoneGradientFunction 结构体，继承自 Function<NoneGradientFunction>
  struct NoneGradientFunction : public Function<NoneGradientFunction> {
    // 前向传播函数，接受 AutogradContext 指针和变量 x、y，返回变量列表
    static variable_list forward(AutogradContext* ctx, Variable x, Variable y) {
      return {x, y};
    }

    // 反向传播函数，接受 AutogradContext 指针和梯度变量列表 grad，返回变量列表
    static variable_list backward(AutogradContext* ctx, variable_list grad) {
      return {grad[0], Variable()};
    }
  };

  // 初始化标志变量 was_called
  bool was_called = false;

  // 定义 lambda 表达式 hook，用于检查梯度变量是否定义，并设置 was_called 为 true
  auto hook = ([&was_called](Variable grad) {
    ASSERT_TRUE(grad.defined());
    was_called = true;
  });

  // 创建随机张量 x 和 y，并标记 x 需要计算梯度
  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto y = torch::randn({5, 5});

  // 调用 NoneGradientFunction 的 apply 方法进行前向传播
  auto out = NoneGradientFunction::apply(x, y);
  // 提取 x 和 y 的第一个元素作为变量 rx 和 ry
  Variable rx = x[0], ry = x[1];

  // 为 rx 和 ry 注册 hook lambda 表达式
  rx.register_hook(hook);
  ry.register_hook(hook);

  // 计算 (rx + ry) 的和，并进行反向传播
  (rx + ry).sum().backward();

  // 断言 was_called 已经被设置为 true
  ASSERT_TRUE(was_called);
}

// 定义测试用例 CustomAutogradTest 中的 BackwardWithInputs 测试
TEST(CustomAutogradTest, BackwardWithInputs) {
  // 创建随机张量 x 和 y，并标记需要计算梯度
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());

  // 定义 z 作为 x * x + x * y + y * y 的计算结果
  Variable z = x * x + x * y + y * y;
  // 定义 x 和 y 的梯度期望值
  Variable x_grad_expected = 2 * x + y;
  Variable y_grad_expected = x + 2 * y;

  // 对 z 进行反向传播，并传入梯度 torch::ones({5, 5})，不创建计算图，不保留中间结果，仅关注变量 x 的梯度
  z.backward(torch::ones({5, 5}), false, false, {x});

  // 断言变量 x 的梯度与期望值 x_grad_expected 相等
  ASSERT_VARIABLE_EQ(x.grad(), x_grad_expected);
  // 断言变量 y 的梯度未定义
  ASSERT_FALSE(y.grad().defined());
}

// 定义测试用例 CustomAutogradTest 中的 BackwardWithEmptyInputs 测试
TEST(CustomAutogradTest, BackwardWithEmptyInputs) {
  // 创建随机张量 x 和 y，并标记需要计算梯度
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());

  // 定义 z 作为 x * x + x * y + y * y 的计算结果
  Variable z = x * x + x * y + y * y;
  // 定义 x 和 y 的梯度期望值
  Variable x_grad_expected = 2 * x + y;
  Variable y_grad_expected = x + 2 * y;

  // 断言在传入空的变量列表时，调用 z 的反向传播会抛出异常 "cannot be empty"
  ASSERT_THROWS_WITH(
      z.backward(torch::ones({5, 5}), false, false, std::vector<Variable>{}),
      "cannot be empty");
}

// 定义测试用例 CustomAutogradTest 中的 BackwardWithNonLeafInputs 测试
TEST(CustomAutogradTest, BackwardWithNonLeafInputs) {
  // 创建随机张量 x 和 y，并标记需要计算梯度
  Variable x = torch::randn({5, 5}, torch::requires_grad());
  Variable y = torch::randn({5, 5}, torch::requires_grad());

  // 定义 z 和 w 的计算过程
  Variable z = x * x;
  Variable w = y * z + x * y + y * y;

  // 定义 x 和 z 的梯度期望值
  Variable x_grad_expected = 2 * x * y + y;
  Variable z_grad_expected = y;

  // 对 w 进行反向传播，并传入梯度 torch::ones({5, 5})，不创建计算图，不保留中间结果，仅关注变量 x 和 z 的梯度
  w.backward(torch::ones({5, 5}), false, false, std::vector<Variable>{x, z});

  // 断言变量 x 的梯度与期望值 x_grad_expected 相等
  ASSERT_VARIABLE_EQ(x.grad(), x_grad_expected);
  // 断言变量 z 的梯度与期望值 z_grad_expected 相等
  ASSERT_VARIABLE_EQ(z.grad(), z_grad_expected);
  // 断言变量 y 的梯度未定义
  ASSERT_FALSE(y.grad().defined());
}

// 定义测试用例 CustomAutogradTest 中的 BackwardWithCreateGraphWarns 测试
TEST(CustomAutogradTest, BackwardWithCreateGraphWarns) {
  // 设置警告工具始终警告
  c10::WarningUtils::WarnAlways guard(true);

  // 创建随机张量 x，并标记需要计算梯度
  torch::Tensor x = torch::randn({5, 5}).set_requires_grad(true);
  // 计算 z = x * x
  auto z = x * x;

  {
    // 捕获警告信息
    WarningCapture warnings;
    // 调用 z 的反向传播，并传入梯度 torch::ones({5, 5})，创建计算图
    z.backward(torch::ones({5, 5}), c10::nullopt, true);
    // 断言警告信息中包含 "Using backward() with create_graph=True"
    ASSERT_TRUE(
        warnings.str().find("Using backward() with create_graph=True") !=
        std::string::npos);
  }

  {
    // 捕获警告信息
    WarningCapture warnings;
    // 使用 torch::autograd::backward 调用 z 的反向传播，并传入梯度 torch::ones({5, 5})，创建计算图
    torch::autograd::backward({z}, {torch::ones({5, 5})}, c10::nullopt, true);
    // 断言警告信息中包含 "Using backward() with create_graph=True"
    ASSERT_TRUE(
        warnings.str().find("Using backward() with create_graph=True") !=
        std::string::npos);
  }
}
/**
 * Tests for AutogradNotImplementedFallback
 * - Check that we created the NotImplemented kernel when inputs require grad
 *   but when no inputs require grad, we should not create this node
 * - check_inplace logic
 * - view ops
 * - TODO: Tests for debug-only checks? Don't need for now because CI doesn't
 * test non-NDEBUG builds.
 * - tensorlist input and output
 * - multiple outputs / non-tensor output
 * - rebase_history vs set_history
 */
namespace {

/**
 * Performs an inplace operation on a tensor.
 *
 * @param self Tensor on which the operation is performed.
 * @param other Tensor containing values to be added inplace.
 * @return Modified tensor after adding 'other' inplace.
 */
torch::Tensor inplace_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return self.add_(other);
}

/**
 * Performs two inplace operations on tensors and returns the modified tensors.
 *
 * @param self First tensor to be operated on.
 * @param other Second tensor to be operated on.
 * @return Tuple containing two modified tensors after inplace operations.
 */
std::tuple<torch::Tensor, torch::Tensor> two_arg_inplace_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  other.add_(self);
  self.add_(other);
  return std::tuple<torch::Tensor, torch::Tensor>(self, other);
}

/**
 * Performs view operations on two tensors.
 *
 * @param self First tensor for view operation.
 * @param other Second tensor for view operation.
 * @return Tuple of tensors after view operation, which is not allowed and
 *         expected to raise an error.
 */
std::tuple<torch::Tensor, torch::Tensor> two_pairs_of_view_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  // This is not allowed. We test below that this calling into the boxed kernel
  // will raise an error
  return std::tuple<torch::Tensor, torch::Tensor>(self, other);
}

/**
 * Performs view operation on tensors, which is not allowed and expected
 * to raise an error.
 *
 * @param self First tensor for view operation.
 * @param other Second tensor for view operation.
 * @return Tuple of tensors after view operation, which is not allowed and
 *         expected to raise an error.
 */
std::tuple<torch::Tensor, torch::Tensor> non_first_view_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  // This is not allowed. We test below that this calling into the boxed kernel
  // will raise an error
  return std::tuple<torch::Tensor, torch::Tensor>(self.clone(), other);
}

/**
 * Returns a single non-tensor value.
 *
 * @param self First tensor.
 * @param other Second tensor.
 * @return A constant integer value (12).
 */
int64_t ret_single_non_tensor(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return 12;
}

/**
 * Performs an optional operation on a tensor.
 *
 * @param self First tensor.
 * @param other Optional second tensor, may be null.
 * @return Sum of 'self' and 'other' if 'other' is valid, otherwise a clone of 'self'.
 */
torch::Tensor opt_op(
    const torch::Tensor& self,
    const std::optional<at::Tensor>& other) {
  if (other.has_value()) {
    return self + other.value();
  } else {
    return self.clone();
  }
}

/**
 * Performs a custom operation on two tensors.
 *
 * @param self First tensor.
 * @param other Second tensor.
 * @return Tensor resulting from addition of 'self' and 'other'.
 */
torch::Tensor my_custom_op(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return self + other;
}

/**
 * Returns a tuple of tensors and a non-tensor value.
 *
 * @param self First tensor.
 * @param other Second tensor.
 * @return Tuple containing two tensors and a constant integer value (12).
 */
std::tuple<torch::Tensor, torch::Tensor, int64_t> ret_tuple_non_tensor(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  auto a = self - other;
  auto b = self + other;
  return std::tuple<torch::Tensor, torch::Tensor, int64_t>(a, b, 12);
}

/**
 * Performs a view operation on a tensor.
 *
 * @param self Tensor to create an alias of.
 * @return Alias tensor of 'self'.
 */
torch::Tensor view_op(const torch::Tensor& self) {
  return self.alias();
}

/**
 * Performs a view operation on a tensor with an additional argument.
 *
 * @param self First tensor.
 * @param other Second tensor (unused in this operation).
 * @return Alias tensor of 'self'.
 */
torch::Tensor view_op_with_extra_arg(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return self.alias();
}

/**
 * Returns a vector of alias tensors.
 *
 * @param self First tensor.
 * @param other Second tensor.
 * @return Vector containing alias tensors of 'self'.
 */
std::vector<torch::Tensor> ret_tensor_vector_view(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return {self.alias(), self.alias()};
}

/**
 * Returns a vector of tensors.
 *
 * @param self First tensor.
 * @param other Second tensor.
 * @return Vector of tensors computed by adding and subtracting 'self' and 'other'.
 */
std::vector<at::Tensor> ret_tensor_vector(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  std::vector<at::Tensor> out;
  out.push_back(self + other);
  out.push_back(self - other);
  return out;
}

/**
 * Performs an operation using a list of tensors.
 *
 * @param self First tensor.
 * @param other List of tensors to be added to 'self'.
 * @return Resultant tensor after adding all tensors in 'other' to 'self'.
 */
torch::Tensor tensorlist_op(const torch::Tensor& self, at::TensorList other) {
  const auto& res = self.clone();
  for (const auto& t : other) {
    res.add_(t);
  }
  return res;
}

} // namespace
#define REGISTER_TEST_OP(name, schema, fn)                                 \
  auto m = MAKE_TORCH_LIBRARY(_test);                                      \
  // 创建名为 m 的 Torch 库，用于注册测试操作
  m.def(schema);                                                           \
  // 在库 m 中定义指定的 schema
  auto m_autograd = MAKE_TORCH_LIBRARY_IMPL(_test, Autograd);              \
  // 创建名为 m_autograd 的 Torch 自动求导库
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);                        \
  // 创建名为 m_cpu 的 Torch CPU 库
  auto m_inplaceorview = MAKE_TORCH_LIBRARY_IMPL(_test, ADInplaceOrView);  \
  // 创建名为 m_inplaceorview 的 Torch ADInplaceOrView 库
  m_cpu.impl(name, c10::DispatchKey::CPU, TORCH_FN(fn));                   \
  // 在 m_cpu 库中实现名为 name 的操作，指定 CPU 调度键和实现函数 fn
  m_autograd.impl(                                                         \
      name, c10::DispatchKey::Autograd, autogradNotImplementedFallback()); \
  // 在 m_autograd 库中实现名为 name 的操作，指定 Autograd 调度键和自动求导未实现的回退函数
  m_inplaceorview.impl(                                                    \
      name,                                                                \
      c10::DispatchKey::ADInplaceOrView,                                   \
      autogradNotImplementedInplaceOrViewFallback());
  // 在 m_inplaceorview 库中实现名为 name 的操作，指定 ADInplaceOrView 调度键和自动求导不支持的就地操作回退函数

template <typename F>
void assertBasicChecks(F op) {
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});
  auto c = torch::tensor({1.}, {torch::kFloat32});

  // If any inputs require grad,
  auto out1 = op(a, b);
  // 调用 op 函数，传入 a 和 b，计算结果存储在 out1 中
  ASSERT_THROWS_WITH(out1.backward(), "is not implemented");
  // 断言 out1 调用 backward() 时抛出指定异常信息

  // # Should not have grad_fn if none require grad
  auto out2 = op(b, c);
  // 调用 op 函数，传入 b 和 c，计算结果存储在 out2 中
  ASSERT_THROWS_WITH(
      out2.backward(),
      "element 0 of tensors does not require grad and does not have a grad_fn");
  // 断言 out2 调用 backward() 时抛出指定异常信息

  // TODO: Forward AD Tests?
}

} // namespace

TEST(TestAutogradNotImplementedFallback, RetSingleNonTensor) {
  REGISTER_TEST_OP(
      "ret_single_non_tensor",
      "_test::ret_single_non_tensor(Tensor self, Tensor other) -> int",
      ret_single_non_tensor);
  // 注册名为 "ret_single_non_tensor" 的测试操作，指定其 schema 和实现函数 ret_single_non_tensor
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_single_non_tensor", "");
  // 获取名为 "_test::ret_single_non_tensor" 的操作的处理句柄
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<int64_t, const torch::Tensor&, const torch::Tensor&>(
        opHandle, _1, _2);
  };
  // 定义 lambda 函数 op，调用指定操作 opHandle 处理句柄进行非盒张量返回类型的操作

  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  ASSERT_EQ(op(a, b), ret_single_non_tensor(a, b));
  // 断言调用 op 函数与直接调用 ret_single_non_tensor 函数得到的结果相等
}

TEST(TestAutogradNotImplementedFallback, InplaceOp) {
  REGISTER_TEST_OP(
      "inplace_op",
      "_test::inplace_op(Tensor(a!) self, Tensor other) -> Tensor(a!)",
      inplace_op);
  // 注册名为 "inplace_op" 的测试操作，指定其 schema 和实现函数 inplace_op
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::inplace_op", "");
  // 获取名为 "_test::inplace_op" 的操作的处理句柄
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    // 定义 lambda 函数 op，调用指定操作 opHandle 处理句柄进行 inplace 操作

    return callOpUnboxed<int64_t, const torch::Tensor&, const torch::Tensor&>(
        opHandle, _1, _2);
  };
  // 调用 callOpUnboxed 函数，传入操作 opHandle 和参数 _1、_2，返回结果

  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  ASSERT_EQ(op(a, b), inplace_op(a, b));
  // 断言调用 op 函数与直接调用 inplace_op 函数得到的结果相等
}
    // 调用一个接受未包装类型参数的函数，并返回结果
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };

  // 创建一个张量 a，包含单个元素 1.0，数据类型为 Float32，并标记为需要梯度
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  // 创建一个张量 b，包含单个元素 1.0，数据类型为 Float32
  auto b = torch::tensor({1.}, {torch::kFloat32});

  // 检查原地操作
  ASSERT_THROWS_WITH(
      op(a, b),
      "a leaf Variable that requires grad is being used in an in-place operation");
  // 调用 op 函数进行原地操作，交换参数 a 和 b
  op(b, a);
  // 克隆张量 a 和 b
  a = a.clone();
  b = b.clone();
  // 调用 op 函数对克隆后的 a 和 b 进行操作，结果保存在 c 中
  auto c = op(a, b);
  // 断言 c 和使用 inplace_op 函数在 a 和 b 上执行操作后的结果是否近似相等
  ASSERT_TRUE(torch::allclose(c, inplace_op(a, b)));

  // 在视图上测试原地操作
  // 创建一个张量 base，包含单个元素 1.0，数据类型为 Float32，并标记为需要梯度，然后进行克隆
  auto base =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();
  // 创建 base 的视图 view
  auto view = base.view(-1);
  // 创建一个张量 t，包含单个元素 1.0，数据类型为 Float32
  auto t = torch::tensor({1.}, {torch::kFloat32});

  // 使用 no_grad 模式下创建张量 v_nograd，作为 base 的视图
  torch::Tensor v_nograd;
  {
    c10::NoGradGuard guard;
    v_nograd = base.view(-1);
    // 在 no_grad 模式下调用 op 函数
    op(v_nograd, t);
  }

  // 断言在 no_grad 模式下调用 op 函数会抛出异常
  ASSERT_THROWS_WITH(op(v_nograd, t), "A view was created in no_grad mode");
  // 断言 op(view, t) 的结果张量与 view 的实现相同
  ASSERT_EQ(op(view, t).unsafeGetTensorImpl(), view.unsafeGetTensorImpl());
  // 断言 op(view, t) 的梯度函数名包含 "AsStridedBackward"
  ASSERT_THAT(
      op(view, t).grad_fn()->name(), ::testing::HasSubstr("AsStridedBackward"));
TEST(TestAutogradNotImplementedFallback, DoubleInplaceOp) {
  // 注册名为 "two_arg_inplace_op" 的测试操作，接受两个张量并进行原地操作
  REGISTER_TEST_OP(
      "two_arg_inplace_op",
      "_test::two_arg_inplace_op(Tensor(a!) self, Tensor(b!) other) -> (Tensor(a!), Tensor(b!))",
      two_arg_inplace_op);
  // 获取操作的句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::two_arg_inplace_op", "");
  // 定义 lambda 函数 op，用于调用未打包的操作
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  // 创建两个张量 a 和 b，设置其中一个需要梯度
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  // 断言在原地操作时抛出异常，因为包含需要梯度的叶子变量
  ASSERT_THROWS_WITH(
      op(a, b),
      "a leaf Variable that requires grad is being used in an in-place operation");
  ASSERT_THROWS_WITH(
      op(b, a),
      "a leaf Variable that requires grad is being used in an in-place operation");

  // 克隆带有梯度信息的张量 c 和 d
  auto c =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();
  auto d =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();

  // 保存当前版本号
  auto saved_version_c = c._version();
  auto saved_version_d = d._version();
  // 执行操作 op，并断言版本号已更新
  op(c, d);
  ASSERT_NE(c._version(), saved_version_c);
  ASSERT_NE(d._version(), saved_version_d);
}

TEST(TestAutogradNotImplementedFallback, OptOp) {
  // 注册名为 "opt_op" 的测试操作，接受一个张量和一个可选的张量参数
  REGISTER_TEST_OP(
      "opt_op", "_test::opt_op(Tensor self, Tensor? other) -> Tensor", opt_op);
  // 获取操作的句柄
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::opt_op", "");
  // 定义 lambda 函数 op，用于调用未打包的操作
  auto op = [&](const torch::Tensor& _1,
                const std::optional<torch::Tensor>& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const std::optional<torch::Tensor>&>(opHandle, _1, _2);
  };

  // 创建两个张量 a 和 b，设置其中一个需要梯度
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  auto b = torch::tensor({1.}, {torch::kFloat32});

  // 断言使用操作 op 对张量进行操作后结果近似相等
  ASSERT_TRUE(torch::allclose(op(a, b), opt_op(a, b)));
  ASSERT_TRUE(torch::allclose(op(a, {}), opt_op(a, {})));
}

TEST(TestAutogradNotImplementedFallback, OutOfPlaceAddition) {
  // 注册名为 "my_custom_op" 的测试操作，接受两个张量进行非原地加法操作
  REGISTER_TEST_OP(
      "my_custom_op",
      "_test::my_custom_op(Tensor self, Tensor other) -> Tensor",
      my_custom_op);
  // 获取操作的句柄
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::my_custom_op", "");
  // 定义 lambda 函数 op，用于调用未打包的操作
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };

  // 执行基本的检查
  assertBasicChecks(op);
}

TEST(TestAutogradNotImplementedFallback, RetTupleNonTensor) {
  // 注册名为 "ret_tuple_non_tensor" 的测试操作，返回一个元组包含两个张量和一个整数
  REGISTER_TEST_OP(
      "ret_tuple_non_tensor",
      "_test::ret_tuple_non_tensor(Tensor self, Tensor other) -> (Tensor, Tensor, int)",
      ret_tuple_non_tensor);
  // 获取操作的句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_tuple_non_tensor", "");
  // 定义 lambda 函数 op，用于调用未打包的操作
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    //`
    // 调用未包装的操作，传入 opHandle 和两个张量，返回一个包含三个元素的元组，分别是 torch::Tensor, torch::Tensor 和 int64_t
    auto out = callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor, int64_t>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
    // 解构元组 out，将其内容分别赋值给 out0, out1 和 out2
    auto [out0, out1, out2] = std::move(out);
    // 返回元组中的第一个元素，即 torch::Tensor 类型的 out0
    return out0;
  };

  // 执行基本检查，确保操作符合预期条件
  assertBasicChecks(op);
TEST(TestAutogradNotImplementedFallback, ViewOp) {
  // 注册测试操作 'view_op'，指定其签名和函数指针
  REGISTER_TEST_OP(
      "view_op", "_test::view_op(Tensor(a) self) -> Tensor(a)", view_op);
  // 查找 'view_op' 的操作处理器
  auto opHandle =
      c10::Dispatcher::singleton().findSchemaOrThrow("_test::view_op", "");
  // 定义操作函数 op，调用包装的 callOpUnboxed 函数来执行操作
  auto op = [&](const torch::Tensor& _1) {
    return callOpUnboxed<torch::Tensor, const torch::Tensor&>(opHandle, _1);
  };
  // 创建一个 Float32 类型的张量 b
  auto b = torch::tensor({1.}, {torch::kFloat32});
  // 执行操作 op，返回结果 v，并断言 v 是一个视图
  auto v = op(b);
  ASSERT_TRUE(v.is_view());
  // 断言 v 的底层张量实现与 b 相同
  ASSERT_EQ(v._base().unsafeGetTensorImpl(), b.unsafeGetTensorImpl());

  // 创建一个设置了 requires_grad 的克隆张量 b1
  auto b1 =
      torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true).clone();
  // 执行操作 op，返回结果 v1，断言 v1 是一个视图
  auto v1 = op(b1);
  ASSERT_TRUE(v1.is_view());
  // 断言 v1 的底层张量实现与 b1 相同
  ASSERT_EQ(v1._base().unsafeGetTensorImpl(), b1.unsafeGetTensorImpl());

  // 在视图上测试原地操作
  auto t = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);

  // 当它刷新梯度函数时，对 rebase_history 报错
  ASSERT_THROWS_WITH(
      v1.add_(t), "which does not have a derivative implemented is forbidden");
  // 底层不应该意识到视图，所以这仍然可以进行
  b1.add_(t);
  ASSERT_THROWS_WITH(
      v1.grad_fn(),
      "which does not have a derivative implemented is forbidden");
}

TEST(TestAutogradNotImplementedFallback, ViewOpWithExtraArg) {
  // 注册测试操作 'view_op_with_extra_arg'，指定其签名和函数指针
  REGISTER_TEST_OP(
      "view_op_with_extra_arg",
      "_test::view_op_with_extra_arg(Tensor(a) self, Tensor other) -> Tensor(a)",
      view_op_with_extra_arg);
  // 查找 'view_op_with_extra_arg' 的操作处理器
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::view_op_with_extra_arg", "");
  // 定义操作函数 op，调用包装的 callOpUnboxed 函数来执行操作
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        torch::Tensor,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  // 断言基本检查
  assertBasicChecks(op);
  // 创建 Float32 类型的张量 a 和 b
  auto a = torch::tensor({1.}, {torch::kFloat32});
  auto b = torch::tensor({2.}, {torch::kFloat32});
  // 执行操作 op，返回结果 out1，断言 out1 是一个视图
  auto out1 = op(a, b);
  ASSERT_TRUE(out1.is_view());
  // 断言 out1 的底层张量实现与 a 相同
  ASSERT_EQ(out1._base().unsafeGetTensorImpl(), a.unsafeGetTensorImpl());
}

TEST(TestAutogradNotImplementedFallback, RetTensorVectorView) {
  // 注册测试操作 'ret_tensor_vector_view'，指定其签名和函数指针
  REGISTER_TEST_OP(
      "ret_tensor_vector_view",
      "_test::ret_tensor_vector_view(Tensor(a) self, Tensor other) -> Tensor[](a)",
      ret_tensor_vector_view);
  // 查找 'ret_tensor_vector_view' 的操作处理器
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_tensor_vector_view", "");
  // 定义操作函数 op，调用包装的 callOpUnboxed 函数来执行操作
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::vector<at::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  // 创建 Float32 类型的张量 a 和 b
  auto a = torch::tensor({1.}, {torch::kFloat32});
  auto b = torch::tensor({1.}, {torch::kFloat32});
  // 执行操作 op，返回结果 out，out 是一个张量向量，断言 out[0] 和 out[1] 都是视图
  auto out = op(a, b);
  ASSERT_TRUE(out[0].is_view());
  ASSERT_EQ(out[0]._base().unsafeGetTensorImpl(), a.unsafeGetTensorImpl());
  ASSERT_TRUE(out[1].is_view());
  ASSERT_EQ(out[1]._base().unsafeGetTensorImpl(), a.unsafeGetTensorImpl());
}
TEST(TestAutogradNotImplementedFallback, DoubleViewOP) {
  // 注册测试操作"two_pairs_of_view_op"，指定其输入输出参数及函数指针
  REGISTER_TEST_OP(
      "two_pairs_of_view_op",
      "_test::two_pairs_of_view_op(Tensor(a) self, Tensor(b) other) -> (Tensor(a), Tensor(b))",
      two_pairs_of_view_op);
  // 获取操作的句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::two_pairs_of_view_op", "");
  // 定义 lambda 函数 op，调用注册的操作并返回结果
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  // 创建张量 a，设置 requires_grad 标志为 true
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  // 创建张量 b
  auto b = torch::tensor({1.}, {torch::kFloat32});
  // 断言操作 op(a, b) 抛出特定错误信息
  ASSERT_THROWS_WITH(
      op(a, b),
      "Expected only a single output in the operator schema to have a non-write alias annotation");
}

TEST(TestAutogradNotImplementedFallback, NonFirstViewOP) {
  // 注册测试操作"non_first_view_op"，指定其输入输出参数及函数指针
  REGISTER_TEST_OP(
      "non_first_view_op",
      "_test::non_first_view_op(Tensor self, Tensor(b) other) -> (Tensor, Tensor(b))",
      non_first_view_op);
  // 获取操作的句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::non_first_view_op", "");
  // 定义 lambda 函数 op，调用注册的操作并返回结果
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::tuple<torch::Tensor, torch::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2);
  };
  // 创建张量 a，设置 requires_grad 标志为 true
  auto a = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  // 创建张量 b
  auto b = torch::tensor({1.}, {torch::kFloat32});
  // 断言操作 op(a, b) 抛出特定错误信息
  ASSERT_THROWS_WITH(
      op(a, b), "can only create view relationships between the first");
}

TEST(TestAutogradNotImplementedFallback, RetTensorVector) {
  // 注册测试操作"ret_tensor_vector"，指定其输入输出参数及函数指针
  REGISTER_TEST_OP(
      "ret_tensor_vector",
      "_test::ret_tensor_vector(Tensor self, Tensor other) -> Tensor[]",
      ret_tensor_vector);
  // 获取操作的句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::ret_tensor_vector", "");
  // 定义 lambda 函数 op，调用注册的操作并返回结果中的第一个张量
  auto op = [&](const torch::Tensor& _1, const torch::Tensor& _2) {
    return callOpUnboxed<
        std::vector<at::Tensor>,
        const torch::Tensor&,
        const torch::Tensor&>(opHandle, _1, _2)[0];
  };
  // 调用基本检查函数 assertBasicChecks，传入操作 op
  assertBasicChecks(op);
}

TEST(TestAutogradNotImplementedFallback, TensorlistOp) {
  // 注册测试操作"tensorlist_op"，指定其输入输出参数及函数指针
  REGISTER_TEST_OP(
      "tensorlist_op",
      "_test::tensorlist_op(Tensor self, Tensor[] other) -> Tensor",
      tensorlist_op);
  // 获取操作的句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::tensorlist_op", "");
  // 定义 lambda 函数 op，调用注册的操作并返回结果
  auto op = [&](torch::Tensor _1, at::TensorList _2) {
    // 调用一个返回未装箱的 torch::Tensor 的函数模板，并传入相应参数
    return callOpUnboxed<torch::Tensor, const torch::Tensor&, at::TensorList>(
        opHandle, _1, _2);
  };

  // 创建一个包含单个浮点数 1.0 的张量 a，并指定数据类型为 torch::kFloat32
  auto a = torch::tensor({1.}, {torch::kFloat32});
  // 创建一个包含单个浮点数 1.0 的张量 b，并指定数据类型为 torch::kFloat32
  auto b = torch::tensor({1.}, {torch::kFloat32});
  // 创建一个包含单个浮点数 1.0 的张量 c，并指定数据类型为 torch::kFloat32，并设置 requires_grad 为 true
  auto c = torch::tensor({1.}, {torch::kFloat32}).set_requires_grad(true);
  // 创建一个包含张量 b 和 c 的张量向量 vec
  std::vector<torch::Tensor> vec = {b, c};
  // 调用 op 函数，传入张量 a 和张量向量 vec，并将结果存储在 out 中
  auto out = op(a, vec);

  // 断言：调用 torch::autograd::grad 函数，对 out 求导，期望抛出异常 "element 0 of the input tensors does not require grad"
  ASSERT_THROWS_WITH(
      torch::autograd::grad({out}, {vec[0]}),
      "element 0 of the input tensors does not require grad");
  // 断言：调用 torch::autograd::grad 函数，对 out 求关于 vec[1] 的导数，期望抛出异常 "is not implemented"
  ASSERT_THROWS_WITH(
      torch::autograd::grad({out}, {vec[1]}), "is not implemented");

  // 断言：验证 op(a, vec) 和 tensorlist_op(a, vec) 的输出是否在误差允许范围内相等
  ASSERT_TRUE(at::allclose(op(a, vec), tensorlist_op(a, vec)));
}
// 结束当前的代码块或函数定义，这里是一个单独的右花括号

// TODO 如果需要的话，添加以下测试
// test_once_differentiable
// test_sparse_backward
// test_save_output_nr
// test_free_deep_graph_pyfunction
// test_naughty_anomaly_access
// test_naughty_autograd-function_stashing_ctx
// test_custom_autograd_repeated_grad_grad
// test_return_leaf
// test_anomaly_detect_nan
// test_no_grad_copy
// 这些是待办事项，可能是未来需要添加的测试用例名称的列表
```