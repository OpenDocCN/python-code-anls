# `.\pytorch\test\cpp\tensorexpr\test_ops.cpp`

```py
# 包含 Google Test 的头文件
#include <gtest/gtest.h>
# 包含 TensorExpr 库的评估和表达式相关的头文件
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/operators.h>
# 包含 PyTorch 的头文件
#include <torch/torch.h>

# 使用 torch::jit::tensorexpr 命名空间
using namespace torch::jit::tensorexpr;

# 定义类型别名 Tensors 为 Tensor 的 vector
using Tensors = std::vector<Tensor>;
# 定义类型别名 Args 为 CodeGen::BufferArg 的 vector
using Args = std::vector<CodeGen::BufferArg>;

# 编译函数，接受输入参数和输出张量，返回 SimpleIREvaluator 的唯一指针
std::unique_ptr<SimpleIREvaluator> compile(
    const Args& inputs,
    const Tensors& outputs) {
  # 创建循环嵌套对象，使用给定的输出张量初始化
  LoopNest nest({outputs});
  # 准备代码生成前的准备工作
  nest.prepareForCodegen();
  # 简化循环嵌套
  nest.simplify();
  # 将输入参数和输出张量合并到一个列表中
  auto join = inputs;
  join.insert(join.end(), outputs.begin(), outputs.end());
  # 返回创建的 SimpleIREvaluator 对象
  return std::make_unique<SimpleIREvaluator>(nest.root_stmt(), join);
}

# 单元测试，验证求和操作的正确性
TEST(Ops, Sum) {
  # 常量定义：矩阵尺寸 M=8, N=16
  constexpr int M = 8;
  constexpr int N = 16;
  # 测试维度列表
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};
  # 输出形状列表
  std::vector<std::vector<ExprHandle>> outputShapes = {{N}, {M}, {}};
  # 遍历测试维度列表
  for (unsigned idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    # 创建缓冲区句柄 a，表示一个 MxN 的浮点数矩阵
    BufHandle a("a", {M, N}, kFloat);
    # 计算输出张量的步长
    std::vector<ExprHandle> outStrides =
        c10::fmap<ExprHandle>(make_contiguous_strides(outShape));
    # 计算张量的和，返回张量 b
    Tensor b = computeSum(
        {a, dims, false}, outShape, outStrides, c10::kFloat, at::kCPU);
    # 编译计算图
    auto cg = compile({a}, {b});

    # 创建张量 at，包含从 0 到 M*N 的浮点数，并按照指定形状视图
    auto at = at::arange(M * N, at::kFloat).view({M, N});
    # 计算参考结果 ref，对张量 at 按给定维度 dims 求和
    auto ref = at::sum(at, dims);
    # 创建张量 bt，与 ref 形状相同
    auto bt = at::empty_like(ref);

    # 调用编译后的计算图，计算结果写入 bt
    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    # 断言：验证 bt 与 ref 是否全部相近
    ASSERT_TRUE(at::allclose(bt, ref));
  }
}

# 单元测试，验证通道最后求和操作的正确性
TEST(Ops, ChannelsLastSum) {
  # 常量定义：五维张量尺寸 A=2, B=3, C=4, D=5, E=6
  constexpr int A = 2;
  constexpr int B = 3;
  constexpr int C = 4;
  constexpr int D = 5;
  constexpr int E = 6;
  # 测试维度列表
  std::vector<IntList> testDims = {{0}, {1}, {0, 1}};
  # 输出形状列表
  std::vector<std::vector<ExprHandle>> outputShapes = {
      {B, C, D, E}, {A, C, D, E}, {C, D, E}};
  # 遍历测试维度列表
  for (unsigned idx = 0; idx < testDims.size(); idx++) {
    const auto& dims = testDims[idx];
    const auto& outShape = outputShapes[idx];

    # 创建缓冲区句柄 a，表示一个 ABCDE 维度的浮点数张量
    BufHandle a("a", {A, B, C, D, E}, kFloat);
    # 计算输出张量的步长
    std::vector<ExprHandle> outStrides =
        c10::fmap<ExprHandle>(make_channels_last_strides(outShape));
    # 计算张量的和，返回张量 b
    Tensor b = computeSum(
        {a, dims, false}, outShape, outStrides, c10::kFloat, at::kCPU);
    # 编译计算图
    auto cg = compile({a}, {b});

    # 创建张量 at，包含从 0 到 ABCDE 的浮点数，并按照指定形状视图
    auto at = at::arange(A * B * C * D * E, at::kFloat).view({A, B, C, D, E});
    # 计算参考结果 ref，对张量 at 按给定维度 dims 求和
    auto ref = at::sum(at, dims);
    # 创建张量 bt，与 ref 形状相同
    auto bt = at::empty_like(ref);

    # 调用编译后的计算图，计算结果写入 bt
    cg->call({at.data_ptr<float>(), bt.data_ptr<float>()});

    # 断言：验证 bt 与 ref 是否全部相近
    ASSERT_TRUE(at::allclose(bt, ref));
  }
}
```