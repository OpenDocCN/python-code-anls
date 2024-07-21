# `.\pytorch\test\cpp\tensorexpr\test_conv.cpp`

```
#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/operators/conv2d.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

namespace te = torch::jit::tensorexpr;
namespace F = torch::nn::functional;

#ifdef TORCH_ENABLE_LLVM

// 生成具有少量精度位数的测试数据，以最小化由浮点重排序引起的误差积累。
static at::Tensor genTestData(c10::IntArrayRef args) {
  return at::trunc(at::randn(args) * 256.0f) / 256.0f;
}

// 测试深度可分离卷积的实现
TEST(Conv, DepthwiseConv2D) {
  constexpr int N = 1, C = 72, H = 56, W = 56;  // 定义输入张量的尺寸
  constexpr int K = 72, R = 3, S = 3;           // 定义卷积核的尺寸
  constexpr int kPad = 1, kStride = 2, kGroups = C;  // 定义卷积参数
  constexpr int CperG = C / kGroups;            // 每个分组的通道数

  // 定义输入、权重、偏置的缓冲区
  te::BufHandle input("input", {N, C, H, W}, te::kFloat);
  te::BufHandle weight("weight", {K, CperG, R, S}, te::kFloat);
  te::BufHandle bias("bias", {K}, te::kFloat);

  // 执行深度可分离卷积操作，生成输出张量
  te::Tensor output =
      te::conv2d_depthwise(input, weight, bias, kStride, kPad, kGroups);

  // 创建循环嵌套对象，并对其进行简化和准备以进行代码生成
  te::LoopNest loop({output});
  loop.simplify();
  loop.prepareForCodegen();

  // 使用 LLVM 进行代码生成，传入输入、权重、偏置以及输出张量
  te::LLVMCodeGen cg(loop.root_stmt(), {input, weight, bias, output});

  // 生成测试数据并进行卷积计算，比较结果与参考值
  auto it = genTestData({N, C, H, W});
  auto wt = genTestData({K, CperG, R, S});
  auto bt = genTestData({K});
  auto ref = at::conv2d(it, wt, bt, kStride, kPad, /*dilation=*/1, kGroups);
  auto ot = at::zeros_like(ref);
  cg.call(
      {it.data_ptr<float>(),
       wt.data_ptr<float>(),
       bt.data_ptr<float>(),
       ot.data_ptr<float>()});

  // 使用断言验证生成的输出与参考输出是否全部接近
  ASSERT_TRUE(at::allclose(ref, ot));
}

// 测试不带偏置的深度可分离卷积实现
TEST(Conv, DepthwiseConv2DNoBias) {
  constexpr int N = 1, C = 72, H = 56, W = 56;  // 定义输入张量的尺寸
  constexpr int K = 72, R = 3, S = 3;           // 定义卷积核的尺寸
  constexpr int kPad = 1, kStride = 2, kGroups = C;  // 定义卷积参数
  constexpr int CperG = C / kGroups;            // 每个分组的通道数

  // 定义输入、权重的缓冲区
  te::BufHandle input("input", {N, C, H, W}, te::kFloat);
  te::BufHandle weight("weight", {K, CperG, R, S}, te::kFloat);

  // 执行不带偏置的深度可分离卷积操作，生成输出张量
  te::Tensor output =
      te::conv2d_depthwise(input, weight, kStride, kPad, kGroups);

  // 创建循环嵌套对象，并对其进行简化和准备以进行代码生成
  te::LoopNest loop({output});
  loop.simplify();
  loop.prepareForCodegen();

  // 使用 LLVM 进行代码生成，传入输入、权重以及输出张量
  te::LLVMCodeGen cg(loop.root_stmt(), {input, weight, output});

  // 生成测试数据并进行卷积计算，比较结果与参考值
  auto it = genTestData({N, C, H, W});
  auto wt = genTestData({K, CperG, R, S});
  auto ref =
      at::conv2d(it, wt, at::Tensor(), kStride, kPad, /*dilation=*/1, kGroups);
  auto ot = at::zeros_like(ref);
  cg.call({it.data_ptr<float>(), wt.data_ptr<float>(), ot.data_ptr<float>()});

  // 使用断言验证生成的输出与参考输出是否全部接近
  ASSERT_TRUE(at::allclose(ref, ot));
}

#endif  // TORCH_ENABLE_LLVM

}  // namespace jit
}  // namespace torch
TEST(Conv, DepthwiseConv2DDynamicShapes) {
  // 定义输入变量
  te::VarHandle N_var("N", te::kInt);
  te::VarHandle C_var("C", te::kInt);
  te::VarHandle H_var("H", te::kInt);
  te::VarHandle W_var("W", te::kInt);
  te::VarHandle K_var("K", te::kInt);
  te::VarHandle CperG_var("CperG", te::kInt);
  te::VarHandle R_var("R", te::kInt);
  te::VarHandle S_var("S", te::kInt);
  te::VarHandle kPad_var("kPad", te::kInt);
  te::VarHandle kStride_var("kStride", te::kInt);
  te::VarHandle kGroups_var("kGroups", te::kInt);

  // 定义输入缓冲区和权重缓冲区
  te::BufHandle input("input", {N_var, C_var, H_var, W_var}, te::kFloat);
  te::BufHandle weight("weight", {K_var, CperG_var, R_var, S_var}, te::kFloat);

  // 调用深度可分离卷积函数
  te::Tensor output = te::conv2d_depthwise(
      input,
      weight,
      N_var,
      C_var,
      H_var,
      W_var,
      K_var,
      CperG_var,
      R_var,
      S_var,
      kStride_var,
      kPad_var,
      kGroups_var);

  // 初始化循环嵌套对象
  te::LoopNest loop({output});
  loop.simplify();
  loop.prepareForCodegen();

  // 准备用于代码生成的缓冲区参数
  std::vector<te::CodeGen::BufferArg> buffer_args = {
      input,
      weight,
      N_var,
      C_var,
      H_var,
      W_var,
      K_var,
      CperG_var,
      R_var,
      S_var,
      kPad_var,
      kStride_var,
      kGroups_var,
      output};

  // 使用LLVM进行代码生成
  te::LLVMCodeGen cg(loop.root_stmt(), buffer_args);

  // 定义常量和输入数据
  constexpr int N = 1, C = 72, H = 56, W = 56;
  constexpr int K = 72, R = 3, S = 3;
  constexpr int kPad = 1, kStride = 2, kGroups = C;
  constexpr int CperG = C / kGroups;

  // 生成测试数据
  auto it = genTestData({N, C, H, W});
  auto wt = genTestData({K, CperG, R, S});

  // 调用PyTorch中的conv2d函数生成参考结果
  auto ref =
      at::conv2d(it, wt, at::Tensor(), kStride, kPad, /*dilation=*/1, kGroups);

  // 创建输出张量
  auto ot = at::zeros_like(ref);

  // 准备调用LLVM生成的代码
  std::vector<te::CodeGen::CallArg> call_args = {
      it.data_ptr<float>(),
      wt.data_ptr<float>(),
      N,
      C,
      H,
      W,
      K,
      CperG,
      R,
      S,
      kPad,
      kStride,
      kGroups,
      ot.data_ptr<float>()};

  // 调用LLVM生成的代码进行计算
  cg.call(call_args);

  // 断言生成的结果与参考结果一致
  ASSERT_TRUE(at::allclose(ref, ot));
}

#endif
TEST(Conv, Conv2D) {
  // Input dimensions.
  constexpr int N = 1;    // 定义输入数据的批量大小
  constexpr int C = 3;    // 定义输入数据的通道数
  constexpr int H = 11;   // 定义输入数据的高度
  constexpr int W = 11;   // 定义输入数据的宽度

  // Filter dimensions.
  constexpr int K = 8;    // 定义滤波器的数量（输出通道数）
  constexpr int R = 3;    // 定义滤波器的高度
  constexpr int S = 3;    // 定义滤波器的宽度

  // Output dims.
  constexpr int OH = H - R + 1;  // 计算卷积输出的高度
  constexpr int OW = W - S + 1;  // 计算卷积输出的宽度

  // Compute reference result.
  at::Tensor input = torch::randn({N, C, H, W});     // 生成随机输入数据张量
  at::Tensor filter = torch::randn({K, C, R, S});   // 生成随机滤波器张量
  at::Tensor ref = F::conv2d(input, filter);        // 计算参考的卷积结果

  // Double check the output size is as expected.
  ASSERT_EQ(ref.size(0), N);   // 检查输出张量的批量大小是否正确
  ASSERT_EQ(ref.size(1), K);   // 检查输出张量的通道数（滤波器数量）是否正确
  ASSERT_EQ(ref.size(2), OH);  // 检查输出张量的高度是否正确
  ASSERT_EQ(ref.size(3), OW);  // 检查输出张量的宽度是否正确

  te::BufHandle inputB("input", {N, C, H, W}, te::kFloat);      // 创建输入数据的缓冲区句柄
  te::BufHandle filterB("filter", {K, C, R, S}, te::kFloat);    // 创建滤波器的缓冲区句柄

  te::Tensor conv = te::Reduce(
      "conv",
      {N, K, OH, OW},
      te::Sum(),
      // FIXME: We have to use a `std::vector` parameter here and then unpack
      // it, because we don't have an overload allowing for an arbitrary number
      // of ExprHandle/VarHandle parameters.
      [&](const std::vector<te::VarHandle>& v) {
        auto const& n = v[0];
        auto const& k = v[1];
        auto const& oh = v[2];
        auto const& ow = v[3];
        auto const& c = v[4];
        auto const& r = v[5];
        auto const& s = v[6];
        // FIXME: We have to use `call` and construct a `std::vector` here
        // because the `operator()` overload is only specialized for a small
        // number of arguments.
        return inputB.load(n, c, oh + r, ow + s) * filterB.load(k, c, r, s);
      },
      // FIXME: If you forget one of the reduction dims, you get a segfault.
      // Could that be caught by a verifier?
      {C, R, S});   // 对卷积操作进行定义和实现

  // FIXME: It'd be nice to have a single header that pulls in things like
  // LoopNest, IRSimplifier, etc.
  te::LoopNest loop({conv});    // 创建一个循环嵌套对象，包含卷积操作
  loop.prepareForCodegen();     // 准备用于代码生成
  te::StmtPtr s = loop.root_stmt();   // 获取循环嵌套的根语句
  s = te::IRSimplifier::simplify(s);  // 简化中间表示的语句

  at::Tensor result = at::empty_like(ref);    // 创建一个和参考结果相同形状的空张量
  te::SimpleIREvaluator cg(s, {inputB, filterB, conv});   // 创建简单的中间表示求值器
  cg.call(
      {input.data_ptr<float>(),
       filter.data_ptr<float>(),
       result.data_ptr<float>()});    // 调用中间表示求值器计算结果

  ASSERT_TRUE(at::allclose(ref, result, 1e-3, 1e-3));   // 断言参考结果和计算结果的近似程度
}

} // namespace jit
} // namespace torch
```