# `.\pytorch\test\cpp\tensorexpr\test_approx.cpp`

```py
#ifdef TORCH_ENABLE_LLVM
#ifdef TORCH_ENABLE_LLVM 检查是否定义了 TORCH_ENABLE_LLVM 宏，用于条件编译

#include <gtest/gtest.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/torch.h>
#include <cstring>
包含必要的头文件，用于测试、TensorExpr IR 简化、LLVM 代码生成、循环嵌套、张量处理等

using namespace torch::indexing;
namespace te = torch::jit::tensorexpr;
使用 torch::indexing 命名空间，并定义别名 te 为 torch::jit::tensorexpr

static void vectorize(te::LoopNest* ln, te::Tensor target, int width) {
  static void vectorize(te::LoopNest* ln, te::Tensor target, int width) {
  定义静态函数 vectorize，用于向量化循环嵌套对象中的目标张量

  auto loops = ln->getLoopStmtsFor(target);
  获取循环嵌套对象 ln 中目标张量 target 的循环语句列表

  te::ForPtr inner, tail;
  te::ForPtr inner, tail;
  定义 te::ForPtr 类型的指针 inner 和 tail，用于表示内部循环和尾部循环

  ln->splitWithTail(loops[0], width, &inner, &tail);
  使用宽度 width 将 loops[0] 中的循环分割为内部循环和尾部循环，存储在 inner 和 tail 中

  ASSERT_TRUE(te::LoopNest::vectorize(inner));
  ASSERT_TRUE(te::LoopNest::vectorize(inner));
  使用 te::LoopNest::vectorize 函数尝试向量化内部循环，并进行断言确认

}

std::string diffs(const at::Tensor& a, const at::Tensor& b) {
std::生成 diffs(const at::Tensor& a, const at::Tensor& b) {
  定义 diffs 函数，计算两个张量 a 和 b 之间的差异，并返回描述性字符串

  auto diff = torch::abs(a.flatten() - b.flatten());
  计算张量 a 和 b 展平后的绝对差异

  auto count_diffs = torch::sum(diff > 0.f);
  计算差异大于 0 的元素数量

  auto greatest_diff_index = torch::argmax(diff);
  计算最大差异的索引位置

  std::stringstream ss;
  创建字符串流 ss 用于构建结果描述

  ss << "Found " << count_diffs << " unequal element(s). "
     << "The greatest difference was " << diff.index({greatest_diff_index})
     << " at index " << greatest_diff_index;
  构建描述性字符串，包括不匹配元素数量、最大差异值及其索引位置

  return ss.str();
  返回生成的描述性字符串
}

TEST(Approx, log_vml) {
测试用例定义：Approx 下的 log_vml

  te::VarHandle N("N", te::kInt);
  创建整型变量 N

  te::BufHandle A("A", {N}, te::kFloat);
  创建浮点数缓冲区 A，维度为 N

  te::Tensor B = te::Compute(
      "B", {N}, [&](const te::VarHandle& i) { return log_vml(A.load(i)); });
  创建张量 B，用于计算 log_vml(A.load(i))，其中 A 是缓冲区，i 是循环变量

  te::LoopNest ln({B});
  创建循环嵌套对象 ln，包含张量 B

  ln.prepareForCodegen();
  准备循环嵌套对象 ln 进行代码生成

  vectorize(&ln, B, 8);
  向量化 ln 中的张量 B，向量化宽度为 8

  te::StmtPtr s = ln.root_stmt();
  获取 ln 的根语句，并存储在 s 中

  s = te::IRSimplifier::simplify(s);
  使用 IRSimplifier 对 s 进行简化

  te::LLVMCodeGen cg(s, {A, B, N});
  创建 LLVMCodeGen 对象 cg，用于将简化后的 IR 代码生成 LLVM IR，包括 A、B、N 作为参数

  auto eps = std::numeric_limits<float>::epsilon();
  设置浮点数精度限制 eps 为最小正浮点数的差值

  auto test = [&](const at::Tensor& A_t) {
  定义测试函数 test，接收张量 A_t 作为参数

    at::Tensor B_ref = at::log(A_t);
    计算参考结果 B_ref，使用 torch::log 函数计算张量 A_t 的对数

    at::Tensor B_t = at::empty_like(A_t);
    创建与 A_t 相同形状的空张量 B_t

    auto ap = A_t.data_ptr<float>();
    获取张量 A_t 的数据指针 ap

    auto bp = B_t.data_ptr<float>();
    获取张量 B_t 的数据指针 bp

    cg.call({ap, bp, A_t.numel()});
    调用 LLVMCodeGen 对象 cg，计算结果并存储到 B_t 中

    // Results should be bit-identical.
    // 断言结果应该是位相同的
    ASSERT_TRUE(torch::allclose(
        B_t, B_ref, /*rtol=*/eps, /*atol=*/0.0f, /*equal_nan=*/true))
        << "Input[:8]\n"
        << A_t.index({Slice(0, 8)}) << "\n"
        << "Test[:8]\n"
        << B_t.index({Slice(0, 8)}) << "\n"
        << "Ref[:8]\n"
        << B_ref.index({Slice(0, 8)}) << diffs(B_t, B_ref);
    检查 B_t 和 B_ref 的所有元素是否非常接近，并在不匹配时输出详细信息

  };

  // Generate every single-precision FP value in [1.0, 2.0).
  生成位于 [1.0, 2.0) 范围内的单精度浮点数值 A_t

  at::Tensor A_t = torch::arange(1.0f, 2.0f, eps);
  使用 torch::arange 函数生成指定范围内的单精度浮点数张量 A_t

  ASSERT_EQ(A_t.numel(), 1 << 23);
  断言 A_t 的元素数量等于 1 左移 23 次方

  test(A_t);
  执行测试函数 test，传入 A_t 进行测试

  test(A_t * 2.0f);
  test(A_t * 0.5f);
  test(A_t * 4.0f);
  test(A_t * 0.25f);
  test(A_t * powf(2.0f, 16));
  test(A_t * powf(2.0f, -16));
  test(A_t * powf(2.0f, 126));
  test(A_t * powf(2.0f, -126));
  test(torch::full({32}, INFINITY));
  test(torch::full({32}, NAN));
  对多种输入进行测试，包括倍数变化、指数变化、特殊值如 INFINITY 和 NAN

  auto min = std::numeric_limits<float>::min();
  auto denorm_min = std::numeric_limits<float>::denorm_min();

  A_t = torch::arange(0.0f, min, denorm_min);
  使用 torch::arange 生成从 0.0 到 min 之间以 denorm_min 为步长的张量 A_t

  ASSERT_EQ(A_t.numel(), 1 << 23);
  断言 A_t 的元素数量等于 1 左移 23 次方

  auto B_ref = at::log(A_t);
  计算 A_t 的对数作为参考结果 B_ref

  auto B_t = at::empty_like(B_ref);
  创建与 B_ref 形状相同的空张量 B_t

  cg.call({A_t.data_ptr<float>(), B_t.data_ptr<float>(), A_t.numel()});
  使用 LLVMCodeGen 对象 cg 计算结果
#endif // TORCH_ENABLE_LLVM
```