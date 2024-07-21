# `.\pytorch\test\cpp\tensorexpr\test_memplanning.cpp`

```
#include <gtest/gtest.h>
#include <test/cpp/tensorexpr/test_base.h>

#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// 定义一个测试类 BufLiveRange，用于测试缓冲区生存期的情况
TEST(BufLiveRange, SingleRangeLine) {
  // 定义两个整数变量 i 和 j
  VarHandle i("i", kInt), j("j", kInt);
  // 定义两个缓冲区变量 a 和 b，分别为一维和二维的浮点数数组
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32, 32}, kFloat);

  // 构造一个嵌套循环语句：
  // {
  //   for (int i = 0; i < 32; i++) {
  //     a[i] = 0;
  //     for (int j = 0; j < 32; j++) {
  //       a[i] = (a[i]) + (b[i, j]);
  //     }
  //   }
  // }

  // 初始化 a[i] = 0 的存储操作
  StorePtr aInit = Store::make(a, {i}, 0);
  // 计算 a[i] = a[i] + b[i, j] 的表达式
  ExprHandle reduce = a.load({i}) + b.load({i, j});
  // 存储 a[i] = a[i] + b[i, j] 的存储操作
  StorePtr aReduce = Store::make(a, {i}, reduce);
  // 构造循环语句
  StmtPtr loop =
      For::make(i, 0, 32, Block::make({aInit, For::make(j, 0, 32, aReduce)}));

  // 构造整体语句块
  StmtPtr stmt = Block::make({loop});

  // 获取缓冲区 a 的生存期范围
  auto range = BufLiveRange::liveRange(stmt, a.node());
  // 断言生存期起始和结束位置
  ASSERT_TRUE(std::get<0>(range) == 0);
  ASSERT_TRUE(std::get<1>(range) == 0);
}

// 定义第二个测试案例 BufLiveRange，用于测试多个范围的情况
TEST(BufLiveRange, MulRangeLine) {
  // 定义整数变量 i
  VarHandle i("i", kInt);
  // 定义两个缓冲区变量 a 和 b，都是一维的浮点数数组
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32}, kFloat);

  // 构造两个嵌套循环语句：
  // {
  //   for (int i = 0; i < 32; i++) {
  //     if (i<10 ? 1 : 0) {
  //       a[i] = i + i;
  //       b[i] = i * i;
  //     }
  //   }
  //   for (int i = 0; i < 32; i++) {
  //     if (i>10 ? 1 : 0) {
  //       a[i] = i * i;
  //       b[i] = i + i;
  //     }
  //   }
  // }

  // 第一个循环中的存储操作
  StorePtr aStore_1 = Store::make(a, {i}, i + i);
  StorePtr bStore_1 = Store::make(b, {i}, i * i);
  // 第一个循环
  StmtPtr loop_1 = For::make(
      i, 0, 32, Cond::make(i < 10, Block::make({aStore_1, bStore_1}), NULL));

  // 第二个循环中的存储操作
  StorePtr aStore_2 = Store::make(a, {i}, i * i);
  StorePtr bStore_2 = Store::make(b, {i}, i + i);
  // 第二个循环
  StmtPtr loop_2 = For::make(
      i, 0, 32, Cond::make(i > 10, Block::make({aStore_2, bStore_2}), NULL));

  // 构造整体语句块
  StmtPtr stmt = Block::make({loop_1, loop_2});

  // 获取缓冲区 a 的生存期范围
  auto range_a = BufLiveRange::liveRange(stmt, a.node());
  // 断言缓冲区 a 的生存期起始和结束位置
  ASSERT_TRUE(std::get<0>(range_a) == 0);
  ASSERT_TRUE(std::get<1>(range_a) == 1);

  // 获取缓冲区 b 的生存期范围
  auto range_b = BufLiveRange::liveRange(stmt, b.node());
  // 断言缓冲区 b 的生存期起始和结束位置
  ASSERT_TRUE(std::get<0>(range_b) == 0);
  ASSERT_TRUE(std::get<1>(range_b) == 1);
}

} // namespace jit
} // namespace torch
TEST(MemPlanning, MemReuseWithTypeCast) {
  // 定义矩阵维度
  int M = 4;
  int N = 4;
  int K = 4;

  // 创建名为 AP 和 BP 的缓冲区对象，每个对象有指定的维度和数据类型
  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  // 定义张量 CT，表示矩阵乘法的结果
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        // 返回矩阵乘法的计算结果
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});

  // 定义张量 DT，应用 ReLU 激活函数
  Tensor DT =
      Compute("relu", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 根据 CT 张量的值应用 ReLU 函数
        return CompareSelect::make(
            CT.load(m, n), 0.0f, 0.0f, CT.load(m, n), kLT);
      });

  // 定义张量 ET，将 DT 的值转换为无符号 8 位整数类型
  Tensor ET =
      Compute("E", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 将 DT 张量的值类型转换为无符号 8 位整数类型
        return Cast::make(kQUInt8, DT.load(m, n) + DT.load(m, n));
      });

  // 定义张量 FT，复制 ET 张量的值
  Tensor FT =
      Compute("F", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 返回 ET 张量的值
        return ET.load(m, n);
      });

  // 创建一个包含所有计算语句的块语句对象
  StmtPtr stmt =
      tensorexpr::Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  // 为了代码生成做准备
  LoopNest l(stmt, {FT.buf()});
  l.prepareForCodegen();
  // 创建一个简单的 IR 评估器
  SimpleIREvaluator cg(Stmt::clone(l.root_stmt()), {AP, BP, FT});

  // 检查生成的 IR 是否符合预期
  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[4, 4]
# CHECK: Allocate(relu); // dtype=float, dims=[4, 4]
# CHECK: Alias(E,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

  // 创建输入和输出缓冲区对象
  PaddedBuffer<float> a_v(M, K, "a");
  PaddedBuffer<float> b_v(K, N, "b");
  PaddedBuffer<uint8_t> o1(M, N, "e_before");
  PaddedBuffer<uint8_t> o2(M, N, "e_after");

  // 填充输入缓冲区 a_v 和 b_v 的随机值
  for (const auto m : c10::irange(M)) {
    for (const auto k : c10::irange(K)) {
      a_v(m, k) = at::randn({1}).item().to<float>();
    }
  }

  for (const auto k : c10::irange(K)) {
    for (const auto n : c10::irange(N)) {
      b_v(k, n) = at::randn({1}).item().to<float>();
    }
  }

  // 调用代码生成器进行计算
  cg.call({a_v, b_v, o1});

#ifdef TORCH_ENABLE_LLVM
  // 如果启用 LLVM 支持，创建 LLVM 代码生成器
  LLVMCodeGen cg_llvm(Stmt::clone(l.root_stmt()), {AP, BP, FT});

  // 检查生成的 LLVM IR 是否符合预期
  checkIR(cg_llvm.stmt(), R"IR(
// 在这段测试中，定义了三个矩阵的计算过程，gemm、relu 和 E，并且在后续释放了部分内存资源。

// 调用 LLVM 生成的计算图 IR，传入三个缓冲区的句柄作为参数
cg_llvm.call({a_v, b_v, o2});

// 使用 NOLINTNEXTLINE 禁用 lint 提示，比较两个张量 o1 和 o2 是否近似相等
ExpectAllNear(o1, o2, 1e-5);

}

// 测试：对于更大的类型不进行内存重用
TEST(MemPlanning, NoMemReuseForLargerType) {
  // 定义矩阵的维度
  int M = 4;
  int N = 4;
  int K = 4;

  // 创建两个缓冲区对象，AP 和 BP，用于存储矩阵 A 和 B 的数据
  BufHandle AP("A", {M, K}, kShort);
  BufHandle BP("B", {K, N}, kShort);

  // 定义矩阵 CT，其计算方式为矩阵乘法的求和
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});

  // 定义矩阵 DT，使用 relu 函数进行计算
  auto zero = Cast::make(CT.buf()->dtype(), 0);
  Tensor DT =
      Compute("relu", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });

  // 定义矩阵 ET，对矩阵 DT 中的元素进行 float 类型的转换，并且乘以 2
  Tensor ET =
      Compute("E", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return Cast::make(kFloat, DT.load(m, n) + DT.load(m, n));
      });

  // 定义矩阵 FT，其元素与矩阵 ET 中的元素相等
  Tensor FT =
      Compute("F", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return ET.load(m, n);
      });

  // 将 CT、DT、ET、FT 的计算语句合并为一个块语句
  StmtPtr stmt =
      tensorexpr::Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  // 创建循环嵌套对象 l，其中 FT 缓冲区作为参数
  LoopNest l(stmt, {FT.buf()});
  // 准备为代码生成做准备工作
  l.prepareForCodegen();
  // 创建简单的 IR 评估器 cg，传入 l 根语句的克隆和三个缓冲区句柄作为参数
  SimpleIREvaluator cg(Stmt::clone(l.root_stmt()), {AP, BP, FT.buf()});

  // 检查生成的 IR，期望看到 gemm、relu 和 E 的分配及释放操作
  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=int16_t, dims=[4, 4]
# CHECK: Allocate(relu); // dtype=int16_t, dims=[4, 4]
# CHECK: Allocate(E); // dtype=float, dims=[4, 4]
# CHECK: Free(E);
# CHECK: Free(relu);
// CHECK: Free(gemm))IR");

// 创建并初始化大小为 MxK 的 short 类型的缓冲区 a_v，用于存储随机生成的浮点数转换为 short 后的值
PaddedBuffer<short> a_v(M, K, "a");

// 创建并初始化大小为 KxN 的 short 类型的缓冲区 b_v，用于存储随机生成的浮点数转换为 short 后的值
PaddedBuffer<short> b_v(K, N, "b");

// 创建大小为 MxN 的 float 类型的缓冲区 o1，用于存储计算前的结果
PaddedBuffer<float> o1(M, N, "e_before");

// 创建大小为 MxN 的 float 类型的缓冲区 o2，用于存储计算后的结果
PaddedBuffer<float> o2(M, N, "e_after");

// 对于每个 m 在范围 M 内，对于每个 k 在范围 K 内，生成一个随机浮点数，转换为 float 后赋值给 a_v(m, k)
for (const auto m : c10::irange(M)) {
  for (const auto k : c10::irange(K)) {
    a_v(m, k) = at::randn({1}).item().to<float>();
  }
}

// 对于每个 k 在范围 K 内，对于每个 n 在范围 N 内，生成一个随机浮点数，转换为 float 后赋值给 b_v(k, n)
for (const auto k : c10::irange(K)) {
  for (const auto n : c10::irange(N)) {
    b_v(k, n) = at::randn({1}).item().to<float>();
  }
}

// 调用 cg 对象的 call 方法，传入 a_v, b_v, o1 进行计算
cg.call({a_v, b_v, o1});

#ifdef TORCH_ENABLE_LLVM
// 使用 LLVMCodeGen 对象 cg_llvm，克隆 l.root_stmt() 并传入 AP, BP, FT 初始化
LLVMCodeGen cg_llvm(Stmt::clone(l.root_stmt()), {AP, BP, FT});

// 检查 LLVMCodeGen 对象 cg_llvm 生成的 IR 代码，与预期的 IR 代码进行比对
checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=int16_t, dims=[4, 4]
# CHECK: Allocate(relu); // dtype=int16_t, dims=[4, 4]
# CHECK: Allocate(E); // dtype=float, dims=[4, 4]
# CHECK: Free(E);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

// 调用 cg_llvm 对象的 call 方法，传入 a_v, b_v, o2 进行 LLVM IR 代码生成的计算
cg_llvm.call({a_v, b_v, o2});

// 检查 o1 和 o2 的结果是否在 1e-5 的误差范围内相等
// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
ExpectAllNear(o1, o2, 1e-5);
#endif
}
// 定义一个测试函数，测试内存规划中相同缓冲区大小的内存重用
TEST(MemPlanning, SameBufSizeMemReuse) {
  // 定义三个整数变量，分别表示矩阵的维度
  int M = 1024;
  int N = 1024;
  int K = 2048;

  // 创建两个缓冲区句柄对象 AP 和 BP，分别表示矩阵 A 和 B，数据类型为 float
  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  // 定义一个张量 CT，表示矩阵乘法的结果，通过 Reduce 函数计算
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        // 返回矩阵乘法中每个元素的计算表达式
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});
  
  // 定义一个张量 DT，表示对 CT 进行 ReLU 操作的结果
  Tensor DT =
      Compute("relu", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 创建一个表示零的常量
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        // 根据条件比较和选择，实现 ReLU 激活函数
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });

  // 定义一个张量 ET，表示 DT 张量的每个元素自身相加的结果
  Tensor ET =
      Compute("add", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return DT.load(m, n) + DT.load(m, n);
      });

  // 定义一个张量 FT，表示 ET 张量的每个元素自身相乘的结果
  Tensor FT =
      Compute("mul", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return ET.load(m, n) * ET.load(m, n);
      });

  // 创建一个代码块，包含 CT、DT、ET、FT 四个张量的语句
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  // 输出构建好的语句块，包含注释，描述了每个张量在计算中的作用和逻辑流程
  // Constructed stmt:
  // Intermediate buffers and their liveness ranges: gemm [0, 1], relu [1, 2],
  // add [2, 3] Buffer 'gemm' and 'add' are the same size; we'll reuse 'gemm'
  // for 'add'.
  //{
  //  for (int M = 0; M < 1024; M++) {
  //    for (int N = 0; N < 1024; N++) {
  //      gemm[M, N] = float(0);
  //      for (int K = 0; K < 2048; K++) {
  //        gemm[M, N] = ReduceOp((gemm[M, N]) + (A[M, K]) * (B[K, N]),
  //        reduce_args={K});
  //      }
  //    }
  //  }
  //  for (int M_1 = 0; M_1 < 1024; M_1++) {
  //    for (int N_1 = 0; N_1 < 1024; N_1++) {
  //      relu[M_1, N_1] = (gemm[M_1, N_1])<float(0) ? float(0) : (gemm[M_1,
  //      N_1]);
  //    }
  //  }
  //  for (int M_2 = 0; M_2 < 1024; M_2++) {
  //    for (int N_2 = 0; N_2 < 1024; N_2++) {
  //      add[M_2, N_2] = (relu[M_2, N_2]) + (relu[M_2, N_2]);
  //    }
  //  }
  //  for (int M_3 = 0; M_3 < 1024; M_3++) {
  //    for (int N_3 = 0; N_3 < 1024; N_3++) {
  //      mul[M_3, N_3] = (add[M_3, N_3]) * (add[M_3, N_3]);
  //    }
  //  }
  //}

  // 创建一个简单的 IR 评估器 cg，用于执行张量计算并进行内存分配和释放的检查
  SimpleIREvaluator cg(stmt, {AP, BP, FT});

  // 检查生成的 IR 语句，验证内存分配和释放的正确性
  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

#ifdef TORCH_ENABLE_LLVM
  // 如果启用了 LLVM，创建一个循环嵌套对象 loop，并准备进行 LLVM 代码生成
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  loop.prepareForCodegen();
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  // 检查 LLVM 代码生成的 IR，验证内存分配和释放的正确性
  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}
```cpp`
TEST(MemPlanning, SameBufSizeMultiMemReuses) {
  // 定义矩阵维度 M, N, K
  int M = 1024;
  int N = 1024;
  int K = 2048;

  // 创建名为 "A" 的缓冲区，形状为 {M, K}，数据类型为浮点数
  BufHandle AP("A", {M, K}, kFloat);
  // 创建名为 "B" 的缓冲区，形状为 {K, N}，数据类型为浮点数
  BufHandle BP("B", {K, N}, kFloat);

  // 定义一个张量 CT，执行矩阵乘法，形状为 {M, N}
  Tensor CT = Reduce(
      "gemm",       // 操作名为 "gemm"
      {M, N},       // 输出张量形状为 {M, N}
      Sum(),        // 使用求和操作
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        // 定义计算表达式，AP.load(m, k) * BP.load(k, n)
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});         // 迭代变量 K
  // 定义一个张量 DT，执行 ReLU 激活函数，形状为 {M, N}
  Tensor DT =
      Compute("relu", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 将 CT 的元素与 0 进行比较，如果小于零则返回 0，否则返回 CT.load(m, n)
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });
  // 定义一个张量 ET，执行加法操作，形状为 {M, N}
  Tensor ET =
      Compute("add", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 计算 DT.load(m, n) + DT.load(m, n)
        return DT.load(m, n) + DT.load(m, n);
      });
  // 定义一个张量 FT，执行乘法操作，形状为 {M, N}
  Tensor FT =
      Compute("mul", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 计算 ET.load(m, n) * ET.load(m, n)
        return ET.load(m, n) * ET.load(m, n);
      });
  // 定义一个张量 GT，执行减法操作，形状为 {M, N}
  Tensor GT =
      Compute("sub", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        // 计算 FT.load(m, n) - ET.load(m, n)
        return FT.load(m, n) - ET.load(m, n);
      });

  // 创建一个包含所有张量语句的块
  auto stmt =
      Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt(), GT.stmt()});

  // 构建的语句块：
  // 中间缓冲区及其生命周期范围：gemm [0, 1], relu [1, 2], add [2, 3], mul [3, 4]
  // 缓冲区 'gemm'，'relu'，'add' 和 'mul' 大小相同；我们将 'gemm' 重用为 'add'，并将 'relu' 重用为 'mul'
  //{
  //  // 矩阵乘法计算部分，初始化 gemm 缓冲区
  //  for (int M = 0; M < 1024; M++) {
  //    for (int N = 0; N < 1024; N++) {
  //      gemm[M, N] = float(0);
  //      for (int K = 0; K < 2048; K++) {
  //        gemm[M, N] = ReduceOp((gemm[M, N]) + (A[M, K]) * (B[K, N]),
  //        reduce_args={K});
  //      }
  //    }
  //  }
  //  // ReLU 操作部分
  //  for (int M_1 = 0; M_1 < 1024; M_1++) {
  //    for (int N_1 = 0; N_1 < 1024; N_1++) {
  //      relu[M_1, N_1] = (gemm[M_1, N_1])<float(0) ? float(0) : (gemm[M_1,
  //      N_1]);
  //    }
  //  }
  //  // 加法操作部分
  //  for (int M_2 = 0; M_2 < 1024; M_2++) {
  //    for (int N_2 = 0; N_2 < 1024; N_2++) {
  //      add[M_2, N_2] = (relu[M_2, N_2]) + (relu[M_2, N_2]);
  //    }
  //  }
  //  // 乘法操作部分
  //  for (int M_3 = 0; M_3 < 1024; M_3++) {
  //    for (int N_3 = 0; N_3 < 1024; N_3++) {
  //      mul[M_3, N_3] = (add[M_3, N_3]) * (add[M_3, N_3]);
  //    }
  //  }
  //  // 减法操作部分
  //  for (int M_4 = 0; M_4 < 1024; M_4++) {
  //    for (int N_4 = 0; N_4 < 1024; N_4++) {
  //      sub[M_4, N_4] = (mul[M_4, N_4]) - (add[M_4, N_4]);
  //    }
  //  }
  //}

  // 创建一个简单的 IR 执行器，传入构建的语句块和缓冲区 AP, BP, GT
  SimpleIREvaluator cg(stmt, {AP, BP, GT});

  // 检查生成的 IR 语句是否符合预期
  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Alias(mul,relu);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");

#ifdef TORCH_ENABLE_LLVM
  // 创建一个循环嵌套，准备将语句块转换为 LLVM 代码
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  loop.prepareForCodegen();
  // 创建 LLVM 代码生成器，传入循环根语句和缓冲区 AP, BP, FT
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  // 检查生成的 LLVM IR 语句是否符合预期
  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // 分配 relu 变量内存空间；数据类型为 float，维度为 [1024, 1024]
# CHECK: Alias(add,gemm);  // 将 add 别名为 gemm，表示它们引用相同的内存位置
# CHECK: Alias(mul,relu);  // 将 mul 别名为 relu，表示它们引用相同的内存位置
# CHECK: Free(relu);       // 释放 relu 变量所占用的内存空间
# CHECK: Free(gemm))IR");  // 释放 gemm 变量所占用的内存空间，并结束该部分代码的注释
#endif
TEST(MemPlanning, SameBufSizeMultiMemReusesOfOneBuf) {
  // 定义三个整型变量 M, N, K，分别赋值为 1024, 1024, 2048
  int M = 1024;
  int N = 1024;
  int K = 2048;

  // 创建两个缓冲区对象 AP 和 BP，分别表示大小为 {M, K} 和 {K, N} 的浮点数缓冲区
  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  // 定义一个计算表达式 CT，执行矩阵乘法操作 gemm = sum(AP[m, k] * BP[k, n])，其中 m, n, k 分别是循环变量
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});

  // 定义一个计算表达式 DT，执行逐元素的 relu 操作，即 relu = max(CT[m, n], 0)
  Tensor DT =
      Compute("relu", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });

  // 定义一个计算表达式 ET，执行逐元素的加法操作，即 add = DT[m, n] + DT[m, n]
  Tensor ET =
      Compute("add", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return DT.load(m, n) + DT.load(m, n);
      });

  // 定义一个计算表达式 FT，执行逐元素的乘法操作，即 mul = ET[m, n] * ET[m, n]
  Tensor FT =
      Compute("mul", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return ET.load(m, n) * ET.load(m, n);
      });

  // 定义一个计算表达式 GT，执行逐元素的减法操作，即 sub = FT[m, n] - 1
  Tensor GT =
      Compute("sub", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return FT.load(m, n) - 1;
      });

  // 定义一个计算表达式 HT，执行逐元素的除法操作，即 div = GT[m, n] / 2
  Tensor HT =
      Compute("div", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        return GT.load(m, n) / 2;
      });

  // 将所有计算表达式放入语句块中
  auto stmt = Block::make(
      {CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt(), GT.stmt(), HT.stmt()});

  // 构建一个简单的IR评估器对象，用于评估计算表达式的结果
  SimpleIREvaluator cg(stmt, {AP, BP, HT});

  // 检查IR的语句，并进行匹配验证
  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
#ifdef TORCH_ENABLE_LLVM
  // 创建循环嵌套对象，以stmt为基础，包含FT.buf()作为参数
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  // 准备代码生成前的准备工作
  loop.prepareForCodegen();
  // 使用LLVM进行代码生成
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  // 检查生成的IR代码是否符合预期
  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK: Alias(add,gemm);
# CHECK: Alias(mul,relu);
# CHECK: Alias(sub,gemm);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}

TEST(MemPlanning, SmallerBufSizeNonMemReuse) {
  int M = 1024;
  int N = 1024;
  int K = 2048;

  // 创建缓冲区对象，用于存储矩阵数据
  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);

  // 创建张量CT，表示矩阵乘法结果的累加
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});

  // 创建张量DT，表示使用ReLU激活函数处理CT的结果
  Tensor DT =
      Compute("relu", {M, N}, [&](const ExprHandle& m, const ExprHandle& n) {
        auto zero = Cast::make(CT.buf()->dtype(), 0);
        return CompareSelect::make(
            CT.load(m, n), zero, zero, CT.load(m, n), kLT);
      });

  // 创建张量ET，表示对DT结果进行二倍增加操作
  Tensor ET = Compute(
      "add", {M * 2, N * 2}, [&](const ExprHandle& em, const ExprHandle& en) {
        return DT.load(em / 2, en / 2) + DT.load(em / 2, en / 2);
      });

  // 创建张量FT，表示对ET结果进行平方操作
  Tensor FT = Compute(
      "mul", {M * 2, N * 2}, [&](const ExprHandle& fm, const ExprHandle& fn) {
        return ET.load(fm, fn) * ET.load(fm, fn);
      });

  // 创建包含所有中间计算张量的语句块
  auto stmt = Block::make({CT.stmt(), DT.stmt(), ET.stmt(), FT.stmt()});

  // 创建SimpleIREvaluator对象，用于执行中间计算张量的评估
  SimpleIREvaluator cg(stmt, {AP, BP, FT});

  // 检查生成的IR代码是否符合预期
  checkIR(cg.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK-NOT: Alias(add,gemm);
# CHECK: Allocate(add); // dtype=float, dims=[2048, 2048]
# CHECK: Free(add);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
}
#ifdef TORCH_ENABLE_LLVM
  // 使用给定的语句克隆创建一个循环嵌套对象，仅包含FT.buf()作为外部缓冲区
  LoopNest loop(Stmt::clone(stmt), {FT.buf()});
  // 为LLVM代码生成做准备，这会修改内部表示
  loop.prepareForCodegen();
  // 使用LLVM进行代码生成，基于给定的根语句和缓冲区列表（AP, BP, FT）
  LLVMCodeGen cg_llvm(loop.root_stmt(), {AP, BP, FT});

  // 检查生成的LLVM IR代码是否符合预期
  checkIR(cg_llvm.stmt(), R"IR(
# CHECK: Allocate(gemm); // dtype=float, dims=[1024, 1024]
# CHECK: Allocate(relu); // dtype=float, dims=[1024, 1024]
# CHECK-NOT: Alias(add,gemm);
# CHECK: Allocate(add); // dtype=float, dims=[2048, 2048]
# CHECK: Free(add);
# CHECK: Free(relu);
# CHECK: Free(gemm))IR");
#endif
}

} // namespace jit
} // namespace torch
```