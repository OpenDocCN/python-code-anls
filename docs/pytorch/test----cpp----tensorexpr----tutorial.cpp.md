# `.\pytorch\test\cpp\tensorexpr\tutorial.cpp`

```py
// *** Tensor Expressions ***
//
// This tutorial covers basics of NNC's tensor expressions, shows basic APIs to
// work with them, and outlines how they are used in the overall TorchScript
// compilation pipeline. This doc is permanently a "work in progress" since NNC
// is under active development and things change fast.
//
// This Tutorial's code is compiled in the standard pytorch build, and the
// executable can be found in `build/bin/tutorial_tensorexpr`.
//
// *** What is NNC ***
//
// NNC stands for Neural Net Compiler. It is a component of TorchScript JIT
// and it performs on-the-fly code generation for kernels, which are often a
// combination of multiple aten (torch) operators.
//
// When the JIT interpreter executes a torchscript model, it automatically
// extracts subgraphs from the torchscript IR graph for which specialized code
// can be JIT generated. This usually improves performance as the 'combined'
// kernel created from the subgraph could avoid unnecessary memory traffic that
// is unavoidable when the subgraph is interpreted as-is, operator by operator.
// This optimization is often referred to as 'fusion'. Relatedly, the process of
// finding and extracting subgraphs suitable for NNC code generation is done by
// a JIT pass called 'fuser'.
//
// *** What is TE ***
//
// TE stands for Tensor Expressions. TE is a commonly used approach for
// compiling kernels performing tensor (~matrix) computation. The idea behind it
// is that operators are represented as a mathematical formula describing what
// computation they do (as TEs) and then the TE engine can perform mathematical
// simplification and other optimizations using those formulas and eventually
// generate executable code that would produce the same results as the original
// sequence of operators, but more efficiently.
//
// NNC's design and implementation of TE was heavily inspired by Halide and TVM
// projects.

#include <iostream>                             // 包含标准输入输出流库
#include <string>                               // 包含字符串处理库

#include <c10/util/irange.h>                    // 包含c10的范围处理工具
#include <torch/csrc/jit/ir/ir.h>               // Torch JIT的IR模块
#include <torch/csrc/jit/ir/irparser.h>         // Torch JIT的IR解析器
#include <torch/csrc/jit/tensorexpr/eval.h>     // Torch TE的表达式求值器
#include <torch/csrc/jit/tensorexpr/expr.h>     // Torch TE的表达式定义
#include <torch/csrc/jit/tensorexpr/ir.h>       // Torch TE的IR定义
#include <torch/csrc/jit/tensorexpr/ir_printer.h>// Torch TE的IR打印器
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>// Torch TE的IR简化器
#include <torch/csrc/jit/tensorexpr/kernel.h>   // Torch TE的内核定义
#include <torch/csrc/jit/tensorexpr/loopnest.h> // Torch TE的循环嵌套定义
#include <torch/csrc/jit/tensorexpr/stmt.h>     // Torch TE的语句定义
#include <torch/csrc/jit/tensorexpr/tensor.h>   // Torch TE的张量定义
#include <torch/torch.h>                        // Torch主库

using namespace torch::jit::tensorexpr;         // 使用torch JIT的tensor表达式命名空间

#ifdef TORCH_ENABLE_LLVM

// Helper function to print a snippet from a big multi-line string
static void printLinesToFrom(const std::string& input_str, int from, int to);

#endif

int main(int argc, char* argv[]) {
  std::cout << "*** Structure of tensor expressions and statements ***"
            << std::endl;                       // 打印主程序的欢迎信息

  {
    // A tensor expression is a tree of expressions. Each expression has a type,
    // 简述张量表达式是一个表达式树，每个表达式都有一个类型，


这样，每行代码都得到了适当的注释，解释了它们的作用和上下文。
    // and that type defines what sub-expressions the current expression has.
    // For instance, an expression of type 'Mul' would have a type 'kMul' and
    // two subexpressions: LHS and RHS. Each of these two sub-expressions could
    // also be a 'Mul' or some other expression.
    //
    // Let's construct a simple TE:
    // 分析表达式类型，'Mul' 表示乘法表达式，具有类型 'kMul' 和两个子表达式：LHS 和 RHS。
    // 每个子表达式也可能是 'Mul' 或其他表达式类型。

    ExprPtr lhs = alloc<IntImm>(5);
    // 创建一个整数常量表达式，值为 5
    ExprPtr rhs = alloc<Var>("x", kInt);
    // 创建一个整数类型的变量表达式，名称为 "x"
    ExprPtr mul = alloc<Mul>(lhs, rhs);
    // 创建一个乘法表达式，左操作数为 lhs，右操作数为 rhs
    std::cout << "Tensor expression: " << *mul << std::endl;
    // 打印结果：Tensor expression: 5 * x

    // Here we created an expression representing a 5*x computation, where x is
    // an int variable.
    // 创建了一个表示 5*x 计算的表达式，其中 x 是一个整数变量。

    // Another, probably a more convenient, way to construct tensor expressions
    // is to use so called expression handles (as opposed to raw expressions
    // like we did in the previous example). Expression handles overload common
    // operations and allow us to express the same semantics in a more natural
    // way:
    // 另一种更方便的构造张量表达式的方法是使用表达式句柄（而不是像前面的例子中使用的原始表达式）。表达式句柄重载了常见操作，允许我们以更自然的方式表达相同的语义。
    ExprHandle l = 5;
    // 创建一个整数常量表达式，值为 5
    ExprHandle r = Var::make("x", kInt);
    // 创建一个整数类型的变量表达式，名称为 "x"
    ExprHandle m = l * r;
    // 创建一个乘法表达式，左操作数为 l，右操作数为 r
    std::cout << "Tensor expression: " << *m.node() << std::endl;
    // 打印结果：Tensor expression: 5 * x

    // Converting from handles to raw expressions and back is easy:
    // 从表达式句柄转换为原始表达式以及反向转换都很容易：
    ExprHandle handle = Var::make("x", kInt);
    // 创建一个整数类型的变量表达式句柄，名称为 "x"
    ExprPtr raw_expr_from_handle = handle.node();
    // 从表达式句柄获取原始表达式
    ExprPtr raw_expr = alloc<Var>("x", kInt);
    // 创建一个整数类型的变量原始表达式，名称为 "x"
    ExprHandle handle_from_raw_expr = ExprHandle(raw_expr);
    // 使用原始表达式创建表达式句柄

    // We could construct arbitrarily complex expressions using mathematical
    // and logical operations, casts between various data types, and a bunch of
    // intrinsics.
    // 我们可以使用数学和逻辑操作、不同数据类型之间的转换以及许多内置函数构建任意复杂的表达式。
    ExprHandle a = Var::make("a", kInt);
    // 创建一个整数类型的变量表达式，名称为 "a"
    ExprHandle b = Var::make("b", kFloat);
    // 创建一个浮点数类型的变量表达式，名称为 "b"
    ExprHandle c = Var::make("c", kFloat);
    // 创建一个浮点数类型的变量表达式，名称为 "c"
    ExprHandle x = ExprHandle(5) * a + b / (sigmoid(c) - 3.0f);
    // 构建一个复杂的表达式，包括乘法、除法、函数调用（sigmoid）等
    std::cout << "Tensor expression: " << *x.node() << std::endl;
    // 打印结果：Tensor expression: float(5 * a) + b / ((sigmoid(c)) - 3.f)

    // An ultimate purpose of tensor expressions is to optimize tensor
    // computations, and in order to represent accesses to tensors data, there
    // is a special kind of expression - a load.
    // 张量表达式的最终目的是优化张量计算，并且为了表示对张量数据的访问，有一种特殊的表达式 - 载入（load）。
    //
    // To construct a load we need two pieces: the base and the indices. The
    // base of a load is a Buf expression, which could be thought of as a
    // placeholder similar to Var, but with dimensions info.
    // 要构造一个载入（load），我们需要两个部分：基础（base）和索引（indices）。载入的基础是 Buf 表达式，可以视为类似于 Var 的占位符，但包含维度信息。
    //
    // Let's construct a simple load:
    // 让我们构建一个简单的载入（load）：
    BufHandle A("A", {64, 32}, kInt);
    // 创建一个整数类型的缓冲区表达式句柄，名称为 "A"，维度为 {64, 32}
    VarPtr i_var = alloc<Var>("i", kInt), j_var = alloc<Var>("j", kInt);
    // 创建两个整数类型的变量原始表达式，名称分别为 "i" 和 "j"
    ExprHandle i(i_var), j(j_var);
    // 使用原始表达式创建表达式句柄
    ExprHandle load = Load::make(A.dtype(), A, {i, j});
    // 构造一个载入（load），加载缓冲区 A 中索引为 {i, j} 的数据
    std::cout << "Tensor expression: " << *load.node() << std::endl;
    // 打印结果：Tensor expression: A[i, j]

    // Tensor Expressions constitute Tensor Statements, which are used to
    // represent computation of a given operator or a group of operators from a
    // fusion group.
    // 张量表达式构成张量语句，用于表示从融合组中的给定运算符或一组运算符进行计算。
    //
    // There are three main kinds of tensor statements:
    //  - block
    //  - store
    // 张量语句主要有三种类型：
    //  - block
    //  - store
    // 创建一个 Store 语句，将表达式 i + j 存储到矩阵 A 的位置 (i, j)
    StmtPtr store_a = Store::make(A, {i, j}, i + j);
    // 打印 Store 语句，显示为 A[i, j] = i + j;
    std::cout << "Store statement: " << *store_a << std::endl;

    // 创建一个 For 语句，用于迭代变量 j_var 从 0 到 31，包含之前创建的 store_a 语句
    ForPtr loop_j_a = For::make(VarHandle(j_var), 0, 32, store_a);
    // 创建一个 For 语句，用于迭代变量 i_var 从 0 到 63，包含之前创建的 loop_j_a For 语句
    ForPtr loop_i_a = For::make(VarHandle(i_var), 0, 64, loop_j_a);

    // 打印嵌套的 For 循环语句
    std::cout << "Nested for loops: " << std::endl << *loop_i_a << std::endl;
    // 输出示例:
    // Nested for loops:
    // for (const auto i : c10::irange(64)) {
    //   for (const auto j : c10::irange(32)) {
    //     A[i, j] = i + j;
    //   }
    // }

    // 创建一个 BufHandle 对象 B，表示一个大小为 64x32 的整数矩阵
    BufHandle B("B", {64, 32}, kInt);
    // 创建一个 Store 语句，将矩阵 A 中 (i, j) 处的值存储到矩阵 B 的对应位置
    StmtPtr store_b = Store::make(B, {i, j}, A.load(i, j));
    // 创建一个 For 语句，用于迭代变量 j_var 从 0 到 31，包含之前创建的 store_b 语句
    ForPtr loop_j_b = For::make(VarHandle(j_var), 0, 32, store_b);
    // 创建一个 For 语句，用于迭代变量 i_var 从 0 到 63，包含之前创建的 loop_j_b For 语句
    ForPtr loop_i_b = For::make(VarHandle(i_var), 0, 64, loop_j_b);

    // 创建一个 Block 语句，将之前创建的两个 For 语句组合在一起
    BlockPtr block = Block::make({loop_i_a, loop_i_b});
    // 打印复合 Block 语句
    std::cout << "Compound Block statement: " << std::endl << *block << std::endl;
    // 输出示例:
    // Compound Block statement:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       A[i, j] = i + j;
    //     }
    //   }
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       B[i, j] = A[i, j];
    //     }
    //   }
    // }

    // 使用 Compute API 手动构建一个计算语句，计算矩阵 C 的每个元素值为 i * j
    Tensor C =
        Compute("C", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
          return i * j;
        });
    // 打印由 Compute API 生成的语句
    std::cout << "Stmt produced by 'Compute' API: " << std::endl << *C.stmt() << std::endl;
    // 打印结果为空，因为示例中的输出被省略
    {
      // 声明一个名为 C 的 Tensor 对象，表示一个 64x32 的计算，每个元素的值为 i * (j + 1)
      Tensor C =
          Compute("C", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
            return i * (j + 1);
          });
    
      // 获取 C 对应的缓冲区句柄
      BufHandle c_buf(C.buf());
    
      // 声明一个名为 D 的 Tensor 对象，表示一个 64x32 的计算，每个元素的值为 c_buf.load(i, j) - i
      Tensor D =
          Compute("D", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
            return c_buf.load(i, j) - i;
          });
    
      // 将 C 和 D 的计算封装在一个 Block 语句中
      StmtPtr block = Block::make({C.stmt(), D.stmt()});
    
      // 输出使用 'Compute' API 生成的语句块
      std::cout << "Stmt produced by 'Compute' API: " << std::endl
                << *block << std::endl;
      // 打印：
      // Stmt produced by 'Compute' API:
      // {
      //   for (const auto i : c10::irange(64)) {
      //     for (const auto j : c10::irange(32)) {
      //       C[i, j] = i * (j + 1);
      //     }
      //   }
      //   for (const auto i_1 : c10::irange(64)) {
      //     for (const auto j_1 : c10::irange(32)) {
      //       D[i_1, j_1] = (C[i_1, j_1]) - i_1;
      //     }
      //   }
      // }
    
      // 创建一个 LoopNest 对象，用于对生成的计算语句进行变换和优化
      LoopNest nest(block, {D.buf()});
    
      // 输出 LoopNest 对象的根语句
      std::cout << "LoopNest root stmt: " << std::endl
                << *nest.root_stmt() << std::endl;
      // 打印：
      // LoopNest root stmt:
      // {
      //   for (const auto i : c10::irange(64)) {
      //     for (const auto j : c10::irange(32)) {
      //       C[i, j] = i * (j + 1);
      //     }
      //   }
      //   for (const auto i_1 : c10::irange(64)) {
      //     for (const auto j_1 : c10::irange(32)) {
      //
    }
    // Now we can apply the inlining transformation:
    // 使用嵌入转换进行代码内联操作
    nest.computeInline(C.buf());
    // 输出内联后的语句
    std::cout << "Stmt after inlining:" << std::endl
              << *nest.root_stmt() << std::endl;
    // 打印结果:
    // Stmt after inlining:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       D[i, j] = i * (j + 1) - i;
    //     }
    //   }
    // }

    // We can also apply algebraic simplification to a statement:
    // 对语句进行代数简化
    StmtPtr simplified = IRSimplifier::simplify(nest.root_stmt());
    // 输出简化后的语句
    std::cout << "Stmt after simplification:" << std::endl
              << *simplified << std::endl;
    // 打印结果:
    // Stmt after simplification:
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       D[i, j] = i * j;
    //     }
    //   }
    // }

    // Many loopnest transformations are stateless and can be applied without
    // creating a LoopNest object. In fact, we plan to make all transformations
    // stateless.
    // splitWithTail 是一种这样的转换: 它将给定循环的迭代空间分割为两部分，使用给定的因子。
    ForPtr outer_loop = to<For>(to<Block>(simplified)->stmts().front());
    // 使用 splitWithTail 将外部循环分割
    LoopNest::splitWithTail(outer_loop, 13);
    // 再次调用简化器以进行算术折叠
    simplified = IRSimplifier::simplify(simplified);
    // 输出 splitWithTail 后的语句
    std::cout << "Stmt after splitWithTail:" << std::endl
              << *simplified << std::endl;
    // 打印结果:
    // Stmt after splitWithTail:
    // {
    //   for (const auto i_outer : c10::irange(4)) {
    //     for (const auto i_inner : c10::irange(13)) {
    //       for (const auto j : c10::irange(32)) {
    //         D[i_inner + 13 * i_outer, j] = i_inner * j + 13 * (i_outer * j);
    //       }
    //     }
    //   }
    //   for (const auto i_tail : c10::irange(12)) {
    //     for (const auto j : c10::irange(32)) {
    //       D[i_tail + 52, j] = i_tail * j + 52 * j;
    //     }
    //   }
    // }

    // NNC 支持广泛的循环嵌套转换，此处未列出所有。详细信息请参阅文档:
    // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/tensorexpr/loopnest.h
  }

  std::cout << "*** Codegen ***" << std::endl;
  {
    // 张量表达式的最终目标是提供一种在最快速度下执行给定计算的机制。
    // 到目前为止，我们描述了我们感兴趣的计算，但还未涉及如何实际执行它。

    // 让我们从构建一个简单的计算开始:
    // 创建一个名为 A 的缓冲区对象，形状为 (64, 32)，元素类型为整数
    BufHandle A("A", {64, 32}, kInt);
    // 创建一个名为 B 的缓冲区句柄，表示一个大小为 64x32 的整数类型的二维数组
    BufHandle B("B", {64, 32}, kInt);
    
    // 定义一个名为 X 的张量，大小为 64x32，计算规则是将 A 和 B 对应位置的元素相加
    Tensor X =
        Compute("X", {64, 32}, [&](const VarHandle& i, const VarHandle& j) {
          return A.load(i, j) + B.load(i, j);
        });
    
    // 将张量 X 转换为一个循环嵌套结构，以便进行后续的代码生成
    LoopNest loopnest({X});
    
    // 打印生成的循环嵌套结构，展示 X 的计算方式
    std::cout << *loopnest.root_stmt() << std::endl;
    // 打印结果为：
    // {
    //   for (const auto i : c10::irange(64)) {
    //     for (const auto j : c10::irange(32)) {
    //       X[i, j] = (A[i, j]) + (B[i, j]);
    //     }
    //   }
    
    // 创建一个 SimpleIREvaluator 对象 ir_eval，用于执行简单的 IR 评估器
    // 参数为循环嵌套的根语句，以及在计算中使用的占位符 A、B 和张量 X
    SimpleIREvaluator ir_eval(loopnest.root_stmt(), {A, B, X});
    
    // 创建三个大小为 64x32 的整数向量，分别作为输入 A、B 和输出 X 的数据
    std::vector<int> data_A(64 * 32, 3); // 输入 A 的数据
    std::vector<int> data_B(64 * 32, 5); // 输入 B 的数据
    std::vector<int> data_X(64 * 32, 0); // 用于存储 X 的结果数据
    ir_eval(data_A, data_B, data_X);
    // 调用一个函数 ir_eval，传入 data_A, data_B, data_X 作为参数进行评估

    // 让我们打印每个数组中的一个元素，以验证计算是否已经进行：
    std::cout << "A[10] = " << data_A[10] << std::endl
              << "B[10] = " << data_B[10] << std::endl
              << "X[10] = A[10] + B[10] = " << data_X[10] << std::endl;
    // 输出：
    // A[10] = 3
    // B[10] = 5
    // X[10] = A[10] + B[10] = 8
  }

  std::cout << "*** Lowering TorchScript IR to TensorExpr IR ***" << std::endl;
  {
    // 这部分需要一个支持 LLVM 的 PyTorch 构建，因此我们需要使用一个保护条件：
#ifdef TORCH_ENABLE_LLVM
    // 如果定义了 TORCH_ENABLE_LLVM 宏，则执行以下代码块

    // 常常我们希望将 TorchScript IR 转换为 TE 而不是从头开始构建 TE IR。
    // NNC 提供了一个 API 来执行这样的降低：它接受一个 TorchScript 图，并返回一个
    // 可用于调用生成的内核的对象。
    // 当前 TorchScript JIT 融合器使用此 API，也可以提前编译模型的部分。
    //
    // 为了熟悉这个 API，让我们先从定义一个简单的 TorchScript 图开始：
    const auto graph_string = R"IR(
        graph(%A : Float(5, 3, strides=[3, 1], device=cpu),
              %B : Float(5, 3, strides=[3, 1], device=cpu)):
          %AB : Float(5, 3, strides=[3, 1]) = aten::mul(%A, %B)
          %one : int = prim::Constant[value=1]()
          %AAB : Float(5, 3, strides=[3, 1]) = aten::mul(%A, %AB)
          %AAB_plus_B: Float(5, 3, strides=[3, 1]) = aten::add(%AAB, %B, %one)
          return (%AAB_plus_B))IR";
    // 创建一个共享指针指向 TorchScript 图对象
    auto graph = std::make_shared<torch::jit::Graph>();
    // 解析图的字符串表示，将其填充到 graph 对象中
    parseIR(graph_string, &*graph);

    // 此图定义了一个简单的计算 A*A*B + B，其中 A 和 B 是输入的 5x3 张量。

    // 要将此 TorchScript 图降低到 TE，我们只需创建一个 TensorExprKernel 对象。
    // 在其构造函数中，它构建相应的 TE IR 并为给定的后端（在此示例中为使用 LLVM 编译器的 CPU）编译它。
    TensorExprKernel kernel(graph);

    // 我们可以从内核对象中检索生成的 TE stmt：
    StmtPtr kernel_stmt = kernel.getCodeGenStmt();
    // 打印生成的 TE stmt：
    std::cout << "TE Stmt constructed from TorchScript: " << std::endl
              << *kernel_stmt << std::endl;
    // 打印：
    // TE Stmt constructed from TorchScript:
    // {
    //   for (const auto v : c10::irange(5)) {
    //     for (const auto _tail_tail : c10::irange(3)) {
    //       aten_add[_tail_tail + 3 * v] = (tA[_tail_tail + 3 * v]) *
    //       ((tA[_tail_tail + 3 * v]) * (tB[_tail_tail + 3 * v])) +
    //       (tB[_tail_tail + 3 * v]);
    //     }
    //   }
    // }

    // 我们还可以查看生成的 LLVM IR 和汇编代码：
    std::cout << "Generated LLVM IR: " << std::endl;
    // 获取 LLVM IR 的字符串表示
    auto ir_str = kernel.getCodeText("ir");
    // 打印 LLVM IR 的部分行数：
    printLinesToFrom(ir_str, 15, 20);
    // 打印：
    // Generated LLVM IR:
    //   %9 = bitcast float* %2 to <8 x float>*
    //   %10 = load <8 x float>, <8 x float>* %9 ...
    //   %11 = bitcast float* %5 to <8 x float>*
    //   %12 = load <8 x float>, <8 x float>* %11 ...
    //   %13 = fmul <8 x float> %10, %12
    //   %14 = fmul <8 x float> %10, %13

    std::cout << "Generated assembly: " << std::endl;
    // 获取汇编代码的字符串表示
    auto asm_str = kernel.getCodeText("asm");
    // 打印汇编代码的部分行数：
    printLinesToFrom(asm_str, 10, 15);
    // 打印：
    // Generated assembly:
    //         vmulps  %ymm1, %ymm0, %ymm2
    //         vfmadd213ps     %ymm1, %ymm0, %ymm2
    //         vmovups %ymm2, (%rax)
    //         vmovss  32(%rcx), %xmm0;
#endif  // TORCH_ENABLE_LLVM
    // 使用汇编指令vmovss，从内存地址rdx偏移32字节处加载单精度浮点数到寄存器xmm1
    // 使用汇编指令vmulss，将寄存器xmm0和xmm1中的单精度浮点数相乘，结果存储在xmm2中

    // 创建大小为5x3的全为2.0的张量A，数据类型为float，存储在CPU上
    auto A =
        at::ones({5, 3}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) *
        2.0;
    // 创建大小为5x3的全为3.0的张量B，数据类型为float，存储在CPU上
    auto B =
        at::ones({5, 3}, torch::TensorOptions(torch::kCPU).dtype(at::kFloat)) *
        3.0;
    // 将张量A和B存入std::vector<at::Tensor>中
    std::vector<at::Tensor> inputs = {A, B};
    // 将inputs中的张量转换为torch::IValue类型，存入std::vector<torch::IValue> stack中
    std::vector<torch::IValue> stack = torch::fmap<torch::IValue>(inputs);
    // 运行kernel，将输入张量传递给kernel
    kernel.run(stack);
    // 从stack中获取第一个元素，并转换为Tensor类型，存入R中
    auto R = stack[0].toTensor();

    // 打印结果张量R中第二行第二列的元素，验证计算结果是否正确
    std::cout << "R[2][2] = " << R[2][2] << std::endl;
    // 打印示例：
    // R[2][2] = 15
    // [ CPUFloatType{} ]
#endif
  }
  // 函数执行完毕，返回值 0
  return 0;
}

void printLinesToFrom(const std::string& input_str, int from, int to) {
  // 使用输入字符串创建字符串流
  std::istringstream f(input_str);
  // 声明字符串变量 s，用于存储每行内容
  std::string s;
  // 初始化行号索引为 0
  int idx = 0;
  // 逐行读取字符串流中的内容
  while (getline(f, s)) {
    // 如果行号大于 from，则输出当前行内容
    if (idx > from) {
      std::cout << s << "\n";
    }
    // 如果行号超过 to，则跳出循环
    if (idx++ > to) {
      break;
    }
  }
}
```