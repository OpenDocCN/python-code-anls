# `.\pytorch\test\cpp\tensorexpr\test_kernel.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 ATen 库的代码模板
#include <ATen/code_template.h>

// 包含 C10 库的工具函数，如范围迭代器
#include <c10/util/irange.h>

// 包含 TensorExpr 的测试基类
#include <test/cpp/tensorexpr/test_base.h>

// 包含 Torch 的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>

// 包含 Torch 的 IR 优化 passes 头文件
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

// 包含 Torch 的 TensorExpr 内核定义头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

// 包含 Torch 的文件检查工具头文件
#include <torch/csrc/jit/testing/file_check.h>

// 包含 Torch 的核心头文件
#include <torch/torch.h>

// 包含标准数学函数库
#include <cmath>

// 包含字符串流处理头文件
#include <sstream>

// 包含标准异常处理头文件
#include <stdexcept>

// Torch 的 JIT 命名空间
namespace torch {
namespace jit {

// 使用 torch::indexing 命名空间
using namespace torch::indexing;

// 使用 torch::jit::tensorexpr 命名空间
using namespace torch::jit::tensorexpr;

// 定义一个测试类 Kernel，继承自 ::testing::Test
class Kernel : public ::testing::Test {
 public:
  // 设置测试环境，在每个测试之前执行
  void SetUp() override {
    // 设置不强制使用 LLVM 在 CPU 上运行
    getTEMustUseLLVMOnCPU() = false;
  }
};

// 定义一个测试用例 ParallelExternalCallBuf
TEST_F(Kernel, ParallelExternalCallBuf) {
  // 定义包含 IR 字符串
  const auto graph_string = R"IR(
    graph(%0 : Float(1000, 5000, strides=[5000, 1], device=cpu),
          %1 : Float(1000, 5000, strides=[5000, 1], device=cpu),
          %2 : Float(5000, 1000, strides=[5000, 1], device=cpu)):
      %3 : Float(1000, 5000, strides=[5000, 1], device=cpu) = aten::mul(%0, %1)
      %4 : Float(1000, 5000, strides=[5000, 1], device=cpu) = aten::matmul(%3, %2)
      return (%4))IR";

  // 创建一个新的图形对象
  auto graph = std::make_shared<Graph>();

  // 解析 IR 字符串并填充到图形对象中
  torch::jit::parseIR(graph_string, &*graph);

  // 定义验证模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i = 0ll; i < 5000ll; i++)  /* parallel */{)IR";

#ifdef TORCH_ENABLE_LLVM
  // 创建一个 TensorExprKernel 对象，传入图形对象
  TensorExprKernel k(graph);

  // 获取代码生成语句
  StmtPtr s = k.getCodeGenStmt();

  // 创建一个字符串流对象
  std::ostringstream oss;

  // 将代码生成语句输出到字符串流中
  oss << *s;

  // 运行文件检查工具，验证输出是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
#endif
}

// 定义另一个测试用例 InliningIntermediates
TEST_F(Kernel, InliningIntermediates) {
  // 内联中间结果测试
  {
    // 定义包含 IR 字符串
    const auto graph_string = R"IR(
        graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
              %1 : Float(5, 3, strides=[3, 1], device=cpu)):
          %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
          %one : int = prim::Constant[value=1]()
          %4 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
          %5: Float(5, 3, strides=[3, 1]) = aten::add(%4, %1, %one)
          return (%5))IR";

    // 创建一个新的图形对象
    auto graph = std::make_shared<Graph>();

    // 解析 IR 字符串并填充到图形对象中
    parseIR(graph_string, &*graph);

    // 创建 TensorExprKernel 对象，传入图形对象
    TensorExprKernel k(graph);

    // 获取代码生成语句
    auto stmt = k.getCodeGenStmt();

    // 创建一个字符串流对象
    std::ostringstream oss;

    // 将代码生成语句输出到字符串流中
    oss << *stmt;

    // 运行文件检查工具，检查是否不存在 "aten_mul" 的出现
    torch::jit::testing::FileCheck().check_not("aten_mul")->run(oss.str());
  }
}
    const auto graph_template = R"IR(
        graph(%0 : Float(5, 3, strides=[3, 1], device=${device}),
              %1 : Float(5, 3, strides=[3, 1], device=${device})):
          %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
          %one : int = prim::Constant[value=1]()
          %3 : Float(5, 3, strides=[3, 1]) = aten::sub(%0, %2, %one)
          %4 : Float(5, 3, strides=[3, 1]) = aten::add(%3, %0, %one)
          %5 : Float(5, 3, strides=[3, 1]) = aten::div(%3, %0)
          return (%4, %5))IR";

    // 遍历是否使用 CUDA 的布尔值列表
    for (bool use_cuda : {false, true}) {
      // 如果当前系统不支持 CUDA 且需要使用 CUDA，则跳过此次循环
      if (!torch::cuda::is_available() && use_cuda) {
        continue;
      }

      // 创建模板环境并设置设备为 CPU 或 CUDA
      at::jit::TemplateEnv env;
      env.s("device", use_cuda ? "cuda:0" : "cpu");
      
      // 使用环境变量替换模板字符串，生成具体的图形描述字符串
      const auto graph_string = format(graph_template, env);
      
      // 创建共享的图对象指针
      auto graph = std::make_shared<Graph>();
      
      // 解析图形描述字符串，生成图形对象
      parseIR(graph_string, &*graph);
      
      // 根据图对象创建张量表达式内核
      TensorExprKernel k(graph);
      
      // 获取代码生成的语句对象
      auto stmt = k.getCodeGenStmt();
      
      // 创建字符串流对象，用于将代码生成语句转换为字符串
      std::ostringstream oss;
      oss << *stmt;
      
      // 执行文件检查，确保代码中不包含 "aten_mul"
      // aten_mul 只被一个使用处完全内联
      torch::jit::testing::FileCheck().check_not("aten_mul")->run(oss.str());

      // 执行文件检查，确保代码中不包含 "aten_sub"
      // aten_sub 应该由 CUDA 后端通过 metavar 重写移除，由 CPU 后端通过水平融合移除
      torch::jit::testing::FileCheck().check_not("aten_sub")->run(oss.str());
    }
}

// 在 Kernel 类中定义 PreAllocIntermediateBufs 测试方法
TEST_F(Kernel, PreAllocIntermediateBufs) {
  // 定义包含 IR 表达式的字符串
  const auto graph_string = R"IR(
graph(%a.1 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu),
      %b.1 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu)):
  %2 : int = prim::Constant[value=1]()
  %c.2 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::matmul(%a.1, %b.1) # test_matmul.py:12:12
  %3 : Float(8, 8, strides=[8, 1], requires_grad=0, device=cpu) = aten::add(%a.1, %c.2, %2) # test_matmul.py:13:15
  return (%3))IR";
  
  // 创建共享指针指向新的 Graph 对象，并解析 IR 表达式字符串到该 Graph
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 创建随机初始化的张量 a, b 和零张量 o，以及预期的参考结果 ref
  auto a = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = at::matmul(a, b) + a;

  // 使用 Graph 创建 TensorExprKernel 对象 k，不传入额外的参数和常量，开启优化
  TensorExprKernel k(graph, {}, {}, true);

  // 构建输入张量的向量
  std::vector<at::Tensor> inputs = {a, b};

  // 获取代码生成语句的指针 stmt
  auto stmt = k.getCodeGenStmt();

  // 创建一个输出字符串流 oss
  std::ostringstream oss;
  oss << *stmt;

  // 检查是否已将中间缓冲区添加到常量中
  auto constants = k.getConstantDescriptors();
  ASSERT_EQ(constants.size(), 1);

  // 检查生成的 IR
  torch::jit::testing::FileCheck().check_not("Alloc")->run(oss.str());
  torch::jit::testing::FileCheck().check_not("Free")->run(oss.str());

  // 检查正确性
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  ASSERT_TRUE(at::allclose(o, ref));
}

// 在 Kernel 类中定义 _1 测试方法
TEST_F(Kernel, _1) {
  // 定义包含 IR 表达式的字符串
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  
  // 创建共享指针指向新的 Graph 对象，并解析 IR 表达式字符串到该 Graph
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 创建随机初始化的张量 a, b 和零张量 o，以及预期的参考结果 ref
  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);

  // 使用 Graph 创建 TensorExprKernel 对象 k
  TensorExprKernel k(graph);

  // 构建输入张量的向量
  std::vector<at::Tensor> inputs = {a, b};

  // 获取代码生成语句的指针 s
  StmtPtr s = k.getCodeGenStmt();

  // 创建一个输出字符串流 oss
  std::ostringstream oss;
  oss << *s;

  // 定义用于验证的 IR 模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";

  // 运行 FileCheck 来验证生成的 IR
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 检查正确性
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  for (size_t i = 0; i < 5 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}
TEST_F(Kernel, _2) {
  // 定义一个包含 IR 字符串的常量，描述了一个计算图
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();
  // 解析 IR 字符串，填充到创建的计算图对象中
  parseIR(graph_string, &*graph);

  // 创建随机生成的张量 a, b 和一个全零张量 o，以及参考结果张量 ref
  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);

  // 使用填充了 IR 的计算图对象创建 TensorExprKernel 对象
  TensorExprKernel k(graph);
  // 准备输入张量的向量
  std::vector<at::Tensor> inputs = {a, b};
  // 获取代码生成后的语句对象
  StmtPtr s = k.getCodeGenStmt();

  // 创建一个字符串流对象，将语句对象的内容输出到流中
  std::ostringstream oss;
  oss << *s;

  // 检查生成的 IR 是否符合指定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 运行 TensorExprKernel 对象，传入输入张量的堆栈
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  // 将运行结果写回输出张量 o
  o = stack[0].toTensor();

  // 检查计算结果与参考结果是否一致
  for (size_t i = 0; i < 5 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

TEST_F(Kernel, _3) {
  // 定义一个包含 IR 字符串的常量，描述了一个计算图
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[12, 2], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();
  // 解析 IR 字符串，填充到创建的计算图对象中
  parseIR(graph_string, &*graph);

  // 创建随机生成的张量 a, b 和一个全零张量 o，以及参考结果张量 ref
  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
               .index({Slice(None, None, 2), Slice(None, None, 2)});
  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto ref = a * (a * b);

  // 使用填充了 IR 的计算图对象创建 TensorExprKernel 对象
  TensorExprKernel k(graph);
  // 准备输入张量的向量
  std::vector<at::Tensor> inputs = {a, b};
  // 获取代码生成后的语句对象
  StmtPtr s = k.getCodeGenStmt();

  // 创建一个字符串流对象，将语句对象的内容输出到流中
  std::ostringstream oss;
  oss << *s;

  // 检查生成的 IR 是否符合指定的模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NOT: for)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 运行 TensorExprKernel 对象，传入输入张量的堆栈
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  // 将运行结果写回输出张量 o
  o = stack[0].toTensor();

  // 检查计算结果与参考结果是否一致
  for (size_t i = 0; i < 5 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}
TEST_F(Kernel, Huge) {
  // 定义一个包含大量元素的张量计算图的字符串表示
  const auto graph_string = R"IR(
      graph(%x.1 : Float(4000000000, strides=[1], requires_grad=0, device=cpu)):
        %1 : int = prim::Constant[value=0]()
        %2 : Float(1, 4000000000, strides=[4000000000, 1], requires_grad=0, device=cpu) = aten::unsqueeze(%x.1, %1)
        %3 : Float(1, 4000000000, strides=[4000000000, 1], requires_grad=0, device=cpu) = aten::relu(%2)
        return (%3))IR";
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();
  // 解析上述字符串表示的计算图，并将其加载到 graph 中
  parseIR(graph_string, &*graph);
  // 使用加载后的计算图创建一个 TensorExprKernel 对象
  TensorExprKernel k(graph);
  // 创建一个字符串流对象 oss
  std::ostringstream oss;
  // 将 TensorExprKernel 对象的代码生成语句输出到 oss 中
  oss << *k.getCodeGenStmt();
  // 定义用于验证输出的字符串模式
  // 如果 LLVM 存在，将 4000000000 次迭代循环分割成 500000000 x 8，并且外部循环将是并行的。
  // 如果 LLVM 不存在，循环不会被分割，我们需要检查输出中是否包含 "00000000ll;"。
  const std::string& verification_pattern = R"IR(# CHECK: 00000000ll;)IR";
  // 运行 FileCheck 工具验证输出是否符合上述模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());
}

TEST_F(Kernel, ParallelStrided) {
  // 定义一个包含并行步幅的张量计算图的字符串表示
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, 40005, strides=[120015, 40005, 1], device=cpu),
            %1 : Float(5, 3, 40005, strides=[960120, 160020, 2], device=cpu)):
        %2 : Float(5, 3, 40005, strides=[120015, 40005, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, 40005, strides=[120015, 40005, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  // 创建一个空的计算图对象
  auto graph = std::make_shared<Graph>();
  // 解析上述字符串表示的计算图，并将其加载到 graph 中
  parseIR(graph_string, &*graph);

  // 创建输入张量 a 和 b，使用随机数据填充
  auto a = at::rand({5, 3, 40005}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({10, 6, 80010}, TensorOptions(kCPU).dtype(at::kFloat))
               .index(
                   {Slice(None, None, 2),
                    Slice(None, None, 2),
                    Slice(None, None, 2)});
  // 计算参考输出 ref = a * (a * b)
  auto ref = a * (a * b);
  // 创建一个和 ref 相同形状的零张量 o
  auto o = at::zeros_like(ref);
  // 使用 TensorExprKernel 对象 k 运行计算图，并将结果存储在 o 中
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  o = stack[0].toTensor();
  // 检查计算结果 o 是否与参考输出 ref 相等
  for (size_t i = 0; i < 5 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}

TEST_F(Kernel, DISABLED_Shape_Inference) {
  // disabled: 不执行形状推断，当前也没有使用

  // 测试 TensorExpr 的形状推断功能：应该只需要输入张量的形状信息
  {
    const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[12, 2], device=cpu)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor = aten::mul(%0, %2)
        return (%3))IR";
    // 创建一个空的计算图对象
    auto graph = std::make_shared<Graph>();
    // 解析上述字符串表示的计算图，并将其加载到 graph 中
    parseIR(graph_string, &*graph);

    // 创建输入张量 a 和 b，使用随机数据填充
    auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 6}, TensorOptions(kCPU).dtype(at::kFloat))
                 .index({Slice(None, None, 2), Slice(None, None, 2)});
    // 创建一个形状与 a 相同的零张量 o
    auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
    // 计算参考输出 ref = a * (a * b)
    auto ref = a * (a * b);
    // 使用给定的计算图(graph)创建张量表达式内核对象
    TensorExprKernel k(graph);
    // 创建包含张量a和b的输入向量
    std::vector<at::Tensor> inputs = {a, b};
    // 获取代码生成语句(stmt)，该语句用于生成张量表达式的代码
    StmtPtr s = k.getCodeGenStmt();

    // 创建一个字符串流，将生成的代码语句(s)输出到流中
    std::ostringstream oss;
    oss << *s;

    // 检查生成的中间表示(IR)
    // 验证模式字符串定义了期望的中间表示(IR)的格式和内容
    const std::string& verification_pattern =
        R"IR(
  {
    // 定义一个表示计算图的字符串，使用 R"IR(...)IR" 语法
    const auto graph_string = R"IR(
      graph(%0 : Float(8, 8, strides=[8, 1], device=cpu),
            %1 : Float(8, 8, strides=[8, 1], device=cpu)):
        %2 : Tensor = aten::mul(%0, %1)
        %3 : Tensor, %4 : Tensor = prim::ConstantChunk[dim=1,chunks=2](%2)
        %r : Tensor = aten::mul(%3, %4)
        return (%r))IR";
    
    // 创建一个共享指针指向 Graph 对象，并解析上面定义的计算图字符串
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    // 创建三个随机初始化的张量 a, b, 和 o
    auto a = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({8, 8}, TensorOptions(kCPU).dtype(at::kFloat));
    auto o = at::zeros({8, 4}, TensorOptions(kCPU).dtype(at::kFloat));
    
    // 对张量 a 和 b 进行分块操作，将结果保存在 t 中
    auto t = torch::chunk(a * b, 2, 1);
    
    // 计算参考结果 ref
    auto ref = t[0] * t[1];
    
    // 创建一个 TensorExprKernel 对象 k，用上述定义的计算图初始化它
    TensorExprKernel k(graph);
    
    // 创建一个包含张量 a 和 b 的输入向量
    std::vector<at::Tensor> inputs = {a, b};
    
    // 获取由 TensorExprKernel 生成的代码块语句
    StmtPtr s = k.getCodeGenStmt();

    // 创建一个字符串流 oss，并将代码块语句 s 输出到 oss 中
    std::ostringstream oss;
    oss << *s;

    // 检查生成的 IR 是否符合预期模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for)IR";
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // 将输入向量转换成 IValue 向量并运行张量表达式 kernel
    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);
    
    // 将计算结果写回张量 o
    o = stack[0].toTensor();
    
    // 验证 o 的尺寸和数据与 ref 相符
    TORCH_CHECK_EQ(o.sizes()[0], 8);
    TORCH_CHECK_EQ(o.sizes()[1], 4);
    for (size_t i = 0; i < 8 * 4; i++) {
      TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    }
  }
  {
    // 测试 shape 推断处理 aten::unsqueeze
    
    // 定义一个表示计算图的字符串，使用 R"IR(...)IR" 语法
    const auto graph_string = R"IR(
      graph(%a : Float(4, 2, strides=[2, 1], device=cpu),
            %b : Float(4, 3, 2, strides=[6, 2, 1], device=cpu),
            %c : Float(3, 2, 2, strides=[4, 2, 1], device=cpu)):
        %one : int = prim::Constant[value=1]()
        %minus_one : int = prim::Constant[value=-1]()
        %three : int = prim::Constant[value=3]()
        %minus_four : int = prim::Constant[value=-4]()
        %a1 : Tensor = aten::unsqueeze(%a, %one)        # new size: [4,1,2]
        %a2 : Tensor = aten::unsqueeze(%a1, %minus_one) # new size: [4,1,2,1]
        %b1 : Tensor = aten::unsqueeze(%b, %three)      # new size: [4,3,2,1]
        %c1 : Tensor = aten::unsqueeze(%c, %minus_four) # new size: [1,3,2,2]
        %ab : Tensor = aten::mul(%a2, %b1)         # 期望尺寸: [4,3,2,1]
        %abc : Tensor = aten::mul(%ab, %c1)        # 期望尺寸: [4,3,2,2]
        return (%abc))IR";
    
    // 创建一个共享指针指向 Graph 对象，并解析上面定义的计算图字符串
    auto graph = std::make_shared<Graph>();
    parseIR(graph_string, &*graph);

    // 创建三个随机初始化的张量 a, b, 和 c
    auto a = at::rand({4, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({4, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    // 创建一个形状为 {4, 3, 2, 2} 的全零张量 `o`，使用 CPU 上的浮点数数据类型
    auto o = at::zeros({4, 3, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    
    // 对张量 `a` 进行两次 unsqueeze 操作，并将张量 `b` 和 `c` 进行 unsqueeze 操作，然后进行逐元素乘法操作
    auto ref = at::unsqueeze(at::unsqueeze(a, 1), -1) * at::unsqueeze(b, 3) *
        at::unsqueeze(c, -4);
    
    // 创建一个 TensorExprKernel 对象 `k`，使用给定的计算图 `graph` 初始化
    TensorExprKernel k(graph);
    
    // 创建一个包含张量 `a`, `b`, `c` 的向量 `inputs`
    std::vector<at::Tensor> inputs = {a, b, c};
    
    // 获取代码生成语句的指针 `s`
    StmtPtr s = k.getCodeGenStmt();
    
    // 创建一个 ostringstream 对象 `oss`，并将代码生成语句 `s` 的内容写入其中
    std::ostringstream oss;
    oss << *s;
    
    // 检查我们生成的中间表示(IR)
    // 验证模式字符串，包含生成的 IR 表示形式
    const std::string& verification_pattern =
        R"IR(
{
  // Test the correctness of generated IR against expected pattern using FileCheck
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // Convert input tensors to IValues and execute the kernel
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  // Retrieve output tensor from stack
  o = stack[0].toTensor();

  // Check sizes of output tensor against reference tensor
  TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
  size_t num_el = 1;
  // Compute total number of elements in the tensors
  for (const auto idx : c10::irange(ref.sizes().size())) {
    TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
    num_el *= ref.sizes()[idx];
  }

  // Check each element of the output tensor against reference tensor
  for (const auto i : c10::irange(num_el)) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}
{
  // Test that shape inference correctly handles aten::cat in the IR graph

  // Define the IR graph string
  const auto graph_string = R"IR(
    graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
          %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
          %c : Float(5, 9, 2, strides=[18, 2, 1], device=cpu)):
      %dim : int = prim::Constant[value=1]()
      %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
      %r : Tensor = aten::cat(%inputs, %dim)               # new size: [5,19,2]
      return (%r))IR";

  // Parse the IR graph string into a Graph object
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // Generate random tensors a, b, c and an output tensor o with zeros
  auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto o = at::zeros({5, 19, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  
  // Concatenate tensors a, b, c along dimension 1 to create a reference tensor ref
  auto ref = at::cat({a, b, c}, 1);

  // Instantiate a TensorExprKernel with the parsed IR graph
  TensorExprKernel k(graph);
  
  // Prepare input tensors as a vector for kernel execution
  std::vector<at::Tensor> inputs = {a, b, c};
  
  // Get the code generation statement (StmtPtr) from the kernel
  StmtPtr s = k.getCodeGenStmt();

  // Convert the statement to a string representation
  std::ostringstream oss;
  oss << *s;

  // Check the generated IR against the expected pattern using FileCheck
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // Convert input tensors to IValues and execute the kernel
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  // Retrieve output tensor from stack
  o = stack[0].toTensor();

  // Check sizes of output tensor against reference tensor
  TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
  size_t num_el = 1;
  // Compute total number of elements in the tensors
  for (const auto idx : c10::irange(ref.sizes().size())) {
    TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
    num_el *= ref.sizes()[idx];
  }

  // Check each element of the output tensor against reference tensor
  for (const auto i : c10::irange(num_el)) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }
}
{
  // Test that an error is thrown when input list for aten::cat is empty

  // Define the IR graph string with an empty list construct
  const auto graph_string = R"IR(
    graph():
      %dim : int = prim::Constant[value=1]()
      %inputs : Tensor[] = prim::ListConstruct()
      %r : Tensor = aten::cat(%inputs, %dim)
      return (%r))IR";

  // Parse the IR graph string into a Graph object
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // Attempt to create a TensorExprKernel with the empty graph (should throw an error)
  auto compile = [&]() {
    TensorExprKernel k(graph);
    k.getCodeGenStmt();
  };

  // The actual test is performed in the lambda compile
}
  {
    // 当传递给 aten::cat 的 'dim' 参数无效时，测试抛出错误的情况

    // 定义包含无效 'dim' 的 IR 字符串
    const auto ir_dim_99 = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 3, 2, strides=[6, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=99]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b)
        %r : Float(5, 3, 2, strides=[6, 2, 1], device=cpu) = aten::cat(%inputs, %dim)
        return (%r))IR";
    
    // 定义包含负数 'dim' 的 IR 字符串
    const auto ir_dim_minus_6 = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 3, 2, strides=[6, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=-6]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b)
        %r : Float(5, 3, 2, strides=[6, 2, 1], device=cpu) = aten::cat(%inputs, %dim)
        return (%r))IR";

    // 定义编译函数，解析 IR 字符串并生成张量表达式内核
    auto compile = [](const std::string& graph_string) {
      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph); // 解析 IR 字符串并填充到图中
      TensorExprKernel k(graph); // 创建张量表达式内核对象
      k.getCodeGenStmt(); // 获取代码生成语句
    };

    // 断言编译函数对无效 'dim' 抛出预期的错误消息
    ASSERT_THROWS_WITH(compile(ir_dim_99), "Invalid index");
    // 断言编译函数对负数 'dim' 抛出预期的错误消息
    ASSERT_THROWS_WITH(compile(ir_dim_minus_6), "Invalid index");
  }
// 定义测试用例 `CatInputTypesPromotion`，测试 `aten::cat` 函数输入类型的正确提升
TEST_F(Kernel, CatInputTypesPromotion) {
  {
    // IR 字符串表示图形的中间表示（IR），定义了包含三个输入张量的计算图
    const auto graph_string = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
            %c : Double(5, 9, 2, strides=[18, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Double(5, 19, 2, strides=[38, 2, 1]) = aten::cat(%inputs, %dim)
        return (%r))IR";
    // 创建共享指针指向一个空的图形对象
    auto graph = std::make_shared<Graph>();
    // 解析 IR 字符串到图形对象
    parseIR(graph_string, &*graph);

    // 创建三个随机张量 a, b, c，分别指定其设备和数据类型
    auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
    auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kDouble));
    // 创建参考张量 ref，使用 aten::cat 函数将 a, b, c 连接在维度 1 上
    auto ref = at::cat({a, b, c}, 1);

    // 创建 TensorExprKernel 对象，并传入图形对象进行初始化
    TensorExprKernel k(graph);
    // 创建输入张量的向量
    std::vector<at::Tensor> inputs = {a, b, c};
    // 获取代码生成后的语句对象
    StmtPtr s = k.getCodeGenStmt();

    // 创建输出字符串流
    std::ostringstream oss;
    // 将生成的语句对象输出到字符串流中
    oss << *s;

    // 检查生成的 IR 是否符合指定的验证模式
    const std::string& verification_pattern =
        R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_cat)IR";
    // 使用 FileCheck 工具检查字符串流中的 IR 是否匹配验证模式
    torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

    // 创建输入值的向量，并将输入张量转换为 IValue
    std::vector<IValue> stack = fmap<IValue>(inputs);
    // 运行 TensorExprKernel 对象，将结果存储在 stack 中
    k.run(stack);
    // 获取运行结果张量 o
    auto o = stack[0].toTensor();

    // 检查张量尺寸是否相等
    TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
    // 检查张量数据类型是否相等
    TORCH_CHECK_EQ(o.dtype(), ref.dtype());
    // 计算参考张量中元素的总数
    size_t num_el = 1;
    for (const auto idx : c10::irange(ref.sizes().size())) {
      // 检查每个维度的尺寸是否一致
      TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
      // 计算参考张量中元素的总数
      num_el *= ref.sizes()[idx];
    }

    // 检查张量内容是否一致
    for (const auto i : c10::irange(num_el)) {
      // 逐元素比较张量的浮点数值
      TORCH_CHECK_EQ(((double*)o.data_ptr())[i], ((double*)ref.data_ptr())[i]);
    }
  }
}
#ifdef TORCH_ENABLE_LLVM
  // 定义字符串，包含 LLVM IR 表示的计算图
  const auto graph_string = R"IR(
      graph(%x.1 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
        %1 : NoneType = prim::Constant()
        %2 : bool = prim::Constant[value=0]()
        %3 : int = prim::Constant[value=6]()
        %4 : int = prim::Constant[value=15]()
        %5 : int = prim::Constant[value=5]()
        %6 : bool = prim::Constant[value=1]()
        %y.3 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::sigmoid(%x.1)
        %z.3 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::_autocast_to_reduced_precision(%y.3, %6, %6, %5, %4)
        %h.3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::_autocast_to_full_precision(%z.3, %6, %6)
        %i.3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::to(%h.3, %3, %2, %2, %1)
        %j.3 : BFloat16(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::to(%i.3, %4, %2, %2, %1)
        %k.3 : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::to(%j.3, %3, %2, %2, %1)
        return (%k.3))IR";

  // 创建共享指针，用于存储解析后的计算图
  auto graph = std::make_shared<Graph>();
  // 解析 LLVM IR 字符串，填充图对象
  parseIR(graph_string, &*graph);
  // 使用图对象构建 TensorExprKernel
  TensorExprKernel k(graph);
  // 获取代码生成后的语句对象
  StmtPtr s = k.getCodeGenStmt();
  // 使用流将语句对象转换为字符串
  std::ostringstream oss;
  oss << *s;

  // 定义字符串，包含用于验证的模式信息
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_to
# CHECK-NEXT: }
# CHECK-NEXT: })IR";
  // 运行 FileCheck 验证生成的代码与模式匹配
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 创建随机生成的 BFloat16 张量 a
  auto a = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kBFloat16));
  // 创建参考结果 ref，对 a 执行 sigmoid 操作并转换为 Float 类型
  auto ref =
      at::_to_copy(at::sigmoid(a), TensorOptions(kCPU).dtype(at::kFloat));

  // 将输入张量放入输入向量
  std::vector<at::Tensor> inputs = {a};
  // 使用 fmap 将输入向量转换为 IValue 向量
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 运行 TensorExprKernel
  k.run(stack);
  // 获取输出张量 o
  auto o = stack[0].toTensor();
  // 断言输出张量 o 的尺寸与参考结果 ref 相同
  ASSERT_EQ(o.sizes(), ref.sizes());
  // 断言输出张量 o 的数据类型与参考结果 ref 相同
  ASSERT_EQ(o.dtype(), ref.dtype());
  // 断言输出张量 o 与参考结果 ref 的数值近似程度在给定误差范围内
  ASSERT_TRUE(at::allclose(o, ref, 4E-3, 4E-3));
#endif
}
TEST_F(Kernel, CatAndInlineWithAConstantDim) {
  // 定义一个包含两个输入张量的图形字符串
  const auto graph_string = R"IR(
      graph(%0 : Float(1, 512, strides=[1024, 1], requires_grad=0, device=cpu),
            %1 : Float(1, 512, strides=[1024, 1], requires_grad=0, device=cpu)):
        %2 : bool = prim::Constant[value=0]()  // 创建布尔类型常量值为 0
        %3 : int = prim::Constant[value=1]()   // 创建整数类型常量值为 1
        %4 : Tensor[] = prim::ListConstruct(%0, %1)  // 构建张量列表包含 %0 和 %1
        %5 : Float(1, 1024, strides=[1024, 1], requires_grad=0, device=cpu) = aten::cat(%4, %3)  // 在维度 1 上拼接张量列表 %4
        %6 : Tensor[] = prim::ListConstruct(%5)  // 构建包含张量 %5 的列表
        %7 : Float(1, 1024, strides=[1024, 1], requires_grad=0, device=cpu) = aten::cat(%6, %3)  // 在维度 1 上再次拼接张量列表 %6
        %8 : Float(1, 1024, strides=[1024, 1], requires_grad=0, device=cpu) = aten::_cast_Float(%7, %2)  // 将张量 %7 转换为 Float 类型
        return (%8, %7))IR";  // 返回 %8 和 %7 作为结果

  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向图对象
  parseIR(graph_string, &*graph);  // 解析图形字符串并填充到图对象中
  TensorExprKernel k(graph);  // 创建一个张量表达式内核对象

  auto a = at::rand({1, 512}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建一个随机张量 a
  auto b = at::rand({1, 512}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建一个随机张量 b
  auto ref = at::_cast_Float(at::cat({a, b}, 1), 0);  // 计算标准参考输出 ref

  std::vector<at::Tensor> inputs = {a, b};  // 创建输入张量向量包含 a 和 b
  std::vector<IValue> stack = fmap<IValue>(inputs);  // 将输入张量向量转换为 IValue 类型向量
  k.run(stack);  // 运行张量表达式内核
  auto o = stack[0].toTensor();  // 获取运行后的输出张量 o
  ASSERT_EQ(o.sizes(), ref.sizes());  // 断言输出张量 o 的尺寸与 ref 的尺寸相同
  ASSERT_EQ(o.dtype(), ref.dtype());  // 断言输出张量 o 的数据类型与 ref 的数据类型相同
  ASSERT_TRUE(at::allclose(o, ref));  // 断言输出张量 o 与 ref 在数值上近似

}

TEST_F(Kernel, CatWithEmptyInputs) {
  bool curr_cat_wo_conditionals = getCatWoConditionals();  // 获取当前的 cat_wo_conditionals 值
  for (auto cat_wo_conditionals : {true, false}) {  // 遍历 cat_wo_conditionals 的两个可能取值
    getCatWoConditionals() = cat_wo_conditionals;  // 设置 cat_wo_conditionals 的值为当前遍历到的值
    const auto graph_string = R"IR(
        graph(%0 : Float(0, 64, strides=[64, 1], requires_grad=0, device=cpu),
              %1 : Float(10, 64, strides=[64, 1], requires_grad=0, device=cpu)):
          %3 : int = prim::Constant[value=0]()  // 创建整数类型常量值为 0
          %6 : Float(0, 64, strides=[64, 1], requires_grad=0, device=cpu) = aten::tanh(%0)  // 对 %0 进行 tanh 运算
          %7 : Float(10, 64, strides=[64, 1], requires_grad=0, device=cpu) = aten::tanh(%1)  // 对 %1 进行 tanh 运算
          %10 : Tensor[] = prim::ListConstruct(%6, %7)  // 构建张量列表包含 %6 和 %7
          %11 : Float(10, 64, strides=[64, 1], requires_grad=0, device=cpu) = aten::cat(%10, %3)  // 在维度 0 上拼接张量列表 %10
          return (%11))IR";  // 返回 %11 作为结果

    auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向图对象
    parseIR(graph_string, &*graph);  // 解析图形字符串并填充到图对象中
    TensorExprKernel k(graph);  // 创建一个张量表达式内核对象

    auto a = at::rand({0, 64}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建一个随机张量 a
    auto b = at::rand({10, 64}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建一个随机张量 b
    auto ref = at::cat({at::tanh(a), at::tanh(b)}, 0);  // 计算标准参考输出 ref

    std::vector<at::Tensor> inputs = {a, b};  // 创建输入张量向量包含 a 和 b
    std::vector<IValue> stack = fmap<IValue>(inputs);  // 将输入张量向量转换为 IValue 类型向量
    k.run(stack);  // 运行张量表达式内核
    auto o = stack[0].toTensor();  // 获取运行后的输出张量 o
    ASSERT_EQ(o.sizes(), ref.sizes());  // 断言输出张量 o 的尺寸与 ref 的尺寸相同
    ASSERT_EQ(o.dtype(), ref.dtype());  // 断言输出张量 o 的数据类型与 ref 的数据类型相同
    ASSERT_TRUE(at::allclose(o, ref));  // 断言输出张量 o 与 ref 在数值上近似
  }
  getCatWoConditionals() = curr_cat_wo_conditionals;  // 恢复原来的 cat_wo_conditionals 值
}
TEST_F(Kernel, CatWoConditionals) {
  // 保存旧的 'cat_wo_conditionals' 标志状态
  bool old_cat_wo_conditionals = getCatWoConditionals();
  // 设置 'cat_wo_conditionals' 标志为 true
  getCatWoConditionals() = true;

  // 定义表示计算图的字符串
  const auto graph_string = R"IR(
      graph(%a : Float(5, 3, 2, strides=[6, 2, 1], device=cpu),
            %b : Float(5, 7, 2, strides=[14, 2, 1], device=cpu),
            %c : Float(5, 9, 2, strides=[18, 2, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(5, 19, 2, strides=[38, 2, 1]) = aten::cat(%inputs, %dim)
        return (%r))IR";

  // 创建计算图对象
  auto graph = std::make_shared<Graph>();
  // 解析计算图字符串并存储到 graph 中
  parseIR(graph_string, &*graph);

  // 使用 TensorExprKernel 对象处理计算图
  TensorExprKernel k(graph);
  // 获取生成的代码语句
  StmtPtr s = k.getCodeGenStmt();
  // 将生成的代码语句转换为字符串形式
  std::ostringstream oss;
  oss << *s;

  // 定义用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK: for
# CHECK: for
# CHECK: aten_cat
# CHECK: for
# CHECK: for
# CHECK: aten_cat
# CHECK: for
# CHECK: for
# CHECK: aten_cat)IR";
  
  // 使用 FileCheck 运行验证模式，检查生成的代码
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 生成测试用的随机张量 a, b, c
  auto a = at::rand({5, 3, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 7, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto c = at::rand({5, 9, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  // 创建参考结果张量 ref，使用 PyTorch 的 cat 函数连接张量
  auto ref = at::cat({a, b, c}, 1);

  // 将输入张量存储在 vector 中
  std::vector<at::Tensor> inputs = {a, b, c};
  // 将输入张量转换为 IValue 格式并存储在 stack 中
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 运行 TensorExprKernel 对象处理的计算图
  k.run(stack);
  // 获取输出张量 o
  auto o = stack[0].toTensor();

  // 检查输出张量 o 和参考张量 ref 的尺寸是否相等
  TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
  // 检查输出张量 o 和参考张量 ref 的数据类型是否相同
  TORCH_CHECK_EQ(o.dtype(), ref.dtype());
  size_t num_el = 1;
  // 检查输出张量 o 和参考张量 ref 的每个维度的大小是否相同，并计算总元素数
  for (const auto idx : c10::irange(ref.sizes().size())) {
    TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
    num_el *= ref.sizes()[idx];
  }

  // 检查输出张量 o 和参考张量 ref 的内容是否一致
  for (const auto i : c10::irange(num_el)) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
  }

  // 恢复旧的 'cat_wo_conditionals' 标志状态
  getCatWoConditionals() = old_cat_wo_conditionals;
}

TEST_F(Kernel, OptimizeConditionals) {
  // 保存旧的 'cat_wo_conditionals' 和 'opt_conditionals' 标志状态
  bool old_cat_wo_conditionals = getCatWoConditionals();
  bool old_opt_conditionals = getOptConditionals();
  // 设置 'cat_wo_conditionals' 标志为 false，'opt_conditionals' 标志为 true
  getCatWoConditionals() = false;
  getOptConditionals() = true;

  // 定义表示计算图的字符串
  const auto graph_string = R"IR(
      graph(%a : Float(5, 3, strides=[3, 1], device=cpu),
            %b : Float(5, 7, strides=[7, 1], device=cpu),
            %c : Float(5, 9, strides=[9, 1], device=cpu)):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(5, 19, strides=[19, 1]) = aten::cat(%inputs, %dim)
        %t : Float(5, 19, strides=[19, 1]) = aten::relu(%r)
        return (%t))IR";

  // 创建计算图对象
  auto graph = std::make_shared<Graph>();
  // 解析计算图字符串并存储到 graph 中
  parseIR(graph_string, &*graph);

  // 使用 TensorExprKernel 对象处理计算图
  TensorExprKernel k(graph);
  // 获取生成的代码语句
  StmtPtr s = k.getCodeGenStmt();
  // 将生成的代码语句转换为字符串形式
  std::ostringstream oss;
  oss << *s;

  // 定义用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for
# CHECK-NEXT: for
# CHECK-NEXT: aten_relu
# CHECK: for
# CHECK-NEXT: aten_relu
# CHECK: for
# CHECK-NEXT: aten_relu
# CHECK-NOT: Allocate)IR";
  
  // 使用 FileCheck 运行验证模式，检查生成的代码
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 恢复旧的 'cat_wo_conditionals' 和 'opt_conditionals' 标志状态
  getCatWoConditionals() = old_cat_wo_conditionals;
  getOptConditionals() = old_opt_conditionals;
}
// 运行自定义的文件检查，验证 oss 字符串是否满足 verification_pattern 的要求
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

// 创建一个大小为 [5, 3] 的随机张量 a，数据类型为 Float，存储在 CPU 上
auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
// 创建一个大小为 [5, 7] 的随机张量 b，数据类型为 Float，存储在 CPU 上
auto b = at::rand({5, 7}, TensorOptions(kCPU).dtype(at::kFloat));
// 创建一个大小为 [5, 9] 的随机张量 c，数据类型为 Float，存储在 CPU 上
auto c = at::rand({5, 9}, TensorOptions(kCPU).dtype(at::kFloat));
// 计算将张量 a、b、c 沿第一个维度拼接后的张量，并对其应用 relu 激活函数
auto ref = at::relu(at::cat({a, b, c}, 1));

// 创建一个张量向量 inputs 包含张量 a、b、c
std::vector<at::Tensor> inputs = {a, b, c};
// 将 inputs 中的每个张量转换为 IValue 类型，存储在 stack 中
std::vector<IValue> stack = fmap<IValue>(inputs);
// 调用某个运行时对象 k 的方法 run，传入 stack 中的数据执行计算
k.run(stack);
// 从 stack 中取出第一个元素，并将其转换为张量类型，存储在 o 中
auto o = stack[0].toTensor();

// 检查张量 o 和参考张量 ref 的尺寸是否相等
TORCH_CHECK_EQ(o.sizes().size(), ref.sizes().size());
// 检查张量 o 和参考张量 ref 的数据类型是否相等
TORCH_CHECK_EQ(o.dtype(), ref.dtype());
// 计算参考张量 ref 中元素的总数
size_t num_el = 1;
for (const auto idx : c10::irange(ref.sizes().size())) {
  // 检查张量 o 和参考张量 ref 在每个维度上的大小是否相等，并累计元素总数
  TORCH_CHECK_EQ(o.sizes()[idx], ref.sizes()[idx]);
  num_el *= ref.sizes()[idx];
}

// 检查张量 o 和参考张量 ref 的内容是否一致
for (const auto i : c10::irange(num_el)) {
  TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
}
// 恢复旧的优化条件
getOptConditionals() = old_opt_conditionals;
getCatWoConditionals() = old_cat_wo_conditionals;
}

namespace {

// 根据标量类型返回对应的字符串表示
std::string dtypeConstant(ScalarType scalar_type) {
  if (scalar_type == ScalarType::Undefined) {
    return "None = prim::Constant()";
  } else {
    at::jit::TemplateEnv env_dtype;
    env_dtype.d("scalar_type", static_cast<int>(scalar_type));
    return format("int = prim::Constant[value=${scalar_type}]()", env_dtype);
  }
}

// 创建一个 iota 张量，元素从 0 开始连续排列，形状由 sizes 指定，使用给定的选项
at::Tensor iotaTensor(IntArrayRef sizes, const at::TensorOptions& options) {
  // 计算张量的元素总数
  int64_t numel = std::accumulate(
      sizes.begin(),
      sizes.end(),
      1,
      // 使用乘法运算符计算元素总数
      std::multiplies<int64_t>());
  // 创建一个元素为从 0 开始到 numel-1 的连续浮点数的向量
  std::vector<float> values(numel);
  std::iota(values.begin(), values.end(), 0);
  // 使用给定选项创建张量 a，并将向量 values 的数据填充到张量中，然后重塑为指定形状
  auto a = at::tensor(values, options);
  return a.reshape(sizes);
}

} // namespace

TEST_F(Kernel, SumAllAxes) {
  // 测试在所有轴上求和的降低操作
  // 定义一个 IR 图形板，用于描述对给定张量进行求和操作的计算图
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : ${dtype}
        %2 : ${out_dtype}(requires_grad=0, device=cpu) = aten::sum(%0, %1)
        return (%2))IR";
  // 创建一个元素为从 0 开始到 14 的连续浮点数的 iota 张量 a，形状为 [5, 3]，数据类型为 Float，存储在 CPU 上
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // 针对每种标量类型进行迭代测试
  for (auto scalar_type : {ScalarType::Undefined, ScalarType::Double}) {
    // 创建一个模板环境对象 env
    at::jit::TemplateEnv env;
    env.s("dtype", dtypeConstant(scalar_type));
    // 根据标量类型选择输出数据类型的字符串表示
    if (scalar_type == ScalarType::Undefined) {
      env.s("out_dtype", "Float");
    } else {
      env.s("out_dtype", "Double");
    }
    // 使用环境变量 env 格式化图形模板，生成实际的图形字符串
    const auto graph_string = format(graph_template, env);

    // 创建一个空张量 o，用于存储计算结果
    auto o = at::empty({}, TensorOptions(kCPU));
    std::optional<c10::ScalarType> dtype;
    // 如果标量类型不是 Undefined，则将其转换为 c10::ScalarType 类型
    if (scalar_type != ScalarType::Undefined) {
      dtype = static_cast<c10::ScalarType>(scalar_type);
    }
    // 调用 Tensor a 的 sum 方法，并将结果赋值给 auto 类型的 ref 变量
    auto ref = a.sum(/*dtype=*/dtype);

    // 创建一个 TensorExprKernel 对象 k，传入计算图 graph
    TensorExprKernel k(graph);

    // 创建一个包含 Tensor a 的向量 inputs，用于传递给内核对象 k
    std::vector<at::Tensor> inputs = {a};

    // 获取代码生成语句，并将其赋值给 StmtPtr 类型的变量 s
    StmtPtr s = k.getCodeGenStmt();

    // 创建一个字符串输出流 oss，并将 s 的内容写入其中
    std::ostringstream oss;
    oss << *s;

    // 检查生成的中间表示（IR）是否符合预期的验证模式
    const std::string& verification_pattern =
        R"IR(
// 定义测试类 Kernel，用于测试张量表达式的内核操作
TEST_F(Kernel, SumOneAxis) {
  // 测试在单个轴上求和的降维操作

  // 定义张量表达式的模板字符串
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int[] = prim::Constant[value=[${dim}]]()
        %2 : bool = prim::Constant[value=${keepdim}]()
        %3 : ${dtype}
        %4 : ${out_dtype}(${size}, strides=[${strides}], device=cpu) = aten::sum(%0, %1, %2, %3)
        return (%4))IR";

  // 创建一个大小为 [5, 3] 的浮点型张量 a
  auto a = iotaTensor({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // 遍历不同的维度和保持维度标志
  for (int dim = -a.dim(); dim < a.dim(); ++dim) {
    for (bool keepdim : {false, true}) {
      // 遍历不同的标量类型
      for (auto scalar_type : {ScalarType::Undefined, ScalarType::Double}) {
        // 创建模板环境对象
        at::jit::TemplateEnv env;
        env.d("dim", dim);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(scalar_type));

        std::optional<c10::ScalarType> dtype;
        // 如果标量类型不是未定义的，则转换为对应的枚举类型
        if (scalar_type != ScalarType::Undefined) {
          dtype = static_cast<c10::ScalarType>(scalar_type);
        }

        // 计算张量在指定维度上的和，并获得参考结果
        auto ref = a.sum({dim}, /*keepdim=*/keepdim, /*dtype=*/dtype);

        // 根据标量类型设置输出张量的数据类型
        if (scalar_type == ScalarType::Undefined) {
          env.s("out_dtype", "Float");
        } else {
          env.s("out_dtype", "Double");
        }

        // 将张量的大小和步幅转换为字符串形式
        env.s("size", li_to_str(ref.sizes()));
        env.s("strides", li_to_str(ref.strides()));

        // 根据模板和环境字符串格式化生成图形字符串
        const auto graph_string = format(graph_template, env);

        // 创建共享的图对象
        auto graph = std::make_shared<Graph>();
        // 解析图形字符串到图对象中
        parseIR(graph_string, &*graph);

        // 创建一个空张量 o 用于存储运行结果
        auto o = at::empty({}, TensorOptions(kCPU));
        // 创建张量表达式内核对象 k
        TensorExprKernel k(graph);
        // 创建输入张量的向量
        std::vector<at::Tensor> inputs = {a};
        // 获取代码生成语句对象
        StmtPtr s = k.getCodeGenStmt();

        // 创建字符串流对象 oss，用于输出代码生成的字符串
        std::ostringstream oss;
        oss << *s;

        // 检查生成的中间表示是否符合预期
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int64_t
# CHECK-NEXT: sum
# CHECK-NEXT: for (int64_t
# CHECK-NEXT:   sum)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        // 将输入张量转换为 IValue 的向量 stack
        std::vector<IValue> stack = fmap<IValue>(inputs);
        // 运行张量表达式内核
        k.run(stack);
        // 将运行结果转换为张量 o
        o = stack[0].toTensor();

        // 断言结果张量 o 的大小与参考结果的大小一致
        ASSERT_EQ(o.sizes(), ref.sizes());
        // 断言结果张量 o 的数据类型与参考结果的数据类型一致
        ASSERT_EQ(o.dtype(), ref.dtype());
        // 断言结果张量 o 与参考结果 ref 在一定误差范围内相等
        ASSERT_TRUE(at::allclose(o, ref, 4E-3, 4E-3));
      }
    }
  }
}
TEST_F(Kernel, SumMultipleAxes) {
  // Test lowering of sum on multiple axes.

  // 定义一个包含模板字符串的常量，用于描述一个计算图
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], requires_grad=0, device=cpu)):
        %1 : int = prim::Constant[value=${dim1}]()
        %2 : int = prim::Constant[value=${dim2}]()
        %3 : int[] = prim::ListConstruct(%1, %2)
        %4 : bool = prim::Constant[value=${keepdim}]()
        %5 : ${dtype}
        %6 : Float(${size}, strides=[${strides}], requires_grad=0, device=cpu) = aten::sum(%0, %3, %4, %5)
        return (%6))IR";

  // 创建一个具有给定形状和数据类型的新张量
  auto a = iotaTensor({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // 仅迭代正值的轴，以保持运行时间合理，因为成对轴的数量是二次的。
  for (const auto dim1 : c10::irange(a.dim())) {
    for (int dim2 = dim1 + 1; dim2 < a.dim(); ++dim2) {
      for (bool keepdim : {false, true}) {
        // 创建模板环境并填充维度、保持维度信息
        at::jit::TemplateEnv env;
        env.d("dim1", dim1);
        env.d("dim2", dim2);
        env.d("keepdim", keepdim);
        env.s("dtype", dtypeConstant(ScalarType::Undefined));

        // 创建一个空张量o作为输出
        auto o = at::empty({}, TensorOptions(kCPU));

        // 计算在指定轴上求和的参考值
        auto ref = a.sum(IntArrayRef{dim1, dim2}, /*keepdim=*/keepdim);

        // 将参考张量的大小和步幅信息填充到环境中
        env.s("size", li_to_str(ref.sizes()));
        env.s("strides", li_to_str(ref.strides()));

        // 根据模板字符串和环境创建计算图
        const auto graph_string = format(graph_template, env);
        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        // 创建张量表达式内核对象
        TensorExprKernel k(graph);

        // 准备输入张量列表
        std::vector<at::Tensor> inputs = {a};

        // 获取代码生成的语句
        StmtPtr s = k.getCodeGenStmt();

        // 创建输出流，将生成的代码打印到流中
        std::ostringstream oss;
        oss << *s;

        // 检查生成的IR代码
        const std::string& verification_pattern =
            R"IR(
# CHECK: for (int64_t
# CHECK: for (int64_t
# CHECK: for (int64_t
# CHECK: for (int64_t
# CHECK: sum)IR";
        torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        // 运行张量表达式内核
        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);

        // 将运行后得到的张量结果存入o
        o = stack[0].toTensor();

        // 断言输出张量o与参考张量ref的形状相同
        ASSERT_EQ(o.sizes(), ref.sizes());

        // 断言输出张量o与参考张量ref的数据类型相同
        ASSERT_EQ(o.dtype(), ref.dtype());

        // 断言输出张量o与参考张量ref的数值接近
        ASSERT_TRUE(at::allclose(o, ref));
      }
    }
  }
}

// This test and the following ones testing Softmax only tests with dim set
// to one of the valid input dimensions. It does not test with dim=None
// because that is supposed to be deprecated.
TEST_F(Kernel, Softmax2D) {
  const auto graph_template = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %dt_float : int = prim::Constant[value=7]()
        %dt_none : NoneType = prim::Constant()
        %4 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %${dt})
        return (%4))IR";

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // 定义字符串模板，用于验证生成的 IR 是否符合预期
  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i${other_dim} = 0; i${other_dim} < ${other_dim_size}
        # CHECK: for (int i${softmax_dim} = 0; i${softmax_dim} < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i${other_dim}_1 = 0; i${other_dim}_1 < ${other_dim_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_2 = 0; i0_2 < 5
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 3
        # CHECK-NEXT: aten_softmax)IR";

  // 遍历条件，包括空数据类型和对数 softmax 的两种情况
  for (bool empty_dtype : {false, true}) {
    for (auto log_softmax : {false, true}) {
      // 遍历张量的维度
      for (const auto softmax_dim : c10::irange(a.dim())) {
        auto softmax_dim_size = a.sizes()[softmax_dim];
        auto other_dim = (softmax_dim + 1) % a.dim();
        auto ref =
            log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);
        
        // 创建模板环境
        at::jit::TemplateEnv env;
        env.d("dim", softmax_dim);
        env.s("op", log_softmax ? "log_softmax" : "softmax");
        env.s("size", li_to_str(ref.sizes()));
        env.s("strides", li_to_str(ref.strides()));
        env.s("dt", empty_dtype ? "dt_none" : "dt_float");

        // 格式化生成图的字符串
        const auto graph_string = format(graph_template, env);

        // 解析生成的 IR
        auto graph = std::make_shared<Graph>();
        parseIR(graph_string, &*graph);

        // 创建 TensorExprKernel 对象
        TensorExprKernel k(graph);
        std::vector<at::Tensor> inputs = {a};

        // 获取代码生成的语句
        StmtPtr s = k.getCodeGenStmt();

        // 创建输出流，用于生成代码的字符串表示
        std::ostringstream oss;
        oss << *s;

        // 创建验证模板环境
        at::jit::TemplateEnv ver_env;
        ver_env.d("other_dim", other_dim);
        ver_env.d("other_dim_size", a.sizes()[other_dim]);
        ver_env.d("softmax_dim", softmax_dim);
        ver_env.d("softmax_dim_size", softmax_dim_size);
        const auto verification_pattern =
            format(verification_template, ver_env);

        // 运行验证模板（暂时禁用，待评估 exp() 内联后再启用）
        // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

        // 运行生成的代码
        std::vector<IValue> stack = fmap<IValue>(inputs);
        k.run(stack);
        auto output = stack[0].toTensor();

        // 断言生成的输出张量与参考张量的大小相等
        ASSERT_EQ(output.sizes(), ref.sizes());

        // 断言生成的输出张量与参考张量在数值上是否接近
        ASSERT_TRUE(at::allclose(output, ref));
      }
    }
  }
}
TEST_F(Kernel, Softmax3D) {
  // 定义一个模板字符串，表示一个带有特定参数的计算图
  const auto graph_template = R"IR(
      graph(%0 : Float(3, 4, 5, strides=[20, 5, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %2)
        return (%3))IR";

  // 生成一个随机的 3x4x5 的 FloatTensor
  auto a = at::rand({3, 4, 5}, TensorOptions(kCPU).dtype(at::kFloat));

  // 定义一个用于验证的模板字符串，包含待检查的循环和函数调用
  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i${dim1} = 0; i${dim1} < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2} = 0; i${dim2} < ${dim2_size}
        # CHECK: for (int i${softmax_dim} = 0; i${softmax_dim} < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i${dim1}_1 = 0; i${dim1}_1 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_1 = 0; i${dim2}_1 < ${dim2_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_2 = 0; i0_2 < 3
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 4
        # CHECK-NEXT: for (int i2_2 = 0; i2_2 < 5
        # CHECK-NEXT: aten_softmax)IR";

  // 遍历两种 softmax 操作：log_softmax 和 softmax
  for (auto log_softmax : {false, true}) {
    // 遍历输入张量的所有维度
    for (const auto softmax_dim : c10::irange(a.dim())) {
      // 获取当前 softmax 维度的大小
      auto softmax_dim_size = a.sizes()[softmax_dim];
      std::vector<int> other_dims;
      // 构建除了 softmax 维度之外的其它维度列表
      for (const auto i : c10::irange(a.dim())) {
        if (i != softmax_dim) {
          other_dims.push_back(i);
        }
      }
      // 计算参考输出（reference output）
      auto ref =
          log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);

      // 创建一个模板环境变量并设置其值
      at::jit::TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      // 格式化计算图模板字符串并解析为图对象
      const auto graph_string = format(graph_template, env);
      auto graph = std::make_shared<Graph>();
      parseIR(graph_string, &*graph);

      // 构建张量表达式内核对象
      TensorExprKernel k(graph);
      std::vector<at::Tensor> inputs = {a};
      StmtPtr s = k.getCodeGenStmt();

      // 创建一个输出流，以将生成的代码写入其中
      std::ostringstream oss;
      oss << *s;

      // 创建一个模板环境变量用于验证，并设置其值
      at::jit::TemplateEnv ver_env;
      ver_env.d("dim1", other_dims[0]);
      ver_env.d("dim1_size", a.sizes()[other_dims[0]]);
      ver_env.d("dim2", other_dims[1]);
      ver_env.d("dim2_size", a.sizes()[other_dims[1]]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);

      // 格式化验证模板字符串并进行格式化
      const auto verification_pattern = format(verification_template, ver_env);

      // 临时禁用验证字符串，直到内联 exp() 函数的性能影响测量和确定
      // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

      // 运行张量表达式内核，传入输入张量，并获取输出结果
      std::vector<IValue> stack = fmap<IValue>(inputs);
      k.run(stack);
      auto output = stack[0].toTensor();

      // 使用断言检查输出张量的大小和内容是否与参考输出一致
      ASSERT_EQ(output.sizes(), ref.sizes());
      ASSERT_TRUE(at::allclose(output, ref));
    }
  }
}
TEST_F(Kernel, Softmax4D) {
  // 定义 IR 字符串模板，包含占位符 ${dim}、${size}、${strides}、${op}
  const auto graph_template = R"IR(
      graph(%0 : Float(2, 3, 2, 3, strides=[18, 6, 3, 1], device=cpu)):
        %1 : int = prim::Constant[value=${dim}]()
        %2 : int = prim::Constant[value=7]()
        %3 : Float(${size}, strides=[${strides}]) = aten::${op}(%0, %1, %2)
        return (%3))IR";

  // 生成随机张量 a，形状为 [2, 3, 2, 3]，CPU 上的 Float 类型
  auto a = at::rand({2, 3, 2, 3}, TensorOptions(kCPU).dtype(at::kFloat));

  // 定义 IR 字符串模板，包含占位符 ${dim1}、${dim2}、${dim3}、${softmax_dim} 等
  const std::string& verification_template =
      R"IR(
        # CHECK: for (int i${dim1} = 0; i${dim1} < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2} = 0; i${dim2} < ${dim2_size}
        # CHECK-NEXT: for (int i${dim3} = 0; i${dim3} < ${dim3_size}
        # CHECK: for (int i${softmax_dim} = 0; i${softmax_dim} < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_max
        # CHECK: for (int i${dim1}_1 = 0; i${dim1}_1 < ${dim1_size}
        # CHECK-NEXT: for (int i${dim2}_1 = 0; i${dim2}_1 < ${dim2_size}
        # CHECK-NEXT: for (int i${dim3}_1 = 0; i${dim3}_1 < ${dim3_size}
        # CHECK: for (int i${softmax_dim}_1 = 0; i${softmax_dim}_1 < ${softmax_dim_size}
        # CHECK-NEXT: aten_softmax_sum
        # CHECK: for (int i0_2 = 0; i0_2 < 2
        # CHECK-NEXT: for (int i1_2 = 0; i1_2 < 3
        # CHECK-NEXT: for (int i2_2 = 0; i2_2 < 2
        # CHECK-NEXT: for (int i3_2 = 0; i3_2 < 3
        # CHECK-NEXT: aten_softmax)IR";

  // 针对 log_softmax 为 false 和 true 分别进行迭代
  for (auto log_softmax : {false, true}) {
    // 遍历张量 a 的维度
    for (const auto softmax_dim : c10::irange(a.dim())) {
      // 获取当前维度的大小
      auto softmax_dim_size = a.sizes()[softmax_dim];
      // 创建一个存储非当前维度的索引的向量
      std::vector<int> other_dims;
      // 遍历所有维度
      for (const auto i : c10::irange(a.dim())) {
        // 如果当前维度不是 softmax_dim，则将其索引添加到 other_dims 中
        if (i != softmax_dim) {
          other_dims.push_back(i);
        }
      }
      // 根据 log_softmax 标志选择计算 log_softmax 或 softmax
      auto ref = log_softmax ? a.log_softmax(softmax_dim) : a.softmax(softmax_dim);

      // 创建模板环境
      at::jit::TemplateEnv env;
      env.d("dim", softmax_dim);
      env.s("op", log_softmax ? "log_softmax" : "softmax");
      env.s("size", li_to_str(ref.sizes()));
      env.s("strides", li_to_str(ref.strides()));

      // 格式化模板字符串
      const auto graph_string = format(graph_template, env);

      // 创建图对象
      auto graph = std::make_shared<Graph>();
      // 解析 IR 字符串并填充图对象
      parseIR(graph_string, &*graph);

      // 创建 TensorExprKernel 对象
      TensorExprKernel k(graph);
      // 创建输入张量向量
      std::vector<at::Tensor> inputs = {a};
      // 获取代码生成语句
      StmtPtr s = k.getCodeGenStmt();

      // 创建输出流对象
      std::ostringstream oss;
      oss << *s;

      // 创建验证模板环境
      at::jit::TemplateEnv ver_env;
      ver_env.d("dim1", other_dims[0]);
      ver_env.d("dim1_size", a.sizes()[other_dims[0]);
      ver_env.d("dim2", other_dims[1]);
      ver_env.d("dim2_size", a.sizes()[other_dims[1]);
      ver_env.d("dim3", other_dims[2]);
      ver_env.d("dim3_size", a.sizes()[other_dims[2]);
      ver_env.d("softmax_dim", softmax_dim);
      ver_env.d("softmax_dim_size", softmax_dim_size);
      // 格式化验证模板字符串
      const auto verification_pattern = format(verification_template, ver_env);

      // 暂时禁用验证字符串，直到 exp() 的内联化被基准测试确定
      // torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

      // 创建输入值向量并运行 TensorExprKernel
      std::vector<IValue> stack = fmap<IValue>(inputs);
      k.run(stack);
      // 获取输出张量
      auto output = stack[0].toTensor();
      // 断言输出张量的大小与参考张量的大小相同
      ASSERT_EQ(output.sizes(), ref.sizes());
      // 断言输出张量与参考张量在误差范围内相等
      ASSERT_TRUE(at::allclose(output, ref));
    }
  }
}

TEST_F(Kernel, SignTest) {
  // 定义一个包含 IR 的模板字符串，使用 ${dtype} 和 ${size} 作为占位符
  const auto graph_template = R"IR(
      graph(%0 : ${dtype}(${size}, strides=[1], device=cpu)):
        %2 : ${dtype}(${size}, strides=[1]) = aten::sign(%0)
        return (%2))IR";

  // 定义一个 lambda 函数 run_test，用于执行测试
  auto run_test = [](const std::string& graph_string, const at::Tensor& input) {
    // 创建一个共享指针指向 Graph 对象
    auto graph = std::make_shared<Graph>();
    // 解析输入的 IR 字符串并填充到 graph 对象中
    parseIR(graph_string, &*graph);

    // 创建 TensorExprKernel 对象 k，用于生成代码语句
    TensorExprKernel k(graph);
    // 获取代码生成后的语句
    StmtPtr s = k.getCodeGenStmt();

    // 准备输入数据和栈，运行代码生成的内核
    std::vector<at::Tensor> inputs = {input};
    std::vector<IValue> stack = fmap<IValue>(inputs);
    k.run(stack);

    // 从栈中获取输出张量 o
    auto o = stack[0].toTensor();
    // 计算输入张量 input 的标志函数的参考结果 ref
    auto ref = at::sign(input);
    // 使用 ASSERT_TRUE 确保 o 和 ref 的所有元素接近
    ASSERT_TRUE(at::allclose(o, ref));
  };

  // 定义通用的 TensorOptions
  auto common_options = at::TensorOptions()
                            .layout(at::kStrided)
                            .device(at::kCPU)
                            .requires_grad(false);
  // 定义默认输入的大小
  int default_input_size = 100;

  // 遍历浮点类型和双精度类型的标量类型
  for (auto scalar_type : {ScalarType::Float, ScalarType::Double}) {
    // 初始化 corner_case_inputs 和 env
    at::Tensor corner_case_inputs;
    at::jit::TemplateEnv env;

    // 根据标量类型设置相应的环境变量和选项
    switch (scalar_type) {
      case ScalarType::Float: {
        // 设置环境变量 dtype 和选项 dtype 为 Float
        env.s("dtype", "Float");
        auto options = common_options.dtype(at::kFloat);

        // 创建浮点数输入向量 input_float，包含特殊值
        std::vector<float> input_float = {
            0.0f,
            -0.0f,
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::nanf("1"),
            -std::nanf("1")};

        // 使用 from_blob 创建 corner_case_inputs
        corner_case_inputs = at::from_blob(
            input_float.data(),
            {static_cast<long>(input_float.size())},
            options);

        // 生成随机输入和合并输入
        auto rand_input = at::rand({default_input_size}, options);
        auto input = at::cat({rand_input, corner_case_inputs});

        // 设置环境变量 size 为输入张量的元素数
        env.d("size", at::numel(input));

        // 格式化 graph_template，生成具体的 IR 字符串 graph_string
        const auto graph_string = format(graph_template, env);

        // 运行测试
        run_test(graph_string, input);
        break;
      }
      case ScalarType::Double: {
        // 设置环境变量 dtype 和选项 dtype 为 Double
        env.s("dtype", "Double");
        auto options = common_options.dtype(at::kDouble);

        // 创建双精度输入向量 input_double，包含特殊值
        std::vector<double> input_double = {
            0.0,
            -0.0,
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            std::nan("1"),
            -std::nan("1")};

        // 使用 from_blob 创建 corner_case_inputs
        corner_case_inputs = at::from_blob(
            input_double.data(),
            {static_cast<long>(input_double.size())},
            options);

        // 生成随机输入和合并输入
        auto rand_input = at::rand({default_input_size}, options);
        auto input = at::cat({rand_input, corner_case_inputs});

        // 设置环境变量 size 为输入张量的元素数
        env.d("size", at::numel(input));

        // 格式化 graph_template，生成具体的 IR 字符串 graph_string
        const auto graph_string = format(graph_template, env);

        // 运行测试
        run_test(graph_string, input);
        break;
      }
      default:
        throw unsupported_dtype();
    }
  }
}
TEST_F(Kernel, InlineProducerIntoReduction) {
  // Inline producer (mul) into reduction (sum).
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%0, %1)
        %3 : int = prim::Constant[value=7]()
        %4 : Double(device=cpu) = aten::sum(%2, %3)
        return (%4))IR";
  auto graph = std::make_shared<Graph>();
  // 解析图形字符串并将结果存储在共享指针graph中
  parseIR(graph_string, &*graph);

  // 创建TensorExprKernel对象k，并传入解析后的图形
  TensorExprKernel k(graph);
  // 获取代码生成语句并存储在StmtPtr对象s中
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  // 检查生成的IR代码。
  // 应该最终只有一个循环。
  const std::string& verification_pattern =
      R"IR(
        # CHECK: for (int64_t i_1 = 0ll; i_1 < 5
        # CHECK-NEXT: for (int64_t j_1 = 0ll; j_1 < 3
        # CHECK-NEXT:   sum
        # CHECK-NOT: for)IR";
  // 运行FileCheck来验证生成的IR代码是否符合指定模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 生成随机输入张量a和b，均为5x3的浮点数张量
  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {a, b};
  // 将输入张量转换为IValue类型的堆栈
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 运行TensorExprKernel对象k上的代码生成语句
  k.run(stack);
  // 获取运行后的输出张量o
  auto o = stack[0].toTensor();
  // 计算参考结果(ref)，即a*b的元素和，并以双精度数值返回
  auto ref = (a * b).sum(at::kDouble);
  // 断言运行后的输出张量o与参考结果ref在误差允许范围内相等
  ASSERT_TRUE(at::allclose(o, ref));
}

TEST_F(Kernel, InlineReductionIntoConsumer) {
  // Inline producer (mul %2) into reduction (sum %4) but DO NOT
  // inline the reduction into consumer (mul %4).
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[3, 1], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : int = prim::Constant[value=6]()
        %4 : Float(device=cpu) = aten::sum(%2, %3)
        %5 : Float(5, 3, strides=[3, 1], device=cpu) = aten::mul(%2, %4)
        return (%5))IR";
  auto graph = std::make_shared<Graph>();
  // 解析图形字符串并将结果存储在共享指针graph中
  parseIR(graph_string, &*graph);

  // 创建TensorExprKernel对象k，并传入解析后的图形
  TensorExprKernel k(graph);
  // 获取代码生成语句并存储在StmtPtr对象s中
  StmtPtr s = k.getCodeGenStmt();
  std::ostringstream oss;
  oss << *s;

  // 检查生成的IR代码。
  // 应该最终有两个循环。
  const std::string& verification_pattern =
      R"IR(
        # CHECK: for (int64_t i_1 = 0ll; i_1 < 5
        # CHECK-NEXT: for (int64_t j_1 = 0ll; j_1 < 3
        # CHECK-NEXT:   sum
        # CHECK: for (int64_t i_2 = 0ll; i_2 < 5
        # CHECK-NEXT: for (int64_t j_2 = 0ll; j_2 < 3
        # CHECK-NEXT:   aten_mul
        # CHECK-NOT: for)IR";
  // 运行FileCheck来验证生成的IR代码是否符合指定模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 生成随机输入张量a和b，均为5x3的浮点数张量
  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  std::vector<at::Tensor> inputs = {a, b};
  // 将输入张量转换为IValue类型的堆栈
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 运行TensorExprKernel对象k上的代码生成语句
  k.run(stack);
  // 获取运行后的输出张量o
  auto o = stack[0].toTensor();
  // 计算参考结果(ref)，即(a*b).sum() * (a*b)，并以浮点数值返回
  auto ref = (a * b).sum(at::kFloat) * (a * b);
  // 断言运行后的输出张量o与参考结果ref在误差允许范围内相等
  ASSERT_TRUE(at::allclose(o, ref));
}
TEST_F(Kernel, SanitizeNames_CUDA) {
  // 定义一个包含两个 CUDA 设备上的 Float 张量的 IR 图的字符串表示
  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cuda:0),
            %1 : Float(5, 3, strides=[3, 1], device=cuda:0)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %4 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%4))IR";
  
  // 创建一个空的图对象 shared_ptr
  auto graph = std::make_shared<Graph>();
  
  // 将 IR 字符串解析到图对象中
  parseIR(graph_string, &*graph);
  
  // 设置输入张量的调试名称
  graph->inputs().at(0)->setDebugName("aten::add:");
  graph->inputs().at(1)->setDebugName("aten::add_");
  
  // 创建 TensorExprKernel 对象，传入图对象
  TensorExprKernel k(graph);
  
  // 创建两个随机初始化的 CUDA Float 张量 a 和 b
  auto a = at::rand({5, 3}, TensorOptions(kCUDA).dtype(at::kFloat));
  auto b = at::rand({5, 3}, TensorOptions(kCUDA).dtype(at::kFloat));
  
  // 计算参考结果 ref，即 a * (a * b)
  auto ref = a * (a * b);
  
  // 准备输入张量的 vector 和 IValue 的 stack
  std::vector<at::Tensor> inputs = {a, b};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  
  // 运行 TensorExprKernel 的 run 方法，传入 stack
  k.run(stack);
  
  // 获取运行后的输出张量 o
  auto o = stack[0].toTensor();
  
  // 断言输出张量 o 与参考结果 ref 在数值上是否全部接近
  ASSERT_TRUE(at::allclose(o, ref));
}

TEST_F(Kernel, SanitizeConstants_CUDA) {
  // 定义一个包含 CUDA 设备上常量处理的 IR 图的字符串表示
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cuda:0)):
          %none : NoneType = prim::Constant()
          %size : int = prim::Constant[value=16]()
          %sizes : int[] = prim::ListConstruct(%size, %size)
          %30 : Device = prim::Constant[value="cuda"]()
          %y : Float(16, 16, strides=[16, 1], device=cuda:0) = aten::ones(%sizes, %none, %none, %30, %none)
          %z : Float(16, 16, strides=[16, 1], device=cuda:0) = aten::mul(%x, %y)
          return (%z))IR";
  
  // 创建一个空的图对象 shared_ptr
  auto graph = std::make_shared<Graph>();
  
  // 将 IR 字符串解析到图对象中
  parseIR(graph_string, &*graph);
  
  // 执行常量传播，因为 IRParser 不支持张量常量，我们插入一个调用来代替它们
  ConstantPropagation(graph);
  
  // 设置常量节点的调试名称，包含不允许的特殊字符
  graph->nodes().front()->output()->setDebugName("illegal.name");
  
  // 检查图中是否有带有非法名称的常量节点
  auto const_node = graph->nodes().front();
  ASSERT_EQ(const_node->kind(), prim::Constant);
  ASSERT_NE(const_node->output()->debugName().find('.'), std::string::npos);
  
  // 创建 TensorExprKernel 对象，传入图对象
  TensorExprKernel k(graph);
  
  // 创建一个随机初始化的 CUDA Float 张量 x
  auto x = at::rand({16, 16}, TensorOptions(kCUDA).dtype(at::kFloat));
  
  // 准备输入张量的 vector 和 IValue 的 stack
  std::vector<at::Tensor> inputs = {x};
  std::vector<IValue> stack = fmap<IValue>(inputs);
  
  // 运行 TensorExprKernel 的 run 方法，传入 stack
  k.run(stack);
  
  // 获取运行后的输出张量 o
  auto o = stack[0].toTensor();
  
  // 创建一个与 x 形状相同的全为 1 的张量 y
  auto y = at::ones({16, 16}, TensorOptions(kCUDA).dtype(at::kFloat));
  
  // 计算参考结果 ref，即 x * y
  auto ref = x * y;
  
  // 断言输出张量 o 与参考结果 ref 在数值上是否全部接近
  ASSERT_TRUE(at::allclose(o, ref));
}
TEST_F(Kernel, ConstantTensors) {
  // 定义包含IR图形字符串的常量，描述操作序列及其参数
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()  // 创建一个空类型常量
          %size : int = prim::Constant[value=16]()  // 创建一个整数常量值为16
          %sizes : int[] = prim::ListConstruct(%size, %size)  // 构造一个整数数组常量，包含两个16
          %y : Float(16, 16, strides=[16, 1], device=cpu) = aten::ones(%sizes, %none, %none, %none, %none)  // 用aten::ones创建一个全1的Tensor
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)  // 用aten::mul计算x和y的元素乘积
          return (%z))IR";  // 返回计算结果z
  auto graph = std::make_shared<Graph>();  // 创建一个IR图的共享指针
  parseIR(graph_string, &*graph);  // 解析IR字符串填充图形对象graph
  // IRParser不支持张量常量，因此我们插入一个aten::ones调用，然后进行常量传播
  ConstantPropagation(graph);  // 进行常量传播优化

  TensorExprKernel k(graph);  // 创建Tensor表达式内核对象，传入优化后的图形

  auto x = at::rand({16, 16}, TensorOptions(kCPU).dtype(at::kFloat));  // 生成一个16x16的随机张量x
  std::vector<at::Tensor> inputs = {x};  // 创建输入张量列表
  std::vector<IValue> stack = fmap<IValue>(inputs);  // 转换为IValue向量
  k.run(stack);  // 运行Tensor表达式内核，计算输出
  auto o = stack[0].toTensor();  // 获取输出张量o
  auto y = at::ones({16, 16}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建一个全1的张量y，作为参考
  auto ref = x * y;  // 计算参考结果ref，为x和y的元素乘积
  ASSERT_TRUE(at::allclose(o, ref));  // 断言输出o与参考ref相似
}

TEST_F(Kernel, ConstantTensorsNonContiguous) {
  // 定义包含IR图形字符串的常量，描述操作序列及其参数
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()  // 创建一个空类型常量
          %dtype : int = prim::Constant[value=6]()  // 创建一个整数常量值为6
          %c0 : int = prim::Constant[value=0]()  // 创建一个整数常量值为0
          %c256 : int = prim::Constant[value=256]()  // 创建一个整数常量值为256
          %c16 : int = prim::Constant[value=16]()  // 创建一个整数常量值为16
          %y_flat : Tensor = aten::arange(%c0, %c256, %dtype, %none, %none, %none)  // 使用aten::arange生成一个张量y_flat
          %sizes : int[] = prim::ListConstruct(%c16, %c16)  // 构造一个整数数组常量，包含两个16
          %y_t : Tensor = aten::view(%y_flat, %sizes)  // 使用aten::view调整y_flat的形状为sizes
          %y : Tensor = aten::t(%y_t)  // 使用aten::t对y_t进行转置得到y
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)  // 使用aten::mul计算x和y的元素乘积得到z
          return (%z))IR";  // 返回计算结果z
  auto graph = std::make_shared<Graph>();  // 创建一个IR图的共享指针
  parseIR(graph_string, &*graph);  // 解析IR字符串填充图形对象graph
  // IRParser不支持张量常量，因此我们生成几个aten调用来生成非连续的常量张量，然后进行常量传播
  ConstantPropagation(graph);  // 进行常量传播优化

  TensorExprKernel k(graph);  // 创建Tensor表达式内核对象，传入优化后的图形

  auto x = at::rand({16, 16}, TensorOptions(kCPU).dtype(at::kFloat));  // 生成一个16x16的随机张量x
  std::vector<at::Tensor> inputs = {x};  // 创建输入张量列表
  std::vector<IValue> stack = fmap<IValue>(inputs);  // 转换为IValue向量
  k.run(stack);  // 运行Tensor表达式内核，计算输出
  auto o = stack[0].toTensor();  // 获取输出张量o
  auto y = at::arange(0, 256, TensorOptions(kCPU).dtype(at::kFloat))  // 创建一个0到255的张量，数据类型为Float
               .view({16, 16})  // 调整形状为16x16
               .t();  // 对张量进行转置得到y
  auto ref = x * y;  // 计算参考结果ref，为x和y的元素乘积
  ASSERT_TRUE(at::allclose(o, ref));  // 断言输出o与参考ref相似
}
#ifdef TORCH_ENABLE_LLVM
  // 如果 Torch 的 LLVM 支持被启用，则执行以下代码段

  const auto graph_string = R"IR(
      graph(%0 : Float(5, 3, strides=[3, 1], device=cpu),
            %1 : Float(5, 3, strides=[1, 5], device=cpu)):
        %2 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %1)
        %3 : Float(5, 3, strides=[3, 1]) = aten::mul(%0, %2)
        return (%3))IR";
  // 定义一个表示计算图的字符串，描述了一个简单的张量计算图

  auto graph = std::make_shared<Graph>();
  // 创建一个指向 Graph 对象的共享指针

  parseIR(graph_string, &*graph);
  // 解析上述定义的 IR 字符串，将计算图加载到 graph 对象中

  auto a = at::rand({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  // 创建一个大小为 5x3 的随机浮点数张量 a

  auto b =
      at::rand({3, 5}, TensorOptions(kCPU).dtype(at::kFloat)).transpose(0, 1);
  // 创建一个大小为 3x5 的随机浮点数张量 b，并将其转置

  auto o = at::zeros({5, 3}, TensorOptions(kCPU).dtype(at::kFloat));
  // 创建一个全零的大小为 5x3 的浮点数张量 o，作为输出

  auto ref = a * (a * b);
  // 计算参考结果张量，等于 a * (a * b)

  TensorExprKernel k(graph);
  // 创建一个基于给定计算图的 TensorExprKernel 对象 k

  k.runFast({a.data_ptr(), b.data_ptr()}, {o.data_ptr()});
  // 使用快速运行模式执行 TensorExprKernel 对象 k，传入 a 和 b 的数据指针，并将结果存储到 o 中

  for (size_t i = 0; i < 5 * 3; i++) {
    // 遍历结果张量 o 的每个元素
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
    // 检查计算结果是否与预期结果 ref 相符
  }
#endif
}
#ifdef TORCH_ENABLE_LLVM
  // 定义一个包含 IR 的字符串，描述一个计算图，输入为 Float 类型的张量
  const auto graph_string = R"IR(
        graph(%x : Float(16, 16, strides=[16, 1], device=cpu)):
          %none : NoneType = prim::Constant()  // 定义一个常量 NoneType
          %dtype : int = prim::Constant[value=6]()  // 定义一个值为 6 的整数常量
          %c0 : int = prim::Constant[value=0]()  // 定义一个值为 0 的整数常量
          %c256 : int = prim::Constant[value=256]()  // 定义一个值为 256 的整数常量
          %c16 : int = prim::Constant[value=16]()  // 定义一个值为 16 的整数常量
          %y_flat : Tensor = aten::arange(%c0, %c256, %dtype, %none, %none, %none)  // 使用 arange 函数创建一个张量
          %sizes : int[] = prim::ListConstruct(%c16, %c16)  // 创建一个包含两个元素的整数列表
          %y_t : Tensor = aten::view(%y_flat, %sizes)  // 使用 view 函数对张量进行形状变换
          %y : Tensor = aten::t(%y_t)  // 对张量进行转置操作
          %z : Float(16, 16, strides=[16, 1], device=cpu) = aten::mul(%x, %y)  // 对输入张量与 y 进行逐元素乘法操作
          return (%z))IR";  // 返回操作结果张量 z
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针指向 Graph 对象
  parseIR(graph_string, &*graph);  // 解析 IR 字符串，生成计算图并存储在 graph 中
  // IRParser 不支持张量常量，因此通过生成多个 aten 调用来生成非连续的常量张量，然后进行常量传播
  ConstantPropagation(graph);  // 对图进行常量传播优化

  TensorExprKernel k(graph);  // 根据优化后的计算图创建 TensorExprKernel 对象

  // 检查是否能够获取生成的汇编代码
  auto asm_str = k.getCodeText("asm");  // 获取指定名称（"asm"）的汇编代码文本
  const std::string& asm_verification_pattern =
      R"ASM(
        # CHECK: .text
        # CHECK: retq)ASM";  // 汇编代码验证模式
  torch::jit::testing::FileCheck().run(asm_verification_pattern, asm_str);  // 使用 FileCheck 运行汇编代码验证模式

  // 检查是否能够获取代码生成参数的信息
  auto constants = k.getConstantDescriptors();  // 获取常量描述符
  auto buf_args = k.getBufferArgs();  // 获取缓冲区参数
  // 预期的缓冲区参数应为：[input0, output0, constant0]
  ASSERT_EQ(buf_args.size(), 3);  // 断言缓冲区参数的数量为 3
  ASSERT_EQ(constants.size(), 1);  // 断言常量描述符的数量为 1
  ASSERT_TRUE(
      !buf_args[0].isVar() && !buf_args[1].isVar() && !buf_args[2].isVar());  // 断言缓冲区参数不是变量
#endif
}

Tensor lowerNanToNum(
    const std::vector<ArgValue>& inputs,  // 输入参数列表
    const std::vector<ExprHandle>& outputShape,  // 输出张量的形状描述
    const std::vector<ExprHandle>& outputStrides,  // 输出张量的步幅描述
    const std::optional<ScalarType>& outputType,  // 输出张量的数据类型（可选）
    at::Device device) {  // 张量所在设备
  auto input_buf = std::get<BufHandle>(inputs[0]);  // 获取输入缓冲区句柄
  auto e = Compute(
      "custom_nan_to_num",  // 计算的名称
      outputShape,  // 输出张量的形状
      outputStrides,  // 输出张量的步幅
      [&](const std::vector<VarHandle>& axes) {  // 匿名函数，对各轴进行操作
        std::vector<ExprHandle> indices(axes.begin(), axes.end());  // 构造索引向量
        auto load = input_buf.load(indices);  // 根据索引加载输入张量数据
        return IfThenElse::make(Cast::make(kBool, isnan(load)), 0.0f, load);  // 如果加载的数据是 NaN，则返回 0.0，否则返回加载的数据
      });
  return e;  // 返回计算结果张量
}

TEST_F(Kernel, CustomLowering) {
  const auto graph_string = R"IR(
      graph(%x : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu)):
          %none : NoneType = prim::Constant()  // 定义一个常量 NoneType
          %y : Float(2, 2, strides=[2, 1], requires_grad=0, device=cpu) = aten::nan_to_num(%x, %none, %none, %none)  // 使用 nan_to_num 函数处理输入张量 x
          return (%y)  // 返回处理后的张量 y
      )IR";  // IR 字符串描述一个计算图
// 使用R"IR(...)IR"语法定义一个字符串，表示一个Torch的IR图，描述了一些张量操作
const auto graph_string = R"IR(
    graph(%0 : Float(100, 3, strides=[3, 1], device=cpu),
          %1 : Float(100, 3, strides=[3, 1], device=cpu)):
      %2 : Float(100, 3, strides=[3, 1]) = aten::mul(%0, %1)
      %3 : Float(100, 3, strides=[3, 1]) = aten::mul(%0, %2)
      return (%3))IR";
// 创建一个指向Graph对象的shared_ptr，用于存储解析后的IR图
auto graph = std::make_shared<Graph>();
// 解析graph_string中的IR代码，将结果存储到graph所指向的对象中
parseIR(graph_string, &*graph);

// 生成两个大小为[100, 3]的随机浮点数张量a和b，以及一个全零张量o，用作输出
auto a = at::rand({100, 3}, TensorOptions(kCPU).dtype(at::kFloat));
auto b = at::rand({100, 3}, TensorOptions(kCPU).dtype(at::kFloat));
auto o = at::zeros({100, 3}, TensorOptions(kCPU).dtype(at::kFloat));
// 计算参考结果，即a * (a * b)，存储在ref张量中
auto ref = a * (a * b);

// 使用解析后的IR图创建TensorExprKernel对象k
TensorExprKernel k(graph);
// 将输入张量a和b作为输入列表传递给张量表达式内核对象
std::vector<at::Tensor> inputs = {a, b};
// 获取生成的代码语句的指针
StmtPtr s = k.getCodeGenStmt();

// 创建一个ostringstream对象oss，用于构建生成代码语句的字符串表示
std::ostringstream oss;
oss << *s;

// 检查我们生成的IR代码是否符合指定的验证模式
const std::string& verification_pattern = R"IR(# CHECK: Ramp)IR";
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

// 将输入张量转换为IValue向量并传递给TensorExprKernel的run方法执行
std::vector<IValue> stack = fmap<IValue>(inputs);
k.run(stack);
// 将输出张量o更新为运行后的结果张量
o = stack[0].toTensor();

// 使用循环检查o和ref张量的每个元素是否相等
for (size_t i = 0; i < 100 * 3; i++) {
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
}
    # 使用 TORCH_CHECK_EQ 宏比较两个浮点数数组中的元素是否相等
    TORCH_CHECK_EQ(((float*)o.data_ptr())[i], ((float*)ref.data_ptr())[i]);
graph(%x : int, %y : int):
  // 定义一个整数变量 %z，值为 %x 与 %y 的乘积
  %z : int = aten::mul(%x, %y)
  // 定义一个整数变量 %r，值为 %z 与 %x 的乘积
  %r : int = aten::mul(%z, %x)
  // 返回 %r 和 %z 两个整数变量
  return (%r, %z))IR";
auto graph = std::make_shared<Graph>();
std::unordered_map<std::string, Value*> vmap;
// 解析上述的 IR 字符串，并将结果填充到 graph 中，同时将对应的变量映射到 vmap 中
parseIR(ir, graph.get(), vmap);
TensorExprKernel k(graph);

auto stmt = k.getCodeGenStmt();
std::ostringstream oss;
oss << *stmt;

// 验证生成的 IR。预期会看到一个标量变量（Let），后跟对零维缓冲区的存储。
const std::string& verification_pattern = R"IR(
# CHECK: int64_t
# CHECK-NEXT: [0ll] =
# CHECK-NEXT: int64_t
# CHECK-NEXT: [0ll] =
)IR";
torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

int64_t x = 2, y = 3, r = 0, z = 0;

// 使用标量输出验证 TEK::runFast 是否正常工作
std::vector<void*> inputs = {&x, &y};
std::vector<void*> outputs = {&r, &z};
k.runFast(inputs, outputs);
TORCH_CHECK_EQ(z, x * y);
TORCH_CHECK_EQ(r, z * x);

// 使用标量输出验证 TEK::run 是否正常工作
std::vector<IValue> stack = {x, y};
k.run(stack);
TORCH_CHECK_EQ(stack[0], x * y * x);
TORCH_CHECK_EQ(stack[1], x * y);
// 定义一个函数 graph，接受四个参数：两个整数 %x 和 %y，两个长整型张量 %xt 和 %yt
graph(%x : int,
      %xt : Long(3, strides=[1], device=cpu),
      %y : int,
      %yt : Long(3, strides=[1], device=cpu)):
  // 计算 %z = %x * %y
  %z : int = aten::mul(%x, %y)
  // 计算 %r = %z * %x
  %r : int = aten::mul(%z, %x)
  // 计算 %zt = %xt * %y
  %zt : Long(3, strides=[1], device=cpu) = aten::mul(%xt, %y)
  // 计算 %rt = %zt * %xt
  %rt : Long(3, strides=[1], device=cpu) = aten::mul(%zt, %xt)
  // 返回四个结果 %r, %rt, %z, %zt
  return (%r, %rt, %z, %zt))IR";
  
// 创建一个名为 graph 的共享指针 Graph 对象
auto graph = std::make_shared<Graph>();
// 创建一个空的无序映射 vmap，用于存储字符串到 Value* 的映射关系
std::unordered_map<std::string, Value*> vmap;
// 解析输入的 IR 字符串 ir，将解析结果存储到 graph 中，并更新 vmap
parseIR(ir, graph.get(), vmap);
// 基于 graph 创建一个 TensorExprKernel 对象 k
TensorExprKernel k(graph);

// 初始化整数变量 x = 2, y = 3, r = 0, z = 0
int64_t x = 2, y = 3, r = 0, z = 0;
// 创建一个包含三个元素，元素值为 2 的长整型张量 xt
auto xt = at::ones({3}, TensorOptions(kCPU).dtype(at::kLong)) * 2;
// 创建一个包含三个元素，元素值为 3 的长整型张量 yt
auto yt = at::ones({3}, TensorOptions(kCPU).dtype(at::kLong)) * 3;
// 创建一个包含三个元素，元素值为 0 的长整型张量 zt
auto zt = at::zeros({3}, TensorOptions(kCPU).dtype(at::kLong));
// 创建一个包含三个元素，元素值为 0 的长整型张量 rt
auto rt = at::zeros({3}, TensorOptions(kCPU).dtype(at::kLong));

// 准备输入向量 inputs 和输出向量 outputs，用于 TensorExprKernel 的快速运行
std::vector<void*> inputs = {&x, xt.data_ptr(), &y, yt.data_ptr()};
std::vector<void*> outputs = {&r, rt.data_ptr(), &z, zt.data_ptr()};
// 调用 TensorExprKernel 的快速运行方法，执行计算
k.runFast(inputs, outputs);

// 使用断言验证结果
TORCH_CHECK_EQ(z, x * y);
TORCH_CHECK_EQ(r, z * x);
// 使用 ASSERT_TRUE 断言验证两个张量相等性
ASSERT_TRUE(at::equal(zt, xt * yt));
ASSERT_TRUE(at::equal(rt, zt * xt));

// 准备堆栈 stack，用于 TensorExprKernel 的运行
std::vector<IValue> stack = {x, xt, y, yt};
// 调用 TensorExprKernel 的运行方法，执行计算
k.run(stack);

// 使用断言验证堆栈中的值
TORCH_CHECK_EQ(stack[0], x * y * x);
ASSERT_TRUE(at::equal(stack[1].toTensor(), xt * yt * xt));
TORCH_CHECK_EQ(stack[2], x * y);
ASSERT_TRUE(at::equal(stack[3].toTensor(), xt * yt));
}
#ifdef TORCH_ENABLE_LLVM
  // 保存当前的 getCatWoConditionals() 状态
  bool old_cat_wo_conditionals = getCatWoConditionals();
  // 设置 getCatWoConditionals() 为 true，以便在运行期间禁用条件语句
  getCatWoConditionals() = true;
  // 定义一个包含 IR 字符串的常量，描述一个 TorchScript 图
  const auto graph_string = R"IR(
      graph(%a : Float(SS(-2), 3, SS(-3), requires_grad=0, device=cpu),
            %b : Float(SS(-2), 7, SS(-3), requires_grad=0, device=cpu),
            %c : Float(SS(-2), 9, SS(-3), requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(SS(-2), 19, SS(-3), requires_grad=0, device=cpu) = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
  // 创建一个新的 TorchScript 图对象
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 解析 IR 字符串并填充到图对象中
  torch::jit::parseIR(graph_string, graph.get());

  // 定义一个符号形状输入向量
  std::vector<int64_t> symbolic_shape_inputs = {-2, -3};

  // 定义输入描述的符号步长
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 定义一个从值到符号步长输入的映射
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  // 设置图对象的输入和输出的符号步长描述
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->inputs().at(2)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建一个 TensorExprKernel 对象，用于表示优化的张量表达式内核
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 创建一个输出流对象，将内核的代码生成语句写入流中
  std::ostringstream oss;
  oss << *kernel.getCodeGenStmt();
  // 定义一个用于验证的字符串模式
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i
# CHECK-NEXT: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK-NOT: for (int64_t i
      )IR";
  // 使用 FileCheck 工具验证生成的代码是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 定义一个运行内核的 Lambda 函数，测试不同维度大小的情况
  auto run_kernel = [&](int dim1, int dim2) {
    // 创建三个随机张量 a, b, c，分别具有不同的维度
    auto a =
        at::rand({dim1, 3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto b =
        at::rand({dim1, 7, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto c =
        at::rand({dim1, 9, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));

    // 使用 PyTorch 的 at::cat 函数合并这三个张量在第1维上
    auto ref = at::cat({a, b, c}, 1);

    // 创建一个输入堆栈，包含张量和两个维度参数
    std::vector<IValue> stack =
        fmap<IValue>(std::vector<at::Tensor>({a, b, c}));
    stack.emplace_back(dim1);
    stack.emplace_back(dim2);
    // 运行优化内核
    kernel.run(stack);

    // 获取内核运行后的输出张量
    auto o = stack[0].toTensor();
    // 断言输出张量与预期的参考张量 ref 是否接近
    ASSERT_TRUE(at::allclose(o, ref));
  };

  // 在不同的维度大小下运行内核函数
  run_kernel(10, 20);
  // 恢复旧的 getCatWoConditionals() 状态
  getCatWoConditionals() = old_cat_wo_conditionals;
#endif
}
#ifdef TORCH_ENABLE_LLVM
  // 保存当前的 getCatWoConditionals() 值
  bool old_cat_wo_conditionals = getCatWoConditionals();
  // 设置 getCatWoConditionals() 为 true
  getCatWoConditionals() = true;
  // 定义一个包含图形信息的字符串
  const auto graph_string = R"IR(
      graph(%a : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %b : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %c : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int,
            %SS_4 : int,
            %SS_5 : int):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b, %c)
        %r : Float(SS(-2), SS(-5), SS(-3), requires_grad=0, device=cpu) = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
  // 创建一个共享指针的图对象
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 解析图形字符串到图对象中
  torch::jit::parseIR(graph_string, graph.get());

  // 定义符号形状输入的数组
  std::vector<int64_t> symbolic_shape_inputs = {-2, -3, -4, -5};

  // 定义输入描述的向量
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 创建符号步幅的无序映射
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  // 将输入值与输入描述关联起来
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->inputs().at(2)] = input_desc;
  // 将输出值与输入描述关联起来
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建一个 TensorExprKernel 对象
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 创建一个字符串流对象
  std::ostringstream oss;
  // 将内核代码生成语句写入字符串流
  oss << *kernel.getCodeGenStmt();
  // 验证生成的代码与预期的模式是否匹配
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i
# CHECK-NEXT: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK-NOT: for (int64_t i
      )IR";
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 定义一个运行内核的 lambda 函数
  auto run_kernel = [&](int dim1, int dim2, int dim3) {
    // 创建随机张量 a, b, c
    auto a =
        at::rand({dim1, dim3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto b =
        at::rand({dim1, dim3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto c =
        at::rand({dim1, dim3, dim2}, at::TensorOptions(kCPU).dtype(at::kFloat));

    // 计算预期的合并结果 ref
    auto ref = at::cat({a, b, c}, 1);

    // 创建一个输入值向量
    std::vector<IValue> stack =
        fmap<IValue>(std::vector<at::Tensor>({a, b, c}));
    stack.emplace_back(dim1);
    stack.emplace_back(dim2);
    stack.emplace_back(dim3);
    stack.emplace_back(3 * dim3);
    // 运行内核
    kernel.run(stack);

    // 获取内核运行后的输出张量 o
    auto o = stack[0].toTensor();
    // 断言内核运行结果与预期的合并结果一致
    ASSERT_TRUE(at::allclose(o, ref));
  };

  // 运行内核函数的测试案例
  run_kernel(10, 20, 15);
  // 恢复 getCatWoConditionals() 到之前的值
  getCatWoConditionals() = old_cat_wo_conditionals;
#endif
}
#ifdef TORCH_ENABLE_LLVM
  // 保存旧的条件下不使用条件语句的设置，并设置为强制启用条件语句
  bool old_cat_wo_conditionals = getCatWoConditionals();
  getCatWoConditionals() = true;
  // 定义一个字符串，包含表示图形的内部表示(IR)
  const auto graph_string = R"IR(
      graph(%a : Float(SS(-2), SS(-4), SS(-3), requires_grad=0, device=cpu),
            %b : Float(SS(-2), SS(-5), SS(-3), requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int,
            %SS_4 : int,
            %SS_5 : int,
            %SS_6 : int):
        %dim : int = prim::Constant[value=1]()
        %inputs : Tensor[] = prim::ListConstruct(%a, %b)
        %r : Float(SS(-2), SS(-6), SS(-3), requires_grad=0, device=cpu) = aten::cat(%inputs, %dim)               # new size: [5,19,2]
        return (%r))IR";
  // 创建共享指针指向图形对象，并解析上面定义的图形字符串
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());

  // 定义符号形状输入的数组
  std::vector<int64_t> symbolic_shape_inputs = {-2, -3, -4, -5, -6};

  // 定义输入描述的向量，指定张量的连续性
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 创建映射，将符号步长映射到图形的输入和输出
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建张量表达式的核心对象，使用定义的图形、空的其他参数、符号形状输入和步长映射
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 创建字符串流，将张量表达式核心的代码生成语句写入流中
  std::ostringstream oss;
  oss << *kernel.getCodeGenStmt();
  // 定义用于验证生成代码的模式字符串
  const std::string& verification_pattern =
      R"IR(
# CHECK: for (int64_t i
# CHECK-NEXT: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK: for (int64_t j
# CHECK-NEXT: for (int64_t k
# CHECK-NOT: for (int64_t j
# CHECK-NOT: for (int64_t i
      )IR";
  // 运行文件检查工具，验证生成的代码是否符合预期模式
  torch::jit::testing::FileCheck().run(verification_pattern, oss.str());

  // 定义一个匿名函数，运行张量表达式核心，并进行结果验证
  auto run_kernel = [&](int dim2, int dim3, int dim4, int dim5) {
    auto a =
        at::rand({dim2, dim4, dim3}, at::TensorOptions(kCPU).dtype(at::kFloat));
    auto b =
        at::rand({dim2, dim5, dim3}, at::TensorOptions(kCPU).dtype(at::kFloat));

    auto ref = at::cat({a, b}, 1);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.emplace_back(dim2);
    stack.emplace_back(dim3);
    stack.emplace_back(dim4);
    stack.emplace_back(dim5);
    stack.emplace_back(dim4 + dim5);
    // 运行张量表达式核心
    kernel.run(stack);

    // 从栈中获取输出张量，并与预期结果进行比较
    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  };

  // 使用指定的维度参数运行匿名函数
  run_kernel(10, 20, 15, 8);
  // 恢复旧的条件下不使用条件语句的设置
  getCatWoConditionals() = old_cat_wo_conditionals;
#endif
}
```