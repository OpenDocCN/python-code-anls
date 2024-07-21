# `.\pytorch\test\cpp\tensorexpr\test_quantization.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <ATen/native/quantized/PackedParams.h>  // 包含 ATen 库的量化参数处理相关头文件
#include <test/cpp/tensorexpr/test_base.h>  // 包含测试基础库的头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的 IR 相关头文件
#include <torch/csrc/jit/ir/irparser.h>  // 包含 Torch 的 IR 解析器头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>  // 包含 Torch 的 Tensor 表达式核心头文件
#include <torch/csrc/jit/tensorexpr/loopnest.h>  // 包含 Torch 的循环嵌套表达式头文件
#include <torch/csrc/jit/tensorexpr/tensor.h>  // 包含 Torch 的张量表达式头文件
#include <torch/csrc/jit/testing/file_check.h>  // 包含 Torch 的文件检查测试头文件
#include <torch/torch.h>  // 包含 Torch 主头文件
#include <cmath>  // 包含数学函数库的头文件
#include <sstream>  // 包含字符串流处理的头文件
#include "torch/csrc/jit/tensorexpr/eval.h"  // 包含 Torch 的表达式求值头文件
#include "torch/csrc/jit/tensorexpr/ir.h"  // 包含 Torch 的 IR 表达式头文件

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;  // 使用 Torch 的 Tensor 表达式命名空间
using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;  // 定义简单 IR 表达式求值类型
using namespace torch::indexing;  // 使用 Torch 的索引命名空间
using namespace torch::jit::tensorexpr;  // 再次使用 Torch 的 Tensor 表达式命名空间

class Quantization : public ::testing::Test {
 public:
  void SetUp() override {
    getTEMustUseLLVMOnCPU() = false;  // 设置测试环境不强制使用 LLVM 在 CPU 上运行
  }
};

TEST_F(Quantization, QuantDequantInt8) {
  // 定义 IR 图的字符串表示，用于量化和反量化操作
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=12]()
        %3 : int = prim::Constant[value=13]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : QInt8(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        %6 : Float(2, 2) = aten::dequantize(%q.1)
        return (%6))IR";
  
  auto graph = std::make_shared<Graph>();  // 创建共享的图对象
  parseIR(graph_string, &*graph);  // 解析 IR 字符串并填充图对象

  auto x = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建随机浮点张量 x
  auto q = at::quantize_per_tensor(x, 0.1f, 13, at::kQInt8);  // 对张量 x 进行量化
  auto y_expected = at::dequantize(q);  // 对量化后的张量 q 进行反量化
  TensorExprKernel k(graph);  // 创建 Tensor 表达式核心对象 k
  std::vector<at::Tensor> inputs = {x};  // 构建输入张量列表

  StmtPtr s = k.getCodeGenStmt();  // 获取代码生成语句

  std::vector<IValue> stack = fmap<IValue>(inputs);  // 将输入张量映射为 IValue 列表
  k.run(stack);  // 运行 Tensor 表达式核心对象 k
  auto y = stack[0].toTensor();  // 获取运行后的张量结果
  bool check = at::allclose(y_expected, y);  // 检查预期结果和实际结果是否接近
  if (!check) {
    std::cout << "y_expected:\n" << y_expected << std::endl;  // 输出预期结果
    std::cout << "y:\n" << y << std::endl;  // 输出实际结果
  }
  TORCH_CHECK_EQ(check, 1);  // 使用 Torch 的检查机制确保结果一致
}

TEST_F(Quantization, QuantDequantUInt8) {
  // 定义 IR 图的字符串表示，用于量化和反量化操作
  const auto graph_string = R"IR(
      graph(%x.1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %3 : int = prim::Constant[value=122]()
        %4 : float = prim::Constant[value=0.1]()
        %q.1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)
        %6 : Float(2, 2) = aten::dequantize(%q.1)
        return (%6))IR";

  auto graph = std::make_shared<Graph>();  // 创建共享的图对象
  parseIR(graph_string, &*graph);  // 解析 IR 字符串并填充图对象

  auto x = 2 * at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));  // 创建随机浮点张量 x 的倍数
  auto q = at::quantize_per_tensor(x, 0.1f, 122, at::kQUInt8);  // 对张量 x 进行量化
  auto y_expected = at::dequantize(q);  // 对量化后的张量 q 进行反量化
  TensorExprKernel k(graph);  // 创建 Tensor 表达式核心对象 k
  std::vector<at::Tensor> inputs = {x};  // 构建输入张量列表

  StmtPtr s = k.getCodeGenStmt();  // 获取代码生成语句

  std::vector<IValue> stack = fmap<IValue>(inputs);  // 将输入张量映射为 IValue 列表
  k.run(stack);  // 运行 Tensor 表达式核心对象 k
  auto y = stack[0].toTensor();  // 获取运行后的张量结果
  bool check = at::allclose(y_expected, y);  // 检查预期结果和实际结果是否接近
  if (!check) {
    std::cout << "y_expected:\n" << y_expected << std::endl;  // 输出预期结果
  }
  // 注意：此处省略了实际结果输出部分，仅演示输出预期结果的情况

  TORCH_CHECK_EQ(check, 1);  // 使用 Torch 的检查机制确保结果一致
}
    // 输出 y 的值到标准输出流，并换行
    std::cout << "y:\n" << y << std::endl;
  }
  // 使用 Torch 的检查宏，验证 check 的值是否等于 1
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantDequantUInt8_NLC) {
  // 定义一个包含 IR 的字符串，描述一个计算图，接受大小为 (1, 2, 2) 的浮点张量作为输入
  const auto graph_string = R"IR(
      graph(%x.1 : Float(1, 2, 2, strides=[4, 1, 2], device=cpu)):
        %2 : int = prim::Constant[value=13]()  // 创建一个整数常量，值为 13
        %3 : int = prim::Constant[value=122]()  // 创建一个整数常量，值为 122
        %4 : float = prim::Constant[value=0.1]()  // 创建一个浮点数常量，值为 0.1
        %q.1 : QUInt8(1, 2, 2) = aten::quantize_per_tensor(%x.1, %4, %3, %2)  // 对输入张量进行量化为 QUInt8 类型
        %6 : Float(1, 2, 2) = aten::dequantize(%q.1)  // 对量化后的张量进行反量化
        return (%6))IR";  // 返回反量化后的张量作为结果

  // 创建一个共享指针指向 Graph 对象，并解析 graph_string 描述的计算图
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 创建一个大小为 (1, 2, 2) 的随机浮点张量 x，并设置其大小和步幅
  auto x = 2 * at::rand({1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  x.unsafeGetTensorImpl()->set_sizes_and_strides(
      std::initializer_list<int64_t>{1, 2, 2}, {4, 1, 2});

  // 对输入张量 x 进行量化，得到 q，量化参数为 0.1，零点值为 122，量化类型为 QUInt8
  auto q = at::quantize_per_tensor(x, 0.1f, 122, at::kQUInt8);

  // 对量化后的张量 q 进行反量化，得到期望的输出 y_expected
  auto y_expected = at::dequantize(q);

  // 创建一个 TensorExprKernel 对象 k，传入解析后的计算图
  TensorExprKernel k(graph);

  // 创建输入张量的 vector，用于运行 Kernel
  std::vector<at::Tensor> inputs = {x};

  // 获取代码生成语句的指针 s
  StmtPtr s = k.getCodeGenStmt();

  // 创建一个 IValue 类型的 stack，包含输入张量 x 的值
  std::vector<IValue> stack = fmap<IValue>(inputs);

  // 运行 Kernel，将结果存储在 stack 中
  k.run(stack);

  // 从 stack 中获取运行后的输出张量 y
  auto y = stack[0].toTensor();

  // 检查 y_expected 和 y 是否接近
  bool check = at::allclose(y_expected, y);

  // 如果检查不通过，则输出 x、y_expected 和 y 的值
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }

  // 使用 TORCH_CHECK_EQ 确保检查通过
  TORCH_CHECK_EQ(check, 1);
}

// 定义一个函数 quantized_add，接受两个张量 x1 和 x2，以及 scale 和 zero 作为参数
at::Tensor quantized_add(
    at::Tensor x1,
    at::Tensor x2,
    double scale,
    int64_t zero) {
  // 查找并调用 quantized::add 操作的分发器，使用 x1、x2、scale 和 zero 作为参数
  const auto qadd_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("quantized::add", "")
          .typed<at::Tensor(at::Tensor, at::Tensor, double, int64_t)>();
  return qadd_op.call(x1, x2, scale, zero);  // 返回 quantized::add 操作的结果张量
}
// 定义测试函数 QuantAddDequantInt8，使用 Google Test 的 TEST_F 宏
TEST_F(Quantization, QuantAddDequantInt8) {
  // 定义包含 IR 表示的计算图的字符串
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=12]()   // 定义整数常量 12
        %qz1 : int = prim::Constant[value=13]()  // 定义整数常量 13，用于量化参数
        %qs1 : float = prim::Constant[value=0.1]()  // 定义浮点常量 0.1，用于量化参数
        %qz2 : int = prim::Constant[value=13]()  // 定义整数常量 13，用于量化参数
        %qs2 : float = prim::Constant[value=0.1]()  // 定义浮点常量 0.1，用于量化参数
        %qza : int = prim::Constant[value=13]()  // 定义整数常量 13，用于量化参数
        %qsa : float = prim::Constant[value=0.1]()  // 定义浮点常量 0.1，用于量化参数
        %q1 : QInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)  // 对输入张量 %x1 进行量化
        %q2 : QInt8(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)  // 对输入张量 %x2 进行量化
        %qa : QInt8(2, 2) = quantized::add(%q1, %q2, %qsa, %qza)  // 使用量化后的张量进行加法运算
        %6 : Float(2, 2) = aten::dequantize(%qa)  // 对量化后的结果 %qa 进行反量化
        return (%6))IR";  // 返回反量化结果

  // 创建共享指针指向图对象，并解析 IR 表示的计算图
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 生成随机输入张量 x1 和 x2
  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));

  // 对输入张量 x1 和 x2 进行量化
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.1f, 13, at::kQInt8);

  // 使用量化后的张量进行加法运算
  auto qa = quantized_add(q1, q2, 0.1f, 13);

  // 计算预期的反量化结果
  auto y_expected = at::dequantize(qa);

  // 创建 TensorExprKernel 对象，并获取代码生成的语句
  TensorExprKernel k(graph);
  std::vector<at::Tensor> inputs = {x1, x2};
  StmtPtr s = k.getCodeGenStmt();

  // 将输入张量转换为 IValue 格式并运行计算图
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);

  // 获取运行后的输出张量 y
  auto y = stack[0].toTensor();

  // 检查计算得到的输出张量 y 与预期结果 y_expected 是否近似相等
  bool check = at::allclose(y_expected, y);
  if (!check) {
    // 如果结果不匹配，打印调试信息
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }

  // 使用 TORCH_CHECK_EQ 断言检查结果是否符合预期
  TORCH_CHECK_EQ(check, 1);
}
TEST_F(Quantization, QuantAddDequantUInt8) {
  // 定义包含计算图的字符串表示
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()   // 常量节点，值为整数 13
        %qz1 : int = prim::Constant[value=13]() // 常量节点，值为整数 13
        %qs1 : float = prim::Constant[value=0.1]() // 常量节点，值为浮点数 0.1
        %qz2 : int = prim::Constant[value=13]() // 常量节点，值为整数 13
        %qs2 : float = prim::Constant[value=0.1]() // 常量节点，值为浮点数 0.1
        %qza : int = prim::Constant[value=13]() // 常量节点，值为整数 13
        %qsa : float = prim::Constant[value=0.1]() // 常量节点，值为浮点数 0.1
        %q1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2) // 将输入张量 %x1 量化为 QUInt8 类型
        %q2 : QUInt8(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2) // 将输入张量 %x2 量化为 QUInt8 类型
        %qa : QUInt8(2, 2) = quantized::add(%q1, %q2, %qsa, %qza) // 对量化后的张量 %q1 和 %q2 执行加法
        %6 : Float(2, 2) = aten::dequantize(%qa) // 将结果张量 %qa 反量化为浮点张量
        return (%6))IR"; // 返回反量化后的浮点张量

  // 创建共享指针，解析计算图字符串并存储在 graph 中
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 生成随机浮点张量 x1 和 x2
  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));

  // 将 x1 和 x2 分别量化为 QUInt8 类型的张量 q1 和 q2
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQUInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.1f, 13, at::kQUInt8);

  // 对量化后的张量进行加法操作
  auto qa = quantized_add(q1, q2, 0.1f, 13);

  // 计算预期的反量化结果
  auto y_expected = at::dequantize(qa);

  // 创建 TensorExprKernel 对象，封装计算图
  TensorExprKernel k(graph);

  // 准备输入张量的向量
  std::vector<at::Tensor> inputs = {x1, x2};

  // 获取代码生成的语句
  StmtPtr s = k.getCodeGenStmt();

  // 将输入张量转换为 IValue 栈
  std::vector<IValue> stack = fmap<IValue>(inputs);

  // 运行计算图
  k.run(stack);

  // 从栈中获取输出张量 y
  auto y = stack[0].toTensor();

  // 检查预期输出和实际输出是否相似
  bool check = at::allclose(y_expected, y);

  // 如果检查失败，打印调试信息
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }

  // 使用 Torch 的断言检查最终结果
  TORCH_CHECK_EQ(check, 1);
}

TEST_F(Quantization, QuantSigmoidDequantUInt8) {
  // 定义包含计算图的字符串表示
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()   // 常量节点，值为整数 13
        %qz1 : int = prim::Constant[value=13]() // 常量节点，值为整数 13
        %qs1 : float = prim::Constant[value=0.1]() // 常量节点，值为浮点数 0.1
        %q1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2) // 将输入张量 %x1 量化为 QUInt8 类型
        %qa : QUInt8(2, 2) = aten::sigmoid(%q1) // 对量化后的张量 %q1 执行 sigmoid 操作
        %6 : Float(2, 2) = aten::dequantize(%qa) // 将结果张量 %qa 反量化为浮点张量
        return (%6))IR"; // 返回反量化后的浮点张量

  // 创建共享指针，解析计算图字符串并存储在 graph 中
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 生成随机浮点张量 x1
  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));

  // 将 x1 量化为 QUInt8 类型的张量 q1
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQUInt8);

  // 对量化后的张量执行 sigmoid 操作
  auto qs = at::sigmoid(q1);

  // 计算预期的反量化结果
  auto y_expected = at::dequantize(qs);

  // 创建 TensorExprKernel 对象，封装计算图
  TensorExprKernel k(graph);

  // 准备输入张量的向量
  std::vector<at::Tensor> inputs = {x1};

  // 获取代码生成的语句
  StmtPtr s = k.getCodeGenStmt();

  // 将输入张量转换为 IValue 栈
  std::vector<IValue> stack = fmap<IValue>(inputs);

  // 运行计算图
  k.run(stack);

  // 从栈中获取输出张量 y
  auto y = stack[0].toTensor();

  // 检查预期输出和实际输出是否相似
  bool check = at::allclose(y_expected, y);

  // 如果检查失败，打印调试信息
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
  }
}
    // 输出变量 qs 的值到标准输出流，末尾换行
    std::cout << "qs:\n" << qs << std::endl;
    // 输出变量 y_expected 的值到标准输出流，末尾换行
    std::cout << "y_expected:\n" << y_expected << std::endl;
    // 输出变量 y 的值到标准输出流，末尾换行
    std::cout << "y:\n" << y << std::endl;
  }
  // 使用 TORCH_CHECK_EQ 宏检查 check 的值是否等于 1
  TORCH_CHECK_EQ(check, 1);
TEST_F(Quantization, QuantMulDequantUInt8) {
  // 定义内嵌的 IR 表示图，描述了一个计算图
  const auto graph_string = R"IR(
      graph(%x1 : Float(2, 2, strides=[2, 1], device=cpu), %x2 : Float(2, 2, strides=[2, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()  // 创建一个整数常量 13
        %qz1 : int = prim::Constant[value=13]()  // 创建一个整数常量 13
        %qs1 : float = prim::Constant[value=0.1]()  // 创建一个浮点数常量 0.1
        %qz2 : int = prim::Constant[value=13]()  // 创建一个整数常量 13
        %qs2 : float = prim::Constant[value=0.1]()  // 创建一个浮点数常量 0.1
        %qza : int = prim::Constant[value=13]()  // 创建一个整数常量 13
        %qsa : float = prim::Constant[value=0.1]()  // 创建一个浮点数常量 0.1
        // 使用 quantize_per_tensor 操作对输入张量进行量化
        %q1 : QUInt8(2, 2) = aten::quantize_per_tensor(%x1, %qs1, %qz1, %2)
        %q2 : QUInt8(2, 2) = aten::quantize_per_tensor(%x2, %qs2, %qz2, %2)
        // 使用 quantized::mul 操作进行量化张量的乘法运算
        %qa : QUInt8(2, 2) = quantized::mul(%q1, %q2, %qsa, %qza)
        // 使用 dequantize 操作对量化结果进行反量化
        %6 : Float(2, 2) = aten::dequantize(%qa)
        return (%6))IR";
  // 创建共享指针 graph，用于存储和操作图
  auto graph = std::make_shared<Graph>();
  // 解析 IR 字符串，并将解析结果存储到 graph 中
  parseIR(graph_string, &*graph);

  // 生成随机张量 x1 和 x2，用于测试
  auto x1 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  auto x2 = at::rand({2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  // 对 x1 和 x2 进行量化操作，得到量化后的张量 q1 和 q2
  auto q1 = at::quantize_per_tensor(x1, 0.1f, 13, at::kQUInt8);
  auto q2 = at::quantize_per_tensor(x2, 0.1f, 13, at::kQUInt8);
  // 使用 quantized_mul 函数对量化张量 q1 和 q2 进行乘法运算，得到量化结果 qa
  auto qa = quantized_mul(q1, q2, 0.1f, 13);
  // 计算预期的反量化结果 y_expected
  auto y_expected = at::dequantize(qa);

  // 创建 TensorExprKernel 对象 k，基于 graph 生成代码
  TensorExprKernel k(graph);
  // 构建输入张量列表
  std::vector<at::Tensor> inputs = {x1, x2};
  // 获取代码生成后的语句
  StmtPtr s = k.getCodeGenStmt();

  // 构建输入张量的 IValue 列表
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 运行生成的代码
  k.run(stack);
  // 从运行结果中获取输出张量 y
  auto y = stack[0].toTensor();
  // 检查生成的 y 是否与预期的 y_expected 接近
  bool check = at::allclose(y_expected, y);
  // 如果检查不通过，则输出相关张量的值
  if (!check) {
    std::cout << "x1:\n" << x1 << std::endl;
    std::cout << "q1:\n" << q1 << std::endl;
    std::cout << "x2:\n" << x2 << std::endl;
    std::cout << "q2:\n" << q2 << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  // 使用 TORCH_CHECK_EQ 确保检查通过
  TORCH_CHECK_EQ(check, 1);
}
// 定义测试函数 QuantUpsampleNearst2dDequantUInt8，用于测试量化、上采样和反量化操作
TEST_F(Quantization, QuantUpsampleNearst2dDequantUInt8) {
  // 定义 IR 字符串，描述一个计算图
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 4, 4, strides=[16, 16, 4, 1], device=cpu)):
        %2 : int = prim::Constant[value=13]()
        %4 : NoneType = prim::Constant()
        %3 : int[] = prim::Constant[value=[6, 6]]()
        %qz : int = prim::Constant[value=13]()
        %qs : float = prim::Constant[value=0.1]()
        %q : QUInt8(1, 1, 4, 4) = aten::quantize_per_tensor(%x, %qs, %qz, %2)
        %qu : QUInt8(1, 1, 6, 6) = aten::upsample_nearest2d(%q, %3, %4)
        %6 : Float(1, 1, 6, 6) = aten::dequantize(%qu)
        return (%6))IR";
  
  // 创建一个计算图对象，并解析上面定义的 IR 字符串
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 生成一个随机的输入张量 x，尺寸为 [1, 1, 4, 4]，CPU 上的 float 类型
  auto x = at::rand({1, 1, 4, 4}, TensorOptions(kCPU).dtype(at::kFloat));
  
  // 对输入张量 x 进行量化操作，得到量化后的张量 q
  auto q = at::quantize_per_tensor(x, 0.1f, 13, at::kQUInt8);
  
  // 对量化后的张量 q 进行最近邻插值的上采样操作，得到上采样后的张量 qu
  auto qu = at::upsample_nearest2d(q, {6, 6});
  
  // 对上采样后的张量 qu 进行反量化操作，得到浮点型的输出张量 y_expected
  auto y_expected = at::dequantize(qu);

  // 创建一个 TensorExprKernel 对象，并传入计算图
  TensorExprKernel k(graph);
  
  // 准备输入张量的列表，并获取生成的代码块语句
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  // 将输入张量列表转换为 IValue 格式的栈，执行计算图的运行
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  
  // 从运行后的栈中获取输出张量 y
  auto y = stack[0].toTensor();
  
  // 检查输出张量 y 是否与预期的 y_expected 接近
  bool check = at::allclose(y_expected, y);
  
  // 如果检查不通过，则输出相关张量内容
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "q:\n" << q << std::endl;
    std::cout << "qu:\n" << qu << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  
  // 使用 TORCH_CHECK 确认检查结果为真
  TORCH_CHECK_EQ(check, 1);
}

// 定义测试函数 UpsampleNearst2d，用于测试最近邻插值上采样操作
TEST_F(Quantization, UpsampleNearst2d) {
  // 定义 IR 字符串，描述一个计算图
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu)):
        %4 : NoneType = prim::Constant()
        %3 : int[] = prim::Constant[value=[4, 4]]()
        %u : Float(1, 1, 4, 4) = aten::upsample_nearest2d(%x, %3, %4)
        return (%u))IR";
  
  // 创建一个计算图对象，并解析上面定义的 IR 字符串
  auto graph = std::make_shared<Graph>();
  parseIR(graph_string, &*graph);

  // 生成一个随机的输入张量 x，尺寸为 [1, 1, 2, 2]，CPU 上的 float 类型
  auto x = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));
  
  // 对输入张量 x 进行最近邻插值的上采样操作，得到上采样后的张量 y_expected
  auto y_expected = at::upsample_nearest2d(x, {4, 4});

  // 创建一个 TensorExprKernel 对象，并传入计算图
  TensorExprKernel k(graph);
  
  // 准备输入张量的列表，并获取生成的代码块语句
  std::vector<at::Tensor> inputs = {x};
  StmtPtr s = k.getCodeGenStmt();

  // 将输入张量列表转换为 IValue 格式的栈，执行计算图的运行
  std::vector<IValue> stack = fmap<IValue>(inputs);
  k.run(stack);
  
  // 从运行后的栈中获取输出张量 y
  auto y = stack[0].toTensor();
  
  // 检查输出张量 y 是否与预期的 y_expected 接近
  bool check = at::allclose(y_expected, y);
  
  // 如果检查不通过，则输出相关张量内容
  if (!check) {
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y_expected:\n" << y_expected << std::endl;
    std::cout << "y:\n" << y << std::endl;
  }
  
  // 使用 TORCH_CHECK 确认检查结果为真
  TORCH_CHECK_EQ(check, 1);
}
    # 定义一个接受多个张量输入并执行拼接操作的函数调用
    const auto op = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("quantized::cat", "")
                      .typed<at::Tensor(
                          c10::List<at::Tensor> const&,  # 输入参数：张量列表的常量引用
                          int64_t,                       # 输入参数：拼接的维度
                          std::optional<double>,         # 输入参数：可选的缩放因子
                          std::optional<int64_t>)>();    # 输入参数：可选的零点值

    # 通过重新分发操作调用定义好的函数，指定分发键集为QuantizedCPU，传递给定的参数
    return op.redispatch(
        DispatchKeySet({DispatchKey::QuantizedCPU}),  # 指定分发键集为QuantizedCPU
        xs,                                           # 张量列表
        dim,                                          # 拼接的维度
        scale,                                        # 可选的缩放因子
        zero);                                        # 可选的零点值
}

TEST_F(Quantization, QuantCatDequantUInt8) {
  // 定义一个字符串，表示一个简单的计算图
  const auto graph_string = R"IR(
      graph(%x : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu), %y : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu), %z : Float(1, 1, 2, 2, strides=[2, 2, 2, 1], device=cpu)):
        %qdt : int = prim::Constant[value=13]()  // 创建一个常量节点 qdt，值为 13
        %qxz : int = prim::Constant[value=13]()  // 创建一个常量节点 qxz，值为 13
        %qxs : float = prim::Constant[value=0.1]()  // 创建一个常量节点 qxs，值为 0.1
        %qyz : int = prim::Constant[value=16]()  // 创建一个常量节点 qyz，值为 16
        %qys : float = prim::Constant[value=0.15]()  // 创建一个常量节点 qys，值为 0.15
        %qzz : int = prim::Constant[value=19]()  // 创建一个常量节点 qzz，值为 19
        %qzs : float = prim::Constant[value=0.2]()  // 创建一个常量节点 qzs，值为 0.2
        %qx : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%x, %qxs, %qxz, %qdt)  // 对输入张量 x 进行量化
        %qy : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%y, %qys, %qyz, %qdt)  // 对输入张量 y 进行量化
        %qz : QUInt8(1, 1, 2, 2) = aten::quantize_per_tensor(%z, %qzs, %qzz, %qdt)  // 对输入张量 z 进行量化
        %catx : Tensor[] = prim::ListConstruct(%qx, %qy, %qz)  // 构建一个包含量化后张量 qx, qy, qz 的列表
        %catd : int = prim::Constant[value=0]()  // 创建一个常量节点 catd，值为 0
        %qcat : QUInt8(3, 1, 2, 2) = quantized::cat(%catx, %catd, %qxs, %qxz)  // 对量化后的张量列表进行合并
        %cat : Float(3, 1, 2, 2) = aten::dequantize(%qcat)  // 对合并后的量化张量进行反量化
        return (%cat))IR";  // 返回反量化结果
  auto graph = std::make_shared<Graph>();  // 创建一个共享指针，指向计算图对象
  parseIR(graph_string, &*graph);  // 解析上述字符串表示的计算图，并将其存储到 graph 对象中

  auto x = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));  // 生成一个随机浮点张量 x
  auto y = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));  // 生成一个随机浮点张量 y
  auto z = at::rand({1, 1, 2, 2}, TensorOptions(kCPU).dtype(at::kFloat));  // 生成一个随机浮点张量 z
  auto qx = at::quantize_per_tensor(x, 0.1f, 13, at::kQUInt8);  // 对张量 x 进行按张量量化
  auto qy = at::quantize_per_tensor(y, 0.15f, 16, at::kQUInt8);  // 对张量 y 进行按张量量化
  auto qz = at::quantize_per_tensor(z, 0.2f, 19, at::kQUInt8);  // 对张量 z 进行按张量量化
  auto qcat = quantized_cat({qx, qy, qz}, 0, 0.1f, 13);  // 对量化后的张量列表进行合并
  auto expected = at::dequantize(qcat);  // 对合并后的量化张量进行反量化

  TensorExprKernel k(graph);  // 创建一个张量表达式内核对象，使用之前构建的计算图
  std::vector<at::Tensor> inputs = {x, y, z};  // 将 x, y, z 张量放入输入张量列表
  StmtPtr s = k.getCodeGenStmt();  // 获取生成的代码语句

  std::vector<IValue> stack = fmap<IValue>(inputs);  // 将输入张量列表转换为 IValue 列表
  k.run(stack);  // 运行张量表达式内核，对输入进行计算
  auto result = stack[0].toTensor();  // 获取计算结果张量
  bool check = at::allclose(expected, result);  // 检查计算结果和预期结果是否接近
  if (!check) {
    // 如果计算结果与预期结果不接近，则输出相关信息
    std::cout << "x:\n" << x << std::endl;
    std::cout << "y:\n" << y << std::endl;
    std::cout << "z:\n" << z << std::endl;
    std::cout << "qx:\n" << qx << std::endl;
    std::cout << "qy:\n" << qy << std::endl;
    std::cout << "qz:\n" << qz << std::endl;
    std::cout << "qcat:\n" << qcat << std::endl;
    std::cout << "expected:\n" << expected << std::endl;
    std::cout << "result:\n" << result << std::endl;
  }
  TORCH_CHECK_EQ(check, 1);  // 使用 Torch 的断言检查结果是否正确
}

} // namespace jit
} // namespace torch
```