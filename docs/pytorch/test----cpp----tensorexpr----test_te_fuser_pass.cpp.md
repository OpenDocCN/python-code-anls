# `.\pytorch\test\cpp\tensorexpr\test_te_fuser_pass.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <test/cpp/tensorexpr/test_base.h>  // 包含测试基础库的头文件
#include <torch/csrc/jit/codegen/fuser/interface.h>  // 包含 Torch 的代码生成模块接口头文件
#include <torch/csrc/jit/ir/ir.h>  // 包含 Torch 的中间表示(IR)相关头文件
#include <torch/csrc/jit/ir/irparser.h>  // 包含 Torch 的 IR 解析器头文件
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>  // 包含 Torch 的张量表达式融合头文件
#include <torch/csrc/jit/runtime/interpreter.h>  // 包含 Torch 的解释器头文件
#include <torch/csrc/jit/testing/file_check.h>  // 包含 Torch 的文件检查测试头文件
#include <sstream>  // 包含标准库的字符串流头文件

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

struct WithCPUFuser {
  WithCPUFuser(bool val = true) : cpuFuserEnabled(canFuseOnCPU()) {
    overrideCanFuseOnCPU(val);
  }

  ~WithCPUFuser() {
    overrideCanFuseOnCPU(cpuFuserEnabled);
  }

  bool cpuFuserEnabled;
};

TEST(TEFuserPass, FuserPass_1) {
  WithCPUFuser cf;  // 创建 WithCPUFuser 结构体实例，用于控制 CPU 上的张量表达式融合
  const auto graph_string = R"IR(
    graph(%0 : Float(128, strides=[1], device=cpu),
          %1 : Float(128, strides=[1], device=cpu)):
      %12 : int = prim::Constant[value=1]()  // 创建常量节点值为 1
      %2.1 : Float(128, strides=[1], device=cpu) = aten::mul(%0, %1)  // 执行张量乘法操作
      %2 : Float(128, strides=[1], device=cpu) = aten::mul(%2.1, %1)  // 继续执行张量乘法操作
      %3 : Float(128, strides=[1], device=cpu) = aten::add_(%2, %1, %12)  // 执行原位加法操作
      %4 : Float(128, strides=[1], device=cpu) = aten::mul(%2, %1)  // 执行张量乘法操作
      %5 : Float(128, strides=[1], device=cpu) = aten::add(%2, %4, %12)  // 执行张量加法操作
      return (%5))IR";  // 返回张量 %5

  auto g = std::make_shared<Graph>();  // 创建共享指针 g 指向 Graph 对象
  torch::jit::parseIR(graph_string, g.get());  // 解析图形字符串并填充到 Graph 对象 g 中

  g->lint();  // 对图形进行 lint 检查
  FuseTensorExprs(g);  // 对图形进行张量表达式融合

  // 我们不能在这里跨越原位操作进行融合。
  testing::FileCheck()
      .check("prim::TensorExprGroup_")
      ->check("aten::add_")
      ->check("prim::TensorExprGroup_")
      ->run(*g);  // 运行文件检查，验证融合结果
}

TEST(TEFuserPass, FuserPass_2) {
  WithCPUFuser cf;  // 创建 WithCPUFuser 结构体实例，用于控制 CPU 上的张量表达式融合
  const auto graph_string = R"IR(
    graph(%0 : Float(128, strides=[1], device=cpu),
          %1 : Float(128, strides=[1], device=cpu)):
      %12 : int = prim::Constant[value=1]()  // 创建常量节点值为 1
      %a : Float(128, strides=[1], device=cpu) = aten::mul(%0, %1)  // 执行张量乘法操作
      %b : Float(128, strides=[1], device=cpu) = aten::add(%0, %1, %12)  // 执行张量加法操作
      %c : Float(128, strides=[1], device=cpu) = aten::add_(%b, %1, %12)  // 执行原位加法操作
      %d : Float(128, strides=[1], device=cpu) = aten::mul(%c, %a)  // 执行张量乘法操作
      return (%d))IR";  // 返回张量 %d

  auto g = std::make_shared<Graph>();  // 创建共享指针 g 指向 Graph 对象
  torch::jit::parseIR(graph_string, g.get());  // 解析图形字符串并填充到 Graph 对象 g 中

  g->lint();  // 对图形进行 lint 检查
  FuseTensorExprs(g);  // 对图形进行张量表达式融合

  // 我们不能在这里跨越原位操作进行融合。
  testing::FileCheck()
      .check("aten::add_")
      ->check("prim::TensorExprGroup_0")
      ->run(*g);  // 运行文件检查，验证融合结果
}

TEST(TEFuserPass, FuserPass_3) {
  WithCPUFuser cf;  // 创建 WithCPUFuser 结构体实例，用于控制 CPU 上的张量表达式融合
  const auto graph_string = R"IR(
    graph(%x : Float(128, strides=[1], device=cpu),
          %y : Float(128, strides=[1], device=cpu)):
      %r : Float(128, strides=[1], device=cpu) = aten::mul(%x, %y)  // 执行张量乘法操作
      return (%r))IR";  // 返回张量 %r

  {
    auto g = std::make_shared<Graph>();  // 创建共享指针 g 指向 Graph 对象
    torch::jit::parseIR(graph_string, g.get());  // 解析图形字符串并填充到 Graph 对象 g 中

    g->lint();  // 对图形进行 lint 检查
    FuseTensorExprs(g, /* min_group_size= */ 2);  // 对图形进行张量表达式融合，设置最小融合组大小为 2

    // 我们不应该创建融合组，因为其大小太小。
    // 使用 FileCheck 检查是否不存在 "prim::TensorExprGroup"，并运行检查
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    // 创建一个名为 g 的共享指针，指向一个新的 Graph 对象
    auto g = std::make_shared<Graph>();
    // 使用 Torch JIT 解析输入的图形字符串，将结果存储在 g 中
    torch::jit::parseIR(graph_string, g.get());

    // 对图形 g 进行 lint 检查
    g->lint();
    // 对图形 g 中的张量表达式进行融合，要求最小融合组大小为 1
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // 由于融合组的大小超过阈值，应该创建一个融合组
    // 使用 FileCheck 检查是否存在 "prim::TensorExprGroup"，并运行检查
    testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
  }
TEST(TEFuserPass, FuserPass_Multidevice) {
  {
    // 创建一个 CPUFuser 对象，用于在 CPU 上进行融合操作
    WithCPUFuser cf;
    // 定义表示计算图的字符串，描述了包含多设备张量的计算
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      return (%cat))IR";
    // 创建共享指针指向一个新的计算图对象
    auto g = std::make_shared<Graph>();
    // 解析上述定义的 IR 字符串并将结果存储在 g 中
    torch::jit::parseIR(graph_string, g.get());

    // 对计算图进行静态检查，确保没有错误
    g->lint();
    // 对计算图进行张量表达式融合，指定最小融合组大小为 1
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // 断言：应该能够成功将张量表达式进行融合
    testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
  }
  {
    // 创建一个 CPUFuser 对象，用于在 CPU 上进行融合操作
    WithCPUFuser cf;
    // 定义表示计算图的字符串，描述了包含多设备张量的计算（包括 CUDA 设备）
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cuda:0),
          %z : Float(30, strides=[1], device=cpu)):
      %dim : int = prim::Constant[value=0]()
      %xyz_list : Tensor[] = prim::ListConstruct(%x, %y, %z)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xyz_list, %dim)
      return (%cat))IR";
    // 创建共享指针指向一个新的计算图对象
    auto g = std::make_shared<Graph>();
    // 解析上述定义的 IR 字符串并将结果存储在 g 中
    torch::jit::parseIR(graph_string, g.get());

    // 对计算图进行静态检查，确保没有错误
    g->lint();
    // 对计算图进行张量表达式融合，指定最小融合组大小为 1
    FuseTensorExprs(g, /* min_group_size= */ 1);
  {
    // 创建一个 CPUFuser 对象，用于处理 CPU 上的融合操作
    WithCPUFuser cf;
    // 定义表示计算图的字符串，包含三个输入节点，分别设备为 CPU 和 CUDA
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(10, strides=[1], device=cuda:0)):
      %dim : int = prim::Constant[value=0]()
      %xy_list : Tensor[] = prim::ListConstruct(%x, %y)
      %xy_cat : Float(30, strides=[1], device=cpu) = aten::cat(%xy_list, %dim)
      %r : Float(30, strides=[1], device=cpu) = aten::mul(%xy_cat, %z)
      return (%r))IR";
    // 创建共享指针 g，解析上述字符串并存储为计算图
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    // 对计算图进行检查，确保没有问题
    g->lint();
    // 执行张量表达式的融合，设置最小组大小为 2
    FuseTensorExprs(g, /* min_group_size= */ 2);

    // 检查不应该将一个节点（cat）融合到另一个节点（mul）之前的设备匹配情况
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    // 创建一个 CPUFuser 对象，用于处理 CPU 上的融合操作
    WithCPUFuser cf;
    // 定义表示计算图的字符串，包含三个输入节点，分别设备为 CPU 和 CUDA
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cpu),
          %z : Float(10, strides=[1], device=cuda:0)):
      %z2 : Tensor = aten::mul(%z, %z)
      %dim : int = prim::Constant[value=0]()
      %xy_list : Tensor[] = prim::ListConstruct(%x, %y, %z2)
      %cat : Float(60, strides=[1], device=cpu) = aten::cat(%xy_list, %dim)
      return (%cat))IR";
    // 创建共享指针 g，解析上述字符串并存储为计算图
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    // 对计算图进行检查，确保没有问题
    g->lint();
    // 执行张量表达式的融合，设置最小组大小为 2
    FuseTensorExprs(g, /* min_group_size= */ 2);

    // 检查不应该将一个节点（mul）融合到另一个节点（cat）之前的设备匹配情况
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    // 创建一个 CPUFuser 对象，用于处理 CPU 上的融合操作
    WithCPUFuser cf;
    // 定义表示计算图的字符串，包含两个输入节点，分别设备为 CPU 和 CUDA
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cpu),
          %y : Float(20, strides=[1], device=cuda:0)):
      %r : Float(10, strides=[1], device=cpu) = aten::mul(%x, %y)
      return (%r))IR";
    // 创建共享指针 g，解析上述字符串并存储为计算图
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    // 对计算图进行检查，确保没有问题
    g->lint();
    // 执行张量表达式的融合，设置最小组大小为 1
    FuseTensorExprs(g, /* min_group_size= */ 1);

    // 检查不应该融合这个图形，因为其输入来自不同的设备
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
  }
  {
    // 创建一个 CPUFuser 对象，用于处理 CPU 上的融合操作
    WithCPUFuser cf;
    // 定义表示计算图的字符串，包含三个输入节点，分别设备为 CUDA
    const auto graph_string = R"IR(
    graph(%x : Float(10, strides=[1], device=cuda:0),
          %y : Float(20, strides=[1], device=cuda:1),
          %z : Float(20, strides=[1], device=cpu)):
      %x2 : Float(10, strides=[1], device=cpu) = aten::mul(%x, %x)
      %y2 : Float(10, strides=[1], device=cpu) = aten::mul(%y, %y)
      %z2 : Float(10, strides=[1], device=cpu) = aten::mul(%z, %z)
      return (%x2, %y2, %z2))IR";
    // 创建共享指针 g，解析上述字符串并存储为计算图
    auto g = std::make_shared<Graph>();
    torch::jit::parseIR(graph_string, g.get());

    // 对计算图进行检查，确保没有问题
    g->lint();
    // 执行张量表达式的融合，设置最小组大小为 2
    FuseTensorExprs(g, /* min_group_size= */ 2);
    // 我们不应该合并这两个计算，因为它们使用了不同的设备
    testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
TEST(TEFuserPass, FuserPass_MergeGroups) {
  // 创建一个 CPU Fuser 的环境
  WithCPUFuser cf;
  // 定义一个包含两个输入的图形字符串
  const auto graph_string = R"IR(
    graph(%a : Float(128, strides=[1], device=cpu),
          %b : Float(128, strides=[1], device=cpu)):
      %x : Float(128, strides=[1], device=cpu) = aten::mul(%a, %a)
      %y : Float(128, strides=[1], device=cpu) = aten::mul(%b, %b)
      return (%x, %y))IR";
  // 创建一个指向图形对象的智能指针
  auto g = std::make_shared<Graph>();
  // 解析图形字符串到图形对象中
  torch::jit::parseIR(graph_string, g.get());

  // 执行图形的静态检查
  g->lint();
  // 对图形执行张量表达式的融合，最小组大小为 1
  FuseTensorExprs(g, /* min_group_size= */ 1);

  // %x 和 %y 的计算是完全独立的，但是我们应该将它们放入单个融合组中，而不是分开两个组
  testing::FileCheck()
      .check("= prim::TensorExprGroup_")
      ->check_not("= prim::TensorExprGroup_")
      ->run(*g);
}

TEST(TEFuserPass, FuserPass_IgnoreUnknownShapeAtStart) {
  // 创建一个 CPU Fuser 的环境
  WithCPUFuser cf;
  // 定义一个包含两个输入的图形字符串
  const auto graph_string = R"IR(
    graph(%x : Bool(8, strides=[1], device=cpu),
          %y : Bool(8, strides=[1], device=cpu)):
      %a : Bool(8, strides=[1], device=cpu) = aten::__and__(%x, %y)
      %b : Tensor = aten::__or__(%a, %y)
      return (%b)
    )IR";
  // 创建一个指向图形对象的智能指针
  auto g = std::make_shared<Graph>();
  // 解析图形字符串到图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 执行图形的静态检查
  g->lint();
  // 对图形执行张量表达式的融合，最小组大小为 2
  FuseTensorExprs(g, /* min_group_size= */ 2);
  // 检查不应存在 prim::TensorExprGroup
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

TEST(TEFuserPass, FuserPass_Where) {
  // 创建一个 CPU Fuser 的环境
  WithCPUFuser cf;
  // 定义一个包含三个输入的图形字符串
  const auto graph_string = R"IR(
    graph(%x : Float(8, strides=[1], device=cpu),
          %y : Float(8, strides=[1], device=cpu),
          %z : Float(8, strides=[1], device=cpu)):
      %cond : Bool(8, strides=[1], device=cpu) = aten::eq(%x, %y)
      %b : Float(8, strides=[1], device=cpu) = aten::where(%cond, %y, %z)
      return (%b)
    )IR";
  // 创建一个指向图形对象的智能指针
  auto g = std::make_shared<Graph>();
  // 解析图形字符串到图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 执行图形的静态检查
  g->lint();
  // 对图形执行张量表达式的融合，最小组大小为 2
  FuseTensorExprs(g, /* min_group_size= */ 2);
  // 检查应存在 prim::TensorExprGroup
  testing::FileCheck().check("prim::TensorExprGroup")->run(*g);
}

TEST(TEFuserPass, FuserPass_WhereList) {
  // 创建一个 CPU Fuser 的环境
  WithCPUFuser cf;
  // 定义一个包含三个输入的图形字符串
  const auto graph_string = R"IR(
    graph(%x : Float(8, strides=[1], device=cpu),
          %y : Float(8, strides=[1], device=cpu),
          %z : Float(8, strides=[1], device=cpu)):
      %cond : Bool(8, strides=[1], device=cpu) = aten::eq(%x, %y)
      %b : Tensor[] = aten::where(%cond)
      return (%b)
    )IR";
  // 创建一个指向图形对象的智能指针
  auto g = std::make_shared<Graph>();
  // 解析图形字符串到图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 执行图形的静态检查
  g->lint();
  // 对图形执行张量表达式的融合，最小组大小为 2
  FuseTensorExprs(g, /* min_group_size= */ 2);
  // 检查不应存在 prim::TensorExprGroup
  testing::FileCheck().check_not("prim::TensorExprGroup")->run(*g);
}

TEST(TEFuserPass, DynamicShapeFusion) {
  // 创建一个 CPU Fuser 的环境
  WithCPUFuser cf;
  // 省略的部分，无需添加注释
  const auto graph_string = R"IR(
  // 定义一个图结构，包含两个输入张量 %0 和 %1，形状为 (10, 5)，存储在 CPU 上
  graph(%0 : Float(10, 5, strides=[5, 1], device=cpu),
        %1 : Float(10, 5, strides=[5, 1], device=cpu)):
    // 计算两个输入张量的逐元素乘积，结果保存在 %2 中
    %2 : Float(10, 5, strides=[5, 1], device=cpu) = aten::mul(%0, %1)
    // 计算 %2 和 %1 的逐元素乘积，结果保存在 %3 中
    %3 : Float(10, 5, strides=[5, 1], device=cpu) = aten::mul(%2, %1)
    // 返回 %3 作为函数的输出
    return (%3))IR";

// 创建一个共享指针指向图对象 g
auto g = std::make_shared<Graph>();
// 解析图的 IR 字符串，填充到图对象 g 中
torch::jit::parseIR(graph_string, g.get());

// 对图 g 进行静态检查
g->lint();

// 对图 g 进行张量表达式的融合，设定最小分组大小为 2，允许合成操作，并支持动态形状融合
FuseTensorExprs(
    g,
    /* min_group_size = */ 2,
    /* add_composed_op = */ true,
    /* fuse_to_dynamic_shapes = */ true);

// 根据图 g 和空字符串创建代码对象
Code code(g, "");

// 使用 FileCheck 对图 g 进行检查，确保包含特定的操作模式
testing::FileCheck()
    .check("prim::TensorExprDynamicGroup_")
    ->check("prim::TensorExprDynamicGuard")
    ->check("prim::TensorExprGroup_")
    ->run(*g);

// 定义一个 Lambda 函数 run_and_compare，用于运行输入的张量，并与预期结果进行比较
auto run_and_compare = [&](const std::vector<at::Tensor>& inputs) {
  // 断言输入张量的数量为 2
  TORCH_INTERNAL_ASSERT(inputs.size() == 2);

  // 计算参考结果 ref，即 inputs[0] * inputs[1] * inputs[1]
  auto ref = at::mul(at::mul(inputs[0], inputs[1]), inputs[1]);

  // 使用解释器状态 interp 执行代码对象 code
  InterpreterState interp(code);
  // 将输入张量压入堆栈 stack
  Stack stack(inputs.begin(), inputs.end());
  // 运行解释器
  interp.run(stack);
  // 弹出堆栈顶部元素，并将其转换为张量类型
  at::Tensor out = pop(stack).toTensor();
  // 断言输出张量 out 与参考结果 ref 在数值上是否接近
  ASSERT_TRUE(at::allclose(out, ref));
};

// 创建三组不同形状的随机输入张量，并分别调用 run_and_compare 进行比较
std::vector<at::Tensor> inputs = {at::rand({10, 5}), at::rand({10, 5})};
run_and_compare(inputs);

std::vector<at::Tensor> inputs2 = {at::rand({20, 5}), at::rand({20, 5})};
run_and_compare(inputs2);

std::vector<at::Tensor> inputs3 = {at::rand({25, 60}), at::rand({25, 60})};
run_and_compare(inputs3);
}

} // namespace jit
} // namespace torch
```