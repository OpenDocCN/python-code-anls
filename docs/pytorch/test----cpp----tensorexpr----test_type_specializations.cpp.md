# `.\pytorch\test\cpp\tensorexpr\test_type_specializations.cpp`

```py
// 包含 Google Test 框架头文件
#include <gtest/gtest.h>

// 包含 Torch 自动微分库生成的变量工厂头文件
#include <torch/csrc/autograd/generated/variable_factories.h>
// 包含 Torch 前端 IR 发射器头文件
#include <torch/csrc/jit/frontend/ir_emitter.h>
// 包含 Torch IR 结构定义头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch IR 解析器头文件
#include <torch/csrc/jit/ir/irparser.h>
// 包含 Torch JIT 日志头文件
#include <torch/csrc/jit/jit_log.h>
// 包含 Torch JIT 通行证管理头文件
#include <torch/csrc/jit/passes/pass_manager.h>
// 包含 Torch JIT 张量表达式融合器头文件
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

// 测试 tensor 类型特化在自定义通行证中是否可用

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 匿名命名空间，内部函数和变量仅在本文件可见
namespace {

// 判断给定基本块是否包含 tensor 类型特化
bool hasTensorTypeSpecializations(torch::jit::Block* block) {
  // 遍历基本块的输入值，检查是否有 tensor 类型特化
  for (Value* v : block->inputs()) {
    if (hasTensorTypeSpecialization(v))
      return true;
  }
  // 遍历基本块的节点，递归检查其子块是否有 tensor 类型特化
  for (Node* n : block->nodes()) {
    for (torch::jit::Block* b : n->blocks()) {
      if (hasTensorTypeSpecializations(b))
        return true;
    }
    // 检查节点的输出值是否有 tensor 类型特化
    for (Value* v : n->outputs()) {
      if (hasTensorTypeSpecialization(v))
        return true;
    }
  }
  return false; // 默认未找到 tensor 类型特化
}

// 静态变量，用于标记是否存在特化
static bool hasSpecializations = false;

// 检测 tensor 类型特化的通行证函数
void detectTTSpecializationPass(std::shared_ptr<Graph>& graph) {
  // 输出图形结构的调试信息
  GRAPH_DUMP("In detectTTSpecialization Custom Post Pass: ", graph);
  // 检测图形中是否存在 tensor 类型特化
  hasSpecializations = hasTensorTypeSpecializations(graph->block());
}

} // namespace

// 测试用例，验证自定义通行证中的特化功能
TEST(SpecializationsInCustomPasses, Basic) {
  // 注册检测 tensor 类型特化的通行证
  RegisterPass p(detectTTSpecializationPass);
  // 初始化特化存在标记为假
  hasSpecializations = false;
  // 创建一个空图形对象
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 解析指定的 IR 字符串并填充图形对象
  parseIR(
      R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %c.1 : Tensor = aten::mul(%a.1, %b.1) # misc/test_specializations.py:5:8
  %d.1 : Tensor = aten::mul(%c.1, %b.1) # misc/test_specializations.py:6:8
  return (%d.1)
  )IR",
      &*graph);

  // 创建一个随机张量值作为输入
  IValue ival = IValue(torch::randn({22}, at::kCPU));
  // 创建输入栈，并复制相同的随机张量值
  std::vector<IValue> stack = {ival, ival};
  
  // 定义一个执行函数，执行图形对象和输入栈，返回输出栈
  auto run = [&](std::shared_ptr<Graph>& graph, std::vector<IValue> stack) {
    GraphExecutor executor(graph, "");
    executor.run(stack);
    return stack;
  };
  
  // 执行运行函数，传入图形对象和输入栈
  run(graph, stack);

  // 如果非执行器模式，断言存在 tensor 类型特化
  if (!getExecutorMode()) {
    EXPECT_TRUE(hasSpecializations);
  }
}

} // namespace jit
} // namespace torch
```