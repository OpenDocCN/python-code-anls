# `.\pytorch\torch\csrc\jit\codegen\fuser\interface.cpp`

```py
// 引入 Torch 的 JIT Fuser 接口头文件
#include <torch/csrc/jit/codegen/fuser/interface.h>

// 引入 Torch 的 JIT Fuser 编译器头文件
#include <torch/csrc/jit/codegen/fuser/compiler.h>
// 引入 Torch 的 JIT Fuser 执行器头文件
#include <torch/csrc/jit/codegen/fuser/executor.h>
// 引入 Torch 的 JIT Fuser 回退机制头文件
#include <torch/csrc/jit/codegen/fuser/fallback.h>
// 引入 Torch 的 JIT Fuser 内核缓存头文件
#include <torch/csrc/jit/codegen/fuser/kernel_cache.h>

// 引入 C10 库的 Flags 头文件
#include <c10/util/Flags.h>

// 引入标准异常处理头文件
#include <stdexcept>

// Torch 的命名空间
namespace torch {
// JIT 的命名空间
namespace jit {

// JIT Fuser 的细节命名空间
namespace detail {

// 根据 TORCH_ENABLE_LLVM 宏定义决定 CPU Fuser 是否启用
#ifdef TORCH_ENABLE_LLVM
bool cpu_fuser_enabled = true;
#else
bool cpu_fuser_enabled = false;
#endif

// GPU Fuser 默认启用
bool gpu_fuser_enabled = true;

} // namespace detail

// 注册融合操作的函数，返回融合操作的键值
int64_t registerFusion(const Node* fusion_group) {
  return fuser::registerFusion(fusion_group);
}

// 执行融合操作，给定键值和栈作为参数
void runFusion(const int64_t key, Stack& stack) {
  // 尝试运行融合操作
  const auto result = fuser::runFusion(key, stack);
  // 如果运行失败，执行回退操作
  if (!result)
    fuser::runFallback(key, stack);
}

// 检查是否可以在 CPU 上进行融合操作
bool canFuseOnCPU() {
  return fuser::hasFusionBackend(DeviceType::CPU) && detail::cpu_fuser_enabled;
}

// 检查是否可以在 GPU 上进行融合操作
bool canFuseOnGPU() {
  return fuser::hasFusionBackend(DeviceType::CUDA) && detail::gpu_fuser_enabled;
}

// 覆盖 CPU 上的融合操作使能状态
void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

// 覆盖 GPU 上的融合操作使能状态
void overrideCanFuseOnGPU(bool value) {
  detail::gpu_fuser_enabled = value;
}

// 使用上述接口将图形插入节点并将该节点视为融合组，并调试启动该图形
std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // 创建包装图形
  auto wrapper_graph = std::make_shared<Graph>();
  // 插入融合组节点
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  // 将图形复制到融合组的子图属性中
  fusion_group->g_(attr::Subgraph, graph.copy());
  // 添加输入节点到融合组
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  // 注册融合组的输出节点
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // 创建栈，注册并运行融合操作
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);
  fuser::runFusion(key, stack);
  // 返回栈中的张量作为结果
  return fmap(stack, [](const IValue& iv) { return iv.toTensor(); });
}

// 获取融合内核代码的调试接口
std::string debugGetFusedKernelCode(
    Graph& graph,
    at::ArrayRef<at::Tensor> inputs) {
  // 创建包装图形
  auto wrapper_graph = std::make_shared<Graph>();
  // 插入融合组节点
  Node* fusion_group = wrapper_graph->insertNode(
      wrapper_graph->createWithSubgraph(prim::FusionGroup));
  // 将图形复制到融合组的子图属性中
  fusion_group->g_(attr::Subgraph, graph.copy());
  // 添加输入节点到融合组
  for (size_t i = 0; i < graph.inputs().size(); ++i) {
    fusion_group->addInput(wrapper_graph->addInput());
  }
  // 注册融合组的输出节点
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    wrapper_graph->registerOutput(fusion_group->addOutput());
  }

  // 创建栈，注册并运行融合操作，获取生成的内核代码
  Stack stack = fmap<IValue>(inputs);
  const auto key = fuser::registerFusion(fusion_group);

  std::string code;
  // 尝试运行融合操作并获取内核代码
  if (!fuser::runFusion(key, stack, &code)) {
    // 抛出一个 std::runtime_error 异常，指示图形的融合运行失败
    throw std::runtime_error("Could not run fusion for graph");
  }
  
  // 返回从融合操作中获取的代码
  return code;
// 返回编译的内核数目，调用了 fuser 命名空间中的 nCompiledKernels 函数
size_t nCompiledKernels() {
    return fuser::nCompiledKernels();
}

// 结束 jit 命名空间
} // namespace jit

// 结束 torch 命名空间
} // namespace torch
```