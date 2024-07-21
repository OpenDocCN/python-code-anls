# `.\pytorch\torch\csrc\jit\codegen\onednn\kernel.h`

```
#pragma once

// `#pragma once` 是预处理指令，用于确保头文件只被包含一次，提高编译效率。


#include <unordered_map>

// 包含 `<unordered_map>` 头文件，用于使用标准库中的无序映射容器。


#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>

// 包含一系列的头文件，用于使用相应的库或模块，例如 Intel DNNL 图模块、PyTorch JIT 编码器相关模块等。


#include <c10/util/CallOnce.h>

// 包含 `<c10/util/CallOnce.h>` 头文件，该头文件提供了 `c10::call_once` 函数，用于实现线程安全的一次性初始化。


namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 进入命名空间 `torch::jit::fuser::onednn`，用于组织和隔离代码，避免命名冲突。


using ArgSpec = LlgaTensorDesc;
using ArgSpecs = std::vector<ArgSpec>;
using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using TensorArgs = std::vector<at::Tensor>;

// 使用别名定义，简化类型名称：`ArgSpec` 和 `ArgSpecs` 是关于张量描述的类型；`RunArg` 和 `RunArgs` 是用于执行图张量的类型；`TensorArgs` 是包含 PyTorch 张量的向量。


class LlgaKernel {
 public:
  explicit LlgaKernel(const Node* fusionNode);

  void run(Stack& stack);

  void initialize(const TensorArgs& inputs);

  const std::string& debugName() const {
    return debugName_;
  }

// `LlgaKernel` 类声明开始，包含公共和私有成员函数，如构造函数 `LlgaKernel`，`run` 函数用于执行核心操作，`initialize` 函数用于初始化操作，`debugName()` 函数返回调试名称的引用。


 private:
  bool useOpaqueLayout(size_t offset) const;

  // PyTorch copy constants inside the subgraph instead of referencing them.
  // Constants inputs to the partition are no longer in the graph->inputs().
  // Need use the tid retrieved from the partition to find the missing
  // constant inputs.
  void initializeConstantInputs();

  ArgSpecs initializeInputSpecs(const TensorArgs& inputs);

  ArgSpecs initializeOutputSpecs() const;

  dnnl::graph::compiled_partition compile(
      const dnnl::graph::partition& partition);

  std::map<size_t, int64_t> initializeTensorIdToOccurence() const;

  std::tuple<RunArgs, RunArgs> prepareRunArgs(
      const TensorArgs& inputs,
      TensorArgs& outputs) const;

  static std::string genDebugName() {
    static size_t debugId = 0;
    return "LlgaPartition_" + std::to_string(debugId++);
  }

  static dnnl::graph::logical_tensor toLogicalTensor(const ArgSpec& s) {
    return s.logical_tensor();
  }

  at::Device device_ = at::kCPU;
  const Node* fusionNode_;
  std::shared_ptr<Graph> graph_;
  int64_t nGraphInputs_ = 0; // number of inputs to graph_ on the IR
  int64_t nOutputs_ = 0;
  std::map<size_t, Value*> tensorIdToValue_;
  std::vector<int64_t> runArgsIdx_;
  dnnl::graph::partition partition_;
  // nPartitionInputs_ is the actual number of inputs to partition_ of graph_
  // needed by the backend.
  // nPartitionInputs_ = nGraphInputs_ + constantInputs_.size() since Constant
  // inputs are copied to the inside of the subgraph
  int64_t nPartitionInputs_;
  dnnl::graph::compiled_partition compilation_;
  std::set<size_t> initializedInputIds_;
  std::vector<Value*> constantValues_;
  TensorArgs constantInputs_;
  ArgSpecs inputSpecs_;
  ArgSpecs outputSpecs_;
  std::vector<dnnl::graph::logical_tensor> constantLogicalTensors_;
  std::string debugName_;
  c10::once_flag initialized_flag;
  bool is_initialized_ = false;
};

// `LlgaKernel` 类的私有成员包括一系列方法和属性：`useOpaqueLayout` 判断是否使用不透明布局；`initializeConstantInputs` 初始化常量输入；`initializeInputSpecs` 和 `initializeOutputSpecs` 初始化输入和输出规格；`compile` 编译图分区；`initializeTensorIdToOccurence` 初始化张量 ID 到出现次数的映射；`prepareRunArgs` 准备运行参数；`genDebugName()` 生成调试名称的静态方法；`toLogicalTensor` 将 `ArgSpec` 转换为逻辑张量。还有一系列成员变量用于存储状态和数据。


} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch

// 结束命名空间的声明，确保代码组织清晰，避免全局命名冲突。


#endif // LLGA_KERNEL_H

// `#endif` 是条件预处理指令的结尾，与 `#pragma once` 对应，用于结束条件编译区块，确保头文件只被包含一次。
```