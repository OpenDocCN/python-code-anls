# `.\pytorch\torch\csrc\lazy\ts_backend\ts_lowering_context.h`

```
#pragma once
// 预处理指令：确保头文件只被包含一次

#include <sstream>
// 包含头文件：用于字符串流操作

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>
// 包含头文件：导入 Torch C++ API 和相关模块

namespace torch {
namespace lazy {

using TSOpVector = std::vector<torch::jit::Value*>;
// 命名空间定义：定义 torch::lazy 命名空间和 TSOpVector 别名

class TORCH_API TSComputation : public Computation {
 public:
  TSComputation(const std::shared_ptr<torch::jit::Graph>& graph)
      : graph_(graph), graph_executor_(graph, "") {
    for (torch::jit::Value* input : graph_->inputs()) {
      parameter_names_.push_back(input->debugName());
    }
  }
  // 类定义：TSComputation 类，继承自 Computation

  int parameters_size() const override {
    return parameter_names_.size();
  }
  // 方法：返回参数数量

  const std::vector<Shape>& parameter_shapes() const override {
    throw std::runtime_error(
        "TODO(whc) implement TS computation shapes or change interface");
    return parameter_shapes_;
  }
  // 方法：返回参数形状（未实现，抛出运行时错误）

  const std::vector<std::string>& parameter_names() const override {
    return parameter_names_;
  }
  // 方法：返回参数名称列表

  const Shape& result_shape() const override {
    throw std::runtime_error(
        "TODO(whc) implement TS computation shapes or change interface");
    return result_shape_;
  }
  // 方法：返回结果形状（未实现，抛出运行时错误）

  const std::string to_string() const override {
    std::ostringstream oss;
    oss << *graph_;
    return oss.str();
  }
  // 方法：返回对象的字符串表示形式

  std::shared_ptr<torch::jit::Graph> graph() const {
    return graph_;
  }
  // 方法：返回内部 Torch 图对象的共享指针

  torch::jit::GraphExecutor& graph_executor() {
    return graph_executor_;
  }
  // 方法：返回内部 Torch 图执行器的引用

 private:
  std::shared_ptr<torch::jit::Graph> graph_;
  torch::jit::GraphExecutor graph_executor_;
  std::vector<std::string> parameter_names_;
  std::vector<Shape> parameter_shapes_;
  Shape result_shape_;
};
// 类定义：TSComputation 类的私有成员和公共方法实现

class TORCH_API TSLoweringContext : public LoweringContext {
 public:
  TSLoweringContext(const std::string& name, const BackendDevice device);
  // 构造函数声明：接受名称和后端设备参数

  TSLoweringContext(
      const std::string& name,
      BackendDevice device,
      c10::ArrayRef<const Node*> post_order,
      Util::EmissionMap emit_status);
  // 构造函数声明：接受名称、后端设备、节点后序数组和发射状态映射参数

  size_t AddResult(const Output& output) override {
    return AddResult(GetOutputOp(output));
  }
  // 方法：添加结果输出，调用 GetOutputOp 获取输出操作

  void AddParameter(
      const torch::lazy::Output& output,
      size_t index,
      const Shape& shape,
      const std::string& name) override {
    TORCH_INTERNAL_ASSERT(false, "not implemented");
  }
  // 方法：添加参数（未实现，断言失败）

  void Lower(const Node* node);
  // 方法声明：降低节点操作

  ComputationPtr Build() override {
    for (torch::jit::Value* output : root_tuple_) {
      graph_->block()->registerOutput(output);
    }
    return std::shared_ptr<Computation>(new TSComputation(graph_));
  }
  // 方法：构建计算对象，注册根节点输出并返回 TSComputation 对象的共享指针

  // Retrieves the lowered operation for an output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding TS operation returned.
  torch::jit::Value* GetOutputOp(const Output& output) {
    auto it = emitted_outputs_.find(output);
    // 如果输出在已发射的输出集合中不存在
    if (it == emitted_outputs_.end()) {
      // 计算输出节点的后序遍历顺序
      auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
      // 对后序遍历中的每个节点进行降级操作
      for (auto node : post_order) {
        Lower(node);
      }
      // 现在输出应该已经存在，否则降级代码存在问题
      // 查找输出在已发射的输出集合中的位置
      it = emitted_outputs_.find(output);
      // 如果找不到输出，抛出错误
      TORCH_CHECK(
          it != emitted_outputs_.end(),
          "No TS operation emitted for output: ",
          output.ToString());
    }
    // 返回找到的输出对应的值
    return it->second;
  }

  // 将给定的 TS 操作分配给指定的输出。由于输出以后序遍历的方式降级，
  // 后续节点应始终在已发射的输出集合中找到其操作数。
  void AssignOutputOp(const Output& output, torch::jit::Value* op);

  // 如果与数据关联的参数已经声明，则返回该参数；否则将创建一个新的参数，
  // 与数据中持有的张量相关联。
  torch::jit::Value* GetParameter(BackendDataPtr data);

  std::shared_ptr<torch::jit::Graph> graph() const {
    // 返回对象持有的图
    return graph_;
  }

 private:
  // 参数结构体定义
  struct Parameter {
    torch::jit::Value* param{nullptr};  // 参数值指针，默认为空
    size_t index = 0;  // 参数索引，默认为0
  };

  // 添加结果到根元组中
  size_t AddResult(torch::jit::Value* op) {
    root_tuple_.push_back(std::move(op));
    // 返回根元组中添加的结果的索引
    return root_tuple_.size() - 1;
  }

  std::shared_ptr<torch::jit::Graph> graph_;  // 图对象的共享指针
  std::shared_ptr<torch::jit::GraphFunction> function_;  // 图函数的共享指针
  std::unordered_map<BackendData::Handle, Parameter> parameters_map_;  // 后端数据句柄到参数映射的无序映射表
  std::vector<torch::jit::Value*> root_tuple_;  // 根元组，存储了一组值的向量
  OutputMap<torch::jit::Value*> emitted_outputs_;  // 已发射输出的映射
};

} // namespace lazy
} // namespace torch
```