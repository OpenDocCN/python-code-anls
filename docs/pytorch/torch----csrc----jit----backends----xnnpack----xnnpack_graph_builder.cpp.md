# `.\pytorch\torch\csrc\jit\backends\xnnpack\xnnpack_graph_builder.cpp`

```
// 包含头文件 xnnpack_graph_builder.h，用于构建 XNNPack 图的后端
#include <caffe2/torch/csrc/jit/backends/xnnpack/xnnpack_graph_builder.h>

// 包含图迭代器头文件，用于遍历 Torch JIT 图
#include <torch/csrc/jit/runtime/graph_iterator.h>

// 包含 XNNPack 库的头文件
#include <xnnpack.h>

// 包含 Torch JIT 中的图优化 pass
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>

// 命名空间声明开始
namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

// 优化和跟踪 Torch JIT 图的函数，返回优化后的图
std::shared_ptr<torch::jit::Graph> XNNGraph::optimizeAndTraceGraph(
    std::shared_ptr<torch::jit::Graph> graph,
    std::vector<c10::IValue>& example_inputs) {
  // 优化冻结图
  OptimizeFrozenGraph(graph, true);

  // 移除列表变异
  RemoveListMutation(graph);

  // 移除张量变异
  RemoveTensorMutation(graph);

  // 将所有元组降级
  LowerAllTuples(graph);

  // 常量传播
  ConstantPropagation(graph);

  // 跟踪图，并更新图对象
  graph = TraceGraph(graph, example_inputs);

  return graph;
}

// 构建 XNNPack 图
void XNNGraph::buildXNNGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    std::vector<c10::IValue> example_inputs) {
  // 优化和跟踪 Torch JIT 图
  graph = optimizeAndTraceGraph(graph, example_inputs);

  // 检查需要委托的操作
  checkOpsToDelegate(graph);

  // 收集张量值
  gatherTensorValues(graph);

  // 统计唯一输入/输出值（某些输入可能是输出）
  std::unordered_set<torch::jit::Value*> externals;
  for (auto inp : _inputs) {
    externals.insert(inp);
  }
  for (auto out : _outputs) {
    externals.insert(out);
  }

  // 创建子图
  xnn_status status = xnn_create_subgraph(
      /*external_value_ids=*/externals.size(),
      /*flags=*/0,
      &_subgraph_ptr);
  TORCH_CHECK(xnn_status_success == status, "Failed to create xnn subgraph");

  // 定义所有张量值
  defineAllTensorValues();

  // 定义所有节点
  defineAllNodes(graph);

  // 在此时图已完成，为了测试目的，在此处进行运行时设置和使用默认值运行
}

// 在输入上运行图
void XNNGraph::runGraphOnInputs(
    std::vector<at::Tensor> tensor_inputs,
    std::vector<at::Tensor> tensor_outputs) {
  // 确保子图指针不为空
  TORCH_CHECK(
      _subgraph_ptr != nullptr,
      "run buildXNNGraph before running graph on inputs");

  // 创建 XNN 运行时
  xnn_runtime_t runtime = nullptr;
  xnn_status status =
      xnn_create_runtime_v2(_subgraph_ptr, nullptr, /*flags=*/0, &runtime);
  TORCH_CHECK(
      xnn_status_success == status,
      "failed to create runtime for running inputs");

  // 智能指针管理运行时
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  // 外部值向量
  std::vector<xnn_external_value> external_values;

  // 确保输入张量大小与预期输入大小匹配
  TORCH_CHECK(
      tensor_inputs.size() == _inputs.size(),
      "supplied inputs does not match expected inputs");

  // 遍历输入张量并添加到外部值中
  for (int i = 0; i < tensor_inputs.size(); i++) {
    external_values.push_back(
        {_val_to_ids[_inputs[i]], tensor_inputs[i].data_ptr<float>()});


    // 将输入张量的标识符和数据指针加入外部数值列表
    external_values.push_back(
        {_val_to_ids[_inputs[i]], tensor_inputs[i].data_ptr<float>()});



  TORCH_CHECK(
      tensor_outputs.size() == _outputs.size(),
      "supplied outputs does not match expected outputs");


  // 使用 Torch 的检查功能确保输出张量的数量与期望输出相匹配
  TORCH_CHECK(
      tensor_outputs.size() == _outputs.size(),
      "supplied outputs does not match expected outputs");



  for (int i = 0; i < tensor_outputs.size(); i++) {
    external_values.push_back(
        {_val_to_ids[_outputs[i]], tensor_outputs[i].data_ptr<float>()});
  }


  // 将输出张量的标识符和数据指针加入外部数值列表
  for (int i = 0; i < tensor_outputs.size(); i++) {
    external_values.push_back(
        {_val_to_ids[_outputs[i]], tensor_outputs[i].data_ptr<float>()});
  }



  status = xnn_setup_runtime(
      auto_runtime.get(), external_values.size(), external_values.data());


  // 设置运行时环境，使用外部数值列表初始化自动运行时对象
  status = xnn_setup_runtime(
      auto_runtime.get(), external_values.size(), external_values.data());



  TORCH_CHECK(xnn_status_success == status, "runtime not properly setup");


  // 使用 Torch 的检查功能确保运行时环境设置成功
  TORCH_CHECK(xnn_status_success == status, "runtime not properly setup");



  TORCH_CHECK(xnn_status_success == xnn_invoke_runtime(auto_runtime.get()));


  // 使用 Torch 的检查功能确保调用运行时环境成功
  TORCH_CHECK(xnn_status_success == xnn_invoke_runtime(auto_runtime.get()));
}

void XNNGraph::checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph) {
  // 创建一个集合，用于存储不支持的操作类型
  std::unordered_set<string> unsupported_ops;
  // 创建深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(graph);
  // 初始化节点指针
  Node* node = nullptr;
  // 迭代遍历图中的每个节点
  while ((node = it.next()) != nullptr) {
    // 根据节点类型进行分类处理
    switch (node->kind()) {
      case prim::Constant:
      case aten::add: {
        // 对于常量和加法操作，不做特殊处理
        break;
      }
      default: {
        // 将不支持的操作类型添加到集合中
        unsupported_ops.insert(node->kind().toDisplayString());
      }
    }
  }
  // 创建字符串流，用于存储不支持操作的错误信息
  std::stringstream error;
  // 遍历不支持操作的集合，并将每个操作类型写入错误流中
  for (auto itr = unsupported_ops.begin(); itr != unsupported_ops.end();
       itr++) {
    error << *itr << std::endl;
    ;
  }
  // 检查是否存在不支持的操作类型，若存在则抛出错误
  TORCH_CHECK(
      unsupported_ops.empty(),
      "the module contains the following unsupported ops:\n" + error.str());
}

std::string XNNGraph::serializedXNNGraph() {
  // 创建输入节点和输出节点的标识符向量
  std::vector<uint32_t> input_ids;
  std::vector<uint32_t> output_ids;
  // 创建用于记录外部节点数量的集合
  std::unordered_set<uint32_t> num_externs;

  // 遍历输入节点，并将每个节点的标识符和外部节点记录到对应的数据结构中
  for (auto val : _inputs) {
    input_ids.push_back(_val_to_ids[val]);
    num_externs.emplace(_val_to_ids[val]);
  }

  // 遍历输出节点，并将每个节点的标识符和外部节点记录到对应的数据结构中
  for (auto val : _outputs) {
    output_ids.push_back(_val_to_ids[val]);
    num_externs.emplace(_val_to_ids[val]);
  }

  // 调用序列化器的方法完成并序列化图结构，返回序列化结果
  return _serializer.finishAndSerialize(
      input_ids, output_ids, num_externs.size());
}

std::vector<std::vector<long>> XNNGraph::getGraphOutputShapes() {
  // 创建用于存储输出形状的二维向量
  std::vector<std::vector<long>> output_shapes;
  // 遍历每个输出节点，并获取其张量类型的大小信息，存储到输出形状向量中
  for (auto val : _outputs) {
    auto tensor_ptr = val->type()->cast<TensorType>();
    std::vector<long> sizes = tensor_ptr->sizes().concrete_sizes().value();
    output_shapes.push_back(sizes);
  }

  // 返回所有输出节点的形状信息
  return output_shapes;
}

void XNNGraph::defineAllNodes(std::shared_ptr<torch::jit::Graph>& graph) {
  // 创建深度优先图节点迭代器
  DepthFirstGraphNodeIterator it(graph);
  // 初始化节点指针
  Node* node = nullptr;
  // 迭代遍历图中的每个节点
  while ((node = it.next()) != nullptr) {
    // 根据节点类型进行分类处理
    switch (node->kind()) {
      case prim::Constant: {
        // 对于常量节点，不做特殊处理
        break;
      }
      case aten::add: {
        // 处理加法操作节点，包括 alpha 值的检查和节点定义
        uint32_t input1_id = _val_to_ids[node->inputs()[0]];
        uint32_t input2_id = _val_to_ids[node->inputs()[1]];
        TORCH_CHECK(
            node->inputs()[2]->type()->cast<IntType>() == 1,
            "non-1 alpha values not supported");
        uint32_t output_id = _val_to_ids[node->outputs()[0]];

        // 使用神经网络定义工具定义加法操作节点，并进行序列化
        xnn_status status = xnn_define_add2(
            _subgraph_ptr,
            output_min,
            output_max,
            input1_id,
            input2_id,
            output_id,
            /*flags=*/0);
        _serializer.serializeAddNode(input1_id, input2_id, output_id, 0);
        // 检查节点定义的状态，如果失败则抛出错误
        TORCH_CHECK(status == xnn_status_success, "failed to create add node");
        break;
      }
      default: {
        // 对于不支持的节点类型，抛出异常并抛出相应的错误信息
        throw std::exception();
        TORCH_CHECK(
            false,
            "The node of ",
            node->kind().toQualString(),
            " is not supported yet");
        break;
      }
    }
  }
}
void XNNGraph::defineAllTensorValues() {
  // 从最小的无效值ID开始为每个外部值分配唯一ID
  uint32_t external_id =
      std::numeric_limits<decltype(XNN_INVALID_VALUE_ID)>::min();
  // 遍历所有中间张量
  for (auto val : _intermediate_tensors) {
    // 如果当前值尚未在映射中，则进行处理
    if (_val_to_ids.find(val) == _val_to_ids.end()) {
      // 初始化为无效值ID
      uint32_t id = XNN_INVALID_VALUE_ID;

      // 将值转换为张量类型指针
      auto tensor_ptr = val->type()->cast<TensorType>();
      auto num_dims = tensor_ptr->dim().value();

      // 创建张量形状的size_t*，将long转换为size_t
      std::vector<long> sizes = tensor_ptr->sizes().concrete_sizes().value();
      std::vector<size_t> tensor_shape;
      tensor_shape.reserve(sizes.size());
      for (auto dim : sizes) {
        // 确保维度非负
        TORCH_CHECK(dim >= 0, "Input Dims should be unsigned");
        tensor_shape.push_back(static_cast<size_t>(dim));
      }

      // 外部ID值初始化为无效值ID
      uint32_t ext_id = XNN_INVALID_VALUE_ID;

      // 标志位，用于指示张量是否为图的输入或输出
      uint32_t flags = 0;

      // 检查值是否由prim::Constant生成
      void* value_data = nullptr;
      size_t buffer_idx = 0;
      size_t num_bytes = 0;
      if (val->node()->kind() == prim::Constant) {
        // 提取常量值的数据，确保数据连续以便序列化
        std::optional<IValue> constant = val->node()->t(attr::value);
        auto const_val = constant->toIValue().toTensor();
        auto cont_const_val = const_val.contiguous();
        value_data = cont_const_val.data_ptr();

        num_bytes = const_val.storage().nbytes();
        // 序列化数据并获取缓冲区索引
        buffer_idx = _serializer.serializeData(
            static_cast<const uint8_t*>(value_data), num_bytes);
      }

      // 如果值是图的输入或输出，则更新标志位和外部ID
      if (isGraphInput(val) || isGraphOutput(val)) {
        if (isGraphInput(val)) {
          flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
        }
        if (isGraphOutput(val)) {
          flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
        }
        ext_id = external_id++;
      }

      // 定义张量值并获取其ID
      xnn_status status = xnn_define_tensor_value(
          /*subgraph=*/_subgraph_ptr,
          /*datatype=*/xnn_datatype_fp32,
          /*num_dims=*/num_dims,
          /*dims=*/tensor_shape.data(),
          /*data=*/value_data,
          /*external_id=*/ext_id,
          /*flags=*/flags,
          /*id_out=*/&id);
      // 检查张量值定义是否成功
      TORCH_CHECK(
          status == xnn_status_success,
          "failed to define xnn_tensor_id for: " + val->debugName());
      
      // 序列化张量值
      _serializer.serializeTensorValue(
          xnn_datatype_fp32,
          num_dims,
          tensor_shape,
          buffer_idx,
          ext_id,
          flags,
          id);
      
      // 将值与其ID映射存入映射表
      _val_to_ids.insert({val, id});
    }
  }
}

void XNNGraph::gatherTensorValues(std::shared_ptr<torch::jit::Graph>& graph) {
  // 将图的输入张量添加到中间张量集合和输入列表中
  for (auto input : graph->inputs()) {
    if (input->isCompleteTensor()) {
      _intermediate_tensors.insert(input);
      _inputs.push_back(input);
    }
  }

  // 使用深度优先方式遍历图中的每个节点
  DepthFirstGraphNodeIterator it(graph);
  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    gatherNodeInputs(*n);
  }

  # 遍历图中所有输出节点
  for (auto output : graph->outputs()) {
    # 检查输出节点是否为完整的张量
    if (output->isCompleteTensor()) {
      # 将完整的张量添加到中间张量集合中
      _intermediate_tensors.insert(output);
      # 将输出节点添加到输出列表中
      _outputs.push_back(output);
    }
  }
}

void XNNGraph::gatherNodeInputs(torch::jit::Node& node) {
  switch (node.kind()) {
    case aten::add: {
      // 处理所有只有两个输入的操作，如 sub、add 等
      for (auto value : node.inputs()) {
        // 如果值是完整的张量
        if (value->isCompleteTensor()) {
          // 将完整的张量插入到中间张量集合中
          _intermediate_tensors.insert(value);
        }
      }
    }
  }
}

bool XNNGraph::isGraphInput(torch::jit::Value* val) {
  // 检查给定值是否是图的输入
  return std::count(_inputs.begin(), _inputs.end(), val) > 0;
};

bool XNNGraph::isGraphOutput(torch::jit::Value* val) {
  // 检查给定值是否是图的输出
  return std::count(_outputs.begin(), _outputs.end(), val) > 0;
};

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
```