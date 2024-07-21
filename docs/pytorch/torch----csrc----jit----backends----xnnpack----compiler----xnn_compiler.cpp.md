# `.\pytorch\torch\csrc\jit\backends\xnnpack\compiler\xnn_compiler.cpp`

```
// 版权声明及许可信息
//
// 该源代码使用 BSD 风格许可证授权，许可条款详见根目录下的 LICENSE 文件。

#include <caffe2/torch/csrc/jit/backends/xnnpack/compiler/xnn_compiler.h>
#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>

#include <ATen/Utils.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

// XNNCompiler 类的方法，用于编译模型
void XNNCompiler::compileModel(
    const void* buffer_pointer,
    size_t num_bytes,
    XNNExecutor* executor) {
  // 设置输出的最小值为负无穷大，最大值为正无穷大
  auto output_min = -std::numeric_limits<float>::infinity();
  auto output_max = std::numeric_limits<float>::infinity();

  // 从 flatbuffer 数据中获取 XNNGraph 对象
  auto flatbuffer_graph = fb_xnnpack::GetXNNGraph(buffer_pointer);

  // 初始化 XNNPack
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  TORCH_CHECK(xnn_status_success == status, "Failed to initialize xnnpack");

  // 创建 XNNPack 子图
  xnn_subgraph_t subgraph_ptr = nullptr;
  status = xnn_create_subgraph(
      /*external_value_ids=*/flatbuffer_graph->num_externs(),
      /*flags=*/0,
      &subgraph_ptr);
  TORCH_CHECK(xnn_status_success == status, "Failed to create xnn subgraph");

  // 用于将旧的 ID 映射到新创建的值 ID 的哈希映射表
  std::unordered_map<uint32_t, uint32_t> remapped_ids;

  // 遍历 XNNGraph 中的每个值
  for (auto value : *flatbuffer_graph->xvalues()) {
    switch (value->xvalue_type()) {
      case fb_xnnpack::XValueUnion::XNNTensorValue: {
        // 如果是 XNNTensorValue 类型的值
        auto tensor_value = value->xvalue_as_XNNTensorValue();

        // 从 flatbuffer 中获取张量的维度信息
        std::vector<size_t> dims_data;
        for (auto dim : *tensor_value->dims()) {
          dims_data.push_back(static_cast<size_t>(dim));
        }

        // 定义张量值并获取其 ID
        uint32_t id = XNN_INVALID_VALUE_ID;
        const auto& constant_buffer = *flatbuffer_graph->constant_buffer();
        auto buffer_idx = tensor_value->constant_buffer_idx();
        const auto buffer_ptr = buffer_idx == 0
            ? nullptr
            : constant_buffer[buffer_idx]->storage()->data();
        status = xnn_define_tensor_value(
            /*subgraph=*/subgraph_ptr,
            /*datatype=*/xnn_datatype_fp32,
            /*num_dims=*/tensor_value->num_dims(),
            /*dims=*/dims_data.data(),
            /*data=*/buffer_ptr,
            /*external_id=*/tensor_value->external_id(),
            /*flags=*/tensor_value->flags(),
            /*id_out=*/&id);
        TORCH_CHECK(
            status == xnn_status_success,
            "Failed to define tensor values in graph")
        // 将序列化的 ID 映射到新生成的 ID
        remapped_ids.emplace(std::make_pair(tensor_value->id_out(), id));
        break;
      }
      default: {
        // 处理未处理的值类型异常情况
        TORCH_CHECK(false, "Unhandled value type found in deserialization");
      }
    }
  }
  // 遍历 flatbuffer 图中的每个节点
  for (auto node : *flatbuffer_graph->xnodes()) {
    // 根据节点类型进行不同的处理
    switch (node->xnode_type()) {
      case fb_xnnpack::XNodeUnion::XNNAdd: {
        // 如果节点类型是 XNNAdd，则处理加法节点
        auto graph_node = node->xnode_as_XNNAdd();
        // 调用 xnn_define_add2 定义加法操作
        status = xnn_define_add2(
            subgraph_ptr,
            output_min,
            output_max,
            remapped_ids.at(graph_node->input1_id()),
            remapped_ids.at(graph_node->input2_id()),
            remapped_ids.at(graph_node->output_id()),
            graph_node->flags());
        // 检查操作执行状态，如果失败则抛出异常
        TORCH_CHECK(status == xnn_status_success, "Failed to create add node")
        break;
      }
      default:
        // 对于未处理的节点类型，抛出异常
        TORCH_CHECK(false, "Unhandled node type found in deserialization");
    }
  }

  // 创建 XNN 运行时环境
  xnn_runtime_t runtime_ptr = nullptr;
  status = xnn_create_runtime_v2(subgraph_ptr, nullptr, 0, &runtime_ptr);
  // 检查运行时创建状态，确保成功
  TORCH_CHECK(xnn_status_success == status);

  // 将创建的运行时环境与执行器关联
  executor->runtime_ =
      std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)>(
          runtime_ptr, xnn_delete_runtime);

  // 将重映射后的输入节点 ID 添加到执行器的输入 ID 列表中
  for (auto old_id : *flatbuffer_graph->input_ids()) {
    executor->input_ids_.emplace_back(remapped_ids.at(old_id));
  }

  // 将重映射后的输出节点 ID 添加到执行器的输出 ID 列表中
  for (auto old_id : *flatbuffer_graph->output_ids()) {
    executor->output_ids_.emplace_back(remapped_ids.at(old_id));
  }
};

// 结束 xnnpack 命名空间
} // namespace delegate
// 结束 delegate 命名空间
} // namespace xnnpack
// 结束 xnnpack 命名空间
} // namespace jit
// 结束 jit 命名空间
} // namespace torch
// 结束 torch 命名空间
```