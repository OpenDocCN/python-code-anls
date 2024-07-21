# `.\pytorch\torch\csrc\jit\backends\xnnpack\serialization\serializer.cpp`

```
// 包含必要的头文件
#include <caffe2/torch/csrc/jit/backends/xnnpack/serialization/serializer.h>
#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>

// 包含标准库
#include <sstream>

// 定义命名空间
namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

// 使用 XNN 库的命名空间
using namespace fb_xnnpack;

// 实现 XNNSerializer 类的 serializeAddNode 方法
void XNNSerializer::serializeAddNode(
    uint32_t input1_id,
    uint32_t input2_id,
    uint32_t output_id,
    uint32_t flags) {
  // 创建 XNNAdd 节点
  const auto addNode =
      CreateXNNAdd(_builder, input1_id, input2_id, output_id, flags);
  // 创建 FlatBuffer 节点
  const auto flatbufferNode =
      CreateXNode(_builder, XNodeUnion::XNNAdd, addNode.Union());
  // 将节点添加到 _nodes 向量
  _nodes.push_back(flatbufferNode);
}

// 实现 XNNSerializer 类的 serializeData 方法
size_t XNNSerializer::serializeData(const uint8_t* data_ptr, size_t num_bytes) {
  // 初始化常量缓冲区索引
  size_t constant_buffer_idx = 0;
  // 处理包含数据的张量 _values
  if (data_ptr != nullptr) {
    // 创建 FlatBuffer 的字节向量存储张量数据
    auto storage = _builder.CreateVector(data_ptr, num_bytes);
    // 将其放入常量缓冲区
    constant_buffer_idx = _constantBuffer.size();
    _constantBuffer.emplace_back(CreateBuffer(_builder, storage));
    // 记录缓冲区大小到 _bufferSizes
    _bufferSizes.push_back(num_bytes);
    // 断言确保 _bufferSizes 和 _constantBuffer 大小相等
    assert(_bufferSizes.size() == _constantBuffer.size());
  }
  return constant_buffer_idx;
}

// 实现 XNNSerializer 类的 serializeTensorValue 方法
void XNNSerializer::serializeTensorValue(
    uint32_t xnn_datatype,
    size_t num_dims,
    std::vector<size_t> dims,
    size_t data_buffer_idx,
    uint32_t external_id,
    uint32_t flags,
    uint32_t id_out) {
  // 序列化张量维度信息
  std::vector<uint32_t> serialized_dims;
  serialized_dims.reserve(dims.size());
  for (auto dim : dims) {
    serialized_dims.push_back(static_cast<uint32_t>(dim));
  }

  // 创建 XNNTensorValueDirect
  const auto tensorValue = CreateXNNTensorValueDirect(
      _builder,
      XNNDatatype(xnn_datatype),
      num_dims,
      &serialized_dims,
      data_buffer_idx,
      external_id,
      flags,
      id_out);

  // 创建 FlatBuffer 的 XValueUnion 节点
  const auto flatbufferValue =
      CreateXValue(_builder, XValueUnion::XNNTensorValue, tensorValue.Union());
  // 将值添加到 _values 向量
  _values.push_back(flatbufferValue);
}

// 实现 XNNSerializer 类的 finishAndSerialize 方法
std::string XNNSerializer::finishAndSerialize(
    std::vector<uint32_t> input_ids,
    std::vector<uint32_t> output_ids,
    size_t num_extern_ids) {
  // 创建 XNNGraphDirect
  auto xnnGraph = CreateXNNGraphDirect(
      _builder,
      _version_sha1,
      &_nodes,
      &_values,
      num_extern_ids,
      &input_ids,
      &output_ids,
      &_constantBuffer,
      &_bufferSizes);

  // 完成 FlatBuffer 构建
  _builder.Finish(xnnGraph);

  // 使用 stringstream 将 FlatBuffer 内容转换为字符串
  std::stringstream ss;
  ss.write(
      reinterpret_cast<char*>(_builder.GetBufferPointer()), _builder.GetSize());

  return ss.str();
}

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
```