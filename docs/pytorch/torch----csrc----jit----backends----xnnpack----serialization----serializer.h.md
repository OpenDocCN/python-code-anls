# `.\pytorch\torch\csrc\jit\backends\xnnpack\serialization\serializer.h`

```py
// 包含所需的头文件
#include <torch/csrc/jit/backends/xnnpack/serialization/schema_generated.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 命名空间声明：torch -> jit -> xnnpack -> delegate
namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

// 使用 flatbuffers_fbsource 命名空间中的类型和函数
using namespace fb_xnnpack;

// XNNSerializer 类定义
class XNNSerializer {
 public:
  // 默认构造函数，初始化缓冲区大小为 1024
  XNNSerializer() : XNNSerializer(1024) {}

  // 带参构造函数，初始化私有成员变量
  explicit XNNSerializer(size_t bufferSize)
      : _builder(bufferSize),  // 初始化 FlatBufferBuilder
        _nodes(),               // 初始化节点序列
        _values(),              // 初始化值序列
        _constantBuffer({CreateBuffer(
            _builder,
            {})}),               // 初始化常量缓冲区
        _bufferSizes({0}) {}    // 初始化缓冲区大小数组

  // 序列化节点：添加一个 add 节点的序列化方法
  void serializeAddNode(
      uint32_t input1_id,
      uint32_t input2_id,
      uint32_t output_id,
      uint32_t flags);

  // 序列化值：添加一个张量值的序列化方法
  void serializeTensorValue(
      uint32_t xnn_datatype,
      size_t num_dims,
      std::vector<size_t> dims,
      size_t buffer_data_idx,
      uint32_t external_id,
      uint32_t flags,
      uint32_t id_out);

  // 完成并序列化 XNN 图，返回序列化后的数据
  std::string finishAndSerialize(
      std::vector<uint32_t> input_ids,
      std::vector<uint32_t> output_ids,
      size_t num_extern_ids);

  // 序列化数据：将数据序列化并返回其索引
  size_t serializeData(const uint8_t* data_ptr, size_t num_bytes);

 private:
  // 正在序列化的 XNNPack 版本的 SHA-1 标识
  const char* _version_sha1 = "ae108ef49aa5623b896fc93d4298c49d1750d9ba";

  // FlatBuffer 构建器对象，用于创建 FlatBuffer 对象
  flatbuffers_fbsource::FlatBufferBuilder _builder;

  // 序列化后的节点对象的向量
  std::vector<flatbuffers_fbsource::Offset<XNode>> _nodes;

  // 序列化后的值对象的向量
  std::vector<flatbuffers_fbsource::Offset<XValue>> _values;

  // 序列化后的常量缓冲区对象的向量
  std::vector<flatbuffers_fbsource::Offset<Buffer>> _constantBuffer;

  // 缓冲区大小的向量，用于跟踪每个缓冲区的大小
  std::vector<uint32_t> _bufferSizes;
};

}  // namespace delegate
}  // namespace xnnpack
}  // namespace jit
}  // namespace torch
```