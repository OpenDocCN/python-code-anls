# `.\pytorch\torch\csrc\lazy\ts_backend\ops\device_data.cpp`

```
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_builder.h>

#include <sstream>

namespace torch {
namespace lazy {

// DeviceData 类的构造函数，接受一个 BackendData 共享指针作为参数
DeviceData::DeviceData(std::shared_ptr<BackendData> data)
    : TsNode(
          ClassOpKind(),        // 使用 ClassOpKind() 作为 TsNode 的操作类型
          data->shape(),        // 使用 data 的形状作为 TsNode 的形状
          /*num_outputs=*/1,    // 设置输出数量为 1
          /*hash_seed=*/static_cast<uint32_t>(101)),  // 使用静态转型的哈希种子为 101
      data_(std::move(data)) {}  // 移动构造函数初始化 data_

// 返回对象的字符串表示形式，包含 TsNode 的字符串和数据的设备信息
std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << TsNode::ToString() << ", device=" << data_->device();
  return ss.str();  // 返回串流的字符串表示
}

// 将 Node 指针向下转型为 DeviceData 指针的静态方法
const DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node);  // 使用 NodeCast 进行转型
}

// 创建 DeviceData 节点的静态方法，接受一个 BackendData 共享指针作为参数
NodePtr DeviceData::Create(std::shared_ptr<BackendData> data) {
  // 重用已有的 DeviceData 节点或者创建新节点
  NodePtr node = ReuseOrMakeNode<DeviceData>(data);
  // ReuseOrMakeNode 可能返回一个已重用的节点，其形状相同，
  // 然而，我们需要用新数据替换旧数据 data_
  DeviceData* device_data = static_cast<DeviceData*>(node.get());
  device_data->SetData(data);  // 设置新的数据到 device_data
  return node;  // 返回创建或重用的节点
}

} // namespace lazy
} // namespace torch
```