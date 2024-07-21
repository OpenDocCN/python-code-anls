# `.\pytorch\torch\csrc\lazy\ts_backend\ops\device_data.h`

```py
#pragma once
// 包含 Torch Lazy 模块的相关头文件

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// DeviceData 类继承自 TsNode 类
class TORCH_API DeviceData : public TsNode {
 public:
  // 静态方法，返回类的操作种类 OpKind
  static OpKind ClassOpKind() {
    return ltc_device_data;
  }

  // 显式构造函数，接受一个 BackendData 类型的 shared_ptr 参数
  explicit DeviceData(std::shared_ptr<BackendData> data);

  // 判断当前 DeviceData 节点是否可以被重用
  // 条件是传入的数据的形状与当前数据的形状相同
  bool CanBeReused(std::shared_ptr<BackendData> data) const {
    return data_->shape() == data->shape();
  }

  // 返回当前对象的字符串表示形式
  std::string ToString() const override;

  // 返回当前 DeviceData 对象持有的数据的 shared_ptr 引用
  const std::shared_ptr<BackendData>& data() const {
    return data_;
  }

  // 设置当前 DeviceData 对象持有的数据
  void SetData(std::shared_ptr<BackendData> data) {
    data_ = data;
  }

  // 将 Node 类型转换为 DeviceData 类型的静态方法
  static const DeviceData* Cast(const Node* node);

  // 创建 DeviceData 节点的静态方法，用于重用 IR 节点
  // 而不是直接调用构造函数
  static NodePtr Create(std::shared_ptr<BackendData> data);

  // 覆盖基类的 Lower 方法，用于降低节点，生成 TSOpVector 对象
  TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;

 private:
  // 私有成员变量，持有 BackendData 类型的 shared_ptr 数据
  std::shared_ptr<BackendData> data_;
};

} // namespace lazy
} // namespace torch
```