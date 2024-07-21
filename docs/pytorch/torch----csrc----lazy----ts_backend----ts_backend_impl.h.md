# `.\pytorch\torch\csrc\lazy\ts_backend\ts_backend_impl.h`

```
#pragma once

#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch {
namespace lazy {

// TSData 类继承自 BackendData 类，表示 TorchScript 后端数据
class TORCH_API TSData : public torch::lazy::BackendData {
 public:
  // 构造函数，接受标量和设备参数，初始化 scalar 成员和 BackendData 基类
  TSData(const at::Scalar& scalar, const torch::lazy::BackendDevice& device)
      : torch::lazy::BackendData(device, torch::lazy::Shape(scalar.type(), {})),
        scalar(scalar) {}

  // 构造函数，接受张量、形状和设备参数，初始化 data_ 成员和 BackendData 基类
  TSData(
      const at::Tensor& data,
      const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device)
      : torch::lazy::BackendData(device, shape), data_(data) {}

  // 构造函数，接受形状和设备参数，初始化 BackendData 基类
  TSData(
      const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device)
      : torch::lazy::BackendData(device, shape) {}

  // 实现 BackendData 的虚函数，返回对象的句柄
  Handle GetHandle() override {
    return reinterpret_cast<int64_t>(this);
  }

  // 实现 BackendData 的虚函数，从另一个 BackendData 对象中复制数据
  void Assign(const torch::lazy::BackendData& data) override {
    data_ = static_cast<const TSData&>(data).data_;
  }

  // 实现 BackendData 的虚函数，检查是否有有效数据
  bool HasValue() const override {
    return data_.defined();
  }

  // 返回数据张量
  at::Tensor data() {
    return data_;
  }

  // 可选的标量成员
  std::optional<at::Scalar> scalar;

 private:
  // 数据张量成员
  at::Tensor data_;
};

// 返回 TorchScript 后端实现接口的指针
TORCH_API torch::lazy::BackendImplInterface* GetTSBackendImpl();

// 初始化 TorchScript 后端
TORCH_API void InitTorchScriptBackend();

} // namespace lazy
} // namespace torch
```