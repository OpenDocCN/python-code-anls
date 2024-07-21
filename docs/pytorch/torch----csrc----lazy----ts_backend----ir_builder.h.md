# `.\pytorch\torch\csrc\lazy\ts_backend\ir_builder.h`

```py
#pragma once


// 在 C++ 中，#pragma once 指令确保头文件只被包含一次，防止重复包含
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <torch/csrc/lazy/generated/LazyNonNativeIr.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/generic.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// TorchScriptIrBuilder 结构体继承自 IrBuilder，用于构建 TorchScript IR
struct TorchScriptIrBuilder : IrBuilder {
  // 创建设备数据节点，接受 BackendData 共享指针作为参数，并返回节点指针
  NodePtr MakeDeviceData(
      const std::shared_ptr<BackendData>& data) const override {
    return DeviceData::Create(data);
  }
  // 创建标量节点，暂时未被 ts_backend 使用。如有需要后续可以启用复用标量节点
  NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type)
      const override {
    return MakeNode<Scalar>(value, type);
  }
  // 创建扩展节点，接受输入值、大小和是否标量扩展作为参数，并返回节点指针
  NodePtr MakeExpand(
      const Value& input0,
      const std::vector<int64_t>& size,
      const bool& is_scalar_expand) const override {
    return ReuseOrMakeNode<Expand>(input0, size, is_scalar_expand);
  }
  // 创建类型转换节点，接受输入值、目标数据类型和可选的源数据类型作为参数，并返回节点指针
  NodePtr MakeCast(
      const Value& input0,
      const at::ScalarType& dtype,
      const std::optional<at::ScalarType>& stype =
          c10::nullopt) const override {
    return ReuseOrMakeNode<Cast>(input0, dtype, stype);
  }
  // 创建张量列表节点，接受操作列表作为参数，并返回节点指针
  NodePtr MakeTensorList(const OpList& inputs) const override {
    return ReuseOrMakeNode<TensorList>(inputs);
  }
  // 创建通用节点，接受操作类型、操作数列表、形状、输出数量和哈希种子作为参数，并返回节点指针
  NodePtr MakeGeneric(
      const OpKind& op,
      const OpList& operands,
      const Shape& shape,
      const size_t& num_outputs = 1,
      const hash_t& hash_seed =
          static_cast<uint32_t>(0x5a2d296e9)) const override {
    return MakeNode<Generic>(op, operands, shape, num_outputs, hash_seed);
  }

  // 动态 IR 节点
  // TODO: 验证 IR 节点复用在动态形状操作中是否有效
  // 创建尺寸节点，接受输入值和维度作为参数，并返回节点指针
  NodePtr MakeSizeNode(const Value& input, size_t dim) const override {
    return MakeNode<SizeNode>(input, dim);
  }
  // 创建尺寸加法节点，接受两个输入值作为参数，并返回节点指针
  NodePtr MakeSizeAdd(const Value& a, const Value& b) const override {
    return MakeNode<SizeAdd>(a, b);
  }
  // 创建尺寸乘法节点，接受两个输入值作为参数，并返回节点指针
  NodePtr MakeSizeMul(const Value& a, const Value& b) const override {
    return MakeNode<SizeMul>(a, b);
  }
  // 创建尺寸除法节点，接受两个输入值作为参数，并返回节点指针
  NodePtr MakeSizeDiv(const Value& a, const Value& b) const override {
    return MakeNode<SizeDiv>(a, b);
  }
};

} // namespace lazy
} // namespace torch
```