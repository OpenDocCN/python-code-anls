# `.\pytorch\torch\csrc\lazy\core\ir_builder.h`

```py
#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/trie.h>
#include <vector>

// This file is part of the backend interface. So, ops shouldn't be added or
// removed without due process The exception to this being the view ops which
// will be removed soon pending functionalization

namespace torch {
namespace lazy {

// Template function to reuse an existing IR node if caching is enabled
template <typename T, typename... Args>
NodePtr ReuseNode(Args&&... args) {
  if (FLAGS_torch_lazy_reuse_ir) {
    // Lookup and return an existing node from trie cache if available
    return LookupNodeFromTrieCache<T>(std::forward<Args>(args)...);
  }
  return nullptr;
}

// Function to cache an IR node into trie cache if caching is enabled
static inline void CacheNode(NodePtr node) {
  if (FLAGS_torch_lazy_reuse_ir) {
    // Insert the node into trie cache
    TrieCache::Get()->Insert(std::move(node));
  }
}

// Template function to create a new IR node
template <typename T, typename... Args>
NodePtr MakeNode(Args&&... args) {
  // Create and return a new node of type T with given arguments
  return std::make_shared<T>(std::forward<Args>(args)...);
}

// Template function to reuse an existing IR node or create a new one
template <typename T, typename... Args>
NodePtr ReuseOrMakeNode(Args&&... args) {
  // Attempt to reuse an existing node
  NodePtr node = ReuseNode<T>(std::forward<Args>(args)...);
  if (!node) {
    // If no existing node found, create a new one and cache it
    node = MakeNode<T>(std::forward<Args>(args)...);
    CacheNode(node);
  }
  return node;
}

// Interface for building IR nodes
struct IrBuilder {
  // Virtual function to create a device data node
  virtual NodePtr MakeDeviceData(const std::shared_ptr<BackendData>& data) const = 0;
  // Virtual function to create a scalar node
  virtual NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type) const = 0;
  // Virtual function to create an expand node
  virtual NodePtr MakeExpand(const Value& input0, const std::vector<int64_t>& size, const bool& is_scalar_expand) const = 0;
  // Virtual function to create a cast node
  virtual NodePtr MakeCast(const Value& input0, const at::ScalarType& dtype, const std::optional<at::ScalarType>& stype = c10::nullopt) const = 0;
  // Virtual function to create a tensor list node
  virtual NodePtr MakeTensorList(const OpList& inputs) const = 0;
  // Virtual function to create a generic node
  virtual NodePtr MakeGeneric(const OpKind& op, const OpList& operands, const Shape& shape, const size_t& num_outputs = 1, const hash_t& hash_seed = static_cast<uint32_t>(0x5a2d296e9)) const = 0;

  // Virtual functions for dynamic IR nodes
  virtual NodePtr MakeSizeNode(const Value& input, size_t dim) const = 0;
  virtual NodePtr MakeSizeAdd(const Value& a, const Value& b) const = 0;
  virtual NodePtr MakeSizeMul(const Value& a, const Value& b) const = 0;
  virtual NodePtr MakeSizeDiv(const Value& a, const Value& b) const = 0;

  // Virtual destructor for the interface
  virtual ~IrBuilder() = default;
};

// Function to create a device data node using the global IR builder
static inline NodePtr MakeDeviceData(const std::shared_ptr<BackendData>& data) {
  return getIrBuilder()->MakeDeviceData(data);
}

// Function to create a scalar node using the global IR builder
static inline NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type) {
  return getIrBuilder()->MakeScalar(value, type);
}

// Function to create an expand node using the global IR builder
static inline NodePtr MakeExpand(const Value& input0, const std::vector<int64_t>& size,
                                 const bool& is_scalar_expand) {
  return getIrBuilder()->MakeExpand(input0, size, is_scalar_expand);
}
    // 调用当前对象的 getIrBuilder 方法获取 IR 构建器，并调用其 MakeExpand 方法
    // 用于创建并返回一个扩展操作，接受三个参数：input0（输入数据）、size（扩展的尺寸）、is_scalar_expand（是否标量扩展）
    return getIrBuilder()->MakeExpand(input0, size, is_scalar_expand);
// 结束 lazy 命名空间
namespace lazy {

// 结束 torch 命名空间
namespace torch {

// 根据输入值、目标数据类型和可选的源数据类型，创建一个类型转换节点
static inline NodePtr MakeCast(
    const Value& input0,
    const at::ScalarType& dtype,
    const std::optional<at::ScalarType>& stype = c10::nullopt) {
  return getIrBuilder()->MakeCast(input0, dtype, stype);
}

// 根据输入的操作列表，创建一个张量列表节点
static inline NodePtr MakeTensorList(const OpList& inputs) {
  return getIrBuilder()->MakeTensorList(inputs);
}

// 根据操作种类、操作数列表、形状和输出数量，创建一个通用节点
static inline NodePtr MakeGeneric(
    const OpKind& op,
    const OpList& operands,
    const Shape& shape,
    const size_t& num_outputs = 1,
    const hash_t& hash_seed = static_cast<uint32_t>(0x5a2d296e9)) {
  return getIrBuilder()->MakeGeneric(
      op, operands, shape, num_outputs, hash_seed);
}

// 创建一个表示输入值大小的节点
static inline NodePtr MakeSizeNode(const Value& input, size_t dim) {
  return getIrBuilder()->MakeSizeNode(input, dim);
}

// 创建一个表示两个输入值大小相加的节点
static inline NodePtr MakeSizeAdd(const Value& a, const Value& b) {
  return getIrBuilder()->MakeSizeAdd(a, b);
}

// 创建一个表示两个输入值大小相乘的节点
static inline NodePtr MakeSizeMul(const Value& a, const Value& b) {
  return getIrBuilder()->MakeSizeAdd(a, b);  // 此处应为 MakeSizeMul
}

// 创建一个表示两个输入值大小相除的节点
static inline NodePtr MakeSizeDiv(const Value& a, const Value& b) {
  return getIrBuilder()->MakeSizeDiv(a, b);
}

// 获取符号整数值的包装函数
inline Value GetSymIntValue(c10::SymInt a) {
  if (auto ma = a.maybe_as_int()) {
    // 如果可转换为整数，则创建一个标量值节点
    return Value(MakeScalar(*ma, at::kLong), 0);
  } else {
    // 否则，创建一个与符号节点相关的值节点
    return Value(
        dynamic_cast<torch::lazy::SymNodeImpl*>(a.toSymNodeImplUnowned())
            ->node_,
        0);
  }
}

// TODO: 此函数应该返回一个值节点
// 获取符号整数数组引用的值
inline std::vector<int64_t> GetSymIntArrayRefValue(c10::SymIntArrayRef arr) {
  std::vector<int64_t> r;
  for (const auto& a : arr) {
    // 遍历数组并获取每个元素的整数值，添加到结果向量中
    r.emplace_back(a.guard_int(__FILE__, __LINE__));
  }
  return r;
}

} // namespace torch
} // namespace lazy
```