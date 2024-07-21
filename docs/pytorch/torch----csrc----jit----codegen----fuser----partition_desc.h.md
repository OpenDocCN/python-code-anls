# `.\pytorch\torch\csrc\jit\codegen\fuser\partition_desc.h`

```
// 防止头文件被多次包含
#pragma once

// 引入异常处理工具头文件
#include <c10/util/Exception.h>
// 引入 Torch 导出工具头文件
#include <torch/csrc/Export.h>
// 引入张量描述头文件
#include <torch/csrc/jit/codegen/fuser/tensor_desc.h>

// 引入标准整数类型头文件
#include <cstdint>
// 引入智能指针头文件
#include <memory>
// 引入向量容器头文件
#include <vector>

// Torch 命名空间
namespace torch {
// Torch JIT 命名空间
namespace jit {
// Torch JIT Fuser 命名空间
namespace fuser {

// 描述将输入张量分块为子张量或将输出张量连接成子张量的数据结构
// 注意：默认构造用于不参与分块或连接操作的张量
struct TORCH_API PartitionDesc {
  // 默认构造函数，初始化子张量数量为1，分块/连接维度为0
  PartitionDesc() : nSubTensors_{1}, dim_{0} {}

  // 构造函数，根据给定的张量描述 _desc，子张量数量 _nSubTensors 和分块/连接维度 _dim 初始化对象
  PartitionDesc(const TensorDesc& _desc, size_t _nSubTensors, size_t _dim)
      : nSubTensors_{_nSubTensors}, dim_{_dim} {
    // 断言子张量数量大于1
    AT_ASSERT(nSubTensors_ > 1);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 复制张量的连续性信息
    std::vector<bool> cont = _desc.contiguity;
    // 如果分块/连接维度大于0
    if (dim_ > 0) {
      // 当我们缩小输出的连接/输入的分块时
      // 我们使维度[dim]的尺寸变小，但保持维度[dim]的步长不变，
      // 这意味着：步长[dim - 1] != 步长[dim]*尺寸[dim]
      // 因此，维度[dim - 1]不再是连续的
      cont[dim_ - 1] = false;
    }
    // NOLINTNEXTLINE(modernize-make-shared)
    // 为子张量描述分配新的智能指针对象
    subTensorDesc_.reset(new TensorDesc(_desc.scalar_type, cont));
  }

  // 检查对象是否为无操作（即子张量数量为1）
  bool isNoop() const {
    return (nSubTensors_ == 1);
  }

  // 返回子张量数量
  size_t nSubTensors() const {
    return nSubTensors_;
  }

  // 返回分块/连接维度
  size_t dim() const {
    return dim_;
  }

  // 返回子张量描述的智能指针
  std::shared_ptr<TensorDesc> subTensorDesc() {
    return subTensorDesc_;
  }

  // 返回子张量描述的常量智能指针
  const std::shared_ptr<TensorDesc> subTensorDesc() const {
    return subTensorDesc_;
  }

 private:
  size_t nSubTensors_; // 子张量的数量，对于不进行分块/连接操作的张量为1
  size_t dim_; // 分块/连接操作发生的维度
  std::shared_ptr<TensorDesc>
      subTensorDesc_; // 子张量的描述对象，如果存在的话
};

} // namespace fuser
} // namespace jit
} // namespace torch
```