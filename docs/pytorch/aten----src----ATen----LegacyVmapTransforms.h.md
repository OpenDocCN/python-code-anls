# `.\pytorch\aten\src\ATen\LegacyVmapTransforms.h`

```py
#pragma once

#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/core/IListRef.h>

namespace at {

// This file contains abstractions used for transforming *logical* vmap
// arguments into *physical* arguments. (Keep reading for definitions of these
// terms).

// NOTE: [Logical vs physical args]
// Consider the following vmap.
//   vmap(vmap(func, in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3, 4))
// This would produce a BatchedTensor wrapping a Tensor of size [2, 3, 4],
// with batch dims 0 and 2:
//   BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])
//
// We say the *logical* view of the tensor has size [3] -- tensors inside
// `func` appear to have size [3].
// However, the *physical* underlying tensor (the one passed to vmap) has size
// [2, 3, 4].
//
// This notion of logical vs physical also extends to non-tensor arguments.
// Consider the previous tensor; let's assume the user called
// `torch.sum(tensor, dim=0)` inside of `func`. Then the logical
// dimension they are reducing over is dim 0 but the physical dim is dim 1
// (the first non-batch dimension)

// Forward declared; see NOTE: [What is a VmapPhysicalView?]
struct VmapPhysicalView;

// Most PyTorch operators take 4 or fewer inputs.
constexpr int64_t kVmapTransformStaticInputSize = 4;
using VmapPhysicalViewVec =
    SmallVector<VmapPhysicalView, kVmapTransformStaticInputSize>;

// Pytorch generally advertises good performance for <= 5 dims.
// (see ATen/core/DimVector.h). We add a few extra dims (~3) for vmap
// dimensions to get 8. Adjust this number as necessary
constexpr int64_t kVmapStaticDimVecSize = 8;
using VmapDimVector = SmallVector<int64_t, kVmapStaticDimVecSize>;
using VmapSymDimVector = SmallVector<c10::SymInt, kVmapStaticDimVecSize>;

// NOTE: [What is an VmapTransform?]
// An *VmapTransform* converts logical views of tensors to physical views.
//
// Batching rules use VmapTransforms to convert logical arguments to
// physical arguments, then call one or more at:: operator that handles the
// physical arguments, and then converts the physical result back to a logical
// argument.

// VmapTransform for operators that take tensors with multiple batch dims.
// Given one or more logical views on Tensors, `logicalToPhysical`
// permutes all of the batch dims to the front of the tensor, aligns
// and expands the batch dims to match each other (according to their `level`),
// and returns a VmapPhysicalView on the tensor(s).
struct TORCH_API MultiBatchVmapTransform {
  // Converts a single logical tensor into its corresponding physical view.
  static VmapPhysicalView logicalToPhysical(const Tensor& logical_tensor);
  
  // Converts a list of logical tensors into their corresponding physical views.
  static VmapPhysicalViewVec logicalToPhysical(ITensorListRef logical_tensors);
};

// VmapTransform for operators that broadcast all inputs.
// Given some logical views on Tensors, `logicalToPhysical`:
// - permutes all of the batch dims to the front of the tensors
// - aligns all the batch dims to the collective levels of all of the tensors.
// If a tensor does not have a batch dim for a vmap level, then it receives
// a size-one dimension for said level.
// - aligns the non-batch dims to have the same dimensionality, adding extra
//   size-1 dimensions in between the batch dimensions and the non-batch
//   dimensions so that the batch dimensions are lined up from the right.
//
// For example: given inputs of size (B, 2) and (B, 3, 2) where B is the batch
// dimension, BroadcastingVmapTransform returns VmapPhysicalViews that wrap
// tensors of size (B, 1, 2) and (B, 3, 2).
//
// Given inputs of size (B, 2) and (2,), BroadcastingVmapTransform returns
// VmapPhysicalViews wrapping tensors of size (B, 2) and (1, 2). We don't
// actually *need* to return a tensor of size (1, 2) for the second tensor
// because the broadcasting operation takes care of that for us, but we do
// it anyways to keep things simple.
struct TORCH_API BroadcastingVmapTransform {
  // Static method that converts logical tensors to physical views adjusted for vmap batch dimensions.
  static VmapPhysicalViewVec logicalToPhysical(TensorList logical_tensors);
};

// Forward declared, if you're reading this file head to toe, don't worry about
// it yet.
struct VmapPhysicalToLogicalMap;

// NOTE: [What is a VmapPhysicalView?]
// VmapPhysicalView represents a physical view on a Tensor.
//
// One can use it to further convert logical dimension indices, logical shapes,
// and more to their physical variants, or convert a new (physical) tensor into
// a logical BatchedTensor. (TODO(rzou): some of these are not yet implemented).
//
// VmapPhysicalView stores a physical tensor with all of its batch dimensions at
// the front and some levels that correspond to said batch dimensions.
//
// The levels bitset specifies which vmap levels correspond to the batch
// dimensions at the front of the tensor. In particular, the number of set bits
// corresponds to the number of batch dimensions on `tensor` and the rightmost
// bit of `levels` specifies the maximum number of nested vmaps we are in at
// this point in time.
// For example, given:
//   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5, 6), levels={1, 3})
//
// Rightmost bit of `levels` is 3 indicating the number of nested vmaps less
// than or equal to 3.
//   bitset: 010100
//              ^
//              |
//   levels: 012345
struct TORCH_API VmapPhysicalView {
  // Constructor initializing a VmapPhysicalView object with a tensor and vmap level information.
  VmapPhysicalView(Tensor&& tensor, std::bitset<kVmapNumLevels> levels)
      : levels_(levels), tensor_(std::move(tensor)) {
    TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor_));
  }

  // Accessor to get mutable reference to the stored tensor.
  Tensor& tensor() {
    return tensor_;
  }

  // Accessor to get const reference to the stored tensor.
  const Tensor& tensor() const {
    return tensor_;
  }
    // 返回成员变量 tensor_
    return tensor_;
    }
    
    // 将逻辑维度索引映射到物理维度索引，并处理维度包装。
    //
    // 例如，假设：
    //   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5), levels={1, 3})
    //
    // 那么 physical_view.getPhysicalDims({0, 1}) 返回 {2, 3}。
    // 这是因为 levels 的大小告诉我们 tensor_ 的前两个维度是批量维度，
    // 所以逻辑维度 `n` 实际上是物理维度 `n + 2`。
    VmapDimVector getPhysicalDims(OptionalIntArrayRef logical_dims) const;
    
    // 返回一个 VmapPhysicalToLogicalMap 对象。这可用于将物理张量映射到新的逻辑张量（BatchedTensor）。
    VmapPhysicalToLogicalMap getPhysicalToLogicalMap() const;
    
    // 通过将批量大小预置到逻辑形状前面，将逻辑形状映射到物理形状。
    VmapDimVector getPhysicalShape(IntArrayRef logical_shape) const;
    
    // 返回批量维度的数量。
    int64_t numBatchDims() const;
    
    private:
    // 返回逻辑维度的数量。
    int64_t numLogicalDims() const;
    
    // 表示哪些维度是批量维度的位集合。
    std::bitset<kVmapNumLevels> levels_;
    // 内部存储的张量对象。
    Tensor tensor_;
};

// Convenience struct used for mapping a physical tensor (a non-BatchedTensor)
// to a logical one (BatchedTensor). It holds some levels that are used to do
// the mapping and assumes that the batch dimensions in the physical tensor all
// occur at the front of the tensor.
// 用于将物理张量（非批量张量）映射到逻辑张量（批量张量）的便捷结构体。它包含用于进行映射的一些级别，并假设物理张量中的批量维度都位于张量的前部。

struct TORCH_API VmapPhysicalToLogicalMap {
  VmapPhysicalToLogicalMap(std::bitset<kVmapNumLevels> levels)
      : levels_(levels) {}

  // Maps a physical tensor to a new logical tensor (BatchedTensor).
  // Assumes that all of the "batch dimensions" are at the front
  // of the physical tensor. For example, given:
  // - x = rank-4 Tensor with size 2, 3, 5, 7
  // - levels = (2, 4)
  // Returns:
  // - BatchedTensor(x, bdims=[(dim=0,lvl=2), (dim=1, lvl=4)])
  // 将物理张量映射到新的逻辑张量（批量张量）。假设所有的“批量维度”都位于物理张量的前部。例如，给定：
  // - x = 大小为 2, 3, 5, 7 的四阶张量
  // - levels = (2, 4)
  // 返回：
  // - BatchedTensor(x, bdims=[(dim=0,lvl=2), (dim=1, lvl=4)])
  Tensor apply(const Tensor& physical_tensor) const;

  // Given a vector of physical tensors,
  // 1. maps each tensor to a new logical tensor. Assumes that all of the
  //    "batch dimensions" are at the front of the physical tensors.
  // 2. stores the new logical tensors back into the passed-in vector. This is
  //    to avoid additional dynamic allocations.
  // 给定一个物理张量的向量，
  // 1. 将每个张量映射到一个新的逻辑张量。假设所有的“批量维度”都位于物理张量的前部。
  // 2. 将新的逻辑张量存储回传入的向量中。这样做是为了避免额外的动态分配。
  void applyInplace(std::vector<Tensor>& physical_tensors) const;

  std::bitset<kVmapNumLevels> levels_;
};

} // namespace at


这段代码定义了一个结构体 `VmapPhysicalToLogicalMap`，用于将物理张量映射到逻辑张量，假设所有的批量维度都位于张量的前部。
```