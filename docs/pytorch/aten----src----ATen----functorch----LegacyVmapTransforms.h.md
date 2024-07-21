# `.\pytorch\aten\src\ATen\functorch\LegacyVmapTransforms.h`

```
// 版权声明和许可信息，指出该源代码受BSD风格许可证保护，许可证文件位于根目录的LICENSE文件中
// 声明本文件为functorch库中已废弃的批处理规则API，建议使用新的批处理规则API（参见writing_batch_rules.md）

#pragma once

#include <ATen/functorch/Macros.h>
#include <ATen/functorch/BatchedTensorImpl.h>

namespace at::functorch {

// 该文件包含用于将“逻辑”vmap参数转换为“物理”参数的抽象表示。请继续阅读以了解这些术语的定义。

// NOTE: [Logical vs physical args]
// 考虑以下vmap示例：
//   vmap(vmap(func, in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3, 4))
// 这将产生一个BatchedTensor，封装了一个大小为[2, 3, 4]的Tensor，
// 其中的批处理维度为0和2：
//   BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])
//
// 我们说这个张量的“逻辑”视图大小为[3] -- 在func函数内部，张量看起来是大小为[3]。
// 然而，“物理”底层张量（传递给vmap的那个）的大小是[2, 3, 4]。
//
// 这种逻辑与物理的概念也适用于非张量参数。考虑前面的张量；假设用户在func内部调用了
// `torch.sum(tensor, dim=0)`。那么他们正在减少的逻辑维度是dim 0，但物理维度是dim 1
// （第一个非批处理维度）

// 前向声明；请参阅 NOTE: [What is a VmapPhysicalView?]
struct VmapPhysicalView;

// 大多数PyTorch操作符最多接受4个输入。
constexpr int64_t kVmapTransformStaticInputSize = 4;
using VmapPhysicalViewVec = SmallVector<VmapPhysicalView, kVmapTransformStaticInputSize>;

// PyTorch通常推荐对<= 5维张量性能良好。
// （参见 ATen/core/DimVector.h）。我们为vmap维度添加了几个额外的维度（约3个），
// 得到了8。根据需要调整此数值。
constexpr int64_t kVmapStaticDimVecSize = 8;
using VmapDimVector = SmallVector<int64_t, kVmapStaticDimVecSize>;
using VmapSymDimVector = SmallVector<c10::SymInt, kVmapStaticDimVecSize>;

// NOTE: [What is an VmapTransform?]
// VmapTransform用于将张量的逻辑视图转换为物理视图。
//
// 批处理规则使用VmapTransform将逻辑参数转换为物理参数，然后调用一个或多个处理物理参数的at::操作符，
// 然后将物理结果转换回逻辑参数。

// 用于接受多个批处理维度的张量的VmapTransform。
// 给定一个或多个Tensor的逻辑视图，logicalToPhysical
// 将所有批处理维度置于张量的前面，根据它们的“级别”对齐和扩展批处理维度，并返回张量的VmapPhysicalView。
// 定义一个结构体 MultiBatchVmapTransform，这个结构体提供了一些静态方法用于多批次 Vmap 变换。
struct TORCH_API MultiBatchVmapTransform {
  // 声明静态方法 logicalToPhysical，接收一个 Tensor 参数，返回一个 VmapPhysicalView 对象。
  static VmapPhysicalView logicalToPhysical(const Tensor& logical_tensor);
  // 声明静态方法 logicalToPhysical，接收一个 ITensorListRef 参数，返回一个 VmapPhysicalViewVec 对象。
  static VmapPhysicalViewVec logicalToPhysical(ITensorListRef logical_tensors);
};

// 声明一个结构体 BroadcastingVmapTransform，用于支持广播操作的 Vmap 变换。
// 它对于给定的逻辑视图 Tensors，`logicalToPhysical` 方法：
// - 将所有批次维度排列到张量的前面
// - 将所有批次维度对齐到所有张量的集体级别
// - 如果张量在 Vmap 级别没有批次维度，则为该级别添加一个大小为一的维度
// - 将非批次维度对齐为相同的维度，添加额外的大小为一的维度，使批次维度从右侧对齐
struct TORCH_API BroadcastingVmapTransform {
  // 声明静态方法 logicalToPhysical，接收一个 TensorList 参数，返回一个 VmapPhysicalViewVec 对象。
  static VmapPhysicalViewVec logicalToPhysical(TensorList logical_tensors);
};

// 提前声明，如果你正在逐行阅读这个文件，暂时可以忽略它。
struct VmapPhysicalToLogicalMap;

// 注意：[什么是 VmapPhysicalView?]
// VmapPhysicalView 表示张量的物理视图。
//
// 可以使用它进一步将逻辑维度索引、逻辑形状等转换为它们的物理变体，或将新的（物理）张量转换为逻辑 BatchedTensor。
//
// VmapPhysicalView 存储一个物理张量，其中所有的批次维度在前面，并且一些级别对应于这些批次维度。
//
// levels 位集指定哪些 Vmap 级别对应于张量前面的批次维度。
// 特别地，设置的位数对应于 `tensor` 上的批次维度数量，levels 的最右侧位指定此时我们处于的最大嵌套 Vmap 数量。
// 例如，给定：
//   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5, 6), levels={1, 3})
//
// levels 的最右侧位是 3，指示嵌套 Vmap 的数量不超过 3。
//   位集：010100
//             ^
//             |
//   levels：012345
struct TORCH_API VmapPhysicalView {
  // 构造函数，接收一个 Tensor&& 参数和一个 std::bitset<kVmapNumLevels> 参数。
  VmapPhysicalView(Tensor&& tensor, std::bitset<kVmapNumLevels> levels)
      : levels_(levels), tensor_(std::move(tensor)) {
    // TORCH_INTERNAL_ASSERT(!isBatchedTensor(tensor));
    // 对给定的张量进行断言，确保其不是批处理张量
    
    Tensor& tensor() { return tensor_; }
    const Tensor& tensor() const { return tensor_; }
    // 返回非常量和常量引用的成员张量对象
    
    // Maps logical dim indices to physical dim indices. Also does dim wrapping.
    //
    // For example, given:
    //   physical_view = VmapPhysicalView(tensor=ones(2, 3, 4, 5), levels={1, 3})
    //
    // Then physical_view.getPhysicalDims({0, 1}) returns {2, 3}.
    // This is because the size of levels tell us that the first two dimensions
    // of `tensor_` are batch dimensions, so a logical dim of `n` is actually
    // a physical dim of `n + 2`.
    VmapDimVector getPhysicalDims(IntArrayRef logical_dims) const;
    // 将逻辑维度索引映射到物理维度索引，并处理维度包装
    
    int64_t getPhysicalDim(int64_t logical_dim) const;
    // 返回指定逻辑维度索引对应的物理维度索引
    
    // Returns a VmapPhysicalToLogicalMap object. This can be used for
    // mapping a physical tensor to a new logical tensor (BatchedTensor)
    VmapPhysicalToLogicalMap getPhysicalToLogicalMap() const;
    // 返回一个 VmapPhysicalToLogicalMap 对象，用于将物理张量映射到新的逻辑张量（批处理张量）
    
    // Maps a logical shape to a physical shape by pre-pending the batch
    // sizes to the logical shape.
    VmapDimVector getPhysicalShape(IntArrayRef logical_shape) const;
    // 将逻辑形状映射到物理形状，通过将批处理大小前置到逻辑形状中
    
    SymDimVector getPhysicalShape(c10::SymIntArrayRef logical_shape) const;
    // 返回符号化的物理形状，通过将批处理大小前置到逻辑形状中
    
    int64_t numBatchDims() const;
    // 返回批处理维度的数量
    
    private:
    int64_t numLogicalDims() const;
    // 返回逻辑维度的数量
    
    std::bitset<kVmapNumLevels> levels_;
    // 用于存储 VmapPhysicalView 的级别信息的位集合
    
    Tensor tensor_;
    // 存储物理视图所关联的张量
};

// 用于将物理张量（非批处理张量）映射到逻辑张量（BatchedTensor）的便捷结构体。
// 它保存了用于进行映射的一些级别，并假设物理张量中的批处理维度都位于张量的前部。
struct TORCH_API VmapPhysicalToLogicalMap {
  // 构造函数，接受一个 kVmapNumLevels 大小的位集合作为参数，用于初始化 levels_ 成员变量。
  VmapPhysicalToLogicalMap(std::bitset<kVmapNumLevels> levels): levels_(levels) {}

  // 将物理张量映射到新的逻辑张量（BatchedTensor）。
  // 假设所有的“批处理维度”都位于物理张量的前部。
  // 例如，给定：
  // - x 是一个秩为 4 的张量，大小为 2, 3, 5, 7
  // - levels = (2, 4)
  // 返回：
  // - BatchedTensor(x, bdims=[(dim=0,lvl=2), (dim=1, lvl=4)])
  Tensor apply(const Tensor& physical_tensor) const;

  // 给定一个物理张量的向量，
  // 1. 将每个张量映射到一个新的逻辑张量。假设所有的“批处理维度”都位于物理张量的前部。
  // 2. 将新的逻辑张量存储回传入的向量中。这样做是为了避免额外的动态分配。
  void applyInplace(std::vector<Tensor>& physical_tensors) const;

  // 位集合，用于存储映射级别。
  std::bitset<kVmapNumLevels> levels_;
};

// 命名空间：at::functorch
} // namespace at::functorch
```