# `.\pytorch\aten\src\ATen\LegacyBatchedTensorImpl.h`

```
#pragma once
// 使用预处理器指令 #pragma once，确保该头文件只被包含一次

#include <bitset>
// 包含标准库头文件 <bitset>

#include <ATen/ArrayRef.h>
#include <ATen/SmallVector.h>
#include <ATen/Tensor.h>
// 包含 ATen 库的头文件 ArrayRef.h, SmallVector.h 和 Tensor.h

namespace at {
// 命名空间 at 开始

// 我们在代码库的其他地方假设这个值，但没有集中定义。
// 定义一个常量 kVmapMaxTensorDims，表示 Vmap 支持的最大张量维度数为 64
constexpr int64_t kVmapMaxTensorDims = 64;

// 有效的 vmap 级别范围为 [0, 64)。这意味着我们最多支持 64 层嵌套的 vmap。
// 定义一个常量 kVmapNumLevels，表示最大支持的 vmap 嵌套层数为 64
constexpr int64_t kVmapNumLevels = 64;

// 在堆栈上存储 BatchDims 的元素数量。大多数情况下，可能最多使用 <= 5 层嵌套的 vmaps，
// 但根据需要可以调整这个数字。
// 定义一个常量 kBatchDimsStackSize，表示在堆栈上存储的 BatchDims 元素数量为 5
constexpr int64_t kBatchDimsStackSize = 5;

// BatchDim 表示在 vmap 内部创建的 "私有" 张量维度。它是一个 (level, dim) 元组，
// 其中 `dim` 表示正在进行 vmap 的维度，`level` 是标识在哪个 vmap 内部创建的维度。
// `dim` 对应于 "物理维度" - 它是底层物理张量上的维度索引，正在进行 vmap 操作。
struct BatchDim {
  BatchDim(int64_t level, int64_t dim) : dim_(dim), level_(level) {}
  // 构造函数，初始化 BatchDim 的 level 和 dim

  int64_t dim() const {
    return dim_;
  }
  // 返回 BatchDim 的 dim

  int64_t level() const {
    return level_;
  }
  // 返回 BatchDim 的 level

 private:
  int64_t dim_;
  int64_t level_;
  // BatchDim 的私有成员变量，分别表示维度 dim 和层级 level
};

// 使用 SmallVector 定义 BatchDims，存储 BatchDim 结构的容器，大小为 kBatchDimsStackSize
using BatchDims = SmallVector<BatchDim, kBatchDimsStackSize>;
// 使用 ArrayRef 定义 BatchDimsRef，表示 BatchDims 的不可变引用
using BatchDimsRef = ArrayRef<BatchDim>;

// BatchedTensorImpl 包含一个底层 Tensor 和一个 BatchDims 列表
// 注：我们使用术语 "BatchedTensor" 来表示由 BatchedTensorImpl 支持的张量。
//
// 这些批量维度被视为 "私有"；它们对用户不可见。例如，在以下张量中，
//    bt = BatchedTensorImpl(ones(2, 3, 5, 7), [(lvl=1, dim=0), (lvl=2, dim=1)])
// 维度 0 和 1 是批处理维度。
//
// bt.sizes() 返回 (5, 7); bt.sum(0) 对 (公共) 维度 0 执行约简，这等效于底层 ones(2, 3, 5, 7) 张量的维度 3。
struct TORCH_API BatchedTensorImpl : public c10::TensorImpl {
  // 构造函数，初始化 BatchedTensorImpl，接受一个 Tensor 和 BatchDims
  explicit BatchedTensorImpl(Tensor value, BatchDims bdims);

  // 返回表示此张量的私有维度 BatchDims 的引用
  BatchDimsRef bdims() const {
    return bdims_;
  }

  // BatchedTensorImpl 包装了一个 Tensor
  const Tensor& value() const {
    // 返回成员变量 value_ 的值
    return value_;
    };
    
    // 给定一个公共维度索引，返回底层 value() 张量中的维度索引。
    // 例如，如果我们有 bt = BatchedTensorImpl(ones(2, 3, 5, 7), [(lvl=1, dim=0), (lvl=2, dim=2)])
    // bt.actualDim(0) -> 1
    // bt.actualDim(1) -> 3
    // bt.actualDim(2) -> Error
    int64_t actualDim(int64_t dim, bool wrap_dim = true) const;
    
    // 我们必须重写此方法，因为我们选择了 CustomStrides
    IntArrayRef strides_custom() const override;
    
    // 覆盖从 TensorImpl 继承的一些方法，返回错误消息。
    bool is_contiguous_custom(at::MemoryFormat memory_format) const override;
    void set_size(int64_t dim, int64_t new_size) override;
    void set_stride(int64_t dim, int64_t new_stride) override;
    void set_storage_offset(int64_t storage_offset) override;
#ifdef DEBUG
  // 声明一个虚函数，用于检查对象是否有存储
  bool has_storage() const override;
#endif

 private:
  // 查看批量张量实现中的不变性，确保不变性条件得到满足
  void checkInvariants() const;
  // 返回张量实现类型的名称
  const char* tensorimpl_type_name() const override;

  // 存储实际的张量数据
  Tensor value_;

  // 注意：批量张量实现中的不变性
  // 存在一个不变性条件，即批量维度必须按照递增的级别顺序存储。
  // 换句话说，对于 i < j，bdims_[i].level 必须小于 bdims_[j].level。
  BatchDims bdims_;
};

// 注意：我们使用术语“BatchedTensor”来指代由BatchedTensorImpl支持的张量。
inline bool isBatchedTensor(const Tensor& tensor) {
  // 检查张量是否具有批量维度的实现
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::Batched);
}

// 在非由BatchedTensorImpl支持的张量上调用此函数是不安全的。
// 尽可能使用`maybeGetBatchedImpl`。
inline BatchedTensorImpl* unsafeGetBatchedImpl(const Tensor& tensor) {
  // 强制转换为BatchedTensorImpl指针，假定张量由BatchedTensorImpl支持
  return static_cast<BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

inline BatchedTensorImpl* maybeGetBatchedImpl(const Tensor& tensor) {
  // 如果张量具有批量维度的实现，则返回该实现指针；否则返回空指针。
  if (!isBatchedTensor(tensor)) {
    return nullptr;
  }
  return unsafeGetBatchedImpl(tensor);
}

// 返回一个位集。如果第 i 位被设置，则表示维度 i 是批量维度。
inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(
    BatchDimsRef bdims) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  // 遍历批量维度，设置对应维度的位
  for (const auto& bdim : bdims) {
    is_bdim.set(bdim.dim());
  }
  return is_bdim;
}

// 创建包含`bdims`中所有级别的位集
inline std::bitset<kVmapNumLevels> createVmapLevelsBitset(BatchDimsRef bdims) {
  std::bitset<kVmapNumLevels> result;
  // 遍历批量维度，设置对应级别的位
  for (const auto& bdim : bdims) {
    result.set(bdim.level());
  }
  return result;
}

inline std::ostream& operator<<(std::ostream& out, const BatchDim& bdim) {
  // 将批量维度输出为格式化字符串 "(lvl=<level>, dim=<dimension>)"
  out << "(lvl=" << bdim.level() << ", dim=" << bdim.dim() << ")";
  return out;
}

// 使用此函数从常规张量构造BatchedTensor
TORCH_API Tensor makeBatched(const Tensor& tensor, BatchDims bdims);

// 将批量维度添加到`tensor`，返回一个BatchedTensor
TORCH_API Tensor addBatchDim(const Tensor& tensor, int64_t level, int64_t dim);

// 检查self和other上的原地操作是否“vmap兼容”。
// 有关此定义，请参阅注释：[vmap-incompatible in-place operations]。
TORCH_API bool inplaceIsVmapCompatible(const Tensor& self, const Tensor& other);

} // namespace at
```