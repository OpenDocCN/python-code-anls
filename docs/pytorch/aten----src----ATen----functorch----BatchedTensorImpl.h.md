# `.\pytorch\aten\src\ATen\functorch\BatchedTensorImpl.h`

```
// 版权声明和许可声明，指出此源代码受到 BSD 风格许可证保护，许可文件位于源代码根目录下的 LICENSE 文件中
// 包含 C++ 标准库中的 <bitset> 头文件
// 包含 ATen 库中的头文件，如 ArrayRef、SmallVector 和 Tensor
#pragma once

// 命名空间 at::functorch，用于组织代码，避免命名冲突
namespace at::functorch {

// 使用别名 Tensor 表示 at::Tensor
using Tensor = at::Tensor;

// 假设在代码库的其他地方也有类似的假设，但没有集中定义
constexpr int64_t kVmapMaxTensorDims = 64;

// 有效的 vmap 级别范围从 [0, 64)，这意味着我们支持最多 64 层嵌套的 vmap
constexpr int64_t kVmapNumLevels = 64;

// 在堆栈上存储 BatchDims 的元素数量。大多数人可能会使用 <= 5 层嵌套的 vmaps，但根据需要调整此数字
constexpr int64_t kBatchDimsStackSize = 5;

// BatchedTensorImpl 包含一个基础 Tensor 和单个批处理维度
// 注意：我们使用术语 "BatchedTensor" 来表示一个由 BatchedTensorImpl 支持的 Tensor
//
// 批处理维度被视为 "私有"，不可见给用户。
// 例如，在以下 Tensor 中，
//    bt = BatchedTensorImpl(ones(2, 3, 5, 7), lvl=1, dim=0)
// 维度 0 是批处理维度。
//
// bt.sizes() 返回 (5, 7); bt.sum(0) 执行对 (公共) 维度 0 的约简，等效于基础 ones(2, 3, 5, 7) Tensor 中的维度 3。
struct TORCH_API BatchedTensorImpl : public c10::TensorImpl {
  // 构造函数：初始化 BatchedTensorImpl 对象
  explicit BatchedTensorImpl(at::DispatchKeySet key_set, Tensor value, int64_t dim, int64_t level);

  // 返回此张量的批处理维度
  int64_t bdim() const { return bdim_; }

  // 返回此张量的 vmap 级别
  int64_t level() const { return level_; }

  // 返回 BatchedTensorImpl 包装的 Tensor
  const Tensor& value() const { return value_; }

  // 给定一个公共维度索引，返回基础 value() Tensor 中的维度索引
  // 例如，如果有
  //    bt = BatchedTensorImpl(ones(2, 3, 5, 7), lvl=1, dim=0)
  // bt.actualDim(0) -> 1
  // bt.actualDim(1) -> 2
  // bt.actualDim(2) -> 3
  // bt.actualDim(3) -> 错误
  int64_t actualDim(int64_t dim, bool wrap_dim = true) const;

  // 重写继承自 TensorImpl 的 sizes_custom 方法
  IntArrayRef sizes_custom() const override;

  // 重写继承自 TensorImpl 的 sym_sizes_custom 方法
  SymIntArrayRef sym_sizes_custom() const override;

  // 重写继承自 TensorImpl 的 size_custom 方法
  int64_t size_custom(int64_t d) const override;

  // 重写继承自 TensorImpl 的 sym_size_custom 方法
  c10::SymInt sym_size_custom(int64_t d) const override;

  // 重写继承自 TensorImpl 的 strides_custom 方法
  IntArrayRef strides_custom() const override;

  // 重写继承自 TensorImpl 的 sym_strides_custom 方法
  SymIntArrayRef sym_strides_custom() const override;

  // 重写继承自 TensorImpl 的 is_contiguous_custom 方法
  bool is_contiguous_custom(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const override;

  // 重写继承自 TensorImpl 的 set_size 方法
  void set_size(int64_t dim, int64_t new_size) override;

  // 重写继承自 TensorImpl 的 set_stride 方法
  void set_stride(int64_t dim, int64_t new_stride) override;

  // 重写继承自 TensorImpl 的 shallow_copy_and_detach 方法
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach() override;
    // 声明一个常量引用参数 version_counter，类型为 c10::VariableVersion，该函数为虚函数并被重写
    const c10::VariableVersion& version_counter,
    // 声明一个布尔类型参数 allow_tensor_metadata_change，该函数为虚函数并被重写
    bool allow_tensor_metadata_change) const override;
    
    // 声明一个返回类型为 c10::intrusive_ptr<TensorImpl> 的函数 shallow_copy_and_detach
    // 接受一个右值引用参数 version_counter，类型为 c10::VariableVersion
    // 接受一个布尔类型参数 allow_tensor_metadata_change，该函数为虚函数并被重写
    c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
        c10::VariableVersion&& version_counter,
        bool allow_tensor_metadata_change) const override;
    
    // 声明一个无返回值的函数 shallow_copy_from
    // 接受一个类型为 c10::intrusive_ptr<TensorImpl> 的常量引用参数 impl
    // 该函数为虚函数并被重写
    void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;
#ifdef DEBUG
  // 声明一个虚函数，用于检查对象是否有存储空间，覆盖父类接口
  bool has_storage() const override;
#endif

// 刷新张量元数据的函数声明
void refreshTensorMetadata();

// 在 torchdim 中使用，用于设置 BatchedTensor 的级别，实现与当前 vmap 变换级别匹配的非词法 BatchedTensor
void _unsafe_set_level(int64_t level) {
  // 直接设置私有成员变量 level_
  level_ = level;
}

// 用于处理就地视图操作的批处理规则，这些操作可以改变 bdim 的索引（例如 squeeze_、unsqueeze_）
void unsafe_set_bdim(int64_t bdim) {
  // 必须注意：在执行此操作后，必须调用 refreshTensorMetadata 函数。
  bdim_ = bdim;
}

private:
// 见注释: [BatchedTensorImpl levels invariant]，用于检查不变量的私有方法
void checkInvariants() const;
// 返回 TensorImpl 的类型名称，覆盖父类接口
const char* tensorimpl_type_name() const override;

// 内部使用的 Tensor 对象
Tensor value_;

// 用于表示 BatchedTensor 级别的私有成员变量
int64_t level_;
// 用于表示批处理维度 bdim 的私有成员变量
int64_t bdim_;
};

// 注意：我们使用术语 "BatchedTensor" 来表示由 BatchedTensorImpl 支持的 Tensor。
// 检查给定的 Tensor 是否是 BatchedTensor
inline bool isBatchedTensor(const Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::FuncTorchBatched) ||
      tensor.unsafeGetTensorImpl()->key_set().has(DispatchKey::BatchedNestedTensor);
}

// 在不安全情况下获取 BatchedTensorImpl 指针，如果 Tensor 不是由 BatchedTensorImpl 支持，则调用此函数是不安全的。
// 尽可能使用 `maybeGetBatchedImpl` 替代。
inline BatchedTensorImpl* unsafeGetBatchedImpl(const Tensor& tensor) {
  return static_cast<BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
}

// 尝试获取 BatchedTensorImpl 指针，如果 Tensor 不是 BatchedTensor，则返回 nullptr。
inline BatchedTensorImpl* maybeGetBatchedImpl(const Tensor& tensor) {
  if (!isBatchedTensor(tensor)) {
    return nullptr;
  }
  return unsafeGetBatchedImpl(tensor);
}

// 创建指定维度的批处理位集
inline std::bitset<kVmapMaxTensorDims> createBatchDimBitset(int64_t dim) {
  std::bitset<kVmapMaxTensorDims> is_bdim;
  // 设置指定维度为批处理维度
  is_bdim.set(dim);
  return is_bdim;
}

// 创建给定级别的 vmap 位集
inline std::bitset<kVmapNumLevels> createVmapLevelsBitset(int64_t level) {
  std::bitset<kVmapNumLevels> result;
  // 设置指定级别
  result.set(level);
  return result;
}

// 从普通 Tensor 构造 BatchedTensor
TORCH_API Tensor makeBatched(const Tensor& tensor, int64_t dim, int64_t level);

// 在 `tensor` 上添加一个批处理维度，返回一个 BatchedTensor
TORCH_API Tensor addBatchDim(const Tensor& tensor, int64_t dim, int64_t level);

// 某些分发键必须传播到 BatchedTensor（或一般情况下的任何包装 Tensor 子类）。
// 这是因为 Tensor 上有些方法会跳过分发并检查分发键的存在（例如 is_cpu()）。
// TODO: 可能应该包含更多（或全部？）后端键
constexpr DispatchKeySet kKeysToPropagateToWrapper({
  DispatchKey::Negative,
  DispatchKey::Conjugate,
  DispatchKey::XLA,
  DispatchKey::CUDA,
  DispatchKey::CPU,
});
// 返回应传播到包装器的键集合，根据给定的张量和可选的传播键集合
inline DispatchKeySet getKeysToPropagateToWrapper(const Tensor& tensor, DispatchKeySet to_propagate=kKeysToPropagateToWrapper) {
  // 获取张量底层实现的键集合
  auto key_set = tensor.unsafeGetTensorImpl()->key_set();
  // 返回实际传播到包装器的键集合，通过与预定义的传播键集合进行按位与操作
  return key_set & kKeysToPropagateToWrapper;
}

} // namespace at::functorch
```