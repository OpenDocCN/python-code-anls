# `.\pytorch\aten\src\ATen\functorch\BatchedTensorImpl.cpp`

```py
// 定义命名空间 at::functorch，包含了批量张量的实现细节
namespace at::functorch {

// 批量张量的构造函数，初始化成员变量并设置关键字集
BatchedTensorImpl::BatchedTensorImpl(DispatchKeySet key_set, Tensor value, int64_t bdim, int64_t level)
  : TensorImpl(
      key_set.add(
          value.is_nested() ? DispatchKey::BatchedNestedTensor : DispatchKey::FuncTorchBatched), // 根据是否嵌套选择合适的分发键
      value.dtype(),  // 使用输入张量的数据类型
      value.device()  // 使用输入张量的设备
    )
  , value_(std::move(value))  // 移动赋值输入张量到成员变量
  , level_(level)  // 设置批量级别
  , bdim_(bdim)  // 设置批量维度
{
  TORCH_INTERNAL_ASSERT(value_.defined());  // 断言确保值张量已定义
  // 对于嵌套张量或使用批量嵌套分发键的情况，进行进一步检查
  if (value_.is_nested() || value_.key_set().has(DispatchKey::BatchedNestedTensor)) {
    TORCH_CHECK(bdim_ == 0,
        "Nested tensors can only be vmapped over dim=0, but got dim=", bdim_);  // 如果是嵌套张量，只能在维度0上进行 vmapped
    TORCH_CHECK(level_ == 1,
        "Only one level of vmap is supported when vmapping over nested tensors");  // 如果是嵌套张量，只支持一级的 vmap
  }
  set_storage_access_should_throw();  // 设置存储访问时抛出异常
  // 根据是否嵌套设置自定义尺寸和步幅策略
  set_custom_sizes_strides(
      value_.is_nested() ? SizesStridesPolicy::CustomSizes : SizesStridesPolicy::CustomStrides);
  checkInvariants();  // 检查不变量
  refreshTensorMetadata();  // 刷新张量元数据
}

// 刷新张量元数据的方法
void BatchedTensorImpl::refreshTensorMetadata() {
  const auto public_dims = value_.dim() - 1;  // 获取公共维度数量
  if (value_.is_nested()) {  // 如果是嵌套张量
    sizes_and_strides_.resize(public_dims);  // 调整尺寸和步幅的大小
    storage_offset_= value_.storage_offset();  // 设置存储偏移量
    refresh_numel();  // 刷新元素数量
    refresh_contiguous();  // 刷新连续性
  } else {  // 如果不是嵌套张量
    c10::SymDimVector new_sizes;  // 新的尺寸向量
    c10::SymDimVector new_strides;  // 新的步幅向量
    new_sizes.reserve(public_dims);
    new_strides.reserve(public_dims);

    // 更新具有符号尺寸和步幅的张量的大小、步幅和存储偏移量
    const auto value_sizes = value_.sym_sizes();
    const auto value_strides = value_.sym_strides();

    for (const auto dim : c10::irange(0, public_dims)) {
      auto actual_dim = actualDim(dim, /*wrap_dim=*/false);  // 获取实际维度
      new_sizes.push_back(value_sizes.at(actual_dim));  // 添加新的尺寸
      new_strides.push_back(value_strides.at(actual_dim));  // 添加新的步幅
    }

    // `set_sizes_and_strides` 方法负责调用 `refresh_numel` 和 `refresh_contiguous`
    set_sizes_and_strides(new_sizes, new_strides, value_.sym_storage_offset());  // 设置尺寸、步幅和存储偏移量
  }
}

// 计算实际维度的方法
int64_t BatchedTensorImpl::actualDim(int64_t dim, bool wrap_dim) const {
  if (wrap_dim) {  // 如果需要包装维度
    const auto ndim = sizes_and_strides_.size();  // 获取尺寸和步幅的大小
    dim = maybe_wrap_dim(dim, static_cast<int64_t>(ndim));  // 可能包装维度
  }
  if (bdim_ <= dim) {  // 如果批量维度小于或等于维度
    return dim + 1;  // 返回维度加一
  } else {
    return dim;  // 否则返回维度本身
  }
}

// 检查不变量的方法
void BatchedTensorImpl::checkInvariants() const {
  TORCH_INTERNAL_ASSERT(level_ > -1);  // 断言确保批量级别大于-1
}

// 获取指定维度尺寸的自定义方法
int64_t BatchedTensorImpl::size_custom(int64_t d) const {
  if (!value_.is_nested()) {  // 如果不是嵌套张量
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);  // 可能包装维度

    // 更新维度，如果需要包装标量
    const auto ndim = sizes_and_strides_.size();  // 获取尺寸和步幅的大小
    d = maybe_wrap_dim(d, ndim, /*wrap_scalar=*/false);  // 可能包装维度

    // 返回实际维度的尺寸
    return sizes_and_strides_.at(d).size();
  }
  // 如果是嵌套张量，暂不支持自定义尺寸
  TORCH_CHECK(false, "Custom size API is not yet supported for nested tensors");
  return -1;  // 返回无效值
}

} // namespace at::functorch
    return sizes_default()[d];
  }



    // 返回默认尺寸中索引为 d 的尺寸值
    return sizes_default()[d];
  }



  // TODO: Error messages will mention the actualDim, which could be confusing; fix this
  auto actual_dim = actualDim(d, /*wrap_dim=*/ true);
  return value_.size(actual_dim);



  // TODO: 错误消息将提到 actualDim，这可能会造成混淆；需要修复此问题
  // 计算实际维度，使用 actualDim 函数，并确保 wrap_dim 参数为 true
  auto actual_dim = actualDim(d, /*wrap_dim=*/ true);
  // 返回 value_ 对象在实际维度 actual_dim 上的大小
  return value_.size(actual_dim);
}

c10::SymInt BatchedTensorImpl::sym_size_custom(int64_t d) const {
  // 如果值不是嵌套的，则调整维度d，确保在当前维度范围内，不包装标量
  if (!value_.is_nested()) {
    d = maybe_wrap_dim(d, dim(), /*wrap_scalar=*/false);
    // 返回默认的符号大小列表中的第d个元素
    return sym_sizes_default()[d];
  }
  // TODO: 错误消息将提及实际维度，这可能会令人困惑；需要修复此问题
  auto actual_dim = actualDim(d, /*wrap_dim=*/ true);
  // 返回值对象在实际维度上的符号大小
  return value_.sym_size(actual_dim);
}

IntArrayRef BatchedTensorImpl::sizes_custom() const {
  // 检查值对象是否为嵌套的，如果是则抛出异常，不支持批次嵌套张量的sizes()操作
  TORCH_CHECK(!value_.is_nested(), "sizes() is not supported for batched nested tensors");
  // 返回默认的尺寸列表
  return sizes_default();
}

SymIntArrayRef BatchedTensorImpl::sym_sizes_custom() const {
  // 检查值对象是否为嵌套的，如果是则抛出异常，不支持批次嵌套张量的sizes()操作
  TORCH_CHECK(!value_.is_nested(), "sizes() is not supported for batched nested tensors");
  // 返回默认的符号尺寸列表
  return sym_sizes_default();
}

// 以下方法作为Tensor的公开方法

IntArrayRef BatchedTensorImpl::strides_custom() const {
  // 返回默认的步幅列表
  return strides_default();
}

SymIntArrayRef BatchedTensorImpl::sym_strides_custom() const {
  // 返回默认的符号步幅列表
  return sym_strides_default();
}

// TODO: 实现批次张量的适当连续性，然后将sizes_strides_policy恢复为Default
bool BatchedTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  // 检查内存格式是否为Contiguous，如果不是则抛出异常
  TORCH_CHECK(memory_format == MemoryFormat::Contiguous,
      "NYI: querying is_contiguous inside of vmap for memory_format ",
      "other than torch.contiguous_format");
  // 查询默认的连续性状态并返回
  return is_contiguous_default(memory_format);
}

// 以下是一些内部继承方法，我们不支持它们。
// 它们不应该被调用。

void BatchedTensorImpl::set_size(int64_t dim, int64_t new_size) {
  // 不支持在BatchedTensorImpl上直接设置尺寸，抛出断言错误
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for BatchedTensorImpl");
}

void BatchedTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  // 不支持在BatchedTensorImpl上直接设置步幅，抛出断言错误
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for BatchedTensorImpl");
}

#ifdef DEBUG
bool BatchedTensorImpl::has_storage() const {
  // 调试模式下，断言BatchedTensorImpl不应该有存储
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "BatchedTensorImpl assumes that storage_ is never set");
  return false;
}
#endif

const char* BatchedTensorImpl::tensorimpl_type_name() const {
  // 返回BatchedTensorImpl的类型名称字符串
  return "BatchedTensorImpl";
}

c10::intrusive_ptr<TensorImpl> BatchedTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  // 在vmap变换下，禁止访问`data`，抛出异常
  TORCH_CHECK(false, "accessing `data` under vmap transform is not allowed");
  return nullptr;
}

c10::intrusive_ptr<TensorImpl> BatchedTensorImpl::shallow_copy_and_detach(
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  // 在vmap变换下，禁止访问`data`，抛出异常
  TORCH_CHECK(false, "accessing `data` under vmap transform is not allowed");
  return nullptr;
}

void BatchedTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  // 在vmap变换下，禁止直接使用`.data`进行突变操作，抛出异常
  TORCH_CHECK(false, "mutating directly with `.data` under vmap transform is not allowed.");
}
# 创建一个批处理张量，将给定张量封装在批处理实现中
Tensor makeBatched(const Tensor& tensor, int64_t bdim, int64_t level) {
  # 获取需要传播到包装器的调度键集合
  DispatchKeySet key_set = getKeysToPropagateToWrapper(tensor);
  # 尝试获取张量的批处理实现
  auto* batched = maybeGetBatchedImpl(tensor);
  if (batched) {
    # 如果批处理实现存在，则获取其当前级别
    auto batched_level = batched->level();
    # 内部断言：确保目标级别高于当前批处理级别，否则抛出错误信息
    TORCH_INTERNAL_ASSERT(level > batched_level, " batched_level: ", batched_level, " level: ", level);
  }
  # 利用批处理张量实现创建一个新的张量
  return at::detail::make_tensor<BatchedTensorImpl>(key_set, tensor, bdim, level);
}

# 将给定张量添加批处理维度并返回
Tensor addBatchDim(const Tensor& tensor, int64_t dim, int64_t level) {
  # 调用makeBatched函数，将给定维度和级别作为参数
  return makeBatched(tensor, dim, level);
}

# 结束命名空间at::functorch
} // namespace at::functorch
```