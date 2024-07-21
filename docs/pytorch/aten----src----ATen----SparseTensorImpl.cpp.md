# `.\pytorch\aten\src\ATen\SparseTensorImpl.cpp`

```
// 匿名命名空间中定义了一个函数，根据分发键集返回稀疏张量的设备类型
DeviceType sparseTensorSetToDeviceType(DispatchKeySet key_set) {
  // 获取具有最高优先级的后端类型 ID
  auto k = c10::highestPriorityBackendTypeId(key_set);
  // 检查该后端类型是否为稀疏类型，否则抛出错误
  TORCH_CHECK(c10::toFunctionalityKey(k) == DispatchKey::Sparse,
    "cannot create sparse tensor with non sparse dispatch key ", k);
  // 返回与指定后端类型对应的设备类型
  return c10::dispatchKeyToDeviceType(k);
}

// 稀疏张量的实现类构造函数，用于创建空的稀疏张量
SparseTensorImpl::SparseTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta data_type)
  :   SparseTensorImpl(key_set, data_type
      // 创建形状为 [1,0] 的空索引张量，并设置其设备和数据类型
      , at::empty({1, 0}, at::initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(ScalarType::Long))
      // 创建形状为 [0] 的空值张量，并设置其设备和数据类型
      , at::empty({0}, at::initialTensorOptions().device(sparseTensorSetToDeviceType(key_set)).dtype(data_type))) {}

// 稀疏张量的实现类构造函数，用于指定索引和值张量的初始化
SparseTensorImpl::SparseTensorImpl(at::DispatchKeySet key_set, const caffe2::TypeMeta data_type, at::Tensor indices, at::Tensor values)
    : TensorImpl(key_set, data_type, values.device())
    , sparse_dim_(1)  // 稀疏维度为 1
    , indices_(std::move(indices))  // 设置索引张量
    , values_(std::move(values)) {  // 设置值张量
  // 断言索引张量的形状为 [1,0]
  AT_ASSERT(indices_.sizes() == IntArrayRef({1, 0}));
  // 断言值张量的形状为 [0]
  AT_ASSERT(values_.sizes() == IntArrayRef({0}));
  // 断言值张量和索引张量的设备相同
  AT_ASSERT(values_.device() == indices_.device());
  // 断言值张量和当前张量的设备相同
  AT_ASSERT(values_.device() == device());

  // 稀疏张量通常不是重叠且稠密的
  is_non_overlapping_and_dense_ = false;
  // 设置存储访问时应抛出异常
  set_storage_access_should_throw();
  // 设置自定义大小和步幅策略
  set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
}

// 释放稀疏张量的资源，重写了父类的方法
void SparseTensorImpl::release_resources() {
  // 调用父类的释放资源方法
  TensorImpl::release_resources();
  // 释放值张量的资源
  values_.reset();
  // 释放索引张量的资源
  indices_.reset();
}

// 设置稀疏张量维度大小的方法，因稀疏张量不支持动态大小设置，抛出错误
void SparseTensorImpl::set_size(int64_t dim, int64_t new_size) {
  AT_ERROR("sparse tensors do not have set_size");
}

// 设置稀疏张量步幅的方法，因稀疏张量不支持动态步幅设置，抛出错误
void SparseTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  AT_ERROR("sparse tensors do not have set_stride");
}

// 设置稀疏张量存储偏移量的方法，因稀疏张量不支持存储偏移设置，抛出错误
void SparseTensorImpl::set_storage_offset(int64_t storage_offset) {
  AT_ERROR("sparse tensors do not have set_storage_offset");
}
#ifdef DEBUG
bool SparseTensorImpl::has_storage() const {
  // 断言确保 storage_ 未设置，以确保 SparseTensorImpl 假设中 storage_ 从不被设置
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!storage_, "SparseTensorImpl assumes that storage_ is never set");
  // 返回 false，表明 SparseTensorImpl 对象没有存储空间
  return false;
}
#endif

const char* SparseTensorImpl::tensorimpl_type_name() const {
  // 返回稀疏张量实现的类型名称字符串 "SparseTensorImpl"
  return "SparseTensorImpl";
}

void SparseTensorImpl::set_indices_and_values_unsafe(const Tensor& indices, const Tensor& values) {
  // 检查是否允许修改张量元数据，否则抛出错误信息
  TORCH_CHECK(allow_tensor_metadata_change(), "set_indices_and_values_unsafe ", err_msg_tensor_metadata_change_not_allowed);

  // 检查 indices 张量是否为稠密张量，否则抛出带有布局信息的错误信息
  TORCH_CHECK(!indices.is_sparse(), "expected indices to be a dense tensor, but got indices of layout ", indices.layout());
  // 检查 values 张量是否为稠密张量，否则抛出带有布局信息的错误信息
  TORCH_CHECK(!values.is_sparse(), "expected values to be a dense tensor, but got values of layout ", values.layout());

  // 检查 values 张量的设备类型是否与当前稀疏张量的设备类型匹配，否则抛出错误信息
  TORCH_CHECK(values.device().type() == device().type(), "device type of values (", values.device().type(), ") must match device type of device().type()", device().type(), ")");
  // 检查 values 张量的数据类型是否与当前稀疏张量的数据类型匹配，否则抛出错误信息
  TORCH_CHECK(values.scalar_type() == typeMetaToScalarType(dtype()), "dtype of values (", values.scalar_type(), ") must match dtype of sparse tensor (", typeMetaToScalarType(dtype()), ")");
  // 检查 indices 张量是否为 int64 类型，否则抛出错误信息
  TORCH_CHECK(indices.scalar_type() == kLong, "indices must be an int64 tensor");
  // 检查 indices 张量和 values 张量的后端（backend）是否匹配，否则抛出错误信息
  TORCH_CHECK(indices.options().backend() == values.options().backend(), "backend of indices (", indices.options().backend(), ") must match backend of values (", values.options().backend(), ")");
  // 如果 indices 张量在 CUDA 设备上，检查其设备与 values 张量的设备是否匹配，否则抛出错误信息
  TORCH_CHECK(!indices.is_cuda() || indices.get_device() == values.get_device(), "device of indices (", indices.get_device(), ") must match device of values (", values.get_device(), ")");

  // 检查 indices 张量的维度是否为2，表明其应为稀疏_dim x nnz 的形式，否则抛出错误信息
  TORCH_CHECK(indices.dim() == 2, "indices must be sparse_dim x nnz, but got: ", indices.sym_sizes());
  // 检查 indices 张量的第二个符号尺寸是否与 values 张量的第一个符号尺寸相匹配，否则抛出错误信息
  TORCH_CHECK(indices.sym_size(1) == values.sym_size(0), "indices and values must have same nnz, but got nnz from indices: ", indices.sym_size(1), ", nnz from values: ", values.sym_size(0));
  // 检查 indices 张量的第一个符号尺寸是否与 sparse_dim_ 匹配，否则抛出错误信息
  TORCH_CHECK(indices.sym_size(0) == sparse_dim_, "indices has incorrect first dimension, expected ", sparse_dim_, ", got ", indices.sym_size(0));
  // 检查 values 张量的维度是否为 dense_dim_ + 1，表明其具有正确的维度数目，否则抛出错误信息
  TORCH_CHECK(values.dim() == dense_dim_ + 1, "values has incorrect number of dimensions, expected ", dense_dim_ + 1, ", got ", values.dim());

  // 计算期望的 values 张量尺寸，确保其与预期的尺寸匹配，否则抛出错误信息
  auto dense_size_original = sym_sizes().slice(sparse_dim_);
  std::vector<c10::SymInt> expected_values_size_vec = {values.sym_size(0)};
  expected_values_size_vec.insert(expected_values_size_vec.end(), dense_size_original.begin(), dense_size_original.end());
  SymIntArrayRef expected_values_size(expected_values_size_vec);
  auto new_values_size = values.sym_sizes();
  TORCH_CHECK(
    std::equal(expected_values_size.begin(), expected_values_size.end(), new_values_size.begin()),
    "values has incorrect size, expected ", expected_values_size, ", got ", new_values_size
  );

  // 设置 indices_ 和 values_ 成员变量为传入的 indices 和 values 张量
  indices_ = indices;
  values_ = values;
  // 断言当前稀疏张量的设备与 values_ 张量的设备匹配
  AT_ASSERT(device() == values_.device());
  // 断言 values_ 张量的设备与 indices_ 张量的设备匹配
  AT_ASSERT(values_.device() == indices_.device());

  // 根据 sym_nnz() 的结果设置 coalesced_ 成员变量
  coalesced_ = TORCH_GUARD_SIZE_OBLIVIOUS(sym_nnz().sym_lt(2));
}

} // namespace at
```