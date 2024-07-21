# `.\pytorch\aten\src\ATen\SparseCsrTensorImpl.cpp`

```
// 在 at 命名空间内定义 SparseCsrTensorImpl 的构造函数
SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,            // 设置分派键集
    at::Device device,                     // 指定设备
    at::Layout layout,                     // 指定张量布局
    const caffe2::TypeMeta data_type)      // 指定数据类型
    : SparseCsrTensorImpl(                 // 调用另一个构造函数进行初始化列表初始化
          key_set,                         // 将分派键集传递给另一个构造函数
          data_type,                       // 将数据类型传递给另一个构造函数
          at::empty(                      // 创建一个空张量用于存储行索引，传递给另一个构造函数
              {0},                         // 空张量的形状为 {0}
              at::initialTensorOptions()   // 使用初始张量选项设置
                  .device(device)          // 指定设备
                  .dtype(ScalarType::Int)) // 指定数据类型为 Int
          ,                                // 分隔符，表示下一个参数的开始
          at::empty(                      // 创建一个空张量用于存储列索引，传递给另一个构造函数
              {0},                         // 空张量的形状为 {0}
              at::initialTensorOptions()   // 使用初始张量选项设置
                  .device(device)          // 指定设备
                  .dtype(ScalarType::Int)) // 指定数据类型为 Int
          ,                                // 分隔符，表示下一个参数的开始
          at::empty(                      // 创建一个空张量用于存储值，传递给另一个构造函数
              {0},                         // 空张量的形状为 {0}
              at::initialTensorOptions()   // 使用初始张量选项设置
                  .device(device)          // 指定设备
                  .dtype(data_type))       // 指定数据类型
          ,                                // 分隔符，表示下一个参数的开始
          layout                           // 将布局传递给另一个构造函数
      ) {}                                 // 结束构造函数定义

// 在 at 命名空间内定义 SparseCsrTensorImpl 的构造函数
SparseCsrTensorImpl::SparseCsrTensorImpl(
    at::DispatchKeySet key_set,            // 设置分派键集
    const caffe2::TypeMeta data_type,      // 指定数据类型
    at::Tensor crow_indices,               // 行索引张量
    at::Tensor col_indices,                // 列索引张量
    at::Tensor values,                     // 值张量
    at::Layout layout)                     // 指定布局
    // 使用给定的键集、数据类型、设备和布局构造稀疏张量的实现
    : TensorImpl(key_set, data_type, values.device()),
      // 移动传入的行索引和列索引到当前对象的成员变量
      crow_indices_(std::move(crow_indices)),
      col_indices_(std::move(col_indices)),
      // 移动传入的值到当前对象的成员变量
      values_(std::move(values)),
      // 设置布局
      layout_(layout) {
    // 发出一次性警告，指示稀疏 CSR 格式张量的支持处于测试阶段
      TORCH_WARN_ONCE("Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensor support is in beta state. "
                      "If you miss a functionality in the sparse tensor support, please submit a feature request "
                      "to https://github.com/pytorch/pytorch/issues.");
    // 内部断言，确保键集与设备类型之间的一致性
      TORCH_INTERNAL_ASSERT(((key_set.has(DispatchKey::SparseCsrCPU) && device().type() == kCPU)
                             || (key_set.has(DispatchKey::SparseCsrCUDA) && device().type() == kCUDA)
                             || (key_set.has(DispatchKey::SparseCsrMeta) && device().type() == kMeta)
                             || (key_set.has(DispatchKey::SparseCsrCPU) && device().type() == kMeta)   // fake tensor
                             || (key_set.has(DispatchKey::SparseCsrCUDA) && device().type() == kMeta)  // fake tensor
                             || (key_set.has(DispatchKey::SparseCsrPrivateUse1) && device().type() == kPrivateUse1)),
                            "Inconsistent key_set (=", key_set, ") and device (=", device(), ")");
    // 设置存储访问时抛出异常
      set_storage_access_should_throw();
    // 设置标志，指示稀疏张量不是非重叠且稠密的
      is_non_overlapping_and_dense_ = false;
    // 设置自定义大小和步幅策略为自定义步幅
      set_custom_sizes_strides(SizesStridesPolicy::CustomStrides);
    // 检查值的设备与行索引的设备是否相同，如果不同则抛出错误
      TORCH_CHECK(values_.device() == crow_indices_.device(), "Values and ",
                  at::sparse_csr::compressedIndicesName(layout_), " need to be on the same device.");
    // 检查值的设备与列索引的设备是否相同，如果不同则抛出错误
      TORCH_CHECK(values_.device() == col_indices_.device(), "Values and ",
                  at::sparse_csr::plainIndicesName(layout_), " need to be on the same device.");
    // 内部断言，确保值的设备与稀疏张量实例的设备相同
      TORCH_INTERNAL_ASSERT(values_.device() == device(),
                            "Values and compressed sparse tensor instance need to have the same device.");
}

const char* SparseCsrTensorImpl::tensorimpl_type_name() const {
  返回一个常量字符串指针，表示该稀疏CSR张量实现的类型名称为"SparseCsrTensorImpl"。
}

void SparseCsrTensorImpl::resize_(int64_t nnz, IntArrayRef size) {
  // 检查张量是否具有符号形状和步幅，如果是，则抛出错误信息
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "resize_ called on tensor with symbolic shape")
  
  // 获取张量的倒数第二和最后一个维度作为行数和列数
  auto rows = size[size.size() - 2];
  auto cols = size[size.size() - 1];

  // 记录当前crow_indices_的最后一个维度的尺寸
  auto old_crow_indices_size = crow_indices_.size(-1);

  // 创建一个新的crow_indices_的尺寸，从size中取前两维作为维度，最后一个维度设为rows+1
  auto new_crow_indices_size = DimVector(size.slice(0, size.size() - 2));
  new_crow_indices_size.push_back(rows + 1);
  // 调整crow_indices_的尺寸为new_crow_indices_size
  crow_indices_.resize_(new_crow_indices_size);

  // 如果rows + 1大于等于old_crow_indices_size，则在crow_indices_的最后一个维度中从old_crow_indices_size开始填充nnz
  if (rows + 1 >= old_crow_indices_size) {
    crow_indices_.narrow(-1, old_crow_indices_size, rows + 1 - old_crow_indices_size).fill_(nnz);
  } else {
    // 否则，在crow_indices_的倒数第二个维度中从rows开始，填充std::min<int64_t>(nnz, rows*cols)
    crow_indices_.narrow(-1, rows, 1).fill_(std::min<int64_t>(nnz, rows*cols));
  }

  // 创建一个新的col_indices_和values_的尺寸，从size中取前两维作为维度，最后一个维度设为std::min<int64_t>(nnz, rows*cols)
  auto col_indices_values_size = DimVector(size.slice(0, size.size() - 2));
  col_indices_values_size.push_back(std::min<int64_t>(nnz, rows*cols));
  // 调整col_indices_和values_的尺寸为col_indices_values_size
  col_indices_.resize_(col_indices_values_size);
  values_.resize_(col_indices_values_size);

  // 设置sizes_and_strides_的尺寸为size
  sizes_and_strides_.set_sizes(size);

  // 刷新张量的元素数目
  refresh_numel();
}
// 调整稀疏 CSR 张量的大小并清除内容。这是 SparseCsrTensorImpl 类的成员函数。
void SparseCsrTensorImpl::resize_and_clear_(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size) {
  // 检查张量是否具有符号形状和步长，如果是则抛出错误信息
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "resize_and_clear_ called on tensor with symbolic shape");
  // 检查稀疏维度是否为2，否则抛出错误信息
  TORCH_CHECK(sparse_dim == 2, "resize_and_clear_ sparse dimensionality must be 2, got ", sparse_dim);
  // 检查输入的 size 数组长度是否至少为稀疏维度（=sparse_dim）加上密集维度（=dense_dim），否则抛出错误信息
  TORCH_CHECK(static_cast<int64_t>(size.size()) >= sparse_dim + dense_dim, "resize_and_clear_ size length must be at least sparse dimensionality (=",
              sparse_dim, ") plus dense dimensionality (=", dense_dim, "), got ", size.size());
  
  // 计算批处理维度的大小
  auto batch_dim = size.size() - sparse_dim - dense_dim;
  auto batchsize = size.slice(0, batch_dim);
  auto densesize = size.slice(batch_dim + sparse_dim, dense_dim);

  // 设置列索引的大小，初始化为 batchsize 加一个零元素
  auto col_indices_size = DimVector(batchsize);
  col_indices_size.push_back(0); // nse

  // 根据当前的布局类型，分派调用以获取压缩索引的大小
  auto n_compressed_indices = AT_DISPATCH_ROW_SPARSE_COMPRESSED_LAYOUTS(layout_, "resize_and_clear_",
                                                                        [&] () -> int64_t { return size[batch_dim]; },
                                                                        [&] () -> int64_t { return size[batch_dim + 1]; }
                                                                        );

  // 设置值的大小，初始化为 batchsize 加一个零元素
  auto values_size = DimVector(batchsize);
  values_size.push_back(0); // nse

  // 在块张量情况下，警告当前值形状定义了块大小
  // 设置块因子，并根据不同的布局类型获取块大小
  int64_t block_factor = 1;
  AT_DISPATCH_PLAIN_SPARSE_COMPRESSED_LAYOUTS(layout_,
                                              "resize_and_clear_",
                                              [] () {},
                                              [&] () {
                                                auto blocksize = this->values_.sizes().slice(this->batch_dim() + 1, 2);
                                                values_size.append(blocksize.begin(), blocksize.end());
                                                block_factor = blocksize[(the_layout == kSparseBsr ? 0 : 1)];
                                              });

  // 检查压缩维度的大小是否能被块大小整除，否则抛出错误信息
  TORCH_CHECK(n_compressed_indices % block_factor == 0,
              "The size of the compressed dimension (=", n_compressed_indices,
              ") must be divisible with the corresponding block size (=", block_factor,")");

  // 更新压缩维度大小
  n_compressed_indices /= block_factor;
  values_size.append(densesize.begin(), densesize.end());

  // 设置行索引的大小，初始化为 batchsize 加 n_compressed_indices 加 1
  auto crow_indices_size = DimVector(batchsize);
  crow_indices_size.push_back(n_compressed_indices + 1);

  // 调整行索引、列索引和值的大小
  crow_indices_.resize_(crow_indices_size);
  crow_indices_.zero_();
  col_indices_.resize_(col_indices_size);
  values_.resize_(values_size);

  // 设置 sizes_and_strides_ 的尺寸
  sizes_and_strides_.set_sizes(size);

  // 刷新 numel（张量的元素数量）
  refresh_numel();
}
    const Tensor& src) {
  // 检查是否具有符号形状和步幅，如果是，则抛出错误，不能调整大小
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "resize_as_sparse_compressed_tensor_ called on tensor with symbolic shape");

  // 检查自身布局与源张量布局是否相同，必须保持一致性以维护自身的布局不变性
  TORCH_CHECK(
      src.layout() == layout_,
      "resize_as_sparse_compressed_tensor_: self and src must have the same layout, but got: self (",
      layout_,
      ") and source (",
      src.layout(),
      ")");

  // 调用函数获取压缩的稀疏索引和普通索引
  auto [compressed_indices, plain_indices] =
      sparse_csr::getCompressedPlainIndices(src);

  // 重用自身的稀疏行索引存储空间，如果尺寸不同则调整
  if (crow_indices_.sizes() != compressed_indices.sizes()) {
    crow_indices_.resize_as_(compressed_indices);
  }
  // 重用自身的列索引存储空间，如果尺寸不同则调整
  if (col_indices_.sizes() != plain_indices.sizes()) {
    col_indices_.resize_as_(plain_indices);
  }

  // 当尺寸或密集维度不同时，更新索引数据以确保结果在不变性检查下有效
  if ((sizes() != src.sizes()) || (dense_dim() != src.dense_dim())) {
    crow_indices_.copy_(compressed_indices);
    col_indices_.copy_(plain_indices);
  }

  // 重用值存储空间，如果尺寸不同则调整
  if (values_.sizes() != src.values().sizes()) {
    values_.resize_as_(src.values());
  }

  // 设置当前对象的大小和步幅，以匹配源张量的大小
  sizes_and_strides_.set_sizes(src.sizes());
  // 刷新元素总数
  refresh_numel();
// 设置稀疏 CSR 张量的成员张量，接受行索引、列索引、数值和尺寸信息
void SparseCsrTensorImpl::set_member_tensors(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    c10::SymIntArrayRef size) {
  // 检查是否有符号形状和步长
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "set_member_tensors called on tensor with symbolic shape");

  // CSR 类型不变性检查：值的数据类型必须与稀疏张量的数据类型匹配
  TORCH_CHECK(
      values.scalar_type() == typeMetaToScalarType(dtype()),
      "dtype of values (",
      values.scalar_type(),
      ") must match dtype of sparse tensor (",
      typeMetaToScalarType(dtype()),
      ")");
  
  // 设置成员变量：行索引、列索引、值
  crow_indices_ = crow_indices;
  col_indices_ = col_indices;
  values_ = values;

  // 设置尺寸和步长
  sizes_and_strides_.set_sizes(C10_AS_INTARRAYREF_SLOW(size));
  
  // 刷新元素数量信息
  refresh_numel();

  // TODO: 如果此检查出现性能瓶颈（这是不太可能的，因为设备比较仅涉及比较类型和索引（两个整数）），我们可以将其移动到 DEBUG 模式下的断言中。目前这个检查确保并维持一个关键不变性。
  TORCH_CHECK(values_.device() == crow_indices_.device(), "Values and ",
              at::sparse_csr::compressedIndicesName(layout_), " need to be on the same device.");
  TORCH_CHECK(values_.device() == col_indices_.device(), "Values and ",
              at::sparse_csr::plainIndicesName(layout_), " need to be on the same device.");
  TORCH_CHECK(values_.device() == device(),
              "Values and compressed tensor instance need to be on the same device.");
}

// 接受 IntArrayRef 尺寸信息的重载函数
void SparseCsrTensorImpl::set_member_tensors(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const Tensor& values,
    IntArrayRef size) {
  // 调用基础的 set_member_tensors 函数，转换 IntArrayRef 尺寸信息为 SymIntArrayRef
  set_member_tensors(crow_indices, col_indices, values, c10::fromIntArrayRefSlow(size));
}

// 返回自定义步长的 IntArrayRef
IntArrayRef SparseCsrTensorImpl::strides_custom() const {
  // 报错：稀疏 CSR 张量没有步长
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have strides");
}

// 返回符号形状自定义步长的 SymIntArrayRef
SymIntArrayRef SparseCsrTensorImpl::sym_strides_custom() const {
  // 报错：稀疏 CSR 张量没有步长
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have strides");
}

// 设置指定维度的大小（报错：稀疏 CSR 张量不支持设置大小）
void SparseCsrTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have set_size.");
}

// 设置指定维度的步长（报错：稀疏 CSR 张量不支持设置步长）
void SparseCsrTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have set_stride.");
}

// 设置存储偏移（报错：稀疏 CSR 张量不支持设置存储偏移）
void SparseCsrTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have set_storage_offset.");
}

// 检查是否是连续的自定义内存格式（报错：稀疏 CSR 张量不支持连续性检查）
bool SparseCsrTensorImpl::is_contiguous_custom(MemoryFormat) const {
  TORCH_CHECK(false, "Sparse ", at::sparse_csr::layoutToString(layout_, /*upper=*/true), " tensors do not have is_contiguous");
}
```