# `.\pytorch\aten\src\ATen\native\mkldnn\xpu\detail\Utils.cpp`

```
// 包含 oneDNN 的头文件 Utils.h

#include <ATen/native/mkldnn/xpu/detail/Utils.h>

// 定义命名空间 at::native::onednn，包含了与 oneDNN 相关的函数和类
namespace at::native::onednn {

// 创建一个 oneDNN 内存对象，基于给定的描述符、引擎和指针
dnnl::memory make_onednn_memory(
    dnnl::memory::desc md,       // oneDNN 内存描述符
    dnnl::engine& engine,        // oneDNN 引擎
    void* ptr) {                 // 可选的内存指针
  // 使用 oneDNN 的 SYCL 互操作接口创建内存对象
  return dnnl::sycl_interop::make_memory(
      md,                         // 内存描述符
      engine,                     // 引擎
      dnnl::sycl_interop::memory_kind::usm,  // 内存类型为 USM
      ptr == nullptr ? DNNL_MEMORY_ALLOCATE : ptr);  // 根据指针是否为空决定分配方式
}

// 获取指定维度下的 oneDNN 默认格式标签
dnnl::memory::format_tag get_dnnl_default_format(
    int ndims,                   // 张量的维度
    bool is_channels_last,       // 是否采用 channels last 的布局
    bool allow_undef) {          // 是否允许未定义的格式
  // 根据维度选择合适的默认格式标签
  switch (ndims) {
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return is_channels_last ? dnnl::memory::format_tag::acb
                              : dnnl::memory::format_tag::abc;
    case 4:
      return is_channels_last ? dnnl::memory::format_tag::acdb
                              : dnnl::memory::format_tag::abcd;
    case 5:
      return is_channels_last ? dnnl::memory::format_tag::acdeb
                              : dnnl::memory::format_tag::abcde;
    case 6:
      return dnnl::memory::format_tag::abcdef;
    case 7:
      return dnnl::memory::format_tag::abcdefg;
    case 8:
      return dnnl::memory::format_tag::abcdefgh;
    case 9:
      return dnnl::memory::format_tag::abcdefghi;
    case 10:
      return dnnl::memory::format_tag::abcdefghij;
    case 11:
      return dnnl::memory::format_tag::abcdefghijk;
    case 12:
      return dnnl::memory::format_tag::abcdefghijkl;
    default:
      // 如果不允许未定义的格式，并且维度超过 12，则抛出异常
      if (!allow_undef) {
        TORCH_CHECK(false, "oneDNN doesn't support tensor dimension > 12");
      }
      // 否则返回未定义的格式标签
      return dnnl::memory::format_tag::undef;
  }
}

// 获取指定张量的 oneDNN 数据类型
dnnl::memory::data_type get_onednn_dtype(
    const at::Tensor& tensor,    // 输入张量
    bool allow_undef) {          // 是否允许未定义的数据类型
  // 根据张量的标量类型选择相应的 oneDNN 数据类型
  switch (tensor.scalar_type()) {
    case at::ScalarType::Byte:
      return dnnl::memory::data_type::u8;
    case at::ScalarType::Char:
      return dnnl::memory::data_type::s8;
    case at::ScalarType::QInt8:
      return dnnl::memory::data_type::s8;
    case at::ScalarType::QUInt8:
      return dnnl::memory::data_type::u8;
    case at::ScalarType::Int:
      return dnnl::memory::data_type::s32;
    case at::ScalarType::Half:
      return dnnl::memory::data_type::f16;
    case at::ScalarType::Float:
      return dnnl::memory::data_type::f32;
    case at::ScalarType::BFloat16:
      return dnnl::memory::data_type::bf16;
    default:
      // 如果不允许未定义的数据类型，则抛出异常
      if (!allow_undef) {
        TORCH_CHECK(
            false,
            c10::toString(tensor.scalar_type()),
            " is not supported in oneDNN!");
      }
      // 否则返回未定义的数据类型
      return dnnl::memory::data_type::undef;
  };
}

// 获取指定张量的 oneDNN 数据类型，包括对 Double 类型的特殊处理
dnnl::memory::data_type get_onednn_dtype_include_double(
    const at::Tensor& tensor,    // 输入张量
    bool allow_undef) {          // 是否允许未定义的数据类型
  // 如果输入张量的标量类型是 Double，则返回 f64 的数据类型
  if (tensor.scalar_type() == at::ScalarType::Double)
    return dnnl::memory::data_type::f64;
  // 否则调用 get_onednn_dtype 函数获取数据类型
  return get_onednn_dtype(tensor, allow_undef);
}

// 命名空间结束
} // namespace at::native::onednn
// 检查给定张量在 OneDNN 中是否支持的数据类型
bool is_supported_onednn_dtype(const at::Tensor& tensor) {
  // 获取张量的 OneDNN 数据类型，允许未定义类型
  return get_onednn_dtype(tensor, /*allow_undef*/ true) ==
          dnnl::memory::data_type::undef
      ? false  // 如果类型未定义，则不支持
      : true;  // 否则支持
}

// 获取张量的维度，转换为 OneDNN 的内存维度格式
dnnl::memory::dims get_onednn_dims(const at::Tensor& tensor) {
  dnnl::memory::dims dims;
  // 遍历张量的每个维度，将其添加到 OneDNN 的维度对象中
  for (size_t i = 0; i < tensor.sizes().size(); i++)
    dims.push_back(tensor.size(i));
  return dims;
}

// 获取张量的步长信息，转换为 OneDNN 的内存步长格式
dnnl::memory::dims get_onednn_strides(const at::Tensor& tensor) {
  dnnl::memory::dims strides;
  // 遍历张量的每个维度的步长，将其添加到 OneDNN 的步长对象中
  for (size_t i = 0; i < tensor.strides().size(); i++)
    strides.push_back(tensor.stride(i));
  return strides;
}

// 根据张量构造 OneDNN 的内存描述对象
dnnl::memory::desc get_onednn_md(const at::Tensor& tensor) {
  return {
      get_onednn_dims(tensor),     // 设置维度
      get_onednn_dtype(tensor),    // 设置数据类型
      get_onednn_strides(tensor)}; // 设置步长信息
}

// 检查张量的步长信息是否符合 OneDNN 的要求
bool onednn_strides_check(const at::Tensor& src) {
  auto adims = get_onednn_dims(src);  // 获取张量的维度信息
  int ndims = (int)adims.size();      // 获取维度数量
  auto dims = adims.data();           // 获取维度数据指针
  auto data_type = static_cast<dnnl_data_type_t>(
      get_onednn_dtype(src, /*allow_undef*/ true));  // 获取数据类型
  auto strides_info = get_onednn_strides(src);       // 获取步长信息
  auto strides = strides_info.empty() ? nullptr : &strides_info[0];  // 获取步长数组指针

  dnnl_memory_desc_t md;
  dnnl_memory_desc_create_with_strides(&md, ndims, dims, data_type, strides);  // 创建 OneDNN 内存描述对象
  dnnl_format_kind_t md_fmt_kind;
  int md_ndims;
  int md_inner_nblks;
  dnnl_dims_t* md_padded_dims = nullptr;

  dnnl_memory_desc_query(md, dnnl_query_inner_nblks_s32, &md_inner_nblks);  // 查询内部块数
  dnnl_memory_desc_query(md, dnnl_query_format_kind, &md_fmt_kind);          // 查询格式种类
  dnnl_memory_desc_query(md, dnnl_query_ndims_s32, &md_ndims);               // 查询维度数量
  dnnl_memory_desc_query(md, dnnl_query_padded_dims, &md_padded_dims);       // 查询填充维度信息

  if (strides == nullptr || md_ndims == 0 ||
      md_fmt_kind != dnnl_format_kind_t::dnnl_blocked)
    return true;  // 如果步长为空或者维度为零或者格式种类不是 blocked，则返回 true

  dnnl_dims_t blocks = {0};
  int perm[DNNL_MAX_NDIMS] = {0};
  for (int d = 0; d < md_ndims; ++d) {
    // 对于空张量不需要检查步长
    if (md_padded_dims[d] == 0)
      return true;
    
    // 对于运行时维度不需要步长验证
    if (strides[d] == DNNL_RUNTIME_DIM_VAL)
      return true;

    perm[d] = d;
    blocks[d] = 1;
  }

  auto block_size = 1;
  dnnl_dims_t md_inner_blks;
  dnnl_dims_t md_blk_inner_idxs;
  dnnl_memory_desc_query(md, dnnl_query_inner_idxs, &md_blk_inner_idxs);  // 查询内部索引
  dnnl_memory_desc_query(md, dnnl_query_inner_blks, &md_inner_blks);      // 查询内部块大小
  for (int iblk = 0; iblk < md_inner_nblks; ++iblk) {
    blocks[md_blk_inner_idxs[iblk]] *= md_inner_blks[iblk];
    block_size *= md_inner_blks[iblk];
  }

  // 自定义比较器，在 perm 上生成线性顺序
  auto idx_sorter = [&](const int a, const int b) -> bool {
    if (strides[a] == strides[b] && md_padded_dims[a] == md_padded_dims[b])
      return a < b;
    else if (strides[a] == strides[b])
      return md_padded_dims[a] < md_padded_dims[b];
    else
      return strides[a] < strides[b];
  };
  std::sort(perm, perm + md_ndims, idx_sorter);  // 使用自定义比较器对 perm 数组排序

  auto min_stride = block_size;
  for (int idx = 0; idx < md_ndims; ++idx) {
    const int d = perm[idx];
    // 从排列数组中获取当前索引位置的维度

    // 如果 strides[d] == 0，表示具有广播语义，可以跳过当前维度的检查
    // 注意：由于是有序的，这些是初始的步长
    if (strides[d] == 0)
      continue;
    // 如果 strides[d] 小于最小步长 min_stride，则返回 false
    else if (strides[d] < min_stride)
      return false;

    // 更新下一次迭代的最小步长 min_stride
    const auto padded_dim = *md_padded_dims[d];
    // 计算新的最小步长，考虑到块大小、步长和维度的填充
    min_stride = block_size * strides[d] * (padded_dim / blocks[d]);
  }
  // 所有维度检查通过，返回 true
  return true;
}

// 检查张量是否具有广播特性
bool is_broadcast(const at::Tensor& t) {
  // 遍历张量的维度
  for (int i = 0; i < t.dim(); i++) {
    // 如果某个维度的步长为0，表示存在广播
    if (t.stride(i) == 0)
      return true;
  }
  // 如果所有维度都没有步长为0，则不是广播
  return false;
}

// 检查张量是否符合oneDNN矩阵乘法的步长要求
bool is_onednn_matmul_strides(
    const at::Tensor& tensor,
    bool is_dst) {
  // 检查张量的维度数，并且只支持2维和3维的张量
  auto sizes = tensor.sizes();
  auto tensor_dim = sizes.size();
  if (tensor_dim != 2 && tensor_dim != 3)
    return false;

  // 如果张量是连续的，直接返回true
  if (tensor.is_contiguous())
    return true;

  // 获取张量的步长信息
  dnnl::memory::dims strides = get_onednn_strides(tensor);
  int64_t storage_size = 1;
  // 计算存储尺寸
  for (size_t dim = 0; dim < tensor_dim; ++dim)
    storage_size += (sizes[dim] - 1) * strides[dim];
  // 如果存储尺寸小于元素数量，则不符合要求
  if (storage_size < tensor.numel())
    return false;

  // 如果存在广播，则不支持
  if (is_broadcast(tensor)) {
    return false;
  }

  // 如果是目标张量，检查其内存格式是否为plain，且最后一个轴的步长应为1
  if (is_dst) {
    if (strides[-1] != 1)
      return false;
  } else {
    // 如果是源张量，检查倒数第二和倒数第一轴的步长是否为1
    if (strides[tensor_dim - 1] != 1 && strides[tensor_dim - 2] != 1)
      return false;
  }

  // 进一步检查oneDNN的步长要求
  if (!onednn_strides_check(tensor))
    return false;
  // 符合所有条件，返回true
  return true;
}

// 检查张量是否从其他张量广播到自身
bool is_broadcast_from_other_to_self(
    const at::Tensor& self,
    const at::Tensor& other) {
  // 如果张量大小不相等且可以扩展到相同大小，则认为是广播
  return (
      self.sizes() != other.sizes() &&
      at::is_expandable_to(other.sizes(), self.sizes()));
}

// 根据维度数获取内存格式标签
at::MemoryFormat get_cl_tag_by_ndim(const int64_t ndim) {
  // 检查维度数是否为3, 4或5，用于获取内存格式标签
  TORCH_CHECK(
      3 == ndim || 4 == ndim || 5 == ndim,
      "ndim must be 3, 4 or 5 when get cl tag");
  if (3 == ndim) {
    return at::MemoryFormat::Contiguous;
  } else if (5 == ndim) {
    return at::MemoryFormat::ChannelsLast3d;
  } else {
    return at::MemoryFormat::ChannelsLast;
  }
}

// 检查二元操作的有效性
bool binary_valid(
    const at::Tensor& self,
    const at::Tensor& other,
    bool is_fusion) {
  // 如果张量大小不相等且不能从其他张量广播到自身，则不有效
  if (self.sizes() != other.sizes() &&
      !is_broadcast_from_other_to_self(self, other))
    return false;

  /* 如果以下条件满足，将选择oneDNN路径：
     * 1. self和other应为xpu张量并且已定义。
     * 2. self或other不应为标量（包装张量）。
     * 3. self和other的维度应相等且大于0且小于7。
     * 4. 数据类型应由oneDNN原语支持。
     * 5. self和other的数据类型应相同。
     * 6. self和other应为连续或通道最后连续。*/

  // 1. self和other应为xpu张量并且已定义。
  if ((!self.defined()) || (!other.defined()) || (!self.is_xpu()) ||
      (!other.is_xpu()))
    // 返回 false，表示条件不满足
    return false;

  // 2. self 或 other 不应为标量（封装的张量）。
  if (self.unsafeGetTensorImpl()->is_wrapped_number() || other.unsafeGetTensorImpl()->is_wrapped_number())
    // 返回 false，表示条件不满足
    return false;

  // 3. self 和 other 的维度应相等且大于 0 且小于 7。
  if ((self.dim() <= 0) || (other.dim() <= 0) || (self.dim() != other.dim()) ||
      (self.dim() > 6) || (other.dim() > 6))
    // 返回 false，表示条件不满足
    return false;

  // 4. 数据类型应该被 oneDNN 原语支持。
  switch (self.scalar_type()) {
    // 支持的数据类型之一
    case at::ScalarType::Char:
      break;
    case at::ScalarType::Byte:
      break;
    case at::ScalarType::Half:
      break;
    case at::ScalarType::Float:
      break;
    case at::ScalarType::BFloat16:
      break;
    // 不支持的数据类型，返回 false
    default:
      return false;
  };

  // 5. 数据类型检查
  if (is_fusion) {
    // 对于融合情况，可以在 scalar_type 或 Float 数据类型上执行融合。
    if (self.scalar_type() != other.scalar_type() &&
        other.scalar_type() != at::ScalarType::Float) {
      // 返回 false，表示条件不满足
      return false;
    }
  } else {
    // 对于非融合情况，self 和 other 应处于相同的数据类型。
    if (self.scalar_type() != other.scalar_type()) {
      // 返回 false，表示条件不满足
      return false;
    }
  }

  // 6. self 和 other 应该是连续的或通道最后连续的。
  const auto ndim = self.ndimension();
  auto cl_tag = at::MemoryFormat::ChannelsLast;
  // 根据维度确定通道最后的标签
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    cl_tag = get_cl_tag_by_ndim(ndim);
  }
  // 如果 self 和 other 都是连续的或者通道最后连续的，返回 true；否则返回 false。
  if ((self.is_contiguous() && other.is_contiguous()) ||
      (self.is_contiguous(cl_tag) && other.is_contiguous(cl_tag)))
    return true;
  // 返回 false，表示条件不满足
  return false;
}

// 检查给定的内存格式是否为ChannelsLast或ChannelsLast3d
static inline bool is_channels_last(at::MemoryFormat fmt){
  return (at::MemoryFormat::ChannelsLast == fmt) || (at::MemoryFormat::ChannelsLast3d == fmt);
}

// 检查张量的建议内存格式是否为ChannelsLast
static inline bool is_smf_channels_last(const Tensor& t){
  return is_channels_last(t.suggest_memory_format());
}

// 判断是否应该在卷积操作中使用ChannelsLast格式
bool use_channels_last_for_conv(
    const at::Tensor& src,
    const at::Tensor& weight,
    bool is_transpose){

  if (!src.defined() || src.is_sparse()) {
    // 如果源张量未定义或是稀疏张量，则建议使用channels_first格式
    return false;
  }

  auto suggest_channels_last_format =
      (is_smf_channels_last(src) || is_smf_channels_last(weight));
  if (suggest_channels_last_format) {
    // 如果源张量或权重张量的建议内存格式为ChannelsLast，则建议使用channels_last格式
    return true;
  }

  // 否则建议使用channels_first格式
  return false;
}

}


**注释解释：**
- `}`：结束处，标志着某个代码块的结束，这里没有前文的开头部分。
- `static inline bool is_channels_last(at::MemoryFormat fmt){`：定义了一个静态内联函数，用于判断给定的内存格式是否为ChannelsLast或ChannelsLast3d。
- `static inline bool is_smf_channels_last(const Tensor& t){`：定义了一个静态内联函数，用于检查张量的建议内存格式是否为ChannelsLast。
- `bool use_channels_last_for_conv(`：定义了一个函数，判断在卷积操作中是否应该使用ChannelsLast格式。
- `if (!src.defined() || src.is_sparse()) {`：条件语句，如果源张量未定义或者是稀疏张量，则建议使用channels_first格式。
- `auto suggest_channels_last_format =`：赋值语句，判断源张量或权重张量的建议内存格式是否为ChannelsLast。
- `if (suggest_channels_last_format) {`：条件语句，如果建议使用ChannelsLast格式，则返回true。
- `return false;`：返回语句，表示建议不使用ChannelsLast格式。
```