# `.\pytorch\aten\src\ATen\NestedTensorImpl.cpp`

```py
/**
 * 匿名命名空间用于封装内部函数和变量，限定其作用域在当前编译单元中。
 * 这些函数和变量在命名空间外部不可见，有助于避免命名冲突和提高代码模块化。
 */
#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Exception.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Logging.h>

#include <numeric>
#include <functional>
#include <utility>

/**
 * validate_nested_tensor_metadata 函数用于验证嵌套张量的元数据。
 * 
 * @param nested_sizes 嵌套尺寸张量，用于存储每个维度的大小。
 * @param nested_strides 嵌套步长张量，用于存储每个维度的步长。
 * @param offsets 偏移量张量，根据嵌套维度的不同可能为空或包含偏移信息。
 */
namespace {
inline void validate_nested_tensor_metadata(
    const at::Tensor& nested_sizes,
    const at::Tensor& nested_strides,
    const at::Tensor& offsets) {
  TORCH_INTERNAL_ASSERT(nested_sizes.is_contiguous());
  int64_t size_dim = nested_sizes.dim();
  TORCH_INTERNAL_ASSERT(size_dim == 0 || size_dim == 2);
  TORCH_INTERNAL_ASSERT(nested_strides.is_contiguous());
  TORCH_INTERNAL_ASSERT(nested_strides.dim() == size_dim);
  TORCH_INTERNAL_ASSERT(nested_sizes.sizes() == nested_strides.sizes());
  TORCH_INTERNAL_ASSERT(
      (size_dim == 0 && offsets.size(0) == 0) ||
      (size_dim == 2 && nested_sizes.size(0) == offsets.size(0)));
}

/**
 * generate_nested_key_set_from_buffer 函数从非嵌套张量生成嵌套键集。
 * 
 * @param buffer 非嵌套张量，用于生成相应的嵌套键集。
 * @return 生成的嵌套键集。
 */
inline c10::DispatchKeySet generate_nested_key_set_from_buffer(
    const at::Tensor& buffer) {
  auto nested_key_set = buffer.key_set();
  const bool has_autograd = nested_key_set.has_any(c10::autograd_dispatch_keyset);
  // 移除非嵌套张量特定的键
  nested_key_set = nested_key_set -
      c10::DispatchKeySet{c10::DispatchKey::Dense, c10::DispatchKey::Autograd};

  // 添加嵌套张量特定的键
  nested_key_set =
      nested_key_set | c10::DispatchKeySet{c10::DispatchKey::NestedTensor};
  nested_key_set =
      has_autograd ? nested_key_set | c10::autograd_nested : nested_key_set;
  return nested_key_set;
}

/**
 * get_view_key_set 函数获取视图张量的正确键集。
 * 
 * @param base 基础张量，用于根据其嵌套状态确定适当的键集。
 * @return 生成的视图张量的键集。
 */
c10::DispatchKeySet get_view_key_set(const at::Tensor& base) {
  return base.is_nested() ? base.key_set()
                          : generate_nested_key_set_from_buffer(base);
}

} // namespace

/**
 * construct_opt_sizes 函数用于构造可选尺寸的向量。
 * 
 * @param sizes 尺寸张量，用于提取尺寸信息构造向量。
 * @return 构造的可选尺寸向量。
 */
namespace at::native {

inline std::vector<int64_t> construct_opt_sizes(const at::Tensor& sizes) {
  // torch.tensor([]) 被认为具有 `dim() = 1` 和 `size(0) = 0`
  // torch.nested_tensor([]) 也应该具有 `dim() = 1` 和 `size(0) = 0`
  if (sizes.dim() == 0) {
    return std::vector<int64_t>({0});
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  std::vector<int64_t> result(1, sizes.sizes()[0]);
  if (sizes.dim() > 0) {
    size_t nested_dim = result.size();
    const int64_t* sizes_ptr = sizes.const_data_ptr<int64_t>();
    result.resize(nested_dim + sizes.sizes()[1]);
    int64_t sizes_size_0 = sizes.sizes()[0];
    # 获取 sizes 的第二个维度大小
    int64_t sizes_size_1 = sizes.sizes()[1];
    
    # 遍历第二个维度的索引，将 sizes_ptr 中的值复制到 result 中对应位置
    for (const auto i : c10::irange(sizes_size_1)) {
      result[nested_dim + i] = sizes_ptr[i];
    }
    
    # 嵌套循环：遍历第二个维度和第一个维度的所有索引
    for (const auto j : c10::irange(sizes_size_1)) {
      for (const auto i : c10::irange(sizes_size_0)) {
        # 检查 result[nested_dim + j] 是否为真值，并且其值不等于 sizes_ptr 中对应位置的值
        if (result[nested_dim + j] &&
            (result[nested_dim + j] != sizes_ptr[i * sizes.size(1) + j])) {
          # 如果条件成立，将 result[nested_dim + j] 设置为 -1
          result[nested_dim + j] = -1;
        }
      }
    }
  }
  # 返回处理后的结果数组 result
  return result;
}

// 结束当前函数或代码块的定义

// 假设 `sizes` 是连续的，我们可以根据大小构造步长
at::Tensor construct_nested_strides(const at::Tensor& sizes) {
  // 如果 `sizes` 的维度为 0，表示空的嵌套张量，返回空的步长
  if (sizes.dim() == 0) {
    return sizes;
  }
  // 断言调试条件，确保 `sizes` 的维度为 2
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(sizes.dim() == 2);
  int64_t orig_dim = sizes.size(1);
  // 如果原始维度为 0，则返回 `sizes` 自身，因为步长也是空的但是有形状
  if (orig_dim == 0) {
    return sizes;
  }
  // 创建一个空的步长张量，形状与 `sizes` 相同
  at::Tensor strides = sizes.new_empty(sizes.sizes());
  // 获取 `sizes` 和 `strides` 的指针
  const int64_t* sizes_ptr = sizes.const_data_ptr<int64_t>();
  int64_t* strides_ptr = strides.data_ptr<int64_t>();
  // 遍历 `sizes` 的第一维度
  for (int64_t i = 0; i < sizes.size(0); i++) {
    // 初始化最后一个步长为 1
    strides_ptr[orig_dim - 1] = 1;
    int64_t product = sizes_ptr[orig_dim - 1];
    // 计算步长
    for (int64_t j = orig_dim - 2; j >= 0; j--) {
      strides_ptr[j] = product;
      product *= sizes_ptr[j];
    }
    sizes_ptr += orig_dim;
    strides_ptr += orig_dim;
  }
  // 返回计算得到的步长张量
  return strides;
}

/**
   * 创建一个偏移量张量，假设嵌套张量是连续的
   *
   * 此函数迭代隐式的 ntensor 外部维度，
   * 用每个隐式张量中的元素数量填充一个张量。
   * 返回的张量的第一个元素始终为 0，其长度为 n_tensor。
   *
   * @return 一个偏移量张量
  */
at::Tensor construct_offsets(const at::Tensor& sizes) {
  // 如果 `sizes` 的维度为 0，表示空的嵌套张量，返回一个空张量
  if (sizes.dim() == 0) {
    return at::empty({0}, sizes.options().dtype(kLong));
  }
  int64_t ntensors = sizes.size(0), orig_dim = sizes.size(1);
  // 创建一个空的偏移量张量
  auto offsets = at::empty({ntensors}, sizes.options());
  int64_t *offsets_ptr = offsets.mutable_data_ptr<int64_t>();
  // 如果原始维度为 0，则偏移量为简单的索引
  if (orig_dim == 0) {
    std::iota(offsets_ptr, offsets_ptr + ntensors, 0);
    return offsets;
  }
  // 获取 `sizes` 的指针
  const int64_t* sizes_ptr = sizes.const_data_ptr<int64_t>();
  // 第一个偏移量为 0
  offsets_ptr[0] = 0;
  // 迭代计算偏移量
  for (const auto i : c10::irange(ntensors - 1)) {
    const int64_t row_product = std::accumulate(sizes_ptr, sizes_ptr + orig_dim, 1, std::multiplies());
    offsets_ptr[i + 1] = offsets_ptr[i] + row_product;
    sizes_ptr += orig_dim;
  }
  // 返回计算得到的偏移量张量
  return offsets;
}

// NestedTensorImpl 构造函数的实现，初始化嵌套张量的各个属性
NestedTensorImpl::NestedTensorImpl(
    Storage storage,
    c10::DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets)
    // 使用移动语义初始化 TensorImpl 对象，传入 storage、key_set 和 data_type
    TensorImpl(std::move(storage), key_set, data_type),
    // 使用移动语义初始化 nested_sizes_ 对象
    nested_sizes_(std::move(nested_sizes)),
    // 使用移动语义初始化 nested_strides_ 对象
    nested_strides_(std::move(nested_strides)),
    // 使用移动语义初始化 storage_offsets_ 对象
    storage_offsets_(std::move(storage_offsets)),
    // 初始化 opt_sizes_ 为 c10::nullopt
    opt_sizes_(c10::nullopt) {
    // 记录使用 "torch.NestedTensor" API 的一次使用
    C10_LOG_API_USAGE_ONCE("torch.NestedTensor");
    // 发出一次性警告，说明嵌套张量的 PyTorch API 处于原型阶段，将来可能会更改
    TORCH_WARN_ONCE(
        "The PyTorch API of nested tensors is in prototype stage and will change "
        "in the near future.");
    // 获取 storage_ 对象的设备信息
    auto storage_device = storage_.device();
    // 内部断言，确保 NestedTensorImpl 的存储必须是 CUDA、CPU、XPU 或私有的使用一的后端
    TORCH_INTERNAL_ASSERT(
        storage_device.is_cpu() || storage_device.is_cuda() || storage_device.is_xpu() || storage_device.is_privateuseone(),
        "NestedTensorImpl storage must be either CUDA, CPU, XPU or ", get_privateuse1_backend(), " but got ",
        storage_device);
    // 验证嵌套张量的元数据，包括 nested_sizes_、nested_strides_ 和 storage_offsets_
    validate_nested_tensor_metadata(nested_sizes_, nested_strides_, storage_offsets_);
    // 刷新维度信息
    refresh_dim();
    // 设置自定义的尺寸和步长策略为 CustomSizes
    set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

// NestedTensorImpl 的构造函数，接受一个 buffer 引用、嵌套大小张量、嵌套步长张量和存储偏移张量作为参数
NestedTensorImpl::NestedTensorImpl(
    const at::Tensor& buffer,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets)
    : NestedTensorImpl(
          buffer.storage(), // 调用另一个构造函数，传递 buffer 的存储、从 buffer 生成的嵌套键集、数据类型和移动语义的嵌套大小、嵌套步长、存储偏移
          generate_nested_key_set_from_buffer(buffer),
          buffer.dtype(),
          std::move(nested_sizes),
          std::move(nested_strides),
          std::move(storage_offsets)) {

  // 断言 buffer 的维度必须为 1
  TORCH_INTERNAL_ASSERT(
      buffer.dim() == 1,
      "NestedTensorImpl buffer is required to be 1 dimensional but got a buffer with ",
      buffer.dim(),
      " dimensions.");
}

// 假设是连续的，可以从 nested_sizes 推断出 nested_strides 和 offsets
// 这是 NestedTensorImpl 的另一个构造函数，接受 buffer 和 nested_sizes 作为参数
NestedTensorImpl::NestedTensorImpl(
    const at::Tensor& buffer,
    const at::Tensor& nested_sizes)
    : NestedTensorImpl(
          buffer,
          nested_sizes,
          construct_nested_strides(nested_sizes), // 使用 nested_sizes 构造嵌套步长张量
          construct_offsets(nested_sizes)) // 使用 nested_sizes 构造存储偏移张量
{}

// NestedTensorImpl 的构造函数，接受 impl_type、基础张量、嵌套大小张量、嵌套步长张量和存储偏移张量作为参数
NestedTensorImpl::NestedTensorImpl(
    c10::TensorImpl::ImplType impl_type,
    const at::Tensor& base_tensor,
    at::Tensor nested_sizes,
    at::Tensor nested_strides,
    at::Tensor storage_offsets)
    : TensorImpl(impl_type, Storage(base_tensor.storage()), get_view_key_set(base_tensor), base_tensor.dtype()), // 调用 TensorImpl 的构造函数，传递实现类型、基础张量的存储、视图键集和数据类型
      nested_sizes_(std::move(nested_sizes)), // 初始化嵌套大小张量
      nested_strides_(std::move(nested_strides)), // 初始化嵌套步长张量
      storage_offsets_(std::move(storage_offsets)), // 初始化存储偏移张量
      opt_sizes_(c10::nullopt) { // 初始化可选大小为无值
  // 验证嵌套张量的元数据是否有效
  validate_nested_tensor_metadata(nested_sizes_, nested_strides_, storage_offsets_);
  // 刷新维度信息
  refresh_dim();
  // 设置自定义大小和步长策略
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::CustomSizes);
}

// 返回指定维度的可选大小
std::optional<int64_t> NestedTensorImpl::opt_size(int64_t d) const {
  if (C10_UNLIKELY(!opt_sizes_.has_value())) {
    // 缓存元数据以避免每次重新计算
    opt_sizes_ = c10::make_optional(construct_opt_sizes(nested_sizes_));
  }
  // 将维度 d 包装在边界内，确保不超过张量的维数
  d = at::maybe_wrap_dim(d, dim(), false);
  // 如果大小为 -1，则返回无值
  if ((*opt_sizes_)[d] == -1) {
    return c10::nullopt;
  }
  return (*opt_sizes_)[d];
}

// 刷新维度信息
void NestedTensorImpl::refresh_dim() {
  // 计算当前维度，如果 nested_sizes_ 的维度为零，则返回 0，否则返回嵌套大小张量的第二个维度大小加 1
  const auto my_dim = nested_sizes_.dim() ? nested_sizes_.sizes()[1] + 1 : 1;
  // 调整 sizes_and_strides_ 的大小以匹配当前维度
  sizes_and_strides_.resize(my_dim);
  // 断言当前维度与实际维度相同
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dim() == my_dim);
}

// 返回自定义维度数，等同于默认维度数
int64_t NestedTensorImpl::dim_custom() const {
  return dim_default();
}

// 当前假设大小和步长是连续的
int64_t NestedTensorImpl::numel_custom() const {
  // 如果 nested_sizes_ 的维度为 0，则返回 0；否则计算从嵌套大小张量获取的元素总数
  if (nested_sizes_.dim() == 0) {
    return 0;
  }
  return get_numel_from_nested_size_tensor(nested_sizes_);
}

// 返回符号整数类型的元素数量，等同于 numel_custom 的返回值
c10::SymInt NestedTensorImpl::sym_numel_custom() const {
  return NestedTensorImpl::numel_custom();
}

// 检查是否是特定内存格式下的连续张量，通过调用 nested_tensor_impl_is_contiguous 函数实现
bool NestedTensorImpl::is_contiguous_custom(MemoryFormat) const {
  return nested_tensor_impl_is_contiguous(this);
}

// 返回 sizes_custom 不支持的错误消息，因为 NestedTensorImpl 不支持返回大小
IntArrayRef NestedTensorImpl::sizes_custom() const {
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue.");
}
// 返回空的符号整数数组引用，表示 NestedTensorImpl 不支持大小（sizes）
c10::SymIntArrayRef NestedTensorImpl::sym_sizes_custom() const {
  // 抛出错误并显示消息，指示 NestedTensorImpl 不支持大小，建议提交问题报告
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support sizes. Please file an issue.");
}

// 返回空的符号整数数组引用，表示 NestedTensorImpl 不支持步长（strides）
c10::SymIntArrayRef NestedTensorImpl::sym_strides_custom() const {
  // 抛出错误并显示消息，指示 NestedTensorImpl 不支持步长，建议提交问题报告
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue.");
}

// 返回空的整数数组引用，表示 NestedTensorImpl 不支持步长（strides）
IntArrayRef NestedTensorImpl::strides_custom() const {
  // 抛出错误并显示消息，指示 NestedTensorImpl 不支持步长，建议提交问题报告
  TORCH_CHECK(false, "Internal error: NestedTensorImpl doesn't support strides. Please file an issue.");
}

// 返回字符串 "NestedTensorImpl"，表示 NestedTensorImpl 的类型名称
const char* NestedTensorImpl::tensorimpl_type_name() const {
  return "NestedTensorImpl";
}

// 返回一个新的 TensorImpl 实例，浅拷贝当前实例，并分离（detach）底层数据
template <typename VariableVersion>
c10::intrusive_ptr<TensorImpl> NestedTensorImpl::shallow_copy_and_detach_core(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  // 如果关联有 Python DispatchKey 并且未排除 Python DispatchKey
  if (key_set_.has(DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    // 尝试通过 Python 解释器分离（detach）当前实例
    auto r = pyobj_slot_.load_pyobj_interpreter()->detach(this);
    if (r) {
      // 设置新实例的版本计数器和允许元数据修改的标志
      r->set_version_counter(std::forward<VariableVersion>(version_counter));
      r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      return r;
    }
    // 否则只复制 TensorImpl 而不复制 PyObject。因为解释器已经不可用，不会有人指责我们
  }
  // 创建新的 NestedTensorImpl 实例并复制元数据
  auto impl = c10::make_intrusive<NestedTensorImpl>(
      storage_,
      key_set_,
      data_type_,
      nested_sizes_,
      nested_strides_,
      storage_offsets_);

  // 复制张量元数据到新实例
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::forward<VariableVersion>(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

// 返回一个新的 TensorImpl 实例，浅拷贝当前实例，并分离（detach）底层数据
c10::intrusive_ptr<TensorImpl> NestedTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

// 返回一个新的 TensorImpl 实例，浅拷贝当前实例，并分离（detach）底层数据
c10::intrusive_ptr<TensorImpl> NestedTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      std::move(version_counter), allow_tensor_metadata_change);
}

// 计算给定张量的元素数量
int64_t get_numel_from_nested_size_tensor(const at::Tensor& tensor) {
  // 定义最大的元素数量，防止整数溢出
  constexpr auto numel_max = std::min(
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      static_cast<uint64_t>(std::numeric_limits<size_t>::max()));

  // 获取张量中大小（sizes）的指针
  const int64_t* sizes_ptr = tensor.const_data_ptr<int64_t>();
  // 获取嵌套维度的数量
  const auto nt_dim = tensor.size(1);
  uint64_t num_elements{0};

  // 遍历张量的第一个维度
  for (const auto i : c10::irange(tensor.size(0))) {
    uint64_t n = 1;
    // 计算当前维度的元素数量，并检查是否溢出
    const auto start{sizes_ptr + i * nt_dim};
    const auto end{start + nt_dim};
    bool overflows = c10::safe_multiplies_u64(start, end, &n);
    num_elements += n;
    // 检查总元素数量是否超过最大限制
    overflows |= (num_elements > numel_max);
    // 使用 TORCH_CHECK 宏来检查 !overflows 的值，如果为 true，则输出错误信息 "numel: integer multiplication overflow"
    TORCH_CHECK(!overflows, "numel: integer multiplication overflow");
  }
  // 将 num_elements 转换为 int64_t 类型，并作为函数的返回值
  return static_cast<int64_t>(num_elements);
}

} // namespace at::native
```