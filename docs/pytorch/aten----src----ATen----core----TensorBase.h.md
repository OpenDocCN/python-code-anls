# `.\pytorch\aten\src\ATen\core\TensorBase.h`

```py
```cpp`
#pragma once

// 包含核心设备相关的头文件
#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/core/WrapDimMinimal.h>

// 包含 C++17 特性和异常处理相关的头文件
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/ExclusivelyOwnedTensorTraits.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>

// 包含 ATen 库的头文件
#include <ATen/core/NamedTensor.h>
#include <ATen/core/QuantizerBase.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/StorageUtils.h>

// 命名空间 c10 中的 Scalar 类声明
namespace c10 {
class Scalar;
}

// 命名空间 torch::autograd 中的 Node 结构体声明
namespace torch::autograd {

struct Node;

} // namespace torch::autograd

// 命名空间 at 开始
namespace at {

class Tensor;
class TensorBase;

// 将 Tensor 转换为 TensorBase，避免包含 Tensor.h 头文件
TORCH_API const TensorBase& get_tensor_base(const Tensor& t);

namespace impl {
// 实现函数，用于判断是否排除变量的调度
inline bool variable_excluded_from_dispatch() {
#ifdef C10_MOBILE
  // 请阅读 `VariableFallbackKernel.cpp` 中的注释，了解此更改的背景。
  return true;
#else
  // 检查当前调度键集是否包含 autograd_dispatch_keyset
  return c10::impl::tls_local_dispatch_key_set().excluded_.isSupersetOf(c10::autograd_dispatch_keyset);
#endif
}

}

// NOTE: [Tensor vs. TensorBase]
//
// Tensor 是 PyTorch 中的核心数据结构，几乎在所有地方使用，并且包含在头文件中。
// 这意味着每次在 native_functions.yaml 中更新或更改操作符签名时，
// 你（以及其他 PyTorch 开发者）都需要重新编译整个 ATen 及其依赖项。
//
// TensorBase 旨在分离这些头文件依赖关系，并提高所有 PyTorch 开发者的增量构建时间。
// TensorBase 表示对 TensorImpl 的引用计数句柄，与 Tensor 完全相同。
// 但是，TensorBase 的 API 中没有代码生成的方法，因此不依赖于 native_functions.yaml。
//
// 使用提示
// ----------
// - 你可以在 .cpp 或 .cu 文件的顶部定义 `TORCH_ASSERT_NO_OPERATORS`，
//   以确保它没有直接或间接的 native_functions.yaml 的头文件依赖。
// - Tensor 继承自 TensorBase，因此接受 `const TensorBase &` 的函数也可以使用 Tensor。
// - 可以使用 `Tensor(tensor_base)` 将 TensorBase 转换为 tensor，但这需要增加引用计数。
//   另一方面，OptionalTensorRef 可以在不触及引用计数的情况下实现 `const Tensor &`。
# 定义一个 TensorBase 类，用于封装 PyTorch 张量的基本功能
class TORCH_API TensorBase {
 public:
  // 定义一个结构体 unsafe_borrow_t，用于表示不安全的借用情况
  struct unsafe_borrow_t { explicit unsafe_borrow_t() = default; };

 protected:
  // 使用 rhs 的 TensorBase 对象创建一个具有 +0 引用计数的 TensorBase 对象。
  // 在析构时需要特别注意避免减少此引用计数。
  // 用于支持 MaybeOwnedTraits<Tensor>.
  explicit TensorBase(unsafe_borrow_t, const TensorBase& rhs)
      : impl_(c10::intrusive_ptr<at::TensorImpl, UndefinedTensorImpl>::reclaim(rhs.impl_.get())) {}
  friend MaybeOwnedTraits<TensorBase>;

 public:
  // 默认构造函数
  TensorBase() = default;
  // 构造函数，不应该被最终用户使用，是由自动生成的代码调用的实现细节
  explicit TensorBase(
      c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
      : impl_(std::move(tensor_impl)) {
    // 如果 impl_ 为 nullptr，则抛出运行时错误
    if (impl_.get() == nullptr) {
      throw std::runtime_error("TensorImpl with nullptr is not supported");
    }
  }
  // 拷贝构造函数
  TensorBase(const TensorBase&) = default;
  // 移动构造函数
  TensorBase(TensorBase&&) noexcept = default;

 public:
  // 静态方法，从 TensorImpl 创建一个新的 TensorBase 包装器。
  // 注意这是一个自由方法，应谨慎使用。检查必要的不变量。
  static TensorBase wrap_tensor_impl(
      c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl) {
    TensorBase r(std::move(tensor_impl));
    r.enforce_invariants();
    return r;
  }

  // 返回张量的维度
  int64_t dim() const {
    return impl_->dim();
  }
  // 返回张量的存储偏移量
  int64_t storage_offset() const {
    return impl_->storage_offset();
  }

  // 返回一个连续存储的张量，可指定内存格式
  TensorBase contiguous(MemoryFormat memory_format=MemoryFormat::Contiguous) const {
    if (is_contiguous(memory_format)) {
      return *this;
    } else {
      return __dispatch_contiguous(memory_format);
    }
  }

  /// 如果可以合理期待 *this 是连续的且性能很重要，则应使用此方法。
  /// 相比 contiguous 方法，如果 *this 已经是连续的，则保存了一个引用计数的增加/减少，
  /// 但额外增加了栈使用的指针，访问时的额外分支，以及析构时的额外分支。
  c10::MaybeOwned<TensorBase> expect_contiguous(
      MemoryFormat memory_format=MemoryFormat::Contiguous) const &;

  // 禁止使用移动语义版本的 expect_contiguous 方法
  c10::MaybeOwned<TensorBase> expect_contiguous(
      MemoryFormat memory_format=MemoryFormat::Contiguous) && = delete;

  // 将张量填充为指定的标量值，并返回自身
  const TensorBase& fill_(const c10::Scalar& scalar) const;
  // 将张量填充为零，并返回自身
  const TensorBase& zero_() const;

  // 将张量转换为指定选项的张量，可选的非阻塞和拷贝标志，以及内存格式
  TensorBase to(at::TensorOptions options={}, bool non_blocking=false, bool copy=false, std::optional<at::MemoryFormat> memory_format=c10::nullopt) const;

  // 判断张量是否为复数类型
  bool is_complex() const {
    return at::isComplexType(this->scalar_type());
  }

  // 判断张量是否为浮点数类型
  bool is_floating_point() const {
    return at::isFloatingType(this->scalar_type());
  }

  // 判断张量是否为有符号数类型
  bool is_signed() const {
    return at::isSignedType(this->scalar_type());
  }

  // 返回指定维度的符号整数大小
  c10::SymInt sym_size(int64_t dim) const {
  // 返回当前维度的符号步长
  c10::SymInt sym_stride(int64_t dim) const {
    // 获取符号步长的向量
    const auto sizes = this->sym_strides();
    // 确定向量的维度
    const auto ndim = static_cast<int64_t>(sizes.size());
    // 传递 false 给 maybe_wrap_dim，使得行为与数组访问相同（但带有环绕）
    return sizes[c10::maybe_wrap_dim(dim, ndim, /*wrap_scalar=*/false)];
  }

  // 返回当前维度的大小
  int64_t size(int64_t dim) const {
    return impl_->size(dim);
  }

  // 返回当前维度的步长
  int64_t stride(int64_t dim) const {
    // 获取步长的向量
    const auto strides = this->strides();
    // 确定向量的维度
    const auto ndim = static_cast<int64_t>(strides.size());
    // 传递 false 给 maybe_wrap_dim，使得行为与数组访问相同（但带有环绕）
    return strides[c10::maybe_wrap_dim(dim, ndim, /*wrap_scalar=*/false)];
  }

  // 返回不安全的 TensorImpl 指针
  TensorImpl * unsafeGetTensorImpl() const {
    return impl_.get();
  }

  // 释放并返回不安全的 TensorImpl 指针
  TensorImpl * unsafeReleaseTensorImpl() {
    return impl_.release();
  }

  // 返回不安全的内部 TensorImpl 指针的引用
  const c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>& getIntrusivePtr() const {
    return impl_;
  }

  // 释放并返回不安全的内部 TensorImpl 指针的引用
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> unsafeReleaseIntrusivePtr() {
    return std::move(impl_);
  }

  // 检查当前 Tensor 是否已定义
  bool defined() const {
    return impl_;
  }

  // 重置 TensorImpl 的内部指针
  void reset() {
    impl_.reset();
  }
#if defined (_MSC_VER)
  // 定义赋值运算符=（复制赋值）用于左值对象
  TensorBase& operator=(const TensorBase& x) & {
    impl_ = x.impl_;
    return *this;
  };
  // 定义移动赋值运算符=（移动赋值）用于左值对象，允许 noexcept
  TensorBase& operator=(TensorBase&& x) & noexcept {
    impl_ = std::move(x.impl_);
    return *this;
  }
#else
  // 对于非 MSVC 编译器，使用默认的复制赋值运算符=
  TensorBase& operator=(const TensorBase& x) & = default;
  // 对于非 MSVC 编译器，使用默认的移动赋值运算符=，允许 noexcept
  TensorBase& operator=(TensorBase&& x) & noexcept = default;
#endif

  // 禁止对右值进行复制赋值，因为 at::Tensor 在这里执行深度复制
  TensorBase& operator=(const TensorBase&) && = delete;
  // 禁止对右值进行移动赋值，不允许 noexcept
  TensorBase& operator=(TensorBase&&) && noexcept = delete;

  // 检查是否与另一个 TensorBase 对象相同
  bool is_same(const TensorBase& other) const noexcept {
    return impl_ == other.impl_;
  }

  // 返回当前对象的引用计数
  size_t use_count() const noexcept {
    return impl_.use_count();
  }

  // 返回当前对象的弱引用计数
  size_t weak_use_count() const noexcept {
    return impl_.weak_use_count();
  }

  // 返回当前对象的字符串表示
  std::string toString() const;

  // 返回当前对象的尺寸数组的引用
  IntArrayRef sizes() const {
    return impl_->sizes();
  }

  // 返回当前对象的符号化尺寸数组的引用
  c10::SymIntArrayRef sym_sizes() const {
    return impl_->sym_sizes();
  }

  // 返回当前对象的符号化步幅数组的引用
  c10::SymIntArrayRef sym_strides() const {
    return impl_->sym_strides();
  }

  // 返回当前对象的步幅数组的引用
  IntArrayRef strides() const {
    return impl_->strides();
  }

  // 返回当前对象的可选维度名称列表
  // 参考 ATen/NamedTensor.h 中的 impl::get_opt_names
  std::optional<DimnameList> opt_names() const {
    return impl::get_opt_names(unsafeGetTensorImpl());
  }

  // 返回当前对象的维度名称列表
  // 参考 ATen/NamedTensor.h 中的 impl::get_names
  DimnameList names() const {
    return impl::get_names(unsafeGetTensorImpl());
  }

  // 返回当前对象的维度数量（等同于 dim()）
  int64_t ndimension() const {
    return dim();
  }

  // 检查当前对象是否是连续的（按照指定的内存格式）
  bool is_contiguous(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const {
    return impl_->is_contiguous(memory_format);
  }

  // 检查当前对象是否是非重叠且稠密的
  bool is_non_overlapping_and_dense() const {
    return impl_->is_non_overlapping_and_dense();
  }

  // 建议最佳内存格式
  at::MemoryFormat suggest_memory_format(
      bool channels_last_strides_exact_match = false) const {
    // 当布局为 kStrided 时进行检查
    if (layout() == at::kStrided) {
      // 如果符合通道最后步幅的条件
      if (impl_->is_strides_like_channels_last()) {
        // 如果 channels_last_strides_exact_match 为 true 或者尺寸和步幅匹配，则使用 ChannelsLast 格式
        if (!channels_last_strides_exact_match ||
            get_channels_last_strides_2d(sizes()) == strides()) {
          return at::MemoryFormat::ChannelsLast;
        }
      }
      // 如果符合三维通道最后步幅的条件
      else if (impl_->is_strides_like_channels_last_3d()) {
        // 如果 channels_last_strides_exact_match 为 true 或者尺寸和步幅匹配，则使用 ChannelsLast3d 格式
        if (!channels_last_strides_exact_match ||
            get_channels_last_strides_3d(sizes()) == strides()) {
          return at::MemoryFormat::ChannelsLast3d;
        }
      }
    }
    // 默认情况返回连续格式
    return at::MemoryFormat::Contiguous;
  }

  // 返回当前视图占用的总字节数（不包括元数据大小）
  // 这里返回的数字不一定等同于张量实际占用的物理内存，而是连续排列时的占用
  size_t nbytes() const {
  // 检查张量的布局是否为稀疏格式，如果是，抛出错误信息说明稀疏张量不支持 nbytes 操作
  TORCH_CHECK(layout() != at::kSparse,
              "nbytes is not defined for sparse tensors. If you want the size of the constituent "
              "tensors, add the nbytes of the indices and values. If you want the size of the "
              "equivalent dense tensor, multiply numel() by element_size()");
  // 返回张量元素的总字节数
  return impl_->numel() * impl_->itemsize();
}

c10::SymInt sym_nbytes() const {
  // 检查张量的布局是否为稀疏格式，如果是，抛出错误信息说明稀疏张量不支持 nbytes 操作
  TORCH_CHECK(layout() != at::kSparse,
              "nbytes is not defined for sparse tensors. If you want the size of the constituent "
              "tensors, add the nbytes of the indices and values. If you want the size of the "
              "equivalent dense tensor, multiply numel() by element_size()");
  // 返回张量的符号表示的元素总字节数
  return impl_->sym_numel() * impl_->itemsize();
}

int64_t numel() const {
  // 返回张量的元素个数
  return impl_->numel();
}

c10::SymInt sym_numel() const {
  // 返回张量的符号表示的元素个数
  return impl_->sym_numel();
}

c10::SymInt sym_storage_offset() const {
  // 返回张量的符号表示的存储偏移量
  return impl_->sym_storage_offset();
}

// 返回单个数组元素的字节数，这是传统的 NumPy 命名方式
size_t itemsize() const {
  // 返回单个数组元素的字节数
  return impl_->itemsize();
}

// 返回单个数组元素的字节数，这是 PyTorch 的命名方式
int64_t element_size() const {
  // 返回单个数组元素的字节数，强制类型转换为 int64_t
  return static_cast<int64_t>(impl_->itemsize());
}

DispatchKeySet key_set() const {
  // 返回张量的分发键集合
  return impl_->key_set();
}

ScalarType scalar_type() const {
  // 返回张量的标量类型
  return typeMetaToScalarType(impl_->dtype());
}

bool has_storage() const {
  // 检查张量是否有有效的存储
  return defined() && impl_->has_storage();
}

const Storage& storage() const {
  // 返回张量的存储引用
  return impl_->storage();
}

bool is_alias_of(const at::TensorBase& other) const {
  // 检查张量是否是另一个张量的别名
  return impl_->storage().is_alias_of(other.storage());
}

// 将存储后端移动到基于 shm 的共享内存，以实现跨进程内存共享
//
// 注意1：此 API 的理想行为还需要进一步讨论，但目前我们倾向于保持与现有 THP 行为的一致性
// https://github.com/pytorch/pytorch/blob/4dca9bde0552afc67b5b74f4a0696fe6055709c4/torch/storage.py#L196-L212
// 因此在此不对任何内容进行断言，依赖调用者了解其操作。
//
// 注意2：目前仅提供基于 Linux fd 的 shm 支持，以简化 ATen 中的存储生命周期管理逻辑，
// 并且暂时不添加像 THP 中那样的基于文件系统的 shm 支持，因为需要额外的 GC 管理器支持以防止泄漏。
// 因此，从不支持的系统（例如 Windows）调用此函数将失败。
void share_memory_() {
  // 调用 ATen 的 share_memory_ 函数实现共享内存操作
  at::share_memory_(*this);
}

inline bool _is_zerotensor() const {
  // 检查张量是否为全零张量
  return impl_->_is_zerotensor();
}

inline void _set_zero(bool zero) const {
  // 设置张量是否为全零张量
  impl_->_set_zero(zero);
}

inline bool is_conj() const {
  // 返回一个张量是否为共轭的。
  return impl_->is_conj();
}

// 设置张量的共轭位。
// 注意：共轭位应该是一个只读字段。仅在确定需要改变时才修改它，否则可能导致不正确的行为，
// 因为共轭是一种延迟操作，我们依赖这一位来确定是否需要生成共轭。
inline void _set_conj(bool conjugate) const {
  impl_->_set_conj(conjugate);
}

inline bool is_neg() const {
  // 返回一个张量是否为负数的。
  return impl_->is_neg();
}

// 设置张量的负数位。
// 注意：负数位应该是一个只读字段。仅在确定需要改变时才修改它，否则可能导致不正确的行为，
// 因为我们依赖这一位来确定是否需要生成负数。
inline void _set_neg(bool negative) const {
  impl_->_set_neg(negative);
}

/// 返回张量的布局。
Layout layout() const {
  return impl_->layout();
}

/// 返回张量的数据类型（TypeMeta）。
caffe2::TypeMeta dtype() const {
  return impl_->dtype();
}

/// 返回张量的设备。
inline Device device() const {
  return impl_->device();
}

/// 返回张量的设备索引。
DeviceIndex get_device() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->get_device();
}

/// 返回张量是否具有 CPU 后端。
bool is_cpu() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->is_cpu();
}

/// 返回张量是否具有 CUDA 后端。
bool is_cuda() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->is_cuda();
}

/// 返回张量是否具有 IPU 后端。
bool is_ipu() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->is_ipu();
}

/// 返回张量是否具有 XPU 后端。
bool is_xpu() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->is_xpu();
}

/// 返回张量是否具有 XLA 后端。
bool is_xla() const {
  return impl_->is_xla();
}

/// 返回张量是否具有 MTIA 后端。
bool is_mtia() const {
  return impl_->is_mtia();
}

/// 返回张量是否具有 HPU 后端。
bool is_hpu() const {
  return impl_->is_hpu();
}

/// 返回张量是否具有 Lazy 后端。
bool is_lazy() const {
  return impl_->is_lazy();
}

/// 返回张量是否具有 HIP 后端。
bool is_hip() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->is_hip();
}

/// 返回张量是否具有 VE 后端。
bool is_ve() const {
  // 注意：这不是一个本地函数，以避免调度开销。
  return impl_->is_ve();
}

/// 返回张量是否具有 PrivateUse1 后端。
bool is_privateuseone() const {
  // 返回实现对象是否为私有使用的一个函数，避免调度开销。
  return impl_->is_privateuseone();
}

/// 返回一个 `Tensor` 是否具有稀疏后端。
bool is_sparse() const {
  // 返回实现对象是否为稀疏的一个函数，避免调度开销。
  return impl_->is_sparse();
}

/// 返回一个 `Tensor` 是否具有稀疏 CSR 后端。
bool is_sparse_csr() const {
  // 返回实现对象是否为稀疏 CSR 的一个函数，避免调度开销。
  return impl_->is_sparse_csr();
}

/// 返回一个 `Tensor` 是否为 mkldnn 张量。
bool is_mkldnn() const {
  // 返回实现对象是否为 mkldnn 张量的一个函数，避免调度开销。
  return impl_->is_mkldnn();
}

/// 返回一个 `Tensor` 是否为 mps 张量。
bool is_mps() const {
  // 返回实现对象是否为 mps 张量的一个函数，避免调度开销。
  return impl_->is_mps();
}

/// 返回一个 `Tensor` 是否为 maia 张量。
bool is_maia() const {
  // 返回实现对象是否为 maia 张量的一个函数，避免调度开销。
  return impl_->is_maia();
}

/// 返回一个 `Tensor` 是否为 vulkan 张量。
bool is_vulkan() const {
  // 返回实现对象是否为 vulkan 张量的一个函数，避免调度开销。
  return impl_->is_vulkan();
}

/// 返回一个 `Tensor` 是否为 metal 张量。
bool is_metal() const {
  // 返回实现对象是否为 metal 张量的一个函数，避免调度开销。
  return impl_->is_metal();
}

/// 返回一个 `Tensor` 是否具有量化后端。
bool is_quantized() const {
  // 返回实现对象是否为量化的一个函数，避免调度开销。
  return impl_->is_quantized();
}

/// 返回一个 `Tensor` 是否为元张量。元张量也可能具有其他标识。
bool is_meta() const {
  // 返回实现对象是否为元张量的一个函数。
  return impl_->is_meta();
}

/// 返回一个 `Tensor` 是否为推断张量。
bool is_inference() const {
  // 返回实现对象是否为推断张量的一个函数。
  return impl_->is_inference();
}

// 返回一个 `Tensor` 是否为 NestedTensor。
bool is_nested() const {
  // 返回实现对象是否为 NestedTensor 的一个函数。
  return impl_->is_nested();
}

/// 如果张量是量化张量，则返回其量化器。
/// TODO: 由于尚未在 native_functions.yaml 中暴露给 Python，此处待处理。
QuantizerPtr quantizer() const;

/// 返回一个 `Tensor` 是否具有任何维度名称。
bool has_names() const {
  // 如果用户使用未命名的张量，则可以在此处进行短路处理。
  // 否则，impl::has_names 尝试检索名称。
  if (!impl_->has_named_tensor_meta()) {
    return false;
  }
  return impl::has_names(unsafeGetTensorImpl());
}

/// 返回一个 `Tensor` 的维度名称数据结构。
const NamedTensorMeta* get_named_tensor_meta() const {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

NamedTensorMeta* get_named_tensor_meta() {
  return static_cast<NamedTensorMeta*>(impl_->named_tensor_meta());
}

/// 返回与此 `Tensor` 对应的 `TensorOptions`。定义在 TensorOptions.h 中。
TensorOptions options() const {
    ```py`
      // 返回一个 TensorOptions 对象，设置其数据类型、设备和布局，然后返回该对象
      return TensorOptions().dtype(dtype())
                            .device(device())
                            .layout(layout());
    }
    
    // 返回指向常量数据的指针，通过 unsafeGetTensorImpl() 获取 tensor 实现对象并返回其数据指针
    const void* const_data_ptr() const {
      return this->unsafeGetTensorImpl()->data();
    }
    
    // 返回指向可变数据的指针，通过 unsafeGetTensorImpl() 获取 tensor 实现对象并返回其可变数据指针
    void* mutable_data_ptr() const {
      return this->unsafeGetTensorImpl()->mutable_data();
    }
    
    // 返回数据的指针，这里调用 mutable_data_ptr() 获取可变数据指针
    // TODO(#97856) Make this return a const pointer. This currently
    //              returns a non-const pointer because of the large
    //              number of clients that we still want to audit before
    //              migrating to mutable_data_ptr().
    void* data_ptr() const {
      return mutable_data_ptr();
    }
    
    // 返回常量数据的指针，具体类型 T 由模板参数确定，函数未定义
    template <typename T, std::enable_if_t<!std::is_const_v<T>, int> = 0>
    const T* const_data_ptr() const;
    
    // 返回常量数据的指针，具体类型 T 由模板参数确定，函数未定义
    template <typename T, std::enable_if_t<std::is_const_v<T>, int> = 0>
    const std::remove_const_t<T>* const_data_ptr() const;
    
    // 返回可变数据的指针，具体类型 T 由模板参数确定，函数未定义
    template <typename T>
    T* mutable_data_ptr() const;
    
    // 打印函数，未定义实现，用于禁止内联
    void print() const;
    
    // 返回 CPU 上 Tensor 的 TensorAccessor，具体类型 T 和维度 N 由模板参数确定
    // 使用 const_data_ptr() 或 mutable_data_ptr() 获取数据指针，构造 TensorAccessor 对象并返回
    template<typename T, size_t N>
    TensorAccessor<T,N> accessor() const& {
      static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
      TORCH_CHECK(dim() == N, "TensorAccessor expected ", N, " dims but tensor has ", dim());
      T* ptr = nullptr;
      if constexpr (std::is_const<T>::value) {
        ptr = const_data_ptr<T>();
      } else {
        ptr = mutable_data_ptr<T>();
      }
      return TensorAccessor<T,N>(ptr,sizes().data(),strides().data());
    }
    template<typename T, size_t N>
    TensorAccessor<T,N> accessor() && = delete;
    
    // 返回 CUDA 上 Tensor 的 GenericPackedTensorAccessor，具体类型 T、维度 N 和模板参数 PtrTraits 由模板参数确定
    // 使用 const_data_ptr() 或 mutable_data_ptr() 获取数据指针，构造 GenericPackedTensorAccessor 对象并返回
    template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
    GenericPackedTensorAccessor<T,N,PtrTraits,index_t> generic_packed_accessor() const& {
      static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
      TORCH_CHECK(dim() == N, "TensorAccessor expected ", N, " dims but tensor has ", dim());
      T* ptr = nullptr;
      if constexpr (std::is_const<T>::value) {
        ptr = const_data_ptr<T>();
  }
  // 如果不是可变数据指针，则使用常规数据指针
  } else {
    ptr = mutable_data_ptr<T>();
  }
  // 返回一个通用的打包张量访问器，使用给定的指针类型和尺寸/步长数据
  return GenericPackedTensorAccessor<T,N,PtrTraits,index_t>(static_cast<typename PtrTraits<T>::PtrType>(ptr),sizes().data(),strides().data());
}

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
// 删除右值引用版本的泛型打包张量访问器
GenericPackedTensorAccessor<T,N> generic_packed_accessor() && = delete;

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
// 返回一个32位打包张量访问器的常量左值引用版本
PackedTensorAccessor32<T,N,PtrTraits> packed_accessor32() const& {
  // 检查元素数量是否小于等于 int32_t 的最大值
  TORCH_CHECK(
      impl_->numel() <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "numel needs to be smaller than int32_t max; otherwise, please use packed_accessor64");
  // 返回通用的32位打包张量访问器
  return generic_packed_accessor<T,N,PtrTraits,int32_t>();
}

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
// 删除右值引用版本的32位打包张量访问器
PackedTensorAccessor32<T,N,PtrTraits> packed_accessor32() && = delete;

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
// 返回一个64位打包张量访问器的常量左值引用版本
PackedTensorAccessor64<T,N,PtrTraits> packed_accessor64() const& {
  // 设置是否需要梯度
  impl_->set_requires_grad(requires_grad);
  // 返回当前对象自身
  return *this;
}

bool requires_grad() const {
  // 返回当前对象是否需要梯度
  return impl_->requires_grad();
}

// 下面的前向自动微分（Forward AD）API函数是低级别的，不应由终端用户使用，终端用户应使用 torch/csrc/autograd.h 中提供的API

/// 返回给定级别下此张量的前向梯度
const Tensor& _fw_grad(uint64_t level) const {
  // 调用实现对象的 _fw_grad 函数返回前向梯度
  return impl_->_fw_grad(level, *this);
}

/// 用于设置前向梯度的值
/// 注意，如果给定的新梯度与当前张量的元数据（大小/步长/存储偏移）不同，则新梯度内容将被复制到新的张量中
void _set_fw_grad(const TensorBase& new_grad, uint64_t level, bool is_inplace_op) const {
// 在派生类中注册钩子函数，接受一个函数对象作为参数，返回一个注册的位置
protected:
  unsigned _register_hook(std::function<TensorBase(const TensorBase&)> hook) const;

public:

  /// 在给定位置移除钩子函数
  void remove_hook(unsigned pos) const;

  // Variable methods
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// 返回是否为叶子节点
  bool is_leaf() const;

  /// 返回输出编号
  int64_t output_nr() const;

  /// 设置数据为新数据
  void set_data(const TensorBase & new_data) const;

  /// 返回数据
  TensorBase data() const;

  /// 返回版本号
  int64_t _version() const;

  /// 保留梯度
  void retain_grad() const;

  /// 返回是否保留梯度
  bool retains_grad() const;

  /// 设置是否需要计算梯度
  const TensorBase& requires_grad_(bool _requires_grad=true) const;

  // View Variables
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// 返回是否为视图变量
  bool is_view() const;

  /// 返回该变量所依赖的基础变量，如果不是视图则抛出 std::runtime_error
  const TensorBase& _base() const;

  // Miscellaneous
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// 返回变量的名称
  const std::string& name() const;

protected:
  /// 强制执行不变量
  void enforce_invariants();
  c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> impl_;

private:
  /// 分发连续的操作，使用指定的内存格式
  TensorBase __dispatch_contiguous(c10::MemoryFormat) const;
};

/// 获取张量所在的设备索引
inline DeviceIndex get_device(const TensorBase& self) {
  return self.get_device();
}

template <typename T>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
/// 注册钩子函数，支持返回 void 类型的钩子
auto TensorBase::register_hook(T&& hook) const -> TensorBase::hook_return_void_t<T> {
  // 如果钩子函数返回类型是 void，则返回 grad 参数
  static_assert(std::is_same_v<decltype(hook(TensorBase())), void>,
                "Expected hook to return void");
  return _register_hook([fn=std::forward<T>(hook)](const TensorBase& grad) {
    fn(grad);
    return TensorBase();
  });
}

template <typename T>
/// 注册钩子函数，支持返回 TensorBase 类型的钩子
auto TensorBase::register_hook(T&& hook) const -> TensorBase::hook_return_var_t<T> {
  return _register_hook(std::forward<T>(hook));
}

namespace detail {
// 为 Tensor 类型创建辅助函数，使用户无需直接传递 intrusive_ptr，而是自动转换为请求的 intrusive_ptr 类型
template <typename T, typename... Args>
TensorBase make_tensor_base(Args&&... args) {
  return TensorBase(c10::make_intrusive<T>(std::forward<Args>(args)...));
}

} // namespace detail

/// 从 TensorBase 中提取遗留的调度键
inline DispatchKey legacyExtractDispatchKey(const TensorBase& t) {
  return legacyExtractDispatchKey(t.key_set());
}

} // namespace at

namespace c10 {
template <>
struct MaybeOwnedTraits<at::TensorBase> {
  using owned_type = at::TensorBase;
  using borrow_type = at::TensorBase;

  static borrow_type createBorrow(const owned_type& from) {
    // 注意：这可以实现为不使用特殊的 unsafe_borrow_t Tensor 构造函数
    //
    // 返回一个 borrow_type 对象，利用 intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim 方法从 from.unsafeGetTensorImpl() 中重建
    return borrow_type(c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(from.unsafeGetTensorImpl()));
}
  // 返回一个借用类型的实例，使用从另一个借用类型实例构造的数据，省略了空指针检查
  return borrow_type(borrow_type::unsafe_borrow_t{}, from);
}

static void assignBorrow(borrow_type& lhs, const borrow_type& rhs) {
  // 释放左操作数的张量实现，允许在不进行空指针检查的情况下直接赋值
  lhs.unsafeReleaseTensorImpl();
  // 同上述说明：可以使用公共 API 来实现与 createBorrow() 类似的方法，但这会影响内联优化
  lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
}

static void destroyBorrow(borrow_type& toDestroy) {
  // 销毁借用类型对象，通过释放张量实现来模拟"泄漏"，尽管引用计数已经是 +0
  toDestroy.unsafeReleaseTensorImpl();
}

static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
  // 从借用类型获取其所引用的所有权类型的常量引用
  return borrow;
}

static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
  // 从借用类型获取其所引用的所有权类型的指针
  return &borrow;
}

static bool debugBorrowIsValid(const borrow_type& /*borrow*/) {
  // 调试用函数：始终返回 true，表明借用类型有效
  return true;
}
};

// 特化模板，指定 ExclusivelyOwnedTraits 类型为 at::TensorBase 的派生类
template <>
struct ExclusivelyOwnedTraits<at::TensorBase> : public c10::ExclusivelyOwnedTensorTraits<at::TensorBase> {};

} // namespace c10

namespace at {

// 从可选的 TensorBase 类型借用对象，返回一个 MaybeOwned<TensorBase> 对象
inline c10::MaybeOwned<TensorBase> borrow_from_optional_tensor(
    const std::optional<TensorBase>& opt) {
  return opt.has_value()
    ? c10::MaybeOwned<TensorBase>::borrowed(*opt)  // 如果 optional 中有值，则借用该值
    : c10::MaybeOwned<TensorBase>::owned(std::in_place);  // 如果 optional 中没有值，则创建一个新的对象
}

// 对 TensorBase 类型的对象执行 expect_contiguous 操作，返回 MaybeOwned<TensorBase> 对象
inline c10::MaybeOwned<TensorBase> TensorBase::expect_contiguous(MemoryFormat memory_format) const & {
  if (is_contiguous(memory_format)) {
    return c10::MaybeOwned<TensorBase>::borrowed(*this);  // 如果对象是连续的，则直接借用当前对象
  } else {
    return c10::MaybeOwned<TensorBase>::owned(__dispatch_contiguous(memory_format));  // 否则，调用 __dispatch_contiguous 创建一个新的对象
  }
}

namespace symint {

// 如果 T 是 c10::SymInt 类型，则返回 TensorBase 对象的符号化尺寸引用
template <typename T>
using enable_if_symint = std::enable_if_t<std::is_same_v<T, c10::SymInt>>;
template <typename T>
c10::SymIntArrayRef sizes(const TensorBase& t) { return t.sym_sizes(); }

// 如果 T 是 int64_t 类型，则返回 TensorBase 对象的尺寸引用
template <typename T>
using enable_if_int = std::enable_if_t<std::is_same_v<T, int64_t>>;
template <typename T>
IntArrayRef sizes(const TensorBase& t) { return t.sizes(); }

// 如果 T 是 c10::SymInt 类型，则返回 TensorBase 对象的指定维度的符号化尺寸
template <typename T>
c10::SymInt size(const TensorBase& t, int64_t dim) { return t.sym_size(dim); }

// 如果 T 是 int64_t 类型，则返回 TensorBase 对象的指定维度的尺寸
template <typename T>
int64_t size(const TensorBase& t, int64_t dim) { return t.size(dim); }

// 如果 T 是 c10::SymInt 类型，则返回 TensorBase 对象的符号化步长引用
template <typename T>
c10::SymIntArrayRef strides(const TensorBase& t) { return t.sym_strides(); }

// 如果 T 是 int64_t 类型，则返回 TensorBase 对象的步长引用
template <typename T>
IntArrayRef strides(const TensorBase& t) { return t.strides(); }

// 如果 T 是 c10::SymInt 类型，则返回 TensorBase 对象的符号化元素数
template <typename T>
c10::SymInt numel(const TensorBase& t) { return t.sym_numel(); }

// 如果 T 是 int64_t 类型，则返回 TensorBase 对象的元素数
template <typename T>
int64_t numel(const TensorBase& t) { return t.numel(); }

} // namespace symint

} // namespace at
```