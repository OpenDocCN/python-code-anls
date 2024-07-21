# `.\pytorch\torch\csrc\lazy\core\tensor_impl.cpp`

```
// 包含 Torch 中用于 Lazy 模块的张量实现的头文件
#include <torch/csrc/lazy/core/tensor_impl.h>

// 包含 C10 核心功能的分配器、标量类型等头文件
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

// 包含 Torch 中 Lazy 模块的 IR 构建器和张量工具的头文件
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/tensor_util.h>

// Torch 的命名空间
namespace torch {
// Lazy 模块的命名空间
namespace lazy {
// 匿名命名空间，用于定义本文件内部使用的类和函数

// LTCGuardImpl 被 CompositeExplicitAutograd 操作或 eager fallback 使用，
// 确保在 guard 生命周期内的特定张量位于相同的设备上。例如，在 RegisterCompositeExplicitAutograd.cpp 中，
// 检查每个操作的输出是否与提供的 TensorOptions 上的设备相同。有关详细信息，请参见 DeviceGuard.h。
// 对于具有 LTC 本机函数实现的操作，此 guard 被省略。
thread_local c10::Device g_device(c10::DeviceType::Lazy);

// LTCGuardImpl 类继承自 c10::impl::DeviceGuardImplInterface 接口
struct LTCGuardImpl : public c10::impl::DeviceGuardImplInterface {
  // 返回设备类型为 Lazy
  at::DeviceType type() const override {
    return at::DeviceType::Lazy;
  }

  // 交换当前设备并返回旧设备
  c10::Device exchangeDevice(c10::Device device) const override {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::Lazy);
    auto old_device = g_device;
    g_device = device;
    return old_device;
  }

  // 返回当前设备
  c10::Device getDevice() const override {
    return g_device;
  }

  // 设置当前设备
  void setDevice(c10::Device device) const override {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::Lazy);
    g_device = device;
  }

  // 不安全地设置当前设备
  void uncheckedSetDevice(c10::Device device) const noexcept override {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::Lazy);
    g_device = device;
  }

  // 返回给定设备的流
  c10::Stream getStream(c10::Device device) const noexcept override {
    TORCH_INTERNAL_ASSERT(device.type() == c10::DeviceType::Lazy);
    return c10::Stream(c10::Stream::DEFAULT, device);
  }

  // 交换当前流并返回旧流
  c10::Stream exchangeStream(c10::Stream _unused) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, g_device);
  }

  // 返回设备数量
  c10::DeviceIndex deviceCount() const noexcept override {
    // 当 autograd 初始化其设备池时调用此函数，无论我们是否事先注册了后端。
    if (!hasBackend()) {
      return 0;
    }

    return getBackend()->GetBackendDevices().size();
  }
};

// 在 C10 中注册 Lazy 设备的 Guard 实现
C10_REGISTER_GUARD_IMPL(Lazy, LTCGuardImpl);

} // namespace lazy
} // namespace torch

// LTCTensorImpl 类的构造函数，接受 LazyTensorPtr 类型的参数
LTCTensorImpl::LTCTensorImpl(const LazyTensorPtr& tensor)
    : LTCTensorImpl(LazyTensor(*tensor)) {}

// LTCTensorImpl 类的构造函数，接受 LazyTensor 类型的参数
LTCTensorImpl::LTCTensorImpl(const LazyTensor& tensor)
    : LTCTensorImpl(LazyTensor(tensor)) {}

// LTCTensorImpl 类的构造函数，接受 LazyTensor&& 类型的参数
LTCTensorImpl::LTCTensorImpl(LazyTensor&& tensor)
    : c10::TensorImpl(
          c10::DispatchKeySet{
              c10::DispatchKey::Lazy,
              c10::DispatchKey::AutogradLazy},
          c10::scalarTypeToTypeMeta(tensor.dtype()),
          backendDeviceToAtenDevice(tensor.GetDevice())),

创建一个 `c10::TensorImpl` 对象，使用给定的调度键集合包括 `c10::DispatchKey::Lazy` 和 `c10::DispatchKey::AutogradLazy`，以及从 `tensor.dtype()` 转换得到的类型元数据 (`c10::scalarTypeToTypeMeta(tensor.dtype())`)，以及从 `tensor.GetDevice()` 转换得到的后端设备到 ATen 设备的映射 (`backendDeviceToAtenDevice(tensor.GetDevice())`)。


      tensor_(c10::make_intrusive<LazyTensor>(std::move(tensor))) {

使用 `c10::make_intrusive<LazyTensor>(std::move(tensor))` 创建一个懒惰计算张量 (`LazyTensor`)，并将其作为成员变量 `tensor_` 的初始化值。


  set_custom_sizes_strides(SizesStridesPolicy::CustomSizes);

调用 `set_custom_sizes_strides` 方法，将 `SizesStridesPolicy::CustomSizes` 作为参数，用于设置自定义的尺寸和步幅策略。
}

// 设置 LTCTensorImpl 的 tensor_ 成员变量为给定的 LazyTensorPtr 对象，并将 generation_ 设为 0
void LTCTensorImpl::set_tensor(const LazyTensorPtr& lazy_tensor) {
    tensor_ = c10::make_intrusive<LazyTensor>(*lazy_tensor);
    generation_ = 0;
}

// 创建 LTCTensorImpl 的浅拷贝并分离，返回一个 c10::intrusive_ptr<c10::TensorImpl> 对象
c10::intrusive_ptr<c10::TensorImpl> LTCTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<LTCTensorImpl>(tensor_);
  // 复制张量的元数据
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

// 创建 LTCTensorImpl 的浅拷贝并分离，使用移动语义的 version_counter
c10::intrusive_ptr<c10::TensorImpl> LTCTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<LTCTensorImpl>(tensor_);
  // 复制张量的元数据
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  return impl;
}

// 从给定的 TensorImpl 指针浅拷贝到当前对象，同时复制元数据
void LTCTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl>& impl) {
  LTCTensorImpl* ltc_impl = dynamic_cast<LTCTensorImpl*>(impl.get());
  TORCH_INTERNAL_ASSERT(ltc_impl);
  // 复制张量的元数据
  copy_tensor_metadata(
      /*src_impl=*/ltc_impl,
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
  // 将 ltc_impl 的 tensor_ 浅拷贝到当前对象的 tensor_ 中
  ltc_impl->tensor_->ShallowCopyTo(tensor_);
  generation_ = 0;
}

// 返回包含自定义步长的 SymIntArrayRef
c10::SymIntArrayRef LTCTensorImpl::sym_strides_custom() const {
  return c10::fromIntArrayRefKnownNonNegative(strides_custom());
}

// 返回包含自定义尺寸的 SymIntArrayRef
c10::SymIntArrayRef LTCTensorImpl::sym_sizes_custom() const {
  return c10::fromIntArrayRefKnownNonNegative(sizes_custom());
}

// 返回自定义元素数量的 SymInt
c10::SymInt LTCTensorImpl::sym_numel_custom() const {
  return numel_custom();
}

// 设置张量的尺寸属性，如大小、步长等
void LTCTensorImpl::setup_size_properties() {
  size_t generation = tensor_->generation();
  if (generation != generation_) {
    // 填充基本的维度数据成员，供基类实现在其 API 中使用
    auto shape = tensor_->shape();
    // 由于我们重写了 sizes()，因此无法调用 refresh_numel()
    numel_ = shape.Get().numel();
    sizes_and_strides_.set_sizes(shape.Get().sizes());
    // 由于我们重写了 sizes()，因此无法调用 empty_tensor_restride(c10::MemoryFormat::Contiguous)
    std::vector<int64_t> updated_strides;
    updated_strides = ComputeArrayStrides(shape.Get().sizes());
    for (const auto i : c10::irange(updated_strides.size())) {
      sizes_and_strides_.stride_at_unchecked(i) = updated_strides[i];
    }
    generation_ = generation;
  }
}

// 返回自定义尺寸的 IntArrayRef
at::IntArrayRef LTCTensorImpl::sizes_custom() const {
  // 强制将 const_cast 使用在 LTCTensorImpl 上，并设置尺寸属性
  const_cast<LTCTensorImpl*>(this)->setup_size_properties();
  return sizes_default();
}
at::IntArrayRef LTCTensorImpl::strides_custom() const {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  // 强制去除常量属性，以确保可以在常量方法中修改对象状态
  const_cast<LTCTensorImpl*>(this)->setup_size_properties();
  // 调用默认的步长计算方法并返回结果
  return strides_default();
}

int64_t LTCTensorImpl::dim_custom() const {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  // 强制去除常量属性，以确保可以在常量方法中修改对象状态
  const_cast<LTCTensorImpl*>(this)->setup_size_properties();
  // 返回默认维度数
  return dim_default();
}

int64_t LTCTensorImpl::numel_custom() const {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  // 强制去除常量属性，以确保可以在常量方法中修改对象状态
  const_cast<LTCTensorImpl*>(this)->setup_size_properties();
  // 返回默认元素数量
  return numel_default();
}

int64_t LTCTensorImpl::storage_offset_custom() const {
  // 返回存储偏移量，对于自定义实现，默认为0
  return 0;
}

bool LTCTensorImpl::is_strides_like_custom(
    c10::MemoryFormat memory_format) const {
  // 断言不是连续内存格式，因为是自定义的特殊情况
  TORCH_INTERNAL_ASSERT(memory_format != at::MemoryFormat::Contiguous);
  // 返回false，表明不是类似连续内存格式的张量
  return false;
}

bool LTCTensorImpl::is_non_overlapping_and_dense_custom() const {
  // 作为PyTorch核心问题的临时修复，暂时返回false
  // 参考 https://github.com/pytorch/xla/pull/2682
  return false;
}

bool LTCTensorImpl::is_contiguous_custom(c10::MemoryFormat _unused) const {
  // TODO(ezyang): 我认为这个分支实际上是不必要的
  // TODO(ezyang): 我认为这个逻辑是错误的，我们应该传递内存格式吗？
  if (tensor_->CurrentTensorData()) {
    // 如果有当前张量数据，则检查其是否连续
    return tensor_->CurrentTensorData()->is_contiguous();
  }
  // 只检查存储是否已经是连续的
  TORCH_CHECK(is_contiguous_, "Non-contiguous storage for lazy tensor");
  // TODO: 我认为逻辑是错误的，我们应该在返回true之前检查请求的内存格式
  // 暂时返回true，因为没有更精确的逻辑
  return true;
}

} // namespace lazy
} // namespace torch
```