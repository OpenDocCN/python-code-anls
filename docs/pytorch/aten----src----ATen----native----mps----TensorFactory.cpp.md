# `.\pytorch\aten\src\ATen\native\mps\TensorFactory.cpp`

```py
// 2022年苹果公司版权声明

#include <ATen/ATen.h>  // 引入 ATen 库，提供张量操作的基本功能
#include <ATen/Tensor.h>  // 引入 ATen 中的 Tensor 类定义
#include <ATen/Utils.h>  // 引入 ATen 提供的一些实用工具函数
#include <torch/library.h>  // 引入 Torch 库，支持与 PyTorch 的互操作
#include <ATen/mps/EmptyTensor.h>  // 引入 ATen 中 MPS 模块的空张量定义
#include <ATen/mps/MPSDevice.h>  // 引入 ATen 中 MPS 设备的定义
#include <ATen/native/Resize.h>  // 引入 ATen 中张量调整大小的实现
#include <ATen/native/ResizeCommon.h>  // 引入 ATen 中通用的张量调整大小操作
#include <ATen/native/mps/Copy.h>  // 引入 ATen 中 MPS 模块的复制操作
#include <ATen/native/mps/TensorFactory.h>  // 引入 ATen 中 MPS 模块的张量工厂函数
#include <ATen/Dispatch.h>  // 引入 ATen 中的调度器实现

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>  // 引入 ATen 中的函数定义
#include <ATen/NativeFunctions.h>  // 引入 ATen 中的原生函数定义
#endif
#include <ATen/ops/_efficientzerotensor_native.h>  // 引入 ATen 中的高效零张量操作

namespace at::native {

// 以 MPS 设备为目标，可能调整张量的存储大小
static inline void maybe_resize_storage_mps(TensorImpl* self, uint64_t new_size) {
  if (new_size == 0) {  // 如果新大小为0，则直接返回，不进行调整
    return;
  }

  auto storage = self->storage().unsafeGetStorageImpl();  // 获取张量的存储实现
  if (!storage) {
    TORCH_CHECK(false, "Tensor: invalid null storage");  // 如果存储为空，则抛出异常
  }
  uint64_t new_size_bytes = (new_size + self->storage_offset()) * self->dtype().itemsize();  // 计算新的字节大小
  if (new_size_bytes > self->storage().nbytes()) {  // 如果新的字节大小大于当前存储的字节数
    if (new_size_bytes == 0) {
      // 如果新字节大小为0，则设置数据指针为空，并且设置字节数为0
      storage->set_data_ptr_noswap(at::DataPtr(nullptr, at::Device(at::DeviceType::MPS, 0)));
      storage->set_nbytes(0);
    } else {
      // 分配新的数据指针，并拷贝已有数据（如果存在）
      at::DataPtr new_data = storage->allocator()->allocate(new_size_bytes);
      size_t copy_capacity = std::min<size_t>(new_size_bytes, storage->nbytes());
      if (storage->data() && copy_capacity > 0) {
        at::native::mps::copy_blit_mps(new_data.get(), storage->data(), copy_capacity);
      }
      // 替换数据指针
      storage->set_data_ptr_noswap(std::move(new_data));
      storage->set_nbytes(new_size_bytes);
    }
  }
}

// 以 MPS 设备为目标，调整张量的大小实现
inline TensorImpl* resize_impl_mps_(
    TensorImpl* self,
    IntArrayRef size,
    std::optional<IntArrayRef> stride,
    bool device_guard = true) {
  if (self->sizes() == size && (!stride || self->strides() == stride)) {
    return self;  // 如果当前大小与目标大小相同，则直接返回张量本身
  }

  int64_t storage_size = 1;
  if (stride) {
    self->set_sizes_and_strides(size, *stride);  // 设置新的大小和步长
    storage_size = storage_size_for(size, *stride);  // 计算存储大小
  } else {
    self->set_sizes_contiguous(size);  // 设置新的连续大小
    storage_size = self->numel();  // 获取张量元素数
  }
  maybe_resize_storage_mps(self, storage_size);  // 可能调整存储大小

  return self;  // 返回调整后的张量实现
}

// 使用 MPS 设备创建空张量
Tensor empty_mps(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {

  return at::detail::empty_mps(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);  // 调用内部函数创建空张量
}

// 使用 MPS 设备创建空步长张量
Tensor empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
```  
// 声明一个名为pin_memory_opt的可选布尔值参数，表示是否启用内存固定选项


  check_size_nonnegative(size);
```py  
// 调用函数check_size_nonnegative，用于检查尺寸参数size是否为非负数


  // empty memory formatempty
  auto t = at::native::empty_mps(
      {0},
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt);
```  
// 调用at::native::empty_mps函数创建一个空张量t，该函数接受参数{0}表示张量的形状为空，dtype_opt表示数据类型选项，layout_opt表示布局选项，device_opt表示设备选项，pin_memory_opt表示内存固定选项


  resize_impl_mps_(t.unsafeGetTensorImpl(), size, stride);
```py  
// 调用resize_impl_mps_函数，用于重新设置张量t的实现，参数包括张量实现、尺寸和步长


  return t;
```  
// 返回创建并调整大小后的张量t作为函数的结果
}

// 重新调整张量大小，可能改变内存布局
const Tensor& resize_mps_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 如果张量有命名维度，则调用带命名维度的重调整函数
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  // 获取旧存储空间的字节数
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  // 调用 MPS 特定的重调整函数，不设置步长
  resize_impl_mps_(self_, size, /*strides=*/c10::nullopt);
  // 如果指定了内存格式，则设置新的内存格式
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    // 检查是否支持指定的内存格式
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    // 重新调整张量的内存布局以适应指定的内存格式
    self_->empty_tensor_restride(memory_format);
  }
  // 见注释 [启用确定性操作]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    // 在启用确定性算法和填充未初始化内存时，填充调整后的张量以保持确定性
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  // 返回调整大小后的张量
  return self;
}

// 设置 MPS 专用的零张量
Tensor& set_mps_(Tensor& result) {
  // 获取结果张量的数据类型
  caffe2::TypeMeta dtype = result.dtype();
  // 使用 MPS 分配器创建一个空的存储空间
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::mps::GetMPSAllocator(),
      true);
  // 将结果张量设置为使用新的存储空间
  result.set_(storage, 0, {0}, {});
  // 内部断言，检查设置后张量的数据类型与之前是否一致
  TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  // 返回设置后的结果张量
  return result;
}

// 设置 MPS 专用的存储空间
Tensor& set_storage_mps_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  // 检查设置新存储空间的合法性
  checkSetStorage(result, storage, storage_offset, size, stride);
  // 设置结果张量的存储偏移量
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  // 如果指定了步长，则将其作为可选参数传递给重调整函数；否则传递空值
  std::optional<IntArrayRef> stride_opt = stride.data() != nullptr ?
                                          std::optional<IntArrayRef>(stride) : c10::nullopt;
  // 调用 MPS 特定的重调整函数，设置新的存储空间和可能的步长
  at::native::resize_impl_mps_(result.unsafeGetTensorImpl(), size, stride_opt);
  // 返回设置后的结果张量
  return result;
}

// 创建一个 MPS 专用的高效零张量
Tensor _efficientzerotensor_mps(IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
    // 获取设备信息
    auto device_ = device_or_default(device);
    // 使用零张量分配器创建一个 MPS 专用的零张量
    auto allocator = at::native::ZeroTensorAllocator(device_);
    // 获取数据类型信息
    auto dtype_ = dtype_or_default(dtype);
    // 设置调度键集合为 MPS 和 ZeroTensor
    auto zero_ks = at::DispatchKeySet(c10::DispatchKey::MPS) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
    // 调用通用的空张量创建函数，使用指定的分配器和调度键，设置数据类型和布局（可选参数为空）
    auto out = at::detail::empty_generic(size, &allocator, zero_ks, dtype_, c10::nullopt);
    // 返回创建的零张量
    return out;
}

} // namespace at::native
```