# `.\pytorch\aten\src\ATen\native\Resize.cpp`

```py
// 定义预处理宏，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 包含必要的头文件，用于张量操作
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>

// 根据情况选择是否包含各个操作符的头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/resize_as_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/resize.h>
#include <ATen/ops/_resize_output.h>
#include <ATen/ops/_resize_output_native.h>
#endif

// 定义命名空间 at::native
namespace at::native {

// 检查是否需要进行张量调整大小的函数模板
// T 表示要检查的形状类型
template <typename T>
bool _resize_output_check(const Tensor& output, ArrayRef<T> shape) {
  // 检测张量是否需要调整大小，比较当前形状与目标形状是否一致
  if (at::symint::sizes<T>(output).equals(shape)) {
    return false;
  }
  // 如果输出张量非空且形状不匹配，则发出警告信息
  if (at::symint::numel<T>(output) != 0) {
    TORCH_WARN(
      "An output with one or more elements was resized since it had ",
      "shape ", at::symint::sizes<T>(output), ", which does not match the required ",
      "output shape ", shape, ". ",
      "This behavior is deprecated, and in a future PyTorch release outputs ",
      "will not be resized unless they have zero elements. You can explicitly ",
      "reuse an out tensor t by resizing it, inplace, to zero elements with ",
      "t.resize_(0).");
  }
  return true;
}

// 检查是否需要进行张量调整大小，针对普通整数数组形状
bool resize_output_check(const Tensor& output, IntArrayRef shape) {
  return _resize_output_check(output, shape);
}

// 检查是否需要进行张量调整大小，针对符号整数数组形状
bool resize_output_check_symint(const Tensor& output, SymIntArrayRef shape) {
  return _resize_output_check(output, shape);
}

// 调整输出张量大小的实际操作函数，针对普通整数数组形状
static void native_resize_(const Tensor& output, IntArrayRef shape) {
  native::resize_(output, shape);
}

// 调整输出张量大小的实际操作函数，针对符号整数数组形状
static void native_resize_(const Tensor& output, SymIntArrayRef shape) {
  native::resize__symint(output, shape);
}

// 实现张量调整大小的主要函数模板
template <typename T>
bool _resize_output(const Tensor& output, ArrayRef<T> shape) {
  if (_resize_output_check<T>(output, shape)) {
    // 针对 CPU 并且不是类似张量子类的情况，调用快速路径的调整大小函数
    if (output.is_cpu() && !isTensorSubclassLike(output)) {
      native_resize_(output, shape);
    } else {
      at::symint::resize_<T>(output, shape);
    }
    return true;
  } else {
    return false;
  }
}

// 调整输出张量大小的主函数，针对普通整数数组形状
bool resize_output(const Tensor& output, IntArrayRef shape) {
  return _resize_output(output, shape);
}

// 调整输出张量大小的主函数，针对符号整数数组形状
bool resize_output_symint(const Tensor& output, SymIntArrayRef shape) {
  return _resize_output(output, shape);
}

// 调整输出张量大小的函数，返回调整后的自身张量，针对普通整数数组形状和指定设备
const Tensor& _resize_output_(const Tensor& self, IntArrayRef shape, c10::Device device) {
  // 检查输出张量是否在正确的设备上
  TORCH_CHECK(self.device() == device, "out Tensor doesn't have the correct device set");
  // 调整输出张量的大小
  at::native::resize_output(self, shape);
  return self;
}

} // namespace at::native
// 调整给定存储的大小（以字节为单位），同时进行必要的内存复制和重排列操作
void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes) {
  // 检查存储是否可调整大小，如果不可调整，则抛出错误信息
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");

  // 分配新的数据指针，如果大小不为0
  at::DataPtr new_data;
  if (size_bytes != 0) {
    new_data = storage->allocator()->allocate(size_bytes);
  }

  // 获取旧数据指针和容量
  const at::DataPtr& old_data = storage->data_ptr();
  const auto old_capacity = storage->nbytes();

  // 计算实际复制的容量，即新旧数据指针大小的较小值
  const auto copy_capacity = std::min(size_bytes, old_capacity);

  // 如果旧数据指针不为空且复制容量大于0，则进行内存复制
  if (old_data != nullptr && copy_capacity > 0) {
    memcpy(new_data.get(), old_data.get(), copy_capacity);
  }

  // 更新存储的数据指针，不交换新旧数据指针
  storage->set_data_ptr_noswap(std::move(new_data));

  // 更新存储的字节大小
  storage->set_nbytes(size_bytes);
}

// 直接调用 SparseTensor.cpp 中的稀疏实现函数。
// 此处不需要动态分派，因此未将此函数包含在 native_functions.yaml 中
const Tensor& resize_as_sparse_(const Tensor& self, const Tensor& src);

// TODO(VitalyFedyunin): 移动到 HTML 文档中。
//
// `resize_as_` 操作的输出张量步幅由输入张量的步幅和 memory_format 参数的值定义。
//
// 如果省略了 memory_format 参数并且输入张量与输出张量形状相同，则输出张量的步幅将保持不变。
// 如果形状不同，则步幅将设置为连续。
//
// 如果 memory_format 等于 MemoryFormat::Contiguous（torch.contiguous_format），
// 输出张量将具有连续的步幅。
//
// 如果 memory_format 等于 MemoryFormat::ChannelsLast（torch.channels_last）且输入张量为4D，
// 输出张量将采用通道最后的内存布局。
//
// 如果 memory_format 等于 MemoryFormat::Preserve（torch.preserve_format），
// 输出张量的步幅将由输入张量的步幅定义，遵循内存格式保留规则：
// - 如果输入张量的步幅采用通道最后的格式，则输出张量将采用通道最后的内存布局。
// - 否则，输出张量将具有连续的内存布局。
//
const Tensor& resize_as_(
    const Tensor& self,
    const Tensor& the_template,
    std::optional<MemoryFormat> optional_memory_format) {
  // 如果输入张量和模板张量均为稀疏张量，则直接调用稀疏实现的 resize_as_ 函数
  if (self.is_sparse() && the_template.is_sparse()) {
    TORCH_CHECK(
        !optional_memory_format.has_value(),
        "Unsupported memory format for sparse tensor resize_as_ :",
        optional_memory_format.value());
    return at::native::resize_as_sparse_(self, the_template);
  }
  
  // 否则，调整输入张量的大小以匹配模板张量的大小
  const Tensor& result = self.resize_(the_template.sizes());
  
  // 如果指定了内存格式参数，根据参数值重新调整张量的步幅
  if (optional_memory_format.has_value()) {
    auto memory_format = optional_memory_format.value();
    if (memory_format == MemoryFormat::Preserve) {
      memory_format = the_template.suggest_memory_format();
    }
    self.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  }
  
  // 将结果张量的命名信息传播到模板张量
  namedinference::propagate_names(result, the_template);
  
  // 返回调整大小后的结果张量
  return result;
}
void resize_bytes_meta(StorageImpl* storage, c10::SymInt size_bytes) {
  // 检查存储是否可调整大小，如果不可调整，则抛出错误信息
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  // 设置存储的字节数为给定的大小
  storage->set_nbytes(std::move(size_bytes));
}

static void maybe_resize_storage_meta(TensorImpl* self, c10::SymInt new_size_bytes) {
  // 如果张量元素数为0，则直接返回，因为无法调整大小
  if (self->sym_numel() == 0) {
    return;
  }

  // 获取不安全的存储引用
  const Storage& storage = self->unsafe_storage();
  // 如果存储不存在，则断言出错（只能是Caffe2，尚未实现）
  if (!storage) {
    TORCH_INTERNAL_ASSERT(0, "NYI, this should only be Caffe2");
  } else if (new_size_bytes > storage.sym_nbytes()) {
    // 如果新的大小大于存储的当前字节数，则调整存储的字节数
    resize_bytes_meta(storage.unsafeGetStorageImpl(), std::move(new_size_bytes));
  }
}

static void _maybe_resize_storage(TensorImpl* self, int64_t new_size_bytes) {
  // 调用CPU版本的调整存储大小函数
  maybe_resize_storage_cpu(self, new_size_bytes);
}

static void _maybe_resize_storage(TensorImpl* self, c10::SymInt new_size_bytes) {
  // 如果张量在CPU上，则调用CPU版本的调整存储大小函数
  if (self->is_cpu()) {
    maybe_resize_storage_cpu(self, new_size_bytes.expect_int());
    return;
  }
  // 断言张量是元数据类型
  TORCH_INTERNAL_ASSERT(self->is_meta());
  // 调用元数据版本的调整存储大小函数
  maybe_resize_storage_meta(self, std::move(new_size_bytes));
}

template <typename T>
TensorImpl* _resize_impl_(
    TensorImpl* self,
    ArrayRef<T> size,
    at::OptionalArrayRef<T> stride,
    bool resize_storage) {
  // 如果张量已经具有指定的大小和步长，则直接返回自身
  if (self->generic_sizes<T>() == size && (!stride || self->generic_strides<T>() == stride.value())) {
    return self;
  }

  // 计算张量元素的字节大小、存储偏移量等
  const auto itemsize = self->dtype().itemsize();
  const auto storage_offset = self->generic_storage_offset<T>();
  T storage_size = T(1);
  if (stride) {
    // 设置张量的大小和步长，并计算存储的字节数
    self->set_sizes_and_strides(size, *stride);
    storage_size = at::detail::computeStorageNbytes(
        size, *stride, itemsize, storage_offset);
  } else {
    // 设置张量的大小（连续存储情况下），并计算存储的字节数
    self->generic_set_sizes_contiguous(size);
    storage_size = at::detail::computeStorageNbytesContiguous(
        size, itemsize, storage_offset);
  }

  // 如果需要调整存储大小，则调用调整存储大小的函数
  if (resize_storage) {
    _maybe_resize_storage(self, std::move(storage_size));
  }

  // 返回调整后的张量实现
  return self;
}

TensorImpl* resize_impl_cpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool resize_storage) {
  // 调用通用的调整大小实现函数，针对CPU版本
  return _resize_impl_(self, size, stride, resize_storage);
}

template <typename T>
const Tensor& _resize_(
    const Tensor& self,
    ArrayRef<T> size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 获取不安全的张量实现指针
  auto* self_ = self.unsafeGetTensorImpl();
  // 获取当前存储的字节数，如果不存在存储，则为-1
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().sym_nbytes().maybe_as_int().value_or(-1) : 0;
  // 调用调整大小的实现函数，强制为真（即总是调整存储大小）
  _resize_impl_<T>(self_, size, /*strides=*/c10::nullopt, true);
  // 如果指定了内存格式，检查是否支持该内存格式
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    # 调用C++对象self_的方法，使用给定的内存格式来重构空张量的步长信息
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  # 检查全局上下文中是否启用了确定性操作，并且需要在初始化内存时填充确定性的值，
  # 同时旧存储空间大小不为-1（即旧存储空间有效）
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory() && old_storage_nbytes != -1)) {
    # 调用本地实现函数，根据旧存储空间大小填充张量，保证操作的确定性
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  # 返回更新后的自身张量对象
  return self;
}

namespace at::native {

const Tensor& resize_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 如果张量有命名，则调用 resize_named_tensor_ 处理
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  // 否则调用普通的 _resize_ 处理
  return _resize_(self, size, optional_memory_format);
}

const Tensor& resize__symint(
    const Tensor& self,
    c10::SymIntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // 内部断言，确保张量没有命名
  TORCH_INTERNAL_ASSERT(!self.has_names())
  // 调用 _resize_ 处理，用于处理 c10::SymIntArrayRef 类型的尺寸
  return _resize_(self, size, optional_memory_format);
}

void resize_bytes_nocuda(const Storage& storage, c10::SymInt newsize) {
  // 处理除了 cuda 设备类型的所有设备
  c10::DeviceType device_type = storage.device_type();
  if (device_type == at::kCPU) {
    // 调用 CPU 设备的 resize_bytes_cpu 处理
    at::native::resize_bytes_cpu(storage.unsafeGetStorageImpl(), newsize.expect_int());
  } else if (device_type == at::kMeta) {
    // 调用 Meta 设备的 resize_bytes_meta 处理
    at::native::resize_bytes_meta(storage.unsafeGetStorageImpl(), newsize);
  } else if (device_type == at::kPrivateUse1) {
    // 调用 PrivateUse1 设备的 resizePrivateUse1Bytes 处理
    at::GetPrivateUse1HooksInterface()->resizePrivateUse1Bytes(
        storage, newsize.expect_int());
  } else if (device_type == at::kXPU || device_type == at::kHPU) {
    // XPU 或 HPU 设备的处理分支
    ptrdiff_t size_bytes_i = newsize.expect_int();
    // 检查 size_bytes_i 是否会溢出 int64_t 类型
    TORCH_CHECK(
        !c10::overflows<int64_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a int64_t");
    const auto size_bytes = static_cast<int64_t>(size_bytes_i);
    // 获取原始数据指针
    void* original_data_ptr = storage.data_ptr().get();

    // 创建一个与 storage 关联的新张量，设备类型和数据类型为指定设备类型和字节类型
    auto src_option =
        c10::TensorOptions().device(storage.device()).dtype(at::kByte);
    auto src_tensor = at::empty({0}, src_option).set_(storage);
    // 调整新张量的大小为 size_bytes
    src_tensor.resize_({size_bytes});

    // 当使用 resize_ 替换 resize_bytes_xxx 时，在某些情况下，原始数据指针仍然会被返回，
    // 这与 resize_bytes_xxx 的行为不一致。对于这些情况，需要额外的内存复制和存储更新。
    if (original_data_ptr == src_tensor.storage().data_ptr().get()) {
      // 创建一个新的张量并将数据复制到新张量
      auto new_tensor = at::empty(src_tensor.sizes(), src_tensor.options());
      new_tensor.copy_(src_tensor);
      // 更新 storage 的数据指针
      storage.set_data_ptr_noswap(
          std::move(new_tensor.storage().mutable_data_ptr()));
      // 设置 storage 的分配器
      storage.unsafeGetStorageImpl()->set_allocator(
          new_tensor.storage().unsafeGetStorageImpl()->allocator());
      // 设置 storage 的字节大小
      storage.set_nbytes(new_tensor.storage().nbytes());
    }
  } else {
    // 如果设备类型不是已知类型，则抛出错误
    TORCH_CHECK(
        false,
        "UntypedStorage.resize_: got unexpected device type ",
        device_type);
  }
}

} // namespace at::native
```