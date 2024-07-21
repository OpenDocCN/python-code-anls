# `.\pytorch\aten\src\ATen\templates\Functions.cpp`

```py
// 包含所需的头文件

#include <array>  // 包含数组容器的头文件

#include <ATen/Functions.h>  // 包含 ATen 库的功能函数头文件
#include <ATen/Utils.h>      // 包含 ATen 库的实用工具头文件
#include <c10/core/Allocator.h>  // 包含 c10 核心库中的分配器头文件

namespace at {

// 实现 TensorMaker 类的 make_tensor 方法
Tensor TensorMaker::make_tensor() {
   AutoDispatchBelowADInplaceOrView guard{};  // 自动分发下的暂时占位符，待移除
   tracer::impl::NoTracerDispatchMode tracer_guard{};  // 跟踪器的无跟踪调度模式

   check_size_nonnegative(sizes_);  // 检查尺寸是否非负

   // 检查删除器和上下文参数的互斥性
   TORCH_CHECK_VALUE(
       !deleter_ || !ctx_,
       "The deleter and context arguments are mutually exclusive.");

   // 如果设备未指定，则根据数据指针获取全局上下文中的设备
   if (device_ == nullopt) {
     device_ = globalContext().getDeviceFromPtr(data_, opts_.device().type());
   }

   // 如果选项中指定了设备索引
   if (opts_.device().has_index()) {
     // clang-format off
     TORCH_CHECK_VALUE(
         opts_.device() == *device_,
         "Specified device ", opts_.device(), " does not match device of data ", *device_);
     // clang-format on
   }

   // 计算存储空间的字节数
   std::size_t size_bytes = computeStorageSize();

   DataPtr data_ptr{};
   // 根据是否有删除器选择数据指针的生成方式
   if (deleter_) {
     data_ptr = makeDataPtrFromDeleter();
   } else {
     data_ptr = makeDataPtrFromContext();
   }

   // 如果允许调整大小且分配器不为空，则创建可调整大小的存储
   TORCH_CHECK(!resizeable_ || allocator_ != nullptr, "Must specify an allocator with allocator() if you want to use resizeable_storage()");
   Storage storage{Storage::use_byte_size_t{}, size_bytes, std::move(data_ptr), /*allocator=*/allocator_, /*resizeable=*/resizeable_};

   // 使用详细方法创建张量
   Tensor tensor = detail::make_tensor<TensorImpl>(
       std::move(storage), opts_.computeDispatchKey(), opts_.dtype());

  // 获取张量实现指针
  TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  if (strides_) {
    tensor_impl->set_sizes_and_strides(sizes_, *strides_);  // 设置尺寸和步幅
  } else {
    tensor_impl->set_sizes_contiguous(sizes_);  // 设置连续尺寸
  }
  if (storage_offset_) {
    // 这里还有代码，需要继续完成
  // 设置张量实现的存储偏移量为指定的存储偏移量
  tensor_impl->set_storage_offset(*storage_offset_);
}

// 返回创建的张量对象
return tensor;
}

// 计算张量的存储大小，以字节为单位，不会抛出异常
std::size_t TensorMaker::computeStorageSize() const noexcept {
  // 计算数据类型的每个项的字节大小
  std::size_t itemsize = opts_.dtype().itemsize();

  // 如果存在步长信息
  if (strides_) {
    // 根据大小、步长和项大小计算存储的总字节数
    auto storage_size = detail::computeStorageNbytes(sizes_, *strides_, itemsize);
    // 如果存在存储偏移量，则将其添加到存储大小中
    if (storage_offset_) {
      storage_size += storage_offset_.value();
    }
    return storage_size;
  }

  // 计算张量的总元素个数
  std::size_t size = 1;
  for (std::int64_t s : sizes_) {
    size *= static_cast<std::size_t>(s);
  }
  // 计算未指定步长时的存储大小
  auto storage_size = size * itemsize;
  // 如果存在存储偏移量，则将其添加到存储大小中
  if (storage_offset_) {
    storage_size += storage_offset_.value();
  }
  return storage_size;
}

// 从委托器创建数据指针的内联函数，不会抛出异常
inline DataPtr TensorMaker::makeDataPtrFromDeleter() noexcept {
  return InefficientStdFunctionContext::makeDataPtr(data_, std::move(deleter_), *device_);
}

// 从上下文创建数据指针的内联函数，不会抛出异常
inline DataPtr TensorMaker::makeDataPtrFromContext() noexcept {
  return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), *device_};
}

// 创建临时大小数组的方法，不会抛出异常
IntArrayRef TensorMaker::makeTempSizes() const noexcept {
  // 静态数组，用于当内存格式为ChannelsLast或ChannelsLast3d时返回对应的临时大小数组
  static std::int64_t zeros[5] = {0, 0, 0, 0, 0};
  // 如果选项指定了内存格式
  if (opts_.has_memory_format()) {
    MemoryFormat format = *opts_.memory_format_opt();
    // 如果内存格式为ChannelsLast，则返回四维数组的引用
    if (format == MemoryFormat::ChannelsLast) {
      return IntArrayRef(zeros, 4);
    }
    // 如果内存格式为ChannelsLast3d，则返回五维数组的引用
    if (format == MemoryFormat::ChannelsLast3d) {
      return IntArrayRef(zeros, 5);
    }
  }
  // 默认返回一维的零数组的引用
  return IntArrayRef(zeros, 1);
}
} // 结束 at 命名空间的定义
```