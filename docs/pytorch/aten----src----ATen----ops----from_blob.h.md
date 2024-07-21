# `.\pytorch\aten\src\ATen\ops\from_blob.h`

```
#pragma once
#include <ATen/core/Tensor.h>

namespace at {

namespace detail {

// 空操作删除器，用于默认的上下文删除器
TORCH_API inline void noopDelete(void*) {}

} // namespace detail

/// 提供了一个流畅的 API 从外部数据构造张量。
///
/// 当现有的 `from_blob` 函数的参数不符合要求时，可以使用流畅的 API 替代。
///
///     at::Tensor tensor = at::for_blob(data, sizes)
///             .strides(strides)
///             .context(context, [](void *ctx) { delete static_cast<Ctx*>(ctx); })
///             .options(...)
///             .make_tensor();
///
class TORCH_API TensorMaker {
  friend TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept;

 public:
  using ContextDeleter = DeleterFnPtr;

  // 设置 strides 参数
  TensorMaker& strides(OptionalIntArrayRef value) noexcept {
    strides_ = value;
    return *this;
  }

  // 设置 storage_offset 参数
  TensorMaker& storage_offset(std::optional<int64_t> value) noexcept {
    storage_offset_ = value;
    return *this;
  }

  // 设置 deleter 参数
  TensorMaker& deleter(std::function<void(void*)> value) noexcept {
    deleter_ = std::move(value);
    return *this;
  }

  // 设置 context 参数
  TensorMaker& context(void* value, ContextDeleter deleter = nullptr) noexcept {
    // 使用指定的上下文和删除器创建唯一指针
    ctx_ = std::unique_ptr<void, ContextDeleter>{
        value, deleter != nullptr ? deleter : detail::noopDelete};
    return *this;
  }

  // 设置 target_device 参数
  TensorMaker& target_device(std::optional<Device> value) noexcept {
    device_ = value;
    return *this;
  }

  // 设置 options 参数
  TensorMaker& options(TensorOptions value) noexcept {
    opts_ = value;
    return *this;
  }

  // 标记可调整大小的存储
  TensorMaker& resizeable_storage() noexcept {
    resizeable_ = true;
    return *this;
  }

  // 设置 allocator 参数
  TensorMaker& allocator(c10::Allocator* allocator) noexcept {
    allocator_ = allocator;
    return *this;
  }

  // 构造并返回张量
  Tensor make_tensor();

 private:
  explicit TensorMaker(void* data, IntArrayRef sizes) noexcept
      : data_{data}, sizes_{sizes} {}

  // 计算存储大小
  std::size_t computeStorageSize() const noexcept;

  // 从 deleter 创建 DataPtr
  DataPtr makeDataPtrFromDeleter() noexcept;

  // 从 context 创建 DataPtr
  DataPtr makeDataPtrFromContext() noexcept;

  // 创建临时尺寸
  IntArrayRef makeTempSizes() const noexcept;

  void* data_;
  IntArrayRef sizes_;
  OptionalIntArrayRef strides_{};
  std::optional<int64_t> storage_offset_{};
  std::function<void(void*)> deleter_{};
  std::unique_ptr<void, ContextDeleter> ctx_{nullptr, detail::noopDelete};
  std::optional<Device> device_{};
  TensorOptions opts_{};
  bool resizeable_{};
  c10::Allocator* allocator_{};
};

// 创建一个 TensorMaker 对象，用于从给定的数据和尺寸构造张量
inline TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept {
  return TensorMaker{data, sizes};
}

// 从给定的数据和尺寸构造张量，使用指定的 strides、deleter 和 options
inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const std::function<void(void*)>& deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = c10::nullopt) {
  return for_blob(data, sizes)
      .strides(strides)
      .deleter(deleter)
      .options(options)
      .target_device(target_device)
      .make_tensor();
}

// 这是 from_blob 函数的另一个重载，声明未完成
inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    // 使用给定的数据和大小创建一个 Blob 对象，并返回其构造器对象
    IntArrayRef strides,
    // 设置存储偏移量
    int64_t storage_offset,
    // 设置数据销毁器的函数对象
    const std::function<void(void*)>& deleter,
    // 设置张量的选项，默认为空选项
    const TensorOptions& options = {},
    // 设置目标设备，如果未指定则为空
    const std::optional<Device> target_device = c10::nullopt) {
  // 调用 Blob 对象的 for_blob 方法，并传入数据和大小
  return for_blob(data, sizes)
      // 设置 Blob 对象的步长
      .strides(strides)
      // 设置 Blob 对象的存储偏移量
      .storage_offset(storage_offset)
      // 设置 Blob 对象的数据销毁器
      .deleter(deleter)
      // 设置 Blob 对象的张量选项
      .options(options)
      // 设置 Blob 对象的目标设备
      .target_device(target_device)
      // 调用 Blob 对象的 make_tensor 方法，返回生成的张量
      .make_tensor();
}
}

// 命名空间结束

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const TensorOptions& options = {},
    const std::optional<Device> target_device = c10::nullopt) {
  // 使用给定的 data 和 sizes 创建一个 for_blob 对象，返回其结果
  return for_blob(data, sizes)
      // 设置释放器函数，并移动所有权
      .deleter(std::move(deleter))
      // 设置张量选项
      .options(options)
      // 设置目标设备，如果提供的话
      .target_device(target_device)
      // 创建并返回张量对象
      .make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options = {}) {
  // 使用给定的 data 和 sizes 创建一个 for_blob 对象，并设置 strides 和选项，返回其结果
  return for_blob(data, sizes).strides(strides).options(options).make_tensor();
}

inline Tensor from_blob(
    void* data,
    IntArrayRef sizes,
    const TensorOptions& options = {}) {
  // 使用给定的 data 和 sizes 创建一个 for_blob 对象，并设置选项，返回其结果
  return for_blob(data, sizes).options(options).make_tensor();
}

} // namespace at


这些注释详细解释了每个函数的目的和每行代码的作用，确保了代码的每个部分都得到了充分的解释和理解。
```