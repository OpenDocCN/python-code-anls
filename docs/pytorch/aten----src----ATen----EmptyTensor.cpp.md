# `.\pytorch\aten\src\ATen\EmptyTensor.cpp`

```
// 定义宏 TORCH_ASSERT_NO_OPERATORS，用于禁用运算符
#define TORCH_ASSERT_NO_OPERATORS
// 包含 ATen 库的空张量、CUDA 和 XPU 接口、运行环境上下文、私有使用接口、CPU 分配器等头文件
#include <ATen/EmptyTensor.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <ATen/Context.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/util/safe_numerics.h>

// 匿名命名空间，用于定义私有函数和变量
namespace at::detail {
namespace {

// 根据是否需要固定内存，返回对应的 CPU 分配器
c10::Allocator* GetCPUAllocatorMaybePinned(bool pin_memory) {
  if (pin_memory) {
    // 如果有 CUDA 初始化，返回 CUDA 固定内存分配器
    if (at::globalContext().hasCUDA()) {
      return at::detail::getCUDAHooks().getPinnedMemoryAllocator();
    }
    // 如果有 XPU 初始化，返回 XPU 固定内存分配器
    else if (at::globalContext().hasXPU()) {
      return at::detail::getXPUHooks().getPinnedMemoryAllocator();
    }
    // 如果私有使用接口注册了，返回私有使用接口的固定内存分配器
    else if (at::isPrivateUse1HooksRegistered()) {
      return at::GetPrivateUse1HooksInterface()->getPinnedMemoryAllocator();
    } else {
      // 否则抛出错误，需要提供固定内存分配器以使用固定内存
      TORCH_CHECK(false, "Need to provide pin_memory allocator to use pin memory.")
    }
  }
  // 如果不需要固定内存，返回普通的 CPU 分配器
  return c10::GetCPUAllocator();
}

// 计算 int64_t 和 size_t 在 ATen 中的一致性，返回它们的最大值
constexpr uint64_t storage_max() {
  // 计算 int64_t 和 size_t 的最大值
  constexpr auto int64_max = static_cast<uint64_t>(
      std::numeric_limits<int64_t>::max());
  constexpr auto size_max = static_cast<uint64_t>(
      std::numeric_limits<size_t>::max());
  // 返回它们的最小值
  return std::min(int64_max, size_max);
}

// 如果数据类型是 kComplexHalf，则发出警告，此处支持实验性特性
inline void raise_warning_for_complex_half(ScalarType dtype) {
  if (dtype == kComplexHalf) {
    TORCH_WARN_ONCE(
        "ComplexHalf support is experimental and many operators don't support it yet.");
  }
}

}  // namespace (anonymous)

// 计算连续存储的字节大小，包括溢出检查
size_t computeStorageNbytesContiguous(
    IntArrayRef sizes,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  // 在移动设备上忽略溢出检查
#ifndef C10_MOBILE
  uint64_t size = 1;
  bool overflowed = c10::safe_multiplies_u64(sizes, &size);
  overflowed |= c10::add_overflows(size, storage_offset, &size);
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  overflowed |= size > storage_max();
  // 抛出错误，如果存储大小计算溢出
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=", sizes);
  return static_cast<size_t>(size);
#else
  // 在移动设备上简单计算存储字节大小，不进行溢出检查
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
#endif
}

// 计算存储的字节大小，包括维度大小和步幅，进行维度匹配检查
size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes,
    size_t storage_offset
  ) {
  // 检查 sizes 和 strides 的维度是否匹配
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // 在移动设备上忽略溢出检查
#ifndef C10_MOBILE
  // 如果不是移动平台，则计算存储空间的大小
  // 存储空间大小比最后一个元素的偏移多 1
  uint64_t size = storage_offset + 1;
  bool overflowed = false;
  for (const auto i : c10::irange(sizes.size())) {
    // 如果某个维度的大小为 0，则返回 0
    if (sizes[i] == 0) {
      return 0;
    }

    // 计算按步幅计算的大小，并检查是否溢出
    uint64_t strided_size = 0;
    overflowed |= c10::mul_overflows(strides[i], sizes[i] - 1, &strided_size);
    overflowed |= c10::add_overflows(size, strided_size, &size);
  }
  // 计算存储空间大小乘以每个元素的字节数，并检查是否溢出
  overflowed |= c10::mul_overflows(size, itemsize_bytes, &size);
  // 检查存储空间大小是否超过了最大允许的值
  overflowed |= size > storage_max();
  // 如果发生溢出，则抛出错误信息
  TORCH_CHECK(!overflowed,
              "Storage size calculation overflowed with sizes=",
              sizes, " and strides=", strides);
  // 返回计算得到的存储空间大小
  return static_cast<size_t>(size);
#else
  // 如果是移动平台，则按移动平台的方式计算存储空间大小
  // 存储空间大小比最后一个元素的偏移多 1
  uint64_t size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    // 如果某个维度的大小为 0，则返回 0
    if (sizes[i] == 0) {
      return 0;
    }

    // 计算按步幅计算的大小
    size += strides[i] * (sizes[i] - 1);
  }
  // 返回计算得到的存储空间大小乘以每个元素的字节数
  return itemsize_bytes * (storage_offset + size);
#endif
}

SymInt computeStorageNbytesContiguous(
    SymIntArrayRef sizes,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  // 计算连续存储空间的字节数
  const auto numel = c10::multiply_integers(sizes);
  return itemsize_bytes * (storage_offset + numel);
}

// not including mobile-only macros in this function,
// since mobile shouldn't be using symints.
SymInt computeStorageNbytes(
    SymIntArrayRef sizes,
    SymIntArrayRef strides,
    const SymInt& itemsize_bytes,
    const SymInt& storage_offset
  ) {
  // 检查维度大小和步幅大小是否匹配
  TORCH_CHECK(
    sizes.size() == strides.size(),
    "dimensionality of sizes (",
    sizes.size(),
    ") must match dimensionality of strides (",
    strides.size(),
    ")");

  // 计算存储空间的字节数
  // 存储空间大小比最后一个元素的偏移多 1
  SymInt size = 1;
  for (const auto i : c10::irange(sizes.size())) {
    // 如果某个维度的大小为 0，则返回 0
    if (TORCH_GUARD_SIZE_OBLIVIOUS(sizes[i].sym_eq(0))) {
      return 0;
    }

    // 计算按步幅计算的大小
    size += strides[i] * (sizes[i] - 1);
  }
  // 返回计算得到的存储空间大小乘以每个元素的字节数
  return itemsize_bytes * (storage_offset + size);
}

template <typename T>
TensorBase _empty_generic(
    ArrayRef<T> size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    // 检查尺寸是否非负，若负数则抛出异常
    at::detail::check_size_nonnegative(size);
    // 如果标量类型是复数或者半精度浮点数，发出警告
    at::detail::raise_warning_for_complex_half(scalar_type);
    // 将标量类型转换为对应的TypeMeta类型
    caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    // 计算存储空间大小（字节），确保连续存储
    auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
    // 创建具有指定大小的可调整存储实现对象，使用指定的分配器
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator,
        /*resizeable=*/true);
    
    // 使用存储实现对象创建基础张量对象
    auto tensor = detail::make_tensor_base<TensorImpl>(
        std::move(storage_impl), ks, dtype);
    // 默认的张量实现对象具有大小 [0]
    // 注意：在元分发键上进行测试，以避免在大小为零时进行处理
    if (ks.has(c10::DispatchKey::Meta) || size.size() != 1 || size[0] != 0) {
      // 设置张量的大小为给定的大小，并保证其是连续的
      tensor.unsafeGetTensorImpl()->generic_set_sizes_contiguous(size);
    }
    
    // 如果有指定的内存格式选项
    if (memory_format_opt.has_value()) {
      // 对于刚创建的空连续张量，重新调整内存布局不会产生任何效果
      if (*memory_format_opt != MemoryFormat::Contiguous) {
        // 使用指定的内存格式重新调整张量的存储方式
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
      }
    }
    
    // 返回创建的张量对象
    return tensor;
}

TensorBase empty_generic(
    IntArrayRef size,                             # 接受整数数组引用作为大小参数
    c10::Allocator* allocator,                    # 分配器指针参数
    c10::DispatchKeySet ks,                       # 分发键集参数
    ScalarType scalar_type,                       # 标量类型参数
    std::optional<c10::MemoryFormat> memory_format_opt) {  # 可选的内存格式参数
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);  # 调用内部通用空张量创建函数并返回结果
}

TensorBase empty_generic_symint(
    SymIntArrayRef size,                          # 接受符号整数数组引用作为大小参数
    c10::Allocator* allocator,                    # 分配器指针参数
    c10::DispatchKeySet ks,                       # 分发键集参数
    ScalarType scalar_type,                       # 标量类型参数
    std::optional<c10::MemoryFormat> memory_format_opt) {  # 可选的内存格式参数
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt);  # 调用内部通用空张量创建函数并返回结果
}

template <typename T>
TensorBase _empty_strided_generic(
    T size,                                       # 大小参数（泛型）
    T stride,                                     # 步幅参数（泛型）
    c10::Allocator* allocator,                    # 分配器指针参数
    c10::DispatchKeySet ks,                       # 分发键集参数
    ScalarType scalar_type) {                     # 标量类型参数
  at::detail::check_size_nonnegative(size);       # 检查大小是否为非负数
  at::detail::raise_warning_for_complex_half(scalar_type);  # 对于复杂半精度引发警告
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);  # 将标量类型转换为类型元数据
  auto size_bytes = computeStorageNbytes(size, stride, dtype.itemsize());  # 计算存储字节数
  auto storage_impl = c10::make_intrusive<StorageImpl>(  # 创建存储实现
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor_base<TensorImpl>(  # 创建张量基类对象
      std::move(storage_impl), ks, dtype);
  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);  # 设置张量的大小和步幅
  return tensor;                                   # 返回创建的张量对象
}

TensorBase empty_strided_generic(
    IntArrayRef size,                             # 接受整数数组引用作为大小参数
    IntArrayRef stride,                           # 接受整数数组引用作为步幅参数
    c10::Allocator* allocator,                    # 分配器指针参数
    c10::DispatchKeySet ks,                       # 分发键集参数
    ScalarType scalar_type) {                      # 标量类型参数
  return _empty_strided_generic<IntArrayRef>(size, stride, allocator, ks, scalar_type);  # 调用泛型空张量创建函数并返回结果
}

TensorBase empty_strided_symint_generic(
    SymIntArrayRef size,                          # 接受符号整数数组引用作为大小参数
    SymIntArrayRef stride,                        # 接受符号整数数组引用作为步幅参数
    c10::Allocator* allocator,                    # 分配器指针参数
    c10::DispatchKeySet ks,                       # 分发键集参数
    ScalarType scalar_type) {                      # 标量类型参数
  return _empty_strided_generic<SymIntArrayRef>(size, stride, allocator, ks, scalar_type);  # 调用泛型空张量创建函数并返回结果
}

TensorBase empty_cpu(IntArrayRef size,             # 接受整数数组引用作为大小参数
                     ScalarType dtype,            # 标量类型参数
                     bool pin_memory,             # 布尔值，指示是否使用固定内存
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);  # 根据固定内存标志获取 CPU 分配器
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);  # 创建 CPU 分发键集
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);  # 调用通用空张量创建函数并返回结果
}

TensorBase empty_cpu(
    IntArrayRef size,                             # 接受整数数组引用作为大小参数
    std::optional<ScalarType> dtype_opt,          # 可选的标量类型参数
    std::optional<Layout> layout_opt,             # 可选的布局参数
    std::optional<Device> device_opt,             # 可选的设备参数
    std::optional<bool> pin_memory_opt,           # 可选的固定内存参数
    std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);  # 断言设备类型为 CPU
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);  # 断言布局为 Strided

  auto pin_memory = pinned_memory_or_default(pin_memory_opt);  # 获取或默认固定内存标志
  auto dtype = dtype_or_default(dtype_opt);        # 获取或默认标量类型
  return empty_cpu(size, dtype, pin_memory, memory_format_opt);  # 调用 CPU 特定的空张量创建函数并返回结果
}

TensorBase empty_cpu(
    ```cpp`
    # 定义一个函数，接受一个整数数组引用和一个 TensorOptions 对象作为参数
    return at::detail::empty_cpu(
        size,  # 调用 at::detail::empty_cpu 函数，传入 size 参数，指定张量的大小
        optTypeMetaToScalarType(options.dtype_opt()),  # 将 options 对象的数据类型选项转换为标量类型
        options.layout_opt(),  # 传入 options 对象的布局选项
        options.device_opt(),  # 传入 options 对象的设备选项
        options.pinned_memory_opt(),  # 传入 options 对象的固定内存选项
        options.memory_format_opt()  # 传入 options 对象的内存格式选项
    );
}

// 返回一个基于 CPU 的空的张量，使用指定的大小、步长、数据类型和是否固定内存选项
TensorBase empty_strided_cpu(IntArrayRef size, IntArrayRef stride,
                             ScalarType dtype, bool pin_memory) {
  // 获取可能固定内存的 CPU 分配器
  auto allocator = at::detail::GetCPUAllocatorMaybePinned(pin_memory);
  // 定义 CPU 的调度键集合
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  // 调用通用的 empty_strided_generic 函数创建空的步长张量
  return at::detail::empty_strided_generic(
      size, stride, allocator, cpu_ks, dtype);
}

// 返回一个基于 CPU 的空的张量，支持可选的数据类型、布局、设备和固定内存选项
TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  // 断言调试模式下的设备为 CPU
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::CPU);
  // 断言调试模式下的布局为 Strided
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  // 获取固定内存选项的值
  auto pin_memory = pinned_memory_or_default(pin_memory_opt);
  // 获取数据类型的值
  auto dtype = dtype_or_default(dtype_opt);
  // 调用前一个函数 empty_strided_cpu 创建空的步长张量
  return at::detail::empty_strided_cpu(size, stride, dtype, pin_memory);
}

// 返回一个基于 CPU 的空的张量，使用 TensorOptions 对象来指定参数
TensorBase empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  // 调用前一个函数 empty_strided_cpu，使用 TensorOptions 中的参数
  return at::detail::empty_strided_cpu(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

// 元分配器会忽略请求的任何分配并始终返回 nullptr
struct MetaAllocator final : public at::Allocator {
  MetaAllocator() = default;
  ~MetaAllocator() override = default;
  // 删除器函数，确保分配的指针始终为 nullptr
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  // 分配函数，返回空的 DataPtr 对象
  DataPtr allocate(const size_t nbytes) override {
    return {nullptr, nullptr, &deleter, at::Device(DeviceType::Meta)};
  }
  // 获取删除器函数指针
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
  // 复制数据的函数，此处为空实现
  void copy_data(void* dest, const void* src, std::size_t count) const final {}
};

// 创建全局的 MetaAllocator 对象
static MetaAllocator g_meta_alloc;

// 将 MetaAllocator 注册为 kMeta 的分配器
REGISTER_ALLOCATOR(kMeta, &g_meta_alloc);

// 返回一个基于 Meta 的空的张量，使用指定的大小、数据类型和内存格式选项
TensorBase empty_meta(IntArrayRef size, ScalarType dtype,
                     std::optional<c10::MemoryFormat> memory_format_opt) {
  // 获取 MetaAllocator 分配器
  auto *allocator = GetAllocator(kMeta);
  // 定义 Meta 的调度键集合
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);
  // 调用通用的 empty_generic 函数创建空的 Meta 张量
  return at::detail::empty_generic(
      size, allocator, meta_dks, dtype, memory_format_opt);
}

// 返回一个基于 Meta 的空的张量，支持可选的数据类型、布局、设备、固定内存和内存格式选项
TensorBase empty_meta(
  IntArrayRef size,
  std::optional<ScalarType> dtype_opt,
  std::optional<Layout> layout_opt,
  std::optional<Device> device_opt,
  std::optional<bool> pin_memory_opt,
  std::optional<c10::MemoryFormat> memory_format_opt
) {
  // 断言调试模式下的设备为 Meta
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);
  // 断言调试模式下的布局为 Strided
  // 注意：因为目前没有 SparseMeta，非步长布局是可以支持的
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout_opt) == Layout::Strided,
    "non-strided meta tensors not supported yet"
  );

  // 获取数据类型的值
  auto dtype = dtype_or_default(dtype_opt);
  // 调用前一个函数 empty_meta 创建空的 Meta 张量
  return empty_meta(size, dtype, memory_format_opt);
}
TensorBase empty_symint_meta(
  SymIntArrayRef size,                     // 接受一个SymIntArrayRef类型的参数size，表示大小信息
  std::optional<ScalarType> dtype_opt,     // 可选参数，指定张量的数据类型
  std::optional<Layout> layout_opt,        // 可选参数，指定张量的布局
  std::optional<Device> device_opt,        // 可选参数，指定张量的设备
  std::optional<bool> pin_memory_opt,      // 可选参数，指定是否使用固定内存
  std::optional<c10::MemoryFormat> memory_format_opt // 可选参数，指定张量的内存格式
) {
  auto *allocator = GetAllocator(kMeta);   // 获取用于Meta键的分配器
  constexpr c10::DispatchKeySet ks(c10::DispatchKey::Meta); // 定义一个Meta分发键集
  auto scalar_type = dtype_or_default(dtype_opt);  // 获得或者默认指定的数据类型
  return _empty_generic(size, allocator, ks, scalar_type, memory_format_opt); // 调用_empty_generic函数创建一个空的张量
}

TensorBase empty_meta(
    IntArrayRef size,                       // 接受一个IntArrayRef类型的参数size，表示大小信息
    const TensorOptions &options            // 接受一个TensorOptions类型的引用options，包含张量的各种选项
) {
  return at::detail::empty_meta(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),    // 使用options中的dtype_opt()转换为标量类型
      options.layout_opt(),                           // 使用options中的layout_opt()获取布局信息
      options.device_opt(),                           // 使用options中的device_opt()获取设备信息
      options.pinned_memory_opt(),                    // 使用options中的pinned_memory_opt()获取固定内存标志
      options.memory_format_opt());                   // 使用options中的memory_format_opt()获取内存格式信息
}

TensorBase empty_strided_meta(IntArrayRef size, IntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);             // 获取用于Meta键的分配器
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);  // 定义一个Meta分发键集
  return at::detail::empty_strided_generic(
      size, stride, allocator, meta_dks, dtype);     // 调用empty_strided_generic创建一个带步长信息的张量
}

TensorBase empty_strided_meta(
    IntArrayRef size,                               // 接受一个IntArrayRef类型的参数size，表示大小信息
    IntArrayRef stride,                             // 接受一个IntArrayRef类型的参数stride，表示步长信息
    std::optional<ScalarType> dtype_opt,            // 可选参数，指定张量的数据类型
    std::optional<Layout> layout_opt,               // 可选参数，指定张量的布局
    std::optional<Device> device_opt,               // 可选参数，指定张量的设备
    std::optional<bool> pin_memory_opt              // 可选参数，指定是否使用固定内存
) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);  // 断言设备类型为Meta
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);           // 断言布局为Strided

  auto dtype = dtype_or_default(dtype_opt);       // 获得或者默认指定的数据类型
  return at::detail::empty_strided_meta(size, stride, dtype);  // 调用empty_strided_meta创建带步长信息的张量
}

TensorBase empty_strided_meta(
    IntArrayRef size,                               // 接受一个IntArrayRef类型的参数size，表示大小信息
    IntArrayRef stride,                             // 接受一个IntArrayRef类型的参数stride，表示步长信息
    const TensorOptions &options                    // 接受一个TensorOptions类型的引用options，包含张量的各种选项
) {
  return at::detail::empty_strided_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),    // 使用options中的dtype_opt()转换为标量类型
      options.layout_opt(),                           // 使用options中的layout_opt()获取布局信息
      options.device_opt(),                           // 使用options中的device_opt()获取设备信息
      options.pinned_memory_opt());                    // 使用options中的pinned_memory_opt()获取固定内存标志
}

TensorBase empty_strided_symint_meta(SymIntArrayRef size, SymIntArrayRef stride,
                              ScalarType dtype) {
  auto *allocator = GetAllocator(kMeta);             // 获取用于Meta键的分配器
  constexpr c10::DispatchKeySet meta_dks(c10::DispatchKey::Meta);  // 定义一个Meta分发键集
  return at::detail::empty_strided_symint_generic(
      size, stride, allocator, meta_dks, dtype);     // 调用empty_strided_symint_generic创建一个带步长信息的SymInt张量
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,                            // 接受一个SymIntArrayRef类型的参数size，表示大小信息
    SymIntArrayRef stride,                          // 接受一个SymIntArrayRef类型的参数stride，表示步长信息
    std::optional<ScalarType> dtype_opt,            // 可选参数，指定张量的数据类型
    std::optional<Layout> layout_opt,               // 可选参数，指定张量的布局
    std::optional<Device> device_opt,               // 可选参数，指定张量的设备
    std::optional<bool> pin_memory_opt              // 可选参数，指定是否使用固定内存
) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_or_default(device_opt).type() == DeviceType::Meta);  // 断言设备类型为Meta
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);           // 断言布局为Strided

  auto dtype = dtype_or_default(dtype_opt);       // 获得或者默认指定的数据类型
  return at::detail::empty_strided_symint_meta(size, stride, dtype);  // 调用empty_strided_symint_meta创建带步长信息的SymInt张量
}

TensorBase empty_strided_symint_meta(
    SymIntArrayRef size,                            // 接受一个SymIntArrayRef类型的参数size，表示大小信息
    SymIntArrayRef stride,                          // 接受一个SymIntArrayRef类型的参数stride，表示步长信息
    const TensorOptions &options                    // 接受一个TensorOptions类型的引用options，包含张量的各种选项
) {
  return at::detail::empty_strided_symint_meta(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),    // 使用options中的dtype_opt()转换为标量类型
      options.layout_opt(),                           // 使用options中的layout_opt()获取布局信息
      options.device_opt(),                           // 使用options中的device_opt()获取设备信息
      options.pinned_memory_opt());                    // 使用options中的pinned_memory_opt()获取固定内存标志
}
    `
    const TensorOptions &options) {
      return at::detail::empty_strided_symint_meta(
          size,
          stride,
          optTypeMetaToScalarType(options.dtype_opt()),
          options.layout_opt(),
          options.device_opt(),
          options.pinned_memory_opt());
    }
}

} // namespace at::detail
```