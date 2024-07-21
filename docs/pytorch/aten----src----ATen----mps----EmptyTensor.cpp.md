# `.\pytorch\aten\src\ATen\mps\EmptyTensor.cpp`

```
// 定义错误消息：如果PyTorch代码未启用MPS，则报告错误
#define MPS_ERROR_NOT_COMPILED "PyTorch code is not compiled with MPS enabled"
// 定义错误消息：MPS后端仅支持MacOS 12.3+版本
#define MPS_ERROR_RUNTIME_TOO_LOW \
  "The MPS backend is supported on MacOS 12.3+.", \
  "Current OS version can be queried using `sw_vers`"
// 定义错误消息：MPS框架不支持将MPS张量转换为float64类型
#define MPS_ERROR_DOUBLE_NOT_SUPPORTED "Cannot convert a MPS Tensor to float64 dtype " \
  "as the MPS framework doesn't support float64. Please use float32 instead."

// 进入ATen命名空间的detail命名空间
namespace at::detail {
// 定义函数：创建一个空的MPS张量
TensorBase empty_mps(
    IntArrayRef size,                                           // 张量的尺寸
    std::optional<ScalarType> dtype_opt,                        // 可选的数据类型
    std::optional<Layout> layout_opt,                           // 可选的布局
    std::optional<Device> device_opt,                           // 可选的设备
    std::optional<bool> pin_memory_opt,                         // 可选的内存固定
    std::optional<c10::MemoryFormat> memory_format_opt) {       // 可选的内存格式

  // 如果目标操作系统是macOS且MPS已启用，则执行以下操作
#if defined(__APPLE__)
#if __is_target_os(macOS)
  if (at::hasMPS()) {                                          // 检查是否支持MPS
    auto device = device_or_default(device_opt);                // 获取设备类型，默认为参数中的设备类型
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::MPS); // 断言设备类型为MPS

    // 检查张量布局是否为Strided，MPS仅支持Strided布局
    TORCH_CHECK_NOT_IMPLEMENTED(
        layout_or_default(layout_opt) == Layout::Strided,
        "only strided tensors are supported on MPS");

    // 检查张量维度是否小于等于16，MPS支持的最大维度为16
    TORCH_CHECK(size.size() <= 16, "MPS supports tensors with dimensions <= 16, but got ", size.size(), ".");

    // 检查张量尺寸是否为非负数
    check_size_nonnegative(size);

    // 获取MPS的分配器
    auto* allocator = at::mps::GetMPSAllocator();
    // 计算张量元素个数
    int64_t nelements = c10::multiply_integers(size);
    // 获取数据类型，如果是Double类型则抛出错误
    auto dtype = dtype_or_default(dtype_opt);
    TORCH_CHECK_TYPE(dtype != ScalarType::Double, MPS_ERROR_DOUBLE_NOT_SUPPORTED);
    // 检查BFloat16类型在MPS下的支持情况，仅在macOS 14或更新版本支持
    TORCH_CHECK_TYPE(dtype != ScalarType::BFloat16 || is_macos_13_or_newer(mps::MacOSVersion::MACOS_VER_14_0_PLUS), "MPS BFloat16 is only supported on MacOS 14 or newer");

    // 获取数据类型的元数据信息
    auto dtype_meta = scalarTypeToTypeMeta(dtype);
    // 计算存储空间的字节大小
    int64_t size_bytes = nelements * dtype_meta.itemsize();
    // 创建存储实现
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        size_bytes,
        allocator->allocate(size_bytes),
        allocator,
        /*resizeable=*/true);

    // 创建张量对象，使用MPS分发键和指定的数据类型元数据
    auto tensor =
        detail::make_tensor<TensorImpl>(storage_impl, DispatchKey::MPS, dtype_meta);
    
    // 如果张量尺寸不为[0]，则设置其尺寸为连续的
    if (size.size() != 1 || size[0] != 0) {
      tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }

    // 获取内存格式，默认为连续
    auto memory_format = memory_format_opt.value_or(MemoryFormat::Contiguous);
    // 重排空的张量以适应指定的内存格式
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

    // 如果全局上下文启用确定性操作并且启用了填充未初始化内存的确定性填充，则执行填充操作
    if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
      at::native::fill_empty_deterministic_(tensor);
    }
    // 返回创建的张量
    return tensor;
  } else {
    // 如果不支持MPS，则抛出运行时错误
    TORCH_CHECK(false, MPS_ERROR_RUNTIME_TOO_LOW)
  }
#else
  // 如果不是macOS系统，则抛出未编译MPS的错误
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)
#endif
#else
  // 如果未定义__APPLE__，则执行以下代码块
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)
#endif
}

TensorBase empty_mps(
    IntArrayRef size, const TensorOptions &options) {
  // 调用detail命名空间中的empty_mps函数，返回一个MPS（Metal Performance Shaders）优化的空TensorBase对象
  return at::detail::empty_mps(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),  // 将TensorOptions中的dtype转换为ScalarType
      options.layout_opt(),                         // 获取TensorOptions中的layout选项
      options.device_opt(),                         // 获取TensorOptions中的device选项
      options.pinned_memory_opt(),                  // 获取TensorOptions中的pinned_memory选项
      options.memory_format_opt());                 // 获取TensorOptions中的memory_format选项
}

TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt) {
#if defined(__APPLE__)
#if __is_target_os(macOS)
  // 如果在苹果操作系统上，并且支持MPS（Metal Performance Shaders）
  if (at::hasMPS()) {
    auto device = device_or_default(device_opt);  // 获取默认设备或者传入的设备选项
    TORCH_INTERNAL_ASSERT(device.is_mps());       // 确保设备是MPS（Metal Performance Shaders）设备
    TORCH_CHECK_TYPE(dtype != ScalarType::Double, MPS_ERROR_DOUBLE_NOT_SUPPORTED);  // 检查不支持Double类型的错误
    const DeviceGuard device_guard(device);       // 使用DeviceGuard管理当前设备
    auto* allocator = at::mps::GetMPSAllocator(); // 获取MPS分配器
    constexpr c10::DispatchKeySet mps_dks(c10::DispatchKey::MPS);  // 定义MPS的DispatchKeySet
    // 调用detail命名空间中的empty_strided_generic函数，创建一个带有指定大小、步长和数据类型的Tensor对象
    Tensor result = at::detail::empty_strided_generic(
        size, stride, allocator, mps_dks, dtype);
    // 查看"注释 [Enabling Deterministic Operations]"，如果全局上下文启用确定性操作并且填充未初始化内存，则调用fill_empty_deterministic_函数
    if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
      at::native::fill_empty_deterministic_(result);
    }
    return result;  // 返回创建的Tensor对象
  } else {
    TORCH_CHECK(false, MPS_ERROR_RUNTIME_TOO_LOW)  // 如果不支持MPS，抛出运行时版本过低的错误
  }
#else
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)  // 如果不是macOS操作系统，抛出未编译MPS的错误
#endif
#else
  TORCH_CHECK(false, MPS_ERROR_NOT_COMPILED)  // 如果未定义__APPLE__，抛出未编译MPS的错误
#endif
}

TensorBase empty_strided_mps(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  // 调用native命名空间中的empty_strided_mps函数，返回一个MPS（Metal Performance Shaders）优化的空TensorBase对象
  return at::native::empty_strided_mps(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),  // 将TensorOptions中的dtype转换为ScalarType
      options.layout_opt(),                         // 获取TensorOptions中的layout选项
      options.device_opt(),                         // 获取TensorOptions中的device选项
      options.pinned_memory_opt());                 // 获取TensorOptions中的pinned_memory选项
}

} // namespace at::detail
```