# `.\pytorch\aten\src\ATen\AccumulateType.cpp`

```
// 包含 ATen 库中的 AccumulateType.h 文件

namespace at {

// 根据给定的标量类型和设备类型返回累积类型
c10::ScalarType toAccumulateType(c10::ScalarType type, c10::DeviceType device) {
  // 根据标量类型进行切换
  switch (type) {
    // 定义每种标量类型对应的累积类型转换
#define DEFINE_CASE(scalar_t, TypeNum)                                                             \
    case ScalarType::TypeNum:                                                                      \
      // 根据设备类型进行进一步切换
      switch (device) {                                                                            \
        // 如果设备是 CUDA
        case DeviceType::CUDA:                                                                     \
          // 返回对应 CUDA 设备的累积类型
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CUDA>>::value; \
        // 如果设备是 MPS
        case DeviceType::MPS:                                                                      \
          // 返回对应 MPS 设备的累积类型
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::MPS>>::value;  \
        // 默认情况下返回 CPU 设备的累积类型
        default:                                                                                   \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CPU>>::value;  \
      }

    // 针对所有非复数半精度标量类型，使用宏展开进行定义处理
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(DEFINE_CASE)
#undef DEFINE_CASE

    // 如果给定的标量类型未被识别，则抛出内部断言错误
    default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

// 根据给定的标量类型和 CUDA 设备标志，返回累积类型
c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda) {
  // 如果是 CUDA 设备，则调用 toAccumulateType 函数并指定设备类型为 CUDA
  return is_cuda ? toAccumulateType(type, c10::DeviceType::CUDA) : toAccumulateType(type, c10::DeviceType::CPU);
}

} // namespace at
```