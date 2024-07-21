# `.\pytorch\aten\src\ATen\native\DispatchStub.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#include <cstdlib>
#include <cstring>

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
#include <sys/auxv.h>
#endif

namespace at::native {

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
// 检查当前 CPU 是否支持 VXE（Vector eXtensions for Enhanced Performance）
static inline bool cpu_has_vxe() {
  return (getauxval(AT_HWCAP) & HWCAP_S390_VXE);
}
#endif

// 计算当前 CPU 的能力
static CPUCapability compute_cpu_capability() {
  auto envar = std::getenv("ATEN_CPU_CAPABILITY");
  if (envar) {
    // 检查环境变量，确定 CPU 的能力
#if defined(HAVE_VSX_CPU_DEFINITION)
    if (strcmp(envar, "vsx") == 0) {
      return CPUCapability::VSX;
    }
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
    if (strcmp(envar, "zvector") == 0) {
      return CPUCapability::ZVECTOR;
    }
#else
#ifdef HAVE_AVX512_CPU_DEFINITION
    if (strcmp(envar, "avx512") == 0) {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    if (strcmp(envar, "avx2") == 0) {
      return CPUCapability::AVX2;
    }
#endif
#endif
    // 默认返回值
    if (strcmp(envar, "default") == 0) {
      return CPUCapability::DEFAULT;
    }
    // 输出警告信息并忽略无效的 ATEN_CPU_CAPABILITY 值
    TORCH_WARN("ignoring invalid value for ATEN_CPU_CAPABILITY: ", envar);
  }

  // 在没有指定 ATEN_CPU_CAPABILITY 的情况下，根据当前 CPU 特性确定能力
#if !defined(__powerpc__) && !defined(__s390x__)
  if (cpuinfo_initialize()) {
#if defined(HAVE_AVX512_CPU_DEFINITION)
    // 只有在支持的硬件和编译器条件下，返回 AVX512 的能力
    if (cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() &&
        cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX512;
    }
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    // 在支持 AVX2 的硬件上，返回 AVX2 的能力
    if (cpuinfo_has_x86_avx2() && cpuinfo_has_x86_fma3()) {
      return CPUCapability::AVX2;
    }
#endif
  }
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  // 如果 CPU 支持 VXE，则返回 ZVECTOR 的能力
  if (cpu_has_vxe()) {
    return CPUCapability::ZVECTOR;
  }
#endif

#ifdef HAVE_VSX_CPU_DEFINITION
  // 默认返回 VSX 的能力
  return CPUCapability::VSX;
#else
  // 如果没有定义特定的 CPU 定义，返回默认能力
  return CPUCapability::DEFAULT;
#endif
}

// 获取当前 CPU 的能力
CPUCapability get_cpu_capability() {
  static CPUCapability capability = compute_cpu_capability();
  return capability;
}

// 尝试获取调用指针，根据设备类型和 CPU 能力
DispatchResult DispatchStubImpl::try_get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
) {
  // 支持的设备类型数组
  constexpr auto supported_devices = c10::array_of<c10::DeviceType>(
        c10::DeviceType::CPU,
        c10::DeviceType::CUDA,
        c10::DeviceType::HIP,
        c10::DeviceType::MPS,
        c10::DeviceType::PrivateUse1
    );
    // 检查设备类型是否在支持的范围内。
    # 如果 device_type 不在 supported_devices 列表中，则设备不被支持，返回相应错误类型
    if (std::find(supported_devices.begin(), supported_devices.end(), device_type) == supported_devices.end()) {
        return ErrorType::DeviceNotSupported;
    }

  switch (device_type) {
    case DeviceType::CPU: {
      // 在这里使用 memory_order_relaxed，因为即使两个线程竞争，
      // 它们仍会为 cpu_dispatch_ptr 计算相同的值。
      auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
      // 如果 fptr 为空指针，则尝试选择默认的 CPU 实现
      if (!fptr) {
        auto result = try_choose_cpu_impl(
          DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
          , AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
          , AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
          , VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
          , ZVECTOR
#endif
        );
        // 如果定义了 AVX512_CPU，添加 AVX512 到参数列表中
        // 如果定义了 AVX2_CPU，添加 AVX2 到参数列表中
        // 如果定义了 VSX_CPU，添加 VSX 到参数列表中
        // 如果定义了 ZVECTOR_CPU，添加 ZVECTOR 到参数列表中
        // 这些宏定义用于根据不同的 CPU 支持情况，在函数调用时选择不同的实现
        if (!std::holds_alternative<ErrorType>(result)) {
          // 如果结果不是 ErrorType 变体，则将 fptr 存储在 cpu_dispatch_ptr 中
          cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
        }
      // 返回结果
      return result;
      }
      // 如果结果是 ErrorType 变体，则返回 DispatchResult(fptr)
      return DispatchResult(fptr);
    }

    case DeviceType::CUDA:
      // 对于 CUDA 设备类型，如果 cuda_dispatch_ptr 不为 nullptr，则返回其 DispatchResult，否则返回 MissingDeviceKernel 错误
      return cuda_dispatch_ptr != nullptr ? DispatchResult(cuda_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    case DeviceType::HIP:
      // 对于 HIP 设备类型，如果 hip_dispatch_ptr 不为 nullptr，则返回其 DispatchResult，否则返回 MissingDeviceKernel 错误
      return hip_dispatch_ptr != nullptr ? DispatchResult(hip_dispatch_ptr) : ErrorType::MissingDeviceKernel;

#if defined(USE_MPS)
    case DeviceType::MPS:
      // 对于 MPS 设备类型，如果 mps_dispatch_ptr 不为 nullptr，则返回其 DispatchResult，否则返回 MissingDeviceKernel 错误
      return mps_dispatch_ptr != nullptr ? DispatchResult(mps_dispatch_ptr) : ErrorType::MissingDeviceKernel;
#endif

    case DeviceType::PrivateUse1:
      // 对于 PrivateUse1 设备类型，如果 privateuse1_dispatch_ptr 不为 nullptr，则返回其 DispatchResult，否则返回 MissingDeviceKernel 错误
      return privateuse1_dispatch_ptr != nullptr ? DispatchResult(privateuse1_dispatch_ptr) : ErrorType::MissingDeviceKernel;

    default:
      // 对于未知的设备类型，引发断言错误，并显示错误消息和提供的设备类型
      TORCH_INTERNAL_ASSERT(false, "An unexpected device type was provided ", device_type);
      return ErrorType::DeviceNotSupported;
  }
}

// 获取调用指针的具体实现，根据设备类型和 CPU 支持情况进行选择
void* DispatchStubImpl::get_call_ptr(
  const DeviceType device_type
  , void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
) {

  // 尝试获取调用指针
  auto result = try_get_call_ptr(
      device_type,
      DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
      ,
      AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
      ,
      AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
      ,
      VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
      ,
      ZVECTOR
#endif
  );
  // 如果返回结果是 ErrorType 变体，则处理错误
  if (std::holds_alternative<ErrorType>(result)) {
    auto error = std::get<ErrorType>(result);
    switch (error) {
      case ErrorType::MissingDeviceKernel:
        // 如果缺少设备内核，引发断言错误，并显示错误消息和设备类型
        TORCH_INTERNAL_ASSERT(
            false, "DispatchStub: missing kernel for ", device_type);
        return nullptr;
      case ErrorType::DeviceNotSupported:
        // 如果设备类型不受支持，引发 AT_ERROR，并显示错误消息和设备类型
        AT_ERROR("DispatchStub: unsupported device type", device_type);
    }
  }

  // 否则，获取函数指针并返回
  void* fptr = std::get<void*>(result);
  return fptr;
}

// 尝试选择 CPU 实现的具体方法
DispatchResult DispatchStubImpl::try_choose_cpu_impl(
    void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
    , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
    , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
    , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
    , void *ZVECTOR
#endif
  ){

  // 获取当前 CPU 的能力值并转换为整数
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
#ifdef HAVE_AVX512_CPU_DEFINITION
  // 如果支持 AVX512，则执行以下操作
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // 由于一些测试在 Windows 平台上对 AVX512 的支持不稳定，因此禁用量化内核的 AVX512 版本
    // 理想情况下，应该为所有内核提供 AVX512 版本

    // 如果支持 AVX512，则执行以下操作
    if (capability >= static_cast<int>(CPUCapability::AVX512)) {
        // 由于一些测试在 Windows 平台上对 AVX512 的支持不稳定，因此禁用量化内核的 AVX512 版本
        // 理想情况下，应该为所有内核提供 AVX512 版本。
        // 这里的代码逻辑涉及到根据 CPU 能力选择合适的实现方式，同时考虑了平台特定的稳定性问题。
        // 在这个条件下执行的代码，仅在 CPU 支持 AVX512 时才会被执行。
        // 注意：此处缺少具体的操作内容，可能在后续的代码中继续实现 AVX512 版本的内核操作。
    # 如果 AVX512 不可用，则执行以下逻辑
    if (C10_UNLIKELY(!AVX512)) {
      # 如果 AVX512 内核不存在，将任务分派到 AVX2 内核
      return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
    } else {
      # 如果 AVX512 可用，则直接分派到 AVX512 内核
      return DispatchResult(AVX512);
    }
#ifdef HAVE_AVX2_CPU_DEFINITION
  // 如果具有 AVX2 CPU 定义，并且当前能力大于等于 AVX2，则选择 AVX2 内核
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    // 如果 AVX2 内核存在，则返回 AVX2 内核；否则返回缺少设备内核错误
    return AVX2 != nullptr ? DispatchResult(AVX2) : ErrorType::MissingDeviceKernel;
  }
#endif

#ifdef HAVE_VSX_CPU_DEFINITION
  // 如果具有 VSX CPU 定义，并且当前能力大于等于 VSX，则选择 VSX 内核
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    // 如果 VSX 内核存在，则返回 VSX 内核；否则返回缺少设备内核错误
    return VSX != nullptr ? DispatchResult(VSX) : ErrorType::MissingDeviceKernel;
  }
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  // 如果具有 ZVECTOR CPU 定义，并且当前能力大于等于 ZVECTOR，则选择 ZVECTOR 内核
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    // 如果 ZVECTOR 内核存在，则返回 ZVECTOR 内核；否则返回缺少设备内核错误
    return ZVECTOR != nullptr ? DispatchResult(ZVECTOR) : ErrorType::MissingDeviceKernel;
  }
#endif

// 如果以上条件均不满足，则返回默认内核
return DEFAULT != nullptr ? DispatchResult(DEFAULT) : ErrorType::MissingDeviceKernel;
}

void* DispatchStubImpl::choose_cpu_impl(
  void *DEFAULT
#ifdef HAVE_AVX512_CPU_DEFINITION
  , void *AVX512
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  , void *AVX2
#endif
#ifdef HAVE_VSX_CPU_DEFINITION
  , void *VSX
#endif
#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  , void *ZVECTOR
#endif
) {
  auto capability = static_cast<int>(get_cpu_capability());
  (void)capability;
  
#ifdef HAVE_AVX512_CPU_DEFINITION
  // 如果具有 AVX512 CPU 定义，并且当前能力大于等于 AVX512，则选择 AVX512 内核
  if (capability >= static_cast<int>(CPUCapability::AVX512)) {
    // 在 Windows 上，由于某些测试不稳定，AVX512 的量化内核也被禁用了。
    // 理想情况下，所有内核都应该有 AVX512 版本。
    if (C10_UNLIKELY(!AVX512)) {
      // 如果 AVX512 内核不存在，则分派到 AVX2，因为缺少 AVX512 内核
      TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
      return AVX2;
    } else {
      return AVX512;
    }
  }
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
  // 如果具有 AVX2 CPU 定义，并且当前能力大于等于 AVX2，则选择 AVX2 内核
  if (capability >= static_cast<int>(CPUCapability::AVX2)) {
    TORCH_INTERNAL_ASSERT(AVX2, "DispatchStub: missing AVX2 kernel");
    return AVX2;
  }
#endif

#ifdef HAVE_VSX_CPU_DEFINITION
  // 如果具有 VSX CPU 定义，并且当前能力大于等于 VSX，则选择 VSX 内核
  if (capability >= static_cast<int>(CPUCapability::VSX)) {
    TORCH_INTERNAL_ASSERT(VSX, "DispatchStub: missing VSX kernel");
    return VSX;
  }
#endif

#ifdef HAVE_ZVECTOR_CPU_DEFINITION
  // 如果具有 ZVECTOR CPU 定义，并且当前能力大于等于 ZVECTOR，则选择 ZVECTOR 内核
  if (capability >= static_cast<int>(CPUCapability::ZVECTOR)) {
    TORCH_INTERNAL_ASSERT(ZVECTOR, "DispatchStub: missing ZVECTOR kernel");
    return ZVECTOR;
  }
#endif

// 如果以上条件均不满足，则返回默认内核
TORCH_INTERNAL_ASSERT(DEFAULT, "DispatchStub: missing default kernel");
return DEFAULT;
}

}  // namespace at::native
```