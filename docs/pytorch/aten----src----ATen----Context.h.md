# `.\pytorch\aten\src\ATen\Context.h`

```py
#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/LinalgBackend.h>
#include <ATen/core/ATenGeneral.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/Generator.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/detail/AcceleratorHooksInterface.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HIPHooksInterface.h>
#include <ATen/detail/IPUHooksInterface.h>
#include <ATen/detail/MAIAHooksInterface.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/detail/MTIAHooksInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <c10/core/QEngine.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>

#include <cstdint>
#include <mutex>

namespace at {

class Tensor;

enum class TORCH_API Float32MatmulPrecision { HIGHEST, HIGH, MEDIUM };

// Context 类的定义
class TORCH_API Context {
 public:
  // 构造函数
  Context();

  // 返回特定设备上的默认生成器对象
  const Generator& defaultGenerator(Device device) {
    // 获取设备类型
    c10::DeviceType device_type = device.type();
    // 如果是 CUDA 设备，初始化 CUDA 相关设置
    initCUDAIfNeeded(device_type);
    // 如果是 HIP 设备，初始化 HIP 相关设置
    initHIPIfNeeded(device_type);

    // 根据设备类型返回默认生成器对象
    if (device_type == at::kCPU) {
      return at::detail::getDefaultCPUGenerator();
    } else if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks().getDefaultCUDAGenerator(device.index());
    } else if (device_type == at::kMPS) {
      return at::detail::getMPSHooks().getDefaultMPSGenerator();
    } else if (device_type == at::kXPU) {
      return at::detail::getXPUHooks().getDefaultXPUGenerator(device.index());
    } else if (device_type == at::kIPU) {
      return at::detail::getIPUHooks().getDefaultIPUGenerator(device.index());
    } else if (device_type == at::kPrivateUse1) {
      return at::GetPrivateUse1HooksInterface()->getDefaultGenerator(
          device.index());
    } else {
      // 抛出错误，指示设备类型未启用
      AT_ERROR(c10::DeviceTypeName(device_type), " device type not enabled.");
    }
  }

  // 获取加速器钩子接口对象
  const AcceleratorHooksInterface& getAcceleratorHooksInterface(
      std::optional<c10::DeviceType> opt_device_type = c10::nullopt) {
    // 如果提供了设备类型，则使用提供的设备类型，否则获取默认加速器类型
    c10::DeviceType device_type = opt_device_type.has_value()
        ? opt_device_type.value()
        : at::getAccelerator(true).value();

    // 根据设备类型返回相应的加速器钩子接口对象
    if (device_type == at::kCUDA) {
      return at::detail::getCUDAHooks();
    } else if (device_type == at::kMPS) {
      return at::detail::getMPSHooks();
    } else if (device_type == at::kPrivateUse1) {
      return at::detail::getPrivateUse1Hooks();
    } else if (device_type == at::kMTIA) {
      return at::detail::getMTIAHooks();
    } else {
      // 抛出错误，指示设备类型不是加速器类型
      AT_ERROR(
          c10::DeviceTypeName(device_type), " device type not an accelerator.");
    }
  }

  // 根据指针获取设备对象
  Device getDeviceFromPtr(void* data, c10::DeviceType device_type) {
    // 根据设备类型初始化相关设置（如 CUDA）
    initCUDAIfNeeded(device_type);
    initHIPIfNeeded(device_type);
    initXPUIfNeeded(device_type);
    // 如果设备类型是 CPU，则返回对应的 DeviceType::CPU
    if (device_type == at::kCPU) {
      return c10::DeviceType::CPU;
    } else if (device_type == at::kCUDA) {
      // 如果设备类型是 CUDA，则调用 getCUDAHooks().getDeviceFromPtr(data) 获取 CUDA 设备信息
      return at::detail::getCUDAHooks().getDeviceFromPtr(data);
    } else if (device_type == at::kXPU) {
      // 如果设备类型是 XPU，则调用 getXPUHooks().getDeviceFromPtr(data) 获取 XPU 设备信息
      return at::detail::getXPUHooks().getDeviceFromPtr(data);
    } else if (device_type == at::kPrivateUse1) {
      // 如果设备类型是 PrivateUse1，则调用 GetPrivateUse1HooksInterface()->getDeviceFromPtr(data) 获取 PrivateUse1 设备信息
      return at::GetPrivateUse1HooksInterface()->getDeviceFromPtr(data);
    } else {
      // 如果设备类型未知或不支持，抛出错误信息
      AT_ERROR(c10::DeviceTypeName(device_type), " device type not enabled.");
    }
  }
  // 检查给定指针是否是 CUDA 中的固定内存指针
  static bool isPinnedPtr(const void* data) {
    return detail::getCUDAHooks().isPinnedPtr(data);
  }
  // 下面几个函数声明，用于检查系统是否支持特定的库或功能
  static bool hasOpenMP();
  static bool hasMKL();
  static bool hasLAPACK();
  static bool hasMKLDNN();
  // 检查当前 CUDA 环境是否支持 MAGMA 库
  static bool hasMAGMA() {
    return detail::getCUDAHooks().hasMAGMA();
  }
  // 检查当前系统是否支持 CUDA
  static bool hasCUDA() {
    return detail::getCUDAHooks().hasCUDA();
  }
  // 检查当前系统是否支持 MTIA
  static bool hasMTIA() {
    return detail::getMTIAHooks().hasMTIA();
  }
  // 检查当前系统是否支持 CUDART
  static bool hasCUDART() {
    return detail::getCUDAHooks().hasCUDART();
  }
  // 获取当前 CUDART 的版本号
  static long versionCUDART() {
    return detail::getCUDAHooks().versionCUDART();
  }
  // 检查当前系统是否支持 CuDNN
  static bool hasCuDNN() {
    return detail::getCUDAHooks().hasCuDNN();
  }
  // 获取当前 CuDNN 的版本号
  static long versionCuDNN() {
    return detail::getCUDAHooks().versionCuDNN();
  }
  // 检查当前系统是否支持 CuSOLVER
  static bool hasCuSOLVER() {
    return detail::getCUDAHooks().hasCuSOLVER();
  }
  // 检查当前系统是否支持 CuBLASLt
  static bool hasCuBLASLt() {
    return detail::getCUDAHooks().hasCuBLASLt();
  }
  // 检查当前系统是否支持 HIP
  static bool hasHIP() {
    return detail::getHIPHooks().hasHIP();
  }
  // 检查当前系统是否支持 MPS
  static bool hasMPS() {
    return detail::getMPSHooks().hasMPS();
  }
  // 检查当前系统是否支持 IPU 设备
  static bool hasIPU() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::IPU);
  }
  // 检查当前系统是否支持 XLA 设备
  static bool hasXLA() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::XLA);
  }
  // 检查当前系统是否支持 XPU
  static bool hasXPU() {
    return detail::getXPUHooks().hasXPU();
  }
  // 检查当前系统是否支持 Lazy 设备
  static bool hasLazy() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::Lazy);
  }
  // 检查当前系统是否支持 MAIA 设备
  static bool hasMAIA() {
    return c10::impl::hasDeviceGuardImpl(c10::DeviceType::MAIA);
  }
  // 初始化 CUDA 环境，使用 call_once 确保只初始化一次
  // 此函数用于 lazyInitCUDA
  void lazyInitCUDA() {
    c10::call_once(thc_init, [&] { detail::getCUDAHooks().initCUDA(); });
  }
  // 初始化 HIP 环境，使用 call_once 确保只初始化一次
  // 此函数用于 lazyInitHIP
  void lazyInitHIP() {
    c10::call_once(thh_init, [&] { detail::getHIPHooks().initHIP(); });
  }
  // 初始化 XPU 环境，使用 call_once 确保只初始化一次
  // 此函数用于 lazyInitXPU
  void lazyInitXPU() {
    c10::call_once(thx_init, [&] { detail::getXPUHooks().initXPU(); });
  }
  // 初始化 MTIA 环境，使用 call_once 确保只初始化一次
  // 此函数用于 lazyInitMTIA
  void lazyInitMTIA() {
    c10::call_once(th_mtia_init, [&] { detail::getMTIAHooks().initMTIA(); });
  }
  // 初始化 PrivateUse1 环境，使用 call_once 确保只初始化一次
  // 此函数用于 lazyInitPrivateUse1
  void lazyInitPrivateUse1() {
    c10::call_once(thp_init, [&] {
      // 如果注册了 PrivateUse1 的钩子接口，则初始化 PrivateUse1
      if (isPrivateUse1HooksRegistered()) {
        at::GetPrivateUse1HooksInterface()->initPrivateUse1();
      }
    });
  }
  // 获取 NVRTC 相关信息
  static const at::cuda::NVRTC& getNVRTC() {
    if (p == c10::DeviceType::CUDA) {
      // 如果是 CUDA 设备，则调用 lazyInitCUDA() 初始化 CUDA 环境
      lazyInitCUDA();
    }
  }
  // 根据指定的设备类型初始化 HIP 环境
  void initHIPIfNeeded(c10::DeviceType p) {
    if (p == c10::DeviceType::HIP) {
      // 如果设备类型是 HIP，则调用 lazyInitHIP() 初始化 HIP 环境
      lazyInitHIP();
    }
    }
  }

这两行代码似乎是空白行，可能是代码中的格式错误或者是其他部分的一部分。


  void initXPUIfNeeded(c10::DeviceType p) {
    // 如果设备类型为XPU，初始化XPU（假设这里调用了一个lazy初始化函数）
    if (p == c10::DeviceType::XPU) {
      lazyInitXPU();
    }
  }

定义了一个函数 `initXPUIfNeeded`，用于根据给定的设备类型初始化XPU。如果设备类型为XPU (`c10::DeviceType::XPU`)，则调用 `lazyInitXPU()` 函数进行初始化。


  static bool checkCuBLASConfigDeterministic();

声明了一个静态函数 `checkCuBLASConfigDeterministic()`，但没有提供实现细节。


  c10::once_flag thc_init;
  c10::once_flag thh_init;
  c10::once_flag thx_init;
  c10::once_flag th_mtia_init;
  c10::once_flag thp_init;

声明了多个 `c10::once_flag` 类型的变量，用于在多线程环境下确保只执行一次初始化。


  bool enabled_cudnn = true;
  bool deterministic_cudnn = false;
  bool deterministic_mkldnn = false;
  bool _deterministic_algorithms = false;
  bool _deterministic_algorithms_warn_only = false;
  bool _deterministic_fill_uninitialized_memory = true;
  bool enabled_flashSDP = true;
  bool enabled_mem_efficientSDP = true;
  bool enabled_mathSDP = true;
  bool enabled_cudnnSDP = false;
  bool enabled_overrideable = true;

声明了多个布尔类型的变量，用于配置不同的算法和库的启用或禁用状态，例如 `cudnn`、`mkldnn`、以及一些算法的确定性设置和内存管理选项。
#ifdef USE_ROCM
  // 如果使用 ROCm，启用 CUDNN 的基准测试
  bool benchmark_cudnn = true;
#else
  // 如果未使用 ROCm，禁用 CUDNN 的基准测试
  bool benchmark_cudnn = false;
#endif

// 设置浮点32位矩阵乘法的精度
Float32MatmulPrecision float32_matmul_precision =
    // 检查环境变量 "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" 是否为 true，选择精度高的设置
    c10::utils::check_env("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == true
    ? at::Float32MatmulPrecision::HIGH
    : at::Float32MatmulPrecision::HIGHEST;

// CUDNN 基准测试的次数限制
int benchmark_limit_cudnn = 10;

// 允许使用 TF32 的 CUDNN 加速
bool allow_tf32_cudnn = true;

// 允许使用 FP16 降维的 CUBLAS 加速
bool allow_fp16_reduction_cublas = true;

// 允许使用 BF16 降维的 CUBLAS 加速
bool allow_bf16_reduction_cublas = true;

// 启用 MKLDNN 加速
bool enabled_mkldnn = true;

// 启用 NNPACK 加速
bool enabled_nnpack = true;

// 线性代数操作首选的后端设置
at::LinalgBackend linalg_preferred_backend =
    // 检查环境变量 "TORCH_LINALG_PREFER_CUSOLVER" 是否为 true，选择 CUSOLVER 作为后端
    c10::utils::check_env("TORCH_LINALG_PREFER_CUSOLVER") == true
    ? at::LinalgBackend::Cusolver
    : at::LinalgBackend::Default;

// BLAS 加速首选的后端设置
at::BlasBackend blas_preferred_backend =
#ifdef USE_ROCM
    // 如果使用 ROCm 并且 "TORCH_BLAS_PREFER_HIPBLASLT" 不为 false，则选择 HIPBLASLT 作为后端
    (c10::utils::check_env("TORCH_BLAS_PREFER_HIPBLASLT") != false)
#else
    // 如果未使用 ROCm 并且 "TORCH_BLAS_PREFER_CUBLASLT" 为 true，则选择 CUBLASLT 作为后端
    (c10::utils::check_env("TORCH_BLAS_PREFER_CUBLASLT") == true)
#endif
    ? at::BlasBackend::Cublaslt
    : at::BlasBackend::Cublas;

#ifdef C10_MOBILE
// 如果是在移动平台，释放原始权重
bool release_original_weights = true;
#else
// 如果不是在移动平台，不释放原始权重
bool release_original_weights = false;
#endif

// 是否显示 VMap 回退警告
bool display_vmap_fallback_warnings_ = false;

// 量化引擎设置为无
std::optional<at::QEngine> quantized_engine = c10::nullopt;

// 启用稀疏张量不变性检查
bool enable_sparse_tensor_invariant_checks = false;

// 禁止在 CPU 上使用 FP16 降维
bool allow_fp16_reduction_cpu = false;
// 返回当前系统中可用的 GPU 数量，该函数内部不处理其他设备类型的逻辑。
// 如果要查询特定设备类型的设备数量，应该在相应的库中添加对应的函数（例如，类似于 at::cuda::device_count()）。
static inline size_t getNumGPUs() {
  // 检查当前环境既支持 CUDA 又支持 HIP，这种情况不受支持，会抛出异常。
  // HIP 在 ATen 中会伪装成 CUDA，因此在 HIP 构建的 ATen 中，CUDA 实际上是指 HIP。
  // 如果需要支持 CUDA 或 HIP，请重新构建 PyTorch，禁用其中一个。
  if (hasCUDA() && hasHIP()) {
    throw std::runtime_error(
        "Enabling both CUDA and HIP in ATen is not supported, as HIP masquerades "
        "to be CUDA (e.g., when you say CUDA, on a HIP build of ATen, this actually "
        "means HIP.  Rebuild PyTorch with one or the other disabled.");
  } else if (hasCUDA()) {
    // 如果系统支持 CUDA，则调用 CUDA 的钩子函数获取 GPU 数量。
    return detail::getCUDAHooks().getNumGPUs();
  } else if (hasHIP()) {
    // 如果系统支持 HIP，则调用 HIP 的钩子函数获取 GPU 数量。
    return detail::getHIPHooks().getNumGPUs();
  } else {
    // 如果系统既不支持 CUDA 也不支持 HIP，则返回 GPU 数量为 0。
    return 0;
  }
}

// 检查当前环境是否支持 OpenMP。
static inline bool hasOpenMP() {
  return globalContext().hasOpenMP();
}

// 检查当前环境是否支持 MKL（Math Kernel Library）。
static inline bool hasMKL() {
  return globalContext().hasMKL();
}

// 检查当前环境是否支持 LAPACK。
static inline bool hasLAPACK() {
  return globalContext().hasLAPACK();
}

// 检查当前环境是否支持 MAGMA。
static inline bool hasMAGMA() {
  return globalContext().hasMAGMA();
}

// 检查当前环境是否支持 MKLDNN。
static inline bool hasMKLDNN() {
  return globalContext().hasMKLDNN();
}

// 设置随机数生成器的种子，用于生成随机数。
// 注意：在使用随机数生成器时，必须获取锁，确保线程安全性。
void manual_seed(uint64_t seed) {
  // 获取默认的 CPU 随机数生成器
  auto gen = globalContext().defaultGenerator(c10::DeviceType::CPU);
  {
    // 获取锁，确保在使用随机数生成器时的线程安全性
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_current_seed(seed);
  }

  // 如果系统中有 CUDA 设备且数量大于 0，则为每个 CUDA 设备设置相同的随机种子。
  const auto cuda_num_gpus = detail::getCUDAHooks().getNumGPUs();
  if (hasCUDA() && cuda_num_gpus > 0) {
    for (const auto i : c10::irange(cuda_num_gpus)) {
      auto cuda_gen = globalContext().defaultGenerator(
          Device(at::kCUDA, static_cast<c10::DeviceIndex>(i)));
      {
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        cuda_gen.set_current_seed(seed);
      }
    }
  }

  // 如果系统中有 XPU 设备，则为每个 XPU 设备设置相同的随机种子。
  const auto xpu_num_gpus = detail::getXPUHooks().getNumGPUs();
  if (hasXPU() && xpu_num_gpus) {
    for (const auto i : c10::irange(xpu_num_gpus)) {
      auto xpu_gen = globalContext().defaultGenerator(
          Device(at::kXPU, static_cast<c10::DeviceIndex>(i)));
      {
        std::lock_guard<std::mutex> lock(xpu_gen.mutex());
        xpu_gen.set_current_seed(seed);
      }
    }
  }

  // 如果系统中支持 MPS（Multi-Process Service），则为 MPS 设置相同的随机种子。
  if (hasMPS()) {
    auto mps_gen = globalContext().defaultGenerator(c10::DeviceType::MPS);
    std::lock_guard<std::mutex> lock(mps_gen.mutex());
    mps_gen.set_current_seed(seed);
  }
}
// 这段代码定义了两个与TensorFlow32位精度相关的RAII保护类：NoTF32Guard和ROCmBackwardPassGuard。
// RAII（Resource Acquisition Is Initialization）是一种C++编程中的资源管理策略，利用对象的生命周期来管理资源的获取和释放。
// NoTF32Guard类用于在其作用域内禁用TF32，以防止精度丢失。
// ROCmBackwardPassGuard类用于管理ROCm（Radeon Open Compute平台）的反向传播过程。
// 这些类提供了静态方法和成员变量来操作和检查TF32状态以及ROCm反向传播状态。
struct TORCH_API NoTF32Guard {
  NoTF32Guard();        // 构造函数，用于禁用TF32
  ~NoTF32Guard();       // 析构函数，用于恢复TF32状态
  static bool should_disable_tf32();  // 静态方法，用于检查是否应禁用TF32

 private:
  bool changed = false;  // 成员变量，标记TF32状态是否已更改
};

struct TORCH_API ROCmBackwardPassGuard {
  ROCmBackwardPassGuard();   // 构造函数，用于ROCm反向传播的管理
  ~ROCmBackwardPassGuard();  // 析构函数，用于清理ROCm反向传播状态
  static bool is_backward_pass();  // 静态方法，用于检查是否处于ROCm反向传播过程中
};

} // namespace at
```