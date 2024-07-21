# `.\pytorch\aten\src\ATen\cuda\detail\CUDAHooks.cpp`

```py
// 包含 ATen CUDA 相关的头文件

#include <ATen/cuda/detail/CUDAHooks.h>

// 包含其他 ATen CUDA 组件的头文件
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/core/Vitals.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/cuda/PinnedMemoryAllocator.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/irange.h>

// 根据条件包含额外的 CUDA 相关头文件

#if AT_CUDNN_ENABLED()
#include <ATen/cudnn/cudnn-wrapper.h>
#endif

#if AT_MAGMA_ENABLED()
#include <magma_v2.h>
#endif

#if defined(USE_ROCM)
#include <miopen/version.h>
#endif

#ifndef USE_ROCM
#include <ATen/cuda/detail/LazyNVRTC.h>
#endif

// 包含 CUDA 核心头文件

#include <cuda.h>

// 包含标准库头文件

#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>

// 定义命名空间 c10::cuda::_internal 中的函数 setHasPrimaryContext

namespace c10::cuda::_internal {
void setHasPrimaryContext(bool (*func)(DeviceIndex));
}

// 定义命名空间 at::cuda::detail

namespace at::cuda::detail {

// 声明函数 nvrtc() 和 current_device()，并定义一个静态成员函数 magma_init_fn

const at::cuda::NVRTC& nvrtc();
DeviceIndex current_device();

static void (*magma_init_fn)() = nullptr;

// 定义函数 set_magma_init_fn，设置 magma_init_fn 的函数指针

void set_magma_init_fn(void (*fn)()) {
  magma_init_fn = fn;
}

// 定义匿名命名空间，包含 _hasPrimaryContext 函数的实现

namespace {
bool _hasPrimaryContext(DeviceIndex device_index) {
  // 检查设备索引是否有效
  TORCH_CHECK(device_index >= 0 && device_index < at::cuda::device_count(),
              "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
  
  unsigned int ctx_flags;
  int ctx_is_active = 0;
  
  // 调用 cuDevicePrimaryCtxGetState 获取主要上下文的状态
  AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  
  // 返回主要上下文是否活跃的布尔值
  return ctx_is_active == 1;
}

// 定义 _Initializer 结构体，在构造函数中注册 _hasPrimaryContext 函数到 c10::cuda::_internal 命名空间

struct _Initializer {
  _Initializer() {
      c10::cuda::_internal::setHasPrimaryContext(_hasPrimaryContext);
  }
  ~_Initializer() {
      // 在析构函数中取消注册 _hasPrimaryContext 函数
      c10::cuda::_internal::setHasPrimaryContext(nullptr);
  }
} initializer;
} // anonymous namespace

// 定义函数 maybe_set_cuda_module_loading，设置 CUDA_MODULE_LOADING 环境变量的默认值

void maybe_set_cuda_module_loading(const std::string &def_value) {
  auto value = std::getenv("CUDA_MODULE_LOADING");
  
  // 如果 CUDA_MODULE_LOADING 环境变量未设置，则设置其默认值
  if (!value) {
#ifdef _WIN32
    auto env_var = "CUDA_MODULE_LOADING=" + def_value;
    _putenv(env_var.c_str());
#else
    setenv("CUDA_MODULE_LOADING", def_value.c_str(), 1);
#endif
  }
}

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
void CUDAHooks::initCUDA() const {
  C10_LOG_API_USAGE_ONCE("aten.init.cuda");
  // 强制更新以启用单元测试。此代码在单元测试有机会启用 vitals 之前执行。
  // 设置 CUDA 使用情况为 true，可能会强制更新
  at::vitals::VitalsAPI.setVital("CUDA", "used", "true", /* force = */ true);

  // 可能设置 CUDA 模块加载方式为 LAZY
  maybe_set_cuda_module_loading("LAZY");
  // 获取 CUDA 设备数量，确保至少一个 CUDA 设备可用
  const auto num_devices = c10::cuda::device_count_ensure_non_zero();
  // 初始化 CUDA 缓存分配器
  c10::cuda::CUDACachingAllocator::init(num_devices);
  // 初始化 CUDA 设备之间的点对点访问缓存
  at::cuda::detail::init_p2p_access_cache(num_devices);

#if AT_MAGMA_ENABLED()
  // 如果启用了 MAGMA，则执行其初始化函数，否则抛出断言错误
  TORCH_INTERNAL_ASSERT(magma_init_fn != nullptr, "Cannot initialize magma, init routine not set");
  magma_init_fn();
#endif
}

// 获取默认的 CUDA 生成器
const Generator& CUDAHooks::getDefaultCUDAGenerator(DeviceIndex device_index) const {
  return at::cuda::detail::getDefaultCUDAGenerator(device_index);
}

// 从指针获取设备信息
Device CUDAHooks::getDeviceFromPtr(void* data) const {
  return at::cuda::getDeviceFromPtr(data);
}

// 检查指针是否为 pinned memory
bool CUDAHooks::isPinnedPtr(const void* data) const {
  // 如果 CUDA 不可用，则返回 false
  if (!at::cuda::is_available()) {
    return false;
  }
  // 设置当前设备的上下文来获取 CUDA 指针属性
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(at::Device(at::DeviceType::CUDA, *primary_ctx_device_index));
  }
  cudaPointerAttributes attr;
  // 获取 CUDA 指针的属性，不需要对数据进行修改
  cudaError_t err = cudaPointerGetAttributes(&attr, data);
#if !defined(USE_ROCM)
  // 如果错误为 cudaErrorInvalidValue，则清除 CUDA 错误并返回 false
  if (err == cudaErrorInvalidValue) {
    (void)cudaGetLastError(); // 清除 CUDA 错误
    return false;
  }
  // 检查 CUDA 错误
  AT_CUDA_CHECK(err);
#else
  // 在 HIP 环境下，如果出现错误不是 cudaSuccess，则清除 HIP 错误并返回 false
  if (err != cudaSuccess) {
    (void)cudaGetLastError(); // 清除 HIP 错误
    return false;
  }
#endif
  // 返回指针属性中是否为 Host 内存的信息
  return attr.type == cudaMemoryTypeHost;
}

// 检查系统中是否有 CUDA 设备可用
bool CUDAHooks::hasCUDA() const {
  return at::cuda::is_available();
}

// 检查是否启用了 MAGMA
bool CUDAHooks::hasMAGMA() const {
#if AT_MAGMA_ENABLED()
  return true;
#else
  return false;
#endif
}

// 检查系统中是否有 CuDNN 可用
bool CUDAHooks::hasCuDNN() const {
  return AT_CUDNN_ENABLED();
}

// 检查系统中是否有 CuSOLVER 可用
bool CUDAHooks::hasCuSOLVER() const {
#if defined(CUDART_VERSION) && defined(CUSOLVER_VERSION)
  return true;
#elif AT_ROCM_ENABLED()
  return true;
#else
  return false;
#endif
}

// 检查系统中是否有 CuBLASLt 可用
bool CUDAHooks::hasCuBLASLt() const {
#if defined(CUDART_VERSION)
  return true;
#elif AT_ROCM_ENABLED()
  return true;
#else
  return false;
#endif
}

// 检查系统是否编译了 ROCm
bool CUDAHooks::hasROCM() const {
  // 目前与 compiledWithMIOpen 相同，未来可能会有不包含 MIOpen 的 ROCm 构建
  return AT_ROCM_ENABLED();
}

#if defined(USE_DIRECT_NVRTC)
#if AT_CUDNN_ENABLED()
  // 如果编译时启用了 CuDNN，返回 true
  return true;
#else
  // 否则返回 false
  return false;
#endif


cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
// 获取当前 CUDA 设备的属性
// 检查是否支持 Volta 架构的深度卷积
if (prop->major >= 7) {
  // 如果设备主版本号大于等于 7，返回 true
  return true;
} else {
  // 否则返回 false
  return false;
}


cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
// 获取当前 CUDA 设备的属性
// 检查是否支持 Volta 架构的 BFloat16 卷积（CuDNN 版本 8）
if (prop->major >= 8) {
  // 如果设备主版本号大于等于 8，返回 true
  return true;
} else {
  // 否则返回 false
  return false;
}


#if AT_CUDNN_ENABLED()
  // 如果编译时启用了 CuDNN，返回当前 CuDNN 的版本号
  return CUDNN_VERSION;
#else
  // 否则抛出错误，因为 ATen_cuda 没有使用 CuDNN 编译
  AT_ERROR("Cannot query CuDNN version if ATen_cuda is not built with CuDNN");
#endif


#ifdef CUDART_VERSION
  // 如果定义了 CUDART_VERSION，返回当前 CUDA 运行时的版本号
  return CUDART_VERSION;
#else
  // 否则抛出错误，通常不应该发生，除非 CUDA 运行时版本未定义
  TORCH_CHECK(
    false,
    "CUDA Runtime version is not defined. Please rebuild with CUDART_VERSION defined."
  );
#endif
    "Cannot query CUDART version because CUDART is not available");


    # 输出错误信息，指示无法查询 CUDART 版本因为 CUDART 不可用
// 结束函数定义
#endif
}

// 检查是否定义了 CUDART_VERSION 宏，若定义则返回 true，否则返回 false
bool CUDAHooks::hasCUDART() const {
#ifdef CUDART_VERSION
  return true;
#else
  return false;
#endif
}

// 返回一个描述 CUDA 或 ROCm 运行时配置的字符串
std::string CUDAHooks::showConfig() const {
  std::ostringstream oss;

  int runtimeVersion;
  // 获取 CUDA 运行时版本号
  cudaRuntimeGetVersion(&runtimeVersion);

  // 定义一个 lambda 函数，根据输入的版本号 v，输出 CUDA 风格的版本号字符串
  auto printCudaStyleVersion = [&](int v) {
#ifdef USE_ROCM
    // 如果版本号 v 小于 500，则格式为 xxyy
    if(v < 500) {
      oss << (v / 100) << "." << (v % 10);
    }
    else {
      // 如果版本号 v 大于等于 500，则格式为 xxyyzzzzz
      oss << (v / 10000000) << "." << (v / 100000 % 100) << "." << (v % 100000);
    }
#else
    // 否则，格式为 x.y 或 x.y.z
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
#endif
  };

// 如果未定义 USE_ROCM 宏
#if !defined(USE_ROCM)
  oss << "  - CUDA Runtime ";
#else
  oss << "  - HIP Runtime ";
#endif
  // 打印 CUDA 运行时版本号
  printCudaStyleVersion(runtimeVersion);
  oss << "\n";

  // TODO: 让 HIPIFY 理解 CUDART_VERSION 宏
#if !defined(USE_ROCM)
  // 如果运行时版本号不等于 CUDART_VERSION，则打印构建时使用的 CUDA 运行时版本号
  if (runtimeVersion != CUDART_VERSION) {
    oss << "  - Built with CUDA Runtime ";
    printCudaStyleVersion(CUDART_VERSION);
    oss << "\n";
  }
  // 打印 NVCC 架构标志
  oss << "  - NVCC architecture flags: " << NVCC_FLAGS_EXTRA << "\n";
#endif

// 如果未定义 USE_ROCM 宏，并且定义了 AT_CUDNN_ENABLED 宏
#if !defined(USE_ROCM)
#if AT_CUDNN_ENABLED()

  // 定义一个 lambda 函数，根据输入的版本号 v，输出 CuDNN 风格的版本号字符串
  auto printCudnnStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 100 % 10);
    if (v % 100 != 0) {
      oss << "." << (v % 100);
    }
  };

  // 获取当前 CuDNN 版本号
  size_t cudnnVersion = cudnnGetVersion();
  oss << "  - CuDNN ";
  // 打印 CuDNN 版本号
  printCudnnStyleVersion(cudnnVersion);
  size_t cudnnCudartVersion = cudnnGetCudartVersion();
  // 如果 CuDNN 版本号与 CUDA 运行时版本号不同，则打印构建时使用的 CUDA 运行时版本号
  if (cudnnCudartVersion != CUDART_VERSION) {
    oss << "  (built against CUDA ";
    printCudaStyleVersion(cudnnCudartVersion);
    oss << ")";
  }
  oss << "\n";
  // 如果当前 CuDNN 版本号不等于预定义的 CUDNN_VERSION，则打印预定义的 CuDNN 版本号
  if (cudnnVersion != CUDNN_VERSION) {
    oss << "    - Built with CuDNN ";
    printCudnnStyleVersion(CUDNN_VERSION);
    oss << "\n";
  }
#endif
#else
  // TODO: 检查 miopen 是否具有上述功能，并进行统一处理
  oss << "  - MIOpen " << MIOPEN_VERSION_MAJOR << "." << MIOPEN_VERSION_MINOR << "." << MIOPEN_VERSION_PATCH << "\n";
#endif

// 如果 AT_MAGMA_ENABLED 宏已定义
#if AT_MAGMA_ENABLED()
  // 打印 Magma 版本号
  oss << "  - Magma " << MAGMA_VERSION_MAJOR << "." << MAGMA_VERSION_MINOR << "." << MAGMA_VERSION_MICRO << "\n";
#endif

  // 返回配置信息字符串
  return oss.str();
}

// 返回 CUDNN_BN_MIN_EPSILON 常量，如果未启用 CuDNN 则抛出错误信息
double CUDAHooks::batchnormMinEpsilonCuDNN() const {
#if AT_CUDNN_ENABLED()
  return CUDNN_BN_MIN_EPSILON;
#else
  // 如果未构建 ATen_cuda 使用 CuDNN，则抛出错误信息
  AT_ERROR(
      "Cannot query CUDNN_BN_MIN_EPSILON if ATen_cuda is not built with CuDNN");
#endif
}

// 返回指定设备索引的 cuFFT 计划缓存的最大大小
int64_t CUDAHooks::cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const {
  return at::native::detail::cufft_get_plan_cache_max_size_impl(device_index);
}

// 设置指定设备索引的 cuFFT 计划缓存的最大大小
void CUDAHooks::cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size) const {
  at::native::detail::cufft_set_plan_cache_max_size_impl(device_index, max_size);
}
// 获取 cuFFT 计划缓存的大小，通过调用 cuFFT 的实现函数
int64_t CUDAHooks::cuFFTGetPlanCacheSize(DeviceIndex device_index) const {
    // 调用 cufft_get_plan_cache_size_impl 函数获取 cuFFT 计划缓存的大小并返回
    return at::native::detail::cufft_get_plan_cache_size_impl(device_index);
}

// 清除 cuFFT 计划缓存，通过调用 cuFFT 的实现函数
void CUDAHooks::cuFFTClearPlanCache(DeviceIndex device_index) const {
    // 调用 cufft_clear_plan_cache_impl 函数清除 cuFFT 计划缓存
    at::native::detail::cufft_clear_plan_cache_impl(device_index);
}

// 获取 CUDA 设备的数量
int CUDAHooks::getNumGPUs() const {
    // 调用 at::cuda::device_count() 函数返回 CUDA 设备的数量
    return at::cuda::device_count();
}

// 在特定 CUDA 设备上同步所有流中的操作
void CUDAHooks::deviceSynchronize(DeviceIndex device_index) const {
    // 通过 at::DeviceGuard 确保操作在指定设备上进行
    at::DeviceGuard device_guard(at::Device(at::DeviceType::CUDA, device_index));
    // 调用 c10::cuda::device_synchronize() 函数实现设备上所有流的同步
    c10::cuda::device_synchronize();
}

// 注册 CUDA 钩子，由于注册表不支持命名空间，使用全局命名
using at::CUDAHooksRegistry;
using at::RegistererCUDAHooksRegistry;

// 将 CUDAHooks 类注册到 CUDA 钩子注册表中
REGISTER_CUDA_HOOKS(CUDAHooks);

// 结束 at::cuda::detail 命名空间
} // namespace at::cuda::detail
```