# `.\pytorch\aten\src\ATen\detail\CUDAHooksInterface.h`

```
#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <ATen/detail/AcceleratorHooksInterface.h>

// Forward-declares at::Generator and at::cuda::NVRTC
namespace at {
struct Generator;
namespace cuda {
struct NVRTC;
} // namespace cuda
} // namespace at

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

#ifdef _MSC_VER
// 定义错误帮助信息，用于指导用户解决 CUDA 库加载问题，特别是在 Windows 平台上
constexpr const char* CUDA_HELP =
  "PyTorch splits its backend into two shared libraries: a CPU library "
  "and a CUDA library; this error has occurred because you are trying "
  "to use some CUDA functionality, but the CUDA library has not been "
  "loaded by the dynamic linker for some reason.  The CUDA library MUST "
  "be loaded, EVEN IF you don't directly use any symbols from the CUDA library! "
  "One common culprit is a lack of -INCLUDE:?warp_size@cuda@at@@YAHXZ "
  "in your link arguments; many dynamic linkers will delete dynamic library "
  "dependencies if you don't depend on any of their symbols.  You can check "
  "if this has occurred by using link on your binary to see if there is a "
  "dependency on *_cuda.dll library.";
#else
// 定义错误帮助信息，用于指导用户解决 CUDA 库加载问题，特别是在非 Windows 平台上
constexpr const char* CUDA_HELP =
  "PyTorch splits its backend into two shared libraries: a CPU library "
  "and a CUDA library; this error has occurred because you are trying "
  "to use some CUDA functionality, but the CUDA library has not been "
  "loaded by the dynamic linker for some reason.  The CUDA library MUST "
  "be loaded, EVEN IF you don't directly use any symbols from the CUDA library! "
  "One common culprit is a lack of -Wl,--no-as-needed in your link arguments; many "
  "dynamic linkers will delete dynamic library dependencies if you don't "
  "depend on any of their symbols.  You can check if this has occurred by "
  "using ldd on your binary to see if there is a dependency on *_cuda.so "
  "library.";
#endif

// The CUDAHooksInterface is an omnibus interface for any CUDA functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of CUDA code).  How do I
// decide if a function should live in this class?  There are two tests:
//
//  1. Does the *implementation* of this function require linking against
//     CUDA libraries?
//
//  2. Is this function *called* from non-CUDA ATen code?
//
// (2) should filter out many ostensible use-cases, since many times a CUDA
// function provided by ATen is only really ever used by actual CUDA code.
//
// TODO: Consider putting the stub definitions in another class, so that one
// never forgets to implement each virtual function in the real implementation
// in CUDAHooks.  This probably doesn't buy us much though.
// 定义一个 CUDA 接口类 CUDAHooksInterface，继承自 AcceleratorHooksInterface 接口类
struct TORCH_API CUDAHooksInterface : AcceleratorHooksInterface {

  // 虚析构函数，通常不会实际实现，用于避免 -Werror=non-virtual-dtor 的警告
  ~CUDAHooksInterface() override = default;

  // 初始化 THCState 和 CUDA 状态
  virtual void initCUDA() const {
    // 抛出错误，指示无法在缺少 ATen_cuda 库时初始化 CUDA
    TORCH_CHECK(false, "Cannot initialize CUDA without ATen_cuda library. ", CUDA_HELP);
  }

  // 获取默认的 CUDA 生成器，需要 ATen_cuda 库
  virtual const Generator& getDefaultCUDAGenerator(C10_UNUSED DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default CUDA generator without ATen_cuda library. ", CUDA_HELP);
  }

  // 获取指针所在的 CUDA 设备
  virtual Device getDeviceFromPtr(void* /*data*/) const {
    TORCH_CHECK(false, "Cannot get device of pointer on CUDA without ATen_cuda library. ", CUDA_HELP);
  }

  // 检查指针是否为固定内存
  virtual bool isPinnedPtr(const void* /*data*/) const {
    return false;
  }

  // 检查是否支持 CUDA
  virtual bool hasCUDA() const {
    return false;
  }

  // 检查是否支持 CUDART
  virtual bool hasCUDART() const {
    return false;
  }

  // 检查是否支持 MAGMA
  virtual bool hasMAGMA() const {
    return false;
  }

  // 检查是否支持 CuDNN
  virtual bool hasCuDNN() const {
    return false;
  }

  // 检查是否支持 CuSOLVER
  virtual bool hasCuSOLVER() const {
    return false;
  }

  // 检查是否支持 CuBLASLt
  virtual bool hasCuBLASLt() const {
    return false;
  }

  // 检查是否支持 ROCm
  virtual bool hasROCM() const {
    return false;
  }

  // 返回 NVRTC 对象，需要 CUDA 支持
  virtual const at::cuda::NVRTC& nvrtc() const {
    TORCH_CHECK(false, "NVRTC requires CUDA. ", CUDA_HELP);
  }

  // 检查是否具有主要 CUDA 上下文
  bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(false, "Cannot call hasPrimaryContext(", device_index, ") without ATen_cuda library. ", CUDA_HELP);
  }

  // 获取当前设备索引，未初始化 CUDA 时返回 -1
  virtual DeviceIndex current_device() const {
    return -1;
  }

  // 获取固定内存分配器，需要 CUDA 支持
  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Pinned memory requires CUDA. ", CUDA_HELP);
  }

  // 获取 CUDA 设备分配器，需要 CUDA 支持
  virtual Allocator* getCUDADeviceAllocator() const {
    TORCH_CHECK(false, "CUDADeviceAllocator requires CUDA. ", CUDA_HELP);
  }

  // 检查是否使用了 CuDNN 编译
  virtual bool compiledWithCuDNN() const {
    return false;
  }

  // 检查是否使用了 MIOpen 编译
  virtual bool compiledWithMIOpen() const {
    return false;
  }

  // 检查是否支持带有 CuDNN 的扩展卷积
  virtual bool supportsDilatedConvolutionWithCuDNN() const {
    return false;
  }

  // 检查是否支持带有 CuDNN 的深度卷积
  virtual bool supportsDepthwiseConvolutionWithCuDNN() const {
    return false;
  }

  // 检查是否支持使用 CuDNNv8 进行 BFloat16 卷积
  virtual bool supportsBFloat16ConvolutionWithCuDNNv8() const {
    return false;
  }

  // 查询 CuDNN 版本号，需要 CUDA 支持
  virtual long versionCuDNN() const {
    TORCH_CHECK(false, "Cannot query cuDNN version without ATen_cuda library. ", CUDA_HELP);
  }

  // 查询 CUDART 版本号，需要 CUDA 支持
  virtual long versionCUDART() const {
    TORCH_CHECK(false, "Cannot query CUDART version without ATen_cuda library. ", CUDA_HELP);
  }

  // 查询详细的 CUDA 配置信息，需要 CUDA 支持
  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed CUDA version without ATen_cuda library. ", CUDA_HELP);
  }

  // 查询 CuDNN 中的 batchnormMinEpsilon，需要 CUDA 支持
  virtual double batchnormMinEpsilonCuDNN() const {
    TORCH_CHECK(false,
        "Cannot query batchnormMinEpsilonCuDNN() without ATen_cuda library. ", CUDA_HELP);
  }

  // 获取 CuFFT 的计划缓存最大大小，未初始化 CUDA 时返回默认值
  virtual int64_t cuFFTGetPlanCacheMaxSize(DeviceIndex /*device_index*/) const {
    # 检查条件为 false，抛出错误并显示相关 CUDA 帮助信息，表示无法访问 cuFFT 计划缓存
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual void cuFFTSetPlanCacheMaxSize(DeviceIndex /*device_index*/, int64_t /*max_size*/) const {
    # 检查条件为 false，抛出错误并显示相关 CUDA 帮助信息，表示无法访问 cuFFT 计划缓存
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual int64_t cuFFTGetPlanCacheSize(DeviceIndex /*device_index*/) const {
    # 检查条件为 false，抛出错误并显示相关 CUDA 帮助信息，表示无法访问 cuFFT 计划缓存
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual void cuFFTClearPlanCache(DeviceIndex /*device_index*/) const {
    # 检查条件为 false，抛出错误并显示相关 CUDA 帮助信息，表示无法访问 cuFFT 计划缓存
    TORCH_CHECK(false, "Cannot access cuFFT plan cache without ATen_cuda library. ", CUDA_HELP);
  }

  virtual int getNumGPUs() const {
    # 返回值为 0，表示当前没有 GPU 设备可用
    return 0;
  }

  virtual void deviceSynchronize(DeviceIndex /*device_index*/) const {
    # 检查条件为 false，抛出错误并显示相关 CUDA 帮助信息，表示无法同步 CUDA 设备
    TORCH_CHECK(false, "Cannot synchronize CUDA device without ATen_cuda library. ", CUDA_HELP);
  }
};

// 闭合了一个匿名命名空间，这里用来结束命名空间的定义

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
// 注释：这是一个宏定义的说明，用于在定义可变参数宏时传递一个虚拟的参数，
// 避免编译器报错提示需要至少一个参数的情况。

// 声明一个结构体 CUDAHooksArgs，用作 C++ 结构体的占位符参数
struct TORCH_API CUDAHooksArgs {};

// 声明 CUDAHooksRegistry 类型的注册表，注册的对象类型为 CUDAHooksInterface，参数类型为 CUDAHooksArgs
TORCH_DECLARE_REGISTRY(CUDAHooksRegistry, CUDAHooksInterface, CUDAHooksArgs);

// 定义宏 REGISTER_CUDA_HOOKS(clsname)，注册 CUDA 钩子的类名称
#define REGISTER_CUDA_HOOKS(clsname) \
  C10_REGISTER_CLASS(CUDAHooksRegistry, clsname, clsname)

// 进入命名空间 detail，定义了一个名为 getCUDAHooks 的函数或者对象，返回类型为 CUDAHooksInterface 的引用
namespace detail {
TORCH_API const CUDAHooksInterface& getCUDAHooks();
} // namespace detail

// 结束命名空间 at
} // namespace at
```