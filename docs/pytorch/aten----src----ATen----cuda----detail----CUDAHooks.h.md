# `.\pytorch\aten\src\ATen\cuda\detail\CUDAHooks.h`

```
#pragma once
// 使用#pragma once确保此头文件仅被编译一次

#include <ATen/detail/CUDAHooksInterface.h>
// 引入ATen库中的CUDAHooksInterface头文件

#include <ATen/Generator.h>
// 引入ATen库中的Generator头文件

#include <c10/util/Optional.h>
// 引入c10库中的Optional头文件

// TODO: No need to have this whole header, we can just put it all in
// the cpp file
// TODO注释，建议将整个定义移到cpp文件中，不需要保留在头文件中

namespace at::cuda::detail {
// 命名空间at::cuda::detail，包含下面的CUDAHooks结构体的实现

// Set the callback to initialize Magma, which is set by
// torch_cuda_cu. This indirection is required so magma_init is called
// in the same library where Magma will be used.
// 设置回调函数以初始化Magma，该函数由torch_cuda_cu设置。这种间接性是必需的，
// 因此在使用Magma的同一库中调用magma_init。

TORCH_CUDA_CPP_API void set_magma_init_fn(void (*magma_init_fn)());
// 声明了一个名为set_magma_init_fn的函数，接受一个指向函数的指针作为参数，该函数没有返回值。
// 该函数的作用是设置初始化Magma的回调函数。

// The real implementation of CUDAHooksInterface
// CUDAHooksInterface的实际实现

struct CUDAHooks : public at::CUDAHooksInterface {
  CUDAHooks(at::CUDAHooksArgs) {}
  // CUDAHooks结构体的构造函数，接受一个at::CUDAHooksArgs参数，并在成员初始化列表中将其传递给基类构造函数。

  void initCUDA() const override;
  // 实现CUDAHooksInterface中的initCUDA方法，用于初始化CUDA环境。

  Device getDeviceFromPtr(void* data) const override;
  // 实现CUDAHooksInterface中的getDeviceFromPtr方法，根据指针获取设备信息。

  bool isPinnedPtr(const void* data) const override;
  // 实现CUDAHooksInterface中的isPinnedPtr方法，判断指针是否为固定内存指针。

  const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1) const override;
  // 实现CUDAHooksInterface中的getDefaultCUDAGenerator方法，获取默认的CUDA生成器。

  bool hasCUDA() const override;
  // 实现CUDAHooksInterface中的hasCUDA方法，检查系统是否支持CUDA。

  bool hasMAGMA() const override;
  // 实现CUDAHooksInterface中的hasMAGMA方法，检查系统是否支持MAGMA。

  bool hasCuDNN() const override;
  // 实现CUDAHooksInterface中的hasCuDNN方法，检查系统是否支持CuDNN。

  bool hasCuSOLVER() const override;
  // 实现CUDAHooksInterface中的hasCuSOLVER方法，检查系统是否支持CuSOLVER。

  bool hasCuBLASLt() const override;
  // 实现CUDAHooksInterface中的hasCuBLASLt方法，检查系统是否支持CuBLASLt。

  bool hasROCM() const override;
  // 实现CUDAHooksInterface中的hasROCM方法，检查系统是否支持ROCM。

  const at::cuda::NVRTC& nvrtc() const override;
  // 实现CUDAHooksInterface中的nvrtc方法，获取CUDA的NVRTC接口。

  DeviceIndex current_device() const override;
  // 实现CUDAHooksInterface中的current_device方法，获取当前使用的CUDA设备索引。

  bool hasPrimaryContext(DeviceIndex device_index) const override;
  // 实现CUDAHooksInterface中的hasPrimaryContext方法，检查指定设备是否有主上下文。

  Allocator* getCUDADeviceAllocator() const override;
  // 实现CUDAHooksInterface中的getCUDADeviceAllocator方法，获取CUDA设备的分配器。

  Allocator* getPinnedMemoryAllocator() const override;
  // 实现CUDAHooksInterface中的getPinnedMemoryAllocator方法，获取固定内存的分配器。

  bool compiledWithCuDNN() const override;
  // 实现CUDAHooksInterface中的compiledWithCuDNN方法，检查编译是否支持CuDNN。

  bool compiledWithMIOpen() const override;
  // 实现CUDAHooksInterface中的compiledWithMIOpen方法，检查编译是否支持MIOpen。

  bool supportsDilatedConvolutionWithCuDNN() const override;
  // 实现CUDAHooksInterface中的supportsDilatedConvolutionWithCuDNN方法，检查系统是否支持带扩张的CuDNN卷积。

  bool supportsDepthwiseConvolutionWithCuDNN() const override;
  // 实现CUDAHooksInterface中的supportsDepthwiseConvolutionWithCuDNN方法，检查系统是否支持深度卷积。

  bool supportsBFloat16ConvolutionWithCuDNNv8() const override;
  // 实现CUDAHooksInterface中的supportsBFloat16ConvolutionWithCuDNNv8方法，检查系统是否支持CuDNN v8的BFloat16卷积。

  bool hasCUDART() const override;
  // 实现CUDAHooksInterface中的hasCUDART方法，检查系统是否支持CUDART。

  long versionCUDART() const override;
  // 实现CUDAHooksInterface中的versionCUDART方法，获取CUDART的版本号。

  long versionCuDNN() const override;
  // 实现CUDAHooksInterface中的versionCuDNN方法，获取CuDNN的版本号。

  std::string showConfig() const override;
  // 实现CUDAHooksInterface中的showConfig方法，显示CUDA的配置信息。

  double batchnormMinEpsilonCuDNN() const override;
  // 实现CUDAHooksInterface中的batchnormMinEpsilonCuDNN方法，获取CuDNN中batchnorm的最小epsilon值。

  int64_t cuFFTGetPlanCacheMaxSize(DeviceIndex device_index) const override;
  // 实现CUDAHooksInterface中的cuFFTGetPlanCacheMaxSize方法，获取指定设备上cuFFT的计划缓存最大大小。

  void cuFFTSetPlanCacheMaxSize(DeviceIndex device_index, int64_t max_size) const override;
  // 实现CUDAHooksInterface中的cuFFTSetPlanCacheMaxSize方法，设置指定设备上cuFFT的计划缓存最大大小。

  int64_t cuFFTGetPlanCacheSize(DeviceIndex device_index) const override;
  // 实现CUDAHooksInterface中的cuFFTGetPlanCacheSize方法，获取指定设备上cuFFT的计划缓存大小。

  void cuFFTClearPlanCache(DeviceIndex device_index) const override;
  // 实现CUDAHooksInterface中的cuFFTClearPlanCache方法，清除指定设备上cuFFT的计划缓存。

  int getNumGPUs() const override;
  // 实现CUDAHooksInterface中的getNumGPUs方法，获取系统中GPU的数量。

  void deviceSynchronize(DeviceIndex device_index) const override;
  // 实现CUDAHooksInterface中的deviceSynchronize方法，同步指定设备的所有流。
};

} // at::cuda::detail
// 命名空间at::cuda::detail的结束标记
```