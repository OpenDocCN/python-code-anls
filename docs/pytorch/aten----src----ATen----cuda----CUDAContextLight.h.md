# `.\pytorch\aten\src\ATen\cuda\CUDAContextLight.h`

```py
#pragma once
// 表示这个头文件只会被编译一次

// 包含标准整数类型
#include <cstdint>

// 包含 CUDA 运行时 API 的头文件
#include <cuda_runtime_api.h>
// 包含 cusparse 库的头文件
#include <cusparse.h>
// 包含 cublas 库的头文件
#include <cublas_v2.h>

// 在 CUDA 10.1 中引入 cublasLt，但我们只为支持了 bf16 的 CUDA 11.1 开启
#include <cublasLt.h>

// 如果定义了 CUDART_VERSION，则包含 cusolverDn 库的头文件
#ifdef CUDART_VERSION
#include <cusolverDn.h>
#endif

// 如果定义了 USE_ROCM，则包含 hipsolver 库的头文件
#if defined(USE_ROCM)
#include <hipsolver/hipsolver.h>
#endif

// 包含 C10 库的 Allocator 头文件
#include <c10/core/Allocator.h>
// 包含 C10 CUDA 相关函数的头文件
#include <c10/cuda/CUDAFunctions.h>

// 进入 c10 命名空间
namespace c10 {
    // 声明 Allocator 结构体
    struct Allocator;
}

// 进入 at::cuda 命名空间
namespace at::cuda {

/*
 * ATen 的 CUDA 通用接口。
 * 
 * 该接口与 CUDAHooks 是不同的，CUDAHooks 定义了一个链接到 CPU-only 和 CUDA 构建的接口。
 * 那个接口用于运行时调度，应该在同时包含 CPU-only 和 CUDA 构建的文件中使用。
 * 
 * CUDAContext 则应该被仅包含在 CUDA 构建中的文件所优先使用。它旨在以一致的方式暴露 CUDA 功能。
 * 
 * 这意味着 CUDAContext 和 CUDAHooks 之间存在一些重叠，但使用的选择很简单：
 * 在 CUDA-only 文件中使用 CUDAContext，在其他情况下使用 CUDAHooks。
 * 
 * 注意，CUDAContext 只是定义了一个接口，没有关联的类。预期组成该接口的功能模块将管理自己的状态。
 * 只有一个 CUDA 上下文/状态。
 */

/**
 * 不推荐使用：请使用 device_count() 替代
 */
inline int64_t getNumGPUs() {
    // 调用 c10::cuda::device_count() 函数并返回结果
    return c10::cuda::device_count();
}

/**
 * 如果我们编译时启用了 CUDA 并且有一个或多个设备可用，则 CUDA 可用。
 * 如果编译时启用了 CUDA 但存在驱动程序问题等，则此函数将报告 CUDA 不可用（而不是引发错误）。
 */
inline bool is_available() {
    // 返回 c10::cuda::device_count() 是否大于 0 的结果
    return c10::cuda::device_count() > 0;
}

// 获取当前设备的属性并返回 cudaDeviceProp 指针
TORCH_CUDA_CPP_API cudaDeviceProp* getCurrentDeviceProperties();

// 返回当前设备的线程束大小（warp size）
TORCH_CUDA_CPP_API int warp_size();

// 获取指定设备的属性并返回 cudaDeviceProp 指针
TORCH_CUDA_CPP_API cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device);

// 检查设备是否可以访问对等设备
TORCH_CUDA_CPP_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

// 返回 CUDA 设备的分配器 Allocator 指针
TORCH_CUDA_CPP_API c10::Allocator* getCUDADeviceAllocator();

/* Handles */

// 获取当前的 cusparse 句柄并返回
TORCH_CUDA_CPP_API cusparseHandle_t getCurrentCUDASparseHandle();
// 获取当前的 cublas 句柄并返回
TORCH_CUDA_CPP_API cublasHandle_t getCurrentCUDABlasHandle();
// 获取当前的 cublasLt 句柄并返回
TORCH_CUDA_CPP_API cublasLtHandle_t getCurrentCUDABlasLtHandle();

// 清理 cublas 的工作空间
TORCH_CUDA_CPP_API void clearCublasWorkspaces();

// 如果定义了 CUDART_VERSION 或 USE_ROCM，则获取当前的 cusolverDn 句柄并返回
#if defined(CUDART_VERSION) || defined(USE_ROCM)
TORCH_CUDA_CPP_API cusolverDnHandle_t getCurrentCUDASolverDnHandle();
#endif

} // namespace at::cuda
```