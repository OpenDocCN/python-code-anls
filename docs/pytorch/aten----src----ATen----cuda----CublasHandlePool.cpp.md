# `.\pytorch\aten\src\ATen\cuda\CublasHandlePool.cpp`

```
/**
 * 清除 cuBLAS 工作空间
 * ~~~~~~~~~~~~~~~~~~~~~~
 * 清除当前设备的 cuBLAS 工作空间，根据编译选项决定是否清除 ROCm 构建中的工作空间。
 */

namespace at::cuda {

namespace {

/**
 * 创建 cuBLAS 句柄
 * ~~~~~~~~~~~~~~~~~~
 * 创建一个新的 cuBLAS 句柄，并使用 TORCH_CUDABLAS_CHECK 进行错误检查。
 *
 * @param handle 指向 cublasHandle_t 的指针，用于存储新创建的句柄
 */
void createCublasHandle(cublasHandle_t *handle) {
  TORCH_CUDABLAS_CHECK(cublasCreate(handle));
}

/**
 * 销毁 cuBLAS 句柄
 * ~~~~~~~~~~~~~~~~~~
 * 销毁给定的 cuBLAS 句柄，根据编译选项决定是否实际执行销毁操作。
 *
 * @param handle 待销毁的 cublasHandle_t 句柄
 */
void destroyCublasHandle(cublasHandle_t handle) {
// 由于销毁顺序不当，在某些情况下会发生问题。在 fbcode 环境中已观察到 CUDA 上下文
// 或其他组件在销毁时已不再有效的情况。@colesbury 和 @soumith 决定不销毁句柄作为临时解决方案。
// - @soumith 的评论，摘自 cuDNN 句柄池实现
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cublasDestroy(handle);
#endif
}

#if defined(USE_ROCM)
/**
 * 创建 cuBLASLt 句柄
 * ~~~~~~~~~~~~~~~~~~~~
 * 创建一个新的 cuBLASLt 句柄，并使用 TORCH_CUDABLAS_CHECK 进行错误检查。
 *
 * @param handle 指向 cublasLtHandle_t 的指针，用于存储新创建的 cuBLASLt 句柄
 */
void createCublasLtHandle(cublasLtHandle_t *handle) {
  TORCH_CUDABLAS_CHECK(cublasLtCreate(handle));
}

/**
 * 销毁 cuBLASLt 句柄
 * ~~~~~~~~~~~~~~~~~~~~
 * 销毁给定的 cuBLASLt 句柄，根据编译选项决定是否实际执行销毁操作。
 * 在某些 fbcode 环境中，观察到由于销毁顺序问题，有时 CUDA 上下文或其他组件已经销毁。
 * @colesbury 和 @soumith 决定不销毁句柄作为临时解决方案。
 *
 * @param handle 待销毁的 cublasLtHandle_t 句柄
 */
void destroyCublasLtHandle(cublasLtHandle_t handle) {
// 由于销毁顺序不当，在某些情况下会发生问题。在 fbcode 环境中已观察到 CUDA 上下文
// 或其他组件在销毁时已不再有效的情况。@colesbury 和 @soumith 决定不销毁句柄作为临时解决方案。
// - @soumith 的评论，摘自 cuDNN 句柄池实现
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
    cublasLtDestroy(handle);
#endif
}

/**
 * cuBLASLt 句柄池类型
 * ~~~~~~~~~~~~~~~~~~~~~
 * 使用 DeviceThreadHandlePool 模板定义的 cuBLASLt 句柄池类型，用于管理多个 cuBLASLt 句柄。
 */
using CuBlasLtPoolType = DeviceThreadHandlePool<cublasLtHandle_t, createCublasLtHandle, destroyCublasLtHandle>;
#endif

/**
 * cuBLAS 句柄池类型
 * ~~~~~~~~~~~~~~~~~~
 * 使用 DeviceThreadHandlePool 模板定义的 cuBLAS 句柄池类型，用于管理多个 cuBLAS 句柄。
 */
using CuBlasPoolType = DeviceThreadHandlePool<cublasHandle_t, createCublasHandle, destroyCublasHandle>;

} // namespace

/**
 * 清除 cuBLAS 工作空间
 * ~~~~~~~~~~~~~~~~~~~~~~~
 * 清除 cuBLAS 句柄与流到工作空间的映射关系，仅当未使用 ROCm 构建时才执行清除操作。
 */
void clearCublasWorkspaces() {
  #if !defined(USE_ROCM)
      cublas_handle_stream_to_workspace().clear();
  #endif
}

} // namespace at::cuda
size_t parseChosenWorkspaceSize() {
  // 获取环境变量"CUBLAS_WORKSPACE_CONFIG"的值
  const char * val = getenv("CUBLAS_WORKSPACE_CONFIG");
  /* :4096:2:16:8 default, 32MiB for Hopper */
  // 获取当前 CUDA 设备的属性
  cudaDeviceProp* properties = at::cuda::getCurrentDeviceProperties();
  // 检查当前设备是否是 sm_90 架构
  const bool sm90 = properties != nullptr && properties->major == 9 && properties->minor == 0;
  // 根据 sm_90 架构确定默认的工作空间大小
  const size_t default_size = sm90 ? 4096 * 8 * 1024 : 4096 * 1024 * 2 + 16 * 1024 * 8;

  if (val) {
    size_t total_size = 0;
    // 将环境变量值转换为字符串进行配置解析
    const std::string config(val);
    // 使用正则表达式解析配置字符串，匹配格式为:SIZE:COUNT
    std::regex exp(":([0-9]+):([0-9]+)");
    std::sregex_iterator next(config.begin(), config.end(), exp);
    std::sregex_iterator end;
    // 如果没有匹配到有效配置，使用默认的工作空间大小并发出警告
    if (next == end) {
      TORCH_WARN("Could not parse CUBLAS_WORKSPACE_CONFIG, using default workspace size of ", default_size, " bytes.");
      return default_size;
    }
    // 解析匹配到的配置项，计算总的工作空间大小
    while (next != end) {
      std::smatch match = *next;
      TORCH_CHECK(match.size() == 3, "Expected CUBLAS_WORKSPACE_SPACE_CONFIG match of size 3 (Format :SIZE:COUNT)");
      size_t curr_size = (size_t) std::stoi(match.str(1));
      size_t count = (size_t) std::stoi(match.str(2));
      total_size += curr_size * 1024 * count;
      next++;
    }
    return total_size;
  } else {
    // 如果未设置环境变量，则返回默认的工作空间大小
    return default_size;
  }
}

size_t getChosenWorkspaceSize() {
  // 获取选择的工作空间大小
  size_t pool_size = parseChosenWorkspaceSize();
  return pool_size;
}

at::DataPtr getNewWorkspace() {
  // 分配新的工作空间内存
  return c10::cuda::CUDACachingAllocator::get()->allocate(getChosenWorkspaceSize());
}

cublasHandle_t getCurrentCUDABlasHandle() {
  // 获取当前 CUDA Blas 句柄

  // 获取当前设备索引
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

#if !defined(USE_ROCM)
  // 获取当前 CUDA 上下文
  CUcontext pctx = nullptr;
  at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx);
  if (C10_UNLIKELY(!pctx)) {
    // 解决当没有当前 CUDA 上下文时的问题
    TORCH_WARN_ONCE("Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context...");
    at::globalContext().getNVRTC().cuDevicePrimaryCtxRetain(&pctx, device);
    at::globalContext().getNVRTC().cuCtxSetCurrent(pctx);
  }
#endif

  // 线程局部变量，延迟初始化 PoolWindow，避免在 Windows 上出现的初始化问题
  // 参考：https://github.com/pytorch/pytorch/pull/22405
  // 线程局部唯一指针在线程终止时将被销毁，将其保留的句柄返回到池中

  // 使用泄漏单例模式管理池，遵循单例的标准实践
  static auto pool = std::shared_ptr<CuBlasPoolType>(
      new CuBlasPoolType(), [](CuBlasPoolType* p) {
        // 内存泄漏处理：故意泄漏内存，因为单例模式需要保持内存不释放
      });
  // 线程局部唯一指针，用于管理 CuBlasPoolType::PoolWindow
  thread_local std::unique_ptr<CuBlasPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  // 获取当前设备的 CuBLAS 句柄
  auto handle = myPoolWindow->reserve(device);
  // 获取当前 CUDA 流
  auto stream = c10::cuda::getCurrentCUDAStream();
  // 将 CuBLAS 句柄关联到当前 CUDA 流
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
#if !defined(USE_ROCM)
  // 如果未定义 USE_ROCM

  // 我们明确设置 cublas 的工作空间，即使 CUDA 12.2+ 修复了在图形捕获期间内存使用增加的问题。
  // 原始问题见：https://github.com/pytorch/pytorch/pull/83461
  // 这是因为在 CUDA 12.2+ 中，cublas 中使用 cudaMallocAsync 将动态分配内存（即使它们很便宜），
  // 超出了 PyTorch 的 CUDA 缓存分配器。可能是 CUDA 缓存分配器耗尽了所有内存，
  // 而 cublas 的 cudaMallocAsync 将返回内存不足的错误。
  cudaStream_t _stream = stream;
  auto key = std::make_tuple(static_cast<void *>(handle), static_cast<void *>(_stream));
  auto workspace_it = cublas_handle_stream_to_workspace().find(key);
  if (workspace_it == cublas_handle_stream_to_workspace().end()) {
    workspace_it = cublas_handle_stream_to_workspace().insert(workspace_it, {key, getNewWorkspace()});
  }
  TORCH_CUDABLAS_CHECK(cublasSetWorkspace(handle, workspace_it->second.get(), getChosenWorkspaceSize()));

  // 在 CUDA >= 11 且架构 >= Ampere 的情况下，cuBLAS 可以根据 allow_tf32 标志
  // 加速 FP32 数据类型的计算，设置数学模式为 CUBLAS_TF32_TENSOR_OP_MATH。
  if (!NoTF32Guard::should_disable_tf32() && at::globalContext().allowTF32CuBLAS()) {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  } else {
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  }
#else
  // 如果定义了 USE_ROCM

  // 根据全局上下文中的 deterministicAlgorithms() 方法返回值确定 hipblas 的原子操作模式。
  hipblasAtomicsMode_t hipblas_mode;
  if (at::globalContext().deterministicAlgorithms()) {
    hipblas_mode = HIPBLAS_ATOMICS_NOT_ALLOWED;
  } else {
    hipblas_mode = HIPBLAS_ATOMICS_ALLOWED;
  }
  TORCH_CUDABLAS_CHECK(hipblasSetAtomicsMode(handle, hipblas_mode));
#endif

// 返回处理器句柄
return handle;
}

// 获取当前的 cuBLAS LT 句柄
cublasLtHandle_t getCurrentCUDABlasLtHandle() {
#ifdef USE_ROCM
  // 如果定义了 USE_ROCM

  // 获取当前 CUDA 设备索引
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(c10::cuda::GetDevice(&device));

  // 线程本地的 PoolWindows 是延迟初始化的，以避免在 Windows 上导致 hang 的初始化问题。
  // 参见：https://github.com/pytorch/pytorch/pull/22405
  // 这些线程本地的唯一指针在线程终止时将被销毁，将其预留的句柄释放回池中。

  // 使用泄漏的单例模式管理池，遵循单例的标准实践：https://isocpp.org/wiki/faq/ctors#construct-on-first-use-v2
  static auto pool = std::shared_ptr<CuBlasLtPoolType>(
      new CuBlasLtPoolType(), [](CuBlasLtPoolType* p) {
        // 泄漏内存
      });
  thread_local std::unique_ptr<CuBlasLtPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  return handle;
#else
  // 如果未定义 USE_ROCM

  // 返回当前 cuBLAS 句柄的转换类型
  return reinterpret_cast<cublasLtHandle_t>(getCurrentCUDABlasHandle());
#endif
}

// 命名空间结束
} // namespace at::cuda
```