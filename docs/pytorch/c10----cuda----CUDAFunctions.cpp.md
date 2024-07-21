# `.\pytorch\c10\cuda\CUDAFunctions.cpp`

```
// 包含 CUDA 相关头文件
#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Macros.h>

// 匿名命名空间，用于定义内部函数和变量，限制其作用域
namespace c10::cuda {

namespace {
// 获取 CUDA 驱动程序版本，如果失败返回 -1
int32_t driver_version() {
  int driver_version = -1;
  // 调用 CUDA API 获取驱动程序版本号
  C10_CUDA_IGNORE_ERROR(cudaDriverGetVersion(&driver_version));
  return driver_version;
}

// 获取 CUDA 设备数量的实现函数，如果没有驱动程序则根据 fail_if_no_driver 返回值处理
int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  // 调用 PyTorch 封装的获取设备数量的函数
  auto err = C10_CUDA_ERROR_HANDLED(c10::cuda::GetDeviceCount(&count));
  if (err == cudaSuccess) {
    // 如果成功获取设备数量，则返回 count
    return count;
  }
  // 清除错误状态，以避免误导其他部分代码
  cudaError_t last_err C10_UNUSED = cudaGetLastError();
  switch (err) {
    case cudaErrorNoDevice:
      // 没有 CUDA 设备时，设备数量置为 0
      count = 0;
      break;
    case cudaErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // 如果没有 CUDA 驱动程序且不要求失败，则设备数量置为 0
          count = 0;
          break;
        }
        // 抛出错误，要求安装 NVIDIA 驱动程序
        TORCH_CHECK(
            false,
            "Found no NVIDIA driver on your system. Please check that you "
            "have an NVIDIA GPU and installed a driver from "
            "http://www.nvidia.com/Download/index.aspx");
      } else {
        // 抛出错误，要求更新 CUDA 驱动程序版本
        TORCH_CHECK(
            false,
            "The NVIDIA driver on your system is too old (found version ",
            version,
            "). Please update your GPU driver by downloading and installing "
            "a new version from the URL: "
            "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
            "https://pytorch.org to install a PyTorch version that has been "
            "compiled with your version of the CUDA driver.");
      }
    } break;
    case cudaErrorInitializationError:
      // CUDA 驱动程序初始化失败时抛出错误
      TORCH_CHECK(
          false,
          "CUDA driver initialization failed, you might not "
          "have a CUDA gpu.");
      break;
    case cudaErrorUnknown:
      // 出现未知 CUDA 错误时抛出错误
      TORCH_CHECK(
          false,
          "CUDA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable CUDA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
      break;
#if C10_ASAN_ENABLED
    case cudaErrorMemoryAllocation:
      // 在 ASAN 模式下，由于 nvcc 与 ASAN 不兼容可能导致内存分配错误
      TORCH_CHECK(
          false,
          "Got 'out of memory' error while trying to initialize CUDA. "
          "CUDA with nvcc does not work well with ASAN and it's probably "
          "the reason. We will simply shut down CUDA support. If you "
          "would like to use GPUs, turn off ASAN.");
      break;
#endif // C10_ASAN_ENABLED
    default:
      # 在switch语句中，处理所有未被显式处理的情况
      TORCH_CHECK(
          false,
          # 使用TORCH_CHECK宏来断言条件是否为真，如果为假，抛出错误信息
          "Unexpected error from cudaGetDeviceCount(). Did you run "
          "some cuda functions before calling NumCudaDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          cudaGetErrorString(err));
  }
  # 返回存储CUDA设备数量的计数变量
  return count;
} // namespace
// 结束命名空间的定义

DeviceIndex device_count() noexcept {
  // 返回当前系统中 CUDA 设备的数量，仅初始化一次
  static int count = []() {
    try {
      // 调用实现函数获取设备数量，允许失败时不报错
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      // 断言确保设备数量不超过 DeviceIndex 的最大值
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many CUDA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // 捕获异常，记录警告信息并返回 0
      TORCH_WARN("CUDA initialization: ", ex.msg());
      return 0;
    }
  }();
  // 返回设备数量的静态计数值
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  // 每次调用都获取设备数量并强制要求至少有一个 CUDA GPU
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // 如果没有 CUDA GPU，则产生错误
  TORCH_CHECK(count, "No CUDA GPUs are available");
  // 断言确保设备数量不超过 DeviceIndex 的最大值
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<DeviceIndex>::max(),
      "Too many CUDA devices, DeviceIndex overflowed");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  // 获取当前设备的索引
  DeviceIndex cur_device = -1;
  // 使用 C10_CUDA_CHECK 确保获取设备的正确性
  C10_CUDA_CHECK(c10::cuda::GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device) {
  // 使用 C10_CUDA_CHECK 设置当前操作设备
  C10_CUDA_CHECK(c10::cuda::SetDevice(device));
}

void device_synchronize() {
  // 获取 GPU 追踪器的实例
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  // 如果存在 GPU 追踪器实例，则进行 GPU 设备同步追踪
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kCUDA);
  }
  // 使用 C10_CUDA_CHECK 同步 CUDA 设备
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

// 当执行 CUDA 同步操作时，必须由调用方调用此函数以引发正确的错误或警告
void warn_or_error_on_sync() {
  // 检查当前同步调试模式，如果为错误模式则抛出异常
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing CUDA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    // 如果为警告模式则记录警告信息
    TORCH_WARN("called a synchronizing CUDA operation");
  }
}

std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // 检查当前设备的主要上下文
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    // 如果当前设备有主要上下文，则返回该设备索引
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  // 遍历所有 CUDA 设备，寻找有主要上下文的设备
  for (const auto device_index : c10::irange(at::cuda::device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  // 如果没有找到符合条件的设备，则返回空值
  return c10::nullopt;
}

namespace _internal {
// 虚拟函数，当被调用时应该发生错误
bool dummyHasPrimaryContext(C10_UNUSED DeviceIndex device_index) {
  TORCH_CHECK(false, "Should never been called");
}
// 指向虚拟函数的函数指针，用于判断设备是否有主要上下文
bool (*hasPrimaryContext)(DeviceIndex) = dummyHasPrimaryContext;

// 私有 API，由 CUDAHooks.cpp 调用
C10_CUDA_API void setHasPrimaryContext(bool (*func)(DeviceIndex)) {
  // 设置是否有主要上下文的函数指针，如果为空则指向虚拟函数
  hasPrimaryContext = func ? func : dummyHasPrimaryContext;
}
} // namespace _internal
// 检查给定设备索引是否存在主要上下文
bool hasPrimaryContext(DeviceIndex device_index) {
  return _internal::hasPrimaryContext(device_index);
}

// CUDA 设备管理原始函数的包装器
cudaError_t GetDeviceCount(int* dev_count) {
  return cudaGetDeviceCount(dev_count);
}

// 这是 CUDA 12 的代码路径，引入了`cudaSetDevice`行为的重要变更。
// 不同于之前的 CUDA 版本在需要时分配上下文，CUDA 12.x 在调用`cudaSetDevice`时会
// 立即分配主要上下文。这可能导致分布式运行中的严重后果，并且可能污染设备内存。
// 为了避免不必要的上下文创建，引入了一个新函数`MaybeSetDevice`。该函数应在设备守卫析构函数
// 和 torch.cuda.device 上下文管理器的退出时调用。`MaybeSetDevice`函数的行为非常简单，
// 如果上下文已经存在，则调用`cudaSetDevice`；如果在目标设备上尚未分配上下文，则仅保存设备索引。
// 这样我们可以保持 PyTorch 对如下应用的向后兼容性：
//
// ```
// import torch
// x = torch.empty(1, device=“cuda:1”) # 此后 cuda:0 上没有 CUDA 上下文
// y = torch.empty(1, device=“cuda”) # CUDA 上下文将在 cuda:0 上创建
// ```
#if CUDA_VERSION >= 12000
thread_local DeviceIndex targetDeviceIndex = -1;

// 获取当前设备索引
cudaError_t GetDevice(DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return cudaSuccess;
  }
  int tmp_device = -1;
  auto err = cudaGetDevice(&tmp_device);
  if (err == cudaSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "cudaGetDevice 返回无效设备 ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

// 设置当前设备索引
cudaError_t SetDevice(DeviceIndex device) {
  TORCH_CHECK(device >= 0, "设备 ID 必须为正数！", device);
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  if (device == cur_device) {
    return cudaSuccess;
  }
  return cudaSetDevice(device);
}

// 如果存在主要上下文，则设置设备索引
cudaError_t MaybeSetDevice(DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return c10::cuda::SetDevice(device);
  }
  targetDeviceIndex = device;
  return cudaSuccess;
}

// 此函数始终在 to_device 上初始化 CUDA 上下文
DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_CUDA_CHECK(cudaSetDevice(to_device));
  return cur_device;
}
#endif
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  // 临时变量，存储当前设备索引
  int tmp_cur_device = -1;
  // 调用 CUDA API 获取当前设备索引，并将结果存入 tmp_cur_device
  C10_CUDA_CHECK(cudaGetDevice(&tmp_cur_device));
  // 检查获取的设备索引是否有效
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 &&
          tmp_cur_device <= std::numeric_limits<DeviceIndex>::max(),
      "cudaGetDevice returns invalid device ",
      tmp_cur_device);
  // 将 tmp_cur_device 转换为 DeviceIndex 类型
  auto cur_device = static_cast<DeviceIndex>(tmp_cur_device);
  // 如果目标设备索引与当前设备索引相同，则直接返回当前设备索引
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  // 如果目标设备索引是主要上下文中的设备，则设置当前设备为 to_device
  if (hasPrimaryContext(to_device)) {
    C10_CUDA_CHECK(cudaSetDevice(to_device));
  } else {
    // 否则，设置目标设备索引为 targetDeviceIndex
    targetDeviceIndex = to_device;
  }
  // 返回当前设备索引
  return cur_device;
}

void SetTargetDevice() {
  // 如果目标设备索引大于等于 0，则设置当前 CUDA 设备为 targetDeviceIndex
  if (targetDeviceIndex >= 0) {
    C10_CUDA_CHECK(c10::cuda::SetDevice(targetDeviceIndex));
  }
}
#else
cudaError_t GetDevice(DeviceIndex* device) {
  // 临时变量，存储当前设备索引
  int tmp_device = -1;
  // 调用 CUDA API 获取当前设备索引，并将结果存入 tmp_device
  auto err = cudaGetDevice(&tmp_device);
  // 如果成功获取设备索引，则进行进一步检查
  if (err == cudaSuccess) {
    // 检查获取的设备索引是否有效
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "cudaGetDevice returns invalid device ",
        tmp_device);
    // 将 tmp_device 转换为 DeviceIndex 类型，并赋值给 *device
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  // 返回 CUDA API 调用的结果
  return err;
}

cudaError_t SetDevice(DeviceIndex device) {
  // 检查设备索引是否大于等于 0
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  // 临时变量，存储当前设备索引
  int cur_device = -1;
  // 获取当前 CUDA 设备索引，并将结果存入 cur_device
  C10_CUDA_CHECK(cudaGetDevice(&cur_device));
  // 如果设备索引与当前设备索引相同，则返回成功
  if (device == cur_device) {
    return cudaSuccess;
  }
  // 否则，设置当前 CUDA 设备为指定的设备索引 device
  return cudaSetDevice(device);
}

cudaError_t MaybeSetDevice(DeviceIndex device) {
  // 调用 c10::cuda::SetDevice 设置当前 CUDA 设备为 device
  return c10::cuda::SetDevice(device);
}

DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  // 临时变量，存储当前设备索引
  DeviceIndex cur_device = -1;
  // 调用 c10::cuda::GetDevice 获取当前设备索引，并将结果存入 cur_device
  C10_CUDA_CHECK(c10::cuda::GetDevice(&cur_device));
  // 如果目标设备索引与当前设备索引相同，则返回当前设备索引
  if (to_device == cur_device) {
    return cur_device;
  }
  // 否则，设置当前 CUDA 设备为目标设备索引 to_device
  C10_CUDA_CHECK(cudaSetDevice(to_device));
  // 返回当前设备索引
  return cur_device;
}

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  // 调用 c10::cuda::ExchangeDevice 交换设备，并返回结果
  return c10::cuda::ExchangeDevice(to_device);
}

void SetTargetDevice() {
  // 在 CUDA 版本小于 12.x 时，不执行任何操作
  // no-op on CUDA version < 12.x
}
#endif

} // namespace c10::cuda
```