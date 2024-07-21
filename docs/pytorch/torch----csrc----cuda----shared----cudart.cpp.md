# `.\pytorch\torch\csrc\cuda\shared\cudart.cpp`

```
// 包含 CUDA 相关的头文件
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/utils/pybind.h>

// 如果不是使用 ROCm 平台，则包含 CUDA Profiler 的头文件
#if !defined(USE_ROCM)
#include <cuda_profiler_api.h>
// 否则，包含 ROCm 的头文件
#else
#include <hip/hip_runtime_api.h>
#endif

// 包含 C10 库中的 CUDA 异常处理和 CUDA 设备管理相关头文件
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

// 定义了 torch::cuda::shared 命名空间
namespace torch::cuda::shared {

#ifdef USE_ROCM
// ROCm 平台下的命名空间，定义了一个返回 hipSuccess 的函数
namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace
#endif

// 初始化 CUDArt 绑定的函数，接收一个 Python 模块对象作为参数
void initCudartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // 在 Python 模块中定义一个名为 _cudart 的子模块，描述为 "libcudart.so bindings"
  auto cudart = m.def_submodule("_cudart", "libcudart.so bindings");

  // 如果不是使用 ROCm 平台，并且 CUDA 版本小于 12，定义一个枚举类型 cudaOutputMode_t
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  py::enum_<cudaOutputMode_t>(
      cudart,
      "cuda"
      "OutputMode")
      .value("KeyValuePair", cudaKeyValuePair)  // 枚举值 KeyValuePair 对应 cudaKeyValuePair
      .value("CSV", cudaCSV);  // 枚举值 CSV 对应 cudaCSV
#endif

  // 定义一个枚举类型 cudaError_t，描述 CUDA 错误，其中定义了一个枚举值 success 对应 cudaSuccess
  py::enum_<cudaError_t>(
      cudart,
      "cuda"
      "Error")
      .value("success", cudaSuccess);

  // 定义一个函数 cudart.cudaGetErrorString，对应 CUDA 的函数 cudaGetErrorString
  cudart.def(
      "cuda"
      "GetErrorString",
      cudaGetErrorString);

  // 定义一个函数 cudart.cudaProfilerStart，条件编译时根据使用平台选择对应的函数：CUDA 下使用 cudaProfilerStart，ROCm 下使用 hipReturnSuccess
  cudart.def(
      "cuda"
      "ProfilerStart",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStart
#endif
  );

  // 定义一个函数 cudart.cudaProfilerStop，条件编译时根据使用平台选择对应的函数：CUDA 下使用 cudaProfilerStop，ROCm 下使用 hipReturnSuccess
  cudart.def(
      "cuda"
      "ProfilerStop",
#ifdef USE_ROCM
      hipReturnSuccess
#else
      cudaProfilerStop
#endif
  );

  // 定义一个函数 cudart.cudaHostRegister，注册主机内存地址并返回 CUDA 错误码，使用 C10_CUDA_ERROR_HANDLED 宏来处理异常
  cudart.def(
      "cuda"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(
            cudaHostRegister((void*)ptr, size, flags));
      });

  // 定义一个函数 cudart.cudaHostUnregister，取消注册主机内存地址并返回 CUDA 错误码，使用 C10_CUDA_ERROR_HANDLED 宏来处理异常
  cudart.def(
      "cuda"
      "HostUnregister",
      [](uintptr_t ptr) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(cudaHostUnregister((void*)ptr));
      });

  // 定义一个函数 cudart.cudaStreamCreate，创建 CUDA 流并返回 CUDA 错误码，使用 C10_CUDA_ERROR_HANDLED 宏来处理异常
  cudart.def(
      "cuda"
      "StreamCreate",
      [](uintptr_t ptr) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(cudaStreamCreate((cudaStream_t*)ptr));
      });

  // 定义一个函数 cudart.cudaStreamDestroy，销毁 CUDA 流并返回 CUDA 错误码，使用 C10_CUDA_ERROR_HANDLED 宏来处理异常
  cudart.def(
      "cuda"
      "StreamDestroy",
      [](uintptr_t ptr) -> cudaError_t {
        return C10_CUDA_ERROR_HANDLED(cudaStreamDestroy((cudaStream_t)ptr));
      });

  // 如果不是使用 ROCm 平台，并且 CUDA 版本小于 12，定义一个函数 cudart.cudaProfilerInitialize，用于 CUDA Profiler 初始化
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION < 12000
  cudart.def(
      "cuda"
      "ProfilerInitialize",
      cudaProfilerInitialize);
#endif

  // 定义一个函数 cudart.cudaMemGetInfo，获取 CUDA 设备内存信息并返回一个包含空闲和总内存大小的 std::pair 对象
  cudart.def(
      "cuda"
      "MemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        // 使用 CUDAGuard 对象进行 CUDA 设备的上下文管理
        c10::cuda::CUDAGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        // 调用 CUDA API 获取设备内存信息，并通过 C10_CUDA_CHECK 宏处理异常
        C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::cuda::shared
```