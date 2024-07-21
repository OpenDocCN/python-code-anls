# `.\pytorch\torch\csrc\inductor\aoti_torch\shim_cuda.cpp`

```py
// 包含 Torch 的头文件 shim.h 和 utils.h
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

// 包含 CUDA 相关的头文件
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// 创建 CUDA Guard 对象，用于管理指定设备上的 CUDA 操作
AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard // 返回新的引用
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 在指定设备上创建一个 CUDA Guard 对象
    at::cuda::CUDAGuard* guard = new at::cuda::CUDAGuard(device_index);
    // 将 guard 转换为 CUDAGuardHandle 类型并返回
    *ret_guard = reinterpret_cast<CUDAGuardHandle>(guard);
  });
}

// 删除指定的 CUDA Guard 对象
AOTITorchError aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAGuard*>(guard); });
}

// 设置 CUDA Guard 对象的设备索引
AOTITorchError aoti_torch_cuda_guard_set_index(
    CUDAGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 设置 guard 指向的 CUDA Guard 对象的设备索引
    reinterpret_cast<at::cuda::CUDAGuard*>(guard)->set_index(device_index);
  });
}

// 创建 CUDA Stream Guard 对象，用于管理指定设备上的 CUDA 流
AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 根据外部提供的 CUDA 流创建 CUDA Stream Guard 对象
    at::cuda::CUDAStreamGuard* guard =
        new at::cuda::CUDAStreamGuard(at::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(stream), device_index));
    // 将 guard 转换为 CUDAStreamGuardHandle 类型并返回
    *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
  });
}

// 删除指定的 CUDA Stream Guard 对象
AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAStreamGuard*>(guard); });
}

// 获取当前指定设备上的 CUDA 流
AOTITorch_EXPORT AOTITorchError
aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // 获取当前指定设备上的 CUDA 流并返回
    *(cudaStream_t*)(ret_stream) = at::cuda::getCurrentCUDAStream(device_index);
  });
}


这段代码是用于 Torch 和 CUDA 相关操作的 C++ 接口实现。
```