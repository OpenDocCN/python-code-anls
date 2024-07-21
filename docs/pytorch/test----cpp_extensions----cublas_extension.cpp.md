# `.\pytorch\test\cpp_extensions\cublas_extension.cpp`

```
#include <iostream>

#include <torch/extension.h>  // 导入 PyTorch C++ 扩展模块的头文件
#include <ATen/cuda/CUDAContext.h>  // 导入 ATen CUDA 上下文的头文件

#include <cublas_v2.h>  // 导入 cuBLAS 头文件

// 定义一个无操作的 cuBLAS 函数，接受一个张量 x 作为输入
torch::Tensor noop_cublas_function(torch::Tensor x) {
  cublasHandle_t handle;  // 声明 cuBLAS 句柄
  TORCH_CUDABLAS_CHECK(cublasCreate(&handle));  // 创建 cuBLAS 句柄并检查创建状态
  TORCH_CUDABLAS_CHECK(cublasDestroy(handle));  // 销毁 cuBLAS 句柄并检查销毁状态
  return x;  // 返回输入张量 x
}

// 定义 Python 绑定模块的入口函数，注册 noop_cublas_function 为一个名为 "noop_cublas_function" 的 cuBLAS 函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("noop_cublas_function", &noop_cublas_function, "a cublas function");  // 将 noop_cublas_function 绑定到 Python 模块中
}
```