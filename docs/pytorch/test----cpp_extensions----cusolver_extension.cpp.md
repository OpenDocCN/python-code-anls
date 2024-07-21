# `.\pytorch\test\cpp_extensions\cusolver_extension.cpp`

```
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cusolverDn.h>


// 定义一个 Torch 扩展模块的函数，用于调用 cusolver 库进行操作
torch::Tensor noop_cusolver_function(torch::Tensor x) {
  // 声明 cusolver 句柄
  cusolverDnHandle_t handle;
  // 创建 cusolver 句柄并检查是否成功
  TORCH_CUSOLVER_CHECK(cusolverDnCreate(&handle));
  // 销毁 cusolver 句柄
  TORCH_CUSOLVER_CHECK(cusolverDnDestroy(handle));
  // 返回输入的张量，此函数仅为示例，实际操作可以在 cusolver 上进行
  return x;
}


// 定义 Python 绑定模块的入口点
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 定义一个 Python 函数，将 noop_cusolver_function 绑定到 Python 中，说明其作用
    m.def("noop_cusolver_function", &noop_cusolver_function, "a cusolver function");
}
```