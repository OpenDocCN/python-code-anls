# `ChatRWKV\rwkv_pip_package\src\rwkv\cuda\rwkv6_op.cpp`

```
#include <torch/extension.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAGuard.h>

// 定义别名，分别表示 BFloat16、Half 和 Float 类型
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

// 声明使用 BFloat16 类型进行前向传播的函数
void cuda_forward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);
// 声明使用 Half 类型进行前向传播的函数
void cuda_forward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, fp16 *y);
// 声明使用 Float 类型进行前向传播的函数
void cuda_forward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, fp32 *y);

// 使用 BFloat16 类型进行前向传播的函数实现
void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    // 获取当前设备的 CUDA Guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    // 调用使用 BFloat16 类型进行前向传播的函数
    cuda_forward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}

// 使用 Half 类型进行前向传播的函数实现
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    // 获取当前设备的 CUDA Guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    // 调用使用 Half 类型进行前向传播的函数
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), u.data_ptr<fp16>(), y.data_ptr<fp16>());
}

// 使用 Float 类型进行前向传播的函数实现
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    // 获取当前设备的 CUDA Guard
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    // 调用使用 Float 类型进行前向传播的函数
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}

// 绑定前向传播函数到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 绑定使用 BFloat16 类型进行前向传播的函数
    m.def("forward_bf16", &forward_bf16, "rwkv6 forward_bf16");
    # 定义 Python 模块中的函数 forward_fp16，映射到 C++ 函数 forward_fp16，提供描述信息 "rwkv6 forward_fp16"
    m.def("forward_fp16", &forward_fp16, "rwkv6 forward_fp16");
    # 定义 Python 模块中的函数 forward_fp32，映射到 C++ 函数 forward_fp32，提供描述信息 "rwkv6 forward_fp32"
    m.def("forward_fp32", &forward_fp32, "rwkv6 forward_fp32");
# 定义一个名为rwkv6的Torch库
TORCH_LIBRARY(rwkv6, m) {
    # 定义名为forward_bf16的函数，并将其添加到Torch库中
    m.def("forward_bf16", forward_bf16);
    # 定义名为forward_fp16的函数，并将其添加到Torch库中
    m.def("forward_fp16", forward_fp16);
    # 定义名为forward_fp32的函数，并将其添加到Torch库中
    m.def("forward_fp32", forward_fp32);
}
```