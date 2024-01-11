# `ChatRWKV\rwkv_pip_package\src\rwkv\cuda\rwkv5_op.cpp`

```
#include <torch/extension.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAGuard.h>

// 定义别名，分别代表BFloat16、Half和float类型
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

// 声明三个CUDA前向函数，分别处理BFloat16、Half和float类型的数据
void cuda_forward_bf16(int B, int T, int C, int H, float *state, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);
void cuda_forward_fp16(int B, int T, int C, int H, float *state, fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, fp16 *y);
void cuda_forward_fp32(int B, int T, int C, int H, float *state, fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, fp32 *y);

// 定义BFloat16类型的前向函数，接受Tensor参数，并调用对应的CUDA前向函数
void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    cuda_forward_bf16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}

// 定义Half类型的前向函数，接受Tensor参数，并调用对应的CUDA前向函数
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    cuda_forward_fp16(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), u.data_ptr<fp16>(), y.data_ptr<fp16>());
}

// 定义float类型的前向函数，接受Tensor参数，并调用对应的CUDA前向函数
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &state, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    cuda_forward_fp32(B, T, C, H, state.data_ptr<float>(), r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), u.data_ptr<fp32>(), y.data_ptr<fp32>());
}

// 绑定前向函数，使其可以在Python中调用
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_bf16", &forward_bf16, "rwkv5 forward_bf16");
    # 使用 pybind11 将 C++ 函数 forward_fp16 包装成 Python 可调用的函数
    m.def("forward_fp16", &forward_fp16, "rwkv5 forward_fp16");
    # 使用 pybind11 将 C++ 函数 forward_fp32 包装成 Python 可调用的函数
    m.def("forward_fp32", &forward_fp32, "rwkv5 forward_fp32");
# 定义一个名为rwkv5的Torch库
TORCH_LIBRARY(rwkv5, m) {
    # 定义名为forward_bf16的函数，并将其添加到Torch库中
    m.def("forward_bf16", forward_bf16);
    # 定义名为forward_fp16的函数，并将其添加到Torch库中
    m.def("forward_fp16", forward_fp16);
    # 定义名为forward_fp32的函数，并将其添加到Torch库中
    m.def("forward_fp32", forward_fp32);
}
```