# `.\transformers\kernels\rwkv\wkv_op.cpp`

```py
// 包含 Torch C++ 扩展头文件
#include <torch/extension.h>
// 包含 ATen 库的头文件
#include "ATen/ATen.h"
// 使用别名定义 BFloat16 类型为 bf16
typedef at::BFloat16 bf16;

// 定义 CUDA 前向传播函数，接受 float 类型参数
void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
// 定义 CUDA 前向传播函数，接受 BFloat16 类型参数
void cuda_forward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y);
// 定义 CUDA 带状态前向传播函数，接受 float 类型参数
void cuda_forward_with_state(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *s);
// 定义 CUDA 带状态前向传播函数，接受 BFloat16 类型参数
void cuda_forward_with_state_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s);
// 定义 CUDA 反向传播函数，接受 float 类型参数
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *gy, float *gw, float *gu, float *gk, float *gv);
// 定义 CUDA 反向传播函数，接受 BFloat16 类型参数
void cuda_backward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv);

// 定义前向传播函数，接受 Torch 张量参数
void forward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    // 获取张量维度信息
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    // 调用相应的 CUDA 前向传播函数
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}
// 定义 BFloat16 类型参数的前向传播函数
void forward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    // 获取张量维度信息
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    // 调用相应的 CUDA 前向传播函数，参数类型为 BFloat16
    cuda_forward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
}
// 定义带状态前向传播函数
void forward_with_state(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    // 获取张量维度信息
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    // 调用相应的 CUDA 带状态前向传播函数
    cuda_forward_with_state(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), s.data_ptr<float>());
}
// 定义带状态前向传播函数，参数类型为 BFloat16
void forward_with_state_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    // 获取张量维度信息
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    // 调用相应的 CUDA 带状态前向传播函数，参数类型为 BFloat16
    cuda_forward_with_state_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), s.data_ptr<float>());
}
// 定义反向传播函数
void backward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    // 获取张量维度信息
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    // 调用相应的 CUDA 反向传播函数
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}
// 定义反向传播函数，参数类型为 BFloat16
void backward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    // 获取张量维度信息
    const int B = k.size(0);
    // 定义常量 T，表示 k 张量的第二维大小
    const int T = k.size(1);
    // 定义常量 C，表示 k 张量的第三维大小
    const int C = k.size(2);
    // 调用 CUDA 后向传播函数，传入参数 B、T、C、以及各张量的数据指针
    cuda_backward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(),
        gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}

# 定义一个 Python 模块，名为 TORCH_EXTENSION_NAME，其中包含了下列函数的绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    # 定义一个名为 "forward" 的 Python 函数，其底层实现为 C++ 中的 forward 函数，描述为 "wkv forward"
    m.def("forward", &forward, "wkv forward");
    # 定义一个名为 "forward_bf16" 的 Python 函数，其底层实现为 C++ 中的 forward_bf16 函数，描述为 "wkv forward bf16"
    m.def("forward_bf16", &forward_bf16, "wkv forward bf16");
    # 定义一个名为 "forward_with_state" 的 Python 函数，其底层实现为 C++ 中的 forward_with_state 函数，描述为 "wkv forward with state"
    m.def("forward_with_state", &forward_with_state, "wkv forward with state");
    # 定义一个名为 "forward_with_state_bf16" 的 Python 函数，其底层实现为 C++ 中的 forward_with_state_bf16 函数，描述为 "wkv forward with state bf16"
    m.def("forward_with_state_bf16", &forward_with_state_bf16, "wkv forward with state bf16");
    # 定义一个名为 "backward" 的 Python 函数，其底层实现为 C++ 中的 backward 函数，描述为 "wkv backward"
    m.def("backward", &backward, "wkv backward");
    # 定义一个名为 "backward_bf16" 的 Python 函数，其底层实现为 C++ 中的 backward_bf16 函数，描述为 "wkv backward bf16"
    m.def("backward_bf16", &backward_bf16, "wkv backward bf16");
}

# 注册一个名为 wkv 的 Torch 库，其中包含了下列函数的定义
TORCH_LIBRARY(wkv, m) {
    # 将名为 forward 的函数注册到 wkv 库中
    m.def("forward", forward);
    # 将名为 forward_bf16 的函数注册到 wkv 库中
    m.def("forward_bf16", forward_bf16);
    # 将名为 forward_with_state 的函数注册到 wkv 库中
    m.def("forward_with_state", forward_with_state);
    # 将名为 forward_with_state_bf16 的函数注册到 wkv 库中
    m.def("forward_with_state_bf16", forward_with_state_bf16);
    # 将名为 backward 的函数注册到 wkv 库中
    m.def("backward", backward);
    # 将名为 backward_bf16 的函数注册到 wkv 库中
    m.def("backward_bf16", backward_bf16);
}
```