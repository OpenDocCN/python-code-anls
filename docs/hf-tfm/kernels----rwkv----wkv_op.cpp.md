# `.\kernels\rwkv\wkv_op.cpp`

```
# 包含 Torch 扩展和 ATen 库的头文件
#include <torch/extension.h>
#include "ATen/ATen.h"

# 定义一个别名 bf16 代表 ATen 库中的 BFloat16 类型
typedef at::BFloat16 bf16;

# 声明 CUDA 前向传播函数，接受 float 类型数据
void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);

# 声明 CUDA 前向传播函数，接受 BFloat16 类型数据
void cuda_forward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y);

# 声明带状态的 CUDA 前向传播函数，接受 float 类型数据
void cuda_forward_with_state(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *s);

# 声明带状态的 CUDA 前向传播函数，接受 BFloat16 类型数据
void cuda_forward_with_state_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, float *s);

# 声明 CUDA 反向传播函数，接受 float 类型数据
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *gy, float *gw, float *gu, float *gk, float *gv);

# 声明 CUDA 反向传播函数，接受 BFloat16 类型数据
void cuda_backward_bf16(int B, int T, int C, float *w, bf16 *u, bf16 *k, bf16 *v, bf16 *y, bf16 *gy, bf16 *gw, bf16 *gu, bf16 *gk, bf16 *gv);

# 定义无状态的前向传播函数，接受 Torch 张量参数
void forward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    # 获取张量的批量大小 B，时间步长 T，通道数 C
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    # 调用 CUDA 前向传播函数，传递 float 类型数据指针
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}

# 定义无状态的前向传播函数，接受 BFloat16 类型的 Torch 张量参数
void forward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    # 获取张量的批量大小 B，时间步长 T，通道数 C
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    # 调用 CUDA 前向传播函数，传递 BFloat16 类型数据指针
    cuda_forward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>());
}

# 定义带状态的前向传播函数，接受 Torch 张量参数及状态张量
void forward_with_state(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    # 获取张量的批量大小 B，时间步长 T，通道数 C
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    # 调用带状态的 CUDA 前向传播函数，传递 float 类型数据指针及状态张量数据指针
    cuda_forward_with_state(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), s.data_ptr<float>());
}

# 定义带状态的前向传播函数，接受 BFloat16 类型的 Torch 张量参数及状态张量
void forward_with_state_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &s) {
    # 获取张量的批量大小 B，时间步长 T，通道数 C
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    # 调用带状态的 CUDA 前向传播函数，传递 BFloat16 类型数据指针及状态张量数据指针
    cuda_forward_with_state_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), s.data_ptr<float>());
}

# 定义反向传播函数，接受 Torch 张量参数
void backward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    # 获取张量的批量大小 B，时间步长 T，通道数 C
    const int B = k.size(0);
    const int T = k.size(1);
    const int C = k.size(2);
    # 调用 CUDA 反向传播函数，传递 float 类型数据指针
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}

# 定义反向传播函数，接受 BFloat16 类型的 Torch 张量参数
void backward_bf16(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    # 获取张量的批量大小 B
    const int B = k.size(0);
    # 以下代码与 backward 函数相同，但接受 BFloat16 类型数据
    cuda_backward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(), gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}
    # 定义常量 T，表示 k 张量的第一个维度大小
    const int T = k.size(1);
    # 定义常量 C，表示 k 张量的第二个维度大小
    const int C = k.size(2);
    # 调用 CUDA 后向传播函数 cuda_backward_bf16，传递以下参数：
    # - B: 未明确提到，可能是某种批量大小或其它参数
    # - T: k 张量的第一个维度大小
    # - C: k 张量的第二个维度大小
    # - w.data_ptr<float>()：权重张量 w 的 float 类型数据指针
    # - u.data_ptr<bf16>()：输入张量 u 的 bf16 类型数据指针
    # - k.data_ptr<bf16>()：内核张量 k 的 bf16 类型数据指针
    # - v.data_ptr<bf16>()：中间变量 v 的 bf16 类型数据指针
    # - y.data_ptr<bf16>()：输出张量 y 的 bf16 类型数据指针
    # - gy.data_ptr<bf16>()：输出梯度 gy 的 bf16 类型数据指针
    # - gw.data_ptr<bf16>()：权重梯度 gw 的 bf16 类型数据指针
    # - gu.data_ptr<bf16>()：输入梯度 gu 的 bf16 类型数据指针
    # - gk.data_ptr<bf16>()：内核梯度 gk 的 bf16 类型数据指针
    # - gv.data_ptr<bf16>()：中间变量梯度 gv 的 bf16 类型数据指针
    cuda_backward_bf16(B, T, C, w.data_ptr<float>(), u.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), y.data_ptr<bf16>(),
        gy.data_ptr<bf16>(), gw.data_ptr<bf16>(), gu.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 定义 Python 绑定模块的函数 "forward"，与 C++ 函数 &forward 绑定，描述为 "wkv forward"
    m.def("forward", &forward, "wkv forward");
    // 定义 Python 绑定模块的函数 "forward_bf16"，与 C++ 函数 &forward_bf16 绑定，描述为 "wkv forward bf16"
    m.def("forward_bf16", &forward_bf16, "wkv forward bf16");
    // 定义 Python 绑定模块的函数 "forward_with_state"，与 C++ 函数 &forward_with_state 绑定，描述为 "wkv forward with state"
    m.def("forward_with_state", &forward_with_state, "wkv forward with state");
    // 定义 Python 绑定模块的函数 "forward_with_state_bf16"，与 C++ 函数 &forward_with_state_bf16 绑定，描述为 "wkv forward with state bf16"
    m.def("forward_with_state_bf16", &forward_with_state_bf16, "wkv forward with state bf16");
    // 定义 Python 绑定模块的函数 "backward"，与 C++ 函数 &backward 绑定，描述为 "wkv backward"
    m.def("backward", &backward, "wkv backward");
    // 定义 Python 绑定模块的函数 "backward_bf16"，与 C++ 函数 &backward_bf16 绑定，描述为 "wkv backward bf16"
    m.def("backward_bf16", &backward_bf16, "wkv backward bf16");
}

TORCH_LIBRARY(wkv, m) {
    // 在 Torch 的 wkv 库中注册函数 "forward"，与 C++ 函数 forward 绑定
    m.def("forward", forward);
    // 在 Torch 的 wkv 库中注册函数 "forward_bf16"，与 C++ 函数 forward_bf16 绑定
    m.def("forward_bf16", forward_bf16);
    // 在 Torch 的 wkv 库中注册函数 "forward_with_state"，与 C++ 函数 forward_with_state 绑定
    m.def("forward_with_state", forward_with_state);
    // 在 Torch 的 wkv 库中注册函数 "forward_with_state_bf16"，与 C++ 函数 forward_with_state_bf16 绑定
    m.def("forward_with_state_bf16", forward_with_state_bf16);
    // 在 Torch 的 wkv 库中注册函数 "backward"，与 C++ 函数 backward 绑定
    m.def("backward", backward);
    // 在 Torch 的 wkv 库中注册函数 "backward_bf16"，与 C++ 函数 backward_bf16 绑定
    m.def("backward_bf16", backward_bf16);
}
```