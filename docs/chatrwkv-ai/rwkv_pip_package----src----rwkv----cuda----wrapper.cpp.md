# `ChatRWKV\rwkv_pip_package\src\rwkv\cuda\wrapper.cpp`

```
#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

typedef at::Half fp16;  // 定义类型别名，将 at::Half 重命名为 fp16

template <typename F>
void cuda_wkv_forward(int B, int T, int C,
                      float *w, float *u, F *k, F *v, F *y,
                      float *aa, float *bb, float *pp);  // 声明一个模板函数，用于在 CUDA 上执行前向传播操作

template <typename F>
void cuda_mm8_seq(int B, int N, int M,
                  F *x, int x_stride,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  F *y, int y_stride);  // 声明一个模板函数，用于在 CUDA 上执行矩阵乘法操作

template <typename F>
void cuda_mm8_one(int N, int M,
                  F *x,
                  uint8_t *w, int w_stride,
                  F *mx, F *rx,
                  F *my, F *ry,
                  float *y);  // 声明一个模板函数，用于在 CUDA 上执行单个矩阵乘法操作

void wkv_forward(int64_t B, int64_t T, int64_t C,
                 torch::Tensor &w, torch::Tensor &u,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                 torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));  // 创建一个 CUDA 设备卫士，确保在正确的 CUDA 设备上执行操作
    switch (k.scalar_type()) {  // 根据输入张量 k 的数据类型进行分支选择
    case c10::ScalarType::Half:  // 如果数据类型为半精度浮点数
        cuda_wkv_forward(B, T, C,
                         w.data_ptr<float>(), u.data_ptr<float>(),  // 获取张量的数据指针
                         k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(),  // 获取张量的数据指针
                         aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());  // 获取张量的数据指针
        break;
    case c10::ScalarType::Float:  // 如果数据类型为单精度浮点数
        cuda_wkv_forward(B, T, C,
                         w.data_ptr<float>(), u.data_ptr<float>(),  // 获取张量的数据指针
                         k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(),  // 获取张量的数据指针
                         aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());  // 获取张量的数据指针
        break;
    default:  // 如果数据类型不是半精度浮点数或单精度浮点数
        assert(false && "Only FP16 and FP32 are currently supported");  // 抛出断言错误，表示当前仅支持半精度浮点数和单精度浮点数
    }
}
// 定义一个名为 mm8_seq 的函数，接受多个参数，包括输入和输出的张量
void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    // 断言输入张量 x 的第一维的步长为 1
    assert(x.stride(1) == 1);
    // 断言输入张量 w 的第一维的步长为 1
    assert(w.stride(1) == 1);
    // 断言输入张量 mx 和 rx 的第零维的步长为 1
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    // 断言输入张量 my 和 ry 的第零维的步长为 1
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    // 断言输出张量 y 的第一维的步长为 1
    assert(y.stride(1) == 1);
    // 创建一个 CUDA 设备的可选守卫，确保在当前设备上执行操作
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    // 根据输入张量 x 的数据类型进行不同的操作
    switch (x.scalar_type()) {
    // 如果数据类型为半精度浮点数
    case c10::ScalarType::Half:
        // 调用 CUDA 函数 cuda_mm8_seq 进行矩阵乘法运算
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<fp16>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<fp16>(), y.stride(0));
        break;
    // 如果数据类型为单精度浮点数
    case c10::ScalarType::Float:
        // 调用 CUDA 函数 cuda_mm8_seq 进行矩阵乘法运算
        cuda_mm8_seq(
            B, N, M,
            x.data_ptr<float>(), x.stride(0),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>(), y.stride(0));
        break;
    // 如果数据类型不是半精度浮点数或单精度浮点数
    default:
        // 断言并输出错误信息
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}
// 定义一个名为 mm8_one 的函数，接受多个参数，包括输入和输出的张量
void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y) {
    // 断言输入张量 x 的第零维的步长为 1
    assert(x.stride(0) == 1);
    // 断言输入张量 w 的第一维的步长为 1
    assert(w.stride(1) == 1);
    // 断言输入张量 mx 和 rx 的第零维的步长为 1
    assert(mx.stride(0) == 1 && rx.stride(0) == 1);
    // 断言输入张量 my 和 ry 的第零维的步长为 1
    assert(my.stride(0) == 1 && ry.stride(0) == 1);
    // 断言输出张量 y 的第零维的步长为 1
    assert(y.stride(0) == 1);
    // 创建一个 CUDA 设备的可选守卫，确保在当前设备上执行操作
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    // 根据输入张量 x 的数据类型进行不同的操作
    switch (x.scalar_type()) {
    # 根据不同的数据类型进行不同的计算操作
    case c10::ScalarType::Half:
        # 调用 CUDA 函数进行半精度浮点数的矩阵乘法运算
        cuda_mm8_one(
            N, M,
            x.data_ptr<fp16>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<fp16>(), rx.data_ptr<fp16>(),
            my.data_ptr<fp16>(), ry.data_ptr<fp16>(),
            y.data_ptr<float>());
        # 结束当前 case 的执行
        break;
    case c10::ScalarType::Float:
        # 调用 CUDA 函数进行单精度浮点数的矩阵乘法运算
        cuda_mm8_one(
            N, M,
            x.data_ptr<float>(),
            w.data_ptr<uint8_t>(), w.stride(0),
            mx.data_ptr<float>(), rx.data_ptr<float>(),
            my.data_ptr<float>(), ry.data_ptr<float>(),
            y.data_ptr<float>());
        # 结束当前 case 的执行
        break;
    default:
        # 如果不是半精度或单精度浮点数，则断言报错
        assert(false && "Only FP16 and FP32 are currently supported");
    }
}

using torch::Tensor;

#ifndef DISABLE_CUBLAS_GEMM
// 声明使用 CUBLAS 进行 FP16 矩阵乘法的函数
void gemm_fp16_cublas(Tensor a, Tensor b, Tensor c);
#endif

// 定义 Python 绑定模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 绑定 wkv_forward 函数
    m.def("wkv_forward", &wkv_forward, "wkv forward");
    // 绑定 mm8_seq 函数
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    // 绑定 mm8_one 函数
    m.def("mm8_one", &mm8_one, "mm8 one");
#ifndef DISABLE_CUBLAS_GEMM
    // 如果未禁用 CUBLAS GEMM，则绑定 gemm_fp16_cublas 函数
    m.def("gemm_fp16_cublas", &gemm_fp16_cublas, "gemv fp16 cublas");
#endif
}

// 定义 Torch 库 rwkv
TORCH_LIBRARY(rwkv, m) {
    // 绑定 wkv_forward 函数
    m.def("wkv_forward", wkv_forward);
    // 绑定 mm8_seq 函数
    m.def("mm8_seq", mm8_seq);
    // 绑定 mm8_one 函数
    m.def("mm8_one", mm8_one);
#ifndef DISABLE_CUBLAS_GEMM
    // 如果未禁用 CUBLAS GEMM，则绑定 gemm_fp16_cublas 函数
    m.def("gemm_fp16_cublas", gemm_fp16_cublas);
#endif
}
```