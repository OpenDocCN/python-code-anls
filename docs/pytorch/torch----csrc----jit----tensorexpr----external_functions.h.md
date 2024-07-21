# `.\pytorch\torch\csrc\jit\tensorexpr\external_functions.h`

```
#pragma once

#include <ATen/Config.h>   // 引入 ATen 库的配置信息
#include <ATen/Functions.h>   // 引入 ATen 库的函数
#include <c10/macros/Macros.h>   // 引入 c10 库的宏定义
#include <torch/csrc/Export.h>   // 引入 torch 库的导出相关功能头文件
#include <cstdint>   // 引入标准整数类型
#include <vector>   // 引入向量（动态数组）相关功能

#define FOR_ALL_EXTERNAL_FUNCTIONS(_)   \   // 定义宏，用于列出所有外部函数
  _(nnc_aten_adaptive_avg_pool2d)       \   // 外部函数：自适应平均池化2D
  _(nnc_aten_addmm)                     \   // 外部函数：矩阵相加乘
  _(nnc_aten_conv2d)                    \   // 外部函数：2D 卷积
  _(nnc_aten_conv1d)                    \   // 外部函数：1D 卷积
  _(nnc_aten_conv1d_out)                \   // 外部函数：1D 卷积输出
  _(nnc_aten_dequantize)                \   // 外部函数：去量化
  _(nnc_aten_dequantize_out)            \   // 外部函数：去量化输出
  _(nnc_aten_embedding)                 \   // 外部函数：嵌入
  _(nnc_aten_matmul)                    \   // 外部函数：矩阵乘法
  _(nnc_aten_mv)                        \   // 外部函数：矩阵向量乘法
  _(nnc_aten_mm)                        \   // 外部函数：矩阵乘法
  _(nnc_aten_mean)                      \   // 外部函数：均值
  _(nnc_aten_max_red)                   \   // 外部函数：最大值约减
  _(nnc_aten_max_red_out)               \   // 外部函数：最大值约减输出
  _(nnc_aten_quantized_conv1d)          \   // 外部函数：量化1D卷积
  _(nnc_aten_quantized_conv1d_out)      \   // 外部函数：量化1D卷积输出
  _(nnc_aten_quantized_conv2d)          \   // 外部函数：量化2D卷积
  _(nnc_aten_quantized_conv2d_out)      \   // 外部函数：量化2D卷积输出
  _(nnc_aten_quantized_conv2d_relu)     \   // 外部函数：带ReLU的量化2D卷积
  _(nnc_aten_quantized_conv2d_relu_out) \   // 外部函数：带ReLU的量化2D卷积输出
  _(nnc_aten_quantized_linear)          \   // 外部函数：量化线性层
  _(nnc_aten_quantized_linear_out)      \   // 外部函数：量化线性层输出
  _(nnc_aten_quantized_linear_relu)     \   // 外部函数：带ReLU的量化线性层
  _(nnc_aten_quantized_add)             \   // 外部函数：量化加法
  _(nnc_aten_quantized_cat)             \   // 外部函数：量化拼接
  _(nnc_aten_quantized_mul)             \   // 外部函数：量化乘法
  _(nnc_aten_quantized_mul_out)         \   // 外部函数：量化乘法输出
  _(nnc_aten_quantized_mul_scalar)      \   // 外部函数：标量乘法
  _(nnc_aten_quantized_mul_scalar_out)  \   // 外部函数：标量乘法输出
  _(nnc_aten_quantized_relu)            \   // 外部函数：量化ReLU
  _(nnc_aten_quantized_sigmoid)         \   // 外部函数：量化Sigmoid
  _(nnc_aten_quantized_sigmoid_out)     \   // 外部函数：量化Sigmoid输出
  _(nnc_aten_quantize_per_tensor)       \   // 外部函数：按张量量化
  _(nnc_aten_quantize_per_tensor_out)   \   // 外部函数：按张量量化输出
  _(nnc_aten_triangular_solve)          \   // 外部函数：三角解
  _(nnc_aten_upsample_nearest2d)        \   // 外部函数：最近邻插值2D上采样
  _(nnc_aten_upsample_nearest2d_out)    \   // 外部函数：最近邻插值2D上采样输出
  _(nnc_prepacked_conv2d_clamp_run)     \   // 外部函数：预打包的带clamp运行的2D卷积

#define DECLARE_EXTERNAL_FUNCTION(NAME) \   // 定义宏，用于声明外部函数的接口
  TORCH_API void NAME(                  \   // 声明外部函数的名称和参数列表
      int64_t bufs_num,                 \   // 参数：缓冲区数量
      void** buf_data,                  \   // 参数：缓冲区数据指针数组
      int64_t* buf_ranks,               \   // 参数：缓冲区秩（维度数量）数组
      int64_t* buf_dims,                \   // 参数：缓冲区维度大小数组
      int64_t* buf_strides,             \   // 参数：缓冲区步长数组
      int8_t* buf_dtypes,               \   // 参数：缓冲区数据类型数组
      int64_t args_num,                 \   // 参数：额外参数数量
      int64_t* extra_args);             \   // 参数：额外参数数组

namespace torch {   // 命名空间：torch
namespace jit {   // 命名空间：jit
namespace tensorexpr {   // 命名空间：tensorexpr
struct QIData final {   // 结构体：QIData
  double scale;   // 成员变量：缩放因子
  int64_t zero;   // 成员变量：零点
  c10::ScalarType scalarType;   // 成员变量：标量类型
};
std::vector<at::Tensor> constructTensors(   // 函数声明：构建张量
    int64_t bufs_num,   // 参数：缓冲区数量
    void** buf_data,    // 参数：缓冲区数据指针数组
    int64_t* buf_ranks,   // 参数：缓冲区秩（维度数量）数组
    int64_t* buf_dims,    // 参数：缓冲区维度大小数组
    int64_t* buf_strides,   // 参数：缓冲区步长数组
    int8_t* buf_dtypes,   // 参数：缓冲区数据类型数组
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg =   // 参数：QIData 可选参数
        c10::nullopt);   // 默认值：空

std::vector<at::Tensor> constructTensors2(   // 函数声明：构建张量2
    int64_t bufs_in_num,   // 参数：输入缓冲区数量
    void** buf_data,    // 参数：缓冲区数据指针数组
    int64_t* buf_ranks,   // 参数：缓冲区秩（维度数量）数组
    int64_t* buf_dims,    // 参数：缓冲区维度大小数组
    int64_t* buf_strides,   // 参数：缓冲区步长数组
    int8_t* buf_dtypes,   // 参数：缓冲区数据类型数组
    std::optional<std::vector<std::pair<size_t, QIData>>> qdataArg =   // 参数：QIData 可选参数
        c10::nullopt,   // 默认值：空
    size_t bufs_out_num = 0);   // 参数：输出缓冲区数量，默认值为0

#ifdef C10_MOBILE   // 如果定义了 C10_MOBILE 宏，则编译以下内容
#ifdef C10_MOBILE
```  
在 C10_MOBILE 宏定义被设置时执行以下代码块


} // extern "C"
#endif
```  
结束 extern "C" 的作用域，并结束 C10_MOBILE 宏定义的条件编译块
```