# `.\pytorch\aten\src\ATen\nnapi\nnapi_model_loader.h`

```py
#ifndef NNAPI_MODEL_LOADER_H_  // 如果未定义 NNAPI_MODEL_LOADER_H_ 宏，则进入条件编译
#define NNAPI_MODEL_LOADER_H_

#include <stdint.h>  // 包含标准整数类型的头文件

#include <ATen/nnapi/NeuralNetworks.h>  // 包含 ATen 的 NNAPI 相关头文件
#include <ATen/nnapi/nnapi_wrapper.h>  // 包含 ATen 的 NNAPI 封装头文件

namespace caffe2 {
namespace nnapi {

// 加载 NNAPI 模型的函数声明，返回整型
int load_nnapi_model(
    struct nnapi_wrapper* nnapi,  // 指向 nnapi_wrapper 结构体的指针
    ANeuralNetworksModel* model,  // 指向 ANeuralNetworksModel 结构体的指针，用于存储模型
    const void* serialized_model,  // 指向序列化模型数据的指针
    int64_t model_length,  // 模型数据的长度，64 位整数
    size_t num_buffers,  // 缓冲区的数量，无符号整数
    const void** buffer_ptrs,  // 指向缓冲区指针数组的指针
    int32_t* buffer_sizes,  // 缓冲区大小的数组指针，32 位整数
    size_t num_memories,  // 内存块的数量，无符号整数
    ANeuralNetworksMemory** memories,  // 指向 ANeuralNetworksMemory 指针数组的指针
    int32_t* memory_sizes,  // 内存块大小的数组指针，32 位整数
    int32_t* out_input_count,  // 输出：输入数量，32 位整数
    int32_t* out_output_count,  // 输出：输出数量，32 位整数
    size_t* out_bytes_consumed  // 输出：消耗的字节数，无符号整数
);

}} // namespace caffe2::nnapi

#endif // NNAPI_MODEL_LOADER_H_  // 结束条件编译，定义宏结束
```