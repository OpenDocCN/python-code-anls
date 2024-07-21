# `.\pytorch\torch\csrc\jit\tensorexpr\external_functions_registry.h`

```py
// 预处理指令，用于确保头文件只被包含一次
#pragma once

// 引入Torch库的导出头文件
#include <torch/csrc/Export.h>

// 引入C标准整数类型头文件
#include <cstdint>

// 引入C++标准字符串头文件
#include <string>

// 引入C++标准无序映射头文件
#include <unordered_map>

// Torch命名空间
namespace torch {

// JIT子命名空间
namespace jit {

// TensorExpr子命名空间
namespace tensorexpr {

// 定义外部函数指针类型NNCExternalFunction
// 这些函数可以从NNC调用，必须符合此指定的签名
using NNCExternalFunction = void (*)(
    int64_t bufs_num,        // 缓冲区数量
    void** buf_data,         // 缓冲区数据数组的指针
    int64_t* buf_ranks,      // 缓冲区秩数组的指针
    int64_t* buf_dims,       // 缓冲区维度数组的指针
    int64_t* buf_strides,    // 缓冲区步长数组的指针
    int8_t* buf_dtypes,      // 缓冲区数据类型数组的指针
    int64_t args_num,        // 额外参数数量
    int64_t* extra_args);    // 额外参数数组的指针

// 返回一个全局的映射表，将函数名映射到对应的NNCExternalFunction函数指针
TORCH_API std::unordered_map<std::string, NNCExternalFunction>& getNNCFunctionRegistry();

// 用于在NNC中注册新的外部函数，需要创建此结构体的实例
struct RegisterNNCExternalFunction {
  RegisterNNCExternalFunction(const std::string& name, NNCExternalFunction fn) {
    // 将函数名和对应的函数指针注册到NNC的函数映射表中
    getNNCFunctionRegistry()[name] = fn;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
```