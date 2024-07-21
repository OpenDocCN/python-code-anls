# `.\pytorch\aten\src\ATen\cuda\nvrtc_stub\ATenNVRTC.cpp`

```py
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>  // 包含 ATenNVRTC 头文件，提供了与 CUDA 相关的 NVRTC 接口
#include <iostream>  // 包含标准输入输出流的头文件，用于输出信息到控制台

namespace at { namespace cuda {  // 进入命名空间 at::cuda

NVRTC* load_nvrtc() {  // 定义 load_nvrtc 函数，返回类型为 NVRTC 指针

  auto self = new NVRTC();  // 创建一个 NVRTC 对象的指针，并赋值给 self
  
#define CREATE_ASSIGN(name) self->name = name;  // 使用宏定义创建对应成员变量的赋值语句，用于初始化 NVRTC 对象的成员变量
  AT_FORALL_NVRTC(CREATE_ASSIGN)  // 执行宏展开，对 NVRTC 对象的所有成员变量进行赋值

  return self;  // 返回初始化后的 NVRTC 对象指针
}

}} // at::cuda  // 退出命名空间 at::cuda
```