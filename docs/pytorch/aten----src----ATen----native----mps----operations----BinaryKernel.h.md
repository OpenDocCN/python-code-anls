# `.\pytorch\aten\src\ATen\native\mps\operations\BinaryKernel.h`

```
#pragma once


// 使用预处理指令#pragma once，确保头文件只被编译一次，防止多重包含的问题



namespace at::native::mps {


// 定义命名空间at::native::mps，用于组织和封装特定的函数和类型



void complex_mul_out(const Tensor& input, const Tensor& other, const Tensor& output);


// 声明一个函数complex_mul_out，接受三个Tensor类型的引用参数input、other和output，无返回值
// 函数用途可能是执行复杂的元素级乘法操作，并将结果存储在output中
```