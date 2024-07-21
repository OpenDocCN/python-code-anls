# `.\pytorch\torch\csrc\autograd\python_fft_functions.h`

```py
#pragma once


// 使用 pragma once 预处理指令，确保当前头文件只被编译一次，防止重复包含


namespace torch::autograd {


// 定义命名空间 torch::autograd，用于封装 Torch 自动求导模块的功能


void initFFTFunctions(PyObject* module);


// 声明函数 initFFTFunctions，该函数用于初始化 FFT 相关的函数，并接收一个 PyObject 模块对象作为参数


}


// 结束 torch::autograd 命名空间的定义
```