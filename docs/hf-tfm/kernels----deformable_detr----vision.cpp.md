# `.\kernels\deformable_detr\vision.cpp`

```
/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "ms_deform_attn.h"

// 使用 PYBIND11_MODULE 宏定义，将 C++ 函数绑定到 Python 中
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 定义 Python 可调用函数 ms_deform_attn_forward，对应 C++ 中的 ms_deform_attn_forward 函数
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  // 定义 Python 可调用函数 ms_deform_attn_backward，对应 C++ 中的 ms_deform_attn_backward 函数
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
```