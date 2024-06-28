# `.\kernels\deta\vision.cpp`

```py
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

// 使用 Pybind11 构建一个 Python 模块，名字为 TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 定义 Python 接口函数 ms_deform_attn_forward，与 C++ 函数 ms_deform_attn_forward 绑定
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  // 定义 Python 接口函数 ms_deform_attn_backward，与 C++ 函数 ms_deform_attn_backward 绑定
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
```