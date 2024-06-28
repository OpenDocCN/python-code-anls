# `.\models\deformable_detr\load_custom.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Loading of Deformable DETR's CUDA kernels"""

import os  # 导入操作系统相关的模块
from pathlib import Path  # 导入处理文件路径的模块


def load_cuda_kernels():
    from torch.utils.cpp_extension import load  # 导入加载自定义C++扩展的函数

    # 获取当前脚本文件的父目录的父目录，并拼接出CUDA内核源文件所在路径
    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr"
    # 定义需要加载的所有源文件的路径列表
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    # 使用torch的cpp_extension模块加载CUDA扩展，并指定相关配置
    load(
        "MultiScaleDeformableAttention",  # 扩展名
        src_files,  # 源文件路径列表
        with_cuda=True,  # 指定是否包含CUDA支持
        extra_include_paths=[str(root)],  # 额外的头文件包含路径
        extra_cflags=["-DWITH_CUDA=1"],  # 额外的C编译标志
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",  # CUDA支持的FP16
            "-D__CUDA_NO_HALF_OPERATORS__",  # 禁用CUDA半精度操作符
            "-D__CUDA_NO_HALF_CONVERSIONS__",  # 禁用CUDA半精度转换
            "-D__CUDA_NO_HALF2_OPERATORS__",  # 禁用CUDA半精度操作符
        ],
    )

    import MultiScaleDeformableAttention as MSDA  # 导入加载的扩展模块作为MSDA

    return MSDA  # 返回加载后的扩展模块对象
```