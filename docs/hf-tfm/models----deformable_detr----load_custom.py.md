# `.\models\deformable_detr\load_custom.py`

```py
# coding=utf-8  # 设置编码格式为UTF-8
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

# 导入必要的库
import os
from pathlib import Path


# 加载 CUDA 内核
def load_cuda_kernels():
    # 从 torch.utils.cpp_extension 导入 load模块
    from torch.utils.cpp_extension import load
    
    # 获取内核所在路径
    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr"

    # 获取源文件列表
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]
    
    # 使用load函数加载内核
    load(
        "MultiScaleDeformableAttention",  # 内核的名称
        src_files,  # 源文件列表
        with_cuda=True,  # 设置使用CUDA
        extra_include_paths=[str(root)],  # 指定附加的包含路径
        extra_cflags=["-DWITH_CUDA=1"],  # 附加的CFLAGS参数
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],  # 附加的CUDA_CFLAGS参数
    )

    # 导入加载的内核
    import MultiScaleDeformableAttention as MSDA

    # 返回加载的内核
    return MSDA
```