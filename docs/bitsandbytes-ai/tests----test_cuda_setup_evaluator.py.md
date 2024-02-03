# `bitsandbytes\tests\test_cuda_setup_evaluator.py`

```py
# 导入必要的库
import os
from pathlib import Path

import torch

# 手动覆盖测试。虽然不好，但目前是一个健全性检查
# TODO: 改进这个
def test_manual_override(requires_cuda):
    # 手动设置 CUDA 路径
    manual_cuda_path = str(Path('/mmfs1/home/dettmers/data/local/cuda-12.2'))

    # 获取 PyTorch 版本的 CUDA 版本号
    pytorch_version = torch.version.cuda.replace('.', '')

    # 断言 PyTorch 版本的 CUDA 版本号不等于 122（这个断言永远不会为真）
    assert pytorch_version != 122

    # 设置环境变量 CUDA_HOME 为手动设置的 CUDA 路径
    os.environ['CUDA_HOME']='{manual_cuda_path}'
    # 设置环境变量 BNB_CUDA_VERSION 为 122
    os.environ['BNB_CUDA_VERSION']='122'
    # 断言手动设置的 CUDA 路径在 LD_LIBRARY_PATH 环境变量中
    #assert str(manual_cuda_path) in os.environ['LD_LIBRARY_PATH']
    # 导入 bitsandbytes 库
    import bitsandbytes as bnb
    # 获取加载的库文件名
    loaded_lib = bnb.cuda_setup.main.CUDASetup.get_instance().binary_name
    # 断言加载的库文件名为 'libbitsandbytes_cuda122.so'
    #assert loaded_lib == 'libbitsandbytes_cuda122.so'
```