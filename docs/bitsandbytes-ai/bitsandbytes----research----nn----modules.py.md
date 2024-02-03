# `bitsandbytes\bitsandbytes\research\nn\modules.py`

```py
from typing import TypeVar
# 导入 TypeVar 模块

import torch
# 导入 torch 模块
from torch import nn
# 从 torch 模块中导入 nn 模块

import bitsandbytes as bnb
# 导入 bitsandbytes 模块并重命名为 bnb

T = TypeVar("T", bound="torch.nn.Module")
# 定义一个类型变量 T，限定为 torch.nn.Module 类型的子类

class LinearFP8Mixed(nn.Linear):
    # 定义一个名为 LinearFP8Mixed 的类，继承自 nn.Linear 类
    def __init__(self, input_features, output_features, bias=True):
        # 初始化方法，接受输入特征数、输出特征数和是否包含偏置参数
        super().__init__(input_features, output_features, bias)
        # 调用父类的初始化方法
        self.bw_code = None
        # 初始化 bw_code 属性为 None
        self.fw_code = None
        # 初始化 fw_code 属性为 None
        array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
        # 定义一个数组 array 包含一系列数值
        for i, k in enumerate(array):
            # 遍历数组
            if input_features > array[i + 1]:
                # 如果输入特征数大于数组中下一个元素
                self.bsz = k
                # 将 k 赋值给 self.bsz
                break
        for i, k in enumerate(array):
            # 再次遍历数组
            if output_features > array[i + 1]:
                # 如果输出特征数大于数组中下一个元素
                self.bsz2 = k
                # 将 k 赋值给 self.bsz2

    def forward(self, x: torch.Tensor):
        # 前向传播方法，接受输入张量 x
        if self.fw_code is None:
            # 如果 fw_code 为 None
            self.bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(x.device)
            # 使用 bitsandbytes 模块中的 create_fp8_map 方法创建 bw_code
            self.fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(x.device)
            # 使用 bitsandbytes 模块中的 create_fp8_map 方法创建 fw_code

        out = bnb.research.matmul_fp8_mixed(x, self.weight.t(), fw_code=self.fw_code, bw_code=self.bw_code, bsz=self.bsz, bsz2=self.bsz2)
        # 调用 bitsandbytes 模块中的 matmul_fp8_mixed 方法进行矩阵乘法计算
        if self.bias is not None:
            # 如果存在偏置参数
            out += self.bias
            # 将偏置参数加到输出结果中

        return out
        # 返回输出结果

class LinearFP8Global(nn.Linear):
    # 定义一个名为 LinearFP8Global 的类，继承自 nn.Linear 类
    def __init__(self, input_features, output_features, bias=True):
        # 初始化方法，接受输入特征数、输出特征数和是否包含偏置参数
        super().__init__(input_features, output_features, bias)
        # 调用父类的初始化方法
        self.bw_code = None
        # 初始化 bw_code 属性为 None
        self.fw_code = None
        # 初始化 fw_code 属性为 None
        array = [4096, 2048, 1024, 512, 256, 128, 64, 0]
        # 定义一个数组 array 包含一系列数值
        for i, k in enumerate(array):
            # 遍历数组
            if input_features > array[i + 1]:
                # 如果输入特征数大于数组中下一个元素
                self.bsz = k
                # 将 k 赋值给 self.bsz
                break
        for i, k in enumerate(array):
            # 再次遍历数组
            if output_features > array[i + 1]:
                # 如果输出特征数大于数组中下一个元素
                self.bsz2 = k
                # 将 k 赋值给 self.bsz2
    # 定义一个前向传播函数，接受一个 torch.Tensor 类型的输入 x
    def forward(self, x: torch.Tensor):
        # 如果前向码为空，则创建前向码和后向码
        if self.fw_code is None:
            # 创建后向码，使用 bnb.functional.create_fp8_map 函数生成，参数为(True, 5, 2, 8)，并移到 x 的设备上
            self.bw_code = bnb.functional.create_fp8_map(True, 5, 2, 8).to(x.device)
            # 创建前向码，使用 bnb.functional.create_fp8_map 函数生成，参数为(True, 4, 3, 8)，并移到 x 的设备上
            self.fw_code = bnb.functional.create_fp8_map(True, 4, 3, 8).to(x.device)

        # 使用 bnb.matmul_fp8_global 函数进行矩阵乘法运算，传入输入 x、权重的转置、前向码、后向码、bsz 和 bsz2 参数
        out = bnb.matmul_fp8_global(x, self.weight.t(), fw_code=self.fw_code, bw_code=self.bw_code, bsz=self.bsz, bsz2=self.bsz2)
        # 如果存在偏置项，则将其加到输出上
        if self.bias is not None:
            out += self.bias

        # 返回输出结果
        return out
```