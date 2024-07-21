# `.\pytorch\torch\ao\quantization\experimental\observer.py`

```
"""
This module implements nonuniform observers used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

import torch
import itertools
import matplotlib.pyplot as plt
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.experimental.apot_utils import float_to_apot, apot_to_float

# TODO: Consider adding NonUniformQuantizationObserverBase class
# when more than one non-uniform method is implemented

class APoTObserver(ObserverBase):
    b: int
    k: int
    n: int
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        b,
        k,
        dtype=torch.quint8) -> None:
        super().__init__(dtype)
        self.b = b
        self.k = k

        self.min_val = torch.tensor([])  # 初始化最小值为空的张量
        self.max_val = torch.tensor([])  # 初始化最大值为空的张量

    # min_val and max_val are optional args to override
    # the min_val and max_val observed by forward
    def calculate_qparams(self, signed):
        return self._calculate_qparams(signed, self.min_val, self.max_val)

    r""" Calculates nonuniform quantization parameters according to APoT paper:
    https://arxiv.org/pdf/1909.13144.pdf.
    Arg:
        signed: specifies whether to include signed values in quantization level calculations
        min_val: optional arg that can override min_val internal attribute
        max_val: optional arg that can override max_val internal attribute
    Returns:
        alpha: alpha quantization parameter, max of abs value of observed values
        gamma: gamma quantization parameter, defined to ensure that alpha is the maximum of the range
        quantization_levels: non-uniform quantization levels (fp representation)
        level_indices: int representation of quantization_levels indices
    """
    # 计算量化参数，用于量化神经网络权重
    def _calculate_qparams(self, signed: bool, min_val=None, max_val=None):
        # 如果给定最小值，则设置实例变量 self.min_val
        if min_val is not None:
            self.min_val = min_val
        # 如果给定最大值，则设置实例变量 self.max_val
        if max_val is not None:
            self.max_val = max_val

        # 计算 alpha，取最大的绝对值作为范围的上界
        alpha = torch.max(-self.min_val, self.max_val)

        # 检查 b 和 k 的有效性
        assert self.k and self.k != 0
        assert self.b % self.k == 0

        # 计算 n，并存储为成员变量
        self.n = self.b // self.k

        # 存储子张量的张量（所有级别）
        p_all = []

        # 创建级别
        for i in range(0, self.n):
            p_curr = torch.tensor([0])

            for j in range(0, (2 ** self.k - 2) + 1):
                curr_ele = 2 ** (- (i + j * self.n))
                p_append = torch.tensor([curr_ele])
                p_curr = torch.cat((p_curr, p_append))
                # 引入有符号数字
                if signed:
                    p_curr = torch.cat((p_curr, torch.tensor([-curr_ele])))

            if signed:
                # 如果有符号，则按降序排序张量
                sorted, indices = torch.sort(p_curr, descending=True)
                p_all.append(sorted)
            else:
                p_all.append(p_curr)

        # gamma 计算:
        # 遍历所有张量
        # 如果有符号，则将每个张量的第一个元素加入
        # 否则，将每个张量的第二个元素加入
        # gamma 被定义为确保 alpha 处于范围的最大值
        p_sum = 0.0
        for tens in p_all:
            if signed:
                p_sum += float(tens[0])
            else:
                p_sum += float(tens[1])

        # 分配 gamma
        gamma = alpha / p_sum

        # 计算笛卡尔积
        cartesian_product = list(itertools.product(*p_all))

        quantization_levels_list = []

        # 计算每行的总和
        for row in cartesian_product:
            sum = 0.0
            for ele in row:
                sum += ele
            quantization_levels_list.append(sum)

        # 使用 gamma 缩放量化级别列表中的每个元素
        quantization_levels_gamma = [float(gamma) * ele for ele in quantization_levels_list]
        # 将量化级别转换为张量并排序
        quantization_levels = torch.tensor(quantization_levels_gamma)
        level_indices = torch.tensor([])
        quantization_levels, level_indices = quantization_levels.sort()

        return (alpha, gamma, quantization_levels, level_indices)

    r"""记录 ``x`` 的运行最小值和最大值。
        Args:
            x_orig: 要观察最小值和最大值的张量"""
    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        # 分离 x_orig 的副本 x
        x = x_orig.detach()
        # 计算 x 的最小值和最大值
        min_val, max_val = torch.aminmax(x)
        # 如果 self.min_val 非空，则取最小值的最小值
        if self.min_val.numel():
            min_val = torch.min(min_val, self.min_val)
        # 如果 self.max_val 非空，则取最大值的最大值
        if self.max_val.numel():
            max_val = torch.max(max_val, self.max_val)
        # 更新实例变量 self.min_val 和 self.max_val
        self.min_val = min_val
        self.max_val = max_val
        return x_orig
    r"""Displays visualization of APoT quantization levels
        Args:
            observer: APoTObserver to calculate qparams
            signed: bool to indicate if qparams should be signed/unsigned
    """
    # 定义一个方法用于显示APoT量化水平的可视化效果
    def quant_levels_visualization(self, signed=False):
        # 计算量化参数alpha, gamma, quantization_levels和level_indices
        alpha, gamma, quantization_levels, level_indices = self.calculate_qparams(signed)

        # 创建用于绘图的x轴数据，范围是[0, 1)，每隔0.001一个点
        xs = [float(x) / 1000.0 for x in range(1000)]
        
        # 根据APoT量化转换函数将每个x值映射到量化后的y值
        ys = [apot_to_float(float_to_apot(x, quantization_levels, level_indices, alpha),
                            quantization_levels, level_indices).item() for x in xs]

        # 创建一个图形对象，并设置图形的大小
        f = plt.figure(figsize=(15, 10))

        # 绘制x轴为xs，y轴为ys的曲线图
        plt.plot(xs, ys)
        # 设置图形的标题
        plt.title("APoT Quantization Plot")
        # 设置x轴标签
        plt.xlabel("Full Precision")
        # 设置y轴标签
        plt.ylabel("Quantized")
        # 显示图形
        plt.show()
```