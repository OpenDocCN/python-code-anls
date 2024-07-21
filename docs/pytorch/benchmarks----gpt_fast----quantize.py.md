# `.\pytorch\benchmarks\gpt_fast\quantize.py`

```py
# flake8: noqa: E266, C417, B950
# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F


##### 量化基元 ######


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # 假设对称量化
    # 假设轴 == 0
    # 假设密集内存格式
    # TODO（未来）：根据需要放宽上述限制

    # 默认设置用于激活的仿射量化
    eps = torch.finfo(torch.float32).eps

    # 获取最小值和最大值
    min_val, max_val = torch.aminmax(x, dim=1)

    # 基于最小值和最大值计算比例尺和零点
    # 参考：https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # 参考：https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # 确保 scales 与原始张量具有相同的数据类型
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # 基于 qmin/qmax/scales/zp 进行量化
    # 参考：https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales, zero_points


##### 仅权重 int8 每通道量化代码 ######


def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(child.in_features, child.out_features),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight.to("cpu")
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype).to("cpu")

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor
    # 初始化函数，用于创建一个线性层的对象
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # 创建一个字典，包含设备和数据类型参数
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入特征数量
        self.in_features = in_features
        # 设置输出特征数量
        self.out_features = out_features
        # 注册一个缓冲区（不参与梯度计算）来存储权重张量，初始化为空的 torch.int8 类型
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        # 注册一个缓冲区（不参与梯度计算）来存储缩放因子张量，初始化为全1的 torch.bfloat16 类型
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    # 前向传播函数，接受输入张量并返回输出张量
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 使用函数 F.linear 对输入张量进行线性变换，使用当前权重张量并将数据类型转换为输入张量的数据类型
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales
```