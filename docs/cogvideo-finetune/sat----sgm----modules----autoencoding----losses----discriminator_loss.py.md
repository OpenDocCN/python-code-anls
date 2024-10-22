# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\losses\discriminator_loss.py`

```py
# 导入所需的类型定义
from typing import Dict, Iterator, List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 torchvision 库
import torchvision
# 从 einops 库导入 rearrange 函数
from einops import rearrange
# 从 matplotlib 库导入 colormaps
from matplotlib import colormaps
# 从 matplotlib 库导入 pyplot
from matplotlib import pyplot as plt

# 导入自定义的工具函数
from ....util import default, instantiate_from_config
# 导入 LPIPS 损失函数
from ..lpips.loss.lpips import LPIPS
# 导入模型的权重初始化函数
from ..lpips.model.model import weights_init
# 导入两种感知损失函数
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss


# 定义一个带有鉴别器的通用 LPIPS 类
class GeneralLPIPSWithDiscriminator(nn.Module):
    # 初始化方法，接受多个参数进行配置
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, Dict[str, float]] = None,
        additional_log_keys: Optional[List[str]] = None,
        discriminator_config: Optional[Dict] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存维度信息
        self.dims = dims
        # 如果维度大于2，打印相关信息
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss "
                f"calculation, the LPIPS loss will be applied to each frame "
                f"independently."
            )
        # 保存是否缩放输入至目标大小
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        # 确保鉴别器损失是有效的
        assert disc_loss in ["hinge", "vanilla"]
        # 初始化感知损失为 LPIPS 模型并设置为评估模式
        self.perceptual_loss = LPIPS().eval()
        # 保存感知损失的权重
        self.perceptual_weight = perceptual_weight
        # 输出对数方差，设置为可训练参数
        self.logvar = nn.Parameter(torch.full((), logvar_init), requires_grad=learn_logvar)
        # 保存是否学习对数方差
        self.learn_logvar = learn_logvar

        # 使用默认配置创建鉴别器配置
        discriminator_config = default(
            discriminator_config,
            {
                "target": "sgm.modules.autoencoding.lpips.model.model.NLayerDiscriminator",
                "params": {
                    "input_nc": disc_in_channels,
                    "n_layers": disc_num_layers,
                    "use_actnorm": False,
                },
            },
        )

        # 实例化鉴别器并应用权重初始化
        self.discriminator = instantiate_from_config(discriminator_config).apply(weights_init)
        # 保存鉴别器开始训练的迭代次数
        self.discriminator_iter_start = disc_start
        # 根据损失类型选择相应的损失函数
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        # 保存鉴别器的因子和权重
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        # 设置正则化权重
        self.regularization_weights = default(regularization_weights, {})

        # 定义前向传播时需要的键
        self.forward_keys = [
            "optimizer_idx",
            "global_step",
            "last_layer",
            "split",
            "regularization_log",
        ]

        # 创建额外的日志键集
        self.additional_log_keys = set(default(additional_log_keys, []))
        # 更新日志键集合，包含正则化权重的键
        self.additional_log_keys.update(set(self.regularization_weights.keys()))

    # 获取可训练参数的迭代器
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminator.parameters()
    # 获取可训练的自编码器参数的生成器
    def get_trainable_autoencoder_parameters(self) -> Iterator[nn.Parameter]:
        # 如果需要学习对数方差，则生成 logvar 参数
        if self.learn_logvar:
            yield self.logvar
        # 生成器为空，表示没有其他可训练参数
        yield from ()

    # 计算自适应权重，使用 torch.no_grad() 防止计算梯度
    @torch.no_grad()
    def calculate_adaptive_weight(
        self, nll_loss: torch.Tensor, g_loss: torch.Tensor, last_layer: torch.Tensor
    ) -> torch.Tensor:
        # 计算负对数似然损失相对于最后一层输出的梯度
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        # 计算生成损失相对于最后一层输出的梯度
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        # 计算自适应权重，使用负对数似然梯度的范数与生成损失梯度的范数的比值
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        # 将权重限制在 0.0 到 1e4 之间，并分离梯度
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        # 将权重乘以判别器权重
        d_weight = d_weight * self.discriminator_weight
        # 返回计算得到的自适应权重
        return d_weight

    # 定义前向传播方法，处理输入和重构数据
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        *,  # 用于确保参数顺序正确，强制使用关键字参数
        regularization_log: Dict[str, torch.Tensor],
        optimizer_idx: int,
        global_step: int,
        last_layer: torch.Tensor,
        split: str = "train",
        weights: Union[None, float, torch.Tensor] = None,
    ):
        # 前向传播的具体实现逻辑（未提供）

    # 计算负对数似然损失
    def get_nll_loss(
        self,
        rec_loss: torch.Tensor,
        weights: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算基本的负对数似然损失
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # 初始化加权负对数似然损失为基本损失
        weighted_nll_loss = nll_loss
        # 如果提供了权重，则计算加权损失
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        # 计算加权负对数似然损失的平均值
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        # 计算负对数似然损失的平均值
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # 返回负对数似然损失和加权负对数似然损失
        return nll_loss, weighted_nll_loss
```