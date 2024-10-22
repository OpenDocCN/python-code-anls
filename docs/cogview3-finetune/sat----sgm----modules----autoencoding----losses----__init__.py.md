# `.\cogview3-finetune\sat\sgm\modules\autoencoding\losses\__init__.py`

```
# 导入类型提示 Any 和 Union
from typing import Any, Union

# 导入 PyTorch 及其神经网络模块
import torch
import torch.nn as nn
# 从 einops 导入 rearrange 函数
from einops import rearrange

# 导入自定义工具函数和类
from ....util import default, instantiate_from_config
# 从 lpips 库导入 LPIPS 损失类
from ..lpips.loss.lpips import LPIPS
# 从 lpips 模型导入 NLayerDiscriminator 和权重初始化函数
from ..lpips.model.model import NLayerDiscriminator, weights_init
# 从 vqperceptual 模块导入两种损失函数
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss


# 定义 adopt_weight 函数，调整权重值
def adopt_weight(weight, global_step, threshold=0, value=0.0):
    # 如果全局步数小于阈值，则将权重设为给定值
    if global_step < threshold:
        weight = value
    # 返回调整后的权重
    return weight


# 定义 LatentLPIPS 类，继承自 nn.Module
class LatentLPIPS(nn.Module):
    # 初始化方法，设置相关参数
    def __init__(
        self,
        decoder_config,
        perceptual_weight=1.0,
        latent_weight=1.0,
        scale_input_to_tgt_size=False,
        scale_tgt_to_input_size=False,
        perceptual_weight_on_inputs=0.0,
    ):
        # 调用父类构造方法
        super().__init__()
        # 设置输入大小缩放标志
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        # 初始化解码器
        self.init_decoder(decoder_config)
        # 初始化感知损失模型，并设置为评估模式
        self.perceptual_loss = LPIPS().eval()
        # 设置感知损失和潜在损失的权重
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        # 设置对输入的感知权重
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    # 定义初始化解码器的方法
    def init_decoder(self, config):
        # 从配置实例化解码器
        self.decoder = instantiate_from_config(config)
        # 如果解码器有 encoder 属性，则删除该属性
        if hasattr(self.decoder, "encoder"):
            del self.decoder.encoder
    # 定义前向传播函数，接收潜在输入、潜在预测、图像输入和数据集切分信息
    def forward(self, latent_inputs, latent_predictions, image_inputs, split="train"):
        # 初始化一个字典用于记录日志信息
        log = dict()
        # 计算潜在输入与潜在预测之间的均方差损失
        loss = (latent_inputs - latent_predictions) ** 2
        # 将均方差损失的平均值添加到日志中，使用切分名称作为键
        log[f"{split}/latent_l2_loss"] = loss.mean().detach()
        # 初始化图像重建变量
        image_reconstructions = None
        # 如果感知损失权重大于0，则进行感知损失的计算
        if self.perceptual_weight > 0.0:
            # 解码潜在预测生成图像重建
            image_reconstructions = self.decoder.decode(latent_predictions)
            # 解码潜在输入生成目标图像
            image_targets = self.decoder.decode(latent_inputs)
            # 计算感知损失
            perceptual_loss = self.perceptual_loss(
                image_targets.contiguous(), image_reconstructions.contiguous()
            )
            # 综合潜在损失和感知损失，更新总损失
            loss = (
                self.latent_weight * loss.mean()
                + self.perceptual_weight * perceptual_loss.mean()
            )
            # 将感知损失的平均值添加到日志中
            log[f"{split}/perceptual_loss"] = perceptual_loss.mean().detach()
    
        # 如果感知权重在输入上大于0，则进行相应的处理
        if self.perceptual_weight_on_inputs > 0.0:
            # 如果重建图像为空，则解码潜在预测生成图像重建
            image_reconstructions = default(
                image_reconstructions, self.decoder.decode(latent_predictions)
            )
            # 如果需要将输入图像缩放到目标图像大小
            if self.scale_input_to_tgt_size:
                image_inputs = torch.nn.functional.interpolate(
                    image_inputs,
                    image_reconstructions.shape[2:],
                    mode="bicubic",  # 使用双三次插值法
                    antialias=True,  # 使用抗锯齿
                )
            # 如果需要将目标图像缩放到输入图像大小
            elif self.scale_tgt_to_input_size:
                image_reconstructions = torch.nn.functional.interpolate(
                    image_reconstructions,
                    image_inputs.shape[2:],
                    mode="bicubic",  # 使用双三次插值法
                    antialias=True,  # 使用抗锯齿
                )
    
            # 计算与输入图像的感知损失
            perceptual_loss2 = self.perceptual_loss(
                image_inputs.contiguous(), image_reconstructions.contiguous()
            )
            # 更新总损失，加入输入的感知损失
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            # 将输入的感知损失的平均值添加到日志中
            log[f"{split}/perceptual_loss_on_inputs"] = perceptual_loss2.mean().detach()
        # 返回总损失和日志信息
        return loss, log
# 定义一个带有判别器的通用 LPIPS 类，继承自 nn.Module
class GeneralLPIPSWithDiscriminator(nn.Module):
    # 初始化方法，接收多个参数以配置模型
    def __init__(
        self,
        disc_start: int,  # 判别器开始训练的迭代次数
        logvar_init: float = 0.0,  # 日志方差的初始值
        pixelloss_weight=1.0,  # 像素损失的权重
        disc_num_layers: int = 3,  # 判别器的层数
        disc_in_channels: int = 3,  # 判别器输入的通道数
        disc_factor: float = 1.0,  # 判别器的缩放因子
        disc_weight: float = 1.0,  # 判别器损失的权重
        perceptual_weight: float = 1.0,  # 感知损失的权重
        disc_loss: str = "hinge",  # 判别器使用的损失类型
        scale_input_to_tgt_size: bool = False,  # 是否将输入缩放到目标大小
        dims: int = 2,  # 数据的维度
        learn_logvar: bool = False,  # 是否学习日志方差
        regularization_weights: Union[None, dict] = None,  # 正则化权重
    ):
        # 调用父类的初始化方法
        super().__init__()
        self.dims = dims  # 保存维度信息
        # 如果维度大于2，打印警告信息
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss calculation, "
                f"the LPIPS loss will be applied to each frame independently. "
            )
        self.scale_input_to_tgt_size = scale_input_to_tgt_size  # 保存输入缩放标志
        # 确保判别器损失类型为 hinge 或 vanilla
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight  # 保存像素损失权重
        self.perceptual_loss = LPIPS().eval()  # 初始化 LPIPS 感知损失并设置为评估模式
        self.perceptual_weight = perceptual_weight  # 保存感知损失权重
        # 输出日志方差，作为可学习的参数
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar  # 保存是否学习日志方差的标志

        # 初始化 NLayerDiscriminator 作为判别器
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False
        ).apply(weights_init)  # 应用权重初始化
        self.discriminator_iter_start = disc_start  # 保存判别器开始训练的迭代次数
        # 根据损失类型选择合适的损失函数
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor  # 保存判别器缩放因子
        self.discriminator_weight = disc_weight  # 保存判别器损失权重
        self.regularization_weights = default(regularization_weights, {})  # 设置正则化权重，默认为空字典

    # 获取可训练的参数
    def get_trainable_parameters(self) -> Any:
        return self.discriminator.parameters()  # 返回判别器的参数

    # 获取可训练的自编码器参数
    def get_trainable_autoencoder_parameters(self) -> Any:
        if self.learn_logvar:  # 如果学习日志方差
            yield self.logvar  # 生成日志方差
        yield from ()  # 返回空生成器

    # 计算自适应权重
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:  # 如果提供了最后一层
            # 计算负对数似然损失的梯度
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            # 计算生成器损失的梯度
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # 如果没有提供最后一层，使用类属性中的最后一层
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        # 计算判别器权重
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()  # 限制权重范围并分离计算图
        d_weight = d_weight * self.discriminator_weight  # 应用判别器权重
        return d_weight  # 返回计算得到的权重

    # 前向传播方法
    def forward(
        self,
        regularization_log,  # 正则化日志
        inputs,  # 输入数据
        reconstructions,  # 重建数据
        optimizer_idx,  # 优化器索引
        global_step,  # 全局步数
        last_layer=None,  # 最后一层（可选）
        split="train",  # 数据集划分（训练或验证）
        weights=None,  # 权重（可选）
```