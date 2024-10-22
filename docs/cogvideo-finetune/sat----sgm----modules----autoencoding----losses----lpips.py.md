# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\losses\lpips.py`

```py
# 导入 PyTorch 和相关模块
import torch
import torch.nn as nn

# 从 util 模块导入默认值和配置实例化函数
from ....util import default, instantiate_from_config
# 从 lpips 模块导入 LPIPS 损失函数
from ..lpips.loss.lpips import LPIPS


# 定义 LatentLPIPS 类，继承自 nn.Module
class LatentLPIPS(nn.Module):
    # 初始化函数，接收多个参数进行配置
    def __init__(
        self,
        decoder_config,
        perceptual_weight=1.0,
        latent_weight=1.0,
        scale_input_to_tgt_size=False,
        scale_tgt_to_input_size=False,
        perceptual_weight_on_inputs=0.0,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置输入参数以控制输入大小缩放
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        # 初始化解码器
        self.init_decoder(decoder_config)
        # 创建 LPIPS 实例并设置为评估模式
        self.perceptual_loss = LPIPS().eval()
        # 设置感知损失权重
        self.perceptual_weight = perceptual_weight
        # 设置潜在损失权重
        self.latent_weight = latent_weight
        # 设置输入的感知损失权重
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    # 初始化解码器的函数
    def init_decoder(self, config):
        # 根据配置实例化解码器
        self.decoder = instantiate_from_config(config)
        # 如果解码器有编码器，则将其删除
        if hasattr(self.decoder, "encoder"):
            del self.decoder.encoder

    # 前向传播函数，接收潜在输入和预测，以及图像输入
    def forward(self, latent_inputs, latent_predictions, image_inputs, split="train"):
        # 初始化日志字典
        log = dict()
        # 计算潜在输入和预测之间的均方损失
        loss = (latent_inputs - latent_predictions) ** 2
        # 记录潜在损失到日志
        log[f"{split}/latent_l2_loss"] = loss.mean().detach()
        # 初始化图像重建变量
        image_reconstructions = None
        # 如果感知权重大于 0，进行感知损失计算
        if self.perceptual_weight > 0.0:
            # 解码潜在预测得到图像重建
            image_reconstructions = self.decoder.decode(latent_predictions)
            # 解码潜在输入得到目标图像
            image_targets = self.decoder.decode(latent_inputs)
            # 计算感知损失
            perceptual_loss = self.perceptual_loss(image_targets.contiguous(), image_reconstructions.contiguous())
            # 结合潜在损失和感知损失
            loss = self.latent_weight * loss.mean() + self.perceptual_weight * perceptual_loss.mean()
            # 记录感知损失到日志
            log[f"{split}/perceptual_loss"] = perceptual_loss.mean().detach()

        # 如果输入的感知损失权重大于 0
        if self.perceptual_weight_on_inputs > 0.0:
            # 如果没有重建图像，重新解码潜在预测
            image_reconstructions = default(image_reconstructions, self.decoder.decode(latent_predictions))
            # 根据配置缩放输入图像到目标大小
            if self.scale_input_to_tgt_size:
                image_inputs = torch.nn.functional.interpolate(
                    image_inputs,
                    image_reconstructions.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )
            # 根据配置缩放重建图像到输入大小
            elif self.scale_tgt_to_input_size:
                image_reconstructions = torch.nn.functional.interpolate(
                    image_reconstructions,
                    image_inputs.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )

            # 计算第二次感知损失
            perceptual_loss2 = self.perceptual_loss(image_inputs.contiguous(), image_reconstructions.contiguous())
            # 将第二次感知损失加入总损失
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            # 记录第二次感知损失到日志
            log[f"{split}/perceptual_loss_on_inputs"] = perceptual_loss2.mean().detach()
        # 返回最终损失和日志
        return loss, log
```