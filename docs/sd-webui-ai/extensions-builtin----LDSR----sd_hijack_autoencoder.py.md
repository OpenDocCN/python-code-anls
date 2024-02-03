# `stable-diffusion-webui\extensions-builtin\LDSR\sd_hijack_autoencoder.py`

```py
# 从 compvis/stable-diffusion 仓库的 ldm/models/autoencoder.py 文件中获取内容
# 后来在切换到 stability-ai/stablediffusion 仓库时，VQModel 和 VQModelInterface 从 ldm/models/autoencoder.py 中被移除
# 由于 LDSR 上采样器依赖于 VQModel 和 VQModelInterface，hijack 旨在将它们放回 ldm.models.autoencoder 中
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from torch.optim.lr_scheduler import LambdaLR

from ldm.modules.ema import LitEma
from vqvae_quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.util import instantiate_from_config

import ldm.models.autoencoder
from packaging import version

# 定义 VQModel 类，继承自 pl.LightningModule
class VQModel(pl.LightningModule):
    # 定义 ema_scope 上下文管理器
    @contextmanager
    def ema_scope(self, context=None):
        # 如果使用 EMA
        if self.use_ema:
            # 将模型参数存储到 EMA 中
            self.model_ema.store(self.parameters())
            # 将 EMA 参数复制到当前模型中
            self.model_ema.copy_to(self)
            # 如果有上下文信息，则打印切换到 EMA 权重的消息
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            # 返回 None
            yield None
        finally:
            # 如果使用 EMA
            if self.use_ema:
                # 恢复模型参数为训练权重
                self.model_ema.restore(self.parameters())
                # 如果有上下文信息，则打印恢复训练权重的消息
                if context is not None:
                    print(f"{context}: Restored training weights")
    # 从给定的检查点路径初始化模型参数，可以选择忽略某些键
    def init_from_ckpt(self, path, ignore_keys=None):
        # 加载检查点文件中的状态字典
        sd = torch.load(path, map_location="cpu")["state_dict"]
        # 获取状态字典中的所有键
        keys = list(sd.keys())
        # 遍历所有键
        for k in keys:
            # 检查是否需要忽略该键
            for ik in ignore_keys or []:
                if k.startswith(ik):
                    # 如果需要忽略，则删除该键
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # 加载状态字典到模型中，允许缺失键
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if missing:
            print(f"Missing Keys: {missing}")
        if unexpected:
            print(f"Unexpected Keys: {unexpected}")

    # 在训练批次结束时执行的操作，如果使用指数移动平均，则更新模型EMA
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # 对输入进行编码操作，返回量化后的结果、损失和信息
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    # 对输入进行编码操作，返回编码后的结果
    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    # 对量化后的结果进行解码操作
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    # 对编码后的代码进行解码操作
    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    # 前向传播函数，对输入进行编码和解码操作，可选择返回预测索引
    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff
    # 从批量数据中获取输入数据，根据指定的键值 k
    def get_input(self, batch, k):
        # 获取指定键值对应的数据 x
        x = batch[k]
        # 如果数据 x 的维度为 3，则在最后添加一个维度
        if len(x.shape) == 3:
            x = x[..., None]
        # 将数据 x 的维度重新排列，转换为指定的内存格式，并转换为浮点型数据
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        # 如果设置了批量调整大小范围
        if self.batch_resize_range is not None:
            # 获取调整大小的下限和上限
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            # 如果全局步数小于等于 4
            if self.global_step <= 4:
                # 初始几个批次使用最大大小，以避免后续内存溢出
                new_resize = upper_size
            else:
                # 随机选择一个新的调整大小值
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            # 如果新的调整大小值不等于数据 x 的第三维大小
            if new_resize != x.shape[2]:
                # 使用双三次插值调整数据 x 的大小
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            # 分离数据 x 的梯度信息
            x = x.detach()
        # 返回处理后的数据 x
        return x

    # 训练步骤函数，处理每个批次的训练过程
    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # 尽量不要欺骗启发式算法
        # 获取输入数据 x
        x = self.get_input(batch, self.image_key)
        # 调用模型进行前向传播，获取重构数据 xrec、量化损失 qloss 和预测索引 ind
        xrec, qloss, ind = self(x, return_pred_indices=True)

        # 如果优化器索引为 0
        if optimizer_idx == 0:
            # 自编码器
            # 计算自编码器损失 aeloss，并获取日志信息 log_dict_ae
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)
            # 记录日志信息
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            # 返回自编码器损失
            return aeloss

        # 如果优化器索引为 1
        if optimizer_idx == 1:
            # 判别器
            # 计算判别器损失 discloss，并获取日志信息 log_dict_disc
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            # 记录日志信息
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            # 返回判别器损失
            return discloss
    # 执行验证步骤，返回日志字典
    def validation_step(self, batch, batch_idx):
        # 调用内部验证步骤函数，获取日志字典
        log_dict = self._validation_step(batch, batch_idx)
        # 使用指数移动平均进行验证步骤，获取日志字典
        with self.ema_scope():
            self._validation_step(batch, batch_idx, suffix="_ema")
        # 返回日志字典
        return log_dict

    # 内部验证步骤函数，接收批次数据和批次索引，可指定后缀
    def _validation_step(self, batch, batch_idx, suffix=""):
        # 获取输入数据
        x = self.get_input(batch, self.image_key)
        # 获取重构数据、量化损失、预测索引
        xrec, qloss, ind = self(x, return_pred_indices=True)
        # 计算自编码器损失和日志字典
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=ind
                                        )

        # 计算判别器损失和日志字典
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
        # 获取重构损失
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        # 记录重构损失到日志
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # 记录自编码器损失到日志
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        # 如果 PyTorch-Lightning 版本大于等于 1.4.0，则删除指定键
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        # 记录自编码器日志字典
        self.log_dict(log_dict_ae)
        # 记录判别器日志字典
        self.log_dict(log_dict_disc)
        # 返回日志字典
        return self.log_dict
    # 配置优化器，设置判别器和生成器的学习率
    def configure_optimizers(self):
        # 获取初始学习率
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        # 打印判别器和生成器的学习率
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        # 创建 Adam 优化器，包括编码器、解码器、量化器等参数
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        # 创建 Adam 优化器，包括损失函数的判别器参数
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        # 如果存在调度器配置，则实例化调度器
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            # 设置 LambdaLR 调度器，分别应用于生成器和判别器的优化器
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            # 返回优化器和调度器
            return [opt_ae, opt_disc], scheduler
        # 如果不存在调度器配置，则返回空的调度器
        return [opt_ae, opt_disc], []

    # 获取最后一层的权重
    def get_last_layer(self):
        return self.decoder.conv_out.weight
    # 记录图像数据，返回一个包含输入和重建图像的字典
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        # 初始化一个空字典用于记录数据
        log = {}
        # 获取输入图像数据
        x = self.get_input(batch, self.image_key)
        # 将输入图像数据移动到指定设备上
        x = x.to(self.device)
        # 如果只需要记录输入数据，则将输入数据记录到字典中并返回
        if only_inputs:
            log["inputs"] = x
            return log
        # 对输入数据进行重建
        xrec, _ = self(x)
        # 如果输入数据的通道数大于3
        if x.shape[1] > 3:
            # 使用随机投影进行着色
            assert xrec.shape[1] > 3
            # 将输入数据和重建数据转换为RGB格式
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        # 记录输入数据和重建数据到字典中
        log["inputs"] = x
        log["reconstructions"] = xrec
        # 如果需要绘制指数移动平均图像
        if plot_ema:
            # 在指数移动平均范围内进行操作
            with self.ema_scope():
                # 获取指数移动平均的重建图像
                xrec_ema, _ = self(x)
                # 如果输入数据的通道数大于3
                if x.shape[1] > 3:
                    # 将指数移动平均的重建图像转换为RGB格式
                    xrec_ema = self.to_rgb(xrec_ema)
                # 记录指数移动平均的重建图像到字典中
                log["reconstructions_ema"] = xrec_ema
        # 返回记录的数据字典
        return log

    # 将输入数据转换为RGB格式
    def to_rgb(self, x):
        # 断言输入数据的关键字为"segmentation"
        assert self.image_key == "segmentation"
        # 如果模型没有colorize属性，则初始化一个随机颜色化参数
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        # 使用卷积操作将输入数据转换为RGB格式
        x = F.conv2d(x, weight=self.colorize)
        # 将数据归一化到[-1, 1]范围内
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        # 返回转换后的RGB格式数据
        return x
# 定义 VQModelInterface 类，继承自 VQModel 类
class VQModelInterface(VQModel):
    # 初始化方法，接受嵌入维度参数和其他参数
    def __init__(self, embed_dim, *args, **kwargs):
        # 调用父类的初始化方法，传入其他参数和嵌入维度参数
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        # 设置对象的嵌入维度属性
        self.embed_dim = embed_dim

    # 编码方法，接受输入 x，经过编码器和量化卷积层处理后返回结果
    def encode(self, x):
        # 经过编码器处理得到 h
        h = self.encoder(x)
        # 经过量化卷积层处理得到 h
        h = self.quant_conv(h)
        return h

    # 解码方法，接受编码结果 h 和是否强制不量化的标志，经过量化和解码器处理后返回结果
    def decode(self, h, force_not_quantize=False):
        # 如果不强制不量化，则通过量化方法 quantize 处理 h，得到量化结果 quant、嵌入损失 emb_loss 和信息 info
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            # 否则直接使用 h 作为量化结果
            quant = h
        # 通过后量化卷积层处理量化结果 quant
        quant = self.post_quant_conv(quant)
        # 经过解码器处理得到解码结果 dec
        dec = self.decoder(quant)
        return dec

# 将 VQModel 和 VQModelInterface 类添加到 ldm.models.autoencoder 模块中
ldm.models.autoencoder.VQModel = VQModel
ldm.models.autoencoder.VQModelInterface = VQModelInterface
```