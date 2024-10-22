# `.\cogview3-finetune\sat\sgm\models\autoencoder.py`

```
# 导入标准库和第三方库
import logging  # 用于记录日志信息
import math  # 提供数学函数
import re  # 提供正则表达式支持
from abc import abstractmethod  # 用于定义抽象方法
from contextlib import contextmanager  # 提供上下文管理器功能
from typing import Any, Dict, Tuple, Union  # 提供类型提示支持

import pytorch_lightning as pl  # 引入 PyTorch Lightning 框架
import torch  # 引入 PyTorch 库
from omegaconf import ListConfig  # 用于处理配置文件
from packaging import version  # 用于版本比较
from safetensors.torch import load_file as load_safetensors  # 用于加载 safetensors 格式文件

from ..modules.diffusionmodules.model import Decoder, Encoder  # 导入解码器和编码器
from ..modules.distributions.distributions import DiagonalGaussianDistribution  # 导入对角高斯分布
from ..modules.ema import LitEma  # 导入指数移动平均类
from ..util import default, get_obj_from_str, instantiate_from_config  # 导入实用工具函数


class AbstractAutoencoder(pl.LightningModule):
    """
    这是所有自编码器的基类，包括图像自编码器、带判别器的图像自编码器、
    unCLIP 模型等。因此，它是相当通用的，特定功能
    （例如判别器训练、编码、解码）必须在子类中实现。
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,  # 指定 EMA 衰减值
        monitor: Union[None, str] = None,  # 用于监控的指标名称
        input_key: str = "jpg",  # 输入数据的键名，默认为 "jpg"
        ckpt_path: Union[None, str] = None,  # 检查点文件路径
        ignore_keys: Union[Tuple, list, ListConfig] = (),  # 需要忽略的键
    ):
        super().__init__()  # 调用父类构造函数
        self.input_key = input_key  # 保存输入数据的键名
        self.use_ema = ema_decay is not None  # 判断是否使用 EMA
        if monitor is not None:  # 如果提供监控指标
            self.monitor = monitor  # 保存监控指标

        if self.use_ema:  # 如果使用 EMA
            self.model_ema = LitEma(self, decay=ema_decay)  # 初始化 EMA 对象
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")  # 打印 EMA 缓冲区的数量

        if ckpt_path is not None:  # 如果提供检查点路径
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 从检查点初始化模型

        if version.parse(torch.__version__) >= version.parse("2.0.0"):  # 如果 PyTorch 版本 >= 2.0.0
            self.automatic_optimization = False  # 禁用自动优化

    def init_from_ckpt(
        self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple()  # 初始化检查点
    ) -> None:
        if path.endswith("ckpt"):  # 如果路径以 "ckpt" 结尾
            sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载检查点的状态字典
        elif path.endswith("safetensors"):  # 如果路径以 "safetensors" 结尾
            sd = load_safetensors(path)  # 加载 safetensors 文件
        else:  # 如果路径不符合以上两种格式
            raise NotImplementedError  # 抛出未实现异常

        keys = list(sd.keys())  # 获取状态字典的所有键
        for k in keys:  # 遍历每个键
            for ik in ignore_keys:  # 遍历忽略的键
                if re.match(ik, k):  # 如果键匹配忽略模式
                    print("Deleting key {} from state_dict.".format(k))  # 打印被删除的键
                    del sd[k]  # 从状态字典中删除该键
        missing, unexpected = self.load_state_dict(sd, strict=False)  # 加载状态字典，允许非严格匹配
        # print(
        #     f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        # )  # 打印恢复信息
        # if len(missing) > 0:  # 如果有缺失的键
        #     print(f"Missing Keys: {missing}")  # 打印缺失的键
        # if len(unexpected) > 0:  # 如果有意外的键
        #     print(f"Unexpected Keys: {unexpected}")  # 打印意外的键
    # 应用检查点，参数可以是 None、路径字符串或字典
    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        # 如果检查点为 None，直接返回
        if ckpt is None:
            return
        # 如果检查点是字符串，将其转换为字典格式
        if isinstance(ckpt, str):
            ckpt = {
                # 指定检查点引擎的目标
                "target": "sgm.modules.checkpoint.CheckpointEngine",
                # 指定检查点路径
                "params": {"ckpt_path": ckpt},
            }
        # 根据配置实例化检查点引擎
        engine = instantiate_from_config(ckpt)
        # 调用引擎并传入当前对象
        engine(self)
    
    # 抽象方法，获取输入数据，参数为 batch，返回类型为 Any
    @abstractmethod
    def get_input(self, batch) -> Any:
        # 抛出未实现错误
        raise NotImplementedError()
    
    # 训练批次结束时的回调函数
    def on_train_batch_end(self, *args, **kwargs):
        # 用于 EMA（Exponential Moving Average）计算
        if self.use_ema:
            # 调用 EMA 模型
            self.model_ema(self)
    
    # 上下文管理器，处理 EMA 权重的切换
    @contextmanager
    def ema_scope(self, context=None):
        # 如果使用 EMA
        if self.use_ema:
            # 存储当前参数
            self.model_ema.store(self.parameters())
            # 将 EMA 权重复制到当前模型
            self.model_ema.copy_to(self)
            # 如果提供上下文，打印切换消息
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            # 进入上下文
            yield None
        finally:
            # 离开上下文时恢复参数
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                # 如果提供上下文，打印恢复消息
                if context is not None:
                    print(f"{context}: Restored training weights")
    
    # 抽象方法，进行编码，返回类型为 torch.Tensor
    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        # 抛出未实现错误
        raise NotImplementedError("encode()-method of abstract base class called")
    
    # 抽象方法，进行解码，返回类型为 torch.Tensor
    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        # 抛出未实现错误
        raise NotImplementedError("decode()-method of abstract base class called")
    
    # 从配置实例化优化器
    def instantiate_optimizer_from_config(self, params, lr, cfg):
        # 打印正在加载的优化器信息
        print(f"loading >>> {cfg['target']} <<< optimizer from config")
        # 根据配置获取优化器对象，并初始化
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )
    
    # 配置优化器，返回类型为 Any
    def configure_optimizers(self) -> Any:
        # 抛出未实现错误
        raise NotImplementedError()
# 自动编码器引擎的基类，供所有图像自动编码器使用，如 VQGAN 或 AutoencoderKL
class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    # 初始化方法，配置编码器、解码器、损失函数等
    def __init__(
        self,
        *args,
        encoder_config: Dict,  # 编码器配置字典
        decoder_config: Dict,  # 解码器配置字典
        loss_config: Dict,  # 损失函数配置字典
        regularizer_config: Dict,  # 正则化器配置字典
        optimizer_config: Union[Dict, None] = None,  # 优化器配置字典（可选）
        lr_g_factor: float = 1.0,  # 学习率缩放因子
        ckpt_path=None,  # 检查点路径（可选）
        ignore_keys=[],  # 忽略的键列表
        **kwargs,  # 额外的关键字参数
    ):
        super().__init__(*args, **kwargs)  # 调用父类初始化方法
        # todo: add options to freeze encoder/decoder
        # 实例化编码器
        self.encoder = instantiate_from_config(encoder_config)
        # 实例化解码器
        self.decoder = instantiate_from_config(decoder_config)
        # 实例化损失函数
        self.loss = instantiate_from_config(loss_config)
        # 实例化正则化器
        self.regularization = instantiate_from_config(regularizer_config)
        # 设置优化器配置，默认为 Adam
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam"}
        )
        # 设置学习率缩放因子
        self.lr_g_factor = lr_g_factor
        # 如果检查点路径不为空，初始化从检查点
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    # 从检查点加载模型状态
    def init_from_ckpt(self, path, ignore_keys=list()):
        # 根据文件扩展名加载状态字典
        if path.endswith("ckpt") or path.endswith("pt"):
            sd = torch.load(path, map_location="cpu")['state_dict']  # 加载 PyTorch 检查点
        elif path.endswith("safetensors"):
            sd = load_safetensors(path)  # 加载 safetensors 格式
        else:
            raise NotImplementedError  # 未实现的文件格式处理
        keys = list(sd.keys())  # 获取状态字典中的所有键
        for k in keys:
            for ik in ignore_keys:
                # 如果键以忽略键开头，删除该键
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # 加载状态字典，返回缺失和意外的键
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print("Missing keys: ", missing_keys)  # 打印缺失的键
        print("Unexpected keys: ", unexpected_keys)  # 打印意外的键
        print(f"Restored from {path}")  # 打印恢复信息

    # 获取输入数据
    def get_input(self, batch: Dict) -> torch.Tensor:
        # 假设统一数据格式，数据加载器返回一个字典。
        # 图像张量应缩放至 -1 ... 1 并采用通道优先格式（如 bchw 而非 bhwc）
        return batch[self.input_key]  # 返回指定的输入键对应的张量

    # 获取自动编码器的所有参数
    def get_autoencoder_params(self) -> list:
        # 收集编码器、解码器、正则化器和损失函数的可训练参数
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.regularization.get_trainable_parameters())
            + list(self.loss.get_trainable_autoencoder_parameters())
        )
        return params  # 返回所有参数的列表

    # 获取鉴别器的参数
    def get_discriminator_params(self) -> list:
        # 获取损失函数中的可训练参数，例如鉴别器
        params = list(self.loss.get_trainable_parameters())  
        return params  # 返回鉴别器参数的列表

    # 获取解码器的最后一层
    def get_last_layer(self):
        return self.decoder.get_last_layer()  # 返回解码器的最后一层

    # 编码输入张量
    def encode(
        self,
        x: torch.Tensor,  # 输入张量
        return_reg_log: bool = False,  # 是否返回正则化日志
        unregularized: bool = False,  # 是否使用未正则化的编码
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # 使用编码器对输入 x 进行编码，生成潜在表示 z
        z = self.encoder(x)
        # 如果 unregularized 为真，返回 z 和空字典
        if unregularized:
            return z, dict()
        # 对 z 进行正则化处理，得到正则化后的 z 和正则化日志 reg_log
        z, reg_log = self.regularization(z)
        # 如果需要返回正则化日志，返回 z 和 reg_log
        if return_reg_log:
            return z, reg_log
        # 否则只返回 z
        return z

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        # 使用解码器对潜在表示 z 进行解码，生成输出 x
        x = self.decoder(z, **kwargs)
        # 返回解码后的输出
        return x

    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 执行编码过程，获取潜在表示 z 和正则化日志 reg_log
        z, reg_log = self.encode(x, return_reg_log=True)
        # 对潜在表示 z 进行解码，获取重建的输入 dec
        dec = self.decode(z)
        # 返回潜在表示 z，重建输入 dec 和正则化日志 reg_log
        return z, dec, reg_log

    def training_step(self, batch, batch_idx, optimizer_idx) -> Any:
        # 获取当前 batch 的输入数据 x
        x = self.get_input(batch)
        # 执行前向传播，获取潜在表示 z，重建输入 xrec 和正则化日志
        z, xrec, regularization_log = self(x)

        # 判断优化器的索引，进行不同的训练步骤
        if optimizer_idx == 0:
            # 处理自编码器的损失
            aeloss, log_dict_ae = self.loss(
                # 计算自编码器损失
                regularization_log,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            # 记录自编码器的日志
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            # 返回自编码器的损失
            return aeloss

        if optimizer_idx == 1:
            # 处理判别器的损失
            discloss, log_dict_disc = self.loss(
                # 计算判别器损失
                regularization_log,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            # 记录判别器的日志
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
            # 返回判别器的损失
            return discloss

    def validation_step(self, batch, batch_idx) -> Dict:
        # 执行验证步骤，获取日志字典
        log_dict = self._validation_step(batch, batch_idx)
        # 在 EMA 范围内执行验证步骤，获取 EMA 日志
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            # 更新日志字典，合并 EMA 日志
            log_dict.update(log_dict_ema)
        # 返回更新后的日志字典
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix="") -> Dict:
        # 获取当前 batch 的输入数据 x
        x = self.get_input(batch)

        # 执行前向传播，获取潜在表示 z，重建输入 xrec 和正则化日志
        z, xrec, regularization_log = self(x)
        # 计算自编码器的损失
        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        # 计算判别器的损失
        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )
        # 记录重建损失
        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        # 更新自编码器日志字典，合并判别器日志
        log_dict_ae.update(log_dict_disc)
        # 记录合并后的日志字典
        self.log_dict(log_dict_ae)
        # 返回自编码器日志字典
        return log_dict_ae
    # 配置优化器，返回优化器和其他信息
    def configure_optimizers(self) -> Any:
        # 获取自动编码器的参数
        ae_params = self.get_autoencoder_params()
        # 获取鉴别器的参数
        disc_params = self.get_discriminator_params()

        # 从配置中实例化自动编码器的优化器，使用学习率的默认值
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        # 从配置中实例化鉴别器的优化器
        opt_disc = self.instantiate_optimizer_from_config(
            disc_params, self.learning_rate, self.optimizer_config
        )

        # 返回自动编码器和鉴别器的优化器，以及一个空列表
        return [opt_ae, opt_disc], []

    # 禁止梯度计算，避免在日志记录时影响计算图
    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        # 初始化一个字典用于存储日志
        log = dict()
        # 从批次中获取输入数据
        x = self.get_input(batch)
        # 通过模型进行前向传播，获取重构结果
        _, xrec, _ = self(x)
        # 将输入数据和重构结果存入日志字典
        log["inputs"] = x
        log["reconstructions"] = xrec
        # 在 EMA（指数移动平均）作用域内进行操作
        with self.ema_scope():
            # 获取 EMA 重构结果
            _, xrec_ema, _ = self(x)
            # 将 EMA 重构结果存入日志字典
            log["reconstructions_ema"] = xrec_ema
        # 返回日志字典
        return log
# 定义 AutoencodingEngineLegacy 类，继承自 AutoencodingEngine
class AutoencodingEngineLegacy(AutoencodingEngine):
    # 初始化方法，接受嵌入维度及其他可选参数
    def __init__(self, embed_dim: int, **kwargs):
        # 从 kwargs 中提取最大批处理大小，如果没有则为 None
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        # 从 kwargs 中提取 ddconfig 配置
        ddconfig = kwargs.pop("ddconfig")
        # 从 kwargs 中提取检查点路径，如果没有则为 None
        ckpt_path = kwargs.pop("ckpt_path", None)
        # 从 kwargs 中提取检查点引擎，如果没有则为 None
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        # 调用父类的初始化方法，配置编码器和解码器
        super().__init__(
            encoder_config={
                "target": "sgm.modules.diffusionmodules.model.Encoder",  # 编码器目标
                "params": ddconfig,  # 编码器参数
            },
            decoder_config={
                "target": "sgm.modules.diffusionmodules.model.Decoder",  # 解码器目标
                "params": ddconfig,  # 解码器参数
            },
            **kwargs,
        )
        # 定义量化卷积层，输入通道和输出通道根据配置计算
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],  # 输入通道数
            (1 + ddconfig["double_z"]) * embed_dim,  # 输出通道数
            1,  # 卷积核大小
        )
        # 定义后量化卷积层，将嵌入维度映射回 z_channels
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)  
        # 保存嵌入维度
        self.embed_dim = embed_dim

        # 应用检查点设置，使用默认方法获取路径和引擎
        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    # 获取自动编码器的参数
    def get_autoencoder_params(self) -> list:
        # 调用父类方法获取参数
        params = super().get_autoencoder_params()
        return params

    # 编码输入张量 x，返回量化后的表示 z
    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # 如果没有最大批处理大小
        if self.max_batch_size is None:
            # 直接编码并量化
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            # 获取输入样本数
            N = x.shape[0]
            # 获取批处理大小
            bs = self.max_batch_size
            # 计算总批次数
            n_batches = int(math.ceil(N / bs))
            z = list()  # 初始化存储编码结果的列表
            # 遍历每个批次
            for i_batch in range(n_batches):
                # 对当前批次进行编码
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)  # 量化编码结果
                z.append(z_batch)  # 添加到结果列表中
            # 将所有批次的结果连接成一个张量
            z = torch.cat(z, 0)

        # 应用正则化方法
        z, reg_log = self.regularization(z)
        # 如果需要返回正则化日志
        if return_reg_log:
            return z, reg_log  # 返回量化结果和日志
        return z  # 返回量化结果

    # 解码输入张量 z，返回重构结果
    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        # 如果没有最大批处理大小
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)  # 先经过后量化卷积
            dec = self.decoder(dec, **decoder_kwargs)  # 再经过解码器
        else:
            # 获取输入样本数
            N = z.shape[0]
            # 获取批处理大小
            bs = self.max_batch_size
            # 计算总批次数
            n_batches = int(math.ceil(N / bs))
            dec = list()  # 初始化存储解码结果的列表
            # 遍历每个批次
            for i_batch in range(n_batches):
                # 对当前批次进行后量化处理
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)  # 解码
                dec.append(dec_batch)  # 添加到结果列表中
            # 将所有批次的结果连接成一个张量
            dec = torch.cat(dec, 0)

        return dec  # 返回解码结果
    

# 定义 AutoencoderKL 类，继承自 AutoencodingEngine
class AutoencoderKL(AutoencodingEngine):
    # 初始化方法，接受嵌入维度和其他可选参数
        def __init__(self, embed_dim: int, **kwargs):
            # 从 kwargs 中提取 ddconfig 配置
            ddconfig = kwargs.pop("ddconfig")
            # 从 kwargs 中提取检查点路径，如果不存在则为 None
            ckpt_path = kwargs.pop("ckpt_path", None)
            # 从 kwargs 中提取忽略的键，默认为空元组
            ignore_keys = kwargs.pop("ignore_keys", ())
            # 调用父类初始化，配置编码器、解码器和正则化器
            super().__init__(
                encoder_config={"target": "torch.nn.Identity"},
                decoder_config={"target": "torch.nn.Identity"},
                regularizer_config={"target": "torch.nn.Identity"},
                loss_config=kwargs.pop("lossconfig"),
                **kwargs,
            )
            # 确保 ddconfig 中的 double_z 为真
            assert ddconfig["double_z"]
            # 初始化编码器，传入 ddconfig 参数
            self.encoder = Encoder(**ddconfig)
            # 初始化解码器，传入 ddconfig 参数
            self.decoder = Decoder(**ddconfig)
            # 创建一个卷积层，将输入的通道数映射到嵌入维度的两倍
            self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
            # 创建一个卷积层，将嵌入维度映射回原始通道数
            self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            # 存储嵌入维度
            self.embed_dim = embed_dim
    
            # 如果检查点路径不为空，则从该路径初始化模型
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
        # 编码方法，将输入数据编码为后验分布
        def encode(self, x):
            # 确保当前模式为推理模式
            assert (
                not self.training
            ), f"{self.__class__.__name__} only supports inference currently"
            # 通过编码器处理输入数据
            h = self.encoder(x)
            # 通过量化卷积层处理编码结果
            moments = self.quant_conv(h)
            # 创建一个对角高斯分布，基于量化结果
            posterior = DiagonalGaussianDistribution(moments)
            # 返回后验分布
            return posterior
    
        # 解码方法，将后验分布样本解码为输出
        def decode(self, z, **decoder_kwargs):
            # 通过反量化卷积层处理输入样本
            z = self.post_quant_conv(z)
            # 通过解码器生成最终输出
            dec = self.decoder(z, **decoder_kwargs)
            # 返回解码结果
            return dec
# 定义一个名为 AutoencoderKLInferenceWrapper 的类，继承自 AutoencoderKL
class AutoencoderKLInferenceWrapper(AutoencoderKL):
    # 定义 encode 方法，接受参数 x
    def encode(self, x):
        # 调用父类的 encode 方法并返回其结果的样本
        return super().encode(x).sample()

# 定义一个名为 IdentityFirstStage 的类，继承自 AbstractAutoencoder
class IdentityFirstStage(AbstractAutoencoder):
    # 定义初始化方法，接受可变数量的参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

    # 定义 get_input 方法，接受参数 x，返回输入
    def get_input(self, x: Any) -> Any:
        return x

    # 定义 encode 方法，接受参数 x 和其他可变参数
    def encode(self, x: Any, *args, **kwargs) -> Any:
        # 返回输入 x
        return x

    # 定义 decode 方法，接受参数 x 和其他可变参数
    def decode(self, x: Any, *args, **kwargs) -> Any:
        # 返回输入 x
        return x


# 定义一个名为 AutoencoderKLModeOnly 的类，继承自 AutoencodingEngineLegacy
class AutoencoderKLModeOnly(AutoencodingEngineLegacy):
    # 定义初始化方法，接受关键字参数
    def __init__(self, **kwargs):
        # 如果 kwargs 中包含 'lossconfig' 键，则将其改名为 'loss_config'
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        # 调用父类的初始化方法，并传入 regularizer_config 和 kwargs
        super().__init__(
            regularizer_config={
                # 定义目标为 DiagonalGaussianRegularizer
                "target": (
                    "sgm.modules.autoencoding.regularizers"
                    ".DiagonalGaussianRegularizer"
                ),
                # 设置参数 sample 为 False
                "params": {"sample": False},
            },
            **kwargs,
        )
```