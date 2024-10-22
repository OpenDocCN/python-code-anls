# `.\cogvideo-finetune\sat\vae_modules\autoencoder.py`

```py
# 导入所需的标准库和第三方库
import logging  # 日志库，用于记录信息
import math  # 数学库，提供数学函数
import re  # 正则表达式库，用于字符串匹配
import random  # 随机数库，提供随机数生成
from abc import abstractmethod  # 导入抽象方法装饰器，用于定义抽象基类
from contextlib import contextmanager  # 上下文管理器，用于资源管理
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型注解，提供类型提示

# 导入额外的第三方库
import numpy as np  # 数值计算库，支持大规模数组和矩阵运算
import pytorch_lightning as pl  # PyTorch 的轻量级封装，简化模型训练过程
import torch  # PyTorch 的核心库，支持深度学习
import torch.distributed  # PyTorch 的分布式计算库
import torch.nn as nn  # PyTorch 的神经网络模块
from einops import rearrange  # einops 库，用于数组重排
from packaging import version  # 版本控制库，处理版本比较

# 从自定义模块导入必要的函数和类
from vae_modules.ema import LitEma  # 导入 EMA 类，用于指数移动平均
from sgm.util import (
    instantiate_from_config,  # 从配置中实例化对象
    get_obj_from_str,  # 从字符串获取对象
    default,  # 提供默认值功能
    is_context_parallel_initialized,  # 检查上下文并行是否初始化
    initialize_context_parallel,  # 初始化上下文并行
    get_context_parallel_group,  # 获取上下文并行组
    get_context_parallel_group_rank,  # 获取上下文并行组的排名
)
from vae_modules.cp_enc_dec import _conv_split, _conv_gather  # 导入卷积相关函数

# 创建一个日志记录器
logpy = logging.getLogger(__name__)  # 使用模块名称创建一个日志记录器


class AbstractAutoencoder(pl.LightningModule):
    """
    这是所有自动编码器的基类，包括图像自动编码器、带有判别器的图像自动编码器、
    unCLIP 模型等。因此，它非常通用，特定特性
    （例如判别器训练、编码、解码）必须在子类中实现。
    """

    def __init__(
        self,
        ema_decay: Union[None, float] = None,  # 指定 EMA 衰减率，默认 None
        monitor: Union[None, str] = None,  # 监控指标，默认 None
        input_key: str = "jpg",  # 输入数据的键，默认设置为 "jpg"
    ):
        super().__init__()  # 调用父类构造函数

        self.input_key = input_key  # 保存输入键
        self.use_ema = ema_decay is not None  # 判断是否使用 EMA
        if monitor is not None:  # 如果监控指标不为 None
            self.monitor = monitor  # 保存监控指标

        if self.use_ema:  # 如果使用 EMA
            self.model_ema = LitEma(self, decay=ema_decay)  # 初始化 EMA 对象
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")  # 记录 EMA 的数量

        if version.parse(torch.__version__) >= version.parse("2.0.0"):  # 检查 PyTorch 版本
            self.automatic_optimization = False  # 禁用自动优化

    # def apply_ckpt(self, ckpt: Union[None, str, dict]):
    #     if ckpt is None:
    #         return
    #     if isinstance(ckpt, str):
    #         ckpt = {
    #             "target": "sgm.modules.checkpoint.CheckpointEngine",
    #             "params": {"ckpt_path": ckpt},
    #         }
    #     engine = instantiate_from_config(ckpt)
    #     engine(self)

    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:  # 如果检查点为 None
            return  # 不进行任何操作
        self.init_from_ckpt(ckpt)  # 从检查点初始化

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]  # 从指定路径加载模型状态字典
        keys = list(sd.keys())  # 获取状态字典的所有键
        for k in keys:  # 遍历所有键
            for ik in ignore_keys:  # 遍历要忽略的键
                if k.startswith(ik):  # 如果键以忽略的键开头
                    print("Deleting key {} from state_dict.".format(k))  # 输出被删除的键
                    del sd[k]  # 从状态字典中删除该键
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)  # 加载状态字典，允许不严格匹配
        print("Missing keys: ", missing_keys)  # 输出缺失的键
        print("Unexpected keys: ", unexpected_keys)  # 输出意外的键
        print(f"Restored from {path}")  # 输出恢复模型的信息

    @abstractmethod
    def get_input(self, batch) -> Any:  # 抽象方法，获取输入数据
        raise NotImplementedError()  # 抛出未实现错误

    def on_train_batch_end(self, *args, **kwargs):
        # 在训练批次结束时调用的函数
        if self.use_ema:  # 如果使用 EMA
            self.model_ema(self)  # 更新 EMA
    # 上下文管理器，用于在 EMA 模式下管理模型权重
        @contextmanager
        def ema_scope(self, context=None):
            # 如果启用 EMA，存储当前模型参数
            if self.use_ema:
                self.model_ema.store(self.parameters())
                # 将 EMA 权重复制到当前模型
                self.model_ema.copy_to(self)
                # 如果提供上下文，则记录信息
                if context is not None:
                    logpy.info(f"{context}: Switched to EMA weights")
            # 生成器，允许在上下文内执行代码
            try:
                yield None
            finally:
                # 如果启用 EMA，恢复之前存储的模型参数
                if self.use_ema:
                    self.model_ema.restore(self.parameters())
                    # 如果提供上下文，则记录恢复信息
                    if context is not None:
                        logpy.info(f"{context}: Restored training weights")
    
        # 抽象方法，用于编码输入，必须在子类中实现
        @abstractmethod
        def encode(self, *args, **kwargs) -> torch.Tensor:
            # 抛出未实现错误，指明该方法需被子类实现
            raise NotImplementedError("encode()-method of abstract base class called")
    
        # 抽象方法，用于解码输入，必须在子类中实现
        @abstractmethod
        def decode(self, *args, **kwargs) -> torch.Tensor:
            # 抛出未实现错误，指明该方法需被子类实现
            raise NotImplementedError("decode()-method of abstract base class called")
    
        # 根据配置实例化优化器
        def instantiate_optimizer_from_config(self, params, lr, cfg):
            # 记录加载的优化器信息
            logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
            # 返回根据配置创建的优化器对象
            return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))
    
        # 抽象方法，用于配置优化器，必须在子类中实现
        def configure_optimizers(self) -> Any:
            # 抛出未实现错误，指明该方法需被子类实现
            raise NotImplementedError()
# 定义图像自编码器的基类，例如 VQGAN 或 AutoencoderKL
class AutoencodingEngine(AbstractAutoencoder):
    """
    Base class for all image autoencoders that we train, like VQGAN or AutoencoderKL
    (we also restore them explicitly as special cases for legacy reasons).
    Regularizations such as KL or VQ are moved to the regularizer class.
    """

    # 初始化方法，接收多个配置参数
    def __init__(
        self,
        *args,
        encoder_config: Dict,  # 编码器配置字典
        decoder_config: Dict,  # 解码器配置字典
        loss_config: Dict,  # 损失函数配置字典
        regularizer_config: Dict,  # 正则化配置字典
        optimizer_config: Union[Dict, None] = None,  # 优化器配置字典或 None
        lr_g_factor: float = 1.0,  # 学习率增益因子
        trainable_ae_params: Optional[List[List[str]]] = None,  # 可训练的自编码器参数
        ae_optimizer_args: Optional[List[dict]] = None,  # 自编码器优化器参数
        trainable_disc_params: Optional[List[List[str]]] = None,  # 可训练的判别器参数
        disc_optimizer_args: Optional[List[dict]] = None,  # 判别器优化器参数
        disc_start_iter: int = 0,  # 判别器开始迭代的轮数
        diff_boost_factor: float = 3.0,  # 差异提升因子
        ckpt_engine: Union[None, str, dict] = None,  # 检查点引擎
        ckpt_path: Optional[str] = None,  # 检查点路径
        additional_decode_keys: Optional[List[str]] = None,  # 附加解码键
        **kwargs,  # 其他可选参数
    ):
        # 调用父类构造函数
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False  # 设置为 False，禁用自动优化

        # 根据配置实例化编码器
        self.encoder = instantiate_from_config(encoder_config)
        # 根据配置实例化解码器
        self.decoder = instantiate_from_config(decoder_config)
        # 根据配置实例化损失函数
        self.loss = instantiate_from_config(loss_config)
        # 根据配置实例化正则化器
        self.regularization = instantiate_from_config(regularizer_config)
        # 设置优化器配置，默认为 Adam 优化器
        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.Adam"})
        # 设置差异提升因子
        self.diff_boost_factor = diff_boost_factor
        # 设置判别器开始迭代的轮数
        self.disc_start_iter = disc_start_iter
        # 设置学习率增益因子
        self.lr_g_factor = lr_g_factor
        # 设置可训练的自编码器参数
        self.trainable_ae_params = trainable_ae_params
        # 如果有可训练的自编码器参数
        if self.trainable_ae_params is not None:
            # 设置自编码器优化器参数，默认每个参数一个空字典
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            # 确保优化器参数与可训练参数数量匹配
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            # 默认情况下，设置为一个空字典以保持类型一致
            self.ae_optimizer_args = [{}]  # makes type consitent

        # 设置可训练的判别器参数
        self.trainable_disc_params = trainable_disc_params
        # 如果有可训练的判别器参数
        if self.trainable_disc_params is not None:
            # 设置判别器优化器参数，默认每个参数一个空字典
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            # 确保优化器参数与可训练参数数量匹配
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            # 默认情况下，设置为一个空字典以保持类型一致
            self.disc_optimizer_args = [{}]  # makes type consitent

        # 如果指定了检查点路径
        if ckpt_path is not None:
            # 确保不同时设置检查点引擎和路径
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            # 记录警告，说明检查点路径已弃用
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")
        # 应用检查点配置
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        # 设置附加解码键的集合
        self.additional_decode_keys = set(default(additional_decode_keys, []))
    # 从给定的 batch 字典中获取输入张量
    def get_input(self, batch: Dict) -> torch.Tensor:
        # 假设统一的数据格式，数据加载器返回一个字典。
        # 图像张量应缩放到 -1 ... 1 并采用通道优先格式
        # （例如，bchw 而不是 bhwc）
        return batch[self.input_key]  # 返回与输入键对应的张量

    # 获取自动编码器的可训练参数
    def get_autoencoder_params(self) -> list:
        params = []  # 初始化参数列表
        # 如果损失对象有获取可训练自动编码器参数的方法
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            # 将这些参数添加到参数列表中
            params += list(self.loss.get_trainable_autoencoder_parameters())
        # 如果正则化对象有获取可训练参数的方法
        if hasattr(self.regularization, "get_trainable_parameters"):
            # 将这些参数添加到参数列表中
            params += list(self.regularization.get_trainable_parameters())
        # 将编码器的参数添加到参数列表中
        params = params + list(self.encoder.parameters())
        # 将解码器的参数添加到参数列表中
        params = params + list(self.decoder.parameters())
        return params  # 返回所有可训练参数的列表

    # 获取鉴别器的可训练参数
    def get_discriminator_params(self) -> list:
        # 如果损失对象有获取可训练参数的方法
        if hasattr(self.loss, "get_trainable_parameters"):
            # 获取这些参数，例如，鉴别器的参数
            params = list(self.loss.get_trainable_parameters())
        else:
            params = []  # 如果没有，则返回空列表
        return params  # 返回鉴别器的可训练参数列表

    # 获取解码器的最后一层
    def get_last_layer(self):
        return self.decoder.get_last_layer()  # 返回解码器的最后一层

    # 编码输入张量
    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,  # 是否返回正则化日志
        unregularized: bool = False,  # 是否返回未正则化的编码
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x)  # 使用编码器对输入 x 进行编码
        if unregularized:
            return z, dict()  # 如果未正则化，返回编码结果和空字典
        z, reg_log = self.regularization(z)  # 对编码结果进行正则化
        if return_reg_log:
            return z, reg_log  # 如果请求，返回编码结果和正则化日志
        return z  # 返回编码结果

    # 解码输入的编码
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)  # 使用解码器对编码 z 进行解码
        return x  # 返回解码结果

    # 前向传播过程
    def forward(self, x: torch.Tensor, **additional_decode_kwargs) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)  # 编码输入并获取正则化日志
        dec = self.decode(z, **additional_decode_kwargs)  # 解码编码结果
        return z, dec, reg_log  # 返回编码结果、解码结果和正则化日志
    # 定义内部训练步骤方法，接收批次数据、批次索引和优化器索引，返回张量
        def inner_training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0) -> torch.Tensor:
            # 获取输入数据
            x = self.get_input(batch)
            # 创建包含附加解码参数的字典，来自于批次数据
            additional_decode_kwargs = {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
            # 调用模型进行前向传播，得到潜变量z、重建数据xrec和正则化日志
            z, xrec, regularization_log = self(x, **additional_decode_kwargs)
            # 如果损失对象具有前向键属性
            if hasattr(self.loss, "forward_keys"):
                # 创建额外信息字典，包含潜变量和训练相关信息
                extra_info = {
                    "z": z,
                    "optimizer_idx": optimizer_idx,
                    "global_step": self.global_step,
                    "last_layer": self.get_last_layer(),
                    "split": "train",
                    "regularization_log": regularization_log,
                    "autoencoder": self,
                }
                # 过滤额外信息，保留损失函数所需的键
                extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
            else:
                # 如果没有前向键，初始化为空字典
                extra_info = dict()
    
            # 如果优化器索引为0，表示处理自动编码器
            if optimizer_idx == 0:
                # 计算重建损失
                out_loss = self.loss(x, xrec, **extra_info)
                # 如果损失是元组，拆分为自编码器损失和日志字典
                if isinstance(out_loss, tuple):
                    aeloss, log_dict_ae = out_loss
                else:
                    # 否则直接赋值自编码器损失，并初始化日志字典
                    aeloss = out_loss
                    log_dict_ae = {"train/loss/rec": aeloss.detach()}
    
                # 记录自编码器损失的日志字典
                self.log_dict(
                    log_dict_ae,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                )
                # 记录平均损失到进度条
                self.log(
                    "loss",
                    aeloss.mean().detach(),
                    prog_bar=True,
                    logger=False,
                    on_epoch=False,
                    on_step=True,
                )
                # 返回自编码器损失
                return aeloss
            # 如果优化器索引为1，表示处理判别器
            elif optimizer_idx == 1:
                # 计算判别器损失
                discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
                # 判别器总是需要返回一个元组
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                # 返回判别器损失
                return discloss
            else:
                # 如果优化器索引未知，抛出未实现错误
                raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")
    
        # 定义训练步骤方法，接收批次数据和批次索引
        def training_step(self, batch: dict, batch_idx: int):
            # 获取当前优化器列表
            opts = self.optimizers()
            # 如果优化器不是列表，转化为列表
            if not isinstance(opts, list):
                # 非对抗性情况，将优化器放入列表
                opts = [opts]
            # 根据批次索引计算当前优化器索引
            optimizer_idx = batch_idx % len(opts)
            # 如果当前全局步骤小于判别器开始迭代步骤，选择第一个优化器
            if self.global_step < self.disc_start_iter:
                optimizer_idx = 0
            # 获取当前优化器
            opt = opts[optimizer_idx]
            # 清零优化器的梯度
            opt.zero_grad()
            # 在优化器的模型上下文中
            with opt.toggle_model():
                # 调用内部训练步骤，计算损失
                loss = self.inner_training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
                # 手动进行反向传播
                self.manual_backward(loss)
            # 更新优化器的参数
            opt.step()
    
        # 定义验证步骤方法，接收批次数据和批次索引，返回日志字典
        def validation_step(self, batch: dict, batch_idx: int) -> Dict:
            # 调用内部验证步骤，获取日志字典
            log_dict = self._validation_step(batch, batch_idx)
            # 在 EMA 上下文中进行验证步骤
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
                # 更新日志字典，合并 EMA 结果
                log_dict.update(log_dict_ema)
            # 返回最终的日志字典
            return log_dict
    # 验证步骤，处理输入批次并计算损失
    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        # 从输入批次中获取输入数据
        x = self.get_input(batch)

        # 使用模型前向传播，获取隐变量、重构输出和正则化日志
        z, xrec, regularization_log = self(x)
        # 检查损失对象是否有前向键
        if hasattr(self.loss, "forward_keys"):
            # 创建包含额外信息的字典
            extra_info = {
                "z": z,  # 隐变量
                "optimizer_idx": 0,  # 优化器索引
                "global_step": self.global_step,  # 全局步数
                "last_layer": self.get_last_layer(),  # 获取最后一层
                "split": "val" + postfix,  # 验证集标记
                "regularization_log": regularization_log,  # 正则化日志
                "autoencoder": self,  # 自编码器实例
            }
            # 根据前向键过滤额外信息
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            # 如果没有前向键，则创建空字典
            extra_info = dict()
        # 计算损失
        out_loss = self.loss(x, xrec, **extra_info)
        # 检查损失是否为元组
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss  # 分别获取损失和日志字典
        else:
            # 简单损失函数的情况
            aeloss = out_loss  # 直接获取损失值
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}  # 创建日志字典
        full_log_dict = log_dict_ae  # 初始化完整日志字典

        # 如果额外信息中有优化器索引
        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1  # 更新优化器索引
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)  # 计算鉴别损失
            full_log_dict.update(log_dict_disc)  # 更新完整日志字典
        # 记录重构损失
        self.log(
            f"val{postfix}/loss/rec",  # 日志标签
            log_dict_ae[f"val{postfix}/loss/rec"],  # 重构损失值
            sync_dist=True,  # 同步分布式日志
        )
        # 记录完整日志字典
        self.log_dict(full_log_dict, sync_dist=True)
        # 返回完整日志字典
        return full_log_dict

    # 获取参数组
    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []  # 初始化参数组列表
        num_params = 0  # 初始化参数计数
        # 遍历参数名称和优化器参数
        for names, args in zip(parameter_names, optimizer_args):
            params = []  # 存储匹配的参数
            # 遍历每个参数模式
            for pattern_ in names:
                pattern_params = []  # 存储符合模式的参数
                pattern = re.compile(pattern_)  # 编译正则表达式
                # 遍历命名参数
                for p_name, param in self.named_parameters():
                    # 检查参数名称是否匹配模式
                    if re.match(pattern, p_name):
                        pattern_params.append(param)  # 添加匹配的参数
                        num_params += param.numel()  # 更新参数计数
                # 如果没有找到匹配参数，发出警告
                if len(pattern_params) == 0:
                    logpy.warn(f"Did not find parameters for pattern {pattern_}")
                # 扩展参数列表
                params.extend(pattern_params)
            # 将参数组添加到列表中
            groups.append({"params": params, **args})
        # 返回参数组和参数总数
        return groups, num_params
    # 配置优化器并返回优化器列表
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        # 如果可训练的自编码器参数为空
        if self.trainable_ae_params is None:
            # 获取自编码器的参数
            ae_params = self.get_autoencoder_params()
        else:
            # 获取自编码器参数组及其数量
            ae_params, num_ae_params = self.get_param_groups(self.trainable_ae_params, self.ae_optimizer_args)
            # 记录可训练自编码器参数的数量
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        # 如果可训练的判别器参数为空
        if self.trainable_disc_params is None:
            # 获取判别器的参数
            disc_params = self.get_discriminator_params()
        else:
            # 获取判别器参数组及其数量
            disc_params, num_disc_params = self.get_param_groups(self.trainable_disc_params, self.disc_optimizer_args)
            # 记录可训练判别器参数的数量
            logpy.info(f"Number of trainable discriminator parameters: {num_disc_params:,}")
        # 从配置中实例化自编码器优化器
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        # 创建优化器列表，首先添加自编码器优化器
        opts = [opt_ae]
        # 如果判别器参数列表不为空
        if len(disc_params) > 0:
            # 从配置中实例化判别器优化器
            opt_disc = self.instantiate_optimizer_from_config(disc_params, self.learning_rate, self.optimizer_config)
            # 将判别器优化器添加到优化器列表中
            opts.append(opt_disc)

        # 返回所有配置好的优化器
        return opts

    # 不计算梯度的图像记录函数
    @torch.no_grad()
    def log_images(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
        # 创建一个字典用于存储日志信息
        log = dict()
        # 创建字典用于存储额外的解码参数
        additional_decode_kwargs = {}
        # 获取输入数据
        x = self.get_input(batch)
        # 更新额外解码参数字典，包含在批次中的附加解码键
        additional_decode_kwargs.update({key: batch[key] for key in self.additional_decode_keys.intersection(batch)})

        # 通过自编码器模型处理输入，获得重构结果
        _, xrec, _ = self(x, **additional_decode_kwargs)
        # 将输入和重构结果存入日志
        log["inputs"] = x
        log["reconstructions"] = xrec
        # 计算输入和重构之间的差异，并进行归一化
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        # 将差异限制在[0, 1]范围内
        diff.clamp_(0, 1.0)
        # 存储归一化后的差异
        log["diff"] = 2.0 * diff - 1.0
        # diff_boost 显示小错误的位置，通过增强它们的亮度
        log["diff_boost"] = 2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        # 如果损失对象有 log_images 方法，更新日志
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, xrec))
        # 在 EMA 范围内进行操作
        with self.ema_scope():
            # 通过自编码器模型处理输入，获得 EMA 重构结果
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            # 将 EMA 重构结果存入日志
            log["reconstructions_ema"] = xrec_ema
            # 计算 EMA 重构与输入之间的差异
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            # 将 EMA 差异限制在[0, 1]范围内
            diff_ema.clamp_(0, 1.0)
            # 存储 EMA 归一化后的差异
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            # 存储 EMA 的差异增强
            log["diff_boost_ema"] = 2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
        # 如果有附加日志参数
        if additional_log_kwargs:
            # 更新解码参数字典
            additional_decode_kwargs.update(additional_log_kwargs)
            # 通过自编码器模型处理输入，获得附加重构结果
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            # 创建日志字符串
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            # 将附加重构结果存入日志
            log[log_str] = xrec_add
        # 返回日志字典
        return log
# 定义一个遗留的自编码引擎类，继承自 AutoencodingEngine
class AutoencodingEngineLegacy(AutoencodingEngine):
    # 初始化方法，接受嵌入维度和其他可选参数
    def __init__(self, embed_dim: int, **kwargs):
        # 从参数中提取最大批量大小，如果没有则为 None
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        # 从参数中提取 ddconfig 配置
        ddconfig = kwargs.pop("ddconfig")
        # 从参数中提取检查点路径，如果没有则为 None
        ckpt_path = kwargs.pop("ckpt_path", None)
        # 从参数中提取检查点引擎，如果没有则为 None
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        # 调用父类构造函数，初始化编码器和解码器配置
        super().__init__(
            encoder_config={
                "target": "sgm.modules.diffusionmodules.model.Encoder",
                "params": ddconfig,
            },
            decoder_config={
                "target": "sgm.modules.diffusionmodules.model.Decoder",
                "params": ddconfig,
            },
            **kwargs,
        )
        # 创建量化卷积层，输入和输出通道依赖于 ddconfig
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        # 创建后量化卷积层，将嵌入维度映射回 z_channels
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # 保存嵌入维度
        self.embed_dim = embed_dim

        # 应用检查点，传入检查点路径和引擎
        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    # 获取自编码器参数的方法
    def get_autoencoder_params(self) -> list:
        # 调用父类方法获取参数
        params = super().get_autoencoder_params()
        return params

    # 编码方法，接收输入张量并返回编码结果
    def encode(self, x: torch.Tensor, return_reg_log: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # 如果没有最大批量大小，则直接编码
        if self.max_batch_size is None:
            z = self.encoder(x)
            # 应用量化卷积
            z = self.quant_conv(z)
        else:
            # 获取输入的批量大小
            N = x.shape[0]
            bs = self.max_batch_size
            # 计算需要的批次数
            n_batches = int(math.ceil(N / bs))
            z = list()
            # 按批处理输入
            for i_batch in range(n_batches):
                # 编码当前批次
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                # 应用量化卷积
                z_batch = self.quant_conv(z_batch)
                # 收集编码结果
                z.append(z_batch)
            # 将所有批次的编码结果连接起来
            z = torch.cat(z, 0)

        # 应用正则化
        z, reg_log = self.regularization(z)
        # 如果需要返回正则化日志，则返回
        if return_reg_log:
            return z, reg_log
        # 否则只返回编码结果
        return z

    # 解码方法，接收编码结果并返回解码后的张量
    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        # 如果没有最大批量大小，则直接解码
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            # 使用解码器进行解码
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            # 获取输入的批量大小
            N = z.shape[0]
            bs = self.max_batch_size
            # 计算需要的批次数
            n_batches = int(math.ceil(N / bs))
            dec = list()
            # 按批处理输入
            for i_batch in range(n_batches):
                # 应用后量化卷积
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                # 使用解码器进行解码
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                # 收集解码结果
                dec.append(dec_batch)
            # 将所有批次的解码结果连接起来
            dec = torch.cat(dec, 0)

        # 返回解码后的结果
        return dec


# 定义一个自编码器类，继承自 AutoencodingEngineLegacy
class AutoencoderKL(AutoencodingEngineLegacy):
    # 初始化方法
    def __init__(self, **kwargs):
        # 如果提供了 lossconfig，则将其重命名为 loss_config
        if "lossconfig" in kwargs:
            kwargs["loss_config"] = kwargs.pop("lossconfig")
        # 调用父类构造函数，初始化正则化器配置
        super().__init__(
            regularizer_config={"target": ("sgm.modules.autoencoding.regularizers" ".DiagonalGaussianRegularizer")},
            **kwargs,
        )


# 定义一个身份自编码器类，继承自 AbstractAutoencoder
class IdentityFirstStage(AbstractAutoencoder):
    # 初始化方法，用于创建类的实例
        def __init__(self, *args, **kwargs):
            # 调用父类的初始化方法，传递参数
            super().__init__(*args, **kwargs)
    
    # 获取输入的方法，接收任意类型的输入
        def get_input(self, x: Any) -> Any:
            # 直接返回传入的输入
            return x
    
    # 编码方法，接收任意类型的输入，并可以接收额外参数
        def encode(self, x: Any, *args, **kwargs) -> Any:
            # 直接返回传入的输入，未做任何处理
            return x
    
    # 解码方法，接收任意类型的输入，并可以接收额外参数
        def decode(self, x: Any, *args, **kwargs) -> Any:
            # 直接返回传入的输入，未做任何处理
            return x
# 定义视频自动编码引擎类，继承自自动编码引擎类
class VideoAutoencodingEngine(AutoencodingEngine):
    # 初始化函数，定义构造参数
    def __init__(
        # 检查点路径，可选，默认为 None
        ckpt_path: Union[None, str] = None,
        # 要忽略的键，可选，默认为空元组
        ignore_keys: Union[Tuple, list] = (),
        # 图像和视频权重列表，默认值为 [1, 1]
        image_video_weights=[1, 1],
        # 是否仅训练解码器，默认为 False
        only_train_decoder=False,
        # 上下文并行大小，默认为 0
        context_parallel_size=0,
        # 其他关键字参数
        **kwargs,
    ):
        # 调用父类构造函数
        super().__init__(**kwargs)
        # 保存上下文并行大小到实例变量
        self.context_parallel_size = context_parallel_size
        # 如果检查点路径不为 None，初始化检查点
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    # 日志视频的方法，接受批量数据和额外日志参数
    def log_videos(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
        # 调用日志图像的方法并返回结果
        return self.log_images(batch, additional_log_kwargs, **kwargs)

    # 获取输入数据的方法
    def get_input(self, batch: dict) -> torch.Tensor:
        # 如果上下文并行大小大于 0
        if self.context_parallel_size > 0:
            # 检查上下文并行是否已初始化
            if not is_context_parallel_initialized():
                # 初始化上下文并行
                initialize_context_parallel(self.context_parallel_size)

            # 从批量中提取输入键对应的数据
            batch = batch[self.input_key]

            # 获取全局源排名并进行分布式广播
            global_src_rank = get_context_parallel_group_rank() * self.context_parallel_size
            torch.distributed.broadcast(batch, src=global_src_rank, group=get_context_parallel_group())

            # 在指定维度上进行卷积分割
            batch = _conv_split(batch, dim=2, kernel_size=1)
            # 返回处理后的批量数据
            return batch

        # 如果上下文并行大小为 0，直接返回输入键对应的数据
        return batch[self.input_key]

    # 应用检查点的方法
    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        # 如果检查点为 None，直接返回
        if ckpt is None:
            return
        # 初始化检查点
        self.init_from_ckpt(ckpt)

    # 从检查点初始化状态字典的方法
    def init_from_ckpt(self, path, ignore_keys=list()):
        # 加载指定路径的状态字典
        sd = torch.load(path, map_location="cpu")["state_dict"]
        # 获取状态字典的键列表
        keys = list(sd.keys())
        # 遍历每个键，检查是否需要忽略
        for k in keys:
            for ik in ignore_keys:
                # 如果键以忽略键开头，则删除该键
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        # 加载状态字典，并返回缺失和意外的键
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        # 打印缺失和意外的键
        print("Missing keys: ", missing_keys)
        print("Unexpected keys: ", unexpected_keys)
        # 打印恢复信息
        print(f"Restored from {path}")

# 定义视频自动编码推理包装类，继承自视频自动编码引擎类
class VideoAutoencoderInferenceWrapper(VideoAutoencodingEngine):
    # 初始化函数，定义构造参数
    def __init__(
        # 上下文并行大小，默认为 0
        cp_size=0,
        # 其他位置参数
        *args,
        # 其他关键字参数
        **kwargs,
    ):
        # 保存上下文并行大小到实例变量
        self.cp_size = cp_size
        # 调用父类构造函数
        return super().__init__(*args, **kwargs)

    # 编码方法，接收输入张量和一些可选参数
    def encode(
        self,
        # 输入张量
        x: torch.Tensor,
        # 是否返回正则日志，默认为 False
        return_reg_log: bool = False,
        # 是否使用非正则化，默认为 False
        unregularized: bool = False,
        # 是否输入上下文并行，默认为 False
        input_cp: bool = False,
        # 是否输出上下文并行，默认为 False
        output_cp: bool = False,
    # 定义返回类型为 Tensor 或包含 Tensor 和字典的元组
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
            # 如果 cp_size 大于 0 且未传入输入的 cp 标志
            if self.cp_size > 0 and not input_cp:
                # 如果尚未初始化上下文并行
                if not is_context_parallel_initialized:
                    # 初始化上下文并行，指定 cp_size
                    initialize_context_parallel(self.cp_size)
    
                # 获取全局源排名，并根据 cp_size 计算
                global_src_rank = get_context_parallel_group_rank() * self.cp_size
                # 从全局源排名广播张量 x
                torch.distributed.broadcast(x, src=global_src_rank, group=get_context_parallel_group())
    
                # 对张量 x 进行分割操作，沿着维度 2 进行卷积
                x = _conv_split(x, dim=2, kernel_size=1)
    
            # 如果需要返回正则化日志
            if return_reg_log:
                # 调用父类的编码方法，返回编码结果 z 和正则化日志 reg_log
                z, reg_log = super().encode(x, return_reg_log, unregularized)
            else:
                # 只返回编码结果 z
                z = super().encode(x, return_reg_log, unregularized)
    
            # 如果 cp_size 大于 0 且未传入输出的 cp 标志
            if self.cp_size > 0 and not output_cp:
                # 对 z 进行汇聚操作，沿着维度 2 进行卷积
                z = _conv_gather(z, dim=2, kernel_size=1)
    
            # 如果需要返回正则化日志，则返回 z 和 reg_log
            if return_reg_log:
                return z, reg_log
            # 否则只返回 z
            return z
    
        # 定义解码方法
        def decode(
            self,
            z: torch.Tensor,
            input_cp: bool = False,
            output_cp: bool = False,
            split_kernel_size: int = 1,
            **kwargs,
        ):
            # 如果 cp_size 大于 0 且未传入输入的 cp 标志
            if self.cp_size > 0 and not input_cp:
                # 如果尚未初始化上下文并行
                if not is_context_parallel_initialized:
                    # 初始化上下文并行，指定 cp_size
                    initialize_context_parallel(self.cp_size)
    
                # 获取全局源排名，并根据 cp_size 计算
                global_src_rank = get_context_parallel_group_rank() * self.cp_size
                # 从全局源排名广播张量 z
                torch.distributed.broadcast(z, src=global_src_rank, group=get_context_parallel_group())
    
                # 对张量 z 进行分割操作，沿着维度 2 进行卷积
                z = _conv_split(z, dim=2, kernel_size=split_kernel_size)
    
            # 调用父类的解码方法，返回解码结果 x
            x = super().decode(z, **kwargs)
    
            # 如果 cp_size 大于 0 且未传入输出的 cp 标志
            if self.cp_size > 0 and not output_cp:
                # 对 x 进行汇聚操作，沿着维度 2 进行卷积
                x = _conv_gather(x, dim=2, kernel_size=split_kernel_size)
    
            # 返回解码结果 x
            return x
    
        # 定义前向传播方法
        def forward(
            self,
            x: torch.Tensor,
            input_cp: bool = False,
            latent_cp: bool = False,
            output_cp: bool = False,
            **additional_decode_kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
            # 调用编码方法，返回编码结果 z 和正则化日志 reg_log
            z, reg_log = self.encode(x, return_reg_log=True, input_cp=input_cp, output_cp=latent_cp)
            # 调用解码方法，返回解码结果 dec
            dec = self.decode(z, input_cp=latent_cp, output_cp=output_cp, **additional_decode_kwargs)
            # 返回编码结果 z、解码结果 dec 和正则化日志 reg_log
            return z, dec, reg_log
```