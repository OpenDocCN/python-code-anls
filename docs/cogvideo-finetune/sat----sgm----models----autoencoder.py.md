# `.\cogvideo-finetune\sat\sgm\models\autoencoder.py`

```py
# 导入标准库的日志模块
import logging
# 导入数学库
import math
# 导入正则表达式库
import re
# 导入随机数生成库
import random
# 从 abc 模块导入抽象方法装饰器
from abc import abstractmethod
# 从上下文管理库导入上下文管理器装饰器
from contextlib import contextmanager
# 导入类型提示相关的类型
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch Lightning 库
import pytorch_lightning as pl
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的分布式模块
import torch.distributed
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 einops 库导入重排函数
from einops import rearrange
# 导入版本管理库
from packaging import version

# 从自定义模块中导入所需的类和函数
from ..modules.autoencoding.regularizers import AbstractRegularizer
from ..modules.ema import LitEma
from ..util import (
    default,  # 默认值函数
    get_nested_attribute,  # 获取嵌套属性函数
    get_obj_from_str,  # 从字符串获取对象函数
    instantiate_from_config,  # 从配置实例化对象函数
    initialize_context_parallel,  # 初始化上下文并行函数
    get_context_parallel_group,  # 获取上下文并行组函数
    get_context_parallel_group_rank,  # 获取上下文并行组的排名函数
    is_context_parallel_initialized,  # 检查上下文并行是否已初始化函数
)
from ..modules.cp_enc_dec import _conv_split, _conv_gather  # 导入卷积拆分和聚合函数

# 创建日志记录器
logpy = logging.getLogger(__name__)

# 定义抽象自编码器类，继承自 PyTorch Lightning 模块
class AbstractAutoencoder(pl.LightningModule):
    """
    这是所有自编码器的基类，包括图像自编码器、带鉴别器的图像自编码器、unCLIP 模型等。
    因此，它是相当通用的，具体特性（例如，鉴别器训练、编码、解码）必须在子类中实现。
    """

    # 初始化方法，设置自编码器的属性
    def __init__(
        self,
        ema_decay: Union[None, float] = None,  # 指定 EMA 衰减参数
        monitor: Union[None, str] = None,  # 指定监控指标
        input_key: str = "jpg",  # 输入数据的键，默认为 "jpg"
    ):
        super().__init__()  # 调用父类的初始化方法

        self.input_key = input_key  # 存储输入键
        self.use_ema = ema_decay is not None  # 检查是否使用 EMA
        if monitor is not None:  # 如果监控指标不为 None
            self.monitor = monitor  # 存储监控指标

        if self.use_ema:  # 如果使用 EMA
            self.model_ema = LitEma(self, decay=ema_decay)  # 创建 EMA 实例
            logpy.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")  # 记录 EMA 缓冲区的数量

        # 检查 PyTorch 版本是否大于或等于 2.0.0
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False  # 禁用自动优化

    # 应用检查点的方法
    def apply_ckpt(self, ckpt: Union[None, str, dict]):
        if ckpt is None:  # 如果检查点为 None
            return  # 直接返回
        if isinstance(ckpt, str):  # 如果检查点是字符串
            ckpt = {
                "target": "sgm.modules.checkpoint.CheckpointEngine",  # 设置目标为检查点引擎
                "params": {"ckpt_path": ckpt},  # 设置检查点路径
            }
        engine = instantiate_from_config(ckpt)  # 根据配置实例化引擎
        engine(self)  # 将自编码器传入引擎

    @abstractmethod  # 声明此方法为抽象方法
    def get_input(self, batch) -> Any:  # 获取输入的方法
        raise NotImplementedError()  # 抛出未实现错误

    # 训练批次结束后的回调方法
    def on_train_batch_end(self, *args, **kwargs):
        # 用于 EMA 计算
        if self.use_ema:  # 如果使用 EMA
            self.model_ema(self)  # 更新 EMA

    # 定义 EMA 上下文管理器
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:  # 如果使用 EMA
            self.model_ema.store(self.parameters())  # 存储当前参数
            self.model_ema.copy_to(self)  # 复制 EMA 权重到模型
            if context is not None:  # 如果有上下文信息
                logpy.info(f"{context}: Switched to EMA weights")  # 记录切换到 EMA 权重的信息
        try:
            yield None  # 允许在上下文中执行
        finally:
            if self.use_ema:  # 如果使用 EMA
                self.model_ema.restore(self.parameters())  # 恢复模型参数
                if context is not None:  # 如果有上下文信息
                    logpy.info(f"{context}: Restored training weights")  # 记录恢复训练权重的信息

    @abstractmethod  # 声明此方法为抽象方法
    # 定义一个编码方法，接受可变参数，返回一个张量
        def encode(self, *args, **kwargs) -> torch.Tensor:
            # 抛出未实现错误，指示这是一个抽象基类的方法
            raise NotImplementedError("encode()-method of abstract base class called")
    
        # 定义一个解码方法，接受可变参数，返回一个张量
        @abstractmethod
        def decode(self, *args, **kwargs) -> torch.Tensor:
            # 抛出未实现错误，指示这是一个抽象基类的方法
            raise NotImplementedError("decode()-method of abstract base class called")
    
        # 根据配置实例化优化器，接受参数列表、学习率和配置字典
        def instantiate_optimizer_from_config(self, params, lr, cfg):
            # 记录加载优化器的目标信息
            logpy.info(f"loading >>> {cfg['target']} <<< optimizer from config")
            # 从配置中获取优化器对象，并返回实例化的优化器
            return get_obj_from_str(cfg["target"])(params, lr=lr, **cfg.get("params", dict()))
    
        # 配置优化器，返回任意类型
        def configure_optimizers(self) -> Any:
            # 抛出未实现错误，指示这是一个抽象基类的方法
            raise NotImplementedError()
# 定义图像自编码器的基类，例如 VQGAN 或 AutoencoderKL
class AutoencodingEngine(AbstractAutoencoder):
    """
    所有图像自编码器的基类，我们训练的如 VQGAN 或 AutoencoderKL
    （出于遗留原因，我们也显式恢复它们作为特例）。
    正则化如 KL 或 VQ 被移动到正则化器类中。
    """

    # 初始化自编码器
    def __init__(
        self,
        *args,
        encoder_config: Dict,  # 编码器配置字典
        decoder_config: Dict,  # 解码器配置字典
        loss_config: Dict,  # 损失函数配置字典
        regularizer_config: Dict,  # 正则化器配置字典
        optimizer_config: Union[Dict, None] = None,  # 优化器配置字典，可选
        lr_g_factor: float = 1.0,  # 学习率缩放因子
        trainable_ae_params: Optional[List[List[str]]] = None,  # 可训练的自编码器参数
        ae_optimizer_args: Optional[List[dict]] = None,  # 自编码器优化器参数
        trainable_disc_params: Optional[List[List[str]]] = None,  # 可训练的判别器参数
        disc_optimizer_args: Optional[List[dict]] = None,  # 判别器优化器参数
        disc_start_iter: int = 0,  # 判别器开始迭代的初始迭代次数
        diff_boost_factor: float = 3.0,  # 差异提升因子
        ckpt_engine: Union[None, str, dict] = None,  # 检查点引擎配置
        ckpt_path: Optional[str] = None,  # 检查点路径
        additional_decode_keys: Optional[List[str]] = None,  # 额外解码键
        **kwargs,  # 其他参数
    ):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.automatic_optimization = False  # 禁用自动优化，适用于 PyTorch Lightning

        # 根据配置实例化编码器
        self.encoder: torch.nn.Module = instantiate_from_config(encoder_config)
        # 根据配置实例化解码器
        self.decoder: torch.nn.Module = instantiate_from_config(decoder_config)
        # 根据配置实例化损失函数
        self.loss: torch.nn.Module = instantiate_from_config(loss_config)
        # 根据配置实例化正则化器
        self.regularization: AbstractRegularizer = instantiate_from_config(regularizer_config)
        # 设置优化器配置，默认为 Adam
        self.optimizer_config = default(optimizer_config, {"target": "torch.optim.Adam"})
        # 设置差异提升因子
        self.diff_boost_factor = diff_boost_factor
        # 设置判别器开始迭代的初始值
        self.disc_start_iter = disc_start_iter
        # 设置学习率缩放因子
        self.lr_g_factor = lr_g_factor
        # 存储可训练的自编码器参数
        self.trainable_ae_params = trainable_ae_params
        if self.trainable_ae_params is not None:
            # 设置自编码器优化器参数，默认为空字典
            self.ae_optimizer_args = default(
                ae_optimizer_args,
                [{} for _ in range(len(self.trainable_ae_params))],
            )
            # 确保优化器参数和可训练参数数量一致
            assert len(self.ae_optimizer_args) == len(self.trainable_ae_params)
        else:
            self.ae_optimizer_args = [{}]  # 使类型一致

        # 存储可训练的判别器参数
        self.trainable_disc_params = trainable_disc_params
        if self.trainable_disc_params is not None:
            # 设置判别器优化器参数，默认为空字典
            self.disc_optimizer_args = default(
                disc_optimizer_args,
                [{} for _ in range(len(self.trainable_disc_params))],
            )
            # 确保优化器参数和可训练参数数量一致
            assert len(self.disc_optimizer_args) == len(self.trainable_disc_params)
        else:
            self.disc_optimizer_args = [{}]  # 使类型一致

        # 如果设置了检查点路径
        if ckpt_path is not None:
            # 确保不能同时设置检查点引擎和检查点路径
            assert ckpt_engine is None, "Can't set ckpt_engine and ckpt_path"
            logpy.warn("Checkpoint path is deprecated, use `checkpoint_egnine` instead")  # 记录警告
        # 应用检查点，默认使用给定的路径或引擎
        self.apply_ckpt(default(ckpt_path, ckpt_engine))
        # 设置额外解码键的集合
        self.additional_decode_keys = set(default(additional_decode_keys, []))
    # 获取输入数据，返回一个张量
    def get_input(self, batch: Dict) -> torch.Tensor:
        # 假设统一的数据格式，数据加载器返回一个字典
        # 图像张量应缩放到 -1 到 1，并采用通道优先格式（例如 bchw 而不是 bhwc）
        return batch[self.input_key]  # 从批次中提取指定的输入键的数据

    # 获取自动编码器的可训练参数
    def get_autoencoder_params(self) -> list:
        params = []  # 初始化参数列表
        # 检查损失对象是否具有获取可训练自动编码器参数的方法
        if hasattr(self.loss, "get_trainable_autoencoder_parameters"):
            # 将损失对象的可训练参数添加到参数列表
            params += list(self.loss.get_trainable_autoencoder_parameters())
        # 检查正则化对象是否具有获取可训练参数的方法
        if hasattr(self.regularization, "get_trainable_parameters"):
            # 将正则化对象的可训练参数添加到参数列表
            params += list(self.regularization.get_trainable_parameters())
        # 添加编码器的参数到参数列表
        params = params + list(self.encoder.parameters())
        # 添加解码器的参数到参数列表
        params = params + list(self.decoder.parameters())
        return params  # 返回所有可训练参数的列表

    # 获取判别器的可训练参数
    def get_discriminator_params(self) -> list:
        # 检查损失对象是否具有获取可训练参数的方法
        if hasattr(self.loss, "get_trainable_parameters"):
            # 获取并返回损失对象的可训练参数（例如，判别器）
            params = list(self.loss.get_trainable_parameters())
        else:
            params = []  # 如果没有，初始化为空列表
        return params  # 返回判别器的可训练参数列表

    # 获取解码器的最后一层
    def get_last_layer(self):
        return self.decoder.get_last_layer()  # 返回解码器的最后一层

    # 编码输入张量并可选择返回正则化日志
    def encode(
        self,
        x: torch.Tensor,
        return_reg_log: bool = False,
        unregularized: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        z = self.encoder(x, **kwargs)  # 使用编码器对输入进行编码
        if unregularized:
            return z, dict()  # 如果未正则化，返回编码结果和空字典
        z, reg_log = self.regularization(z)  # 对编码结果进行正则化，并获取日志
        if return_reg_log:
            return z, reg_log  # 如果要求返回正则化日志，返回编码结果和日志
        return z  # 返回编码结果

    # 解码输入张量
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.decoder(z, **kwargs)  # 使用解码器对编码结果进行解码
        return x  # 返回解码后的结果

    # 前向传播方法，处理输入并返回编码、解码结果和正则化日志
    def forward(self, x: torch.Tensor, **additional_decode_kwargs) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        z, reg_log = self.encode(x, return_reg_log=True)  # 编码输入并要求返回正则化日志
        dec = self.decode(z, **additional_decode_kwargs)  # 解码编码结果
        return z, dec, reg_log  # 返回编码结果、解码结果和正则化日志
    # 定义内部训练步骤方法，接受批次数据、批次索引和优化器索引
        def inner_training_step(self, batch: dict, batch_idx: int, optimizer_idx: int = 0) -> torch.Tensor:
            # 从批次中获取输入数据
            x = self.get_input(batch)
            # 创建额外解码参数字典，包含批次中额外的解码键
            additional_decode_kwargs = {key: batch[key] for key in self.additional_decode_keys.intersection(batch)}
            # 执行前向传播，得到潜在变量 z、重构的输入 xrec 和正则化日志
            z, xrec, regularization_log = self(x, **additional_decode_kwargs)
            # 检查损失对象是否有 forward_keys 属性
            if hasattr(self.loss, "forward_keys"):
                # 构建额外信息字典，包括潜在变量和训练相关信息
                extra_info = {
                    "z": z,
                    "optimizer_idx": optimizer_idx,
                    "global_step": self.global_step,
                    "last_layer": self.get_last_layer(),
                    "split": "train",
                    "regularization_log": regularization_log,
                    "autoencoder": self,
                }
                # 仅保留在损失对象中定义的额外信息键
                extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
            else:
                # 初始化额外信息为空字典
                extra_info = dict()
    
            # 检查优化器索引，如果是第一个优化器
            if optimizer_idx == 0:
                # 计算自编码器损失
                out_loss = self.loss(x, xrec, **extra_info)
                # 检查损失是否为元组，分解损失和日志字典
                if isinstance(out_loss, tuple):
                    aeloss, log_dict_ae = out_loss
                else:
                    # 简单损失函数，初始化损失和日志字典
                    aeloss = out_loss
                    log_dict_ae = {"train/loss/rec": aeloss.detach()}
    
                # 记录字典中的损失信息
                self.log_dict(
                    log_dict_ae,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                )
                # 在进度条上记录平均损失
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
            # 如果是第二个优化器
            elif optimizer_idx == 1:
                # 计算判别器损失
                discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
                # 判别器总是需要返回一个元组
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                # 返回判别器损失
                return discloss
            else:
                # 抛出未实现的错误，表示未知的优化器索引
                raise NotImplementedError(f"Unknown optimizer {optimizer_idx}")
    
        # 定义训练步骤方法，接受批次数据和批次索引
        def training_step(self, batch: dict, batch_idx: int):
            # 获取当前模型的优化器
            opts = self.optimizers()
            # 检查优化器是否为列表，如果不是则将其转为列表
            if not isinstance(opts, list):
                # 非对抗情况，将优化器放入列表
                opts = [opts]
            # 根据批次索引计算当前优化器的索引
            optimizer_idx = batch_idx % len(opts)
            # 如果全局步骤小于判别器开始迭代，则使用第一个优化器
            if self.global_step < self.disc_start_iter:
                optimizer_idx = 0
            # 选择当前优化器
            opt = opts[optimizer_idx]
            # 将优化器的梯度置为零
            opt.zero_grad()
            # 在优化器的模型切换上下文中执行
            with opt.toggle_model():
                # 调用内部训练步骤，计算损失
                loss = self.inner_training_step(batch, batch_idx, optimizer_idx=optimizer_idx)
                # 手动进行反向传播
                self.manual_backward(loss)
            # 更新优化器的参数
            opt.step()
    
        # 定义验证步骤方法，接受批次数据和批次索引
        def validation_step(self, batch: dict, batch_idx: int) -> Dict:
            # 执行基本的验证步骤，获取日志字典
            log_dict = self._validation_step(batch, batch_idx)
            # 在 EMA（指数移动平均）上下文中执行验证步骤
            with self.ema_scope():
                # 获取 EMA 验证步骤的日志字典
                log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
                # 更新日志字典，合并 EMA 结果
                log_dict.update(log_dict_ema)
            # 返回合并后的日志字典
            return log_dict
    # 定义验证步骤的方法，接受一个批次的数据、批次索引和可选的后缀
    def _validation_step(self, batch: dict, batch_idx: int, postfix: str = "") -> Dict:
        # 从批次数据中获取输入
        x = self.get_input(batch)
    
        # 前向传播，得到潜在变量 z、重建的输入 xrec 和正则化日志
        z, xrec, regularization_log = self(x)
        # 检查损失对象是否有前向键
        if hasattr(self.loss, "forward_keys"):
            # 构建额外信息字典，包含多个状态信息
            extra_info = {
                "z": z,  # 潜在变量
                "optimizer_idx": 0,  # 优化器索引初始化为 0
                "global_step": self.global_step,  # 全局步数
                "last_layer": self.get_last_layer(),  # 获取最后一层的输出
                "split": "val" + postfix,  # 验证数据集的标识
                "regularization_log": regularization_log,  # 正则化日志
                "autoencoder": self,  # 自编码器对象
            }
            # 仅保留损失对象中定义的前向键
            extra_info = {k: extra_info[k] for k in self.loss.forward_keys}
        else:
            # 如果没有前向键，初始化为空字典
            extra_info = dict()
        # 计算损失值
        out_loss = self.loss(x, xrec, **extra_info)
        # 检查损失值是否为元组
        if isinstance(out_loss, tuple):
            aeloss, log_dict_ae = out_loss  # 解包自编码器损失和日志字典
        else:
            # 简单的损失函数处理
            aeloss = out_loss  # 直接将损失赋值给 aeloss
            log_dict_ae = {f"val{postfix}/loss/rec": aeloss.detach()}  # 创建日志字典
        full_log_dict = log_dict_ae  # 初始化完整日志字典
    
        # 如果额外信息中有优化器索引
        if "optimizer_idx" in extra_info:
            extra_info["optimizer_idx"] = 1  # 更新优化器索引
            # 计算判别器损失
            discloss, log_dict_disc = self.loss(x, xrec, **extra_info)
            # 更新完整日志字典，包含判别器的日志
            full_log_dict.update(log_dict_disc)
        # 记录重建损失
        self.log(
            f"val{postfix}/loss/rec",  # 日志名称
            log_dict_ae[f"val{postfix}/loss/rec"],  # 日志值
            sync_dist=True,  # 进行分布式同步
        )
        # 记录完整日志字典
        self.log_dict(full_log_dict, sync_dist=True)
        # 返回完整日志字典
        return full_log_dict
    
    # 定义获取参数组的方法，接受参数名称列表和优化器参数列表
    def get_param_groups(
        self, parameter_names: List[List[str]], optimizer_args: List[dict]
    ) -> Tuple[List[Dict[str, Any]], int]:
        groups = []  # 初始化参数组列表
        num_params = 0  # 初始化参数数量计数器
        # 遍历参数名称和优化器参数
        for names, args in zip(parameter_names, optimizer_args):
            params = []  # 初始化当前参数列表
            # 遍历每个参数模式
            for pattern_ in names:
                pattern_params = []  # 初始化匹配到的参数列表
                pattern = re.compile(pattern_)  # 编译正则表达式模式
                # 遍历命名参数
                for p_name, param in self.named_parameters():
                    # 检查参数名称是否与模式匹配
                    if re.match(pattern, p_name):
                        pattern_params.append(param)  # 添加匹配的参数
                        num_params += param.numel()  # 更新参数数量计数
                # 如果没有找到匹配的参数，发出警告
                if len(pattern_params) == 0:
                    logpy.warn(f"Did not find parameters for pattern {pattern_}")
                params.extend(pattern_params)  # 扩展当前参数列表
            # 将当前参数及其参数设置添加到参数组中
            groups.append({"params": params, **args})
        # 返回参数组和参数数量
        return groups, num_params
    # 配置优化器，返回优化器列表
    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        # 检查自编码器可训练参数是否为 None
        if self.trainable_ae_params is None:
            # 获取自编码器参数
            ae_params = self.get_autoencoder_params()
        else:
            # 获取指定的参数组及其数量
            ae_params, num_ae_params = self.get_param_groups(self.trainable_ae_params, self.ae_optimizer_args)
            # 记录可训练自编码器参数的数量
            logpy.info(f"Number of trainable autoencoder parameters: {num_ae_params:,}")
        # 检查鉴别器可训练参数是否为 None
        if self.trainable_disc_params is None:
            # 获取鉴别器参数
            disc_params = self.get_discriminator_params()
        else:
            # 获取指定的参数组及其数量
            disc_params, num_disc_params = self.get_param_groups(self.trainable_disc_params, self.disc_optimizer_args)
            # 记录可训练鉴别器参数的数量
            logpy.info(f"Number of trainable discriminator parameters: {num_disc_params:,}")
        # 根据自编码器参数和学习率配置实例化优化器
        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        # 初始化优化器列表
        opts = [opt_ae]
        # 如果鉴别器参数不为空，则实例化鉴别器优化器
        if len(disc_params) > 0:
            opt_disc = self.instantiate_optimizer_from_config(disc_params, self.learning_rate, self.optimizer_config)
            opts.append(opt_disc)
        # 返回优化器列表
        return opts
    
    # 不计算梯度，记录图像
    @torch.no_grad()
    def log_images(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
        # 初始化日志字典
        log = dict()
        # 初始化额外解码参数字典
        additional_decode_kwargs = {}
        # 从批次中获取输入
        x = self.get_input(batch)
        # 更新额外解码参数字典
        additional_decode_kwargs.update({key: batch[key] for key in self.additional_decode_keys.intersection(batch)})
        
        # 获取输入的重构结果
        _, xrec, _ = self(x, **additional_decode_kwargs)
        # 记录输入
        log["inputs"] = x
        # 记录重构结果
        log["reconstructions"] = xrec
        # 计算输入与重构之间的差异
        diff = 0.5 * torch.abs(torch.clamp(xrec, -1.0, 1.0) - x)
        # 将差异值限制在 [0, 1] 范围内
        diff.clamp_(0, 1.0)
        # 记录差异值
        log["diff"] = 2.0 * diff - 1.0
        # 通过增强小误差的亮度来显示误差位置
        log["diff_boost"] = 2.0 * torch.clamp(self.diff_boost_factor * diff, 0.0, 1.0) - 1
        # 如果损失对象有 log_images 方法，则更新日志
        if hasattr(self.loss, "log_images"):
            log.update(self.loss.log_images(x, xrec))
        # 进入 EMA 作用域
        with self.ema_scope():
            # 获取 EMA 重构结果
            _, xrec_ema, _ = self(x, **additional_decode_kwargs)
            # 记录 EMA 重构结果
            log["reconstructions_ema"] = xrec_ema
            # 计算 EMA 输入与重构之间的差异
            diff_ema = 0.5 * torch.abs(torch.clamp(xrec_ema, -1.0, 1.0) - x)
            # 将差异值限制在 [0, 1] 范围内
            diff_ema.clamp_(0, 1.0)
            # 记录 EMA 差异值
            log["diff_ema"] = 2.0 * diff_ema - 1.0
            # 记录 EMA 差异增强值
            log["diff_boost_ema"] = 2.0 * torch.clamp(self.diff_boost_factor * diff_ema, 0.0, 1.0) - 1
        # 如果有额外的日志参数，则进行处理
        if additional_log_kwargs:
            additional_decode_kwargs.update(additional_log_kwargs)
            # 获取额外重构结果
            _, xrec_add, _ = self(x, **additional_decode_kwargs)
            # 构造日志字符串
            log_str = "reconstructions-" + "-".join(
                [f"{key}={additional_log_kwargs[key]}" for key in additional_log_kwargs]
            )
            # 记录额外重构结果
            log[log_str] = xrec_add
        # 返回日志字典
        return log
# 定义一个继承自 AutoencodingEngine 的类 AutoencodingEngineLegacy
class AutoencodingEngineLegacy(AutoencodingEngine):
    # 初始化方法，接受嵌入维度和其他关键字参数
    def __init__(self, embed_dim: int, **kwargs):
        # 从 kwargs 中提取最大批次大小，默认值为 None
        self.max_batch_size = kwargs.pop("max_batch_size", None)
        # 从 kwargs 中提取 ddconfig 参数
        ddconfig = kwargs.pop("ddconfig")
        # 从 kwargs 中提取检查点路径，默认值为 None
        ckpt_path = kwargs.pop("ckpt_path", None)
        # 从 kwargs 中提取检查点引擎，默认值为 None
        ckpt_engine = kwargs.pop("ckpt_engine", None)
        # 调用父类构造函数，设置编码器和解码器配置
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
        # 定义量化卷积层，输入通道数由 ddconfig 决定
        self.quant_conv = torch.nn.Conv2d(
            (1 + ddconfig["double_z"]) * ddconfig["z_channels"],
            (1 + ddconfig["double_z"]) * embed_dim,
            1,
        )
        # 定义后量化卷积层，输出通道数为 z_channels
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # 保存嵌入维度
        self.embed_dim = embed_dim
        # 应用检查点，初始化模型状态
        self.apply_ckpt(default(ckpt_path, ckpt_engine))

    # 获取自动编码器参数
    def get_autoencoder_params(self) -> list:
        # 调用父类方法获取参数
        params = super().get_autoencoder_params()
        return params

    # 编码输入张量，返回编码结果和可选的正则化日志
    def encode(self, x: torch.Tensor, return_reg_log: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # 如果没有设置最大批次大小，直接编码并量化
        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            # 获取输入张量的批次大小
            N = x.shape[0]
            bs = self.max_batch_size
            # 计算需要的批次数量
            n_batches = int(math.ceil(N / bs))
            z = list()
            # 遍历每个批次进行编码和量化
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            # 将所有批次的结果连接成一个张量
            z = torch.cat(z, 0)

        # 应用正则化到编码结果
        z, reg_log = self.regularization(z)
        # 根据参数决定返回结果
        if return_reg_log:
            return z, reg_log
        return z

    # 解码输入张量
    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:
        # 如果没有设置最大批次大小，直接解码
        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            # 获取输入张量的批次大小
            N = z.shape[0]
            bs = self.max_batch_size
            # 计算需要的批次数量
            n_batches = int(math.ceil(N / bs))
            dec = list()
            # 遍历每个批次进行解码
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            # 将所有批次的结果连接成一个张量
            dec = torch.cat(dec, 0)

        # 返回解码结果
        return dec


# 定义一个继承自 AbstractAutoencoder 的类 IdentityFirstStage
class IdentityFirstStage(AbstractAutoencoder):
    # 初始化方法，调用父类构造函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 获取输入，返回原始输入
    def get_input(self, x: Any) -> Any:
        return x

    # 编码方法，返回原始输入
    def encode(self, x: Any, *args, **kwargs) -> Any:
        return x

    # 解码方法，什么也不返回
    def decode(self, x: Any, *args, **kwargs) -> Any:
        return


# 定义一个继承自 AutoencodingEngine 的类 VideoAutoencodingEngine
class VideoAutoencodingEngine(AutoencodingEngine):
    # 初始化方法，用于设置模型参数
        def __init__(
            self,
            ckpt_path: Union[None, str] = None,  # 可选的检查点路径，用于加载模型
            ignore_keys: Union[Tuple, list] = (),  # 指定忽略的键列表
            image_video_weights=[1, 1],  # 图像和视频的权重设置
            only_train_decoder=False,  # 仅训练解码器的标志
            context_parallel_size=0,  # 上下文并行的大小
            **kwargs,  # 其他额外参数
        ):
            super().__init__(**kwargs)  # 调用父类的初始化方法
            self.context_parallel_size = context_parallel_size  # 保存上下文并行大小
            if ckpt_path is not None:  # 如果提供了检查点路径
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 从检查点初始化模型
    
        # 日志记录视频的方法，接受一个批次和额外的日志参数
        def log_videos(self, batch: dict, additional_log_kwargs: Optional[Dict] = None, **kwargs) -> dict:
            return self.log_images(batch, additional_log_kwargs, **kwargs)  # 调用 log_images 方法记录视频
    
        # 获取输入数据的方法
        def get_input(self, batch: dict) -> torch.Tensor:
            if self.context_parallel_size > 0:  # 如果上下文并行大小大于0
                if not is_context_parallel_initialized():  # 检查上下文并行是否已初始化
                    initialize_context_parallel(self.context_parallel_size)  # 初始化上下文并行
    
                batch = batch[self.input_key]  # 获取指定键的批次数据
    
                global_src_rank = get_context_parallel_group_rank() * self.context_parallel_size  # 计算全局源排名
                torch.distributed.broadcast(batch, src=global_src_rank, group=get_context_parallel_group())  # 广播批次数据
    
                batch = _conv_split(batch, dim=2, kernel_size=1)  # 在指定维度上分割批次数据
                return batch  # 返回处理后的批次数据
    
            return batch[self.input_key]  # 返回指定键的批次数据
    
        # 应用检查点的方法
        def apply_ckpt(self, ckpt: Union[None, str, dict]):
            if ckpt is None:  # 如果检查点为 None
                return  # 直接返回
            self.init_from_ckpt(ckpt)  # 从检查点初始化模型
    
        # 从检查点初始化模型的方法
        def init_from_ckpt(self, path, ignore_keys=list()):
            sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载检查点的状态字典
            keys = list(sd.keys())  # 获取状态字典中的所有键
            for k in keys:  # 遍历所有键
                for ik in ignore_keys:  # 遍历忽略键列表
                    if k.startswith(ik):  # 如果键以忽略键开头
                        del sd[k]  # 删除该键
    
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)  # 加载状态字典，允许不严格匹配
            print("Missing keys: ", missing_keys)  # 打印缺失的键
            print("Unexpected keys: ", unexpected_keys)  # 打印意外的键
            print(f"Restored from {path}")  # 打印恢复信息
```