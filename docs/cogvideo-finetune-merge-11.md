# CogVideo & CogVideoX 微调代码源码解析（十二）



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

# `.\cogvideo-finetune\sat\vae_modules\cp_enc_dec.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 导入分布式训练模块
import torch.distributed
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络功能模块
import torch.nn.functional as F
# 导入 NumPy 库
import numpy as np

# 从 beartype 导入装饰器和类型
from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
# 从 einops 导入重排功能
from einops import rearrange

# 从自定义模块中导入上下文相关函数
from sgm.util import (
    get_context_parallel_group,  # 获取并行组
    get_context_parallel_rank,   # 获取当前并行任务的排名
    get_context_parallel_world_size,  # 获取并行世界的大小
    get_context_parallel_group_rank,   # 获取当前组的排名
)

# 尝试导入 SafeConv3d，如果失败则注释掉
# try:
from vae_modules.utils import SafeConv3d as Conv3d  # 从 utils 中导入安全的 3D 卷积
# except:
#     # 如果 SafeConv3d 不可用，则降级为普通的 Conv3d
#     from torch.nn import Conv3d  # 从 PyTorch 导入标准的 3D 卷积


def cast_tuple(t, length=1):
    # 如果 t 不是元组，则返回由 t 组成的元组，长度为 length
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    # 检查 num 是否能被 den 整除
    return (num % den) == 0


def is_odd(n):
    # 检查 n 是否为奇数
    return not divisible_by(n, 2)


def exists(v):
    # 检查 v 是否存在（不为 None）
    return v is not None


def pair(t):
    # 如果 t 不是元组，则返回一个由 t 组成的元组
    return t if isinstance(t, tuple) else (t, t)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    该函数实现与去噪扩散概率模型中的相匹配：
    来自 Fairseq。
    构建正弦嵌入。
    该实现与 tensor2tensor 中的相匹配，但与“Attention Is All You Need”第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维的
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2  # 计算嵌入维度的一半
    emb = math.log(10000) / (half_dim - 1)  # 计算基础值
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  # 计算正弦和余弦嵌入的值
    emb = emb.to(device=timesteps.device)  # 将嵌入移动到与 timesteps 相同的设备
    emb = timesteps.float()[:, None] * emb[None, :]  # 根据时间步调整嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 合并正弦和余弦嵌入
    if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数，则进行零填充
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb  # 返回最终的嵌入


def nonlinearity(x):
    # 实现 swish 非线性激活函数
    return x * torch.sigmoid(x)


def leaky_relu(p=0.1):
    # 返回具有指定负斜率的 Leaky ReLU 激活函数
    return nn.LeakyReLU(p)


def _split(input_, dim):
    cp_world_size = get_context_parallel_world_size()  # 获取并行世界的大小

    if cp_world_size == 1:  # 如果只有一个并行任务，直接返回输入
        return input_

    cp_rank = get_context_parallel_rank()  # 获取当前并行任务的排名

    # print('in _split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧并进行转换
    inpu_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()  # 处理输入，去除第一帧
    dim_size = input_.size()[dim] // cp_world_size  # 计算分割后的维度大小

    input_list = torch.split(input_, dim_size, dim=dim)  # 根据维度大小进行分割
    output = input_list[cp_rank]  # 获取当前任务对应的输出

    if cp_rank == 0:  # 如果是第一个任务，将第一帧和输出合并
        output = torch.cat([inpu_first_frame_, output], dim=dim)
    output = output.contiguous()  # 确保输出是连续的内存块

    # print('out _split, cp_rank:', cp_rank, 'output_size:', output.shape)

    return output  # 返回最终的输出


def _gather(input_, dim):
    cp_world_size = get_context_parallel_world_size()  # 获取并行世界的大小

    # 如果只有一个并行任务，直接返回输入
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group()  # 获取并行组
    cp_rank = get_context_parallel_rank()  # 获取当前并行任务的排名

    # print('in _gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取输入的第一帧并进行转换
    input_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    # 如果当前进程的排名为 0
        if cp_rank == 0:
            # 对输入张量进行转置，取出第一个维度之后的所有数据，再转置回原形状
            input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()
    
        # 创建一个张量列表，包含一个空张量和 cp_world_size - 1 个与 input_ 相同形状的空张量
        tensor_list = [torch.empty_like(torch.cat([input_first_frame_, input_], dim=dim))] + [
            torch.empty_like(input_) for _ in range(cp_world_size - 1)
        ]
    
        # 如果当前进程的排名为 0
        if cp_rank == 0:
            # 将第一个帧的输入与当前输入张量在指定维度上拼接
            input_ = torch.cat([input_first_frame_, input_], dim=dim)
    
        # 将当前进程的输入张量赋值给张量列表中的相应位置
        tensor_list[cp_rank] = input_
        # 在所有进程之间收集输入张量，存入张量列表
        torch.distributed.all_gather(tensor_list, input_, group=group)
    
        # 在指定维度上拼接所有收集到的张量，并返回连续的内存块
        output = torch.cat(tensor_list, dim=dim).contiguous()
    
        # 可能用于调试输出当前进程的排名和输出张量的形状
        # print('out _gather, cp_rank:', cp_rank, 'output_size:', output.shape)
    
        # 返回最终的输出张量
        return output
# 定义一个用于分割输入的函数，参数包括输入张量、维度和卷积核大小
def _conv_split(input_, dim, kernel_size):
    # 获取并行环境中的总进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果进程数量为1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前进程的排名
    cp_rank = get_context_parallel_rank()

    # 计算在指定维度上，每个进程处理的元素数量
    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    # 对于第一个进程，计算输出张量
    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # 对于其他进程，计算输出张量
        output = input_.transpose(dim, 0)[
            cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size
        ].transpose(dim, 0)
    # 确保输出张量在内存中是连续的
    output = output.contiguous()

    # 返回输出张量
    return output


# 定义一个用于聚合输入的函数，参数包括输入张量、维度和卷积核大小
def _conv_gather(input_, dim, kernel_size):
    # 获取并行环境中的总进程数量
    cp_world_size = get_context_parallel_world_size()

    # 如果进程数量为1，则直接返回输入
    if cp_world_size == 1:
        return input_

    # 获取当前进程的组和排名
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()

    # 获取输入张量的首个卷积核并确保是连续的
    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    # 对于第一个进程，更新输入张量
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        # 对于其他进程，处理输入张量
        input_ = input_.transpose(0, dim)[max(kernel_size - 1, 0) :].transpose(0, dim).contiguous()

    # 创建一个张量列表，用于存储各进程的输入
    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    # 对于第一个进程，合并张量
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    # 将当前进程的输入放入列表
    tensor_list[cp_rank] = input_
    # 收集所有进程的输入
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # 合并张量列表并确保是连续的
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # 返回输出张量
    return output


# 定义一个用于从前一个进程传递输入的函数，参数包括输入张量、维度和卷积核大小
def _pass_from_previous_rank(input_, dim, kernel_size):
    # 如果卷积核大小为1，则直接返回输入
    if kernel_size == 1:
        return input_

    # 获取当前进程的组、排名和总进程数量
    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    # 获取全局进程排名和数量
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    # 将输入张量在指定维度上转置
    input_ = input_.transpose(0, dim)

    # 确定发送和接收的进程排名
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    # 如果发送排名能被总进程数量整除，则调整发送排名
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    # 检查接收进程的排名，决定是否需要增加排名值以避免冲突
        if recv_rank % cp_world_size == cp_world_size - 1:
            recv_rank += cp_world_size
    
        # 检查当前进程排名，是否小于世界大小减一
        if cp_rank < cp_world_size - 1:
            # 发送输入的最后一部分到指定发送进程
            req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
        # 检查当前进程排名，是否大于零
        if cp_rank > 0:
            # 创建与输入相同形状的空缓冲区以接收数据
            recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
            # 异步接收数据到接收缓冲区
            req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)
    
        # 检查当前进程是否为主进程
        if cp_rank == 0:
            # 将输入的第一部分重复并与原始输入连接
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
        else:
            # 等待接收请求完成
            req_recv.wait()
            # 将接收到的缓冲区与原始输入连接
            input_ = torch.cat([recv_buffer, input_], dim=0)
    
        # 转置输入张量的维度以适应后续操作
        input_ = input_.transpose(0, dim).contiguous()
    
        # 打印当前进程排名及输入大小（调试用）
        # print('out _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)
    
        # 返回最终处理后的输入张量
        return input_
# 定义一个私有函数，用于从之前的并行排名获取数据，处理卷积运算
def _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding=None):
    # 如果卷积核大小为1，则直接返回输入数据
    if kernel_size == 1:
        return input_

    # 获取当前并行上下文的组
    group = get_context_parallel_group()
    # 获取当前的并行排名
    cp_rank = get_context_parallel_rank()
    # 获取当前组的排名
    cp_group_rank = get_context_parallel_group_rank()
    # 获取当前并行组的世界大小
    cp_world_size = get_context_parallel_world_size()

    # print('in _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)

    # 获取全局排名和全局世界大小
    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    # 在指定维度上转置输入数据
    input_ = input_.transpose(0, dim)

    # 从上一个排名传递数据
    send_rank = global_rank + 1  # 发送给下一个排名
    recv_rank = global_rank - 1  # 接收来自上一个排名的数据
    # 如果发送排名超出范围，则调整为循环发送
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    # 如果接收排名是最后一个，则调整为循环接收
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    # 创建接收缓冲区，用于存储接收到的数据
    recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
    # 如果当前排名小于最后一个，则发送数据
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    # 如果当前排名大于0，则接收数据
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    # 如果当前排名是0，则处理输入数据
    if cp_rank == 0:
        # 如果有缓存填充，则将其与输入数据拼接
        if cache_padding is not None:
            input_ = torch.cat([cache_padding.transpose(0, dim).to(input_.device), input_], dim=0)
        # 否则，重复输入数据的第一项以填充
        else:
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
    else:
        # 等待接收请求完成，然后拼接接收到的数据
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    # 再次转置输入数据并确保其内存连续
    input_ = input_.transpose(0, dim).contiguous()
    # 返回处理后的输入数据
    return input_


# 定义一个私有函数，用于从之前的排名丢弃数据，处理卷积运算
def _drop_from_previous_rank(input_, dim, kernel_size):
    # 转置输入数据，然后丢弃前 kernel_size - 1 个元素
    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    # 返回处理后的输入数据
    return input_


# 定义一个类，继承自 torch.autograd.Function，表示卷积散射操作
class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    # 前向传播函数
    def forward(ctx, input_, dim, kernel_size):
        # 保存维度和卷积核大小到上下文
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用分卷积函数进行处理
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    # 反向传播函数
    def backward(ctx, grad_output):
        # 调用收集卷积函数进行处理，并返回结果
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None


# 定义一个类，继承自 torch.autograd.Function，表示卷积收集操作
class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    # 前向传播函数
    def forward(ctx, input_, dim, kernel_size):
        # 保存维度和卷积核大小到上下文
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用收集卷积函数进行处理
        return _conv_gather(input_, dim, kernel_size)

    @staticmethod
    # 反向传播函数
    def backward(ctx, grad_output):
        # 调用分卷积函数进行处理，并返回结果
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size), None, None
# 定义一个用于前一秩的卷积操作的自定义 PyTorch 函数
class _ConvolutionPassFromPreviousRank(torch.autograd.Function):
    # 定义前向传播的静态方法
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        # 将维度和卷积核大小存储在上下文中以供后向传播使用
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用外部函数执行前向卷积操作并返回结果
        return _pass_from_previous_rank(input_, dim, kernel_size)

    # 定义后向传播的静态方法
    @staticmethod
    def backward(ctx, grad_output):
        # 调用外部函数处理梯度并返回，None 表示不返回额外的梯度
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None


# 定义一个用于前一秩的假 CP 卷积操作的自定义 PyTorch 函数
class _FakeCPConvolutionPassFromPreviousRank(torch.autograd.Function):
    # 定义前向传播的静态方法
    @staticmethod
    def forward(ctx, input_, dim, kernel_size, cache_padding):
        # 将维度、卷积核大小和缓存填充存储在上下文中
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        # 调用外部函数执行假 CP 卷积操作并返回结果
        return _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding)

    # 定义后向传播的静态方法
    @staticmethod
    def backward(ctx, grad_output):
        # 调用外部函数处理梯度并返回，None 表示不返回额外的梯度
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None, None


# 定义一个函数用于将输入数据分散到上下文并进行并行区域处理
def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    # 调用自定义函数并返回结果
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)


# 定义一个函数用于从上下文并行区域汇聚输入数据
def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    # 调用自定义函数并返回结果
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)


# 定义一个函数用于从最后一秩进行卷积传递
def conv_pass_from_last_rank(input_, dim, kernel_size):
    # 调用自定义函数并返回结果
    return _ConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)


# 定义一个函数用于进行假 CP 从前一秩的传递
def fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding):
    # 调用自定义函数并返回结果
    return _FakeCPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size, cache_padding)


# 定义一个 3D 因果卷积的上下文并行模块
class ContextParallelCausalConv3d(nn.Module):
    # 初始化模块，设置输入输出通道、卷积核大小、步幅等参数
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride=1, **kwargs):
        # 调用父类初始化方法
        super().__init__()
        # 将卷积核大小转换为元组，确保有三个维度
        kernel_size = cast_tuple(kernel_size, 3)

        # 分别获取时间、高度和宽度的卷积核大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 确保高度和宽度的卷积核大小为奇数，以便中心对齐
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 计算填充大小，以保持卷积输出的维度
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 存储填充和卷积核的相关参数
        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 2

        # 设置步幅和扩张参数为三维的相同值
        stride = (stride, stride, stride)
        dilation = (1, 1, 1)
        # 初始化 3D 卷积层
        self.conv = Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        # 初始化缓存填充为 None
        self.cache_padding = None
    # 定义前向传播函数，接受输入数据和清除缓存的标志
        def forward(self, input_, clear_cache=True):
            # 如果输入的形状第三维为1，处理图像数据
            #     # 对第一帧进行填充
            #     input_parallel = torch.cat([input_] * self.time_kernel_size, dim=2)
            # else:
            #     # 从最后一维进行卷积处理
            #     input_parallel = conv_pass_from_last_rank(input_, self.temporal_dim, self.time_kernel_size)
    
            # 设置2D填充的大小
            # padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对输入进行填充，填充值为0
            # input_parallel = F.pad(input_parallel, padding_2d, mode = 'constant', value = 0)
    
            # 对填充后的输入进行卷积操作
            # output_parallel = self.conv(input_parallel)
            # 赋值输出
            # output = output_parallel
            # 返回输出结果
            # return output
    
            # 从上一个层的输出中获取并处理输入，添加时间维度和缓存填充
            input_parallel = fake_cp_pass_from_previous_rank(
                input_, self.temporal_dim, self.time_kernel_size, self.cache_padding
            )
    
            # 删除旧的缓存填充
            del self.cache_padding
            # 将缓存填充设置为None
            self.cache_padding = None
            # 如果不清除缓存
            if not clear_cache:
                # 获取并行计算的当前排名和总大小
                cp_rank, cp_world_size = get_context_parallel_rank(), get_context_parallel_world_size()
                # 获取全局排名
                global_rank = torch.distributed.get_rank()
                # 如果并行计算的大小为1
                if cp_world_size == 1:
                    # 保存最后一帧的缓存填充
                    self.cache_padding = (
                        input_parallel[:, :, -self.time_kernel_size + 1 :].contiguous().detach().clone().cpu()
                    )
                else:
                    # 如果是最后一个并行计算的排名
                    if cp_rank == cp_world_size - 1:
                        # 发送最后一帧数据到下一个全局排名
                        torch.distributed.isend(
                            input_parallel[:, :, -self.time_kernel_size + 1 :].contiguous(),
                            global_rank + 1 - cp_world_size,
                            group=get_context_parallel_group(),
                        )
                    # 如果是第一个并行计算的排名
                    if cp_rank == 0:
                        # 创建接收缓存并接收数据
                        recv_buffer = torch.empty_like(input_parallel[:, :, -self.time_kernel_size + 1 :]).contiguous()
                        torch.distributed.recv(
                            recv_buffer, global_rank - 1 + cp_world_size, group=get_context_parallel_group()
                        )
                        # 保存接收的数据作为缓存填充
                        self.cache_padding = recv_buffer.contiguous().detach().clone().cpu()
    
            # 设置2D填充的大小
            padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            # 对输入进行填充，填充值为0
            input_parallel = F.pad(input_parallel, padding_2d, mode="constant", value=0)
    
            # 对填充后的输入进行卷积操作
            output_parallel = self.conv(input_parallel)
            # 赋值输出
            output = output_parallel
            # 返回输出结果
            return output
# 定义一个名为 ContextParallelGroupNorm 的类，继承自 torch.nn.GroupNorm
class ContextParallelGroupNorm(torch.nn.GroupNorm):
    # 定义前向传播方法
    def forward(self, input_):
        # 检查输入的第三个维度大小是否大于1，用于决定是否进行上下文并行处理
        gather_flag = input_.shape[2] > 1
        # 如果需要进行上下文并行处理
        if gather_flag:
            # 从上下文并行区域聚合输入数据，维度为2，卷积核大小为1
            input_ = conv_gather_from_context_parallel_region(input_, dim=2, kernel_size=1)
        # 调用父类的前向传播方法，处理输入数据
        output = super().forward(input_)
        # 如果需要进行上下文并行处理
        if gather_flag:
            # 从上下文并行区域散播输出数据，维度为2，卷积核大小为1
            output = conv_scatter_to_context_parallel_region(output, dim=2, kernel_size=1)
        # 返回处理后的输出数据
        return output


# 定义一个函数 Normalize，接受输入通道数和其他参数
def Normalize(in_channels, gather=False, **kwargs):  # 适用于3D和2D情况
    # 如果需要聚合
    if gather:
        # 返回一个上下文并行的 GroupNorm 实例
        return ContextParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        # 返回一个普通的 GroupNorm 实例
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 定义一个名为 SpatialNorm3D 的类，继承自 nn.Module
class SpatialNorm3D(nn.Module):
    # 定义初始化方法，接受多个参数
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        gather=False,
        **norm_layer_params,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果需要聚合
        if gather:
            # 初始化上下文并行的归一化层
            self.norm_layer = ContextParallelGroupNorm(num_channels=f_channels, **norm_layer_params)
        else:
            # 初始化普通的归一化层
            self.norm_layer = torch.nn.GroupNorm(num_channels=f_channels, **norm_layer_params)
        # self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)  # 注释掉的代码
        # 如果需要冻结归一化层的参数
        if freeze_norm_layer:
            # 遍历归一化层的参数，将其 requires_grad 属性设置为 False
            for p in self.norm_layer.parameters:
                p.requires_grad = False

        # 保存是否添加卷积层的标志
        self.add_conv = add_conv
        # 如果需要添加卷积层
        if add_conv:
            # 初始化上下文并行的因果卷积层，输入和输出通道均为 zq_channels，卷积核大小为3
            self.conv = ContextParallelCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
            )

        # 初始化上下文并行的因果卷积层，输入通道为 zq_channels，输出通道为 f_channels，卷积核大小为1
        self.conv_y = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )
        # 初始化另一个上下文并行的因果卷积层，参数同上
        self.conv_b = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )
    # 定义前向传播方法，接受输入张量 f、zq 和一个可选参数
        def forward(self, f, zq, clear_fake_cp_cache=True):
            # 检查 f 的第三维度是否大于 1 且为奇数
            if f.shape[2] > 1 and f.shape[2] % 2 == 1:
                # 将 f 分为第一帧和其余帧
                f_first, f_rest = f[:, :, :1], f[:, :, 1:]
                # 获取第一帧和其余帧的大小
                f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
                # 将 zq 分为第一帧和其余帧
                zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
                # 使用最近邻插值调整 zq_first 的大小
                zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
                # 使用最近邻插值调整 zq_rest 的大小
                zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
                # 在第三维度上连接调整后的 zq_first 和 zq_rest
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                # 对 zq 进行最近邻插值调整，匹配 f 的大小
                zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")
    
            # 如果需要，使用卷积层处理 zq
            if self.add_conv:
                zq = self.conv(zq, clear_cache=clear_fake_cp_cache)
    
            # 对输入 f 进行归一化处理
            norm_f = self.norm_layer(f)
            # norm_f = conv_scatter_to_context_parallel_region(norm_f, dim=2, kernel_size=1)
    
            # 计算新的特征 f，结合 norm_f 和 zq 的卷积输出
            new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
            # 返回新的特征 f
            return new_f
# 定义一个用于3D标准化的函数
def Normalize3D(
    # 输入通道数
    in_channels,
    # 量化通道
    zq_ch,
    # 是否添加卷积层
    add_conv,
    # 是否进行聚合操作
    gather=False,
):
    # 返回经过3D空间标准化的结果
    return SpatialNorm3D(
        # 输入通道数
        in_channels,
        # 量化通道
        zq_ch,
        # 聚合参数
        gather=gather,
        # 不冻结标准化层
        freeze_norm_layer=False,
        # 添加卷积参数
        add_conv=add_conv,
        # 组数设置
        num_groups=32,
        # 防止除零的微小值
        eps=1e-6,
        # 启用仿射变换
        affine=True,
    )


# 定义一个3D上采样的神经网络模块
class Upsample3D(nn.Module):
    # 初始化方法
    def __init__(
        # 输入通道数
        self,
        in_channels,
        # 是否添加卷积层
        with_conv,
        # 是否压缩时间维度
        compress_time=False,
    ):
        # 调用父类构造方法
        super().__init__()
        # 设置卷积参数
        self.with_conv = with_conv
        # 如果需要卷积层
        if self.with_conv:
            # 创建一个卷积层
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 设置时间压缩参数
        self.compress_time = compress_time

    # 前向传播方法
    def forward(self, x):
        # 如果压缩时间且时间维度大于1
        if self.compress_time and x.shape[2] > 1:
            # 如果时间维度是奇数
            if x.shape[2] % 2 == 1:
                # 分离第一帧
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                # 对第一帧进行插值上采样
                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                # 对剩余帧进行插值上采样
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")
                # 将第一帧和剩余帧合并
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:
                # 对所有帧进行插值上采样
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

        else:
            # 仅进行2D插值上采样
            t = x.shape[2]
            # 调整维度以便插值处理
            x = rearrange(x, "b c t h w -> (b t) c h w")
            # 进行插值上采样
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            # 调整维度回到原形状
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        # 如果需要卷积层
        if self.with_conv:
            t = x.shape[2]
            # 调整维度以便卷积处理
            x = rearrange(x, "b c t h w -> (b t) c h w")
            # 进行卷积操作
            x = self.conv(x)
            # 调整维度回到原形状
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        # 返回处理后的张量
        return x


# 定义一个3D下采样的神经网络模块
class DownSample3D(nn.Module):
    # 初始化方法
    def __init__(self, in_channels, with_conv, compress_time=False, out_channels=None):
        # 调用父类构造方法
        super().__init__()
        # 设置卷积参数
        self.with_conv = with_conv
        # 如果未指定输出通道数，使用输入通道数
        if out_channels is None:
            out_channels = in_channels
        # 如果需要卷积层
        if self.with_conv:
            # 因为 PyTorch 的卷积不支持不对称填充，手动处理填充
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        # 设置时间压缩参数
        self.compress_time = compress_time
    # 定义前向传播函数，输入为 x
        def forward(self, x):
            # 如果启用了压缩且输入的时间维度大于 1
            if self.compress_time and x.shape[2] > 1:
                # 获取输入的高和宽
                h, w = x.shape[-2:]
                # 重新排列张量的维度
                x = rearrange(x, "b c t h w -> (b h w) c t")
    
                # 如果最后一个维度的大小为奇数
                if x.shape[-1] % 2 == 1:
                    # 分离第一帧和其余帧
                    x_first, x_rest = x[..., 0], x[..., 1:]
    
                    # 如果其余帧的时间维度大于 0，进行平均池化
                    if x_rest.shape[-1] > 0:
                        x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)
                    # 将第一帧与池化后的其余帧连接
                    x = torch.cat([x_first[..., None], x_rest], dim=-1)
                    # 重新排列张量的维度
                    x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
                else:
                    # 对输入进行平均池化
                    x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
                    # 重新排列张量的维度
                    x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
    
            # 如果启用了卷积操作
            if self.with_conv:
                # 定义填充的大小
                pad = (0, 1, 0, 1)
                # 对输入进行填充
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
                # 获取时间维度的大小
                t = x.shape[2]
                # 重新排列张量的维度
                x = rearrange(x, "b c t h w -> (b t) c h w")
                # 进行卷积操作
                x = self.conv(x)
                # 重新排列张量的维度
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            else:
                # 获取时间维度的大小
                t = x.shape[2]
                # 重新排列张量的维度
                x = rearrange(x, "b c t h w -> (b t) c h w")
                # 对输入进行平均池化
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
                # 重新排列张量的维度
                x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            # 返回处理后的张量
            return x
# 定义一个三维卷积的上下文并行残差块，继承自 nn.Module
class ContextParallelResnetBlock3D(nn.Module):
    # 初始化函数，设置各项参数
    def __init__(
        self,
        *,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数，可选
        conv_shortcut=False,  # 是否使用卷积捷径
        dropout,  # dropout 概率
        temb_channels=512,  # 时间嵌入通道数
        zq_ch=None,  # 可选的 zq 通道数
        add_conv=False,  # 是否添加卷积
        gather_norm=False,  # 是否使用聚合归一化
        normalization=Normalize,  # 归一化方法
    ):
        super().__init__()  # 调用父类的初始化方法
        self.in_channels = in_channels  # 设置输入通道数
        out_channels = in_channels if out_channels is None else out_channels  # 设置输出通道数，若未指定则与输入通道数相同
        self.out_channels = out_channels  # 保存输出通道数
        self.use_conv_shortcut = conv_shortcut  # 保存是否使用卷积捷径的标志

        # 初始化归一化层，输入通道数和其他参数
        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )

        # 创建上下文并行因果卷积层，输入和输出通道数以及卷积核大小
        self.conv1 = ContextParallelCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        # 如果时间嵌入通道数大于0，则创建线性投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层，输出通道数和其他参数
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )
        # 创建 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 创建第二个上下文并行因果卷积层，输入和输出通道数
        self.conv2 = ContextParallelCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        # 如果输入通道数与输出通道数不相等
        if self.in_channels != self.out_channels:
            # 如果使用卷积捷径，创建卷积捷径层
            if self.use_conv_shortcut:
                self.conv_shortcut = ContextParallelCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            # 否则创建 1x1 卷积捷径层
            else:
                self.nin_shortcut = Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
    # 定义前向传播函数，接收输入数据x、时间嵌入temb、可选参数zq和是否清除虚假缓存的标志
    def forward(self, x, temb, zq=None, clear_fake_cp_cache=True):
        # 初始化隐藏状态h为输入x
        h = x

        # 判断self.norm1是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm1, torch.nn.GroupNorm):
        #     # 在并行区域中从上下文聚合输入h
        #     h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        # 如果zq不为None，则使用zq和清除缓存标志调用规范化层norm1
        if zq is not None:
            h = self.norm1(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            # 否则仅使用规范化层norm1
            h = self.norm1(h)
        # 判断self.norm1是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm1, torch.nn.GroupNorm):
        #     # 在并行区域中将输入h散射到上下文
        #     h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过卷积层conv1处理h，并清除虚假缓存
        h = self.conv1(h, clear_cache=clear_fake_cp_cache)

        # 如果temb不为None，将其嵌入到h中
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        # 判断self.norm2是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm2, torch.nn.GroupNorm):
        #     # 在并行区域中从上下文聚合输入h
        #     h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        # 如果zq不为None，则使用zq和清除缓存标志调用规范化层norm2
        if zq is not None:
            h = self.norm2(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            # 否则仅使用规范化层norm2
            h = self.norm2(h)
        # 判断self.norm2是否为GroupNorm类型（此行被注释）
        # if isinstance(self.norm2, torch.nn.GroupNorm):
        #     # 在并行区域中将输入h散射到上下文
        #     h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过dropout层进行正则化
        h = self.dropout(h)
        # 通过卷积层conv2处理h，并清除虚假缓存
        h = self.conv2(h, clear_cache=clear_fake_cp_cache)

        # 如果输入通道数与输出通道数不同
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷方式
            if self.use_conv_shortcut:
                # 通过卷积快捷方式处理输入x，并清除虚假缓存
                x = self.conv_shortcut(x, clear_cache=clear_fake_cp_cache)
            else:
                # 否则通过nin快捷方式处理输入x
                x = self.nin_shortcut(x)

        # 返回x和h的相加结果
        return x + h
# 定义一个名为 ContextParallelEncoder3D 的类，继承自 nn.Module
class ContextParallelEncoder3D(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        # 使用关键字参数定义初始化需要的参数
        self,
        *,
        # 输入通道数
        ch,
        # 输出通道数
        out_ch,
        # 通道倍增的元组，控制不同层的通道数
        ch_mult=(1, 2, 4, 8),
        # 残差块的数量
        num_res_blocks,
        # 注意力分辨率
        attn_resolutions,
        # dropout 的比例，默认为 0
        dropout=0.0,
        # 是否使用卷积进行上采样，默认为 True
        resamp_with_conv=True,
        # 输入数据的通道数
        in_channels,
        # 输入数据的分辨率
        resolution,
        # 潜在空间的通道数
        z_channels,
        # 是否使用双重潜在空间，默认为 True
        double_z=True,
        # 填充模式，默认为 "first"
        pad_mode="first",
        # 时间压缩次数，默认为 4
        temporal_compress_times=4,
        # 是否收集归一化，默认为 False
        gather_norm=False,
        # 其余不需要的关键字参数
        **ignore_kwargs,
    ):
        # 调用父类构造函数
        super().__init__()
        # 设置当前类的通道数
        self.ch = ch
        # 初始化时间嵌入通道数
        self.temb_ch = 0
        # 计算分辨率数量
        self.num_resolutions = len(ch_mult)
        # 记录残差块数量
        self.num_res_blocks = num_res_blocks
        # 设置分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels

        # 计算 temporal_compress_times 的以 2 为底的对数值
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # 初始化输入卷积层，使用 3x3 卷积
        self.conv_in = ContextParallelCausalConv3d(
            chan_in=in_channels,  # 输入通道数
            chan_out=self.ch,     # 输出通道数
            kernel_size=3,        # 卷积核大小
        )

        # 当前分辨率
        curr_res = resolution
        # 输入通道数的倍数，包含 1 作为初始值
        in_ch_mult = (1,) + tuple(ch_mult)
        # 创建一个模块列表，用于存储每个分辨率的网络层
        self.down = nn.ModuleList()
        # 遍历每个分辨率
        for i_level in range(self.num_resolutions):
            # 创建模块列表用于存储块和注意力层
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 当前块的输入通道数
            block_in = ch * in_ch_mult[i_level]
            # 当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks):
                # 添加一个残差块到块列表
                block.append(
                    ContextParallelResnetBlock3D(
                        in_channels=block_in,   # 输入通道数
                        out_channels=block_out,  # 输出通道数
                        dropout=dropout,         # dropout 参数
                        temb_channels=self.temb_ch,  # 时间嵌入通道数
                        gather_norm=gather_norm,      # 归一化设置
                    )
                )
                # 更新输入通道数为输出通道数
                block_in = block_out
            # 创建一个新的模块，用于下采样
            down = nn.Module()
            down.block = block  # 将块赋值给下采样模块
            down.attn = attn    # 将注意力层赋值给下采样模块
            # 如果不是最后一个分辨率
            if i_level != self.num_resolutions - 1:
                # 如果当前层小于时间压缩层
                if i_level < self.temporal_compress_level:
                    # 使用卷积下采样
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=True)
                else:
                    # 不使用卷积下采样
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=False)
                # 更新当前分辨率为一半
                curr_res = curr_res // 2
            # 将下采样模块添加到下采样列表中
            self.down.append(down)

        # middle
        # 创建中间模块
        self.mid = nn.Module()
        # 添加第一个中间残差块
        self.mid.block_1 = ContextParallelResnetBlock3D(
            in_channels=block_in,    # 输入通道数
            out_channels=block_in,    # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,         # dropout 参数
            gather_norm=gather_norm,  # 归一化设置
        )

        # 添加第二个中间残差块
        self.mid.block_2 = ContextParallelResnetBlock3D(
            in_channels=block_in,    # 输入通道数
            out_channels=block_in,    # 输出通道数
            temb_channels=self.temb_ch,  # 时间嵌入通道数
            dropout=dropout,         # dropout 参数
            gather_norm=gather_norm,  # 归一化设置
        )

        # end
        # 初始化输出归一化层
        self.norm_out = Normalize(block_in, gather=gather_norm)

        # 初始化输出卷积层，使用 3x3 卷积
        self.conv_out = ContextParallelCausalConv3d(
            chan_in=block_in,                       # 输入通道数
            chan_out=2 * z_channels if double_z else z_channels,  # 输出通道数，根据条件决定
            kernel_size=3,                          # 卷积核大小
        )
    # 定义前向传播方法，接收输入 x 和其他可选参数
    def forward(self, x, **kwargs):
        # 初始化时间步嵌入为 None
        temb = None
    
        # 进行下采样操作
        h = self.conv_in(x)  # 输入通过初始卷积层处理
        for i_level in range(self.num_resolutions):  # 遍历每个分辨率级别
            for i_block in range(self.num_res_blocks):  # 遍历每个残差块
                h = self.down[i_level].block[i_block](h, temb)  # 通过当前块处理 h
                if len(self.down[i_level].attn) > 0:  # 如果有注意力机制
                    h = self.down[i_level].attn[i_block](h)  # 通过注意力机制处理 h
            if i_level != self.num_resolutions - 1:  # 如果不是最后一个分辨率级别
                h = self.down[i_level].downsample(h)  # 对 h 进行下采样
    
        # 经过中间处理
        h = self.mid.block_1(h, temb)  # 通过中间块 1 处理 h
        h = self.mid.block_2(h, temb)  # 通过中间块 2 处理 h
    
        # 最终处理
        # h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)  # 选择性操作，未启用
        h = self.norm_out(h)  # 对 h 进行归一化处理
        # h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)  # 选择性操作，未启用
    
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h)  # 通过输出卷积层处理 h
    
        return h  # 返回处理后的结果
# 定义一个名为 ContextParallelDecoder3D 的类，继承自 nn.Module
class ContextParallelDecoder3D(nn.Module):
    # 初始化方法，接受多种参数用于配置
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制的分辨率
        dropout=0.0,  # dropout 概率
        resamp_with_conv=True,  # 是否使用卷积进行上采样
        in_channels,  # 输入的通道数
        resolution,  # 输入分辨率
        z_channels,  # z 的通道数
        give_pre_end=False,  # 是否给出前置结束输出
        zq_ch=None,  # 可选的 zq 通道数
        add_conv=False,  # 是否添加卷积
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩次数
        gather_norm=False,  # 是否聚集归一化
        **ignorekwargs,  # 其他忽略的关键字参数
    ):
        # 省略具体的初始化实现
        pass

    # 前向传播方法，定义网络的前向计算过程
    def forward(self, z, clear_fake_cp_cache=True, **kwargs):
        # 保存输入 z 的形状，用于后续处理
        self.last_z_shape = z.shape

        # 时间步嵌入初始化为 None
        temb = None

        # 获取 z 的时间维度大小
        t = z.shape[2]
        # 将 z 赋值给 zq，准备后续处理
        zq = z
        # 使用输入 z 进行初步卷积处理，生成特征图 h
        h = self.conv_in(z, clear_cache=clear_fake_cp_cache)

        # 中间层处理
        h = self.mid.block_1(h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过第一个中间块处理 h
        h = self.mid.block_2(h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过第二个中间块处理 h

        # 上采样过程
        for i_level in reversed(range(self.num_resolutions)):  # 从高分辨率到低分辨率反向遍历
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过上采样块处理 h
                if len(self.up[i_level].attn) > 0:  # 如果当前级别有注意力机制
                    h = self.up[i_level].attn[i_block](h, zq)  # 通过注意力机制处理 h
            if i_level != 0:  # 如果不是最后一层
                h = self.up[i_level].upsample(h)  # 对 h 进行上采样

        # 结束层处理
        if self.give_pre_end:  # 如果需要前置结束输出
            return h  # 返回当前的特征图 h

        # 归一化输出
        h = self.norm_out(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)  # 通过归一化层处理 h
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h, clear_cache=clear_fake_cp_cache)  # 通过输出卷积层处理 h

        return h  # 返回最终的特征图 h

    # 获取最后一层的卷积权重
    def get_last_layer(self):
        return self.conv_out.conv.weight  # 返回最后输出卷积层的权重
```

# `.\cogvideo-finetune\sat\vae_modules\ema.py`

```py
# 导入 PyTorch 库和神经网络模块
import torch
from torch import nn

# 定义 LitEma 类，继承自 nn.Module
class LitEma(nn.Module):
    # 初始化方法，接收模型、衰减率和是否使用更新计数
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        # 调用父类的初始化方法
        super().__init__()
        # 检查衰减率是否在有效范围内
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        # 创建一个空字典用于存储模型参数名称到阴影参数名称的映射
        self.m_name2s_name = {}
        # 注册衰减率的缓冲区
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        # 根据是否使用更新计数注册相应的缓冲区
        self.register_buffer(
            "num_updates",
            torch.tensor(0, dtype=torch.int) if use_num_upates else torch.tensor(-1, dtype=torch.int),
        )

        # 遍历模型的命名参数
        for name, p in model.named_parameters():
            # 如果参数需要梯度
            if p.requires_grad:
                # 将名称中的 '.' 替换为 ''
                s_name = name.replace(".", "")
                # 更新模型参数名称到阴影参数名称的映射
                self.m_name2s_name.update({name: s_name})
                # 注册克隆的参数数据为缓冲区
                self.register_buffer(s_name, p.clone().detach().data)

        # 初始化收集的参数列表
        self.collected_params = []

    # 重置更新计数的方法
    def reset_num_updates(self):
        # 删除当前的更新计数缓冲区
        del self.num_updates
        # 注册一个新的更新计数缓冲区，初始值为 0
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.int))

    # 前向传播方法，接收一个模型作为输入
    def forward(self, model):
        # 获取当前的衰减率
        decay = self.decay

        # 如果更新计数为非负
        if self.num_updates >= 0:
            # 更新计数加 1
            self.num_updates += 1
            # 更新衰减率
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        # 计算 1 减去衰减率
        one_minus_decay = 1.0 - decay

        # 在无梯度计算的上下文中执行
        with torch.no_grad():
            # 获取模型的参数字典
            m_param = dict(model.named_parameters())
            # 获取阴影参数的字典
            shadow_params = dict(self.named_buffers())

            # 遍历模型的参数
            for key in m_param:
                # 如果参数需要梯度
                if m_param[key].requires_grad:
                    # 获取对应的阴影参数名称
                    sname = self.m_name2s_name[key]
                    # 将阴影参数转换为与模型参数相同的类型
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    # 更新阴影参数
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    # 确保该参数不在映射中
                    assert not key in self.m_name2s_name

    # 将当前的阴影参数复制到模型参数的方法
    def copy_to(self, model):
        # 获取模型的参数字典
        m_param = dict(model.named_parameters())
        # 获取阴影参数的字典
        shadow_params = dict(self.named_buffers())
        # 遍历模型的参数
        for key in m_param:
            # 如果参数需要梯度
            if m_param[key].requires_grad:
                # 将阴影参数的数据复制到模型参数
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                # 确保该参数不在映射中
                assert not key in self.m_name2s_name

    # 存储当前参数以备后续恢复的方法
    def store(self, parameters):
        """
        保存当前参数以便稍后恢复。
        参数:
          parameters: 可迭代的 `torch.nn.Parameter`；需要临时存储的参数。
        """
        # 克隆参数并存储在 collected_params 列表中
        self.collected_params = [param.clone() for param in parameters]
    # 定义恢复方法，接受参数以恢复存储的模型参数
    def restore(self, parameters):
        # 文档字符串，说明此方法的作用和参数
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        # 遍历收集到的参数和传入的参数，并将收集到的参数数据复制到对应的参数
        for c_param, param in zip(self.collected_params, parameters):
            # 将收集到的参数数据复制到当前参数的数据
            param.data.copy_(c_param.data)
```

# `.\cogvideo-finetune\sat\vae_modules\regularizers.py`

```py
# 从 abc 模块导入抽象方法装饰器
from abc import abstractmethod
# 导入 Any 和 Tuple 类型注解
from typing import Any, Tuple

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能函数
import torch.nn.functional as F
# 从 PyTorch 导入神经网络模块
from torch import nn


# 定义对角高斯分布类
class DiagonalGaussianDistribution(object):
    # 初始化方法，接收参数和是否为确定性
    def __init__(self, parameters, deterministic=False):
        # 存储传入的参数
        self.parameters = parameters
        # 将参数分为均值和对数方差
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # 将对数方差限制在-30到20的范围内
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # 记录是否为确定性
        self.deterministic = deterministic
        # 计算标准差
        self.std = torch.exp(0.5 * self.logvar)
        # 计算方差
        self.var = torch.exp(self.logvar)
        # 如果是确定性，则标准差和方差设置为0
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    # 生成样本的方法
    def sample(self):
        # x = self.mean + self.std * torch.randn(self.mean.shape).to(
        #     device=self.parameters.device
        # )
        # 从均值和标准差生成样本
        x = self.mean + self.std * torch.randn_like(self.mean)
        # 返回生成的样本
        return x

    # 计算KL散度的方法
    def kl(self, other=None):
        # 如果是确定性，则返回0
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            # 如果没有提供其他分布，则计算与标准正态分布的KL散度
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                # 否则计算与另一个分布的KL散度
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    # 计算负对数似然的方法
    def nll(self, sample, dims=[1, 2, 3]):
        # 如果是确定性，则返回0
        if self.deterministic:
            return torch.Tensor([0.0])
        # 计算2π的对数
        logtwopi = np.log(2.0 * np.pi)
        # 计算负对数似然
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    # 返回模式（均值）的方法
    def mode(self):
        return self.mean


# 定义抽象正则化器类，继承自nn.Module
class AbstractRegularizer(nn.Module):
    # 初始化方法
    def __init__(self):
        super().__init__()

    # 前向传播的方法，需实现
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    # 获取可训练参数的抽象方法
    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


# 定义身份正则化器类，继承自抽象正则化器
class IdentityRegularizer(AbstractRegularizer):
    # 前向传播方法，返回输入和空字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, dict()

    # 获取可训练参数的方法，返回空生成器
    def get_trainable_parameters(self) -> Any:
        yield from ()


# 定义测量困惑度的函数
def measure_perplexity(predicted_indices: torch.Tensor, num_centroids: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # 评估聚类困惑度。当困惑度 == num_embeddings 时，所有聚类被完全均匀使用
    # 对预测索引进行独热编码并重塑为二维张量
    encodings = F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    # 计算平均概率
    avg_probs = encodings.mean(0)
    # 计算困惑度
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    # 计算使用的聚类数量
    cluster_use = torch.sum(avg_probs > 0)
    # 返回困惑度和聚类使用情况
    return perplexity, cluster_use
# 定义一个对角高斯正则化器类，继承自抽象正则化器
class DiagonalGaussianRegularizer(AbstractRegularizer):
    # 初始化方法，接受一个布尔参数 sample，默认为 True
    def __init__(self, sample: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 保存 sample 参数
        self.sample = sample

    # 获取可训练参数的方法，返回生成器
    def get_trainable_parameters(self) -> Any:
        # 生成一个空的生成器
        yield from ()

    # 前向传播方法，接收一个张量 z，返回一个张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 初始化一个空字典用于存储日志信息
        log = dict()
        # 创建一个对角高斯分布对象，基于输入张量 z
        posterior = DiagonalGaussianDistribution(z)
        # 根据 sample 参数决定如何获取样本
        if self.sample:
            # 从后验分布中采样
            z = posterior.sample()
        else:
            # 获取后验分布的众数
            z = posterior.mode()
        # 计算 KL 散度损失
        kl_loss = posterior.kl()
        # 对 KL 散度损失取平均
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # 将 KL 损失存储在日志字典中
        log["kl_loss"] = kl_loss
        # 返回处理后的张量和日志字典
        return z, log
```

# `.\cogvideo-finetune\sat\vae_modules\utils.py`

```py
# 导入所需的模块
import functools  # 引入 functools 模块，提供高阶函数
import importlib  # 引入 importlib 模块，用于动态导入模块
import os  # 引入 os 模块，提供与操作系统交互的功能
from functools import partial  # 从 functools 导入 partial，用于部分函数应用
from inspect import isfunction  # 从 inspect 导入 isfunction，用于检查对象是否为函数

import fsspec  # 导入 fsspec 库，用于文件系统规范
import numpy as np  # 导入 numpy 库并简化为 np，提供数组和数值计算功能
import torch  # 导入 PyTorch 库，提供深度学习功能
from PIL import Image, ImageDraw, ImageFont  # 从 PIL 导入图像处理相关模块
from safetensors.torch import load_file as load_safetensors  # 导入 safetensors 加载函数并重命名
import torch.distributed  # 导入 PyTorch 的分布式模块

# 初始化上下文并行组的变量
_CONTEXT_PARALLEL_GROUP = None  # 用于存储上下文并行组
_CONTEXT_PARALLEL_SIZE = None  # 用于存储上下文并行大小


def is_context_parallel_initialized():
    # 检查上下文并行组是否已初始化
    if _CONTEXT_PARALLEL_GROUP is None:
        return False  # 如果未初始化，返回 False
    else:
        return True  # 否则返回 True


def initialize_context_parallel(context_parallel_size):
    # 初始化上下文并行组
    global _CONTEXT_PARALLEL_GROUP  # 声明全局变量
    global _CONTEXT_PARALLEL_SIZE  # 声明全局变量

    # 确保上下文并行组尚未初始化
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    _CONTEXT_PARALLEL_SIZE = context_parallel_size  # 设置上下文并行大小

    rank = torch.distributed.get_rank()  # 获取当前进程的排名
    world_size = torch.distributed.get_world_size()  # 获取所有进程的总数

    # 根据上下文并行大小创建新的分组
    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)  # 获取当前分组的排名
        group = torch.distributed.new_group(ranks)  # 创建新的分组
        if rank in ranks:  # 如果当前排名在分组中
            _CONTEXT_PARALLEL_GROUP = group  # 设置全局上下文并行组
            break  # 退出循环


def get_context_parallel_group():
    # 获取当前上下文并行组
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"  # 确保已初始化

    return _CONTEXT_PARALLEL_GROUP  # 返回上下文并行组


def get_context_parallel_world_size():
    # 获取上下文并行的世界大小
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"  # 确保已初始化

    return _CONTEXT_PARALLEL_SIZE  # 返回上下文并行大小


def get_context_parallel_rank():
    # 获取当前上下文并行组的排名
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"  # 确保已初始化

    rank = torch.distributed.get_rank()  # 获取当前进程的排名
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE  # 计算上下文并行组排名
    return cp_rank  # 返回上下文并行组排名


def get_context_parallel_group_rank():
    # 获取当前上下文并行组的组排名
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"  # 确保已初始化

    rank = torch.distributed.get_rank()  # 获取当前进程的排名
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE  # 计算组排名

    return cp_group_rank  # 返回组排名


class SafeConv3d(torch.nn.Conv3d):
    # 自定义 3D 卷积类，继承自 torch.nn.Conv3d
    def forward(self, input):
        # 前向传播方法
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3  # 计算输入所需内存
        if memory_count > 2:  # 如果内存需求超过 2 GB
            kernel_size = self.kernel_size[0]  # 获取卷积核大小
            part_num = int(memory_count / 2) + 1  # 计算分块数量
            input_chunks = torch.chunk(input, part_num, dim=2)  # 将输入按维度 2 分块，格式为 NCTHW
            if kernel_size > 1:  # 如果卷积核大于 1
                input_chunks = [input_chunks[0]] + [  # 将第一块加入结果
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)  # 拼接相邻块
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []  # 初始化输出块列表
            for input_chunk in input_chunks:  # 遍历每个输入块
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))  # 调用父类的 forward 方法
            output = torch.cat(output_chunks, dim=2)  # 将输出块拼接
            return output  # 返回拼接后的输出
        else:
            return super(SafeConv3d, self).forward(input)  # 否则直接调用父类的 forward 方法


def disabled_train(self, mode=True):
    # 重写模型的 train 方法，确保训练/评估模式不再改变
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self  # 返回当前对象
# 定义一个函数，通过元组字符串返回其第一个元素
def get_string_from_tuple(s):
    # 尝试执行下面的代码块，处理可能出现的异常
    try:
        # 检查字符串是否以括号开头和结尾
        if s[0] == "(" and s[-1] == ")":
            # 将字符串转换为元组
            t = eval(s)
            # 检查 t 的类型是否为元组
            if type(t) == tuple:
                # 返回元组的第一个元素
                return t[0]
            else:
                pass
    # 捕获所有异常，防止程序崩溃
    except:
        pass
    # 如果不满足条件，返回原始字符串
    return s


# 定义一个函数，检查一个整数是否是 2 的幂
def is_power_of_two(n):
    """
    chat.openai.com/chat
    如果 n 是 2 的幂，返回 True；否则返回 False。

    函数 is_power_of_two 接受一个整数 n 作为输入，如果 n 是 2 的幂，则返回 True；否则返回 False。
    该函数首先检查 n 是否小于或等于 0。如果 n 小于或等于 0，则它不能是 2 的幂，因此函数返回 False。
    如果 n 大于 0，函数通过使用 n 和 n-1 之间的按位与操作来检查 n 是否是 2 的幂。
    如果 n 是 2 的幂，它的二进制表示中只有一个位被设置为 1。当我们从 2 的幂中减去 1 时，所有右侧的位变为 1，该位本身变为 0。
    因此，当我们对 n 和 n-1 进行按位与操作时，如果 n 是 2 的幂，结果为 0，否则为非零值。
    因此，如果按位与操作的结果为 0，则 n 是 2 的幂，函数返回 True；否则返回 False。
    """
    # 检查 n 是否小于等于 0
    if n <= 0:
        # 如果小于等于 0，返回 False
        return False
    # 返回 n 和 n-1 进行按位与操作的结果是否为 0
    return (n & (n - 1)) == 0


# 定义一个函数，实现自动类型转换的功能
def autocast(f, enabled=True):
    # 定义一个内部函数，执行自动类型转换
    def do_autocast(*args, **kwargs):
        # 使用自动类型转换上下文管理器
        with torch.cuda.amp.autocast(
            enabled=enabled,  # 是否启用自动类型转换
            dtype=torch.get_autocast_gpu_dtype(),  # 获取 GPU 的自动类型转换数据类型
            cache_enabled=torch.is_autocast_cache_enabled(),  # 检查是否启用缓存
        ):
            # 执行传入的函数 f，并返回结果
            return f(*args, **kwargs)

    # 返回内部函数
    return do_autocast


# 定义一个函数，从配置中加载部分对象
def load_partial_from_config(config):
    # 使用部分应用函数返回目标对象，带有指定的参数
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


# 定义一个函数，将文本以图像形式记录
def log_txt_as_img(wh, xc, size=10):
    # wh 是一个包含 (宽度, 高度) 的元组
    # xc 是要绘制的字幕列表
    b = len(xc)  # 获取字幕列表的长度
    txts = list()  # 初始化一个空列表，用于存储图像数据
    # 遍历每个字幕
    for bi in range(b):
        # 创建一个新的白色背景图像
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)  # 创建可用于绘图的对象
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)  # 加载指定字体
        nc = int(40 * (wh[0] / 256))  # 根据图像宽度计算每行的字符数
        # 检查当前字幕是否为列表
        if isinstance(xc[bi], list):
            text_seq = xc[bi][0]  # 获取列表中的第一个元素
        else:
            text_seq = xc[bi]  # 直接使用字幕

        # 将文本序列分行，每行不超过 nc 个字符
        lines = "\n".join(text_seq[start : start + nc] for start in range(0, len(text_seq), nc))

        try:
            # 在图像上绘制文本
            draw.text((0, 0), lines, fill="black", font=font)
        # 捕获文本编码错误
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")  # 输出错误信息

        # 将图像数据转换为 NumPy 数组并进行归一化
        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)  # 将处理后的图像数据添加到列表
    # 将列表中的图像数据堆叠成一个 NumPy 数组
    txts = np.stack(txts)
    # 将 NumPy 数组转换为 PyTorch 张量
    txts = torch.tensor(txts)
    # 返回最终的张量
    return txts


# 定义一个部分类，允许使用部分应用创建新类
def partialclass(cls, *args, **kwargs):
    # 定义一个新类，继承自原始类
    class NewCls(cls):
        # 使用部分应用替换原始类的初始化方法
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    # 返回新类
    return NewCls
# 根据给定路径生成绝对路径
def make_path_absolute(path):
    # 解析路径并获取文件系统和路径
    fs, p = fsspec.core.url_to_fs(path)
    # 如果协议是文件，则返回绝对路径
    if fs.protocol == "file":
        return os.path.abspath(p)
    # 否则返回原始路径
    return path


# 判断输入是否为一个特定形状的张量
def ismap(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度和通道数
    return (len(x.shape) == 4) and (x.shape[1] > 3)


# 判断输入是否为图像张量
def isimage(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度和通道数
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


# 判断输入是否为热图张量
def isheatmap(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度
    return x.ndim == 2


# 判断输入是否为邻居张量
def isneighbors(x):
    # 检查 x 是否为张量
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量的维度和通道数
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


# 检查输入是否存在
def exists(x):
    # 返回 x 是否不为 None
    return x is not None


# 将张量的维度扩展到与另一个张量相同
def expand_dims_like(x, y):
    # 在 x 的维度与 y 不同的情况下循环扩展
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    # 返回扩展后的张量
    return x


# 返回给定值或默认值
def default(val, d):
    # 如果 val 存在，返回 val
    if exists(val):
        return val
    # 如果 d 是函数，则调用并返回其结果，否则返回 d
    return d() if isfunction(d) else d


# 计算张量的扁平化均值
def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    对所有非批处理维度进行均值计算。
    """
    # 计算并返回指定维度的均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 统计模型参数数量
def count_params(model, verbose=False):
    # 计算模型所有参数的总数
    total_params = sum(p.numel() for p in model.parameters())
    # 如果 verbose 为真，打印参数数量
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    # 返回总参数数量
    return total_params


# 根据配置实例化对象
def instantiate_from_config(config):
    # 检查配置中是否包含 'target' 键
    if not "target" in config:
        # 返回 None，如果配置是特定字符串
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        # 否则抛出异常
        raise KeyError("Expected key `target` to instantiate.")
    # 从配置中获取目标对象并返回实例化结果
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# 从字符串获取对象
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    # 将字符串分解为模块和类
    module, cls = string.rsplit(".", 1)
    # 如果需要，失效缓存
    if invalidate_cache:
        importlib.invalidate_caches()
    # 如果需要重新加载模块
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 返回模块中的类对象
    return getattr(importlib.import_module(module, package=None), cls)


# 在张量末尾追加一个零
def append_zero(x):
    # 将 x 与一个新零张量连接
    return torch.cat([x, x.new_zeros([1])])


# 将张量扩展到目标维度
def append_dims(x, target_dims):
    """将维度附加到张量末尾，直到它具有 target_dims 维度。"""
    # 计算需要附加的维度数量
    dims_to_append = target_dims - x.ndim
    # 如果目标维度小于当前维度，抛出异常
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    # 返回附加维度后的张量
    return x[(...,) + (None,) * dims_to_append]


# 从配置加载模型
def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    # 打印加载模型的检查点信息
    print(f"Loading model from {ckpt}")
    # 如果检查点是以 'ckpt' 结尾，加载相应的数据
    if ckpt.endswith("ckpt"):
        pl_sd = torch.load(ckpt, map_location="cpu")
        # 如果包含全局步骤信息，打印
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    # 如果检查点是以 'safetensors' 结尾，加载相应的数据
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    # 否则抛出未实现异常
    else:
        raise NotImplementedError
    # 根据配置文件实例化模型
        model = instantiate_from_config(config.model)
    
        # 加载状态字典（权重）到模型中，返回缺失和意外的键
        m, u = model.load_state_dict(sd, strict=False)
    
        # 如果有缺失的键且需要详细输出，则打印缺失的键
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        # 如果有意外的键且需要详细输出，则打印意外的键
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    
        # 如果需要冻结模型参数，禁止其更新
        if freeze:
            for param in model.parameters():
                # 设置参数的 requires_grad 属性为 False，停止梯度计算
                param.requires_grad = False
    
        # 将模型设置为评估模式，禁用训练时的特性（如 Dropout）
        model.eval()
        # 返回实例化后的模型
        return model
# 获取配置文件的路径
def get_configs_path() -> str:
    """
    获取 `configs` 目录。
    对于工作副本，这是存储库根目录下的目录，
    而对于已安装副本，则在 `sgm` 包中（见 pyproject.toml）。
    """
    # 获取当前文件所在目录的路径
    this_dir = os.path.dirname(__file__)
    # 定义可能的配置目录路径
    candidates = (
        os.path.join(this_dir, "configs"),  # 当前目录下的 configs
        os.path.join(this_dir, "..", "configs"),  # 上级目录下的 configs
    )
    # 遍历候选目录
    for candidate in candidates:
        # 将候选路径转换为绝对路径
        candidate = os.path.abspath(candidate)
        # 检查该路径是否为目录
        if os.path.isdir(candidate):
            # 如果是目录，返回该路径
            return candidate
    # 如果没有找到任何目录，抛出文件未找到异常
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")

# 获取嵌套属性的函数
def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    """
    将返回递归获取属性调用的结果。
    例如：
        a.b.c
        = getattr(getattr(a, "b"), "c")
        = get_nested_attribute(a, "b.c")
    如果属性调用的任何部分是整数 x，并且当前 obj 为 a，将
    尝试首先调用 a[x] 而不是 a.x。
    """
    # 将属性路径以 "." 分割为列表
    attributes = attribute_path.split(".")
    # 如果设置了深度，截取属性列表
    if depth is not None and depth > 0:
        attributes = attributes[:depth]
    # 确保至少选择了一个属性
    assert len(attributes) > 0, "At least one attribute should be selected"
    # 初始化当前属性为传入对象
    current_attribute = obj
    current_key = None
    # 遍历每一层的属性
    for level, attribute in enumerate(attributes):
        # 生成当前的属性路径字符串
        current_key = ".".join(attributes[: level + 1])
        try:
            # 尝试将属性转换为整数
            id_ = int(attribute)
            # 如果成功，将当前属性设置为索引访问的结果
            current_attribute = current_attribute[id_]
        except ValueError:
            # 否则使用 getattr 获取属性
            current_attribute = getattr(current_attribute, attribute)

    # 返回当前属性和当前键，或仅返回当前属性
    return (current_attribute, current_key) if return_key else current_attribute

# 定义检查点函数
def checkpoint(func, inputs, params, flag):
    """
    在不缓存中间激活的情况下评估函数，从而减少内存，
    代价是反向传播时增加额外计算。
    :param func: 要评估的函数。
    :param inputs: 要传递给 `func` 的参数序列。
    :param params: `func` 依赖的参数序列，但不作为参数显式传递。
    :param flag: 如果为 False，则禁用梯度检查点。
    """
    # 如果启用标志
    if flag:
        # 将输入和参数组合成元组
        args = tuple(inputs) + tuple(params)
        # 使用 CheckpointFunction 应用函数并返回结果
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        # 否则直接调用函数并返回结果
        return func(*inputs)

# 定义检查点功能的类
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    # 前向计算方法
    def forward(ctx, run_function, length, *args):
        # 保存要运行的函数
        ctx.run_function = run_function
        # 保存输入张量
        ctx.input_tensors = list(args[:length])
        # 保存输入参数
        ctx.input_params = list(args[length:])
        # 保存 GPU 自动混合精度的设置
        ctx.gpu_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(),
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
        # 在无梯度上下文中执行函数
        with torch.no_grad():
            # 获取输出张量
            output_tensors = ctx.run_function(*ctx.input_tensors)
        # 返回输出张量
        return output_tensors

    @staticmethod
    # 定义反向传播函数，接收上下文和输出梯度作为参数
    def backward(ctx, *output_grads):
        # 对输入张量进行detach操作，并设置requires_grad为True，以便计算梯度
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        # 启用梯度计算，并在自动混合精度模式下执行代码
        with torch.enable_grad(), torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # 修复一个bug：第一个操作会修改Tensor存储，这对已detach的Tensors不允许
            # 创建输入张量的浅拷贝，确保原始张量不被修改
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            # 调用运行函数，传入浅拷贝的输入张量，获得输出张量
            output_tensors = ctx.run_function(*shallow_copies)
        # 计算输入张量和参数的梯度，output_grads作为输出梯度
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        # 删除上下文中的输入张量以释放内存
        del ctx.input_tensors
        # 删除上下文中的输入参数以释放内存
        del ctx.input_params
        # 删除输出张量以释放内存
        del output_tensors
        # 返回None和输入梯度的元组
        return (None, None) + input_grads
```