# `.\diffusers\pipelines\wuerstchen\modeling_wuerstchen_prior.py`

```py
# 版权声明，说明文件的版权所有者和许可证信息
# Copyright (c) 2023 Dominic Rampas MIT License
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 许可证许可使用本文件
# 仅在遵循许可证的情况下使用此文件
# 可以在此获取许可证的副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下按 "现状" 基础提供
# 不提供任何形式的担保或条件
# 查看许可证以获取特定语言的权限和限制

# 导入数学库
import math
# 从 typing 模块导入字典和联合类型
from typing import Dict, Union

# 导入 PyTorch 及其神经网络模块
import torch
import torch.nn as nn

# 导入配置工具和适配器相关的类
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
# 导入注意力处理器相关的类
from ...models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
# 导入模型相关的基类
from ...models.modeling_utils import ModelMixin
# 导入工具函数以检查 PyTorch 版本
from ...utils import is_torch_version
# 导入模型组件
from .modeling_wuerstchen_common import AttnBlock, ResBlock, TimestepBlock, WuerstchenLayerNorm

# 定义 WuerstchenPrior 类，继承自多个基类
class WuerstchenPrior(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    # 设置 UNet 名称为 "prior"
    unet_name = "prior"
    # 启用梯度检查点功能
    _supports_gradient_checkpointing = True

    # 注册初始化方法，定义类的构造函数
    @register_to_config
    def __init__(self, c_in=16, c=1280, c_cond=1024, c_r=64, depth=16, nhead=16, dropout=0.1):
        # 调用父类的构造函数
        super().__init__()

        # 设置压缩通道数
        self.c_r = c_r
        # 定义一个卷积层用于输入到中间通道的映射
        self.projection = nn.Conv2d(c_in, c, kernel_size=1)
        # 定义条件映射层，由两个线性层和一个激活函数组成
        self.cond_mapper = nn.Sequential(
            nn.Linear(c_cond, c),  # 将条件输入映射到中间通道
            nn.LeakyReLU(0.2),      # 应用 Leaky ReLU 激活函数
            nn.Linear(c, c),        # 再次映射到中间通道
        )

        # 创建一个模块列表用于存储多个块
        self.blocks = nn.ModuleList()
        # 根据深度参数添加多个残差块、时间步块和注意力块
        for _ in range(depth):
            self.blocks.append(ResBlock(c, dropout=dropout))  # 添加残差块
            self.blocks.append(TimestepBlock(c, c_r))         # 添加时间步块
            self.blocks.append(AttnBlock(c, c, nhead, self_attn=True, dropout=dropout))  # 添加注意力块
        # 定义输出层，由归一化层和卷积层组成
        self.out = nn.Sequential(
            WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6),  # 归一化
            nn.Conv2d(c, c_in * 2, kernel_size=1),  # 输出卷积层
        )

        # 默认禁用梯度检查点
        self.gradient_checkpointing = False
        # 设置默认的注意力处理器
        self.set_default_attn_processor()

    # 定义一个只读属性
    @property
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制的属性
    # 定义一个返回注意力处理器字典的方法
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 的注意力处理器: 一个字典，包含模型中使用的所有注意力处理器，并以其权重名称索引。
        """
        # 初始化一个空字典以存储处理器
        processors = {}
    
        # 定义递归添加处理器的内部函数
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块具有获取处理器的方法，则将其添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用以添加子模块的处理器
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            # 返回更新后的处理器字典
            return processors
    
        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用内部函数以添加所有子模块的处理器
            fn_recursive_add_processors(name, module, processors)
    
        # 返回所有处理器的字典
        return processors
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制的方法
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的注意力处理器。
    
        参数:
            processor (`dict` of `AttentionProcessor` 或仅 `AttentionProcessor`):
                实例化的处理器类或将作为所有 `Attention` 层的处理器设置的处理器类字典。
    
                如果 `processor` 是一个字典，则键需要定义对应的交叉注意力处理器的路径。
                在设置可训练的注意力处理器时，强烈建议这样做。
        """
        # 计算当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 检查传入的处理器字典的大小是否与注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器的字典，但处理器的数量 {len(processor)} 与注意力层的数量: {count} 不匹配。"
                f" 请确保传入 {count} 个处理器类。"
            )
    
        # 定义递归设置处理器的内部函数
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块具有设置处理器的方法，则设置处理器
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，则直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历子模块
            for sub_name, child in module.named_children():
                # 递归调用以设置子模块的处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用内部函数以设置所有子模块的处理器
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor 复制的方法
    # 定义一个方法，用于设置默认的注意力处理器
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器，并设置默认的注意力实现。
        """
        # 检查所有注意力处理器是否属于新增的键值注意力处理器
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，使用新增的键值注意力处理器
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否属于交叉注意力处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 如果是，使用标准的注意力处理器
            processor = AttnProcessor()
        else:
            # 否则，抛出一个值错误，说明无法设置默认注意力处理器
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        # 调用设置方法，将选择的处理器应用于当前对象
        self.set_attn_processor(processor)

    # 定义一个私有方法，用于设置梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 将梯度检查点的值设置为传入的布尔值
        self.gradient_checkpointing = value

    # 定义生成位置嵌入的方法
    def gen_r_embedding(self, r, max_positions=10000):
        # 将输入的 r 乘以最大位置数
        r = r * max_positions
        # 计算嵌入的半维度
        half_dim = self.c_r // 2
        # 计算嵌入的缩放因子
        emb = math.log(max_positions) / (half_dim - 1)
        # 创建一个张量，并根据半维度生成指数嵌入
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        # 根据 r 生成最终的嵌入
        emb = r[:, None] * emb[None, :]
        # 将正弦和余弦嵌入拼接在一起
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        # 如果 c_r 是奇数，则进行零填充
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode="constant")
        # 返回嵌入，确保数据类型与 r 一致
        return emb.to(dtype=r.dtype)
    # 定义前向传播函数，接收输入张量 x、条件 r 和 c
        def forward(self, x, r, c):
            # 保存输入张量的原始值
            x_in = x
            # 对输入张量进行投影处理
            x = self.projection(x)
            # 将条件 c 转换为嵌入表示
            c_embed = self.cond_mapper(c)
            # 生成条件 r 的嵌入表示
            r_embed = self.gen_r_embedding(r)
    
            # 如果处于训练模式并且开启梯度检查点
            if self.training and self.gradient_checkpointing:
    
                # 创建自定义前向传播函数的辅助函数
                def create_custom_forward(module):
                    # 定义接受任意输入的自定义前向函数
                    def custom_forward(*inputs):
                        return module(*inputs)
    
                    return custom_forward
    
                # 检查 PyTorch 版本是否大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 遍历所有块进行处理
                    for block in self.blocks:
                        # 如果块是注意力块
                        if isinstance(block, AttnBlock):
                            # 使用检查点来保存内存
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, c_embed, use_reentrant=False
                            )
                        # 如果块是时间步块
                        elif isinstance(block, TimestepBlock):
                            # 使用检查点来保存内存
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block), x, r_embed, use_reentrant=False
                            )
                        else:
                            # 处理其他类型的块
                            x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, use_reentrant=False)
                else:
                    # 对于旧版本的 PyTorch
                    for block in self.blocks:
                        # 如果块是注意力块
                        if isinstance(block, AttnBlock):
                            # 使用检查点来保存内存
                            x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, c_embed)
                        # 如果块是时间步块
                        elif isinstance(block, TimestepBlock):
                            # 使用检查点来保存内存
                            x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, r_embed)
                        else:
                            # 处理其他类型的块
                            x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x)
            else:
                # 如果不在训练模式下
                for block in self.blocks:
                    # 如果块是注意力块
                    if isinstance(block, AttnBlock):
                        # 直接进行前向传播
                        x = block(x, c_embed)
                    # 如果块是时间步块
                    elif isinstance(block, TimestepBlock):
                        # 直接进行前向传播
                        x = block(x, r_embed)
                    else:
                        # 处理其他类型的块
                        x = block(x)
            # 将输出分割为两个部分 a 和 b
            a, b = self.out(x).chunk(2, dim=1)
            # 返回经过归一化处理的结果
            return (x_in - a) / ((1 - b).abs() + 1e-5)
```