# `.\diffusers\models\lora.py`

```py
# 版权信息，指明文件的版权所有者和保留权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 按照 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则不得使用本文件。
# 可在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，
# 否则根据许可证分发的软件在“按原样”基础上提供，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证，以获取有关权限和
# 限制的具体条款。

# 重要提示：                                                      #
###################################################################
# ----------------------------------------------------------------#
# 此文件已被弃用，将很快删除                                   #
# （一旦 PEFT 成为 LoRA 的必需依赖项）                          #
# ----------------------------------------------------------------#
###################################################################

from typing import Optional, Tuple, Union  # 导入可选类型、元组和联合类型以用于类型注解

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from torch import nn  # 从 PyTorch 导入神经网络模块

from ..utils import deprecate, logging  # 从上级目录导入工具函数 deprecate 和 logging
from ..utils.import_utils import is_transformers_available  # 导入检查 transformers 库可用性的函数


# 如果 transformers 库可用，则导入相关模型
if is_transformers_available():
    from transformers import CLIPTextModel, CLIPTextModelWithProjection  # 导入 CLIP 文本模型及其变体


logger = logging.get_logger(__name__)  # 创建一个记录器实例，用于日志记录，禁用 pylint 的名称检查


def text_encoder_attn_modules(text_encoder):
    attn_modules = []  # 初始化一个空列表，用于存储注意力模块

    # 检查文本编码器是否为 CLIPTextModel 或 CLIPTextModelWithProjection 的实例
    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        # 遍历编码器层，收集每一层的自注意力模块
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"  # 构造注意力模块的名称
            mod = layer.self_attn  # 获取当前层的自注意力模块
            attn_modules.append((name, mod))  # 将名称和模块元组添加到列表中
    else:
        # 如果文本编码器不是预期的类型，抛出值错误
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules  # 返回注意力模块的列表


def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []  # 初始化一个空列表，用于存储 MLP 模块

    # 检查文本编码器是否为 CLIPTextModel 或 CLIPTextModelWithProjection 的实例
    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        # 遍历编码器层，收集每一层的 MLP 模块
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            mlp_mod = layer.mlp  # 获取当前层的 MLP 模块
            name = f"text_model.encoder.layers.{i}.mlp"  # 构造 MLP 模块的名称
            mlp_modules.append((name, mlp_mod))  # 将名称和模块元组添加到列表中
    else:
        # 如果文本编码器不是预期的类型，抛出值错误
        raise ValueError(f"do not know how to get mlp modules for: {text_encoder.__class__.__name__}")

    return mlp_modules  # 返回 MLP 模块的列表


def adjust_lora_scale_text_encoder(text_encoder, lora_scale: float = 1.0):
    # 遍历文本编码器中的注意力模块
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        # 检查当前注意力模块的查询投影是否为 PatchedLoraProjection 实例
        if isinstance(attn_module.q_proj, PatchedLoraProjection):
            attn_module.q_proj.lora_scale = lora_scale  # 调整查询投影的 Lora 缩放因子
            attn_module.k_proj.lora_scale = lora_scale  # 调整键投影的 Lora 缩放因子
            attn_module.v_proj.lora_scale = lora_scale  # 调整值投影的 Lora 缩放因子
            attn_module.out_proj.lora_scale = lora_scale  # 调整输出投影的 Lora 缩放因子
    # 遍历文本编码器中的 MLP 模块
        for _, mlp_module in text_encoder_mlp_modules(text_encoder):
            # 检查当前模块的 fc1 层是否为 PatchedLoraProjection 类型
            if isinstance(mlp_module.fc1, PatchedLoraProjection):
                # 设置 fc1 层的 lora_scale 属性
                mlp_module.fc1.lora_scale = lora_scale
                # 设置 fc2 层的 lora_scale 属性
                mlp_module.fc2.lora_scale = lora_scale
# 定义一个名为 PatchedLoraProjection 的类，继承自 PyTorch 的 nn.Module
class PatchedLoraProjection(torch.nn.Module):
    # 初始化方法，接受多个参数以设置 LoraProjection
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        # 设置弃用警告信息
        deprecation_message = "Use of `PatchedLoraProjection` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        # 调用 deprecate 函数记录弃用信息
        deprecate("PatchedLoraProjection", "1.0.0", deprecation_message)

        # 调用父类的初始化方法
        super().__init__()
        # 从 lora 模块导入 LoRALinearLayer 类
        from ..models.lora import LoRALinearLayer

        # 保存传入的常规线性层
        self.regular_linear_layer = regular_linear_layer

        # 获取常规线性层的设备信息
        device = self.regular_linear_layer.weight.device

        # 如果未指定数据类型，则使用常规线性层的权重数据类型
        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        # 创建 LoRALinearLayer 实例
        self.lora_linear_layer = LoRALinearLayer(
            self.regular_linear_layer.in_features,
            self.regular_linear_layer.out_features,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype,
            rank=rank,
        )

        # 保存 LoRA 的缩放因子
        self.lora_scale = lora_scale

    # 重写 PyTorch 的 state_dict 方法以确保仅保存 'regular_linear_layer' 权重
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # 如果没有 LoRA 线性层，返回常规线性层的状态字典
        if self.lora_linear_layer is None:
            return self.regular_linear_layer.state_dict(
                *args, destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        # 否则调用父类的 state_dict 方法
        return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)

    # 定义一个融合 LoRA 权重的方法
    def _fuse_lora(self, lora_scale=1.0, safe_fusing=False):
        # 如果没有 LoRA 线性层，则直接返回
        if self.lora_linear_layer is None:
            return

        # 获取常规线性层的权重数据类型和设备
        dtype, device = self.regular_linear_layer.weight.data.dtype, self.regular_linear_layer.weight.data.device

        # 将常规线性层的权重转换为浮点类型
        w_orig = self.regular_linear_layer.weight.data.float()
        # 将 LoRA 层的上权重转换为浮点类型
        w_up = self.lora_linear_layer.up.weight.data.float()
        # 将 LoRA 层的下权重转换为浮点类型
        w_down = self.lora_linear_layer.down.weight.data.float()

        # 如果 network_alpha 不为 None，则调整上权重
        if self.lora_linear_layer.network_alpha is not None:
            w_up = w_up * self.lora_linear_layer.network_alpha / self.lora_linear_layer.rank

        # 计算融合后的权重
        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        # 如果安全融合并且融合权重中包含 NaN，抛出异常
        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        # 更新常规线性层的权重数据
        self.regular_linear_layer.weight.data = fused_weight.to(device=device, dtype=dtype)

        # 将 LoRA 线性层设为 None，表示已经融合
        self.lora_linear_layer = None

        # 将上、下权重矩阵转移到 CPU 以节省内存
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        # 更新 LoRA 的缩放因子
        self.lora_scale = lora_scale
    # 定义解融合 Lora 的私有方法
    def _unfuse_lora(self):
        # 检查 w_up 和 w_down 属性是否存在且不为 None
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            # 如果任一属性为 None，则直接返回
            return

        # 获取常规线性层的权重数据
        fused_weight = self.regular_linear_layer.weight.data
        # 保存权重的数据类型和设备信息
        dtype, device = fused_weight.dtype, fused_weight.device

        # 将 w_up 转换为目标设备并转为浮点类型
        w_up = self.w_up.to(device=device).float()
        # 将 w_down 转换为目标设备并转为浮点类型
        w_down = self.w_down.to(device).float()

        # 计算未融合的权重，通过从融合权重中减去 Lora 的贡献
        unfused_weight = fused_weight.float() - (self.lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        # 将未融合的权重赋值回常规线性层
        self.regular_linear_layer.weight.data = unfused_weight.to(device=device, dtype=dtype)

        # 清空 w_up 和 w_down 属性
        self.w_up = None
        self.w_down = None

    # 定义前向传播方法
    def forward(self, input):
        # 如果 lora_scale 为 None，则设置为 1.0
        if self.lora_scale is None:
            self.lora_scale = 1.0
        # 如果 lora_linear_layer 为 None，则直接返回常规线性层的输出
        if self.lora_linear_layer is None:
            return self.regular_linear_layer(input)
        # 返回常规线性层的输出加上 Lora 的贡献
        return self.regular_linear_layer(input) + (self.lora_scale * self.lora_linear_layer(input))
# 定义一个用于 LoRA 的线性层，继承自 nn.Module
class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    # 初始化方法，定义输入输出特征和其他参数
    def __init__(
        self,
        in_features: int,  # 输入特征数量
        out_features: int,  # 输出特征数量
        rank: int = 4,  # LoRA 层的秩，默认为 4
        network_alpha: Optional[float] = None,  # 用于稳定学习的网络 alpha，默认为 None
        device: Optional[Union[torch.device, str]] = None,  # 权重使用的设备，默认为 None
        dtype: Optional[torch.dtype] = None,  # 权重使用的数据类型，默认为 None
    ):
        super().__init__()  # 调用父类的初始化方法

        # 弃用提示消息，提醒用户切换到 PEFT 后端
        deprecation_message = "Use of `LoRALinearLayer` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        deprecate("LoRALinearLayer", "1.0.0", deprecation_message)  # 记录弃用信息

        # 定义向下线性层，不使用偏置
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        # 定义向上线性层，不使用偏置
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # 将网络 alpha 值赋给实例变量
        self.network_alpha = network_alpha
        self.rank = rank  # 保存秩
        self.out_features = out_features  # 保存输出特征数量
        self.in_features = in_features  # 保存输入特征数量

        # 使用正态分布初始化向下权重
        nn.init.normal_(self.down.weight, std=1 / rank)
        # 将向上权重初始化为零
        nn.init.zeros_(self.up.weight)

    # 前向传播方法，接受隐藏状态并返回处理后的结果
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype  # 保存输入数据类型
        dtype = self.down.weight.dtype  # 获取向下层权重的数据类型

        # 通过向下层处理隐藏状态
        down_hidden_states = self.down(hidden_states.to(dtype))
        # 通过向上层处理向下层输出
        up_hidden_states = self.up(down_hidden_states)

        # 如果网络 alpha 不为 None，则调整向上层输出
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        # 返回与原始数据类型相同的输出
        return up_hidden_states.to(orig_dtype)


# 定义一个用于 LoRA 的卷积层，继承自 nn.Module
class LoRAConv2dLayer(nn.Module):
    r"""
    A convolutional layer that is used with LoRA.
    # 参数说明
    Parameters:
        in_features (`int`):  # 输入特征的数量
            Number of input features.  # 输入特征的数量
        out_features (`int`):  # 输出特征的数量
            Number of output features.  # 输出特征的数量
        rank (`int`, `optional`, defaults to 4):  # LoRA 层的秩，默认为 4
            The rank of the LoRA layer.  # LoRA 层的秩
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):  # 卷积核的大小，默认为 (1, 1)
            The kernel size of the convolution.  # 卷积核的大小
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):  # 卷积的步幅，默认为 (1, 1)
            The stride of the convolution.  # 卷积的步幅
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):  # 卷积的填充方式，默认为 0
            The padding of the convolution.  # 卷积的填充方式
        network_alpha (`float`, `optional`, defaults to `None`):  # 网络 alpha 的值，用于稳定学习，防止下溢
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See  # 与 kohya-ss 训练脚本中的 `--network_alpha` 选项含义相同
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning  # 参考链接

    # 初始化方法
    def __init__(
        self,
        in_features: int,  # 输入特征数量
        out_features: int,  # 输出特征数量
        rank: int = 4,  # LoRA 层的秩，默认为 4
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),  # 卷积核大小，默认为 (1, 1)
        stride: Union[int, Tuple[int, int]] = (1, 1),  # 卷积步幅，默认为 (1, 1)
        padding: Union[int, Tuple[int, int], str] = 0,  # 卷积填充，默认为 0
        network_alpha: Optional[float] = None,  # 网络 alpha 的值，默认为 None
    ):
        super().__init__()  # 调用父类的初始化方法

        # 弃用警告信息，提示用户切换到 PEFT 后端
        deprecation_message = "Use of `LoRAConv2dLayer` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        deprecate("LoRAConv2dLayer", "1.0.0", deprecation_message)  # 发出弃用警告

        # 定义下卷积层，输入为 in_features，输出为 rank，使用指定的卷积参数
        self.down = nn.Conv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # 根据官方 kohya_ss 训练器，向上卷积层的卷积核大小始终固定
        # # 参考链接: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        # 定义上卷积层，输入为 rank，输出为 out_features，使用固定的卷积核大小 (1, 1)
        self.up = nn.Conv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # 保存网络 alpha 值，与训练脚本中的相同含义
        # 参考链接: https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha  # 设置网络 alpha 值
        self.rank = rank  # 设置秩

        # 初始化下卷积层的权重为均值为 0，标准差为 1/rank 的正态分布
        nn.init.normal_(self.down.weight, std=1 / rank)
        # 初始化上卷积层的权重为 0
        nn.init.zeros_(self.up.weight)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数
        orig_dtype = hidden_states.dtype  # 保存输入张量的原始数据类型
        dtype = self.down.weight.dtype  # 获取下卷积层权重的数据类型

        # 将输入的隐状态张量通过下卷积层
        down_hidden_states = self.down(hidden_states.to(dtype))
        # 将下卷积层的输出通过上卷积层
        up_hidden_states = self.up(down_hidden_states)

        # 如果 network_alpha 不为 None，则进行缩放
        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank  # 根据 network_alpha 进行缩放

        # 返回转换回原始数据类型的输出张量
        return up_hidden_states.to(orig_dtype)  # 返回最终输出
# 定义一个可以与 LoRA 兼容的卷积层，继承自 nn.Conv2d
class LoRACompatibleConv(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    # 初始化方法，接受可变数量的参数，lora_layer 为可选参数，其他参数通过 kwargs 接收
    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        # 设置弃用消息，提示用户切换到 PEFT 后端
        deprecation_message = "Use of `LoRACompatibleConv` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        # 调用弃用函数，记录此类的弃用信息
        deprecate("LoRACompatibleConv", "1.0.0", deprecation_message)

        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 将 lora_layer 赋值给实例变量
        self.lora_layer = lora_layer

    # 设置 lora_layer 的方法
    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        # 设置弃用消息，提示用户切换到 PEFT 后端
        deprecation_message = "Use of `set_lora_layer()` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        # 调用弃用函数，记录此方法的弃用信息
        deprecate("set_lora_layer", "1.0.0", deprecation_message)

        # 将传入的 lora_layer 赋值给实例变量
        self.lora_layer = lora_layer

    # 融合 LoRA 权重的方法
    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        # 如果 lora_layer 为 None，直接返回
        if self.lora_layer is None:
            return

        # 获取当前权重的数据类型和设备
        dtype, device = self.weight.data.dtype, self.weight.data.device

        # 将权重转换为浮点型
        w_orig = self.weight.data.float()
        # 获取 lora_layer 的上升和下降权重，并转换为浮点型
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        # 如果 network_alpha 不为 None，调整上升权重
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        # 进行矩阵乘法，融合上升和下降权重
        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        # 将融合的结果调整为原始权重的形状
        fusion = fusion.reshape((w_orig.shape))
        # 计算最终融合权重
        fused_weight = w_orig + (lora_scale * fusion)

        # 如果安全融合为 True，检查融合权重中是否有 NaN 值
        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        # 将融合后的权重赋值回实例的权重，保持设备和数据类型
        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # 融合后可以删除 lora_layer
        self.lora_layer = None

        # 将上升和下降矩阵转移到 CPU，以减少内存占用
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        # 存储 lora_scale
        self._lora_scale = lora_scale

    # 解融合 LoRA 权重的方法
    def _unfuse_lora(self):
        # 检查 w_up 和 w_down 是否存在
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        # 获取当前融合权重
        fused_weight = self.weight.data
        # 获取当前权重的数据类型和设备
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        # 将 w_up 和 w_down 转移到正确的设备并转换为浮点型
        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        # 进行矩阵乘法，重新计算未融合权重
        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        # 将融合结果调整为融合权重的形状
        fusion = fusion.reshape((fused_weight.shape))
        # 计算最终的未融合权重
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        # 更新实例的权重
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        # 清空 w_up 和 w_down
        self.w_up = None
        self.w_down = None
    # 定义前向传播函数，接收隐藏状态和缩放因子，返回张量
    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # 检查填充模式是否不是“零”，若是则进行相应填充
        if self.padding_mode != "zeros":
            # 对隐藏状态进行填充，使用反向填充参数和指定的填充模式
            hidden_states = F.pad(hidden_states, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            # 设置填充为 (0, 0)
            padding = (0, 0)
        else:
            # 使用类中的填充属性
            padding = self.padding
    
        # 进行二维卷积操作，返回卷积结果
        original_outputs = F.conv2d(
            hidden_states, self.weight, self.bias, self.stride, padding, self.dilation, self.groups
        )
    
        # 如果 Lora 层不存在，则返回卷积结果
        if self.lora_layer is None:
            return original_outputs
        else:
            # 否则，将卷积结果与 Lora 层的结果按比例相加并返回
            return original_outputs + (scale * self.lora_layer(hidden_states))
# 定义一个兼容 LoRA 的线性层，继承自 nn.Linear
class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    # 初始化方法，接收参数并可选传入 LoRA 层
    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        # 定义弃用提示信息，建议用户切换到 PEFT 后端
        deprecation_message = "Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        # 调用弃用函数提示用户
        deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)

        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        # 设置 LoRA 层
        self.lora_layer = lora_layer

    # 设置 LoRA 层的方法
    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        # 定义弃用提示信息，建议用户切换到 PEFT 后端
        deprecation_message = "Use of `set_lora_layer()` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        # 调用弃用函数提示用户
        deprecate("set_lora_layer", "1.0.0", deprecation_message)
        # 设置 LoRA 层
        self.lora_layer = lora_layer

    # 融合 LoRA 权重的方法
    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        # 如果没有 LoRA 层，直接返回
        if self.lora_layer is None:
            return

        # 获取权重的数据类型和设备
        dtype, device = self.weight.data.dtype, self.weight.data.device

        # 将原始权重转换为浮点型
        w_orig = self.weight.data.float()
        # 获取 LoRA 层的上权重并转换为浮点型
        w_up = self.lora_layer.up.weight.data.float()
        # 获取 LoRA 层的下权重并转换为浮点型
        w_down = self.lora_layer.down.weight.data.float()

        # 如果网络 alpha 不为 None，则调整上权重
        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        # 融合权重的计算
        fused_weight = w_orig + (lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])

        # 如果进行安全融合且融合权重存在 NaN，则抛出错误
        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        # 更新当前权重为融合后的权重
        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # 将 LoRA 层设为 None，表示已融合
        self.lora_layer = None

        # 将上权重和下权重移到 CPU，防止内存溢出
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        # 保存 LoRA 融合的缩放因子
        self._lora_scale = lora_scale

    # 反融合 LoRA 权重的方法
    def _unfuse_lora(self):
        # 如果上权重和下权重不存在，直接返回
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        # 获取当前融合权重
        fused_weight = self.weight.data
        # 获取当前权重的数据类型和设备
        dtype, device = fused_weight.dtype, fused_weight.device

        # 将上权重和下权重移到对应设备并转换为浮点型
        w_up = self.w_up.to(device=device).float()
        w_down = self.w_down.to(device).float()

        # 计算未融合的权重
        unfused_weight = fused_weight.float() - (self._lora_scale * torch.bmm(w_up[None, :], w_down[None, :])[0])
        # 更新当前权重为未融合的权重
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        # 将上权重和下权重设为 None
        self.w_up = None
        self.w_down = None

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        # 如果没有 LoRA 层，直接使用父类的前向传播
        if self.lora_layer is None:
            out = super().forward(hidden_states)
            return out
        else:
            # 使用父类的前向传播加上 LoRA 层的输出
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out
```