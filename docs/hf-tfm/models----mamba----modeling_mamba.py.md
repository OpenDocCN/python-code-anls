# `.\models\mamba\modeling_mamba.py`

```py
# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MAMBA model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig


logger = logging.get_logger(__name__)

# Check if MAMBA SSM (Selective State Memory Access) functionalities are available
if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

# Check if causal conv1d functionalities are available
if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

# Check if all fast path functionalities are available
is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

# Documentation variables
_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"

# List of pretrained model archive names for MAMBA models
MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST = []  # See all MAMBA models at https://huggingface.co/models?filter=mamba


class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        """
        Initialize MambaCache object.

        Args:
            config (MambaConfig): The configuration object for MAMBA model.
            batch_size (int): Batch size for the cache.
            dtype (torch.dtype, optional): Data type for tensors in cache (default: torch.float16).
            device (torch.device, optional): Device for tensors in cache (default: None).
        """
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        # Initialize convolutional states dictionary
        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

        # Initialize SSM (Selective State Memory) states dictionary
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }


class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    """
    """
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    # 初始化方法，接受配置对象和层索引作为参数
    def __init__(self, config, layer_idx):
        super().__init__()
        # 从配置对象中获取隐藏层大小和状态空间大小
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        # 获取卷积核大小、中间层大小、时间步级别的排名、层索引以及是否使用卷积偏置的配置
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias

        # 创建一个 1D 卷积层，用于特征转换
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        # 激活函数和激活函数名称的映射
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # 投影输入隐藏状态的线性层
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # 选择性投影，用于使 dt、B 和 C 依赖于输入
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # 时间步投影（离散化）
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D 真实初始化。这些值不是离散化的！
        # 核心是加载它们，计算离散状态，然后写入更新后的状态。保持内存有限
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        # 初始化 A 的对数，作为可学习参数
        self.A_log = nn.Parameter(torch.log(A))
        # 初始化 D，作为可学习参数
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        # 输出投影线性层
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        # 如果快速路径不可用，则发出警告
        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    # fmt: off
    # fmt: on

    # 前向传播方法，接受隐藏状态和缓存参数作为输入
    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        # 如果快速路径可用且在 GPU 上，则调用 CUDA 版本的前向传播
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params)
        # 否则，调用慢速的 Python 实现的前向传播
        return self.slow_forward(hidden_states, cache_params)
# 定义一个自定义的神经网络模块，用于实现MambaRMSNorm规范化层
class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        # 初始化可学习参数weight，初始值为1，用于缩放规范化后的输出
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 规范化中的一个小常数，用于防止除以零
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 记录输入的数据类型，后续会将hidden_states转换为float32
        input_dtype = hidden_states.dtype
        # 将输入hidden_states转换为float32类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算hidden_states的方差，并在最后一个维度上保持维度不变
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 对hidden_states进行RMS（均方根）规范化操作
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回规范化后乘以权重的结果，并转回输入数据类型
        return self.weight * hidden_states.to(input_dtype)


# 定义一个自定义的神经网络模块，表示Mamba模型中的一个块
class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 记录配置和层索引
        self.config = config
        self.layer_idx = layer_idx
        # 是否在浮点数（float32）中处理残差连接
        self.residual_in_fp32 = config.residual_in_fp32
        # 初始化规范化层，使用MambaRMSNorm，并传入隐藏大小和层归一化的小常数值
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化MambaMixer模块，用于处理隐藏状态
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        # 备份原始的hidden_states作为残差连接
        residual = hidden_states
        # 对隐藏状态进行规范化，并将数据类型转换为self.norm.weight的数据类型
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        # 如果配置要求，将残差连接转换为float32类型
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        # 经过MambaMixer处理后的隐藏状态与残差连接相加，作为本模块的输出
        hidden_states = self.mixer(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


# MambaPreTrainedModel是一个抽象类，用于处理权重初始化，下载和加载预训练模型的简单接口
class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定该类使用的配置类
    config_class = MambaConfig
    # 指定模型的主要前缀字符串
    base_model_prefix = "backbone"
    # 不需要拆分的模块名称列表
    _no_split_modules = ["MambaBlock"]
    # 支持梯度检查点的标志
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果 module 是 MambaMixer 类的实例
        if isinstance(module, MambaMixer):
            # 设置权重不参与权重衰减的标志位
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            # 根据配置参数初始化时间步长的标准差
            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            # 根据初始化方案初始化时间步长投影权重
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            # 生成指数分布的时间步长，并进行上下限截断
            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # 计算逆 softplus 函数的结果，用于初始化偏置
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            # 用逆 softplus 函数的结果设置偏置，不进行梯度计算
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        # 如果 module 是 nn.Linear 类的实例
        if isinstance(module, nn.Linear):
            # 如果存在偏置项且未标记为不重新初始化，则将其初始化为零
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        
        # 如果 module 是 nn.Embedding 类的实例
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        # 如果配置要求重新缩放预正则化残差
        if self.config.rescale_prenorm_residual:
            # 针对选定的参数进行重新初始化，参考 OpenAI GPT-2 论文中的方案
            # 对于 "out_proj.weight" 参数，特殊的缩放初始化策略
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # 使用特殊的 Kaiming 均匀初始化，除以 sqrt(2 * num_layers) 进行缩放
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)
"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
"""

MambaOutput:
"""
Class for the MAMBA model outputs.

Args:
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
    cache_params (`MambaCache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
"""
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


MambaCausalLMOutput:
"""
Base class for causal language model (or autoregressive) outputs.

Args:
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    cache_params (`MambaCache`):
        The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
        avoid providing the old `input_ids`.

        Includes both the State space model state matrices after the selective scan, and the Convolutional states
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
"""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)



    # 该库实现了其所有模型的各种功能，如下载或保存模型、调整输入嵌入的大小、修剪模型头等。



    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.



    # 这个模型也是 PyTorch 的 torch.nn.Module 的子类。
    # 可以像使用常规的 PyTorch 模块一样使用它，关于一般用法和行为的所有事项，请参考 PyTorch 文档。



    Parameters:
        config ([`MambaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.



    # 参数:
    #     config ([`MambaConfig`]): 包含模型所有参数的配置类。
    #         使用配置文件初始化不会加载模型的权重，只会加载配置信息。
    #         若要加载模型权重，请查阅 [`~PreTrainedModel.from_pretrained`] 方法。
"""
MAMBA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.

            If `cache_params.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cache_params (`MambaCache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare MAMBA Model transformer outputting raw hidden-states without any specific head on top.",
    MAMBA_START_DOCSTRING,
)
class MambaModel(MambaPreTrainedModel):
    """
    MAMBA 模型的核心类，输出未经特定头部处理的原始隐藏状态。
    """

    def __init__(self, config):
        """
        初始化 MambaModel 类。

        Args:
            config (MambaConfig): 包含模型配置信息的实例。

        Attributes:
            embeddings (nn.Embedding): 输入 token 的嵌入表示。
            layers (nn.ModuleList): MambaBlock 层的列表，构成模型的主体。
            gradient_checkpointing (bool): 是否使用梯度检查点。
            norm_f (MambaRMSNorm): 应用于隐藏状态的 RMS 标准化器。
        """
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        """
        获取输入嵌入层。

        Returns:
            nn.Embedding: 输入嵌入层对象。
        """
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        设置输入嵌入层。

        Args:
            new_embeddings (nn.Embedding): 新的输入嵌入层对象。
        """
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self, input_ids=None, inputs_embeds=None, cache_params=None, use_cache=False,
                output_hidden_states=False, return_dict=True):
        """
        模型的前向传播。

        Args:
            input_ids (torch.LongTensor, optional): 输入 token 的索引序列。
            inputs_embeds (torch.FloatTensor, optional): 输入 token 的嵌入表示。
            cache_params (MambaCache, optional): 缓存参数，用于模型的历史状态。
            use_cache (bool, optional): 如果为 True，则返回缓存参数以便快速生成下一个 logit。
            output_hidden_states (bool, optional): 是否返回所有层的隐藏状态。
            return_dict (bool, optional): 是否返回 ModelOutput 对象而不是普通元组。

        Returns:
            ModelOutput or tuple: 模型输出对象或普通元组，具体取决于 return_dict 参数的设置。
        """
        pass
    # 定义模型的前向传播方法，接受多个输入参数，并返回一个元组或MambaOutput对象
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ids序列，可选
        inputs_embeds: Optional[torch.LongTensor] = None,  # 输入的嵌入表示，可选
        cache_params: Optional[MambaCache] = None,  # 缓存参数对象，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出所有隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
        **kwargs,  # 其他关键字参数，例如attention_mask由分词器传递，不需要处理
    ) -> Union[Tuple, MambaOutput]:  # 返回值可以是元组或MambaOutput对象
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )  # 如果没有显式指定输出隐藏状态，则使用配置中的默认设置

        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        # 如果没有显式指定是否使用缓存，则根据训练状态和模型配置进行设定

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果没有显式指定是否返回字典形式的输出，则根据模型配置进行设定

        if (input_ids is None) ^ (inputs_embeds is not None):  # 异或运算符判断输入参数的合法性
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # 如果没有提供嵌入表示，则根据输入的token ids生成嵌入表示

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False  # 如果启用了梯度检查点且处于训练模式并且使用缓存，则禁用缓存

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )  # 如果没有提供缓存参数且需要使用缓存，则创建新的MambaCache对象

        hidden_states = inputs_embeds  # 将嵌入表示作为初始隐藏状态
        all_hidden_states = () if output_hidden_states else None  # 如果需要输出所有隐藏状态，则初始化空元组

        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
                # 如果启用了梯度检查点并且处于训练模式，则使用梯度检查点函数计算mixer_block的输出
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)
                # 否则直接调用mixer_block计算隐藏状态

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # 如果需要输出所有隐藏状态，则保存当前隐藏状态

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]  # 更新缓存参数的序列长度偏移量

        hidden_states = self.norm_f(hidden_states)  # 对最终的隐藏状态进行归一化处理

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # 如果需要输出所有隐藏状态，则保存最终的隐藏状态

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)
            # 如果不需要返回字典形式的输出，则返回一个元组，包含非空的hidden_states、cache_params和all_hidden_states

        return MambaOutput(
            last_hidden_state=hidden_states,  # 返回MambaOutput对象，包括最终的隐藏状态
            cache_params=cache_params if use_cache else None,  # 如果使用缓存，则返回缓存参数，否则为None
            hidden_states=all_hidden_states,  # 返回所有的隐藏状态
        )
@add_start_docstrings(
    """
    The MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MAMBA_START_DOCSTRING,
)
class MambaForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)  # 初始化 MambaModel 作为 backbone
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 初始化线性层 lm_head，用于语言建模头部，权重与输入嵌入层相关联

        # 初始化后处理
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head  # 返回 lm_head 作为输出嵌入层

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings  # 设置新的输出嵌入层

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()  # 获取输入嵌入层

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)  # 设置新的输入嵌入层

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs
        # 更新用于生成的模型参数，包括缓存参数

    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # 如果传递了状态，则只使用输入 IDs 的最后一个标记
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs
        # 为生成准备输入数据，支持输入 IDs 或嵌入张量，以及缓存参数

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ):
        # 此处是模型的前向传播方法，输入参数包括 input_ids、inputs_embeds 等等，用于生成或训练模型
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 根据需要确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的主体部分进行前向计算，获取模型输出
        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        # 获取模型主体输出中的隐藏状态
        hidden_states = mamba_outputs[0]

        # 使用语言模型头部计算逻辑回归结果
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        # 初始化损失为None
        loss = None
        if labels is not None:
            # 将标签移到正确的设备上，以支持模型并行计算
            labels = labels.to(logits.device)
            # 将预测的logits向左移动一个位置，以对齐标签
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 如果不返回字典形式的输出，构造输出元组
        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回自定义的输出类MambaCausalLMOutput，包括损失、logits和其他额外的模型输出
        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
```