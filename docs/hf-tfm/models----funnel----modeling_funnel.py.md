# `.\models\funnel\modeling_funnel.py`

```
# coding=utf-8
# Copyright 2020-present Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" PyTorch Funnel Transformer model."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_funnel import FunnelConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点名称
_CONFIG_FOR_DOC = "FunnelConfig"
_CHECKPOINT_FOR_DOC = "funnel-transformer/small"

# 预训练模型的存档列表
FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

# 无穷大常量
INF = 1e6

# 加载 TensorFlow 模型权重到 PyTorch 模型
def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    
    # 从 TensorFlow 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 的 API 加载指定路径下的变量数据
        array = tf.train.load_variable(tf_path, name)
        # 将加载的变量名添加到列表中
        names.append(name)
        # 将加载的变量数据添加到数组中
        arrays.append(array)

    _layer_map = {
        "k": "k_head",
        "q": "q_head",
        "v": "v_head",
        "o": "post_proj",
        "layer_1": "linear_1",
        "layer_2": "linear_2",
        "rel_attn": "attention",
        "ff": "ffn",
        "kernel": "weight",
        "gamma": "weight",
        "beta": "bias",
        "lookup_table": "weight",
        "word_embedding": "word_embeddings",
        "input": "embeddings",
    }

    for name, array in zip(names, arrays):
        # 将变量名按 '/' 分割
        name = name.split("/")
        # 如果变量名中包含以下任意一个，跳过加载：
        # "adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 记录日志，显示跳过的变量名
            logger.info(f"Skipping {'/'.join(name)}")
            # 继续下一个变量的处理
            continue
        # 如果变量名的第一个部分是 "generator"，跳过处理
        if name[0] == "generator":
            continue
        # 初始化指针为模型本身
        pointer = model
        skipped = False
        # 遍历变量名中的每个部分
        for m_name in name[1:]:
            # 如果指针不是 FunnelPositionwiseFFN 类型，并且 m_name 符合 "layer_\d+" 的格式
            if not isinstance(pointer, FunnelPositionwiseFFN) and re.fullmatch(r"layer_\d+", m_name):
                # 提取出层索引
                layer_index = int(re.search(r"layer_(\d+)", m_name).groups()[0])
                # 如果层索引小于配置中的隐藏层数量
                if layer_index < config.num_hidden_layers:
                    block_idx = 0
                    # 找到对应的块和层
                    while layer_index >= config.block_sizes[block_idx]:
                        layer_index -= config.block_sizes[block_idx]
                        block_idx += 1
                    pointer = pointer.blocks[block_idx][layer_index]
                else:
                    # 如果层索引大于等于配置中的隐藏层数量，使用层索引来访问指针的层
                    layer_index -= config.num_hidden_layers
                    pointer = pointer.layers[layer_index]
            elif m_name == "r" and isinstance(pointer, FunnelRelMultiheadAttention):
                # 如果 m_name 是 "r"，且指针是 FunnelRelMultiheadAttention 类型，则访问 r_kernel
                pointer = pointer.r_kernel
                break
            elif m_name in _layer_map:
                # 如果 m_name 在 _layer_map 中，根据映射找到对应的指针属性
                pointer = getattr(pointer, _layer_map[m_name])
            else:
                try:
                    # 尝试获取指针中的属性
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    # 如果属性不存在，记录日志并跳过当前变量
                    print(f"Skipping {'/'.join(name)}", array.shape)
                    skipped = True
                    break
        # 如果没有跳过当前变量的处理
        if not skipped:
            # 如果指针的形状与加载的数组形状不匹配，重新调整数组形状
            if len(pointer.shape) != len(array.shape):
                array = array.reshape(pointer.shape)
            # 如果 m_name 是 "kernel"，对数组进行转置操作
            if m_name == "kernel":
                array = np.transpose(array)
            # 使用 torch.from_numpy 将数组数据转换为 Torch 张量，并赋值给指针的数据
            pointer.data = torch.from_numpy(array)

    # 返回加载并更新后的模型
    return model
class FunnelEmbeddings(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        # 初始化词嵌入层，将词汇表大小设为 config.vocab_size，隐藏单元大小设为 config.hidden_size，
        # 并设置填充标记为 config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化 Layer Normalization 层，输入维度为 config.d_model，epsilon 设为 config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，丢弃率为 config.hidden_dropout
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            # 如果 inputs_embeds 为 None，则使用词嵌入层将 input_ids 转换为词嵌入向量
            inputs_embeds = self.word_embeddings(input_ids)
        # 对输入的词嵌入向量进行 Layer Normalization 处理
        embeddings = self.layer_norm(inputs_embeds)
        # 对经过 Layer Normalization 的向量应用 Dropout
        embeddings = self.dropout(embeddings)
        return embeddings


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        # 初始化 Sinusoidal Dropout 层，丢弃率为 config.hidden_dropout
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        # 初始化 Cosinusoidal Dropout 层，丢弃率为 config.hidden_dropout
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # 用于跟踪从原始输入进行池化的进度，例如，通过将序列长度除以多少
        self.pooling_mult = None

    def init_attention_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Returns the attention inputs associated to the inputs of the model."""
        # 设置 pooling_mult 为 1，表示尚未进行任何池化
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.size(1)
        # 获取位置嵌入，形状为 seq_len x config.d_model，数据类型为 inputs_embeds 的数据类型，
        # 设备为 inputs_embeds 的设备
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        # 如果存在 token_type_ids，则将其转换为 token_type_mat
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        # 如果配置要求分离 <cls> 标记，则创建对应的掩码
        cls_mask = (
            nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config.separate_cls
            else None
        )
        # 返回初始化的注意力输入元组
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Convert `token_type_ids` to `token_type_mat`."""
        # 将 token_type_ids 转换为 token_type_mat，形状为 batch_size x seq_len x seq_len
        token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        # 将 <cls> 标记视为与 A 和 B 同一段
        cls_ids = token_type_ids == self.cls_token_type_id
        cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        return cls_mat | token_type_mat

    def get_position_embeds(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # 返回位置嵌入向量，形状为 seq_len x config.d_model，数据类型为 dtype，设备为 device
        pass  # 实现在此处
    def stride_pool_pos(self, pos_id: torch.Tensor, block_index: int):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        if self.config.separate_cls:
            # 在分离 <cls> 的情况下，我们将 <cls> 视为第一个真实块的前一个块。
            # 由于第一个真实块的位置始终为1，前一个块的位置将为 `1 - 2 ** block_index`。
            cls_pos = pos_id.new_tensor([-(2**block_index) + 1])
            # 如果设置了截断序列，排除第一个和最后一个位置
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            # 返回合并后的位置信息，首先是 <cls> 的位置，然后是按步长为2抽取的池化位置
            return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            # 如果不分离 <cls>，则直接按步长为2抽取池化位置
            return pos_id[::2]

    def relative_pos(self, pos: torch.Tensor, stride: int, pooled_pos=None, shift: int = 1) -> torch.Tensor:
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos

        # 参考点是池化后位置的第一个元素减去原始位置的第一个元素
        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        # 计算最大距离和最小距离
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        # 构建相对位置向量，从最大距离开始到最小距离结束，步长为负的步长值
        return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)

    def stride_pool(
        self,
        tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        axis: Union[int, Tuple[int], List[int]],
    ) -> torch.Tensor:
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None

        # 如果轴是整数，则递归地沿着给定轴进行步长池化
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # 如果张量是列表或元组的列表，则递归地对每个张量进行步长池化
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # 处理负轴值，将轴值映射到张量的维度上
        axis %= tensor.ndim

        # 确定切片方式，根据配置决定是否分离 <cls> 并是否截断序列
        axis_slice = (
            slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
        )
        enc_slice = [slice(None)] * axis + [axis_slice]

        # 如果配置分离 <cls>，则在第一个位置前添加 <cls> 的位置信息
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)

        # 返回根据切片后的张量
        return tensor[enc_slice]

    def pool_tensor(
        self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], mode: str = "mean", stride: int = 2
    ) -> torch.Tensor:
        """
        Perform pooling operation on the input tensor.
        """
        # 这里将根据模式（平均或其他）和步长执行张量池化操作
        # 具体的池化操作在实际代码中会根据 mode 参数实现
        # 在这里我们省略了详细的具体实现方式
        pass
    ) -> torch.Tensor:
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        # Adjust tensor format if separate_cls flag is enabled.
        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]  # Expand tensor dimensions for pooling
        elif ndim == 3:
            tensor = tensor[:, None, :, :]  # Expand tensor dimensions for pooling

        # Define stride specifically for pooling operation
        stride = (stride, 1)

        # Apply pooling based on selected mode
        if mode == "mean":
            tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        # Adjust tensor format back after pooling operation
        if ndim == 2:
            return tensor[:, 0, :, 0]  # Squeeze extra dimensions for 2D tensor
        elif ndim == 3:
            return tensor[:, 0]  # Squeeze extra dimension for 3D tensor
        return tensor  # Return pooled tensor

    def pre_attention_pooling(
        self, output, attention_inputs: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        # Unpack attention_inputs into individual tensors
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        # Adjust position embeddings based on configuration
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)  # Apply stride pooling on token_type_mat
            cls_mask = self.stride_pool(cls_mask, 0)  # Apply stride pooling on cls_mask
            output = self.pool_tensor(output, mode=self.config.pooling_type)  # Apply pooling on output tensor
        else:
            self.pooling_mult *= 2  # Update pooling multiplier
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)  # Apply stride pooling on position_embeds
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])  # Apply stride pooling on token_type_mat
            cls_mask = self.stride_pool(cls_mask, [1, 2])  # Apply stride pooling on cls_mask
            attention_mask = self.pool_tensor(attention_mask, mode="min")  # Apply min pooling on attention_mask
            output = self.pool_tensor(output, mode=self.config.pooling_type)  # Apply pooling on output tensor

        # Pack adjusted tensors back into attention_inputs
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)

        # Return pooled output and adjusted attention_inputs
        return output, attention_inputs
    def post_attention_pooling(self, attention_inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        # 解包输入的注意力部分：位置嵌入、标记类型矩阵、注意力掩码、CLS掩码
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        
        # 如果配置要求仅对查询进行池化
        if self.config.pool_q_only:
            # 增加池化倍数
            self.pooling_mult *= 2
            
            # 如果注意力类型为"factorized"
            if self.config.attention_type == "factorized":
                # 对位置嵌入的前两部分进行池化，并与后续部分拼接
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            
            # 对标记类型矩阵进行池化
            token_type_mat = self.stride_pool(token_type_mat, 2)
            
            # 对CLS掩码进行池化
            cls_mask = self.stride_pool(cls_mask, 1)
            
            # 对注意力掩码进行池化，使用最小值池化模式
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        
        # 更新注意力输入为池化后的部分
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        
        # 返回更新后的注意力输入
        return attention_inputs
def _relative_shift_gather(positional_attn: torch.Tensor, context_len: int, shift: int) -> torch.Tensor:
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    # 定义函数参数和返回值的类型注解

    # 将 positional_attn 重新形状为 [batch_size, n_head, max_rel_len, seq_len]
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    # 从第 shift 列开始，截取后续的数据
    positional_attn = positional_attn[:, :, shift:, :]
    # 将 positional_attn 重新形状为 [batch_size, n_head, seq_len, max_rel_len - shift]
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    # 仅保留最后一维度中的前 context_len 个元素
    positional_attn = positional_attn[..., :context_len]
    # 返回处理后的 positional_attn
    return positional_attn


class FunnelRelMultiheadAttention(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int) -> None:
        super().__init__()
        # 初始化 FunnelRelMultiheadAttention 类，设置参数和属性

        self.config = config
        self.block_index = block_index
        d_model, n_head, d_head = config.d_model, config.n_head, config.d_head

        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        # 初始化 q、k、v 头部线性映射
        self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_head = nn.Linear(d_model, n_head * d_head)
        self.v_head = nn.Linear(d_model, n_head * d_head)

        # 初始化 r_w_bias、r_r_bias、r_kernel、r_s_bias 和 seg_embed 作为参数
        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_r_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.r_kernel = nn.Parameter(torch.zeros([d_model, n_head, d_head]))
        self.r_s_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        self.seg_embed = nn.Parameter(torch.zeros([2, n_head, d_head]))

        # 初始化后处理的线性映射和层归一化
        self.post_proj = nn.Linear(n_head * d_head, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.scale = 1.0 / (d_head**0.5)
    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        
        # Check if the attention type is factorized
        if self.config.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            phi, pi, psi, omega = position_embeds
            
            # Calculate relative bias term u with shape n_head x d_head
            u = self.r_r_bias * self.scale  # Shape n_head x d_head
            
            # Retrieve the kernel for relative attention with shape d_model x n_head x d_head
            w_r = self.r_kernel  # Shape d_model x n_head x d_head
            
            # Compute q_r_attention with shape batch_size x sea_len x n_head x d_model
            q_r_attention = torch.einsum("binh,dnh->bind", q_head + u, w_r)
            
            # Compute scaled attention scores based on positional embeddings phi and pi
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]
            
            # Combine positional attention contributions from phi, pi, psi, and omega
            # Resulting shape: batch_size x n_head x seq_len x context_len
            positional_attn = torch.einsum("bind,jd->bnij", q_r_attention_1, psi) + torch.einsum(
                "bind,jd->bnij", q_r_attention_2, omega
            )
        else:
            # For other attention types, determine the shift value
            shift = 2 if q_head.shape[1] != context_len else 1
            
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the positional encoding for the given shift
            r = position_embeds[self.block_index][shift - 1]  # Shape max_rel_len x d_model
            
            # Compute relative bias term v with shape n_head x d_head
            v = self.r_r_bias * self.scale  # Shape n_head x d_head
            
            # Retrieve the kernel for relative attention with shape d_model x n_head x d_head
            w_r = self.r_kernel  # Shape d_model x n_head x d_head
            
            # Compute r_head using the positional encoding r and kernel w_r
            r_head = torch.einsum("td,dnh->tnh", r, w_r)  # Shape max_rel_len x n_head x d_model
            
            # Compute positional attention scores based on q_head and r_head
            # Resulting shape: batch_size x n_head x seq_len x max_rel_len
            positional_attn = torch.einsum("binh,tnh->bnit", q_head + v, r_head)
            
            # Adjust positional attention scores based on relative shift and context_len
            # Resulting shape: batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

        # Apply class token masking if cls_mask is provided
        if cls_mask is not None:
            positional_attn *= cls_mask
        
        # Return the computed positional attention scores
        return positional_attn
    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        # 如果token_type_mat为空，则返回0
        if token_type_mat is None:
            return 0
        # 获取batch_size, seq_len, context_len的维度
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head的形状为batch_size x seq_len x n_head x d_head
        # 形状为n_head x d_head的r_s_bias乘以scale
        r_s_bias = self.r_s_bias * self.scale

        # 形状为batch_size x n_head x seq_len x 2的token_type_bias
        token_type_bias = torch.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # token_type_mat的形状扩展为batch_size x n_head x seq_len x context_len
        token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.shape[2], seq_len, context_len])
        # 在最后一个维度上分割为形状为batch_size x n_head x seq_len x 1的diff_token_type和same_token_type
        diff_token_type, same_token_type = torch.split(token_type_bias, 1, dim=-1)
        # 形状为batch_size x n_head x seq_len x context_len的token_type_attn
        token_type_attn = torch.where(
            token_type_mat,  # 条件是token_type_mat
            same_token_type.expand(token_type_mat.shape),  # 如果条件成立，使用same_token_type扩展形状
            diff_token_type.expand(token_type_mat.shape)   # 否则，使用diff_token_type扩展形状
        )

        # 如果有cls_mask，则将token_type_attn与其相乘
        if cls_mask is not None:
            token_type_attn *= cls_mask
        # 返回token_type_attn
        return token_type_attn
    ) -> Tuple[torch.Tensor, ...]:
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config.n_head, self.config.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, d_head)

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len
        content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        # Calculate relative positional attention
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        # Calculate relative token type attention
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        attn_score = attn_score.float()
        # perform masking
        if attention_mask is not None:
            # Apply attention mask to attention scores
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
        # attention probability
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
        attn_prob = self.attention_dropout(attn_prob)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(attn_vec.reshape(batch_size, seq_len, n_head * d_head))
        attn_out = self.hidden_dropout(attn_out)

        # Residual connection and layer normalization
        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)
# 定义一个用于Funnel模型中的编码器的类
class FunnelEncoder(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config  # 保存Funnel配置对象

        # 初始化Funnel注意力结构对象
        self.attention_structure = FunnelAttentionStructure(config)

        # 创建多层模块列表，每一层由多个FunnelLayer组成，根据配置中的块大小生成
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    # 前向传播函数，接受输入嵌入向量及其它可选参数，并返回输出结果
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, Tuple], torch.Tensor]:
        # 省略了前向传播函数体的注释
    ) -> Union[Tuple, BaseModelOutput]:
        # 定义函数的输入和输出类型，此函数返回一个元组或者BaseModelOutput类型的对象

        # 将注意力掩码转换为与输入嵌入张量相同的数据类型
        attention_mask = attention_mask.type_as(inputs_embeds)

        # 使用注意力结构初始化注意力输入，包括输入嵌入张量、注意力掩码、标记类型ID
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 将隐藏状态初始化为输入嵌入张量
        hidden = inputs_embeds

        # 如果需要输出所有隐藏状态，则初始化存储所有隐藏状态的元组
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None

        # 如果需要输出所有注意力权重，则初始化存储所有注意力权重的元组
        all_attentions = () if output_attentions else None

        # 遍历每一个块
        for block_index, block in enumerate(self.blocks):
            # 根据配置和块索引确定是否进行池化
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0

            # 如果需要池化，则执行前注意力池化操作
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )

            # 遍历块内的每一层
            for layer_index, layer in enumerate(block):
                # 根据块配置的重复次数遍历每一层
                for repeat_index in range(self.config.block_repeats[block_index]):
                    # 确定当前是否需要进行池化操作
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag

                    # 根据是否需要池化，选择不同的查询（query）、键（key）、值（value）
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden

                    # 调用当前层的前向方法，获取层的输出
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)

                    # 更新隐藏状态为当前层的输出
                    hidden = layer_output[0]

                    # 如果执行了池化操作，则执行后注意力池化操作
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)

                    # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_attentions中
                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]

                    # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states中
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)

        # 根据return_dict标志返回不同的输出形式
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)
# 定义一个函数 `upsample`，用于对输入的张量 `x` 进行上采样操作，使其长度与 `target_len` 相匹配，
# 方法是在序列长度维度上重复每个标记 `stride` 次。
def upsample(
    x: torch.Tensor, stride: int, target_len: int, separate_cls: bool = True, truncate_seq: bool = False
) -> torch.Tensor:
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    # 如果 `stride` 为 1，则直接返回输入的张量 `x`
    if stride == 1:
        return x
    # 如果 `separate_cls` 为 True，则从 `x` 中分离出特殊标记（CLS 标记）
    if separate_cls:
        cls = x[:, :1]  # 提取第一个标记作为特殊标记
        x = x[:, 1:]    # 剩余部分作为序列数据
    # 在序列长度维度上重复每个标记 `stride` 次，形成上采样后的输出
    output = torch.repeat_interleave(x, repeats=stride, dim=1)
    # 如果 `separate_cls` 为 True，则根据需要截断序列并重新连接特殊标记
    if separate_cls:
        # 如果需要截断序列 (`truncate_seq` 为 True)，则在末尾进行零填充
        if truncate_seq:
            output = nn.functional.pad(output, (0, 0, 0, stride - 1, 0, 0))
        # 截取序列长度至 `target_len - 1`，并重新连接特殊标记
        output = output[:, : target_len - 1]
        output = torch.cat([cls, output], dim=1)
    else:
        # 如果 `separate_cls` 为 False，则直接截取序列长度至 `target_len`
        output = output[:, :target_len]
    # 返回经过上述处理后的输出张量
    return output


class FunnelDecoder(nn.Module):
    # 定义 FunnelDecoder 类，继承自 nn.Module 类
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.config = config
        # 初始化注意力结构模块 FunnelAttentionStructure，并传入配置参数
        self.attention_structure = FunnelAttentionStructure(config)
        # 使用列表推导式创建多个 FunnelLayer 层，并存储在 layers 属性中
        self.layers = nn.ModuleList([FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    def forward(
        self,
        final_hidden: torch.Tensor,
        first_block_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        # 对 final_hidden 进行上采样操作，使其与 first_block_hidden 的长度相匹配
        upsampled_hidden = upsample(
            final_hidden,
            stride=2 ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            separate_cls=self.config.separate_cls,
            truncate_seq=self.config.truncate_seq,
        )

        # 将上采样后的 hidden 与 first_block_hidden 相加得到新的 hidden 张量
        hidden = upsampled_hidden + first_block_hidden
        # 初始化空列表，用于存储所有的隐藏状态
        all_hidden_states = (hidden,) if output_hidden_states else None
        # 初始化空元组，用于存储所有的注意力权重
        all_attentions = () if output_attentions else None

        # 初始化注意力结构输入参数，并传入相应参数
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 遍历所有的 FunnelLayer 层，并依次进行前向传播计算
        for layer in self.layers:
            # 调用每一层的前向传播方法，并获取输出
            layer_output = layer(hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions)
            hidden = layer_output[0]  # 更新 hidden 为当前层的输出

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)

        # 如果不需要返回字典形式的结果，则按需返回 hidden、all_hidden_states 和 all_attentions
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        # 如果需要返回字典形式的结果，则使用 BaseModelOutput 构造器返回结果
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


class FunnelDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""
    # FunnelDiscriminatorPredictions 类，用于判别器的预测模块，由两个全连接层组成
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 创建一个线性层，输入和输出维度都为 config.d_model
        self.dense = nn.Linear(config.d_model, config.d_model)
        # 创建一个线性层，输入维度为 config.d_model，输出维度为 1
        self.dense_prediction = nn.Linear(config.d_model, 1)

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, discriminator_hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行转换，输入为 discriminator_hidden_states
        hidden_states = self.dense(discriminator_hidden_states)
        # 根据配置中的激活函数选择对 hidden_states 进行非线性变换
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        # 使用预测线性层得到最终的 logits，将结果的最后一个维度压缩为 1
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        # 返回处理后的 logits 结果
        return logits
class FunnelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 引入配置类，用于处理模型配置
    config_class = FunnelConfig
    # 载入 TensorFlow 权重的方法
    load_tf_weights = load_tf_weights_in_funnel
    # 基础模型的前缀
    base_model_prefix = "funnel"

    def _init_weights(self, module):
        # 获取当前模块的类名
        classname = module.__class__.__name__
        # 如果类名中包含 "Linear" 字符串，表示是线性层
        if classname.find("Linear") != -1:
            # 如果模块具有权重属性
            if getattr(module, "weight", None) is not None:
                # 如果初始化标准差未指定，则计算标准差为平方根值
                if self.config.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                # 使用正态分布初始化权重
                nn.init.normal_(module.weight, std=std)
            # 如果模块具有偏置属性，则将偏置初始化为 0
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0.0)
        # 如果类名是 "FunnelRelMultiheadAttention"，表示是多头注意力层
        elif classname == "FunnelRelMultiheadAttention":
            # 使用均匀分布初始化特定参数
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.r_r_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.r_kernel, b=self.config.initializer_range)
            nn.init.uniform_(module.r_s_bias, b=self.config.initializer_range)
            nn.init.uniform_(module.seg_embed, b=self.config.initializer_range)
        # 如果类名是 "FunnelEmbeddings"，表示是嵌入层
        elif classname == "FunnelEmbeddings":
            # 如果未指定初始化标准差，则使用默认值 1.0
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            # 使用正态分布初始化词嵌入权重
            nn.init.normal_(module.word_embeddings.weight, std=std)
            # 如果嵌入层具有填充索引，则将填充索引位置的权重置为零
            if module.word_embeddings.padding_idx is not None:
                module.word_embeddings.weight.data[module.padding_idx].zero_()


class FunnelClassificationHead(nn.Module):
    def __init__(self, config: FunnelConfig, n_labels: int) -> None:
        super().__init__()
        # 线性层，用于隐藏层
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 线性层，用于输出层
        self.linear_out = nn.Linear(config.d_model, n_labels)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # 隐藏层的线性变换
        hidden = self.linear_hidden(hidden)
        # 使用双曲正切函数进行激活
        hidden = torch.tanh(hidden)
        # Dropout 操作
        hidden = self.dropout(hidden)
        # 输出层的线性变换
        return self.linear_out(hidden)


@dataclass
class FunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FunnelForPreTraining`].
    """
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            ELECTRA-style目标函数的总损失。
            如果提供了`labels`，则返回该损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            模型头部的预测分数（每个token的分数，未经过SoftMax）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态，以及初始嵌入输出的元组。
            每个元素是`torch.FloatTensor`，形状为`(batch_size, sequence_length, hidden_size)`。
            当参数`output_hidden_states=True`或`config.output_hidden_states=True`时返回。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            每层注意力权重的元组。
            每个元素是`torch.FloatTensor`，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
            在注意力softmax后得到的注意力权重，用于计算自注意力头部的加权平均值。
            当参数`output_attentions=True`或`config.output_attentions=True`时返回。
FUNNEL_START_DOCSTRING = r"""

    The Funnel Transformer model was proposed in [Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing](https://arxiv.org/abs/2006.03236) by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FunnelConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


注释：
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获得这些索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖掩码，用于在填充标记索引上避免执行注意力操作。
            # 遮盖值选择在 `[0, 1]` 之间：
            # - 1 表示 **未遮盖** 的标记，
            # - 0 表示 **被遮盖** 的标记。
            # [什么是注意力遮盖？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，用于指示输入的第一部分和第二部分。
            # 索引在 `[0, 1]` 之间选择：
            # - 0 对应于 *句子 A* 的标记，
            # - 1 对应于 *句子 B* 的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示，而不是传递 `input_ids`。
            # 如果您希望更加控制将 `input_ids` 索引转换为相关向量的方式，而不是使用模型内部的嵌入查找矩阵，则这很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 更多细节请参见返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 更多细节请参见返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
"""
FunnelTransformer 模型的基础类，输出原始的隐藏状态，没有上采样头（也称为解码器）或任何特定任务的顶部头部。

该类包含了 FunnelTransformer 模型的基本结构和方法。
"""
@add_start_docstrings(
    """
    The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        """
        初始化 FunnelBaseModel 类的实例。

        Args:
            config (FunnelConfig): 包含模型配置信息的对象。
        """
        super().__init__(config)

        # 初始化嵌入层和编码器
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """
        获取输入嵌入层的方法。

        Returns:
            nn.Embedding: 返回用于词嵌入的 nn.Embedding 对象。
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        """
        设置输入嵌入层的方法。

        Args:
            new_embeddings (nn.Embedding): 新的词嵌入对象。
        """
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,


这段代码定义了一个名为 `FunnelBaseModel` 的类，作为 Funnel Transformer 模型的基础实现。它包含了模型的初始化方法、嵌入层和编码器的设置方法，以及模型的前向传播方法，用于处理输入并生成输出隐藏状态。
        # 如果 output_attentions 参数未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果既指定了 input_ids 又指定了 inputs_embeds，则抛出异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids
        elif input_ids is not None:
            # 检查是否需要警告，即是否存在填充并且未提供 attention_mask
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        # 如果指定了 inputs_embeds
        elif inputs_embeds is not None:
            # 获取 inputs_embeds 的形状，排除最后一维
            input_shape = inputs_embeds.size()[:-1]
        # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出异常
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 根据是否指定了 input_ids，确定设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未指定 attention_mask，则创建全为 1 的默认 attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果未指定 token_type_ids，则创建全为 0 的 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: 处理 head_mask，这部分代码目前尚未实现

        # 如果未指定 inputs_embeds，则使用 self.embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # 将嵌入后的 inputs_embeds 输入到 encoder 中进行编码
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回编码器的输出
        return encoder_outputs
# 使用装饰器为模型类添加文档字符串，描述此模型是一个输出原始隐藏状态的Funnel Transformer模型，没有特定的头部处理。
@add_start_docstrings(
    "The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.",
    FUNNEL_START_DOCSTRING,
)
# 定义FunnelModel类，继承自FunnelPreTrainedModel类
class FunnelModel(FunnelPreTrainedModel):
    
    # 初始化方法，接收一个FunnelConfig类型的config对象
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的config对象赋值给实例变量self.config
        self.config = config
        # 创建FunnelEmbeddings对象并赋值给实例变量self.embeddings
        self.embeddings = FunnelEmbeddings(config)
        # 创建FunnelEncoder对象并赋值给实例变量self.encoder
        self.encoder = FunnelEncoder(config)
        # 创建FunnelDecoder对象并赋值给实例变量self.decoder
        self.decoder = FunnelDecoder(config)

        # 调用模型后处理方法，用于初始化权重并进行最终处理
        self.post_init()

    # 返回输入嵌入层的方法
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    # 设置输入嵌入层的方法，接收一个新的nn.Embedding对象作为参数
    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings.word_embeddings = new_embeddings

    # 使用装饰器为前向传播方法添加文档字符串，描述前向传播的输入参数和返回值
    # 同时添加代码示例的文档字符串，展示如何调用此方法进行推理
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 设置是否返回注意力矩阵，默认从配置中获取
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否返回隐藏层状态，默认从配置中获取
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回类型，默认从配置中获取
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出 ValueError
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果只指定了 input_ids，则检查是否需要警告无 attention_mask 的情况，并获取输入的形状
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 如果只指定了 inputs_embeds，则获取输入的形状（去掉最后一维，即 batch 维度）
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出 ValueError
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 根据 input_ids 或 inputs_embeds 确定设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有提供 attention_mask，则创建一个全为 1 的 mask，形状与输入数据相同，放置在指定的设备上
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果没有提供 token_type_ids，则创建一个全为 0 的 token 类型 ID，形状与输入数据相同，数据类型为 long，放置在指定的设备上
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: 处理 head_mask（头部遮罩），待实现

        # 如果没有提供 inputs_embeds，则使用 self.embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # 使用 self.encoder 进行编码器的前向传播计算
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 强制输出隐藏状态
            return_dict=return_dict,
        )

        # 使用 self.decoder 进行解码器的前向传播计算
        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs[0],  # 使用编码器的最终隐藏状态作为解码器的输入
            first_block_hidden=encoder_outputs[1][self.config.block_sizes[0]],  # 使用编码器第一个块的隐藏状态作为解码器的输入
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不需要返回字典形式的结果，则根据需要构建返回的元组
        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)  # 将解码器输出的最后隐藏状态作为输出的第一个元素
            if output_hidden_states:
                idx += 1
                # 如果需要输出隐藏状态，则将编码器和解码器的隐藏状态拼接起来作为输出的一部分
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                # 如果需要输出注意力矩阵，则将编码器和解码器的注意力矩阵拼接起来作为输出的一部分
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs

        # 如果需要返回字典形式的结果，则构建一个 BaseModelOutput 对象作为输出
        return BaseModelOutput(
            last_hidden_state=decoder_outputs[0],  # 最后的隐藏状态来自解码器的输出
            hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states  # 如果需要输出隐藏状态，则将编码器和解码器的隐藏状态列表合并
            else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions)
            if output_attentions  # 如果需要输出注意力矩阵，则将编码器和解码器的注意力矩阵列表合并
            else None,
        )
add_start_docstrings(
    """
    Funnel Transformer model with a binary classification head on top as used during pretraining for identifying
    generated tokens.
    """,
    FUNNEL_START_DOCSTRING,
)



class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)

        # 初始化 Funnel 模型
        self.funnel = FunnelModel(config)
        # 初始化用于判别预测的组件
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        # 初始化权重并应用最终处理
        self.post_init()



    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



        ):
        """
        Funnel 模型的前向传播方法，支持的输入参数包括:
        - input_ids: 输入的 token IDs
        - attention_mask: 注意力掩码
        - token_type_ids: token 类型 IDs
        - inputs_embeds: 输入的嵌入向量
        - labels: 标签
        - output_attentions: 是否输出注意力权重
        - output_hidden_states: 是否输出隐藏状态
        - return_dict: 是否返回结果字典形式

        返回一个包含预测输出的 FunnelForPreTrainingOutput 对象。
        """
        ) -> Union[Tuple, FunnelForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the ELECTRA-style loss. Input should be a sequence of tokens (see `input_ids`
            docstring) Indices should be in `[0, 1]`:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, FunnelForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("funnel-transformer/small")
        >>> model = FunnelForPreTraining.from_pretrained("funnel-transformer/small")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> logits = model(**inputs).logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取鉴别器的隐藏状态，通过调用Funnel模型进行计算
        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取鉴别器输出的序列结果
        discriminator_sequence_output = discriminator_hidden_states[0]

        # 将鉴别器输出序列传入鉴别器预测模块，生成logits
        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        # 如果提供了labels，则计算损失
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                # 计算有效的损失，只考虑attention_mask标记为1的部分
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                # 计算所有位置的损失
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        # 如果不要求返回字典，则输出一个元组
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则输出一个FunnelForPreTrainingOutput对象
        return FunnelForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
# 使用装饰器为类添加文档字符串，描述其为在Funnel Transformer模型基础上带有语言建模头部的模型
@add_start_docstrings("""Funnel Transformer Model with a `language modeling` head on top.""", FUNNEL_START_DOCSTRING)
class FunnelForMaskedLM(FunnelPreTrainedModel):
    # 定义权重共享的键值对列表，这里指定了语言建模头部权重
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建Funnel模型实例，使用给定的配置
        self.funnel = FunnelModel(config)
        # 创建一个线性层作为语言建模头部，输入维度为配置中定义的d_model，输出维度为词汇表大小（vocab_size）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # 调用后续初始化方法，用于权重初始化和最终处理
        self.post_init()

    # 返回语言建模头部的线性层对象
    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    # 设置新的输出嵌入层作为语言建模头部
    def set_output_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.lm_head = new_embeddings

    # 使用装饰器为前向方法添加文档字符串，描述其输入参数和使用示例
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 根据需要确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给Funnel模型进行前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取Funnel模型输出的最后一层隐藏状态
        last_hidden_state = outputs[0]

        # 使用语言模型头部对最后一层隐藏状态进行预测
        prediction_logits = self.lm_head(last_hidden_state)

        masked_lm_loss = None
        # 如果提供了标签，则计算masked language modeling的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数，-100索引对应填充标记
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回字典形式的输出，则将结果按顺序打包返回
        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回MaskedLMOutput对象，其中包含损失、预测logits、隐藏状态和注意力分布
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Funnel Transformer Model with a sequence classification/regression head on top (two linear layer on top of the
    first timestep of the last hidden state) e.g. for GLUE tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForSequenceClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.funnel = FunnelBaseModel(config)  # 初始化FunnelBaseModel模型
        self.classifier = FunnelClassificationHead(config, config.num_labels)  # 初始化FunnelClassificationHead分类头
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        将输入传递给Funnel模型以执行前向传播。
        """
        # 略
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 默认情况下，如果 return_dict 为 None，则根据 self.config.use_return_dict 来确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Funnel 模型进行处理，获取输出
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的最后一层隐藏状态
        last_hidden_state = outputs[0]
        # 提取池化后的输出，通常是最后一层隐藏状态的第一个位置的输出
        pooled_output = last_hidden_state[:, 0]
        # 将池化后的输出传递给分类器，得到预测的 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果有提供标签
        if labels is not None:
            # 如果问题类型未定义，则根据标签的类型自动推断问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单个标签的回归任务，计算损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签的回归任务，计算损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                # 计算单标签分类任务的损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                # 计算多标签分类任务的损失
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 返回一个元组，包含 logits 和可能的其他输出
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Funnel Transformer Model with a multiple choice classification head on top (two linear layer on top of the first
    timestep of the last hidden state, and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForMultipleChoice(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)

        # 初始化 FunnelBaseModel，用于处理 Transformer 的主体部分
        self.funnel = FunnelBaseModel(config)
        # 初始化 FunnelClassificationHead，用于多选分类任务的头部
        self.classifier = FunnelClassificationHead(config, 1)
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="funnel-transformer/small-base",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        FunnelForMultipleChoice 模型的前向传播方法，接收多个输入参数，返回模型输出结果。

        Args:
            input_ids (Optional[torch.Tensor], optional): 输入序列的 token IDs. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): 注意力遮罩，掩盖无效输入. Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): token 类型 IDs, 用于区分 segment. Defaults to None.
            inputs_embeds (Optional[torch.Tensor], optional): 替代输入 token IDs 的嵌入. Defaults to None.
            labels (Optional[torch.Tensor], optional): 真实标签. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典格式的输出. Defaults to None.
        
        Returns:
            输出结果，根据 return_dict 的设置返回不同的格式，可能包括分类结果、注意力权重或隐藏状态等信息.
        """
        # 省略部分方法内容...
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据返回字典的存在性来确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新整形输入数据，将其变为二维张量
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用 Funnel 模型进行前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一层隐藏状态和池化输出
        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        # 使用分类器得到 logits
        logits = self.classifier(pooled_output)
        # 重新整形 logits，以匹配 num_choices 的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None
        # 如果存在 labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不使用返回字典，则返回 reshaped_logits 和额外的 outputs
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用返回字典类 MultipleChoiceModelOutput 返回结果
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Funnel Transformer Model with a span classification head on top for extractive question-answering tasks like SQuAD
"""
@add_start_docstrings(
    """
    Funnel Transformer Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    FUNNEL_START_DOCSTRING,
)
class FunnelForTokenClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize Funnel Transformer model
        self.funnel = FunnelModel(config)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout)
        # Linear layer for token classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through Funnel Transformer model
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply dropout to the output of the transformer
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        # Project the hidden states to logits using a linear layer
        logits = self.classifier(last_hidden_state)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Prepare the output according to return_dict flag
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 这里是字符串常量，描述了在隐藏状态输出之上的线性层，用于计算“起始位置标志”和“结束位置标志”
    (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    # 导入了名为 FUNNEL_START_DOCSTRING 的文档字符串常量
    FUNNEL_START_DOCSTRING,
    )
    # 定义 FunnelForQuestionAnswering 类，继承自 FunnelPreTrainedModel
    class FunnelForQuestionAnswering(FunnelPreTrainedModel):
        def __init__(self, config: FunnelConfig) -> None:
            super().__init__(config)
            self.num_labels = config.num_labels

            # 初始化 FunnelModel 和 QA 输出层
            self.funnel = FunnelModel(config)
            self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

            # 初始化权重并进行最终处理
            self.post_init()

        @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=QuestionAnsweringModelOutput,
            config_class=_CONFIG_FOR_DOC,
        )
        # 定义 forward 方法，接受一系列输入参数并返回相应的输出
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.Tensor] = None,
            end_positions: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 根据返回字典的设置，确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Funnel 模型进行推理
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一层隐藏状态
        last_hidden_state = outputs[0]

        # 使用 QA 输出层得到起始和结束 logits
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上运行，增加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典，则返回元组形式的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果需要返回字典，则创建 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```