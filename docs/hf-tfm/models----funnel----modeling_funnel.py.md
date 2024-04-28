# `.\models\funnel\modeling_funnel.py`

```
# 设置 UTF-8 编码
# 版权声明和许可证信息
# 导入必要的库和模块
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入模型相关的类和函数
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

# 设置日志记录器
logger = logging.get_logger(__name__)

# 文档中用到的配置和检查点名称
_CONFIG_FOR_DOC = "FunnelConfig"
_CHECKPOINT_FOR_DOC = "funnel-transformer/small"

# 预训练模型存档列表
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

# 从 TensorFlow 模型加载权重到 PyTorch 模型
def load_tf_weights_in_funnel(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，输出错误消息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 输出信息，显示正在从 TensorFlow 检查点中加载权重
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 初始化权重和权重名称列表
    names = []
    arrays = []
```  
    # 循环遍历初始化变量名和形状
    for name, shape in init_vars:
        # 记录当前正在加载的 TensorFlow 权重名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 从 TensorFlow 模型路径中加载指定名称的变量
        array = tf.train.load_variable(tf_path, name)
        # 将变量名添加到列表中
        names.append(name)
        # 将加载的变量数组添加到列表中
        arrays.append(array)

    # 定义一个层次名称的映射字典
    _layer_map = {
        "k": "k_head",  # key的简写映射
        "q": "q_head",  # query的简写映射
        "v": "v_head",  # value的简写映射
        "o": "post_proj",  # 输出投影层
        "layer_1": "linear_1",  # 第一线性层
        "layer_2": "linear_2",  # 第二线性层
        "rel_attn": "attention",  # 相对注意力层
        "ff": "ffn",  # 前馈网络
        "kernel": "weight",  # 核心权重
        "gamma": "weight",  # 权重（标准化用）
        "beta": "bias",  # 偏置（标准化用）
        "lookup_table": "weight",  # 查找表权重
        "word_embedding": "word_embeddings",  # 词嵌入层
        "input": "embeddings",  # 输入嵌入层
    }

    # 循环遍历名称列表和数组列表
    for name, array in zip(names, arrays):
        # 将完整名称分割为路径的各个部分
        name = name.split("/")
        # 判断是否为不需要的优化器参数或者全局步数
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 如果是优化器相关的变量，则记录跳过的信息
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # 如果是生成器的部分，也跳过
        if name[0] == "generator":
            continue
        # 初始设置模型的指针为模型对象本身
        pointer = model
        # 初始化跳过标志为假
        skipped = False
        # 遍历除第一个元素之外的所有元素（路径中的每个组成部分）
        for m_name in name[1:]:
            # 判断当前指针是否不是指向 FFN 层且名称匹配层级模式
            if not isinstance(pointer, FunnelPositionwiseFFN) and re.fullmatch(r"layer_\d+", m_name):
                # 通过正则表达式提取层数索引
                layer_index = int(re.search(r"layer_(\d+)", m_name).groups()[0])
                # 如果层数索引小于隐藏层数量
                if layer_index < config.num_hidden_layers:
                    block_idx = 0
                    # 确定层数在哪个块中
                    while layer_index >= config.block_sizes[block_idx]:
                        layer_index -= config.block_sizes[block_idx]
                        block_idx += 1
                    # 设置指针为相应的块和层
                    pointer = pointer.blocks[block_idx][layer_index]
                else:
                    # 否则调整层索引并设置指针为相应的层
                    layer_index -= config.num_hidden_layers
                    pointer = pointer.layers[layer_index]
            # 如果名称是 "r" 且当前指针是关联多头注意力层
            elif m_name == "r" and isinstance(pointer, FunnelRelMultiheadAttention):
                # 设置指针为 r 核心
                pointer = pointer.r_kernel
                break
            # 如果名称在映射字典中
            elif m_name in _layer_map:
                # 通过映射更新指针
                pointer = getattr(pointer, _layer_map[m_name])
            else:
                # 尝试获取属性，如果失败打印跳过信息，并设置跳过标志为真
                try:
                    pointer = getattr(pointer, m_name)
                except AttributeError:
                    print(f"Skipping {'/'.join(name)}", array.shape)
                    skipped = True
                    break
        # 如果未跳过
        if not skipped:
            # 如果指针的形状和数组的形状不一致，重塑数组
            if len(pointer.shape) != len(array.shape):
                array = array.reshape(pointer.shape)
            # 如果最后一个名称是核心
            if m_name == "kernel":
                # 对数组进行转置
                array = np.transpose(array)
            # 更新指针的数据
            pointer.data = torch.from_numpy(array)

    # 返回更新后的模型对象
    return model
class FunnelEmbeddings(nn.Module):
    # 初始化函数，接受配置参数并初始化模型
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        # 创建词嵌入层，用于将输入的词索引映射为词嵌入向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建 LayerNorm 层，用于对输入进行归一化处理
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # 创建 Dropout 层，用于对输入进行随机失活处理
        self.dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播函数，接受输入并返回输出
    def forward(
        self, input_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 如果未提供嵌入向量，则通过词嵌入层获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 对输入的嵌入向量进行 LayerNorm 处理
        embeddings = self.layer_norm(inputs_embeds)
        # 对处理后的向量进行随机失活处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入向量
        return embeddings


class FunnelAttentionStructure(nn.Module):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    # 类型标识符，表示 <cls> 标记的类型ID
    cls_token_type_id: int = 2

    # 初始化函数，接受配置参数并初始化辅助结构
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        # 存储配置参数
        self.config = config
        # 创建 Dropout 层，用于对正弦部分的注意力权重进行随机失活处理
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        # 创建 Dropout 层，用于对余弦部分的注意力权重进行随机失活处理
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # 跟踪从原始输入池化的位置，例如，序列长度被减少了多少倍
        self.pooling_mult = None

    # 初始化注意力输入，返回与模型输入相关的注意力输入
    def init_attention_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """Returns the attention inputs associated to the inputs of the model."""
        # inputs_embeds 的形状为 batch_size x seq_len x d_model
        # attention_mask 和 token_type_ids 的形状为 batch_size x seq_len
        # 初始化池化倍数为1
        self.pooling_mult = 1
        # 获取输入序列的长度
        self.seq_len = seq_len = inputs_embeds.size(1)
        # 获取位置嵌入向量
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        # 将 token_type_ids 转换为 token_type_mat
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        # 如果配置了分开处理 <cls> 标记，则创建 <cls> 掩码
        cls_mask = (
            nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config.separate_cls
            else None
        )
        # 返回注意力输入
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    # 将 token_type_ids 转换为 token_type_mat
    def token_type_ids_to_mat(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Convert `token_type_ids` to `token_type_mat`."""
        # 创建 token_type_mat，判断哪些位置的 token_type_id 相同
        token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        # 将 <cls> 视为与 A 和 B 相同段的标记
        cls_ids = token_type_ids == self.cls_token_type_id
        cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        # 返回 token_type_mat
        return cls_mat | token_type_mat

    # 获取位置嵌入向量
    def get_position_embeds(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    def stride_pool_pos(self, pos_id: torch.Tensor, block_index: int):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        # 如果 `config.separate_cls=True`，则在保持 cls 令牌单独的同时对 pos_id 进行汇集。
        if self.config.separate_cls:
            # 在分离的 <cls> 下，我们将 <cls> 视为上一个块的第一个令牌。由于第一个真实块的位置始终为1，上一个块的位置将为 `1 - 2 ** block_index`。
            cls_pos = pos_id.new_tensor([-(2**block_index) + 1])
            # 对 pos_id 进行汇集的操作，忽略首尾的 cls 令牌，如果 `config.truncate_seq=True`，则忽略末尾的 cls 令牌
            pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:]
            return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos: torch.Tensor, stride: int, pooled_pos=None, shift: int = 1) -> torch.Tensor:
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        # 构建 `pos` 和 `pooled_pos` 之间的相对位置向量。
        if pooled_pos is None:
            pooled_pos = pos

        # 参考点为 pooled_pos 的第一个值减去 pos 的第一个值
        ref_point = pooled_pos[0] - pos[0]
        # 移除的数量为 shift 乘以 pooled_pos 长度
        num_remove = shift * len(pooled_pos)
        # 最大距离为参考点加上移除数量乘以 stride
        max_dist = ref_point + num_remove * stride
        # 最小距离为 pooled_pos 的第一个值减去 pos 的最后一个值
        min_dist = pooled_pos[0] - pos[-1]

        return torch.arange(max_dist, min_dist - 1, -stride, dtype=torch.long, device=pos.device)

    def stride_pool(
        self,
        tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
        axis: Union[int, Tuple[int], List[int]],
    ) -> torch.Tensor:
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        # 沿着给定的轴进行 stride 切片来执行汇集。
        if tensor is None:
            return None

        # 如果 axis 是整数的列表或元组，则递归地进行 stride 汇集。
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # 如果 tensor 是张量的列表或元组，则递归地进行 stride 汇集。
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # 处理负轴
        axis %= tensor.ndim

        # 切片操作，在同时满足 `config.separate_cls=True` 和 `config.truncate_seq=True` 的情况下，获取奇数索引的切片
        axis_slice = (
            slice(None, -1, 2) if self.config.separate_cls and self.config.truncate_seq else slice(None, None, 2)
        )
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            # 在 axis 位置上，拼接 cls 切片和 tensor
            tensor = torch.cat([tensor[cls_slice], tensor], axis=axis)
        return tensor[enc_slice]

    def pool_tensor(
        self, tensor: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]], mode: str = "mean", stride: int = 2
):
    ) -> torch.Tensor:
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        # 如果输入的张量为空，则返回空
        if tensor is None:
            return None

        # 如果输入的张量是张量列表或元组，则递归地对每个张量进行池化操作
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor)

        # 如果参数配置中指定分开处理分类向量，则重新排列张量内容
        if self.config.separate_cls:
            suffix = tensor[:, :-1] if self.config.truncate_seq else tensor
            tensor = torch.cat([tensor[:, :1], suffix], dim=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor[:, None, :, None]
        elif ndim == 3:
            tensor = tensor[:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        # 根据不同的池化模式进行池化操作
        if mode == "mean":
            tensor = nn.functional.avg_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = nn.functional.max_pool2d(tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -nn.functional.max_pool2d(-tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")

        # 根据原始张量的维度，返回池化后的张量
        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(
        self, output, attention_inputs: Tuple[torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        # 如果参数配置中仅对查询向量进行池化处理
        if self.config.pool_q_only:
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        else:
            # 池化倍数增加
            self.pooling_mult *= 2
            if self.config.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.config.pooling_type)
        # 返回池化后的输出和更新后的注意力输入
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return output, attention_inputs
    def post_attention_pooling(self, attention_inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        # 解包 attention_inputs，包含位置嵌入、标记类型矩阵、注意力掩码和类别掩码
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        # 如果仅池化查询向量
        if self.config.pool_q_only:
            # 将池化倍数乘以 2
            self.pooling_mult *= 2
            # 如果注意力类型为 "factorized"
            if self.config.attention_type == "factorized":
                # 对位置嵌入进行切片，保留前两个部分，对其余部分进行步幅池化
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            # 对标记类型矩阵进行步幅池化
            token_type_mat = self.stride_pool(token_type_mat, 2)
            # 对类别掩码进行步幅池化
            cls_mask = self.stride_pool(cls_mask, 1)
            # 对注意力掩码进行池化，取最小值
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        # 更新 attention_inputs
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        # 返回更新后的 attention_inputs
        return attention_inputs
# 定义一个函数_relative_shift_gather，传入参数positional_attn（类型：torch.Tensor）、context_len（类型：int）、shift（类型：int），返回类型为torch.Tensor
def _relative_shift_gather(positional_attn: torch.Tensor, context_len: int, shift: int) -> torch.Tensor:
    # 获取positional_attn的shape的参数，分别赋值给batch_size、n_head、seq_len、max_rel_len
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    # 定义max_rel_len，即max_rel_len = 2 * context_len + shift -1，表示可能的相对位置i-j的数量

    # 下面的操作与下面的gather操作相同，但可能代码更为清晰，但效率上较低
    # idxs = context_len + torch.arange(0, context_len).unsqueeze(0) - torch.arange(0, seq_len).unsqueeze(1)
    # # 矩阵为context_len + i-j
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    # 对positional_attn进行形状变换，将其reshape为[batch_size, n_head, max_rel_len, seq_len]
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    # 截取positional_attn的左移后的部分，即positional_attn[:, :, shift:, :]
    positional_attn = positional_attn[:, :, shift:, :]
    # 对positional_attn进行再次形状变换，将其reshape为[batch_size, n_head, seq_len, max_rel_len - shift]
    positional_attn = torch.reshape(positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    # 截取positional_attn的右移前的部分，即positional_attn[..., :context_len]
    positional_attn = positional_attn[..., :context_len]
    # 返回positional_attn
    return positional_attn


# 定义一个类FunnelRelMultiheadAttention，继承自nn.Module类
class FunnelRelMultiheadAttention(nn.Module):
    # 初始化函数，传入参数config（类型：FunnelConfig）、block_index（类型：int），无返回值
    def __init__(self, config: FunnelConfig, block_index: int) -> None:
        super().__init__()
        self.config = config
        self.block_index = block_index
        d_model, n_head, d_head = config.d_model, config.n_head, config.d_head

        # 定义模型的隐藏层dropout
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        # 定义模型的attention层dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        # 定义模型的q_head线性层，输入维度为d_model，输出维度为n_head * d_head，无偏置
        self.q_head = nn.Linear(d_model, n_head * d_head, bias=False)
        # 定义模型的k_head线性层，输入维度为d_model，输出维度为n_head * d_head
        self.k_head = nn.Linear(d_model, n_head * d_head)
        # 定义模型的v_head线性层，输入维度为d_model，输出维度为n_head * d_head
        self.v_head = nn.Linear(d_model, n_head * d_head)

        # 定义模型的r_w_bias参数，维度为[n_head, d_head]，初始值为0
        self.r_w_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        # 定义模型的r_r_bias参数，维度为[n_head, d_head]，初始值为0
        self.r_r_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        # 定义模型的r_kernel参数，维度为[d_model, n_head, d_head]，初始值为0
        self.r_kernel = nn.Parameter(torch.zeros([d_model, n_head, d_head]))
        # 定义模型的r_s_bias参数，维度为[n_head, d_head]，初始值为0
        self.r_s_bias = nn.Parameter(torch.zeros([n_head, d_head]))
        # 定义模型的seg_embed参数，维度为[2, n_head, d_head]，初始值为0
        self.seg_embed = nn.Parameter(torch.zeros([2, n_head, d_head]))

        # 定义模型的post_proj线性层，输入维度为n_head * d_head，输出维度为d_model
        self.post_proj = nn.Linear(n_head * d_head, d_model)
        # 定义模型的layer_norm层，输入维度为d_model，计算均值和方差进行归一化，eps为config.layer_norm_eps
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        # 定义模型的scale参数，值为(1.0 / (d_head**0.5))
        self.scale = 1.0 / (d_head**0.5)
        def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
            """Relative attention score for the positional encodings"""
            # q_head has shape batch_size x sea_len x n_head x d_head
            # 根据论文A.2.2的最终公式，进行因式分解的注意力计算
            if self.config.attention_type == "factorized":
                # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
                # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
                phi, pi, psi, omega = position_embeds
                # Shape n_head x d_head
                u = self.r_r_bias * self.scale
                # Shape d_model x n_head x d_head
                w_r = self.r_kernel

                # Shape batch_size x sea_len x n_head x d_model
                q_r_attention = torch.einsum("binh,dnh->bind", q_head + u, w_r)
                q_r_attention_1 = q_r_attention * phi[:, None]
                q_r_attention_2 = q_r_attention * pi[:, None]

                # Shape batch_size x n_head x seq_len x context_len
                positional_attn = torch.einsum("bind,jd->bnij", q_r_attention_1, psi) + torch.einsum(
                    "bind,jd->bnij", q_r_attention_2, omega
                )
            else:
                # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
                shift = 2 if q_head.shape[1] != context_len else 1
                # Grab the proper positional encoding, shape max_rel_len x d_model
                # 根据论文A.2.1的最终公式，进行普通注意力计算
                r = position_embeds[self.block_index][shift - 1]
                # Shape n_head x d_head
                v = self.r_r_bias * self.scale
                # Shape d_model x n_head x d_head
                w_r = self.r_kernel

                # Shape max_rel_len x n_head x d_model
                r_head = torch.einsum("td,dnh->tnh", r, w_r)
                # Shape batch_size x n_head x seq_len x max_rel_len
                positional_attn = torch.einsum("binh,tnh->bnit", q_head + v, r_head)
                # Shape batch_size x n_head x seq_len x context_len
                # 调整注意力矩阵的相对位置关系
                positional_attn = _relative_shift_gather(positional_attn, context_len, shift)

            if cls_mask is not None:
                # 如果存在 cls_mask，则将注意力矩阵与 cls_mask 相乘
                positional_attn *= cls_mask
            return positional_attn
    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """计算token_type_ids的相对注意力分数"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head的形状为 batch_size x seq_len x n_head x d_head
        # 形状为 n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # 形状为 batch_size x n_head x seq_len x 2
        token_type_bias = torch.einsum("bind,snd->bnis", q_head + r_s_bias, self.seg_embed)
        # 形状为 batch_size x n_head x seq_len x context_len
        token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.shape[2], seq_len, context_len])
        # 形状为 batch_size x n_head x seq_len
        diff_token_type, same_token_type = torch.split(token_type_bias, 1, dim=-1)
        # 形状为 batch_size x n_head x seq_len x context_len
        token_type_attn = torch.where(
            token_type_mat, same_token_type.expand(token_type_mat.shape), diff_token_type.expand(token_type_mat.shape)
        )

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs: Tuple[torch.Tensor],
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        # query 的形状为 batch_size x seq_len x d_model
        # key 和 value 的形状为 batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config.n_head, self.config.d_head

        # 形状为 batch_size x seq_len x n_head x d_head
        q_head = self.q_head(query).view(batch_size, seq_len, n_head, d_head)
        # 形状为 batch_size x context_len x n_head x d_head
        k_head = self.k_head(key).view(batch_size, context_len, n_head, d_head)
        v_head = self.v_head(value).view(batch_size, context_len, n_head, d_head)

        q_head = q_head * self.scale
        # 形状为 n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # 形状为 batch_size x n_head x seq_len x context_len
        content_score = torch.einsum("bind,bjnd->bnij", q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)

        # 合并注意力分数
        attn_score = content_score + positional_attn + token_type_attn

        # 在混合精度训练时保证精度安全
        dtype = attn_score.dtype
        attn_score = attn_score.float()
        # 执行掩码操作
        if attention_mask is not None:
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
        # 注意力概率
        attn_prob = torch.softmax(attn_score, dim=-1, dtype=dtype)
        attn_prob = self.attention_dropout(attn_prob)

        # 注意力输出，形状为 batch_size x seq_len x n_head x d_head
        attn_vec = torch.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # 形状为 batch_size x seq_len x d_model
        attn_out = self.post_proj(attn_vec.reshape(batch_size, seq_len, n_head * d_head))
        attn_out = self.hidden_dropout(attn_out)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)
class FunnelPositionwiseFFN(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        # 初始化神经网络模块
        super().__init__()
        # 线性变换层，将输入特征维度从config.d_model转换到config.d_inner
        self.linear_1 = nn.Linear(config.d_model, config.d_inner)
        # 激活函数
        self.activation_function = ACT2FN[config.hidden_act]
        # 激活函数后的dropout
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        # 再次线性变换层，将特征维度转换回到config.d_model
        self.linear_2 = nn.Linear(config.d_inner, config.d_model)
        # 全连接层的dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # Layer normalization 层
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # 第一个线性变换层
        h = self.linear_1(hidden)
        # 使用激活函数
        h = self.activation_function(h)
        # 应用激活函数后的dropout
        h = self.activation_dropout(h)
        # 第二个线性变换层
        h = self.linear_2(h)
        # dropout
        h = self.dropout(h)
        # 残差连接和Layer normalization操作
        return self.layer_norm(hidden + h)


class FunnelLayer(nn.Module):
    def __init__(self, config: FunnelConfig, block_index: int) -> None:
        # 初始化神经网络模块
        super().__init__()
        # FunnelRelMultiheadAttention模块
        self.attention = FunnelRelMultiheadAttention(config, block_index)
        # FunnelPositionwiseFFN模块
        self.ffn = FunnelPositionwiseFFN(config)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_inputs,
        output_attentions: bool = False,
    ) -> Tuple:
        # 调用attention模块
        attn = self.attention(query, key, value, attention_inputs, output_attentions=output_attentions)
        # 调用FFN模块
        output = self.ffn(attn[0])
        # 返回输出和注意力权重
        return (output, attn[1]) if output_attentions else (output,)


class FunnelEncoder(nn.Module):
    def __init__(self, config: FunnelConfig) -> None:
        # 初始化神经网络模块
        super().__init__()
        # 保存config
        self.config = config
        # FunnelAttentionStructure模块
        self.attention_structure = FunnelAttentionStructure(config)
        # 多层FunnelLayer模块
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([FunnelLayer(config, block_index) for _ in range(block_size)])
                for block_index, block_size in enumerate(config.block_sizes)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义一个方法，其参数包括输入数据、注意力掩码和token类型ID，并返回模型输出的元组或BaseModelOutput对象
    ) -> Union[Tuple, BaseModelOutput]:
        # 没有实现对长张量的池化，所以我们将此掩码转换为与输入嵌入张量相同的类型
        attention_mask = attention_mask.type_as(inputs_embeds)
        # 初始化注意力输入，包括输入嵌入张量、注意力掩码和token类型ID
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # 初始化隐藏状态为输入嵌入张量
        hidden = inputs_embeds
    
        # 如果需要输出隐藏状态，则初始化包含输入嵌入张量的元组，否则为None
        all_hidden_states = (inputs_embeds,) if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组，否则为None
        all_attentions = () if output_attentions else None
    
        # 遍历每个块
        for block_index, block in enumerate(self.blocks):
            # 检查是否需要进行池化
            pooling_flag = hidden.size(1) > (2 if self.config.separate_cls else 1)
            pooling_flag = pooling_flag and block_index > 0
            # 如果需要进行池化，则执行前池化操作
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs
                )
            # 遍历每个层
            for layer_index, layer in enumerate(block):
                # 遍历每个重复次数
                for repeat_index in range(self.config.block_repeats[block_index]):
                    # 检查是否需要进行池化
                    do_pooling = (repeat_index == 0) and (layer_index == 0) and pooling_flag
                    # 如果需要进行池化，则使用池化后的隐藏状态作为查询，否则使用原始隐藏状态
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    # 对当前层进行前向传播，获取层的输出
                    layer_output = layer(query, key, value, attention_inputs, output_attentions=output_attentions)
                    # 更新隐藏状态为当前层的输出的第一个元素
                    hidden = layer_output[0]
                    # 如果进行了池化，则执行后池化操作
                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(attention_inputs)
    
                    # 如果需要输出注意力权重，则将当前层输出的注意力权重追加到all_attentions中
                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    # 如果需要输出隐藏状态，则将当前隐藏状态追加到all_hidden_states中
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden,)
    
        # 如果不需要返回字典形式的输出，则返回包含隐藏状态、所有隐藏状态和所有注意力权重的元组
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        # 否则，返回BaseModelOutput对象，包括最后的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)
# 定义一个函数，将张量 `x` 上采样到 `target_len` 长度，通过在序列长度维度上重复 `stride` 次
def upsample(
    x: torch.Tensor, stride: int, target_len: int, separate_cls: bool = True, truncate_seq: bool = False
) -> torch.Tensor:
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    # 如果 `stride` 等于 1，直接返回输入张量 `x`
    if stride == 1:
        return x
    # 如果 `separate_cls` 为 True，将第一个单元素切片为 `cls`，剩余部分切片为 `x`
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    # 对张量 `x` 在维度 1 上重复 `stride` 次，得到 `output`
    output = torch.repeat_interleave(x, repeats=stride, dim=1)
    # 如果 `separate_cls` 为 True
    if separate_cls:
        # 如果 `truncate_seq` 为 True，在 `output` 上进行填充
        if truncate_seq:
            output = nn.functional.pad(output, (0, 0, 0, stride - 1, 0, 0))
        # 从 `output` 中切片出指定长度 `target_len - 1`，并与 `cls` 连接起来
        output = output[:, : target_len - 1]
        output = torch.cat([cls, output], dim=1)
    else:
        # 从 `output` 中切片出指定长度 `target_len`
        output = output[:, :target_len]
    # 返回结果张量 `output`
    return output


# 定义一个类，表示FunnelDecoder模型
class FunnelDecoder(nn.Module):
    # 初始化函数
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        # 将传入的配置参数保存到类中
        self.config = config
        # 创建FunnelAttentionStructure对象
        self.attention_structure = FunnelAttentionStructure(config)
        # 创建FunnelLayer对象列表，并重复创建 `config.num_decoder_layers` 次
        self.layers = nn.ModuleList([FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    # 前向传播函数
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
        # 将 `final_hidden` 上采样到与 `first_block_hidden` 相同长度，并保存在变量 `upsampled_hidden` 中
        upsampled_hidden = upsample(
            final_hidden,
            stride=2 ** (len(self.config.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            separate_cls=self.config.separate_cls,
            truncate_seq=self.config.truncate_seq,
        )

        # 将 `upsampled_hidden` 与 `first_block_hidden` 相加，得到 `hidden`
        hidden = upsampled_hidden + first_block_hidden
        # 初始化 `all_hidden_states` 和 `all_attentions`
        all_hidden_states = (hidden,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 初始化 `attention_inputs`，并获得初始的注意力输入
        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 遍历每个层
        for layer in self.layers:
            # 将 `hidden`、`hidden`、`hidden` 和 `attention_inputs` 传入FunnelLayer模型中进行前向传播
            layer_output = layer(hidden, hidden, hidden, attention_inputs, output_attentions=output_attentions)
            # 更新 `hidden`
            hidden = layer_output[0]

            # 如果 `output_attentions` 为 True，将 `layer_output` 中的注意力张量加入 `all_attentions` 中
            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            # 如果 `output_hidden_states` 为 True，将 `hidden` 加入 `all_hidden_states` 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden,)

        # 如果 `return_dict` 为 False，返回 `hidden`、`all_hidden_states` 和 `all_attentions`
        if not return_dict:
            return tuple(v for v in [hidden, all_hidden_states, all_attentions] if v is not None)
        # 返回BaseModelOutput对象，其中 `hidden` 为 `hidden`，`all_hidden_states` 为 `all_hidden_states`，`all_attentions` 为 `all_attentions`
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=all_hidden_states, attentions=all_attentions)


# 定义一个类，表示FunnelDiscriminatorPredictions模型
class FunnelDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""
    # 初始化函数，接受FunnelConfig对象作为参数
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置参数保存在对象中
        self.config = config
        # 创建一个线性层，输入和输出维度都为config中指定的d_model
        self.dense = nn.Linear(config.d_model, config.d_model)
        # 创建一个线性层，输入为config中指定的d_model，输出为1
        self.dense_prediction = nn.Linear(config.d_model, 1)

    # 前向传播函数，接受一个torch.Tensor类型的参数并返回一个torch.Tensor类型的结果
    def forward(self, discriminator_hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性层得到隐藏状态
        hidden_states = self.dense(discriminator_hidden_states)
        # 使用配置文件中指定的激活函数对隐藏状态进行处理
        hidden_states = ACT2FN[self.config.hidden_act](hidden_states)
        # 经过输出层得到预测结果，并压缩成一维
        logits = self.dense_prediction(hidden_states).squeeze()
        # 返回预测结果
        return logits
# 定义FunnelPreTrainedModel类为PreTrainedModel的子类，用于处理权重的初始化和预训练模型的下载和加载。
class FunnelPreTrainedModel(PreTrainedModel):
    # 定义config_class类属性为FunnelConfig，load_tf_weights类属性为load_tf_weights_in_funnel，base_model_prefix类属性为"funnel"
    config_class = FunnelConfig
    load_tf_weights = load_tf_weights_in_funnel
    base_model_prefix = "funnel"

    # 初始化权重的方法
    def _init_weights(self, module):
        # 获取当前模块的类名
        classname = module.__class__.__name__
        # 如果类名包含"Linear"
        if classname.find("Linear") != -1:
            # 如果模块有权重属性
            if getattr(module, "weight", None) is not None:
                # 如果没有指定初始化标准差
                if self.config.initializer_std is None:
                    # 获取权重的输入和输出维度
                    fan_out, fan_in = module.weight.shape
                    # 计算标准差
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config.initializer_std
                # 使用正态分布初始化权重
                nn.init.normal_(module.weight, std=std)
            # 如果模块有偏置属性
            if getattr(module, "bias", None) is not None:
                # 使用常数0初始化偏置
                nn.init.constant_(module.bias, 0.0)
        # 如果类名是"FunnelRelMultiheadAttention"
        elif classname == "FunnelRelMultiheadAttention":
            # 使用均匀分布初始化r_w_bias属性
            nn.init.uniform_(module.r_w_bias, b=self.config.initializer_range)
            # 使用均匀分布初始化r_r_bias属性
            nn.init.uniform_(module.r_r_bias, b=self.config.initializer_range)
            # 使用均匀分布初始化r_kernel属性
            nn.init.uniform_(module.r_kernel, b=self.config.initializer_range)
            # 使用均匀分布初始化r_s_bias属性
            nn.init.uniform_(module.r_s_bias, b=self.config.initializer_range)
            # 使用均匀分布初始化seg_embed属性
            nn.init.uniform_(module.seg_embed, b=self.config.initializer_range)
        # 如果类名是"FunnelEmbeddings"
        elif classname == "FunnelEmbeddings":
            # 如果没有指定初始化标准差，默认为1.0
            std = 1.0 if self.config.initializer_std is None else self.config.initializer_std
            # 使用正态分布初始化word_embeddings权重
            nn.init.normal_(module.word_embeddings.weight, std=std)
            # 如果word_embeddings有padding_idx属性
            if module.word_embeddings.padding_idx is not None:
                # 将padding_idx对应的权重初始化为0
                module.word_embeddings.weight.data[module.padding_idx].zero_()


# 定义FunnelClassificationHead类为nn.Module的子类，用于分类任务的头部
class FunnelClassificationHead(nn.Module):
    # 初始化方法，接收一个FunnelConfig对象和一个整数，表示分类类别数
    def __init__(self, config: FunnelConfig, n_labels: int) -> None:
        super().__init__()
        # 定义linear_hidden属性为nn.Linear类，输入维度为config.d_model，输出维度为config.d_model
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        # 定义dropout属性为nn.Dropout类，设置dropout比例为config.hidden_dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 定义linear_out属性为nn.Linear类，输入维度为config.d_model，输出维度为n_labels
        self.linear_out = nn.Linear(config.d_model, n_labels)

    # 前向传播函数，接收一个torch.Tensor类型的hidden输入，返回一个torch.Tensor类型的输出
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # 经过线性变换linear_hidden
        hidden = self.linear_hidden(hidden)
        # 使用tanh激活函数
        hidden = torch.tanh(hidden)
        # 使用dropout进行随机失活
        hidden = self.dropout(hidden)
        # 经过线性变换linear_out
        return self.linear_out(hidden)


# 定义FunnelForPreTrainingOutput类为ModelOutput的子类，表示FunnelForPreTraining的输出类型
@dataclass
class FunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FunnelForPreTraining`].
    """
    Args:
        # 如果`labels`参数提供，则返回损失，类型为`torch.FloatTensor`，形状为`(1,)`
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss of the ELECTRA-style objective.
        
        # 预测头部的预测分数（SoftMax之前每个标记的分数），类型为`torch.FloatTensor`，形状为`(batch_size, sequence_length)`
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        
        # 如果设置了`output_hidden_states=True`或`config.output_hidden_states=True`，则返回模型隐藏状态（嵌入输出以及每一层的输出），类型为`tuple(torch.FloatTensor)`
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        
        # 如果设置了`output_attentions=True`或`config.output_attentions=True`，则返回注意力权重，用于计算自注意力头中的加权平均值，类型为`tuple(torch.FloatTensor)`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    
    # 可选的损失值，类型为`torch.FloatTensor`
    loss: Optional[torch.FloatTensor] = None
    
    # 预测分数，类型为`torch.FloatTensor`
    logits: torch.FloatTensor = None
    
    # 可选的隐藏状态，类型为`tuple(torch.FloatTensor)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 可选的注意力权重，类型为`tuple(torch.FloatTensor)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义变量 FUNNEL_START_DOCSTRING，包含了关于Funnel Transformer模型的说明文档
# 提供了论文链接和模型相关信息
# 继承自PreTrainedModel类，具有该类的通用方法，如下载、保存、调整输入嵌入大小和修剪头部等
# 也是一个PyTorch torch.nn.Module子类，可以按照常规的PyTorch模块使用，并参考PyTorch文档
# 参数：config（FunnelConfig）：包含模型所有参数的模型配置类。用配置文件初始化不加载与模型相关的权重，只是配置。使用from_pretrained方法加载模型权重
# 定义变量 FUNNEL_INPUTS_DOCSTRING，暂时为空
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记在词汇表中的索引。
                # 可以使用[`AutoTokenizer`]获取索引。有关详细信息，请参阅[`PreTrainedTokenizer.encode`]和[PreTrainedTokenizer.__call__`]。
                # [什么是输入ID？](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 避免在填充令牌索引上执行注意力的掩码。在`[0, 1]`中选择掩码值：
                # - 对于**未屏蔽**的标记为1，
                # - 对于**屏蔽**的标记为0。
                # [什么是注意力掩码？](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 段标记索引，指示输入的第一部分和第二部分。在`[0, 1]`中选择索引：
                # - 0对应一个*句子A*标记，
                # - 1对应一个*句子B*标记。
                # [什么是标记类型ID？](../glossary#token-type-ids)
            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选，可以直接传入嵌入表示，而不是传入`input_ids`。如果您希望更多地控制如何将`input_ids`索引转换为关联向量，那么这是有用的，而不是使用模型的内部嵌入查找矩阵。
            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量。
            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                # 是否返回[`~utils.ModelOutput`]而不是普通元组。
# 从基类 FunnelPreTrainedModel 继承的基本 Funnel Transformer 模型，输出原始的隐藏状态，不带有上采样头（也称为解码器）或任何特定任务的头部
class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        # 调用基类的构造方法
        super().__init__(config)

        # 初始化嵌入层
        self.embeddings = FunnelEmbeddings(config)
        # 初始化编码层
        self.encoder = FunnelEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings.word_embeddings = new_embeddings

    # 前向传播函数
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
        # 指定函数返回类型为元组或者BaseModelOutput类型
        ) -> Union[Tuple, BaseModelOutput]:
        # 如果未指定output_attentions则使用配置中的output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states则使用配置中的output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict则使用配置中的use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果input_ids和inputs_embeds同时指定，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了input_ids
        elif input_ids is not None:
            # 如果input_ids存在且没有attention_mask则发出警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取input_ids的形状
            input_shape = input_ids.size()
        # 如果只指定了inputs_embeds
        elif inputs_embeds is not None:
            # 获取inputs_embeds的形状
            input_shape = inputs_embeds.size()[:-1]
        # 如果既没有指定input_ids也没有指定inputs_embeds则抛出数值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 根据input_ids是否存在来确定设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果attention_mask未指定，则创建shape为input_shape的全1张量
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果token_type_ids未指定，则创建shape为input_shape的全0张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # TODO: 处理head_mask

        # 如果未指定inputs_embeds，则使用embeddings方法对input_ids进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # 使用encoder进行编码得到encoder_outputs
        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
# 定义一个 Funnel 模型类，继承自 FunnelPreTrainedModel 类，用于生成原始隐藏状态而不添加特定的输出头部
@add_start_docstrings(
    "The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.",
    FUNNEL_START_DOCSTRING,
)
class FunnelModel(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        super().__init__(config)
        self.config = config
        # 初始化嵌入层
        self.embeddings = FunnelEmbeddings(config)
        # 初始化编码器
        self.encoder = FunnelEncoder(config)
        # 初始化解码器
        self.decoder = FunnelDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.embeddings.word_embeddings = new_embeddings

    # 前向传播函数，用于模型推理
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数，接收输入，并返回输出，输出类型是一个元组或者BaseModelOutput
    def forward(
        ) -> Union[Tuple, BaseModelOutput]:
            # 如果output_attentions不为None，则使用给定值，否则使用self.config.output_attentions的值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果output_hidden_states不为None，则使用给定值，否则使用self.config.output_hidden_states的值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果return_dict不为None，则使用给定值，否则使用self.config.use_return_dict的值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            # 如果input_ids和inputs_embeds都不为None，则抛出异常
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            # 如果只有input_ids，则进行一些预处理
            elif input_ids is not None:
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
            # 如果只有inputs_embeds，则获取其shape
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            # 如果既没有input_ids也没有inputs_embeds，则抛出异常
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
    
            # 如果input_ids不为None，则获取其设备信息，否则获取inputs_embeds的设备信息
            device = input_ids.device if input_ids is not None else inputs_embeds.device
    
            # 如果attention_mask为None，则创建一个全为1的attention_mask
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            # 如果token_type_ids为None，则创建一个全为0的token_type_ids
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    
            # TODO: deal with head_mask
            # 如果inputs_embeds为None，则使用self.embeddings函数获取其嵌入
            if inputs_embeds is None:
                inputs_embeds = self.embeddings(input_ids)
    
            # 使用self.encoder函数进行编码器计算
            encoder_outputs = self.encoder(
                inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
    
            # 使用self.decoder函数进行解码器计算
            decoder_outputs = self.decoder(
                final_hidden=encoder_outputs[0],
                first_block_hidden=encoder_outputs[1][self.config.block_sizes[0]],
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
    
            # 如果return_dict为False，则将输出整合为一个元组进行返回
            if not return_dict:
                idx = 0
                outputs = (decoder_outputs[0],)
                if output_hidden_states:
                    idx += 1
                    outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
                if output_attentions:
                    idx += 1
                    outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
                return outputs
    
            # 如果return_dict为True，则使用BaseModelOutput整合输出进行返回
            return BaseModelOutput(
                last_hidden_state=decoder_outputs[0],
                hidden_states=(encoder_outputs.hidden_states + decoder_outputs.hidden_states)
                if output_hidden_states
                else None,
                attentions=(encoder_outputs.attentions + decoder_outputs.attentions) if output_attentions else None,
            )
# 添加起始的文档字符串，描述该模型用途
add_start_docstrings(
    """
    Funnel Transformer model with a binary classification head on top as used during pretraining for identifying
    generated tokens.
    """,
    FUNNEL_START_DOCSTRING,
)

# 定义用于预训练的Funnel模型
class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的构造函数初始化模型
        super().__init__(config)

        # 初始化Funnel模型
        self.funnel = FunnelModel(config)
        # 初始化用于预测的判别器预测器
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        # 初始化权重并进行最终处理
        self.post_init()

    # 重写forward方法，对模型进行前向传播
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

        # 使用输入数据进行判别器的前向传播
        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取判别器的序列输出
        discriminator_sequence_output = discriminator_hidden_states[0]

        # 使用判别器序列输出进行预测并得到输出日志概率
        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        # 如果存在标签数据
        if labels is not None:
            # 计算二元交叉熵损失函数
            loss_fct = nn.BCEWithLogitsLoss()
            # 如果存在注意力掩码
            if attention_mask is not None:
                active_loss = attention_mask.view(-1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.view(-1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                # 计算损失
                loss = loss_fct(active_logits, active_labels.float())
            else:
                # 计算损失
                loss = loss_fct(logits.view(-1, discriminator_sequence_output.shape[1]), labels.float())

        # 如果不需要返回字典形式的结果
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回FunnelForPreTrainingOutput对象
        return FunnelForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
# 使用装饰器添加文档字符串描述，指定语言模型头部的 Funnel Transformer 模型
@add_start_docstrings("""Funnel Transformer Model with a `language modeling` head on top.""", FUNNEL_START_DOCSTRING)
# 定义 FunnelForMaskedLM 类，继承自 FunnelPreTrainedModel
class FunnelForMaskedLM(FunnelPreTrainedModel):
    # 定义权重绑定的键
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数，接受 FunnelConfig 类型的参数配置
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 FunnelModel 对象，使用给定的配置参数
        self.funnel = FunnelModel(config)
        # 创建线性层作为语言模型的头部，输入维度为配置中的 d_model，输出维度为词汇表大小
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入的线性层
    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    # 设置输出嵌入的新嵌入层
    def set_output_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.lm_head = new_embeddings

    # 使用装饰器添加文档字符串描述和示例代码，描述模型的前向传播逻辑
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    # 定义前向传播方法，接受多种输入参数
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
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None, token_type_ids: torch.LongTensor = None, position_ids: torch.LongTensor = None, inputs_embeds: torch.Tensor = None, head_mask: torch.Tensor = None, labels: torch.LongTensor = None, output_attentions: bool = None, output_hidden_states: bool = None, return_dict: bool = None) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 设置是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用funnel模型处理输入数据
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型的最后隐藏状态
        last_hidden_state = outputs[0]

        # 对最后隐藏状态应用语言模型头部进行预测
        prediction_logits = self.lm_head(last_hidden_state)

        # 初始化masked_lm_loss为None
        masked_lm_loss = None

        # 如果存在labels，则计算masked_lm_loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典格式的结果
        if not return_dict:
            output = (prediction_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回MaskedLMOutput对象，包括loss、logits、hidden_states和attentions
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个在顶部具有序列分类/回归头的Funnel变换器模型（在最后一个隐藏状态的第一个时间步上有两个线性层），例如用于GLUE任务
# 此处是添加了模型文档注释的类装饰器
class FunnelForSequenceClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类构造函数
        super().__init__(config)
        # 设定标签数量
        self.num_labels = config.num_labels
        # 存储配置信息
        self.config = config

        # 初始化Funnel基础模型
        self.funnel = FunnelBaseModel(config)
        # 初始化分类器
        self.classifier = FunnelClassificationHead(config, config.num_labels)
        # 初始化权重并应用最终处理
        self.post_init()

    # 此处是为模型定义文档注释的方法装饰器
    # 此处是为模型定义示例代码的方法装饰器
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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Funnel 模型进行前向传播
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
        # 获取池化后的输出
        pooled_output = last_hidden_state[:, 0]
        # 将池化后的输出传递给分类器，得到分类的 logits
        logits = self.classifier(pooled_output)

        # 初始化损失值
        loss = None
        # 如果有标签
        if labels is not None:
            # 如果问题类型未定义
            if self.config.problem_type is None:
                # 如果标签数量为1，则问题类型为回归
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                # 如果标签数量大于1且标签的数据类型是 long 或 int，则问题类型为单标签分类
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                # 否则为多标签分类
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数并计算损失值
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不返回字典，则返回 logits 和其他输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 SequenceClassifierOutput 对象，其中包含损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型文档字符串，描述了该模型是在Funnel Transformer模型基础上添加了一个多选分类头部的模型
# 包含了对模型配置的文档说明
class FunnelForMultipleChoice(FunnelPreTrainedModel):
    # 初始化方法
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类初始化方法
        super().__init__(config)
        # 创建FunnelBaseModel实例
        self.funnel = FunnelBaseModel(config)
        # 创建FunnelClassificationHead实例，用于多选分类
        self.classifier = FunnelClassificationHead(config, 1)
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受多个输入参数
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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择个数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的input_ids重塑为二维张量
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将attention_mask重塑为二维张量
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将token_type_ids重塑为二维张量
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将inputs_embeds重塑为三维张量
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用funnel模型进行前向传播
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
        # 获取池化后的输出
        pooled_output = last_hidden_state[:, 0]
        # 将池化后的输出传入分类器，得到logits
        logits = self.classifier(pooled_output)
        # 重塑logits为二维张量
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典格式的结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回多项选择模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为了在 Funnel 模型的基础上构建一个标记分类任务的模型，例如命名实体识别（NER）任务，添加了一个线性层作为隐藏状态输出的顶部
# 权重的处理参考了 Funnel 预训练模型
class FunnelForTokenClassification(FunnelPreTrainedModel):
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类构造函数初始化模型参数
        super().__init__(config)
        # 设置模型标签的数量
        self.num_labels = config.num_labels

        # 初始化 Funnel 模型
        self.funnel = FunnelModel(config)
        # 添加丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 添加线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受输入并返回模型输出
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
        # 如果 return_dict 为 None，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对输入进行处理并调用 Funnel 模型的前向传播
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型的最后隐藏状态并添加丢弃层
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        # 使用分类器进行分类，得到预测的 logits
        logits = self.classifier(last_hidden_state)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用返回字典，则将输出包装成元组返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 类封装输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 为了在 Funnel 模型的基础上构建一个用于提取式问答任务（如 SQuAD）的跨度分类任务的模型
    # 一个线性层，用于在隐藏状态输出之上计算“跨度起始 logits”和“跨度结束 logits”
    """,
    # 使用 FUNNEL_START_DOCSTRING 的值
    FUNNEL_START_DOCSTRING,
)
# 定义用于问题回答的Funnel模型类，继承自FunnelPreTrainedModel
class FunnelForQuestionAnswering(FunnelPreTrainedModel):
    # 初始化方法，接受一个FunnelConfig类的实例对象
    def __init__(self, config: FunnelConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置num_labels属性为config的num_labels属性
        self.num_labels = config.num_labels

        # 创建Funnel模型对象
        self.funnel = FunnelModel(config)
        # 创建一个线性层，输入维度为config的hidden_size，输出维度为config的num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播的方法
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
    # 定义一个函数，用于执行问答模型的推理操作
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
    
        # 判断是否设置了 return_dict，若未设置则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传入 Funnel 模型中进行推理，获取输出
        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取最后一层的隐藏状态
        last_hidden_state = outputs[0]

        # 通过最后一层隐藏状态计算答案起始位置和结束位置的预测值
        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # 如果设置了起始位置和结束位置，计算损失
        if start_positions is not None and end_positions is not None:
            # 如果是多GPU情况，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典，直接返回输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问答模型的输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```