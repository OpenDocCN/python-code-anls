# `.\transformers\models\chinese_clip\modeling_chinese_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 2022 年 OFA-Sys 团队作者和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，不附带任何明示或暗示的担保。
# 请查看许可证以获取特定语言的权限和限制。
""" PyTorch Chinese-CLIP 模型。"""

# 导入所需的库
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入相关模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_chinese_clip import ChineseCLIPConfig, ChineseCLIPTextConfig, ChineseCLIPVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "OFA-Sys/chinese-clip-vit-base-patch16"
_CONFIG_FOR_DOC = "ChineseCLIPConfig"

# Chinese-CLIP 预训练模型存档列表
CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "OFA-Sys/chinese-clip-vit-base-patch16",
    # 查看所有 Chinese-CLIP 模型 https://huggingface.co/models?filter=chinese_clip
]

# 对比损失函数，参考 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
# 从 transformers.models.clip.modeling_clip.contrastive_loss 复制而来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# Chinese-CLIP 损失函数
def chinese_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

# Chinese-CLIP 模型输出类
@dataclass
class ChineseCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`ChineseCLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`ChineseCLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ChineseCLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ChineseCLIPVisionModel`].
    """

    # 初始化变量并指定其类型，这些变量用于存储损失、预测结果、文本和图像的嵌入向量以及模型的输出
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None

    # 将实例对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 将所有属性存入元组，但跳过文本和视觉模型输出，因为它们需要进一步处理为元组
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 从 transformers.models.bert.modeling_bert.BertEmbeddings 复制并修改为 ChineseCLIPTextEmbeddings 类
class ChineseCLIPTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position, and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 创建词嵌入层，将词索引映射为隐藏状态向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，将位置索引映射为隐藏状态向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建 token_type 嵌入层，将 token 类型索引映射为隐藏状态向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 不使用蛇形命名以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        # 创建 LayerNorm 层，用于对隐藏状态向量进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，用于在模型训练时随机断开部分神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 在内存中是连续的，并在序列化时被导出
        # 根据 position_ids 创建 position_embedding_type（绝对位置编码或相对位置编码）
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册 position_ids 缓冲区，持久性为 False
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册 token_type_ids 缓冲区，持久性为 False
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果输入的是 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状，除去最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为 None，则设置其为从预训练模型的位置编码中提取的长度为 seq_length 的片段
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置 token_type_ids 为构造函数中注册的缓冲区，其中全为零，通常在自动生成时出现，
        # 注册的缓冲区在模型跟踪时有助于用户，无需传递 token_type_ids，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入的嵌入是空的，则使用 word_embeddings 函数将 input_ids 转换为嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token_type_embeddings 相加
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置嵌入的类型是 "absolute"，则添加位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对 embeddings 进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回 embeddings
        return embeddings
# 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings复制代码，并将CLIP->ChineseCLIP
class ChineseCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# 从transformers.models.bert.modeling_bert.BertSelfAttention复制代码，并将Bert->ChineseCLIPText
class ChineseCLIPTextSelfAttention(nn.Module):
    # 初始化函数，用于创建一个多头注意力层对象
    def __init__(self, config, position_embedding_type=None):
        # 调用父类初始化函数
        super().__init__()
        # 检查隐藏层大小是否能被注意力头的数量整除，如果不能且配置中没有嵌入大小，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置对象的属性，包括注意力头的数量、每个注意力头的大小以及所有注意力头的总大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建用于丢弃部分注意力权重的 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键-查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量转置以便进行注意力分数计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量的形状调整为指定的形状，以便用于计算注意力分数
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        # 对调张量的维度顺序，以便进行多头注意力计算
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，用于计算多头自注意力或注意力交叉项
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从transformers.models.bert.modeling_bert.BertSelfOutput中复制并修改为ChineseCLIPTextSelfOutput
class ChineseCLIPTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将输入特征映射到相同维度的输出特征
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义 LayerNormalization 层，用于归一化输入数据
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义 Dropout 层，以一定概率丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对输入的隐藏状态进行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对全连接操作的结果进行随机丢弃一定比例的神经元
        hidden_states = self.dropout(hidden_states)
        # 将丢弃部分神经元后的结果与输入张量相加，并进行 LayerNormalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的结果张量
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertAttention中复制并修改为ChineseCLIPTextAttention
class ChineseCLIPTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 定义 ChineseCLIPTextSelfAttention 层，用于计算自注意力
        self.self = ChineseCLIPTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # 定义 ChineseCLIPTextSelfOutput 层，用于处理自注意力的输出
        self.output = ChineseCLIPTextSelfOutput(config)
        # 初始化需要剪枝的注意力头集合为空集
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录已剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 ChineseCLIPTextSelfAttention 处理输入的隐藏状态
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用 ChineseCLIPTextSelfOutput 处理自注意力的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则添加到输出中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回处理后的结果
        return outputs


class ChineseCLIPVisionAttention(nn.Module):
    # 多头注意力机制，参考 'Attention Is All You Need' 论文
    class MultiHeadAttention(nn.Module):
    
        # 初始化函数，接受配置参数
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_dim = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.embed_dim // self.num_heads
            # 检查是否能够整除
            if self.head_dim * self.num_heads != self.embed_dim:
                raise ValueError(
                    f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                    f" {self.num_heads})."
                )
            # 缩放因子
            self.scale = self.head_dim**-0.5
            self.dropout = config.attention_dropout
    
            # 线性变换层
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
        # 将张量重塑为指定形状
        def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
        # 前向传播函数
        def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # 获取隐藏状态张量的维度信息
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 获取查询投影
        query_states = self.q_proj(hidden_states) * self.scale
        # 获取键投影，并调整形状以适应多头注意力机制
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        # 获取值投影，并调整形状以适应多头注意力机制
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # 设置投影后的形状
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # 计算注意力权重
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 检查注意力权重的形状是否正确
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 使用softmax函数计算注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 如果需要输出注意力权重，则对其进行处理以保持梯度
        if output_attentions:
            # 这个操作有点奇怪，但是它是必需的，
            # 为了确保 attn_weights 保持梯度。
            # 为了做到这一点，attn_weights 必须重塑两次，
            # 并且必须在接下来的操作中重新使用
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # 使用dropout对注意力权重进行处理
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 计算注意力输出
        attn_output = torch.bmm(attn_probs, value_states)

        # 检查注意力输出的形状是否正确
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 调整注意力输出的形状以符合预期
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        # 应用输出投影
        attn_output = self.out_proj(attn_output)

        # 返回注意力输出和重塑后的注意力权重
        return attn_output, attn_weights_reshaped
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制代码，并将类名从 Bert 修改为 ChineseCLIPText
class ChineseCLIPTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征大小转换为中间层大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏层激活函数是字符串形式，则将其映射到对应的函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行特征转换
        hidden_states = self.dense(hidden_states)
        # 应用中间层激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制代码，并将类名从 Bert 修改为 ChineseCLIPText
class ChineseCLIPTextOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将中间层大小转换为隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 使用 LayerNorm 进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 添加 Dropout 层，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行特征转换
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states)
        # 将输入特征与转换后的特征相加，并通过 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.clip.modeling_clip.CLIPMLP 复制代码，并将类名从 CLIP 修改为 ChineseCLIPVision
class ChineseCLIPVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 将激活函数字符串映射到对应的函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 创建第一个全连接层，将输入特征大小转换为中间层大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建第二个全连接层，将中间层大小转换为隐藏层大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过第一个全连接层进行特征转换，并应用激活函数
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        # 通过第二个全连接层进行特征转换
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertLayer 复制代码，并将类名从 Bert 修改为 ChineseCLIPText
class ChineseCLIPTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设定用于前向传播的 FeedForward 层的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度索引
        self.seq_len_dim = 1
        # 创建自注意力层
        self.attention = ChineseCLIPTextAttention(config)
        # 判断是否是解码器
        self.is_decoder = config.is_decoder
        # 判断是否添加了跨层注意力
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果不是解码器，则抛出异常
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建跨层注意力层
            self.crossattention = ChineseCLIPTextAttention(config, position_embedding_type="absolute")
        # 创建中间层
        self.intermediate = ChineseCLIPTextIntermediate(config)
        # 创建输出层
        self.output = ChineseCLIPTextOutput(config)
    # 定义一个前向传播函数，用于模型的前向推断
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 如果有过去的键/值对存在且不为None，则获取decoder的单向自注意力缓存键/值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力机制，得到自注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出的第一个元素，即注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 输出为自注意力输出的除最后一个元素之外的所有元素
            outputs = self_attention_outputs[1:-1]
            # 获取当前的键/值对
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是decoder，则输出包括自注意力权重
            outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，则添加自注意力
            present_key_value = None

        # 如果是decoder且存在encoder_hidden_states
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果当前层不具有交叉注意力层，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 获取交叉注意力的过去键/值对，位置在past_key_value元组的倒数第二个到最后一个
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力机制
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出的第一个元素，即注意力输出
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力的输出到outputs中
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果我们输出注意力权重，则添加交叉注意力
            # 将交叉注意力的当前键/值对添加到当前的键/值对中
            present_key_value = present_key_value + cross_attention_outputs[-1]

        # 将注意力输出传递给前馈网络，并应用分块机制
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将前馈网络的输出添加到outputs中
        outputs = (layer_output,) + outputs

        # 如果是decoder，将注意力键/值对作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回outputs
        return outputs
    # 执行神经网络的前向传播，处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 中间层的输出，传入注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 输出层的输出，传入中间层的输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层的输出
        return layer_output
# 定义一个自定义的神经网络层，用于处理中文CLIP模型的视觉输入
class ChineseCLIPVisionLayer(nn.Module):
    def __init__(self, config: ChineseCLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = ChineseCLIPVisionAttention(config)  # 初始化自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 初始化LayerNorm层
        self.mlp = ChineseCLIPVisionMLP(config)  # 初始化MLP层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 初始化第二个LayerNorm层

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)  # 对输入进行LayerNorm处理
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )  # 使用自注意力机制处理输入
        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)  # 对处理后的输入再次进行LayerNorm处理
        hidden_states = self.mlp(hidden_states)  # 使用MLP处理输入
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs


# 从transformers.models.bert.modeling_bert.BertPooler复制并修改为ChineseCLIPTextPooler
class ChineseCLIPTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 初始化线性层
        self.activation = nn.Tanh()  # 初始化激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]  # 取第一个token对应的隐藏状态
        pooled_output = self.dense(first_token_tensor)  # 使用线性层处理隐藏状态
        pooled_output = self.activation(pooled_output)  # 使用激活函数处理输出
        return pooled_output


class ChineseCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ChineseCLIPConfig  # 设置配置类
    base_model_prefix = "chinese_clip"  # 设置基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        # 如果是 ChineseCLIPVisionEmbeddings 模块
        if isinstance(module, ChineseCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            # 初始化 class_embedding
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 初始化 patch_embedding
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 初始化 position_embedding
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果是 ChineseCLIPTextEmbeddings 模块
        elif isinstance(module, ChineseCLIPTextEmbeddings):
            # 初始化 word_embeddings
            nn.init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            # 初始化 position_embeddings
            nn.init.normal_(module.position_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            # 初始化 token_type_embeddings
            nn.init.normal_(module.token_type_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            # 对于有 padding_idx 的 embedding，将其对应位置的权重置为 0
            for embedding in [module.word_embeddings, module.position_embeddings, module.token_type_embeddings]:
                if embedding.padding_idx is not None:
                    embedding.weight.data[embedding.padding_idx].zero_()
        # 如果是 ChineseCLIPVisionAttention 模块
        elif isinstance(module, ChineseCLIPVisionAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化 q_proj, k_proj, v_proj, out_proj
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果是 ChineseCLIPVisionMLP 模块
        elif isinstance(module, ChineseCLIPVisionMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化 fc1, fc2
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果是 ChineseCLIPModel 模块
        elif isinstance(module, ChineseCLIPModel):
            # 初始化 text_projection
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 初始化 visual_projection
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        # 如果是 nn.LayerNorm 模块
        if isinstance(module, nn.LayerNorm):
            # 将 bias 置为 0，将 weight 置为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 nn.Linear 模块
        if isinstance(module, nn.Linear):
            # 初始化权重和偏置
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
# 定义中文 CLIP 模型的起始文档字符串，提供模型的基本说明和参数信息
CHINESE_CLIP_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ChineseCLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# 定义中文 CLIP 模型输入文本的文档字符串，提供关于输入的说明
CHINESE_CLIP_TEXT_INPUTS_DOCSTRING = r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                输入序列标记在词汇表中的索引。

                可以使用 [`AutoTokenizer`] 获取这些索引。详情请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

                [什么是输入ID？](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                避免对填充标记索引执行注意力的掩码。掩码值在 `[0, 1]` 之间：

                - 对于**未屏蔽**的标记，为 1，
                - 对于**已屏蔽**的标记，为 0。

                [什么是注意力掩码？](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                分段标记索引，指示输入的第一部分和第二部分。索引在 `[0, 1]` 之间：

                - 0 对应于*句子 A* 标记，
                - 1 对应于*句子 B* 标记。

                [什么是分段标记ID？](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                每个输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。

                [什么是位置ID？](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                选择自注意力模块的特定头部进行空值化的掩码。掩码值在 `[0, 1]` 之间：

                - 1 表示该头部**未被屏蔽**，
                - 0 表示该头部**被屏蔽**。

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                可选择直接传递嵌入表示，而不是传递 `input_ids`。如果您希望更多地控制如何将 `input_ids` 索引转换为相关向量，则这很有用，而不是使用模型的内部嵌入查找矩阵。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的 `attentions`。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量下的 `hidden_states`。
            return_dict (`bool`, *optional*):
                是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 用于文档字符串的中文 CLIP 模型输入的参数说明
CHINESE_CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下，如果提供了填充(padding)，将会被忽略。可以使用 [`AutoImageProcessor`] 获得像素值。有关详细信息，请参见 [`ChineseCLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多细节，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 用于文档字符串的中文 CLIP 模型输入的参数说明
CHINESE_CLIP_INPUTS_DOCSTRING = r"""
# 空字符串，用作占位符，暂无代码需要注释
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列 token 在词汇表中的索引。默认情况下会忽略填充。可通过 [`AutoTokenizer`] 获取。
            # 参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取详细信息。
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩以避免在填充的 token 索引上执行注意力。遮罩值为 `[0, 1]`：
            # - 1 表示**未遮罩**的 token，
            # - 0 表示**已遮罩**的 token。
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选在 `[0, 1]`：
            # - 0 对应于 *句子 A* 的 token，
            # - 1 对应于 *句子 B* 的 token。
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列 token 在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]`。
            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下会忽略填充。可通过 [`AutoImageProcessor`] 获取。
            # 参见 [`ChineseCLIPImageProcessor.__call__`] 获取详细信息。
        return_loss (`bool`, *optional*):
            # 是否返回对比损失。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
# 从transformers.models.bert.modeling_bert.BertEncoder复制并修改为ChineseCLIPText
class ChineseCLIPTextEncoder(nn.Module):
    # 初始化ChineseCLIPTextEncoder类
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建包含多个ChineseCLIPTextLayer对象的层
        self.layer = nn.ModuleList([ChineseCLIPTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数，接受一个torch.Tensor类型的参数并返回一个元组，其中包含torch.Tensor类型的元素，或者一个BaseModelOutputWithPastAndCrossAttentions类型的元素
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，则初始化一个空元组，否则初始化为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组，否则初始化为None
        all_self_attentions = () if output_attentions else None
        # 如果输出交叉注意力权重且配置中包含交叉注意力，则初始化一个空元组，否则初始化为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用渐变检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果使用缓存，则发出警告并将use_cache设置为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果使用缓存，则初始化一个空元组，否则初始化为None
        next_decoder_cache = () if use_cache else None
        # 对每个解码器层进行迭代
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果头掩码不为None，则获取当前层的头掩码，否则初始化为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果过去的键值不为None，则获取当前层的过去键值，否则初始化为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用渐变检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用渐变检查点函数来计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层的__call__方法来计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出的最后一个元素添加到next_decoder_cache中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的输出的第二个元素添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中包含交叉注意力，则将当前层的输出的第三个元素添加到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 返回一个元组，其中包含不为None的元素
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则，返回一个BaseModelOutputWithPastAndCrossAttentions类型的结果，包含指定的属性
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class ChineseCLIPVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ChineseCLIPVisionEncoderLayer`].

    Args:
        config: ChineseCLIPConfig
    """

    def __init__(self, config: ChineseCLIPConfig):
        # 初始化 ChineseCLIPVisionEncoder 类
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建包含多个 ChineseCLIPVisionLayer 实例的模块列表，数量为配置中指定的隐藏层数
        self.layers = nn.ModuleList([ChineseCLIPVisionLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # 如果指定了输出的注意力权重张量，则使用指定的值；否则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果指定了输出的隐藏状态张量，则使用指定的值；否则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了返回字典形式的输出，则使用指定的值；否则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果不需要返回隐藏状态，则将其初始化为空元组；否则为 None
        encoder_states = () if output_hidden_states else None
        # 如果不需要返回所有注意力权重，则将其初始化为空元组；否则为 None
        all_attentions = () if output_attentions else None

        # 将输入的嵌入向量赋值给隐藏状态张量
        hidden_states = inputs_embeds
        # 遍历编码器的每一层
        for idx, encoder_layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到隐藏状态元组中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # 如果启用了渐变检查点并且处于训练模式，则使用渐变检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    output_attentions,
                )
            # 否则直接调用编码器层的前向传播函数
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    output_attentions=output_attentions,
                )

            # 更新隐藏状态张量为编码器层的输出的第一个张量
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到注意力元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回元组形式的输出
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则返回一个基本模型输出对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
# 定义一个用于处理中文 CLIP 视觉模型的类
class ChineseCLIPVisionTransformer(nn.Module):
    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 初始化中文 CLIP 视觉嵌入层
        self.embeddings = ChineseCLIPVisionEmbeddings(config)
        # 添加预层归一化层，用于处理嵌入向量
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化中文 CLIP 视觉编码器
        self.encoder = ChineseCLIPVisionEncoder(config)
        # 添加后层归一化层，用于处理编码器输出
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 为模型的前向传播方法添加文档字符串和返回值注释
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ChineseCLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        # 根据参数设置是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数设置是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则引发数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值转换为嵌入向量
        hidden_states = self.embeddings(pixel_values)
        # 对嵌入向量进行预层归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用编码器处理嵌入向量
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 提取池化输出，即序列第一个位置的隐藏状态
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不使用返回字典，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果使用返回字典，则返回包含各种输出的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 定义一个中文 CLIP 文本模型类，继承自 ChineseCLIPPreTrainedModel
@add_start_docstrings(
    "The text model from CHINESE_CLIP without any head or projection on top.",
    CHINESE_CLIP_START_DOCSTRING,
)
class ChineseCLIPTextModel(ChineseCLIPPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    # 使用 ChineseCLIPTextConfig 类作为配置类
    config_class = ChineseCLIPTextConfig

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        self.config = config

        # 初始化文本嵌入层
        self.embeddings = ChineseCLIPTextEmbeddings(config)
        # 初始化文本编码器
        self.encoder = ChineseCLIPTextEncoder(config)

        # 如果需要添加池化层，则初始化池化层，否则为 None
        self.pooler = ChineseCLIPTextPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的注意力头进行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CHINESE_CLIP_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用 add_start_docstrings 装饰器为 ChineseCLIPVisionModel 添加文档字符串，
# 描述该模型是从 CHINESE_CLIP 模型中提取的视觉模型，没有额外的头或顶部投影。
# CHINESE_CLIP_START_DOCSTRING 是 CHINESE_CLIP 模型的文档字符串。
class ChineseCLIPVisionModel(ChineseCLIPPreTrainedModel):
    # 指定配置类为 ChineseCLIPVisionConfig
    config_class = ChineseCLIPVisionConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法
    def __init__(self, config: ChineseCLIPVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建视觉模型，使用 ChineseCLIPVisionTransformer 初始化
        self.vision_model = ChineseCLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        # 返回视觉模型的嵌入补丁嵌入层
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ChineseCLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, ChineseCLIPVisionModel

        >>> model = ChineseCLIPVisionModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```py"""
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播方法，并返回结果
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 使用 add_start_docstrings 装饰器为 ChineseCLIPModel 添加文档字符串，
# 描述该模型是 CHINESE_CLIP 模型的衍生类。
# CHINESE_CLIP_START_DOCSTRING 是 CHINESE_CLIP 模型的文档字符串。
class ChineseCLIPModel(ChineseCLIPPreTrainedModel):
    # 指定配置类为 ChineseCLIPConfig
    config_class = ChineseCLIPConfig
    # 初始化方法，接受一个 ChineseCLIPConfig 类型的参数
    def __init__(self, config: ChineseCLIPConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 检查文本配置是否为 ChineseCLIPTextConfig 类型，若不是则引发 ValueError 异常
        if not isinstance(config.text_config, ChineseCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ChineseCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查视觉配置是否为 ChineseCLIPVisionConfig 类型，若不是则引发 ValueError 异常
        if not isinstance(config.vision_config, ChineseCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type ChineseCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将文本配置和视觉配置存储到局部变量中
        text_config = config.text_config
        vision_config = config.vision_config

        # 将投影维度、文本嵌入维度和视觉嵌入维度存储到实例变量中
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建中文文本模型对象，不添加池化层
        self.text_model = ChineseCLIPTextModel(text_config, add_pooling_layer=False)
        # 创建中文视觉 Transformer 模型对象
        self.vision_model = ChineseCLIPVisionTransformer(vision_config)

        # 创建视觉投影层，将视觉嵌入维度映射到投影维度
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # 创建文本投影层，将文本嵌入维度映射到投影维度
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        # 创建对数尺度参数，用于缩放模型输出
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型前向方法添加文档字符串
    @add_start_docstrings_to_model_forward(CHINESE_CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the final [CLS] hidden state of Text-Transformer.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ChineseCLIPModel

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> inputs = tokenizer(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        >>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        ```py"""
        # Use CHINESE_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        # 设置输出注意力权重的标志，如果未指定则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的标志，如果未指定则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的标志，如果未指定则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取文本模型的输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化后的文本输出
        pooled_output = text_outputs[0][:, 0, :]
        # 对池化后的文本输出应用文本投影层
        text_features = self.text_projection(pooled_output)

        # 返回文本特征
        return text_features

    @add_start_docstrings_to_model_forward(CHINESE_CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the final [CLS] hidden state of Vision-Transformer.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, ChineseCLIPModel

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        >>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        ```py"""
        # Use CHINESE_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the pixel values and other optional arguments to the vision model
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the pooled output from vision model
        pooled_output = vision_outputs[1]  # pooled_output
        # Apply visual projection to the pooled output to get image features
        image_features = self.visual_projection(pooled_output)

        # Return the image features
        return image_features

    @add_start_docstrings_to_model_forward(CHINESE_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ChineseCLIPOutput, config_class=ChineseCLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```