# `.\models\chinese_clip\modeling_chinese_clip.py`

```
# 定义了编码为 UTF-8 的文件头声明
# 版权声明及许可信息，使用 Apache License, Version 2.0 许可协议
# 导入所需的库和模块
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入特定的自定义模块和类
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

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档化的模型检查点信息
_CHECKPOINT_FOR_DOC = "OFA-Sys/chinese-clip-vit-base-patch16"
# 用于文档化的配置信息
_CONFIG_FOR_DOC = "ChineseCLIPConfig"

# 可用的预训练模型列表
CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "OFA-Sys/chinese-clip-vit-base-patch16",
    # 可在 https://huggingface.co/models?filter=chinese_clip 查看所有 Chinese-CLIP 模型
]


# 定义对比损失函数，来自 transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    对比损失函数，计算交叉熵损失

    Args:
        logits (torch.Tensor): 模型预测的对比结果

    Returns:
        torch.Tensor: 计算的对比损失
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# 定义 Chinese-CLIP 的损失函数，包括文本和图像对比损失的平均值
def chinese_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    Chinese-CLIP 损失函数，包括文本和图像对比损失的平均值

    Args:
        similarity (torch.Tensor): 模型预测的相似性分数

    Returns:
        torch.Tensor: 计算的 Chinese-CLIP 总损失
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())  # 转置后计算图像损失
    return (caption_loss + image_loss) / 2.0


@dataclass
class ChineseCLIPOutput(ModelOutput):
    """
    Chinese-CLIP 模型输出类，继承自 ModelOutput 类
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image: (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text: (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`ChineseCLIPTextModel`].
        image_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`ChineseCLIPVisionModel`].
        text_model_output: (`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ChineseCLIPTextModel`].
        vision_model_output: (`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`ChineseCLIPVisionModel`].
    """

    # Optional attribute: Loss value representing contrastive loss for image-text similarity
    loss: Optional[torch.FloatTensor] = None
    # Attribute: Scores representing similarity between image and text embeddings
    logits_per_image: torch.FloatTensor = None
    # Attribute: Scores representing similarity between text and image embeddings
    logits_per_text: torch.FloatTensor = None
    # Attribute: Embeddings of text data after projection from ChineseCLIPTextModel
    text_embeds: torch.FloatTensor = None
    # Attribute: Embeddings of image data after projection from ChineseCLIPVisionModel
    image_embeds: torch.FloatTensor = None
    # Attribute: Output object from ChineseCLIPTextModel, including pooling and cross-attentions
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    # Attribute: Output object from ChineseCLIPVisionModel, including pooling and cross-attentions
    vision_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None

    def to_tuple(self) -> Tuple[Any]:
        # Method: Converts all attributes to a tuple; certain attributes are converted to tuples recursively
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 从 transformers.models.bert.modeling_bert.BertEmbeddings 复制并修改为 ChineseCLIPTextEmbeddings 类
class ChineseCLIPTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 定义词嵌入层，将词汇 ID 映射到隐藏表示大小的向量空间
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，将位置 ID 映射到隐藏表示大小的向量空间
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 定义类型嵌入层，将类型 ID（如句子 A 或句子 B）映射到隐藏表示大小的向量空间
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用 TensorFlow 模型变量名，保持与 TensorFlow 模型兼容，方便加载 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义 dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置编码类型，通常是绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，保存位置 ID 的张量，用于序列化时持久化保存
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，保存类型 ID 的张量，初始为全零张量，用于序列化时持久化保存
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
    # 定义方法forward，接收输入参数input_ids, inputs_embeds, token_type_ids, position_ids, past_key_values_length，并返回torch.Tensor对象
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果输入参数input_ids不为None，则获取其形状作为input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取inputs_embeds的形状除了最后一个维度的所有维度作为input_shape
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度seq_length，从input_shape的第二个维度获取
        seq_length = input_shape[1]

        # 如果position_ids为None，则从self.position_ids中获取一部分切片，其范围从past_key_values_length到seq_length+past_key_values_length
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置token_type_ids为已注册缓冲区中的值，通常是全零。这种情况通常发生在自动生成时，注册缓冲区帮助在模型跟踪过程中没有传递token_type_ids时的用户，解决问题＃5664
        if token_type_ids is None:
            # 如果self对象具有属性"token_type_ids"
            if hasattr(self, "token_type_ids"):
                # 则从self.token_type_ids中获取一部分切片，其范围从第二维度的第一个元素到seq_length
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 将buffered_token_type_ids在第一维度复制input_shape[0]次，在第二维度复制seq_length次
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，将token_type_ids设置为全零的torch.Tensor对象，形状为input_shape，数据类型为torch.long，设备为self.position_ids所在的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果inputs_embeds为None，则使用self.word_embeddings对input_ids进行嵌入处理得到inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 使用self.token_type_embeddings对token_type_ids进行嵌入处理得到token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings为inputs_embeds和token_type_embeddings的加和
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型为"absolute"，则使用self.position_embeddings对position_ids进行嵌入处理并加到embeddings上
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对embeddings进行LayerNorm归一化处理
        embeddings = self.LayerNorm(embeddings)

        # 对embeddings进行dropout处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的embeddings作为方法的输出
        return embeddings
# 从 transformers.models.clip.modeling_clip.CLIPVisionEmbeddings 复制而来，将 CLIP 模型改为了 ChineseCLIP
class ChineseCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为隐藏大小
        self.image_size = config.image_size  # 图像大小
        self.patch_size = config.patch_size  # 补丁大小

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # 类别嵌入作为可学习参数

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )  # 创建卷积层用于图像补丁嵌入

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中补丁的数量
        self.num_positions = self.num_patches + 1  # 位置嵌入的数量为补丁数加一
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 创建位置嵌入层
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)  # 注册位置 ID 缓冲区，非持久性

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # 获取批次大小
        target_dtype = self.patch_embedding.weight.dtype  # 目标数据类型为补丁嵌入权重的数据类型
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 将像素值通过卷积层得到补丁嵌入

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 对补丁嵌入进行展平和转置

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # 扩展类别嵌入以匹配批次大小
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接类别嵌入和补丁嵌入

        embeddings = embeddings + self.position_embedding(self.position_ids)  # 添加位置嵌入到嵌入张量中
        return embeddings


# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制而来，将 Bert 自注意力机制改为了 ChineseCLIPText
class ChineseCLIPTextSelfAttention(nn.Module):
    # 初始化函数，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 检查隐藏大小是否能被注意力头的数量整除，如果不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对位置编码，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并修改为 ChineseCLIPTextSelfOutput 类
class ChineseCLIPTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization 层，归一化操作，eps 是归一化过程中的稳定性参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，以 config.hidden_dropout_prob 的概率随机置零输入张量
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换操作，将 hidden_states 映射到相同维度空间
        hidden_states = self.dense(hidden_states)
        # Dropout 操作，防止过拟合
        hidden_states = self.dropout(hidden_states)
        # Layer normalization，通过归一化操作来稳定和加速训练
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertAttention 复制并修改为 ChineseCLIPTextAttention 类
class ChineseCLIPTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # ChineseCLIPTextSelfAttention 对象，用于计算自注意力机制
        self.self = ChineseCLIPTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # ChineseCLIPTextSelfOutput 对象，将自注意力机制的输出进行线性变换、归一化和 dropout 处理
        self.output = ChineseCLIPTextSelfOutput(config)
        # 存储需要剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可以剪枝的注意力头并获取对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头
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
        # 执行自注意力机制，计算输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 对自注意力机制的输出进行处理，再次线性变换和归一化
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其包含在输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，添加注意力权重到输出中
        return outputs


class ChineseCLIPVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        super().__init__()  # 调用父类的初始化方法
        self.config = config  # 保存配置对象
        self.embed_dim = config.hidden_size  # 获取隐藏大小作为嵌入维度
        self.num_heads = config.num_attention_heads  # 获取注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads  # 计算每个头的维度
        # 检查嵌入维度是否可以被注意力头的数量整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5  # 缩放因子，按照论文设定
        self.dropout = config.attention_dropout  # 注意力机制中的丢弃率

        # 初始化线性层，用于投影键、值、查询和输出
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 将张量重塑为适合多头注意力的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，接受隐藏状态张量和是否输出注意力权重的选项
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # 获取隐藏状态张量的形状信息：批量大小、时间步长、嵌入维度
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 获取查询投影
        query_states = self.q_proj(hidden_states) * self.scale
        # 获取键投影，并调整形状以匹配多头注意力的计算需求
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        # 获取值投影，并调整形状以匹配多头注意力的计算需求
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # 定义投影后张量的形状
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # 调整查询状态的形状，以便进行多头注意力的计算
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # 调整键状态的形状，以便进行多头注意力的计算
        key_states = key_states.view(*proj_shape)
        # 调整值状态的形状，以便进行多头注意力的计算
        value_states = value_states.view(*proj_shape)

        # 获取源序列的长度
        src_len = key_states.size(1)
        # 计算注意力权重
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # 检查注意力权重的尺寸是否符合预期
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 对注意力权重进行 softmax 操作
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 如果需要输出注意力权重，则进行形状调整以保留梯度
        if output_attentions:
            # 这个操作有些笨拙，但是必须执行，以确保 attn_weights 保持其梯度
            # 为了实现这一点，需要两次重塑，并在接下来的操作中重复使用 attn_weights
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        # 对注意力权重应用 dropout
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 计算注意力输出
        attn_output = torch.bmm(attn_probs, value_states)

        # 检查注意力输出的尺寸是否符合预期
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 将注意力输出的形状调整为预期形状
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        # 对输出进行最终投影
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped
# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->ChineseCLIPText
class ChineseCLIPTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入的 hidden_size 维度转换为 intermediate_size 维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置文件中的激活函数名称选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层
        hidden_states = self.dense(hidden_states)
        # 应用选定的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->ChineseCLIPText
class ChineseCLIPTextOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将 intermediate_size 维度转换为 hidden_size 维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化 LayerNorm 层，对 hidden_size 维度进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，应用概率为 hidden_dropout_prob 的 dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层
        hidden_states = self.dense(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 应用 LayerNorm，并将结果与输入的 input_tensor 相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->ChineseCLIPVision
class ChineseCLIPVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 根据配置文件中的激活函数名称选择相应的激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 初始化两个全连接层，分别将 hidden_size 维度转换为 intermediate_size 维度和 intermediate_size 维度转换为 hidden_size 维度
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 应用选定的激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 将结果再次应用到第二个全连接层
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->ChineseCLIPText
class ChineseCLIPTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置 feed forward 操作的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度，通常为 1
        self.seq_len_dim = 1
        # 初始化 self.attention 为 ChineseCLIPTextAttention 实例，使用给定的配置
        self.attention = ChineseCLIPTextAttention(config)
        # 是否为解码器模型
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨注意力机制但不是解码器模型，则引发异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化 self.crossattention 为 ChineseCLIPTextAttention 实例，使用给定的配置和绝对位置嵌入
            self.crossattention = ChineseCLIPTextAttention(config, position_embedding_type="absolute")
        # 初始化 self.intermediate 为 ChineseCLIPTextIntermediate 实例，使用给定的配置
        self.intermediate = ChineseCLIPTextIntermediate(config)
        # 初始化 self.output 为 ChineseCLIPTextOutput 实例，使用给定的配置
        self.output = ChineseCLIPTextOutput(config)
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
        # 获取自注意力的缓存的键/值对，位置在 past_key_value 的第1和第2个位置
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 排除最后一个元素，因为它是自注意力缓存
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
                                              
        cross_attn_present_key_value = None
        # 如果是解码器且存在编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 检查是否具有交叉注意力层，若没有则抛出异常
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 获取交叉注意力的缓存的键/值对，位置在 past_key_value 的倒数第2和倒数第1个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到 present_key_value 的第3、第4个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对 attention_output 应用前向传播分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 定义一个方法，用于执行前向传播的一个步骤，接收注意力输出作为输入参数
    def feed_forward_chunk(self, attention_output):
        # 调用模型的 intermediate 方法处理注意力输出，得到中间层的输出
        intermediate_output = self.intermediate(attention_output)
        # 调用模型的 output 方法处理中间层输出和注意力输出，得到最终层的输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层的输出作为该方法的结果
        return layer_output
# 定义一个用于处理视觉输入的自定义层，继承自 `nn.Module`
class ChineseCLIPVisionLayer(nn.Module):
    def __init__(self, config: ChineseCLIPConfig):
        super().__init__()
        # 设置嵌入维度为配置中的隐藏大小
        self.embed_dim = config.hidden_size
        # 初始化自注意力层，使用给定配置
        self.self_attn = ChineseCLIPVisionAttention(config)
        # 初始化第一个层归一化层，对隐藏状态进行归一化，使用给定的 epsilon
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 初始化MLP（多层感知机）层，用于处理视觉输入
        self.mlp = ChineseCLIPVisionMLP(config)
        # 初始化第二个层归一化层，对隐藏状态进行归一化，使用给定的 epsilon
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

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
        # 保存残差连接
        residual = hidden_states

        # 应用第一个层归一化层到隐藏状态
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力层处理归一化后的隐藏状态，根据需要输出注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 保存当前隐藏状态作为下一步的残差连接输入
        residual = hidden_states
        # 应用第二个层归一化层到隐藏状态
        hidden_states = self.layer_norm2(hidden_states)
        # 使用MLP处理归一化后的隐藏状态
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 输出包含处理后的隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 `transformers.models.bert.modeling_bert.BertPooler` 复制代码，并替换 Bert 为 ChineseCLIPText
class ChineseCLIPTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态转换为与隐藏大小相同的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用双曲正切激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地取第一个标记对应的隐藏状态来“汇集”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记对应的隐藏状态通过线性层
        pooled_output = self.dense(first_token_tensor)
        # 应用双曲正切激活函数
        pooled_output = self.activation(pooled_output)
        # 返回汇集后的输出
        return pooled_output


class ChineseCLIPPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和预训练模型下载加载的抽象类。
    """

    # 指定配置类为 ChineseCLIPConfig
    config_class = ChineseCLIPConfig
    # 设置基础模型前缀为 "chinese_clip"
    base_model_prefix = "chinese_clip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        # 如果是视觉嵌入模块
        if isinstance(module, ChineseCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            # 初始化类别嵌入的权重
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 初始化补丁嵌入的权重
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 初始化位置嵌入的权重
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果是文本嵌入模块
        elif isinstance(module, ChineseCLIPTextEmbeddings):
            # 初始化词嵌入的权重
            nn.init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            # 初始化位置嵌入的权重
            nn.init.normal_(module.position_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            # 初始化标记类型嵌入的权重
            nn.init.normal_(module.token_type_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引对应的权重置零
            for embedding in [module.word_embeddings, module.position_embeddings, module.token_type_embeddings]:
                if embedding.padding_idx is not None:
                    embedding.weight.data[embedding.padding_idx].zero_()
        # 如果是视觉注意力模块
        elif isinstance(module, ChineseCLIPVisionAttention):
            factor = self.config.initializer_factor
            # 输入投影的标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 输出投影的标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化查询投影的权重
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            # 初始化键投影的权重
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            # 初始化值投影的权重
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            # 初始化输出投影的权重
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果是视觉MLP模块
        elif isinstance(module, ChineseCLIPVisionMLP):
            factor = self.config.initializer_factor
            # 输入投影的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 全连接层的标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化第一个全连接层的权重
            nn.init.normal_(module.fc1.weight, std=fc_std)
            # 初始化第二个全连接层的权重
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果是整体模型
        elif isinstance(module, ChineseCLIPModel):
            # 初始化文本投影层的权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 初始化视觉投影层的权重
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        # 如果是 LayerNorm 层
        if isinstance(module, nn.LayerNorm):
            # 将偏置项置零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 初始化线性层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将偏置项置零
            if module.bias is not None:
                module.bias.data.zero_()
# 中文 CLIP 模型的起始文档字符串，描述了该模型是 PyTorch 中的一个子类，应当像常规的 PyTorch Module 一样使用。
# 模型的具体使用和行为相关的事项应参考 PyTorch 文档。
CHINESE_CLIP_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ChineseCLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 中文 CLIP 模型输入文本的文档字符串部分，暂未填写具体内容。
CHINESE_CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的标记索引，用于词汇表中的词汇。
            Indices of input sequence tokens in the vocabulary.

            # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取详细信息。

            [What are input IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 注意力遮罩，用于避免对填充标记索引执行注意力操作。遮罩值选择在 `[0, 1]` 之间：

            - 1 表示**未遮罩**的标记，
            - 0 表示**遮罩**的标记。

            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 段落标记索引，指示输入的第一部分和第二部分。索引在 `[0, 1]` 之间选择：

            - 0 对应于 *句子 A* 的标记，
            - 1 对应于 *句子 B* 的标记。

            [What are token type IDs?](../glossary#token-type-ids)

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。

            [What are position IDs?](../glossary#position-ids)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块中的选定头部置零的遮罩。遮罩值选择在 `[0, 1]` 之间：

            - 1 表示**未遮罩**的头部，
            - 0 表示**遮罩**的头部。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示而不是 `input_ids`。如果需要对如何将 `input_ids` 索引转换为相关向量进行更多控制，这是很有用的。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详细信息请参见返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详细信息请参见返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 `~utils.ModelOutput` 而不是普通元组。
"""
CHINESE_CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`ChineseCLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CHINESE_CLIP_INPUTS_DOCSTRING = r"""
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记的索引，用于表示词汇表中的标记。默认情况下会忽略填充部分。
            # 可以使用 `AutoTokenizer` 获得这些索引。详情请见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            [What are input IDs?](../glossary#input-ids)

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于在填充的标记索引上避免执行注意力操作。遮罩值在 `[0, 1]` 之间：

            - 1 表示**未被遮罩**的标记，
            - 0 表示**被遮罩**的标记。

            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引值在 `[0, 1]` 之间：

            - 0 对应*句子 A* 的标记，
            - 1 对应*句子 B* 的标记。

            [What are token type IDs?](../glossary#token-type-ids)

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选取范围为 `[0, config.max_position_embeddings - 1]`。

            [What are position IDs?](../glossary#position-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下会忽略填充部分。像素值可以使用 `AutoImageProcessor` 获得。
            # 详情请见 `ChineseCLIPImageProcessor.__call__`。

        return_loss (`bool`, *optional*):
            # 是否返回对比损失。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详情请见返回的张量中的 `attentions` 部分。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详情请见返回的张量中的 `hidden_states` 部分。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而非普通元组。
"""
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制并修改为 ChineseCLIPTextEncoder

class ChineseCLIPTextEncoder(nn.Module):
    """
    ChineseCLIPTextEncoder 类，继承自 nn.Module，用于处理中文文本的编码器。

    Args:
        config (object): 配置对象，包含编码器的参数设置。

    Attributes:
        config (object): 编码器的配置参数。
        layer (nn.ModuleList): 包含多个 ChineseCLIPTextLayer 层的模块列表，用于构建编码器的层。
        gradient_checkpointing (bool): 是否启用梯度检查点，默认为 False，表示不启用。

    Methods:
        forward: 前向传播方法，接受多个输入和参数，返回编码器的输出结果。
    """

    def __init__(self, config):
        """
        ChineseCLIPTextEncoder 的初始化方法。

        Args:
            config (object): 配置对象，包含编码器的参数设置。
        """
        super().__init__()
        self.config = config
        # 创建多个 ChineseCLIPTextLayer 层，构成编码器的层列表
        self.layer = nn.ModuleList([ChineseCLIPTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

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
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果设置了输出隐藏状态，初始化空元组；否则置为None
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了输出注意力权重，初始化空元组；否则置为None
        all_self_attentions = () if output_attentions else None
        # 如果设置了输出注意力权重并且模型配置中包含跨注意力，初始化空元组；否则置为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启了梯度检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果use_cache为True，则给出警告并设置use_cache为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果use_cache为True，则初始化空元组；否则置为None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，将当前隐藏状态加入all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有头部掩码，则获取当前解码层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果有过去的键值对，则获取当前解码层的过去键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
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
                # 否则，直接调用解码层模块进行前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为解码层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果use_cache为True，则将解码层输出的最后一个元素加入next_decoder_cache
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，将解码层输出的第二个元素加入all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中包含跨注意力，将解码层输出的第三个元素加入all_cross_attentions
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，将最终的隐藏状态加入all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典格式结果，以元组形式返回多个元素的结果
        if not return_dict:
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
        # 否则，以BaseModelOutputWithPastAndCrossAttentions格式返回结果
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
        super().__init__()
        self.config = config
        # 创建一个由多个 `ChineseCLIPVisionLayer` 实例组成的列表，每个实例代表一个编码器层
        self.layers = nn.ModuleList([ChineseCLIPVisionLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点默认关闭
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
        # Determine whether to use the provided `output_attentions` or fall back to the model's default setting
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # Determine whether to use the provided `output_hidden_states` or fall back to the model's default setting
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine whether to return a dictionary (`return_dict`) or a plain tuple based on the provided setting or model default
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize empty tuples to store encoder states and attentions if not requested
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Start with the input embeddings for processing through the transformer layers
        hidden_states = inputs_embeds
        # Iterate through each transformer layer
        for idx, encoder_layer in enumerate(self.layers):
            # Store hidden states of each layer if requested
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            # Use gradient checkpointing during training if enabled
            if self.gradient_checkpointing and self.training:
                # Apply the encoder layer function with gradient checkpointing
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    output_attentions,
                )
            else:
                # Apply the encoder layer function without gradient checkpointing
                layer_outputs = encoder_layer(
                    hidden_states,
                    output_attentions=output_attentions,
                )

            # Update hidden states with the outputs of the current layer
            hidden_states = layer_outputs[0]

            # Store attentions of each layer if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Store final encoder states if requested
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # Return either a plain tuple or a `BaseModelOutput` depending on `return_dict` setting
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
class ChineseCLIPVisionTransformer(nn.Module):
    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 初始化中文CLIP视觉嵌入层
        self.embeddings = ChineseCLIPVisionEmbeddings(config)
        # 添加预层归一化层
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化中文CLIP视觉编码器
        self.encoder = ChineseCLIPVisionEncoder(config)
        # 添加后层归一化层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

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
        前向传播函数
        Args:
            pixel_values (Optional[torch.FloatTensor]): 输入像素值的张量
            output_attentions (Optional[bool]): 是否输出注意力权重
            output_hidden_states (Optional[bool]): 是否输出隐藏状态
            return_dict (Optional[bool]): 是否使用返回字典格式

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 如果不使用返回字典，返回元组；否则返回带池化的基础模型输出
        """
        # 如果未提供像素值，抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值嵌入到嵌入层中
        hidden_states = self.embeddings(pixel_values)
        # 应用预层归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 将隐藏状态传递给编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后的隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 池化输出，取第一个位置的隐藏状态
        pooled_output = last_hidden_state[:, 0, :]
        # 应用后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不使用返回字典格式，返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 使用返回字典格式，返回带池化的基础模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    "CHINESE_CLIP的文本模型，不包含任何顶部头部或投影。",
    CHINESE_CLIP_START_DOCSTRING,
)
class ChineseCLIPTextModel(ChineseCLIPPreTrainedModel):
    """
    
    模型可以作为编码器（仅自注意力）或解码器使用，在后一种情况下，会在自注意力层之间添加交叉注意力层，遵循[Attention is
    all you need](https://arxiv.org/abs/1706.03762)的架构描述，作者为Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser和Illia Polosukhin。
    """
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """



    # 设置配置类为 ChineseCLIPTextConfig
    config_class = ChineseCLIPTextConfig

    # 模型初始化函数，接受配置和是否添加池化层的标志
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化文本嵌入层
        self.embeddings = ChineseCLIPTextEmbeddings(config)
        
        # 初始化文本编码器
        self.encoder = ChineseCLIPTextEncoder(config)

        # 根据是否添加池化层来初始化池化层，或者设为 None
        self.pooler = ChineseCLIPTextPooler(config) if add_pooling_layer else None

        # 调用后续初始化函数
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数，接受多个输入参数，详细见下方装饰器说明
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
@add_start_docstrings(
    """The vision model from CHINESE_CLIP without any head or projection on top.""",
    CHINESE_CLIP_START_DOCSTRING,
)
class ChineseCLIPVisionModel(ChineseCLIPPreTrainedModel):
    # 设置配置类
    config_class = ChineseCLIPVisionConfig
    # 主要输入名称
    main_input_name = "pixel_values"

    def __init__(self, config: ChineseCLIPVisionConfig):
        # 调用父类构造函数初始化
        super().__init__(config)
        # 初始化视觉模型
        self.vision_model = ChineseCLIPVisionTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回嵌入的补丁嵌入层
        return self.vision_model.embeddings.patch_embedding

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
        ```"""
        # 如果未指定返回字典，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 委托给视觉模型进行前向传播
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(CHINESE_CLIP_START_DOCSTRING)
class ChineseCLIPModel(ChineseCLIPPreTrainedModel):
    # 设置配置类
    config_class = ChineseCLIPConfig
    def __init__(self, config: ChineseCLIPConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 检查文本配置是否为正确类型，若不是则引发数值错误异常
        if not isinstance(config.text_config, ChineseCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ChineseCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查视觉配置是否为正确类型，若不是则引发数值错误异常
        if not isinstance(config.vision_config, ChineseCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type ChineseCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从配置对象中获取文本配置和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型，其中文本模型不添加池化层
        self.text_model = ChineseCLIPTextModel(text_config, add_pooling_layer=False)
        self.vision_model = ChineseCLIPVisionTransformer(vision_config)

        # 初始化视觉投影层和文本投影层，不包含偏置项
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # 初始化对数尺度参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并进行最终处理
        self.post_init()

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
        ```"""
        # 根据需要使用 CHINESE_CLIP 模型的配置中的一些字段（如果指定），而不是视觉和文本组件的配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用文本模型处理输入，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中提取池化后的特征表示
        pooled_output = text_outputs[0][:, 0, :]
        # 将池化后的特征表示投影到文本特征空间
        text_features = self.text_projection(pooled_output)

        # 返回文本特征表示
        return text_features
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
        ```"""
        # 使用 CHINESE_CLIP 模型的配置来覆盖一些字段（如果指定了的话），而不是视觉和文本组件的字段。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型，传入像素值、是否输出注意力权重、是否输出隐藏状态以及是否返回字典等参数
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉模型的输出中获取第二个元素，即汇总的输出（pooled_output）
        pooled_output = vision_outputs[1]  # pooled_output
        # 使用视觉投影层对汇总的输出进行投影，得到最终的图像特征表示
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征表示
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