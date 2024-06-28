# `.\models\altclip\modeling_altclip.py`

```py
# 导入 math 模块，用于数学运算
import math
# 导入 dataclass 用于创建数据类，用于存储数据而无需手动编写常规方法
from dataclasses import dataclass
# 导入类型提示相关的模块
from typing import Any, List, Optional, Tuple, Union

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.utils.checkpoint

# 导入自定义的激活函数映射表
from ...activations import ACT2FN
# 导入模型输出相关的类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndProjection,
)
# 导入预训练模型的基类
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数，用于分块处理前向传播
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入辅助工具函数
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入 AltCLIP 相关的配置类
from .configuration_altclip import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预定义的检查点和配置文档
_CHECKPOINT_FOR_DOC = "BAAI/AltCLIP"
_CONFIG_FOR_DOC = "AltCLIPConfig"

# 预训练模型存档列表
ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BAAI/AltCLIP",
    # 可在 https://huggingface.co/models?filter=altclip 查看所有 AltCLIP 模型
]

# AltCLIP 模型的开始文档字符串，描述了模型继承自 PreTrainedModel，以及模型参数配置等
ALTCLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# AltCLIP 文本输入的文档字符串，用于描述模型接收的文本输入格式
ALTCLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的token索引，形状为(batch_size, sequence_length)，通过AutoTokenizer获取。
            # 参见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__获取详情。
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充token索引上执行注意力的掩码。掩码值在[0, 1]之间：
            # - 1表示**未被掩码**的token，
            # - 0表示**被掩码**的token。
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 输入序列token在位置嵌入中的位置索引。选择范围为[0, config.max_position_embeddings - 1]。
            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中的attentions字段包含更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中的hidden_states字段包含更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回utils.ModelOutput而不是普通元组。
# 定义一个原始字符串文档，描述了输入参数的详细说明和类型
ALTCLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素数值。默认情况下将忽略填充。可以使用 [`AutoImageProcessor`] 获取像素值。
            查看 [`CLIPImageProcessor.__call__`] 获取更多详情。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。返回的张量中的 `attentions` 更多细节。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。返回的张量中的 `hidden_states` 更多细节。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义一个原始字符串文档，描述了输入参数的详细说明和类型
ALTCLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记的索引。默认情况下将忽略填充。
            可以使用 [`AutoTokenizer`] 获取这些索引。
            查看 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取更多详情。
            [什么是输入 IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充的令牌索引上执行注意力的掩码。掩码值在 `[0, 1]` 中选择：

            - 1 表示 **未掩码** 的令牌，
            - 0 表示 **掩码** 的令牌。

            [什么是注意力掩码?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列令牌在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]` 内。

            [什么是位置 ID?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素数值。默认情况下将忽略填充。
            可以使用 [`AutoImageProcessor`] 获取像素值。
            查看 [`CLIPImageProcessor.__call__`] 获取更多详情。
        return_loss (`bool`, *optional*):
            是否返回对比损失。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。返回的张量中的 `attentions` 更多细节。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。返回的张量中的 `hidden_states` 更多细节。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是普通元组。
"""
# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html 改编而来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 使用交叉熵损失计算对比损失，目标标签为 logits 的长度
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算对比损失的加权平均值
    caption_loss = contrastive_loss(similarity)
    # 对转置的相似性张量计算对比损失
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
# 从 transformers.models.clip.modeling_clip.CLIPOutput 复制并更改为 AltCLIP
class AltCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            图像与文本相似性的对比损失。
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            `image_embeds` 和 `text_embeds` 之间的缩放点积分数。表示图像与文本的相似性分数。
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            `text_embeds` 和 `image_embeds` 之间的缩放点积分数。表示文本与图像的相似性分数。
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            通过对 [`AltCLIPTextModel`] 的汇总输出应用投影层获得的文本嵌入。
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            通过对 [`AltCLIPVisionModel`] 的汇总输出应用投影层获得的图像嵌入。
        text_model_output(`BaseModelOutputWithPooling`):
            [`AltCLIPTextModel`] 的输出。
        vision_model_output(`BaseModelOutputWithPooling`):
            [`AltCLIPVisionModel`] 的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        # 将对象转换为元组形式，但排除 `text_model_output` 和 `vision_model_output` 属性
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制并更改为 AltRoberta
class AltRobertaEmbeddings(nn.Module):
    """
    与 BertEmbeddings 相同，但是对位置嵌入的索引进行了微小的调整。
    """

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制
    # 初始化函数，用于初始化一个模型实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，将词汇表大小、隐藏层大小作为参数，指定填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，将最大位置编码数和隐藏层大小作为参数
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，将标记类型词汇表大小和隐藏层大小作为参数
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 不使用蛇形命名以保持与 TensorFlow 模型变量名的一致性，并能够加载任何 TensorFlow 检查点文件
        # 创建层归一化层，将隐藏层大小和层归一化的 epsilon 参数作为参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，将隐藏层的 dropout 概率作为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_embedding_type 指定位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置 id 缓冲区，用于存储从 0 到 config.max_position_embeddings - 1 的序列
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册标记类型 id 缓冲区，用零填充，与位置 id 的大小相同，数据类型为长整型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置填充标记的索引，与 config.pad_token_id 相同
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层，将最大位置编码数、隐藏层大小以及填充标记的索引作为参数
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数，接受输入参数并进行模型前向传播计算
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果没有给定位置标识符，则根据输入的标记标识符创建位置标识符。任何填充的标记保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        else:
            # 如果没有输入标记标识符，则从输入嵌入创建位置标识符
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 将token_type_ids设置为构造函数中注册的缓冲区，该缓冲区全为零，通常在自动生成时出现，
        # 注册的缓冲区在不传递token_type_ids时帮助用户跟踪模型，解决问题＃5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 获取已注册的缓冲区中的token_type_ids，并扩展以匹配输入形状
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果模型未定义token_type_ids，则创建全零的tensor
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果没有输入嵌入，则从输入标记标识符创建嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据token_type_ids获取token类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的嵌入表示，包括输入嵌入、token类型嵌入和位置嵌入（如果是绝对位置编码）
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 应用LayerNorm层进行归一化处理
        embeddings = self.LayerNorm(embeddings)
        # 应用dropout进行正则化
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        提供的是直接的嵌入。我们无法推断哪些是填充的，因此只生成顺序位置标识符。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序位置标识符，从padding_idx + 1到sequence_length + padding_idx + 1
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->AltRoberta
class AltRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否能够被注意力头数整除，如果不是则抛出错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 线性变换层，用于生成查询、键、值
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果是相对位置编码，则需要额外的距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器（Transformer 架构中的一部分）
        self.is_decoder = config.is_decoder

    # 将线性变换后的张量重新组织为注意力分数的张量形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 这里没有直接写 forward 函数，而是展示了函数签名及参数说明
        pass

# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput
class AltRobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从 transformers.models.roberta.modeling_roberta.RobertaAttention 复制并修改为 AltRobertaAttention 类
class AltRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 self 层，用于自注意力机制
        self.self = AltRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 output 层，用于处理自注意力输出
        self.output = AltRobertaSelfOutput(config)
        # 初始化一个空集合，用于存储被修剪的注意力头信息
        self.pruned_heads = set()

    # 修剪不需要的注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找需要修剪的注意力头和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪过的注意力头信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
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
        # 使用 self 层进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用 output 层处理自注意力的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力信息，则添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，还可以添加注意力
        return outputs


# 从 transformers.models.roberta.modeling_roberta.RobertaIntermediate 复制并修改为 AltRobertaIntermediate 类
class AltRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，将隐藏状态映射到中间状态
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 中间激活函数根据配置选择
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性映射
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.roberta.modeling_roberta.RobertaOutput 复制为 AltRobertaOutput 类
class AltRobertaOutput(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为中间大小，输出大小为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个层归一化层，对隐藏状态进行归一化处理，设置 epsilon 为配置中的层归一化 epsilon
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，以配置中的隐藏 dropout 概率为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受两个张量作为输入，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 对 dropout 后的隐藏状态与输入张量进行残差连接，并进行层归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.roberta.modeling_roberta.RobertaLayer复制代码，修改为AltRoberta
class AltRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1
        self.seq_len_dim = 1
        # 初始化注意力层，使用AltRobertaAttention
        self.attention = AltRobertaAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力但不作为解码器使用，抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力层，使用AltRobertaAttention，位置嵌入类型为"absolute"
            self.crossattention = AltRobertaAttention(config, position_embedding_type="absolute")
        # 初始化中间层，使用AltRobertaIntermediate
        self.intermediate = AltRobertaIntermediate(config)
        # 初始化输出层，使用AltRobertaOutput
        self.output = AltRobertaOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention on the input hidden_states using the attention module
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # Retrieve the attention output from self_attention_outputs
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Exclude the first and last elements from self_attention_outputs
            outputs = self_attention_outputs[1:-1]
            # Retrieve the present key/value tuple from self_attention_outputs
            present_key_value = self_attention_outputs[-1]
        else:
            # Include all elements except the first one from self_attention_outputs
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention using crossattention module
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # Retrieve the attention output from cross_attention_outputs
            attention_output = cross_attention_outputs[0]
            # Combine outputs with cross attentions from cross_attention_outputs
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking to the forward pass of feed_forward_chunk method
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # Include layer_output in the outputs tuple
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            # Append present_key_value to outputs tuple
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # Pass attention_output through intermediate and output layers
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.roberta.modeling_roberta.RobertaEncoder 复制并修改为 AltRoberta
class AltRobertaEncoder(nn.Module):
    # 初始化函数，设置模型的配置和层列表
    def __init__(self, config):
        super().__init__()
        # 保存模型配置
        self.config = config
        # 创建包含多个 AltRobertaLayer 层的列表，层数由配置决定
        self.layer = nn.ModuleList([AltRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接收多个输入和控制参数
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
    # 返回值类型注释，指定了函数返回的对象类型
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不需要输出隐藏状态，则初始化为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重或者模型配置中不包含交叉注意力，则初始化为空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点并且在训练阶段，则检查是否允许使用缓存；若允许则发出警告并设置不使用缓存
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果使用缓存，则初始化下一个解码器缓存为空元组；否则设置为None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有头部掩码，则获取当前层的头部掩码；否则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果有过去的键值对，则获取当前层的过去键值对；否则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且在训练阶段，则调用梯度检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
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
                # 否则，直接调用当前层的前向传播函数
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
            # 如果使用缓存，则将当前层的输出的最后一个元素添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的输出的第二个元素添加到所有自注意力权重元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中包含交叉注意力，则将当前层的输出的第三个元素添加到所有交叉注意力权重元组中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回一个元组，包含所有需要输出的元素，且过滤掉空值
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
        # 否则，返回一个包含所有需要输出的对象的BaseModelOutputWithPastAndCrossAttentions对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.roberta.modeling_roberta.RobertaPooler
class AltRobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个激活函数，使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从隐藏状态中选择第一个 token 的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 将选定的第一个 token 的隐藏状态传递给全连接层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数处理全连接层的输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->AltCLIP
class AltCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # 定义用于查询（Q）、键（K）、值（V）投影的线性层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # 定义输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 对输入张量进行形状变换，以便进行多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # ...
    ):
        # 省略了 forward 方法中的其余代码，该方法实现了多头注意力的前向传播


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->AltCLIP
class AltCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 第一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个全连接层，输入维度为 config.intermediate_size，输出维度为 config.hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 第一个全连接层的前向传播
        hidden_states = self.fc1(hidden_states)
        # 激活函数的应用
        hidden_states = self.activation_fn(hidden_states)
        # 第二个全连接层的前向传播
        hidden_states = self.fc2(hidden_states)
        # 返回最终的隐藏状态张量
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->AltCLIP
class AltCLIPEncoderLayer(nn.Module):
    # 这个类定义了 CLIP 模型的编码器层，实现了自注意力机制和前馈神经网络
    # 具体的实现方法将包含在后续的方法中
    pass
    # 初始化函数，接收一个 AltCLIPConfig 类型的配置对象作为参数
    def __init__(self, config: AltCLIPConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置自注意力模块的隐藏单元数作为嵌入维度
        self.embed_dim = config.hidden_size
        # 创建自注意力层对象
        self.self_attn = AltCLIPAttention(config)
        # 第一个 LayerNorm 层，对隐藏状态进行归一化
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建多层感知机 MLP 模块
        self.mlp = AltCLIPMLP(config)
        # 第二个 LayerNorm 层，对隐藏状态进行归一化
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # 前向传播函数，接收多个张量作为输入，并返回一个元组的张量
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量，形状为(batch, seq_len, embed_dim)
        attention_mask: torch.Tensor,  # 注意力掩码张量，形状为(batch, 1, tgt_len, src_len)
        causal_attention_mask: torch.Tensor,  # 因果注意力掩码张量，形状同上
        output_attentions: Optional[bool] = False,  # 是否输出所有层的注意力张量，默认为False
    ) -> Tuple[torch.FloatTensor]:  # 返回一个包含浮点数张量的元组
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        # 保存原始的隐藏状态张量作为残差连接的一部分
        residual = hidden_states

        # 对隐藏状态进行第一次 LayerNorm 归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力层进行注意力计算，并返回计算的注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 将残差连接加回到当前隐藏状态中
        hidden_states = residual + hidden_states

        # 保存更新后的隐藏状态作为残差连接的一部分
        residual = hidden_states
        # 对隐藏状态进行第二次 LayerNorm 归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 使用 MLP 模块进行非线性变换
        hidden_states = self.mlp(hidden_states)
        # 将残差连接加回到当前隐藏状态中
        hidden_states = residual + hidden_states

        # 准备要输出的结果，只包含更新后的隐藏状态张量
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重张量加入到输出结果中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出结果元组
        return outputs
# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->AltCLIP
class AltCLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`AltCLIPEncoderLayer`].

    Args:
        config: AltCLIPConfig
    """

    def __init__(self, config: AltCLIPConfig):
        super().__init__()
        self.config = config
        # 创建包含多个 AltCLIPEncoderLayer 层的模块列表
        self.layers = nn.ModuleList([AltCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处是 AltCLIPEncoder 的前向传播函数，输入参数包括嵌入输入、注意力掩码等
        pass  # 此处省略了具体的前向传播逻辑，实际应根据具体情况填写

# Copied from transformers.models.clip.modeling_clip.CLIPVisionEmbeddings with CLIP->AltCLIP
class AltCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 定义分类嵌入向量作为模块的可学习参数
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 使用卷积层创建图像块嵌入向量，参数包括输入通道数、输出通道数等
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像中的图块数量和位置数量，并定义位置嵌入层
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # 对输入的像素值进行图块嵌入，然后展平和转置操作
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展分类嵌入向量，与图块嵌入连接，并加上位置嵌入
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class AltCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定使用的配置类和模型前缀
    config_class = AltCLIPConfig
    base_model_prefix = "altclip"
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 从配置中获取初始化因子
        factor = self.config.initializer_factor
        
        # 如果模块是 AltCLIPVisionEmbeddings 类型
        if isinstance(module, AltCLIPVisionEmbeddings):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            
            # 初始化 class_embedding 使用均值为 0，标准差为 embed_dim 的倒数乘以 factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            
            # 初始化 patch_embedding 的权重，标准差为 initializer_range 乘以 factor
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            
            # 初始化 position_embedding 的权重，标准差为 initializer_range 乘以 factor
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        
        # 如果模块是 AltCLIPAttention 类型
        elif isinstance(module, AltCLIPAttention):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            
            # 初始化 q_proj 的权重，标准差为 in_proj_std
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            
            # 初始化 k_proj 的权重，标准差为 in_proj_std
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            
            # 初始化 v_proj 的权重，标准差为 in_proj_std
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            
            # 初始化 out_proj 的权重，标准差为 out_proj_std
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        
        # 如果模块是 AltCLIPMLP 类型
        elif isinstance(module, AltCLIPMLP):
            # 重新设置初始化因子
            factor = self.config.initializer_factor
            
            # 初始化 fc1 的权重，标准差为 fc_std
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            
            # 初始化 fc2 的权重，标准差为 in_proj_std
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        
        # 如果模块是 AltCLIPModel 类型
        elif isinstance(module, AltCLIPModel):
            # 初始化 text_projection 的权重，标准差为 text_embed_dim 的倒数乘以 factor
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 将 _is_hf_initialized 设置为 True，表示已经初始化
            module.text_projection._is_hf_initialized = True
            
            # 初始化 visual_projection 的权重，标准差为 vision_embed_dim 的倒数乘以 factor
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 将 _is_hf_initialized 设置为 True，表示已经初始化
            module.visual_projection._is_hf_initialized = True
        
        # 如果模块是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置数据设为零
            module.bias.data.zero_()
            
            # 将权重数据填充为 1.0
            module.weight.data.fill_(1.0)
        
        # 如果模块是 nn.Linear 类型
        elif isinstance(module, nn.Linear):
            # 初始化权重数据，均值为 0，标准差为初始化因子
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            
            # 如果存在偏置数据，将其设为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 初始化权重数据，均值为 0，标准差为初始化因子
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            
            # 如果有 padding_idx，将对应索引的权重数据设为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 从transformers.models.clip.modeling_clip.CLIPVisionTransformer复制而来的AltCLIPVisionTransformer类定义，
# 重命名了一些类和常量，以适应当前环境
class AltCLIPVisionTransformer(nn.Module):
    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 使用配置初始化视觉嵌入层对象
        self.embeddings = AltCLIPVisionEmbeddings(config)
        # 应用 LayerNorm 到嵌入层输出，使用配置中的 epsilon 值
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化 CLIPEncoder 对象，处理视觉输入
        self.encoder = AltCLIPEncoder(config)
        # 应用 LayerNorm 到编码器输出，使用配置中的 epsilon 值
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(ALTCLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=AltCLIPVisionConfig)
    # 定义前向传播函数，接受像素值和其他参数，并返回模型输出
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回:

        """
        # 如果未提供像素值，则引发错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值转换为嵌入表示
        hidden_states = self.embeddings(pixel_values)
        # 应用预层归一化到嵌入表示
        hidden_states = self.pre_layrnorm(hidden_states)

        # 调用编码器处理嵌入表示，传递其他参数和返回类型
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 提取池化输出，通常是最后隐藏状态的首列
        pooled_output = last_hidden_state[:, 0, :]
        # 应用后层归一化到池化输出
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不要求返回字典，则返回一个元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则，返回一个包含池化输出和编码器状态的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AltCLIPVisionModel(AltCLIPPreTrainedModel):
    # 设置配置类为 AltCLIPVisionConfig
    config_class = AltCLIPVisionConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__(config)
        # 初始化 AltCLIPVisionTransformer 对象作为视觉模型
        self.vision_model = AltCLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()
    # 返回当前模型的视觉嵌入层模块
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 将此方法的返回值的文档字符串添加到模型的前向传播方法中
    @add_start_docstrings_to_model_forward(ALTCLIP_VISION_INPUTS_DOCSTRING)
    # 替换返回的文档字符串的输出类型，并使用指定的配置类
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=AltCLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回模型的前向传播结果。

        返回:
            返回一个包含模型输出的元组或BaseModelOutputWithPooling对象。

        示例:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AltCLIPVisionModel

        >>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        # 如果return_dict为None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播方法，传递输入参数并返回结果
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class AltRobertaModel(AltCLIPPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = AltCLIPTextConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->AltRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Initialize the embeddings module for the model based on provided configuration
        self.embeddings = AltRobertaEmbeddings(config)
        
        # Initialize the encoder module for the model based on provided configuration
        self.encoder = AltRobertaEncoder(config)

        # Initialize the pooling layer if `add_pooling_layer` is set to `True`
        self.pooler = AltRobertaPooler(config) if add_pooling_layer else None

        # Initialize model weights and perform final setup
        self.post_init()

    # Retrieve the input word embeddings from the model
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # Set the input word embeddings for the model
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # Prune heads of the model based on the provided `heads_to_prune` dictionary
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    # Forward pass for the model with detailed argument descriptions
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
    # 初始化函数，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 使用 AltRobertaModel 创建 self.roberta 对象，不添加池化层
        self.roberta = AltRobertaModel(config, add_pooling_layer=False)
        # 创建一个线性变换层，从 config.hidden_size 到 config.project_dim
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化，使用 config.layer_norm_eps 作为 epsilon
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 执行初始化后处理
        self.post_init()

    # 获取输入嵌入的方法，返回 self.roberta.embeddings.word_embeddings 模块
    def get_input_embeddings(self) -> nn.Module:
        return self.roberta.embeddings.word_embeddings

    # 设置输入嵌入的方法，将 value 赋值给 self.roberta.embeddings.word_embeddings
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.roberta.embeddings.word_embeddings = value

    # 调整 token 嵌入的大小的方法，调用父类的 resize_token_embeddings 方法
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        return super().resize_token_embeddings(new_num_tokens)

    # 前向传播方法，接受多个输入参数并按照特定的顺序返回结果
    @add_start_docstrings_to_model_forward(ALTCLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndProjection, config_class=AltCLIPTextConfig)
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
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # 设置返回字典，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用给定的输入参数调用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
            input_ids=input_ids,                   # 输入的词语 ID
            attention_mask=attention_mask,         # 注意力遮罩，指示哪些元素应该被忽略
            token_type_ids=token_type_ids,         # 用于区分句子 A 和句子 B 的标识
            position_ids=position_ids,             # 位置 ID，指示每个词语的位置
            head_mask=head_mask,                   # 头部遮罩，用于屏蔽特定的注意力头部
            inputs_embeds=inputs_embeds,           # 输入的嵌入表示
            encoder_hidden_states=encoder_hidden_states,     # 编码器的隐藏状态
            encoder_attention_mask=encoder_attention_mask,   # 编码器的注意力遮罩
            output_attentions=output_attentions,   # 是否输出注意力权重
            output_hidden_states=output_hidden_states,       # 是否输出所有隐藏状态
            return_dict=return_dict,               # 是否返回字典格式的输出
        )

        # 获取模型的序列输出（通常是最后一层的隐藏状态）
        sequence_output = outputs[0]

        # 应用预层标准化（LayerNorm）处理序列输出
        sequence_output = self.pre_LN(sequence_output)

        # 应用变换层对处理后的序列输出进行投影
        projection_state = self.transformation(sequence_output)

        # 提取池化输出，通常是投影后的第一个位置
        pooler_output = projection_state[:, 0]

        # 如果不需要返回字典，则返回投影状态、池化输出以及其他输出状态和注意力权重
        if not return_dict:
            return (projection_state, pooler_output) + outputs[2:4]

        # 返回包含池化输出和投影状态的字典格式的输出
        return BaseModelOutputWithPoolingAndProjection(
            last_hidden_state=projection_state,    # 最后的隐藏状态，通常是投影状态
            pooler_output=pooler_output,           # 池化输出，通常是投影后的第一个位置
            hidden_states=outputs.hidden_states,   # 所有隐藏状态的列表
            attentions=outputs.attentions,         # 所有注意力权重的列表
        )
    class AltCLIPModel(AltCLIPPreTrainedModel):
        # 指定配置类为AltCLIPConfig
        config_class = AltCLIPConfig

        def __init__(self, config: AltCLIPConfig):
            # 调用父类构造函数，初始化模型
            super().__init__(config)

            # 检查config.vision_config是否为AltCLIPVisionConfig类型
            if not isinstance(config.vision_config, AltCLIPVisionConfig):
                # 抛出数值错误异常，提示config.vision_config类型错误
                raise ValueError(
                    "config.vision_config is expected to be of type AltCLIPVisionConfig but is of type"
                    f" {type(config.vision_config)}."
                )
            # 检查config.text_config是否为AltCLIPTextConfig类型
            if not isinstance(config.text_config, AltCLIPTextConfig):
                # 抛出数值错误异常，提示config.text_config类型错误
                raise ValueError(
                    "config.text_config is expected to be of type AltCLIPTextConfig but is of type"
                    f" {type(config.text_config)}."
                )

            # 获取text_config和vision_config对象
            text_config = config.text_config
            vision_config = config.vision_config

            # 设置投影维度、文本嵌入维度和视觉嵌入维度
            self.projection_dim = config.projection_dim
            self.text_embed_dim = text_config.project_dim
            self.vision_embed_dim = vision_config.hidden_size

            # 初始化文本模型和视觉模型
            self.text_model = AltCLIPTextModel(text_config)
            self.vision_model = AltCLIPVisionTransformer(vision_config)

            # 创建视觉投影层和文本投影层，无偏置
            self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
            self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

            # 创建Logit缩放参数
            self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

            # 执行后初始化过程
            self.post_init()

        @add_start_docstrings_to_model_forward(ALTCLIP_TEXT_INPUTS_DOCSTRING)
        def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            token_type_ids=None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`AltCLIPTextModel`].

        Examples:

        ```
        >>> from transformers import AutoProcessor, AltCLIPModel

        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use AltCLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # Use `self.config.output_hidden_states` if `output_hidden_states` is not provided.
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Use `self.config.use_return_dict` if `return_dict` is not provided.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input arguments to the `text_model` of the AltCLIP model and retrieve text outputs.
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Extract the pooled output from `text_outputs`.
        pooled_output = text_outputs[1]
        # Project the pooled output to obtain text features.
        text_features = self.text_projection(pooled_output)

        # Return the computed text features.
        return text_features
        ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`AltCLIPVisionModel`].

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AltCLIPModel

        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # 使用 AltCLIP 模型的配置来设置一些字段（如果指定了），而不是使用视觉和文本组件的配置。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型，传入像素值、注意力输出、隐藏状态输出和返回字典等参数
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉模型的输出中获取池化后的特征向量
        pooled_output = vision_outputs[1]  # pooled_output
        # 将池化后的特征向量投影到特征空间
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征向量
        return image_features

    @add_start_docstrings_to_model_forward(ALTCLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AltCLIPOutput, config_class=AltCLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 从输入的输入标识创建位置标识的函数，源自transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    用输入标识替换非填充符号的位置编号。位置编号从padding_idx + 1开始，填充符号将被忽略。这是根据fairseq的`utils.make_positions`修改的。

    Args:
        input_ids: 输入的标识序列，torch.Tensor类型
        padding_idx: 填充标识的索引，用来确定哪些是填充符号
        past_key_values_length: 过去键值对的长度，用于计算增量索引，默认为0

    Returns:
        incremental_indices: torch.Tensor，包含输入标识的位置标识
    """

    # 创建一个掩码，标记不是填充符号的位置为1，其余为0
    mask = input_ids.ne(padding_idx).int()
    
    # 计算累积和，再加上过去键值对的长度，乘以掩码确保只在非填充符号位置生效
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    
    # 最后将计算出的位置索引转换为长整型，并加上填充索引，以得到最终的位置标识
    return incremental_indices.long() + padding_idx
```