# `.\models\git\modeling_git.py`

```py
# 设置文件编码为utf-8
# 版权声明信息
# 版权归 Microsoft Research 和 The HuggingFace Inc. 团队所有
# 在 Apache 许可证 2.0 版本下许可
# 根据许可证规定，仅在符合许可证条件下才能使用该文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经双方书面同意，否则按原样（AS IS）发布软件
# 没有任何明示或暗示的担保或条件
# 请参考许可证以了解特定语言下的许可证限制
"""PyTorch GIT model.""":专注于PyTorch的GIT模型

# 导入所需的库
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
# 导入相关文件
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_git import GitConfig, GitVisionConfig

# 获取logger
logger = logging.get_logger(__name__)

# 针对文档的检查点信息和配置信息
_CHECKPOINT_FOR_DOC = "microsoft/git-base"
_CONFIG_FOR_DOC = "GitConfig"

# 预训练的GIT模型存档列表
GIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/git-base",
    # 请参考 https://huggingface.co/models?filter=git 查看所有GIT模型
]


@dataclass
# 从transformers.models.clip.modeling_clip中复制的类，将CLIP更改为Git
class GitVisionModelOutput(ModelOutput):
    """
    包含视觉模型输出的基类，还包含最后隐藏状态的池化的图像嵌入
    # 设置函数参数及其类型注释
    
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            通过将投影层应用于池化输出获得的图像嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含每一层输出的隐藏状态的元组，如果模型具有嵌入层，则还包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含每一层注意力权重的元组，用于计算自注意力头中的加权平均值。
    """
    
    # 设置变量类型注释
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义 GitEmbeddings 类，用于构建词嵌入和位置嵌入
class GitEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，vocab_size 表示词汇表大小，hidden_size 表示隐藏层大小，padding_idx 表示填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，max_position_embeddings 表示最大位置嵌入数量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm 不是蛇形命名以保持与 TensorFlow 模型变量名的一致性，并能够加载任何 TensorFlow 检查点文件
        # 初始化 LayerNorm 层，hidden_size 表示隐藏层大小，eps 表示 LayerNorm 中的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，hidden_dropout_prob 表示隐藏层的 Dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 在内存中是连续的，并在序列化时导出
        # position_embedding_type 表示位置嵌入的类型，默认为 "absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册 position_ids 为模型的持久化缓冲区，torch.arange(config.max_position_embeddings) 生成位置嵌入的索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果传入了 input_ids，则获取其形状；否则，获取 inputs_embeds 的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 如果未传入 position_ids，则根据 past_key_values_length 和序列长度生成位置嵌入的索引
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未传入 inputs_embeds，则使用 input_ids 获取词嵌入
        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        # 如果位置嵌入的类型为 "absolute"，则获取位置嵌入并添加到词嵌入中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 Dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的 embeddings
        return embeddings


class GitSelfAttention(nn.Module):
    # 初始化函数，用于初始化注意力头的参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化函数
        super().__init__()
        # 如果隐藏层大小不能被注意力头的数量整除，并且配置中没有嵌入大小这个属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 抛出数值错误
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 记录注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 记录每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算图像分块的数量
        self.image_patch_tokens = int((config.vision_config.image_size / config.vision_config.patch_size) ** 2 + 1)
        # 如果指定了具有嵌入的图像数量，则相应地调整图像分块的数量
        if config.num_image_with_embedding is not None:
            self.image_patch_tokens *= config.num_image_with_embedding

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入的类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入的类型是相对位置嵌入，需要初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 记录最大位置嵌入数量
            self.max_position_embeddings = config.max_position_embeddings
            # 初始化距离嵌入的 Embedding 层
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    # 将输入张量转换为注意力得分计算所需的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 改变张量形状
        x = x.view(new_x_shape)
        # 转置张量维度以便计算注意力得分
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，计算注意力输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
```   
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制而来的类，表示BERT模型中的自注意力机制输出层
class GitSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将隐藏状态的维度映射到相同的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层的前向传播
        hidden_states = self.dense(hidden_states)
        # Dropout 层的前向传播
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 层的前向传播，与输入张量相加后归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回隐藏状态
        return hidden_states


# 表示BERT模型中的自注意力机制
class GitAttention(nn.Module):
    # 从transformers.models.bert.modeling_bert.BertAttention.__init__复制而来的类初始化方法，将Bert替换为Git
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 自注意力机制
        self.self = GitSelfAttention(config, position_embedding_type=position_embedding_type)
        # 自注意力机制的输出层
        self.output = GitSelfOutput(config)
        # 存储要剪枝的头信息
        self.pruned_heads = set()

    # 从transformers.models.bert.modeling_bert.BertAttention.prune_heads复制而来的剪枝方法
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的头
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 自注意力机制的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
            pixel_values_present,
        )
        # 自注意力机制输出层的前向传播
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，将其加入输出元组
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制而来的类，表示BERT模型中的中间层
class GitIntermediate(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则使用预定义的激活函数，否则使用config.hidden_act指定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受一个torch.Tensor类型的hidden_states输入，返回一个torch.Tensor类型的输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入数据进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的数据应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回激活后的数据
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput中复制代码，定义GitOutput类
class GitOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建LayerNorm层，输入维度为config.hidden_size，并设置eps参数为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 对处理后的hidden_states进行dropout
        hidden_states = self.dropout(hidden_states)
        # 对处理后的hidden_states进行LayerNorm并加上input_tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义GitLayer类
class GitLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置chunk_size_feed_forward和seq_len_dim
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建GitAttention、GitIntermediate和GitOutput对象
        self.attention = GitAttention(config)
        self.intermediate = GitIntermediate(config)
        self.output = GitOutput(config)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取self-attention缓存的键/值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 对hidden_states进行self-attention操作
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            pixel_values_present=pixel_values_present,
        )
        # 获取self-attention的输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder，最后一次输出是self-attn缓存的元组
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        # 对attention_output进行分块处理并返回
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是decoder，将attention的键/值对作为最后一个输出返回
        outputs = outputs + (present_key_value,)

        return outputs

    # 定义一个用于分块处理的函数
    def feed_forward_chunk(self, attention_output):
        # 通过intermediate层处理attention_output
        intermediate_output = self.intermediate(attention_output)
        # 通过output层处理intermediate_output
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# 定义GitEncoder类
class GitEncoder(nn.Module):
    # 从transformers.models.bert.modeling_bert.BertEncoder.__init__中复制代码，并将Bert改为Git
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数保存到对象的属性中
        self.config = config
        # 创建一个包含指定数量 GitLayer 对象的列表
        self.layer = nn.ModuleList([GitLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部掩码张量
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 可选的过去键值元组
        use_cache: Optional[bool] = None,  # 可选的使用缓存标志
        output_attentions: Optional[bool] = False,  # 可选的输出注意力标志，默认为 False
        output_hidden_states: Optional[bool] = False,  # 可选的输出隐藏状态标志，默认为 False
        pixel_values_present: Optional[bool] = False,  # 可选的像素值是否存在标志，默认为 False
        return_dict: Optional[bool] = True,  # 可选的返回字典标志，默认为 True
    # 定义函数forward，其作用是前向传播
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        pixel_values_present: bool = False,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        # 若启用梯度检查点方法且处于训练状态
        if self.gradient_checkpointing and self.training:
            
            # 若 use_cache 为 True，则给出警告并将其设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            
        # 初始化用于存储所有隐藏状态的元组
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于存储所有自注意力矩阵的元组
        all_self_attentions = () if output_attentions else None
        
        # 初始化下一个解码器缓存的元组
        next_decoder_cache = () if use_cache else None
        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            
            # 若要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 获取当前层的头掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # 若启用梯度检查点方法且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点方法进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 调用当前层的 forward 方法进行前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    pixel_values_present,
                )
            
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 若 use_cache 为 True，则将当前层输出的缓存添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 若要输出自注意力矩阵，则将当前层的自注意力矩阵添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        # 若要输出隐藏状态，则将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 若 return_dict 为 False，则返回所有结果元素的元组
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        # 若 return_dict 为 True，则返回 BaseModelOutputWithPast 类型的对象
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class GitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为GitConfig
    config_class = GitConfig
    # 设置基本模型前缀为"git"
    base_model_prefix = "git"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是GitVisionEmbeddings
        if isinstance(module, GitVisionEmbeddings):
            # 对class_embedding进行标准正态分布初始化
            nn.init.normal_(module.class_embedding, mean=0.0, std=self.config.initializer_range)
            # 对patch_embedding.weight进行标准正态分布初始化
            nn.init.normal_(module.patch_embedding.weight, std=self.config.initializer_range)
            # 对position_embedding.weight进行标准正态分布初始化
            nn.init.normal_(module.position_embedding.weight, std=self.config.initializer_range)
        # 如果模块是nn.Linear
        if isinstance(module, nn.Linear):
            # 稍微与TF版本不同的初始化方法，“截断正态分布”，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是nn.Embedding
        elif isinstance(module, nn.Embedding):
            # 权重进行标准正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在padding_idx，将其对应的权重置为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是nn.LayerNorm
        elif isinstance(module, nn.LayerNorm):
            # 偏置项置为0
            module.bias.data.zero_()
            # 权重项置为1
            module.weight.data.fill_(1.0)

# 定义GitPreTrainedModel的文档字符串
GIT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义GitPreTrainedModel输入部分的文档字符串
GIT_INPUTS_DOCSTRING = r"""
    # 参数 input_ids，类型为 torch.LongTensor，形状为给定值的形状
    # input_ids 是输入序列标记在词汇表中的索引
    # 可以使用 AutoTokenizer 来获取这些索引。详见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 的详细说明。
    # 什么是 input IDs 可以参考 glossary 中的说明
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        # 参数 attention_mask，类型为 torch.FloatTensor，形状为给定值的形状，可选
        # 用于避免在填充标记的索引上执行注意力，值为 0 或 1：
        # - 1 表示**不被掩蔽**的标记
        # - 0 表示**被掩蔽**的标记
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        # 参数 position_ids，类型为 torch.LongTensor，形状为给定值的形状，可选
        # 用于获取每个输入序列标记在位置嵌入中的位置索引，取值范围是 `[0, config.max_position_embeddings - 1]`
        # 什么是 position IDs 可以参考 glossary 中的说明
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        # 参数 pixel_values，类型为 torch.FloatTensor，形状为(batch_size, num_channels, height, width)
        # 像素值。像素值可以使用 AutoImageProcessor 获取。详见 CLIPImageProcessor.__call__ 的详细说明
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.

        # 参数 head_mask，类型为 torch.FloatTensor，形状为(num_heads,) 或(num_layers, num_heads)，可选
        # 用于将自注意力模块中的选定 heads 置零。值为 0 或 1：
        # - 1 表示头部**未被掩蔽**
        # - 0 表示头部**被掩蔽**
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        # 参数 inputs_embeds，类型为 torch.FloatTensor，形状为给定值的形状，可选
        # 你可以选择直接传递已嵌入的表示而不是传递 input_ids。这对于比模型内部的嵌入查找矩阵更多地控制如何将 input_ids 索引转换为关联向量很有用。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

        # 参数 output_attentions，类型为 bool，可选
        # 是否返回所有注意力层的注意力张量。更多细节可在返回的张量下查看 attentions
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        # 参数 output_hidden_states，类型为 bool，可选
        # 是否返回所有层的隐藏状态。更多细节可在返回的张量下查看 hidden_states
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        # 参数 return_dict，类型为 bool，可选
        # 是否返回 `~utils.ModelOutput` 而不是普通的元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings复制而来，但把CLIP改为Git
class GitVisionEmbeddings(nn.Module):
    def __init__(self, config: GitVisionConfig):
        super().__init__()
        # 初始化GitVisionEmbeddings，设置配置和隐藏层大小
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 创建类别嵌入参数
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 创建图像补丁嵌入层
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算补丁数量和位置数量，并创建位置嵌入层
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播函数，接收像素值张量并返回嵌入张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # 使用图像补丁嵌入层将像素值转换为补丁嵌入张量
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 创建类别嵌入张量并与补丁嵌入张量拼接，然后加上位置嵌入张量
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPMLP复制而来
class GitVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化GitVisionMLP，设置激活函数和线性层
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # 前向传播函数，接收隐藏状态张量并返回变换后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 进行线性变换和激活函数操作
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# 从transformers.models.clip.modeling_clip.CLIPAttention复制而来
class GitVisionAttention(nn.Module):
    """从《Attention Is All You Need》论文中引入的多头注意力机制"""
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置参数中获取隐藏层大小作为嵌入维度
        self.config = config
        self.embed_dim = config.hidden_size
        # 从配置参数中获取注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 如果嵌入维度不能完全被注意力头数量整除，抛出数值错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 计算缩放因子
        self.scale = self.head_dim**-0.5
        # 从配置参数获取注意力层的dropout概率
        self.dropout = config.attention_dropout

        # 初始化线性变换层，用于将输入的隐藏状态映射到查询、键、值、输出空间
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 根据输入的张量形状和批次大小，重塑张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，接受隐藏状态、注意力掩码和因果注意力掩码等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
# 定义一个自定义的 GitVisionEncoderLayer 类，继承自 nn.Module
class GitVisionEncoderLayer(nn.Module):
    # 初始化方法，接受一个 GitVisionConfig 类型的参数 config
    def __init__(self, config: GitVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 初始化自注意力层对象
        self.self_attn = GitVisionAttention(config)
        # 初始化第一个层规范化对象
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 初始化多层感知机对象
        self.mlp = GitVisionMLP(config)
        # 初始化第二个层规范化对象
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入张量，形状为(batch, seq_len, embed_dim)
        attention_mask: torch.Tensor,  # 注意力掩码张量，形状为(batch, 1, tgt_len, src_len)，用于指示填充元素的位置
        causal_attention_mask: torch.Tensor,  # 因果注意力掩码张量，形状为(config.encoder_attention_heads,)
        output_attentions: Optional[bool] = False,  # 是否返回所有注意力层的注意力张量
    ) -> Tuple[torch.FloatTensor]:  # 返回类型为元组，包含一个浮点数张量
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 将输入与残差连接
        residual = hidden_states

        # 进行第一个层规范化
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力层计算注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 将残差与自注意力层的输出相加
        hidden_states = residual + hidden_states

        # 重新赋值残差
        residual = hidden_states
        # 进行第二个层规范化
        hidden_states = self.layer_norm2(hidden_states)
        # 使用多层感知机进行非线性变换
        hidden_states = self.mlp(hidden_states)
        # 将残差与多层感知机的输出相加
        hidden_states = residual + hidden_states

        # 输出结果为隐藏状态张量
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出结果
        return outputs


# 定义一个自定义的 GitVisionEncoder 类，继承自 nn.Module
class GitVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`GitVisionEncoderLayer`].

    Args:
        config: GitVisionConfig
    """

    # 初始化方法，接受一个 GitVisionConfig 类型的参数 config
    def __init__(self, config: GitVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置配置属性
        self.config = config
        # 使用列表推导式创建多个 GitVisionEncoderLayer 对象，数量为 config.num_hidden_layers
        self.layers = nn.ModuleList([GitVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点设为 False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        inputs_embeds,  # 输入嵌入
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        causal_attention_mask: Optional[torch.Tensor] = None,  # 因果注意力掩码张量，可选
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态张量，可选
        return_dict: Optional[bool] = None,  # 是否返回字典类型的结果，可选
GIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 定义一个名为 GitVisionTransformer 的类，该类继承自 nn.Module
class GitVisionTransformer(nn.Module):
    # 以 GitVisionConfig 为参数初始化 GitVisionTransformer 类
    # 该方法是从 transformers.models.clip.modeling_clip.CLIPVisionTransformer.__init__ 复制而来，将 CLIPEncoder->GitVisionEncoder, CLIP->Git
    def __init__(self, config: GitVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 创建 GitVisionEmbeddings 实例
        self.embeddings = GitVisionEmbeddings(config)
        # 创建包含指定 embed_dim 的 LayerNorm 实例
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 创建 GitVisionEncoder 实例
        self.encoder = GitVisionEncoder(config)
        # 创建包含指定 embed_dim 的 LayerNorm 实例
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 对 model.forward 方法添加输入文档字符串（add_start_docstrings_to_model_forward 返回值：输出类型 BaseModelOutput，配置类 GitVisionConfig）
    @add_start_docstrings_to_model_forward(GIT_VISION_INPUTS_DOCSTRING)
    # 替换返回文档字符串（返回值：输出类型 BaseModelOutput，配置类 GitVisionConfig）
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=GitVisionConfig)
    # 定义 model.forward 方法
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果 output_attentions 未指定，则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 未指定，则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 未指定，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用 pixel_values 调用 self.embeddings 方法，得到 hidden_states
        hidden_states = self.embeddings(pixel_values)
        # 对 hidden_states 应用 self.pre_layrnorm 方法
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用 self.encoder 方法处理 hidden_states
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 encoder_outputs 的第一个元素，赋值给 last_hidden_state
        last_hidden_state = encoder_outputs[0]

        # 对 last_hidden_state 应用 self.post_layernorm 方法
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 如果 return_dict 为 False，则返回元组
        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        # 否则，返回 BaseModelOutput 实例
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 对 GitVisionModel 类添加输入文档字符串和模型起始文档字符串
@add_start_docstrings(
    """The vision model from CLIP, used in GIT, without any head or projection on top.""",
    GIT_START_DOCSTRING,
)
# 定义 GitVisionModel 类，该类继承自 GitPreTrainedModel
class GitVisionModel(GitPreTrainedModel):
    # 指定配置类为 GitVisionConfig
    config_class = GitVisionConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 以 GitVisionConfig 为参数初始化 GitVisionModel 类
    # 该方法是从 transformers.models.clip.modeling_clip.CLIPVisionModel.__init__ 复制而来，将 CLIPEncoder->GitVisionEncoder, CLIP->Git
    def __init__(self, config: GitVisionConfig):
        super().__init__(config)
        # 创建 GitVisionTransformer 实例
        self.vision_model = GitVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding
    # 为模型前向传播添加关于输入的文档字符串
    @add_start_docstrings_to_model_forward(GIT_VISION_INPUTS_DOCSTRING)
    # 为模型前向传播替换输出的文档字符串
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=GitVisionConfig)
    # 前向传播函数定义，接收像素值、是否输出注意力、隐藏状态等参数，返回元组或者模型输出对象
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        返回值：

        示例：

        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, GitVisionModel

        >>> processor = AutoProcessor.from_pretrained("microsoft/git-base")
        >>> model = GitVisionModel.from_pretrained("microsoft/git-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        ```"""
        # 如果 return_dict 未设置，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播函数，传入像素值、是否输出注意力、隐藏状态和是否返回字典等参数
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class GitProjection(nn.Module):
    def __init__(self, config: GitConfig):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Sequential(
            # 创建一个线性层，将视觉配置的隐藏尺寸转换为配置的隐藏尺寸
            nn.Linear(config.vision_config.hidden_size, config.hidden_size),
            # 层归一化
            nn.LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(embeddings)


@add_start_docstrings(
    "The bare GIT Model transformer consisting of a CLIP image encoder and text decoder outputting raw hidden-states"
    " without any specific head on top.",
    GIT_START_DOCSTRING,
)
class GitModel(GitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = GitEmbeddings(config)
        self.image_encoder = GitVisionModel(config.vision_config)
        self.encoder = GitEncoder(config)

        self.visual_projection = GitProjection(config)

        if config.num_image_with_embedding is not None:
            # 如果配置了图片嵌入数量，则初始化图片温度嵌入
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size))
                for _ in range(config.num_image_with_embedding)
            )

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
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # 生成未来的遮罩，用于掩盖未来的信息，防止信息泄漏
        # 默认的遮罩是用于向前的方向。向后的方向则翻转
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
    def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
        # 计算目标序列和记忆序列的长度
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        # 获取目标序列所在设备和数据类型
        device = tgt.device
        dtype = tgt.dtype
        # 初始化左上角矩阵为全零
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        # 初始化右上角矩阵为负无穷大
        top_right = torch.full(
            (num_memory, num_tgt + past_key_values_length),
            float("-inf"),
            device=tgt.device,
            dtype=dtype,
        )
        # 初始化左下角矩阵为全零
        bottom_left = torch.zeros(
            (num_tgt, num_memory),
            dtype=dtype,
            device=tgt_mask.device,
        )

        # 如果传入的过去键值的长度大于0，则将目标序列的mask设为全零
        if past_key_values_length > 0:
            tgt_mask = torch.zeros(
                (tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length),
                dtype=dtype,
                device=tgt_mask.device,
            )

        # 将左上角矩阵和左下角矩阵拼接成左侧矩阵
        left = torch.cat((top_left, bottom_left), dim=0)
        # 将右上角矩阵和目标序列mask拼接成右侧矩阵
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        # 将左侧矩阵和右侧矩阵拼接成完整的注意力mask矩阵
        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        # 如果未提供记忆键值的填充mask，则将其初始化为全零
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # 如果填充mask不是布尔类型，则引发数值错误
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError("Memory key padding mask must be a boolean tensor.")
        # 将填充位置的mask设为负无穷大
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float("-inf")
        # 将完整的注意力mask矩阵扩展成与填充mask相同的形状
        full_attention_mask = full_attention_mask.expand(
            (memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt)
        )
        # 对扩展后的注意力mask矩阵进行克隆
        full_attention_mask = full_attention_mask.clone()
        # 更新左侧矩阵与填充mask矩阵的值
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        # 为多头注意力添加一个维度
        full_attention_mask = full_attention_mask[:, None, :, :]

        return full_attention_mask

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义一个基于 GPT 模型的语言建模头的 GIT 模型，继承自 GitPreTrainedModel 类
@add_start_docstrings(
    """GIT Model with a `language modeling` head on top for autoregressive language modeling.""", GIT_START_DOCSTRING
)
class GitForCausalLM(GitPreTrainedModel):
    # 定义绑定权重的键
    _tied_weights_keys = ["output.weight"]

    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 创建一个 GitModel 对象
        self.git = GitModel(config)
        # 创建一个线性层，用于预测词汇
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.output

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 当使用 past_key_values 时，修剪 decoder_input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 如果模型用作编码器-解码器模型中的解码器，动态创建解码器注意力遮罩
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": kwargs.get("pixel_values", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 重新排列缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```