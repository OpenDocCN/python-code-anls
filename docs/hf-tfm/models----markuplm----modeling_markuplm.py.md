# `.\transformers\models\markuplm\modeling_markuplm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可证信息
# 导入必要的库和模块
import math  # 导入数学模块
import os  # 导入操作系统模块
from typing import Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入 PyTorch 的损失函数

# 导入相关的自定义模块和函数
from ...activations import ACT2FN  # 导入激活函数相关的模块
from ...file_utils import (  # 导入文件处理相关的函数和类
    add_start_docstrings,  # 导入添加文档字符串的函数
    add_start_docstrings_to_model_forward,  # 导入给模型前向方法添加文档字符串的函数
    replace_return_docstrings,  # 导入替换返回文档字符串的函数
)
from ...modeling_outputs import (  # 导入模型输出相关的类
    BaseModelOutputWithPastAndCrossAttentions,  # 导入带过去和交叉注意力的基础模型输出类
    BaseModelOutputWithPoolingAndCrossAttentions,  # 导入带池化和交叉注意力的基础模型输出类
    MaskedLMOutput,  # 导入遮蔽语言模型输出类
    QuestionAnsweringModelOutput,  # 导入问答模型输出类
    SequenceClassifierOutput,  # 导入序列分类器输出类
    TokenClassifierOutput,  # 导入标记分类器输出类
)
from ...modeling_utils import (  # 导入模型工具函数和类
    PreTrainedModel,  # 导入预训练模型类
    apply_chunking_to_forward,  # 导入对前向方法应用分块的函数
    find_pruneable_heads_and_indices,  # 导入找到可剪枝头和索引的函数
    prune_linear_layer,  # 导入剪枝线性层的函数
)
from ...utils import logging  # 导入日志记录工具

# 导入配置类
from .configuration_markuplm import MarkupLMConfig  # 导入 MarkupLM 的配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "microsoft/markuplm-base"  # 用于文档的检查点
_CONFIG_FOR_DOC = "MarkupLMConfig"  # 用于文档的配置类

# 预训练模型存档列表
MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/markuplm-base",  # 基础规模的 MarkupLM 模型
    "microsoft/markuplm-large",  # 大规模的 MarkupLM 模型
]


class XPathEmbeddings(nn.Module):
    """构建来自 XPath 标签和下标的嵌入。

    在这个版本中，我们省略了树标识，因为它的信息可以通过 XPath 来覆盖。
    """

    def __init__(self, config):
        super(XPathEmbeddings, self).__init__()  # 初始化父类
        self.max_depth = config.max_depth  # 最大深度参数

        # 将 XPath 单元序列转换为嵌入
        self.xpath_unitseq2_embeddings = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)

        # 丢弃部分数据以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 激活函数
        self.activation = nn.ReLU()

        # 将 XPath 单元序列转换为内部表示
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)

        # 将内部表示转换为最终嵌入
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        # XPath 标签嵌入
        self.xpath_tag_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )

        # XPath 下标嵌入
        self.xpath_subs_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )
    # 前向传播函数，接受 XPath 标签序列和 XPath 子节点序列作为输入参数
    def forward(self, xpath_tags_seq=None, xpath_subs_seq=None):
        # 初始化 XPath 标签和 XPath 子节点的嵌入列表
        xpath_tags_embeddings = []
        xpath_subs_embeddings = []

        # 遍历最大深度范围的迭代
        for i in range(self.max_depth):
            # 将当前深度的 XPath 标签序列传入对应的嵌入层，得到标签的嵌入向量并加入列表中
            xpath_tags_embeddings.append(self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i]))
            # 将当前深度的 XPath 子节点序列传入对应的嵌入层，得到子节点的嵌入向量并加入列表中
            xpath_subs_embeddings.append(self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i]))

        # 在嵌入向量列表维度上连接，得到整个 XPath 标签的嵌入向量
        xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
        # 在嵌入向量列表维度上连接，得到整个 XPath 子节点的嵌入向量
        xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)

        # XPath 嵌入向量为 XPath 标签嵌入向量和 XPath 子节点嵌入向量的和
        xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings

        # 将 XPath 嵌入向量经过全连接层、激活函数、dropout，再经过全连接层映射为最终的嵌入向量
        xpath_embeddings = self.inner2emb(self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))

        # 返回最终的 XPath 嵌入向量
        return xpath_embeddings
```  
# 根据输入的 ID 序列创建对应的位置 ID 序列
# 其中 padding_idx 是填充符的 ID，past_key_values_length 是过去的位置偏移量
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个掩码张量，标记哪些位置不是填充符
    mask = input_ids.ne(padding_idx).int()
    # 根据掩码张量计算出每个位置的位置 ID
    # 首先计算累积和，得到当前位置的相对位置 ID
    # 然后乘以掩码张量，确保只有非填充位置有正确的位置 ID
    # 最后加上过去的位置偏移量，得到最终的位置 ID
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

# 构建用于文本生成任务的嵌入层
class MarkupLMEmbeddings(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super(MarkupLMEmbeddings, self).__init__()
        self.config = config
        # 词嵌入层，将词 ID 映射为隐藏状态向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，将位置 ID 映射为隐藏状态向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 最大深度
        self.max_depth = config.max_depth
        # XPath 嵌入层
        self.xpath_embeddings = XPathEmbeddings(config)
        # 类型嵌入层，将类型 ID 映射为隐藏状态向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # Layer Normalization 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置 ID 缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 填充标记 ID
        self.padding_idx = config.pad_token_id
        # 位置嵌入层，支持可变长度
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 从输入的嵌入向量中创建位置 ID 序列
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        # 获取输入的形状
        input_shape = inputs_embeds.size()[:-1]
        # 计算序列长度
        sequence_length = input_shape[1]
        # 创建位置 ID 序列
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 在批次维度上扩展位置 ID 序列
        return position_ids.unsqueeze(0).expand(input_shape)

    # 前向传播函数
    def forward(
        self,
        input_ids=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        # 代码省略
    ):
        # 如果提供了 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状，去掉最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取输入的设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供 position_ids，则根据 input_ids 创建位置编码
        if position_ids is None:
            if input_ids is not None:
                # 从输入的 token ids 创建位置 ids。任何填充的 token 保持填充状态。
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果未提供 token_type_ids，则创建全零的 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 如果未提供 inputs_embeds，则使用 word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 准备 xpath seq
        if xpath_tags_seq is None:
            # 使用 tag_pad_id 创建全填充的 xpath_tags_seq
            xpath_tags_seq = self.config.tag_pad_id * torch.ones(
                tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
            )
        if xpath_subs_seq is None:
            # 使用 subs_pad_id 创建全填充的 xpath_subs_seq
            xpath_subs_seq = self.config.subs_pad_id * torch.ones(
                tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
            )

        # 获取词嵌入、位置嵌入、token 类型嵌入和 xpath 嵌入
        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)

        # 合并所有嵌入
        embeddings = words_embeddings + position_embeddings + token_type_embeddings + xpath_embeddings

        # 使用 LayerNorm 对嵌入进行归一化
        embeddings = self.LayerNorm(embeddings)
        # 使用 dropout 进行正则化
        embeddings = self.dropout(embeddings)
        # 返回嵌入结果
        return embeddings
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制代码，并将Bert->MarkupLM
class MarkupLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertIntermediate复制代码，将Bert->MarkupLM
class MarkupLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput复制代码，并将Bert->MarkupLM
class MarkupLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertPooler复制代码，并将Bert->MarkupLM
class MarkupLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取对应于第一个令牌的隐藏状态来对模型进行"池化"
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制代码，并将Bert->MarkupLM
class MarkupLMPredictionHeadTransform(nn.Module):
```py  
    # 初始化函数，接受配置参数并进行初始化操作
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入和输出尺寸都是隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果隐藏激活函数是字符串，则通过映射表获取对应的函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用配置中的激活函数
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，输入尺寸为隐藏层大小，epsilon为配置中的值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    # 前向传播函数，接受一个表示隐藏状态的张量，返回经过处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理全连接层输出的隐藏状态张量
        hidden_states = self.transform_act_fn(hidden_states)
        # 使用 LayerNorm 处理隐藏状态张量
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制过来，将Bert改为MarkupLM
class MarkupLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MarkupLMPredictionHeadTransform(config)
        
        # 输出权重与输入嵌入相同，但对于每个标记有一个仅用于输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        
        # 需要两个变量之间的链接，以便偏置能正确地调整大小 `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制过来，将Bert改为MarkupLM
class MarkupLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MarkupLMLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从transformers.models.bert.modeling_bert.BertSelfAttention复制过来，将Bert改为MarkupLM
class MarkupLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        self.is_decoder = config.is_decoder

  
注释：提供了从Bert模型中复制的核心组件，用于创建MarkupLM模型中的预测头和自注意力机制。每个类都有自己的初始化和前向方法，实现具体的功能。同时，包括一些线性层、参数和参数初始化。
    # 将输入张量改变形状，使其适应多头注意力的计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 构造新的形状，即去掉最后一个维度，然后再在最后添加两个维度，分别是注意力头数和注意力头的维度大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 改变张量的形状
        x = x.view(new_x_shape)
        # 对张量进行转置操作，将注意力头维度和序列长度维度交换
        return x.permute(0, 2, 1, 3)

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
# 从 transformers.models.bert.modeling_bert.BertAttention 复制代码，并将Bert->MarkupLM
class MarkupLMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层
        self.self = MarkupLMSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化输出层
        self.output = MarkupLMSelfOutput(config)
        # 初始化一个集合，用于存储被剪枝的注意力头
        self.pruned_heads = set()

    # 剪枝注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可以被剪枝的头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部
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
        # 使用自注意力层进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 获取注意力输出并通过输出层处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果有需要，输出注意力
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从 transformers.models.bert.modeling_bert.BertLayer 复制代码，并将Bert->MarkupLM
class MarkupLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前馈层的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化自注意力层
        self.attention = MarkupLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 如果添加了跨层注意力，则初始化跨层注意力层
            self.crossattention = MarkupLMAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = MarkupLMIntermediate(config)
        # 初始化输出层
        self.output = MarkupLMOutput(config)
    # 前向传播函数，接收隐藏状态、注意力掩码等参数，返回输出元组
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
        # 若前一次的 self-attention 缓存键/值存在，则取出来，位置在 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力机制计算
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
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 报错，必须设置 config.add_cross_attention=True 来实例化交叉注意力层
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 取交叉注意力缓存键/值，位置在过去键/值元组的倒数第二和倒数第一位
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行交叉注意力机制计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注��力

            # 将交叉注意力缓存添加到当前 present_key_value 的末尾
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块处理进行前向传播
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将最后的注意力键/值作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    # 执行神经网络的前向传播，处理一个块的注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层的输出和注意力输出，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层输出
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制得到，将Bert替换为MarkupLM
class MarkupLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建包含多个MarkupLMLayer的ModuleList，层数由config.num_hidden_layers指定
        self.layer = nn.ModuleList([MarkupLMLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点默认关闭
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
        # 初始化变量，根据需求选择是否保存所有隐藏状态、注意力权重和跨注意力权重
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 检查是否启用了梯度检查点和处于训练阶段，如果是则需要关闭缓存功能
        if self.gradient_checkpointing and self.training:
            if use_cache:
                # 如果同时启用了缓存和梯度检查点功能，给出警告并关闭缓存
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化下一个解码器缓存，在不使用缓存时赋值为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个层，并进行相应的操作
        for i, layer_module in enumerate(self.layer):
            # 如果需要保存所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取过去的键值对，用于注意力机制的上下文记忆
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点和处于训练阶段，使用特殊的梯度检查点函数
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
                # 否则使用当前层的前向传播函数
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要使用缓存，则将当前层的输出作为下一个解码器缓存
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要获取注意力权重，将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型具有跨注意力功能，则将当前层的跨注意力权重添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要保存所有隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 参数确定返回值的形式
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
        # 返回封装了不同部分输出的模型输出对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class MarkupLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为MarkupLMConfig
    config_class = MarkupLMConfig
    # 设置预训练模型的资源映射
    pretrained_model_archive_map = MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST
    # 设置基础模型前缀为"markuplm"
    base_model_prefix = "markuplm"

    # 从指定模块中初始化权重
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 略微有所不同于 TF 版本，使用正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将其初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为0，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # 根据预训练模型名称或路径创建实例
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        return super(MarkupLMPreTrainedModel, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


# 模型文档字符串
MARKUPLM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MarkupLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入文档字符串
MARKUPLM_INPUTS_DOCSTRING = r"""
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                输入序列标记在词汇表中的索引。

                可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和
                [`PreTrainedTokenizer.__call__`]。

                [什么是输入 ID？](../glossary#input-ids)

            xpath_tags_seq (`torch.LongTensor` of shape `({0}, config.max_depth)`, *optional*):
                输入序列中每个标记的标签 ID，填充至 config.max_depth。

            xpath_subs_seq (`torch.LongTensor` of shape `({0}, config.max_depth)`, *optional*):
                输入序列中每个标记的下标 ID，填充至 config.max_depth。

            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                避免对填充标记索引执行注意力的掩码。选定范围内的掩码值为 `[0, 1]`：`1` 表示**未被掩码**的标记，`0` 表示**被掩码**的标记。

                [什么是注意力掩码？](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                段标记索引，指示输入的第一部分和第二部分。索引选择范围为 `[0, 1]`：`0` 对应 *句子 A* 标记，`1` 对应 *句子 B* 标记。

                [什么是标记类型 ID？](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                每个输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。

                [什么是位置 ID？](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                用于将自注意力模块的选定头部屏蔽掉的掩码。选定范围内的掩码值为 `[0, 1]`：`1` 表示头部**未被屏蔽**，`0` 表示头部**被屏蔽**。
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                可选地，您可以选择直接传递嵌入表示而不是传递 `input_ids`。如果您想对如何将 *input_ids* 索引转换为相关向量有更多控制权，这将很有用，而不是使用模型的内部嵌入查找矩阵。
            output_attentions (`bool`, *optional*):
                如果设置为 `True`，则返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
            output_hidden_states (`bool`, *optional*):
                如果设置为 `True`，则返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
            return_dict (`bool`, *optional*):
                如果设置为 `True`，模型将返回一个 [`~file_utils.ModelOutput`] 而不是一个普通的元组。
# 定义一个类 MarkupLMModel，继承自 MarkupLMPreTrainedModel 类
@add_start_docstrings("The bare MarkupLM Model transformer outputting raw hidden-states without any specific head on top.", MARKUPLM_START_DOCSTRING)
class MarkupLMModel(MarkupLMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制并修改为 Bert->MarkupLM
    # 初始化方法
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置参数
        self.config = config
        # 创建 MarkupLMEmbeddings 对象
        self.embeddings = MarkupLMEmbeddings(config)
        # 创建 MarkupLMEncoder 对象
        self.encoder = MarkupLMEncoder(config)
        # 如果 add_pooling_layer 为 True 则创建 MarkupLMPooler 对象，否则为 None
        self.pooler = MarkupLMPooler(config) if add_pooling_layer else None
        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入 embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    # 设置输入 embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    # 剪枝模型的 heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    # 前向传播方法
    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        # 定义前向传播的输入参数
        self,
        input_ids: Optional[torch.LongTensor] = None,
        xpath_tags_seq: Optional[torch.LongTensor] = None,
        xpath_subs_seq: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 从 transformers.models.bert.modeling_bert.BertModel.prepare_inputs_for_generation 复制
    # 准备生成的输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        # 获取输入数据的形状
        input_shape = input_ids.shape
        # 如果模型作为编码器在编码-解码模型中使用，则动态创建解码器注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果过去的key-value对被使用，则切割解码器的输入ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保持旧的行为：仅保留最终ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含输入ID、注意力掩码、过去的key-value对和缓存使用情况的字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 从transformers.models.bert.modeling_bert.BertModel._reorder_cache中复制而来
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对过去的key-value对进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
@add_start_docstrings(
    """
    MarkupLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MARKUPLM_START_DOCSTRING,
)
class MarkupLMForQuestionAnswering(MarkupLMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 复制而来，将 bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 初始化标签数量
        self.num_labels = config.num_labels

        # 使用 MarkupLMModel 构建 MarkupLM 对象，不添加池化层
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # 使用线性层构建 QA 输出层，输入维度为隐藏层维度，输出维度为标签数量
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 省略了此处的参数注释
):

@add_start_docstrings("""MarkupLM Model with a `token_classification` head on top.""", MARKUPLM_START_DOCSTRING)
class MarkupLMForTokenClassification(MarkupLMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制而来，将 bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 初始化标签数量
        self.num_labels = config.num_labels

        # 使用 MarkupLMModel 构建 MarkupLM 对象，不添加池化层
        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        # 如果分类器的 dropout 不为 None，则使用其值，否则使用隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 使用线性层构建分类器，输入维度为隐藏层维度，输出维度为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播函数
    def forward(
        # 输入序列 ID 张量
        self,
        input_ids: Optional[torch.Tensor] = None,
        # XPath 标签序列张量  
        xpath_tags_seq: Optional[torch.Tensor] = None,
        # XPath 子节点序列张量
        xpath_subs_seq: Optional[torch.Tensor] = None,
        # 注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 标记类型 ID 张量
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID 张量
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码张量
        head_mask: Optional[torch.Tensor] = None,
        # 嵌入输入张量
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签张量
        labels: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
    ):
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
        >>> processor.parse_html = False
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/markuplm-base", num_labels=7)

        >>> nodes = ["hello", "world"]
        >>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"]
        >>> node_labels = [1, 2]
        >>> encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**encoding)

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.classifier(sequence_output)  # (batch_size, seq_length, node_type_size)

        loss = None
        # 计算损失函数，如果传入了标签
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.num_labels),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有序列分类/回归头部的MarkupLM模型转换器（在汇总输出之上的线性层），例如用于GLUE任务
class MarkupLMForSequenceClassification(MarkupLMPreTrainedModel):
    # 从transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__中复制而来，将bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 保存配置
        self.config = config

        # 创建一个MarkupLM模型
        self.markuplm = MarkupLMModel(config)
        # 设置分类器的dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```