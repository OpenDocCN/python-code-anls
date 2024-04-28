# `.\transformers\models\lilt\modeling_lilt.py`

```
# 设定编码格式为 utf-8
# 版权声明

# 导入所需的模块
import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_lilt import LiltConfig
# 初始化日志
logger = logging.get_logger(__name__)
# 配置模型文档
_CONFIG_FOR_DOC = "LiltConfig"
# 预训练模型存档列表
LILT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "SCUT-DLVCLab/lilt-roberta-en-base",
    # 查看所有LiLT模型 https://huggingface.co/models?filter=lilt
]

# 定义LiltTextEmbeddings类
class LiltTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 词嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 标记类型嵌入
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # LayerNorm，保持与TensorFlow模型变量名称一致，能够加载任何TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # DropOut
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 注册position_ids为持久化缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 位置嵌入类型
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 复制
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ): 
        # 关键字为 if 的函数结束
        if position_ids is None:
            # 如果位置 IDs 为空
            if input_ids is not None:
                # 如果输入 IDs 不为空
                # 从输入的 token IDs 创建位置 IDs。任何填充的 token 仍然保持填充。
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            # 如果输入 IDs 不为空
            input_shape = input_ids.size()
        else:
            # 如果输入 Embeds 不为空
            input_shape = inputs_embeds.size()[:-1]

        # 如果 token 类型 IDs 为空
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果输入 Embeds 为空
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # 返回嵌入和位置 IDs
        return embeddings, position_ids

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        Args:
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
            x: torch.Tensor x:
        Returns: torch.Tensor
        """
        # 通过输入 IDs 替换非填充符号为它们的位置编号。位置编号从 padding_idx+1 开始。填充符号会被忽略。
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        Args:
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成连续的位置 IDs
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
class LiltLayoutEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 在这里我们将隐藏大小除以6，因为有6种不同的布局嵌入，即左位置、上位置、右位置、下位置、高度、宽度
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)  # 创建x位置嵌入层
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)  # 创建y位置嵌入层
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)  # 创建高度位置嵌入层
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)  # 创建宽度位置嵌入层

        self.padding_idx = config.pad_token_id  # 获取配置中的填充标记ID
        self.box_position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size // config.channel_shrink_ratio,
            padding_idx=self.padding_idx,
        )  # 创建框位置嵌入层
        self.box_linear_embeddings = nn.Linear(
            in_features=config.hidden_size, out_features=config.hidden_size // config.channel_shrink_ratio
        )  # 线性变换，将隐藏大小转化为通道收缩比隐藏大小
        self.LayerNorm = nn.LayerNorm(config.hidden_size // config.channel_shrink_ratio, eps=config.layer_norm_eps)  # 创建LayerNorm层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建Dropout层

    def forward(self, bbox=None, position_ids=None):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])  # 计算左侧位置嵌入
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])  # 计算上侧位置嵌入
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])  # 计算右侧位置嵌入
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])  # 计算下侧位置嵌入
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e  # 捕获IndexError并抛出新的IndexError异常

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])  # 计算高度位置嵌入
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])  # 计算宽度位置嵌入

        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,  # 左位置嵌入
                upper_position_embeddings,  # 上位置嵌入
                right_position_embeddings,  # 右位置嵌入
                lower_position_embeddings,  # 下位置嵌入
                h_position_embeddings,  # 高度位置嵌入
                w_position_embeddings,  # 宽度位置嵌入
            ],
            dim=-1,
        )  # 将所有位置嵌入连接成一个张量
        spatial_position_embeddings = self.box_linear_embeddings(spatial_position_embeddings)  # 对位置嵌入进行线性变换
        box_position_embeddings = self.box_position_embeddings(position_ids)  # 获取框位置嵌入

        spatial_position_embeddings = spatial_position_embeddings + box_position_embeddings  # 将位置嵌入和框位置嵌入相加

        spatial_position_embeddings = self.LayerNorm(spatial_position_embeddings)  # LayerNorm归一化
        spatial_position_embeddings = self.dropout(spatial_position_embeddings)  # Dropout

        return spatial_position_embeddings  # 返回空间位置嵌入


class LiltSelfAttention(nn.Module):
    # 初始化函数，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类初始化函数
        super().__init__()
        # 检查隐藏层大小是否为注意力头数的倍数，如果不是且配置中没有嵌入大小，则抛出 ValueError 异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建布局查询、键和值的线性层
        self.layout_query = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )
        self.layout_key = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )
        self.layout_value = nn.Linear(
            config.hidden_size // config.channel_shrink_ratio, self.all_head_size // config.channel_shrink_ratio
        )

        # 创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果是相对键或相对键查询的位置嵌入类型，创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置通道缩减比率
        self.channel_shrink_ratio = config.channel_shrink_ratio

    # 重塑张量以适应注意力得分计算
    def transpose_for_scores(self, x, r=1):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size // r)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states,
        layout_inputs,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
# 从transformers.models.bert.modeling_bert.BertSelfOutput中复制过来的自注意力机制的输出层
class LiltSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，将隐藏状态映射到相同维度空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于规范化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃一部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层映射到相同维度空间
        hidden_states = self.dense(hidden_states)
        # 随机丢弃一部分神经元
        hidden_states = self.dropout(hidden_states)
        # 添加残差连接并进行 LayerNorm 规范化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 自定义的注意力机制模块
class LiltAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 自注意力机制层
        self.self = LiltSelfAttention(config, position_embedding_type=position_embedding_type)
        # 自注意力机制的输出层
        self.output = LiltSelfOutput(config)
        # 存储需要剪枝的注意力头的集合
        self.pruned_heads = set()

        # 保存原始的隐藏大小，用于剪枝操作
        ori_hidden_size = config.hidden_size
        # 根据配置参数缩减隐藏大小
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        # 布局层输出的自注意力机制输出层
        self.layout_output = LiltSelfOutput(config)
        # 恢复原始的隐藏大小
        config.hidden_size = ori_hidden_size

    # 从transformers.models.bert.modeling_bert.BertAttention.prune_heads中复制过来的剪枝函数
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可剪枝的注意力头及其索引
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
        layout_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用自注意力机制层处理输入
        self_outputs = self.self(
            hidden_states,
            layout_inputs,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 将自注意力机制层的输出经过输出层处理
        attention_output = self.output(self_outputs[0][0], hidden_states)
        # 将布局层输出的自注意力机制层的输出经过布局输出层处理
        layout_attention_output = self.layout_output(self_outputs[0][1], layout_inputs)
        # 整合输出结果，包括注意力权重（如果有）并返回
        outputs = ((attention_output, layout_attention_output),) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate中复制过来的中间层
class LiltIntermediate(nn.Module):
    # 初始化函数，接受config参数，初始化线性层dense
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串，则使用ACT2FN字典中对应的函数作为激活函数，否则使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受hidden_states张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LiltOutput(nn.Module):
    # 初始化函数，接受config参数，初始化线性层dense、LayerNorm层和dropout层
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受hidden_states张量和input_tensor张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # dropout处理hidden_states
        hidden_states = self.dropout(hidden_states)
        # LayerNorm处理hidden_states和input_tensor的和
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LiltLayer(nn.Module):
    # 初始化函数，接受config参数，初始化各个层并进行通道缩减操作
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LiltAttention(config)
        self.intermediate = LiltIntermediate(config)
        self.output = LiltOutput(config)

        ori_hidden_size = config.hidden_size
        ori_intermediate_size = config.intermediate_size
        config.hidden_size = config.hidden_size // config.channel_shrink_ratio
        config.intermediate_size = config.intermediate_size // config.channel_shrink_ratio
        # 对layout的中间层进行通道缩减操作
        self.layout_intermediate = LiltIntermediate(config)
        # 对layout的输出层进行通道缩减操作
        self.layout_output = LiltOutput(config)
        config.hidden_size = ori_hidden_size
        config.intermediate_size = ori_intermediate_size

    # 前向传播函数，接受hidden_states张量、layout_inputs张量以及一些可选参数，返回处理后的张量
    def forward(
        self,
        hidden_states: torch.Tensor,
        layout_inputs: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    # 定义一个方法，用于对输入的隐藏状态进行自注意力操作，返回自注意力输出和布局注意力输出
    def forward(
        self, hidden_states: torch.Tensor, layout_inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 self.attention 方法对隐藏状态进行注意力操作，得到自注意力输出和布局注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            layout_inputs,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力输出和布局注意力输出
        attention_output = self_attention_outputs[0][0]
        layout_attention_output = self_attention_outputs[0][1]

        # 如果输出注意力权重，则将其添加到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 对自注意力输出进行分块处理，通过调用 self.feed_forward_chunk 方法
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 对布局注意力输出进行分块处理，通过调用 self.layout_feed_forward_chunk 方法
        layout_layer_output = apply_chunking_to_forward(
            self.layout_feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, layout_attention_output
        )
        # 将处理后的输出添加到输出元组中
        outputs = ((layer_output, layout_layer_output),) + outputs

        # 返回输出元组
        return outputs

    # 从 transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk 复制的方法
    # 对给定的注意力输出进行前馈网络的处理
    def feed_forward_chunk(self, attention_output):
        # 使用中间层模块处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层模块处理中间层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的层输出
        return layer_output

    # 对布局注意力输出进行前馈网络的处理
    def layout_feed_forward_chunk(self, attention_output):
        # 使用布局中间层模块处理注意力输出
        intermediate_output = self.layout_intermediate(attention_output)
        # 使用布局输出层模块处理中间层输出
        layer_output = self.layout_output(intermediate_output, attention_output)
        # 返回处理后的布局层输出
        return layer_output
# LiltEncoder 类，用于编码输入序列
class LiltEncoder(nn.Module):
    # 初始化函数，接受一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 将配置参数保存到对象中
        self.config = config
        # 创建一个由多个 LiltLayer 组成的模块列表，列表长度为配置中指定的隐藏层数量
        self.layer = nn.ModuleList([LiltLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受输入张量和其他可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        layout_inputs: torch.Tensor,  # 布局输入张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩张量，默认为空
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩张量，默认为空
        output_attentions: Optional[bool] = False,  # 是否输出注意力，默认为 False
        output_hidden_states: Optional[bool] = False,  # 是否输出隐藏状态，默认为 False
        return_dict: Optional[bool] = True,  # 是否返回字典形式结果，默认为 True
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:  # 函数返回值的类型注释
        # 如果需要输出隐藏状态，则初始化一个空的隐藏状态元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化一个空的注意力元组
        all_self_attentions = () if output_attentions else None

        # 遍历编码器的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部遮罩
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且处于训练模式，则调用梯度检查点函数，否则直接调用当前层的前向传播函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layout_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    layout_inputs,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )

            # 更新隐藏状态和布局输入
            hidden_states = layer_outputs[0][0]
            layout_inputs = layer_outputs[0][1]

            # 如果需要输出注意力，则将当前层的注意力输出添加到注意力元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式结果，则返回元组形式结果
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        # 返回字典形式结果对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# LiltPooler 类，用于处理编码后的序列以生成池化表示
class LiltPooler(nn.Module):
    # 初始化函数，接受一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 全连接层，将编码后的序列映射到指定大小的特征空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数，使用双曲正切函数
        self.activation = nn.Tanh()
        # 定义一个名为forward的方法，参数为hidden_states，返回类型为torch.Tensor
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # 我们通过简单地取第一个标记对应的隐藏状态来"汇总"模型
            first_token_tensor = hidden_states[:, 0]
            # 通过全连接层处理第一个标记的隐藏状态
            pooled_output = self.dense(first_token_tensor)
            # 通过激活函数处理全连接层的输出
            pooled_output = self.activation(pooled_output)
            # 返回处理后的输出
            return pooled_output
class LiltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LiltConfig  # 设置配置类为 LiltConfig
    base_model_prefix = "lilt"  # 设置基础模型前缀为 "lilt"
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = []  # 初始化不分割模块为空列表

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 如果是线性层，初始化权重为正态分布
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()  # 如果有偏置项，初始化为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有填充索引，初始化为零
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  # 初始化层归一化的偏置为零
            module.weight.data.fill_(1.0)  # 初始化层归一化的权重为 1.0


LILT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LiltConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LILT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare LiLT Model transformer outputting raw hidden-states without any specific head on top.",
    LILT_START_DOCSTRING,  # 添加文档字符串开始标记
)
class LiltModel(LiltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)  # 初始化父类
        self.config = config

        self.embeddings = LiltTextEmbeddings(config)  # 初始化文本嵌入层
        self.layout_embeddings = LiltLayoutEmbeddings(config)  # 初始化布局嵌入层
        self.encoder = LiltEncoder(config)  # 初始化编码器

        self.pooler = LiltPooler(config) if add_pooling_layer else None  # 如果需要添加池化层则初始化池化层，否则为 None

        # Initialize weights and apply final processing
        self.post_init()  # 初始化权重并进行最终处理

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 返回输入嵌入的词嵌入层

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入嵌入的词嵌入层为给定的值
    # 对模型的头部进行修剪，heads_to_prune: 包含要修剪的每一层和头部的字典，参考基类 PreTrainedModel
    def _prune_heads(self, heads_to_prune):
        # 遍历需要修剪的每层和头部
        for layer, heads in heads_to_prune.items():
            # 获取编码器的指定层，然后调用 attention 模块的 prune_heads 方法进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 覆盖基类的 forward 方法，添加一些文档字符串
    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 导入所需模块
@add_start_docstrings(
    """
    LiLT模型转换器，顶部带有一个序列分类/回归头（在池化输出之上的线性层），例如用于GLUE任务。
    """,
    LILT_START_DOCSTRING,
)
# 声明LiltForSequenceClassification类，继承自LiltPreTrainedModel类
class LiltForSequenceClassification(LiltPreTrainedModel):
    # 从transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.__init__复制过来，将Roberta->Lilt，roberta->lilt
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将config中的num_labels赋值给实例变量num_labels
        self.num_labels = config.num_labels
        # 将config赋值给实例变量config
        self.config = config

        # 创建LiltModel实例，不添加池化层
        self.lilt = LiltModel(config, add_pooling_layer=False)
        # 创建LiltClassificationHead实例
        self.classifier = LiltClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加文档字符串
@add_start_docstrings(
    """
    Lilt模型，顶部带有一个标记分类头（隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
    """,
    LILT_START_DOCSTRING,
)
# 声明LiltForTokenClassification类，继承自LiltPreTrainedModel类
class LiltForTokenClassification(LiltPreTrainedModel):
    # 从transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.__init__复制过来，将Roberta->Lilt，roberta->lilt
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将config中的num_labels赋值给实例变量num_labels
        self.num_labels = config.num_labels

        # 创建LiltModel实例，不添加池化层
        self.lilt = LiltModel(config, add_pooling_layer=False)
        # 如果config中classifier_dropout不为空，则将其赋值给classifier_dropout，否则将config中的hidden_dropout_prob赋值给classifier_dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建Dropout层实例
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性层实例
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的词语ID张量，默认为None
        bbox: Optional[torch.LongTensor] = None,  # 边界框张量，默认为None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩张量，默认为None
        token_type_ids: Optional[torch.LongTensor] = None,  # 词语类型ID张量，默认为None
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID张量，默认为None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩张量，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入张量，默认为None
        labels: Optional[torch.LongTensor] = None,  # 用于计算标记分类损失的标签张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:  # 返回类型为torch.Tensor元组或TokenClassifierOutput类

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict为None，则使用self.config.use_return_dict

        outputs = self.lilt(  # 调用lilt模型
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # 获取输出的序列张量

        sequence_output = self.dropout(sequence_output)  # 对序列张量进行dropout处理
        logits = self.classifier(sequence_output)  # 序列张量通过分类器得到logits

        loss = None  # 初始化损失为None
        if labels is not None:  # 如果有标签
            # 将标签移到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失

        if not return_dict:  # 如果不返回字典
            output = (logits,) + outputs[2:]  # 构建输出元组
            return ((loss,) + output) if loss is not None else output  # 如果有损失，则将损失加入输出元组，否则返回输出元组

        return TokenClassifierOutput(  # 返回TokenClassifierOutput对象
            loss=loss,  # 损失
            logits=logits,  # logits
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制而来，将Roberta改为Lilt
class LiltClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        # 密集连接层，输入和输出大小都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的丢弃率，如果config.classifier_dropout不为None，则使用它，否则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 丢弃层，丢弃率为classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层，输入和输出大小都是config.hidden_size
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取出特征的第一个token，相当于[CLS]，形状为(batch_size, config.hidden_size)
        x = features[:, 0, :]
        # 对x进行丢弃
        x = self.dropout(x)
        # 通过密集连接层
        x = self.dense(x)
        # 使用tanh作为激活函数
        x = torch.tanh(x)
        # 再次进行丢弃
        x = self.dropout(x)
        # 通过输出层
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    在提取式问答任务（例如SQuAD）上使用的Lilt模型，其顶部有一个跨度分类头部（在隐藏状态输出的基础上使用线性层来计算
    `跨度起始对数`和`跨度结束对数`）。
    """,
    LILT_START_DOCSTRING,
)
class LiltForQuestionAnswering(LiltPreTrainedModel):
    # 从transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.__init__复制而来，将Roberta改为Lilt，roberta改为lilt
    def __init__(self, config):
        super().__init__(config)
        # 标签数等于config.num_labels
        self.num_labels = config.num_labels

        # 使用LiltModel构建Lilt模型，不添加池化层
        self.lilt = LiltModel(config, add_pooling_layer=False)
        # QA输出层，输入和输出大小都是config.hidden_size
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```