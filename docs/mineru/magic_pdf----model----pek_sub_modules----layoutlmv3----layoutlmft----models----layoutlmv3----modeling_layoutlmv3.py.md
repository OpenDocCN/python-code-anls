# `.\MinerU\magic_pdf\model\pek_sub_modules\layoutlmv3\layoutlmft\models\layoutlmv3\modeling_layoutlmv3.py`

```
# 指定源代码文件的编码格式为 UTF-8
# coding=utf-8
# 版权所有 2018 Google AI Language Team 和 HuggingFace Inc. 团队。
# 版权所有 (c) 2018, NVIDIA CORPORATION. 保留所有权利。
#
# 根据 Apache 2.0 许可协议（“许可”）授权；
# 除非遵循许可，否则您不得使用此文件。
# 您可以在以下地址获得许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可分发的软件在“按原样”基础上分发，
# 不提供任何明示或暗示的保证或条件。
# 有关许可证所涵盖权限和限制的具体信息，请参阅许可证。
"""PyTorch LayoutLMv3 模型。"""
# 导入数学库
import math

# 导入 PyTorch 相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
# 导入损失函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 transformers 导入应用于前向传播的分块处理函数
from transformers import apply_chunking_to_forward
# 导入模型输出类型
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
# 导入模型工具函数
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
# 导入 RoBERTa 模型的相关组件
from transformers.models.roberta.modeling_roberta import (
    RobertaIntermediate,
    RobertaLMHead,
    RobertaOutput,
    RobertaSelfOutput,
)
# 导入日志工具
from transformers.utils import logging

# 创建日志记录器
logger = logging.get_logger(__name__)

# 定义 PatchEmbed 类用于图像到补丁嵌入的转换
class PatchEmbed(nn.Module):
    """ 图像到补丁嵌入
    """
    # 初始化函数，定义图像大小、补丁大小、输入通道和嵌入维度
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 将图像大小转换为二元组
        img_size = to_2tuple(img_size)
        # 将补丁大小转换为二元组
        patch_size = to_2tuple(patch_size)
        # 计算补丁的形状
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 创建卷积层，用于生成补丁嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 以下变量用于 mycheckpointer.py 中的检测
        # 计算补丁总数
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # 保存补丁的宽度
        self.num_patches_w = self.patch_shape[0]
        # 保存补丁的高度
        self.num_patches_h = self.patch_shape[1]

    # 前向传播函数
    def forward(self, x, position_embedding=None):
        # 对输入进行卷积投影
        x = self.proj(x)

        # 如果存在位置嵌入
        if position_embedding is not None:
            # 将位置嵌入调整为相应的大小
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(0, 3, 1, 2)
            # 获取输出的高和宽
            Hp, Wp = x.shape[2], x.shape[3]
            # 使用双线性插值调整位置嵌入的大小
            position_embedding = F.interpolate(position_embedding, size=(Hp, Wp), mode='bicubic')
            # 将位置嵌入加到输入中
            x = x + position_embedding

        # 将张量展平并转置
        x = x.flatten(2).transpose(1, 2)
        # 返回结果
        return x

# 定义 LayoutLMv3Embeddings 类
class LayoutLMv3Embeddings(nn.Module):
    """
    # 与 BertEmbeddings 相同，但对位置嵌入索引进行了微小调整。
    """

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制的内容
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，词汇大小和隐藏层大小由配置指定，指定填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建标记类型嵌入层，类型词汇大小和隐藏层大小由配置指定
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建层归一化层，使用配置的隐层大小和 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建丢弃层，使用配置中的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 在序列化时，位置索引（1，位置嵌入长度）在内存中是连续的，并会被导出
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        # 复制结束
        # 设置填充索引为配置中的填充标记 ID
        self.padding_idx = config.pad_token_id
        # 创建位置嵌入层，最大位置嵌入数量和隐藏层大小由配置指定，填充索引使用 padding_idx
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # 创建二维位置嵌入层，分别用于 x 和 y 位置
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        # 创建高度位置嵌入层
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        # 创建宽度位置嵌入层
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    # 计算空间位置嵌入的方法，输入为边界框
    def _calc_spatial_position_embeddings(self, bbox):
        try:
            # 确保所有边界框坐标值在 0 到 1023 之间
            assert torch.all(0 <= bbox) and torch.all(bbox <= 1023)
            # 根据边界框的左侧坐标计算左位置嵌入
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            # 根据边界框的上侧坐标计算上位置嵌入
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            # 根据边界框的右侧坐标计算右位置嵌入
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            # 根据边界框的下侧坐标计算下位置嵌入
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            # 如果出现索引错误，抛出自定义错误信息
            raise IndexError("The :obj:`bbox` coordinate values should be within 0-1000 range.") from e

        # 根据边界框的高度计算高度位置嵌入，确保值在 0 到 1023 之间
        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        # 根据边界框的宽度计算宽度位置嵌入，确保值在 0 到 1023 之间
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # 以下是 LayoutLMEmbeddingsV2（torch.cat）和 LayoutLMEmbeddingsV1（add）之间的区别
        # 将所有位置嵌入连接成一个张量
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        # 返回计算得到的空间位置嵌入
        return spatial_position_embeddings
    # 创建从输入 ID 生成位置 ID 的方法，允许指定填充索引和过去的键值长度
    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        替换非填充符号为其位置编号。位置编号从 padding_idx + 1 开始。填充符号被忽略。
        该方法修改自 fairseq 的 `utils.make_positions`。

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # 这里的类型转换和类型转换经过精心平衡，以便同时支持 ONNX 导出和 XLA。
        # 创建一个掩码，标记非填充的输入 ID
        mask = input_ids.ne(padding_idx).int()
        # 计算累积的非填充符号索引，并加上过去的键值长度，乘以掩码以保留填充位置
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # 返回增量索引转换为长整型并加上填充索引
        return incremental_indices.long() + padding_idx

    # 前向传播方法，处理输入 ID 和嵌入
    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        # 如果未提供位置 ID
        if position_ids is None:
            # 如果提供了输入 ID
            if input_ids is not None:
                # 从输入的令牌 ID 创建位置 ID，任何填充的令牌保持填充状态
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
            else:
                # 从输入嵌入生成位置 ID
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 获取输入 ID 的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 从输入嵌入的形状获取输入形状（去掉最后一维）
            input_shape = inputs_embeds.size()[:-1]

        # 如果未提供令牌类型 ID，则创建全零的张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入嵌入，则从输入 ID 获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取令牌类型的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算总嵌入，将输入嵌入和令牌类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        # 获取位置嵌入并相加
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        # 计算空间位置嵌入
        spatial_position_embeddings = self._calc_spatial_position_embeddings(bbox)

        # 将空间位置嵌入添加到总嵌入中
        embeddings = embeddings + spatial_position_embeddings

        # 对嵌入进行层归一化
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入应用丢弃层
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入结果
        return embeddings

    # 创建从输入嵌入生成位置 ID 的方法
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        直接提供嵌入。我们无法推断哪些是填充的，因此只需生成连续的位置 ID。

        Args:
            inputs_embeds: torch.Tensor≈

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状（去掉最后一维）
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 创建位置 ID，从填充索引 + 1 开始，直到序列长度 + 填充索引 + 1
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 返回位置 ID，增加维度并扩展到输入形状
        return position_ids.unsqueeze(0).expand(input_shape
# 定义一个抽象类，用于处理权重初始化和提供下载与加载预训练模型的简单接口
class LayoutLMv3PreTrainedModel(PreTrainedModel):
    # 配置类为 LayoutLMv3Config
    config_class = LayoutLMv3Config
    # 基础模型前缀为 "layoutlmv3"
    base_model_prefix = "layoutlmv3"

    # 从 BertPreTrainedModel 复制的权重初始化方法
    def _init_weights(self, module):
        """初始化权重"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，与 TF 版本略有不同
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将其权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)


# 定义 LayoutLMv3 的自注意力模块
class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 验证隐藏层大小是否是注意力头数量的倍数
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"隐藏大小 ({config.hidden_size}) 不是注意力头数量 ({config.num_attention_heads}) 的倍数"
            )

        # 设置注意力头数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # 定义键线性层
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # 定义值线性层
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置是否具有相对注意力偏差
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 设置是否具有空间注意力偏差
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    # 将输入张量转置为适合注意力计算的格式
    def transpose_for_scores(self, x):
        # 新形状为 (batch_size, num_attention_heads, seq_length, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新调整输入张量形状
        x = x.view(*new_x_shape)
        # 返回转置后的张量
        return x.permute(0, 2, 1, 3)
    # 定义一个名为 cogview_attn 的方法，接收注意力分数和一个可选的 alpha 参数，默认为 32
    def cogview_attn(self, attention_scores, alpha=32):
        '''
        # 文档字符串，包含 PB-Relax 方法的引用和说明
        https://arxiv.org/pdf/2105.13290.pdf
        Section 2.4 Stabilization of training: Precision Bottleneck Relaxation (PB-Relax).
        A replacement of the original nn.Softmax(dim=-1)(attention_scores)
        Seems the new attention_probs will result in a slower speed and a little bias
        Can use torch.allclose(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison
        The smaller atol (e.g., 1e-08), the better.
        '''
        # 将注意力分数除以 alpha，以缩放注意力分数
        scaled_attention_scores = attention_scores / alpha
        # 计算缩放后的注意力分数在最后一个维度上的最大值，并在最后添加一个维度
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        # 将缩放后的注意力分数减去最大值，再乘以 alpha
        # max_value = scaled_attention_scores.amax(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        # 对新的注意力分数应用 Softmax，返回结果
        return nn.Softmax(dim=-1)(new_attention_scores)

    # 定义前向传播方法，接收多个参数
    def forward(
        # 隐藏状态，通常是上一层的输出
        hidden_states,
        # 注意力掩码，控制哪些位置参与注意力计算
        attention_mask=None,
        # 头掩码，控制各个注意力头的输出
        head_mask=None,
        # 编码器的隐藏状态，供解码器使用
        encoder_hidden_states=None,
        # 编码器的注意力掩码
        encoder_attention_mask=None,
        # 存储过去的键值对，用于长序列生成
        past_key_value=None,
        # 是否输出注意力分数
        output_attentions=False,
        # 相对位置编码
        rel_pos=None,
        # 2D 相对位置编码
        rel_2d_pos=None,
# 定义 LayoutLMv3Attention 类，继承自 nn.Module
class LayoutLMv3Attention(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 创建自注意力机制实例
        self.self = LayoutLMv3SelfAttention(config)
        # 创建输出层实例
        self.output = RobertaSelfOutput(config)
        # 初始化一个空集合，用于存储已剪枝的头
        self.pruned_heads = set()

    # 剪枝头的方法，接受头的列表
    def prune_heads(self, heads):
        # 如果没有需要剪枝的头，直接返回
        if len(heads) == 0:
            return
        # 找到可剪枝的头及其索引
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

    # 前向传播方法，处理输入数据
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 调用自注意力层进行计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 生成注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将输出结果组合成元组，包含注意力输出和其他输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回最终输出
        return outputs


# 定义 LayoutLMv3Layer 类，继承自 nn.Module
class LayoutLMv3Layer(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 设置前馈网络的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建注意力层实例
        self.attention = LayoutLMv3Attention(config)
        # 断言配置不支持解码器且不添加交叉注意力
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        # 创建中间层实例
        self.intermediate = RobertaIntermediate(config)
        # 创建输出层实例
        self.output = RobertaOutput(config)

    # 前向传播方法，处理输入数据
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 解码器单向自注意力缓存的键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力模块，传入隐藏状态、注意力掩码、头掩码等参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 获取自注意力的输出结果，通常是最后一层的输出
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，则添加自注意力输出
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将注意力输出传入前馈网络的分块处理函数
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将前馈网络输出和其他输出组合在一起
        outputs = (layer_output,) + outputs

        # 返回最终的输出元组
        return outputs

    def feed_forward_chunk(self, attention_output):
        # 将注意力输出传入中间层，得到中间输出
        intermediate_output = self.intermediate(attention_output)
        # 将中间输出和原始注意力输出一起传入输出层，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
# 定义 LayoutLMv3Encoder 类，继承自 nn.Module
class LayoutLMv3Encoder(nn.Module):
    # 初始化方法，接受配置参数和可选参数
    def __init__(self, config, detection=False, out_features=None):
        # 调用父类构造函数
        super().__init__()
        # 保存配置
        self.config = config
        # 保存是否为检测模式的标志
        self.detection = detection
        # 创建包含多个 LayoutLMv3Layer 的模块列表
        self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

        # 保存相对注意力偏置的配置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 保存空间注意力偏置的配置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 如果有相对注意力偏置，则初始化相关参数和线性层
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_onehot_size = config.rel_pos_bins
            # 创建用于相对位置偏置的线性层
            self.rel_pos_bias = nn.Linear(self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        # 如果有空间注意力偏置，则初始化相关参数和线性层
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_2d_pos_onehot_size = config.rel_2d_pos_bins
            # 创建用于二维相对位置偏置的线性层
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False)

        # 如果为检测模式，则启用梯度检查点，并初始化相关参数
        if self.detection:
            self.gradient_checkpointing = True
            embed_dim = self.config.hidden_size
            self.out_features = out_features
            # 提取输出特征的索引
            self.out_indices = [int(name[5:]) for name in out_features]
            # 定义第一组特征金字塔网络 (FPN) 层
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                # nn.SyncBatchNorm(embed_dim),  # 被注释掉的同步批量归一化
                nn.BatchNorm2d(embed_dim),  # 批量归一化层
                nn.GELU(),  # GELU 激活函数
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),  # 转置卷积层
            )

            # 定义第二组特征金字塔网络 (FPN) 层
            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),  # 转置卷积层
            )

            # 定义第三组特征金字塔网络 (FPN) 层，保持输入不变
            self.fpn3 = nn.Identity()

            # 定义第四组特征金字塔网络 (FPN) 层，最大池化
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
            # 将所有操作存储在列表中
            self.ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
    # 计算相对位置桶，用于将相对位置映射到离散的桶中
    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # 初始化返回值为 0
        ret = 0
        # 如果是双向，则将桶数减半
        if bidirectional:
            num_buckets //= 2
            # 如果相对位置大于 0，则将桶数加到返回值上
            ret += (relative_position > 0).long() * num_buckets
            # 取相对位置的绝对值
            n = torch.abs(relative_position)
        else:
            # 否则，将相对位置的负值和零的最大值进行比较，得到 n
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # 现在 n 的范围是 [0, inf)

        # 一半的桶用于确切的位移增量
        max_exact = num_buckets // 2
        # 判断 n 是否小于 max_exact
        is_small = n < max_exact

        # 另一半的桶用于对数增长的更大位置范围，最大距离为 max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        # 将 val_if_large 限制在最大桶数减一的范围内
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        # 根据 n 的大小选择小桶或大桶的值
        ret += torch.where(is_small, n, val_if_large)
        # 返回最终的桶值
        return ret

    # 计算一维位置嵌入
    def _cal_1d_pos_emb(self, hidden_states, position_ids, valid_span):
        # 定义视觉数量，196 + 1
        VISUAL_NUM = 196 + 1

        # 计算相对位置矩阵
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        # 如果 valid_span 不为 None
        if valid_span is not None:
            # 对于文本部分，如果两个词不在同一行，设置它们的距离为最大值
            rel_pos_mat[(rel_pos_mat > 0) & (valid_span == False)] = position_ids.shape[1]
            rel_pos_mat[(rel_pos_mat < 0) & (valid_span == False)] = -position_ids.shape[1]

            # 图像-文本的最小距离设为 0
            rel_pos_mat[:, -VISUAL_NUM:, :-VISUAL_NUM] = 0
            rel_pos_mat[:, :-VISUAL_NUM, -VISUAL_NUM:] = 0

        # 计算相对位置桶
        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # 将相对位置进行 one-hot 编码
        rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_onehot_size).type_as(hidden_states)
        # 应用相对位置偏置，并调整维度顺序
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        # 确保张量在内存中的连续性
        rel_pos = rel_pos.contiguous()
        # 返回相对位置嵌入
        return rel_pos
    # 计算二维位置嵌入
    def _cal_2d_pos_emb(self, hidden_states, bbox):
        # 从边界框中提取 x 坐标
        position_coord_x = bbox[:, :, 0]
        # 从边界框中提取 y 坐标
        position_coord_y = bbox[:, :, 3]
        # 计算 x 坐标之间的相对位置矩阵
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        # 计算 y 坐标之间的相对位置矩阵
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # 将 x 相对位置矩阵映射到离散桶中
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 将 y 相对位置矩阵映射到离散桶中
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 将 x 的相对位置转换为独热编码格式，并与隐藏状态类型一致
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        # 将 y 的相对位置转换为独热编码格式，并与隐藏状态类型一致
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size).type_as(hidden_states)
        # 应用 x 的相对位置偏置，并调整维度顺序
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        # 应用 y 的相对位置偏置，并调整维度顺序
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        # 确保 x 的相对位置数据在内存中是连续的
        rel_pos_x = rel_pos_x.contiguous()
        # 确保 y 的相对位置数据在内存中是连续的
        rel_pos_y = rel_pos_y.contiguous()
        # 合并 x 和 y 的相对位置嵌入
        rel_2d_pos = rel_pos_x + rel_pos_y
        # 返回二维相对位置嵌入
        return rel_2d_pos

    # 定义前向传播方法
    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        Hp=None,
        Wp=None,
        valid_span=None,
# 定义 LayoutLMv3Model 类，继承自 LayoutLMv3PreTrainedModel
class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    """
    类的文档字符串，可以在此处提供关于 LayoutLMv3Model 的描述
    """

    # 在加载模型时忽略的键，主要是缺失的 position_ids
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 从 BertModel 复制的初始化方法，调整为支持 RoBERTa
    def __init__(self, config, detection=False, out_features=None, image_only=False):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置参数
        self.config = config
        # 确保该版本不支持解码器和交叉注意力
        assert not config.is_decoder and not config.add_cross_attention, \
            "This version do not support decoder. Please refer to RoBERTa for implementation of is_decoder."
        # 保存检测标志
        self.detection = detection
        # 如果不是检测模式，设置 image_only 为 False
        if not self.detection:
            self.image_only = False
        else:
            # 如果是检测模式，确保启用了视觉嵌入
            assert config.visual_embed
            # 根据参数设置 image_only
            self.image_only = image_only

        # 如果不是 image_only 模式，初始化嵌入层
        if not self.image_only:
            self.embeddings = LayoutLMv3Embeddings(config)
        # 初始化编码器
        self.encoder = LayoutLMv3Encoder(config, detection=detection, out_features=out_features)

        # 如果配置中启用了视觉嵌入
        if config.visual_embed:
            # 获取隐藏层大小
            embed_dim = self.config.hidden_size
            # 使用默认预训练参数进行微调，处理输入大小
            # 如果微调时输入大小更大，则在前向传播中插值位置嵌入
            self.patch_embed = PatchEmbed(embed_dim=embed_dim)

            # 定义补丁大小
            patch_size = 16
            # 计算输入大小对应的尺寸
            size = int(self.config.input_size / patch_size)
            # 初始化类别标记的参数
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # 初始化位置嵌入的参数
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, embed_dim))
            # 初始化位置丢弃层
            self.pos_drop = nn.Dropout(p=0.)

            # 初始化层归一化
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            # 初始化丢弃层
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            # 如果配置中启用了相对注意力偏置或空间注意力偏置，初始化视觉边界框
            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self._init_visual_bbox(img_size=(size, size))

            # 导入部分函数以简化层归一化的创建
            from functools import partial
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            # 初始化归一化层
            self.norm = norm_layer(embed_dim)

        # 初始化权重
        self.init_weights()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 修剪模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        修剪模型的头部。 heads_to_prune: dict of {layer_num: list of heads to prune in this layer} 参见基类 PreTrainedModel
        """
        # 遍历每一层及其要修剪的头部
        for layer, heads in heads_to_prune.items():
            # 调用编码器的修剪方法
            self.encoder.layer[layer].attention.prune_heads(heads)
    # 初始化视觉边界框，指定图像尺寸和最大长度
        def _init_visual_bbox(self, img_size=(14, 14), max_len=1000):
            # 计算 x 方向的视觉边界框坐标
            visual_bbox_x = torch.div(torch.arange(0, max_len * (img_size[1] + 1), max_len),
                                      img_size[1], rounding_mode='trunc')
            # 计算 y 方向的视觉边界框坐标
            visual_bbox_y = torch.div(torch.arange(0, max_len * (img_size[0] + 1), max_len),
                                      img_size[0], rounding_mode='trunc')
            # 堆叠生成的边界框坐标并调整形状
            visual_bbox = torch.stack(
                [
                    visual_bbox_x[:-1].repeat(img_size[0], 1),  # 重复 x 方向坐标以生成矩形
                    visual_bbox_y[:-1].repeat(img_size[1], 1).transpose(0, 1),  # 重复 y 方向坐标并转置
                    visual_bbox_x[1:].repeat(img_size[0], 1),  # 重复下一个 x 方向坐标
                    visual_bbox_y[1:].repeat(img_size[1], 1).transpose(0, 1),  # 重复下一个 y 方向坐标并转置
                ],
                dim=-1,  # 在最后一个维度堆叠
            ).view(-1, 4)  # 重塑为 (-1, 4) 的形状
    
            # 创建包含分类 token 的边界框
            cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
            # 将分类 token 和视觉边界框连接
            self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)
    
        # 计算视觉边界框，将其复制到指定的设备和数据类型
        def _calc_visual_bbox(self, device, dtype, bsz):  # , img_size=(14, 14), max_len=1000):
            # 扩展视觉边界框以匹配批大小
            visual_bbox = self.visual_bbox.repeat(bsz, 1, 1)
            # 将视觉边界框转换到指定设备和数据类型
            visual_bbox = visual_bbox.to(device).type(dtype)
            # 返回处理后的视觉边界框
            return visual_bbox
    
        # 前向传播处理图像输入
        def forward_image(self, x):
            # 如果开启检测模式，嵌入图像块和位置嵌入
            if self.detection:
                x = self.patch_embed(x, self.pos_embed[:, 1:, :] if self.pos_embed is not None else None)
            else:
                # 如果没有检测模式，仅嵌入图像块
                x = self.patch_embed(x)
            # 获取批大小和序列长度
            batch_size, seq_len, _ = x.size()
    
            # 扩展分类 token 以匹配批大小
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            # 如果存在位置嵌入且处于检测模式，添加位置嵌入到分类 token
            if self.pos_embed is not None and self.detection:
                cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
    
            # 将分类 token 和图像嵌入连接
            x = torch.cat((cls_tokens, x), dim=1)
            # 如果存在位置嵌入且不在检测模式，添加位置嵌入到输入
            if self.pos_embed is not None and not self.detection:
                x = x + self.pos_embed
            # 应用丢弃层
            x = self.pos_drop(x)
    
            # 对输入进行归一化处理
            x = self.norm(x)
            # 返回处理后的输入
            return x
    
        # 前向传播方法，处理多个输入参数
        # Copied from transformers.models.bert.modeling_bert.BertModel.forward
        def forward(
            self,
            input_ids=None,
            bbox=None,
            attention_mask=None,
            token_type_ids=None,
            valid_span=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            images=None,
# 定义 LayoutLMv3ClassificationHead 类，继承自 nn.Module
class LayoutLMv3ClassificationHead(nn.Module):
    """
    用于句子级别分类任务的头部。
    参考: RobertaClassificationHead
    """

    # 初始化方法，接收配置和是否使用池化特征的参数
    def __init__(self, config, pool_feature=False):
        # 调用父类的初始化方法
        super().__init__()
        # 是否使用池化特征的标志
        self.pool_feature = pool_feature
        # 如果使用池化特征，则创建一个线性层，输入维度为 hidden_size 的三倍
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size*3, config.hidden_size)
        # 否则，创建一个线性层，输入维度为 hidden_size
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 获取分类器的 dropout 概率，如果没有设置，则使用隐藏层的 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个 Dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建输出的线性层，输入维度为 hidden_size，输出维度为标签数
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    # 定义前向传播方法
    def forward(self, x):
        # x = features[:, 0, :]  # 取 <s> 令牌（等同于 [CLS]）
        # 对输入 x 应用 dropout
        x = self.dropout(x)
        # 对经过 dropout 的 x 应用线性变换
        x = self.dense(x)
        # 对线性变换后的 x 应用 tanh 激活函数
        x = torch.tanh(x)
        # 再次对经过激活的 x 应用 dropout
        x = self.dropout(x)
        # 对经过 dropout 的 x 应用输出线性层
        x = self.out_proj(x)
        # 返回最终的输出
        return x


# 定义 LayoutLMv3ForTokenClassification 类，继承自 LayoutLMv3PreTrainedModel
class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    # 加载时忽略的键（如果意外存在）
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 加载时忽略的键（如果缺失）
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存标签的数量
        self.num_labels = config.num_labels

        # 创建 LayoutLMv3Model 实例
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果标签数少于 10，使用线性层作为分类器
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 否则，使用 LayoutLMv3ClassificationHead 作为分类器
        else:
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # 初始化权重
        self.init_weights()

    # 定义前向传播方法
    def forward(
        self,
        input_ids=None,           # 输入 ID
        bbox=None,               # 边界框信息
        attention_mask=None,     # 注意力掩码
        token_type_ids=None,     # 令牌类型 ID
        position_ids=None,       # 位置 ID
        valid_span=None,         # 有效范围
        head_mask=None,          # 注意力头掩码
        inputs_embeds=None,      # 输入嵌入
        labels=None,             # 标签
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,# 是否输出隐藏状态
        return_dict=None,        # 是否返回字典格式的输出
        images=None,             # 图像输入
    ):
        r"""  # 文档字符串，描述输入标签的形状和类型
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):  # 标签用于计算令牌分类损失，形状为(batch_size, sequence_length)，可选
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels - 1]``.  # 标签的索引范围应在[0, num_labels - 1]之间
        """
        # 如果未提供 return_dict，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 layoutlmv3 模型，传入多种输入参数
        outputs = self.layoutlmv3(
            input_ids,  # 输入的 ID
            bbox=bbox,  # 边界框信息
            attention_mask=attention_mask,  # 注意力掩码
            token_type_ids=token_type_ids,  # 令牌类型 ID
            position_ids=position_ids,  # 位置 ID
            head_mask=head_mask,  # 头掩码
            inputs_embeds=inputs_embeds,  # 输入嵌入
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 返回的格式
            images=images,  # 图像输入
            valid_span=valid_span,  # 有效的跨度
        )

        # 获取输出的序列部分
        sequence_output = outputs[0]

        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        # 通过分类器生成 logits
        logits = self.classifier(sequence_output)

        loss = None  # 初始化损失为 None
        if labels is not None:  # 如果提供了标签
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数
            # 仅保留损失的有效部分
            if attention_mask is not None:  # 如果存在注意力掩码
                active_loss = attention_mask.view(-1) == 1  # 获取有效的损失部分
                active_logits = logits.view(-1, self.num_labels)  # 变形 logits 为 (batch_size * seq_length, num_labels)
                active_labels = torch.where(  # 选择有效标签
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)  # 有效部分用真实标签填充，无效部分用忽略索引填充
                )
                # 计算损失
                loss = loss_fct(active_logits, active_labels)
            else:  # 如果没有注意力掩码
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失

        if not return_dict:  # 如果不需要返回字典
            output = (logits,) + outputs[2:]  # 将 logits 和其他输出组合
            return ((loss,) + output) if loss is not None else output  # 如果损失不为 None，返回损失和输出，否则仅返回输出

        # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss,  # 损失
            logits=logits,  # 预测的 logits
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力
        )
# 定义一个用于问答的 LayoutLMv3 模型类，继承自预训练模型基类
class LayoutLMv3ForQuestionAnswering(LayoutLMv3PreTrainedModel):
    # 加载时忽略意外出现的键，'pooler' 是可能不需要的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 加载时忽略缺失的键，'position_ids' 是可能缺失的键
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化函数，接收配置对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签的数量
        self.num_labels = config.num_labels

        # 创建 LayoutLMv3 模型实例
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 使用分类头进行问答任务，注释的行是一个线性层的初始化
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # 使用 LayoutLMv3ClassificationHead 进行分类输出
        self.qa_outputs = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # 初始化模型权重
        self.init_weights()

    # 前向传播函数，定义模型的输入和输出
    def forward(
        self,
        input_ids=None,            # 输入的 token ID
        attention_mask=None,      # 注意力掩码，标记哪些 token 是填充的
        token_type_ids=None,      # Token 类型 ID，用于区分不同类型的输入
        position_ids=None,        # 位置 ID，表示每个 token 的位置
        valid_span=None,          # 有效的 span，用于问答的边界
        head_mask=None,           # 掩码，用于选择性地关注某些头
        inputs_embeds=None,       # 直接传入的输入嵌入
        start_positions=None,      # 问题答案的起始位置
        end_positions=None,        # 问题答案的结束位置
        output_attentions=None,   # 是否输出注意力权重
        output_hidden_states=None, # 是否输出隐藏状态
        return_dict=None,         # 是否以字典形式返回输出
        bbox=None,                # 边界框信息，用于处理布局数据
        images=None,              # 图像数据，用于多模态任务
    ):


# 定义一个用于序列分类的 LayoutLMv3 模型类，继承自预训练模型基类
class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    # 加载时忽略缺失的键，'position_ids' 是可能缺失的键
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # 初始化函数，接收配置对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签的数量
        self.num_labels = config.num_labels
        # 保存配置对象
        self.config = config
        # 创建 LayoutLMv3 模型实例
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 使用 LayoutLMv3ClassificationHead 进行分类输出
        self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # 初始化模型权重
        self.init_weights()

    # 前向传播函数，定义模型的输入和输出
    def forward(
        self,
        input_ids=None,            # 输入的 token ID
        attention_mask=None,      # 注意力掩码，标记哪些 token 是填充的
        token_type_ids=None,      # Token 类型 ID，用于区分不同类型的输入
        position_ids=None,        # 位置 ID，表示每个 token 的位置
        valid_span=None,          # 有效的 span，用于分类任务的边界
        head_mask=None,           # 掩码，用于选择性地关注某些头
        inputs_embeds=None,       # 直接传入的输入嵌入
        labels=None,              # 输入的标签，用于训练时的损失计算
        output_attentions=None,   # 是否输出注意力权重
        output_hidden_states=None, # 是否输出隐藏状态
        return_dict=None,         # 是否以字典形式返回输出
        bbox=None,                # 边界框信息，用于处理布局数据
        images=None,              # 图像数据，用于多模态任务
    ):
    ):
        r"""
        # 文档字符串，描述输入标签的格式和用途
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            # 标签用于计算序列分类/回归损失。索引应在 :obj:`[0, ..., config.num_labels - 1]` 范围内。
            # 如果 :obj:`config.num_labels == 1`，计算回归损失（均方损失）。
            # 如果 :obj:`config.num_labels > 1`，计算分类损失（交叉熵）。
        """
        # 如果 return_dict 为空，则使用配置中的 use_return_dict 属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 layoutlmv3 模型，传入各种输入参数并获取输出
        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            images=images,
            valid_span=valid_span,
        )

        # 从模型输出中提取序列输出的第一个时间步的特征
        sequence_output = outputs[0][:, 0, :]
        # 通过分类器将序列输出转换为 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 确定问题类型，如果未指定则根据标签数量推断
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # 设定为回归问题
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    # 设定为单标签分类问题
                    self.config.problem_type = "single_label_classification"
                else:
                    # 设定为多标签分类问题
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 计算单一标签的损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 计算多标签的损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                # 计算单标签分类的损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用带逻辑回归的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                # 计算多标签分类的损失
                loss = loss_fct(logits, labels)

        # 如果不返回字典，则将 logits 和其他输出合并
        if not return_dict:
            output = (logits,) + outputs[2:]
            # 如果有损失，则返回损失和输出；否则仅返回输出
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则返回序列分类输出的详细结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```