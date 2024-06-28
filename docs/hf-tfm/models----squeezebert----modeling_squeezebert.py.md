# `.\models\squeezebert\modeling_squeezebert.py`

```
# 设置文件编码为 UTF-8

# 版权声明，指出 SqueezeBert 项目的版权归属和许可信息
# 根据 Apache License, Version 2.0 许可证，除非符合许可证的要求，否则不得使用该文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，软件以“原样”分发，无任何明示或暗示的保证或条件
# 详细信息请参阅许可证，限制和条件
"""
PyTorch SqueezeBert 模型。
"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入活化函数映射表
from ...activations import ACT2FN
# 导入模型输出类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入工具函数：添加代码示例文档字符串、添加起始文档字符串、添加模型前向方法的起始文档字符串、日志记录
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 SqueezeBert 配置类
from .configuration_squeezebert import SqueezeBertConfig

# 获取 logger 实例，用于日志记录
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "squeezebert/squeezebert-uncased"
_CONFIG_FOR_DOC = "SqueezeBertConfig"

# 预训练模型存档列表
SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "squeezebert/squeezebert-uncased",
    "squeezebert/squeezebert-mnli",
    "squeezebert/squeezebert-mnli-headless",
]

# SqueezeBertEmbeddings 类，构建来自单词、位置和 token 类型嵌入的嵌入层
class SqueezeBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 单词嵌入层，使用配置中的词汇大小、嵌入维度和填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，使用配置中的最大位置嵌入大小和嵌入维度
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # token 类型嵌入层，使用配置中的 token 类型词汇大小和嵌入维度
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # LayerNorm 层，保持与 TensorFlow 模型变量名称一致，以便加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在内存中是连续的，并在序列化时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义一个前向传播方法，用于处理输入的各种信息并生成嵌入表示
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入的 input_ids 不为空，则获取其形状信息
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状信息（排除最后一维）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，这里假设 input_shape 是一个元组，包含 batch_size 和 seq_length
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则使用预定义的 position_ids 矩阵，截取到当前序列长度
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 token_type_ids，则创建全零的张量，形状与 input_shape 相同
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则利用 input_ids 获取嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 根据 position_ids 获取位置嵌入表示
        position_embeddings = self.position_embeddings(position_ids)
        
        # 根据 token_type_ids 获取类型嵌入表示
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入、位置嵌入和类型嵌入相加得到最终的嵌入表示
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        
        # 对嵌入表示进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        
        # 对嵌入表示进行 dropout 处理，防止过拟合
        embeddings = self.dropout(embeddings)
        
        # 返回处理后的嵌入表示
        return embeddings
# 定义了一个名为 MatMulWrapper 的神经网络模块类
class MatMulWrapper(nn.Module):
    """
    Wrapper for torch.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    torch.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        """
        执行前向传播计算

        :param mat1: 第一个 torch 张量
        :param mat2: 第二个 torch 张量
        :return: 两个张量的矩阵乘积

        这里描述了 BERT 中典型的张量维度，mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] 输出形状: [B, <optional extra dims>, M, N]
        """
        return torch.matmul(mat1, mat2)


# 定义了一个名为 SqueezeBertLayerNorm 的 nn.LayerNorm 子类
class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """

    def __init__(self, hidden_size, eps=1e-12):
        # 调用 nn.LayerNorm 的初始化方法来初始化自身
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size, eps=eps)  # instantiates self.{weight, bias, eps}

    def forward(self, x):
        # 将输入张量 x 的维度顺序变换为 NCW
        x = x.permute(0, 2, 1)
        # 调用 nn.LayerNorm 类的前向传播方法对 x 进行归一化
        x = nn.LayerNorm.forward(self, x)
        # 将归一化后的张量 x 的维度顺序变换回原来的维度顺序
        return x.permute(0, 2, 1)


# 定义了一个名为 ConvDropoutLayerNorm 的神经网络模块类
class ConvDropoutLayerNorm(nn.Module):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """

    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        # 定义一个 1 维卷积层
        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        # 定义一个 SqueezeBertLayerNorm 层
        self.layernorm = SqueezeBertLayerNorm(cout)
        # 定义一个 Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 对隐藏状态进行 1 维卷积
        x = self.conv1d(hidden_states)
        # 对卷积结果进行 Dropout 操作
        x = self.dropout(x)
        # 将 Dropout 后的结果与输入张量 input_tensor 相加
        x = x + input_tensor
        # 对相加后的结果进行 LayerNorm 归一化
        x = self.layernorm(x)
        return x


# 定义了一个名为 ConvActivation 的神经网络模块类
class ConvActivation(nn.Module):
    """
    ConvActivation: Conv, Activation
    """

    def __init__(self, cin, cout, groups, act):
        super().__init__()
        # 定义一个 1 维卷积层
        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        # 根据给定的激活函数名称 act，选择相应的激活函数
        self.act = ACT2FN[act]

    def forward(self, x):
        # 对输入张量 x 进行 1 维卷积
        output = self.conv1d(x)
        # 对卷积输出应用选择的激活函数
        return self.act(output)


class SqueezeBertSelfAttention(nn.Module):
    # 继续实现该类，未提供的部分未显示在此处
    pass
    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        """
        super().__init__()  # 调用父类的初始化方法
        if cin % config.num_attention_heads != 0:  # 如果输入通道数不是注意力头数的整数倍
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads  # 设置注意力头的数量
        self.attention_head_size = int(cin / config.num_attention_heads)  # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 计算所有注意力头的总大小

        # 创建查询、键、值的卷积层，用于注意力机制
        self.query = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 设置注意力概率的dropout层
        self.softmax = nn.Softmax(dim=-1)  # 创建softmax层，沿着最后一个维度进行softmax计算

        self.matmul_qk = MatMulWrapper()  # 创建矩阵乘法封装类的实例
        self.matmul_qkv = MatMulWrapper()  # 创建矩阵乘法封装类的实例

    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1])  # 计算新的形状
        x = x.view(*new_x_shape)  # 调整张量的形状
        return x.permute(0, 1, 3, 2)  # 转置张量的维度顺序，使得注意力头的信息在正确的维度上

    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1])  # 计算新的形状
        x = x.view(*new_x_shape)  # 调整张量的形状
        return x  # 返回未进行维度转置的张量

    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        x = x.permute(0, 1, 3, 2).contiguous()  # 转置张量的维度顺序，并保证内存中连续存储
        new_x_shape = (x.size()[0], self.all_head_size, x.size()[3])  # 计算新的形状
        x = x.view(*new_x_shape)  # 调整张量的形状
        return x  # 返回调整后的张量
    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        前向传播函数，用于计算自注意力机制后的结果。

        hidden_states: 输入的隐藏状态张量，数据布局为 [N, C, W]。
        attention_mask: 注意力掩码张量，数据布局为 [N, W]，不需要转置。
        output_attentions: 布尔值，指示是否输出注意力分数。

        返回包含上下文层和（可选）注意力分数的字典结果。
        """

        # 通过查询函数生成混合查询层
        mixed_query_layer = self.query(hidden_states)
        # 通过键函数生成混合键层
        mixed_key_layer = self.key(hidden_states)
        # 通过值函数生成混合值层
        mixed_value_layer = self.value(hidden_states)

        # 将混合查询层转置以获得注意力分数计算所需的格式
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # 将混合键层转置以获得注意力分数计算所需的格式
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        # 将混合值层转置以获得注意力分数计算所需的格式
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算原始的注意力分数，使用查询层和键层的点积
        attention_score = self.matmul_qk(query_layer, key_layer)
        # 将注意力分数除以 sqrt(注意力头大小)，用于稳定训练
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        # 加上预先计算的注意力掩码
        attention_score = attention_score + attention_mask

        # 将注意力分数归一化为注意力概率
        attention_probs = self.softmax(attention_score)

        # 使用 dropout 对注意力概率进行随机置零，以防止过拟合
        attention_probs = self.dropout(attention_probs)

        # 将注意力概率与值层相乘得到上下文层
        context_layer = self.matmul_qkv(attention_probs, value_layer)
        # 将上下文层转置为输出格式
        context_layer = self.transpose_output(context_layer)

        # 构建结果字典，包含上下文层
        result = {"context_layer": context_layer}
        # 如果需要输出注意力分数，则将其添加到结果中
        if output_attentions:
            result["attention_score"] = attention_score
        
        # 返回最终的结果字典
        return result
# 定义 SqueezeBertModule 类，继承自 nn.Module
class SqueezeBertModule(nn.Module):
    def __init__(self, config):
        """
        初始化函数，用于设置模块的各个层次
        - hidden_size = 输入通道数 = 输出通道数（Q、K、V 通道数相同）= 模块的输出通道数
        - intermediate_size = 中间层的输出通道数
        - groups = BertModule 中所有层的分组数（未来可以更改接口以允许不同层的不同分组）
        """
        super().__init__()

        # 从配置中获取各层的通道数
        c0 = config.hidden_size
        c1 = config.hidden_size
        c2 = config.intermediate_size
        c3 = config.hidden_size

        # 初始化注意力层
        self.attention = SqueezeBertSelfAttention(
            config=config, cin=c0, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups
        )
        # 初始化注意力后处理层
        self.post_attention = ConvDropoutLayerNorm(
            cin=c0, cout=c1, groups=config.post_attention_groups, dropout_prob=config.hidden_dropout_prob
        )
        # 初始化中间层
        self.intermediate = ConvActivation(cin=c1, cout=c2, groups=config.intermediate_groups, act=config.hidden_act)
        # 初始化输出层
        self.output = ConvDropoutLayerNorm(
            cin=c2, cout=c3, groups=config.output_groups, dropout_prob=config.hidden_dropout_prob
        )

    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        前向传播函数，用于计算模块的输出
        Args:
        - hidden_states: 输入的隐藏状态张量
        - attention_mask: 注意力掩码张量
        - output_attentions: 是否输出注意力分数

        Returns:
        - output_dict: 包含模块输出的字典，至少包含 "feature_map" 键
        """
        # 计算注意力
        att = self.attention(hidden_states, attention_mask, output_attentions)
        # 获取注意力层的输出
        attention_output = att["context_layer"]

        # 执行注意力后处理
        post_attention_output = self.post_attention(attention_output, hidden_states)
        # 执行中间层计算
        intermediate_output = self.intermediate(post_attention_output)
        # 执行输出层计算
        layer_output = self.output(intermediate_output, post_attention_output)

        # 准备输出字典
        output_dict = {"feature_map": layer_output}
        # 如果需要输出注意力分数，则将其加入输出字典
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]

        return output_dict


# 定义 SqueezeBertEncoder 类，继承自 nn.Module
class SqueezeBertEncoder(nn.Module):
    def __init__(self, config):
        """
        初始化函数，用于设置编码器的层数
        Args:
        - config: 包含模型配置信息的对象
        """
        super().__init__()

        # 确保嵌入尺寸与隐藏尺寸相同
        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size, "
            "please insert a Conv1d layer to adjust the number of channels "
            "before the first SqueezeBertModule."
        )

        # 创建编码器层列表
        self.layers = nn.ModuleList(SqueezeBertModule(config) for _ in range(config.num_hidden_layers))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        """
        前向传播函数，用于计算编码器的输出
        Args:
        - hidden_states: 输入的隐藏状态张量
        - attention_mask: 注意力掩码张量（默认为 None）
        - head_mask: 注意力头掩码张量（默认为 None）
        - output_attentions: 是否输出注意力分数（默认为 False）
        - output_hidden_states: 是否输出隐藏状态（默认为 False）
        - return_dict: 是否返回字典格式的输出（默认为 True）

        Returns:
        - 输出结果，根据 return_dict 参数决定返回类型
        """
        # 遍历每一层编码器模块进行前向传播
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, output_attentions)["feature_map"]

        # 如果需要返回字典格式的输出
        if return_dict:
            return {"last_hidden_state": hidden_states}

        # 否则直接返回隐藏状态张量
        return hidden_states
        if head_mask is None:
            head_mask_is_all_none = True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        else:
            head_mask_is_all_none = False
        assert head_mask_is_all_none is True, "head_mask is not yet supported in the SqueezeBert implementation."

        # 将隐藏状态的维度顺序从 [batch_size, sequence_length, hidden_size] 转换为 [batch_size, hidden_size, sequence_length]
        hidden_states = hidden_states.permute(0, 2, 1)

        # 如果输出隐藏状态，则初始化空元组 all_hidden_states
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化空元组 all_attentions
        all_attentions = () if output_attentions else None

        # 遍历网络层并处理每一层的隐藏状态和注意力权重
        for layer in self.layers:
            if output_hidden_states:
                # 将隐藏状态的维度顺序再次从 [batch_size, hidden_size, sequence_length] 转换回 [batch_size, sequence_length, hidden_size]
                hidden_states = hidden_states.permute(0, 2, 1)
                # 将当前层的隐藏状态添加到 all_hidden_states 中
                all_hidden_states += (hidden_states,)
                # 将隐藏状态的维度顺序恢复为 [batch_size, hidden_size, sequence_length]
                hidden_states = hidden_states.permute(0, 2, 1)

            # 对当前层进行前向传播，获取层的输出特征映射和注意力分数
            layer_output = layer.forward(hidden_states, attention_mask, output_attentions)

            # 更新隐藏状态为当前层的特征映射
            hidden_states = layer_output["feature_map"]

            # 如果输出注意力权重，则将当前层的注意力分数添加到 all_attentions 中
            if output_attentions:
                all_attentions += (layer_output["attention_score"],)

        # 将隐藏状态的维度顺序从 [batch_size, hidden_size, sequence_length] 转换回 [batch_size, sequence_length, hidden_size]
        hidden_states = hidden_states.permute(0, 2, 1)

        # 如果输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 如果不返回字典形式的结果，则返回隐藏状态、所有隐藏状态和所有注意力权重的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        # 返回字典形式的结果，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 定义一个自定义的池化层模块，用于SqueezeBert模型
class SqueezeBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 选择隐藏状态中的第一个 token 的特征向量作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的特征向量输入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 经过激活函数处理
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 定义一个预测头变换模块，用于SqueezeBert模型的预测任务
class SqueezeBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建LayerNorm层，对输入进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 输入经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过激活函数处理
        hidden_states = self.transform_act_fn(hidden_states)
        # 经过LayerNorm处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 定义一个语言模型预测头模块，用于SqueezeBert模型的预测任务
class SqueezeBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建预测头变换模块
        self.transform = SqueezeBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入的权重相同，但每个标记都有一个输出偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个偏置参数，每个词汇表的标记都有一个偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来确保偏置与`resize_token_embeddings`正确地调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 输入经过预测头变换模块
        hidden_states = self.transform(hidden_states)
        # 经过线性层处理
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 定义一个仅含MLM头模块，用于SqueezeBert模型的预测任务
class SqueezeBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建语言模型预测头模块
        self.predictions = SqueezeBertLMPredictionHead(config)

    def forward(self, sequence_output):
        # 输入序列经过语言模型预测头模块
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 定义一个SqueezeBert预训练模型的抽象类，用于处理权重初始化、预训练模型的下载和加载
class SqueezeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为SqueezeBertConfig
    config_class = SqueezeBertConfig
    # 基础模型前缀为"transformer"
    base_model_prefix = "transformer"
    # 定义私有方法 _init_weights，用于初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是全连接层或一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块有偏置项，则将偏置项初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，将该索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是自定义的 SqueezeBertLayerNorm 层
        elif isinstance(module, SqueezeBertLayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
# SQUEEZEBERT_START_DOCSTRING 是一个长字符串，用于描述 SqueezeBERT 模型及其相关信息。
# 提供了论文引用、类继承信息、PyTorch 模块说明和最佳微调建议。
SQUEEZEBERT_START_DOCSTRING = r"""

    The SqueezeBERT model was proposed in [SqueezeBERT: What can computer vision teach NLP about efficient neural
    networks?](https://arxiv.org/abs/2006.11316) by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, and Kurt W.
    Keutzer

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    For best results finetuning SqueezeBERT on text classification tasks, it is recommended to use the
    *squeezebert/squeezebert-mnli-headless* checkpoint as a starting point.

    Parameters:
        config ([`SqueezeBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Hierarchy:

    ```
    Internal class hierarchy:
    SqueezeBertModel
        SqueezeBertEncoder
            SqueezeBertModule
            SqueezeBertSelfAttention
                ConvActivation
                ConvDropoutLayerNorm
    ```

    Data layouts:

    ```
    Input data is in [batch, sequence_length, hidden_size] format.

    Data inside the encoder is in [batch, hidden_size, sequence_length] format. But, if `output_hidden_states == True`, the data from inside the encoder is returned in [batch, sequence_length, hidden_size] format.

    The final output of the encoder is in [batch, sequence_length, hidden_size] format.
    ```
"""

# SQUEEZEBERT_INPUTS_DOCSTRING 是一个空字符串，可能用于描述 SqueezeBERT 模型的输入说明，但当前未填充任何内容。
SQUEEZEBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。

            # 可以使用 `AutoTokenizer` 获得这些索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于在填充标记上避免执行注意力操作。遮罩值选取范围为 `[0, 1]`：

            # - 1 表示 **未遮罩** 的标记，
            # - 0 表示 **已遮罩** 的标记。

            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，用于指示输入的第一部分和第二部分。索引选取范围为 `[0, 1]`：

            # - 0 对应 *句子 A* 的标记，
            # - 1 对应 *句子 B* 的标记。

            # [什么是分段标记 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列每个标记在位置嵌入中的位置索引。索引选取范围为 `[0, config.max_position_embeddings - 1]`。

            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 遮罩，用于将自注意力模块的特定头部置零。遮罩值选取范围为 `[0, 1]`：

            # - 1 表示头部 **未遮罩**，
            # - 0 表示头部 **已遮罩**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选参数，可以直接传入嵌入表示，而不是传递 `input_ids`。如果需要对 `input_ids` 索引转换为相关向量具有更多控制权，则很有用。
            # 如果模型内部嵌入查找矩阵不符合需求，这将非常有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中有关 `attentions` 的详细信息。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中有关 `hidden_states` 的详细信息。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
@add_start_docstrings(
    "The bare SqueezeBERT Model transformer outputting raw hidden-states without any specific head on top.",
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertModel(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化模型的各个组件
        self.embeddings = SqueezeBertEmbeddings(config)  # 初始化嵌入层
        self.encoder = SqueezeBertEncoder(config)        # 初始化编码器
        self.pooler = SqueezeBertPooler(config)          # 初始化池化层

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的注意力头进行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果未指定output_attentions，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了input_ids和inputs_embeds，则抛出异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了input_ids，则检查padding情况，并获取其形状
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        # 如果只指定了inputs_embeds，则获取其形状（去除最后一维）
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既没有指定input_ids也没有指定inputs_embeds，则抛出异常
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 根据input_ids或inputs_embeds确定设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供attention_mask，则创建一个全为1的mask，形状与input_shape相同
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果未提供token_type_ids，则创建一个全为0的token_type_ids，数据类型为long，设备为device
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 获取扩展后的attention_mask，以确保形状匹配
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        # 如果需要，准备头部掩码
        # 在head_mask中的1.0表示保留该头部
        # attention_probs的形状为batch_size x num_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask转换为形状为[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将输入传递给嵌入层，获取嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # 将嵌入输出传递给编码器，获取编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从编码器输出中获取序列输出（通常是最后一层的隐藏状态）
        sequence_output = encoder_outputs[0]
        # 将序列输出传递给池化层，获取池化后的输出
        pooled_output = self.pooler(sequence_output)

        # 如果不要求返回字典，则返回一个元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典，则构建BaseModelOutputWithPooling对象并返回
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用自定义的文档字符串初始化 SqueezeBERT 模型，其顶部带有一个语言建模的头部
@add_start_docstrings("""SqueezeBERT Model with a `language modeling` head on top.""", SQUEEZEBERT_START_DOCSTRING)
# 定义 SqueezeBertForMaskedLM 类，继承自 SqueezeBertPreTrainedModel
class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):
    # 指定共享权重的键列表
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 SqueezeBertModel 对象，并赋值给 self.transformer
        self.transformer = SqueezeBertModel(config)
        # 创建 SqueezeBertOnlyMLMHead 对象，并赋值给 self.cls
        self.cls = SqueezeBertOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输出嵌入的方法，返回预测的解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的方法，将新的嵌入赋给解码器
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接受多个输入参数并返回输出
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
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
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # 确定前向传播是否返回一个字典形式的结果
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 如果 `return_dict` 为 `None`，则使用模型配置中的 `use_return_dict` 参数值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Transformer 模型进行处理
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Transformer 输出中获取序列输出
        sequence_output = outputs[0]
        
        # 将序列输出传递给分类器，生成预测分数
        prediction_scores = self.cls(sequence_output)

        # 初始化 masked_lm_loss 为 None
        masked_lm_loss = None
        
        # 如果提供了 labels，则计算 masked language modeling 的损失
        if labels is not None:
            # 使用交叉熵损失函数，-100 索引对应填充标记
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 `return_dict` 为 False，则返回的 output 包括预测分数和其他输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 `return_dict` 为 True，则返回 MaskedLMOutput 对象，包括损失、预测 logits、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    SqueezeBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    """
    SqueezeBERT模型的序列分类/回归头部变换器（在汇总输出顶部的线性层），例如用于GLUE任务。
    继承自SqueezeBertPreTrainedModel。
    """

    def __init__(self, config):
        """
        初始化方法，设置模型配置和层。
        
        Args:
            config (:class:`~transformers.SqueezeBertConfig`): 模型的配置对象。
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # SqueezeBERT模型，用于特征提取
        self.transformer = SqueezeBertModel(config)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器线性层，将隐藏状态映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
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
        # 根据参数 return_dict 的值确定是否使用配置中的 return_dict 设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Transformer 模型进行处理，并返回相应的输出
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Transformer 输出中获取池化后的表示
        pooled_output = outputs[1]

        # 对池化后的表示应用 dropout
        pooled_output = self.dropout(pooled_output)

        # 将池化后的表示输入分类器，得到预测 logits
        logits = self.classifier(pooled_output)

        # 初始化损失值为 None
        loss = None

        # 如果有指定 labels，则计算损失
        if labels is not None:
            # 如果问题类型未指定，则根据 num_labels 和 labels 的类型确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
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

        # 如果 return_dict 是 False，则返回包含 logits 和可能的隐藏状态的元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 是 True，则返回一个 SequenceClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型文档字符串，描述该模型是基于 SqueezeBERT 的多选分类模型
@add_start_docstrings(
    """
    SqueezeBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
# 定义 SqueezeBertForMultipleChoice 类，继承自 SqueezeBertPreTrainedModel
class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):
    
    # 初始化方法，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 初始化 SqueezeBERT 模型
        self.transformer = SqueezeBertModel(config)
        # Dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器，使用线性层将隐藏状态大小映射到 1（二进制分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(
        SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
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
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 方法接收多个参数，用于进行前向传播
        # 输入参数说明如下：
        # - input_ids: 输入的 token ids
        # - attention_mask: 注意力掩码，指示哪些元素是 padding 的
        # - token_type_ids: token 类型 ids，对于单句或双句模型有用
        # - position_ids: 位置 ids，指示每个 token 在序列中的位置
        # - head_mask: 头部掩码，用于控制哪些注意力头是有效的
        # - inputs_embeds: 可选的输入嵌入向量
        # - labels: 可选的标签，用于训练时的损失计算
        # - output_attentions: 是否输出注意力权重
        # - output_hidden_states: 是否输出隐藏状态
        # - return_dict: 是否返回字典形式的输出
        
        # 返回模型的输出，包括分类器的输出和其他可选的中间状态
        pass
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
            *input_ids* above)
        """
        # 如果 return_dict 参数为 None，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算 num_choices，即输入张量的第二个维度大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 以下四行代码将输入张量展平成二维张量，如果输入为 None 则保持为 None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 Transformer 模型进行前向传播，返回结果保存在 outputs 中
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Transformer 输出中获取 pooled_output，并应用 dropout
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # 将 pooled_output 输入分类器，得到 logits
        logits = self.classifier(pooled_output)

        # 将 logits 重塑成二维张量，形状为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了 labels，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，则返回不同的输出格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
SqueezeBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.
"""
# 导入所需模块和类
@add_start_docstrings(
    """
    SqueezeBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
# 定义SqueezeBertForTokenClassification类，继承自SqueezeBertPreTrainedModel类
class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    
    # 初始化方法，接收一个config对象作为参数
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # SqueezeBERT模型作为transformer层
        self.transformer = SqueezeBertModel(config)
        # Dropout层，根据配置中的hidden_dropout_prob概率丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 线性层，将隐藏状态的输出映射到config.num_labels个类别上
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接收多个输入参数并返回输出结果或损失
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
        # 根据self.config.use_return_dict决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入参数传递给transformer层，并接收输出结果
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取transformer层的输出中的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行dropout操作
        sequence_output = self.dropout(sequence_output)
        # 将dropout后的序列输出通过线性层映射到分类标签空间
        logits = self.classifier(sequence_output)

        # 初始化损失值为None
        loss = None
        # 如果labels不为None，则计算token分类损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 将logits展平为二维张量，计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典形式的输出，则按元组形式构造输出结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput对象，包含损失值、logits、隐藏状态和注意力权重等
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    """
    SqueezeBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    
    
    这段代码是一个字符串文档，描述了SqueezeBERT模型及其用途，特别是在类似SQuAD的抽取式问答任务中使用的特定功能。
)
class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = SqueezeBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs，可选的张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，用于指示哪些tokens要参与attention计算
        token_type_ids: Optional[torch.Tensor] = None,  # token类型IDs，用于区分segment
        position_ids: Optional[torch.Tensor] = None,  # 位置IDs，指示每个token的位置
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，指定要mask的attention头
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，代替input_ids和token_type_ids
        start_positions: Optional[torch.Tensor] = None,  # 起始位置的token索引
        end_positions: Optional[torch.Tensor] = None,  # 结束位置的token索引
        output_attentions: Optional[bool] = None,  # 是否输出attention权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回一个字典格式的输出
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 初始化返回字典，如果未提供则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 transformer 模型处理输入数据
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 使用 qa_outputs 进行问答模型的输出
        logits = self.qa_outputs(sequence_output)
        
        # 将 logits 拆分为 start_logits 和 end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # 去除多余的维度并确保连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # 如果提供了起始位置和结束位置
        if start_positions is not None and end_positions is not None:
            # 如果是在多 GPU 环境下，添加一个维度以便操作
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 忽略超出模型输入范围的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定的索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典形式的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 对象，包含损失、起始位置 logits、结束位置 logits、隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```