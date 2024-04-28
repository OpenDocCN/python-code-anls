# `.\transformers\models\squeezebert\modeling_squeezebert.py`

```
# 设定文件编码为 utf-8
# 版权声明，指明版权所有者和许可协议
# 本代码遵循 Apache License 2.0 许可协议
# 查看许可协议内容请访问指定链接
# 未按照许可协议使用本文件将导致违法
# 此源代码仅供参考，不提供任何担保，明确或隐含
# 请查看许可协议获取更多详细信息

# 引入必要的库与模块
import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引入 Hugging Face 模块、类和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_squeezebert import SqueezeBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 下面的常量用于文档
_CHECKPOINT_FOR_DOC = "squeezebert/squeezebert-uncased"
_CONFIG_FOR_DOC = "SqueezeBertConfig"
SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "squeezebert/squeezebert-uncased",
    "squeezebert/squeezebert-mnli",
    "squeezebert/squeezebert-mnli-headless",
]

# 定义 SqueezeBert 的嵌入模块
class SqueezeBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化函数，构建嵌入层
    def __init__(self, config):
        super().__init__()
        # 定义词嵌入层、位置嵌入层和标记类型嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # 使用 LayerNorm 层和 Dropout 层进行正则化
        # LayerNorm 的命名方式与 TensorFlow 保持一致，以便于加载 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 位置 ID 是内存中连续的，当序列化时会被导出
        # 创建一个不可持久化的缓冲区 position_ids，用于存储从 0 到最大位置嵌入的整数值
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
    # 前向传播函数，接收输入的各种张量参数，返回经过嵌入层处理后的张量表示
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入的input_ids不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 如果input_ids为空，则获取inputs_embeds的形状，不包括最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 如果position_ids为空，则使用预定义的位置编码，截取适当长度
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果token_type_ids为空，则创建一个与input_shape相同形状的全零张量，用于表示token类型
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果inputs_embeds为空，则使用word_embeddings将input_ids转换成嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入表示
        position_embeddings = self.position_embeddings(position_ids)
        # 获取token类型嵌入表示
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入、位置嵌入、token类型嵌入相加得到最终嵌入表示
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 将得到的嵌入表示进行LayerNorm归一化处理
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入表示进行dropout处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入表示
        return embeddings
class MatMulWrapper(nn.Module):
    """
    Wrapper for torch.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    torch.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        """
        Perform matrix multiplication of two input tensors

        :param inputs: two torch tensors
        :return: matmul of these tensors

        Here are the typical dimensions found in BERT (the B is optional)
        mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N]
        output shape: [B, <optional extra dims>, M, N]
        """
        return torch.matmul(mat1, mat2)


class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """

    def __init__(self, hidden_size, eps=1e-12):
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size, eps=eps)  # instantiates self.{weight, bias, eps}

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute the dimensions of the input tensor
        x = nn.LayerNorm.forward(self, x)  # Apply LayerNorm normalization
        return x.permute(0, 2, 1)  # Permute the dimensions back to the original order


class ConvDropoutLayerNorm(nn.Module):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """

    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)  # 1D convolutional layer
        self.layernorm = SqueezeBertLayerNorm(cout)  # Initialize the custom LayerNorm class
        self.dropout = nn.Dropout(dropout_prob)  # Apply dropout with a given probability

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)  # Apply 1D convolution to the input tensor
        x = self.dropout(x)  # Apply dropout to the output of the convolutional layer
        x = x + input_tensor  # Add the input tensor to the dropout output
        x = self.layernorm(x)  # Apply LayerNorm to the sum
        return x


class ConvActivation(nn.Module):
    """
    ConvActivation: Convolutional layer followed by activation
    """

    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)  # 1D convolutional layer
        self.act = ACT2FN[act]  # Get the specified activation function

    def forward(self, x):
        output = self.conv1d(x)  # Apply 1D convolution to the input tensor
        return self.act(output)  # Apply the activation function to the convolutional output


class SqueezeBertSelfAttention(nn.Module):
    # Class definition truncated, please provide the full class definition for commenting
    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        """
        # 初始化函数，接受参数config用于部分操作，cin代表输入通道数和输出通道数
        super().__init__()
        # 如果输入通道数不是注意力头数的倍数，则抛出异常
        if cin % config.num_attention_heads != 0:
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )
        # 设置注意力头数和每个头的大小信息
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(cin / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值对应的一维卷积层
        self.query = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        # 创建Dropout层和Softmax层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)

        # 创建矩阵乘法的包装器
        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        # 转置输入张量的维度顺序并返回
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1])  # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2)  # [N, C1, C2, W] --> [N, C1, W, C2]

    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        # 转置输入张量的维度顺序并返回
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1])  # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        # 不需要进行维度转置
        return x

    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        x = x.permute(0, 1, 3, 2).contiguous()  # [N, C1, C2, W]
        new_x_shape = (x.size()[0], self.all_head_size, x.size()[3])  # [N, C, W]
        x = x.view(*new_x_shape)
        return x
    def forward(self, hidden_states, attention_mask, output_attentions):
        """
        前向传播函数，用于计算自注意力机制后的上下文表示。

        hidden_states 的数据格式为 [N, C, W]，表示输入的隐藏状态。

        attention_mask 的数据格式为 [N, W]，表示注意力掩码，不需要转置。
        """
        # 计算查询、键、值的线性变换
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 将查询、键、值变换后的张量转置，以适应注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算原始的注意力分数，即查询和键的点积
        attention_score = self.matmul_qk(query_layer, key_layer)
        # 对注意力分数进行缩放
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        # 加上注意力掩码
        attention_score = attention_score + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = self.softmax(attention_score)

        # 使用 dropout 对注意力概率进行随机置零
        attention_probs = self.dropout(attention_probs)

        # 计算加权后的值向量，即注意力概率与值的乘积
        context_layer = self.matmul_qkv(attention_probs, value_layer)
        # 将输出的上下文向量转置，恢复为原始的数据格式
        context_layer = self.transpose_output(context_layer)

        # 构建返回结果字典，包含上下文向量和可能的注意力分数（如果需要）
        result = {"context_layer": context_layer}
        if output_attentions:
            result["attention_score"] = attention_score
        return result
class SqueezeBertModule(nn.Module):
    # 定义 SqueezeBertModule 类
    def __init__(self, config):
        # 初始化函数，接收一个配置参数
        """
        - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for
          the module
        - intermediate_size = output chans for intermediate layer
        - groups = number of groups for all layers in the BertModule. (eventually we could change the interface to
          allow different groups for different layers)
        """
        # 初始化函数注释，解释各个参数的含义
        super().__init__()

        # 初始化各个参数
        c0 = config.hidden_size
        c1 = config.hidden_size
        c2 = config.intermediate_size
        c3 = config.hidden_size

        # 创建自注意力机制对象
        self.attention = SqueezeBertSelfAttention(
            config=config, cin=c0, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups
        )
        # 创建后连接层对象
        self.post_attention = ConvDropoutLayerNorm(
            cin=c0, cout=c1, groups=config.post_attention_groups, dropout_prob=config.hidden_dropout_prob
        )
        # 创建中间层对象
        self.intermediate = ConvActivation(cin=c1, cout=c2, groups=config.intermediate_groups, act=config.hidden_act)
        # 创建输出层对象
        self.output = ConvDropoutLayerNorm(
            cin=c2, cout=c3, groups=config.output_groups, dropout_prob=config.hidden_dropout_prob
        )

    def forward(self, hidden_states, attention_mask, output_attentions):
        # 定义前向传播函数
        att = self.attention(hidden_states, attention_mask, output_attentions)
        # 获取自注意力机制输出
        attention_output = att["context_layer"]

        # 计算后连接层输出
        post_attention_output = self.post_attention(attention_output, hidden_states)
        # 计算中间层输出
        intermediate_output = self.intermediate(post_attention_output)
        # 计算最终层输出
        layer_output = self.output(intermediate_output, post_attention_output)

        # 构建输出字典
        output_dict = {"feature_map": layer_output}
        if output_attentions:
            output_dict["attention_score"] = att["attention_score"]

        return output_dict


class SqueezeBertEncoder(nn.Module):
    # 定义 SqueezeBertEncoder 类
    def __init__(self, config):
        # 初始化函数，接收一个配置参数
        super().__init__()

        # 断言确认嵌入大小与隐藏大小相同
        assert config.embedding_size == config.hidden_size, (
            "If you want embedding_size != intermediate hidden_size, "
            "please insert a Conv1d layer to adjust the number of channels "
            "before the first SqueezeBertModule."
        )

        # 创建包含多个 SqueezeBertModule 的模块列表
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
        # 如果头部掩码为None，设置头部掩码全部为None的标志为True
        if head_mask is None:
            head_mask_is_all_none = True
        # 如果头部掩码中的None的数量等于头部掩码的长度，设置头部掩码全部为None的标志为True
        elif head_mask.count(None) == len(head_mask):
            head_mask_is_all_none = True
        # 否则，设置头部掩码全部为None的标志为False
        else:
            head_mask_is_all_none = False
        # 断言头部掩码全部为None的标志为True，如果不是则抛出异常
        assert head_mask_is_all_none is True, "head_mask is not yet supported in the SqueezeBert implementation."

        # [batch_size, sequence_length, hidden_size] --> [batch_size, hidden_size, sequence_length]
        # 调整hidden_states张量的维度，将sequence_length维度移动到hidden_size之后
        hidden_states = hidden_states.permute(0, 2, 1)

        # 如果需要输出隐藏状态，则初始化空元组all_hidden_states
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化空元组all_attentions
        all_attentions = () if output_attentions else None

        # 遍历模型的每一层
        for layer in self.layers:
            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 调整hidden_states张量的维度
                hidden_states = hidden_states.permute(0, 2, 1)
                # 将当前隐藏状态添加到all_hidden_states元组中
                all_hidden_states += (hidden_states,)
                # 恢复hidden_states张量的维度
                hidden_states = hidden_states.permute(0, 2, 1)

            # 调用当前层的forward方法，计算当前层的输出
            layer_output = layer.forward(hidden_states, attention_mask, output_attentions)

            # 更新hidden_states为当前层的输出的特征图
            hidden_states = layer_output["feature_map"]

            # 如果需要输出注意力信息，则将当前层的注意力得分添加到all_attentions元组中
            if output_attentions:
                all_attentions += (layer_output["attention_score"],)

        # [batch_size, hidden_size, sequence_length] --> [batch_size, sequence_length, hidden_size]
        # 调整hidden_states张量的维度，将hidden_size维度移动到sequence_length之后
        hidden_states = hidden_states.permute(0, 2, 1)

        # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回元组形式的输出
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 如果需要返回字典形式的输出，则返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 定义一个名为SqueezeBertPooler的类，继承自nn.Module类
class SqueezeBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将输入的hidden_size进行变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用双曲正切函数作为激活函数
        self.activation = nn.Tanh()

    # 前向传播函数
    def forward(self, hidden_states):
        # 通过取出第一个标记对应的隐藏状态来进行“池化”
        first_token_tensor = hidden_states[:, 0]
        # 将取出的隐藏状态进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数对线性变换的结果进行激活
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 定义一个名为SqueezeBertPredictionHeadTransform的类，继承自nn.Module类
class SqueezeBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将输入的hidden_size进行变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 判断hidden_act是不是字符串类型，如果是则取对应的激活函数，如果不是则直接使用
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用层归一化对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数
    def forward(self, hidden_states):
        # 使用线性变换对隐藏状态进行变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对线性变换的结果进行激活
        hidden_states = self.transform_act_fn(hidden_states)
        # 对变换后的隐藏状态进行归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 定义一个名为SqueezeBertLMPredictionHead的类，继承自nn.Module类
class SqueezeBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用SqueezeBertPredictionHeadTransform对隐藏状态进行变换
        self.transform = SqueezeBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入一样，但每个标记都有一个仅用于输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化偏置参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接使得偏置能够在调整token嵌入时正确调整大小
        self.decoder.bias = self.bias

    # 前向传播函数
    def forward(self, hidden_states):
        # 使用SqueezeBertPredictionHeadTransform对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        # 将变换后的隐藏状态通过线性层得到输出
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 定义一个名为SqueezeBertOnlyMLMHead的类，继承自nn.Module类
class SqueezeBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用SqueezeBertLMPredictionHead对模型的预测做出构建
        self.predictions = SqueezeBertLMPredictionHead(config)

    # 前向传播函数
    def forward(self, sequence_output):
        # 获得模型预测得分
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 定义一个名为SqueezeBertPreTrainedModel的类，继承自PreTrainedModel类
class SqueezeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # SqueezeBertPreTrainedModel的配置类为SqueezeBertConfig
    config_class = SqueezeBertConfig
    # 基础模型前缀为transformer
    base_model_prefix = "transformer"
        def _init_weights(self, module):
            """Initialize the weights"""
            # 如果模块是线性层或一维卷积层
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # 根据配置的初始化范围对权重数据进行正态分布初始化
                # 与 TF 版本略有不同，TF 使用截断正态分布进行初始化，参考 https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                # 如果模块存在偏置项，则将偏置数据初始化为零
                if module.bias is not None:
                    module.bias.data.zero_()
            # 如果模块是嵌入层
            elif isinstance(module, nn.Embedding):
                # 根据配置的初始化范围对权重数据进行正态分布初始化
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                # 如果模块存在填充索引，则将对应位置的权重数据初始化为零
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            # 如果模块是 SqueezeBertLayerNorm 类型
            elif isinstance(module, SqueezeBertLayerNorm):
                # 将偏置数据初始化为零
                module.bias.data.zero_()
                # 将权重数据初始化为1.0
                module.weight.data.fill_(1.0)
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
    `
"""

SQUEEZEBERT_INPUTS_DOCSTRING = r"""

"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引
            # 可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 
            # 输入 IDs 是什么？ (../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 用于避免在填充标记索引上执行注意力的掩码
            # 掩码值选在 `[0, 1]` 之间：
            # - 1 表示**未被掩盖**的标记，
            # - 0 表示**被掩盖**的标记
            # 注意掩码是什么？ (../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，表示输入的第一部分和第二部分
            # 索引选在 `[0, 1]` 之间：
            # - 0 对应于*句子 A*的标记，
            # - 1 对应于*句子 B*的标记
            # 标记类型 IDs 是什么？ (../glossary#token-type-ids)

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引
            # 选择在范围内 `[0, config.max_position_embeddings - 1]`
            # 位置 IDs 是什么？ (../glossary#position-ids)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的选定头部置零的掩码
            # 掩码值选在 `[0, 1]` 之间：
            # - 1 表示该头部**未被掩盖**，
            # - 0 表示该头部**被掩盖**

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示，而不是传递 `input_ids`
            # 如果您希望更好地控制如何将 `input_ids` 索引转换为相关向量，这将很有用
            # 输出注意力张量是否返回所有层的注意力
            # 有关更多详细信息，请参见返回的张量下的 `attentions`

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 有关更多详细信息，请参见返回的张量下的 `attentions`

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 有关更多详细信息，请参见返回的张量下的 `hidden_states`

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是纯元组。
# 定义 SqueezeBERT 模型类，用于输出原始隐藏状态，不包含任何特定的顶层
@add_start_docstrings(
    "The bare SqueezeBERT Model transformer outputting raw hidden-states without any specific head on top.",
    SQUEEZEBERT_START_DOCSTRING,
)
class SqueezeBertModel(SqueezeBertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 初始化词嵌入层
        self.embeddings = SqueezeBertEmbeddings(config)
        # 初始化编码器
        self.encoder = SqueezeBertEncoder(config)
        # 初始化池化器
        self.pooler = SqueezeBertPooler(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    # 精简模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 定义模型的前向传播
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
    # 定义方法，用于模型的前向传播
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则引发错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果只指定了 input_ids
        elif input_ids is not None:
            # 如果 input_ids 和 attention_mask 存在填充，则发出警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        # 如果只指定了 inputs_embeds
        elif inputs_embeds is not None:
            # 获取 inputs_embeds 的形状，排除最后一个维度
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则引发错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果 attention_mask 不存在，则创建全 1 的注意力遮罩
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果 token_type_ids 不存在，则创建全 0 的 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 获取扩展的注意力遮罩
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        # 准备头部遮罩（如果需要）
        # 头部遮罩中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将输入传递给嵌入层
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # 将嵌入输出传递给编码器层
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 获取汇总输出
        pooled_output = self.pooler(sequence_output)

        # 如果不使用返回字典，则返回元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 使用返回字典返回模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 为 SqueezeBERT 模型添加一个基于语言建模的头部
@add_start_docstrings("""SqueezeBERT Model with a `language modeling` head on top.""", SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):
    # 定义共享权重的键值对
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建SqueezeBERT模型
        self.transformer = SqueezeBertModel(config)
        # 创建仅包含MLM头的SqueezeBERT模型
        self.cls = SqueezeBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 定义前向传播方法，包含详细的文档
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
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """
        Perform forward pass for the MaskedLM model.
    
        Args:
            input_ids (torch.LongTensor, optional):
                The input sequence tokens. Shape: [batch_size, sequence_length].
            attention_mask (torch.LongTensor, optional):
                The attention mask. Shape: [batch_size, sequence_length].
            token_type_ids (torch.LongTensor, optional):
                The token type IDs. Shape: [batch_size, sequence_length].
            position_ids (torch.LongTensor, optional):
                The position IDs. Shape: [batch_size, sequence_length].
            head_mask (torch.Tensor, optional):
                The head mask. Shape: [num_heads, sequence_length, sequence_length].
            inputs_embeds (torch.Tensor, optional):
                The embedded inputs. Shape: [batch_size, sequence_length, hidden_size].
            labels (torch.LongTensor, optional):
                The labels for computing the masked language modeling loss. Indices should be in
                [-100, 0, ..., config.vocab_size]. Tokens with indices set to -100 are ignored (masked).
                The loss is only computed for the tokens with labels in [0, ..., config.vocab_size].
            output_attentions (bool, optional):
                Whether to output attentions weights. Default: None.
            output_hidden_states (bool, optional):
                Whether to output hidden states. Default: None.
            return_dict (bool, optional):
                Whether to return a :class:`~transformers.file_utils.TransformerOutput` instead of a tuple. Default: None.
    
        Returns:
            Union[Tuple, MaskedLMOutput]: The output of forward pass.
        """
    
        # If return_dict is not provided, use the value in self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Perform forward pass through the transformer
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
    
        # Get the output of the transformer
        sequence_output = outputs[0]
    
        # Perform forward pass through the classification layer
        prediction_scores = self.cls(sequence_output)
    
        # Compute masked language modeling loss if labels are provided
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    
        # If return_dict is False, return the output as a tuple
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
        # If return_dict is True, return the output as a MaskedLMOutput object
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
    注释：
# 用于在类定义前添加文档字符串的装饰器
@add_start_docstrings(
    # 提供有关模型类的描述
    """
    SqueezeBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    # 使用预定义的 SQUEEZEBERT 文档字符串
    SQUEEZEBERT_START_DOCSTRING,
)
# 定义一个用于序列分类的 SqueezeBert 模型类，继承自 SqueezeBert 预训练模型类
class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    # 构造函数，用于初始化模型
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 保存模型的标签数量
        self.num_labels = config.num_labels
        # 保存配置
        self.config = config

        # 创建一个 SqueezeBert 模型实例，负责 Transformer 部分
        self.transformer = SqueezeBertModel(config)
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，作为序列分类的输出
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # 初始化权重并应用最终的处理步骤
        self.post_init()

    # 装饰器，用于在 forward 方法前添加文档字符串
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 装饰器，添加代码示例的文档字符串
    @add_code_sample_docstrings(
        # 指定文档的检查点和输出类型
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，用于定义模型在给定输入下的行为
    def forward(
        # 输入张量：包含输入 ID
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，控制模型关注的部分
        attention_mask: Optional[torch.Tensor] = None,
        # 令牌类型 ID，区分输入的不同部分
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID，用于位置编码
        position_ids: Optional[torch.Tensor] = None,
        # 头掩码，用于控制模型的注意力头
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入，如果需要自定义嵌入
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签，用于监督学习
        labels: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典
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

        # 从输出中获取 pooled_output
        pooled_output = outputs[1]

        # 对 pooled_output 进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器获取 logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果问题类型未指定
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型不同选择不同的损失函数
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

        # 如果不返回字典，则将输出转换为元组返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加对SqueezeBERT模型的注释，模型包含一个线性层和softmax层的多选分类头部，用于RocStories/SWAG任务
@add_start_docstrings(
    """
    SqueezeBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,
)
# 创建SqueezeBertForMultipleChoice类，继承自SqueezeBertPreTrainedModel类
class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):
    
    # 初始化函数，传入参数为config
    def __init__(self, config):
        # 调用父类SqueezeBertPreTrainedModel的初始化方法
        super().__init__(config)

        # 创建一个转换器对象，用于将输入数据转换成特征向量
        self.transformer = SqueezeBertModel(config)
        # 创建一个dropout对象，用于丢弃一部分网络连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性变换层，将特征向量映射到一个维度为1的输出
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接收一系列输入参数，返回模型的输出结果
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
        
...
    def forward(self,
            ) -> Union[Tuple, MultipleChoiceModelOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
                *input_ids* above)
            """
            # 如果 return_dict 为 None，则使用模型配置中的 use_return_dict 值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # 如果 input_ids 不为 None，则取输入的第二维长度作为选择总数
            num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    
            # 如果 input_ids 不为 None，则将其展平为二维
            input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
            # 如果 attention_mask 不为 None，则将其展平为二维
            attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
            # 如果 token_type_ids 不为 None，则将其展平为二维
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
            # 如果 position_ids 不为 None，则将其展平为二维
            position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
            # 如果 inputs_embeds 不为 None，则将其展平为三维
            inputs_embeds = (
                inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
                if inputs_embeds is not None
                else None
            )
    
            # 通过 transformer 处理输入
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
    
            # 从输出中获取 pooled_output
            pooled_output = outputs[1]
    
            # 对 pooled_output 进行 dropout
            pooled_output = self.dropout(pooled_output)
            # 使用分类器计算 logits
            logits = self.classifier(pooled_output)
            # 重新排列 logits 成为二维
            reshaped_logits = logits.view(-1, num_choices)
    
            loss = None
            # 如果 labels 不为 None，则计算损失
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)
    
            # 如果 return_dict 为 False，返回输出结果
            if not return_dict:
                output = (reshaped_logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
    
            # 返回 MultipleChoiceModelOutput 包含 loss、logits、hidden_states、attentions
            return MultipleChoiceModelOutput(
                loss=loss,
                logits=reshaped_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
# 导入所需模块
@add_start_docstrings(
    """
    SqueezeBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    SQUEEZEBERT_START_DOCSTRING,  # 添加文档字符串的开头
)
# 定义 SqueezeBertForTokenClassification 类，继承自 SqueezeBertPreTrainedModel
class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 使用 SqueezeBertModel 构建 transformer 模型
        self.transformer = SqueezeBertModel(config)
        # 添加 dropout 层，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加线性分类器层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
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
        # 如果 return_dict 为 None，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 transformer 的前向传播函数
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

        # 应用 dropout
        sequence_output = self.dropout(sequence_output)
        # 计算 logits
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加 SqueezeBertForTokenClassification 类的文档字符串
@add_start_docstrings(
    """
    SqueezeBERT 模型，顶部带有用于类似 SQuAD 的抽取式问答任务的跨度分类头部（在隐藏状态输出的顶部有线性层，用于计算“跨度起始对数”和“跨度结束对数”）。
    """,
    SQUEEZEBERT_START_DOCSTRING,
# 定义一个名为 SqueezeBertForQuestionAnswering 的类，这个类继承自 SqueezeBertPreTrainedModel 类
class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    # 初始化 SqueezeBertForQuestionAnswering 类
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类属性 num_labels 为 config 中的 num_labels 值
        self.num_labels = config.num_labels

        # 创建 SqueezeBert 模型对象
        self.transformer = SqueezeBertModel(config)
        # 创建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法，并添加了一些文档注释
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 检查是否需要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给变压器模型进行处理
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

        # 从模型输出中提取序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给问答输出层
        logits = self.qa_outputs(sequence_output)
        # 拆分起始位置和结束位置的logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除不必要的维度
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 初始化总损失为None
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上，则添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出了模型输入，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不需要返回字典，则返回损失和输出
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问答模型输出对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```