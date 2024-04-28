# `.\transformers\models\segformer\modeling_segformer.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明，版权归 NVIDIA The HuggingFace Inc. 团队所有
# 根据 Apache 许可协议 2.0 版本使用代码，除非符合该许可协议，否则不得使用该文件
# 可在以下链接获取许可协议内容：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”方式分发软件，无论是明示还是暗示的，也没有任何形式的担保或条件
# 请查看有关详细信息和遵守条件的许可协议
""" PyTorch SegFormer model."""

# 导入必要的模块
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入 Hugging Face 相关模块
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_segformer import SegformerConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档说明的配置和检查点设置
_CONFIG_FOR_DOC = "SegformerConfig"
_CHECKPOINT_FOR_DOC = "nvidia/mit-b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 16, 16]

# 图像分类
_IMAGE_CLASS_CHECKPOINT = "nvidia/mit-b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# SegFormer 预训练模型存档列表
SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0-finetuned-ade-512-512",
    # 查看所有 SegFormer 模型：https://huggingface.co/models?filter=segformer
]

# 定义图像分类输出类，继承自 ImageClassifierOutput
class SegFormerImageClassifierOutput(ImageClassifierOutput):
    """
    Base class for outputs of image classification models.
    # `loss`参数：分类（如果`config.num_labels==1`则为回归）损失，是一个shape为`(1,)`的张量，当提供`labels`时返回
    # `logits`参数：分类（如果`config.num_labels==1`则为回归）得分，是一个shape为`(batch_size, config.num_labels)`的张量，未经过SoftMax处理
    # `hidden_states`参数：隐藏状态的元组，包含了`torch.FloatTensor`的张量（如果模型有嵌入层，则为嵌入层的输出，以及每个阶段的输出），形状为`(batch_size, num_channels, height, width)`
    # `attentions`参数：注意力的元组，包含了`torch.FloatTensor`的张量（每层一个），形状为`(batch_size, num_heads, patch_size, sequence_length)`，用于计算注意力加权平均值
# 从transformers.models.beit.modeling_beit.drop_path中复制过来的函数drop_path，用于在残差块的主路径中针对每个样本应用随机深度(drop paths)。
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    对每个样本进行随机深度(drop paths)（当应用于残差块的主路径时）。

    由Ross Wightman注释：这与我为EfficientNet等网络创建的DropConnect实现相同，
    但是，原始名称具有误导性，因为"Drop Connect"是一种不同形式的辍学（dropout）在另一篇论文中...
    请参阅讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 
    我选择更改该层和参数名称为'drop path'而不是混合使用DropConnect作为层名称，并使用'survival rate'作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 比特化
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.convnext.modeling_convnext.ConvNextDropPath中复制并更改为Segformer的类SegformerDropPath
class SegformerDropPath(nn.Module):
    """对每个样本进行随机深度（随机深度）（当应用于残差块的主路径中）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 构建重叠的补丁嵌入
class SegformerOverlapPatchEmbeddings(nn.Module):
    """构建重叠的补丁嵌入。"""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        # (batch_size, num_channels, height, width) -> (batch_size, num_channels, height*width) -> (batch_size, height*width, num_channels)
        # 这可以被传递给Transformer层
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


# SegFormer的高效自我关注机制。采用[PvT论文](https://arxiv.org/abs/2102.12122)中引入的序列减少过程。
class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer的高效自我关注机制。采用[PvT论文](https://arxiv.org/abs/2102.12122)中引入的序列减少过程。"""
    # 初始化函数，设置模型参数
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # 检查隐藏层大小是否可以被注意力头数整除
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        # 计算每个注意力头的大小和总的头大小
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 对查询、键和值进行线性变换
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        # 随机删除注意力概率以防止过拟合
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 如果序列缩减比例大于1，使用卷积层和 LayerNorm 进行序列压缩
        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    # 调整形状以便计算注意力分值
    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    # 正向传播函数，执行注意力计算
    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
        ):
        # 通过对隐藏状态进行查询，转置以匹配注意力分数形状
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # 重塑张量形状为 (batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # 应用序列缩减
            hidden_states = self.sr(hidden_states)
            # 将张量重新塑造为 (batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        # 通过对隐藏状态进行键和值的查询，转置以匹配注意力分数的形状
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 取“查询”和“键”之间的点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 这实际上是丢弃整个令牌来进行关注，这可能看起来有点不寻常，但是源自于原始的Transformer论文
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs
class SegformerSelfOutput(nn.Module):
    # 初始化方法，接收配置和隐藏层大小，创建线性层和dropout层
    def __init__(self, config, hidden_size):
        super().__init__()
        # 创建一个线性层
        self.dense = nn.Linear(hidden_size, hidden_size)
        # 创建一个dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，对隐藏状态进行线性变换和dropout
    def forward(self, hidden_states, input_tensor):
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerAttention(nn.Module):
    # 初始化方法，接收配置、隐藏层大小、注意力头数和序列压缩比率
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        # 创建自注意力层和输出层
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        # 初始化一个集合，存储要剪枝的头
        self.pruned_heads = set()

    # 对注意力头进行剪枝
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝后的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法，传入隐藏状态、高度、宽度和是否输出注意力矩阵
    def forward(self, hidden_states, height, width, output_attentions=False):
        # 使用自注意力层对隐藏状态进行处理
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 对注意力输出进行处理
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力矩��，则添加进来
        return outputs


class SegformerDWConv(nn.Module):
    # 初始化方法，接收维度参数，默认为768
    def __init__(self, dim=768):
        super().__init__()
        # 创建一个 depthwise 卷积层
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    # 前向传播方法，传入隐藏状态、高度和宽度
    def forward(self, hidden_states, height, width):
        # 获取隐藏状态的形状信息
        batch_size, seq_len, num_channels = hidden_states.shape
        # 转置并重塑隐藏状态
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        # 进行深度卷积
        hidden_states = self.dwconv(hidden_states)
        # 将卷积后的结果展平并重新调整形状
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):
    # 初始化方法，接受配置、输入特征数、隐藏特征数和输出特征数
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果输出特征数未提供，则将其设为输入特征数
        out_features = out_features or in_features
        # 创建一个全连接层对象，输入特征数为in_features，输出特征数为隐藏特征数
        self.dense1 = nn.Linear(in_features, hidden_features)
        # 创建SegformerDWConv对象，接收隐藏特征数
        self.dwconv = SegformerDWConv(hidden_features)
        # 根据配置确定中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个全连接层对象，输入特征数为隐藏特征数，输出特征数为out_features
        self.dense2 = nn.Linear(hidden_features, out_features)
        # 创建一个Dropout对象，根据配置的概率进行dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态、高度、宽度
    def forward(self, hidden_states, height, width):
        # 对隐藏状态进行第一个全连接层操作
        hidden_states = self.dense1(hidden_states)
        # 对隐藏状态进行深度可分离卷积操作
        hidden_states = self.dwconv(hidden_states, height, width)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对隐藏状态进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 对隐藏状态进行第二个全连接层操作
        hidden_states = self.dense2(hidden_states)
        # 对隐藏状态进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states
class SegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        # 添加 LayerNormalization 层，规范化隐藏状态的维度
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        # 添加 SegformerAttention 层，实现自注意力机制
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        # 添加 DropPath 层，实现随机深度连接，用于防止过拟合
        self.drop_path = SegformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 添加 LayerNormalization 层，规范化隐藏状态的维度
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        # 计算 MLP 的隐藏层大小
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # 添加 SegformerMixFFN 层，实现基于位置编码的 Feed-Forward 网络
        self.mlp = SegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width, output_attentions=False):
        # 对隐藏状态进行规范化
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 实现残差连接，结合自注意力的输出和原始隐藏状态
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        # 通过 MLP 网络处理隐藏状态，并实现残差连接
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # 实现残差连接，结合 MLP 的输出和原始隐藏状态
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class SegformerEncoder(nn.Module):
    # 初始化函数，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置参数
        self.config = config

        # 根据随机深度衰减规则生成衰减率列表
        drop_path_decays = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # 初始化补丁嵌入层
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # 初始化Transformer块
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # 每个块由多层组成
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # 初始化层归一化层
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    # 前向传播函数
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
``` 
        ) -> Union[Tuple, BaseModelOutput]:
            # 如果输出隐藏状态选项为False，则不返回任何隐藏状态，设为None
            all_hidden_states = () if output_hidden_states else None
            # 如果输出注意力权重选项为False，则不返回任何注意力权重，设为None
            all_self_attentions = () if output_attentions else None

            # 获取输入图片的批量大小
            batch_size = pixel_values.shape[0]

            hidden_states = pixel_values
            # 遍历patch embeddings、block和layer norm的组合
            for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
                embedding_layer, block_layer, norm_layer = x
                # 首先，获取patch embeddings
                hidden_states, height, width = embedding_layer(hidden_states)
                # 其次，将嵌入向量传入block层
                for i, blk in enumerate(block_layer):
                    layer_outputs = blk(hidden_states, height, width, output_attentions)
                    hidden_states = layer_outputs[0]
                    if output_attentions:
                        # 累加每个块的注意力权重
                        all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 第三步，应用layer normalization
                hidden_states = norm_layer(hidden_states)
                # 第四步，可选择性地将结果重新整形为(batch_size, num_channels, height, width)
                if idx != len(self.patch_embeddings) - 1 or (
                    idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
                ):
                    hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                # 如果输出隐藏状态选项为True，则累加所有隐藏状态
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                # 如果不返回字典形式的输出
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 返回字典格式的输出
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
class SegformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SegformerConfig
    base_model_prefix = "segformer"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果是线性层或卷积层，使用正态分布初始化权重，均值为0，标准差为设定的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，也使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有padding_idx，则对应位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于LayerNorm层，初始化偏置为0，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SEGFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SegformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEGFORMER_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional`:
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
# 类 SegformerModel 继承自 SegformerPreTrainedModel
class SegformerModel(SegformerPreTrainedModel):
    # 初始化模型类，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 将配置对象保存到当前实例中
        self.config = config
    
        # 创建一个 SegformerEncoder 对象作为模型的编码器
        self.encoder = SegformerEncoder(config)
    
        # 初始化权重并进行最终处理
        self.post_init()
    
    # 对模型中的注意力头进行修剪
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典，其中键是层号，值是需要修剪的注意力头列表
        for layer, heads in heads_to_prune.items():
            # 获取对应层的注意力头并进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    # 重写的 forward 方法，用于前向传播计算
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果未指定是否输出注意力矩阵，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定是否输出隐藏状态，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典形式的输出，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入数据传入编码器进行处理，并根据参数返回相应结果
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
    
        # 如果不需要返回字典形式的输出，则返回元组形式的结果
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
    
        # 如果需要返回字典形式的输出，则构造 BaseModelOutput 对象并返回
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用自定义的文档字符串装饰器，为 SegformerForImageClassification 类添加说明文档
@add_start_docstrings(
    """
    SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden
    states) e.g. for ImageNet.
    """,
    SEGFORMER_START_DOCSTRING,
)
# 定义 SegformerForImageClassification 类，继承自 SegformerPreTrainedModel
class SegformerForImageClassification(SegformerPreTrainedModel):
    # 类初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 从配置中获取标签数
        self.num_labels = config.num_labels
        # 创建 SegformerModel 对象
        self.segformer = SegformerModel(config)

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用自定义的文档字符串装饰器，为 forward 方法添加输入输出文档
    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=SegFormerImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义 forward 方法
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegFormerImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用SegFormer模型进行前向传播
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 将最后一个隐藏状态转换为(batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])

        # 全局平均池化
        sequence_output = sequence_output.mean(dim=1)

        # 使用分类器进行分类
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        # 初始化函数，定义线性映射层
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        # 将输入的 hidden_states 展平并转置
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # 通过线性映射层进行转换
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(SegformerPreTrainedModel):
    def __init__(self, config):
        # 初始化函数，定义解码头部结构
        super().__init__(config)
        # 创建多个 SegformerMLP 实例，用于统一通道维度
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # 实现原始实现中的 ConvModule 的三个层
        # 线性融合层
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        # 批量归一化层
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        # 激活函数层
        self.activation = nn.ReLU()

        # Dropout 层
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 分类器层
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config
    # 前向传播函数，接收编码器隐藏状态作为输入，返回预测的 logits
    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 获取批量大小
        batch_size = encoder_hidden_states[-1].shape[0]

        # 初始化存储所有隐藏状态的元组
        all_hidden_states = ()
        # 遍历编码器隐藏状态和线性层
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            # 如果不需要重塑最后一个阶段，并且编码器隐藏状态的维度为3
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                # 计算高度和宽度
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                # 重塑编码器隐藏状态的形状
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # 统一通道维度
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # 上采样
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        # 将所有隐藏状态连接并通过线性融合层
        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits 的形状为 (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        return logits
# 使用所有MLP解码头的SegFormer模型变换器，例如用于ADE20k、CityScapes等语义分割任务
class SegformerForSemanticSegmentation(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化SegFormer模型和解码头
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受像素值和标签等输入，返回语义分割器输出
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```