# `.\models\glpn\modeling_glpn.py`

```
# 设置源代码文件的编码格式为UTF-8，确保能正确处理中文等特殊字符
# 版权声明，版权归KAIST和The HuggingFace Inc.团队所有，保留所有权利
#
# 根据Apache许可证2.0版进行许可，除非符合许可条件，否则不得使用此文件
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 不提供任何形式的明示或暗示担保或条件
# 请参阅许可证了解具体的法律条款
""" PyTorch GLPN模型。"""

# 导入需要的模块
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 从其他模块导入必要的内容
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_glpn import GLPNConfig

# 获取logger对象用于记录日志信息
logger = logging.get_logger(__name__)

# 模型配置文件的标识符
_CONFIG_FOR_DOC = "GLPNConfig"

# 检查点模型的位置
_CHECKPOINT_FOR_DOC = "vinvino02/glpn-kitti"

# 预期输出的形状
_EXPECTED_OUTPUT_SHAPE = [1, 512, 15, 20]

# GLPN模型的预训练模型存档列表
GLPN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "vinvino02/glpn-kitti",
    # 查看所有GLPN模型的列表：https://huggingface.co/models?filter=glpn
]

# 从transformers.models.beit.modeling_beit.drop_path中复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    每个样本（在残差块的主路径中应用时）丢弃路径（随机深度）。

    评论由Ross Wightman提供：这与我为EfficientNet等网络创建的DropConnect实现相同，
    然而，原始名称具有误导性，因为'Drop Connect'是另一篇论文中不同形式的dropout……
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    …我选择更改层和参数名称为'drop path'而不是将DropConnect作为层名称混合使用，并使用'survival rate'作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.segformer.modeling_segformer.SegformerDropPath中复制的类
class GLPNDropPath(nn.Module):
    """每个样本（在残差块的主路径中应用时）丢弃路径（随机深度）。"""
    # 初始化函数，用于初始化DropPath模块
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 设置实例变量drop_prob，用于存储丢弃概率
        self.drop_prob = drop_prob

    # 前向传播函数，接收隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用drop_path函数，对隐藏状态进行DropPath操作，传入丢弃概率和训练模式标志
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 返回额外的表示信息，此处返回DropPath模块的丢弃概率
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)
# Copied from transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings
class GLPNOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        # 使用二维卷积操作来进行图像的特征提取，将输入的图像通道数变换为指定的隐藏层大小
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        # 对隐藏层输出进行层归一化操作
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        # 使用卷积层进行特征提取
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        # 将卷积层输出的特征张量展平，并进行维度置换，以便于传入Transformer层处理
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 对特征向量进行层归一化操作
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


# Copied from transformers.models.segformer.modeling_segformer.SegformerEfficientSelfAttention
class GLPNEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # 检查隐藏层大小是否能被注意力头数整除
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        # 定义注意力矩阵的dropout操作
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        # 如果序列缩减比率大于1，则使用二维卷积层进行序列缩减，并进行层归一化操作
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, hidden_states):
        # 将隐藏状态变换为注意力分数的形状
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
        output_attentions=False,
        ):
            # 对查询向量进行变换以适应注意力分数计算所需的形状
            query_layer = self.transpose_for_scores(self.query(hidden_states))

            # 如果设置的序列缩减比率大于1，则进行以下操作
            if self.sr_ratio > 1:
                batch_size, seq_len, num_channels = hidden_states.shape
                
                # 将隐藏状态重塑为 (batch_size, num_channels, height, width) 的形状
                hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
                
                # 应用序列缩减操作
                hidden_states = self.sr(hidden_states)
                
                # 将隐藏状态重塑为 (batch_size, seq_len, num_channels) 的形状
                hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
                
                # 应用层归一化
                hidden_states = self.layer_norm(hidden_states)

            # 对键向量进行变换以适应注意力分数计算所需的形状
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            
            # 对值向量进行变换以适应注意力分数计算所需的形状
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            # 计算 "查询" 和 "键" 之间的点积，得到原始注意力分数
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # 将注意力分数除以缩放因子 sqrt(attention_head_size)
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            # 对注意力分数进行归一化处理，转换为概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # 应用 dropout 操作以防止过拟合
            attention_probs = self.dropout(attention_probs)

            # 计算上下文向量，通过注意力概率与值向量的乘积
            context_layer = torch.matmul(attention_probs, value_layer)

            # 调整上下文向量的形状以适应后续层的输入要求
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)

            # 根据需要决定是否返回注意力概率
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            return outputs
# Copied from transformers.models.segformer.modeling_segformer.SegformerSelfOutput
class GLPNSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        # 初始化线性层，用于映射隐藏状态到相同大小的空间
        self.dense = nn.Linear(hidden_size, hidden_size)
        # 初始化丢弃层，根据给定的隐藏丢弃概率丢弃部分神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态通过线性层进行映射
        hidden_states = self.dense(hidden_states)
        # 对映射后的结果进行丢弃操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.segformer.modeling_segformer.SegformerAttention with Segformer->GLPN
class GLPNAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        # 初始化自注意力层，用于计算自注意力机制
        self.self = GLPNEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        # 初始化输出层，用于处理自注意力层的输出
        self.output = GLPNSelfOutput(config, hidden_size=hidden_size)
        # 初始化剪枝头信息的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可剪枝的头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝头部信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        # 进行自注意力计算
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 通过输出层处理自注意力层的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，添加到输出中
        return outputs


# Copied from transformers.models.segformer.modeling_segformer.SegformerDWConv
class GLPNDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # 初始化深度可分离卷积层，使用3x3卷积核
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        # 转置和重塑隐藏状态以适应深度可分离卷积层的输入格式
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        # 展平卷积后的结果并重新转置以恢复原始形状
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states
# 从 transformers.models.segformer.modeling_segformer.SegformerMixFFN 复制到 GLPNMixFFN，并将 Segformer 更名为 GLPN
class GLPNMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        # 创建一个全连接层，输入特征数为 in_features，输出特征数为 hidden_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        # 使用 GLPNDWConv 类来定义深度可分离卷积
        self.dwconv = GLPNDWConv(hidden_features)
        # 根据配置文件中的隐藏层激活函数类型选择合适的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个全连接层，输入特征数为 hidden_features，输出特征数为 out_features
        self.dense2 = nn.Linear(hidden_features, out_features)
        # 创建一个 dropout 层，使用配置文件中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        # 前向传播函数：先通过第一个全连接层
        hidden_states = self.dense1(hidden_states)
        # 然后通过深度可分离卷积层
        hidden_states = self.dwconv(hidden_states, height, width)
        # 再通过选定的中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 最后通过第二个全连接层
        hidden_states = self.dense2(hidden_states)
        # 再次应用 dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从 transformers.models.segformer.modeling_segformer.SegformerLayer 复制到 GLPNLayer，并将 Segformer 更名为 GLPN
class GLPNLayer(nn.Module):
    """这对应于原始实现中的 Block 类。"""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        # 创建一个层归一化层，输入大小为 hidden_size
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        # 创建一个 GLPNAttention 层，根据参数设置
        self.attention = GLPNAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        # 如果 drop_path 大于 0.0，则使用 GLPNDropPath，否则使用 nn.Identity()
        self.drop_path = GLPNDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 创建另一个层归一化层，输入大小为 hidden_size
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        # 计算 MLP 隐藏层大小
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # 创建一个 GLPNMixFFN 对象作为 MLP 层
        self.mlp = GLPNMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)
    # 定义神经网络模型的前向传播方法
    def forward(self, hidden_states, height, width, output_attentions=False):
        # 使用 self.attention 对隐藏状态进行自注意力机制处理，
        # 在 GLPN 中，layernorm 在应用自注意力机制之前进行
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # 对隐藏状态进行层归一化处理
            height,
            width,
            output_attentions=output_attentions,
        )

        # 获取自注意力机制的输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，将其添加到输出中

        # 第一个残差连接（带随机深度）
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        # 将隐藏状态应用 MLP 网络
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # 第二个残差连接（带随机深度）
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        # 将最终的层输出添加到输出中
        outputs = (layer_output,) + outputs

        # 返回所有输出
        return outputs
# 定义 GLPNEncoder 类，继承自 nn.Module，用于编码器部分的实现
class GLPNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        # 使用线性空间生成的概率衰减率列表，用于随机深度（stochastic depth）的实现
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        # 初始化 patch embeddings 列表，用于存储每个编码器块的重叠补丁嵌入层
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                GLPNOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        # 初始化 Transformer 块列表，用于存储每个编码器块的 Transformer 层
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # 每个块包含多个层
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    GLPNLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=dpr[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        # 初始化 LayerNorm 模块列表，用于每个编码器块的层归一化
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    # 前向传播函数定义，接受像素值和其他控制参数作为输入，返回编码器的输出
    def forward(
        self,
        pixel_values,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 初始化空元组，根据需要决定是否保存隐藏状态和注意力信息
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            # 获取批次大小
            batch_size = pixel_values.shape[0]

            # 初始化隐藏状态为输入像素值
            hidden_states = pixel_values
            # 迭代每个嵌入层、块层和规范化层
            for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
                embedding_layer, block_layer, norm_layer = x
                # 第一步，获取补丁嵌入
                hidden_states, height, width = embedding_layer(hidden_states)
                # 第二步，将嵌入通过块层
                for i, blk in enumerate(block_layer):
                    # 获取层输出，包括隐藏状态和可选的注意力信息
                    layer_outputs = blk(hidden_states, height, width, output_attentions)
                    hidden_states = layer_outputs[0]
                    if output_attentions:
                        all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 第三步，应用层规范化
                hidden_states = norm_layer(hidden_states)
                # 第四步，可选地重新整形为(batch_size, num_channels, height, width)
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不需要以字典形式返回结果，则返回相应的元组
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 否则，返回BaseModelOutput对象，包含最终隐藏状态、所有隐藏状态和所有注意力信息
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
class GLPNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 GLPNConfig
    config_class = GLPNConfig
    # 基础模型前缀为 "glpn"
    base_model_prefix = "glpn"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # Copied from transformers.models.segformer.modeling_segformer.SegformerPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或者卷积层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，使用正态分布初始化权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了 padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层，将偏置初始化为零，权重初始化为 1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# GLPN_START_DOCSTRING 文档字符串
GLPN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GLPNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# GLPN_INPUTS_DOCSTRING 文档字符串
GLPN_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`GLPNImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用 @add_start_docstrings 添加文档字符串
@add_start_docstrings(
    "The bare GLPN encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    GLPN_START_DOCSTRING,
)
# 定义 GLPNModel 类，继承自 GLPNPreTrainedModel
class GLPNModel(GLPNPreTrainedModel):
    # 从 transformers.models.segformer.modeling_segformer.SegformerModel.__init__ 处复制而来，将 Segformer 替换为 GLPN
    def __init__(self, config):
        # 调用父类 PreTrainedModel 的初始化方法
        super().__init__(config)
        # 将传入的配置参数保存到对象属性中
        self.config = config
    
        # 初始化 hierarchical Transformer 编码器，使用 GLPNEncoder
        self.encoder = GLPNEncoder(config)
    
        # 执行初始化权重和最终处理
        self.post_init()
    
    # 用于剪枝模型中注意力头部的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的每一层及其对应的注意力头部列表
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力模型进行头部剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    # 从 transformers.models.segformer.modeling_segformer.SegformerModel.forward 处复制而来
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 确定是否输出注意力权重，默认根据配置参数决定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认根据配置参数决定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的输出，默认根据配置参数决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入的像素值传递给编码器进行处理
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]
    
        # 如果不返回字典形式的输出，则返回序列输出及编码器的其他输出
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
    
        # 返回 BaseModelOutput 对象，其中包含序列输出、隐藏状态和注意力权重等信息
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# GLPNSelectiveFeatureFusion 类定义了选择性特征融合模块，参考论文第3.4节
class GLPNSelectiveFeatureFusion(nn.Module):
    """
    Selective Feature Fusion module, as explained in the [paper](https://arxiv.org/abs/2201.07436) (section 3.4). This
    module adaptively selects and integrates local and global features by attaining an attention map for each feature.
    """

    # 初始化方法，设置模块中的各个层
    def __init__(self, in_channel=64):
        super().__init__()

        # 第一个卷积层序列，包括卷积、批归一化和ReLU激活函数
        self.convolutional_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel * 2), out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
        )

        # 第二个卷积层序列，包括卷积、批归一化和ReLU激活函数
        self.convolutional_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU(),
        )

        # 第三个卷积层，直接定义，包括卷积操作
        self.convolutional_layer3 = nn.Conv2d(
            in_channels=int(in_channel / 2), out_channels=2, kernel_size=3, stride=1, padding=1
        )

        # Sigmoid 激活层，用于生成两通道的注意力图
        self.sigmoid = nn.Sigmoid()

    # 前向传播方法，接收局部特征和全局特征作为输入
    def forward(self, local_features, global_features):
        # 在通道维度上连接局部特征和全局特征
        features = torch.cat((local_features, global_features), dim=1)
        # 通过第一个卷积层序列处理特征
        features = self.convolutional_layer1(features)
        # 通过第二个卷积层序列处理特征
        features = self.convolutional_layer2(features)
        # 通过第三个卷积层得到特征，生成两通道的注意力图
        features = self.convolutional_layer3(features)
        # 应用 Sigmoid 激活函数生成注意力图
        attn = self.sigmoid(features)
        # 使用注意力图加权组合局部特征和全局特征，生成混合特征
        hybrid_features = local_features * attn[:, 0, :, :].unsqueeze(1) + global_features * attn[
            :, 1, :, :
        ].unsqueeze(1)

        return hybrid_features


# GLPNDecoderStage 类定义了解码器阶段，根据输入和输出通道的不同，可能会跳过卷积操作和选择性特征融合
class GLPNDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 如果输入和输出通道相同，使用恒等映射，否则使用卷积层
        should_skip = in_channels == out_channels
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1) if not should_skip else nn.Identity()
        # 选择性特征融合模块，根据输出通道数初始化
        self.fusion = GLPNSelectiveFeatureFusion(out_channels)
        # 上采样层，使用双线性插值方法，比例为2倍
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    # 前向传播方法，接收隐藏状态和可能的残差作为输入
    def forward(self, hidden_state, residual=None):
        # 如果需要，通过卷积层处理隐藏状态
        hidden_state = self.convolution(hidden_state)
        # 如果存在残差，通过选择性特征融合模块融合处理后的隐藏状态和残差
        if residual is not None:
            hidden_state = self.fusion(hidden_state, residual)
        # 通过上采样层进行特征图尺寸的放大
        hidden_state = self.upsample(hidden_state)

        return hidden_state
    def __init__(self, config):
        super().__init__()
        # we use features from end -> start
        # 根据配置反转隐藏层大小列表，以便从最后一个开始使用特征
        reserved_hidden_sizes = config.hidden_sizes[::-1]
        # 从配置中获取解码器隐藏层大小作为输出通道数
        out_channels = config.decoder_hidden_size

        # 创建阶段列表，每个阶段使用不同的隐藏层大小
        self.stages = nn.ModuleList(
            [GLPNDecoderStage(hidden_size, out_channels) for hidden_size in reserved_hidden_sizes]
        )
        # 在第一个阶段不进行融合操作
        self.stages[0].fusion = None

        # 创建最终的上采样层，使用双线性插值模式，并不对齐角点
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        # 存储每个阶段的隐藏状态
        stage_hidden_states = []
        # 初始阶段隐藏状态为 None
        stage_hidden_state = None
        # 逆序遍历隐藏状态列表和阶段列表
        for hidden_state, stage in zip(hidden_states[::-1], self.stages):
            # 对当前阶段应用隐藏状态和前一个阶段的隐藏状态，得到新的阶段隐藏状态
            stage_hidden_state = stage(hidden_state, stage_hidden_state)
            # 将新的阶段隐藏状态添加到阶段隐藏状态列表中
            stage_hidden_states.append(stage_hidden_state)

        # 对最后一个阶段的隐藏状态进行上采样
        stage_hidden_states[-1] = self.final_upsample(stage_hidden_state)

        # 返回所有阶段的隐藏状态列表
        return stage_hidden_states
class SiLogLoss(nn.Module):
    r"""
    Implements the Scale-invariant log scale loss [Eigen et al., 2014](https://arxiv.org/abs/1406.2283).

    $$L=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{2 n^{2}}\left(\sum_{i} d_{i}^{2}\right)$$ where $d_{i}=\log y_{i}-\log
    y_{i}^{*}$.

    """

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        # 创建一个有效值掩码，用于排除目标值为零的情况
        valid_mask = (target > 0).detach()
        # 计算预测值和目标值的对数差异
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        # 计算损失函数
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class GLPNDepthEstimationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        channels = config.decoder_hidden_size
        # 定义深度估计头部的神经网络层次结构
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # 使用解码器的最后一个特征作为输入
        hidden_states = hidden_states[self.config.head_in_index]

        hidden_states = self.head(hidden_states)

        # 计算预测的深度，通过 sigmoid 函数和最大深度缩放
        predicted_depth = torch.sigmoid(hidden_states) * self.config.max_depth
        predicted_depth = predicted_depth.squeeze(dim=1)

        return predicted_depth


@add_start_docstrings(
    """GLPN Model transformer with a lightweight depth estimation head on top e.g. for KITTI, NYUv2.""",
    GLPN_START_DOCSTRING,
)
class GLPNForDepthEstimation(GLPNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 GLPN 模型、解码器和深度估计头部
        self.glpn = GLPNModel(config)
        self.decoder = GLPNDecoder(config)
        self.head = GLPNDepthEstimationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(GLPN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> DepthEstimatorOutput:
        # 此处省略部分代码
        ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:
            Depending on the configuration and inputs, returns either a tuple with loss and predicted depth,
            or a `DepthEstimatorOutput` object containing loss, predicted depth, hidden states, and attentions.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, GLPNForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
        >>> model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```
        """
        # Determine whether to use the provided return_dict or the model's default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Determine whether to include hidden states in the outputs based on the provided flag or config
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Pass input through GLPN model to obtain outputs including hidden states if needed
        outputs = self.glpn(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        # Select the appropriate hidden states depending on the return_dict flag
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # Decode hidden states and predict depth map
        out = self.decoder(hidden_states)
        predicted_depth = self.head(out)

        # Compute loss if ground truth labels are provided
        loss = None
        if labels is not None:
            loss_fct = SiLogLoss()
            loss = loss_fct(predicted_depth, labels)

        # Prepare output based on return_dict and output_hidden_states settings
        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]  # Include hidden states in output
            else:
                output = (predicted_depth,) + outputs[2:]  # Exclude hidden states from output
            return ((loss,) + output) if loss is not None else output

        # Return structured DepthEstimatorOutput object with all relevant components
        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```