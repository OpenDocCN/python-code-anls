# `.\models\glpn\modeling_glpn.py`

```
# 定义模型的编码，使用 UTF-8 编码格式
# 版权声明，版权归 KAIST 和 HuggingFace Inc. 团队所有
# 遵循 Apache 许可证 2.0 版本，除非符合许可证，否则禁止使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得分发软件
# 软件按"现状"分发，没有任何形式的明示或暗示担保或条件
# 请参阅许可证了解特定语言的权限和限制
""" PyTorch GLPN 模型。"""

# 导入需要的库和模块
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入相关模块和类
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档生成的通用文本描述
_CONFIG_FOR_DOC = "GLPNConfig"

# 用于文档生成的基础模型检查点
_CHECKPOINT_FOR_DOC = "vinvino02/glpn-kitti"
# 预期输出形状的描述
_EXPECTED_OUTPUT_SHAPE = [1, 512, 15, 20]

# 预训练模型存档列表
GLPN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "vinvino02/glpn-kitti",
    # 查看所有 GLPN 模型：https://huggingface.co/models?filter=glpn
]

# 从 transformers.models.beit.modeling_beit.drop_path 中复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    按样本丢弃路径（随机深度），用于残差块的主路径。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    然而，原始名称可能会误导，因为"Drop Connect"是一篇独立论文中的不同形式的 dropout...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    我选择更改层和参数名称为"drop path"而不是混合使用 DropConnect 作为层名称并使用"survival rate"作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是二维 ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output

# 从 transformers.models.segformer.modeling_segformer.SegformerDropPath 中复制的类
class GLPNDropPath(nn.Module):
    """按样本丢弃路径（随机深度），用于残差块的主路径。"""
    # 初始化函数，初始化DropPath对象
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        # 调用父类初始化函数
        super().__init__()
        # 设置dropout概率
        self.drop_prob = drop_prob

    # 前向传播函数，接受隐藏层状态，返回经DropPath处理后的隐藏层状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用drop_path函数对隐藏层状态进行dropout操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 额外的表示函数，返回DropPath对象的相关表示信息
    def extra_repr(self) -> str:
        # 返回以字符串形式表示的dropout概率信息
        return "p={}".format(self.drop_prob)
# 从transformers.models.segformer.modeling_segformer.SegformerOverlapPatchEmbeddings复制过来的类
class GLPNOverlapPatchEmbeddings(nn.Module):
    """构建重叠的Patch嵌入。"""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        # 使用卷积层将输入的通道转换为隐藏的嵌入向量
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        # 使用Layer Normalization对隐藏嵌入向量进行标准化
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        # 将像素值投射到隐藏嵌入空间
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        # 将嵌入转换为(batch_size, height*width, num_channels)的形状
        # 可以输入到Transformer层中
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width

# 从transformers.models.segformer.modeling_segformer.SegformerEfficientSelfAttention复制过来的类
class GLPNEfficientSelfAttention(nn.Module):
    """SegFormer的高效自注意机制。采用了[PvT paper](https://arxiv.org/abs/2102.12122)中引入的序列缩减过程。"""

    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            # 如果序列缩减比例大于1，则使用卷积层进行序列缩减
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequence_reduction_ratio, stride=sequence_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, hidden_states):
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
        # 通过transpose_for_scores方法对查询向量进行转置处理
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 如果注意力头的数量大于1
        if self.sr_ratio > 1:
            # 获取隐藏状态的batch_size，序列长度和通道数
            batch_size, seq_len, num_channels = hidden_states.shape
            # 重塑形状为(batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # 应用序列降维
            hidden_states = self.sr(hidden_states)
            # 重塑形状为(batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        # 通过transpose_for_scores方法对键向量和值向量进行转置处理
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 通过矩阵乘法计算"query"和"key"之间的原始注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 通过数学计算缩放原始注意力得分
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力得分标准化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 这实际上是丢弃整个令牌以进行注意力，这可能看起来有点不寻常，但取自原始Transformer论文
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文向量进行维度转换和连续操作
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力权重，则返回注意力权重和上下文向量，否则只返回上下文向量
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.segformer.modeling_segformer.SegformerSelfOutput复制而来的GLPNSelfOutput类
class GLPNSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        # 定义一个全连接层，将输入特征维度映射到相同的维度
        self.dense = nn.Linear(hidden_size, hidden_size)
        # 定义一个dropout层，用于随机将输入张量的部分元素设置为0，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 通过全连接层映射输入特征
        hidden_states = self.dense(hidden_states)
        # 对映射后的特征进行dropout操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从transformers.models.segformer.modeling_segformer.SegformerAttention复制而来的GLPNAttention类，将Segformer->GLPN
class GLPNAttention(nn.Module):
    def __init__(self, config, hidden_size, num_attention_heads, sequence_reduction_ratio):
        super().__init__()
        # 初始化GLPNEfficientSelfAttention对象
        self.self = GLPNEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        # 初始化GLPNSelfOutput对象
        self.output = GLPNSelfOutput(config, hidden_size=hidden_size)
        # 初始化一个用于记录被剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用辅助函数找到可剪枝的注意力头和相应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, height, width, output_attentions=False):
        # 调用GLPNEfficientSelfAttention的forward方法
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 将自注意力层的输出通过输出层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 返回输出，如果需要输出注意力矩阵，则添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.segformer.modeling_segformer.SegformerDWConv复制而来的GLPNDWConv类
class GLPNDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # 定义一个2D深度可分离卷积层，用于对输入张量进行卷积操作
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        # 获取输入张量的形状信息
        batch_size, seq_len, num_channels = hidden_states.shape
        # 将输入张量转置并重塑以适应卷积操作的输入要求
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        # 通过深度可分离卷积层进行卷积操作
        hidden_states = self.dwconv(hidden_states)
        # 将卷积后的张量展平，并恢复其形状
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states
# 从 transformers.models.segformer.modeling_segformer.SegformerMixFFN 复制代码并将 Segformer 改为 GLPN
class GLPNMixFFN(nn.Module):
    def __init__(self, config, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        # 创建一个全连接层，输入特征为 in_features，输出特征为 hidden_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        # 创建一个 GLPNDWConv 实例
        self.dwconv = GLPNDWConv(hidden_features)
        # 判断 config.hidden_act 是否为字符串，根据结果选择其中间激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个全连接层，输入特征为 hidden_features，输出特征为 out_features
        self.dense2 = nn.Linear(hidden_features, out_features)
        # 创建一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states, height, width):
        # 层数 1 前向传播
        hidden_states = self.dense1(hidden_states)
        # 使用 GLPNDWConv 实例进行前向传播
        hidden_states = self.dwconv(hidden_states, height, width)
        # 使用中间激活函数进行前向传播
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 使用丢弃层进行前向传播
        hidden_states = self.dropout(hidden_states)
        # 层数 2 前向传播
        hidden_states = self.dense2(hidden_states)
        # 再次使用丢弃层进行前向传播
        hidden_states = self.dropout(hidden_states)
        # 返回结果
        return hidden_states


# 从 transformers.models.segformer.modeling_segformer.SegformerLayer 复制代码并将 Segformer 改为 GLPN
class GLPNLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        # 创建一个层归一化层
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        # 创建一个 GLPNAttention 实例
        self.attention = GLPNAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        # 如果 drop_path 大于 0.0，则创建一个 GLPNDropPath 实例，否则创建一个 nn.Identity()
        self.drop_path = GLPNDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 创建一个层归一化层
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        # 计算 MLP 隐藏层大小
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # 创建一个 GLPNMixFFN 实例
        self.mlp = GLPNMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size)
    # 定义前向传播函数，接受隐藏状态、高度、宽度和输出关注权重作为输入参数
    def forward(self, hidden_states, height, width, output_attentions=False):
        # 使用注意力层处理输入的隐藏状态（在GLPN中，layernorm被应用在self-attention之前）
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # 使用layernorm处理隐藏状态
            height,
            width,
            output_attentions=output_attentions,
        )

        # 从self_attention_outputs中获取注意力输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加到outputs中

        # 第一个残差连接（带有随机深度）
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        # 使用MLP处理经过第一个残差连接后的隐藏状态
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # 第二个残差连接（带有随机深度）
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        # 将layer_output添加到outputs中
        outputs = (layer_output,) + outputs

        # 返回所有输出
        return outputs
class GLPNEncoder(nn.Module):
    # 定义 GLPNEncoder 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接受参数 config
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 保存传入的配置参数

        # stochastic depth decay rule
        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        # 补丁嵌入
        embeddings = []
        for i in range(config.num_encoder_blocks):
            # 循环遍历编码器块数量
            embeddings.append(
                GLPNOverlapPatchEmbeddings(
                    # 添加重叠路径嵌入
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    # 如果是第一个编码器块，通道数使用配置的通道数，否则使用上一个隐层大小
                    hidden_size=config.hidden_sizes[i],
                    # 隐藏层大小为配置中指定的大小
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)
        # 将嵌入模块列表转换为 nn.ModuleList 类型

        # Transformer blocks
        # 变压器块
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # 循环遍历编码器块数量
            # each block consists of layers
            # 每个块由层组成
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                # 循环遍历当前块的深度
                layers.append(
                    GLPNLayer(
                        # 添加 GLPNLayer （变压器层）
                        config,
                        hidden_size=config.hidden_sizes[i],
                        # 隐藏层大小为配置中指定的大小
                        num_attention_heads=config.num_attention_heads[i],
                        # 注意力头的数量为配置中指定的数量
                        drop_path=dpr[cur + j],
                        # drop_path 为随机深度衰减规则中的当前值
                        sequence_reduction_ratio=config.sr_ratios[i],
                        # 序列缩减比例为配置中指定的比例
                        mlp_ratio=config.mlp_ratios[i],
                        # mlp 比例为配置中指定的比例
                    )
                )
            blocks.append(nn.ModuleList(layers))
            # 将层列表转换为 nn.ModuleList 类型并添加到块列表中

        self.block = nn.ModuleList(blocks)
        # 将块列表转换为 nn.ModuleList 类型

        # Layer norms
        # 层标准化
        self.layer_norm = nn.ModuleList(
            # 将每个编码器块的隐层大小作为参数创建相应数量的层标准化模块，并转换为 nn.ModuleList 类型
            [nn.LayerNorm(config.hidden_sizes[i]) for i in range(config.num_encoder_blocks)]
        )

    def forward(
        self,
        pixel_values,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        # 定义前向传播函数及其参数
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # 计算批处理大小
        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # 获取补丁嵌入
            hidden_states, height, width = embedding_layer(hidden_states)
            # 通过块传递嵌入
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # 应用层归一化
            hidden_states = norm_layer(hidden_states)
            # 可选地重新整形为 (batch_size, num_channels, height, width)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义一个 GLPNPreTrainedModel 类，继承自 PreTrainedModel 类
class GLPNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定 GLPNConfig 配置类
    config_class = GLPNConfig
    # 指定基础模型前缀
    base_model_prefix = "glpn"
    # 指定主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 定义 _init_weights 方法来初始化模型的权重
    # 通过分析模块类型进行不同的初始化
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 稍微有些不同于 TF 版本，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化偏置为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，填充索引对应的权重设为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的偏置为零，权重为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 定义 GLPN_START_DOCSTRING 字符串，用于说明 GLPNModel 的用途和参数
GLPN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GLPNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 GLPN_INPUTS_DOCSTRING 字符串，用于说明 GLPNModel 的输入参数
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

# 使用 add_start_docstrings 装饰器添加 GLPNModel 类的注释，说明其为 GLPN encoder，输出原始隐藏状态，无顶部特定头
@add_start_docstrings(
    "The bare GLPN encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    GLPN_START_DOCSTRING,
)
# 定义 GLPNModel 类，继承自 GLPNPreTrainedModel 类
class GLPNModel(GLPNPreTrainedModel):
    # 从transformers.models.segformer.modeling_segformer.SegformerModel.__init__中复制过来，将Segformer替换为GLPN
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 分层Transformer编码器
        self.encoder = GLPNEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型的注意力头。
        heads_to_prune: 每层要剪枝的注意力头的字典 {层编号: 要在此层中剪枝的注意力头列表}
        详见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GLPN_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 从transformers.models.segformer.modeling_segformer.SegformerModel.forward中复制过来
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 编码器的前向传播
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class GLPNSelectiveFeatureFusion(nn.Module):
    """
    Selective Feature Fusion module, as explained in the [paper](https://arxiv.org/abs/2201.07436) (section 3.4). This
    module adaptively selects and integrates local and global features by attaining an attention map for each feature.
    """

    def __init__(self, in_channel=64):
        super().__init__()

        # 第一个卷积层，将输入通道数翻倍后进行卷积，输出通道数为输入通道数，使用3x3的卷积核，填充为1，保持输入输出尺寸一致
        self.convolutional_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel * 2), out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),  # 批量归一化层
            nn.ReLU(),  # ReLU激活函数
        )

        # 第二个卷积层，将输入通道数减半后进行卷积，输出通道数为输入通道数的一半，使用3x3的卷积核，填充为1，保持输入输出尺寸一致
        self.convolutional_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),  # 批量归一化层
            nn.ReLU(),  # ReLU激活函数
        )

        # 第三个卷积层，输入通道数为输入通道数的一半，输出通道数为2，使用3x3的卷积核，填充为1，保持输入输出尺寸一致
        self.convolutional_layer3 = nn.Conv2d(
            in_channels=int(in_channel / 2), out_channels=2, kernel_size=3, stride=1, padding=1
        )

        # Sigmoid激活函数，用于获取两通道的注意力图
        self.sigmoid = nn.Sigmoid()

    def forward(self, local_features, global_features):
        # 沿着通道维度拼接局部特征和全局特征
        features = torch.cat((local_features, global_features), dim=1)
        # 通过第一个卷积层
        features = self.convolutional_layer1(features)
        # 通过第二个卷积层
        features = self.convolutional_layer2(features)
        # 通过第三个卷积层，得到注意力图
        features = self.convolutional_layer3(features)
        # 应用Sigmoid激活函数获取两通道的注意力图
        attn = self.sigmoid(features)
        # 通过注意力图构建混合特征，对局部特征和全局特征加权求和
        hybrid_features = local_features * attn[:, 0, :, :].unsqueeze(1) + global_features * attn[
            :, 1, :, :
        ].unsqueeze(1)

        return hybrid_features


class GLPNDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 如果输入通道数等于输出通道数，则跳过卷积层，否则使用1x1的卷积层进行通道数调整
        should_skip = in_channels == out_channels
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1) if not should_skip else nn.Identity()
        # 特征融合模块
        self.fusion = GLPNSelectiveFeatureFusion(out_channels)
        # 上采样层，使用双线性插值进行上采样，尺寸放大为原来的两倍
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, hidden_state, residual=None):
        # 通过卷积层进行特征调整
        hidden_state = self.convolution(hidden_state)
        # 如果有残差特征，则与当前特征进行融合
        if residual is not None:
            hidden_state = self.fusion(hidden_state, residual)
        # 上采样特征
        hidden_state = self.upsample(hidden_state)

        return hidden_state

        # 无条件返回上采样后的特征，下面的代码将不会执行
        hidden_state = self.upsample(hidden_state)
        return hidden_state


class GLPNDecoder(nn.Module):
    # 定义一个类，继承自nn.Module，用于实现GLPNDecoder的功能
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 将hidden_sizes列表中的元素反转，得到一个新的列表
        reserved_hidden_sizes = config.hidden_sizes[::-1]
        # 从config中获取decoder_hidden_size，然后将它赋值给out_channels
        out_channels = config.decoder_hidden_size
    
        # 创建一个ModuleList，其中的元素是GLPNDecoderStage的实例，其hidden_size从reserved_hidden_sizes列表中获取
        self.stages = nn.ModuleList(
            [GLPNDecoderStage(hidden_size, out_channels) for hidden_size in reserved_hidden_sizes]
        )
        # 在第一个stage中不进行融合
        self.stages[0].fusion = None
    
        # 创建一个将输入数据的尺寸放大两倍的Upsample层
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    
    # 定义forward函数，用于完成前向计算
    # 输入参数hidden_states是一个torch.Tensor类型的列表，返回值是一个torch.Tensor类型的列表
    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        # stage_hidden_states是一个空列表
        stage_hidden_states = []
        # stage_hidden_state是一个None类型的变量
        stage_hidden_state = None
        # 对hidden_states和self.stages进行zip迭代
        for hidden_state, stage in zip(hidden_states[::-1], self.stages):
            # 调用stage的__call__方法，传入hidden_state和stage_hidden_state作为参数，得到stage_hidden_state
            stage_hidden_state = stage(hidden_state, stage_hidden_state)
            # 将stage_hidden_state添加到stage_hidden_states列表中
            stage_hidden_states.append(stage_hidden_state)
    
        # 将stage_hidden_states列表中的最后一个元素通过self.final_upsample放大两倍
        stage_hidden_states[-1] = self.final_upsample(stage_hidden_state)
    
        # 返回stage_hidden_states列表
        return stage_hidden_states
class SiLogLoss(nn.Module):
    r"""
    实现了缩放不变对数刻度损失 [Eigen et al., 2014](https://arxiv.org/abs/1406.2283)。

    $$L=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{2 n^{2}}\left(\sum_{i} d_{i}^{2}\right)$$ 其中 $d_{i}=\log y_{i}-\log
    y_{i}^{*}$.

    """

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()  # 创建有效值掩码
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])  # 计算对数误差
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))  # 计算损失

        return loss


class GLPNDepthEstimationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        channels = config.decoder_hidden_size
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),  # 二维卷积层
            nn.ReLU(inplace=False),  # ReLU激活函数
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1),  # 二维卷积层
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # 使用解码器的最后特征
        hidden_states = hidden_states[self.config.head_in_index]

        hidden_states = self.head(hidden_states)  # 通过头部网络

        predicted_depth = torch.sigmoid(hidden_states) * self.config.max_depth  # 预测深度
        predicted_depth = predicted_depth.squeeze(dim=1)  # 压缩维度

        return predicted_depth


@add_start_docstrings(
    """GLPN Model transformer with a lightweight depth estimation head on top e.g. for KITTI, NYUv2.""",
    GLPN_START_DOCSTRING,
)
class GLPNForDepthEstimation(GLPNPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.glpn = GLPNModel(config)  # GLPN模型
        self.decoder = GLPNDecoder(config)  # GLPN解码器
        self.head = GLPNDepthEstimationHead(config)  # 深度估计头部网络

        # 初始化权重并应用最终处理
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
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, height, width)`, *optional`):
            Ground truth depth estimation maps for computing the loss.
        
        Returns:
            返回模型输出的深度估计结果、隐藏状态等信息
        
        Examples:
            示例代码，用于演示如何使用该函数进行深度估计
        
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
        ```"""
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # 调用GLPN模型，传入像素值，设置是否输出注意力权重、隐藏状态，以及是否使用返回字典
        outputs = self.glpn(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )
        
        # 根据是否使用返回字典，决定如何获取隐藏状态
        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        
        # 使用解码器和头部网络处理隐藏状态，得到深度估计结果
        out = self.decoder(hidden_states)
        predicted_depth = self.head(out)
        
        # 计算损失函数
        loss = None
        if labels is not None:
            loss_fct = SiLogLoss()
            loss = loss_fct(predicted_depth, labels)
        
        # 根据是否使用返回字典和是否输出隐藏状态，决定返回的结果
        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 返回深度估计输出对象
        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```