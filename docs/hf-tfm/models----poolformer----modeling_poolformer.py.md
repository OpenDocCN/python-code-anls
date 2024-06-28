# `.\models\poolformer\modeling_poolformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和所有权信息
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权使用此文件；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证了解特定语言下的权限和限制。
""" PyTorch PoolFormer model."""

# 导入必要的库
import collections.abc
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入活化函数映射
from ...activations import ACT2FN
# 导入模型输出类
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
# 导入预训练模型类
from ...modeling_utils import PreTrainedModel
# 导入工具函数和日志记录
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 PoolFormer 配置类
from .configuration_poolformer import PoolFormerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "PoolFormerConfig"

# 用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "sail/poolformer_s12"
# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 512, 7, 7]

# 图像分类检查点名称
_IMAGE_CLASS_CHECKPOINT = "sail/poolformer_s12"
# 预期的图像分类输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sail/poolformer_s12",
    # 查看所有 PoolFormer 模型 https://huggingface.co/models?filter=poolformer
]


# 从 transformers.models.beit.modeling_beit.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    按样本丢弃路径（随机深度）（在残差块的主路径中应用时）。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    但原始名称具有误导性，因为“Drop Connect”是另一篇论文中不同形式的丢弃...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    我选择改变层和参数名称为“drop path”，而不是将 DropConnect 作为层名称并使用“生存率”作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制的类，并将 Beit 改为 PoolFormer
class PoolFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    # 初始化方法，用于设置实例的初始状态
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.drop_prob = drop_prob  # 初始化实例变量 drop_prob，用于存储丢弃概率

    # 前向传播方法，接收隐藏状态作为输入，返回经过丢弃路径处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 返回该层的额外信息的字符串表示，这里返回丢弃概率的字符串形式
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)
class PoolFormerEmbeddings(nn.Module):
    """
    Construct Patch Embeddings.
    """

    def __init__(self, hidden_size, num_channels, patch_size, stride, padding, norm_layer=None):
        super().__init__()
        # 将 patch_size、stride 和 padding 转换为可迭代对象，如果它们不是的话
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        # 使用卷积层进行投影，将输入的图像通道数转换为隐藏大小的特征图
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=padding)
        # 根据提供的规范化层或者使用身份映射作为默认规范化方法
        self.norm = norm_layer(hidden_size) if norm_layer else nn.Identity()

    def forward(self, pixel_values):
        # 对输入的像素值进行投影处理，得到嵌入表示
        embeddings = self.projection(pixel_values)
        # 对投影后的特征图进行规范化处理
        embeddings = self.norm(embeddings)
        return embeddings


class PoolFormerGroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group. Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PoolFormerPooling(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        # 使用平均池化层进行特征图的平均池化操作
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行池化操作，并返回池化结果减去原始隐藏状态的值
        return self.pool(hidden_states) - hidden_states


class PoolFormerOutput(nn.Module):
    def __init__(self, config, dropout_prob, hidden_size, intermediate_size):
        super().__init__()
        # 使用卷积层将隐藏大小的特征图转换为中间大小的特征图
        self.conv1 = nn.Conv2d(hidden_size, intermediate_size, 1)
        # 使用卷积层将中间大小的特征图转换为隐藏大小的特征图
        self.conv2 = nn.Conv2d(intermediate_size, hidden_size, 1)
        # 使用 PoolFormerDropPath 类来执行丢弃路径(drop path)操作，其中 dropout_prob 是丢弃概率
        self.drop = PoolFormerDropPath(dropout_prob)
        # 根据配置选择相应的激活函数，存储到 self.act_fn 中
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states):
        # 使用第一个卷积层处理隐藏状态
        hidden_states = self.conv1(hidden_states)
        # 应用选择的激活函数
        hidden_states = self.act_fn(hidden_states)
        # 应用丢弃路径操作
        hidden_states = self.drop(hidden_states)
        # 使用第二个卷积层处理更新后的隐藏状态
        hidden_states = self.conv2(hidden_states)
        # 再次应用丢弃路径操作
        hidden_states = self.drop(hidden_states)

        return hidden_states


class PoolFormerLayer(nn.Module):
    """This corresponds to the 'PoolFormerBlock' class in the original implementation."""
    # 初始化函数，用于初始化 PoolFormer 类的实例
    def __init__(self, config, num_channels, pool_size, hidden_size, intermediate_size, drop_path):
        super().__init__()
        # 初始化池化层对象
        self.pooling = PoolFormerPooling(pool_size)
        # 初始化输出层对象
        self.output = PoolFormerOutput(config, drop_path, hidden_size, intermediate_size)
        # 初始化归一化层对象（前）
        self.before_norm = PoolFormerGroupNorm(num_channels)
        # 初始化归一化层对象（后）
        self.after_norm = PoolFormerGroupNorm(num_channels)

        # 根据 drop_path 的值初始化 DropPath 层对象或者使用恒等映射（Identity）
        self.drop_path = PoolFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 是否使用层尺度缩放
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            # 初始化第一层尺度参数
            self.layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((num_channels)), requires_grad=True
            )
            # 初始化第二层尺度参数
            self.layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((num_channels)), requires_grad=True
            )

    # 前向传播函数，处理输入的 hidden_states，并返回处理后的 outputs
    def forward(self, hidden_states):
        # 如果使用层尺度缩放
        if self.use_layer_scale:
            # 执行池化操作，再进行归一化和尺度缩放
            pooling_output = self.pooling(self.before_norm(hidden_states))
            scaled_op = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * pooling_output
            # 第一个残差连接
            hidden_states = hidden_states + self.drop_path(scaled_op)
            outputs = ()

            # 执行输出层操作，再进行归一化和尺度缩放
            layer_output = self.output(self.after_norm(hidden_states))
            scaled_op = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * layer_output
            # 第二个残差连接
            output = hidden_states + self.drop_path(scaled_op)

            outputs = (output,) + outputs
            return outputs

        else:
            # 如果不使用层尺度缩放，执行池化、归一化、DropPath，再进行残差连接
            pooling_output = self.drop_path(self.pooling(self.before_norm(hidden_states)))
            # 第一个残差连接
            hidden_states = pooling_output + hidden_states
            outputs = ()

            # 在 PoolFormerOutput 块内部执行第二个残差连接
            layer_output = self.drop_path(self.output(self.after_norm(hidden_states)))
            output = hidden_states + layer_output

            outputs = (output,) + outputs
            return outputs
class PoolFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        # 生成随机深度衰减规则，根据config.drop_path_rate生成一个线性间隔的衰减率列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                PoolFormerEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    padding=config.padding[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    PoolFormerLayer(
                        config,
                        num_channels=config.hidden_sizes[i],
                        pool_size=config.pool_size,
                        hidden_size=config.hidden_sizes[i],
                        intermediate_size=int(config.hidden_sizes[i] * config.mlp_ratio),
                        drop_path=dpr[cur + j],
                    )
                )
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        hidden_states = pixel_values
        for idx, layers in enumerate(zip(self.patch_embeddings, self.block)):
            embedding_layer, block_layer = layers
            # Get patch embeddings from hidden_states
            # 从隐藏状态中获取补丁嵌入
            hidden_states = embedding_layer(hidden_states)
            # Send the embeddings through the blocks
            # 将嵌入通过Transformer块
            for _, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states)
                hidden_states = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回带有或不带注意力的基础模型输出
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class PoolFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PoolFormerConfig
    base_model_prefix = "poolformer"
    main_input_name = "pixel_values"
    def _init_weights(self, module):
        """Initialize the weights"""
        # 检查模块类型是否为线性层或二维卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对权重进行正态分布初始化，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块类型为 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
# POOLFORMER_START_DOCSTRING 常量，包含 PoolFormerModel 的文档字符串，描述模型作为 PyTorch Module 的用法和配置参数的说明
POOLFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`PoolFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# POOLFORMER_INPUTS_DOCSTRING 常量，包含 PoolFormerModel 的输入文档字符串，描述输入参数 pixel_values 的格式和用途
POOLFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`PoolFormerImageProcessor.__call__`] for details.
"""

# 使用装饰器 @add_start_docstrings，为 PoolFormerModel 类添加文档字符串，描述模型输出原始隐藏状态的特性和配置参数
@add_start_docstrings(
    "The bare PoolFormer Model transformer outputting raw hidden-states without any specific head on top.",
    POOLFORMER_START_DOCSTRING,
)
class PoolFormerModel(PoolFormerPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数并初始化配置
        super().__init__(config)
        self.config = config

        # 初始化编码器部分
        self.encoder = PoolFormerEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入嵌入的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 使用装饰器 @add_start_docstrings_to_model_forward 和 @add_code_sample_docstrings，为 forward 方法添加文档字符串
    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        # 如果 output_hidden_states 和 return_dict 为 None，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将输入传递给编码器，获取编码器的输出
        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            return (sequence_output, None) + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回 BaseModelOutputWithNoAttention 类的对象
        return BaseModelOutputWithNoAttention(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
        )


# PoolFormerFinalPooler 类的定义，继承自 nn.Module
class PoolFormerFinalPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    # 定义一个类方法 `forward`，用于前向传播
    def forward(self, hidden_states):
        # 将输入的隐藏状态 `hidden_states` 输入全连接层 `self.dense` 中进行处理
        output = self.dense(hidden_states)
        # 返回处理后的输出结果 `output`
        return output
# 使用自定义的文档字符串描述 PoolFormerForImageClassification 类，说明它是在 PoolFormerPreTrainedModel 基础上添加了图像分类头的变换器模型
@add_start_docstrings(
    """
    PoolFormer Model transformer with an image classification head on top
    """,
    POOLFORMER_START_DOCSTRING,
)
class PoolFormerForImageClassification(PoolFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.poolformer = PoolFormerModel(config)

        # Final norm
        # 使用 PoolFormerGroupNorm 类对模型最后一层的隐藏表示进行归一化处理
        self.norm = PoolFormerGroupNorm(config.hidden_sizes[-1])
        
        # Classifier head
        # 根据配置决定使用线性分类器或者恒等映射来定义分类头
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        # 调用 post_init 方法来初始化权重并进行最终的处理
        self.post_init()

    # 使用自定义的文档字符串描述 forward 方法的输入和输出，包括输入文档、代码示例和预期输出
    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 其他未列出的参数将由父类处理
        ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 poolformer 进行图像特征提取，可以选择是否返回隐藏状态，根据 return_dict 的设置
        outputs = self.poolformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出的序列特征
        sequence_output = outputs[0]

        # 对序列特征进行归一化，并计算均值，然后通过分类器得到 logits
        logits = self.classifier(self.norm(sequence_output).mean([-2, -1]))

        # 初始化损失值为 None
        loss = None
        # 如果 labels 不为 None，则计算损失函数
        if labels is not None:
            # 根据问题类型动态确定 self.config.problem_type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据不同的问题类型选择不同的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归任务，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典格式的输出，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则创建 ImageClassifierOutputWithNoAttention 对象并返回
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```