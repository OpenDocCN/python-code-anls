# `.\transformers\models\poolformer\modeling_poolformer.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 Sea AI Lab 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
""" PyTorch PoolFormer model."""

# 导入必要的库
import collections.abc
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射
from ...activations import ACT2FN
# 导入模型输出类
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入日志记录工具
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 PoolFormer 配置类
from .configuration_poolformer import PoolFormerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "PoolFormerConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "sail/poolformer_s12"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 7, 7]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "sail/poolformer_s12"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sail/poolformer_s12",
    # 查看所有 PoolFormer 模型，请访问 https://huggingface.co/models?filter=poolformer
]

# 从 transformers.models.beit.modeling_beit.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    为每个样本丢弃路径（在残差块的主路径中应用时的随机深度）。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    但原始名称具有误导性，因为“Drop Connect”是另一篇论文中的不同形式的 dropout...
    请参阅讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择将
    层和参数名称更改为“drop path”，而不是将 DropConnect 作为层名称并使用“survival rate”作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output

# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制的类，将 Beit->PoolFormer
class PoolFormerDropPath(nn.Module):
    # 定义一个类，用于在残差块的主路径中对每个样本进行路径丢弃（随机深度）
    class DropPath(nn.Module):
        
        # 初始化方法，设置丢弃概率
        def __init__(self, drop_prob: Optional[float] = None) -> None:
            super().__init__()
            self.drop_prob = drop_prob
        
        # 前向传播方法，对隐藏状态进行路径丢弃操作
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return drop_path(hidden_states, self.drop_prob, self.training)
        
        # 返回额外的表示信息，包括丢弃概率
        def extra_repr(self) -> str:
            return "p={}".format(self.drop_prob)
class PoolFormerEmbeddings(nn.Module):
    """
    构建 Patch Embeddings。
    """

    def __init__(self, hidden_size, num_channels, patch_size, stride, padding, norm_layer=None):
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        # 使用卷积层将输入通道数转换为隐藏层大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=padding)
        # 使用规范化层对隐藏层进行规范化
        self.norm = norm_layer(hidden_size) if norm_layer else nn.Identity()

    def forward(self, pixel_values):
        # 将输入像素值转换为嵌入向量
        embeddings = self.projection(pixel_values)
        # 对嵌入向量进行规范化
        embeddings = self.norm(embeddings)
        return embeddings


class PoolFormerGroupNorm(nn.GroupNorm):
    """
    具有1个组的组归一化。输入：形状为[B，C，H，W]的张量
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PoolFormerPooling(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        # 使用平均池化层对隐藏状态进行池化
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, hidden_states):
        return self.pool(hidden_states) - hidden_states


class PoolFormerOutput(nn.Module):
    def __init__(self, config, dropout_prob, hidden_size, intermediate_size):
        super().__init__()
        # 使用卷积层将隐藏层大小转换为中间大小
        self.conv1 = nn.Conv2d(hidden_size, intermediate_size, 1)
        self.conv2 = nn.Conv2d(intermediate_size, hidden_size, 1)
        self.drop = PoolFormerDropPath(dropout_prob)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.drop(hidden_states)

        return hidden_states


class PoolFormerLayer(nn.Module):
    """这对应于原始实现中的'PoolFormerBlock'类。"""
    # 初始化 PoolFormer 类，设置各个模块和参数
    def __init__(self, config, num_channels, pool_size, hidden_size, intermediate_size, drop_path):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 PoolFormerPooling 模块
        self.pooling = PoolFormerPooling(pool_size)
        # 初始化 PoolFormerOutput 模块
        self.output = PoolFormerOutput(config, drop_path, hidden_size, intermediate_size)
        # 初始化 PoolFormerGroupNorm 模块，用于归一化
        self.before_norm = PoolFormerGroupNorm(num_channels)
        self.after_norm = PoolFormerGroupNorm(num_channels)

        # 根据是否使用 drop path 初始化 PoolFormerDropPath 模块
        self.drop_path = PoolFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 根据配置决定是否使用层标准化
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            # 初始化层标准化参数
            self.layer_scale_1 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((num_channels)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                config.layer_scale_init_value * torch.ones((num_channels)), requires_grad=True
            )

    # 前向传播函数
    def forward(self, hidden_states):
        # 如果使用层标准化
        if self.use_layer_scale:
            # 进行池化操作
            pooling_output = self.pooling(self.before_norm(hidden_states))
            # 对池化输出进行缩放
            scaled_op = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * pooling_output
            # 第一个残差连接
            hidden_states = hidden_states + self.drop_path(scaled_op)
            outputs = ()

            # 在 PoolFormerOutput 模块中进行处理
            layer_output = self.output(self.after_norm(hidden_states))
            scaled_op = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * layer_output
            # 第二个残差连接
            output = hidden_states + self.drop_path(scaled_op)

            outputs = (output,) + outputs
            return outputs

        else:
            # 如果不使用层标准化
            pooling_output = self.drop_path(self.pooling(self.before_norm(hidden_states)))
            # 第一个残差连接
            hidden_states = pooling_output + hidden_states
            outputs = ()

            # 在 PoolFormerOutput 模块中进行处理，第二个残差连接
            layer_output = self.drop_path(self.output(self.after_norm(hidden_states)))
            output = hidden_states + layer_output

            outputs = (output,) + outputs
            return outputs
# PoolFormerEncoder 类定义，继承自 nn.Module
class PoolFormerEncoder(nn.Module):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置保存到对象中
        self.config = config
        # 根据 dropout 路径衰减率生成列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # 定义嵌入层列表
        embeddings = []
        # 遍历每个编码器块
        for i in range(config.num_encoder_blocks):
            # 创建 PoolFormerEmbeddings 实例，并添加到列表中
            embeddings.append(
                PoolFormerEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    padding=config.padding[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                )
            )
        # 将嵌入层列表转换为模块列表
        self.patch_embeddings = nn.ModuleList(embeddings)

        # 定义 Transformer 块列表
        blocks = []
        cur = 0
        # 遍历每个编码器块
        for i in range(config.num_encoder_blocks):
            # 每个块包含多个层
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            # 遍历每个深度
            for j in range(config.depths[i]):
                # 创建 PoolFormerLayer 实例，并添加到层列表中
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
            # 将层列表转换为模块列表，并添加到块列表中
            blocks.append(nn.ModuleList(layers))

        # 将块列表转换为模块列表
        self.block = nn.ModuleList(blocks)

    # 前向传播函数
    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        # 如果需要输出隐藏状态，则初始化所有隐藏状态为空元组
        all_hidden_states = () if output_hidden_states else None

        # 初始化隐藏状态为输入像素值
        hidden_states = pixel_values
        # 遍历嵌入层和块
        for idx, layers in enumerate(zip(self.patch_embeddings, self.block)):
            embedding_layer, block_layer = layers
            # 获取嵌入层的嵌入
            hidden_states = embedding_layer(hidden_states)
            # 将嵌入通过块层处理
            for _, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states)
                hidden_states = layer_outputs[0]

            # 如果需要输出隐藏状态，则保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回所有隐藏状态
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回带有隐藏状态的 BaseModelOutputWithNoAttention 对象
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


# PoolFormerPreTrainedModel 类定义，继承自 PreTrainedModel
class PoolFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # PoolFormerConfig 用于配置模型
    config_class = PoolFormerConfig
    # 模型参数前缀
    base_model_prefix = "poolformer"
    # 主要输入名称
    main_input_name = "pixel_values"
```  
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 如果是线性层或者卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为全 1
            module.weight.data.fill_(1.0)

    # 定义一个前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用全连接层处理隐藏状态，得到输出
        output = self.dense(hidden_states)
        # 返回输出结果
        return output
# 使用装饰器添加模型文档字符串和样本代码文档字符串
@add_start_docstrings(
    """
    PoolFormer Model transformer with an image classification head on top
    """,
    POOLFORMER_START_DOCSTRING,
)
# 定义一个继承自PoolFormerPreTrainedModel的类
class PoolFormerForImageClassification(PoolFormerPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置self.num_labels为config中的num_labels
        self.num_labels = config.num_labels
        # 创建PoolFormerModel实例并赋值给self.poolformer
        self.poolformer = PoolFormerModel(config)

        # Final norm
        # 创建PoolFormerGroupNorm实例并赋值给self.norm
        self.norm = PoolFormerGroupNorm(config.hidden_sizes[-1])
        # Classifier head
        # 如果config.num_labels大于0，则创建一个全连接层并赋值给self.classifier；否则创建一个恒等映射并赋值给self.classifier
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        # 调用实例的post_init方法
        self.post_init()

    # 使用装饰器添加模型前向方法的文档字符串
    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    # 使用装饰器添加样本代码文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 模型前向方法
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        根据输入参数返回一个元组或一个ImageClassifierOutputWithNoAttention对象
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.poolformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(self.norm(sequence_output).mean([-2, -1]))

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
            output = (logits,) + outputs[2:]
            返回包含logits和outputs的元组，如果loss不为None，则加入到元组中
            return ((loss,) + output) if loss is not None else output

        返回一个ImageClassifierOutputWithNoAttention对象，其中包含loss、logits和hidden_states
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```