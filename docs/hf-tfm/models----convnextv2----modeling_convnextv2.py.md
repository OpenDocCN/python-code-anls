# `.\models\convnextv2\modeling_convnextv2.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 2023 年 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”基础分发软件
# 没有任何种类的担保或条件，无论是明示还是暗示的。
# 有关授予权限和权利的具体语言，请参阅许可证
""" PyTorch ConvNextV2 模型。"""

# 导入所需的库
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_convnextv2 import ConvNextV2Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的通用变量
_CONFIG_FOR_DOC = "ConvNextV2Config"
_CHECKPOINT_FOR_DOC = "facebook/convnextv2-tiny-1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# 图像分类文档说明
_IMAGE_CLASS_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# ConvNextV2 预训练模型存档列表
CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnextv2-tiny-1k-224",
    # 查看所有 ConvNextV2 模型 https://huggingface.co/models?filter=convnextv2
]

# 从 transformers.models.beit.modeling_beit.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    每个样本都会删除路径（随机深度），当应用于残差块的主路径时。
    
    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    但是，原始名称是误导性的，因为“Drop Connect”是另一篇论文中不同形式的 dropout…
    有关讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 …
    我选择将图层和参数名称更改为“drop path”，而不是将 DropConnect 作为图层名称，
    并使用“生存率”作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅限于 2D 卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 对随机生成的张量进行向下取整操作，实现二值化
    random_tensor.floor_()
    # 将输入张量除以保留概率，再乘以二值化后的随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回输出张量
    return output
# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制代码，将 Beit 改为 ConvNextV2
class ConvNextV2DropPath(nn.Module):
    """每个样本的丢弃路径（随机深度），应用在残差块的主路径中。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        # 初始化丢弃概率
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数对隐藏状态进行处理
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class ConvNextV2GRN(nn.Module):
    """全局响应归一化（GRN）层"""

    def __init__(self, dim: int):
        super().__init__()
        # 初始化权重和偏置参数
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 计算并归一化全局空间特征图
        global_features = torch.norm(hidden_states, p=2, dim=(1, 2), keepdim=True)
        norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        # 对隐藏状态进行归一化处理，并应用权重和偏置
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states

        return hidden_states


# 从 transformers.models.convnext.modeling_convnext.ConvNextLayerNorm 复制代码，将 ConvNext 改为 ConvNextV2
class ConvNextV2LayerNorm(nn.Module):
    r"""支持两种数据格式的 LayerNorm：channels_last（默认）或 channels_first。
    输入维度的顺序。channels_last 对应于形状为 (batch_size, height, width, channels) 的输入，
    而 channels_first 对应于形状为 (batch_size, channels, height, width) 的输入。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 初始化权重和偏置参数、eps 和数据格式
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # 如果数据格式为 channels_last，则调用 torch.nn.functional.layer_norm
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果数据格式为 channels_first，则手动计算 LayerNorm
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# 从 transformers.models.convnext.modeling_convnext.ConvNextEmbeddings 复制代码，将 ConvNext 改为 ConvNextV2
class ConvNextV2Embeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        # 用给定的参数配置初始化 Conv2d 模块，用于图像块的嵌入
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        # 初始化归一化层，指定特征大小、epsilon值和数据格式
        self.layernorm = ConvNextV2LayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        # 获取通道数
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入的通道数
        num_channels = pixel_values.shape[1]
        # 如果输入通道数与设置的通道数不匹配，则引发异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入进行图像块嵌入
        embeddings = self.patch_embeddings(pixel_values)
        # 对嵌入结果进行归一化处理
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextV2Layer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        # 深度可分离卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 初始化归一化层，指定特征大小、epsilon值
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        # 点卷积/1x1卷积，使用线性层实现
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        # 应用深度可分离卷积
        x = self.dwconv(hidden_states)
        # 调整维度排列，从(N, C, H, W)到(N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # 对调整后的张量进行归一化处理
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # 再次调整维度排列，从(N, H, W, C)到(N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


# Copied from transformers.models.convnext.modeling_convnext.ConvNextStage with ConvNeXT->ConvNeXTV2, ConvNext->ConvNextV2
class ConvNextV2Stage(nn.Module):
    # ConvNeXTV2 阶段，包括一个可选的下采样层 + 多个残差块
    class ConvNeXTV2(nn.Module):
        # 构造函数，接受模型配置类、输入通道数、输出通道数、卷积核大小、步长、残差块数量、随机深度率列表作为参数
        def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
            super().__init__()
    
            # 如果输入通道数不等于输出通道数或步长大于1
            if in_channels != out_channels or stride > 1:
                # 创建下采样层，包括 ConvNextV2LayerNorm 和 2D 卷积层
                self.downsampling_layer = nn.Sequential(
                    ConvNextV2LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
                )
            # 否则，创建单位矩阵作为下采样层
            else:
                self.downsampling_layer = nn.Identity()
            
            # 如果没有提供随机深度率列表，则初始化为长度为 depth 的全零列表
            drop_path_rates = drop_path_rates or [0.0] * depth
            # 创建包含多个 ConvNextV2Layer 的层序列
            self.layers = nn.Sequential(
                *[ConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
            )
    
        # 前向传播函数，接受输入的张量，并返回处理后的张量
        def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
            # 对输入张量进行下采样处理
            hidden_states = self.downsampling_layer(hidden_states)
            # 通过多个 ConvNextV2Layer 处理输入张量
            hidden_states = self.layers(hidden_states)
            # 返回处理后的张量
            return hidden_states
# 从transformers.models.convnext.modeling_convnext.ConvNextEncoder复制代码，并将ConvNext->ConvNextV2
# 定义ConvNextV2Encoder类，继承自nn.Module
class ConvNextV2Encoder(nn.Module):
    # 初始化方法，接受config参数
    def __init__(self, config):
        super().__init__()
        # 初始化阶段列表
        self.stages = nn.ModuleList()
        # 计算每个阶段的drop path率
        drop_path_rates = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        # 循环构建每个阶段
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个阶段
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# 从transformers.models.convnext.modeling_convnext.ConvNextPreTrainedModel复制代码，并将ConvNext->ConvNextV2, convnext->convnextv2
# 定义ConvNextV2PreTrainedModel类，继承自PreTrainedModel
class ConvNextV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextV2Config
    base_model_prefix = "convnextv2"
    main_input_name = "pixel_values"

    # 初始化权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 检查模块是否是Linear或Conv2d类型并初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是LayerNorm类型的模块，初始化权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CONVNEXTV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
        # 将其作为常规的 PyTorch 模块使用，并参考 PyTorch 文档了解一切相关的一般用法和行为。

        # 参数:
        #     config ([`ConvNextV2Config`]): 模型配置类，包含模型的所有参数。
        #         使用配置文件初始化不会加载与模型关联的权重，只会加载配置信息。
        #         可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

CONVNEXTV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ConvNextImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ConvNextV2 model outputting raw features without any specific head on top.",
    CONVNEXTV2_START_DOCSTRING,
)
# Copied from transformers.models.convnext.modeling_convnext.ConvNextModel with CONVNEXT->CONVNEXTV2, ConvNext->ConvNextV2
class ConvNextV2Model(ConvNextV2PreTrainedModel):
    # 初始化方法，配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置参数
        self.config = config

        # 创建 ConvNextV2Embeddings 对象
        self.embeddings = ConvNextV2Embeddings(config)
        # 创建 ConvNextV2Encoder 对象
        self.encoder = ConvNextV2Encoder(config)

        # 定义最终的层归一化层
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 检查是否需要返回隐藏层状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否需要返回字典类型输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有给定像素值，则抛出异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 获取像素值的嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 获取编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # 全局平均池化，(N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 为 ConvNextV2 模型添加文档字符串，描述其是一个在顶部带有图像分类头的模型(即在池化特征之上的线性层)，例如用于 ImageNet
@add_start_docstrings(
    """
    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    CONVNEXTV2_START_DOCSTRING,
)
# 从transformers.models.convnext.modeling_convnext.ConvNextForImageClassification复制，修改为CONVNEXT->CONVNEXTV2,ConvNext->ConvNextV2,convnext->convnextv2
class ConvNextV2ForImageClassification(ConvNextV2PreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)

        # 初始化模型的标签数和ConvNextV2模型
        self.num_labels = config.num_labels
        self.convnextv2 = ConvNextV2Model(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的forward方法添加文档字符串
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
            self,
            pixel_values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，如果未提供，则使用模型配置的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 进行卷积和下采样操作
        outputs = self.convnextv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
    
        # 如果使用返回字典，则获取池化输出；否则，获取第一个返回值
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
    
        # 使用分类器对池化输出进行分类得到预测的 logits
        logits = self.classifier(pooled_output)
    
        loss = None
        # 如果提供了标签，则计算损失函数
        if labels is not None:
            # 根据标签的数据类型和类别数量，自动推断问题类型
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
    
        # 如果不使用返回字典，则将结果作为元组返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 如果使用返回字典，则将结果保存在 ImageClassifierOutputWithNoAttention 中并返回
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
# 添加起始文档字符串，说明这是 ConvNeXT V2 backbone，可用于像DETR和MaskFormer这样的框架
@add_start_docstrings(
    """
    ConvNeXT V2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    CONVNEXTV2_START_DOCSTRING,
)
# 从transformers.models.convnext.modeling_convnext.ConvNextBackbone复制而来，对于CONVNEXT->CONVNEXTV2,ConvNext->ConvNextV2,facebook/convnext-tiny-224->facebook/convnextv2-tiny-1k-224进行更改
class ConvNextV2Backbone(ConvNextV2PreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        # 初始化backbone
        super()._init_backbone(config)

        # 创建嵌入层和编码器
        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # 为out_features的隐藏状态添加层范数
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加前向传播函数的起始文档字符串
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    # 替换返回文档字符串中的输出类型和配置类
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
        """
        返回 BackboneOutput 对象

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnextv2-tiny-1k-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""

        # 确定是否需要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 获取图像的嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 使用编码器处理嵌入输出
        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # 如果需要返回字典，则获取隐藏状态
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # 初始化特征图集合
        feature_maps = ()
        # 遍历阶段和隐藏状态，获取特征图并规范化
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps += (hidden_state,)

        # 如果不需要返回字典，则构建输出
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (hidden_states,)
            return output

        # 返回 BackboneOutput 对象
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )
```