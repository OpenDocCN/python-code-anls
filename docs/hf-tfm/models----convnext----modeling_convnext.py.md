# `.\models\convnext\modeling_convnext.py`

```py
# 设置文件编码为 UTF-8
# 版权信息
# 版权声明 
# 您可能不会使用此文件，除非遵守许可证进行使用
# 您可以获取许可证的副本，在http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，软件分发将根据“原样”基础上进行，没有任何形式的担保或条件，无论是明示的还是隐含的，
#见许可证的具体语言，管理权限和限制

# 导入依赖库
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
from .configuration_convnext import ConvNextConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档描述的常量
_CONFIG_FOR_DOC = "ConvNextConfig"
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]
_IMAGE_CLASS_CHECKPOINT = "facebook/convnext-tiny-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"
CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnext-tiny-224",
    # 查看所有ConvNext模型，请访问https://huggingface.co/models?filter=convnext
]

# 复制自transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    为每个样本丢弃路径（随机深度）（当应用于残差块的主路径时）。
    
    Ross Wightman的评论：这与我为EfficientNet等网络创建的DropConnect实现相同，
    但是，原始名称是误导性的，因为“Drop Connect”是另一篇论文中不同形式的辍学... 
    请参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 
    我选择将层和参数名称更改为“drop path”，而不是将DropConnect作为层名称并使用“幸存率”作为参数。
    """
    如果 drop_prob == 0.0 或者不处于训练状态，则返回输入
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，不仅仅是2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    # 根据输入张量和dropout保留率计算输出张量
    output = input.div(keep_prob) * random_tensor
    # 返回输出张量
    return output
# 从transformers.models.beit.modeling_beit.BeitDropPath复制并将Beit->ConvNext
class ConvNextDropPath(nn.Module):
    """针对每个样本丢失路径（随机深度）（当应用于残差块的主路径中）"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class ConvNextLayerNorm(nn.Module):
    r"""支持两种数据格式的LayerNorm：channels_last（默认）或channels_first。
    输入维度的顺序。channels_last对应于形状为（batch_size，height，width，channels）的输入，而channels_first对应于形状为（batch_size，channels，height，width）的输入。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNextEmbeddings(nn.Module):
    """这个类与src/transformers/models/swin/modeling_swin.py中找到的SwinEmbeddings类类似（受到启发）。"""

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels
    # 前向传播函数，接受像素值张量并返回张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取像素值的通道数
        num_channels = pixel_values.shape[1]
        # 如果通道数不等于模型设定的通道数，则抛出数值错误
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 使用 patch_embeddings 函数获取嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 对嵌入向量进行 layernorm 处理
        embeddings = self.layernorm(embeddings)
        # 返回处理后的嵌入向量
        return embeddings
# 定义一个新的 PyTorch 模块，表示 ConvNext 层
class ConvNextLayer(nn.Module):
    """对应原始实现中的 `Block` 类。
    
    有两种等效的实现方式：[DwConv, LayerNorm (channels_first), Conv, GELU, 1x1 Conv]；所有操作都在 (N, C, H, W) 维度进行
    (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]；再转换回来

    作者使用了 (2)，因为他们发现它在 PyTorch 中稍微快一点。
    
    Args:
        config ([`ConvNextConfig`]): 模型配置类。
        dim (`int`): 输入通道的数量。
        drop_path (`float`): 随机深度率，默认值为 0.0。
    """

    # 构造函数，接受配置参数、输入通道数和随机深度率
    def __init__(self, config, dim, drop_path=0):
        # 初始化父类
        super().__init__()
        # 深度卷积，维度保持不变
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # 定义卷积层的 LayerNorm（按通道进行归一化）
        self.layernorm = ConvNextLayerNorm(dim, eps=1e-6)
        # 点卷积/1x1 卷积，由线性层实现
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 定义激活函数
        self.act = ACT2FN[config.hidden_act]
        # 第二个点卷积
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # 可学习的层级缩放参数
        self.layer_scale_parameter = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True) 
            if config.layer_scale_init_value > 0
            else None
        )
        # 随机深度路径
        self.drop_path = ConvNextDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # 前向传播方法
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 保留输入，以便在后面进行跳跃连接
        input = hidden_states
        # 深度卷积操作
        x = self.dwconv(hidden_states)
        # 转换维度顺序 (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # 进行 LayerNorm 操作
        x = self.layernorm(x)
        # 第一个点卷积
        x = self.pwconv1(x)
        # 激活函数应用
        x = self.act(x)
        # 第二个点卷积
        x = self.pwconv2(x)
        # 如果存在缩放参数，进行缩放
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        # 转换维度顺序 (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # 应用随机深度路径并添加跳跃连接
        x = input + self.drop_path(x)
        # 返回输出结果
        return x


# 定义 ConvNeXT 阶段，包含可选的下采样层和多个残差块
class ConvNextStage(nn.Module):
    """ConvNeXT 阶段，包含一个可选的下采样层和多个残差块。
    
    Args:
        config ([`ConvNextConfig`]): 模型配置类。
        in_channels (`int`): 输入通道的数量。
        out_channels (`int`): 输出通道的数量。
        depth (`int`): 残差块的数量。
        drop_path_rates(`List[float]`): 每个层的随机深度率。
    """
    # 定义一个类，用于实现具有下采样和多个卷积层的神经网络模型
    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()  # 调用父类的初始化方法
    
        # 如果输入通道数不等于输出通道数，或者步长大于1
        if in_channels != out_channels or stride > 1:
            # 定义下采样层，使用序列容器封装了一系列操作
            self.downsampling_layer = nn.Sequential(
                ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),  # 添加归一化层
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),  # 添加卷积层
            )
        else:
            # 如果输入通道数等于输出通道数且步长为1，则使用恒等映射
            self.downsampling_layer = nn.Identity()  # 创建恒等映射层
        # 如果未提供丢弃路径率，则将其设置为与深度相匹配的零列表
        drop_path_rates = drop_path_rates or [0.0] * depth
        # 使用序列容器创建多个卷积层
        self.layers = nn.Sequential(
            *[ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )
    
    # 定义前向传播方法，接受一个浮点数张量作为输入，并返回一个张量
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 对输入的隐藏状态进行下采样
        hidden_states = self.downsampling_layer(hidden_states)
        # 通过多个卷积层处理隐藏状态
        hidden_states = self.layers(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建包含多个子模块的列表
        self.stages = nn.ModuleList()
        # 计算每一层的 drop_path_rate
        drop_path_rates = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        # 初始化前一层的通道数
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            # 设置当前层输出通道数
            out_chs = config.hidden_sizes[i]
            # 创建 ConvNextStage 实例作为当前层的子模块
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            # 将当前层的实例添加到列表中
            self.stages.append(stage)
            # 更新前一层的通道数
            prev_chs = out_chs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        # 如果需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每一层，实现前向传播
        for i, layer_module in enumerate(self.stages):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 更新隐藏状态
            hidden_states = layer_module(hidden_states)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则返回隐藏状态和隐藏状态列表
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回包含隐藏状态和隐藏状态列表的 BaseModelOutputWithNoAttention 实例
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class ConvNextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类
    config_class = ConvNextConfig
    # 设置基础模型前缀
    base_model_prefix = "convnext"
    # 设置主输入名称
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化线性层和卷积层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 初始化偏置
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的参数
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CONVNEXT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    # 参数：
    # config ([`ConvNextConfig`]): 模型配置类，包含模型的所有参数。
    # 用配置文件初始化不会加载与模型相关的权重，只是加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
# 定义一个文档字符串，解释 ConvNextModel 类的输入参数
CONVNEXT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用 [`AutoImageProcessor`] 获取像素值。详情请参考 [`ConvNextImageProcessor.__call__`]。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详情，请参考返回的张量中的`hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`]，而不是简单的元组。
"""

# 为 ConvNextModel 类添加文档字符串
@add_start_docstrings(
    "The bare ConvNext model outputting raw features without any specific head on top.",
    CONVNEXT_START_DOCSTRING,
)
class ConvNextModel(ConvNextPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 创建 ConvNextEmbeddings 和 ConvNextEncoder 实例
        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)

        # 最终的 layernorm 层
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
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
        # 检查是否要返回隐藏状态和返回的类型
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有像素值，则引发 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 通过 embeddings 计算嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 使用 encoder 进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后的隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 全局平均池化，(N, C, H, W) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回 BaseModelOutputWithPoolingAndNoAttention 实例
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
    # 创建 ConvNext 模型，并在顶部添加了一个图像分类头（线性层在池化特征的顶部），例如用于ImageNet数据集。
        # 定义一个名为 ConvNextForImageClassification 的类，继承自 ConvNextPreTrainedModel
        class ConvNextForImageClassification(ConvNextPreTrainedModel):
            # 初始化方法，接受一个config参数
            def __init__(self, config):
                # 调用父类的初始化方法
                super().__init__(config)

                # 初始化num_labels属性
                self.num_labels = config.num_labels
                # 创建一个ConvNextModel对象
                self.convnext = ConvNextModel(config)

                # 分类器头部
                self.classifier = (
                    # 如果num_labels大于0，则创建一个线性层，否则创建一个恒等映射
                    nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
                )

                # 初始化权重并应用最终处理
                self.post_init()

            # 前向传播方法，接受一些输入参数
            @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取卷积输出
        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果返回字典则获取池化输出，否则获取第一个元素（在outputs里）
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 通过分类器获得预测输出
        logits = self.classifier(pooled_output)

        loss = None
        # 如果有标签
        if labels is not None:
            # 如果未指定问题类型，则根据标签数据类型和类别数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
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

        # 如果不返回字典，则包装输出并返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回图像分类器输出，包括损失、预测输出、隐藏状态
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
# 添加起始文档字符串，描述了 ConvNeXt 骨干网络的作用和使用情况
@add_start_docstrings(
    """
    ConvNeXt backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    CONVNEXT_START_DOCSTRING,
)
# 定义 ConvNeXtBackbone 类，继承自 ConvNextPreTrainedModel 和 BackboneMixin
class ConvNextBackbone(ConvNextPreTrainedModel, BackboneMixin):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 调用父类的 _init_backbone 函数
        super()._init_backbone(config)

        # 创建 ConvNextEmbeddings 对象
        self.embeddings = ConvNextEmbeddings(config)
        # 创建 ConvNextEncoder 对象
        self.encoder = ConvNextEncoder(config)
        # 初始化 num_features 列表
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # 创建 hidden_states_norms 字典，为每个 stage 添加 LayerNorm
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextLayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并进行最终处理
        self.post_init()

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:  # 返回类型为 BackboneOutput
        ) -> BackboneOutput:
        """
        返回：定义函数返回类型为BackboneOutput

        示例：给出函数的使用示例

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnext-tiny-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 定义return_dict变量，如果return_dict不为None，则赋值为return_dict，否则赋值为self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 定义output_hidden_states变量，如果output_hidden_states不为None，则赋值为output_hidden_states，否则赋值为self.config.output_hidden_states

        embedding_output = self.embeddings(pixel_values)
        # 对输入的像素值进行嵌入操作

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        # 使用编码器对嵌入输出进行编码，并指定是否输出隐藏状态，以及返回字典的选项

        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 如果return_dict为True，则隐藏状态为outputs中的隐藏状态，否则为outputs的第二个元素

        feature_maps = ()
        # 初始化特征图为空元组
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                # 如果当前阶段在输出特征中，对隐藏状态进行归一化处理
                feature_maps += (hidden_state,)
                # 将处理后的隐藏状态加入到特征图中

        if not return_dict:
            output = (feature_maps,)
            # 如果不需要返回字典，则输出只包含特征图
            if output_hidden_states:
                output += (hidden_states,)
            # 如果需要输出隐藏状态，则将隐藏状态也加入输出中
            return output
            # 返回结果

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )
        # 返回BackboneOutput对象，包括特征图、隐藏状态和注意力
```