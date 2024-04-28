# `.\transformers\models\mobilenet_v1\modeling_mobilenet_v1.py`

```
# 导入所需的模块和类型定义
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mobilenet_v1 import MobileNetV1Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 设置文档字符串
_CONFIG_FOR_DOC = "MobileNetV1Config"
_CHECKPOINT_FOR_DOC = "google/mobilenet_v1_1.0_224"
_EXPECTED_OUTPUT_SHAPE = [1, 1024, 7, 7]
_IMAGE_CLASS_CHECKPOINT = "google/mobilenet_v1_1.0_224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilenet_v1_1.0_224",
    "google/mobilenet_v1_0.75_192",
    # See all MobileNetV1 models at https://huggingface.co/models?filter=mobilenet_v1
]

# 构建从 TensorFlow 到 PyTorch 的权重映射
def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    tf_to_pt_map = {}

    if isinstance(model, MobileNetV1ForImageClassification):
        backbone = model.mobilenet_v1
    else:
        backbone = model

    prefix = "MobilenetV1/Conv2d_0/"
    tf_to_pt_map[prefix + "weights"] = backbone.conv_stem.convolution.weight
    tf_to_pt_map[prefix + "BatchNorm/beta"] = backbone.conv_stem.normalization.bias
    tf_to_pt_map[prefix + "BatchNorm/gamma"] = backbone.conv_stem.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.normalization.running_var

    return tf_to_pt_map


本段代码定义了 MobileNetV1 模型的相关设置和实用函数。主要包括以下内容：

1. 导入必要的模块和类型定义。
2. 获取日志记录器。
3. 设置文档字符串,包括配置文件、检查点、期望输出形状等。
4. 预训练模型存档列表。
5. 定义一个函数 `_build_tf_to_pytorch_map`,用于构建从 TensorFlow 到 PyTorch 的权重映射。该函数会根据模型类型(分类或非分类)获取模型的主要部分,并映射 TensorFlow 权重到对应的 PyTorch 层。
    # 遍历范围为0到12的数字
    for i in range(13):
        # 计算tf_index和pt_index的值
        tf_index = i + 1
        pt_index = i * 2

        # 获取backbone层的指针
        pointer = backbone.layer[pt_index]
        # 构建prefix字符串
        prefix = f"MobilenetV1/Conv2d_{tf_index}_depthwise/"
        # 将深度可分离卷积的深度权重映射到tf_to_pt_map
        tf_to_pt_map[prefix + "depthwise_weights"] = pointer.convolution.weight
        # 将深度可分离卷积的偏差映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/beta"] = pointer.normalization.bias
        # 将深度可分离卷积的缩放参数映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = pointer.normalization.weight
        # 将深度可分离卷积的移动平均映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.normalization.running_mean
        # 将深度可分离卷积的移动方差映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.normalization.running_var

        # 获取backbone层的指针
        pointer = backbone.layer[pt_index + 1]
        # 构建prefix字符串
        prefix = f"MobilenetV1/Conv2d_{tf_index}_pointwise/"
        # 将逐点卷积的权重映射到tf_to_pt_map
        tf_to_pt_map[prefix + "weights"] = pointer.convolution.weight
        # 将逐点卷积的偏差映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/beta"] = pointer.normalization.bias
        # 将逐点卷积的缩放参数映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = pointer.normalization.weight
        # 将逐点卷积的移动平均映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.normalization.running_mean
        # 将逐点卷积的移动方差映射到tf_to_pt_map
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.normalization.running_var

    # 如果model是MobileNetV1ForImageClassification的实例
    if isinstance(model, MobileNetV1ForImageClassification):
        # 构建prefix字符串
        prefix = "MobilenetV1/Logits/Conv2d_1c_1x1/"
        # 将分类器的权重映射到tf_to_pt_map
        tf_to_pt_map[prefix + "weights"] = model.classifier.weight
        # 将分类器的偏差映射到tf_to_pt_map
        tf_to_pt_map[prefix + "biases"] = model.classifier.bias

    # 返回tf_to_pt_map
    return tf_to_pt_map
# 加载 TensorFlow checkpoints 到 PyTorch 模型中
def load_tf_weights_in_mobilenet_v1(model, config, tf_checkpoint_path):
    # 尝试导入 numpy 和 tensorflow 包
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，则记录错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 从 TF 模型中加载权重
    # 获取 TF 检查点中的所有变量
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    tf_weights = {}
    for name, shape in init_vars:
        # 记录要加载的 TF 权重名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载 TF 变量数据
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array

    # 构建 TF 到 PyTorch 权重加载映射
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    # 遍历映射，导入权重
    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        # 如果权重名称不在 TF 权重列表中，则跳过
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        # 获取权重数组
        array = tf_weights[name]

        # 如果权重名称包含 "depthwise_weights"，则进行转置
        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")
            array = np.transpose(array, (2, 3, 0, 1))
        # 如果权重名称包含 "weights"，也进行转置
        elif "weights" in name:
            logger.info("Transposing")
            # 根据指针形状和数组维度进行转置
            if len(pointer.shape) == 2:  # 将数据复制到线性层
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        # 如果指针形状和数组形状不匹配，则引发错误
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        # 初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name} {array.shape}")
        pointer.data = torch.from_numpy(array)

        # 移除已加载的权重
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)

    # 记录未复制到 PyTorch 模型的权重
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    return model


def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor:
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    # 获取输入特征的高度和宽度
    in_height, in_width = features.shape[-2:]
    # 获取卷积层的步长和卷积核大小
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size

    # 计算垂直方向和水平方向的填充量
    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    # 计算左右填充量
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    # 计算垂直方向上的顶部填充大小
    pad_top = pad_along_height // 2
    # 计算垂直方向上的底部填充大小
    pad_bottom = pad_along_height - pad_top

    # 设置填充边界
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    # 使用常数值 0.0 对输入特征进行填充操作
    return nn.functional.pad(features, padding, "constant", 0.0)
class MobileNetV1ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileNetV1Config,  # MobileNetV1 模型配置对象
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: int,  # 卷积核大小
        stride: Optional[int] = 1,  # 卷积步长，默认为1
        groups: Optional[int] = 1,  # 分组卷积参数，默认为1
        bias: bool = False,  # 是否使用偏置项，默认为False
        use_normalization: Optional[bool] = True,  # 是否使用归一化，默认为True
        use_activation: Optional[bool or str] = True,  # 是否使用激活函数，默认为True，或者可指定激活函数名称
    ) -> None:
        super().__init__()
        self.config = config  # 保存 MobileNetV1 模型配置对象

        # 检查输入通道数是否能够被分组数整除
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        # 检查输出通道数是否能够被分组数整除
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 根据是否使用 TensorFlow 风格的填充来设置填充大小
        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2)

        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        # 如果需要归一化，则创建 BatchNormalization 层对象
        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=config.layer_norm_eps,
                momentum=0.9997,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        # 如果需要激活函数，则创建激活函数对象
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]  # 根据指定名称获取对应的激活函数
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]  # 根据配置对象中的隐藏层激活函数名称获取对应的激活函数
            else:
                self.activation = config.hidden_act  # 使用配置对象中定义的激活函数
        else:
            self.activation = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.config.tf_padding:  # 如果使用 TensorFlow 风格的填充
            features = apply_tf_padding(features, self.convolution)  # 应用 TensorFlow 风格的填充
        features = self.convolution(features)  # 进行卷积操作
        if self.normalization is not None:  # 如果需要归一化
            features = self.normalization(features)  # 进行归一化操作
        if self.activation is not None:  # 如果需要激活函数
            features = self.activation(features)  # 应用激活函数
        return features  # 返回处理后的特征张量


class MobileNetV1PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileNetV1Config  # 指定模型配置类
    load_tf_weights = load_tf_weights_in_mobilenet_v1  # 加载 TensorFlow 权重的函数
    base_model_prefix = "mobilenet_v1"  # 基础模型前缀
    main_input_name = "pixel_values"  # 主输入名称
    supports_gradient_checkpointing = False  # 是否支持梯度检查点
    # 初始化神经网络模块的权重
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        # 如果模块是线性层或者卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是二维卷积层
        elif isinstance(module, nn.BatchNorm2d):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
# 这个文档字符串描述了 MobileNetV1 模型的基本信息
MOBILENET_V1_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileNetV1Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 这个文档字符串描述了 MobileNetV1 模型的输入参数
MOBILENET_V1_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileNetV1ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 这个类是 MobileNetV1 模型的基类，包含了基本的模型结构和功能
@add_start_docstrings(
    "The bare MobileNetV1 model outputting raw hidden-states without any specific head on top.",
    MOBILENET_V1_START_DOCSTRING,
)
class MobileNetV1Model(MobileNetV1PreTrainedModel):
    # 定义 MobileNetV1 模型的初始化函数
    def __init__(self, config: MobileNetV1Config, add_pooling_layer: bool = True):
        # 调用父类 PyTorch 模型的初始化方法
        super().__init__(config)
        # 保存配置信息
        self.config = config
    
        # 设置初始通道数
        depth = 32
        out_channels = max(int(depth * config.depth_multiplier), config.min_depth)
    
        # 创建卷积层, 设置输入通道数、输出通道数、核大小、步长
        self.conv_stem = MobileNetV1ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
        )
    
        # 定义各层的步长
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
    
        # 创建 MobileNetV1 的卷积层列表
        self.layer = nn.ModuleList()
        for i in range(13):
            in_channels = out_channels
    
            # 当步长为 2 或者是第一层时, 调整通道数
            if strides[i] == 2 or i == 0:
                depth *= 2
                out_channels = max(int(depth * config.depth_multiplier), config.min_depth)
    
            # 添加深度可分离卷积层
            self.layer.append(
                MobileNetV1ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=strides[i],
                    groups=in_channels,
                )
            )
    
            # 添加 1x1 卷积层
            self.layer.append(
                MobileNetV1ConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                )
            )
    
        # 如果需要, 添加全局平均池化层
        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None
    
        # 初始化权重并应用最终处理
        self.post_init()
    
    # 不支持裁剪头的操作
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError
    
    # 定义前向传播函数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 在这里实现模型的前向传播逻辑
        pass
    # 定义一个方法，接收输入，并返回元组或者BaseModelOutputWithPoolingAndNoAttention类型的值
    def forward(self, pixel_values, return_dict, output_hidden_states) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 检查是否输出隐藏状态，如果没有则使用配置中的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有明确指定返回字典值，使用配置中的返回字典值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 如果像素值为None，抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
    
        # 使用conv_stem方法来处理像素值
        hidden_states = self.conv_stem(pixel_values)
    
        # 如果输出隐藏状态为空，则将all_hidden_states设置为一个空元组，否则设置为None
        all_hidden_states = () if output_hidden_states else None
    
        # 遍历每一层的module方法
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
    
            # 如果输出隐藏状态不为空，则将所有隐藏状态存储在all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
        last_hidden_state = hidden_states
    
        # 如果pooler不为空，则使用池化器对最终隐藏状态进行处理
        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None
    
        # 如果不需要返回字典，则返回一个元组
        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)
    
        # 返回BaseModelOutputWithPoolingAndNoAttention类型的值
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )
# 添加对MobileNetV1模型的文档字符串，解释其用途，并引用了先前定义的常量
@add_start_docstrings(
    """
    MobileNetV1 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILENET_V1_START_DOCSTRING,
)
# 定义MobileNetV1ForImageClassification类，继承自MobileNetV1PreTrainedModel类
class MobileNetV1ForImageClassification(MobileNetV1PreTrainedModel):
    # 初始化方法，接受一个MobileNetV1Config类型的config对象
    def __init__(self, config: MobileNetV1Config) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置类别数量
        self.num_labels = config.num_labels
        # 创建MobileNetV1Model实例
        self.mobilenet_v1 = MobileNetV1Model(config)

        # 获取最后一层卷积层的输出通道数
        last_hidden_size = self.mobilenet_v1.layer[-1].convolution.out_channels

        # 分类器头部
        # 使用config中的分类器dropout概率创建一个Dropout层
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        # 根据类别数量判断是否创建一个Linear层或者Identity层
        self.classifier = nn.Linear(last_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 为前向传播方法添加文档字符串，引用了先前定义的常量
    @add_start_docstrings_to_model_forward(MOBILENET_V1_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义前向传播方法，接受像素值、是否输出隐藏状态、标签和是否返回字典作为参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 有指定的话，使用指定的返回类型，否则使用config中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 调用 mobilenet_v1 模型进行图像分类
        outputs = self.mobilenet_v1(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，那么通过Pooler层将输出转化为固定维度的特征向量
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将池化后的特征向量通过一个全连接层得到可以用于分类的 logits
        logits = self.classifier(self.dropout(pooled_output))

        # 初始化损失
        loss = None
        # 当有指定labels时开始计算损失
        if labels is not None:
            # 当问题类型为空时，根据标签数量判断问题的类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择合适的损失函数
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

        # 如果 return_dict 为 False，则返回 logits 和隐藏层，否则返回具有 attention 的 ImageClassifierOutputWithNoAttention
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```