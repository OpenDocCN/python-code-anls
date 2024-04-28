# `.\transformers\models\mobilenet_v2\modeling_mobilenet_v2.py`

```py
# 设置编码格式
# 版权声明
# 授权许可，需要遵守Apache License 2.0
# 获取授权许可的网址
# 根据适用法律或书面协议要求，本软件按"原样"基础分发，不附带任何明示或暗示的担保或条件
# 请参阅许可证以了解特定语言管理权限和限制

""" PyTorch MobileNetV2 model."""

# 引入必要的库和模块
from typing import Optional, Union  # 引入typing模块
import torch  # 引入torch模块
from torch import nn  # 引入torch.nn模块中的所有类和函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 引入torch.nn模块中的交叉熵损失函数

# 引入自定义的模块
from ...activations import ACT2FN  # 从...activations模块引入ACT2FN变量
from ...modeling_outputs import (  # 从...modeling_outputs模块引入下面三个类
    BaseModelOutputWithPoolingAndNoAttention,  # BaseModelOutputWithPoolingAndNoAttention类
    ImageClassifierOutputWithNoAttention,  # ImageClassifierOutputWithNoAttention类
    SemanticSegmenterOutput,  # SemanticSegmenterOutput类
)
from ...modeling_utils import PreTrainedModel  # 从...modeling_utils模块引入PreTrainedModel类
from ...utils import (  # 从...utils模块引入下面这些函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilenet_v2 import MobileNetV2Config  # 从.configuration_mobilenet_v2模块引入MobileNetV2Config类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 为文档提供一般信息
_CONFIG_FOR_DOC = "MobileNetV2Config"  # MobileNetV2的配置信息
_CHECKPOINT_FOR_DOC = "google/mobilenet_v2_1.0_224"  # 需要检查点的信息
_EXPECTED_OUTPUT_SHAPE = [1, 1280, 7, 7]  # 预期的输出形状

# 图像分类文档信息
_IMAGE_CLASS_CHECKPOINT = "google/mobilenet_v2_1.0_224"  # 图像分类检查点信息
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"  # 图像分类的预期输出

# MobileNetV2预训练模型
MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/mobilenet_v2_1.4_224",
    "google/mobilenet_v2_1.0_224",
    "google/mobilenet_v2_0.37_160",
    "google/mobilenet_v2_0.35_96",
    # 查看所有的MobileNetV2模型信息
]


def _build_tf_to_pytorch_map(model, config, tf_weights=None):
    """
    A map of modules from TF to PyTorch.
    """
    # 从TensorFlow到PyTorch的模块映射

    tf_to_pt_map = {}  # 创建空的映射字典

    if isinstance(model, (MobileNetV2ForImageClassification, MobileNetV2ForSemanticSegmentation)):  # 如果模型是MobileNetV2ForImageClassification或MobileNetV2ForSemanticSegmentation的实例
        backbone = model.mobilenet_v2  # 获取MobileNetV2模型的实例
    else:
        backbone = model  # 否则，backbone等于model

    # 使用EMA权重（如果可用）
    def ema(x):  # 创建名为ema的函数，参数为x
        return x + "/ExponentialMovingAverage" if x + "/ExponentialMovingAverage" in tf_weights else x  # 如果x+"/ExponentialMovingAverage"在tf_weights中，则返回x+"ExponentialMovingAverage"，否则返回x

    prefix = "MobilenetV2/Conv/"  # 设置前缀
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.first_conv.convolution.weight  # 将EMA前缀和"weights"映射到backbone.conv_stem.first_conv.convolution.weight
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.first_conv.normalization.bias  # 将EMA前缀和"BatchNorm/beta"映射到backbone.conv_stem.first_conv.normalization.bias
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.first_conv.normalization.weight  # 将EMA前缀和"BatchNorm/gamma"映射到backbone.conv_stem.first_conv.normalization.weight
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.first_conv.normalization.running_mean  # 将"BatchNorm/moving_mean"映射到backbone.conv_stem.first_conv.normalization.running_mean
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.first_conv.normalization.running_var  # 将"BatchNorm/moving_variance"映射到backbone.conv_stem.first_conv.normalization.running_var

    prefix = "MobilenetV2/expanded_conv/depthwise/"  # 设置前缀
    # 将权重映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "depthwise_weights")] = backbone.conv_stem.conv_3x3.convolution.weight
    # 将偏置映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.conv_3x3.normalization.bias
    # 将缩放参数映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.conv_3x3.normalization.weight
    # 将移动平均值映射到对应的 TensorFlow 变量
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.conv_3x3.normalization.running_mean
    # 将移动方差映射到对应的 TensorFlow 变量
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.conv_3x3.normalization.running_var

    # 更新 prefix 值
    prefix = "MobilenetV2/expanded_conv/project/"
    # 将权重映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_stem.reduce_1x1.convolution.weight
    # 将偏置映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = backbone.conv_stem.reduce_1x1.normalization.bias
    # 将缩放参数映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = backbone.conv_stem.reduce_1x1.normalization.weight
    # 将移动平均值映射到对应的 TensorFlow 变量
    tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = backbone.conv_stem.reduce_1x1.normalization.running_mean
    # 将移动方差映射到对应的 TensorFlow 变量
    tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = backbone.conv_stem.reduce_1x1.normalization.running_var

    # 遍历 backbone 的 layer
    for i in range(16):
        tf_index = i + 1
        pt_index = i
        pointer = backbone.layer[pt_index]

        # 根据索引更新 prefix 值
        prefix = f"MobilenetV2/expanded_conv_{tf_index}/expand/"
        # 将权重映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "weights")] = pointer.expand_1x1.convolution.weight
        # 将偏置映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.expand_1x1.normalization.bias
        # 将缩放参数映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.expand_1x1.normalization.weight
        # 将移动平均值映射到对应的 TensorFlow 变量
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.expand_1x1.normalization.running_mean
        # 将移动方差映射到对应的 TensorFlow 变量
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.expand_1x1.normalization.running_var

        # 根据索引更新 prefix 值
        prefix = f"MobilenetV2/expanded_conv_{tf_index}/depthwise/"
        # 将深度可分离卷积的权重映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "depthwise_weights")] = pointer.conv_3x3.convolution.weight
        # 将深度可分离卷积的偏置映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.conv_3x3.normalization.bias
        # 将深度可分离卷积的缩放参数映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.conv_3x3.normalization.weight
        # 将深度可分离卷积的移动平均值映射到对应的 TensorFlow 变量
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.conv_3x3.normalization.running_mean
        # 将深度可分离卷积的移动方差映射到对应的 TensorFlow 变量
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.conv_3x3.normalization.running_var

        # 根据索引更新 prefix 值
        prefix = f"MobilenetV2/expanded_conv_{tf_index}/project/"
        # 将权重映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "weights")] = pointer.reduce_1x1.convolution.weight
        # 将偏置映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "BatchNorm/beta")] = pointer.reduce_1x1.normalization.bias
        # 将缩放参数映射到对应的 TensorFlow 变量
        tf_to_pt_map[ema(prefix + "BatchNorm/gamma")] = pointer.reduce_1x1.normalization.weight
        # 将移动平均值映射到对应的 TensorFlow 变量
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = pointer.reduce_1x1.normalization.running_mean
        # 将移动方差映射到对应的 TensorFlow 变量
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = pointer.reduce_1x1.normalization.running_var

    # 更新 prefix 值
    prefix = "MobilenetV2/Conv_1/"
    # 将权重映射到对应的 TensorFlow 变量
    tf_to_pt_map[ema(prefix + "weights")] = backbone.conv_1x1.convolution.weight
    # 将 TensorFlow 模型参数映射到 PyTorch 模型参数的字典中
    
    # 如果模型是 MobileNetV2ForImageClassification 的实例
    if isinstance(model, MobileNetV2ForImageClassification):
        # 更新前缀为 Image Classification 任务的权重和偏置参数
        prefix = "MobilenetV2/Logits/Conv2d_1c_1x1/"
        # 将 TensorFlow 模型中的偏置参数映射到 PyTorch 模型的分类器权重
        tf_to_pt_map[ema(prefix + "weights")] = model.classifier.weight
        # 将 TensorFlow 模型中的权重参数映射到 PyTorch 模型的分类器偏置
        tf_to_pt_map[ema(prefix + "biases")] = model.classifier.bias
    
    # 如果模型是 MobileNetV2ForSemanticSegmentation 的实例
    if isinstance(model, MobileNetV2ForSemanticSegmentation):
        # 更新前缀为 Semantic Segmentation 任务的权重和偏置参数
        prefix = "image_pooling/"
        # 将 TensorFlow 模型中的权重参数映射到 PyTorch 模型的分割头图像池化层权重
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_pool.convolution.weight
        # 将 TensorFlow 模型中的偏置参数映射到 PyTorch 模型的分割头图像池化层偏置
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_pool.normalization.bias
        # 将 TensorFlow 模型中的缩放参数映射到 PyTorch 模型的分割头图像池化层缩放
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_pool.normalization.weight
        # 将 TensorFlow 模型中的移动均值参数映射到 PyTorch 模型的分割头图像池化层移动均值
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_pool.normalization.running_mean
        # 将 TensorFlow 模型中的移动方差参数映射到 PyTorch 模型的分割头图像池化层移动方差
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = model.segmentation_head.conv_pool.normalization.running_var
    
        # 更新前缀为 ASPP 模块的权重和偏置参数
        prefix = "aspp0/"
        # 将 TensorFlow 模型中的权重参数映射到 PyTorch 模型的 ASPP 模块权重
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_aspp.convolution.weight
        # 将 TensorFlow 模型中的偏置参数映射到 PyTorch 模型的 ASPP 模块偏置
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_aspp.normalization.bias
        # 将 TensorFlow 模型中的缩放参数映射到 PyTorch 模型的 ASPP 模块缩放
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_aspp.normalization.weight
        # 将 TensorFlow 模型中的移动均值参数映射到 PyTorch 模型的 ASPP 模块移动均值
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_aspp.normalization.running_mean
        # 将 TensorFlow 模型中的移动方差参数映射到 PyTorch 模型的 ASPP 模块移动方差
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = model.segmentation_head.conv_aspp.normalization.running_var
    
        # 更新前缀为 Concatenation Projection 模块的权重和偏置参数
        prefix = "concat_projection/"
        # 将 TensorFlow 模型中的权重参数映射到 PyTorch 模型的 Concatenation Projection 模块权重
        tf_to_pt_map[prefix + "weights"] = model.segmentation_head.conv_projection.convolution.weight
        # 将 TensorFlow 模型中的偏置参数映射到 PyTorch 模型的 Concatenation Projection 模块偏置
        tf_to_pt_map[prefix + "BatchNorm/beta"] = model.segmentation_head.conv_projection.normalization.bias
        # 将 TensorFlow 模型中的缩放参数映射到 PyTorch 模型的 Concatenation Projection 模块缩放
        tf_to_pt_map[prefix + "BatchNorm/gamma"] = model.segmentation_head.conv_projection.normalization.weight
        # 将 TensorFlow 模型中的移动均值参数映射到 PyTorch 模型的 Concatenation Projection 模块移动均值
        tf_to_pt_map[prefix + "BatchNorm/moving_mean"] = model.segmentation_head.conv_projection.normalization.running_mean
        # 将 TensorFlow 模型中的移动方差参数映射到 PyTorch 模型的 Concatenation Projection 模块移动方差
        tf_to_pt_map[prefix + "BatchNorm/moving_variance"] = model.segmentation_head.conv_projection.normalization.running_var
    
        # 更新前缀为 Semantic Segmentation 模块的权重和偏置参数
        prefix = "logits/semantic/"
        # 将 TensorFlow 模型中的权重参数映射到 PyTorch 模型的 Semantic Segmentation 模块权重
        tf_to_pt_map[ema(prefix + "weights")] = model.segmentation_head.classifier.convolution.weight
        # 将 TensorFlow 模型中的偏置参数映射到 PyTorch 模型的 Semantic Segmentation 模块偏置
        tf_to_pt_map[ema(prefix + "biases")] = model.segmentation_head.classifier.convolution.bias
    
    # 返回 TensorFlow 到 PyTorch 模型参数映射字典
    return tf_to_pt_map
# 加载 TensorFlow checkpoints 到 PyTorch 模型中
def load_tf_weights_in_mobilenet_v2(model, config, tf_checkpoint_path):
    """Load TensorFlow checkpoints in a PyTorch model."""
    尝试导入 numpy 和 tensorflow 库
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError:
        如果导入失败，记录错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 从 TF 模型加载权重
    获取 TF 模型中的所有变量
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    创建空字典用于存储 TF 权重
    tf_weights = {}
    遍历 TF 变量，读取权重数据并保存到字典中
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_checkpoint_path, name)
        tf_weights[name] = array

    # 构建 TF 到 PyTorch 权重加载的映射
    tf_to_pt_map = _build_tf_to_pytorch_map(model, config, tf_weights)

    遍历映射，导入权重到 PyTorch 模型中
    for name, pointer in tf_to_pt_map.items():
        logger.info(f"Importing {name}")
        如果名称不在 TF 权重字典中，记录信息并跳过
        if name not in tf_weights:
            logger.info(f"{name} not in tf pre-trained weights, skipping")
            continue

        获取权重数据
        array = tf_weights[name]

        如果名称中包含"depthwise_weights"，则转置数组
        if "depthwise_weights" in name:
            logger.info("Transposing depthwise")
            array = np.transpose(array, (2, 3, 0, 1))
        如果名称中包含"weights"，则根据不同情况进行转置操作
        elif "weights" in name:
            logger.info("Transposing")
            如果指针的形状是二维的，则压缩数组并转置
            if len(pointer.shape) == 2:  # copying into linear layer
                array = array.squeeze().transpose()
            否则，根据指定的轴进行转置
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        如果指针和数组的形状不匹配，抛出数值错误异常
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        记录信息并初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name} {array.shape}")
        pointer.data = torch.from_numpy(array)

        从 TF 权重字典中删除已经导入的权重
        tf_weights.pop(name, None)
        tf_weights.pop(name + "/RMSProp", None)
        tf_weights.pop(name + "/RMSProp_1", None)
        tf_weights.pop(name + "/ExponentialMovingAverage", None)
        tf_weights.pop(name + "/Momentum", None)

    记录未复制到 PyTorch 模型中的权重
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}")
    返回模型
    return model


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    确保所有图层的通道数可以被 `divisor` 整除。该函数取自原始的 TensorFlow 代码库。可以在以下链接找到：
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    如果未指定最小值，则将最小值设置为除数
    if min_value is None:
        min_value = divisor
    计算新的值，确保其不小于指定的最小值，并且可以被除数整除
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保向下取整不会使值减少超过 10%
    if new_value < 0.9 * value:
        new_value += divisor
    返回新的整数值
    return int(new_value)


def apply_depth_multiplier(config: MobileNetV2Config, channels: int) -> int:
    # 返回一个可被 config.depth_divisible_by 整除的 channels * config.depth_multiplier 的四舍五入后的整数值
    return make_divisible(int(round(channels * config.depth_multiplier)), config.depth_divisible_by, config.min_depth)
# 定义一个函数，应用 TensorFlow 风格的 "SAME" 填充到一个卷积层上
def apply_tf_padding(features: torch.Tensor, conv_layer: nn.Conv2d) -> torch.Tensor:
    """
    Apply TensorFlow-style "SAME" padding to a convolution layer. See the notes at:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    """
    # 获取输入特征的高度和宽度
    in_height = int(features.shape[-2])
    in_width = int(features.shape[-1])
    # 获取卷积层的步长、卷积核大小和膨胀率
    stride_height, stride_width = conv_layer.stride
    kernel_height, kernel_width = conv_layer.kernel_size
    dilation_height, dilation_width = conv_layer.dilation

    # 计算高度方向的填充量
    if in_height % stride_height == 0:
        pad_along_height = max(kernel_height - stride_height, 0)
    else:
        pad_along_height = max(kernel_height - (in_height % stride_height), 0)

    # 计算宽度方向的填充量
    if in_width % stride_width == 0:
        pad_along_width = max(kernel_width - stride_width, 0)
    else:
        pad_along_width = max(kernel_width - (in_width % stride_width), 0)

    # 计算左右上下四个方向的填充值
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    # 生成填充元组
    padding = (
        pad_left * dilation_width,
        pad_right * dilation_width,
        pad_top * dilation_height,
        pad_bottom * dilation_height,
    )
    # 对输入特征进行填充操作
    return nn.functional.pad(features, padding, "constant", 0.0)


class MobileNetV2ConvLayer(nn.Module):
    def __init__(
        self,
        config: MobileNetV2Config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        dilation: int = 1,
        use_normalization: bool = True,
        use_activation: Union[bool, str] = True,
        layer_norm_eps: Optional[float] = None,
    # 初始化函数，设置卷积层的参数和配置
    ) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置信息
        self.config = config

        # 检查输入通道数是否能够被分组数整除
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        # 检查输出通道数是否能够被分组数整除
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 计算填充大小
        padding = 0 if config.tf_padding else int((kernel_size - 1) / 2) * dilation

        # 创建卷积层
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        # 如果使用标准化，则创建 BatchNorm2d 层
        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=config.layer_norm_eps if layer_norm_eps is None else layer_norm_eps,
                momentum=0.997,
                affine=True,
                track_running_stats=True,
            )
        else:
            # 否则将标准化层设为 None
            self.normalization = None

        # 如果使用激活函数
        if use_activation:
            # 如果指定了激活函数的字符串形式，则使用预定义的激活函数
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]
            # 如果未指定激活函数但配置了默认的隐藏层激活函数，则使用默认的隐藏层激活函数
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            # 否则使用配置中的隐藏层激活函数
            else:
                self.activation = config.hidden_act
        else:
            # 否则将激活函数设为 None
            self.activation = None

    # 前向传播函数，执行卷积、标准化和激活操作
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果使用 TensorFlow 风格的填充，则应用 TensorFlow 风格的填充函数
        if self.config.tf_padding:
            features = apply_tf_padding(features, self.convolution)
        # 执行卷积操作
        features = self.convolution(features)
        # 如果有标准化层，则执行标准化操作
        if self.normalization is not None:
            features = self.normalization(features)
        # 如果有激活函数，则执行激活操作
        if self.activation is not None:
            features = self.activation(features)
        # 返回处理后的特征张量
        return features
# 定义 MobileNetV2 模型中的反向残差模块
class MobileNetV2InvertedResidual(nn.Module):
    def __init__(
        self, config: MobileNetV2Config, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()

        # 计算扩展后的通道数，确保可被 config.depth_divisible_by 整除，并且不低于 config.min_depth
        expanded_channels = make_divisible(
            int(round(in_channels * config.expand_ratio)), config.depth_divisible_by, config.min_depth
        )

        # 检查步长值是否合法
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 确定是否使用残差连接
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 执行 1x1 卷积进行通道扩展
        self.expand_1x1 = MobileNetV2ConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        # 执行 3x3 卷积
        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        # 执行 1x1 卷积进行通道缩减
        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    # 前向传播函数
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 保存输入特征，以便用于残差连接
        residual = features

        # 执行通道扩展
        features = self.expand_1x1(features)
        # 执行 3x3 卷积
        features = self.conv_3x3(features)
        # 执行通道缩减
        features = self.reduce_1x1(features)

        # 如果使用残差连接，则将输入特征与处理后的特征相加
        return residual + features if self.use_residual else features


# 定义 MobileNetV2 模型中的起始模块
class MobileNetV2Stem(nn.Module):
    def __init__(self, config: MobileNetV2Config, in_channels: int, expanded_channels: int, out_channels: int) -> None:
        super().__init__()

        # 第一层是普通的 3x3 卷积，步长为 2，将输入通道扩展到 32
        # 所有其他扩展层都使用扩展因子计算输出通道数
        self.first_conv = MobileNetV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=2,
        )

        # 如果 config.first_layer_is_expansion 为 True，则不使用额外的 1x1 卷积进行通道扩展
        if config.first_layer_is_expansion:
            self.expand_1x1 = None
        else:
            # 使用额外的 1x1 卷积进行通道扩展
            self.expand_1x1 = MobileNetV2ConvLayer(
                config, in_channels=expanded_channels, out_channels=expanded_channels, kernel_size=1
            )

        # 执行 3x3 卷积
        self.conv_3x3 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=1,
            groups=expanded_channels,
        )

        # 执行 1x1 卷积进行通道缩减
        self.reduce_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )
    # 定义一个前向传播的方法，接受一个 torch.Tensor 类型的 features 参数，并返回一个 torch.Tensor 类型的结果
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 对输入的 features 进行第一次卷积操作
        features = self.first_conv(features)
        # 如果存在 expand_1x1 层，则对 features 进行 expand_1x1 操作
        if self.expand_1x1 is not None:
            features = self.expand_1x1(features)
        # 对 features 进行卷积操作
        features = self.conv_3x3(features)
        # 对 features 进行 reduce_1x1 操作
        features = self.reduce_1x1(features)
        # 返回处理后的 features
        return features
class MobileNetV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义 MobileNetV2 预训练模型类，继承自 PreTrainedModel

    config_class = MobileNetV2Config
    # 配置类为 MobileNetV2Config
    
    load_tf_weights = load_tf_weights_in_mobilenet_v2
    # 加载 TensorFlow 权重为 mobilenet_v2
    
    base_model_prefix = "mobilenet_v2"
    # 基础模型前缀为 mobilenet_v2
    
    main_input_name = "pixel_values"
    # 主输入名称为 pixel_values
    
    supports_gradient_checkpointing = False
    # 不支持梯度检查点

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d]) -> None:
        """Initialize the weights"""
        # 初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 对线性层和卷积层初始化权重为标准正态分布，标准差为配置中的初始化范围
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果存在偏置，则初始化为零
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # 对批归一化层初始化，偏置为零，权重填充为1.0


MOBILENET_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileNetV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILENET_V2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileNetV2ImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare MobileNetV2 model outputting raw hidden-states without any specific head on top.",
    MOBILENET_V2_START_DOCSTRING,
)
# 添加描述信息至 MobileNetV2Model 模型

class MobileNetV2Model(MobileNetV2PreTrainedModel):
    # 定义 MobileNetV2 模型，继承自 MobileNetV2PreTrainedModel
    # 初始化函数，接受 MobileNetV2Config 类型的配置参数，并有一个默认参数 add_pooling_layer
    def __init__(self, config: MobileNetV2Config, add_pooling_layer: bool = True):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将传入的配置参数赋值给实例变量
        self.config = config

        # 设定投影层的输出通道数
        channels = [16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320]
        # 根据配置参数的深度乘数应用于通道数组
        channels = [apply_depth_multiplier(config, x) for x in channels]

        # 深度层的步长
        strides = [2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]

        # 创建 MobileNetV2Stem 对象
        self.conv_stem = MobileNetV2Stem(
            config,
            in_channels=config.num_channels,
            expanded_channels=apply_depth_multiplier(config, 32),
            out_channels=channels[0],
        )

        current_stride = 2  # 第一个卷积层步长为 2
        dilation = 1

        # 创建 nn.ModuleList 对象
        self.layer = nn.ModuleList()
        for i in range(16):
            # 保持特征图尺寸较小或使用扩展卷积？
            if current_stride == config.output_stride:
                layer_stride = 1
                layer_dilation = dilation
                dilation *= strides[i]  # 较大的扩张从下一个块开始
            else:
                layer_stride = strides[i]
                layer_dilation = 1
                current_stride *= layer_stride

            # 添加 MobileNetV2InvertedResidual 到层列表中
            self.layer.append(
                MobileNetV2InvertedResidual(
                    config,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=layer_stride,
                    dilation=layer_dilation,
                )
            )

        if config.finegrained_output and config.depth_multiplier < 1.0:
            output_channels = 1280
        else:
            output_channels = apply_depth_multiplier(config, 1280)

        # 创建 MobileNetV2ConvLayer 对象
        self.conv_1x1 = MobileNetV2ConvLayer(
            config,
            in_channels=channels[-1],
            out_channels=output_channels,
            kernel_size=1,
        )

        # 根据需要添加自适应平均池化层
        self.pooler = nn.AdaptiveAvgPool2d((1, 1)) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 剪枝头部
    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    # 前向传播函数，接受像素值、是否输出隐藏层、是否返回字典等参数
    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数的返回类型注解为Union类型，可能是元组或者BaseModelOutputWithPoolingAndNoAttention类型
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 如果output_hidden_states参数不为None，则使用该参数值，否则使用配置中的output_hidden_states值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict参数不为None，则使用该参数值，否则使用配置中的use_return_dict值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供pixel_values，则引发ValueError异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用输入的像素值作为输入，通过卷积层进行处理
        hidden_states = self.conv_stem(pixel_values)

        # 如果需要输出所有隐藏状态，则初始化一个空元组，否则为None
        all_hidden_states = () if output_hidden_states else None

        # 遍历所有的层模块
        for i, layer_module in enumerate(self.layer):
            # 通过当前层模块处理隐藏状态
            hidden_states = layer_module(hidden_states)

            # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 对最终的隐藏状态进行1x1卷积处理
        last_hidden_state = self.conv_1x1(hidden_states)

        # 如果存在池化层，则对最终隐藏状态进行池化并展平处理
        if self.pooler is not None:
            pooled_output = torch.flatten(self.pooler(last_hidden_state), start_dim=1)
        else:
            pooled_output = None

        # 如果return_dict为False，则返回一个元组，包含最终的隐藏状态、池化输出和所有隐藏状态
        if not return_dict:
            return tuple(v for v in [last_hidden_state, pooled_output, all_hidden_states] if v is not None)

        # 如果return_dict为True，则返回BaseModelOutputWithPoolingAndNoAttention对象，包含最终的隐藏状态、池化输出和所有隐藏状态
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )
# 使用 MobileNetV2 模型，在顶部添加一个图像分类头部（线性层，用于对池化特征进行分类），例如用于 ImageNet 数据集
@add_start_docstrings(
    """
    MobileNetV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILENET_V2_START_DOCSTRING,
)
class MobileNetV2ForImageClassification(MobileNetV2PreTrainedModel):
    # MobileNetV2ForImageClassification 类的初始化方法
    def __init__(self, config: MobileNetV2Config) -> None:
        # 调用 MobileNetV2PreTrainedModel 类的初始化方法
        super().__init__(config)

        # 从配置中获取标签数量
        self.num_labels = config.num_labels
        # 创建 MobileNetV2Model 模型
        self.mobilenet_v2 = MobileNetV2Model(config)

        # 获取最后一个隐藏层的大小
        last_hidden_size = self.mobilenet_v2.conv_1x1.convolution.out_channels

        # 分类器头部
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)
        # 如果有标签数量大于 0，则添加一个线性层；否则添加一个 Identity 层
        self.classifier = nn.Linear(last_hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # MobileNetV2ForImageClassification 类的前向传播方法
    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        # 输入像素值张量，默认为空
        pixel_values: Optional[torch.Tensor] = None,
        # 是否输出隐藏层状态，默认为空
        output_hidden_states: Optional[bool] = None,
        # 标签张量，默认为空
        labels: Optional[torch.Tensor] = None,
        # 是否返回字典，默认为空
        return_dict: Optional[bool] = None,
        ) -> Union[tuple, ImageClassifierOutputWithNoAttention]: 
            # 函数声明，接受一个参数和一个返回值的类型提示

        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*): 
            # 参数说明，labels是一个形状为(batch_size,)的torch.LongTensor类型的张量，是可选的

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # 对return_dict进行赋值，如果return_dict不为None，则使用return_dict的值，否则使用self.config.use_return_dict的值
        outputs = self.mobilenet_v2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
            # 调用self.mobilenet_v2方法，传入pixel_values参数和output_hidden_states和return_dict两个关键字参数

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
            # 如果return_dict为真，则pooled_output等于outputs.pooler_output，否则等于outputs的第一个元素

        logits = self.classifier(self.dropout(pooled_output))
            # 对pooled_output进行dropout后，再传入self.classifier方法中获取logits

        loss = None
            # 初始化loss为None
        if labels is not None:
            # 如果labels不为None，则进入条件判断
            if self.config.problem_type is None:
                # 如果self.config.problem_type为None，则进入条件判断
                if self.num_labels == 1:
                    # 如果self.num_labels等于1，则config.problem_type赋值为"regression"
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    # 如果self.num_labels大于1且(labels的数据类型为torch.long或torch.int)，则config.problem_type赋值为"single_label_classification"
                    self.config.problem_type = "single_label_classification"
                else:
                    # 否则config.problem_type赋值为"multi_label_classification"
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 如果config.problem_type为"regression"，则使用MSELoss作为loss_fct
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果num_labels等于1，计算logits.squeeze()和labels.squeeze()的均方误差作为loss
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则，计算logits和labels的均方误差作为loss
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 如果config.problem_type为"single_label_classification"，则使用CrossEntropyLoss作为loss_fct
                loss_fct = CrossEntropyLoss()
                # 计算logits.view(-1, self.num_labels)和labels.view(-1)的交叉熵作为loss
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 如果config.problem_type为"multi_label_classification"，则使用BCEWithLogitsLoss作为loss_fct
                loss_fct = BCEWithLogitsLoss()
                # 计算logits和labels的二元交叉熵作为loss
                loss = loss_fct(logits, labels)

        if not return_dict:
            # 如果return_dict为假
            output = (logits,) + outputs[2:]
            # 对output进行赋值，元组中包含logits和outputs的第三个元素以后的所有元素
            return ((loss,) + output) if loss is not None else output
            # 如果loss不为None，返回包含loss和output的元组，否则返回output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
            # 返回ImageClassifierOutputWithNoAttention对象，包括loss、logits和outputs的hidden_states
# 定义 MobileNetV2DeepLabV3Plus 类，用于实现 "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" 论文中的神经网络
class MobileNetV2DeepLabV3Plus(nn.Module):
    """
    The neural network from the paper "Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation" https://arxiv.org/abs/1802.02611
    """

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__()

        # 定义自适应均值池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 定义 MobileNetV2ConvLayer 类型的池化层
        self.conv_pool = MobileNetV2ConvLayer(
            config,
            in_channels=apply_depth_multiplier(config, 320),
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        # 定义 MobileNetV2ConvLayer 类型的 ASPP 层
        self.conv_aspp = MobileNetV2ConvLayer(
            config,
            in_channels=apply_depth_multiplier(config, 320),
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        # 定义 MobileNetV2ConvLayer 类型的投影层
        self.conv_projection = MobileNetV2ConvLayer(
            config,
            in_channels=512,
            out_channels=256,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
            layer_norm_eps=1e-5,
        )

        # 定义二维 Dropout 层
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        # 定义 MobileNetV2ConvLayer 类型的分类器
        self.classifier = MobileNetV2ConvLayer(
            config,
            in_channels=256,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    # 实现前向传播过程
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]

        # 均值池化
        features_pool = self.avg_pool(features)
        features_pool = self.conv_pool(features_pool)
        # 双线性插值调整尺寸
        features_pool = nn.functional.interpolate(
            features_pool, size=spatial_size, mode="bilinear", align_corners=True
        )

        # ASPP 层
        features_aspp = self.conv_aspp(features)

        # 拼接池化特征和 ASPP 特征
        features = torch.cat([features_pool, features_aspp], dim=1)

        # 投影层
        features = self.conv_projection(features)
        # Dropout
        features = self.dropout(features)
        # 分类器
        features = self.classifier(features)
        
        return features


@add_start_docstrings(
    """
    MobileNetV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILENET_V2_START_DOCSTRING,
)
# 定义 MobileNetV2ForSemanticSegmentation 类，继承自 MobileNetV2PreTrainedModel
class MobileNetV2ForSemanticSegmentation(MobileNetV2PreTrainedModel):
    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        # 实例化 MobileNetV2Model，不包含池化层
        self.mobilenet_v2 = MobileNetV2Model(config, add_pooling_layer=False)
        # 实例化 MobileNetV2DeepLabV3Plus，用于语义分割
        self.segmentation_head = MobileNetV2DeepLabV3Plus(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(MOBILENET_V2_INPUTS_DOCSTRING)
    # 用于替换返回文档字符串，指定输出类型为SemanticSegmenterOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接受像素数值、标签、是否返回隐藏状态、是否返回字典作为参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 像素数值的张量，可选参数，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签的张量，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态的布尔值，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典的布尔值，可选参数，默认为None
    # 定义一个方法，用于执行语义分割任务
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            用于计算损失的标签，语义分割地图的真实值。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels > 1`，则计算分类损失（交叉熵）。

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MobileNetV2ForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")
        >>> model = MobileNetV2ForSemanticSegmentation.from_pretrained("google/deeplabv3_mobilenet_v2_1.0_513")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        """
        # 获取是否输出隐藏状态的配置参数值，如果未指定则使用默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 获取是否使用返回字典的配置参数值，如果未指定则使用默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用预训练的 MobileNetV2 模型对输入的像素值进行处理，并得到隐藏状态
        outputs = self.mobilenet_v2(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # 使用分割头网络对隐藏状态进行分类，得到预测的语义分割地图
        logits = self.segmentation_head(encoder_hidden_states[-1])

        # 初始化损失为 None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果标签数目为 1，则抛出异常
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # 将预测结果的大小调整为原始图像的大小
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                # 使用交叉熵损失函数计算损失
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

        # 如果不使用返回字典形式
        if not return_dict:
            # 如果输出隐藏状态，则将预测结果和隐藏状态输出
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            # 如果损失不为 None，则将损失和输出组成元组返回；否则只返回输出
            return ((loss,) + output) if loss is not None else output

        # 使用自定义的类 SemanticSegmenterOutput 封装输出结果
        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```