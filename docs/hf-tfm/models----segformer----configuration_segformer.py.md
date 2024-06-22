# `.\transformers\models\segformer\configuration_segformer.py`

```py
# 导入必要的模块和库
import warnings  # 导入警告模块
from collections import OrderedDict  # 导入有序字典
from typing import Mapping  # 导入类型提示的 Mapping 类型

# 导入版本相关的模块
from packaging import version  # 导入版本模块

# 导入配置工具相关模块和库
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入 ONNX 配置类
from ...utils import logging  # 导入日志工具模块

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 定义 SegFormer 预训练模型配置文件的下载映射
SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nvidia/segformer-b0-finetuned-ade-512-512": (
        "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/config.json"
    ),
    # 可在此链接查看所有 SegFormer 模型：https://huggingface.co/models?filter=segformer
}


class SegformerConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`SegformerModel`] 的配置信息。根据指定的参数实例化 SegFormer 模型时使用这些配置信息，
    定义模型架构。使用默认配置实例化将产生与 SegFormer
    [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    ```
    # 定义函数参数列表
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量，默认为3
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            编码器块的数量（即Mix Transformer编码器中的阶段数），默认为4
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            每个编码器块中的层数，默认为`[2, 2, 2, 2]`
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            每个编码器块中的序列减小比率，默认为`[8, 4, 2, 1]`
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            每个编码器块的维度，默认为`[32, 64, 160, 256]`
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            每个编码器块之前的裁剪大小，默认为`[7, 3, 3, 3]`
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个编码器块之前的步幅，默认为`[4, 2, 2, 2]`
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            每个Transformer编码器块中每个注意层的注意力头数，默认为`[1, 2, 5, 8]`
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            Mix FFNs中隐藏层大小与输入层大小的比率，默认为`[4, 4, 4, 4]`
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串），支持`"gelu"`,`"relu"`,`"selu"`和`"gelu_new"`,默认为`"gelu"`
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的丢失概率，默认为0.0
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            注意概率的丢失比率，默认为0.0
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            分类头部之前的丢失概率，默认为0.1
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为0.02
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            随机深度的丢失概率，用于Transformer编码器的块，默认为0.1
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            层规范化层使用的epsilon，默认为1e-06
        decoder_hidden_size (`int`, *optional*, defaults to 256):
            所有MLP解码头的维度，默认为256
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            语义分割模型的损失函数忽略的索引，默认为255

    Example:

    ```py
    # 示例代码
    >>> from transformers import SegformerModel, SegformerConfig

    >>> # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    # 初始化一个SegFormer nvidia/segformer-b0-finetuned-ade-512-512类型的配置
    # 定义一个模型类型为 "segformer" 的配置类
    model_type = "segformer"
    
    # 定义 SegformerConfig 类，用于初始化模型配置参数
    class SegformerConfig:
        def __init__(
            self,
            num_channels=3,  # 输入图像通道数，默认为 3
            num_encoder_blocks=4,  # 编码器块的数量，默认为 4
            depths=[2, 2, 2, 2],  # 每个编码器块内部的层级深度列表，默认为 [2, 2, 2, 2]
            sr_ratios=[8, 4, 2, 1],  # 每个编码器块内自注意力模块的空间降采样比例列表，默认为 [8, 4, 2, 1]
            hidden_sizes=[32, 64, 160, 256],  # 每个编码器块内隐藏层的大小列表，默认为 [32, 64, 160, 256]
            patch_sizes=[7, 3, 3, 3],  # 每个编码器块内用于划分图像补丁的大小列表，默认为 [7, 3, 3, 3]
            strides=[4, 2, 2, 2],  # 每个编码器块内用于卷积操作的步幅列表，默认为 [4, 2, 2, 2]
            num_attention_heads=[1, 2, 5, 8],  # 每个编码器块内注意力头的数量列表，默认为 [1, 2, 5, 8]
            mlp_ratios=[4, 4, 4, 4],  # 每个编码器块内多层感知机 (MLP) 的隐藏层大小倍率列表，默认为 [4, 4, 4, 4]
            hidden_act="gelu",  # 隐藏层激活函数，默认为 "gelu"
            hidden_dropout_prob=0.0,  # 隐藏层的 dropout 概率，默认为 0.0（不进行 dropout）
            attention_probs_dropout_prob=0.0,  # 注意力层的 dropout 概率，默认为 0.0（不进行 dropout）
            classifier_dropout_prob=0.1,  # 分类器的 dropout 概率，默认为 0.1
            initializer_range=0.02,  # 权重初始化的范围，默认为 0.02
            drop_path_rate=0.1,  # 残差路径（DropPath）的概率，默认为 0.1
            layer_norm_eps=1e-6,  # 层归一化操作的 epsilon 值，默认为 1e-6
            decoder_hidden_size=256,  # 解码器隐藏层的大小，默认为 256
            semantic_loss_ignore_index=255,  # 语义损失函数中忽略的索引值，默认为 255
            **kwargs,  # 允许传入其他关键字参数
        ):
            # 调用父类的初始化方法
            super().__init__(**kwargs)
    
            # 检查是否设置了 reshape_last_stage 参数并且为 False，如果是则发出警告
            if "reshape_last_stage" in kwargs and kwargs["reshape_last_stage"] is False:
                warnings.warn(
                    "Reshape_last_stage is set to False in this config. This argument is deprecated and will soon be"
                    " removed, as the behaviour will default to that of reshape_last_stage = True.",
                    FutureWarning,
                )
    
            # 设置各个配置参数
            self.num_channels = num_channels
            self.num_encoder_blocks = num_encoder_blocks
            self.depths = depths
            self.sr_ratios = sr_ratios
            self.hidden_sizes = hidden_sizes
            self.patch_sizes = patch_sizes
            self.strides = strides
            self.mlp_ratios = mlp_ratios
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.classifier_dropout_prob = classifier_dropout_prob
            self.initializer_range = initializer_range
            self.drop_path_rate = drop_path_rate
            self.layer_norm_eps = layer_norm_eps
            self.decoder_hidden_size = decoder_hidden_size
            self.reshape_last_stage = kwargs.get("reshape_last_stage", True)
            self.semantic_loss_ignore_index = semantic_loss_ignore_index
# 定义 SegformerOnnxConfig 类，继承自 OnnxConfig 类
class SegformerOnnxConfig(OnnxConfig):
    # 定义 torch_onnx_minimum_version 属性，要求 torch 版本至少为 1.11
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义 inputs 属性，返回一个有序字典，指定输入张量的名称和维度顺序
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义 atol_for_validation 属性，设置在验证时的绝对容差
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # 定义 default_onnx_opset 属性，设置默认的 ONNX opset 版本
    @property
    def default_onnx_opset(self) -> int:
        return 12
```