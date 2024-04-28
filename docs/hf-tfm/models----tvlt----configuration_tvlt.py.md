# `.\transformers\models\tvlt\configuration_tvlt.py`

```
# 设置文件编码为utf-8
# 版权声明
# 根据Apache License，Version 2.0进行授权
# 获取许可证的副本：http：//www.apache.org/licenses/LICENSE-2.0
# 软件分发基于“按原样”方式，没有任何形式的担保或条件的分发
# 请查看许可证了解特定语言的限制和权限

""" TVLT模型配置"""

# 导入所需的库和模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ZinengTang/tvlt-base": "https://huggingface.co/ZinengTang/tvlt-base/blob/main/config.json",
}


class TvltConfig(PretrainedConfig):
    r"""
    这是用于存储[`TvltModel`]配置的配置类。它用于根据指定的参数实例化TVLT模型，定义模型架构。使用默认值实例化配置将产生与TVLT[ZinengTang/tvlt-base](https://huggingface.co/ZinengTang/tvlt-base)架构类似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读来自[`PretrainedConfig`]的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import TvltConfig, TvltModel

    >>> # # 初始化TVLT ZinengTang/tvlt-base样式配置
    >>> configuration = TvltConfig()

    >>> # # 从TVLT ZinengTang/tvlt-base样式配置初始化模型（随机权重）
    >>> model = TvltModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "tvlt"

    def __init__(
        self,
        image_size=224,  # 图像大小
        spectrogram_length=2048,  # 光谱长度
        frequency_length=128,  # 频率长度
        image_patch_size=[16, 16],  # 图像补丁大小
        audio_patch_size=[16, 16],  # 音频补丁大小
        num_image_channels=3,  # 图像通道数
        num_audio_channels=1,  # 音频通道数
        num_frames=8,  # 帧数
        hidden_size=768,  # 隐藏层大小
        num_hidden_layers=12,  # 隐藏层层数
        num_attention_heads=12,  # 注意力头数
        intermediate_size=3072,  # 中间层大小
        hidden_act="gelu",  # 隐藏层激活函数
        hidden_dropout_prob=0.0,  # 隐藏层丢弃概率
        attention_probs_dropout_prob=0.0,  # 注意力概率丢弃概率
        initializer_range=0.02,  # 初始化范围
        layer_norm_eps=1e-6,  # 层归一化值
        qkv_bias=True,  # QKV偏置
        use_mean_pooling=False,  # 使用均值池化
        decoder_num_attention_heads=16,  # 解码器注意力头数
        decoder_hidden_size=512,  # 解码器隐藏层大小
        decoder_num_hidden_layers=8,  # 解码器隐藏层层数
        decoder_intermediate_size=2048,  # 解码器中间层大小
        pixel_mask_ratio=0.75,  # 像素蒙版比例
        audio_mask_ratio=0.15,  # 音频蒙版比例
        audio_mask_type="frame-level",  # 音频蒙版类型
        task_matching=True,  # 任务匹配
        task_mae=True,  # 任务MAE
        loss_type="classification",  # 损失类型
        **kwargs,  # 其他关键字参数
        ):
        # 调用父类的构造方法并传入关键字参数
        super().__init__(**kwargs)

        # 检查音频屏蔽类型是否为合法值
        if audio_mask_type not in ("frame-level", "patch_level"):
            # 如果不是合法值则抛出数值错误异常
            raise ValueError(
                "audio_mask_type must be one of two acceptable strategies - {'frame_level', 'patch-level') "
                f"got {audio_mask_type}"
            )

        # 设置图像大小、频谱长度、频率长度、图像补丁大小、音频补丁大小、图像通道数、音频通道数、帧数等参数
        self.image_size = image_size
        self.spectrogram_length = spectrogram_length
        self.frequency_length = frequency_length
        self.image_patch_size = image_patch_size
        self.audio_patch_size = audio_patch_size
        self.num_image_channels = num_image_channels
        self.num_audio_channels = num_audio_channels
        self.num_frames = num_frames

        # 设置隐藏层大小、隐藏层数量、注意力头数量、中间层大小、隐藏层激活函数、隐藏层丢失概率等参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_mean_pooling = use_mean_pooling

        # 设置解码器注意力头数量、解码器隐藏层大小、解码器隐藏层数量、解码器中间层大小、像素屏蔽比率、音频屏蔽比率、音频屏蔽类型等参数
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.pixel_mask_ratio = pixel_mask_ratio
        self.audio_mask_ratio = audio_mask_ratio
        self.audio_mask_type = audio_mask_type

        # 设置任务匹配、任务平均绝对误差、损失类型等参数
        self.task_matching = task_matching
        self.task_mae = task_mae
        self.loss_type = loss_type
```