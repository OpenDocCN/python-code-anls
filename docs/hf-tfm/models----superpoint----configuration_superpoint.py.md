# `.\models\superpoint\configuration_superpoint.py`

```
# 导入必要的模块和类
from typing import List

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型与其配置文件的映射关系
SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "magic-leap-community/superpoint": "https://huggingface.co/magic-leap-community/superpoint/blob/main/config.json"
}

# 定义 SuperPointConfig 类，用于存储 SuperPointForKeypointDetection 模型的配置信息
class SuperPointConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SuperPointForKeypointDetection`]. It is used to instantiate a
    SuperPoint model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SuperPoint
    [magic-leap-community/superpoint](https://huggingface.co/magic-leap-community/superpoint) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_hidden_sizes (`List`, *optional*, defaults to `[64, 64, 128, 128]`):
            The number of channels in each convolutional layer in the encoder.
        decoder_hidden_size (`int`, *optional*, defaults to 256): The hidden size of the decoder.
        keypoint_decoder_dim (`int`, *optional*, defaults to 65): The output dimension of the keypoint decoder.
        descriptor_decoder_dim (`int`, *optional*, defaults to 256): The output dimension of the descriptor decoder.
        keypoint_threshold (`float`, *optional*, defaults to 0.005):
            The threshold to use for extracting keypoints.
        max_keypoints (`int`, *optional*, defaults to -1):
            The maximum number of keypoints to extract. If `-1`, will extract all keypoints.
        nms_radius (`int`, *optional*, defaults to 4):
            The radius for non-maximum suppression.
        border_removal_distance (`int`, *optional*, defaults to 4):
            The distance from the border to remove keypoints.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
    ```python
    >>> from transformers import SuperPointConfig, SuperPointForKeypointDetection

    >>> # Initializing a SuperPoint superpoint style configuration
    # 设置模型类型为 "superpoint"
    model_type = "superpoint"

    # 定义 SuperPointConfig 类，用于配置超点模型的参数
    def __init__(
        self,
        encoder_hidden_sizes: List[int] = [64, 64, 128, 128],  # 编码器隐藏层大小列表
        decoder_hidden_size: int = 256,  # 解码器隐藏层大小
        keypoint_decoder_dim: int = 65,  # 关键点解码器维度
        descriptor_decoder_dim: int = 256,  # 描述符解码器维度
        keypoint_threshold: float = 0.005,  # 关键点阈值
        max_keypoints: int = -1,  # 最大关键点数
        nms_radius: int = 4,  # 非极大值抑制半径
        border_removal_distance: int = 4,  # 边界去除距离
        initializer_range=0.02,  # 初始化器范围
        **kwargs,  # 其他关键字参数
    ):
        # 将参数赋值给对象的属性
        self.encoder_hidden_sizes = encoder_hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.keypoint_decoder_dim = keypoint_decoder_dim
        self.descriptor_decoder_dim = descriptor_decoder_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance
        self.initializer_range = initializer_range

        # 调用父类的构造方法，传递其他关键字参数
        super().__init__(**kwargs)
```