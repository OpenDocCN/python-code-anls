# `.\models\layoutlmv2\configuration_layoutlmv2.py`

```
# 设置编码格式为utf-8
# 版权声明
# 根据 Apache 许可证版本 2.0 授权，除非符合许可证要求，否则不得使用该文件。
# 可在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则不得基于“原样”基础分发软件，
# 没有任何明示或暗示的担保或条件。详细信息请参阅许可证，限制在许可范围内。
""" LayoutLMv2 模型配置"""

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入检测是否安装 Detectron2 的函数和日志记录
from ...utils import is_detectron2_available, logging

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 预训练配置文件的映射
LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/config.json",
    "layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/config.json",
    # 可以在 https://huggingface.co/models?filter=layoutlmv2 查看所有 LayoutLMv2 模型
}

# 软依赖
# 如果安装了 Detectron2，则导入
if is_detectron2_available():
    import detectron2

# LayoutLMv2 配置类，继承自预训练配置类
class LayoutLMv2Config(PretrainedConfig):
    r"""
    这是用于存储 [`LayoutLMv2Model`] 配置的配置类。根据指定的参数来实例化一个 LayoutLMv2 模型，
    定义模型架构。使用默认值实例化配置将得到与 LayoutLMv2 [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    示例：

    ```python
    >>> from transformers import LayoutLMv2Config, LayoutLMv2Model

    >>> # 初始化一个 LayoutLMv2 microsoft/layoutlmv2-base-uncased 风格的配置
    >>> configuration = LayoutLMv2Config()

    >>> # 使用 microsoft/layoutlmv2-base-uncased 风格的配置初始化一个（具有随机权重）模型
    >>> model = LayoutLMv2Model(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    
    # 模型类型为 "layoutlmv2"
    model_type = "layoutlmv2"
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        max_2d_position_embeddings=1024,
        max_rel_pos=128,
        rel_pos_bins=32,
        fast_qkv=True,
        max_rel_2d_pos=256,
        rel_2d_pos_bins=64,
        convert_sync_batchnorm=True,
        image_feature_pool_shape=[7, 7, 256],
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=True,
        has_spatial_attention_bias=True,
        has_visual_segment_embedding=False,
        detectron2_config_args=None,
        **kwargs,
    ):

初始化Transformer编码器的类。参数如下：

- `vocab_size(int)`：词汇表大小，默认为30522
- `hidden_size(int)`：隐藏层的大小，默认为768
- `num_hidden_layers(int)`：编码器中的隐藏层数，默认为12
- `num_attention_heads(int)`：注意力头的数量，默认为12
- `intermediate_size(int)`：中间层的大小，默认为3072
- `hidden_act(str)`：隐藏层的激活函数，默认为"gelu"
- `hidden_dropout_prob(float)`：隐藏层的Dropout概率，默认为0.1
- `attention_probs_dropout_prob(float)`：注意力机制的Dropout概率，默认为0.1
- `max_position_embeddings(int)`：最大位置嵌入数，默认为512
- `type_vocab_size(int)`：类型词汇表大小，默认为2
- `initializer_range(float)`：参数初始化的范围，默认为0.02
- `layer_norm_eps(float)`：层标准化的Epsilon，默认为1e-12
- `pad_token_id(int)`：Pad标记的ID，默认为0
- `max_2d_position_embeddings(int)`：二维位置嵌入的最大数目，默认为1024
- `max_rel_pos(int)`：最大相对位置，默认为128
- `rel_pos_bins(int)`：相对位置的bins数，默认为32
- `fast_qkv(bool)`：是否使用快速QKV，默认为True
- `max_rel_2d_pos(int)`：二维相对位置的最大数目，默认为256
- `rel_2d_pos_bins(int)`：二维相对位置的bins数，默认为64
- `convert_sync_batchnorm(bool)`：是否将BatchNorm转换为SyncBatchNorm，默认为True
- `image_feature_pool_shape(list)`：图像特征池化的形状，默认为[7, 7, 256]
- `coordinate_size(int)`：坐标编码的大小，默认为128
- `shape_size(int)`：形状编码的大小，默认为128
- `has_relative_attention_bias(bool)`：是否使用相对注意力偏置，默认为True
- `has_spatial_attention_bias(bool)`：是否使用空间注意力偏置，默认为True
- `has_visual_segment_embedding(bool)`：是否使用视觉切段嵌入，默认为False
- `detectron2_config_args(dict)`：detectron2配置参数的字典，默认为None
- `**kwargs`：额外的关键字参数


        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )

调用父类的初始化方法，初始化Transformer编码器的超参数。


        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
        self.detectron2_config_args = (
            detectron2_config_args if detectron2_config_args is not None else self.get_default_detectron2_config()
        )

设置Transformer编码器的超参数和其他相关属性。


    @classmethod
```        
定义一个类方法。
    # 获取默认的Detectron2配置
    def get_default_detectron2_config(self):
        return {
            "MODEL.MASK_ON": True,  # 开启掩膜功能
            "MODEL.PIXEL_STD": [57.375, 57.120, 58.395],  # 像素标准差
            "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",  # 构建ResNet FPN主干
            "MODEL.FPN.IN_FEATURES": ["res2", "res3", "res4", "res5"],  # FPN输入特征
            "MODEL.ANCHOR_GENERATOR.SIZES": [[32], [64], [128], [256], [512]],  # 锚点生成器大小
            "MODEL.RPN.IN_FEATURES": ["p2", "p3", "p4", "p5", "p6"],  # RPN输入特征
            "MODEL.RPN.PRE_NMS_TOPK_TRAIN": 2000,  # RPN预NMS训练阈值
            "MODEL.RPN.PRE_NMS_TOPK_TEST": 1000,  # RPN预NMS测试阈值
            "MODEL.RPN.POST_NMS_TOPK_TRAIN": 1000,  # RPN后NMS训练阈值
            "MODEL.POST_NMS_TOPK_TEST": 1000,  # 后NMS测试阈值
            "MODEL.ROI_HEADS.NAME": "StandardROIHeads",  # ROI头部标准名
            "MODEL.ROI_HEADS.NUM_CLASSES": 5,  # ROI头部类别数
            "MODEL.ROI_HEADS.IN_FEATURES": ["p2", "p3", "p4", "p5"],  # ROI头部输入特征
            "MODEL.ROI_BOX_HEAD.NAME": "FastRCNNConvFCHead",  # ROI盒子头部名
            "MODEL.ROI_BOX_HEAD.NUM_FC": 2,  # ROI盒子头部全连接层数量
            "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION": 14,  # ROI盒子头部池化器分辨率
            "MODEL.ROI_MASK_HEAD.NAME": "MaskRCNNConvUpsampleHead",  # ROI掩膜头部名
            "MODEL.ROI_MASK_HEAD.NUM_CONV": 4,  # ROI掩膜头部卷积层数量
            "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION": 7,  # ROI掩膜头部池化器分辨率
            "MODEL.RESNETS.DEPTH": 101,  # ResNet深度
            "MODEL.RESNETS.SIZES": [[32], [64], [128], [256], [512]],  # ResNet大小
            "MODEL.RESNETS.ASPECT_RATIOS": [[0.5, 1.0, 2.0]],  # ResNet宽高比
            "MODEL.RESNETS.OUT_FEATURES": ["res2", "res3", "res4", "res5"],  # ResNet输出特征
            "MODEL.RESNETS.NUM_GROUPS": 32,  # ResNet组数
            "MODEL.RESNETS.WIDTH_PER_GROUP": 8,  # ResNet每组宽度
            "MODEL.RESNETS.STRIDE_IN_1X1": False,  # ResNet 1x1步长
        }

    # 获取Detectron2配置
    def get_detectron2_config(self):
        # 获取Detectron2配置对象
        detectron2_config = detectron2.config.get_cfg()
        # 遍历配置参数字典
        for k, v in self.detectron2_config_args.items():
            # 将配置参数键名切割为属性列表
            attributes = k.split(".")
            # 将需设置的对象指向Detectron2配置对象
            to_set = detectron2_config
            # 遍历属性列表除了最后一个属性
            for attribute in attributes[:-1]:
                # 将to_set对象指向对应属性的值
                to_set = getattr(to_set, attribute)
            # 设置属性的最后一个值为v
            setattr(to_set, attributes[-1], v)

        return detectron2_config
```