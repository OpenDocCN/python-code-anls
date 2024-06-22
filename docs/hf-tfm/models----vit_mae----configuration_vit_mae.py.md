# `.\transformers\models\vit_mae\configuration_vit_mae.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：版权归 Facebook AI 和 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 许可发行
# 除非符合许可要求，否则不能使用此文件
# 可以在以下链接获得许可复本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可分发的软件属于“按原样”分发，没有任何明示或暗示的担保或条件
# 请查看协议中关于限制和特定语言的规定
""" ViT MAE model configuration"""

# 从配置工具中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从工具函数中导入日志记录
from ...utils import logging

# 获取记录器对象
logger = logging.get_logger(__name__)

# ViT MAE 预训练配置文件映射字典，包含不同模型的映射关系
VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/vit-mae-base": "https://huggingface.co/facebook/vit-mae-base/resolve/main/config.json",
    # 查看所有 ViT MAE 模型：https://huggingface.co/models?filter=vit-mae
}

# ViTMAEConfig 类，用于存储 ViTMAEModel 的配置信息
class ViTMAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTMAEModel`]. It is used to instantiate an ViT
    MAE model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the ViT
    [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # ViT MAE 模型的配置类，用于指定模型的各种参数
    class ViTMAEConfig:
        # 编码器层和池化层的维度
        def __init__(
            self,
            hidden_size: int = 768,
            # Transformer 编码器中的隐藏层数量
            num_hidden_layers: int = 12,
            # Transformer 编码器中每个注意力层的注意力头数
            num_attention_heads: int = 12,
            # Transformer 编码器中“中间”（即前馈）层的维度
            intermediate_size: int = 3072,
            # 编码器和池化器中的非线性激活函数
            hidden_act: Union[str, Callable] = "gelu",
            # 嵌入层、编码器和池化器中所有全连接层的 dropout 概率
            hidden_dropout_prob: float = 0.0,
            # 注意力概率的 dropout 比率
            attention_probs_dropout_prob: float = 0.0,
            # 用于初始化所有权重矩阵的截断正态初始化器的标准差
            initializer_range: float = 0.02,
            # 层归一化层使用的 epsilon
            layer_norm_eps: float = 1e-12,
            # 每个图像的大小（分辨率）
            image_size: int = 224,
            # 每个图像块的大小（分辨率）
            patch_size: int = 16,
            # 输入通道数
            num_channels: int = 3,
            # 是否向查询、键和值添加偏置
            qkv_bias: bool = True,
            # 解码器中每个注意力层的注意力头数
            decoder_num_attention_heads: int = 16,
            # 解码器的维度
            decoder_hidden_size: int = 512,
            # 解码器中的隐藏层数量
            decoder_num_hidden_layers: int = 8,
            # 解码器中“中间”（即前馈）层的维度
            decoder_intermediate_size: int = 2048,
            # 输入序列中掩码令牌数量的比例
            mask_ratio: float = 0.75,
            # 是否使用规范化的像素进行训练
            norm_pix_loss: bool = False,
        ):
    
    # 初始化一个 ViT MAE 模型，基于 vit-mae-base 风格的配置
    from transformers import ViTMAEModel
    # 创建一个 ViTMAEConfig 的实例
    configuration = ViTMAEConfig()
    
    # 使用 vit-mae-base 风格的配置初始化一个模型（具有随机权重）
    model = ViTMAEModel(configuration)
    
    # 获取模型的配置信息
    configuration = model.config
    
    # 设置模型类型为 "vit_mae"
    model_type = "vit_mae"
    
    # 定义 ViTMAEModel 类
    def __init__(
        # 设置隐藏层大小为 768
        hidden_size=768,
        # 设置隐藏层数为 12
        num_hidden_layers=12,
        # 设置注意力头数为 12
        num_attention_heads=12,
        # 设置中间层大小为 3072
        intermediate_size=3072,
        # 设置隐藏层激活函数为 "gelu"
        hidden_act="gelu",
        # 设置隐藏层丢失率为 0.0
        hidden_dropout_prob=0.0,
        # 设置注意力丢失率为 0.0
        attention_probs_dropout_prob=0.0,
        # 设置初始化范围为 0.02
        initializer_range=0.02,
        # 设置层归一化参数为 1e-12
        layer_norm_eps=1e-12,
        # 设置图像大小为 224
        image_size=224,
        # 设置图像分块大小为 16
        patch_size=16,
        # 设置图像通道数为 3
        num_channels=3,
        # 设置是否使用 QKV 偏置
        qkv_bias=True,
        # 设置解码器注意力头数为 16
        decoder_num_attention_heads=16,
        # 设置解码器隐藏层大小为 512
        decoder_hidden_size=512,
        # 设置解码器隐藏层数为 8
        decoder_num_hidden_layers=8,
        # 设置解码器中间层大小为 2048
        decoder_intermediate_size=2048,
        # 设置掩码比例为 0.75
        mask_ratio=0.75,
        # 设置是否进行像素损失的归一化
        norm_pix_loss=False,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 设置各种参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
```