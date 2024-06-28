# `.\models\donut\configuration_donut_swin.py`

```
# coding=utf-8
# 定义编码格式为 UTF-8

# 导入所需的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练配置文件的映射字典，将模型名称映射到其预训练配置文件的 URL
DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "naver-clova-ix/donut-base": "https://huggingface.co/naver-clova-ix/donut-base/resolve/main/config.json",
    # 查看所有 Donut 模型的列表：https://huggingface.co/models?filter=donut-swin
}


class DonutSwinConfig(PretrainedConfig):
    r"""
    这是用于存储 [`DonutSwinModel`] 配置信息的配置类。它用于根据指定的参数实例化 Donut 模型，定义模型架构。
    使用默认配置实例化将产生类似于 Donut [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    """
    pass  # 这里只是一个占位符，暂时没有其他配置信息需要添加
    # 定义模型类型为 "donut-swin"
    model_type = "donut-swin"
    
    # 定义一个映射字典，将 "num_attention_heads" 映射到 "num_heads"，"num_hidden_layers" 映射到 "num_layers"
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    # 定义一个初始化方法，初始化一个自定义的神经网络模型
    def __init__(
        self,
        image_size=224,  # 图像大小，默认为224像素
        patch_size=4,  # 图像块大小，默认为4像素
        num_channels=3,  # 图像通道数，默认为3（RGB）
        embed_dim=96,  # 嵌入维度，默认为96
        depths=[2, 2, 6, 2],  # 不同阶段的层数列表
        num_heads=[3, 6, 12, 24],  # 不同阶段的注意力头数列表
        window_size=7,  # 窗口大小，默认为7
        mlp_ratio=4.0,  # MLP扩展比率，默认为4.0
        qkv_bias=True,  # 是否在注意力层中使用QKV偏置，默认为True
        hidden_dropout_prob=0.0,  # 隐藏层dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力层dropout概率，默认为0.0
        drop_path_rate=0.1,  # DropPath率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        use_absolute_embeddings=False,  # 是否使用绝对位置嵌入，默认为False
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm的epsilon，默认为1e-5
        **kwargs,  # 其他关键字参数，用于灵活扩展
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 设置各个参数到对象的属性中
        self.image_size = image_size  # 图像大小
        self.patch_size = patch_size  # 图像块大小
        self.num_channels = num_channels  # 图像通道数
        self.embed_dim = embed_dim  # 嵌入维度
        self.depths = depths  # 不同阶段的层数列表
        self.num_layers = len(depths)  # 网络的总层数
        self.num_heads = num_heads  # 不同阶段的注意力头数列表
        self.window_size = window_size  # 窗口大小
        self.mlp_ratio = mlp_ratio  # MLP扩展比率
        self.qkv_bias = qkv_bias  # 是否使用QKV偏置
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力层dropout概率
        self.drop_path_rate = drop_path_rate  # DropPath率
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.use_absolute_embeddings = use_absolute_embeddings  # 是否使用绝对位置嵌入
        self.layer_norm_eps = layer_norm_eps  # LayerNorm的epsilon
        self.initializer_range = initializer_range  # 初始化范围
    
        # 设置隐藏层大小属性，以使Swin模型与VisionEncoderDecoderModel兼容
        # 这表示模型最后一个阶段之后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
```