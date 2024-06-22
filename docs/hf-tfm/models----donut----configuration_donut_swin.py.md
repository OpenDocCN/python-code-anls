# `.\models\donut\configuration_donut_swin.py`

```py
# 设置文件编码为 utf-8

# 版权声明

# 引入预训练配置和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Donut Swin 预训练配置文件映射表，包含了各种预训练模型的地址
DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "naver-clova-ix/donut-base": "https://huggingface.co/naver-clova-ix/donut-base/resolve/main/config.json",
    # 查看所有 Donut 模型地址：https://huggingface.co/models?filter=donut-swin
}

# 定义 DonutSwinConfig 类，用于存储 DonutSwinModel 的配置
# 该配置用于根据指定参数实例化 Donut 模型，定义模型架构
# 使用默认值实例化配置会产生类似 naver-clova-ix/donut-base 的配置
# 配置对象继承自 PretrainedConfig，可用于控制模型输出，可参考 PretrainedConfig 文档获取更多信息
    # 定义函数的参数及其默认值
    # 图像大小的分辨率
    # 每个补丁的大小
    # 输入通道的数量
    # 补丁嵌入的维度
    # Transformer编码器中每个层的深度
    # Transformer编码器每层的注意力头数
    # 窗口大小
    # MLP隐藏维度与嵌入维度的比率
    # 是否应该为查询、键和值添加可学习的偏差
    # 嵌入和编码器中所有全连接层的丢弃概率
    # 注意力概率的丢弃比例
    # 随机深度率
    # 编码器中的非线性激活函数
    # 是否应该为补丁嵌入添加绝对位置嵌入
    # 初始化所有权重矩阵的截断正态初始化器的标准差
    # 层标准化层使用的ε

    # 例子

    # 从transformers库中导入DonutSwinConfig和DonutSwinModel
    # 初始化一个基于DonutSwinConfig配置的Donut naver-clova-ix/donut-base风格模型
    # 从naver-clova-ix/donut-base风格配置随机初始化一个模型
    # 访问模型的配置

    # 设置模型类型为"donut-swin"
    
    # 定义属性映射，将"num_attention_heads"映射到"num_heads"，将"num_hidden_layers"映射到"num_layers"
    # 初始化函数，用于初始化Swin Transformer模型的各种参数
    def __init__(
        self,
        image_size=224,  # 图像大小，默认为224
        patch_size=4,  # 图像分块大小，默认为4
        num_channels=3,  # 图像通道数，默认为3
        embed_dim=96,  # 嵌入维度，默认为96
        depths=[2, 2, 6, 2],  # 每个阶段的深度，默认为[2, 2, 6, 2]
        num_heads=[3, 6, 12, 24],  # 每个阶段的注意力头数，默认为[3, 6, 12, 24]
        window_size=7,  # 窗口大小，默认为7
        mlp_ratio=4.0,  # MLP展开的比例，默认为4.0
        qkv_bias=True,  # 是否包含QKV的偏置，默认为True
        hidden_dropout_prob=0.0,  # 隐藏层的dropout比例，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的dropout比例，默认为0.0
        drop_path_rate=0.1,  # DropPath的比例，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        use_absolute_embeddings=False,  # 是否使用绝对位置编码，默认为False
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm的epsilon值，默认为1e-5
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置各种参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # 设置hidden_size属性，以使Swin与VisionEncoderDecoderModel兼容
        # 这表示模型最后一个阶段之后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
```  
```