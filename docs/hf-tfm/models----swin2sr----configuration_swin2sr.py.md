# `.\transformers\models\swin2sr\configuration_swin2sr.py`

```py
# 模型的配置文件，用于定义 Swin2SR 模型的参数和架构
class Swin2SRConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`Swin2SRModel`] 的配置信息。它用于根据指定的参数实例化 Swin Transformer v2 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 Swin Transformer v2 [caidas/swin2sr-classicalsr-x2-64](https://huggingface.co/caidas/swin2sr-classicalsr-x2-64) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Example:

    ```python
    >>> from transformers import Swin2SRConfig, Swin2SRModel

    >>> # 初始化一个类似于 caidas/swin2sr-classicalsr-x2-64 风格的 Swin2SR 配置
    >>> configuration = Swin2SRConfig()

    >>> # 从该配置初始化一个模型（随机权重）的 caidas/swin2sr-classicalsr-x2-64 风格
    >>> model = Swin2SRModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py
    """

    # 模型类型名称
    model_type = "swin2sr"

    # 属性映射字典，用于将外部名称映射到内部模型参数名称
    attribute_map = {
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=64,  # 输入图像大小
        patch_size=1,   # 每个图像块的大小
        num_channels=3,  # 输入图像的通道数
        num_channels_out=None,  # 输出图像的通道数
        embed_dim=180,  # 嵌入维度
        depths=[6, 6, 6, 6, 6, 6],  # 每个阶段的层数
        num_heads=[6, 6, 6, 6, 6, 6],  # 每个阶段的注意力头数
        window_size=8,  # 窗口大小
        mlp_ratio=2.0,  # MLP 扩展比率
        qkv_bias=True,  # QKV 注意力是否包含偏置
        hidden_dropout_prob=0.0,  # 隐藏层的 dropout 概率
        attention_probs_dropout_prob=0.0,  # 注意力概率的 dropout 概率
        drop_path_rate=0.1,  # drop path 比率
        hidden_act="gelu",  # 隐藏层激活函数
        use_absolute_embeddings=False,  # 是否使用绝对位置嵌入
        initializer_range=0.02,  # 初始化器范围
        layer_norm_eps=1e-5,  # LayerNorm 的 epsilon
        upscale=2,  # 上采样因子
        img_range=1.0,  # 图像范围
        resi_connection="1conv",  # 残差连接方式
        upsampler="pixelshuffle",  # 上采样器类型
        **kwargs,
    ):
    ):  
        # 调用父类的初始化方法，并传入kwargs参数
        super().__init__(**kwargs)

        # 设置图像尺寸
        self.image_size = image_size
        # 设置补丁尺寸
        self.patch_size = patch_size
        # 设置输入通道数
        self.num_channels = num_channels
        # 如果没有指定输出通道数，则使用输入通道数作为输出通道数
        self.num_channels_out = num_channels if num_channels_out is None else num_channels_out
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置深度列表
        self.depths = depths
        # 计算层数
        self.num_layers = len(depths)
        # 设置注意力头数
        self.num_heads = num_heads
        # 设置窗口大小
        self.window_size = window_size
        # 设置MLP扩展比率
        self.mlp_ratio = mlp_ratio
        # 设置查询、键、值是否带偏置
        self.qkv_bias = qkv_bias
        # 设置隐藏层dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置路径丢弃率
        self.drop_path_rate = drop_path_rate
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 是否使用绝对位置编码
        self.use_absolute_embeddings = use_absolute_embeddings
        # 设置层归一化epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置是否上采样
        self.upscale = upscale
        # 图像数值范围
        self.img_range = img_range
        # 是否使用残差连接
        self.resi_connection = resi_connection
        # 上采样器
        self.upsampler = upsampler
```