# `.\transformers\models\videomae\configuration_videomae.py`

```py
# 基于现有的PretrainedConfig类，定义了VideoMAEConfig类
# VideoMAEConfig类用于存储VideoMAE模型的配置信息，通过指定的参数实例化一个VideoMAE模型
# VideoMAEConfig类继承自PretrainedConfig类，并可以用于控制模型的输出
# 配置对象可以被用于控制模型的输出，可以阅读PretrainedConfig类的文档获取更多信息
    Args:
        image_size (`int`, *optional*, defaults to 224):
            每个图像的尺寸（分辨率）。
        patch_size (`int`, *optional*, defaults to 16):
            每个补丁的大小（分辨率）。
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量。
        num_frames (`int`, *optional*, defaults to 16):
            每个视频中的帧数。
        tubelet_size (`int`, *optional*, defaults to 2):
            管道大小的数量。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器中隐藏层数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer编码器中每个注意力层的注意力头数量。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer编码器中“中间”（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 "gelu"、"relu"、"selu" 和 "gelu_new"。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的dropout概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            注意力概率的dropout比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的epsilon。
        qkv_bias (`bool`, *optional*, defaults to `True`):
            是否向查询、键和值添加偏置。
        use_mean_pooling (`bool`, *optional*, defaults to `True`):
            是否对最终隐藏状态进行均值池化，而不是使用[CLS]令牌的最终隐藏状态。
        decoder_num_attention_heads (`int`, *optional*, defaults to 6):
            解码器中每个注意力层的注意力头数量。
        decoder_hidden_size (`int`, *optional*, defaults to 384):
            解码器的维度。
        decoder_num_hidden_layers (`int`, *optional*, defaults to 4):
            解码器中隐藏层数量。
        decoder_intermediate_size (`int`, *optional*, defaults to 1536):
            解码器中“中间”（即前馈）层的维度。
        norm_pix_loss (`bool`, *optional*, defaults to `True`):
            是否对目标补丁像素进行归一化。

    Example:

    ```python
    >>> from transformers import VideoMAEConfig, VideoMAEModel

    >>> # 导入 VideoMAEConfig 和 VideoMAEModel 类
    >>> configuration = VideoMAEConfig()

    >>> # 创建一个 VideoMAE 风格的配置对象
    >>> configuration = VideoMAEConfig()

    >>> # 从配置对象随机初始化一个模型
    >>> model = VideoMAEModel(configuration)

    >>> # 获取模型的配置信息
    >>> configuration = model.config
    """   

    # 定义模型类型为 "videomae"
    model_type = "videomae"

    # 初始化方法，设置模型参数并调用父类的初始化方法
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=16,
        tubelet_size=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_mean_pooling=True,
        decoder_num_attention_heads=6,
        decoder_hidden_size=384,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=1536,
        norm_pix_loss=True,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 设置模型参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

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

        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.norm_pix_loss = norm_pix_loss
```