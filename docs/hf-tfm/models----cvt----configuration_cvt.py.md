# `.\models\cvt\configuration_cvt.py`

```
# 设置文件编码为 UTF-8
# 版权声明，告知此代码的版权归属及使用许可
# 此处使用了 Apache License, Version 2.0，表示在遵守许可的前提下可以使用此代码
# 详细许可信息可在指定的链接查看
# 如果符合适用法律要求，此软件按"原样"提供，没有任何明示或暗示的担保或条件
# 请查阅许可协议获取更多信息

# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射表，将预训练模型名称映射到其配置文件的 URL 地址
CVT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/cvt-13": "https://huggingface.co/microsoft/cvt-13/resolve/main/config.json",
    # 可以在链接中查看所有的 CvT 模型
}

# CvT 模型的配置类，继承自 PretrainedConfig
class CvtConfig(PretrainedConfig):
    r"""
    这是一个用于存储 [`CvtModel`] 配置的类。它用于根据指定的参数实例化一个 CvT 模型，定义模型的架构。使用默认参数实例化一个配置将产生类似于 CvT [microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    示例:

    ```python
    >>> from transformers import CvtConfig, CvtModel

    >>> # 初始化一个 Cvt msft/cvt 风格的配置
    >>> configuration = CvtConfig()

    >>> # 从 msft/cvt 风格的配置初始化一个（带有随机权重）模型
    >>> model = CvtModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 指定模型类型为 "cvt"
    model_type = "cvt"

    # 初始化方法，设置模型配置的各种参数
    def __init__(
        self,
        num_channels=3,  # 输入图像通道数，默认为 3
        patch_sizes=[7, 3, 3],  # 不同层的 patch 大小列表，默认为 [7, 3, 3]
        patch_stride=[4, 2, 2],  # 不同层的 patch 步幅列表，默认为 [4, 2, 2]
        patch_padding=[2, 1, 1],  # 不同层的 patch 填充列表，默认为 [2, 1, 1]
        embed_dim=[64, 192, 384],  # 不同层的嵌入维度列表，默认为 [64, 192, 384]
        num_heads=[1, 3, 6],  # 不同层的注意力头数列表，默认为 [1, 3, 6]
        depth=[1, 2, 10],  # 不同层的深度列表，默认为 [1, 2, 10]
        mlp_ratio=[4.0, 4.0, 4.0],  # 不同层的 MLP 扩展比例列表，默认为 [4.0, 4.0, 4.0]
        attention_drop_rate=[0.0, 0.0, 0.0],  # 不同层的注意力丢弃率列表，默认为 [0.0, 0.0, 0.0]
        drop_rate=[0.0, 0.0, 0.0],  # 不同层的丢弃率列表，默认为 [0.0, 0.0, 0.0]
        drop_path_rate=[0.0, 0.0, 0.1],  # 不同层的路径丢弃率列表，默认为 [0.0, 0.0, 0.1]
        qkv_bias=[True, True, True],  # 是否对 QKV 进行偏置，默认为 [True, True, True]
        cls_token=[False, False, True],  # 是否在不同层添加类别令牌，默认为 [False, False, True]
        qkv_projection_method=["dw_bn", "dw_bn", "dw_bn"],  # QKV 投影方法列表，默认为 ["dw_bn", "dw_bn", "dw_bn"]
        kernel_qkv=[3, 3, 3],  # QKV 卷积核大小列表，默认为 [3, 3, 3]
        padding_kv=[1, 1, 1],  # KV 填充大小列表，默认为 [1, 1, 1]
        stride_kv=[2, 2, 2],  # KV 步幅大小列表，默认为 [2, 2, 2]
        padding_q=[1, 1, 1],  # Q 填充大小列表，默认为 [1, 1, 1]
        stride_q=[1, 1, 1],  # Q 步幅大小列表，默认为 [1, 1, 1]
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # 层归一化 epsilon，默认为 1e-12
        **kwargs,  # 其他参数
        # 调用父类的初始化方法，传入任意额外参数
        super().__init__(**kwargs)
        # 设置自注意力模块的通道数
        self.num_channels = num_channels
        # 设置自注意力模块的感受野大小列表
        self.patch_sizes = patch_sizes
        # 设置自注意力模块的感受野移动步长
        self.patch_stride = patch_stride
        # 设置自注意力模块的感受野填充大小
        self.patch_padding = patch_padding
        # 设置自注意力模块的嵌入维度
        self.embed_dim = embed_dim
        # 设置自注意力模块的头数
        self.num_heads = num_heads
        # 设置自注意力模块的层数
        self.depth = depth
        # 设置自注意力模块中 MLP 的扩展比例
        self.mlp_ratio = mlp_ratio
        # 设置自注意力模块的注意力丢弃率
        self.attention_drop_rate = attention_drop_rate
        # 设置自注意力模块的总体丢弃率
        self.drop_rate = drop_rate
        # 设置自注意力模块的路径丢弃率
        self.drop_path_rate = drop_path_rate
        # 设置自注意力模块中是否使用偏置项
        self.qkv_bias = qkv_bias
        # 设置自注意力模块的类别标记
        self.cls_token = cls_token
        # 设置自注意力模块中 QKV 投影的方法
        self.qkv_projection_method = qkv_projection_method
        # 设置自注意力模块中 QKV 卷积核大小
        self.kernel_qkv = kernel_qkv
        # 设置自注意力模块中 KV 填充大小
        self.padding_kv = padding_kv
        # 设置自注意力模块中 KV 步长
        self.stride_kv = stride_kv
        # 设置自注意力模块中 Q 填充大小
        self.padding_q = padding_q
        # 设置自注意力模块中 Q 步长
        self.stride_q = stride_q
        # 设置初始化权重范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
```  
```