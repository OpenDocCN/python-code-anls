# `.\models\cvt\configuration_cvt.py`

```
# 设置文件编码为 UTF-8

# 导入所需模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# CvT 预训练模型配置文件映射表，指定了预训练模型名称及其对应的配置文件 URL
CVT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/cvt-13": "https://huggingface.co/microsoft/cvt-13/resolve/main/config.json",
    # 可以在 https://huggingface.co/models?filter=cvt 查看所有 CvT 模型
}

# CvTConfig 类继承自 PretrainedConfig 类，用于存储 CvT 模型的配置信息
class CvtConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`CvtModel`] 的配置信息。根据指定的参数实例化一个 CvT 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 CvT [microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    示例:

    ```python
    >>> from transformers import CvtConfig, CvtModel

    >>> # 初始化一个 CvT msft/cvt 风格的配置
    >>> configuration = CvtConfig()

    >>> # 从配置中初始化一个模型（具有随机权重）
    >>> model = CvtModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 指定模型类型为 "cvt"
    model_type = "cvt"

    # 构造函数，初始化 CvTConfig 类的实例，设置模型的各种配置参数
    def __init__(
        self,
        num_channels=3,
        patch_sizes=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embed_dim=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 2, 10],
        mlp_ratio=[4.0, 4.0, 4.0],
        attention_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        cls_token=[False, False, True],
        qkv_projection_method=["dw_bn", "dw_bn", "dw_bn"],
        kernel_qkv=[3, 3, 3],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs,
        ):
        # 调用父类的初始化方法，并传递所有关键字参数
        super().__init__(**kwargs)
        # 设置自身的通道数
        self.num_channels = num_channels
        # 设置自身的补丁大小列表
        self.patch_sizes = patch_sizes
        # 设置自身的补丁步长
        self.patch_stride = patch_stride
        # 设置自身的补丁填充
        self.patch_padding = patch_padding
        # 设置自身的嵌入维度
        self.embed_dim = embed_dim
        # 设置自身的注意力头数
        self.num_heads = num_heads
        # 设置自身的层数
        self.depth = depth
        # 设置自身的MLP放大比例
        self.mlp_ratio = mlp_ratio
        # 设置自身的注意力机制中的注意力丢弃率
        self.attention_drop_rate = attention_drop_rate
        # 设置自身的全连接层丢弃率
        self.drop_rate = drop_rate
        # 设置自身的路径丢弃率
        self.drop_path_rate = drop_path_rate
        # 设置自身的qkv偏置
        self.qkv_bias = qkv_bias
        # 设置自身的类令牌
        self.cls_token = cls_token
        # 设置自身的qkv投影方法
        self.qkv_projection_method = qkv_projection_method
        # 设置自身的qkv内核
        self.kernel_qkv = kernel_qkv
        # 设置自身的kv填充
        self.padding_kv = padding_kv
        # 设置自身的kv步幅
        self.stride_kv = stride_kv
        # 设置自身的q填充
        self.padding_q = padding_q
        # 设置自身的q步幅
        self.stride_q = stride_q
        # 设置自身的初始化器范围
        self.initializer_range = initializer_range
        # 设置自身的层归一化epsilon
        self.layer_norm_eps = layer_norm_eps
```