# `.\models\xlnet\configuration_xlnet.py`

```
# coding=utf-8
# 文件编码声明为 UTF-8

# XLNet 配置模块的版权声明和许可证信息

# 引入警告模块，用于在特定情况下生成警告信息
import warnings

# 从 transformers 库中引入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig

# 从 transformers 库中引入日志记录工具 logging
from ...utils import logging

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典，将模型名称映射到对应的配置文件 URL
XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlnet/xlnet-base-cased": "https://huggingface.co/xlnet/xlnet-base-cased/resolve/main/config.json",
    "xlnet/xlnet-large-cased": "https://huggingface.co/xlnet/xlnet-large-cased/resolve/main/config.json",
}

# 定义 XLNetConfig 类，用于存储 XLNet 模型的配置信息
class XLNetConfig(PretrainedConfig):
    """
    这是一个配置类，用于存储 [`XLNetModel`] 或 [`TFXLNetModel`] 的配置信息。根据指定的参数实例化一个 XLNet 模型，
    定义模型的架构。使用默认参数实例化配置对象将得到与 [xlnet/xlnet-large-cased](https://huggingface.co/xlnet/xlnet-large-cased)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import XLNetConfig, XLNetModel

    >>> # 初始化一个 XLNet 配置对象
    >>> configuration = XLNetConfig()

    >>> # 使用配置对象初始化一个模型（随机权重）
    >>> model = XLNetModel(configuration)

    >>> # 访问模型的配置信息
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "xlnet"
    model_type = "xlnet"

    # 推理阶段忽略的键列表，这些键不参与推理阶段的处理
    keys_to_ignore_at_inference = ["mems"]

    # 属性映射字典，将旧属性名映射到新属性名，用于向后兼容性
    attribute_map = {
        "n_token": "vocab_size",       # 词汇表大小，向后兼容
        "hidden_size": "d_model",      # 隐藏层大小
        "num_attention_heads": "n_head",  # 注意力头的数量
        "num_hidden_layers": "n_layer",   # 隐藏层的数量
    }
    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,
        n_layer=24,
        n_head=16,
        d_inner=4096,
        ff_activation="gelu",
        untie_r=True,
        attn_type="bi",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        dropout=0.1,
        mem_len=512,
        reuse_len=None,
        use_mems_eval=True,
        use_mems_train=False,
        bi_data=False,
        clamp_len=-1,
        same_length=False,
        summary_type="last",
        summary_use_proj=True,
        summary_activation="tanh",
        summary_last_dropout=0.1,
        start_n_top=5,
        end_n_top=5,
        pad_token_id=5,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        """Constructs XLNetConfig."""

        # 初始化 XLNetConfig 对象，设置各个参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head

        # 检查 d_model 是否能整除 n_head
        if d_model % n_head != 0:
            raise ValueError(f"'d_model % n_head' ({d_model % n_head}) should be equal to 0")

        # 检查 kwargs 中是否存在 d_head 参数，并且其值是否等于 d_model // n_head
        if "d_head" in kwargs:
            if kwargs["d_head"] != d_model // n_head:
                raise ValueError(
                    f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
                )

        # 计算并设置 d_head 的值
        self.d_head = d_model // n_head

        # 设置激活函数类型
        self.ff_activation = ff_activation
        # 设置内部维度大小
        self.d_inner = d_inner
        # 是否解绑 r
        self.untie_r = untie_r
        # 设置注意力类型
        self.attn_type = attn_type

        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps

        # 设置 dropout 概率
        self.dropout = dropout
        # 设置记忆长度
        self.mem_len = mem_len
        # 设置重复长度
        self.reuse_len = reuse_len
        # 是否使用双向数据
        self.bi_data = bi_data
        # 设置 clamp 长度
        self.clamp_len = clamp_len
        # 是否具有相同长度
        self.same_length = same_length

        # 设置摘要类型
        self.summary_type = summary_type
        # 是否使用投影层
        self.summary_use_proj = summary_use_proj
        # 摘要激活函数类型
        self.summary_activation = summary_activation
        # 最后一层摘要的 dropout 概率
        self.summary_last_dropout = summary_last_dropout
        # 开始 top-n
        self.start_n_top = start_n_top
        # 结束 top-n
        self.end_n_top = end_n_top

        # 设置开始、填充、结束 token 的 ID
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # 检查 kwargs 中是否存在 use_cache 参数，如果存在则警告将被弃用
        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems_eval`"
                " instead.",
                FutureWarning,
            )
            # 将 use_cache 的值赋给 use_mems_eval
            use_mems_eval = kwargs["use_cache"]

        # 设置是否在评估中使用记忆
        self.use_mems_eval = use_mems_eval
        # 设置是否在训练中使用记忆
        self.use_mems_train = use_mems_train

        # 调用父类初始化方法，传递 pad_token_id、bos_token_id、eos_token_id 以及其它 kwargs
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def max_position_embeddings(self):
        # 输出日志信息，指出该模型没有序列长度限制
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        # 返回 -1 表示没有序列长度限制
        return -1

    @max_position_embeddings.setter
    # 定义一个方法 max_position_embeddings，接受参数 value
    def max_position_embeddings(self, value):
        # 抛出 NotImplementedError 异常，说明方法尚未实现
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )
```