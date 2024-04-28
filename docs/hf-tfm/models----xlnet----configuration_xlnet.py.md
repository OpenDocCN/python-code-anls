# `.\transformers\models\xlnet\configuration_xlnet.py`

```py
# 设置文件编码为 UTF-8

# 导入警告模块
import warnings

# 从配置工具中导入预训练配置模块
from ...configuration_utils import PretrainedConfig

# 从工具中导入日志记录模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 XLNet 预训练配置存档映射
XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlnet-base-cased": "https://huggingface.co/xlnet-base-cased/resolve/main/config.json",
    "xlnet-large-cased": "https://huggingface.co/xlnet-large-cased/resolve/main/config.json",
}

# 定义 XLNet 配置类
class XLNetConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`XLNetModel`] or a [`TFXLNetModel`]. It is used to
    instantiate a XLNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [xlnet-large-cased](https://huggingface.co/xlnet-large-cased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import XLNetConfig, XLNetModel

    >>> # Initializing a XLNet configuration
    >>> configuration = XLNetConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = XLNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 指定模型类型为 "xlnet"
    model_type = "xlnet"

    # 在推断时忽略的键列表
    keys_to_ignore_at_inference = ["mems"]

    # 属性映射
    attribute_map = {
        "n_token": "vocab_size",  # 向后兼容
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 构造 XLNetConfig 类
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
        # 初始化参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        # 如果模型维度不能被注意力头数整除，抛出错误
        if d_model % n_head != 0:
            raise ValueError(f"'d_model % n_head' ({d_model % n_head}) should be equal to 0")
        # 如果参数中有'd_head'关键字，并且其值不等于d_model // n_head，抛出错误
        if "d_head" in kwargs:
            if kwargs["d_head"] != d_model // n_head:
                raise ValueError(
                    f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
                )
        # d_head 是 d_model // n_head
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # 如果kwargs中有'use_cache'关键字，发出警告，建议使用'use_mems_eval'代替
        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated and will be removed in a future version, use `use_mems_eval`"
                " instead.",
                FutureWarning,
            )
            use_mems_eval = kwargs["use_cache"]

        # 设置use_mems_eval和use_mems_train的值
        self.use_mems_eval = use_mems_eval
        self.use_mems_train = use_mems_train
        # 调用父类的__init__方法
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    # 定义max_position_embeddings属性
    @property
    def max_position_embeddings(self):
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        # 返回-1，表示没有序列长度限制
        return -1

    # 实现max_position_embeddings的setter方法
    # 设置最大位置嵌入数值的方法
    def max_position_embeddings(self, value):
        # 抛出未实现错误，指示该模型没有序列长度限制
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )
```