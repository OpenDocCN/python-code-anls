# `.\models\deprecated\transfo_xl\configuration_transfo_xl.py`

```py
# 设置编码格式为 UTF-8
# 版权声明以及版权信息，包括作者和团队信息
# 获取 Apache 许可证 2.0 的副本链接
# 除非符合许可证规定，否则不能使用该文件
# 在没有任何形式的保证或条件下分发软件
# 查看许可证以获取有关语言的限制和权利的详细信息
""" Transformer XL 配置"""

# 导入所需类和函数
# 配置文件预训练映射字典，通过模型名称映射到配置文件的解析链接
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练配置存档映射
TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "transfo-xl-wt103": "https://huggingface.co/transfo-xl-wt103/resolve/main/config.json",
}

# TransfoXL 模型的配置类
class TransfoXLConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`TransfoXLModel`] or a [`TFTransfoXLModel`]. It is
    used to instantiate a Transformer-XL model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the TransfoXL
    [transfo-xl-wt103](https://huggingface.co/transfo-xl-wt103) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import TransfoXLConfig, TransfoXLModel

    >>> # Initializing a Transformer XL configuration
    >>> configuration = TransfoXLConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = TransfoXLModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""
    # 模型类型为 transfo-xl
    model_type = "transfo-xl"
    # 推理时需要忽略的键
    keys_to_ignore_at_inference = ["mems"]
    # 属性映射字典，将配置属性名映射到相应的名称
    attribute_map = {
        "n_token": "vocab_size",
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 初始化函数，设置模型超参数的默认值
    def __init__(
        self,
        vocab_size=267735,
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        n_head=16,
        d_head=64,
        d_inner=4096,
        div_val=4,
        pre_lnorm=False,
        n_layer=18,
        mem_len=1600,
        clamp_len=1000,
        same_length=True,
        proj_share_all_but_first=True,
        attn_type=0,
        sample_softmax=-1,
        adaptive=True,
        dropout=0.1,
        dropatt=0.0,
        untie_r=True,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        eos_token_id=0,
        **kwargs,
    ):
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 初始化截断点列表为空
        self.cutoffs = []
        # 将传入的截断点列表加入模型的截断点列表中
        self.cutoffs.extend(cutoffs)
        # 如果proj_share_all_but_first为True，则表示层间参数共享，否则不共享
        if proj_share_all_but_first:
            # 构建参数共享列表，第一个元素为False，其余元素均为True
            self.tie_projs = [False] + [True] * len(self.cutoffs)
        else:
            # 构建参数不共享列表，所有元素均为False
            self.tie_projs = [False] + [False] * len(self.cutoffs)
        
        # 设置模型的其他超参数
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_head = d_head
        self.d_inner = d_inner
        self.div_val = div_val
        self.pre_lnorm = pre_lnorm
        self.n_layer = n_layer
        self.n_head = n_head
        self.mem_len = mem_len
        self.same_length = same_length
        self.attn_type = attn_type
        self.clamp_len = clamp_len
        self.sample_softmax = sample_softmax
        self.adaptive = adaptive
        self.dropout = dropout
        self.dropatt = dropatt
        self.untie_r = untie_r
        self.init = init
        self.init_range = init_range
        self.proj_init_std = proj_init_std
        self.init_std = init_std
        self.layer_norm_epsilon = layer_norm_epsilon
        # 调用父类的初始化函数
        super().__init__(eos_token_id=eos_token_id, **kwargs)

    @property
    def max_position_embeddings(self):
        # 获取最大位置编码长度，并记录日志
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        # 返回-1表示没有长度限制
        return -1

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        # 设置最大位置编码长度会抛出异常
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )
```