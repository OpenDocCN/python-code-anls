# `.\models\deprecated\mctct\configuration_mctct.py`

```
# 设置编码格式为 UTF-8

# 导入必要的模块和类
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的下载映射
MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "speechbrain/m-ctc-t-large": "https://huggingface.co/speechbrain/m-ctc-t-large/resolve/main/config.json",
    # 查看所有 M-CTC-T 模型的链接地址：https://huggingface.co/models?filter=mctct
}

# MCTCTConfig 类，用于存储 M-CTC-T 模型的配置信息
class MCTCTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MCTCTModel`]. It is used to instantiate an
    M-CTC-T model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the M-CTC-T
    [speechbrain/m-ctc-t-large](https://huggingface.co/speechbrain/m-ctc-t-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import MCTCTConfig, MCTCTModel

    >>> # Initializing a M-CTC-T mctct-large style configuration
    >>> configuration = MCTCTConfig()

    >>> # Initializing a model (with random weights) from the mctct-large style configuration
    >>> model = MCTCTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 M-CTC-T
    model_type = "mctct"

    # 初始化方法，设置模型的各种参数
    def __init__(
        self,
        vocab_size=8065,
        hidden_size=1536,
        num_hidden_layers=36,
        intermediate_size=6144,
        num_attention_heads=4,
        attention_head_dim=384,
        max_position_embeddings=920,
        layer_norm_eps=1e-5,
        layerdrop=0.3,
        hidden_act="relu",
        initializer_range=0.02,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        conv_glu_dim=1,
        conv_dropout=0.3,
        num_conv_layers=1,
        conv_kernel=(7,),
        conv_stride=(3,),
        input_feat_per_channel=80,
        input_channels=1,
        conv_channels=None,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            layerdrop=layerdrop,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            conv_glu_dim=conv_glu_dim,
            conv_dropout=conv_dropout,
            num_conv_layers=num_conv_layers,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            input_feat_per_channel=input_feat_per_channel,
            input_channels=input_channels,
            conv_channels=conv_channels,
            ctc_loss_reduction=ctc_loss_reduction,
            ctc_zero_infinity=ctc_zero_infinity,
            **kwargs,
        )
        ):
            # 调用父类的构造函数，传递所有关键字参数，并设置特定的标记 ID
            super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
            # 初始化模型配置参数
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.intermediate_size = intermediate_size
            self.num_attention_heads = num_attention_heads
            self.attention_head_dim = attention_head_dim
            self.max_position_embeddings = max_position_embeddings
            self.layer_norm_eps = layer_norm_eps
            self.layerdrop = layerdrop
            self.hidden_act = hidden_act
            self.initializer_range = initializer_range
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.conv_glu_dim = conv_glu_dim
            self.conv_dropout = conv_dropout
            self.num_conv_layers = num_conv_layers
            self.input_feat_per_channel = input_feat_per_channel
            self.input_channels = input_channels
            self.conv_channels = conv_channels
            self.ctc_loss_reduction = ctc_loss_reduction
            self.ctc_zero_infinity = ctc_zero_infinity

            # 防止配置测试失败并导出为 JSON
            self.conv_kernel = list(conv_kernel)  # 将卷积核大小转换为列表
            self.conv_stride = list(conv_stride)  # 将卷积步长转换为列表

            # 检查卷积核大小与卷积层数是否匹配，如果不匹配则抛出错误
            if len(self.conv_kernel) != self.num_conv_layers:
                raise ValueError(
                    "Configuration for convolutional module is incorrect. "
                    "It is required that `len(config.conv_kernel)` == `config.num_conv_layers` "
                    f"but is `len(config.conv_kernel) = {len(self.conv_kernel)}`, "
                    f"`config.num_conv_layers = {self.num_conv_layers}`."
                )
```