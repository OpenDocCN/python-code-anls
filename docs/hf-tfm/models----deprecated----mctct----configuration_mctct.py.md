# `.\models\deprecated\mctct\configuration_mctct.py`

```py
# 定义MCTCTConfig类，用于存储M-CTC-T模型的配置，并用于实例化M-CTC-T模型
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
    ```py"""

    # 模型类型为mctct
    model_type = "mctct"

    # 初始化MCTCTConfig对象
    def __init__(
        self,
        vocab_size=8065,  # 词汇表大小
        hidden_size=1536,  # 隐藏层的尺寸
        num_hidden_layers=36,  # 隐藏层的数量
        intermediate_size=6144,  # 中间层的尺寸
        num_attention_heads=4,  # 注意力头的数量
        attention_head_dim=384,  # 注意力头的维度
        max_position_embeddings=920,  # 最大位置嵌入数量
        layer_norm_eps=1e-5,  # 层归一化的epsilon值
        layerdrop=0.3,  # 层dropout的概率
        hidden_act="relu",  # 隐藏层的激活函数
        initializer_range=0.02,  # 初始化权重的范围
        hidden_dropout_prob=0.3,  # 隐藏层的dropout概率
        attention_probs_dropout_prob=0.3,  # 注意力层的dropout概率
        pad_token_id=1,  # 填充符号的标识符
        bos_token_id=0,  # 起始符号的标识符
        eos_token_id=2,  # 终止符号的标识符
        conv_glu_dim=1,  # 卷积门限的维度
        conv_dropout=0.3,  # 卷积层的dropout概率
        num_conv_layers=1,  # 卷积层数
        conv_kernel=(7,),  # 卷积核大小
        conv_stride=(3,),  # 卷积步长
        input_feat_per_channel=80,  # 每个通道的输入特征大小
        input_channels=1,  # 输入通道数
        conv_channels=None,  # 卷积层的通道数
        ctc_loss_reduction="sum",  # CTC损失的规约方式
        ctc_zero_infinity=False,  # CTC的零极限
        **kwargs,
    # 调用父类的初始化方法，传入kwargs以及额外的参数pad_token_id、bos_token_id、eos_token_id
    super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    # 初始化词汇表大小
    self.vocab_size = vocab_size
    # 初始化隐藏层大小
    self.hidden_size = hidden_size
    # 初始化隐藏层数量
    self.num_hidden_layers = num_hidden_layers
    # 初始化中间层大小
    self.intermediate_size = intermediate_size
    # 初始化注意力头数量
    self.num_attention_heads = num_attention_heads
    # 初始化注意力头维度
    self.attention_head_dim = attention_head_dim
    # 初始化最大位置嵌入
    self.max_position_embeddings = max_position_embeddings
    # 初始化层标准化阈值
    self.layer_norm_eps = layer_norm_eps
    # 初始化层丢弃率
    self.layerdrop = layerdrop
    # 初始化隐藏层激活函数
    self.hidden_act = hidden_act
    # 初始化初始化范围
    self.initializer_range = initializer_range
    # 初始化隐藏层丢弃率
    self.hidden_dropout_prob = hidden_dropout_prob
    # 初始化注意力机制丢弃率
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    # 初始化填充标记ID
    self.pad_token_id = pad_token_id
    # 初始化起始标记ID
    self.bos_token_id = bos_token_id
    # 初始化结束标记ID
    self.eos_token_id = eos_token_id
    # 初始化卷积GLU维度
    self.conv_glu_dim = conv_glu_dim
    # 初始化卷积丢弃率
    self.conv_dropout = conv_dropout
    # 初始化卷积层数
    self.num_conv_layers = num_conv_layers
    # 初始化每个通道的输入特征
    self.input_feat_per_channel = input_feat_per_channel
    # 初始化输入通道数
    self.input_channels = input_channels
    # 初始化卷积通道数
    self.conv_channels = conv_channels
    # 初始化CTC损失缩减方式
    self.ctc_loss_reduction = ctc_loss_reduction
    # 初始化CTC零无穷
    self.ctc_zero_infinity = ctc_zero_infinity

    # 防止配置测试失败并导出为JSON
    # 将卷积核大小和卷积步幅转换为列表形式
    self.conv_kernel = list(conv_kernel)
    self.conv_stride = list(conv_stride)

    # 检查卷积模块的配置是否正确
    if len(self.conv_kernel) != self.num_conv_layers:
        # 如果卷积核列表长度不等于卷积层数，抛出数值错误
        raise ValueError(
            "Configuration for convolutional module is incorrect. "
            "It is required that `len(config.conv_kernel)` == `config.num_conv_layers` "
            f"but is `len(config.conv_kernel) = {len(self.conv_kernel)}`, "
            f"`config.num_conv_layers = {self.num_conv_layers}`."
        )
```