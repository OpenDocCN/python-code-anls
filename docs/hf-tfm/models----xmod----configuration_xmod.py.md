# `.\transformers\models\xmod\configuration_xmod.py`

```py
# 这是一个配置类,用于存储 XmodModel 的配置信息
class XmodConfig(PretrainedConfig):
    # 这是这个配置类的文档说明,描述了它的用途和使用方法
    r"""
    This is the configuration class to store the configuration of a [`XmodModel`]. It is used to instantiate an X-MOD
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [facebook/xmod-base](https://huggingface.co/facebook/xmod-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import XmodConfig, XmodModel

    >>> # Initializing an X-MOD facebook/xmod-base style configuration
    >>> configuration = XmodConfig()

    >>> # Initializing a model (with random weights) from the facebook/xmod-base style configuration
    >>> model = XmodModel(configuration)

    >>> # Accessing the model configuration



# 定义了一个包含预训练配置的字典
XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # 定义了一些 X-MOD 预训练模型的配置文件地址
    "facebook/xmod-base": "https://huggingface.co/facebook/xmod-base/resolve/main/config.json",
    "facebook/xmod-large-prenorm": "https://huggingface.co/facebook/xmod-large-prenorm/resolve/main/config.json",
    "facebook/xmod-base-13-125k": "https://huggingface.co/facebook/xmod-base-13-125k/resolve/main/config.json",
    "facebook/xmod-base-30-125k": "https://huggingface.co/facebook/xmod-base-30-125k/resolve/main/config.json",
    "facebook/xmod-base-30-195k": "https://huggingface.co/facebook/xmod-base-30-195k/resolve/main/config.json",
    "facebook/xmod-base-60-125k": "https://huggingface.co/facebook/xmod-base-60-125k/resolve/main/config.json",
    "facebook/xmod-base-60-265k": "https://huggingface.co/facebook/xmod-base-60-265k/resolve/main/config.json",
    "facebook/xmod-base-75-125k": "https://huggingface.co/facebook/xmod-base-75-125k/resolve/main/config.json",
    "facebook/xmod-base-75-269k": "https://huggingface.co/facebook/xmod-base-75-269k/resolve/main/config.json",
}
    # 获取模型的配置信息
    >>> configuration = model.config
    
    # 设置模型类型为 "xmod"
    model_type = "xmod"
    
    # 初始化模型的一些超参数
    def __init__(
        self,
        # 词表大小
        vocab_size=30522,
        # 隐藏层大小
        hidden_size=768,
        # 隐藏层数量
        num_hidden_layers=12,
        # 注意力头数量
        num_attention_heads=12,
        # 中间层大小
        intermediate_size=3072,
        # 激活函数
        hidden_act="gelu",
        # 隐藏层dropout概率
        hidden_dropout_prob=0.1,
        # 注意力dropout概率
        attention_probs_dropout_prob=0.1,
        # 最大位置编码长度
        max_position_embeddings=512,
        # 类型词表大小
        type_vocab_size=2,
        # 权重初始化范围
        initializer_range=0.02,
        # LayerNorm的epsilon值
        layer_norm_eps=1e-12,
        # 填充token的ID
        pad_token_id=1,
        # 开始token的ID
        bos_token_id=0,
        # 结束token的ID
        eos_token_id=2,
        # 位置编码类型
        position_embedding_type="absolute",
        # 是否使用cache
        use_cache=True,
        # 分类器dropout
        classifier_dropout=None,
        # 是否使用pre_norm
        pre_norm=False,
        # adapter层的缩减因子
        adapter_reduction_factor=2,
        # 是否在adapter层使用LayerNorm
        adapter_layer_norm=False,
        # 是否重用现有的LayerNorm
        adapter_reuse_layer_norm=True,
        # 是否在adapter层之前使用LayerNorm
        ln_before_adapter=True,
        # 支持的语言列表
        languages=("en_XX",),
        # 默认语言
        default_language=None,
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 设置各种超参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pre_norm = pre_norm
        self.adapter_reduction_factor = adapter_reduction_factor
        self.adapter_layer_norm = adapter_layer_norm
        self.adapter_reuse_layer_norm = adapter_reuse_layer_norm
        self.ln_before_adapter = ln_before_adapter
        self.languages = list(languages)
        self.default_language = default_language
# 从transformers.models.roberta.configuration_roberta.RobertaOnnxConfig复制代码，并将Roberta->Xmod
class XmodOnnxConfig(OnnxConfig):
    # 定义inputs属性，返回字典形式的输入格式
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选，设置动态轴
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则设置另一种动态轴
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典作为模型的输入格式
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```