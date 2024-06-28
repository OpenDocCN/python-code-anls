# `.\models\roformer\configuration_roformer.py`

```py
# 定义了 RoFormer 模型的配置类，继承自 PretrainedConfig，用于存储 RoFormer 模型的配置信息
class RoFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RoFormerModel`]. It is used to instantiate an
    RoFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the RoFormer
    [junnyu/roformer_chinese_base](https://huggingface.co/junnyu/roformer_chinese_base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 RoFormer 模型的配置类，用于配置模型的各种参数
    Args:
        vocab_size (`int`, *optional*, defaults to 50000):
            RoFormer 模型的词汇表大小，定义了可以由输入 `inputs_ids` 表示的不同 token 数量。
        embedding_size (`int`, *optional*, defaults to None):
            编码器层和池化层的维度。如果未提供，则默认为 `hidden_size`。
        hidden_size (`int`, *optional*, defaults to 768):
            编码器层和池化层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer 编码器中隐藏层的数量。
        num_attention_heads (`int`, *optional*, defaults to 12):
            Transformer 编码器中每个注意力层的注意力头数。
        intermediate_size (`int`, *optional*, defaults to 3072):
            Transformer 编码器中 "intermediate"（即前馈）层的维度。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率的 dropout 比例。
        max_position_embeddings (`int`, *optional*, defaults to 1536):
            模型可能使用的最大序列长度。通常设置为一个较大的值（例如 512、1024 或 1536）。
        type_vocab_size (`int`, *optional*, defaults to 2):
            调用 [`RoFormerModel`] 或 [`TFRoFormerModel`] 时传递的 `token_type_ids` 的词汇表大小。
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的截断正态初始化器的标准差。
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            层归一化层使用的 epsilon。
        is_decoder (`bool`, *optional*, defaults to `False`):
            模型是否用作解码器。如果为 `False`，则模型用作编码器。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的键/值注意力（不是所有模型都使用）。仅在 `config.is_decoder=True` 时相关。
        rotary_value (`bool`, *optional*, defaults to `False`):
            是否在值层应用旋转位置嵌入。
    # 初始化一个 RoFormer 风格的配置对象
    configuration = RoFormerConfig()
    
    # 使用 RoFormer 配置对象初始化一个模型，模型的权重是随机初始化的
    model = RoFormerModel(configuration)
    
    # 获取模型的配置信息
    configuration = model.config
# 定义 RoFormer 模型在 ONNX 格式中的配置类，继承自 OnnxConfig 类
class RoFormerOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射，将字符串映射到一个映射（字典），其键为整数，值为字符串
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        
        # 如果任务类型为多选题
        if self.task == "multiple-choice":
            # 动态轴定义，将 0 对应到 "batch"，1 对应到 "choice"，2 对应到 "sequence"
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，动态轴定义，将 0 对应到 "batch"，1 对应到 "sequence"
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 重新赋值动态轴定义，将 0 对应到 "batch"，1 对应到 "sequence"
        dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回有序字典，包含输入名称到动态轴定义的映射
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 输入名称 "input_ids" 对应动态轴定义
                ("attention_mask", dynamic_axis),   # 输入名称 "attention_mask" 对应动态轴定义
                ("token_type_ids", dynamic_axis),   # 输入名称 "token_type_ids" 对应动态轴定义
            ]
        )
```