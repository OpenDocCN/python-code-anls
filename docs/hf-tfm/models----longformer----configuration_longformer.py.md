# `.\transformers\models\longformer\configuration_longformer.py`

```
# 设置文件编码为 utf-8

# 版权声明，包括版权声明和许可证信息

# 引入所需的模块和类型

# 如果是类型检查，引入相应的模块

# 获取 logger 对象用于日志记录

# 预训练模型配置文件的映射字典

# LongformerConfig 类，用于存储 Longformer 模型的配置

# 预定义的 LongformerConfig 类
    # 定义了一个函数参数列表，包括 Longformer 模型的各种参数设置
        Args:
            vocab_size (`int`, *optional*, defaults to 30522):
                Longformer 模型的词汇表大小，定义了输入 `inputs_ids` 可以表示的不同 token 数量。
            hidden_size (`int`, *optional*, defaults to 768):
                编码器层和池化层的维度。
            num_hidden_layers (`int`, *optional*, defaults to 12):
                Transformer 编码器中的隐藏层数量。
            num_attention_heads (`int`, *optional*, defaults to 12):
                Transformer 编码器中每个注意力层的注意力头数量。
            intermediate_size (`int`, *optional*, defaults to 3072):
                Transformer 编码器中“中间”（通常称为前馈）层的维度。
            hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                嵌入层、编码器和池化器中所有全连接层的 dropout 概率。        
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                注意力概率的 dropout 比率。
            max_position_embeddings (`int`, *optional*, defaults to 512):
                此模型可能的最大序列长度。通常将其设置为非常大的值 (e.g., 512, 1024, 2048)。
            type_vocab_size (`int`, *optional*, defaults to 2):
                调用 [`LongformerModel`] 或 [`TFLongformerModel`] 时传递的 `token_type_ids` 的词汇表大小。
            initializer_range (`float`, *optional*, defaults to 0.02):
                用于初始化所有权重矩阵的截断正态初始化器的标准差。
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                层归一化层使用的 epsilon。
            attention_window (`int` or `List[int]`, *optional*, defaults to 512):
                每个 token 周围的注意力窗口大小。如果是一个整数，对所有层使用相同大小。要为每个层指定不同的窗口大小，可以使用长度为 `num_hidden_layers` 的 `List[int]` 。
    
        Example:
    
        ```python
        >>> from transformers import LongformerConfig, LongformerModel
    
        >>> # Initializing a Longformer configuration
        >>> configuration = LongformerConfig()
    
        >>> # Initializing a model from the configuration
        >>> model = LongformerModel(configuration)
    
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    
    # 将模型类型赋值为 "longformer"
        model_type = "longformer"
    def __init__(
        self,
        attention_window: Union[List[int], int] = 512, # 初始化注意力窗口大小，可以是单个整数或整数列表，默认为512
        sep_token_id: int = 2, # 分隔符 token 的 id，默认为2
        pad_token_id: int = 1, # 填充 token 的 id，默认为1
        bos_token_id: int = 0, # 文本起始 token 的 id，默认为0
        eos_token_id: int = 2, # 文本结束 token 的 id，默认为2
        vocab_size: int = 30522, # 词汇表大小，默认为30522
        hidden_size: int = 768, # 隐藏层大小，默认为768
        num_hidden_layers: int = 12, # 隐藏层的层数，默认为12
        num_attention_heads: int = 12, # 注意力头的个数，默认为12
        intermediate_size: int = 3072, # 中间层大小，默认为3072
        hidden_act: str = "gelu", # 隐藏层激活函数，默认为"gelu"
        hidden_dropout_prob: float = 0.1, # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob: float = 0.1, # 注意力部分的dropout概率，默认为0.1
        max_position_embeddings: int = 512, # 最大位置嵌入长度，默认为512
        type_vocab_size: int = 2, # 类型词汇表大小，默认为2
        initializer_range: float = 0.02, # 初始化范围，默认为0.02
        layer_norm_eps: float = 1e-12, # 层归一化的 epsilon，默认为1e-12
        onnx_export: bool = False, # 是否导出为 onnx 格式，默认为False
        **kwargs,
    ):
        """Constructs LongformerConfig.""" # 构造 LongformerConfig 类
        super().__init__(pad_token_id=pad_token_id, **kwargs) # 调用父类的初始化方法，传递填充 token id 和其它关键字参数

        self.attention_window = attention_window # 初始化对象的注意力窗口属性
        self.sep_token_id = sep_token_id # 初始化对象的分隔符 token id 属性
        self.bos_token_id = bos_token_id # 初始化对象的文本起始 token id 属性
        self.eos_token_id = eos_token_id # 初始化对象的文本结束 token id 属性
        self.vocab_size = vocab_size # 初始化对象的词汇表大小属性
        self.hidden_size = hidden_size # 初始化对象的隐藏层大小属性
        self.num_hidden_layers = num_hidden_layers # 初始化对象的隐藏层层数属性
        self.num_attention_heads = num_attention_heads # 初始化对象的注意力头个数属性
        self.hidden_act = hidden_act # 初始化对象的隐藏层激活函数属性
        self.intermediate_size = intermediate_size # 初始化对象的中间层大小属性
        self.hidden_dropout_prob = hidden_dropout_prob # 初始化对象的隐藏层的dropout概率属性
        self.attention_probs_dropout_prob = attention_probs_dropout_prob # 初始化对象的注意力部分的dropout概率属性
        self.max_position_embeddings = max_position_embeddings # 初始化对象的最大位置嵌入长度属性
        self.type_vocab_size = type_vocab_size # 初始化对象的类型词汇表大小属性
        self.initializer_range = initializer_range # 初始化对象的初始化范围属性
        self.layer_norm_eps = layer_norm_eps # 初始化对象的层归一化的epsilon属性
        self.onnx_export = onnx_export # 初始化对象的是否导出为 onnx 格式属性
# 定义一个 LongformerOnnxConfig 类，继承自 OnnxConfig 类
class LongformerOnnxConfig(OnnxConfig):
    # 初始化方法，接受配置对象、任务字符串和修补规范列表作为参数
    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: "List[PatchingSpec]" = None):
        # 调用父类的初始化方法
        super().__init__(config, task, patching_specs)
        # 设置配置对象的 onnx_export 属性为 True
        config.onnx_export = True

    # 输入属性，返回输入名称到动态轴的映射字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为多项选择
        if self.task == "multiple-choice":
            # 设置动态轴字典为 {0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则设置动态轴字典为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典，包含输入名称到动态轴的映射
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("global_attention_mask", dynamic_axis),
            ]
        )

    # 输出属性，返回输出名称到动态轴的映射字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 调用父类的输出属性
        outputs = super().outputs
        # 如果任务为默认任务
        if self.task == "default":
            # 添加额外的输出键值对，"pooler_output" 对应动态轴为 {0: "batch"}
            outputs["pooler_output"] = {0: "batch"}
        # 返回输出字典
        return outputs

    # 用于验证的绝对容差值属性
    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        # 返回 1e-4，绝对容差值
        return 1e-4

    # 默认的 ONNX 操作集属性
    @property
    def default_onnx_opset(self) -> int:
        # 需要大于等于 14 来支持 tril 操作符
        return max(super().default_onnx_opset, 14)

    # 生成虚拟输入的方法
    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用父类的生成虚拟输入方法
        inputs = super().generate_dummy_inputs(
            preprocessor=tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        # 导入 torch 模块
        import torch

        # 由于某种原因，使用以下代码替换会导致导出过程随机失败：
        # inputs["global_attention_mask"] = torch.randint(2, inputs["input_ids"].shape, dtype=torch.int64)
        # 将全局注意力掩码设置为与输入 "input_ids" 形状相同的零张量
        inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
        # 将每隔一个标记设为全局
        inputs["global_attention_mask"][:, ::2] = 1

        # 返回输入字典
        return inputs
```