# `.\models\ibert\configuration_ibert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者信息和版权信息
# 根据 Apache 许可证 2.0 版本，授权使用该文件
# 获取许可证的链接
# 根据适用法律或书面同意，按“原样”分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
# I-BERT 配置
# 导入所需的库和模块
# 导入日志记录工具
# 获取 logger 对象
# 预训练模型配置文件映射
# I-BERT 预训练模型配置文件映射
# I-BERT 预训练模型配置文件映射
# I-BERT 预训练模型配置文件映射
# I-BERT 配置类，继承自 PretrainedConfig
# 用于存储 IBertModel 的配置信息
# 根据指定参数实例化 I-BERT 模型的配置
# 默认情况下实例化一个与 IBERT 相似的配置
# 配置对象继承自 PretrainedConfig，用于控制模型输出
# 阅读 PretrainedConfig 的文档以获取更多信息
# 模型类型为 "ibert"
# 初始化方法，设置默认参数
# 词汇表大小
# 隐藏层大小
# 隐藏层数量
# 注意力头数量
# 中间层大小
# 隐藏层激活函数
# 隐藏层丢弃概率
# 注意力概率丢弃概率
# 最大位置嵌入
# 类型词汇表大小
# 初始化范围
# 层归一化 epsilon
# 填充标记 ID
# 开始标记 ID
# 结束标记 ID
# 位置嵌入类型
# 量化模式
# 强制去量化
        # 调用父类的构造函数，初始化模型参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化模型的词汇表大小
        self.vocab_size = vocab_size
        # 初始化模型的隐藏层大小
        self.hidden_size = hidden_size
        # 初始化模型的隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化模型的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 初始化模型的隐藏层激活函数
        self.hidden_act = hidden_act
        # 初始化模型的中间层大小
        self.intermediate_size = intermediate_size
        # 初始化模型的隐藏层dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 初始化模型的注意力概率dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化模型的最大位置嵌入大小
        self.max_position_embeddings = max_position_embeddings
        # 初始化模型的类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 初始化模型的初始化范围
        self.initializer_range = initializer_range
        # 初始化模型的层归一化epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 初始化模型的位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 初始化模型的量化模式
        self.quant_mode = quant_mode
        # 初始化模型的强制去量化标志
        self.force_dequant = force_dequant
# 定义一个类 IBertOnnxConfig，继承自 OnnxConfig 类
class IBertOnnxConfig(OnnxConfig):
    # 定义一个 inputs 属性，返回一个映射类型，键为字符串，值为映射类型，值的键为整数，值为字符串
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择
        if self.task == "multiple-choice":
            # 动态轴为 {0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含两个键值对，键为字符串，值为动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```