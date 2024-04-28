# `.\transformers\models\luke\configuration_luke.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 使用 Apache 许可证版本 2.0，除非遵循许可证，否则不得使用此文件
# 获取许可证的副本
# 基于 LUKE 的预训练配置
从...导入预训练配置，预训练配置用于存储 [`LukeModel`] 的配置，根据指定参数实例化 LUKE 模型，定义模型架构
使用默认值实例化配置将生成与 LUKE [studio-ousia/luke-base](https://huggingface.co/studio-ousia/luke-base) 架构相似的配置
配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出，有关更多信息，请阅读 [`PretrainedConfig`] 的文档


示例：


从transformers导入LukeConfig，LukeModel

# 初始化 LUKE 配置
配置 = LukeConfig()

# 从配置初始化模型
模型 = LukeModel(配置)

# 访问模型配置
配置 = 模型.config
    ):
        """Constructs LukeConfig."""
        # 调用父类的构造方法初始化配置
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置实体词汇表大小
        self.entity_vocab_size = entity_vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置实体嵌入大小
        self.entity_emb_size = entity_emb_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的丢弃率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力矩阵的丢弃率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置嵌入
        self.max_position_embeddings = max_position_embeddings
        # 设置类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用实体感知的注意力机制
        self.use_entity_aware_attention = use_entity_aware_attention
        # 设置分类器的丢弃率
        self.classifier_dropout = classifier_dropout
```