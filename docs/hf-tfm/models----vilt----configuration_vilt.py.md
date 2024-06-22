# `.\transformers\models\vilt\configuration_vilt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证，可以根据许可证规定使用该文件
# 获取许可证的副本可以访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则根据许可证分发的软件是基于“现状”分发的，没有任何明示或暗示的条件或保证
# 请查阅许可证了解更多关于控制模型输出的信息
# 设置日志记录器
        # 调用父类的初始化方法，并传入参数tie_word_embeddings和kwargs
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        # 初始化词汇表大小、类型词汇表大小、模态类型词汇表大小、最大位置嵌入大小等属性
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size
        self.max_position_embeddings = max_position_embeddings

        # 初始化隐藏层大小、隐藏层数量、注意力头数量、中间层大小、隐藏层激活函数等属性
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # 初始化图像大小、路径大小、通道数量、qkv偏置、最大图像长度、图像数量等属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.max_image_length = max_image_length
        self.num_images = num_images
```