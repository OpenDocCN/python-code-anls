# `.\models\gpt_neo\configuration_gpt_neo.py`

```
# 定义模型配置类 GPTNeoConfig，它继承自 PretrainedConfig，用于存储 GPT Neo 模型的配置信息
class GPTNeoConfig(PretrainedConfig):
    # 类属性：模型类型为 "gpt_neo"
    model_type = "gpt_neo"
    # 在推断时忽略的键列表，包括 "past_key_values"
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将 "num_attention_heads" 映射到 "num_heads"，将 "num_hidden_layers" 映射到 "num_layers"
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    # 初始化函数，用于创建一个新的Transformer模型配置对象
    def __init__(
        self,
        vocab_size=50257,  # 词汇表大小，默认为50257
        max_position_embeddings=2048,  # 最大位置嵌入长度，默认为2048
        hidden_size=2048,  # 隐藏层大小，默认为2048
        num_layers=24,  # Transformer层数，默认为24
        attention_types=[[["global", "local"], 12]],  # 注意力类型及其数量，默认为[['global', 'local'], 12]
        num_heads=16,  # 注意力头数，默认为16
        intermediate_size=None,  # 中间层大小，默认为None
        window_size=256,  # 窗口大小，默认为256
        activation_function="gelu_new",  # 激活函数类型，默认为'gelu_new'
        resid_dropout=0.0,  # 残差连接的dropout概率，默认为0.0
        embed_dropout=0.0,  # 嵌入层dropout概率，默认为0.0
        attention_dropout=0.0,  # 注意力层dropout概率，默认为0.0
        classifier_dropout=0.1,  # 分类器dropout概率，默认为0.1
        layer_norm_epsilon=1e-5,  # Layer Norm层的epsilon值，默认为1e-5
        initializer_range=0.02,  # 初始化范围，默认为0.02
        use_cache=True,  # 是否使用缓存，默认为True
        bos_token_id=50256,  # 起始token的ID，默认为50256
        eos_token_id=50256,  # 结束token的ID，默认为50256
        **kwargs,
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入长度
        self.hidden_size = hidden_size  # 初始化隐藏层大小
        self.num_layers = num_layers  # 初始化Transformer层数
        self.num_heads = num_heads  # 初始化注意力头数
        self.intermediate_size = intermediate_size  # 初始化中间层大小
        self.window_size = window_size  # 初始化窗口大小
        self.activation_function = activation_function  # 初始化激活函数类型
        self.resid_dropout = resid_dropout  # 初始化残差连接的dropout概率
        self.embed_dropout = embed_dropout  # 初始化嵌入层dropout概率
        self.attention_dropout = attention_dropout  # 初始化注意力层dropout概率
        self.classifier_dropout = classifier_dropout  # 初始化分类器dropout概率
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化Layer Norm层的epsilon值
        self.initializer_range = initializer_range  # 初始化初始化范围
        self.use_cache = use_cache  # 初始化是否使用缓存

        self.bos_token_id = bos_token_id  # 初始化起始token的ID
        self.eos_token_id = eos_token_id  # 初始化结束token的ID

        self.attention_types = attention_types  # 初始化注意力类型及其数量
        # 根据注意力类型扩展注意力层参数，并将结果赋值给self.attention_layers
        self.attention_layers = self.expand_attention_types_params(attention_types)

        # 如果注意力层数与Transformer层数不匹配，抛出数值错误异常
        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @staticmethod
    # 静态方法：根据注意力类型参数扩展注意力类型并返回扩展后的列表
    def expand_attention_types_params(attention_types):
        attentions = []  # 创建一个空列表来存放扩展后的注意力类型
        for item in attention_types:  # 遍历注意力类型列表中的每个元素
            for _ in range(item[1]):  # 根据每个元素的数量参数进行扩展
                attentions.extend(item[0])  # 将注意力类型按数量添加到列表中
        return attentions  # 返回扩展后的注意力类型列表
# 定义一个自定义函数custom_unfold，实现类似torch.Tensor.unfold的功能，以便导出到ONNX
def custom_unfold(input, dimension, size, step):
    """Custom torch.Tensor.unfold implementation to enable the export to ONNX."""
    import torch
    
    # 获取输入张量的形状
    shape = input.size()
    # 张量的维度数
    rank = len(shape)
    # 指定维度的大小
    sizedim = shape[dimension]
    
    # 创建一个从0开始，步长为step的索引张量
    low_indices = torch.arange(0, sizedim, step)
    # 计算可以完整切分的最小长度
    min_length = torch.div(sizedim - size, step, rounding_mode="floor") + 1
    # 根据最小长度和步长，生成对应的索引
    indices = torch.arange(size) + low_indices[:min_length][:, None]
    
    # 构建一个切片的索引列表
    s = [slice(None)] * rank
    s[dimension] = indices
    # 根据索引对输入张量进行切片操作
    sliced = input[s]
    
    # 创建一个排列列表，用于对切片后的张量进行维度重排
    perm = list(range(0, rank + 1))
    perm.append(perm.pop(dimension + 1))
    
    # 返回重排后的张量
    return sliced.permute(perm)


# 定义一个自定义函数custom_get_block_length_and_num_blocks，实现GPTNeoAttentionMixin._get_block_length_and_num_blocks的功能
def custom_get_block_length_and_num_blocks(seq_length, window_size):
    """
    Custom implementation for GPTNeoAttentionMixin._get_block_length_and_num_blocks to enable the export to ONNX as
    original implementation uses Python variables and control flow.
    """
    import torch
    
    # 创建一个候选的窗口大小张量
    candidates = torch.arange(1, window_size)
    # 计算序列长度与候选窗口大小的余数
    remainders = torch.remainder(seq_length, candidates)
    # 找到能整除序列长度的候选窗口大小
    divisor_indices = remainders == 0
    divisors = candidates[divisor_indices]
    # 找到最大的能整除序列长度的窗口大小
    largest_divisor = torch.max(divisors)
    # 返回最大的能整除序列长度的窗口大小以及能整除后的块数量
    return largest_divisor, torch.div(seq_length, largest_divisor, rounding_mode="floor")


# 定义一个类GPTNeoOnnxConfig，继承自OnnxConfigWithPast类
class GPTNeoOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义常见的输入映射，包括input_ids和attention_mask
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        # 如果使用过去状态（self.use_past为True），则填充输入映射中的attention_mask
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}
        
        # 返回最终的输入映射
        return common_inputs

    @property
    def num_attention_heads(self) -> int:
        # 返回配置中的注意力头数目
        return self._config.num_heads

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用父类方法生成通用输入
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照 forward() 方法中的顺序排序输入
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 需要添加 past_keys
        if self.use_past:
            # 检查是否安装了 PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # 计算 past_key_values 的长度，不使用相同的长度
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                # 为每一层生成零初始化的 past_key_values 对
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        # 添加 attention_mask 到有序输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        
        # 如果使用 past_keys，则扩展 attention_mask 的长度
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回有序的输入字典
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本号
        return 13
```