# `.\models\gptj\configuration_gptj.py`

```
# 设置文件编码为 UTF-8
# 版权声明

# 导入必要的库
from collections import OrderedDict
from typing import Any, List, Mapping, Optional
# 导入 Hugging Face 库
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# GPTJ 预训练模型配置文件的存档映射
GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-j-6B": "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/config.json",
    # 查看所有 GPT-J 模型的存档映射
}


# GPTJ 配置类，用于存储 GPT-J 模型的配置信息
class GPTJConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTJModel`]. It is used to instantiate a GPT-J
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GPT-J
    [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) architecture. Configuration objects inherit from
    [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.
    # 定义一个函数参数列表，包括参数的类型和默认值，并提供简短的描述
    Args:
        vocab_size (`int`, *optional*, defaults to 50400): GPT-J 模型的词汇表大小，默认为 50400，表示模型可以表示的不同 token 的数量。
        n_positions (`int`, *optional*, defaults to 2048): 此模型可能使用的最大序列长度，默认为 2048。通常设置为较大的值（例如 512、1024 或 2048）。
        n_embd (`int`, *optional*, defaults to 4096): 嵌入和隐藏状态的维度，默认为 4096。
        n_layer (`int`, *optional*, defaults to 28): Transformer 编码器中的隐藏层数量，默认为 28。
        n_head (`int`, *optional*, defaults to 16): Transformer 编码器中每个注意力层的注意力头数量，默认为 16。
        rotary_dim (`int`, *optional*, defaults to 64): Apply Rotary Position Embedding 的嵌入中的维度数量，默认为 64。
        n_inner (`int`, *optional*, defaults to None): 内部前馈层的维度，默认为 4 倍的 n_embd。
        activation_function (`str`, *optional*, defaults to `"gelu_new"`): 激活函数，在列表 `["relu", "silu", "gelu", "tanh", "gelu_new"]` 中选择，默认为 "gelu_new"。
        resid_pdrop (`float`, *optional*, defaults to 0.1): 嵌入、编码器和池化器中所有全连接层的 dropout 概率，默认为 0.1。
        embd_pdrop (`int`, *optional*, defaults to 0.1): 嵌入的 dropout 比率，默认为 0.1。
        attn_pdrop (`float`, *optional*, defaults to 0.1): 注意力的 dropout 比率，默认为 0.1。
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5): 层标准化层中使用的 epsilon 值，默认为 1e-5。
        initializer_range (`float`, *optional*, defaults to 0.02): 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为 0.02。
        use_cache (`bool`, *optional*, defaults to `True`): 模型是否应返回最后的 key/values 注意力（并非所有模型都使用）。
    
    Example:
        示例代码的用法示例，使用 GPTJModel 和 GPTJConfig 初始化模型并访问配置。
    
    >>> from transformers import GPTJModel, GPTJConfig
    
    >>> # 初始化 GPT-J 6B 配置
    >>> configuration = GPTJConfig()
    
    >>> # 从配置初始化模型
    >>> model = GPTJModel(configuration)
    
    >>> # 访问模型配置
    >>> configuration = model.config
    
    
    # 定义一些额外的模型属性
    model_type = "gptj": 模型类型为 "gptj"
    attribute_map = {
        "max_position_embeddings": "n_positions", 用于映射模型的最大位置嵌入到参数 n_positions
        "hidden_size": "n_embd", 用于映射模型的隐藏大小到参数 n_embd
        "num_attention_heads": "n_head", 用于映射模型的注意力头数量到参数 n_head
        "num_hidden_layers": "n_layer", 用于映射模型的隐藏层数量到参数 n_layer
    }
    # Transformer 模型的初始化函数
    def __init__(
        self,
        vocab_size=50400,  # 词汇表大小，默认为 50400
        n_positions=2048,  # 序列长度，默认为 2048
        n_embd=4096,  # 词嵌入维度，默认为 4096
        n_layer=28,  # Transformer 层数，默认为 28
        n_head=16,  # 注意力头数，默认为 16
        rotary_dim=64,  # 旋转器维度，默认为 64
        n_inner=None,  # 内部隐藏层维度，默认为 None
        activation_function="gelu_new",  # 激活函数，默认为 "gelu_new"
        resid_pdrop=0.0,  # 残差连接丢弃率，默认为 0.0
        embd_pdrop=0.0,  # 词嵌入丢弃率，默认为 0.0
        attn_pdrop=0.0,  # 注意力丢弃率，默认为 0.0
        layer_norm_epsilon=1e-5,  # LayerNormalization 中的 epsilon，默认为 1e-5
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        use_cache=True,  # 是否使用缓存，默认为 True
        bos_token_id=50256,  # 起始符号的标识，默认为 50256
        eos_token_id=50256,  # 结束符号的标识，默认为 50256
        tie_word_embeddings=False,  # 是否共享词嵌入权重，默认为 False
        **kwargs,
    ):
        # 将参数赋值给对象属性
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
    
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
    
        # 调用父类的初始化函数
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
# 从transformers.models.gpt2.configuration_gpt2.GPT2OnnxConfig中复制代码
class GPTJOnnxConfig(OnnxConfigWithPast):
    # 初始化方法，接受预训练的配置对象，任务名称，默认的参数补丁列表，以及是否使用过去的值
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        # 如果配置对象中不存在pad_token_id属性，则将其设置为0
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    # 返回输入相关的信息
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 公共的输入包括input_ids和attention_mask
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        # 如果使用过去的值，填充past相关的信息到common_inputs中
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}
        return common_inputs

    # 返回层数
    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    # 返回注意力头数
    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    # 生成虚拟的输入
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用父类的方法生成常见的虚拟输入
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照 forward() 方法中输入的顺序对输入进行排序
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 需要添加过去键值（past_keys）
        if self.use_past:
            # 如果没有安装 PyTorch，则无法生成虚拟的 past_keys 输入
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # 过去键值的长度不同
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                # 为每一层创建过去键值的零张量
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        # 将 attention_mask 添加到已排序的输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        # 如果使用 past_keys，则修改 attention_mask 的形状
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回已排序的输入
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本
        return 13
```