# `.\models\gpt2\configuration_gpt2.py`

```
# 导入必要的模块和类
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

# 导入所需的类和函数
from ... import PreTrainedTokenizer, TensorType, is_torch_available
# 导入预训练配置的基类
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 相关的配置类和函数
from ...onnx import OnnxConfigWithPast, PatchingSpec
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置与预训练模型的映射字典
GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
    "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/config.json",
    "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
    "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/config.json",
    "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/config.json",
}

# GPT-2 模型的配置类
class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [gpt2](https://huggingface.co/gpt2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型标识
    model_type = "gpt2"
    # 推断时要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将配置属性映射到标准 GPT-2 参数名
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
```  
    # 初始化方法，设定模型的参数
    def __init__(
        # 词汇表的大小，默认为50257
        self,
        vocab_size=50257,
        # 位置编码的数量，默认为1024
        n_positions=1024,
        # 词嵌入的维度，默认为768
        n_embd=768,
        # Transformer 层的数量，默认为12
        n_layer=12,
        # 注意力头的数量，默认为12
        n_head=12,
        # 内部前馈神经网络的维度，默认为 None
        n_inner=None,
        # 激活函数的类型，默认为 "gelu_new"
        activation_function="gelu_new",
        # 残差连接的丢弃概率，默认为0.1
        resid_pdrop=0.1,
        # 词嵌入的丢弃概率，默认为0.1
        embd_pdrop=0.1,
        # 注意力的丢弃概率，默认为0.1
        attn_pdrop=0.1,
        # 层归一化的 epsilon 值，默认为1e-5
        layer_norm_epsilon=1e-5,
        # 参数初始化范围，默认为0.02
        initializer_range=0.02,
        # 摘要的类型，默认为 "cls_index"
        summary_type="cls_index",
        # 是否使用投影层进行摘要，默认为True
        summary_use_proj=True,
        # 摘要激活函数，默认为 None
        summary_activation=None,
        # 是否在摘要后进行第一次丢弃，默认为0.1
        summary_first_dropout=0.1,
        # 是否将摘要投影到标签，默认为True
        summary_proj_to_labels=True,
        # 是否缩放注意力权重，默认为True
        scale_attn_weights=True,
        # 是否使用缓存，默认为True
        use_cache=True,
        # 起始标记的 id，默认为50256
        bos_token_id=50256,
        # 结束标记的 id，默认为50256
        eos_token_id=50256,
        # 按逆序层索引缩放注意力权重，默认为False
        scale_attn_by_inverse_layer_idx=False,
        # 重新排序并上转注意力，默认为False
        reorder_and_upcast_attn=False,
        # 其它参数
        **kwargs,
    ):
        # 设置模型参数
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        # 设置起始标记 id 和 结束标记 id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 调用父类的初始化方法
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
# 定义继承自OnnxConfigWithPast的GPT2OnnxConfig类
class GPT2OnnxConfig(OnnxConfigWithPast):
    # 初始化方法
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        # 如果配置中的pad_token_id属性为None，则将其设为0
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    # 定义inputs属性
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义常见的输入，包括input_ids和attention_mask
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        # 若使用过去数据，则再添加past_key_values相关的输入
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # 定义num_layers属性
    @property
    def num_layers(self) -> int:
        # 返回配置中的层数n_layer
        return self._config.n_layer

    # 定义num_attention_heads属性
    @property
    def num_attention_heads(self) -> int:
        # 返回配置中的头数n_head
        return self._config.n_head

    # 生成虚拟输入数据的方法
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用父类方法生成常见的输入数据
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照forward()方法中输入的顺序对输入数据进行排序
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 需要添加past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        # 若使用过去数据，则添加相应的mask
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs
    # 定义一个方法，返回默认的ONNX操作集版本号为13
    def default_onnx_opset(self) -> int:
        # 返回操作集版本号
        return 13
```