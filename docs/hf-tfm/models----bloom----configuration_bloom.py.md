# `.\transformers\models\bloom\configuration_bloom.py`

```
# 设置文件编码为UTF-8
# 版权声明和许可协议
""" Bloom配置"""
# 导入所需的模块和类型提示
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional

from packaging import version

# 如果类型检查为真，则导入必要的模块
if TYPE_CHECKING:
    from ... import PreTrainedTokenizer, TensorType

# 导入预训练配置类和其他必要模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import is_torch_available, logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练配置文件的映射字典
BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bigscience/bloom": "https://huggingface.co/bigscience/bloom/resolve/main/config.json",
    "bigscience/bloom-560m": "https://huggingface.co/bigscience/bloom-560m/blob/main/config.json",
    "bigscience/bloom-1b1": "https://huggingface.co/bigscience/bloom-1b1/blob/main/config.json",
    "bigscience/bloom-1b7": "https://huggingface.co/bigscience/bloom-1b7/blob/main/config.json",
    "bigscience/bloom-3b": "https://huggingface.co/bigscience/bloom-3b/blob/main/config.json",
    "bigscience/bloom-7b1": "https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json",
}

# Bloom配置类，继承自预训练配置类
class BloomConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://huggingface.co/bigscience/bloom).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 250880):
            Vocabulary size of the Bloom model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`BloomModel`]. Check [this
            discussion](https://huggingface.co/bigscience/bloom/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
            `slow_but_exact=True`.
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
            merging the TP rank tensors, due to slicing operations the results may be slightly different between the
            model trained on Megatron and our model. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
            enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
            resolved in the future once the main model has been fine-tuned with TP_rank=1.

    Example:
    # 导入 BloomConfig 和 BloomModel 类
    from transformers import BloomConfig, BloomModel
    
    # 初始化一个 Bloom 配置对象
    configuration = BloomConfig()
    
    # 从配置对象初始化一个模型（带有随机权重）
    model = BloomModel(configuration)
    
    # 访问模型的配置信息
    configuration = model.config
    
    # 定义模型类型为 "bloom"
    model_type = "bloom"
    
    # 在推断时忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 属性映射字典，将模型参数名称映射到其他名称
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }
    
    # 初始化 BloomModel 类
    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,  # TP rank used when training with megatron
        slow_but_exact=False,
        **kwargs,
    ):
        # 设置模型的各种参数
        self.vocab_size = vocab_size
        # 与 n_embed 参数的向后兼容性
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.slow_but_exact = slow_but_exact
    
        # 调用父类的初始化方法
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
# 定义 BloomOnnxConfig 类，它继承自 OnnxConfigWithPast 类
class BloomOnnxConfig(OnnxConfigWithPast):
    # 设置 torch_onnx_minimum_version 属性为版本号 1.12
    torch_onnx_minimum_version = version.parse("1.12")

    # 初始化方法，接收配置、任务、补丁规范列表和是否使用过去状态等参数
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        # 如果配置中没有 pad_token_id 属性
        if not getattr(self._config, "pad_token_id", None):
            # 将 pad_token_id 设置为 0
            # TODO: 如何更好地执行此操作？
            self._config.pad_token_id = 0

    # inputs 属性，返回输入的描述信息，是一个映射结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 创建通用输入的有序字典，包含 input_ids 和 attention_mask
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        # 如果使用过去状态
        if self.use_past:
            # BLOOM 在动态轴 2 上存储值。有关详细信息，请参阅：https://github.com/huggingface/transformers/pull/18344
            # 使用 fill_with_past_key_values_ 方法填充输入字典，指定方向为 "inputs"，反转值的形状
            self.fill_with_past_key_values_(common_inputs, direction="inputs", inverted_values_shape=True)
            # 添加 attention_mask 描述信息，动态轴的处理方式不同
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            # 如果不使用过去状态，直接添加 attention_mask 描述信息
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # num_layers 属性，返回配置中的层数
    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    # num_attention_heads 属性，返回配置中的注意力头数
    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    # atol_for_validation 属性，返回验证时的绝对误差阈值
    @property
    def atol_for_validation(self) -> float:
        return 1e-3

    # generate_dummy_inputs 方法，生成虚拟输入
    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizer",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 调用父类方法生成通用输入
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照forward()方法中的顺序对输入进行排序
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 需要添加过去的键（past_keys）
        if self.use_past:
            # 检查是否安装了 PyTorch，如果未安装，则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # 过去键值的长度不同
                past_key_values_length = seqlen + 2
                head_dim = self._config.hidden_size // self.num_attention_heads
                past_key_shape = (
                    batch * self.num_attention_heads,
                    head_dim,
                    past_key_values_length,
                )
                past_value_shape = (
                    batch * self.num_attention_heads,
                    past_key_values_length,
                    head_dim,
                )
                # 为每个层创建过去的键值
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_key_shape), torch.zeros(past_value_shape)) for _ in range(self.num_layers)
                ]

        # 将注意力掩码添加到输入
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            # 将注意力掩码与全为1的张量拼接
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回有序输入
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的ONNX操作集版本
        return 13
```