# `.\models\gpt_neo\configuration_gpt_neo.py`

```
# 设置编码格式为UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授予的许可进行使用
# 你不得使用本文件，除非符合许可证规定
# 你可以获取许可证的副本，访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，根据许可证分发的软件属于“按原样”分发，
# 没有任何担保或条件，无论是明示的还是暗示的。
# 请参阅许可证以了解特定语言控制权限和限制

""" GPT Neo 模型配置"""

# 导入必要的库
from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# GPT Neo 预训练配置文件映射
GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-neo-1.3B": "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/config.json",
    # 查看所有 GPTNeo 模型，访问 https://huggingface.co/models?filter=gpt_neo
}

# GPT Neo 配置类
class GPTNeoConfig(PretrainedConfig):
    r"""
    这是用于存储 [`GPTNeoModel`] 配置信息的配置类。它用于根据指定的参数实例化 GPT Neo 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 GPTNeo [EleutherAI/gpt-neo-1.3B]
    (https://huggingface.co/EleutherAI/gpt-neo-1.3B) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import GPTNeoConfig, GPTNeoModel

    >>> # 初始化类似于 EleutherAI/gpt-neo-1.3B 风格配置的 GPTNeo 模型
    >>> 配置 = GPTNeoConfig()

    >>> # 根据 EleutherAI/gpt-neo-1.3B 风格配置初始化一个模型 (具有随机权重)
    >>> model = GPTNeoModel(配置)

    >>> # 访问模型配置
    >>> 配置 = model.config
    ```
    """

    model_type = "gpt_neo"
    # 推断时需要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    # 初始化 Transformer 模型
    def __init__(
        self,
        vocab_size=50257,
        max_position_embeddings=2048,
        hidden_size=2048,
        num_layers=24,
        attention_types=[[["global", "local"], 12]],
        num_heads=16,
        intermediate_size=None,
        window_size=256,
        activation_function="gelu_new",
        resid_dropout=0.0,
        embed_dropout=0.0,
        attention_dropout=0.0,
        classifier_dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        # 设置模型参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        # 设置起始和结束标记的标识符
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 设置注意力类型参数
        self.attention_types = attention_types
        # 根据注意力类型，扩展注意力层参数
        self.attention_layers = self.expand_attention_types_params(attention_types)

        # 验证注意力层数量与层数是否匹配
        if len(self.attention_layers) != self.num_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.attention_layers)` == `config.num_layers` "
                f"but is `len(config.attention_layers) = {len(self.attention_layers)}`, "
                f"`config.num_layers = {self.num_layers}`. "
                "`config.attention_layers` is prepared using `config.attention_types`. "
                "Please verify the value of `config.attention_types` argument."
            )

        # 调用父类的构造函数
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @staticmethod
    # 扩展注意力类型参数
    def expand_attention_types_params(attention_types):
        attentions = []
        # 遍历注意力类型列表，根据权重值扩展注意力列表
        for item in attention_types:
            for _ in range(item[1]):
                attentions.extend(item[0])
        # 返回扩展后的注意力列表
        return attentions
# 自定义 torch.Tensor.unfold 的实现以便导出到 ONNX
def custom_unfold(input, dimension, size, step):
    """Custom torch.Tensor.unfold implementation to enable the export to ONNX."""
    # 导入 torch 库
    import torch

    # 获取输入张量的形状
    shape = input.size()
    # 获取张量的维度
    rank = len(shape)
    # 获取指定维度的大小
    sizedim = shape[dimension]

    # 创建一个张量，包含指定维度范围内的索引
    low_indices = torch.arange(0, sizedim, step)
    # 计算可能的切片数量
    min_length = torch.div(sizedim - size, step, rounding_mode="floor") + 1
    # 创建一个张量，包含指定大小的索引
    indices = torch.arange(size) + low_indices[:min_length][:, None]

    # 创建一个切片索引，用于对输入进行切片操作
    s = [slice(None)] * rank
    s[dimension] = indices
    # 对输入进行切片操作
    sliced = input[s]

    # 创建一个用于重新排列张量维度的排列顺序
    perm = list(range(0, rank + 1))
    perm.append(perm.pop(dimension + 1))

    # 返回重新排列维度后的张量
    return sliced.permute(perm)


def custom_get_block_length_and_num_blocks(seq_length, window_size):
    """
    Custom implementation for GPTNeoAttentionMixin._get_block_length_and_num_blocks to enable the export to ONNX as
    original implementation uses Python variables and control flow.
    """
    # 导入 torch 库
    import torch

    # 创建候选窗口大小序列
    candidates = torch.arange(1, window_size)
    # 计算序列长度与窗口大小的余数
    remainders = torch.remainder(seq_length, candidates)
    # 筛选出能整除序列长度的候选窗口大小
    divisor_indices = remainders == 0
    divisors = candidates[divisor_indices]
    # 获取最大的能整除序列长度的窗口大小
    largest_divisor = torch.max(divisors)
    # 计算每个块的长度和块的数量
    return largest_divisor, torch.div(seq_length, largest_divisor, rounding_mode="floor")


class GPTNeoOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义输入名称和索引映射的公共部分
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        # 如果使用过去信息，则填充过去信息的键值对
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_attention_heads(self) -> int:
        # 返回注意力头的数量
        return self._config.num_heads

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    # 定义函数，生成带有过去信息的 ONNX 配置的虚拟输入
    ) -> Mapping[str, Any]:
        # 调用父类方法生成普通的虚拟输入
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照 forward() 方法中的顺序对输入进行排序
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 需要添加过去键的输入
        if self.use_past:
            # 检查是否安装了 PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                # 获取 batch 和 seqlen 的大小
                batch, seqlen = common_inputs["input_ids"].shape
                # 设置 past_key_values 的长度，不使用相同的长度
                past_key_values_length = seqlen + 2
                # 设置 past 的形状
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                # 为每一层创建过去键的零张量
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        # 将普通的注意力掩码添加到排序后的输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        # 如果使用过去信息，则调整注意力掩码的形状
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            # 将全为1的张量拼接到原始的注意力掩码后面
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回排序后的输入
        return ordered_inputs

    # 返回默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 13
```