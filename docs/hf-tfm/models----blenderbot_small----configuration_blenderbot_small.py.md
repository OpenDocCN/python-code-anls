# `.\models\blenderbot_small\configuration_blenderbot_small.py`

```
"""
BlenderbotSmall model configuration

This module defines the configuration class `BlenderbotSmallConfig` for the BlenderbotSmall model.
It specifies how the model should be instantiated and configured. It inherits from `PretrainedConfig`
and provides defaults similar to the `facebook/blenderbot_small-90M` architecture.

Example:

>>> from transformers import BlenderbotSmallConfig, BlenderbotSmallModel

>>> # Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration
>>> configuration = BlenderbotSmallConfig()

>>> # Initializing a model (with random weights) from the facebook/blenderbot_small-90M style configuration
>>> model = BlenderbotSmallModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
"""

from collections import OrderedDict  # 导入有序字典类
from typing import Any, Mapping, Optional  # 导入类型提示相关的类和函数

from ... import PreTrainedTokenizer  # 导入预训练标记器类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...file_utils import TensorType, is_torch_available  # 导入文件工具类和检查是否有torch可用的函数
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast  # 导入ONNX相关配置类
from ...onnx.utils import compute_effective_axis_dimension  # 导入计算有效轴维度的函数
from ...utils import logging  # 导入日志工具类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/config.json",
    # 预训练配置存档映射表，指定了模型名称及其配置文件的URL
    # 查看所有BlenderbotSmall模型请访问https://huggingface.co/models?filter=blenderbot_small
}


class BlenderbotSmallConfig(PretrainedConfig):
    r"""
    BlenderbotSmall模型的配置类，用于存储[`BlenderbotSmallModel`]的配置。
    它用于根据指定的参数实例化BlenderbotSmall模型，定义模型架构。
    使用默认值实例化配置将生成类似于BlenderbotSmall [facebook/blenderbot_small-90M](https://huggingface.co/facebook/blenderbot_small-90M)架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读[`PretrainedConfig`]的文档。

    Example:

    ```python
    >>> from transformers import BlenderbotSmallConfig, BlenderbotSmallModel

    >>> # Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration
    >>> configuration = BlenderbotSmallConfig()

    >>> # Initializing a model (with random weights) from the facebook/blenderbot_small-90M style configuration
    >>> model = BlenderbotSmallModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "blenderbot-small"  # 模型类型字符串
    keys_to_ignore_at_inference = ["past_key_values"]  # 推理过程中要忽略的键列表
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}  # 属性映射表
    # 初始化函数，用于创建一个新的Transformer模型实例
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        max_position_embeddings=512,  # 最大位置编码长度，默认为512
        encoder_layers=8,  # 编码器层数，默认为8层
        encoder_ffn_dim=2048,  # 编码器中FFN层的维度，默认为2048
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16个
        decoder_layers=8,  # 解码器层数，默认为8层
        decoder_ffn_dim=2048,  # 解码器中FFN层的维度，默认为2048
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16个
        encoder_layerdrop=0.0,  # 编码器层随机丢弃的概率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层随机丢弃的概率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码解码模型，默认为True
        activation_function="gelu",  # 激活函数类型，默认为GELU
        d_model=512,  # 模型的维度，默认为512
        dropout=0.1,  # 全局dropout概率，默认为0.1
        attention_dropout=0.0,  # 注意力模块的dropout概率，默认为0.0
        activation_dropout=0.0,  # 激活函数的dropout概率，默认为0.0
        init_std=0.02,  # 参数初始化的标准差，默认为0.02
        decoder_start_token_id=1,  # 解码器的起始token ID，默认为1
        scale_embedding=False,  # 是否对嵌入进行缩放，默认为False
        pad_token_id=0,  # 填充token的ID，默认为0
        bos_token_id=1,  # 起始token的ID，默认为1
        eos_token_id=2,  # 结束token的ID，默认为2
        forced_eos_token_id=2,  # 强制结束token的ID，默认为2
        **kwargs,
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小属性
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码长度属性
        self.d_model = d_model  # 设置模型维度属性
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器FFN层维度属性
        self.encoder_layers = encoder_layers  # 设置编码器层数属性
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器注意力头数属性
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器FFN层维度属性
        self.decoder_layers = decoder_layers  # 设置解码器层数属性
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器注意力头数属性
        self.dropout = dropout  # 设置全局dropout概率属性
        self.attention_dropout = attention_dropout  # 设置注意力模块dropout概率属性
        self.activation_dropout = activation_dropout  # 设置激活函数dropout概率属性
        self.activation_function = activation_function  # 设置激活函数类型属性
        self.init_std = init_std  # 设置参数初始化标准差属性
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层随机丢弃概率属性
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层随机丢弃概率属性
        self.use_cache = use_cache  # 设置是否使用缓存属性
        self.num_hidden_layers = encoder_layers  # 设置隐藏层总数属性为编码器层数
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入属性，若为True，则缩放因子为sqrt(d_model)

        # 调用父类Transformer的初始化函数，传递相关参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
# 从 transformers.models.bart.configuration_bart.BartOnnxConfig 复制了 BlenderbotSmallOnnxConfig 类定义
class BlenderbotSmallOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义 inputs 属性，返回输入映射的有序字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是 "default" 或者 "seq2seq-lm"
        if self.task in ["default", "seq2seq-lm"]:
            # 定义常见的输入映射
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入序列的批次和编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码的批次和编码器序列
                ]
            )

            # 如果使用过去状态
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}  # 解码器输入的批次
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}  # 解码器注意力掩码的批次和过去解码器序列 + 序列
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}  # 解码器输入的批次和解码器序列
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}  # 解码器注意力掩码的批次和解码器序列

            # 如果使用过去状态，则填充过去键值
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 如果任务是 "causal-lm"
        elif self.task == "causal-lm":
            # TODO: 解决这种情况。
            # 定义常见的输入映射
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入序列的批次和编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码的批次和编码器序列
                ]
            )
            # 如果使用过去状态
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                # 为每一层的过去键值添加输入映射
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}  # 过去键的批次和过去序列 + 序列
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}  # 过去值的批次和过去序列 + 序列
        else:
            # 定义常见的输入映射
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入序列的批次和编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码的批次和编码器序列
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),  # 解码器输入的批次和解码器序列
                    ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),  # 解码器注意力掩码的批次和解码器序列
                ]
            )

        # 返回输入映射的字典
        return common_inputs

    # 定义 outputs 属性，返回输出映射的字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是 "default" 或者 "seq2seq-lm"
        if self.task in ["default", "seq2seq-lm"]:
            # 调用父类的 outputs 方法获取通用的输出映射
            common_outputs = super().outputs
        else:
            # 调用父类 OnnxConfigWithPast 的 outputs 方法获取通用的输出映射
            common_outputs = super(OnnxConfigWithPast, self).outputs
            # 如果使用过去状态
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                # 为每一层的当前状态添加输出映射
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}  # 当前键的批次和过去序列 + 序列
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}  # 当前值的批次和过去序列 + 序列
        # 返回输出映射的字典
        return common_outputs
    # 定义一个方法 `_generate_dummy_inputs_for_default_and_seq2seq_lm`，用于生成默认和序列到序列语言模型的虚拟输入数据
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,  # 参数：预训练的分词器对象，用于处理输入数据的分词和编码
        batch_size: int = -1,             # 参数：批大小，默认为-1，表示使用预设的批大小
        seq_length: int = -1,             # 参数：序列长度，默认为-1，表示使用预设的序列长度
        is_pair: bool = False,            # 参数：是否为成对数据，默认为False，表示不是成对数据
        framework: Optional[TensorType] = None,  # 参数：框架类型，可选的张量类型，用于特定框架的处理
    ) -> Mapping[str, Any]:
        # 生成编码器输入数据
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 生成解码器输入数据
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 为解码器输入添加前缀
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 整合编码器和解码器的输入数据
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去状态
        if self.use_past:
            # 检查是否安装了PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # 获取输入数据的批次大小和编码器序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            
            # 定义编码器和解码器的形状
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            decoder_past_length = decoder_seq_length + 3
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )

            # 扩展解码器的注意力掩码
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化过去键值列表
            common_inputs["past_key_values"] = []

            # 根据模型配置中的编码器和解码器层数，创建过去键值对
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 为每一层添加初始的过去键值对
            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            
            # TODO: test this.
            # 对于剩余的层数，根据模型的不同，添加适当的过去键值对
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        
        # 返回整合后的输入数据字典
        return common_inputs
    # 生成用于因果语言模型的虚拟输入数据集
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用另一个生成序列分类和问答虚拟输入数据集的方法，获取共同的输入部分
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果需要使用过去的键值对（past_key_values）
        if self.use_past:
            # 检查是否安装了 PyTorch，如果没有则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取 batch 和 seqlen
            batch, seqlen = common_inputs["input_ids"].shape
            # 计算过去键值对的长度，比当前序列长度多 2
            past_key_values_length = seqlen + 2
            # 解析编码器层数和注意力头数
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 定义过去键值对的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            # 获取注意力掩码的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 将新生成的过去键值对长度的注意力掩码拼接到原始注意力掩码后面
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 生成初始的过去键值对列表，每层编码器对应一个空的过去键值对元组
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        
        # 返回生成的共同输入字典
        return common_inputs

    # 生成用于序列分类和问答模型的虚拟输入数据集
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从 OnnxConfig.generate_dummy_inputs 复制的方法，用于保持代码清晰
        # 如果动态轴 (-1)，我们使用固定维度的 2 个样本以避免 ONNX 进行的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴 (-1)，我们使用固定维度的 8 个 token 以避免 ONNX 进行的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的 batch 和 sequence 生成虚拟输入数据
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        # 返回生成的共同输入字典
        return common_inputs
    # 生成虚拟输入数据，返回一个包含各种任务通用输入的字典
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务是默认任务或序列到序列语言模型
        if self.task in ["default", "seq2seq-lm"]:
            # 调用默认任务和序列到序列语言模型的虚拟输入生成函数
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 如果任务是因果语言模型
        elif self.task == "causal-lm":
            # 调用因果语言模型的虚拟输入生成函数
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        else:
            # 调用序列分类和问题回答的虚拟输入生成函数（适用于其它任务）
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回通用输入字典
        return common_inputs

    # 将过去的键值扁平化处理的内部方法
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是默认任务或序列到序列语言模型
        if self.task in ["default", "seq2seq-lm"]:
            # 调用父类的方法来扁平化过去的键值
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 使用带有过去键值的 ONNX 序列到序列配置的父类方法来扁平化过去的键值
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```