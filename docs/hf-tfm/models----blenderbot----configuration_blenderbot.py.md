# `.\transformers\models\blenderbot\configuration_blenderbot.py`

```
# 导入所需的模块和类
from collections import OrderedDict  # 从 collections 模块导入 OrderedDict 类
from typing import Any, Mapping, Optional  # 从 typing 模块导入 Any、Mapping、Optional 类型

# 导入预训练的分词器、配置和文件工具
from ... import PreTrainedTokenizer  # 从 transformers 模块导入 PreTrainedTokenizer 类
from ...configuration_utils import PretrainedConfig  # 从 transformers 模块导入 PretrainedConfig 类
from ...file_utils import TensorType, is_torch_available  # 从 transformers 模块导入 TensorType 类、is_torch_available 函数
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast  # 从 transformers 模块导入 OnnxConfig、OnnxConfigWithPast、OnnxSeq2SeqConfigWithPast 类
from ...onnx.utils import compute_effective_axis_dimension  # 从 transformers 模块导入 compute_effective_axis_dimension 函数
from ...utils import logging  # 从 transformers 模块导入 logging 模块

# 获取记录器实例
logger = logging.get_logger(__name__)

# Blenderbot 预训练配置文件的存档映射
BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/config.json",
    # 查看所有 Blenderbot 模型的列表：https://huggingface.co/models?filter=blenderbot
}


class BlenderbotConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`BlenderbotModel`] 的配置。它用于根据指定的参数实例化 Blenderbot 模型，定义模型的架构。
    使用默认值实例化配置将产生类似 Blenderbot [facebook/blenderbot-3B](https://huggingface.co/facebook/blenderbot-3B) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import BlenderbotConfig, BlenderbotModel

    >>> # 初始化一个 Blenderbot facebook/blenderbot-3B 风格的配置
    >>> configuration = BlenderbotConfig()

    >>> # 使用 facebook/blenderbot-3B 风格的配置初始化一个模型（带有随机权重）
    >>> model = BlenderbotModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "blenderbot"  # 模型类型为 "blenderbot"
    keys_to_ignore_at_inference = ["past_key_values"]  # 推理时要忽略的键列表，包括 "past_key_values"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}  # 属性映射字典，用于转换配置属性名称
    # 初始化函数，用于初始化Transformer模型的参数
    def __init__(
        self,
        vocab_size=8008,  # 词汇表大小，默认为8008
        max_position_embeddings=128,  # 最大位置编码长度，默认为128
        encoder_layers=2,  # 编码器层数，默认为2
        encoder_ffn_dim=10240,  # 编码器中Feed Forward网络的维度，默认为10240
        encoder_attention_heads=32,  # 编码器中注意力头的数量，默认为32
        decoder_layers=24,  # 解码器层数，默认为24
        decoder_ffn_dim=10240,  # 解码器中Feed Forward网络的维度，默认为10240
        decoder_attention_heads=32,  # 解码器中注意力头的数量，默认为32
        encoder_layerdrop=0.0,  # 编码器层的丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层的丢弃率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码-解码模型，默认为True
        activation_function="gelu",  # 激活函数类型，默认为gelu
        d_model=2560,  # 模型的维度，默认为2560
        dropout=0.1,  # 模型的丢弃率，默认为0.1
        attention_dropout=0.0,  # 注意力机制的丢弃率，默认为0.0
        activation_dropout=0.0,  # 激活函数的丢弃率，默认为0.0
        init_std=0.02,  # 初始化参数的标准差，默认为0.02
        decoder_start_token_id=1,  # 解码器起始标记的ID，默认为1
        scale_embedding=False,  # 是否缩放嵌入，默认为False；如果为True，则缩放因子为sqrt(d_model)
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=1,  # 起始标记的ID，默认为1
        eos_token_id=2,  # 结束标记的ID，默认为2
        encoder_no_repeat_ngram_size=3,  # 编码器中不重复N-gram的大小，默认为3
        forced_eos_token_id=2,  # 强制结束标记的ID，默认为2
        **kwargs,
    ):
        # 初始化各个参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers  # 隐藏层的数量等于编码器的层数
        self.scale_embedding = scale_embedding  # 如果为True，则缩放因子为sqrt(d_model)

        # 调用父类的初始化函数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
```  
class BlenderbotOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    # 定义输入属性，返回输入的映射关系
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是默认或者seq2seq-lm
        if self.task in ["default", "seq2seq-lm"]:
            # 定义常见的输入映射关系
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            # 如果使用过去信息
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}
            # 如果使用过去信息，填充过去键值对
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 如果任务是causal-lm
        elif self.task == "causal-lm":
            # 定义常见的输入映射关系
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            # 如果使用过去信息，获取解码器层数
            if self.use_past:
                _, num_decoder_layers = self.num_layers
                for i in range(num_decoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
            # 定义常见的输入映射关系
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),
                    ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),
                ]
            )

        return common_inputs

    @property
    # 从transformers.models.bart.configuration_bart.BartOnnxConfig.outputs中复制而来
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如��任务是默认或者seq2seq-lm
        if self.task in ["default", "seq2seq-lm"]:
            # 获取超类的输出
            common_outputs = super().outputs
        else:
            # 获取带有过去信息的超类的输出
            common_outputs = super(OnnxConfigWithPast, self).outputs
            # 如果使用过去信息，获取编码器层数
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return common_outputs

    # 为默认和seq2seq-lm生成虚拟输入
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    # 定义方法，用于生成用于序列分类和问答任务的虚拟输入
    ) -> Mapping[str, Any]:
        # 生成编码器输入
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )
        # 生成解码器输入
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 为解码器输入添加前缀以区分编码器和解码器
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 将编码器和解码器输入合并为一个字典
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去的键（past keys）
        if self.use_past:
            # 如果没有安装 PyTorch，则无法生成虚拟 past_keys 输入
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取批量大小和编码器序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            # 获取解码器序列长度
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            # 获取编码器和解码器的注意力头数
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 计算编码器和解码器的形状
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            decoder_past_length = decoder_seq_length
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )
            # 更新解码器的注意力掩码，以考虑过去的内容
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )
            # 初始化 past_key_values 列表
            common_inputs["past_key_values"] = []
            # 获取解码器层数
            _, num_decoder_layers = self.num_layers

            # 为每个解码器层生成 past_key_values
            for _ in range(num_decoder_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),  # 初始化解码器的过去键
                        torch.zeros(decoder_shape),  # 初始化解码器的过去值
                        torch.zeros(encoder_shape),  # 初始化编码器的过去键（用于解码器-编码器注意力）
                        torch.zeros(encoder_shape),  # 初始化编码器的过去值（用于解码器-编码器注意力）
                    )
                )
        # 返回合并后的输入字典，包括编码器输入、解码器输入和过去键值
        return common_inputs

    # 定义方法，用于生成用于因果语言模型的虚拟输入
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    # 定义函数，返回一个映射类型的字典，包含通用的输入数据
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 根据给定的参数生成用于序列分类和问答的虚拟输入数据
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去的键值对
        if self.use_past:
            # 如果没有安装 PyTorch，则抛出异常
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取输入数据的批量大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            past_key_values_length = seqlen
            _, num_decoder_layers = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 计算过去键值对的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            mask_dtype = common_inputs["attention_mask"].dtype
            # 在注意力掩码后面添加全为1的张量
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化过去键值对
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_decoder_layers)
            ]
        # 返回通用输入数据
        return common_inputs

    # 复制自 transformers.models.bart.configuration_bart.BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering
    # 定义函数，生成用于序列分类和问答的虚拟输入数据
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 复制自 OnnxConfig.generate_dummy_inputs
        # 为了代码清晰性，没有使用 super(OnnxConfigWithPast, self).generate_dummy_inputs
        # 如果动态轴为-1，则以固定维度的2个样本进行前向传播，以避免 ONNX 进行的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴为-1，则以固定维度的8个标记进行前向传播，以避免 ONNX 进行的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批量和序列生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs

    # 复制自 transformers.models.bart.configuration_bart.BartOnnxConfig.generate_dummy_inputs
    # 生成虚拟输入数据，根据任务类型调用不同的生成函数
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务类型是"default"或"seq2seq-lm"，调用对应的生成函数
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 如果任务类型是"causal-lm"，调用对应的生成函数
        elif self.task == "causal-lm":
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 其他情况，调用默认的生成函数
        else:
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        return common_inputs

    # 从父类中复制函数实现，用于展开过去的键值对
    # 来自transformers.models.bart.configuration_bart.BartOnnxConfig._flatten_past_key_values_
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务类型是"default"或"seq2seq-lm"，调用父类的对应函数
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        # 其他情况，调用特定的父类函数
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )

    # 填充过去的键值对
    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        # 如果方向不是"inputs"或"outputs"，抛出异常
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        _, num_decoder_layers = self.num_layers

        encoder_sequence = "past_encoder_sequence"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"

        # 遍历解码器层，填充键值对
        for i in range(num_decoder_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch", 2: encoder_sequence}
```