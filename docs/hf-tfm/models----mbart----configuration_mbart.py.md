# `.\transformers\models\mbart\configuration_mbart.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 Facebook AI Research Team 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" MBART 模型配置"""
# 导入所需的库
from collections import OrderedDict
from typing import Any, Mapping, Optional

# 导入预训练的分词器
from ... import PreTrainedTokenizer
# 导入预训练配置工具
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
# 导入 ONNX 工具
from ...onnx.utils import compute_effective_axis_dimension
# 导入工具函数
from ...utils import TensorType, is_torch_available, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# MBART 预训练配置存档映射
MBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json",
    # 查看所有 MBART 模型：https://huggingface.co/models?filter=mbart
}

# MBART 配置类，继承自预训练配置类
class MBartConfig(PretrainedConfig):
    r"""
    这是用于存储 [`MBartModel`] 配置的配置类。根据指定的参数实例化 MBART 模型，定义模型架构。使用默认值实例化配置将产生类似于 MBART
    [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import MBartConfig, MBartModel

    >>> # 初始化一个 MBART facebook/mbart-large-cc25 风格的配置
    >>> configuration = MBartConfig()

    >>> # 从 facebook/mbart-large-cc25 风格的配置初始化一个模型（带有随机权重）
    >>> model = MBartModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        max_position_embeddings=1024,  # 最大位置编码长度，默认为1024
        encoder_layers=12,  # 编码器层数，默认为12
        encoder_ffn_dim=4096,  # 编码器中前馈网络的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16
        decoder_layers=12,  # 解码器层数，默认为12
        decoder_ffn_dim=4096,  # 解码器中前馈网络的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16
        encoder_layerdrop=0.0,  # 编码器层的丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层的丢弃率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否为编码-解码模型，默认为True
        activation_function="gelu",  # 激活函数，默认为gelu
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # 通用丢弃率，默认为0.1
        attention_dropout=0.0,  # 注意力丢弃率，默认为0.0
        activation_dropout=0.0,  # 激活函数丢弃率，默认为0.0
        init_std=0.02,  # 初始化标准差，默认为0.02
        classifier_dropout=0.0,  # 分类器丢弃率，默认为0.0
        scale_embedding=False,  # 是否对嵌入进行缩放，默认为False；如果为True，则缩放因子为sqrt(d_model)
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=0,  # 起始标记的ID，默认为0
        eos_token_id=2,  # 结束标记的ID，默认为2
        forced_eos_token_id=2,  # 强制结束标记的ID，默认为2
        **kwargs,  # 其他关键字参数
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
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers  # 编码器层数
        self.scale_embedding = scale_embedding  # 如果为True，则缩放因子为sqrt(d_model)
        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,  # 其他关键字参数
        )
# 从transformers.models.bart.configuration_bart.BartOnnxConfig复制代码，并将Bart->MBart
class MBartOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义inputs属性，返回输入的映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设置不同的输入映射关系
        if self.task in ["default", "seq2seq-lm"]:
            # 对于默认任务和seq2seq-lm任务，设置通用的输入映射关系
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            # 根据是否使用过去信息，设置不同的decoder输入映射关系
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

            # 如果使用过去信息，填充past key values
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        elif self.task == "causal-lm":
            # 对于causal-lm任务，暂时留下TODO标记，需要进一步处理
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
            # 对于其他任务，设置通用的输入映射关系
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),
                    ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),
                ]
            )

        return common_inputs

    # 定义outputs属��，返回输出的映射关系
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设置不同的输出映射关系
        if self.task in ["default", "seq2seq-lm"]:
            # 对于默认任务和seq2seq-lm任务，调用父类的outputs方法
            common_outputs = super().outputs
        else:
            # 对于其他任务，调用父类OnnxConfigWithPast的outputs方法
            common_outputs = super(OnnxConfigWithPast, self).outputs
            # 如果使用过去信息，填充present key values
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return common_outputs
    # 生成用于默认和序列到序列语言模型的虚拟输入数据
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        # 输入参数：分词器
        self,
        # 输入参数：批量大小，默认为-1
        batch_size: int = -1,
        # 输入参数：序列长度，默认为-1
        seq_length: int = -1,
        # 输入参数：是否为成对数据，默认为False
        is_pair: bool = False,
        # 输入参数：框架类型，默认为None
        framework: Optional[TensorType] = None,
    # 定义函数返回类型为 Mapping[str, Any]
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
        # 将解码器输入的键名加上"decoder_"前缀
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 合并编码器和解码器输入
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去信息
        if self.use_past:
            # 检查是否安装了 PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取批次大小和编码器序列长度
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

            # 在解码器注意力掩码后面添加全1的张量
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化过去键值对列表
            common_inputs["past_key_values"] = []
            # 如果模型配置中存在编码器和解码器层数，都会被考虑
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 为每一层添加过去键值对
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
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            # 添加剩余层的过去键值对
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        # 返回合并后的输入
        return common_inputs
    # 生成用于因果语言模型的虚拟输入数据
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 生成用于序列分类和问答的虚拟输入数据
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去的键值
        if self.use_past:
            # 检查是否安装了 PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch, seqlen = common_inputs["input_ids"].shape
            # 设置过去键值的长度
            past_key_values_length = seqlen + 2
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            mask_dtype = common_inputs["attention_mask"].dtype
            # 在注意力掩码后面添加一列全为1的张量
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化过去键值
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        return common_inputs

    # 生成用于序列分类和问答的虚拟输入数据
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 计算有效的轴维度，以避免 ONNX 优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 计算有效的轴维度，以避免 ONNX 优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 生成虚拟输入数据
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
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
        # 如果任务是默认任务或者序列到序列语言模型任务
        if self.task in ["default", "seq2seq-lm"]:
            # 为默认任务和序列到序列语言模型任务生成虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 如果任务是因果语言模型任务
        elif self.task == "causal-lm":
            # 为因果语言模型任务生成虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        else:
            # 为序列分类和问答任务生成虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回通用输入数据字典
        return common_inputs

    # 将过去的键值对展平化
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是默认任务或者序列到序列语言模型任务
        if self.task in ["default", "seq2seq-lm"]:
            # 调用父类方法展平化过去的键值对
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 调用特定类的父类方法展平化过去的键值对
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```