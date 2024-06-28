# `.\models\bigbird_pegasus\configuration_bigbird_pegasus.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google Research 和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证版本 2.0 进行许可，除非符合许可证，否则不得使用此文件
# 可以在以下链接获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据“原样”分发本软件，不提供任何形式的担保或条件
# 有关更多信息，请查阅许可证内容
""" BigBirdPegasus 模型配置"""

# 导入 OrderedDict 类和一些类型提示
from collections import OrderedDict
from typing import Any, Mapping, Optional

# 导入 PreTrainedTokenizer 类，它来自于父级目录中的模块
from ... import PreTrainedTokenizer

# 从 configuration_utils 模块中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig

# 从 onnx 模块中导入一些配置类
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast

# 从 onnx.utils 模块导入 compute_effective_axis_dimension 函数
from ...onnx.utils import compute_effective_axis_dimension

# 导入 utils 模块中的一些实用函数和类
from ...utils import TensorType, is_torch_available, logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# BigBirdPegasus 预训练配置文件的映射字典，包含了几个预训练模型的配置文件 URL
BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/bigbird-pegasus-large-arxiv": (
        "https://huggingface.co/google/bigbird-pegasus-large-arxiv/resolve/main/config.json"
    ),
    "google/bigbird-pegasus-large-pubmed": (
        "https://huggingface.co/google/bigbird-pegasus-large-pubmed/resolve/main/config.json"
    ),
    "google/bigbird-pegasus-large-bigpatent": (
        "https://huggingface.co/google/bigbird-pegasus-large-bigpatent/resolve/main/config.json"
    ),
    # 查看所有 BigBirdPegasus 模型的列表链接：https://huggingface.co/models?filter=bigbird_pegasus
}

class BigBirdPegasusConfig(PretrainedConfig):
    r"""
    这是用于存储 BigBirdPegasusModel 配置的类。它用于根据指定的参数实例化 BigBirdPegasus 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 BigBirdPegasus google/bigbird-pegasus-large-arxiv 架构的配置。

    配置对象继承自 PretrainedConfig，并可用于控制模型输出。阅读 PretrainedConfig 的文档以获取更多信息。

    Example:

    ```python
    >>> from transformers import BigBirdPegasusConfig, BigBirdPegasusModel

    >>> # 初始化一个 BigBirdPegasus bigbird-pegasus-base 风格的配置
    >>> configuration = BigBirdPegasusConfig()

    >>> # 从配置中初始化一个模型（带有随机权重）
    >>> model = BigBirdPegasusModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "bigbird_pegasus"
    model_type = "bigbird_pegasus"

    # 在推理时忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 定义一个字典，用于映射模型的属性名到预训练模型配置中使用的属性名
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
    }

    # 初始化函数，用于创建一个新的预训练模型配置对象
    def __init__(
        self,
        vocab_size=96103,  # 词汇表大小，默认为96103
        max_position_embeddings=4096,  # 最大位置嵌入数，默认为4096
        encoder_layers=16,  # 编码器层数，默认为16层
        encoder_ffn_dim=4096,  # 编码器中FFN层的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16个
        decoder_layers=16,  # 解码器层数，默认为16层
        decoder_ffn_dim=4096,  # 解码器中FFN层的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16个
        encoder_layerdrop=0.0,  # 编码器层dropout率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层dropout率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码解码模型，默认为True
        activation_function="gelu_new",  # 激活函数类型，默认为gelu_new
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # 全局dropout率，默认为0.1
        attention_dropout=0.0,  # 注意力机制的dropout率，默认为0.0
        activation_dropout=0.0,  # 激活函数的dropout率，默认为0.0
        init_std=0.02,  # 参数初始化标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器开始标记的ID，默认为2
        classifier_dropout=0.0,  # 分类器的dropout率，默认为0.0
        scale_embedding=True,  # 是否缩放嵌入，默认为True
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=2,  # 开始标记的ID，默认为2
        eos_token_id=1,  # 结束标记的ID，默认为1
        attention_type="block_sparse",  # 注意力类型，仅用于编码器，默认为block_sparse
        block_size=64,  # 块大小，默认为64
        num_random_blocks=3,  # 随机块的数量，默认为3
        use_bias=False,  # 是否使用偏置，默认为False
        **kwargs,  # 其他关键字参数
    ):
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
        self.num_hidden_layers = encoder_layers  # 将编码器层数赋值给隐藏层数
        self.scale_embedding = scale_embedding  # 如果为True，则嵌入向量将缩放为sqrt(d_model)

        # 额外的配置参数
        self.attention_type = attention_type
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.use_bias = use_bias

        # 调用父类的初始化方法，传入一些预定义的参数和额外的关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
# Copied from transformers.models.bart.configuration_bart.BartOnnxConfig
class BigBirdPegasusOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义 inputs 属性，返回输入映射字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型配置通用输入字典
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            # 如果使用过去信息，添加特定于解码器的输入信息
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

            # 如果使用过去信息，填充过去键值对
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        elif self.task == "causal-lm":
            # 处理因果语言建模任务的情况，暂时标记为待解决
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            # 如果使用过去信息，为每个编码器层添加特定的过去键值对信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
            # 默认情况下配置通用输入字典，包括编码器和解码器信息
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),
                    ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),
                ]
            )

        # 返回通用输入字典
        return common_inputs

    # 定义 outputs 属性，返回输出映射字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型配置通用输出字典
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
            # 如果使用过去信息，为每个编码器层添加特定的现在键值对信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        # 返回通用输出字典
        return common_outputs
    # 定义一个方法用于生成默认和序列到序列语言模型的虚拟输入数据
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # Generate encoder inputs using dummy data for sequence classification and question answering
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # Determine decoder sequence length based on whether past information is used
        decoder_seq_length = seq_length if not self.use_past else 1
        
        # Generate decoder inputs using dummy data, adjusted for sequence length and pairing
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        
        # Prefix decoder input names and create a dictionary for decoder inputs
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        
        # Combine encoder and decoder inputs into a common inputs dictionary
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # Handle the case where past information is used
        if self.use_past:
            # Check if PyTorch is available; if not, raise an error
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # Extract batch size and encoder sequence length from common inputs
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            
            # Determine decoder sequence length and attention heads from model configuration
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            
            # Define shapes for encoder and decoder past key values
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

            # Expand decoder attention mask to accommodate past information
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # Initialize past key values list for storing past states
            common_inputs["past_key_values"] = []

            # Determine the minimum number of layers between encoder and decoder
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)

            # Determine the remaining side (encoder or decoder) for past key values
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # Populate past key values with zero-initialized tensors for each layer
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
            # Extend past key values with zero-initialized tensors for additional layers
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        
        # Return the finalized common inputs dictionary
        return common_inputs
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用生成用于序列分类和问答的虚拟输入方法，获取共享的输入字典
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果需要使用过去的键值（past_key_values）
        if self.use_past:
            # 检查是否安装了 torch 库
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # 获取 batch 和 seqlen 的大小
            batch, seqlen = common_inputs["input_ids"].shape
            
            # 设置过去键值的长度，比 seqlen 多 2
            past_key_values_length = seqlen + 2
            
            # 获取编码器层和注意力头的数量
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            
            # 设置过去键值的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            # 获取掩码的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            
            # 将新生成的掩码与现有掩码连接起来
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            
            # 初始化过去键值的占位符列表
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        
        # 返回最终生成的共享输入字典
        return common_inputs

    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从 OnnxConfig.generate_dummy_inputs 复制而来
        # 为了代码清晰性，没有使用 super(OnnxConfigWithPast, self).generate_dummy_inputs
        # 如果动态轴为 -1，则使用固定的样本维度 2 来避免 ONNX 的优化影响
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴为 -1，则使用固定的序列长度 8 来避免 ONNX 的优化影响
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批处理大小和序列长度生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs
    # 生成虚拟输入数据的方法，返回一个包含各种任务通用输入的字典
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务类型是"default"或"seq2seq-lm"，调用相应的方法生成通用输入数据
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 如果任务类型是"causal-lm"，调用相应的方法生成通用输入数据
        elif self.task == "causal-lm":
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 对于其它任务类型，调用相应的方法生成通用输入数据
        else:
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        
        # 返回生成的通用输入数据字典
        return common_inputs

    # 根据任务类型选择性地扁平化过去的键值对
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务类型是"default"或"seq2seq-lm"，调用父类方法来扁平化过去的键值对
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        # 对于其他任务类型，使用带有历史信息的特定子类调用父类方法来扁平化过去的键值对
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```