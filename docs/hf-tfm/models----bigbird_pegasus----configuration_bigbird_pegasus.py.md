# `.\transformers\models\bigbird_pegasus\configuration_bigbird_pegasus.py`

```py
# coding=utf-8
# 版权所有 Google Research 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（"许可证"）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"按原样"分发的，
# 没有任何担保或条件，明示或暗示。
# 有关特定语言的更多信息，请参阅许可证。
""" BigBirdPegasus 模型配置"""

# 导入必要的库
from collections import OrderedDict
from typing import Any, Mapping, Optional

# 导入预训练标记器、预训练配置等
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件的映射，包含不同预训练模型的配置文件链接
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
    # 查看所有 BigBirdPegasus 模型: https://huggingface.co/models?filter=bigbird_pegasus
}

# BigBirdPegasus 配置类，继承自 PretrainedConfig
class BigBirdPegasusConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`BigBirdPegasusModel`] 的配置。它用于根据指定的参数实例化 BigBirdPegasus 模型，
    定义模型架构。使用默认值实例化配置将产生与 BigBirdPegasus
    [google/bigbird-pegasus-large-arxiv](https://huggingface.co/google/bigbird-pegasus-large-arxiv) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。


    示例:

    ```python
    >>> from transformers import BigBirdPegasusConfig, BigBirdPegasusModel

    >>> # 初始化一个 BigBirdPegasus bigbird-pegasus-base 风格的配置
    >>> configuration = BigBirdPegasusConfig()

    >>> # 使用大鸟 - 佩加索斯大型模型配置初始化一个模型（带有随机权重）
    >>> model = BigBirdPegasusModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "bigbird_pegasus"
    # 推断时要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
```  
    # 定义一个属性映射字典，将模型参数名映射到内部使用的名称
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
    }

    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=96103,
        max_position_embeddings=4096,
        encoder_layers=16,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=16,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu_new",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        classifier_dropout=0.0,
        scale_embedding=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        attention_type="block_sparse",  # 只用于编码器
        block_size=64,
        num_random_blocks=3,
        use_bias=False,
        **kwargs,
    ):
        # 设置模型的各种参数
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
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # 如果为True，则缩放因子为sqrt(d_model)

        # 额外的配置
        self.attention_type = attention_type
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.use_bias = use_bias

        # 调用父类的初始化函数，设置模型的其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
# 从transformers.models.bart.configuration_bart.BartOnnxConfig复制代码，定义BigBirdPegasusOnnxConfig类，继承自OnnxSeq2SeqConfigWithPast类
class BigBirdPegasusOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义inputs属性，返回输入的映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设置不同的输入映射关系
        if self.task in ["default", "seq2seq-lm"]:
            # 对于默认任务或seq2seq-lm任务，设置通用的输入映射关系
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

    # 定义outputs属性，返回输出的映射关系
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设置不同的输出映射关系
        if self.task in ["default", "seq2seq-lm"]:
            # 对于默认任务或seq2seq-lm任务，调用父类的outputs方法
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
    # 定义一个方法，用于生成默认和序列到序列语言模型的虚拟输入
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        # 接受一个预训练的分词器对象作为参数
        tokenizer: PreTrainedTokenizer,
        # 批处理大小，默认为-1，表示未指定
        batch_size: int = -1,
        # 序列长度，默认为-1，表示未指定
        seq_length: int = -1,
        # 是否是成对输入，默认为False
        is_pair: bool = False,
        # 框架类型，默认为None
        framework: Optional[TensorType] = None,
    ```py  
    ) -> Mapping[str, Any]:
        # 为序列分类和问答任务生成虚拟输入数据，用于编码器输入
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 生成解码器输入
        decoder_seq_length = seq_length if not self.use_past else 1
        # 为序列分类和问答任务生成虚拟输入数据，用于解码器输入
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 重命名解码器输入的张量，加上前缀"decoder_"
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 合并编码器和解码器的输入
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去信息
        if self.use_past:
            # 检查是否安装了 PyTorch
            if not is_torch_available():
                # 报错，提示未安装 PyTorch
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取输入的批次大小和编码器序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            # 获取解码器序列长度
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            # 获取编码器和解码器的注意力头数
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

            # 扩展解码器注意力掩码，增加过去信息的长度
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化过去键值的列表
            common_inputs["past_key_values"] = []
            # 如果模型配置中存在编码器和解码器层数信息，则考虑两者
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 为编码器和解码器每一层添加过去键值
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
            # 添加剩余的过去键值，如果编码器或解码器的层数不相等
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        # 返回合并后的输入字典
        return common_inputs
    # 生成用于因果语言建模的虚拟输入数据
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 生成用于序列分类和问答任务的共同输入数据
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去的键值对
        if self.use_past:
            # 检查是否有可用的PyTorch，如果没有则引发错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取batch和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            # 为过去的键值对设置长度，比序列长度多两个
            past_key_values_length = seqlen + 2
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 设置过去键值对的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            # 获取mask的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 将输入的注意力掩码扩展以匹配过去键值对的长度
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化过去键值对为零张量
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        return common_inputs

    # 生成用于序列分类和问答任务的虚拟输入数据
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从OnnxConfig.generate_dummy_inputs复制过来
        # 为了代码清晰性而没有使用super(OnnxConfigWithPast, self).generate_dummy_inputs。
        # 如果动态轴（-1），我们将使用一个固定维度的2个样本，以避免ONNX所做的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴（-1），我们将使用一个固定维度的8个标记，以避免ONNX所做的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批次和序列长度生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs
    # 生成虚拟输入数据，用于模型推理
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
        # 如果任务是序列分类或问题回答任务
        else:
            # 为序列分类和问题回答任务生成虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回通用的输入数据
        return common_inputs

    # 将过去的键值扁平化
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是默认任务或者序列到序列语言模型任务
        if self.task in ["default", "seq2seq-lm"]:
            # 调用父类的方法来扁平化过去的键值
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 调用父类的方法来扁平化过去的键值
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```