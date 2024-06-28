# `.\models\marian\configuration_marian.py`

```
# coding=utf-8
# Copyright 2021 The Marian Team Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Marian model configuration
"""
# 从 collections 模块导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块导入 Any, Mapping, Optional 类型
from typing import Any, Mapping, Optional

# 从 transformers 包中导入 PreTrainedTokenizer 类
from ... import PreTrainedTokenizer
# 从 transformers.configuration_utils 中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从 transformers.onnx 中导入 OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast 类
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
# 从 transformers.onnx.utils 中导入 compute_effective_axis_dimension 函数
from ...onnx.utils import compute_effective_axis_dimension
# 从 transformers.utils 中导入 TensorType, is_torch_available, logging 函数
from ...utils import TensorType, is_torch_available, logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP 字典，映射模型名称到配置文件 URL
MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/config.json",
    # 查看所有 Marian 模型的链接：https://huggingface.co/models?filter=marian
}

# 定义 MarianConfig 类，继承自 PretrainedConfig 类
class MarianConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MarianModel`]. It is used to instantiate an
    Marian model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Marian
    [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import MarianModel, MarianConfig

    >>> # Initializing a Marian Helsinki-NLP/opus-mt-en-de style configuration
    >>> configuration = MarianConfig()

    >>> # Initializing a model from the Helsinki-NLP/opus-mt-en-de style configuration
    >>> model = MarianModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "marian"
    model_type = "marian"
    # 推理阶段忽略的键列表为 ["past_key_values"]
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将 num_attention_heads 映射为 encoder_attention_heads，hidden_size 映射为 d_model
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    def __init__(
        self,
        vocab_size=58101,
        decoder_vocab_size=None,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=58100,
        scale_embedding=False,
        pad_token_id=58100,
        eos_token_id=0,
        forced_eos_token_id=0,
        share_encoder_decoder_embeddings=True,
        **kwargs,
    ):
        # 初始化方法，设置模型的各种参数和选项
        self.vocab_size = vocab_size
        self.decoder_vocab_size = decoder_vocab_size or vocab_size  # 如果未指定解码器词汇大小，则与编码器相同
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.d_model = d_model  # 模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 编码器中全连接层的维度
        self.encoder_layers = encoder_layers  # 编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 解码器中全连接层的维度
        self.decoder_layers = decoder_layers  # 解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 解码器注意力头数
        self.dropout = dropout  # 总体dropout率
        self.attention_dropout = attention_dropout  # 注意力机制中的dropout率
        self.activation_dropout = activation_dropout  # 激活函数中的dropout率
        self.activation_function = activation_function  # 激活函数类型，默认为GELU
        self.init_std = init_std  # 参数初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 编码器层级dropout率
        self.decoder_layerdrop = decoder_layerdrop  # 解码器层级dropout率
        self.use_cache = use_cache  # 是否使用缓存
        self.num_hidden_layers = encoder_layers  # 隐藏层的数量等于编码器层数
        self.scale_embedding = scale_embedding  # 如果为True，则嵌入将按sqrt(d_model)进行缩放
        self.share_encoder_decoder_embeddings = share_encoder_decoder_embeddings  # 是否共享编码器和解码器的嵌入
        super().__init__(
            pad_token_id=pad_token_id,  # 用于填充的标记ID
            eos_token_id=eos_token_id,  # EOS（结束）标记ID
            is_encoder_decoder=is_encoder_decoder,  # 是否是编码-解码模型
            decoder_start_token_id=decoder_start_token_id,  # 解码器起始标记ID
            forced_eos_token_id=forced_eos_token_id,  # 强制EOS（结束）标记ID
            **kwargs,
        )
class MarianOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    # 从 transformers.models.bart.configuration_bart.BartOnnxConfig.inputs 复制而来，定义了模型输入的结构
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型配置输入结构
        if self.task in ["default", "seq2seq-lm"]:
            # 对于默认或序列到序列语言模型任务，设置常规输入
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            if self.use_past:
                # 如果使用过去信息，则调整decoder的输入结构
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

            if self.use_past:
                # 如果使用过去信息，填充对应的键值
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        elif self.task == "causal-lm":
            # 处理因果语言模型任务，暂时留下TODO
            # 目前仅设置常规的输入结构
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            if self.use_past:
                # 如果使用过去信息，根据编码器层数设置键值对
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
            # 处理其他任务类型，设置完整的输入结构，包括编码器和解码器
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
    # 从 transformers.models.bart.configuration_bart.BartOnnxConfig.outputs 复制而来，定义了模型输出的结构
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型配置输出结构
        if self.task in ["default", "seq2seq-lm"]:
            # 对于默认或序列到序列语言模型任务，使用超类的输出结构
            common_outputs = super().outputs
        else:
            # 对于其他任务类型，使用带过去信息的超类的输出结构
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.use_past:
                # 如果使用过去信息，根据编码器层数设置输出结构
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return common_outputs
    # 定义一个私有方法 `_generate_dummy_inputs_for_default_and_seq2seq_lm`
    # 该方法用于生成用于默认语言模型和序列到序列语言模型的虚拟输入数据
    # 参数说明：
    #   - self: 表示类本身，即类的实例对象
    #   - tokenizer: 预训练分词器对象，用于处理文本数据
    #   - batch_size: 批次大小，控制生成的虚拟数据批次的数量
    #   - seq_length: 序列长度，控制每个生成的虚拟数据序列的长度
    #   - is_pair: 布尔值，表示是否生成成对的输入数据（例如用于序列到序列模型）
    #   - framework: 可选参数，指定生成数据的框架类型，如TensorFlow或PyTorch等
        ) -> Mapping[str, Any]:
        # 生成编码器输入数据的虚拟数据，用于模型输入
        encoder_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 生成解码器输入数据的虚拟数据
        # 如果使用过去状态（self.use_past=True），解码器序列长度为1，否则与编码器序列长度相同
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 将解码器输入数据的名称修改为以 "decoder_" 开头的形式
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 合并编码器和解码器的输入数据
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        if self.use_past:
            # 检查是否安装了 PyTorch，否则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取批量大小和编码器序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            # 获取解码器输入序列长度
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            # 获取注意力头的数量，包括编码器和解码器
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 定义编码器和解码器形状
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

            # 在解码器注意力掩码后面添加一个全1张量，以扩展过去的键
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            common_inputs["past_key_values"] = []
            # 根据模型配置中的编码器和解码器层数，生成过去的键值对
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 为每一层生成初始的过去键值对
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
            # 根据剩余的层数，继续生成过去的键值对
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        # 返回整合了所有输入数据的字典
        return common_inputs
    # 为因果语言建模生成虚拟输入数据，返回一个包含各种输入的字典
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用_encoder_and_decoder生成虚拟输入的共同部分
        common_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去的键（past_key_values）
        if self.use_past:
            # 检查是否安装了PyTorch，否则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取输入ids的批次大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            # 为past_key_values设置一个不同于输入ids长度的长度
            past_key_values_length = seqlen + 2
            # 获取编码器层数和注意力头数
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 定义past_key_values的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            # 获取mask的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 扩展attention_mask的长度以包括past_key_values
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化past_key_values为零张量的列表
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        # 返回生成的虚拟输入字典
        return common_inputs

    # 从BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering复制而来
    # 由于Marian模型没有序列分类或问答头，我们重命名了这个函数
    def _generate_dummy_inputs_for_encoder_and_decoder(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
    # 定义方法 generate_dummy_inputs，生成模型的虚拟输入数据
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务类型为 "default" 或者 "seq2seq-lm"
        if self.task in ["default", "seq2seq-lm"]:
            # 调用内部方法生成默认和序列到序列语言模型的虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        else:
            # 调用内部方法生成因果语言模型的虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        
        # 返回生成的虚拟输入数据
        return common_inputs

    # 定义方法 _flatten_past_key_values_，用于处理过去键值对的展平操作
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务类型为 "default" 或者 "seq2seq-lm"，则调用父类方法展平过去键值对
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 否则，调用具有过去信息的序列到序列配置类的父类方法展平过去键值对
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )

    # 定义属性 atol_for_validation，返回验证过程中的绝对误差容限
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```