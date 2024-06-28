# `.\models\m2m_100\configuration_m2m_100.py`

```
# coding=utf-8
# 定义了文件的编码格式为 UTF-8

# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 依据 Apache License, Version 2.0 授权许可，详细条款可在此获取：http://www.apache.org/licenses/LICENSE-2.0

# You may obtain a copy of the License at
# 可在上述网址获取许可证副本

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。
# 详细信息请参阅许可证

""" M2M100 model configuration"""
# M2M100 模型配置

from collections import OrderedDict
# 导入 OrderedDict 数据结构

from typing import Any, Mapping, Optional
# 导入类型提示

from ... import PreTrainedTokenizer
# 导入预训练的 Tokenizer

from ...configuration_utils import PretrainedConfig
# 导入配置工具中的预训练配置

from ...onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
# 导入 ONNX 相关配置

from ...onnx.utils import compute_effective_axis_dimension
# 导入计算有效轴维度的工具函数

from ...utils import TensorType, is_torch_available, logging
# 导入工具函数：张量类型、是否可用 Torch、日志记录

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json",
    # 预训练模型的存档映射，链接指向 M2M100 模型的配置文件
    # 查看所有 M2M100 模型，请访问 https://huggingface.co/models?filter=m2m_100
}


class M2M100Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`M2M100Model`]. It is used to instantiate an
    M2M100 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the M2M100
    [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import M2M100Config, M2M100Model

    >>> # Initializing a M2M100 facebook/m2m100_418M style configuration
    >>> configuration = M2M100Config()

    >>> # Initializing a model (with random weights) from the facebook/m2m100_418M style configuration
    >>> model = M2M100Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # M2M100 模型的配置类，用于存储和实例化模型的配置参数

    model_type = "m2m_100"
    # 模型类型为 "m2m_100"

    keys_to_ignore_at_inference = ["past_key_values"]
    # 推断过程中忽略的键名列表，例如 "past_key_values"

    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 属性映射，将外部命名映射到内部模型使用的命名，例如 "num_attention_heads" 映射到 "encoder_attention_heads"
    # 初始化函数，用于创建一个新的Transformer模型实例
    def __init__(
        self,
        vocab_size=128112,  # 词汇表大小，默认为128112
        max_position_embeddings=1024,  # 最大位置编码数，默认为1024
        encoder_layers=12,  # 编码器层数，默认为12层
        encoder_ffn_dim=4096,  # 编码器中间层维度，默认为4096
        encoder_attention_heads=16,  # 编码器注意力头数，默认为16个
        decoder_layers=12,  # 解码器层数，默认为12层
        decoder_ffn_dim=4096,  # 解码器中间层维度，默认为4096
        decoder_attention_heads=16,  # 解码器注意力头数，默认为16个
        encoder_layerdrop=0.05,  # 编码器层dropout率，默认为0.05
        decoder_layerdrop=0.05,  # 解码器层dropout率，默认为0.05
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码-解码结构，默认为True
        activation_function="relu",  # 激活函数类型，默认为ReLU
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # 全连接层和注意力层的dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力层中的dropout率，默认为0.1
        activation_dropout=0.0,  # 激活函数中的dropout率，默认为0.0
        init_std=0.02,  # 参数初始化标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器起始标记ID，默认为2
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 结束标记ID，默认为2
        **kwargs,  # 其他关键字参数，用于传递给父类初始化函数
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置编码数
        self.d_model = d_model  # 初始化模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 初始化编码器中间层维度
        self.encoder_layers = encoder_layers  # 初始化编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 初始化编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 初始化解码器中间层维度
        self.decoder_layers = decoder_layers  # 初始化解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 初始化解码器注意力头数
        self.dropout = dropout  # 初始化全连接层和注意力层的dropout率
        self.attention_dropout = attention_dropout  # 初始化注意力层中的dropout率
        self.activation_dropout = activation_dropout  # 初始化激活函数中的dropout率
        self.activation_function = activation_function  # 初始化激活函数类型
        self.init_std = init_std  # 初始化参数初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 初始化编码器层dropout率
        self.decoder_layerdrop = decoder_layerdrop  # 初始化解码器层dropout率
        self.use_cache = use_cache  # 初始化是否使用缓存
        self.num_hidden_layers = encoder_layers  # 初始化隐藏层的数量为编码器层数
        self.scale_embedding = scale_embedding  # 初始化是否对嵌入进行缩放

        # 调用父类的初始化函数，传入相关参数
        super().__init__(
            pad_token_id=pad_token_id,  # 传入填充标记ID
            bos_token_id=bos_token_id,  # 传入起始标记ID
            eos_token_id=eos_token_id,  # 传入结束标记ID
            is_encoder_decoder=is_encoder_decoder,  # 传入是否是编码-解码结构
            decoder_start_token_id=decoder_start_token_id,  # 传入解码器起始标记ID
            **kwargs,  # 传入其他关键字参数
        )
class M2M100OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义通用的输入格式字典
        common_inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
            ]
        )

        # 根据是否使用过去状态，确定decoder的输入格式
        if self.use_past:
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去状态，调用填充过去键值的方法，填充通用输入字典
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 返回最终的输入格式字典
        return common_inputs

    # 从BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering复制而来
    # 名称更适合是_generate_dummy_inputs_for_encoder_and_decoder，因为M2M100不支持序列分类和问答，
    # 但保留此名称以便检查副本是否与BART的匹配，并在需要时进行更新。
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从OnnxConfig.generate_dummy_inputs复制而来
        # 为了代码清晰性，没有使用super(OnnxConfigWithPast, self).generate_dummy_inputs。
        # 如果动态轴（-1），则前向传播时采用固定维度的2个样本以避免ONNX做的优化。
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴（-1），则前向传播时采用固定维度的8个标记以避免ONNX做的优化。
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批次和序列长度生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs

    # 从transformers.models.bart.configuration_bart.BartOnnxConfig._generate_dummy_inputs_for_default_and_seq2seq_lm复制而来
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        ) -> Mapping[str, Any]:
        # 生成编码器输入数据
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 生成解码器输入数据
        # 如果使用过去信息，则解码器序列长度为1，否则与编码器序列长度相同
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 将解码器输入数据格式化为以"decoder_"开头的命名格式
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 整合编码器和解码器的输入数据
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        if self.use_past:
            # 检查是否安装了 PyTorch，如果没有则抛出异常
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # 获取批次大小和编码器序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            # 获取解码器输入序列长度
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            # 获取注意力头的数量
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

            # 扩展解码器注意力掩码，以确保其长度与decoder_past_length相同
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化过去键值列表
            common_inputs["past_key_values"] = []

            # 根据模型配置中的编码器和解码器层数，初始化过去键值对
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 对于最小层数，初始化过去键值对为零张量
            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )

            # 添加剩余层数的过去键值对，如果是编码器优先，则使用编码器的形状，否则使用解码器的形状
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))

        # 返回整合了所有输入数据的字典
        return common_inputs
    # 将函数_generate_dummy_inputs_for_default_and_seq2seq_lm赋值给generate_dummy_inputs变量
    generate_dummy_inputs = _generate_dummy_inputs_for_default_and_seq2seq_lm
```