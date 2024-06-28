# `.\models\mbart\configuration_mbart.py`

```py
# coding=utf-8
# 上面是指定文件编码为 UTF-8，确保支持多语言字符集
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
# 版权声明，指出代码版权归 Facebook AI Research Team 和 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 指明采用 Apache 许可证 2.0 版本
# you may not use this file except in compliance with the License.
# 在符合许可证条件的情况下才能使用该文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 软件在适用法律要求或书面同意的情况下按“原样”分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证，了解特定语言管理权限和限制

""" MBART model configuration"""
# 导入必要的模块和类
from collections import OrderedDict  # 导入 OrderedDict 类，用于有序字典
from typing import Any, Mapping, Optional  # 导入必要的类型声明，如 Any、Mapping、Optional

from ... import PreTrainedTokenizer  # 导入预训练 Tokenizer
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast  # 导入 ONNX 相关配置类
from ...onnx.utils import compute_effective_axis_dimension  # 导入计算有效轴维度的函数
from ...utils import TensorType, is_torch_available, logging  # 导入 TensorType、is_torch_available 和 logging

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

MBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/config.json",
    # 预训练模型映射字典，指定 MBART 大型模型的配置文件地址
    # 查看所有 MBART 模型地址 https://huggingface.co/models?filter=mbart
}


class MBartConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MBartModel`]. It is used to instantiate an MBART
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MBART
    [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import MBartConfig, MBartModel

    >>> # Initializing a MBART facebook/mbart-large-cc25 style configuration
    >>> configuration = MBartConfig()

    >>> # Initializing a model (with random weights) from the facebook/mbart-large-cc25 style configuration
    >>> model = MBartModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # MBART 配置类，用于存储 MBART 模型的配置信息

    model_type = "mbart"  # 模型类型为 mbart
    keys_to_ignore_at_inference = ["past_key_values"]  # 推断时忽略的键名列表，包含 "past_key_values"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 属性映射字典，将 num_attention_heads 映射为 encoder_attention_heads，hidden_size 映射为 d_model
    # 定义一个初始化方法，用于初始化Transformer模型的参数和配置
    def __init__(
        self,
        vocab_size=50265,                          # 词汇表大小，默认为50265
        max_position_embeddings=1024,              # 最大位置嵌入长度，默认为1024
        encoder_layers=12,                         # 编码器层数，默认为12层
        encoder_ffn_dim=4096,                      # 编码器中FFN层的维度，默认为4096
        encoder_attention_heads=16,                # 编码器中注意力头的数量，默认为16个
        decoder_layers=12,                         # 解码器层数，默认为12层
        decoder_ffn_dim=4096,                      # 解码器中FFN层的维度，默认为4096
        decoder_attention_heads=16,                # 解码器中注意力头的数量，默认为16个
        encoder_layerdrop=0.0,                     # 编码器层的层丢弃率，默认为0.0（不丢弃）
        decoder_layerdrop=0.0,                     # 解码器层的层丢弃率，默认为0.0（不丢弃）
        use_cache=True,                            # 是否使用缓存，默认为True
        is_encoder_decoder=True,                   # 是否是编码-解码结构，默认为True
        activation_function="gelu",                # 激活函数，默认为GELU
        d_model=1024,                              # 模型维度，默认为1024
        dropout=0.1,                               # 全局Dropout率，默认为0.1
        attention_dropout=0.0,                     # 注意力Dropout率，默认为0.0
        activation_dropout=0.0,                    # 激活函数Dropout率，默认为0.0
        init_std=0.02,                             # 权重初始化标准差，默认为0.02
        classifier_dropout=0.0,                    # 分类器Dropout率，默认为0.0
        scale_embedding=False,                     # 是否缩放嵌入，默认为False
        pad_token_id=1,                            # 填充token的ID，默认为1
        bos_token_id=0,                            # 起始token的ID，默认为0
        eos_token_id=2,                            # 终止token的ID，默认为2
        forced_eos_token_id=2,                     # 强制终止token的ID，默认为2
        **kwargs,                                  # 其他参数，作为关键字参数传递
    ):
        self.vocab_size = vocab_size                # 初始化词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入长度
        self.d_model = d_model                      # 初始化模型维度
        self.encoder_ffn_dim = encoder_ffn_dim      # 初始化编码器中FFN层的维度
        self.encoder_layers = encoder_layers        # 初始化编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 初始化编码器中注意力头的数量
        self.decoder_ffn_dim = decoder_ffn_dim      # 初始化解码器中FFN层的维度
        self.decoder_layers = decoder_layers        # 初始化解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 初始化解码器中注意力头的数量
        self.dropout = dropout                      # 初始化全局Dropout率
        self.attention_dropout = attention_dropout  # 初始化注意力Dropout率
        self.activation_dropout = activation_dropout  # 初始化激活函数Dropout率
        self.activation_function = activation_function  # 初始化激活函数类型
        self.init_std = init_std                    # 初始化权重初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 初始化编码器层的层丢弃率
        self.decoder_layerdrop = decoder_layerdrop  # 初始化解码器层的层丢弃率
        self.classifier_dropout = classifier_dropout  # 初始化分类器Dropout率
        self.use_cache = use_cache                  # 初始化是否使用缓存
        self.num_hidden_layers = encoder_layers     # 初始化隐藏层的数量，与编码器层数相同
        self.scale_embedding = scale_embedding      # 初始化是否缩放嵌入的标志
        super().__init__(                            # 调用父类的初始化方法
            pad_token_id=pad_token_id,               # 传递填充token的ID
            bos_token_id=bos_token_id,               # 传递起始token的ID
            eos_token_id=eos_token_id,               # 传递终止token的ID
            is_encoder_decoder=is_encoder_decoder,   # 传递是否是编码-解码结构的标志
            forced_eos_token_id=forced_eos_token_id, # 传递强制终止token的ID
            **kwargs                                 # 传递其他关键字参数
        )
# 从Bart配置类BartOnnxConfig复制，并将Bart改为MBart
class MBartOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是"default"或"seq2seq-lm"，则设置通用输入字典
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            # 如果使用过去信息，则设置解码器的输入ID和注意力掩码
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

            # 如果使用过去信息，则填充输入中的过去键值对
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        elif self.task == "causal-lm":
            # TODO: 需要处理这种情况。
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            # 如果使用过去信息，则为每个编码器层设置过去键和值
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
            # 否则设置完整的输入字典，包括解码器相关信息
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
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是"default"或"seq2seq-lm"，则获取默认的输出字典
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs
        else:
            # 否则调用父类的输出方法获取输出字典，并为每个编码器层设置当前键和值
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return common_outputs
    # 定义一个方法 `_generate_dummy_inputs_for_default_and_seq2seq_lm`，用于生成默认和序列到序列语言模型的虚拟输入数据
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )
        # 生成编码器的输入数据，用于序列分类和问答任务的虚拟输入
        # 根据参数生成编码器的输入数据，包括tokenization对象、批量大小、序列长度、是否为成对输入、框架类型

        # Generate decoder inputs
        decoder_seq_length = seq_length if not self.use_past else 1
        # 计算解码器的序列长度，若使用过去状态则设为1，否则设为与编码器相同的序列长度
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 生成解码器的输入数据，用于序列分类和问答任务的虚拟输入，根据参数生成解码器的输入数据

        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 将解码器输入数据的键名修改为带有前缀"decoder_"的形式

        common_inputs = dict(**encoder_inputs, **decoder_inputs)
        # 将编码器和解码器的输入数据合并成一个字典，作为公共输入数据

        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 检查是否使用过去状态，并验证是否安装了PyTorch

            batch, encoder_seq_length = common_inputs["input_ids"].shape
            # 获取批量大小和编码器序列长度

            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            # 获取解码器的序列长度

            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 获取编码器和解码器的注意力头数目

            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            # 计算编码器的形状

            decoder_past_length = decoder_seq_length + 3
            # 计算解码器的过去状态长度

            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )
            # 计算解码器的形状

            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )
            # 将解码器的注意力遮罩扩展到包括过去状态长度的维度

            common_inputs["past_key_values"] = []
            # 初始化过去键值列表

            # If the number of encoder and decoder layers are present in the model configuration, both are considered
            num_encoder_layers, num_decoder_layers = self.num_layers
            # 获取编码器和解码器的层数

            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            # 计算最小层数

            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            # 计算最大层数

            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"
            # 根据层数的差异确定剩余的一方是编码器还是解码器

            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            # 为每一层编码器和解码器生成零张量，并添加到过去键值列表中

            # TODO: test this.
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            # 根据剩余一方的名称确定形状

            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
            # 为剩余层数生成零张量，并添加到过去键值列表中

        return common_inputs
    # 生成用于因果语言模型的虚拟输入数据集，返回一个映射字典
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用另一个方法生成用于序列分类和问答的虚拟输入数据集
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        if self.use_past:
            # 检查是否需要使用过去键值（past_key_values）
            if not is_torch_available():
                # 如果没有安装 PyTorch，抛出异常
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch, seqlen = common_inputs["input_ids"].shape
            # 计算过去键值的长度，不使用与输入相同的长度
            past_key_values_length = seqlen + 2
            # 获取编码器层和注意力头的数量
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 定义过去键值的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            # 获取注意力掩码的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 扩展现有的注意力掩码，增加过去键值的长度
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化过去键值列表
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        # 返回生成的输入数据集字典
        return common_inputs

    # 生成用于序列分类和问答的虚拟输入数据集
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从 OnnxConfig.generate_dummy_inputs 复制此方法
        # 为了代码清晰性，没有使用 super(OnnxConfigWithPast, self).generate_dummy_inputs
        # 计算有效的轴维度，以避免 ONNX 的优化影响，固定样本维度为2个样本
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 计算要添加的特殊标记的数量，并计算有效的序列维度，固定令牌维度为8个令牌
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批次和序列生成虚拟输入数据
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        # 使用 tokenizer 将虚拟输入转换为张量并返回作为字典
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
        # 如果任务是"default"或"seq2seq-lm"，调用特定方法生成对应任务的虚拟输入
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 如果任务是"causal-lm"，调用特定方法生成对应任务的虚拟输入
        elif self.task == "causal-lm":
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 对于其他任务，调用特定方法生成适用于序列分类和问答的虚拟输入
        else:
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回生成的通用输入字典
        return common_inputs

    # 根据任务类型调用不同的方法来扁平化过去的键值对
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是"default"或"seq2seq-lm"，调用父类方法来处理
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        # 对于其他任务，调用继承类"OnnxSeq2SeqConfigWithPast"的方法来处理
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```