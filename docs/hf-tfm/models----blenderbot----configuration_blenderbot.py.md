# `.\models\blenderbot\configuration_blenderbot.py`

```py
# coding=utf-8
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
Blenderbot model configuration
"""

# 导入必要的模块和类
from collections import OrderedDict  # 导入有序字典类
from typing import Any, Mapping, Optional  # 导入类型提示相关的类和函数

from ... import PreTrainedTokenizer  # 导入预训练分词器类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...file_utils import TensorType, is_torch_available  # 导入与文件操作相关的函数和类
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast  # 导入与ONNX相关的配置类
from ...onnx.utils import compute_effective_axis_dimension  # 导入计算有效轴维度的函数
from ...utils import logging  # 导入日志记录工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/config.json",
    # 查看所有Blenderbot模型请访问 https://huggingface.co/models?filter=blenderbot
}

class BlenderbotConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlenderbotModel`]. It is used to instantiate an
    Blenderbot model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Blenderbot
    [facebook/blenderbot-3B](https://huggingface.co/facebook/blenderbot-3B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```
    >>> from transformers import BlenderbotConfig, BlenderbotModel

    >>> # Initializing a Blenderbot facebook/blenderbot-3B style configuration
    >>> configuration = BlenderbotConfig()

    >>> # Initializing a model (with random weights) from the facebook/blenderbot-3B style configuration
    >>> model = BlenderbotModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "blenderbot"  # 模型类型为Blenderbot
    keys_to_ignore_at_inference = ["past_key_values"]  # 推理时需要忽略的关键字列表
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}  # 属性映射表，用于重命名属性
    # 初始化函数，用于初始化一个Transformer模型的实例
    def __init__(
        self,
        vocab_size=8008,  # 词汇表大小，默认为8008
        max_position_embeddings=128,  # 最大位置编码长度，默认为128
        encoder_layers=2,  # 编码器层数，默认为2层
        encoder_ffn_dim=10240,  # 编码器中FFN层的维度，默认为10240
        encoder_attention_heads=32,  # 编码器注意力头数，默认为32
        decoder_layers=24,  # 解码器层数，默认为24层
        decoder_ffn_dim=10240,  # 解码器中FFN层的维度，默认为10240
        decoder_attention_heads=32,  # 解码器注意力头数，默认为32
        encoder_layerdrop=0.0,  # 编码器层的dropout比例，默认为0.0（无dropout）
        decoder_layerdrop=0.0,  # 解码器层的dropout比例，默认为0.0（无dropout）
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否为编码-解码模型，默认为True
        activation_function="gelu",  # 激活函数类型，默认为GELU
        d_model=2560,  # 模型维度，默认为2560
        dropout=0.1,  # 全连接层的dropout比例，默认为0.1
        attention_dropout=0.0,  # 注意力层的dropout比例，默认为0.0（无dropout）
        activation_dropout=0.0,  # 激活函数的dropout比例，默认为0.0（无dropout）
        init_std=0.02,  # 初始化标准差，默认为0.02
        decoder_start_token_id=1,  # 解码器起始标记ID，默认为1
        scale_embedding=False,  # 是否缩放嵌入向量，默认为False
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 起始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        encoder_no_repeat_ngram_size=3,  # 编码器中不重复ngram的大小，默认为3
        forced_eos_token_id=2,  # 强制结束标记ID，默认为2
        **kwargs,  # 其他参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码长度
        self.d_model = d_model  # 设置模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器中FFN层的维度
        self.encoder_layers = encoder_layers  # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中FFN层的维度
        self.decoder_layers = decoder_layers  # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器注意力头数
        self.dropout = dropout  # 设置全连接层的dropout比例
        self.attention_dropout = attention_dropout  # 设置注意力层的dropout比例
        self.activation_dropout = activation_dropout  # 设置激活函数的dropout比例
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层的dropout比例
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层的dropout比例
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers  # 设置隐藏层的数量等于编码器层数
        self.scale_embedding = scale_embedding  # 设置是否缩放嵌入向量，如果是，缩放因子为sqrt(d_model)

        # 调用父类的初始化方法，传入相关参数
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
    # 定义 BlenderbotOnnxConfig 类，继承自 OnnxSeq2SeqConfigWithPast 类
    class BlenderbotOnnxConfig(OnnxSeq2SeqConfigWithPast):
        
        # 定义 inputs 属性，返回输入的字典映射
        @property
        def inputs(self) -> Mapping[str, Mapping[int, str]]:
            # 根据任务类型决定通用输入格式
            if self.task in ["default", "seq2seq-lm"]:
                # 如果任务是默认或者序列到序列语言模型
                common_inputs = OrderedDict(
                    [
                        ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                        ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                    ]
                )
                # 如果使用过去状态，则设置额外的输入
                if self.use_past:
                    common_inputs["decoder_input_ids"] = {0: "batch"}
                    common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
                else:
                    common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                    common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}
                # 如果使用过去状态，填充具有过去关键值的公共输入
                if self.use_past:
                    self.fill_with_past_key_values_(common_inputs, direction="inputs")
            elif self.task == "causal-lm":
                # 如果任务是因果语言模型
                common_inputs = OrderedDict(
                    [
                        ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                        ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                    ]
                )
                # 如果使用过去状态，为每个解码器层设置过去关键值的输入格式
                if self.use_past:
                    _, num_decoder_layers = self.num_layers
                    for i in range(num_decoder_layers):
                        common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                        common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
            else:
                # 默认情况下，返回完整的输入格式，包括编码器和解码器的输入
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
        # 定义 outputs 属性，返回输出的字典映射
        # 从 transformers.models.bart.configuration_bart.BartOnnxConfig.outputs 复制
        def outputs(self) -> Mapping[str, Mapping[int, str]]:
            # 根据任务类型决定通用输出格式
            if self.task in ["default", "seq2seq-lm"]:
                # 如果任务是默认或者序列到序列语言模型，使用父类的输出
                common_outputs = super().outputs
            else:
                # 否则，使用父类 OnnxConfigWithPast 的输出
                common_outputs = super(OnnxConfigWithPast, self).outputs
                # 如果使用过去状态，为每个编码器层设置现在状态的输出格式
                if self.use_past:
                    num_encoder_layers, _ = self.num_layers
                    for i in range(num_encoder_layers):
                        common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                        common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
            return common_outputs
    
        # 定义 _generate_dummy_inputs_for_default_and_seq2seq_lm 方法，用于生成默认和序列到序列语言模型的虚拟输入
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
        # 如果使用过去状态，则解码器序列长度为1，否则与编码器序列长度相同
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        
        # 将解码器输入数据的键名前添加"decoder_"前缀，并组成新的字典
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        
        # 整合编码器和解码器的输入数据为一个通用的输入字典
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去状态
        if self.use_past:
            # 检查是否可用 PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # 获取编码器输入的批次大小和序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            
            # 获取解码器输入的序列长度
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            
            # 获取注意力头的数量
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            
            # 定义编码器和解码器的张量形状
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
            
            # 在解码器注意力掩码末尾添加全1张量，用于模拟过去的键值
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )
            
            # 初始化过去的键值列表
            common_inputs["past_key_values"] = []
            
            # 获取解码器的层数
            _, num_decoder_layers = self.num_layers
            
            # 为每一层解码器生成过去的键值对
            for _ in range(num_decoder_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
        
        # 返回整合后的通用输入字典
        return common_inputs
    ) -> Mapping[str, Any]:
        # 生成用于序列分类和问答的虚拟输入数据，根据给定的tokenizer、batch_size、seq_length、is_pair和framework参数
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去的键值（past_key_values）
        if self.use_past:
            # 如果没有安装PyTorch，则抛出数值错误异常
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取输入数据的batch大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            past_key_values_length = seqlen
            # 获取解码器层数
            _, num_decoder_layers = self.num_layers
            # 获取编码器注意力头数和隐藏大小
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 计算过去键值的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            # 获取注意力掩码的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 扩展注意力掩码，以适应过去键值的长度
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化过去键值的占位符列表
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_decoder_layers)
            ]
        # 返回生成的输入数据字典
        return common_inputs

    # 从transformers.models.bart.configuration_bart.BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering中复制而来
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从OnnxConfig.generate_dummy_inputs中复制而来
        # 根据动态轴（-1）的情况，设置一个固定维度的样本数，以避免ONNX进行的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 根据动态轴（-1）的情况，设置一个固定维度的标记数，以避免ONNX进行的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批次和序列生成虚拟输入数据
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        # 使用tokenizer生成字典形式的通用输入数据
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        # 返回通用输入数据字典
        return common_inputs

    # 从transformers.models.bart.configuration_bart.BartOnnxConfig.generate_dummy_inputs中复制而来
    # 根据任务类型生成虚拟输入数据，并返回一个字典
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务类型是"default"或"seq2seq-lm"，调用适用于这两种任务的生成虚拟输入方法
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 如果任务类型是"causal-lm"，调用适用于因果语言模型任务的生成虚拟输入方法
        elif self.task == "causal-lm":
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 否则，调用适用于序列分类和问答任务的生成虚拟输入方法
        else:
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回生成的公共输入数据字典
        return common_inputs

    # 从BartOnnxConfig._flatten_past_key_values_方法复制而来
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务类型是"default"或"seq2seq-lm"，调用父类的_flatten_past_key_values_方法
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        # 否则，调用带有过去信息的OnnxSeq2SeqConfigWithPast类的父类方法_flatten_past_key_values_
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )

    # 填充包含过去信息的键值对到输入或输出的字典中
    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        # 如果方向不是"inputs"或"outputs"，抛出错误
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        # 根据方向选择适当的名称
        name = "past_key_values" if direction == "inputs" else "present"
        # 解构元组以获取编码器层数和解码器层数
        _, num_decoder_layers = self.num_layers

        # 定义编码器和解码器的序列名称
        encoder_sequence = "past_encoder_sequence"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"

        # 对每个解码器层进行迭代，填充键值对到输入或输出的字典中
        for i in range(num_decoder_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch", 2: encoder_sequence}
```