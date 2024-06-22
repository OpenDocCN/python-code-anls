# `.\transformers\models\marian\configuration_marian.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache License, Version 2.0 获取许可，如果不符合许可则不得使用该文件
# 获取许可的网址
# 如果按照适用法律或书面同意需要，在“AS IS” BASIS 分发，而没有任何形式的明示或暗示的保证或条件
# 请参阅特定语言的许可格式和限制
""" Marian model configuration"""
# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块中导入 Any, Mapping, Optional 类
from typing import Any, Mapping, Optional
# 从 transformers 模块中导入 PreTrainedTokenizer 类
from ... import PreTrainedTokenizer
# 从 transformers 模块中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从 transformers 模块中导入 OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast 类
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
# 从 transformers 模块中导入 compute_effective_axis_dimension, TensorType, is_torch_available, logging 函数
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import TensorType, is_torch_available, logging
# 从 logging 模块中获取 logger 对象
logger = logging.get_logger(__name__)
# Marian 预训练配置文件列表
MARIAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/config.json",
    # 查看所有 Marian 模型: https://huggingface.co/models?filter=marian
}

# Marian 配置类，继承自 PretrainedConfig 类
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
    ```py"""
    # 定义 model_type 属性为 "marian"
    model_type = "marian"
    # 在推断阶段要忽略的密钥列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 这是一个 Transformer 模型的初始化函数
    def __init__(
        self,
        # 词汇表大小
        vocab_size=58101,
        # 解码器的词汇表大小，默认与编码器相同
        decoder_vocab_size=None,
        # 最大位置编码长度
        max_position_embeddings=1024,
        # 编码器层数
        encoder_layers=12,
        # 编码器前馈网络维度
        encoder_ffn_dim=4096,
        # 编码器注意力头数
        encoder_attention_heads=16,
        # 解码器层数
        decoder_layers=12,
        # 解码器前馈网络维度
        decoder_ffn_dim=4096,
        # 解码器注意力头数
        decoder_attention_heads=16,
        # 编码器 LayerDrop 系数
        encoder_layerdrop=0.0,
        # 解码器 LayerDrop 系数
        decoder_layerdrop=0.0,
        # 是否使用缓存
        use_cache=True,
        # 是否是编码器-解码器模型
        is_encoder_decoder=True,
        # 激活函数
        activation_function="gelu",
        # 隐藏层维度
        d_model=1024,
        # dropout 系数
        dropout=0.1,
        # 注意力 dropout 系数
        attention_dropout=0.0,
        # 激活函数 dropout 系数
        activation_dropout=0.0,
        # 参数初始化标准差
        init_std=0.02,
        # 解码器起始 token ID
        decoder_start_token_id=58100,
        # 是否缩放词嵌入
        scale_embedding=False,
        # padding token ID
        pad_token_id=58100,
        # 结束 token ID
        eos_token_id=0,
        # 强制添加结束 token
        forced_eos_token_id=0,
        # 编解码器共享词嵌入
        share_encoder_decoder_embeddings=True,
        **kwargs,
    ):
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置解码器词汇表大小
        self.decoder_vocab_size = decoder_vocab_size or vocab_size
        # 设置最大位置编码长度
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层维度
        self.d_model = d_model
        # 设置编码器前馈网络维度
        self.encoder_ffn_dim = encoder_ffn_dim
        # 设置编码器层数
        self.encoder_layers = encoder_layers
        # 设置编码器注意力头数
        self.encoder_attention_heads = encoder_attention_heads
        # 设置解码器前馈网络维度
        self.decoder_ffn_dim = decoder_ffn_dim
        # 设置解码器层数
        self.decoder_layers = decoder_layers
        # 设置解码器注意力头数
        self.decoder_attention_heads = decoder_attention_heads
        # 设置 dropout 系数
        self.dropout = dropout
        # 设置注意力 dropout 系数
        self.attention_dropout = attention_dropout
        # 设置激活函数 dropout 系数
        self.activation_dropout = activation_dropout
        # 设置激活函数
        self.activation_function = activation_function
        # 设置参数初始化标准差
        self.init_std = init_std
        # 设置编码器 LayerDrop 系数
        self.encoder_layerdrop = encoder_layerdrop
        # 设置解码器 LayerDrop 系数
        self.decoder_layerdrop = decoder_layerdrop
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置隐藏层数
        self.num_hidden_layers = encoder_layers
        # 设置是否缩放词嵌入
        self.scale_embedding = scale_embedding
        # 设置编解码器是否共享词嵌入
        self.share_encoder_decoder_embeddings = share_encoder_decoder_embeddings
        # 调用父类初始化
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
class MarianOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    # 返回模型输入的格式
    def inputs(self) -> Mapping[str, Mapping[int, str]:
        # 如果任务是"default"或"seq2seq-lm"
        if self.task in ["default", "seq2seq-lm"]:
            # 定义常见的输入格式
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            # 如果使用过去的信息
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

            # 如果使用过去的信息
            if self.use_past:
                # 填充过去关键值
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 如果任务是"causal-lm"
        elif self.task == "causal-lm":
            # TODO: figure this case out.（TODO: 理清这种情况的处理）
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            # 如果使用过去的信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
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
    # 返回模型输出的格式
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是"default"或"seq2seq-lm"
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
            # 如果使用过去的信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return common_outputs
    # 生成给默认LM和seq2seq LM使用的虚拟输入
    # tokenizer: 预训练的分词器
    # batch_size: 批量大小，默认为-1
    # seq_length: 序列长度，默认为-1
    # is_pair: 是否为成对输入，默认为False
    # framework: 可选的张量类型，默认为None
    # 该函数用于生成用于模型输入的dummy数据
    def _generate_dummy_inputs_for_encoder_and_decoder(
        self, tokenizer: PreTrainedTokenizer, batch_size: int, seq_length: int, is_pair: bool, framework: str
    ) -> Mapping[str, Any]:
        # 生成编码器的dummy输入数据
        encoder_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(
            tokenizer, batch_size, seq_length, is_pair, framework
        )
    
        # 生成解码器的dummy输入数据
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 将解码器input数据添加前缀"decoder_"
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 将编码器和解码器的输入数据合并
        common_inputs = dict(**encoder_inputs, **decoder_inputs)
    
        # 如果使用past_key_values，则需要生成额外的past_key_values数据
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 生成编码器的past_key_values
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            # 生成解码器的past_key_values
            decoder_past_length = decoder_seq_length + 3
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )
    
            # 将解码器的注意力掩码扩展到过去的length
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )
    
            # 生成编码器和解码器的past_key_values
            common_inputs["past_key_values"] = []
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"
    
            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        return common_inputs
    # 生成因果语言模型的虚拟输入数据，返回输入数据的字典
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 生成编码器和解码器共享的虚拟输入数据
        common_inputs = self._generate_dummy_inputs_for_encoder_and_decoder(
            tokenizer, batch_size, seq_length, is_pair, framework
        )
    
        # 如果模型使用过去的键值，需要进行以下处理
        if self.use_past:
            # 如果没有安装 PyTorch，则抛出 ValueError
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取输入数据的批次大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            # 设置过去键值的长度
            past_key_values_length = seqlen + 2
            # 获取编码层和注意力头的数量
            num_encoder_layers, _ = self.num_layers
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
            # 将注意力掩码和全 1 序列拼接在一起
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化过去的键值
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        return common_inputs
    
    # 从 BartOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering 复制而来
    # 由于 Marian 模型没有序列分类或问答头，因此我们修改了此函数的名称
    def _generate_dummy_inputs_for_encoder_and_decoder(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    # 定义函数 generate_dummy_inputs，生成虚拟输入数据，返回字典形式的输入数据
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务是"default"或"seq2seq-lm"，调用 _generate_dummy_inputs_for_default_and_seq2seq_lm 方法生成输入数据
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 否则，调用 _generate_dummy_inputs_for_causal_lm 方法生成输入数据
        else:
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        
        # 返回生成的输入数据
        return common_inputs

    # 定义函数 _flatten_past_key_values_，用于平铺过去的键值对
    # Copied from transformers.models.bart.configuration_bart.BartOnnxConfig._flatten_past_key_values_
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是"default"或"seq2seq-lm"，调用超类的 _flatten_past_key_values_ 方法
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        # 否则，调用 OnnxSeq2SeqConfigWithPast 类的 _flatten_past_key_values_ 方法
        else:
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )

    # 定义属性 atol_for_validation，返回 float 类型的数值 1e-4
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```