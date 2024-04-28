# `.\transformers\models\blenderbot_small\configuration_blenderbot_small.py`

```
# 指定编码格式为 UTF-8

# 导入必要的模块和类
# collections 模块中的 OrderedDict 类用于创建有序字典
# typing 模块中的 Any、Mapping、Optional 类用于类型提示
from collections import OrderedDict
from typing import Any, Mapping, Optional

# 从 transformers 模块中导入 PreTrainedTokenizer、PretrainedConfig 类
# configuration_utils 模块中的 PretrainedConfig 类用于存储预训练模型的配置信息
# file_utils 模块中的 TensorType、is_torch_available 函数用于处理文件和检查 Torch 是否可用
# onnx 模块中的 OnnxConfig、OnnxConfigWithPast、OnnxSeq2SeqConfigWithPast 类用于处理 ONNX 配置
# onnx.utils 模块中的 compute_effective_axis_dimension 函数用于计算有效的轴维度
# utils 模块中的 logging 模块用于记录日志
from ... import PreTrainedTokenizer
from ...configuration_utils import PretrainedConfig
from ...file_utils import TensorType, is_torch_available
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
from ...utils import logging

# 获取 logging 模块中的 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射
BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/config.json",
    # 查看所有 BlenderbotSmall 模型的链接：https://huggingface.co/models?filter=blenderbot_small
}


# BlenderbotSmallConfig 类，用于存储 BlenderbotSmall 模型的配置信息
class BlenderbotSmallConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlenderbotSmallModel`]. It is used to instantiate
    an BlenderbotSmall model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall
    [facebook/blenderbot_small-90M](https://huggingface.co/facebook/blenderbot_small-90M) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import BlenderbotSmallConfig, BlenderbotSmallModel

    >>> # Initializing a BlenderbotSmall facebook/blenderbot_small-90M style configuration
    >>> configuration = BlenderbotSmallConfig()

    >>> # Initializing a model (with random weights) from the facebook/blenderbot_small-90M style configuration
    >>> model = BlenderbotSmallModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型为 blenderbot-small
    model_type = "blenderbot-small"
    # 推理时需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，用于在不同配置之间进行属性映射
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化函数，用于初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        encoder_layers=8,  # 编码器层数，默认为8
        encoder_ffn_dim=2048,  # 编码器中全连接层维度，默认为2048
        encoder_attention_heads=16,  # 编码器中注意力头数，默认为16
        decoder_layers=8,  # 解码器层数，默认为8
        decoder_ffn_dim=2048,  # 解码器中全连接层维度，默认为2048
        decoder_attention_heads=16,  # 解码器中注意力头数，默认为16
        encoder_layerdrop=0.0,  # 编码器层的dropout率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层的dropout率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码器-解码器模型，默认为True
        activation_function="gelu",  # 激活函数，默认为gelu
        d_model=512,  # 模型维度，默认为512
        dropout=0.1,  # 普通dropout率，默认为0.1
        attention_dropout=0.0,  # 注意力dropout率，默认为0.0
        activation_dropout=0.0,  # 激活函数dropout率，默认为0.0
        init_std=0.02,  # 参数初始化标准差，默认为0.02
        decoder_start_token_id=1,  # 解码器起始标记ID，默认为1
        scale_embedding=False,  # 是否对嵌入进行缩放，默认为False，如果为True，则缩放因子为sqrt(d_model)
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 开始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        forced_eos_token_id=2,  # 强制结束标记ID，默认为2
        **kwargs,  # 其他关键字参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置嵌入数
        self.d_model = d_model  # 设置模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 设置编码器中全连接层维度
        self.encoder_layers = encoder_layers  # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器中注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中全连接层维度
        self.decoder_layers = decoder_layers  # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器中注意力头数
        self.dropout = dropout  # 设置普通dropout率
        self.attention_dropout = attention_dropout  # 设置注意力dropout率
        self.activation_dropout = activation_dropout  # 设置激活函数dropout率
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置参数初始化标准差
        self.encoder_layerdrop = encoder_layerdrop  # 设置编码器层的dropout率
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层的dropout率
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers  # 设置隐藏层数，与编码器层数相同
        self.scale_embedding = scale_embedding  # 设置是否对嵌入进行缩放，如果为True，则缩放因子为sqrt(d_model)
    
        # 调用父类的初始化函数，传入相关参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,  # 其他关键字参数
        )
# 定义 BlenderbotSmallOnnxConfig 类，继承自 OnnxSeq2SeqConfigWithPast 类
class BlenderbotSmallOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义 inputs 属性，返回输入映射字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为默认或者 seq2seq-lm，则执行以下代码块
        if self.task in ["default", "seq2seq-lm"]:
            # 定义通用输入字典
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入的编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码
                ]
            )

            # 如果使用过去的信息
            if self.use_past:
                # 添加过去的解码器输入和注意力掩码到通用输入字典
                common_inputs["decoder_input_ids"] = {0: "batch"}  # 解码器的输入
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}  # 解码器的注意力掩码
            else:
                # 否则，添加当前解码器输入和注意力掩码到通用输入字典
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}  # 解码器的输入
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}  # 解码器的注意力掩码

            # 如果使用过去的信息
            if self.use_past:
                # 使用 fill_with_past_key_values_ 方法填充通用输入字典中的过去信息
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 如果任务为 causal-lm
        elif self.task == "causal-lm":
            # TODO: figure this case out.
            # 目前尚未实现该情况，暂时保留注释，可能需要后续处理
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入的编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码
                ]
            )
            # 如果使用过去的信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                # 遍历编码器层，为过去信息中的键和值添加相应的说明
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}  # 过去键
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}  # 过去值
        else:
            # 如果任务不是默认或者 seq2seq-lm，则添加所有输入到通用输入字典
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入的编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 编码器注意力掩码
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),  # 解码器的输入
                    ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),  # 解码器的注意力掩码
                ]
            )

        # 返回通用输入字典
        return common_inputs

    # 定义 outputs 属性，返回输出映射字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务为默认或者 seq2seq-lm，则调用父类的 outputs 方法
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs
        else:
            # 否则，调用父类 OnnxConfigWithPast 的 outputs 方法
            common_outputs = super(OnnxConfigWithPast, self).outputs
            # 如果使用过去的信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                # 遍历编码器层，为当前信息中的键和值添加相应的说明
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}  # 当前键
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}  # 当前值
        # 返回通用输出字典
        return common_outputs
    # 生成默认和序列到序列语言模型的虚拟输入
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,  # 接受一个预训练的分词器对象作为参数
        batch_size: int = -1,  # 批大小，默认为-1
        seq_length: int = -1,  # 序列长度，默认为-1
        is_pair: bool = False,  # 是否为成对数据，默认为False
        framework: Optional[TensorType] = None,  # 框架类型，默认为None
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
    # 生成用于因果语言模型的虚拟输入
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 生成用于序列分类和问答的虚拟输入，继承自父类方法
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去信息
        if self.use_past:
            # 如果未安装 PyTorch，则无法生成虚拟过去键（past_keys）输入
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取批次大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            # 计算过去键（past_keys）的长度，比序列长度多2
            past_key_values_length = seqlen + 2
            # 获取编码器层数和注意力头数
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            # 定义过去键的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )

            # 获取注意力掩码的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 在注意力掩码末尾添加额外的1，以适应过去键的长度
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 生成空的过去键值对
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        # 返回生成的输入
        return common_inputs

    # 生成用于序列分类和问答的虚拟输入
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 根据动态轴（-1）确定批次大小，以避免 ONNX 的优化影响
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 根据动态轴（-1）确定序列长度，以避免 ONNX 的优化影响
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        # 使用 tokenizer 将虚拟输入转换为模型输入格式
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        # 返回生成的输入
        return common_inputs
    # 生成虚拟输入数据的方法，用于模型推理
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务是默认任务或者序列到序列语言建模任务
        if self.task in ["default", "seq2seq-lm"]:
            # 调用默认任务和序列到序列语言建模任务的虚拟输入数据生成方法
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 如果任务是因果语言建模任务
        elif self.task == "causal-lm":
            # 调用因果语言建模任务的虚拟输入数据生成方法
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        else:
            # 调用序列分类和问题回答任务的虚拟输入数据生成方法
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回生成的虚拟输入数据
        return common_inputs

    # 将过去的键值扁平化的方法
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是默认任务或者序列到序列语言建模任务
        if self.task in ["default", "seq2seq-lm"]:
            # 调用父类的方法将过去的键值扁平化
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 调用带过去状态的序列到序列配置类的方法将过去的键值扁平化
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```  
```