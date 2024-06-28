# `.\models\bart\configuration_bart.py`

```py
# 导入警告模块，用于在需要时发出警告
import warnings
# OrderedDict 是一个有序字典，可以记录元素插入的顺序
from collections import OrderedDict
# Any、Mapping 和 Optional 是用于类型提示的特定类型
from typing import Any, Mapping, Optional

# 导入预训练分词器 PreTrainedTokenizer
from ... import PreTrainedTokenizer
# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入 OnnxConfig、OnnxConfigWithPast 和 OnnxSeq2SeqConfigWithPast 用于 ONNX 模型配置
from ...onnx import OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
# 导入计算轴维度的工具函数
from ...onnx.utils import compute_effective_axis_dimension
# 导入 TensorType 用于处理张量类型，is_torch_available 用于检查是否有 torch 库，logging 用于日志记录
from ...utils import TensorType, is_torch_available, logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# BART 预训练模型配置文件的映射，指定每个预训练模型的配置文件 URL
BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
    # 更多 BART 模型配置文件映射可见于 https://huggingface.co/models?filter=bart
}

# BartConfig 是用于存储 BART 模型配置的类，继承自 PretrainedConfig
class BartConfig(PretrainedConfig):
    r"""
    这是用于存储 [`BartModel`] 配置的类。它用于根据指定的参数实例化 BART 模型，定义模型架构。
    使用默认参数实例化配置将得到类似于 BART [facebook/bart-large](https://huggingface.co/facebook/bart-large) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请阅读 [`PretrainedConfig`] 的文档。

    Example:

    ```
    >>> from transformers import BartConfig, BartModel

    >>> # 初始化一个 BART facebook/bart-large 风格的配置
    >>> configuration = BartConfig()

    >>> # 使用该配置初始化一个模型（带有随机权重）
    >>> model = BartModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型设定为 "bart"
    model_type = "bart"
    # 推断过程中忽略的键列表，这里忽略 "past_key_values"
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，将 "num_attention_heads" 映射为 "encoder_attention_heads"，"hidden_size" 映射为 "d_model"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化函数，用于初始化一个 Transformer 模型的参数和配置
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为 50265
        max_position_embeddings=1024,  # 最大位置编码长度，默认为 1024
        encoder_layers=12,  # 编码器层数，默认为 12 层
        encoder_ffn_dim=4096,  # 编码器中 Feed Forward 网络的维度，默认为 4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为 16
        decoder_layers=12,  # 解码器层数，默认为 12 层
        decoder_ffn_dim=4096,  # 解码器中 Feed Forward 网络的维度，默认为 4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为 16
        encoder_layerdrop=0.0,  # 编码器层的 dropout 比例，默认为 0.0（不使用）
        decoder_layerdrop=0.0,  # 解码器层的 dropout 比例，默认为 0.0（不使用）
        activation_function="gelu",  # 激活函数类型，默认为 GELU
        d_model=1024,  # 模型的维度，默认为 1024
        dropout=0.1,  # 全连接层和注意力层的 dropout 比例，默认为 0.1
        attention_dropout=0.0,  # 注意力机制中的 dropout 比例，默认为 0.0（不使用）
        activation_dropout=0.0,  # 激活函数中的 dropout 比例，默认为 0.0（不使用）
        init_std=0.02,  # 参数初始化的标准差，默认为 0.02
        classifier_dropout=0.0,  # 分类器中的 dropout 比例，默认为 0.0（不使用）
        scale_embedding=False,  # 是否对嵌入进行缩放，默认为 False
        use_cache=True,  # 是否使用缓存，默认为 True
        num_labels=3,  # 标签数量，默认为 3
        pad_token_id=1,  # 填充 token 的 ID，默认为 1
        bos_token_id=0,  # 开始 token 的 ID，默认为 0
        eos_token_id=2,  # 结束 token 的 ID，默认为 2
        is_encoder_decoder=True,  # 是否为编码解码模型，默认为 True
        decoder_start_token_id=2,  # 解码器开始 token 的 ID，默认为 2
        forced_eos_token_id=2,  # 强制结束 token 的 ID，默认为 2
        **kwargs,  # 其他关键字参数，用于接收额外的配置
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置编码长度
        self.d_model = d_model  # 初始化模型维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 初始化编码器的 Feed Forward 网络维度
        self.encoder_layers = encoder_layers  # 初始化编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 初始化编码器的注意力头数量
        self.decoder_ffn_dim = decoder_ffn_dim  # 初始化解码器的 Feed Forward 网络维度
        self.decoder_layers = decoder_layers  # 初始化解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 初始化解码器的注意力头数量
        self.dropout = dropout  # 初始化全连接层和注意力层的 dropout 比例
        self.attention_dropout = attention_dropout  # 初始化注意力机制中的 dropout 比例
        self.activation_dropout = activation_dropout  # 初始化激活函数中的 dropout 比例
        self.activation_function = activation_function  # 初始化激活函数类型
        self.init_std = init_std  # 初始化参数初始化的标准差
        self.encoder_layerdrop = encoder_layerdrop  # 初始化编码器层的 dropout 比例
        self.decoder_layerdrop = decoder_layerdrop  # 初始化解码器层的 dropout 比例
        self.classifier_dropout = classifier_dropout  # 初始化分类器中的 dropout 比例
        self.use_cache = use_cache  # 初始化是否使用缓存的标志
        self.num_hidden_layers = encoder_layers  # 初始化隐藏层的数量为编码器层数
        self.scale_embedding = scale_embedding  # 初始化是否对嵌入进行缩放的标志（如果为 True，则缩放因子为 sqrt(d_model)）
    
        super().__init__(
            num_labels=num_labels,  # 调用父类构造函数初始化标签数量
            pad_token_id=pad_token_id,  # 调用父类构造函数初始化填充 token 的 ID
            bos_token_id=bos_token_id,  # 调用父类构造函数初始化开始 token 的 ID
            eos_token_id=eos_token_id,  # 调用父类构造函数初始化结束 token 的 ID
            is_encoder_decoder=is_encoder_decoder,  # 调用父类构造函数初始化是否为编码解码模型的标志
            decoder_start_token_id=decoder_start_token_id,  # 调用父类构造函数初始化解码器开始 token 的 ID
            forced_eos_token_id=forced_eos_token_id,  # 调用父类构造函数初始化强制结束 token 的 ID
            **kwargs,  # 将额外的关键字参数传递给父类构造函数
        )
    
        # 确保对于 BART CNN 模型的向后兼容性
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id  # 如果未指定强制开始 token 的 ID，则使用默认的开始 token 的 ID
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )
class BartOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设定通用输入格式
        if self.task in ["default", "seq2seq-lm"]:
            # 如果任务为默认或序列到序列语言建模，设定常见的输入格式
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )

            if self.use_past:
                # 如果使用过去信息，设定解码器输入和注意力掩码的格式
                common_inputs["decoder_input_ids"] = {0: "batch"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
            else:
                # 否则，设定解码器输入和注意力掩码的另一种格式
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

            if self.use_past:
                # 如果使用过去信息，填充带有过去信息的键值对
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        elif self.task == "causal-lm":
            # 如果任务是因果语言建模，设定输入格式并处理过去的键值对
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),
                ]
            )
            if self.use_past:
                # 如果使用过去信息，为每个编码器层填充过去键和值的格式
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        else:
            # 对于其他任务类型，设定通用的输入格式，包括解码器输入和注意力掩码
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
        # 根据任务类型设定通用输出格式
        if self.task in ["default", "seq2seq-lm"]:
            # 如果任务为默认或序列到序列语言建模，调用父类方法获取常见的输出格式
            common_outputs = super().outputs
        else:
            # 对于其他任务类型，调用父类方法获取带有过去信息的输出格式
            common_outputs = super(OnnxConfigWithPast, self).outputs
            if self.use_past:
                # 如果使用过去信息，为每个编码器层填充当前键和值的格式
                num_encoder_layers, _ = self.num_layers
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}
        return common_outputs

    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        # 生成默认和序列到序列语言建模的虚拟输入
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
        # 将解码器输入中的每个张量命名为"decoder_name"，并存放在decoder_inputs字典中
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 将编码器和解码器的输入合并到common_inputs字典中
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去的信息
        if self.use_past:
            # 检查是否安装了PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # 获取输入张量的形状信息
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

            # 扩展解码器注意力遮罩的长度，并添加到common_inputs中
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化past_key_values列表
            common_inputs["past_key_values"] = []
            
            # 根据编码器和解码器层数的较小值生成past_key_values
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            
            # 确定需要初始化的剩余层次的名称
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 为每一层生成初始化的past_key_values元组
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
            # 如果存在剩余的层次，使用相应的形状生成初始化的past_key_values元组
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        
        # 返回生成的common_inputs字典
        return common_inputs
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用另一个方法生成通用输入，用于序列分类和问答任务的虚拟输入生成
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        if self.use_past:
            # 检查是否使用了 self.use_past，若使用且没有安装 PyTorch，则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            
            # 获取输入数据的批次大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            
            # 计算 past_key_values 的长度，比输入序列长度多 2
            past_key_values_length = seqlen + 2
            
            # 获取编码器层数和注意力头数
            num_encoder_layers, _ = self.num_layers
            num_encoder_attention_heads, _ = self.num_attention_heads
            
            # 计算 past_key_values 的形状
            past_shape = (
                batch,
                num_encoder_attention_heads,
                past_key_values_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            
            # 获取注意力掩码的数据类型，并将其扩展以适应新的 past_key_values 长度
            mask_dtype = common_inputs["attention_mask"].dtype
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            
            # 初始化 past_key_values 列表，每个层级都有一个零填充的 past_key_values 元组
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        
        # 返回生成的通用输入字典
        return common_inputs

    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从 OnnxConfig.generate_dummy_inputs 复制的代码
        # 为了代码清晰性没有使用 super(OnnxConfigWithPast, self).generate_dummy_inputs
        # 如果动态轴为 -1，则使用固定的 2 个样本维度来避免 ONNX 的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴为 -1，则使用固定的 8 个标记来避免 ONNX 的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批次大小和序列长度生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        
        # 使用 tokenizer 将虚拟输入转换为字典形式的通用输入
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        
        # 返回通用输入字典
        return common_inputs
    # 生成虚拟输入数据，根据不同的任务类型调用相应的内部方法来生成
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 如果任务类型是"default"或"seq2seq-lm"，调用适用于这两种任务的虚拟输入生成方法
        if self.task in ["default", "seq2seq-lm"]:
            common_inputs = self._generate_dummy_inputs_for_default_and_seq2seq_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 如果任务类型是"causal-lm"，调用适用于因果语言模型任务的虚拟输入生成方法
        elif self.task == "causal-lm":
            common_inputs = self._generate_dummy_inputs_for_causal_lm(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )
        # 对于其他任务类型，调用适用于序列分类和问答任务的虚拟输入生成方法
        else:
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回生成的通用输入数据
        return common_inputs

    # 根据任务类型选择性地扁平化过去的键值对数据
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务类型是"default"或"seq2seq-lm"，调用父类方法处理扁平化过去的键值对数据
        if self.task in ["default", "seq2seq-lm"]:
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 对于其他任务类型，使用带有过去状态的配置类的父类方法处理扁平化过去的键值对数据
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```