# `.\transformers\models\m2m_100\configuration_m2m_100.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权属于 Fairseq 作者和 HuggingFace Inc. 团队，保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可您不得使用此文件。
# 您可以在以下网址获取许可的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依照此许可分发的软件将按“原样”分布，
# 没有任何明示或暗示的担保或条件。请查看许可证以获得有关权限和限制的更多信息。
"""M2M100 模型配置"""
# 导入所需的类和函数
from collections import OrderedDict
from typing import Any, Mapping, Optional
# 从 transformers 库中导入预训练的 tokenizer
from ... import PreTrainedTokenizer
# 从 configuration_utils 模块中导入预训练配置基类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 onnx 模块中导入相关配置类和函数
from ...onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
from ...onnx.utils import compute_effective_axis_dimension
# 从 utils 模块中导入相关类和函数
from ...utils import TensorType, is_torch_available, logging

# 获取 logger 实例，用于记录日志信息
logger = logging.get_logger(__name__)

# 包含 M2M100 预训练模型配置文件的 URL 映射
M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json",
    # 查看所有 M2M100 模型的信息 https://huggingface.co/models?filter=m2m_100
}


# M2M100 配置类，继承自预训练配置基类 PretrainedConfig
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
    ```py"""

    # 模型类型为 "m2m_100"
    model_type = "m2m_100"
    # 推断时忽略的键列表为 ["past_key_values"]
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将属性映射到 encoder_attention_heads 和 d_model
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化Transformer模型的参数
    def __init__(
        self,
        vocab_size=128112,  # 词汇表大小
        max_position_embeddings=1024,  # 最大位置编码长度
        encoder_layers=12,  # 编码器层数
        encoder_ffn_dim=4096,  # 编码器中前馈网络的维度
        encoder_attention_heads=16,  # 编码器中注意力头的数量
        decoder_layers=12,  # 解码器层数
        decoder_ffn_dim=4096,  # 解码器中前馈网络的维度
        decoder_attention_heads=16,  # 解码器中注意力头的数量
        encoder_layerdrop=0.05,  # 编码器层的随机丢弃比例
        decoder_layerdrop=0.05,  # 解码器层的随机丢弃比例
        use_cache=True,  # 是否使用缓存
        is_encoder_decoder=True,  # 是否是编码-解码模型
        activation_function="relu",  # 激活函数
        d_model=1024,  # 模型维度
        dropout=0.1,  # 普通丢弃比例
        attention_dropout=0.1,  # 注意力层的丢弃比例
        activation_dropout=0.0,  # 激活函数的丢弃比例
        init_std=0.02,  # 参数初始化的标准差
        decoder_start_token_id=2,  # 解码器的起始标记ID
        scale_embedding=True,  # 是否缩放嵌入，如果为True，则缩放因子为sqrt(d_model)
        pad_token_id=1,  # 填充标记ID
        bos_token_id=0,  # 起始标记ID
        eos_token_id=2,  # 结束标记ID
        **kwargs,  # 其他参数
    ):
        # 将参数赋值给实例变量
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
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # 如果为True，则缩放因子为sqrt(d_model)
    
        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
class M2M100OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义常见的输入格式，使用 OrderedDict 以保持顺序
        common_inputs = OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 编码器输入的标识符
                ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 编码器输入的注意力掩码
            ]
        )

        # 如果使用过去的状态，添加解码器的输入格式
        if self.use_past:
            common_inputs["decoder_input_ids"] = {0: "batch"}  # 解码器输入的标识符
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}  # 解码器输入的注意力掩码
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}  # 解码器输入的标识符
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}  # 解码器输入的注意力掩码

        # 如果使用过去的状态，填充常见输入格式
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 返回常见的输入格式
        return common_inputs

    # 从BARTOnnxConfig._generate_dummy_inputs_for_sequence_classification_and_question_answering中复制而来
    # 更好的命名应该是_generate_dummy_inputs_for_encoder_and_decoder，但为了能够检查复制是否与BART所做的一致，将其保留下来，以便在需要时进行更新。
    def _generate_dummy_inputs_for_sequence_classification_and_question_answering(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 从OnnxConfig.generate_dummy_inputs中复制而来，为了代码清晰性，没有使用super(OnnxConfigWithPast, self).generate_dummy_inputs
        # 如果动态轴（-1），则将固定维度设置为2个样本，以避免ONNX所做的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴（-1），则将固定维度设置为8个标记，以避免ONNX所做的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批量和序列生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        return common_inputs

    # 从transformers.models.bart.configuration_bart.BartOnnxConfig._generate_dummy_inputs_for_default_and_seq2seq_lm中复制而来
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        ) -> Mapping[str, Any]:
        # 为序列分类和问答任务生成编码器输入
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 生成解码器输入
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 组合编码器和解码器的输入
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去信息
        if self.use_past:
            # 如果没有安装 PyTorch，则抛出异常
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 计算编码器和解码器的形状
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

            # 在解码器注意力掩码后面添加全 1 张量
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化 past_key_values 列表
            common_inputs["past_key_values"] = []
            # 如果模型配置中存在编码器和解码器层的数量，则都予以考虑
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 为编码器和解码器的 past_key_values 添加初始化值
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
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        # 返回合并后的输入
        return common_inputs
    # 将函数_generate_dummy_inputs_for_default_and_seq2seq_lm赋值给generate_dummy_inputs变量
    generate_dummy_inputs = _generate_dummy_inputs_for_default_and_seq2seq_lm
```