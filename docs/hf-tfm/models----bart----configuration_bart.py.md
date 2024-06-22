# `.\transformers\models\bart\configuration_bart.py`

```py
# 设置编码格式为UTF-8
# 版权声明
# 导入警告模块
# 导入有序字典类
# 导入类型提示相关的模块
# 导入预训练的分词器类
# 导入预训练配置相关的模块
# 导入ONNX相关的配置
# 导入ONNX的过去相关配置
# 导入ONNX的带有过去信息的序列到序列的配置
# 导入ONNX工具函数
# 导入张量类型相关的模块
# 导入是否可用PyTorch的标志
# 导入日志记录相关的模块

# 获取日志记录器
logger = logging.get_logger(__name__)

# BART预训练配置文件的归档映射，键为模型名称，值为配置文件的URL
BART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
    # 查看所有BART模型的列表：https://huggingface.co/models?filter=bart
}

# BART模型配置类，用于存储BART模型的配置信息
class BartConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储[`BartModel`]的配置。根据指定的参数，它用于实例化一个BART模型，定义模型架构。
    使用默认参数实例化一个配置对象会生成一个类似于BART [facebook/bart-large](https://huggingface.co/facebook/bart-large) 架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档获取更多信息。

    示例:

    ```python
    >>> from transformers import BartConfig, BartModel

    >>> # 初始化一个BART facebook/bart-large风格的配置
    >>> configuration = BartConfig()

    >>> # 从facebook/bart-large风格的配置初始化一个（具有随机权重）模型
    >>> model = BartModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py
    """

    # 模型类型为"bart"
    model_type = "bart"
    # 推理时忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为50265
        max_position_embeddings=1024,  # 最大位置编码，默认为1024
        encoder_layers=12,  # 编码器层数，默认为12
        encoder_ffn_dim=4096,  # 编码器中前馈网络的维度，默认为4096
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为16
        decoder_layers=12,  # 解码器层数，默认为12
        decoder_ffn_dim=4096,  # 解码器中前馈网络的维度，默认为4096
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为16
        encoder_layerdrop=0.0,  # 编码器层丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器层丢弃率，默认为0.0
        activation_function="gelu",  # 激活函数，默认为GELU
        d_model=1024,  # 模型维度，默认为1024
        dropout=0.1,  # 通用丢弃率，默认为0.1
        attention_dropout=0.0,  # 注意力机制中的丢弃率，默认为0.0
        activation_dropout=0.0,  # 激活函数的丢弃率，默认为0.0
        init_std=0.02,  # 参数初始化的标准差，默认为0.02
        classifier_dropout=0.0,  # 分类器中的丢弃率，默认为0.0
        scale_embedding=False,  # 是否对嵌入进行缩放，默认为False；如果为True，则缩放因子为sqrt(d_model)
        use_cache=True,  # 是否使用缓存，默认为True
        num_labels=3,  # 标签数量，默认为3
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 终止标记ID，默认为2
        is_encoder_decoder=True,  # 是否为编码-解码结构，默认为True
        decoder_start_token_id=2,  # 解码器起始标记ID，默认为2
        forced_eos_token_id=2,  # 强制终止标记ID，默认为2
        **kwargs,
    ):
        # 将各种参数赋值给对象属性
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
        self.num_hidden_layers = encoder_layers  # 隐藏层数量等于编码器层数
        self.scale_embedding = scale_embedding  # 如果为True，则嵌入将进行缩放，缩放因子为sqrt(d_model)

        # 调用父类初始化方法，并传入相应参数
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        # 确保对BART CNN模型的向后兼容性
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            # 如果强制起始标记ID为None，并且在kwargs中设置了"force_bos_token_to_be_generated"为True，则设置为默认的起始标记ID
            self.forced_bos_token_id = self.bos_token_id
            # 发出警告，提醒在未来版本中包括`forced_bos_token_id={self.bos_token_id}`以确保向后兼容性
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                "The config can simply be saved and uploaded again to be fixed."
            )
class BartOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义 inputs 属性，返回输入数据的映射关系字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是默认或者 seq2seq-lm
        if self.task in ["default", "seq2seq-lm"]:
            # 定义通用输入字典
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入的编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码
                ]
            )

            # 如果使用过去信息
            if self.use_past:
                common_inputs["decoder_input_ids"] = {0: "batch"}  # 解码器输入的 ID
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}  # 解码器的注意力掩码
            else:
                common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}  # 解码器输入的 ID
                common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}  # 解码器的注意力掩码

            # 如果使用过去信息，填充通用输入字典的过去信息
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
        # 如果任务是因果-lm
        elif self.task == "causal-lm":
            # TODO: figure this case out.
            # 定义通用输入字典
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入的编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码
                ]
            )
            # 如果使用过去信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                # 遍历编码器层数，添加过去键值对应的信息
                for i in range(num_encoder_layers):
                    common_inputs[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}  # 过去键
                    common_inputs[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}  # 过去值
        else:
            # 默认情况下的通用输入字典
            common_inputs = OrderedDict(
                [
                    ("input_ids", {0: "batch", 1: "encoder_sequence"}),  # 输入的编码器序列
                    ("attention_mask", {0: "batch", 1: "encoder_sequence"}),  # 注意力掩码
                    ("decoder_input_ids", {0: "batch", 1: "decoder_sequence"}),  # 解码器输入的 ID
                    ("decoder_attention_mask", {0: "batch", 1: "decoder_sequence"}),  # 解码器的注意力掩码
                ]
            )

        return common_inputs  # 返回通用输入字典

    # 定义 outputs 属性，返回输出数据的映射关系字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是默认或者 seq2seq-lm
        if self.task in ["default", "seq2seq-lm"]:
            common_outputs = super().outputs  # 调用父类的 outputs 方法获取通用输出字典
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs  # 调用父类的 outputs 方法获取通用输出字典
            # 如果使用过去信息
            if self.use_past:
                num_encoder_layers, _ = self.num_layers
                # 遍历编码器层数，添加当前键值对应的信息
                for i in range(num_encoder_layers):
                    common_outputs[f"present.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}  # 当前键
                    common_outputs[f"present.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}  # 当前值
        return common_outputs  # 返回通用输出字典

    # 为默认和 seq2seq-lm 生成虚拟输入
    def _generate_dummy_inputs_for_default_and_seq2seq_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    # 定义函数签名，指定输入和输出类型
    ) -> Mapping[str, Any]:
        # 生成用于序列分类和问答任务的虚拟输入，包括编码器输入
        encoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 生成解码器输入
        decoder_seq_length = seq_length if not self.use_past else 1
        # 生成用于序列分类和问答任务的虚拟输入，包括解码器输入
        decoder_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, decoder_seq_length, is_pair, framework
        )
        # 将解码器输入中的张量添加前缀以与编码器输入区分开
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        # 合并编码器和解码器输入
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        # 如果使用过去状态
        if self.use_past:
            # 检查是否安装了 PyTorch
            if not is_torch_available():
                # 抛出错误，因为没有安装 PyTorch
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                # 导入 PyTorch
                import torch
            # 获取批次大小和编码器序列长度
            batch, encoder_seq_length = common_inputs["input_ids"].shape
            # 获取解码器序列长度
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            # 获取编码器和解码器注意力头的数量
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            # 定义编码器张量形状
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            # 定义解码器过去状态长度
            decoder_past_length = decoder_seq_length + 3
            # 定义解码器张量形状
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                decoder_past_length,
                self._config.hidden_size // num_decoder_attention_heads,
            )

            # 将解码器注意力掩码延长以适应过去状态
            common_inputs["decoder_attention_mask"] = torch.cat(
                [common_inputs["decoder_attention_mask"], torch.ones(batch, decoder_past_length)], dim=1
            )

            # 初始化过去键值
            common_inputs["past_key_values"] = []
            # 如果模型配置中存在编码器和解码器层数，则都考虑
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            # 初始化过去键值列表
            for _ in range(min_num_layers):
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )
            # 对剩余的层初始化过去键值列表
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))
        # 返回公共输入
        return common_inputs
    # 生成用于因果语言建模（causal language modeling）的虚拟输入
    def _generate_dummy_inputs_for_causal_lm(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 生成用于序列分类和问答的虚拟输入
        common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
            tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 如果使用过去（past）信息
        if self.use_past:
            # 检查是否有安装 PyTorch
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            # 获取输入的批次大小和序列长度
            batch, seqlen = common_inputs["input_ids"].shape
            # 为 past_key_values 设置不同的长度
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

            # 获取 attention_mask 的数据类型
            mask_dtype = common_inputs["attention_mask"].dtype
            # 将注意力掩码扩展到新的长度
            common_inputs["attention_mask"] = torch.cat(
                [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )
            # 初始化 past_key_values
            common_inputs["past_key_values"] = [
                (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(num_encoder_layers)
            ]
        # 返回通用输入
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
        # 从 OnnxConfig.generate_dummy_inputs 复制代码
        # 为了代码清晰度，没有使用 super(OnnxConfigWithPast, self).generate_dummy_inputs
        # 如果动态轴 (-1)，则以固定维度 2 个样本进行前向传播，以避免 ONNX 所做的优化
        batch_size = compute_effective_axis_dimension(
            batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
        )

        # 如果动态轴 (-1)，则以固定维度 8 个 token 进行前向传播，以避免 ONNX 所做的优化
        token_to_add = tokenizer.num_special_tokens_to_add(is_pair)
        seq_length = compute_effective_axis_dimension(
            seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
        )

        # 根据计算的批次和序列生成虚拟输入
        dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
        common_inputs = dict(tokenizer(dummy_input, return_tensors=framework))
        # 返回通用输入
        return common_inputs
    # 生成虚拟输入数据，返回一个包含各种任务通用输入的字典
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
        else:
            # 为序列分类和问答任务生成虚拟输入数据
            common_inputs = self._generate_dummy_inputs_for_sequence_classification_and_question_answering(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

        # 返回通用输入数据字典
        return common_inputs

    # 将过去的键值对展平化
    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        # 如果任务是默认任务或者序列到序列语言模型任务
        if self.task in ["default", "seq2seq-lm"]:
            # 调用父类方法展平化过去的键值对
            flattened_output = super()._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            # 调用特定类的父类方法展平化过去的键值对
            flattened_output = super(OnnxSeq2SeqConfigWithPast, self)._flatten_past_key_values_(
                flattened_output, name, idx, t
            )
```