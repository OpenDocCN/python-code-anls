# `.\transformers\models\codegen\configuration_codegen.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可信息
# 版权所有 © 2022 Salesforce 作者、EleutherAI 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（"许可证"）获得许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，
# 不作任何担保或条件，无论是明示的还是默示的。
# 有关特定语言的权限，请参阅许可证。
""" CodeGen 模型配置"""
# 导入所需模块
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

# 导入预训练 tokenizer 和其他必需模块
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射
CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/codegen-350M-nl": "https://huggingface.co/Salesforce/codegen-350M-nl/resolve/main/config.json",
    "Salesforce/codegen-350M-multi": "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/config.json",
    "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/config.json",
    "Salesforce/codegen-2B-nl": "https://huggingface.co/Salesforce/codegen-2B-nl/resolve/main/config.json",
    "Salesforce/codegen-2B-multi": "https://huggingface.co/Salesforce/codegen-2B-multi/resolve/main/config.json",
    "Salesforce/codegen-2B-mono": "https://huggingface.co/Salesforce/codegen-2B-mono/resolve/main/config.json",
    "Salesforce/codegen-6B-nl": "https://huggingface.co/Salesforce/codegen-6B-nl/resolve/main/config.json",
    "Salesforce/codegen-6B-multi": "https://huggingface.co/Salesforce/codegen-6B-multi/resolve/main/config.json",
    "Salesforce/codegen-6B-mono": "https://huggingface.co/Salesforce/codegen-6B-mono/resolve/main/config.json",
    "Salesforce/codegen-16B-nl": "https://huggingface.co/Salesforce/codegen-16B-nl/resolve/main/config.json",
    "Salesforce/codegen-16B-multi": "https://huggingface.co/Salesforce/codegen-16B-multi/resolve/main/config.json",
    "Salesforce/codegen-16B-mono": "https://huggingface.co/Salesforce/codegen-16B-mono/resolve/main/config.json",
}

# CodeGen 配置类
class CodeGenConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`CodeGenModel`] 的配置。根据指定的参数实例化一个 CodeGen 模型，定义模型架构。
    使用默认值实例化配置将产生与 CodeGen [Salesforce/codegen-2B-mono](https://huggingface.co/Salesforce/codegen-2B-mono)
    架构相似的配置。配置对象
    class CodeGenConfig(PretrainedConfig):
        # 继承自`PretrainedConfig`，用于控制模型输出。从[`PretrainedConfig`]的文档中了解更多信息。
        def __init__(
            self,
            vocab_size=50400,  # CodeGen 模型的词汇量。定义了调用[`CodeGenModel`]时`inputs_ids`可以表示的不同令牌数量。
            n_positions=2048,  # 该模型可能使用的最大序列长度。通常设置为很大的值，以防万一（例如，512或1024或2048）。
            n_ctx=2048,  # 此属性在 `CodeGenModel.__init__` 中使用，没有实际效果。
            n_embd=4096,  # 嵌入和隐藏状态的维度。
            n_layer=28,  # Transformer 编码器中的隐藏层数量。
            n_head=16,  # Transformer 编码器中每个注意力层的注意力头数量。
            rotary_dim=64,  # 旋转位置嵌入应用的嵌入维度数量。
            n_inner=None,  # 内部前馈层的维度。如果为`None`，则设置为4倍的n_embd。
            activation_function="gelu_new",  # 激活函数，在列表`["relu", "silu", "gelu", "tanh", "gelu_new"]`中选择。
            resid_pdrop=0.0,  # 嵌入、编码器和池化器中所有全连接层的 dropout 概率。
            embd_pdrop=0.0,  # 嵌入的 dropout 比率。
            attn_pdrop=0.0,  # 注意力的 dropout 比率。
            layer_norm_epsilon=1e-05,  # 在层归一化层中使用的 epsilon。
            initializer_range=0.02,  # 用于初始化所有权重矩阵的截断正态初始化器的标准差。
            use_cache=True,  # 模型是否应返回最后的 key/values 注意力（并非所有模型都使用）。
            bos_token_id=50256,  # 流的开始令牌 ID。
            eos_token_id=50256,  # 流的结束令牌 ID。
            tie_word_embeddings=False,  # 模型的输入和输出词嵌入是否应该被捆绑。注意，这仅在模型具有输出词嵌入层时相关。
        ):
            # 调用父类构造函数初始化配置。
            super().__init__(
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                **kwargs,
            )
            self.vocab_size = vocab_size
            self.n_positions = n_positions
            self.n_ctx = n_ctx
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.rotary_dim = rotary_dim
            self.n_inner = n_inner if n_inner is not None else 4 * n_embd
            self.activation_function = activation_function
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
            self.use_cache = use_cache
            self.tie_word_embeddings = tie_word_embeddings
    >>> from transformers import CodeGenConfig, CodeGenModel

    >>> # 导入 CodeGenConfig 和 CodeGenModel 类
    >>> # 从transformers库中导入CodeGenConfig和CodeGenModel类

    >>> # 初始化一个CodeGen 6B配置
    >>> # 创建一个名为configuration的CodeGenConfig对象实例，使用默认参数

    >>> configuration = CodeGenConfig()

    >>> # 初始化一个模型（具有随机权重），使用配置
    >>> # 创建一个名为model的CodeGenModel对象实例，使用configuration配置

    >>> model = CodeGenModel(configuration)

    >>> # 访问模型配置
    >>> # 从model对象中获取配置，存储到configuration变量中
    >>> configuration = model.config
    ```"""

    # 定义模型类型
    model_type = "codegen"
    # 定义属性映射，将模型配置参数名映射为变量名
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50400,
        n_positions=2048,
        n_ctx=2048,
        n_embd=4096,
        n_layer=28,
        n_head=16,
        rotary_dim=64,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        **kwargs,
    ):
        # 初始化方法，设置模型参数
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 调用父类的初始化方法
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
# 从transformers.models.gpt2.configuration_gpt2.GPT2OnnxConfig复制而来的CodeGenOnnxConfig类
class CodeGenOnnxConfig(OnnxConfigWithPast):
    # 初始化函数，接受预训练配置、任务名称、补丁规范列表和是否使用过去信息作为参数
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的初始化函数
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        # 如果配置中没有pad_token_id属性，则将其设置为0
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    # 返回输入的映射，包含input_ids和attention_mask
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        # 如果使用过去信息，则填充输入映射
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # 返回层数
    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    # 返回注意力头数
    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    # 生成虚拟输入，接受分词器、批量大小、序列长度、是否成对、框架类型等参数
    # 重写父类方法，生成包含过去信息的 ONNX 模型的虚拟输入
    def generate_dummy_inputs(
        self, tokenizer: PreTrainedTokenizerBase, batch_size: int, seq_length: int, is_pair: bool, framework: str
    ) -> Mapping[str, Any]:
        # 调用父类方法生成常规虚拟输入
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 需要按照 forward() 中出现的顺序对输入进行排序
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 需要添加过去的键值对
        if self.use_past:
            # 检查是否有可用的 PyTorch，若没有则报错
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # 对于 past_key_values 不使用相同的长度
                past_key_values_length = seqlen + 2
                # 设置 past_key_values 的形状
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                # 生成 past_key_values 输入的初始值
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        # 添加 attention_mask 到有序输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        # 若使用过去信息，则将注意力掩码扩展，以匹配 past_key_values 的长度
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回有序输入字典
        return ordered_inputs

    # 返回默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 13
```