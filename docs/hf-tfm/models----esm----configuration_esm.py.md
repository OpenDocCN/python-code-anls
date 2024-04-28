# `.\models\esm\configuration_esm.py`

```
# 设定编码为 utf-8
# 版权声明，版权所有，Meta 和 HuggingFace 公司团队，保留所有权利。
# 根据 Apache 许可证 2.0 版本，您不得使用此文件，除非符合许可证的规定。
# 您可以从以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的担保或条件。请查看许可证，了解特定语言的具体授权和限制。

# 导入必要的包和模块
from dataclasses import asdict, dataclass
from typing import Optional
# 导入配置工具类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 使用 logging 模块获取记录器
logger = logging.get_logger(__name__)

# TODO 待更新
# ESM 预训练模型配置文件映射，指向对应模型的配置文件链接
ESM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/esm-1b": "https://huggingface.co/facebook/esm-1b/resolve/main/config.json",
    # 查看所有 ESM 模型 https://huggingface.co/models?filter=esm
}

# ESM 模型配置类，用于存储 [`ESMModel`] 的配置信息
class EsmConfig(PretrainedConfig):
    r"""
    这是用于存储 [`ESMModel`] 配置的配置类。根据指定的参数实例化 ESM 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 ESM [facebook/esm-1b](https://huggingface.co/facebook/esm-1b) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    用法示例：

    ```python
    >>> from transformers import EsmModel, EsmConfig

    >>> # 初始化 ESM facebook/esm-1b 风格配置 >>> configuration = EsmConfig()

    >>> # 从配置初始化模型 >>> model = ESMModel(configuration)

    >>> # 访问模型配置 >>> configuration = model.config
    ```
    """

    # 模型类型为 "esm"
    model_type = "esm"

    # 初始化方法定义了 ESM 配置的各个参数
    def __init__(
        self,
        vocab_size=None,
        mask_token_id=None,
        pad_token_id=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        emb_layer_norm_before=None,
        token_dropout=False,
        is_folding_model=False,
        esmfold_config=None,
        vocab_list=None,
        **kwargs,
        ):   
        # 调用父类的构造方法，并传入参数 pad_token_id、mask_token_id 以及其他关键字参数
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)

        # 设置词汇表大小、隐藏层大小、隐藏层数量、注意力头数量等模型参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.is_folding_model = is_folding_model
        
        # 判断是否为折叠模型，如果是，则根据传入的参数初始化折叠配置和词汇表
        if is_folding_model:
            if esmfold_config is None:  # 如果esmfold_config为None，则使用默认值
                logger.info("No esmfold_config supplied for folding model, using default values.")
                esmfold_config = EsmFoldConfig()
            elif isinstance(esmfold_config, dict):  # 如果esmfold_config是字典类型，则使用其提供的配置参数
                esmfold_config = EsmFoldConfig(**esmfold_config)
            self.esmfold_config = esmfold_config
            if vocab_list is None:  # 如果没有提供vocab_list，则使用默认的词汇表
                logger.warning("No vocab_list supplied for folding model, assuming the ESM-2 vocabulary!")
                self.vocab_list = get_default_vocab_list()
            else:
                self.vocab_list = vocab_list
        else:  # 如果不是折叠模型，则将折叠配置和词汇表设置为None
            self.esmfold_config = None
            self.vocab_list = None
        
        # 判断是否配置了使用 ESM 的注意力图，如果配置了，则抛出异常
        if self.esmfold_config is not None and getattr(self.esmfold_config, "use_esm_attn_map", False):
            raise ValueError("The HuggingFace port of ESMFold does not support use_esm_attn_map at this time!")

    # 将当前实例序列化为Python字典
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()  # 调用父类的to_dict方法，获得基本配置字典
        # 如果配置了esmfold_config，则将其序列化为字典并添加到输出字典中
        if isinstance(self.esmfold_config, EsmFoldConfig):
            output["esmfold_config"] = self.esmfold_config.to_dict()
        return output  # 返回输出字典
# 声明一个数据类 EsmFoldConfig，用于配置 ESM 折叠模型的参数
@dataclass
class EsmFoldConfig:
    # ESM 模型类型，默认为 None
    esm_type: str = None
    # 是否使用 FP16 格式的 ESM
    fp16_esm: bool = True
    # 是否使用 ESM 的注意力图
    use_esm_attn_map: bool = False
    # 是否关闭 ESM 的两两注意力
    esm_ablate_pairwise: bool = False
    # 是否关闭 ESM 的序列注意力
    esm_ablate_sequence: bool = False
    # ESM 输入的丢弃率，默认为 0
    esm_input_dropout: float = 0

    # 是否嵌入氨基酸特征，默认为 True
    embed_aa: bool = True
    # 是否绕过语言模型
    bypass_lm: bool = False

    # LDDT 头部隐藏维度，默认为 128
    lddt_head_hid_dim: int = 128
    # TrunkConfig 对象，用于配置 Trunk 模型
    trunk: "TrunkConfig" = None

    # 初始化函数，在初始化对象后执行，如果 trunk 为空，则初始化为 TrunkConfig 对象
    def __post_init__(self):
        if self.trunk is None:
            self.trunk = TrunkConfig()
        # 如果 trunk 是字典，则将其转换为 TrunkConfig 对象
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)

    # 将实例序列化为 Python 字典的方法，覆盖默认的 `~PretrainedConfig.to_dict`
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将实例转换为字典形式
        output = asdict(self)
        # 将 trunk 属性转换为字典形式
        output["trunk"] = self.trunk.to_dict()
        return output


# 声明一个数据类 TrunkConfig，用于配置 Trunk 模型的参数
@dataclass
class TrunkConfig:
    # Trunk 模型的块数，默认为 48
    num_blocks: int = 48
    # 序列状态维度，默认为 1024
    sequence_state_dim: int = 1024
    # 两两状态维度，默认为 128
    pairwise_state_dim: int = 128
    # 序列头部宽度，默认为 32
    sequence_head_width: int = 32
    # 两两头部宽度，默认为 32
    pairwise_head_width: int = 32
    # 位置分箱数，默认为 32
    position_bins: int = 32
    # 丢弃率，默认为 0
    dropout: float = 0
    # 层丢弃率，默认为 0
    layer_drop: float = 0
    # 是否在 CPU 上进行梯度检查点，默认为 False
    cpu_grad_checkpoint: bool = False
    # 最大循环次数，默认为 4
    max_recycles: int = 4
    # 分块大小，默认为 128
    chunk_size: Optional[int] = 128
    # StructureModuleConfig 对象，用于配置结构模块
    structure_module: "StructureModuleConfig" = None
    # 初始化方法，用于在对象创建后进行初始化操作
    def __post_init__(self):
        # 如果结构模块为空，则使用默认配置
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        # 如果结构模块是字典形式，则转换为结构模块配置对象
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)

        # 检查最大循环次数是否为正数，若不是则引发 ValueError 异常
        if self.max_recycles <= 0:
            raise ValueError(f"`max_recycles` should be positive, got {self.max_recycles}.")
        # 检查序列状态维度是否为序列头宽度的整数倍，若不是则引发 ValueError 异常
        if self.sequence_state_dim % self.sequence_state_dim != 0:
            raise ValueError(
                "`sequence_state_dim` should be a round multiple of `sequence_state_dim`, got"
                f" {self.sequence_state_dim} and {self.sequence_state_dim}."
            )
        # 检查成对状态维度是否为成对头宽度的整数倍，若不是则引发 ValueError 异常
        if self.pairwise_state_dim % self.pairwise_state_dim != 0:
            raise ValueError(
                "`pairwise_state_dim` should be a round multiple of `pairwise_state_dim`, got"
                f" {self.pairwise_state_dim} and {self.pairwise_state_dim}."
            )

        # 计算序列头数
        sequence_num_heads = self.sequence_state_dim // self.sequence_head_width
        # 计算成对头数
        pairwise_num_heads = self.pairwise_state_dim // self.pairwise_head_width

        # 检查序列状态维度是否等于序列头数乘以序列头宽度，若不是则引发 ValueError 异常
        if self.sequence_state_dim != sequence_num_heads * self.sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {self.sequence_state_dim} != {sequence_num_heads} * {self.sequence_head_width}."
            )
        # 检查成对状态维度是否等于成对头数乘以成对头宽度，若不是则引发 ValueError 异常
        if self.pairwise_state_dim != pairwise_num_heads * self.pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {self.pairwise_state_dim} != {pairwise_num_heads} * {self.pairwise_head_width}."
            )
        # 检查成对状态维度是否为偶数，若不是则引发 ValueError 异常
        if self.pairwise_state_dim % 2 != 0:
            raise ValueError(f"`pairwise_state_dim` should be even, got {self.pairwise_state_dim}.")

        # 检查丢弃率是否小于0.4，若不是则引发 ValueError 异常
        if self.dropout >= 0.4:
            raise ValueError(f"`dropout` should not be greater than 0.4, got {self.dropout}.")

    # 将配置实例序列化为 Python 字典的方法，覆盖默认的 `~PretrainedConfig.to_dict` 方法
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将对象转换为字典形式，并序列化结构模块配置对象
        output = asdict(self)
        output["structure_module"] = self.structure_module.to_dict()
        return output
# 定义一个数据类，用于存储结构模块的配置信息
@dataclass
class StructureModuleConfig:
    """
    Args:
        sequence_dim:
            Single representation channel dimension  # 单一表示通道维度
        pairwise_dim:
            Pair representation channel dimension  # 成对表示通道维度
        ipa_dim:
            IPA hidden channel dimension  # IPA 隐藏通道维度
        resnet_dim:
            Angle resnet (Alg. 23 lines 11-14) hidden channel dimension  # 角度 Resnet (Alg. 23 lines 11-14) 隐藏通道维度
        num_heads_ipa:
            Number of IPA heads  # IPA 的头数
        num_qk_points:
            Number of query/key points to generate during IPA  # 在 IPA 过程中生成的查询/键点数
        num_v_points:
            Number of value points to generate during IPA  # 在 IPA 过程中生成的值点数
        dropout_rate:
            Dropout rate used throughout the layer  # 在整个层中使用的 dropout 率
        num_blocks:
            Number of structure module blocks  # 结构模块块数
        num_transition_layers:
            Number of layers in the single representation transition (Alg. 23 lines 8-9)  # 单一表示转换中的层数 (Alg. 23 lines 8-9)
        num_resnet_blocks:
            Number of blocks in the angle resnet  # 角度 Resnet 中的块数
        num_angles:
            Number of angles to generate in the angle resnet  # 角度 Resnet 中生成的角度数
        trans_scale_factor:
            Scale of single representation transition hidden dimension  # 单一表示转换隐藏维度的比例
        epsilon:
            Small number used in angle resnet normalization  # 角度 Resnet 归一化中使用的小数
        inf:
            Large number used for attention masking  # 用于注意力掩模的大数字
    """

    sequence_dim: int = 384
    pairwise_dim: int = 128
    ipa_dim: int = 16
    resnet_dim: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_transition_layers: int = 1
    num_resnet_blocks: int = 2
    num_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5

    # 将配置信息转换为字典格式
    def to_dict(self):
        return asdict(self)


# 获取默认的词汇列表
def get_default_vocab_list():
    return (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
    )
```