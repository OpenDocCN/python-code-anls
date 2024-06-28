# `.\models\esm\configuration_esm.py`

```
"""
# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
""" ESM model configuration"""

# Import necessary modules
from dataclasses import asdict, dataclass
from typing import Optional

# Import configuration utilities
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# Get logger for this module
logger = logging.get_logger(__name__)

# TODO Update this
# Mapping of pretrained model names to their configuration URLs
ESM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/esm-1b": "https://huggingface.co/facebook/esm-1b/resolve/main/config.json",
    # See all ESM models at https://huggingface.co/models?filter=esm
}

# Configuration class for the ESM model, inheriting from PretrainedConfig
class EsmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ESMModel`]. It is used to instantiate a ESM model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ESM
    [facebook/esm-1b](https://huggingface.co/facebook/esm-1b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import EsmModel, EsmConfig

    >>> # Initializing a ESM facebook/esm-1b style configuration >>> configuration = EsmConfig()

    >>> # Initializing a model from the configuration >>> model = ESMModel(configuration)

    >>> # Accessing the model configuration >>> configuration = model.config
    ```
    """

    model_type = "esm"

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
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)
        # 调用父类的初始化方法，传入特定的参数来初始化当前类

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
        # 初始化多个模型配置的参数

        if is_folding_model:
            if esmfold_config is None:
                logger.info("No esmfold_config supplied for folding model, using default values.")
                # 如果没有提供 esmfold_config 参数，则使用默认配置并记录日志信息
                esmfold_config = EsmFoldConfig()
            elif isinstance(esmfold_config, dict):
                esmfold_config = EsmFoldConfig(**esmfold_config)
                # 如果 esmfold_config 是一个字典，则根据字典内容创建 EsmFoldConfig 对象
            self.esmfold_config = esmfold_config
            if vocab_list is None:
                logger.warning("No vocab_list supplied for folding model, assuming the ESM-2 vocabulary!")
                # 如果没有提供 vocab_list 参数，则假设使用 ESM-2 词汇表，并记录警告信息
                self.vocab_list = get_default_vocab_list()
            else:
                self.vocab_list = vocab_list
                # 否则，使用提供的 vocab_list 参数
        else:
            self.esmfold_config = None
            self.vocab_list = None
            # 如果不是折叠模型，则将 esmfold_config 和 vocab_list 设置为 None

        if self.esmfold_config is not None and getattr(self.esmfold_config, "use_esm_attn_map", False):
            raise ValueError("The HuggingFace port of ESMFold does not support use_esm_attn_map at this time!")
            # 如果 esmfold_config 不为 None，且其属性 use_esm_attn_map 为 True，则抛出值错误异常

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        # 调用父类的 to_dict 方法，将父类的序列化结果添加到 output 字典中

        if isinstance(self.esmfold_config, EsmFoldConfig):
            output["esmfold_config"] = self.esmfold_config.to_dict()
            # 如果 esmfold_config 是 EsmFoldConfig 类型的对象，则将其序列化为字典并加入 output 中

        return output
        # 返回包含当前实例所有属性的字典作为序列化结果
# 数据类 EsmFoldConfig，用于配置 ESM 折叠模型的参数
@dataclass
class EsmFoldConfig:
    # ESM 类型，默认为 None
    esm_type: str = None
    # 是否使用 FP16 格式的 ESM
    fp16_esm: bool = True
    # 是否使用 ESM 注意力映射
    use_esm_attn_map: bool = False
    # 是否剔除 ESM 的成对序列
    esm_ablate_pairwise: bool = False
    # 是否剔除 ESM 的序列
    esm_ablate_sequence: bool = False
    # ESM 输入的 dropout 概率
    esm_input_dropout: float = 0

    # 是否嵌入氨基酸信息
    embed_aa: bool = True
    # 是否绕过语言模型
    bypass_lm: bool = False

    # LDDT 头部隐藏维度
    lddt_head_hid_dim: int = 128
    # EsmFoldConfig 的 trunk 配置，如果为 None 则使用默认配置
    trunk: "TrunkConfig" = None

    # 初始化方法，在对象创建后调用，处理 trunk 属性
    def __post_init__(self):
        # 如果 trunk 为 None，则使用默认的 TrunkConfig
        if self.trunk is None:
            self.trunk = TrunkConfig()
        # 如果 trunk 是 dict 类型，则将其转换为 TrunkConfig 对象
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)

    # 将当前实例序列化为 Python 字典的方法
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将当前实例转换为字典
        output = asdict(self)
        # 将 trunk 属性也转换为字典
        output["trunk"] = self.trunk.to_dict()
        return output


# 数据类 TrunkConfig，用于配置 ESM 折叠模型的 trunk 参数
@dataclass
class TrunkConfig:
    # trunk 的块数
    num_blocks: int = 48
    # 序列状态维度
    sequence_state_dim: int = 1024
    # 成对状态维度
    pairwise_state_dim: int = 128
    # 序列头部宽度
    sequence_head_width: int = 32
    # 成对头部宽度
    pairwise_head_width: int = 32
    # 位置分箱数
    position_bins: int = 32
    # dropout 概率
    dropout: float = 0
    # 层丢弃概率
    layer_drop: float = 0
    # 是否使用 CPU 梯度检查点
    cpu_grad_checkpoint: bool = False
    # 最大循环次数
    max_recycles: int = 4
    # 分块大小
    chunk_size: Optional[int] = 128
    # 结构模块配置
    structure_module: "StructureModuleConfig" = None
    # 初始化方法，在对象实例化后自动调用。确保配置的正确性和一致性。
    def __post_init__(self):
        # 如果结构模块未指定，则使用默认的结构模块配置
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        # 如果结构模块是一个字典，则将其转换为结构模块配置对象
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)

        # 检查最大循环次数是否大于零，否则抛出数值错误异常
        if self.max_recycles <= 0:
            raise ValueError(f"`max_recycles` should be positive, got {self.max_recycles}.")
        
        # 检查序列状态维度是否是其自身的倍数，否则抛出数值错误异常
        if self.sequence_state_dim % self.sequence_state_dim != 0:
            raise ValueError(
                "`sequence_state_dim` should be a round multiple of `sequence_state_dim`, got"
                f" {self.sequence_state_dim} and {self.sequence_state_dim}."
            )
        
        # 检查成对状态维度是否是其自身的倍数，否则抛出数值错误异常
        if self.pairwise_state_dim % self.pairwise_state_dim != 0:
            raise ValueError(
                "`pairwise_state_dim` should be a round multiple of `pairwise_state_dim`, got"
                f" {self.pairwise_state_dim} and {self.pairwise_state_dim}."
            )

        # 计算序列头的数量，确保序列状态维度与序列头宽度的乘积相等
        sequence_num_heads = self.sequence_state_dim // self.sequence_head_width
        if self.sequence_state_dim != sequence_num_heads * self.sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {self.sequence_state_dim} != {sequence_num_heads} * {self.sequence_head_width}."
            )
        
        # 计算成对头的数量，确保成对状态维度与成对头宽度的乘积相等
        pairwise_num_heads = self.pairwise_state_dim // self.pairwise_head_width
        if self.pairwise_state_dim != pairwise_num_heads * self.pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {self.pairwise_state_dim} != {pairwise_num_heads} * {self.pairwise_head_width}."
            )
        
        # 检查成对状态维度是否为偶数，否则抛出数值错误异常
        if self.pairwise_state_dim % 2 != 0:
            raise ValueError(f"`pairwise_state_dim` should be even, got {self.pairwise_state_dim}.")

        # 检查丢弃率是否小于0.4，否则抛出数值错误异常
        if self.dropout >= 0.4:
            raise ValueError(f"`dropout` should not be greater than 0.4, got {self.dropout}.")

    # 将当前实例序列化为Python字典的方法。覆盖默认的to_dict方法。
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将对象的所有属性转换为字典
        output = asdict(self)
        # 将结构模块属性转换为其对应的字典表示
        output["structure_module"] = self.structure_module.to_dict()
        return output
@dataclass
class StructureModuleConfig:
    """
    定义了结构模块的配置参数的数据类。

    Args:
        sequence_dim:
            单一表示通道的维度
        pairwise_dim:
            成对表示通道的维度
        ipa_dim:
            IPA 隐藏通道的维度
        resnet_dim:
            Angle resnet（Alg. 23 lines 11-14）隐藏通道的维度
        num_heads_ipa:
            IPA 头的数量
        num_qk_points:
            在IPA期间生成的查询/键点的数量
        num_v_points:
            在IPA期间生成的值点的数量
        dropout_rate:
            层中使用的dropout率
        num_blocks:
            结构模块的块数量
        num_transition_layers:
            单一表示转换中的层数（Alg. 23 lines 8-9）
        num_resnet_blocks:
            Angle resnet 中的块数量
        num_angles:
            Angle resnet 中生成的角度数量
        trans_scale_factor:
            单一表示转换的隐藏维度的比例因子
        epsilon:
            Angle resnet 归一化中使用的小数值
        inf:
            用于注意力屏蔽的大数值
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

    def to_dict(self):
        """
        将数据类实例转换为字典的方法。
        """
        return asdict(self)


def get_default_vocab_list():
    """
    返回默认的词汇表列表。

    Returns:
        tuple: 包含默认词汇的元组
    """
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