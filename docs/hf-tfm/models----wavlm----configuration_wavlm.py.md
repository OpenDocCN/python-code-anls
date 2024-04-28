# `.\transformers\models\wavlm\configuration_wavlm.py`

```
# 设置脚本文件的编码格式为UTF-8
# 版权声明
# 根据 Apache 2.0 许可证授权，可以在符合许可证的情况下使用该文件
# 可以在下面网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，
# 没有明示或暗示的保证或条件，不提供任何保证。
# 请查阅许可证以获取特定语言的权限和限制信息
""" WavLM 模型配置"""

import functools
import operator

# 导入预训练配置的工具函数
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 预训练配置映射 (模型名称 : 配置文件 URL)
WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/wavlm-base": "https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json",
    # 查看所有 WavLM 模型：https://huggingface.co/models?filter=wavlm
}


class WavLMConfig(PretrainedConfig):
    r"""
    这是用于存储 [`WavLMModel`] 配置的类。根据指定的参数实例化 WavLM 模型的配置，
    定义模型架构。使用默认值实例化配置将产生类似于 WavLM
    [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读
    [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python

    ```

    示例:

    ```python
    >>> from transformers import WavLMConfig, WavLMModel

    >>> # 初始化一个 WavLM facebook/wavlm-base-960h 风格的配置
    >>> configuration = WavLMConfig()

    >>> # 从 facebook/wavlm-base-960h 风格的配置初始化一个模型（随机权重）
    >>> model = WavLMModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "wavlm"
    # 定义一个类，用于搭建一个模型，其中包含一系列的参数设置
    def __init__(
        self,
        vocab_size=32,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        num_buckets=320,
        max_bucket_distance=800,
        do_stable_layer_norm=False,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,
        proj_codevector_dim=256,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        xvector_output_dim=512,
        num_ctc_classes=80,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        add_adapter=False,
        adapter_kernel_size=3,
        adapter_stride=2,
        num_adapter_layers=3,
        output_hidden_size=None,
        **kwargs,
    ): 
        # 计算输入特征序列长度与输出序列长度的比值，采用reduce函数与operator.mul完成计算
        @property
        def inputs_to_logits_ratio(self):
            return functools.reduce(operator.mul, self.conv_stride, 1)
```