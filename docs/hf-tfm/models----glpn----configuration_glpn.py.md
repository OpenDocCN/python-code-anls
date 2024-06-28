# `.\models\glpn\configuration_glpn.py`

```
# coding=utf-8
# Copyright 2022 KAIST and The HuggingFace Inc. team. All rights reserved.
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

""" GLPN model configuration"""

# 导入预训练配置类和日志模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射字典，包含预训练模型名称和其配置文件的链接
GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "vinvino02/glpn-kitti": "https://huggingface.co/vinvino02/glpn-kitti/resolve/main/config.json",
    # See all GLPN models at https://huggingface.co/models?filter=glpn
}


class GLPNConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GLPNModel`]. It is used to instantiate an GLPN
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GLPN
    [vinvino02/glpn-kitti](https://huggingface.co/vinvino02/glpn-kitti) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义一个配置类GLPNConfig，用于初始化GLPN模型的参数
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量。
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            编码器块的数量（即Mix Transformer编码器中的阶段数）。
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            每个编码器块中的层的数量。
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            每个编码器块中的序列缩减比率。
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            每个编码器块的维度。
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            每个编码器块之前的补丁大小。
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个编码器块之前的步幅。
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Transformer编码器每个注意层中的注意头数量。
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            Mix FFN中隐藏层大小与输入层大小的比率。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持的字符串有："gelu", "relu", "selu"和"gelu_new"。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入、编码器和池化器中所有全连接层的丢弃概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            注意力概率的丢弃比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            随机深度的丢弃概率，用于Transformer编码器块。
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            层归一化层使用的epsilon值。
        decoder_hidden_size (`int`, *optional*, defaults to 64):
            解码器的维度。
        max_depth (`int`, *optional*, defaults to 10):
            解码器的最大深度。
        head_in_index (`int`, *optional*, defaults to -1):
            在头部使用的特征的索引。

    Example:

    ```python
    >>> from transformers import GLPNModel, GLPNConfig

    >>> # 初始化一个GLPN vinvino02/glpn-kitti风格的配置
    >>> configuration = GLPNConfig()

    >>> # 使用vinvino02/glpn-kitti风格的配置初始化一个模型
    >>> model = GLPNModel(configuration)
    ```
    # 访问模型配置
    configuration = model.config
```