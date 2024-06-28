# `.\models\mamba\configuration_mamba.py`

```
# coding=utf-8
# 版权所有 2024 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本使用此文件；除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件分发在“按现状”基础上，
# 没有任何明示或暗示的保证或条件。请查看许可证获取特定语言的权限及限制。
"""MAMBA configuration"""

import math  # 导入 math 模块

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "state-spaces/mamba-2.8b": "https://huggingface.co/state-spaces/mamba-2.8b/resolve/main/config.json",
}

class MambaConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MambaModel`]. It is used to instantiate a MAMBA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MAMBA
    [state-spaces/mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) architecture.
    
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    
    Example:
    
    ```python
    >>> from transformers import MambaConfig, MambaModel
    
    >>> # Initializing a Mamba configuration
    >>> configuration = MambaConfig()
    
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MambaModel(configuration)
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mamba"

    def __init__(
        self,
        vocab_size=50280,
        hidden_size=768,
        state_size=16,
        num_hidden_layers=32,
        layer_norm_epsilon=1e-5,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=True,
        time_step_rank="auto",
        time_step_scale=1.0,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_init_scheme="random",
        time_step_floor=1e-4,
        rescale_prenorm_residual=False,
        use_cache=True,
        **kwargs,
    ):
        """
        初始化 MambaConfig 类，设置 MAMBA 模型的配置参数。
        
        参数：
            vocab_size (int): 词汇表大小，默认为 50280
            hidden_size (int): 隐藏层大小，默认为 768
            state_size (int): 状态大小，默认为 16
            num_hidden_layers (int): 隐藏层层数，默认为 32
            layer_norm_epsilon (float): 层归一化的 epsilon 值，默认为 1e-5
            pad_token_id (int): 填充标记的 ID，默认为 0
            bos_token_id (int): 起始标记的 ID，默认为 0
            eos_token_id (int): 结束标记的 ID，默认为 0
            expand (int): 扩展参数，默认为 2
            conv_kernel (int): 卷积核大小，默认为 4
            use_bias (bool): 是否使用偏置，默认为 False
            use_conv_bias (bool): 是否使用卷积偏置，默认为 True
            hidden_act (str): 隐藏层激活函数，默认为 "silu"
            initializer_range (float): 初始化范围，默认为 0.1
            residual_in_fp32 (bool): 是否在 fp32 下进行残差连接，默认为 True
            time_step_rank (str): 时间步长等级，默认为 "auto"
            time_step_scale (float): 时间步长缩放，默认为 1.0
            time_step_min (float): 最小时间步长，默认为 0.001
            time_step_max (float): 最大时间步长，默认为 0.1
            time_step_init_scheme (str): 时间步长初始化方案，默认为 "random"
            time_step_floor (float): 时间步长下限，默认为 1e-4
            rescale_prenorm_residual (bool): 是否对预归一化残差进行重新缩放，默认为 False
            use_cache (bool): 是否使用缓存，默认为 True
            **kwargs: 其他关键字参数
        """
        super().__init__(**kwargs)  # 调用父类 PretrainedConfig 的初始化方法
        # 初始化模型的各种参数
        self.vocab_size = vocab_size                    # 设置词汇表大小
        self.hidden_size = hidden_size                  # 设置隐藏层大小
        self.state_size = state_size                    # 设置状态大小
        self.num_hidden_layers = num_hidden_layers      # 设置隐藏层的数量
        self.layer_norm_epsilon = layer_norm_epsilon    # 设置层归一化的 epsilon 值
        self.conv_kernel = conv_kernel                  # 设置卷积核大小
        self.expand = expand                            # 设置扩展因子
        self.intermediate_size = int(expand * self.hidden_size)  # 计算中间层大小
        self.bos_token_id = bos_token_id                # 设置起始标记 ID
        self.eos_token_id = eos_token_id                # 设置结束标记 ID
        self.pad_token_id = pad_token_id                # 设置填充标记 ID
        self.use_bias = use_bias                        # 设置是否使用偏置
        self.use_conv_bias = use_conv_bias              # 设置卷积层是否使用偏置
        self.hidden_act = hidden_act                    # 设置隐藏层激活函数类型
        self.initializer_range = initializer_range      # 设置初始化范围
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank  # 设置时间步骤的秩
        self.time_step_scale = time_step_scale          # 设置时间步骤的比例
        self.time_step_min = time_step_min              # 设置时间步骤的最小值
        self.time_step_max = time_step_max              # 设置时间步骤的最大值
        self.time_step_init_scheme = time_step_init_scheme  # 设置时间步骤的初始化方案
        self.time_step_floor = time_step_floor          # 设置时间步骤的下限
        self.rescale_prenorm_residual = rescale_prenorm_residual  # 设置前归一化残差的重新缩放
        self.residual_in_fp32 = residual_in_fp32        # 设置是否在 FP32 下使用残差连接
        self.use_cache = use_cache                      # 设置是否使用缓存

        # 调用父类的初始化方法，传递起始、结束和填充标记 ID 以及其它参数
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)
```