# `.\models\deprecated\trajectory_transformer\configuration_trajectory_transformer.py`

```py
# coding=utf-8
# 声明代码文件的编码格式和版权声明

# 导入所需的包
# 从....包的configuration_utils模块中导入PretrainedConfig类
# 从....包的utils模块中导入logging模块
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取logger实例
logger = logging.get_logger(__name__)

# 预训练配置和模型映射字典
TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2": (
        "https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2/resolve/main/config.json"
    ),
    # 查看所有TrajectoryTransformer模型 https://huggingface.co/models?filter=trajectory_transformer
}

# TrajectoryTransformer配置类，继承自PretrainedConfig
class TrajectoryTransformerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`TrajectoryTransformerModel`] 配置的配置类。它用于根据指定的参数实例化 TrajectoryTransformer 模型，
    定义模型架构。使用默认值实例化配置将生成类似于 TrajectoryTransformer
    [CarlCochet/trajectory-transformer-halfcheetah-medium-v2](https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`] 并可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    ```python
    >>> from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

    >>> # 初始化一个 TrajectoryTransformer 类似 CarlCochet/trajectory-transformer-halfcheetah-medium-v2 的配置
    >>> configuration = TrajectoryTransformerConfig()

    >>> # 使用 CarlCochet/trajectory-transformer-halfcheetah-medium-v2 风格的配置初始化一个具有随机权重的模型
    >>> model = TrajectoryTransformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    
    # 模型类型
    model_type = "trajectory_transformer"
    # 推理过程中要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 初始化模型参数
    def __init__(
        self,
        vocab_size=100,  # 词汇表大小
        action_weight=5,  # 行为损失权重
        reward_weight=1,  # 奖励损失权重
        value_weight=1,  # 价值损失权重
        block_size=249,  # 输入序列最大长度
        action_dim=6,  # 行为维度
        observation_dim=17,  # 观察维度
        transition_dim=25,  # 转移维度
        n_layer=4,  # Transformer编码器层数
        n_head=4,  # Transformer注意力头数
        n_embd=128,  # Transformer隐藏层维度
        embd_pdrop=0.1,  # Transformer嵌入层dropout率
        attn_pdrop=0.1,  # Transformer注意力层dropout率
        resid_pdrop=0.1,  # Transformer残差连接层dropout率
        learning_rate=0.0006,  # 学习率
        max_position_embeddings=512,  # 最大位置嵌入
        initializer_range=0.02,  # 权重初始化范围
        layer_norm_eps=1e-12,  # 层归一化 epsilon
        kaiming_initializer_range=1,  # Kaiming初始化范围
        use_cache=True,  # 是否使用缓存
        pad_token_id=1,  # 填充标记ID
        bos_token_id=50256,  # 起始标记ID
        eos_token_id=50256,  # 结束标记ID
        **kwargs,
    ):
        # 保存模型参数
        self.vocab_size = vocab_size
        self.action_weight = action_weight
        self.reward_weight = reward_weight
        self.value_weight = value_weight
        self.max_position_embeddings = max_position_embeddings
        self.block_size = block_size
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.transition_dim = transition_dim
        self.learning_rate = learning_rate
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.kaiming_initializer_range = kaiming_initializer_range
        self.use_cache = use_cache
        # 调用父类初始化方法
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
```