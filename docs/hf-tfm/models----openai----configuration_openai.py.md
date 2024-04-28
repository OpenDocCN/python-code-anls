# `.\transformers\models\openai\configuration_openai.py`

```
# 导入 OpenAI GPT 配置相关的模块
import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义了一个 OpenAI GPT 预训练配置的映射字典
OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {"openai-gpt": "https://huggingface.co/openai-gpt/resolve/main/config.json"}

# 定义了 OpenAIGPTConfig 类，它继承自 PretrainedConfig
# 这个类用于存储和管理 OpenAI GPT 模型的配置
class OpenAIGPTConfig(PretrainedConfig):
    """
    用于存储和管理 OpenAI GPT 模型的配置
    可用于初始化 OpenAIGPTModel 或 TFOpenAIGPTModel
    """

    # 模型类型
    model_type = "openai-gpt"

    # 一个属性映射字典，用于将配置参数对应到模型参数
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    # 初始化函数，接收多个参数用于配置模型
    def __init__(
        self,
        vocab_size=40478,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        afn="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        **kwargs,
    ):
        # 在这里可以对配置参数进行初始化和设置
        pass
    # 初始化 Transformer 类的属性：词汇表大小
    self.vocab_size = vocab_size
    # 初始化 Transformer 类的属性：位置编码数
    self.n_positions = n_positions
    # 初始化 Transformer 类的属性：嵌入维度数
    self.n_embd = n_embd
    # 初始化 Transformer 类的属性：层数
    self.n_layer = n_layer
    # 初始化 Transformer 类的属性：头数
    self.n_head = n_head
    # 初始化 Transformer 类的属性：激活函数
    self.afn = afn
    # 初始化 Transformer 类的属性：残差 Dropout 概率
    self.resid_pdrop = resid_pdrop
    # 初始化 Transformer 类的属性：嵌入 Dropout 概率
    self.embd_pdrop = embd_pdrop
    # 初始化 Transformer 类的属性：注意力 Dropout 概率
    self.attn_pdrop = attn_pdrop
    # 初始化 Transformer 类的属性：层归一化 epsilon 值
    self.layer_norm_epsilon = layer_norm_epsilon
    # 初始化 Transformer 类的属性：初始化范围
    self.initializer_range = initializer_range
    # 初始化 Transformer 类的属性：摘要类型
    self.summary_type = summary_type
    # 初始化 Transformer 类的属性：摘要是否用投影
    self.summary_use_proj = summary_use_proj
    # 初始化 Transformer 类的属性：摘要激活函数
    self.summary_activation = summary_activation
    # 初始化 Transformer 类的属性：摘要首次 Dropout 概率
    self.summary_first_dropout = summary_first_dropout
    # 初始化 Transformer 类的属性：投影到标签的摘要
    self.summary_proj_to_labels = summary_proj_to_labels
    # 调用父类的构造方法
    super().__init__(**kwargs)
```