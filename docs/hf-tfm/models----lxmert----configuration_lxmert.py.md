# `.\transformers\models\lxmert\configuration_lxmert.py`

```
# 设置编码格式为 UTF-8
# 版权声明和信息
# 本代码使用 Apache 许可证版本 2.0 进行许可
# 除非符合适用法律要求或书面同意，否则软件根据“按原样”基础分发
# 无论是明示的还是默示的，都没有任何保证或条件
# 有关许可的更多信息，请参见 http://www.apache.org/licenses/LICENSE-2.0
""" LXMERT 模型配置"""

# 导入所需的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件的映射字典
LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/config.json",
}

# LXMERT 模型配置类，用于存储 LXMERT 模型的配置信息
class LxmertConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`LxmertModel`] 或 [`TFLxmertModel`] 的配置信息。
    根据指定的参数实例化 LXMERT 模型，定义模型架构。使用默认参数实例化配置将产生与 Lxmert
    [unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读
    [`PretrainedConfig`] 的文档以获取更多信息。

    """

    # 模型类型
    model_type = "lxmert"
    # 属性映射字典
    attribute_map = {}

    # 初始化方法，用于设置配置的各种参数
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        num_qa_labels=9500,
        num_object_labels=1600,
        num_attr_labels=400,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        l_layers=9,
        x_layers=5,
        r_layers=5,
        visual_feat_dim=2048,
        visual_pos_dim=4,
        visual_loss_normalizer=6.67,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        visual_obj_loss=True,
        visual_attr_loss=True,
        visual_feat_loss=True,
        **kwargs,
        ):
        # 初始化模型参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_qa_labels = num_qa_labels
        self.num_object_labels = num_object_labels
        self.num_attr_labels = num_attr_labels
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.visual_loss_normalizer = visual_loss_normalizer
        self.task_matched = task_matched
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_qa = task_qa
        self.visual_obj_loss = visual_obj_loss
        self.visual_attr_loss = visual_attr_loss
        self.visual_feat_loss = visual_feat_loss
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}
        # 调用父类初始化函数
        super().__init__(**kwargs)
```