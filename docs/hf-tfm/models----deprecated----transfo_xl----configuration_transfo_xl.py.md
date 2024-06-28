# `.\models\deprecated\transfo_xl\configuration_transfo_xl.py`

```py
# coding=utf-8
# 定义版权信息和许可证，此处使用Apache License 2.0
# 这个文件包含了Transformer XL的配置信息

# 从transformers模块中导入PretrainedConfig类
from ....configuration_utils import PretrainedConfig
# 从utils模块中导入logging函数
from ....utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个字典，映射预训练模型名称到其配置文件的URL
TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "transfo-xl/transfo-xl-wt103": "https://huggingface.co/transfo-xl/transfo-xl-wt103/resolve/main/config.json",
}

# TransfoXLConfig类，继承自PretrainedConfig
class TransfoXLConfig(PretrainedConfig):
    """
    这是一个配置类，用于存储[`TransfoXLModel`]或[`TFTransfoXLModel`]的配置。它被用来根据指定的参数实例化
    Transformer-XL模型，定义模型架构。使用默认参数实例化配置将得到类似于TransfoXL
    [transfo-xl/transfo-xl-wt103](https://huggingface.co/transfo-xl/transfo-xl-wt103)架构的配置。

    配置对象继承自[`PretrainedConfig`]，可以用来控制模型的输出。阅读[`PretrainedConfig`]的文档获取更多信息。

    示例:

    ```
    >>> from transformers import TransfoXLConfig, TransfoXLModel

    >>> # 初始化一个Transformer XL配置
    >>> configuration = TransfoXLConfig()

    >>> # 使用配置初始化一个模型（随机权重）
    >>> model = TransfoXLModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型
    model_type = "transfo-xl"
    # 推理阶段要忽略的键
    keys_to_ignore_at_inference = ["mems"]
    # 属性映射，将旧属性名映射到新属性名
    attribute_map = {
        "n_token": "vocab_size",
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 初始化函数，用于初始化Transformer-XL模型的各种参数和属性
    def __init__(
        self,
        vocab_size=267735,  # 词汇表大小，默认为267735
        cutoffs=[20000, 40000, 200000],  # 截断点列表，默认为[20000, 40000, 200000]
        d_model=1024,  # 模型的维度，默认为1024
        d_embed=1024,  # 嵌入层的维度，默认为1024
        n_head=16,  # 注意力头的数量，默认为16
        d_head=64,  # 每个注意力头的维度，默认为64
        d_inner=4096,  # Transformer中全连接层的维度，默认为4096
        div_val=4,  # 嵌入分割因子，默认为4
        pre_lnorm=False,  # 是否在层归一化前应用Dropout，默认为False
        n_layer=18,  # Transformer的层数，默认为18
        mem_len=1600,  # Transformer-XL中的记忆长度，默认为1600
        clamp_len=1000,  # 用于限制自注意力长度的上限，默认为1000
        same_length=True,  # 自注意力中每个位置是否具有相同的长度，默认为True
        proj_share_all_but_first=True,  # 是否共享所有投影层的参数除了第一个，默认为True
        attn_type=0,  # 注意力类型的标识，默认为0
        sample_softmax=-1,  # 是否对softmax进行采样的标识，默认为-1（不采样）
        adaptive=True,  # 是否使用自适应计算，如自适应softmax，默认为True
        dropout=0.1,  # 全局Dropout的比例，默认为0.1
        dropatt=0.0,  # Attention分数Dropout的比例，默认为0.0
        untie_r=True,  # 是否对R矩阵进行解绑，默认为True
        init="normal",  # 初始化参数的方法，默认为"normal"
        init_range=0.01,  # 参数初始化的范围，默认为0.01
        proj_init_std=0.01,  # 投影层初始化的标准差，默认为0.01
        init_std=0.02,  # 参数初始化的标准差，默认为0.02
        layer_norm_epsilon=1e-5,  # 层归一化中的epsilon，默认为1e-5
        eos_token_id=0,  # EOS（结束符）的token ID，默认为0
        **kwargs,  # 其他关键字参数，用于传递给父类初始化函数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.cutoffs = []  # 初始化截断点列表
        self.cutoffs.extend(cutoffs)  # 将输入的截断点列表复制到self.cutoffs中
        if proj_share_all_but_first:
            self.tie_projs = [False] + [True] * len(self.cutoffs)  # 设置投影层是否共享参数的列表
        else:
            self.tie_projs = [False] + [False] * len(self.cutoffs)  # 设置投影层是否共享参数的列表
        self.d_model = d_model  # 设置模型的维度
        self.d_embed = d_embed  # 设置嵌入层的维度
        self.d_head = d_head  # 设置注意力头的维度
        self.d_inner = d_inner  # 设置全连接层的维度
        self.div_val = div_val  # 设置嵌入分割因子
        self.pre_lnorm = pre_lnorm  # 设置是否在层归一化前应用Dropout
        self.n_layer = n_layer  # 设置Transformer的层数
        self.n_head = n_head  # 设置注意力头的数量
        self.mem_len = mem_len  # 设置记忆长度
        self.same_length = same_length  # 设置自注意力中每个位置是否具有相同的长度
        self.attn_type = attn_type  # 设置注意力类型
        self.clamp_len = clamp_len  # 设置限制自注意力长度的上限
        self.sample_softmax = sample_softmax  # 设置是否对softmax进行采样
        self.adaptive = adaptive  # 设置是否使用自适应计算
        self.dropout = dropout  # 设置全局Dropout的比例
        self.dropatt = dropatt  # 设置Attention分数Dropout的比例
        self.untie_r = untie_r  # 设置是否对R矩阵进行解绑
        self.init = init  # 设置初始化参数的方法
        self.init_range = init_range  # 设置参数初始化的范围
        self.proj_init_std = proj_init_std  # 设置投影层初始化的标准差
        self.init_std = init_std  # 设置参数初始化的标准差
        self.layer_norm_epsilon = layer_norm_epsilon  # 设置层归一化中的epsilon
        super().__init__(eos_token_id=eos_token_id, **kwargs)  # 调用父类的初始化函数，传递EOS token ID和其他关键字参数

    @property
    def max_position_embeddings(self):
        # 获取最大位置嵌入的属性值
        # 根据Transformer-XL文档的描述，该模型没有序列长度限制
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        return -1  # 返回-1，表示没有长度限制

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        # 设置最大位置嵌入的属性值
        # 根据Transformer-XL文档的描述，该模型没有序列长度限制，因此设置操作未实现
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )
```