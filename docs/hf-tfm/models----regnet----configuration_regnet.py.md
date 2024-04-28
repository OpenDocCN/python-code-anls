# `.\transformers\models\regnet\configuration_regnet.py`

```
# coding=utf-8
# 版权声明
# 根据Apache 2.0许可证使用代码
# 配置类用于存储`RegNetModel`的配置，根据指定的参数实例化RegNet模型，定义模型架构
# 继承自`PretrainedConfig`，用于控制模型输出
#数通道数num_channels，可选， 默认为3
# 嵌入层的维度embedding_size，默认为64
#每个阶段的隐藏层维度hidden_sizes,默认为`[256, 512, 1024, 2048]`
#每个阶段的深度depths，默认为`[3, 4, 6, 3]`
# 层的类型layer_type，默认为`"y"`
# 每个块的非线性激活函数hidden_act，默认为`"relu"`
# 是否在第一个阶段进行下采样downsample_in_first_stage，默认为`False`
#示例
#从transformers模块中导入RegNetConfig和RegNetModel
#初始化一个RegNet regnet-y-40风格的配置
    >>> configuration = RegNetConfig()
    # 创建一个 RegNetConfig 实例，用于配置模型
    >>> model = RegNetModel(configuration)
    # 基于 regnet-y-40 风格的配置初始化一个模型
    >>> configuration = model.config
    # 获取模型的配置信息
    """

# 定义一个 RegNetConfig 类
class RegNetConfig:
    # 初始化一些参数
    model_type = "regnet"
    layer_types = ["x", "y"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=32,
        hidden_sizes=[128, 192, 512, 1088],
        depths=[2, 6, 12, 2],
        groups_width=64,
        layer_type="y",
        hidden_act="relu",
        **kwargs,
    ):
        # 继承父类的初始化方法
        super().__init__(**kwargs)
        # 检查传入的 layer_type 是否为合法值
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        # 设置参数值
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.groups_width = groups_width
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        # 总是在第一个阶段进行下采样
        self.downsample_in_first_stage = True
```