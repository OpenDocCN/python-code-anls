# `.\transformers\models\levit\configuration_levit.py`

```
# 导入必要的库和模块
# 从collections模块中导入OrderedDict类，用于创建有序的字典
# 从typing模块中导入Mapping类，用于指定字典的类型注解
# 从packaging模块中导入version类，用于处理版本号
# 从configuration_utils模块中导入PretrainedConfig类，用于创建预训练模型的配置
# 从onnx模块中导入OnnxConfig类，用于配置ONNX模型的信息
# 从utils模块中导入logging函数，用于记录日志信息
# 设置模块logger记录信息的对象
# 定义键为模型名称，值为模型配置信息JSON文件下载地址的字典
# 定义LevitConfig类，继承自PretrainedConfig类
# 这个配置类用于存储LeViT模型的配置信息，并可用于实例化LeViT模型
# r"""""" 表示多行字符串的开始，用来写长注解
# LeViT模型的配置对象继承自PretrainedConfig类，并可用于控制模型的输出
    # 定义模型类型为"levit"
    model_type = "levit"

    # 初始化LeViT模型类
    def __init__(
        self,
        # 输入图片的尺寸，默认为224
        image_size=224,
        # 输入图片的通道数，默认为3
        num_channels=3,
        # 初始卷积层的卷积核大小，默认为3
        kernel_size=3,
        # 初始卷积层的步长，默认为2
        stride=2,
        # 初始卷积层的填充大小，默认为1
        padding=1,
        # 嵌入的图块大小，默认为16
        patch_size=16,
        # 编码器块的每层隐藏单元数，默认为[128, 256, 384]
        hidden_sizes=[128, 256, 384],
        # 每个变压器编码器块中每个注意力层的注意力头数，默认为[4, 8, 12]
        num_attention_heads=[4, 8, 12],
        # 每个编码器块中的层数，默认为[4, 4, 4]
        depths=[4, 4, 4],
        # 每个编码器块中每个密钥的大小，默认为[16, 16, 16]
        key_dim=[16, 16, 16],
        # 随机深度的丢失概率，用于变压器编码器块，默认为0
        drop_path_rate=0,
        # 混合前馈网络中隐藏层大小与输入层大小的比例，默认为[2, 2, 2]
        mlp_ratio=[2, 2, 2],
        # 注意力层中输出维度与输入维度的比例，默认为[2, 2, 2]
        attention_ratio=[2, 2, 2],
        # 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为0.02
        initializer_range=0.02,
        **kwargs,
        ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化图像大小
        self.image_size = image_size
        # 初始化通道数
        self.num_channels = num_channels
        # 初始化卷积核大小
        self.kernel_size = kernel_size
        # 初始化步长
        self.stride = stride
        # 初始化填充
        self.padding = padding
        # 初始化隐藏层大小
        self.hidden_sizes = hidden_sizes
        # 初始化注意力头数目
        self.num_attention_heads = num_attention_heads
        # 初始化层数
        self.depths = depths
        # 初始化关键字维度
        self.key_dim = key_dim
        # 初始化丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 初始化补丁大小
        self.patch_size = patch_size
        # 初始化注意力比例
        self.attention_ratio = attention_ratio
        # 初始化多层感知机比例
        self.mlp_ratio = mlp_ratio
        # 初始化初始化范围
        self.initializer_range = initializer_range
        # 初始化下采样操作列表
        self.down_ops = [
            # 第一个下采样操作：名称、输入维度、输出维度、卷积核大小、步长、填充
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            # 第二个下采样操作：名称、输入维度、输出维度、卷积核大小、步长、填充
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]
# 从transformers.models.vit.configuration_vit.ViTOnnxConfig中复制LevitOnnxConfig类
class LevitOnnxConfig(OnnxConfig):
    # 设置torch_onnx_minimum_version属性为1.11的版本
    torch_onnx_minimum_version = version.parse("1.11")

    # 定义inputs属性，返回一个OrderedDict对象，用于描述模型输入的维度顺序
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                # 定义模型输入的维度顺序，分别对应像素值、批大小、通道数、高度和宽度
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    # 定义atol_for_validation属性，返回用于验证的绝对容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4
```