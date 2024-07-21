# `.\pytorch\benchmarks\operator_benchmark\pt\configs.py`

```py
# 导入名为 operator_benchmark 的模块，并重命名为 op_bench
import operator_benchmark as op_bench

# 以下为多个基准测试共享的配置

# 定义一个函数，用于移除配置列表中包含 CUDA 设备的配置项
def remove_cuda(config_list):
    # 定义 CUDA 设备的配置字典
    cuda_config = {"device": "cuda"}
    # 使用列表推导式，将不包含 CUDA 设备配置的项重新组成列表返回
    return [config for config in config_list if cuda_config not in config]

# Configs for conv-1d ops

# 定义一个短尺度的 conv-1d 操作配置列表
conv_1d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "L"],
    attrs=[
        [128, 256, 3, 1, 1, 64],  # 第一个配置：输入通道数 IC=128, 输出通道数 OC=256, 卷积核大小 kernel=3, 步幅 stride=1, 批量大小 N=1, 序列长度 L=64
        [256, 256, 3, 2, 4, 64],  # 第二个配置：IC=256, OC=256, kernel=3, stride=2, N=4, L=64
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    },
    tags=["short"],  # 标签为短尺度
)

# 定义一个长尺度的 conv-1d 操作配置
conv_1d_configs_long = op_bench.cross_product_configs(
    IC=[128, 512],  # 输入通道数 IC 可选 128 和 512
    OC=[128, 512],  # 输出通道数 OC 可选 128 和 512
    kernel=[3],  # 卷积核大小固定为 3
    stride=[1, 2],  # 步幅可选 1 或 2
    N=[8],  # 批量大小固定为 8
    L=[128],  # 序列长度固定为 128
    device=["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    tags=["long"],  # 标签为长尺度
)

# Configs for convtranspose-1d ops

# 定义一个短尺度的 convtranspose-1d 操作配置列表
convtranspose_1d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "L"],
    attrs=[
        [2016, 1026, 1024, 256, 1, 224],  # 第一个配置：IC=2016, OC=1026, kernel=1024, stride=256, N=1, L=224
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    },
    tags=["short"],  # 标签为短尺度
)

# Configs for Conv2d ops

# 定义一个短尺度的 Conv2d 操作配置列表
conv_2d_configs_short = op_bench.config_list(
    attr_names=[
        "IC", "OC", "kernel", "stride", "N", "H", "W", "G", "pad"
    ],
    attrs=[
        [256, 256, 3, 1, 1, 16, 16, 1, 0],  # 第一个配置：输入通道数 IC=256, 输出通道数 OC=256, 卷积核大小 kernel=3, 步幅 stride=1, 批量大小 N=1, 高度 H=16, 宽度 W=16, 组数 G=1, 填充 pad=0
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    },
    tags=["short"],  # 标签为短尺度
)

# 定义一个长尺度的 Conv2d 操作配置
conv_2d_configs_long = op_bench.cross_product_configs(
    IC=[128, 256],  # 输入通道数 IC 可选 128 和 256
    OC=[128, 256],  # 输出通道数 OC 可选 128 和 256
    kernel=[3],  # 卷积核大小固定为 3
    stride=[1, 2],  # 步幅可选 1 或 2
    N=[4],  # 批量大小固定为 4
    H=[32],  # 高度固定为 32
    W=[32],  # 宽度固定为 32
    G=[1],  # 组数固定为 1
    pad=[0],  # 填充固定为 0
    device=["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    tags=["long"],  # 标签为长尺度
)

# Configs for Conv2dPointwise ops

# 定义一个短尺度的 Conv2dPointwise 操作配置列表
conv_2d_pw_configs_short = op_bench.config_list(
    attr_names=[
        "IC", "OC", "stride", "N", "H", "W", "G", "pad"
    ],
    attrs=[
        [256, 256, 1, 1, 16, 16, 1, 0],  # 第一个配置：输入通道数 IC=256, 输出通道数 OC=256, 步幅 stride=1, 批量大小 N=1, 高度 H=16, 宽度 W=16, 组数 G=1, 填充 pad=0
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    },
    tags=["short"],  # 标签为短尺度
)

# 定义一个长尺度的 Conv2dPointwise 操作配置
conv_2d_pw_configs_long = op_bench.cross_product_configs(
    IC=[128, 256],  # 输入通道数 IC 可选 128 和 256
    OC=[128, 256],  # 输出通道数 OC 可选 128 和 256
    stride=[1, 2],  # 步幅可选 1 或 2
    N=[4],  # 批量大小固定为 4
    H=[32],  # 高度固定为 32
    W=[32],  # 宽度固定为 32
    G=[1],  # 组数固定为 1
    pad=[0],  # 填充固定为 0
    device=["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    tags=["long"],  # 标签为长尺度
)

# Configs for Conv3d ops

# 定义一个短尺度的 Conv3d 操作配置列表
conv_3d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "D", "H", "W"],
    attrs=[
        [64, 64, 3, 1, 8, 4, 16, 16],  # 第一个配置：输入通道数 IC=64, 输出通道数 OC=64, 卷积核大小 kernel=3, 步幅 stride=1, 批量大小 N=8, 深度 D=4, 高度 H=16, 宽度 W=16
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 设备类型跨产品配置：CPU 和 CUDA
    },
    tags=["short"],  # 标签为短尺度
)

# Configs for Linear ops

# 定义一个短尺度的 Linear 操作配置列表
linear_configs_short = op_bench.config_list(
    attr_names=["N", "IN", "OUT"],
    attrs=[
        [1, 1, 1],  # 第一个配置：批量大小 N=1, 输入维度 IN=1, 输出维度 OUT=1
        [4, 256, 128],  # 第二个配置：N=4, IN=256, OUT=128
        [16,
# 创建嵌入包（embeddingbag）的性能测试配置，对应多种参数组合
embeddingbag_short_configs = op_bench.cross_product_configs(
    # 嵌入包大小的不同取值
    embeddingbags=[10, 120, 1000, 2300],
    # 嵌入维度固定为64维
    dim=[64],
    # 汇总模式为"sum"
    mode=["sum"],
    # 输入数据的尺寸分别为8, 16, 64
    input_size=[8, 16, 64],
    # 偏移量固定为0
    offset=[0],
    # 是否稀疏（sparse）的配置，分别为True和False
    sparse=[True, False],
    # 是否包含最后一个偏移量（include_last_offset），分别为True和False
    include_last_offset=[True, False],
    # 设备选择为CPU
    device=["cpu"],
    # 标记为"short"，表示这是短小的配置示例
    tags=["short"],
)

# 创建嵌入（embedding）的性能测试配置，对应多种参数组合
embedding_short_configs = op_bench.cross_product_configs(
    # 嵌入数量（num_embeddings）的不同取值
    num_embeddings=[10, 120, 1000, 2300],
    # 嵌入维度固定为64维
    embedding_dim=[64],
    # 输入数据的尺寸分别为8, 16, 64
    input_size=[8, 16, 64],
    # 设备选择为CPU
    device=["cpu"],
    # 标记为"short"，表示这是短小的配置示例
    tags=["short"],
)
```