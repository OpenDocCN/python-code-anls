# `.\diffusers\models\resnet_flax.py`

```py
# 版权所有 2024 The HuggingFace Team。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按“原样”基础分发的，不提供任何明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 许可证限制的具体语言。
import flax.linen as nn  # 导入 flax.linen 模块以构建神经网络
import jax  # 导入 jax，用于高效数值计算
import jax.numpy as jnp  # 导入 jax 的 numpy 作为 jnp 以进行数组操作


class FlaxUpsample2D(nn.Module):  # 定义一个用于上采样的 2D 模块
    out_channels: int  # 输出通道数的类型注解
    dtype: jnp.dtype = jnp.float32  # 数据类型默认为 float32

    def setup(self):  # 设置模块参数
        self.conv = nn.Conv(  # 创建卷积层
            self.out_channels,  # 设置输出通道数
            kernel_size=(3, 3),  # 卷积核大小为 3x3
            strides=(1, 1),  # 步幅为 1
            padding=((1, 1), (1, 1)),  # 在每个边缘填充 1 像素
            dtype=self.dtype,  # 设置数据类型
        )

    def __call__(self, hidden_states):  # 定义模块的前向传播
        batch, height, width, channels = hidden_states.shape  # 获取输入张量的维度
        hidden_states = jax.image.resize(  # 使用 nearest 方法调整图像大小
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),  # 输出形状为高和宽各翻倍
            method="nearest",  # 使用最近邻插值法
        )
        hidden_states = self.conv(hidden_states)  # 对调整后的张量应用卷积层
        return hidden_states  # 返回卷积后的结果


class FlaxDownsample2D(nn.Module):  # 定义一个用于下采样的 2D 模块
    out_channels: int  # 输出通道数的类型注解
    dtype: jnp.dtype = jnp.float32  # 数据类型默认为 float32

    def setup(self):  # 设置模块参数
        self.conv = nn.Conv(  # 创建卷积层
            self.out_channels,  # 设置输出通道数
            kernel_size=(3, 3),  # 卷积核大小为 3x3
            strides=(2, 2),  # 步幅为 2
            padding=((1, 1), (1, 1)),  # 在每个边缘填充 1 像素
            dtype=self.dtype,  # 设置数据类型
        )

    def __call__(self, hidden_states):  # 定义模块的前向传播
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # 为高和宽维度填充
        # hidden_states = jnp.pad(hidden_states, pad_width=pad)  # 使用填充来调整张量大小
        hidden_states = self.conv(hidden_states)  # 对输入张量应用卷积层
        return hidden_states  # 返回卷积后的结果


class FlaxResnetBlock2D(nn.Module):  # 定义一个用于 2D ResNet 块的模块
    in_channels: int  # 输入通道数的类型注解
    out_channels: int = None  # 输出通道数，默认为 None
    dropout_prob: float = 0.0  # dropout 概率，默认为 0
    use_nin_shortcut: bool = None  # 是否使用 NIN 短路，默认为 None
    dtype: jnp.dtype = jnp.float32  # 数据类型默认为 float32
    # 设置模型的各个层和参数
        def setup(self):
            # 确定输出通道数，若未指定，则使用输入通道数
            out_channels = self.in_channels if self.out_channels is None else self.out_channels
    
            # 初始化第一个归一化层，使用32个组和小于1e-5的epsilon
            self.norm1 = nn.GroupNorm(num_groups=32, epsilon=1e-5)
            # 初始化第一个卷积层，设置卷积参数
            self.conv1 = nn.Conv(
                out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )
    
            # 初始化时间嵌入投影层，输出通道数与dtype
            self.time_emb_proj = nn.Dense(out_channels, dtype=self.dtype)
    
            # 初始化第二个归一化层
            self.norm2 = nn.GroupNorm(num_groups=32, epsilon=1e-5)
            # 初始化丢弃层，设置丢弃概率
            self.dropout = nn.Dropout(self.dropout_prob)
            # 初始化第二个卷积层，设置卷积参数
            self.conv2 = nn.Conv(
                out_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=((1, 1), (1, 1)),
                dtype=self.dtype,
            )
    
            # 确定是否使用1x1卷积快捷连接，依据输入和输出通道数
            use_nin_shortcut = self.in_channels != out_channels if self.use_nin_shortcut is None else self.use_nin_shortcut
    
            # 初始化快捷连接卷积层为None
            self.conv_shortcut = None
            # 如果需要，初始化1x1卷积快捷连接层
            if use_nin_shortcut:
                self.conv_shortcut = nn.Conv(
                    out_channels,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="VALID",
                    dtype=self.dtype,
                )
    
        # 定义前向传播方法
        def __call__(self, hidden_states, temb, deterministic=True):
            # 保存输入作为残差
            residual = hidden_states
            # 对输入进行归一化处理
            hidden_states = self.norm1(hidden_states)
            # 应用Swish激活函数
            hidden_states = nn.swish(hidden_states)
            # 通过第一个卷积层处理
            hidden_states = self.conv1(hidden_states)
    
            # 对时间嵌入进行Swish激活处理
            temb = self.time_emb_proj(nn.swish(temb))
            # 扩展时间嵌入的维度
            temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
            # 将时间嵌入加到隐藏状态中
            hidden_states = hidden_states + temb
    
            # 对隐藏状态进行第二次归一化
            hidden_states = self.norm2(hidden_states)
            # 应用Swish激活函数
            hidden_states = nn.swish(hidden_states)
            # 应用丢弃层
            hidden_states = self.dropout(hidden_states, deterministic)
            # 通过第二个卷积层处理
            hidden_states = self.conv2(hidden_states)
    
            # 如果存在快捷连接卷积层，则对残差进行处理
            if self.conv_shortcut is not None:
                residual = self.conv_shortcut(residual)
    
            # 返回隐藏状态和残差的和
            return hidden_states + residual
```