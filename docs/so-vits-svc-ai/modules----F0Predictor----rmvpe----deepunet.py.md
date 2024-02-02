# `so-vits-svc\modules\F0Predictor\rmvpe\deepunet.py`

```py
# 导入 torch 库
import torch
# 导入 torch.nn 模块
import torch.nn as nn

# 从当前目录下的 constants 模块中导入 N_MELS 常量
from .constants import N_MELS

# 定义 ConvBlockRes 类，继承自 nn.Module
class ConvBlockRes(nn.Module):
    # 初始化函数，接受输入通道数、输出通道数和动量参数
    def __init__(self, in_channels, out_channels, momentum=0.01):
        # 调用父类的初始化函数
        super(ConvBlockRes, self).__init__()
        # 定义卷积块
        self.conv = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            # 批归一化层
            nn.BatchNorm2d(out_channels, momentum=momentum),
            # ReLU 激活函数
            nn.ReLU(),

            # 第二个卷积层
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1),
                      bias=False),
            # 批归一化层
            nn.BatchNorm2d(out_channels, momentum=momentum),
            # ReLU 激活函数
            nn.ReLU(),
        )
        # 如果输入通道数不等于输出通道数
        if in_channels != out_channels:
            # 添加一个 1x1 的卷积层作为快捷连接
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            # 设置快捷连接标志为 True
            self.is_shortcut = True
        else:
            # 设置快捷连接标志为 False
            self.is_shortcut = False

    # 前向传播函数
    def forward(self, x):
        # 如果存在快捷连接
        if self.is_shortcut:
            # 返回卷积块的输出加上快捷连接的输出
            return self.conv(x) + self.shortcut(x)
        else:
            # 返回卷积块的输出加上输入的输出
            return self.conv(x) + x


# 定义 ResEncoderBlock 类，继承自 nn.Module
class ResEncoderBlock(nn.Module):
    # 初始化函数，接受输入通道数、输出通道数、卷积核大小、块数和动量参数
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        # 调用父类的初始化函数
        super(ResEncoderBlock, self).__init__()
        # 设置块数
        self.n_blocks = n_blocks
        # 定义卷积层列表
        self.conv = nn.ModuleList()
        # 添加第一个 ConvBlockRes 实例
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        # 根据块数添加对应数量的 ConvBlockRes 实例
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 如果卷积核大小不为 None
        if self.kernel_size is not None:
            # 添加平均池化层
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 循环执行 n_blocks 次卷积操作
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        # 如果 kernel_size 不为空
        if self.kernel_size is not None:
            # 返回卷积后的结果 x 和池化后的结果
            return x, self.pool(x)
        else:
            # 如果 kernel_size 为空，只返回卷积后的结果 x
            return x
# 定义一个 ResDecoderBlock 类，继承自 nn.Module
class ResDecoderBlock(nn.Module):
    # 初始化方法，接受输入通道数、输出通道数、步幅、块数和动量参数
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        # 调用父类的初始化方法
        super(ResDecoderBlock, self).__init__()
        # 根据步幅确定输出填充
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        # 保存块数
        self.n_blocks = n_blocks
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            # 转置卷积层
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=stride,
                               padding=(1, 1),
                               output_padding=out_padding,
                               bias=False),
            # 批归一化层
            nn.BatchNorm2d(out_channels, momentum=momentum),
            # ReLU 激活函数
            nn.ReLU(),
        )
        # 第二个卷积层列表
        self.conv2 = nn.ModuleList()
        # 添加第一个 ConvBlockRes 实例
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        # 循环添加 n_blocks-1 个 ConvBlockRes 实例
        for i in range(n_blocks-1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    # 前向传播方法，接受输入张量 x 和连接张量 concat_tensor
    def forward(self, x, concat_tensor):
        # 通过第一个卷积层处理输入张量 x
        x = self.conv1(x)
        # 在通道维度上连接处理后的张量和连接张量
        x = torch.cat((x, concat_tensor), dim=1)
        # 循环处理 n_blocks 次
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        # 返回处理后的张量
        return x


# 定义一个 Encoder 类，继承自 nn.Module
class Encoder(nn.Module):
    # 初始化方法，接受输入通道数、输入尺寸、编码器数量、卷积核尺寸、块数、输出通道数和动量参数
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        # 调用父类的初始化方法
        super(Encoder, self).__init__()
        # 保存编码器数量
        self.n_encoders = n_encoders
        # 批归一化层
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        # 卷积层列表
        self.layers = nn.ModuleList()
        # 潜在通道数列表
        self.latent_channels = []
        # 循环添加编码器
        for i in range(self.n_encoders):
            # 添加 ResEncoderBlock 实例
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            # 添加潜在通道数信息
            self.latent_channels.append([out_channels, in_size])
            # 更新输入通道数、输出通道数和输入尺寸
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        # 保存输出尺寸和输出通道数
        self.out_size = in_size
        self.out_channel = out_channels
    # 定义一个前向传播函数，接受输入参数 x
    def forward(self, x):
        # 初始化一个空列表，用于存储拼接后的张量
        concat_tensors = []
        # 对输入参数 x 进行批量归一化处理
        x = self.bn(x)
        # 遍历编码器的数量次数
        for i in range(self.n_encoders):
            # 调用编码器层的前向传播函数，得到输出和中间结果
            _, x = self.layers[i](x)
            # 将中间结果添加到拼接张量列表中
            concat_tensors.append(_)
        # 返回编码器的输出和拼接后的张量列表
        return x, concat_tensors
# 定义一个名为Intermediate的类，继承自nn.Module
class Intermediate(nn.Module):
    # 初始化方法，接受输入通道数、输出通道数、中间层数量、块数量和动量参数
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        # 设置中间层数量
        self.n_inters = n_inters
        # 创建一个空的模块列表
        self.layers = nn.ModuleList()
        # 向模块列表中添加一个ResEncoderBlock模块
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        # 循环添加n_inters-1个ResEncoderBlock模块到模块列表中
        for i in range(self.n_inters-1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    # 前向传播方法，接受输入张量x
    def forward(self, x):
        # 循环遍历中间层的模块列表，对输入张量进行前向传播
        for i in range(self.n_inters):
            x = self.layers[i](x)
        # 返回处理后的张量
        return x


# 定义一个名为Decoder的类，继承自nn.Module
class Decoder(nn.Module):
    # 初始化方法，接受输入通道数、解码器数量、步长、块数量和动量参数
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        # 创建一个空的模块列表
        self.layers = nn.ModuleList()
        # 设置解码器数量
        self.n_decoders = n_decoders
        # 循环添加n_decoders个ResDecoderBlock模块到模块列表中
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    # 前向传播方法，接受输入张量x和连接张量列表concat_tensors
    def forward(self, x, concat_tensors):
        # 循环遍历解码器的模块列表，对输入张量和连接张量进行前向传播
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1-i])
        # 返回处理后的张量
        return x


# 定义一个名为TimbreFilter的类，继承自nn.Module
class TimbreFilter(nn.Module):
    # 初始化方法，接受潜在表示通道数的列表
    def __init__(self, latent_rep_channels):
        super(TimbreFilter, self).__init__()
        # 创建一个空的模块列表
        self.layers = nn.ModuleList()
        # 循环添加ConvBlockRes模块到模块列表中
        for latent_rep in latent_rep_channels:
            self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0])

    # 前向传播方法，接受输入张量列表x_tensors
    def forward(self, x_tensors):
        # 创建一个空的输出张量列表
        out_tensors = []
        # 循环遍历模块列表，对输入张量列表进行前向传播
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        # 返回处理后的张量列表
        return out_tensors


# 定义一个名为DeepUnet的类，继承自nn.Module
class DeepUnet(nn.Module):
    # 初始化函数，定义了 DeepUnet 类的构造方法
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        # 调用父类的构造方法
        super(DeepUnet, self).__init__()
        # 创建编码器对象，传入输入通道数、输出通道数、编码器-解码器层数、卷积核大小、块数、编码器输出通道数
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        # 创建中间层对象，传入编码器输出通道数的一半、编码器输出通道数、中间层层数、块数
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        # 创建音色滤波器对象，传入编码器潜在通道数
        self.tf = TimbreFilter(self.encoder.latent_channels)
        # 创建解码器对象，传入编码器输出通道数、编码器-解码器层数、卷积核大小、块数
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    # 前向传播函数，定义了 DeepUnet 类的前向传播方法
    def forward(self, x):
        # 调用编码器的前向传播方法，获取编码器输出和连接张量
        x, concat_tensors = self.encoder(x)
        # 将编码器输出传入中间层，获取中间层输出
        x = self.intermediate(x)
        # 将连接张量传入音色滤波器，获取处理后的连接张量
        concat_tensors = self.tf(concat_tensors)
        # 将中间层输出和处理后的连接张量传入解码器，获取解码器输出
        x = self.decoder(x, concat_tensors)
        # 返回解码器输出
        return x
# 定义一个名为 DeepUnet0 的神经网络模型类，继承自 nn.Module
class DeepUnet0(nn.Module):
    # 初始化函数，接受一些参数用于构建模型
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        # 调用父类的初始化函数
        super(DeepUnet0, self).__init__()
        # 创建一个名为 encoder 的编码器对象，使用传入的参数进行初始化
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        # 创建一个名为 intermediate 的中间层对象，使用编码器的输出通道数进行初始化
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        # 创建一个名为 tf 的音色滤波器对象，使用编码器的潜在通道数进行初始化
        self.tf = TimbreFilter(self.encoder.latent_channels)
        # 创建一个名为 decoder 的解码器对象，使用编码器的输出通道数和传入的参数进行初始化
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    # 前向传播函数，接受输入 x，返回模型的输出
    def forward(self, x):
        # 将输入 x 通过编码器得到输出和连接张量
        x, concat_tensors = self.encoder(x)
        # 将输出 x 通过中间层处理
        x = self.intermediate(x)
        # 将处理后的输出 x 通过解码器得到最终输出
        x = self.decoder(x, concat_tensors)
        # 返回最终输出
        return x
```