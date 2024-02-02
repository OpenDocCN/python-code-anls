# `so-vits-svc\modules\F0Predictor\rmvpe\model.py`

```py
# 导入 torch 模块中的 nn 类
from torch import nn

# 从当前目录下的 constants 模块中导入所有内容
from .constants import *  # noqa: F403
# 从当前目录下的 deepunet 模块中导入 DeepUnet 和 DeepUnet0 类
from .deepunet import DeepUnet, DeepUnet0
# 从当前目录下的 seq 模块中导入 BiGRU 类
from .seq import BiGRU
# 从当前目录下的 spec 模块中导入 MelSpectrogram 类
from .spec import MelSpectrogram

# 定义 E2E 类，继承自 nn.Module 类
class E2E(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(self, hop_length, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        # 调用父类的初始化方法
        super(E2E, self).__init__()
        # 创建 MelSpectrogram 对象并赋值给 self.mel 属性
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)  # noqa: F405
        # 创建 DeepUnet 对象并赋值给 self.unet 属性
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        # 创建 nn.Conv2d 对象并赋值给 self.cnn 属性
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        # 根据 n_gru 的值选择不同的网络结构
        if n_gru:
            # 创建包含多个层的神经网络模型并赋值给 self.fc 属性
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),   # noqa: F405
                nn.Linear(512, N_CLASS),   # noqa: F405
                nn.Dropout(0.25),
                nn.Sigmoid()
            )
        else:
            # 创建包含多个层的神经网络模型并赋值给 self.fc 属性
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),  # noqa: F405
                nn.Dropout(0.25),
                nn.Sigmoid()
            )

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 对输入 x 进行变换得到 mel，并进行维度转换和扩展
        mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
        # 将 mel 输入到 self.unet 和 self.cnn 中，得到输出 x
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        # 隐藏向量初始化为 0
        hidden_vec = 0
        # 如果 self.fc 中包含 4 个层
        if len(self.fc) == 4:
            # 遍历 self.fc 中的每一层
            for i in range(len(self.fc)):
                # 将 x 输入到第 i 层，并在特定条件下更新 hidden_vec
                x = self.fc[i](x)
                if i == 0:
                    hidden_vec = x
        # 返回隐藏向量和输出 x
        return hidden_vec, x


# 定义 E2E0 类，继承自 nn.Module 类
class E2E0(nn.Module):
    # 初始化函数，定义了模型的结构和参数
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1,
                 en_out_channels=16):
        # 调用父类的初始化函数
        super(E2E0, self).__init__()
        # 创建一个 DeepUnet0 对象，用于特征提取
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        # 创建一个 2D 卷积层，将特征映射到 3 个通道
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        # 如果存在 GRU 层
        if n_gru:
            # 创建一个包含双向 GRU 层的序列模型
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),  # noqa: F405
                # 创建一个全连接层，将输入映射到 N_CLASS 个类别
                nn.Linear(512, N_CLASS),  # noqa: F405
                # 创建一个丢弃层，以减少过拟合
                nn.Dropout(0.25),
                # 创建一个 Sigmoid 激活函数层
                nn.Sigmoid()
            )
        else:
            # 如果不存在 GRU 层
            self.fc = nn.Sequential(
                # 创建一个全连接层，将输入映射到 N_CLASS 个类别
                nn.Linear(3 * N_MELS, N_CLASS),  # noqa: F405
                # 创建一个丢弃层，以减少过拟合
                nn.Dropout(0.25),
                # 创建一个 Sigmoid 激活函数层
                nn.Sigmoid()
            )

    # 前向传播函数，定义了数据在模型中的传播过程
    def forward(self, mel):
        # 转置输入数据的最后两个维度，并在第二个维度上增加一个维度
        mel = mel.transpose(-1, -2).unsqueeze(1)
        # 将输入数据通过特征提取网络和卷积层，然后进行维度转置和压平操作
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        # 将处理后的数据通过全连接层
        x = self.fc(x)
        # 返回处理后的结果
        return x
```