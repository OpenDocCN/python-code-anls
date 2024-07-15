# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\modules\ecapa_tdnn.py`

```py
# 导入PyTorch库中的相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# 定义一个名为Res2Conv1dReluBn的神经网络模块，继承自nn.Module
class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        # 确保channels可以被scale整除，否则抛出异常
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        # 初始化卷积层和批归一化层的列表
        self.convs = []
        self.bns = []
        # 根据nums的数量，创建对应数量的卷积层和批归一化层，并添加到列表中
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    # 前向传播函数
    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)  # 在维度1上按width分割输入x
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # 执行卷积 -> relu激活 -> 批归一化的顺序
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])  # 如果scale不为1，将剩余的部分也添加到输出中
        out = torch.cat(out, dim=1)  # 在维度1上连接输出列表中的张量
        return out


# 定义一个名为Conv1dReluBn的神经网络模块，继承自nn.Module
class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        # 定义一个卷积层和一个批归一化层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    # 前向传播函数
    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))  # 执行卷积 -> relu激活 -> 批归一化的顺序


# 定义一个名为SE_Connect的神经网络模块，继承自nn.Module
class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        # 确保channels可以被s整除，否则抛出异常
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        # 定义两个线性层
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    # 前向传播函数
    def forward(self, x):
        out = x.mean(dim=2)  # 在维度2上求均值
        out = F.relu(self.linear1(out))  # 执行线性变换 -> relu激活
        out = torch.sigmoid(self.linear2(out))  # 执行线性变换 -> sigmoid激活
        out = x * out.unsqueeze(2)  # 对输入x应用注意力机制
        return out


# 定义一个名为SE_Res2Block的函数，用于构建一个特定结构的神经网络模块
def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


# 定义一个名为AttentiveStatsPool的神经网络模块，继承自nn.Module
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # 使用卷积层而不是线性层，kernel_size=1，stride=1，这样不需要转置输入
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)  # 相当于论文中的W和b
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)  # 相当于论文中的V和k
    # 定义神经网络的前向传播函数，接受输入张量 x
    def forward(self, x):
        # 不要在这里使用 ReLU 激活函数！在实验中发现 ReLU 很难收敛。
        
        # 使用第一个线性层进行计算，然后通过双曲正切函数作为激活
        alpha = torch.tanh(self.linear1(x))
        
        # 对 alpha 进行 softmax 操作，dim=2 表示对第二维进行 softmax
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        
        # 计算加权平均值，alpha * x 表示加权后的输入张量
        mean = torch.sum(alpha * x, dim=2)
        
        # 计算残差的加权平均值，alpha * x ** 2 表示加权后的平方输入张量
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        
        # 计算标准差，使用 clamp 函数保证不小于 1e-9
        std = torch.sqrt(residuals.clamp(min=1e-9))
        
        # 在维度 1 上连接均值和标准差，形成最终输出张量
        return torch.cat([mean, std], dim=1)
class EcapaTdnn(nn.Module):
    def __init__(self, input_size=80, channels=512, embd_dim=192):
        super().__init__()
        # 第一层卷积层，输入大小为input_size，输出通道数为channels，卷积核大小为5，填充为2，扩张率为1
        self.layer1 = Conv1dReluBn(input_size, channels, kernel_size=5, padding=2, dilation=1)
        # 第二层SE-Res2块，输入通道数为channels，卷积核大小为3，步长为1，填充为2，扩张率为2，缩放尺度为8
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        # 第三层SE-Res2块，输入通道数为channels，卷积核大小为3，步长为1，填充为3，扩张率为3，缩放尺度为8
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        # 第四层SE-Res2块，输入通道数为channels，卷积核大小为3，步长为1，填充为4，扩张率为4，缩放尺度为8
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        # 计算合并后的通道数
        cat_channels = channels * 3
        # 计算最终的输出通道数
        out_channels = cat_channels * 2
        # 设置嵌入向量的维度
        self.emb_size = embd_dim
        # 1x1卷积层，输入通道数为cat_channels，输出通道数为cat_channels
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        # 基于注意力的统计池化，输入通道数为cat_channels，输出长度为128
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        # BN层，输入通道数为out_channels
        self.bn1 = nn.BatchNorm1d(out_channels)
        # 线性层，输入维度为out_channels，输出维度为embd_dim
        self.linear = nn.Linear(out_channels, embd_dim)
        # BN层，输入通道数为embd_dim
        self.bn2 = nn.BatchNorm1d(embd_dim)

    def forward(self, x):
        # 第一层前向传播
        out1 = self.layer1(x)
        # 第二层前向传播，加上残差连接
        out2 = self.layer2(out1) + out1
        # 第三层前向传播，加上残差连接
        out3 = self.layer3(out1 + out2) + out1 + out2
        # 第四层前向传播，加上残差连接
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        # 合并所有特征图
        out = torch.cat([out2, out3, out4], dim=1)
        # 使用ReLU激活函数的1x1卷积层
        out = F.relu(self.conv(out))
        # 经过基于注意力的统计池化
        out = self.bn1(self.pooling(out))
        # BN层
        out = self.bn2(self.linear(out))
        # 返回最终输出
        return out


class SpeakerIdetification(nn.Module):
    def __init__(
            self,
            backbone,
            num_class=1,
            lin_blocks=0,
            lin_neurons=192,
            dropout=0.1, ):
        """
        The speaker identification model, which includes the speaker backbone network
        and a linear transform to speaker class num in training

        Args:
            backbone (Paddle.nn.Layer class): the speaker identification backbone network model
            num_class (_type_): the speaker class num in the training dataset
            lin_blocks (int, optional): the number of linear layer blocks between the embedding and the final linear layer. Defaults to 0.
            lin_neurons (int, optional): the output dimension of the final linear layer. Defaults to 192.
            dropout (float, optional): the dropout factor on the embedding. Defaults to 0.1.
        """
        super(SpeakerIdetification, self).__init__()
        # speaker identification backbone network model
        # the output of the backbone network is the target embedding
        self.backbone = backbone

        # Set dropout layer if dropout rate is greater than 0
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Construct the linear blocks for transformation
        input_size = self.backbone.emb_size
        self.blocks = list()
        for i in range(lin_blocks):
            self.blocks.extend([
                nn.BatchNorm1d(input_size),  # Batch normalization layer
                nn.Linear(in_features=input_size, out_features=lin_neurons),  # Linear transformation layer
            ])
            input_size = lin_neurons

        # Initialize the weight parameter for the final classifier
        self.weight = Parameter(torch.FloatTensor(num_class, input_size), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x):
        """
        Perform forward pass of the speaker identification model,
        including the speaker embedding model and the classifier model network

        Args:
            x (paddle.Tensor): input audio feats,
                               shape=[batch, dimension, times]
            lengths (paddle.Tensor, optional): input audio length.
                                        shape=[batch, times]
                                        Defaults to None.

        Returns:
            paddle.Tensor: return the logits of the feats
        """
        # x.shape: (N, C, L)
        x = self.backbone(x)  # Extract embeddings using backbone network (N, emb_size)
        if self.dropout is not None:
            x = self.dropout(x)  # Apply dropout if enabled

        # Apply each linear transformation block
        for fc in self.blocks:
            x = fc(x)

        # Calculate logits using normalized embeddings and weights
        logits = F.linear(F.normalize(x), F.normalize(self.weight, dim=-1))

        return logits
```