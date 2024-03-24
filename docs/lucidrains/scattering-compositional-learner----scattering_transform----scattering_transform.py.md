# `.\lucidrains\scattering-compositional-learner\scattering_transform\scattering_transform.py`

```
# 导入 PyTorch 库
import torch
from torch import nn
import torch.nn.functional as F

# 辅助函数

# 如果 val 不为 None，则返回 val，否则返回 default_val
def default(val, default_val):
    return val if val is not None else default_val

# 在指定维度上扩展张量 t 的大小为 k
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 简单的具有 ReLU 激活函数的多层感知机

class MLP(nn.Module):
    def __init__(self, *dims, activation = None):
        super().__init__()
        assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in and dimension out'
        activation = default(activation, nn.ReLU)

        layers = []
        pairs = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(pairs):
            is_last = ind >= (len(pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if not is_last:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 论文中提到的前馈残差块
# 用于在提取视觉特征后以及提取属性信息后使用

class FeedForwardResidual(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.LayerNorm(dim * mult),
            nn.ReLU(inplace = True),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return x + self.net(x)

# 卷积网络
# 待完成，使其可定制化并添加 Evonorm 以进行批次独立归一化

class ConvNet(nn.Module):
    def __init__(self, image_size, chans, output_dim):
        super().__init__()

        num_conv_layers = len(chans) - 1
        conv_output_size = image_size // (2 ** num_conv_layers)

        convolutions = []
        channel_pairs = list(zip(chans[:-1], chans[1:]))

        for ind, (chan_in, chan_out) in enumerate(channel_pairs):
            is_last = ind >= (len(channel_pairs) - 1)
            convolutions.append(nn.Conv2d(chan_in, chan_out, 3, padding=1, stride=2))
            if not is_last:
                convolutions.append(nn.BatchNorm2d(chan_out))

        self.net = nn.Sequential(
            *convolutions,
            nn.Flatten(1),
            nn.Linear(chans[-1] * (conv_output_size ** 2), output_dim),
            nn.ReLU(inplace=True),
            FeedForwardResidual(output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 散射变换

class ScatteringTransform(nn.Module):
    def __init__(self, dims, heads, activation = None):
        super().__init__()
        assert len(dims) > 2, 'must have at least 3 dimensions, for dimension in, the hidden dimension, and dimension out'

        dim_in, *hidden_sizes, dim_out = dims

        dim_in //= heads
        dim_out //= heads

        self.heads = heads
        self.mlp = MLP(dim_in, *hidden_sizes, dim_out, activation = activation)

    def forward(self, x):
        shape, heads = x.shape, self.heads
        dim = shape[-1]

        assert (dim % heads) == 0, f'the dimension {dim} must be divisible by the number of heads {heads}'

        x = x.reshape(-1, heads, dim // heads)
        x = self.mlp(x)

        return x.reshape(shape)

# 主要的散射组合学习器类

class SCL(nn.Module):
    # 初始化函数，设置模型的参数
    def __init__(
        self,
        image_size = 160,  # 图像大小
        set_size = 9,  # 集合大小
        conv_channels = [1, 16, 16, 32, 32, 32],  # 卷积通道数
        conv_output_dim = 80,  # 卷积输出维度
        attr_heads = 10,  # 属性头数
        attr_net_hidden_dims = [128],  # 属性网络隐藏层维度
        rel_heads = 80,  # 关系头数
        rel_net_hidden_dims = [64, 23, 5]):  # 关系网络隐藏层维度

        super().__init__()
        # 创建视觉模型
        self.vision = ConvNet(image_size, conv_channels, conv_output_dim)

        # 设置属性头数和属性网络
        self.attr_heads = attr_heads
        self.attr_net = ScatteringTransform([conv_output_dim, *attr_net_hidden_dims, conv_output_dim], heads = attr_heads)
        self.ff_residual = FeedForwardResidual(conv_output_dim)

        # 设置关系头数和关系网络
        self.rel_heads = rel_heads
        self.rel_net = MLP(set_size * (conv_output_dim // rel_heads), *rel_net_hidden_dims)

        # 线性层，用于输出logits
        self.to_logit = nn.Linear(rel_net_hidden_dims[-1] * rel_heads, 1)

    # 前向传播函数
    def forward(self, sets):
        # 获取输入集合的形状信息
        b, m, n, c, h, w = sets.shape
        # 将集合展平为二维张量
        images = sets.view(-1, c, h, w)
        # 提取图像特征
        features = self.vision(images)

        # 计算属性
        attrs = self.attr_net(features)
        attrs = self.ff_residual(attrs)

        # 重塑属性张量形状
        attrs = attrs.reshape(b, m, n, self.rel_heads, -1).transpose(-2, -3).flatten(3)
        # 计算关系
        rels = self.rel_net(attrs)
        rels = rels.flatten(2)
        
        # 计算logits
        logits = self.to_logit(rels).flatten(1)
        return logits
# 为了更容易进行训练而创建的包装器类
class SCLTrainingWrapper(nn.Module):
    def __init__(self, scl):
        super().__init__()
        self.scl = scl

    # 前向传播函数，接收问题和答案作为输入
    def forward(self, questions, answers):
        # 在答案张量上增加一个维度
        answers = answers.unsqueeze(2)
        # 在问题张量上扩展维度，维度1扩展为8
        questions = expand_dim(questions, dim=1, k=8)

        # 将问题和答案张量连接在一起，沿着第二个维度
        permutations = torch.cat((questions, answers), dim=2)
        # 将连接后的张量传递给self.scl进行处理
        return self.scl(permutations)
```