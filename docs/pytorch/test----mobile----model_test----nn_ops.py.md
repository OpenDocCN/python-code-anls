# `.\pytorch\test\mobile\model_test\nn_ops.py`

```
# 引入 PyTorch 库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个卷积模块的类，继承自 torch.nn.Module
# 参考链接：https://pytorch.org/docs/stable/nn.html
class NNConvolutionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化不同维度的输入数据
        self.input1d = torch.randn(1, 4, 36)  # 1维输入数据，形状为(1, 4, 36)
        self.input2d = torch.randn(1, 4, 30, 10)  # 2维输入数据，形状为(1, 4, 30, 10)
        self.input3d = torch.randn(1, 4, 10, 4, 4)  # 3维输入数据，形状为(1, 4, 10, 4, 4)
        
        # 初始化不同维度的模块列表
        self.module1d = nn.ModuleList(
            [
                nn.Conv1d(4, 33, 3),  # 1维卷积层，输入通道4，输出通道33，卷积核大小3
                nn.ConvTranspose1d(4, 33, 3),  # 1维转置卷积层，输入通道4，输出通道33，卷积核大小3
                nn.Fold(output_size=(5, 10), kernel_size=(2, 2)),  # 折叠操作，输出大小(5, 10)，卷积核大小(2, 2)
            ]
        )
        self.module2d = nn.ModuleList(
            [
                nn.Conv2d(4, 33, 3),  # 2维卷积层，输入通道4，输出通道33，卷积核大小3x3
                nn.ConvTranspose2d(4, 33, 3),  # 2维转置卷积层，输入通道4，输出通道33，卷积核大小3x3
                nn.Unfold(kernel_size=3),  # 解折叠操作，卷积核大小3x3
            ]
        )
        self.module3d = nn.ModuleList(
            [
                nn.Conv3d(4, 33, 2),  # 3维卷积层，输入通道4，输出通道33，卷积核大小2x2x2
                nn.ConvTranspose3d(4, 33, 3),  # 3维转置卷积层，输入通道4，输出通道33，卷积核大小3x3x3
            ]
        )

    # 前向传播函数
    def forward(self):
        # 返回不同维度模块处理后的结果的长度
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],  # 对1维模块列表中的每个模块应用于1维输入数据
                [module(self.input2d) for i, module in enumerate(self.module2d)],  # 对2维模块列表中的每个模块应用于2维输入数据
                [module(self.input3d) for i, module in enumerate(self.module3d)],  # 对3维模块列表中的每个模块应用于3维输入数据
            )
        )


# 定义一个池化模块的类，继承自 torch.nn.Module
class NNPoolingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化不同维度的输入数据
        self.input1d = torch.randn(1, 16, 50)  # 1维输入数据，形状为(1, 16, 50)
        
        # 初始化1维池化模块列表
        self.module1d = nn.ModuleList(
            [
                nn.MaxPool1d(3, stride=2),  # 最大池化，窗口大小3，步幅2
                nn.AvgPool1d(3, stride=2),  # 平均池化，窗口大小3，步幅2
                nn.LPPool1d(2, 3, stride=2),  # Lp范数池化，p=2，窗口大小3，步幅2
                nn.AdaptiveMaxPool1d(3),  # 自适应最大池化，输出大小为3
                nn.AdaptiveAvgPool1d(3),  # 自适应平均池化，输出大小为3
            ]
        )

        self.input2d = torch.randn(1, 16, 30, 10)  # 2维输入数据，形状为(1, 16, 30, 10)
        
        # 初始化2维池化模块列表
        self.module2d = nn.ModuleList(
            [
                nn.MaxPool2d((3, 2), stride=(2, 1)),  # 最大池化，窗口大小为(3, 2)，步幅为(2, 1)
                nn.AvgPool2d((3, 2), stride=(2, 1)),  # 平均池化，窗口大小为(3, 2)，步幅为(2, 1)
                nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5)),  # 分数最大池化，窗口大小为3，输出比率为(0.5, 0.5)
                nn.LPPool2d(2, 3, stride=(2, 1)),  # Lp范数池化，p=2，窗口大小为(3, 2)，步幅为(2, 1)
                nn.AdaptiveMaxPool2d((5, 7)),  # 自适应最大池化，输出大小为(5, 7)
                nn.AdaptiveAvgPool2d(7),  # 自适应平均池化，输出大小为(7, 7)
            ]
        )

        self.input3d = torch.randn(1, 16, 20, 4, 4)  # 3维输入数据，形状为(1, 16, 20, 4, 4)
        
        # 初始化3维池化模块列表
        self.module3d = nn.ModuleList(
            [
                nn.MaxPool3d(2),  # 最大池化，窗口大小为2
                nn.AvgPool3d(2),  # 平均池化，窗口大小为2
                nn.FractionalMaxPool3d(2, output_ratio=(0.5, 0.5, 0.5)),  # 分数最大池化，窗口大小为2，输出比率为(0.5, 0.5, 0.5)
                nn.AdaptiveMaxPool3d((5, 7, 9)),  # 自适应最大池化，输出大小为(5, 7, 9)
                nn.AdaptiveAvgPool3d((5, 7, 9)),  # 自适应平均池化，输出大小为(5, 7, 9)
            ]
        )
        # TODO max_unpool

    # 前向传播函数
    def forward(self):
        # 返回不同维度模块处理后的结果的长度
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],  # 对1维模块列表中的每个模块应用于1维输入数据
                [module(self.input2d) for i, module in enumerate(self.module2d)],  # 对2维模块列表中的每个模块应用于2维输入数据
                [module(self.input3d) for i, module in enumerate(self.module3d)],  # 对3维模块列表中的每个模块应用于3维输入数据
            )
        )


class NNPaddingModule(torch.nn.Module):
    # 初始化函数，用于创建对象实例时初始化各个属性
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个形状为 (1, 4, 50) 的随机张量作为输入数据 input1d
        self.input1d = torch.randn(1, 4, 50)
        # 创建一个包含三个一维填充模块的模块列表 module1d
        self.module1d = nn.ModuleList(
            [
                nn.ReflectionPad1d(2),        # 使用反射填充的一维填充模块，填充大小为 2
                nn.ReplicationPad1d(2),       # 使用复制填充的一维填充模块，填充大小为 2
                nn.ConstantPad1d(2, 3.5),     # 使用常量填充的一维填充模块，填充大小为 2，常量值为 3.5
            ]
        )

        # 创建一个形状为 (1, 4, 30, 10) 的随机张量作为输入数据 input2d
        self.input2d = torch.randn(1, 4, 30, 10)
        # 创建一个包含四个二维填充模块的模块列表 module2d
        self.module2d = nn.ModuleList(
            [
                nn.ReflectionPad2d(2),        # 使用反射填充的二维填充模块，填充大小为 2
                nn.ReplicationPad2d(2),       # 使用复制填充的二维填充模块，填充大小为 2
                nn.ZeroPad2d(2),              # 使用零填充的二维填充模块，填充大小为 2
                nn.ConstantPad2d(2, 3.5),     # 使用常量填充的二维填充模块，填充大小为 2，常量值为 3.5
            ]
        )

        # 创建一个形状为 (1, 4, 10, 4, 4) 的随机张量作为输入数据 input3d
        self.input3d = torch.randn(1, 4, 10, 4, 4)
        # 创建一个包含三个三维填充模块的模块列表 module3d
        self.module3d = nn.ModuleList(
            [
                nn.ReflectionPad3d(1),        # 使用反射填充的三维填充模块，填充大小为 1
                nn.ReplicationPad3d(3),       # 使用复制填充的三维填充模块，填充大小为 3
                nn.ConstantPad3d(3, 3.5),     # 使用常量填充的三维填充模块，填充大小为 3，常量值为 3.5
            ]
        )

    # 前向传播函数，定义了对象实例的前向计算逻辑
    def forward(self):
        # 返回三个模块列表中每个模块对应输入数据的计算结果组成的元组的长度
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],  # 计算一维模块列表每个模块对应 input1d 的结果
                [module(self.input2d) for i, module in enumerate(self.module2d)],  # 计算二维模块列表每个模块对应 input2d 的结果
                [module(self.input3d) for i, module in enumerate(self.module3d)],  # 计算三维模块列表每个模块对应 input3d 的结果
            )
        )
# 定义一个神经网络模块，用于 NN 归一化处理
class NNNormalizationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个 1 维输入张量，形状为 (1, 4, 50)
        self.input1d = torch.randn(1, 4, 50)
        # 初始化 1 维规范化模块列表
        self.module1d = nn.ModuleList(
            [
                nn.BatchNorm1d(4),      # 批标准化模块，处理 4 个通道
                nn.InstanceNorm1d(4),   # 实例标准化模块，处理 4 个通道
            ]
        )

        # 创建一个 2 维输入张量，形状为 (1, 4, 30, 10)
        self.input2d = torch.randn(1, 4, 30, 10)
        # 初始化 2 维规范化模块列表
        self.module2d = nn.ModuleList(
            [
                nn.BatchNorm2d(4),      # 批标准化模块，处理 4 个通道
                nn.GroupNorm(4, 4),      # 分组标准化模块，每组 4 个通道
                nn.InstanceNorm2d(4),   # 实例标准化模块，处理 4 个通道
                nn.LayerNorm([4, 30, 10]),  # 层标准化模块，处理指定维度的张量
                nn.LocalResponseNorm(2),     # 局部响应标准化模块，参数指定
            ]
        )

        # 创建一个 3 维输入张量，形状为 (1, 4, 10, 4, 4)
        self.input3d = torch.randn(1, 4, 10, 4, 4)
        # 初始化 3 维规范化模块列表
        self.module3d = nn.ModuleList(
            [
                nn.BatchNorm3d(4),      # 批标准化模块，处理 4 个通道
                nn.InstanceNorm3d(4),   # 实例标准化模块，处理 4 个通道
                nn.ChannelShuffle(2),   # 通道混洗模块，每 2 个通道进行一次混洗
            ]
        )

    def forward(self):
        # 执行前向传播，返回各个规范化模块对输入的处理结果长度
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


# 定义一个神经网络模块，用于 NN 激活函数处理
class NNActivationModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化激活函数模块列表
        self.activations = nn.ModuleList(
            [
                nn.ELU(),           # ELU 激活函数
                nn.Hardshrink(),    # Hardshrink 硬收缩激活函数
                nn.Hardsigmoid(),   # Hardsigmoid 硬 Sigmoid 激活函数
                nn.Hardtanh(),      # Hardtanh 硬切线激活函数
                nn.Hardswish(),     # Hardswish 硬 Swish 激活函数
                nn.LeakyReLU(),     # LeakyReLU 泄露线性整流单元激活函数
                nn.LogSigmoid(),    # LogSigmoid 对数 Sigmoid 激活函数
                nn.PReLU(),         # PReLU 参数化 ReLU 激活函数
                nn.ReLU(),          # ReLU 线性整流单元激活函数
                nn.ReLU6(),         # ReLU6 限制输出在 [0, 6] 区间的 ReLU 激活函数
                nn.RReLU(),         # RReLU 随机 ReLU 激活函数
                nn.SELU(),          # SELU 自归一化激活函数
                nn.CELU(),          # CELU 渐进线性激活函数
                nn.GELU(),          # GELU 高斯误差线性单元激活函数
                nn.Sigmoid(),       # Sigmoid S 型激活函数
                nn.SiLU(),          # SiLU Sigmoid 线性整流单元激活函数
                nn.Mish(),          # Mish 激活函数
                nn.Softplus(),      # Softplus 平滑版 ReLU 激活函数
                nn.Softshrink(),    # Softshrink 软收缩激活函数
                nn.Softsign(),      # Softsign 软符号激活函数
                nn.Tanh(),          # Tanh 双曲正切激活函数
                nn.Tanhshrink(),    # Tanhshrink 双曲正切收缩激活函数
                nn.GLU(),           # GLU 门控线性单元激活函数
                nn.Softmin(),       # Softmin 最小值归一化激活函数
                nn.Softmax(),       # Softmax 多类别 Softmax 激活函数
                nn.Softmax2d(),     # Softmax2d 二维 Softmax 激活函数
                nn.LogSoftmax(),    # LogSoftmax 对数 Softmax 激活函数
                # nn.AdaptiveLogSoftmaxWithLoss(),  # 自适应对数 Softmax 损失函数
            ]
        )

    def forward(self):
        # 执行前向传播，返回各个激活函数对输入的处理结果长度
        input = torch.randn(2, 3, 4)
        return len(([module(input) for i, module in enumerate(self.activations)],))


# 定义一个神经网络模块，用于 NN 循环神经网络处理
class NNRecurrentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化 RNN 模块列表
        self.rnn = nn.ModuleList(
            [
                nn.RNN(4, 8, 2),    # RNN 循环神经网络模块，输入 4 维，输出 8 维，层数 2
                nn.RNNCell(4, 8),   # RNNCell 循环神经网络单元模块，输入 4 维，输出 8 维
            ]
        )
        # 初始化 GRU 模块列表
        self.gru = nn.ModuleList(
            [
                nn.GRU(4, 8, 2),    # GRU 门控循环单元模块，输入 4 维，输出 8 维，层数 2
                nn.GRUCell(4, 8),   # GRUCell 门控循环单元单元模块，输入 4 维，输出 8 维
            ]
        )
        # 初始化 LSTM 模块列表
        self.lstm = nn.ModuleList(
            [
                nn.LSTM(4, 8, 2),   # LSTM 长短时记忆网络模块，输入 4 维，输出 8 维，层数 2
                nn.LSTMCell(4, 8),  # LSTMCell 长短时记忆网络单元模块，输入 4 维，输出 8 维
            ]
        )
    # 定义一个方法 `forward`，用于模型前向传播计算
    def forward(self):
        # 生成一个形状为 (5, 3, 4) 的随机张量作为输入数据
        input = torch.randn(5, 3, 4)
        # 生成一个形状为 (2, 3, 8) 的随机张量作为初始隐藏状态 h
        h = torch.randn(2, 3, 8)
        # 生成一个形状为 (2, 3, 8) 的随机张量作为初始细胞状态 c
        c = torch.randn(2, 3, 8)
        
        # 使用第一个 RNN 层处理输入 input 和初始隐藏状态 h
        r = self.rnn[0](input, h)
        # 使用第二个 RNN 层处理输入 input 的第一个时间步和初始隐藏状态 h 的第一个时间步
        r = self.rnn[1](input[0], h[0])
        
        # 使用第一个 GRU 层处理输入 input 和初始隐藏状态 h
        r = self.gru[0](input, h)
        # 使用第二个 GRU 层处理输入 input 的第一个时间步和初始隐藏状态 h 的第一个时间步
        r = self.gru[1](input[0], h[0])
        
        # 使用第一个 LSTM 层处理输入 input 和初始隐藏状态 h、细胞状态 c
        r = self.lstm[0](input, (h, c))
        # 使用第二个 LSTM 层处理输入 input 的第一个时间步和初始隐藏状态 h 的第一个时间步、细胞状态 c 的第一个时间步
        r = self.lstm[1](input[0], (h[0], c[0]))
        
        # 返回变量 r 的长度作为输出
        return len(r)
# 定义一个自定义的神经网络模块，继承自 `torch.nn.Module`
class NNTransformerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 `nn.ModuleList` 存储多个 Transformer 相关的子模块
        self.transformers = nn.ModuleList(
            [
                # 创建一个 Transformer 模型
                nn.Transformer(
                    d_model=2, nhead=2, num_encoder_layers=1, num_decoder_layers=1
                ),
                # 创建一个 TransformerEncoder 模型
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=2, nhead=2), num_layers=1
                ),
                # 创建一个 TransformerDecoder 模型
                nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(d_model=2, nhead=2), num_layers=1
                ),
            ]
        )

    def forward(self):
        # 创建一个随机输入张量 `input`
        input = torch.rand(1, 16, 2)
        # 创建一个随机目标张量 `tgt`
        tgt = torch.rand((1, 16, 2))
        # 使用第一个 Transformer 进行前向传播
        r = self.transformers[0](input, tgt)
        # 使用第二个 TransformerEncoder 进行前向传播
        r = self.transformers[1](input)
        # 使用第三个 TransformerDecoder 进行前向传播
        r = self.transformers[2](input, tgt)
        # 返回结果的长度
        return len(r)


# 定义一个自定义的神经网络模块，继承自 `torch.nn.Module`
class NNLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 `nn.ModuleList` 存储多个线性变换相关的子模块
        self.linears = nn.ModuleList(
            [
                # 创建一个恒等映射模块
                nn.Identity(54),
                # 创建一个线性变换模块
                nn.Linear(20, 20),
                # 创建一个双线性变换模块
                nn.Bilinear(20, 20, 40),
                # 创建一个注释掉的 LazyLinear 模块
                # nn.LazyLinear(20, 30),
            ]
        )

    def forward(self):
        # 创建一个随机输入张量 `input`
        input = torch.randn(32, 20)
        # 使用第一个模块进行前向传播
        r = self.linears[0](input)
        # 使用第二个模块进行前向传播
        r = self.linears[1](input)
        # 使用第三个模块进行前向传播
        r = self.linears[2](input, input)
        # 返回结果的长度
        return len(r)


# 定义一个自定义的神经网络模块，继承自 `torch.nn.Module`
class NNDropoutModule(torch.nn.Module):
    def forward(self):
        # 创建三个不同形状的随机张量 `a`, `b`, `c`
        a = torch.randn(8, 4)
        b = torch.randn(8, 4, 4, 4)
        c = torch.randn(8, 4, 4, 4, 4)
        # 应用不同类型的 dropout 操作，并返回每个操作结果的长度
        return len(
            F.dropout(a),
            F.dropout2d(b),
            F.dropout3d(c),
            F.alpha_dropout(a),
            F.feature_alpha_dropout(c),
        )


# 定义一个自定义的神经网络模块，继承自 `torch.nn.Module`
class NNSparseModule(torch.nn.Module):
    def forward(self):
        # 创建两个示例输入张量 `input` 和 `input2`
        input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        input2 = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        # 创建一个随机的嵌入矩阵 `embedding_matrix`
        embedding_matrix = torch.rand(10, 3)
        # 应用不同的稀疏操作，并返回每个操作结果的长度
        return len(
            F.embedding(input, embedding_matrix),
            F.embedding_bag(input2, embedding_matrix, torch.tensor([0, 4])),
            F.one_hot(torch.arange(0, 5) % 3, num_classes=5),
        )


# 定义一个自定义的神经网络模块，继承自 `torch.nn.Module`
class NNDistanceModule(torch.nn.Module):
    def forward(self):
        # 创建两个随机输入张量 `a`, `b`
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        # 应用不同的距离计算操作，并返回每个操作结果的长度
        return len(
            F.pairwise_distance(a, b),
            F.cosine_similarity(a, b),
            F.pdist(a),
        )


# 定义一个自定义的神经网络模块，继承自 `torch.nn.Module`
class NNLossFunctionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建两个张量 `x`, `y`，用于模块内部使用
        self.x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
        self.y = torch.LongTensor([[3, 0, -1, 1]])
    # 定义一个方法 `forward`，用于模型的前向传播计算
    def forward(self):
        # 创建一个大小为 (3, 2) 的张量 a，其中元素服从标准正态分布
        a = torch.randn(3, 2)
        # 创建一个大小为 (3, 2) 的张量 b，其中元素服从 [0, 1) 的均匀分布
        b = torch.rand(3, 2)
        # 创建一个大小为 (3,) 的张量 c，其中元素服从 [0, 1) 的均匀分布
        c = torch.rand(3)
        # 创建一个大小为 (50, 16, 20) 的张量 log_probs，先对第二个维度做 log_softmax 处理后再 detach
        log_probs = torch.randn(50, 16, 20).log_softmax(2).detach()
        # 创建一个大小为 (16, 30) 的长整型张量 targets，元素在 [1, 20) 范围内随机生成
        targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        # 创建一个大小为 (16,) 的长整型张量 input_lengths，每个元素均为 50
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        # 创建一个大小为 (16,) 的长整型张量 target_lengths，元素在 [10, 30) 范围内随机生成
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        # 返回下面一系列损失函数的长度
        return len(
            # 计算 sigmoid 函数后与目标张量 b 之间的二元交叉熵损失
            F.binary_cross_entropy(torch.sigmoid(a), b),
            # 计算 sigmoid 函数后与目标张量 b 之间的带 logits 的二元交叉熵损失
            F.binary_cross_entropy_with_logits(torch.sigmoid(a), b),
            # 计算 Poisson 分布的负对数似然损失
            F.poisson_nll_loss(a, b),
            # 计算余弦相似度损失，用于度量向量 a 和 b 之间的相似度
            F.cosine_embedding_loss(a, b, c),
            # 计算交叉熵损失，用于多类别分类任务
            F.cross_entropy(a, b),
            # 计算 CTC（Connectionist Temporal Classification）损失
            F.ctc_loss(log_probs, targets, input_lengths, target_lengths),
            # 计算铰链损失，用于二分类问题
            F.hinge_embedding_loss(a, b),
            # 计算 KL 散度损失，度量两个概率分布的差异
            F.kl_div(a, b),
            # 计算 L1 损失，度量向量 a 和 b 之间的绝对误差
            F.l1_loss(a, b),
            # 计算均方误差损失，度量向量 a 和 b 之间的平方误差
            F.mse_loss(a, b),
            # 计算边际排名损失，用于排序学习任务
            F.margin_ranking_loss(c, c, c),
            # 计算多标签边际损失，用于多标签分类任务
            F.multilabel_margin_loss(self.x, self.y),
            # 计算多标签软边际损失，用于多标签分类任务
            F.multilabel_soft_margin_loss(self.x, self.y),
            # 计算多分类边际损失，用于多类别分类任务
            F.multi_margin_loss(self.x, torch.tensor([3])),
            # 计算负对数似然损失，用于多类别分类任务
            F.nll_loss(a, torch.tensor([1, 0, 1])),
            # 计算 Huber 损失，一种平滑的损失函数
            F.huber_loss(a, b),
            # 计算平滑 L1 损失，一种平滑的 L1 损失函数
            F.smooth_l1_loss(a, b),
            # 计算软边际损失，用于二分类问题
            F.soft_margin_loss(a, b),
            # 计算三元组边际损失，用于三元组学习任务
            F.triplet_margin_loss(a, b, -b),
            # F.triplet_margin_with_distance_loss(a, b, -b), # 该函数不支持可变数量的参数
        )
# 定义一个神经网络视觉模块类
class NNVisionModule(torch.nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()
        # 初始化输入数据
        self.input = torch.randn(1, 4, 9, 9)
        # 初始化视觉模块列表
        self.vision_modules = nn.ModuleList(
            [
                nn.PixelShuffle(2),
                nn.PixelUnshuffle(3),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Upsample(scale_factor=2, mode="bicubic"),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.UpsamplingBilinear2d(scale_factor=2),
            ]
        )
        # 初始化线性采样模块
        self.linear_sample = nn.Upsample(scale_factor=2, mode="linear")
        # 初始化三线性采样模块
        self.trilinear_sample = nn.Upsample(scale_factor=2, mode="trilinear")

    # 前向传播函数
    def forward(self):
        # 重新初始化输入数据
        input = torch.randn(1, 3, 16, 16)
        # 遍历视觉模块列表并应用模块
        for i, module in enumerate(self.vision_modules):
            r = module(self.input)
        # 返回结果的长度
        return len(
            r,
            self.linear_sample(torch.randn(4, 9, 9)),
            self.trilinear_sample(torch.randn(1, 3, 4, 9, 9)),
            F.grid_sample(input, torch.ones(1, 4, 4, 2)),
        )


# 定义一个神经网络通道混洗模块类
class NNShuffleModule(torch.nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()
        # 初始化通道混洗模块
        self.shuffle = nn.ChannelShuffle(2)

    # 前向传播函数
    def forward(self):
        # 返回结果的长度
        return len(
            self.shuffle(torch.randn(1, 4, 2, 2)),
        )


# 定义一个神经网络工具模块类
class NNUtilsModule(torch.nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()
        # 初始化展平模块
        self.flatten = nn.Sequential(nn.Linear(50, 50), nn.Unflatten(1, (2, 5, 5)))

    # 前向传播函数
    def forward(self):
        # 创建输入数据列表
        a = [torch.tensor([1, 2, 3]), torch.tensor([3, 4])]
        # 对输入数据进行填充
        b = nn.utils.rnn.pad_sequence(a, batch_first=True)
        # c = nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=torch.tensor([3, 2]))
        # 初始化输入数据
        input = torch.randn(2, 50)
        # 返回结果的长度
        return len(
            self.flatten(input),
            b,
        )
```