# `.\pytorch\test\mobile\model_test\quantization_ops.py`

```py
import torch
import torch.nn as nn

# 定义一个继承自 torch.nn.Module 的通用量化模块类
class GeneralQuantModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 创建一个量化的 Embedding 层，设置词汇表大小为 10，嵌入维度为 12
        self.embedding = torch.ao.nn.quantized.Embedding(
            num_embeddings=10, embedding_dim=12
        )
        # 定义一个输入张量作为 Embedding 层的输入
        self.embedding_input = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        # 创建一个量化的 QFunctional 实例
        self.func = torch.ao.nn.quantized.QFunctional()
        # 创建一个量化的一维转置卷积层，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
        self.conv1 = torch.ao.nn.quantized.ConvTranspose1d(16, 33, 3, stride=2)
        # 创建一个量化的二维转置卷积层，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
        self.conv2 = torch.ao.nn.quantized.ConvTranspose2d(16, 33, 3, stride=2)
        # 创建一个量化的三维转置卷积层，输入通道数为 16，输出通道数为 33，卷积核大小为 3，步长为 2
        self.conv3 = torch.ao.nn.quantized.ConvTranspose3d(16, 33, 3, stride=2)

    # 前向传播方法
    def forward(self):
        # 创建一个量化的标量张量 a，值为 3.0
        a = torch.quantize_per_tensor(torch.tensor([3.0]), 1.0, 0, torch.qint32)
        # 创建一个量化的标量张量 b，值为 4.0
        b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        # 创建一个量化的标量张量 c，值为 3.0
        c = torch.quantize_per_tensor(
            torch.tensor([3.0]), torch.tensor(1.0), torch.tensor(0), torch.qint32
        )
        # 创建三个不同形状的随机输入张量
        input1 = torch.randn(1, 16, 4)
        input2 = torch.randn(1, 16, 4, 4)
        input3 = torch.randn(1, 16, 4, 4, 4)
        
        # 返回多个量化操作的结果
        return len(
            self.func.add(a, b),                      # 加法
            self.func.cat((a, a), 0),                 # 连接
            self.func.mul(a, b),                      # 乘法
            self.func.add_relu(a, b),                 # ReLU 加法
            self.func.add_scalar(a, b),               # 标量加法
            self.func.mul_scalar(a, b),               # 标量乘法
            self.embedding(self.embedding_input),     # Embedding 层的结果
            self.conv1(                               # 第一个转置卷积层的结果
                torch.quantize_per_tensor(
                    input1, scale=1.0, zero_point=0, dtype=torch.quint8
                )
            ),
            self.conv2(                               # 第二个转置卷积层的结果
                torch.quantize_per_tensor(
                    input2, scale=1.0, zero_point=0, dtype=torch.quint8
                )
            ),
            c,                                        # 张量 c
            # self.conv3(torch.quantize_per_tensor(input3, scale=1.0, zero_point=0, dtype=torch.quint8)), # iOS 上执行失败的转置卷积层
        )

# 定义一个动态量化模块类
class DynamicQuantModule:
    def __init__(self):
        super().__init__()
        # 创建一个模块实例
        self.module = self.M()

    # 获取量化后的模块
    def getModule(self):
        return torch.ao.quantization.quantize_dynamic(self.module, dtype=torch.qint8)
    # 定义名为 M 的 PyTorch 模块类
    class M(torch.nn.Module):
        # 初始化方法
        def __init__(self):
            # 调用父类初始化方法
            super(DynamicQuantModule.M, self).__init__()
            # 创建一个具有 4 个输入特征、8 个隐藏单元、2 层的 RNN 模型
            self.rnn = nn.RNN(4, 8, 2)
            # 创建一个 RNN 单元，输入特征为 4，隐藏单元为 8
            self.rnncell = nn.RNNCell(4, 8)
            # 创建一个具有 4 个输入特征、8 个隐藏单元、2 层的 GRU 模型
            self.gru = nn.GRU(4, 8, 2)
            # 创建一个 GRU 单元，输入特征为 4，隐藏单元为 8
            self.grucell = nn.GRUCell(4, 8)
            # 创建一个具有 4 个输入特征、8 个隐藏单元、2 层的 LSTM 模型
            self.lstm = nn.LSTM(4, 8, 2)
            # 创建一个 LSTM 单元，输入特征为 4，隐藏单元为 8
            self.lstmcell = nn.LSTMCell(4, 8)
            # 创建一个模块列表，包含不同类型的线性变换层
            self.linears = nn.ModuleList(
                [
                    nn.Identity(54),    # 恒等映射层，输入和输出维度为 54
                    nn.Linear(20, 20),   # 线性映射层，输入和输出维度为 20
                    nn.Bilinear(20, 20, 40),  # 双线性映射层，输入维度为 20，输出维度为 40
                ]
            )
            # 创建一个模块列表，包含不同类型的 Transformer 模型和相关层
            self.transformers = nn.ModuleList(
                [
                    nn.Transformer(
                        d_model=2, nhead=2, num_encoder_layers=1, num_decoder_layers=1
                    ),  # 创建一个 Transformer 模型，设置维度、头数、编码器和解码器层数
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=2, nhead=2), num_layers=1
                    ),  # 创建一个 Transformer 编码器，设置层数和相关层参数
                    nn.TransformerDecoder(
                        nn.TransformerDecoderLayer(d_model=2, nhead=2), num_layers=1
                    ),  # 创建一个 Transformer 解码器，设置层数和相关层参数
                ]
            )
            # 注释掉的代码：使用 pad_sequence 对列表中的张量序列进行填充
            # self.a = torch.nn.utils.rnn.pad_sequence([torch.tensor([1,2,3]), torch.tensor([3,4])], batch_first=True)

        # 前向传播方法
        def forward(self):
            # 创建随机输入张量
            input = torch.randn(5, 3, 4)
            # 创建随机隐藏状态张量
            h = torch.randn(2, 3, 8)
            # 创建随机细胞状态张量
            c = torch.randn(2, 3, 8)
            # 创建随机线性输入张量
            linear_input = torch.randn(32, 20)
            # 创建随机 Transformer 输入张量
            trans_input = torch.randn(1, 16, 2)
            # 创建随机 Transformer 目标张量
            tgt = torch.rand(1, 16, 2)

            # 返回模型中各组件的长度
            return len(
                (
                    self.rnn(input, h),             # RNN 模型的输出
                    self.rnncell(input[0], h[0]),   # RNN 单元的输出
                    self.gru(input, h),             # GRU 模型的输出
                    self.grucell(input[0], h[0]),   # GRU 单元的输出
                    self.lstm(input, (h, c)),       # LSTM 模型的输出
                    # 注释掉的代码：对列表中的张量序列进行填充，并使用 LSTM 模型处理
                    # self.lstm(torch.nn.utils.rnn.pack_padded_sequence(self.a, lengths=torch.tensor([3,2,1])), (h, c)),
                    self.lstmcell(input[0], (h[0], c[0])),   # LSTM 单元的输出
                    self.transformers[0](trans_input, tgt), # 第一个 Transformer 模型的输出
                    self.transformers[1](trans_input),      # 第二个 Transformer 编码器的输出
                    self.transformers[2](trans_input, tgt), # 第三个 Transformer 解码器的输出
                    self.linears[0](linear_input),          # 第一个线性变换层的输出
                    self.linears[1](linear_input),          # 第二个线性变换层的输出
                    self.linears[2](linear_input, linear_input),  # 第三个线性变换层的输出
                )
            )
class StaticQuantModule:
    # 定义静态量化模块类

    def getModule(self):
        # 获取量化后的模型
        model_fp32 = self.M()  # 创建模型实例
        model_fp32.eval()  # 设置模型为评估模式
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        # 配置模型的量化配置为qnnpack
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        # 准备模型以便进行量化
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        # 将准备好的模型转换为INT8格式
        return model_int8  # 返回INT8格式的量化模型

    class M(torch.nn.Module):
        # 定义包含多个层的模型类M，继承自torch.nn.Module

        def __init__(self):
            # 初始化方法
            super(StaticQuantModule.M, self).__init__()
            self.quant = torch.ao.quantization.QuantStub()
            # 创建量化填充层
            self.input1d = torch.randn(4, 2, 2)
            # 创建1维输入张量
            self.input2d = torch.randn((4, 2, 4, 4))
            # 创建2维输入张量
            self.input3d = torch.randn(4, 2, 2, 4, 4)
            # 创建3维输入张量
            self.linear_input = torch.randn(32, 20)
            # 创建线性层输入张量

            self.layer1 = nn.Sequential(
                nn.Conv1d(2, 2, 1), nn.InstanceNorm1d(1), nn.Hardswish()
            )
            # 创建第一层序列包含1维卷积、实例归一化和Hardswish激活函数
            self.layer2 = nn.Sequential(
                nn.Conv2d(2, 2, 1),
                nn.BatchNorm2d(2),
                nn.InstanceNorm2d(1),
                nn.LeakyReLU(),
            )
            # 创建第二层序列包含2维卷积、批归一化、实例归一化和LeakyReLU激活函数
            self.layer3 = nn.Sequential(
                nn.Conv3d(2, 2, 1), nn.BatchNorm3d(2), nn.InstanceNorm3d(1), nn.ReLU()
            )
            # 创建第三层序列包含3维卷积、批归一化、实例归一化和ReLU激活函数
            self.layer4 = nn.Sequential(nn.Linear(4, 3))
            # 创建第四层序列包含线性层

            self.dequant = torch.ao.quantization.DeQuantStub()
            # 创建反量化填充层

        def forward(self):
            # 前向传播方法
            x = self.quant(self.input1d)  # 对1维输入张量进行量化
            x = self.layer1(x)  # 使用第一层序列处理x
            x = self.dequant(x)  # 对x进行反量化

            y = self.input2d  # 将2维输入张量赋给y
            y = self.quant(y)  # 对y进行量化
            y = self.layer2(y)  # 使用第二层序列处理y
            y = self.layer4(y)  # 使用第四层序列处理y
            y = self.dequant(y)  # 对y进行反量化

            z = self.quant(self.input3d)  # 对3维输入张量进行量化
            z = self.layer3(z)  # 使用第三层序列处理z
            z = self.dequant(z)  # 对z进行反量化

            return (x, y, z)  # 返回处理后的张量x, y, z


class FusedQuantModule:
    # 定义融合量化模块类

    def getModule(self):
        # 获取量化后的模型
        model_fp32 = self.M()  # 创建模型实例
        model_fp32.eval()  # 设置模型为评估模式
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        # 配置模型的量化配置为qnnpack
        model_fp32_fused = torch.ao.quantization.fuse_modules(
            model_fp32,
            [
                ["conv1d", "relu1"],
                ["conv2d", "relu2"],
                ["conv3d", "relu3"],
                ["linear", "relu4"],
            ],
        )
        # 融合模型中的指定模块和激活函数
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        # 准备模型以便进行量化
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        # 将准备好的模型转换为INT8格式
        return model_int8  # 返回INT8格式的量化模型
    # 定义一个名为 M 的神经网络模块，继承自 torch.nn.Module
    class M(torch.nn.Module):
        # 构造函数，初始化模块内部状态
        def __init__(self):
            # 调用父类构造函数初始化
            super(FusedQuantModule.M, self).__init__()
            # 定义量化模块，用于量化输入数据
            self.quant = torch.ao.quantization.QuantStub()
            # 创建三种不同形状的随机输入数据
            self.input1d = torch.randn(4, 2, 2)
            self.input2d = torch.randn((4, 2, 4, 4))
            self.input3d = torch.randn(4, 2, 2, 4, 4)
            # 定义一维、二维、三维的卷积层
            self.conv1d = nn.Conv1d(2, 2, 1)
            self.conv2d = nn.Conv2d(2, 2, 1)
            self.conv3d = nn.Conv3d(2, 2, 1)
            # 定义全连接层，输入维度为 4，输出维度为 2
            self.linear = nn.Linear(4, 2)
            # 定义四个 ReLU 激活函数
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            # 定义反量化模块，用于反量化输出数据
            self.dequant = torch.ao.quantization.DeQuantStub()

        # 前向传播函数，定义了模型的前向计算流程
        def forward(self):
            # 对一维输入进行量化
            x = self.input1d
            x = self.quant(x)
            # 一维卷积计算
            x = self.conv1d(x)
            # 一维 ReLU 激活
            x = self.relu1(x)
            # 反量化输出
            x = self.dequant(x)

            # 对二维输入进行量化
            y = self.input2d
            y = self.quant(y)
            # 二维卷积计算
            y = self.conv2d(y)
            # 二维 ReLU 激活
            y = self.relu2(y)
            # 反量化输出
            y = self.dequant(y)

            # 对三维输入进行量化
            z = self.input3d
            z = self.quant(z)
            # 三维卷积计算
            z = self.conv3d(z)
            # 三维 ReLU 激活
            z = self.relu3(z)
            # 全连接层计算
            z = self.linear(z)
            # 四维 ReLU 激活
            z = self.relu4(z)
            # 反量化输出
            z = self.dequant(z)

            # 返回计算结果，包括一维、二维、三维的输出
            return (x, y, z)
```