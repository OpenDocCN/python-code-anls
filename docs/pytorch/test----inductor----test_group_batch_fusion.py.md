# `.\pytorch\test\inductor\test_group_batch_fusion.py`

```py
# Owner(s): ["module: inductor"]

# 导入必要的库和模块
import collections
import unittest
from typing import List

# 导入 PyTorch 相关模块
import torch
import torch._inductor
import torch._inductor.fx_passes.group_batch_fusion
from torch._dynamo.utils import counters, optimus_scuba_log
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA

try:
    # 尝试导入 deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings 模块，用于注册 fbgemm 降低操作
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    # 如果导入失败，设置 has_fbgemm 为 False
    has_fbgemm = False
    pass

# 如果没有 CUDA 支持，则跳过测试
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


# 定义一个名为 TestHighwaySelfGating 的测试类，继承自 torch.nn.Module
class TestHighwaySelfGating(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        size: int,
        device="cuda",
    ) -> None:
        super().__init__()
        # 初始化函数，设定模型的属性
        self.size = size
        self.device = device
        # 创建一个线性层 gating_proj 和 transform_proj，均输入和输出维度为 d_model
        self.gating_proj = torch.nn.Linear(d_model, d_model).to(self.device)
        self.transform_proj = torch.nn.Linear(d_model, d_model).to(self.device)
        # 创建一个 Sigmoid 激活函数 gating_func
        self.gating_func = torch.nn.Sigmoid().to(self.device)

        self.d_model = d_model

    # 定义前向传播函数 forward
    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        # 初始化一个空列表 results 用于存储每个输入张量处理后的结果
        results = []
        # 对于输入列表中的每个张量，进行处理
        for i in range(self.size):
            x = inputs[i]
            # 对当前输入张量 x 应用 gating_proj 线性层
            gating_proj = self.gating_proj(x)
            # 对当前输入张量 x 应用 transform_proj 线性层
            transform_proj = self.transform_proj(x)
            # 计算 gating_proj 和 gating_func(transform_proj) 的乘积
            x = gating_proj * self.gating_func(transform_proj)
            # 将处理后的结果 x 添加到 results 列表中
            results.append(x)

        # 在维度 -1 上连接 results 中的张量，并返回结果
        return torch.cat(results, dim=-1)


# 定义一个名为 MyModule 的自定义模块，继承自 torch.nn.Module
class MyModule(torch.nn.Module):
    def __init__(self, z: int, has_bias: bool, device="cuda") -> None:
        super().__init__()
        # 初始化函数，设定模型的属性
        self.z = z
        self.device = device
        self.seq_len = 10
        # 创建包含多个线性层的序列 seq1，每个线性层输入和输出维度为 z，是否包含偏置由 has_bias 决定
        self.seq1 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        # 创建包含多个线性层的序列 seq2，每个线性层输入和输出维度为 z，是否包含偏置由 has_bias 决定
        self.seq2 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]
        # 创建包含多个线性层的序列 seq3，每个线性层输入和输出维度为 z，是否包含偏置由 has_bias 决定
        self.seq3 = [
            torch.nn.Linear(z, z, has_bias).to(self.device) for _ in range(self.seq_len)
        ]

    # 定义前向传播函数 forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始化一个列表 x1，包含了对输入张量 x 做轻微处理后的结果
        x1 = [x + 0.1 * i for i in range(self.seq_len)]
        # 对 x1 中的每个处理后的张量应用对应的 seq1 中的线性层，得到 x2
        x2 = [self.seq1[i](x1[i]) for i in range(self.seq_len)]
        # 对 x2 中的每个张量做进一步处理，得到 x3
        x3 = [x2[i] - 0.1 * i for i in range(self.seq_len)]
        # 构造 x4 列表，包含部分 x1 和 x3 中的处理结果
        x4 = [x1[i] for i in range(3)] + [x3[i] for i in range(3, self.seq_len)]
        # 对 x4 中的每个处理后的张量应用对应的 seq2 中的线性层，得到 x5
        x5 = [self.seq2[i](x4[i]) for i in range(self.seq_len)]
        # 对 x5 中的每个张量做进一步处理，得到 x6
        x6 = [x5[i] + 0.1 * (self.seq_len - i) for i in range(self.seq_len)]
        # 构造 x7 列表，包含部分 x1、x3 和 x6 中的处理结果
        x7 = (
            [x1[i] for i in range(4)]
            + [x3[i] for i in range(6, 8)]
            + [x6[i] for i in range(4)]
        )
        # 对 x7 中的每个处理后的张量应用对应的 seq3 中的线性层，得到 x8
        x8 = [self.seq3[i](x7[i]) for i in range(self.seq_len)]
        # 在维度 1 上连接 x8 中的张量，并返回结果
        x9 = torch.cat(x8, dim=1)
        return x9


class MyModule2(torch.nn.Module):
    # 定义一个神经网络模型的初始化函数，继承父类构造方法
    def __init__(self) -> None:
        super().__init__()
        # 定义多个线性层，每个层的输入维度和输出维度分别是固定的
        self.linear0 = torch.nn.Linear(6, 8)
        self.linear1 = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(10, 8)
        self.linear3 = torch.nn.Linear(6, 8)
        self.linear4 = torch.nn.Linear(8, 8)
        self.linear5 = torch.nn.Linear(10, 8)
        # 定义多个批归一化层，每层的输入维度为8
        self.bn0 = torch.nn.BatchNorm1d(8)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.bn2 = torch.nn.BatchNorm1d(8)

    # 定义神经网络模型的前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量 x 按指定维度进行分割成三个张量
        t = torch.split(x, [6, 8, 10], dim=1)
        # 对分割后的第一个张量 t[0] 执行线性变换并加上偏置 0.1，再经过批归一化处理
        a0 = self.bn0(self.linear0(t[0] + 0.1))
        # 对分割后的第二个张量 t[1] 执行线性变换并加上偏置 0.2，再经过批归一化处理
        a1 = self.bn1(self.linear1(t[1] + 0.2))
        # 对分割后的第三个张量 t[2] 执行线性变换并加上偏置 0.3，再经过批归一化处理
        a2 = self.bn2(self.linear2(t[2] + 0.3))
        # 对第一个分割张量 t[0] 执行正弦函数的线性变换
        a3 = self.linear3(torch.sin(t[0]))
        # 对第二个分割张量 t[1] 执行余弦函数的线性变换
        a4 = self.linear4(torch.cos(t[1]))
        # 对第三个分割张量 t[2] 的一半值执行正弦函数的线性变换
        a5 = self.linear5(torch.sin(t[2] * 0.5))

        # 将所有处理后的张量按第一个维度进行拼接
        b = torch.cat([a0, a1, a2, a3, a4, a5])
        # 对拼接后的张量 b 执行 sigmoid 函数，得到最终的输出张量
        return torch.sigmoid(b)
# 定义一个名为 MyModule3 的类，继承自 torch.nn.Module
class MyModule3(torch.nn.Module):
    # 初始化方法，接受 device、has_weight 和 has_bias 三个参数
    def __init__(self, device, has_weight=True, has_bias=True):
        # 调用父类的初始化方法
        super().__init__()
        # 将 device 参数保存到实例变量 self.device 中
        self.device = device
        # 创建名为 scale0 的参数列表，包含 5 个形状为 (10,) 的标准正态分布参数，移到设备上
        self.scale0 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
        ).to(self.device)
        # 创建名为 bias0 的参数列表，包含 5 个形状为 (10,) 的标准正态分布参数，移到设备上
        self.bias0 = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(10)) for _ in range(5)]
        ).to(self.device)
        # 根据 has_weight 参数确定是否创建 scale1 参数列表：
        # 如果 has_weight 为 True，创建包含 5 个形状为 (5, 10) 的标准正态分布参数，移到设备上；
        # 如果 has_weight 为 False，创建一个包含 5 个 None 元素的列表
        self.scale1 = (
            torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_weight
            else [None for _ in range(5)]
        )
        # 根据 has_bias 参数确定是否创建 bias1 参数列表：
        # 如果 has_bias 为 True，创建包含 5 个形状为 (5, 10) 的标准正态分布参数，移到设备上；
        # 如果 has_bias 为 False，创建一个包含 5 个 None 元素的列表
        self.bias1 = (
            torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(5, 10)) for _ in range(5)]
            ).to(self.device)
            if has_bias
            else [None for _ in range(5)]
        )

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 将输入 x 移到指定设备上，并按照维度 dim=2 分割为大小为 10 的张量列表 l1_out
        l1_out = torch.split(x.to(self.device), 10, dim=2)
        # 对 l1_out 中的每个张量进行 layer_norm 操作，使用 scale0 和 bias0 参数列表中对应的参数
        post_l1 = [
            torch.nn.functional.layer_norm(
                l1_out[i], (10,), weight=self.scale0[i], bias=self.bias0[i]
            )
            for i in range(len(l1_out))
        ]
        # 将 post_l1 中的张量按照维度 dim=2 连接起来
        l1_out = torch.cat(post_l1, dim=2)

        # 再次按照维度 dim=2 将 l1_out 分割为大小为 10 的张量列表 l2_out
        l2_out = torch.split(l1_out, 10, dim=2)
        # 对 l2_out 中的每个张量进行 layer_norm 操作，使用 scale1 和 bias1 参数列表中对应的参数
        post_l2 = [
            torch.nn.functional.layer_norm(
                l2_out[i], (5, 10), weight=self.scale1[i], bias=self.bias1[i]
            )
            for i in range(len(l2_out))
        ]

        # 将 post_l2 中的张量按照维度 dim=2 连接起来，并作为前向传播的输出结果返回
        return torch.cat(post_l2, dim=2)


# 定义一个名为 MyModule4 的类，继承自 torch.nn.Module
class MyModule4(torch.nn.Module):
    # 初始化方法，接受 z、device 和 has_bias 三个参数
    def __init__(self, z, device, has_bias):
        # 调用父类的初始化方法
        super().__init__()
        # 将 z 参数保存到实例变量 self.z 中
        self.z = z
        # 将 device 参数保存到实例变量 self.device 中
        self.device = device
        # 将 has_bias 参数保存到实例变量 self.has_bias 中
        self.has_bias = has_bias
        # 设置实例变量 seq_len 为 10
        self.seq_len = 10
        # 创建名为 weights1 的列表，包含 seq_len 个形状为 (z - i % 5, z) 的标准正态分布参数，移到设备上
        self.weights1 = [
            torch.nn.Parameter(torch.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]
        # 创建名为 weights2 的列表，包含 seq_len 个形状为 (z - i % 5, z) 的标准正态分布参数，移到设备上
        self.weights2 = [
            torch.nn.Parameter(torch.randn(z - i % 5, z)).to(self.device)
            for i in range(self.seq_len)
        ]

        # 如果 has_bias 为 True，则创建名为 biases1 和 biases2 的参数列表：
        # biases1 包含 seq_len 个形状为 (z - i % 5) 的标准正态分布参数，移到设备上
        # biases2 包含 seq_len 个形状为 (z - i % 5) 的标准正态分布参数，移到设备上
        if has_bias:
            self.biases1 = [
                torch.nn.Parameter(torch.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]
            self.biases2 = [
                torch.nn.Parameter(torch.randn(z - i % 5)).to(self.device)
                for i in range(self.seq_len)
            ]
    # 定义神经网络前向传播函数，接收输入张量 x
    def forward(self, x):
        # 将输入张量 x 加上常数 1.2
        x = x + 1.2
        # 使用线性函数对输入 x 执行线性变换，self.weights1 和 self.biases1 是权重和偏置列表
        # 如果没有偏置 self.has_bias 则为 None
        x1 = [
            torch.nn.functional.linear(
                x, self.weights1[i], self.biases1[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        # 将 x1 中的张量沿着维度 1 进行拼接
        x2 = torch.cat(x1, dim=1)
        # 沿着维度 1 将 x2 张量拆分成大小为 10 的多个张量
        x3 = torch.split(x2, 10, dim=1)
        # 将 x3 中的张量按照第一个维度拼接起来
        x4 = torch.cat(x3)
        # 使用线性函数对 x4 执行线性变换，self.weights2 和 self.biases2 是权重和偏置列表
        # 如果没有偏置 self.has_bias 则为 None
        x5 = [
            torch.nn.functional.linear(
                x4, self.weights2[i], self.biases2[i] if self.has_bias else None
            )
            for i in range(self.seq_len)
        ]
        # 将 x5 中的张量沿着维度 1 进行拼接
        x6 = torch.cat(x5, dim=1)
        # 对 x6 应用 sigmoid 激活函数并返回结果张量
        return torch.sigmoid(x6)
class MyModule5(torch.nn.Module):
    # 定义一个自定义的 PyTorch 模块，继承自 torch.nn.Module
    def __init__(self, device, has_bias=True):
        # 构造函数，初始化模块参数
        super().__init__()
        # 赋值设备信息到模块属性
        self.device = device

        # 创建权重列表，包含5个随机初始化的参数张量，转移到指定设备上
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(50, 100)).to(self.device) for _ in range(5)]
        )

        # 如果设置了偏置项，创建偏置列表，每个元素为一个参数张量，转移到指定设备上；否则创建空列表
        self.biases = (
            ([torch.nn.Parameter(torch.randn(50)).to(self.device) for _ in range(5)])
            if has_bias
            else [None for _ in range(5)]
        )

    def forward(self, x):
        # 前向传播函数，接收输入张量 x
        # 将输入张量按维度1分割成长度为100的子张量列表，并转移到指定设备上
        l1_out = torch.split(x.to(self.device), 100, dim=1)
        
        # 对每个子张量进行线性变换，使用权重和偏置（如果有的话），得到线性变换结果列表
        l1_linear = [
            torch.nn.functional.linear(l1_out[i], self.weights[i], self.biases[i])
            for i in range(len(l1_out))
        ]
        
        # 将线性变换的结果按维度1连接起来
        l1_out = torch.cat(l1_linear, dim=1)
        
        # 对连接结果逐元素计算正弦值，返回结果张量
        return torch.sin(l1_out)


class TestPoitwiseOps(torch.nn.Module):
    # 定义一个测试点操作的 PyTorch 模块，继承自 torch.nn.Module
    def __init__(self, device, has_bias=True):
        # 构造函数，初始化模块参数
        super().__init__()
        # 赋值设备信息到模块属性
        self.device = device

    def forward(self, x):
        # 前向传播函数，接收输入张量 x
        # 将输入张量按维度1分割成长度为500的子张量列表，并转移到指定设备上
        inputs = torch.split(x.to(self.device), 500, dim=1)
        
        # 将第一个分割后的子张量再按维度1分割成长度为50的子张量列表，并转移到指定设备上
        x_split = torch.split(inputs[0].to(self.device), 50, dim=1)
        
        # 将第二个分割后的子张量再按维度1分割成长度为50的子张量列表，并转移到指定设备上
        y_split = torch.split(inputs[1].to(self.device), 50, dim=1)
        
        # 对每个子张量列表中的张量逐元素计算双曲正切函数，得到结果列表
        tanh_1 = [torch.tanh(x_split[i]) for i in range(len(x_split))]
        tanh_2 = [torch.tanh(y_split[i]) for i in range(len(y_split))]
        
        # 对每个结果列表中的张量逐元素计算 sigmoid 函数，得到结果列表
        sigmoid_1 = [torch.sigmoid(tanh_1[i]) for i in range(len(tanh_1))]
        sigmoid_2 = [torch.sigmoid(tanh_2[i]) for i in range(len(tanh_2))]
        
        # 对每个结果列表中的张量逐元素计算 ReLU 函数，得到结果列表
        relu_1 = [torch.nn.functional.relu(sigmoid_1[i]) for i in range(len(sigmoid_1))]
        relu_2 = [torch.nn.functional.relu(sigmoid_2[i]) for i in range(len(sigmoid_2))]
        
        # 对对应位置的张量逐元素相加，得到结果列表
        add = [torch.add(relu_1[i], relu_2[i]) for i in range(len(relu_1))]
        
        # 对每个结果列表中的张量逐元素相乘，得到结果列表
        mul = [torch.mul(add[i], add[i]) for i in range(len(add))]
        
        # 对每个结果列表中的张量逐元素相减，得到结果列表
        sub = [torch.sub(mul[i], mul[i]) for i in range(len(mul))]
        
        # 对每个结果列表中的张量逐元素除法运算，得到结果列表
        div = [torch.div(sub[i], sub[i]) for i in range(len(sub))]
        
        # 将结果列表中的张量按维度1连接起来，返回结果张量
        return torch.cat(div, dim=1)


class TestPoitwiseOpsPostGrad(torch.nn.Module):
    # 定义一个测试点操作后梯度计算的 PyTorch 模块，继承自 torch.nn.Module
    def __init__(self, device):
        # 构造函数，初始化模块参数
        super().__init__()
        # 赋值设备信息到模块属性
        self.device = device

    def forward(self, x):
        # 前向传播函数，接收输入张量 x
        # 使用 ATen 操作将输入张量按维度1分割成长度为500的子张量列表，并转移到指定设备上
        inputs = torch.ops.aten.split(x.to(self.device), 500, dim=1)
        
        # 使用 ATen 操作将第一个分割后的子张量再按维度1分割成长度为50的子张量列表，并转移到指定设备上
        x_split = torch.ops.aten.split(inputs[0].to(self.device), 50, dim=1)
        
        # 使用 ATen 操作将第二个分割后的子张量再按维度1分割成长度为50的子张量列表，并转移到指定设备上
        y_split = torch.ops.aten.split(inputs[1].to(self.device), 50, dim=1)
        
        # 对每个子张量列表中的张量逐元素计算双曲正切函数，得到结果列表
        tanh_1 = [torch.ops.aten.tanh(x_split[i]) for i in range(len(x_split))]
        tanh_2 = [torch.ops.aten.tanh(y_split[i]) for i in range(len(y_split))]
        
        # 对每个结果列表中的张量逐元素计算 sigmoid 函数，得到结果列表
        sigmoid_1 = [torch.ops.aten.sigmoid(tanh_1[i]) for i in range(len(tanh_1))]
        sigmoid_2 = [torch.ops.aten.sigmoid(tanh_2[i]) for i in range(len(tanh_2))]
        
        # 对每个结果列表中的张量逐元素计算 ReLU 函数，得到结果列表
        relu_1 = [torch.ops.aten.relu(sigmoid_1[i]) for i in range(len(sigmoid_1))]
        relu_2 = [torch.ops.aten.relu(sigmoid_2[i]) for i in range(len(sigmoid_2))]
        
        # 对对应位置的张量逐元素相加，得到结果列表
        add = [torch.ops.aten.add(relu_1[i], relu_2[i]) for i in range(len(relu_1))]
        
        # 将结果列表中的张量按维度1连接起来，返回结果张量
        return torch.cat(add, dim=1)
    # 定义预梯度融合选项字典，包括批量线性操作、左手边批量线性操作、批量层归一化、批量双曲正切操作、批量ReLU激活、批量sigmoid激活
    pre_grad_fusion_options={
        "batch_linear": {},                # 批量线性操作选项为空字典
        "batch_linear_lhs": {},            # 左手边批量线性操作选项为空字典
        "batch_layernorm": {},             # 批量层归一化操作选项为空字典
        "batch_tanh": {},                  # 批量双曲正切操作选项为空字典
        "batch_relu": {},                  # 批量ReLU激活操作选项为空字典
        "batch_sigmoid": {},               # 批量sigmoid激活操作选项为空字典
    },
    
    # 定义后梯度融合选项字典，包括批量aten加法操作、批量aten乘法操作、批量aten减法操作、批量aten除法操作、组线性操作（需要fbgemm支持）
    post_grad_fusion_options={
        "batch_aten_add": {},             # 批量aten加法操作选项为空字典
        "batch_aten_mul": {},             # 批量aten乘法操作选项为空字典
        "batch_aten_sub": {},             # 批量aten减法操作选项为空字典
        "batch_aten_div": {},             # 批量aten除法操作选项为空字典
        "group_linear": {"require_fbgemm": True},  # 组线性操作选项包括require_fbgemm=True
    },
    )

class TestGroupBatchFusion(TestCase):
    # 比较两个字典的张量是否相似，返回布尔值，指定相对和绝对误差容差
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        # 如果参考字典和结果字典的键集合不相等，则返回 False
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        # 遍历参考字典的键
        for key1 in ref_dict.keys():
            # 构建对应的结果字典的键
            key2 = "_orig_mod." + key1
            # 断言结果字典中存在对应的键，否则输出错误信息
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            # 使用 torch.allclose 检查张量是否相似，若不相似则返回 False
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        # 若所有键值对比均通过，则返回 True
        return True

    # 比较预测输出，使用 module 和 traced 模型对输入进行前向计算，并比较输出结果的相似性
    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        # 使用 self.assertEqual 检查两个输出是否相等，若不相等则输出详细信息
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    # 比较模型的参数，使用 module 和 traced 模型的命名参数字典，并比较其张量是否相似
    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        # 使用 self.assertTrue 调用 compare_dict_tensors 方法检查参数字典是否相似
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    # 比较梯度，使用 module 和 traced 模型的参数梯度，并检查其相似性
    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        # 使用 self.assertTrue 调用 compare_dict_tensors 方法检查梯度字典是否相似
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    # 跳过测试，条件为没有 fbgemm 模块时
    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    # 测试组线性融合功能
    def test_group_linear_fusion(self):
        # 设置输入维度大小
        z = 10
        # 对于是否有偏置的情况进行迭代测试
        for has_bias in [True, False]:
            # 清空计数器
            counters.clear()
            # 创建 MyModule 类实例，传入参数并将其移到 CUDA 设备上
            module = MyModule(z, has_bias).to("cuda")
            # 创建输入张量列表，包含随机初始化的张量，并将其移到 CUDA 设备上
            input = [torch.randn(z, z, device="cuda")]
            # 使用 torch.compile 编译模型并追踪计算图
            traced = torch.compile(module)
            # 获取模型的参考输出和追踪后的输出
            ref = module(*input)
            res = traced(*input)
            # 使用 self.compare_pred 方法比较预测输出的相似性
            self.compare_pred(module, traced, input)
            # 使用 self.assertEqual 检查计数器中特定键的值是否为 2
            self.assertEqual(
                counters["inductor"]["group_linear"],
                2,
            )
            # 检查优化器日志中是否不包含特定键
            self.assertNotIn("group_batch_fusion_pre_grad", optimus_scuba_log)
            # 计算参考输出的和，并进行反向传播
            ref.sum().backward()
            # 计算追踪输出的和，并进行反向传播
            res.sum().backward()
            # 使用 self.compare_parameters 方法比较模型的参数
            self.compare_parameters(module, traced)
            # 使用 self.compare_gradients 方法比较模型的梯度
            self.compare_gradients(module, traced)
            # 使用 self.assertEqual 检查计数器中特定键的值是否为 4
            self.assertEqual(
                counters["inductor"]["group_linear"],
                4,
            )
            # 使用 self.assertEqual 检查计数器中特定键的值是否为 3
            self.assertEqual(
                counters["inductor"]["batch_aten_add"],
                3,
            )
            # 检查优化器日志中是否包含特定子字符串
            self.assertIn("GroupLinearFusion", optimus_scuba_log)
            # 清空计数器
            counters.clear()

    # 跳过测试，条件为没有 fbgemm 模块时
    @unittest.skipIf(not has_fbgemm, "requires fbgemm")
    # 测试线性融合不同形状的组
    def test_group_linear_fusion_different_shapes(self):
        # 清空计数器
        counters.clear()
        # 创建并将模块移到 CUDA 设备上，并设置为评估模式
        module = MyModule2().eval().to("cuda")
        # 创建输入张量列表，包含一个形状为 (4, 24) 的随机张量，移动到 CUDA 设备上
        input = [torch.rand(4, 24, device="cuda")]
        # 对模块进行编译（tracing）
        traced = torch.compile(module)
        # 调用模块的正向传播，获取参考结果
        ref = module(*input)
        # 调用编译后的模块进行正向传播，获取结果
        res = traced(*input)
        # 比较预测结果
        self.compare_pred(module, traced, input)
        # 断言计数器中 "inductor" 中的 "group_linear" 为 1
        self.assertEqual(
            counters["inductor"]["group_linear"],
            1,
        )
        # 断言计数器中 "inductor" 中的 "batch_fusion" 为 0
        self.assertEqual(
            counters["inductor"]["batch_fusion"],
            0,
        )
        # 对参考结果的总和进行反向传播
        ref.sum().backward()
        # 对编译后结果的总和进行反向传播
        res.sum().backward()
        # 比较模块的参数
        self.compare_parameters(module, traced)
        # 比较模块的梯度
        self.compare_gradients(module, traced)
        # 断言计数器中 "inductor" 中的 "group_linear" 为 2
        self.assertEqual(
            counters["inductor"]["group_linear"],
            2,
        )
        # 断言计数器中 "inductor" 中的 "batch_aten_mul" 为 1
        self.assertEqual(
            counters["inductor"]["batch_aten_mul"],
            1,
        )
        # 清空计数器
        counters.clear()

    # 测试批处理层归一化融合
    def test_batch_layer_norm_fusion(self):
        # 针对是否有权重和是否有偏置进行循环
        for has_weight in [True, False]:
            for has_bias in [True, False]:
                # 清空计数器
                counters.clear()
                # 创建带有指定配置的模块，将其移到 CUDA 设备上
                module = MyModule3("cuda", has_weight, has_bias).to("cuda")
                # 创建输入张量列表，包含一个形状为 (2, 5, 50) 的随机张量，移动到 CUDA 设备上
                input = [torch.randn(2, 5, 50, device="cuda")]
                # 对模块进行编译（tracing）
                traced = torch.compile(module)
                # 调用模块的正向传播，获取参考结果
                ref = module(*input)
                # 调用编译后的模块进行正向传播，获取结果
                res = traced(*input)
                # 比较预测结果
                self.compare_pred(module, traced, input)
                # 断言计数器中 "inductor" 中的 "batch_layernorm" 为 2
                self.assertEqual(counters["inductor"]["batch_layernorm"], 2)
                # 对参考结果的总和进行反向传播
                ref.sum().backward()
                # 对编译后结果的总和进行反向传播
                res.sum().backward()
                # 比较模块的参数，设置相对误差和绝对误差容忍度为 1e-8
                self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
                # 比较模块的梯度，设置相对误差和绝对误差容忍度为 1e-8
                self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
                # 清空计数器
                counters.clear()

    # 测试批处理线性层左手边融合
    def test_batch_linear_lhs_fusion(self):
        # 设置 z 的值为 10
        z = 10
        # 针对是否有偏置进行循环
        for has_bias in [True, False]:
            # 清空计数器
            counters.clear()
            # 创建带有指定配置的模块，移动到 CUDA 设备上
            module = MyModule4(z, "cuda", has_bias)
            # 创建输入张量列表，包含一个形状为 (20, z) 的随机张量，移动到 CUDA 设备上
            input = [torch.randn(20, z, device="cuda")]
            # 对模块进行编译（tracing）
            traced = torch.compile(module)
            # 调用模块的正向传播，获取参考结果
            ref = module(*input)
            # 调用编译后的模块进行正向传播，获取结果
            res = traced(*input)
            # 比较预测结果
            self.compare_pred(module, traced, input)
            # 断言计数器中 "inductor" 中的 "batch_linear_lhs" 为 2
            self.assertEqual(counters["inductor"]["batch_linear_lhs"], 2)
            # 对参考结果的总和进行反向传播
            ref.sum().backward()
            # 对编译后结果的总和进行反向传播
            res.sum().backward()
            # 比较模块的参数，设置相对误差和绝对误差容忍度为 1e-8
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            # 比较模块的梯度，设置相对误差和绝对误差容忍度为 1e-8
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            # 清空计数器
            counters.clear()
    # 定义测试方法，用于测试线性层预梯度融合功能
    def test_batch_linear_pre_grad_fusion(self):
        # 遍历是否包含偏置的情况
        for has_bias in [True, False]:
            # 清空计数器
            counters.clear()
            # 创建 MyModule5 实例，使用 CUDA 并传入是否包含偏置
            module = MyModule5("cuda", has_bias)
            # 创建输入张量列表，包含一个形状为 (50, 500) 的 CUDA 张量
            input = [torch.randn(50, 500, device="cuda")]
            # 编译模块为跟踪模式
            traced = torch.compile(module)
            # 计算模块的正向传播结果作为参考
            ref = module(*input)
            # 执行跟踪模式下的正向传播
            res = traced(*input)
            # 比较预测结果的一致性
            self.compare_pred(module, traced, input)
            # 断言批量线性计数器为 1
            self.assertEqual(counters["inductor"]["batch_linear"], 1)
            # 对参考结果进行求和并执行反向传播
            ref.sum().backward()
            # 对跟踪结果进行求和并执行反向传播
            res.sum().backward()
            # 比较模块参数的一致性，设置相对容差和绝对容差均为 1e-8
            self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
            # 比较梯度的一致性，设置相对容差和绝对容差均为 1e-8
            self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
            # 清空计数器
            counters.clear()

    # 定义测试方法，用于测试逐点操作融合功能
    def test_pointwise_op_fusion(self):
        # 清空计数器
        counters.clear()
        # 创建 TestPoitwiseOps 实例，使用 CUDA
        module = TestPoitwiseOps("cuda")
        # 创建输入张量列表，包含一个形状为 (50, 1000)、需要梯度的 CUDA 张量
        input = [torch.randn(50, 1000, requires_grad=True, device="cuda")]
        # 编译模块为跟踪模式
        traced = torch.compile(module)
        # 计算模块的正向传播结果作为参考
        ref = module(*input)
        # 执行跟踪模式下的正向传播
        res = traced(*input)
        # 比较预测结果的一致性
        self.compare_pred(module, traced, input)
        # 断言批量计数器中不同操作的出现次数
        self.assertEqual(counters["inductor"]["batch_tanh"], 1)
        self.assertEqual(counters["inductor"]["batch_relu"], 1)
        self.assertEqual(counters["inductor"]["batch_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_add"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_mul"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_sub"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_div"], 1)
        # 对参考结果进行求和并执行反向传播
        ref.sum().backward()
        # 对跟踪结果进行求和并执行反向传播
        res.sum().backward()
        # 比较模块参数的一致性，设置相对容差和绝对容差均为 1e-8
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        # 比较梯度的一致性，设置相对容差和绝对容差均为 1e-8
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        # 清空计数器
        counters.clear()

    # 定义需要 CUDA 支持的测试方法，用于测试逐点操作后梯度融合功能
    @requires_cuda
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "batch_aten_relu": {},
            "batch_aten_sigmoid": {},
            "batch_aten_tanh": {},
            "unbind_stack_aten_pass": {},
        },
    )
    def test_pointwise_op_fusion_post_grad(self):
        # 清空计数器
        counters.clear()
        # 创建 TestPoitwiseOpsPostGrad 实例，使用 CUDA
        module = TestPoitwiseOpsPostGrad("cuda")
        # 创建输入张量列表，包含一个形状为 (50, 1000)、需要梯度的 CUDA 张量
        input = [torch.randn(50, 1000, requires_grad=True, device="cuda")]
        # 编译模块为跟踪模式
        traced = torch.compile(module)
        # 计算模块的正向传播结果作为参考
        ref = module(*input)
        # 执行跟踪模式下的正向传播
        res = traced(*input)
        # 比较预测结果的一致性
        self.compare_pred(module, traced, input)
        # 断言批量计数器中不同操作的出现次数
        self.assertEqual(counters["inductor"]["batch_aten_tanh"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_relu"], 1)
        self.assertEqual(counters["inductor"]["batch_aten_sigmoid"], 1)
        self.assertEqual(counters["inductor"]["unbind_stack_aten_pass"], 2)
        # 对参考结果进行求和并执行反向传播
        ref.sum().backward()
        # 对跟踪结果进行求和并执行反向传播
        res.sum().backward()
        # 比较模块参数的一致性，设置相对容差和绝对容差均为 1e-8
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        # 比较梯度的一致性，设置相对容差和绝对容差均为 1e-8
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        # 清空计数器
        counters.clear()
    # 使用 torch._inductor.config.patch 装饰器来配置梯度融合选项，对模型的梯度计算进行优化设置
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            # 针对 "batch_linear_post_grad" 后向传播融合选项的设置
            "batch_linear_post_grad": {
                "shape_broadcast_batch_linear": True,  # 启用批处理线性层后向传播的形状广播
                "fuse_nodes_with_same_users": True,  # 融合具有相同用户的节点
            },
            # 针对 "batch_aten_mul" 后向传播融合选项的设置
            "batch_aten_mul": {"fuse_nodes_with_same_parent": False},  # 不融合具有相同父节点的节点
            # 针对 "batch_aten_sigmoid" 后向传播融合选项的设置
            "batch_aten_sigmoid": {"fuse_nodes_with_same_parent": True},  # 融合具有相同父节点的节点
            # 针对 "batch_aten_add" 后向传播融合选项的设置
            "batch_aten_add": {"fuse_nodes_with_same_parent": True},  # 融合具有相同父节点的节点
            # "normalization_aten_pass" 后向传播融合选项的设置
            "normalization_aten_pass": {},  # 空设置，无额外选项
            # "unbind_stack_aten_pass" 后向传播融合选项的设置
            "unbind_stack_aten_pass": {},  # 空设置，无额外选项
        },
    )
    # 定义用于测试门控融合后向传播的测试方法
    def test_gate_fusion_post_grad(self):
        # 清空计数器
        counters.clear()
        # 定义输入大小
        size = 20
        # 创建 TestHighwaySelfGating 模型对象
        module = TestHighwaySelfGating(d_model=10, size=size)
        # 构造输入数据，为包含随机张量的列表，需在 GPU 上
        input = [
            [
                torch.randn(10, 10, requires_grad=True, device="cuda")
                for i in range(size)
            ]
        ]
        # 编译模型以进行跟踪
        traced = torch.compile(module)
        # 获取模型的前向传播输出
        ref = module(*input)
        # 使用跟踪后的模型进行前向传播
        res = traced(*input)
        # 比较预测结果
        self.compare_pred(module, traced, input)
        # 检查计数器中 "batch_linear_post_grad" 的值是否为 2
        self.assertEqual(counters["inductor"]["batch_linear_post_grad"], 2)
        # 检查计数器中 "batch_aten_sigmoid" 的值是否为 1
        self.assertEqual(counters["inductor"]["batch_aten_sigmoid"], 1)
        # 检查计数器中 "batch_aten_mul" 的值是否为 1
        self.assertEqual(counters["inductor"]["batch_aten_mul"], 1)
        # 检查计数器中 "batch_aten_add" 的值是否为 2
        self.assertEqual(counters["inductor"]["batch_aten_add"], 2)
        # 检查计数器中 "normalization_aten_pass" 的值是否为 1
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 1)
        # 检查计数器中 "unbind_stack_aten_pass" 的值是否为 5
        self.assertEqual(counters["inductor"]["unbind_stack_aten_pass"], 5)
        # 对参考结果的总和进行反向传播
        ref.sum().backward()
        # 对跟踪结果的总和进行反向传播
        res.sum().backward()
        # 比较模型参数
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        # 比较梯度
        self.compare_gradients(module, traced, rtol=1e-8, atol=1e-8)
        # 清空计数器
        counters.clear()
class TestBMMFusionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_modules = torch.nn.ModuleList()
        for _ in range(10):
            # 向 self.my_modules 中添加 10 个输入维度和输出维度均为 10 的线性层
            self.my_modules.append(torch.nn.Linear(10, 10))

    def forward(self, inputs):
        output = None
        # 遍历 self.my_modules 中的线性层和输入数据，并进行线性变换
        for linear, input in zip(self.my_modules, inputs):
            if output is None:
                output = linear(input)
            else:
                output += linear(input)
        return output


@requires_cuda
@torch._inductor.config.patch(
    post_grad_fusion_options={"batch_linear_post_grad": {"require_fbgemm": False}}
)
class TestPostGradBatchLinearFusion(TestCase):
    def test_batch_linear_post_grad_fusion(self):
        # 创建一个 TestBMMFusionModule 的 CUDA 版本
        pt1_module = TestBMMFusionModule().cuda()
        inputs = []
        # 生成 10 个大小为 (10, 10) 的随机张量，并放入 inputs 列表中
        for _ in range(10):
            inputs.append(torch.randn(10, 10).cuda())
        # 在 pt1_module 上执行前向传播
        eager_output = pt1_module(inputs)
        # 编译 pt1_module 得到 pt2_module
        pt2_module = torch.compile(pt1_module)
        # 在 pt2_module 上执行前向传播
        pt2_output = pt2_module(inputs)
        # 断言 eager_output 和 pt2_output 的所有元素都近似相等
        self.assertTrue(torch.allclose(eager_output, pt2_output))
        # 断言 counters["inductor"]["batch_linear_post_grad"] 等于 2
        self.assertEqual(
            counters["inductor"]["batch_linear_post_grad"],
            2,
        )
        # 断言 "PostGradBatchLinearFusion" 在 optimus_scuba_log 中
        self.assertIn("PostGradBatchLinearFusion", optimus_scuba_log)


class TestFindIndependentSubsetGreedy(TestCase):
    # Helper function to build a Graph from a data description.
    def build_graph(self, desc):
        # 根据数据描述构建图形
        # desc: {
        #   "n1": ["n2", "n3"],
        #   "n2": ["n3"],
        #   "n3": [],
        # }
        g = torch.fx.Graph()
        lookup = {}
        # 将 desc 转换为双向队列，其中每个元素是键值对 (k, v)
        desc = collections.deque((k, v) for k, v in desc.items())
        unsatisfied = 0
        while desc:
            unsatisfied += 1
            # 断言 unsatisfied 不大于 desc 的长度，以防循环或错误输入
            assert unsatisfied <= len(desc)  # cycle or bad input?
            name, v = desc.popleft()
            # 从 lookup 中获取 v 中各节点的参数
            args = tuple(lookup.get(n, None) for n in v)
            if None in args:
                desc.append((name, v))
                continue
            # 创建一个占位符节点并添加到图中，同时更新 lookup
            node = g.create_node("placeholder", "target", name=name, args=args)
            lookup[name] = node
            unsatisfied = 0
        return g, lookup

    def verify(self, tree, subnodes, min_fuse, max_fuse, expected):
        # 根据 tree 构建图形，并验证子节点及期望结果
        g, lookup = self.build_graph(tree)
        subnodes = [lookup[n] for n in subnodes]
        expected = [[lookup[n] for n in sub] for sub in expected]
        opts = {
            "min_fuse_set_size": min_fuse,
            "max_fuse_set_size": max_fuse,
        }
        # 使用贪婪算法查找独立子集并返回结果
        result = list(
            torch._inductor.fx_passes.group_batch_fusion.find_independent_subset_greedy(
                subnodes, opts
            )
        )
        self.assertEqual(expected, result)
    def test_find_independent_subset_greedy_fuse(self):
        # 定义测试方法：测试贪婪查找独立子集并进行融合

        # 创建示例图形并返回图形及其查找表
        g, lookup = self.build_graph(
            {
                "n0": (),
                "n1": (),
                "n2": ("n0",),
                "n3": ("n1",),
                "n4": ("n2",),
                "n5": (),
            }
        )

        # 定义选项字典
        opts = {
            "min_fuse_set_size": 0,
            "max_fuse_set_size": 100,
        }

        # 定义子节点列表
        subnodes = ["n2", "n3", "n4", "n0", "n1", "n5"]
        # 将节点名称转换为图中节点对象的列表
        subnodes = [lookup[n] for n in subnodes]

        # 调用贪婪算法查找独立子集的函数
        i = torch._inductor.fx_passes.group_batch_fusion.find_independent_subset_greedy(
            subnodes, opts
        )

        # 断言第一个迭代结果
        self.assertEqual(next(i), [lookup[n] for n in ["n2", "n3", "n5"]])

        # 融合节点 n2 和 n3，使得 n4 现在依赖于 n1
        args = tuple(lookup[n] for n in ["n0", "n1"])
        # 创建一个新的节点作为融合结果，并更新节点引用
        fused = g.create_node("placeholder", "target", name="n2+n3", args=args)
        lookup["n2"].replace_all_uses_with(fused)
        # 删除原始节点 n2
        g.erase_node(lookup["n2"])
        lookup["n3"].replace_all_uses_with(fused)
        # 删除原始节点 n3
        g.erase_node(lookup["n3"])

        # 断言第二个迭代结果
        self.assertEqual(next(i), [lookup[n] for n in ["n4"]])

        # 断言第三个迭代结果
        self.assertEqual(next(i), [lookup[n] for n in ["n0", "n1"]])

        # 断言迭代结束时会抛出 StopIteration 异常
        self.assertRaises(StopIteration, lambda: next(i))
# 如果当前脚本被作为主程序执行（而不是被导入到其他模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```