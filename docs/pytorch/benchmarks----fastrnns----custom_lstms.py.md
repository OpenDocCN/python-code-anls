# `.\pytorch\benchmarks\fastrnns\custom_lstms.py`

```
# 导入必要的模块和库
import numbers                     # 导入 numbers 模块，用于处理数字类型
import warnings                    # 导入 warnings 模块，用于警告处理
from collections import namedtuple  # 导入 namedtuple 类型，用于创建命名元组
from typing import List, Tuple      # 导入类型提示，指定函数参数和返回值的类型

import torch                       # 导入 PyTorch 库
import torch.jit as jit            # 导入 PyTorch 的 jit 模块，用于脚本化
import torch.nn as nn              # 导入 PyTorch 的神经网络模块
from torch import Tensor           # 导入 Tensor 类型
from torch.nn import Parameter     # 导入 Parameter 类型

"""
一些辅助类，用于编写自定义的 TorchScript LSTM。

目标：
- 类易于阅读、使用和扩展
- 自定义 LSTM 的性能接近融合内核级别的速度。

关于我们可以添加的一些特性来清理下面代码的备注：
- 支持使用 nn.ModuleList 进行枚举：
  https://github.com/pytorch/pytorch/issues/14471
- 支持使用列表进行枚举/zip操作：
  https://github.com/pytorch/pytorch/issues/15952
- 支持重写类方法：
  https://github.com/pytorch/pytorch/issues/10733
- 支持传递用户定义的命名元组类型以提升可读性
- 支持使用 range 进行切片。它可以轻松地反转列表。
  https://github.com/pytorch/pytorch/issues/10774
- 多行类型注释。List[List[Tuple[Tensor,Tensor]]] 很冗长
  https://github.com/pytorch/pytorch/pull/14922
"""


def script_lstm(
    input_size,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
):
    """返回一个脚本化的模块，模拟了 PyTorch 原生的 LSTM。"""

    # 以下功能尚未实现。
    assert bias  # 确保偏置启用
    assert not batch_first  # 确保不支持 batch_first 模式

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    # 返回相应类型的堆叠 LSTM 模型
    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[LSTMCell, input_size, hidden_size],
        other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size],
    )


def script_lnlstm(
    input_size,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
    decompose_layernorm=False,
):
    """返回一个脚本化的模块，模拟了 PyTorch 原生的 LSTM。"""

    # 以下功能尚未实现。
    assert bias  # 确保偏置启用
    assert not batch_first  # 确保不支持 batch_first 模式
    assert not dropout  # 确保不支持 dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    # 返回相应类型的堆叠 LSTM 模型
    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[
            LayerNormLSTMCell,
            input_size,
            hidden_size,
            decompose_layernorm,
        ],
        other_layer_args=[
            LayerNormLSTMCell,
            hidden_size * dirs,
            hidden_size,
            decompose_layernorm,
        ],
    )


# 定义命名元组 LSTMState，包含两个字段 hx 和 cx
LSTMState = namedtuple("LSTMState", ["hx", "cx"])


def reverse(lst: List[Tensor]) -> List[Tensor]:
    """反转输入列表中的张量，并返回结果列表。"""
    return lst[::-1]


class LSTMCell(jit.ScriptModule):
    """自定义的 TorchScript LSTM 单元类，继承自 jit.ScriptModule。"""
    # 初始化方法，用于设置LSTM的输入大小和隐藏层大小
    def __init__(self, input_size, hidden_size):
        # 调用父类（torch.nn.Module）的初始化方法
        super().__init__()
        # 设置输入大小
        self.input_size = input_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 初始化输入到隐藏层的权重矩阵，形状为 (4 * hidden_size, input_size)
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        # 初始化隐藏层到隐藏层的权重矩阵，形状为 (4 * hidden_size, hidden_size)
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # 初始化输入到隐藏层的偏置向量，形状为 (4 * hidden_size)
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        # 初始化隐藏层到隐藏层的偏置向量，形状为 (4 * hidden_size)
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    # 使用JIT编译的前向传播方法，接收输入张量和状态元组作为参数，返回输出张量和更新后的状态元组
    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # 从状态元组中解包出隐藏状态（hx）和细胞状态（cx）
        hx, cx = state
        # 计算 LSTM 的四个门控信号的总输入
        gates = (
            torch.mm(input, self.weight_ih.t())  # 输入与输入到隐藏层权重的乘积
            + self.bias_ih  # 加上输入到隐藏层的偏置
            + torch.mm(hx, self.weight_hh.t())  # 加上隐藏状态与隐藏层到隐藏层权重的乘积
            + self.bias_hh  # 加上隐藏层到隐藏层的偏置
        )
        # 将总输入向量分割成四部分，分别代表输入门、遗忘门、细胞更新和输出门
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # 对每个门控信号应用激活函数
        ingate = torch.sigmoid(ingate)  # 输入门
        forgetgate = torch.sigmoid(forgetgate)  # 遗忘门
        cellgate = torch.tanh(cellgate)  # 细胞更新
        outgate = torch.sigmoid(outgate)  # 输出门

        # 计算新的细胞状态和隐藏状态
        cy = (forgetgate * cx) + (ingate * cellgate)  # 细胞状态更新
        hy = outgate * torch.tanh(cy)  # 隐藏状态更新

        # 返回隐藏状态和更新后的状态元组
        return hy, (hy, cy)
class LayerNorm(jit.ScriptModule):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1  # 断言：确保normalized_shape是长度为1的元组或列表

        self.weight = Parameter(torch.ones(normalized_shape))  # 初始化权重参数为1
        self.bias = Parameter(torch.zeros(normalized_shape))  # 初始化偏置参数为0
        self.normalized_shape = normalized_shape  # 记录归一化层的形状信息

    @jit.script_method
    def compute_layernorm_stats(self, input):
        mu = input.mean(-1, keepdim=True)  # 计算输入在最后一个维度上的均值
        sigma = input.std(-1, keepdim=True, unbiased=False)  # 计算输入在最后一个维度上的标准差
        return mu, sigma

    @jit.script_method
    def forward(self, input):
        mu, sigma = self.compute_layernorm_stats(input)  # 计算输入的均值和标准差
        return (input - mu) / sigma * self.weight + self.bias  # 应用 LayerNorm 公式进行归一化


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, decompose_layernorm=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))  # 初始化输入到隐藏状态权重矩阵
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))  # 初始化隐藏到隐藏状态权重矩阵
        # The layernorms provide learnable biases

        if decompose_layernorm:
            ln = LayerNorm  # 如果解耦归一化，使用自定义的 LayerNorm 类
        else:
            ln = nn.LayerNorm  # 否则使用 PyTorch 的 LayerNorm 类

        self.layernorm_i = ln(4 * hidden_size)  # 初始化输入门的归一化层
        self.layernorm_h = ln(4 * hidden_size)  # 初始化隐藏门的归一化层
        self.layernorm_c = ln(hidden_size)  # 初始化细胞状态的归一化层

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))  # 输入门的归一化
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))  # 隐藏门的归一化
        gates = igates + hgates  # 合并门的输出
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # 将门的输出分割成四部分

        ingate = torch.sigmoid(ingate)  # 输入门的 sigmoid 激活
        forgetgate = torch.sigmoid(forgetgate)  # 遗忘门的 sigmoid 激活
        cellgate = torch.tanh(cellgate)  # 细胞状态的 tanh 激活
        outgate = torch.sigmoid(outgate)  # 输出门的 sigmoid 激活

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))  # 更新细胞状态
        hy = outgate * torch.tanh(cy)  # 更新隐藏状态

        return hy, (hy, cy)  # 返回隐藏状态和细胞状态元组


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)  # 初始化 LSTM 单元

    @jit.script_method
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)  # 沿着序列解绑输入
        outputs = torch.jit.annotate(List[Tensor], [])  # 初始化输出列表
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)  # 执行 LSTM 单元的前向传播
            outputs += [out]  # 将输出添加到列表
        return torch.stack(outputs), state  # 返回堆叠后的输出和最终状态


class ReverseLSTMLayer(jit.ScriptModule):
    # 这部分代码未提供，需要继续补充完整
    # 初始化方法，用于创建对象实例
    def __init__(self, cell, *cell_args):
        # 调用父类的初始化方法
        super().__init__()
        # 根据传入的参数创建一个cell对象，并保存在实例变量self.cell中
        self.cell = cell(*cell_args)

    # 使用JIT编译的方法装饰器，表示该方法将被即时编译优化
    @jit.script_method
    # 前向传播方法，接收输入张量input和状态元组state，返回输出张量和新的状态元组
    def forward(
        self, input: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # 对输入张量按第一个维度（通常是时间步维度）解绑，并反转顺序
        inputs = reverse(input.unbind(0))
        # 初始化一个空列表outputs，用于存储每个时间步的输出张量
        outputs = jit.annotate(List[Tensor], [])
        # 遍历反转后的输入张量列表
        for i in range(len(inputs)):
            # 对当前时间步的输入和状态进行cell处理，返回输出和新的状态
            out, state = self.cell(inputs[i], state)
            # 将当前时间步的输出添加到outputs列表中
            outputs += [out]
        # 将所有输出张量按时间步顺序堆叠成一个张量，并返回以及最终的状态元组
        return torch.stack(reverse(outputs)), state
class BidirLSTMLayer(jit.ScriptModule):
    __constants__ = ["directions"]

    def __init__(self, cell, *cell_args):
        super().__init__()
        # 定义正向和反向的 LSTM 层，并封装成模块列表
        self.directions = nn.ModuleList(
            [
                LSTMLayer(cell, *cell_args),         # 正向 LSTM 层
                ReverseLSTMLayer(cell, *cell_args),  # 反向 LSTM 层
            ]
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # 初始化输出和状态列表
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # 遍历正向和反向 LSTM 层
        i = 0
        for direction in self.directions:
            state = states[i]
            # 调用当前方向的 LSTM 层进行前向传播
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        # 将所有方向的输出张量连接在一起，并返回输出和状态列表
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    # 初始化堆叠 LSTM 层
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class StackedLSTM(jit.ScriptModule):
    __constants__ = ["layers"]  # 必要用于迭代 self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        # 初始化堆叠 LSTM 层的列表
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # 初始化输出状态列表和输出张量
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # 遍历所有堆叠的 LSTM 层
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            # 调用当前 LSTM 层进行前向传播
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        # 返回最终的输出和状态列表
        return output, output_states


# 与 StackedLSTM 不同之处在于其 forward 方法接受 List[List[Tuple[Tensor, Tensor]]]
# 详细参见说明，无法通过子类化 StackedLSTM 来实现，因为不支持覆盖脚本方法。
class StackedLSTM2(jit.ScriptModule):
    __constants__ = ["layers"]  # 必要用于迭代 self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        # 初始化堆叠 LSTM2 层的列表
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, input: Tensor, states: List[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]:
        # 此处缺少 forward 方法的实现，需要根据具体需求进行补充
        pass
        # 定义函数签名和返回类型注解，该函数接受一个 Tensor 作为输入和一个 LSTMState 对象的列表作为状态输入，返回一个包含 Tensor 和 LSTMState 列表的元组。
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])  # 初始化 output_states 变量为一个空的列表，用于存储每个 RNN 层的输出状态
        output = input  # 将输入参数赋值给 output 变量，作为 RNN 层的初始输入

        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        # 使用 enumerate 函数迭代 self.layers 中的每个 RNN 层，i 用于追踪当前状态列表的索引
        i = 0
        for rnn_layer in self.layers:
            state = states[i]  # 获取当前 RNN 层对应的状态
            output, out_state = rnn_layer(output, state)  # 调用当前 RNN 层的前向传播方法，计算输出和新的状态
            output_states += [out_state]  # 将当前 RNN 层计算得到的状态添加到 output_states 列表中
            i += 1  # 更新索引，准备处理下一层 RNN

        # 返回最终的输出结果和所有 RNN 层的状态列表
        return output, output_states
class StackedLSTMWithDropout(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ["layers", "num_layers"]

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        # Initialize LSTM layers using a helper function
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )
        # Number of LSTM layers in the stack
        self.num_layers = num_layers

        # Issue a warning if only one layer is provided since dropout is designed for multiple layers
        if num_layers == 1:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )

        # Dropout layer with a dropout probability of 0.4 applied after each LSTM layer's output (except the last)
        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(
        self, input: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # Iterate through each LSTM layer
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            # Forward pass through the current LSTM layer
            output, out_state = rnn_layer(output, state)
            # Apply dropout to the output of all but the last LSTM layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            # Collect output states of each layer
            output_states += [out_state]
            i += 1
        return output, output_states


def flatten_states(states):
    # Flatten nested states of LSTM layers
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    # Flatten states recursively twice
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]


def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    # Generate random input
    inp = torch.randn(seq_len, batch, input_size)
    # Initialize random LSTM state
    state = LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
    # Initialize custom LSTM layer
    rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
    # Perform forward pass through the custom LSTM layer
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    # Initialize pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    # Convert custom LSTM state to match pytorch's format
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    # Ensure weights of custom LSTM match pytorch LSTM weights
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    # Perform forward pass through pytorch LSTM
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    # Assertion: ensure outputs are approximately equal
    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size, num_layers):
    # This function is incomplete and should be completed elsewhere
    # 使用 torch.randn 生成一个指定形状的随机张量作为输入数据
    inp = torch.randn(seq_len, batch, input_size)
    # 使用列表推导式生成包含多个 LSTMState 对象的列表，每个对象都有指定形状的随机张量
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    # 使用 script_lstm 函数创建一个自定义的 LSTM 模型对象 rnn
    rnn = script_lstm(input_size, hidden_size, num_layers)
    # 将输入数据 inp 和状态列表 states 输入到 rnn 模型中进行计算，得到输出和最终状态
    out, out_state = rnn(inp, states)
    # 将 rnn 模型输出的状态展平成一个自定义状态 custom_state
    custom_state = flatten_states(out_state)

    # 控制组：使用 pytorch 原生的 LSTM 模型
    # 创建一个 nn.LSTM 对象 lstm，指定输入大小、隐藏层大小和层数
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    # 将自定义状态列表 states 展平成一个用于 LSTM 模型的状态 lstm_state
    lstm_state = flatten_states(states)
    # 对每一层的参数进行比较和复制
    for layer in range(num_layers):
        # 获取自定义模型 rnn 的参数，这些参数对应于 LSTM 模型的权重和偏置
        custom_params = list(rnn.parameters())[4 * layer : 4 * (layer + 1)]
        # 遍历每一层的权重和偏置，确保它们的形状相同，并用自定义模型的参数值覆盖 LSTM 模型的参数
        for lstm_param, custom_param in zip(lstm.all_weights[layer], custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    # 将输入数据 inp 和展平后的状态 lstm_state 输入到 LSTM 模型 lstm 中进行计算，得到输出和最终状态
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    # 断言两种模型的输出之间的差异在指定的精度范围内
    assert (out - lstm_out).abs().max() < 1e-5
    # 断言自定义模型的最终状态 custom_state 与 LSTM 模型的最终状态 lstm_out_state 的差异在指定的精度范围内
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5
# 定义一个函数用于测试堆叠的双向循环神经网络（RNN）模型
def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size, num_layers):
    # 生成一个形状为 (seq_len, batch, input_size) 的随机张量作为输入
    inp = torch.randn(seq_len, batch, input_size)
    
    # 初始化多层双向 LSTMState 对象，每层有两个方向的状态
    states = [
        [
            LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
            for _ in range(2)
        ]
        for _ in range(num_layers)
    ]
    
    # 创建一个具有指定输入维度、隐藏层大小、层数和双向标志的脚本 LSTM 模型
    rnn = script_lstm(input_size, hidden_size, num_layers, bidirectional=True)
    
    # 使用输入数据和状态来运行脚本 LSTM 模型，得到输出和最终状态
    out, out_state = rnn(inp, states)
    
    # 将脚本 LSTM 模型的状态双倍扁平化，以便与标准 LSTM 模型进行比较
    custom_state = double_flatten_states(out_state)

    # 控制组：使用 PyTorch 原生的 LSTM 模型
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)
    
    # 遍历每一层的每一个方向，验证自定义 LSTM 模型的参数与标准 LSTM 模型的参数是否一致
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(rnn.parameters())[4 * index : 4 * index + 4]
            for lstm_param, custom_param in zip(lstm.all_weights[index], custom_params):
                # 断言两个模型的参数形状应该完全一致
                assert lstm_param.shape == custom_param.shape
                with torch.no_grad():
                    # 将自定义 LSTM 模型的参数复制到标准 LSTM 模型中
                    lstm_param.copy_(custom_param)
    
    # 使用标准 LSTM 模型进行前向传播，得到输出和最终状态
    lstm_out, lstm_out_state = lstm(inp, lstm_state)
    
    # 断言自定义 LSTM 模型的输出与标准 LSTM 模型的输出之间的最大绝对误差小于 1e-5
    assert (out - lstm_out).abs().max() < 1e-5
    # 断言自定义 LSTM 模型的状态的第一个部分与标准 LSTM 模型的状态的第一个部分之间的最大绝对误差小于 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    # 断言自定义 LSTM 模型的状态的第二个部分与标准 LSTM 模型的状态的第二个部分之间的最大绝对误差小于 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


# 定义一个函数用于测试带有 dropout 的堆叠 LSTM 模型
def test_script_stacked_lstm_dropout(seq_len, batch, input_size, hidden_size, num_layers):
    # 生成一个形状为 (seq_len, batch, input_size) 的随机张量作为输入
    inp = torch.randn(seq_len, batch, input_size)
    
    # 初始化多层 LSTMState 对象作为初始状态
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    
    # 创建一个带有 dropout 的脚本 LSTM 模型
    rnn = script_lstm(input_size, hidden_size, num_layers, dropout=True)
    
    # 只是一个简单的测试，运行脚本 LSTM 模型以获取输出和最终状态
    out, out_state = rnn(inp, states)


# 定义一个函数用于测试堆叠的长短时记忆网络（LNLSTM）模型
def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size, num_layers):
    # 生成一个形状为 (seq_len, batch, input_size) 的随机张量作为输入
    inp = torch.randn(seq_len, batch, input_size)
    
    # 初始化多层 LSTMState 对象作为初始状态
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    
    # 创建一个脚本 LNLSTM 模型
    rnn = script_lnlstm(input_size, hidden_size, num_layers)
    
    # 只是一个简单的测试，运行 LNLSTM 模型以获取输出和最终状态
    out, out_state = rnn(inp, states)


# 分别对不同的函数进行测试
test_script_rnn_layer(5, 2, 3, 7)
test_script_stacked_rnn(5, 2, 3, 7, 4)
test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
test_script_stacked_lnlstm(5, 2, 3, 7, 4)
```