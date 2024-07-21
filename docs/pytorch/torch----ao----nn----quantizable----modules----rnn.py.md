# `.\pytorch\torch\ao\nn\quantizable\modules\rnn.py`

```
# mypy: allow-untyped-defs
# 引入 numbers 模块，用于处理数字类型
import numbers
# 引入 Optional 和 Tuple 类型提示
from typing import Optional, Tuple
# 引入警告模块
import warnings

# 引入 PyTorch 库
import torch
# 从 torch 模块中引入 Tensor 类型
from torch import Tensor

"""
We will recreate all the RNN modules as we require the modules to be decomposed
into its building blocks to be able to observe.
"""

# 定义公开的模块列表
__all__ = [
    "LSTMCell",
    "LSTM"
]

# LSTMCell 类的定义，继承自 torch.nn.Module
class LSTMCell(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM) cell.

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTMCell`

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTMCell(10, 20)
        >>> input = torch.randn(6, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    # _FLOAT_MODULE 类属性，指向 torch.nn.LSTMCell 类
    _FLOAT_MODULE = torch.nn.LSTMCell

    # LSTMCell 的初始化方法
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        # 用于工厂方法的关键字参数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类初始化方法
        super().__init__()
        # 设置输入维度和隐藏状态维度
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias

        # 初始化输入门和隐藏状态门的线性层
        self.igates = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=bias, **factory_kwargs)
        self.hgates = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=bias, **factory_kwargs)
        
        # 定义量化操作的功能组件
        self.gates = torch.ao.nn.quantized.FloatFunctional()

        # 初始化各门的激活函数
        self.input_gate = torch.nn.Sigmoid()
        self.forget_gate = torch.nn.Sigmoid()
        self.cell_gate = torch.nn.Tanh()
        self.output_gate = torch.nn.Sigmoid()

        # 定义量化操作的功能组件
        self.fgate_cx = torch.ao.nn.quantized.FloatFunctional()
        self.igate_cgate = torch.ao.nn.quantized.FloatFunctional()
        self.fgate_cx_igate_cgate = torch.ao.nn.quantized.FloatFunctional()

        # 定义量化操作的功能组件
        self.ogate_cy = torch.ao.nn.quantized.FloatFunctional()

        # 初始化隐藏状态的量化参数和类型
        self.initial_hidden_state_qparams: Tuple[float, int] = (1.0, 0)
        # 初始化细胞状态的量化参数和类型
        self.initial_cell_state_qparams: Tuple[float, int] = (1.0, 0)
        # 隐藏状态的数据类型为无符号 8 位整数
        self.hidden_state_dtype: torch.dtype = torch.quint8
        # 细胞状态的数据类型为无符号 8 位整数
        self.cell_state_dtype: torch.dtype = torch.quint8
    # 前向传播函数，计算 LSTM 单元的输出和隐藏状态
    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        # 如果隐藏状态为空或其中的任一部分为空，则初始化隐藏状态
        if hidden is None or hidden[0] is None or hidden[1] is None:
            hidden = self.initialize_hidden(x.shape[0], x.is_quantized)
        hx, cx = hidden

        # 计算输入门、遗忘门和输出门
        igates = self.igates(x)
        hgates = self.hgates(hx)
        gates = self.gates.add(igates, hgates)

        # 将门控制器的输出切分成四个部分：输入门、遗忘门、细胞状态门、输出门
        input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        # 对每个门应用相应的激活函数
        input_gate = self.input_gate(input_gate)
        forget_gate = self.forget_gate(forget_gate)
        cell_gate = self.cell_gate(cell_gate)
        out_gate = self.output_gate(out_gate)

        # 计算细胞状态
        fgate_cx = self.fgate_cx.mul(forget_gate, cx)
        igate_cgate = self.igate_cgate.mul(input_gate, cell_gate)
        fgate_cx_igate_cgate = self.fgate_cx_igate_cgate.add(fgate_cx, igate_cgate)
        cy = fgate_cx_igate_cgate

        # 对细胞状态应用 tanh 函数
        # TODO: 将此 tanh 函数作为模块的成员，以便可以配置其量化参数
        tanh_cy = torch.tanh(cy)
        hy = self.ogate_cy.mul(out_gate, tanh_cy)
        return hy, cy

    # 初始化隐藏状态
    def initialize_hidden(self, batch_size: int, is_quantized: bool = False) -> Tuple[Tensor, Tensor]:
        h, c = torch.zeros((batch_size, self.hidden_size)), torch.zeros((batch_size, self.hidden_size))
        if is_quantized:
            # 如果量化标志为真，则使用量化参数初始化隐藏状态
            (h_scale, h_zp) = self.initial_hidden_state_qparams
            (c_scale, c_zp) = self.initial_cell_state_qparams
            h = torch.quantize_per_tensor(h, scale=h_scale, zero_point=h_zp, dtype=self.hidden_state_dtype)
            c = torch.quantize_per_tensor(c, scale=c_scale, zero_point=c_zp, dtype=self.cell_state_dtype)
        return h, c

    # 获取模块名称
    def _get_name(self):
        return 'QuantizableLSTMCell'

    @classmethod
    # 从参数中创建一个新的 LSTM 单元
    def from_params(cls, wi, wh, bi=None, bh=None):
        """Uses the weights and biases to create a new LSTM cell.

        Args:
            wi, wh: Weights for the input and hidden layers
            bi, bh: Biases for the input and hidden layers
        """
        assert (bi is None) == (bh is None)  # 断言：bi 和 bh 要么都为 None，要么都有值
        input_size = wi.shape[1]
        hidden_size = wh.shape[1]
        # 创建一个新的 LSTM 单元对象
        cell = cls(input_dim=input_size, hidden_dim=hidden_size,
                   bias=(bi is not None))
        cell.igates.weight = torch.nn.Parameter(wi)
        if bi is not None:
            cell.igates.bias = torch.nn.Parameter(bi)
        cell.hgates.weight = torch.nn.Parameter(wh)
        if bh is not None:
            cell.hgates.bias = torch.nn.Parameter(bh)
        return cell
    # 类方法，用于从浮点数模型转换为量化模型
    def from_float(cls, other, use_precomputed_fake_quant=False):
        # 断言，确保 other 是与当前类的浮点数模块相同类型
        assert type(other) == cls._FLOAT_MODULE
        # 断言，确保 other 模块具有 'qconfig' 属性
        assert hasattr(other, 'qconfig'), "The float module must have 'qconfig'"
        # 根据 other 模块的参数创建一个量化模型实例 observed
        observed = cls.from_params(other.weight_ih, other.weight_hh,
                                   other.bias_ih, other.bias_hh)
        # 将 observed 的 qconfig 设置为与 other 相同的 qconfig
        observed.qconfig = other.qconfig
        # 将 observed 内部的 igates 和 hgates 的 qconfig 设置为与 other 相同的 qconfig
        observed.igates.qconfig = other.qconfig
        observed.hgates.qconfig = other.qconfig
        # 返回转换后的量化模型实例 observed
        return observed
# 定义了一个单向 LSTM 层
class _LSTMSingleLayer(torch.nn.Module):
    r"""A single one-directional LSTM layer.

    The difference between a layer and a cell is that the layer can process a
    sequence, while the cell only expects an instantaneous value.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        # 初始化函数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 LSTM 单元作为该层的核心单元
        self.cell = LSTMCell(input_dim, hidden_dim, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # 前向传播函数，用于处理输入序列 x
        result = []
        seq_len = x.shape[0]
        # 遍历输入序列的每个时间步
        for i in range(seq_len):
            # 对当前时间步的输入和隐藏状态进行 LSTM 计算
            hidden = self.cell(x[i], hidden)
            # 将计算结果添加到结果列表中
            result.append(hidden[0])  # type: ignore[index]
        # 将结果列表转换为张量
        result_tensor = torch.stack(result, 0)
        # 返回结果张量和最终的隐藏状态
        return result_tensor, hidden

    @classmethod
    def from_params(cls, *args, **kwargs):
        # 从参数创建一个单向 LSTM 层的实例方法
        cell = LSTMCell.from_params(*args, **kwargs)
        # 根据 LSTM 单元的参数创建一个单向 LSTM 层
        layer = cls(cell.input_size, cell.hidden_size, cell.bias)
        # 将创建的 LSTM 单元赋值给层的核心单元
        layer.cell = cell
        # 返回创建的单向 LSTM 层
        return layer


# 定义了一个双向 LSTM 层
class _LSTMLayer(torch.nn.Module):
    r"""A single bi-directional LSTM layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True,
                 batch_first: bool = False, bidirectional: bool = False,
                 device=None, dtype=None) -> None:
        # 初始化函数
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类的初始化函数
        super().__init__()
        # 是否按批量为第一维的顺序处理输入数据
        self.batch_first = batch_first
        # 是否使用双向 LSTM
        self.bidirectional = bidirectional
        # 创建一个单向 LSTM 层作为正向层
        self.layer_fw = _LSTMSingleLayer(input_dim, hidden_dim, bias=bias, **factory_kwargs)
        # 如果是双向 LSTM，则创建一个单向 LSTM 层作为反向层
        if self.bidirectional:
            self.layer_bw = _LSTMSingleLayer(input_dim, hidden_dim, bias=bias, **factory_kwargs)
    # 定义前向传播方法，接受输入张量 x 和可选的隐藏状态 hidden
    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # 如果模型设置为 batch_first，则将输入张量 x 的维度进行转置
        if self.batch_first:
            x = x.transpose(0, 1)
        # 初始化前向传播的隐藏状态 hx_fw 和细胞状态 cx_fw
        if hidden is None:
            hx_fw, cx_fw = (None, None)
        else:
            hx_fw, cx_fw = hidden
        # 初始化反向传播的隐藏状态 hidden_bw
        hidden_bw: Optional[Tuple[Tensor, Tensor]] = None
        # 如果模型是双向的
        if self.bidirectional:
            # 设置反向传播的初始隐藏状态 hx_bw 和细胞状态 cx_bw
            if hx_fw is None:
                hx_bw = None
            else:
                hx_bw = hx_fw[1]
                hx_fw = hx_fw[0]
            if cx_fw is None:
                cx_bw = None
            else:
                cx_bw = cx_fw[1]
                cx_fw = cx_fw[0]
            # 如果反向传播的隐藏状态和细胞状态均已设置，则创建 hidden_bw
            if hx_bw is not None and cx_bw is not None:
                hidden_bw = hx_bw, cx_bw
        # 如果前向传播的隐藏状态和细胞状态均未设置，则设置 hidden_fw 为 None
        if hx_fw is None and cx_fw is None:
            hidden_fw = None
        else:
            # 否则，从 torch.jit._unwrap_optional 函数中获取前向传播的隐藏状态和细胞状态
            hidden_fw = torch.jit._unwrap_optional(hx_fw), torch.jit._unwrap_optional(cx_fw)
        # 对输入 x 进行前向传播计算，得到 result_fw 和更新后的隐藏状态 hidden_fw
        result_fw, hidden_fw = self.layer_fw(x, hidden_fw)

        # 如果模型具有反向层 self.layer_bw，并且是双向模型
        if hasattr(self, 'layer_bw') and self.bidirectional:
            # 将输入 x 反转，作为反向层的输入 x_reversed
            x_reversed = x.flip(0)
            # 对反向层进行计算，得到 result_bw 和更新后的隐藏状态 hidden_bw
            result_bw, hidden_bw = self.layer_bw(x_reversed, hidden_bw)
            # 将 result_bw 反转回来，保持与 result_fw 同样的顺序
            result_bw = result_bw.flip(0)

            # 将前向传播和反向传播的结果 result_fw 和 result_bw 沿着最后一个维度拼接起来
            result = torch.cat([result_fw, result_bw], result_fw.dim() - 1)
            # 如果前向和反向的隐藏状态均未设置，则 h 和 c 均设置为 None
            if hidden_fw is None and hidden_bw is None:
                h = None
                c = None
            # 如果只有前向的隐藏状态未设置，则从 hidden_bw 解包得到 h 和 c
            elif hidden_fw is None:
                (h, c) = torch.jit._unwrap_optional(hidden_bw)
            # 如果只有反向的隐藏状态未设置，则从 hidden_fw 解包得到 h 和 c
            elif hidden_bw is None:
                (h, c) = torch.jit._unwrap_optional(hidden_fw)
            else:
                # 否则，将前向和反向的隐藏状态分别堆叠起来，形成 h 和 c
                h = torch.stack([hidden_fw[0], hidden_bw[0]], 0)  # type: ignore[list-item]
                c = torch.stack([hidden_fw[1], hidden_bw[1]], 0)  # type: ignore[list-item]
        else:
            # 如果模型不是双向的，则结果 result 仅为前向传播的结果 result_fw
            result = result_fw
            # 从 torch.jit._unwrap_optional 函数中获取前向传播的隐藏状态和细胞状态
            h, c = torch.jit._unwrap_optional(hidden_fw)  # type: ignore[assignment]

        # 如果模型设置为 batch_first，则将结果 result 的维度进行转置
        if self.batch_first:
            result.transpose_(0, 1)

        # 返回前向传播的结果 result 和最终的隐藏状态 (h, c)
        return result, (h, c)

    @classmethod
    def from_float(cls, other, layer_idx=0, qconfig=None, **kwargs):
        r"""
        There is no FP equivalent of this class. This function is here just to
        mimic the behavior of the `prepare` within the `torch.ao.quantization`
        flow.
        """
        # 断言输入参数 `other` 具有 `qconfig` 属性，或者参数 `qconfig` 不为 None
        assert hasattr(other, 'qconfig') or (qconfig is not None)

        # 获取或者从 `kwargs` 中提取输入大小（input_size）、隐藏层大小（hidden_size）、偏置（bias）、
        # 是否批量优先（batch_first）、是否双向（bidirectional）等参数
        input_size = kwargs.get('input_size', other.input_size)
        hidden_size = kwargs.get('hidden_size', other.hidden_size)
        bias = kwargs.get('bias', other.bias)
        batch_first = kwargs.get('batch_first', other.batch_first)
        bidirectional = kwargs.get('bidirectional', other.bidirectional)

        # 创建一个新的当前类对象 `layer`，使用提取的参数进行初始化
        layer = cls(input_size, hidden_size, bias, batch_first, bidirectional)
        
        # 设置 `layer` 对象的 `qconfig` 属性，从 `other` 对象获取，如果不存在则使用传入的 `qconfig`
        layer.qconfig = getattr(other, 'qconfig', qconfig)
        
        # 从 `other` 对象中获取与当前层索引相关的权重（weight_ih_l{layer_idx}）、隐藏层权重（weight_hh_l{layer_idx}）、
        # 输入偏置（bias_ih_l{layer_idx}）、隐藏层偏置（bias_hh_l{layer_idx}）
        wi = getattr(other, f'weight_ih_l{layer_idx}')
        wh = getattr(other, f'weight_hh_l{layer_idx}')
        bi = getattr(other, f'bias_ih_l{layer_idx}', None)
        bh = getattr(other, f'bias_hh_l{layer_idx}', None)

        # 使用提取的参数初始化 `layer` 对象的前向传播 LSTM 单层对象 `layer_fw`
        layer.layer_fw = _LSTMSingleLayer.from_params(wi, wh, bi, bh)

        # 如果 `other` 对象是双向的，则处理反向传播的 LSTM 单层对象 `layer_bw`
        if other.bidirectional:
            wi = getattr(other, f'weight_ih_l{layer_idx}_reverse')
            wh = getattr(other, f'weight_hh_l{layer_idx}_reverse')
            bi = getattr(other, f'bias_ih_l{layer_idx}_reverse', None)
            bh = getattr(other, f'bias_hh_l{layer_idx}_reverse', None)
            # 使用提取的参数初始化 `layer` 对象的反向传播 LSTM 单层对象 `layer_bw`
            layer.layer_bw = _LSTMSingleLayer.from_params(wi, wh, bi, bh)

        # 返回创建并初始化好的 `layer` 对象
        return layer
class LSTM(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples below.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
        >>> # To get the weights:
        >>> # xdoctest: +SKIP
        >>> print(rnn.layers[0].weight_ih)
        tensor([[...]])
        >>> print(rnn.layers[0].weight_hh)
        AssertionError: There is no reverse path in the non-bidirectional layer
    """
    _FLOAT_MODULE = torch.nn.LSTM



# 定义了一个自定义的 LSTM 模型，继承自 `torch.nn.Module`
class LSTM(torch.nn.Module):
    r"""A quantizable long short-term memory (LSTM).

    描述了一个可量化的长短期记忆（LSTM）模型。

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    详细描述了该类及其参数类型，请参考 :class:`~torch.nn.LSTM`。

    Attributes:
        layers : instances of the `_LSTMLayer`

    属性:
        layers : `_LSTMLayer` 的实例集合

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples below.

    注意:
        要访问权重和偏置，需要逐层访问。请参考下面的示例。

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
        >>> # To get the weights:
        >>> # xdoctest: +SKIP
        >>> print(rnn.layers[0].weight_ih)
        tensor([[...]])
        >>> print(rnn.layers[0].weight_hh)
        AssertionError: There is no reverse path in the non-bidirectional layer
    """
    _FLOAT_MODULE = torch.nn.LSTM


这段代码定义了一个自定义的 LSTM 模型，继承自 PyTorch 的 `torch.nn.Module` 类。
    # 初始化函数，用于创建一个多层 LSTM 模型
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0.,
                 bidirectional: bool = False,
                 device=None, dtype=None) -> None:
        # 定义工厂参数字典，用于传递设备和数据类型信息
        factory_kwargs = {'device': device, 'dtype': dtype}
        # 调用父类初始化方法
        super().__init__()
        # 设置模型的输入大小
        self.input_size = input_size
        # 设置模型的隐藏层大小
        self.hidden_size = hidden_size
        # 设置模型的层数
        self.num_layers = num_layers
        # 设置是否包含偏置项
        self.bias = bias
        # 设置是否按照 batch_first 的顺序处理输入数据
        self.batch_first = batch_first
        # 设置 dropout 概率，如果未设置则为 0
        self.dropout = float(dropout)
        # 设置是否为双向 LSTM
        self.bidirectional = bidirectional
        # 默认为评估模式，如果需要训练，需要显式设置为训练模式
        self.training = False  # Default to eval mode. If we want to train, we will explicitly set to training.
        # 根据是否双向 LSTM 计算方向数量
        num_directions = 2 if bidirectional else 1

        # 检查 dropout 参数的合法性
        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        
        # 如果设置了 dropout，给出警告信息，因为 quantizable LSTM 不支持 dropout
        if dropout > 0:
            warnings.warn("dropout option for quantizable LSTM is ignored. "
                          "If you are training, please, use nn.LSTM version "
                          "followed by `prepare` step.")
            # 如果层数为 1，提示不支持单层非零 dropout
            if num_layers == 1:
                warnings.warn("dropout option adds dropout after all but last "
                              "recurrent layer, so non-zero dropout expects "
                              f"num_layers greater than 1, but got dropout={dropout} "
                              f"and num_layers={num_layers}")

        # 初始化 LSTM 层列表，包含多个 _LSTMLayer 实例
        layers = [_LSTMLayer(self.input_size, self.hidden_size,
                             self.bias, batch_first=False,
                             bidirectional=self.bidirectional, **factory_kwargs)]
        # 添加额外的 LSTM 层到列表中
        for layer in range(1, num_layers):
            layers.append(_LSTMLayer(self.hidden_size, self.hidden_size,
                                     self.bias, batch_first=False,
                                     bidirectional=self.bidirectional,
                                     **factory_kwargs))
        # 将 LSTM 层列表转换为 nn.ModuleList 类型，并赋值给 self.layers
        self.layers = torch.nn.ModuleList(layers)
    # 定义前向传播函数，接受输入张量 x 和隐藏状态 hidden（可选）
    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None):
        # 如果设置了 batch_first 标志，则将输入张量 x 的维度交换，使其第一维度为批次大小
        if self.batch_first:
            x = x.transpose(0, 1)

        # 获取当前批次的最大大小
        max_batch_size = x.size(1)
        # 确定隐藏状态的方向数（单向或双向）
        num_directions = 2 if self.bidirectional else 1
        
        # 如果隐藏状态未提供，则初始化为零张量
        if hidden is None:
            # 创建全零张量用于隐藏状态，形状为 (num_directions, max_batch_size, self.hidden_size)
            zeros = torch.zeros(num_directions, max_batch_size,
                                self.hidden_size, dtype=torch.float,
                                device=x.device)
            # 如果输入张量 x 是量化的，则对零张量进行量化处理
            if x.is_quantized:
                zeros = torch.quantize_per_tensor(zeros, scale=1.0,
                                                  zero_point=0, dtype=x.dtype)
            # 将零张量形状调整为 (max_batch_size, self.hidden_size)
            zeros.squeeze_(0)
            # 初始化隐藏状态列表 hxcx，包含 num_layers 个元组 (zeros, zeros)
            hxcx = [(zeros, zeros) for _ in range(self.num_layers)]
        else:
            # 如果提供了隐藏状态，解包隐藏状态
            hidden_non_opt = torch.jit._unwrap_optional(hidden)
            # 如果隐藏状态的第一个元素是张量，则重新调整形状
            if isinstance(hidden_non_opt[0], Tensor):
                hx = hidden_non_opt[0].reshape(self.num_layers, num_directions,
                                               max_batch_size,
                                               self.hidden_size)
                cx = hidden_non_opt[1].reshape(self.num_layers, num_directions,
                                               max_batch_size,
                                               self.hidden_size)
                # 重新组合隐藏状态列表 hxcx，包含 num_layers 个元组 (hx[idx].squeeze(0), cx[idx].squeeze(0))
                hxcx = [(hx[idx].squeeze(0), cx[idx].squeeze(0)) for idx in range(self.num_layers)]
            else:
                # 否则，直接使用提供的隐藏状态
                hxcx = hidden_non_opt

        # 初始化列表用于保存每个层的输出隐藏状态列表
        hx_list = []
        cx_list = []
        # 对每一层循环处理
        for idx, layer in enumerate(self.layers):
            # 调用每一层的处理方法，更新输入张量 x 和隐藏状态 hxcx[idx]
            x, (h, c) = layer(x, hxcx[idx])
            # 将每一层的输出隐藏状态 h 和 c 解包并保存到列表中
            hx_list.append(torch.jit._unwrap_optional(h))
            cx_list.append(torch.jit._unwrap_optional(c))
        
        # 将隐藏状态列表转换为张量
        hx_tensor = torch.stack(hx_list)
        cx_tensor = torch.stack(cx_list)

        # 将双向情况下的隐藏状态张量形状调整为 (-1, hx_tensor.shape[-2], hx_tensor.shape[-1])
        hx_tensor = hx_tensor.reshape(-1, hx_tensor.shape[-2], hx_tensor.shape[-1])
        cx_tensor = cx_tensor.reshape(-1, cx_tensor.shape[-2], cx_tensor.shape[-1])

        # 如果设置了 batch_first 标志，则将输出张量 x 的维度重新交换回原始形式
        if self.batch_first:
            x = x.transpose(0, 1)

        # 返回输出张量 x 和调整后的隐藏状态张量 (hx_tensor, cx_tensor)
        return x, (hx_tensor, cx_tensor)

    # 返回模型名称的私有方法
    def _get_name(self):
        return 'QuantizableLSTM'

    # 类方法声明的位置
    @classmethod
    # 从浮点数模型转换为观察模型的类方法
    def from_float(cls, other, qconfig=None):
        # 断言输入参数other是cls._FLOAT_MODULE的实例
        assert isinstance(other, cls._FLOAT_MODULE)
        # 断言other具有'qconfig'属性或者qconfig参数不为空
        assert (hasattr(other, 'qconfig') or qconfig)
        # 创建一个观察模型，复制其他参数，并保留batch_first和dropout设置
        observed = cls(other.input_size, other.hidden_size, other.num_layers,
                       other.bias, other.batch_first, other.dropout,
                       other.bidirectional)
        # 设置观察模型的qconfig属性，如果other有'qconfig'属性则继承，否则使用给定的qconfig
        observed.qconfig = getattr(other, 'qconfig', qconfig)
        # 遍历other模型的每一层，通过from_float方法将其转换为观察模型的对应层，并存储在observed模型的layers中
        for idx in range(other.num_layers):
            observed.layers[idx] = _LSTMLayer.from_float(other, idx, qconfig,
                                                         batch_first=False)

        # 准备模型
        if other.training:
            # 如果other模型在训练状态，则设置observed为训练状态并准备量化训练模式
            observed.train()
            observed = torch.ao.quantization.prepare_qat(observed, inplace=True)
        else:
            # 如果other模型不在训练状态，则设置observed为评估状态并准备量化模型
            observed.eval()
            observed = torch.ao.quantization.prepare(observed, inplace=True)
        # 返回准备好的观察模型
        return observed

    @classmethod
    def from_observed(cls, other):
        # 整个流程是从浮点数模型 -> 观察模型 -> 量化模型
        # 本类方法仅支持从浮点数模型 -> 观察模型的转换
        raise NotImplementedError("It looks like you are trying to convert a "
                                  "non-quantizable LSTM module. Please, see "
                                  "the examples on quantizable LSTMs.")
```