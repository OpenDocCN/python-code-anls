# `.\pytorch\torch\ao\nn\quantized\dynamic\modules\rnn.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import numbers  # 导入 numbers 模块，用于数值类型的处理
import warnings  # 导入 warnings 模块，用于警告处理
from typing_extensions import deprecated  # 从 typing_extensions 模块导入 deprecated 装饰器

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from torch import Tensor  # noqa: F401 导入 Tensor 类型，用于类型提示
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401 导入其他类型，用于类型提示
from torch.nn.utils.rnn import PackedSequence  # 导入 PyTorch 中用于序列压缩的工具
from torch.ao.nn.quantized.modules.utils import _quantize_weight  # 从 quantized 模块导入 _quantize_weight 函数

__all__ = ['pack_weight_bias', 'PackedParameter', 'RNNBase', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell', 'LSTMCell',
           'GRUCell', "apply_permutation"]

# 对 tensor 在指定维度 dim 上应用 permutation 张量进行重排序
def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


# 装饰器函数 deprecated，用于标记 apply_permutation 函数已废弃
@deprecated(
    "`apply_permutation` is deprecated, please use `tensor.index_select(dim, permutation)` instead",
    category=FutureWarning,
)
# 对 tensor 在指定维度 dim 上应用 permutation 张量进行重排序
def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return _apply_permutation(tensor, permutation, dim)


# 将权重和偏置量化并打包成 Tensor
def pack_weight_bias(qweight, bias, dtype):

    if dtype == torch.qint8:
        # 对于每一层和每个方向，需要量化并打包权重，按照以下顺序打包参数：
        #
        #   w_ih, w_hh
        packed_weight = \
            torch.ops.quantized.linear_prepack(qweight, bias)

        return packed_weight
    else:
        # 对于每一层和每个方向，需要量化并打包权重，按照以下顺序打包参数：
        #
        #   packed_ih, packed_hh, b_ih, b_hh
        packed_weight = torch.ops.quantized.linear_prepack_fp16(
            qweight, bias)

        return packed_weight


# 定义 PackedParameter 类，继承自 torch.nn.Module
class PackedParameter(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    # 将 param 参数保存到 state_dict 中
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'param'] = self.param

    # 从 state_dict 中加载 param 参数
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.param = state_dict[prefix + 'param']
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)


# 定义 RNNBase 类，继承自 torch.nn.Module
class RNNBase(torch.nn.Module):

    _FLOAT_MODULE = nn.RNNBase  # 设置 FLOAT_MODULE 属性为 nn.RNNBase

    _version = 2  # 设置 _version 属性为 2

    # 返回模型的名称 'DynamicQuantizedRNN'
    def _get_name(self):
        return 'DynamicQuantizedRNN'

    # 返回模型的额外描述信息，包括 input_size、hidden_size 等参数
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
    # 定义自定义的对象表示方法 `__repr__`
    def __repr__(self):
        # 我们不希望显示 `ModuleList` 的子模块，因此自定义 `__repr__` 方法。
        # 该方法与 nn.Module.__repr__ 相同，只是增加了对 `PackedParameter` 和 `nn.ModuleList` 的检查。
        # 你仍然应该覆盖 `extra_repr` 方法以添加更多信息。

        # 初始化额外的行列表
        extra_lines = []
        # 调用对象的 `extra_repr` 方法获取额外的表示信息
        extra_repr = self.extra_repr()
        # 如果额外的表示信息不为空，则按换行符分割成列表
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        # 初始化子模块行列表
        child_lines = []
        # 遍历模块字典 `_modules` 的每个键值对
        for key, module in self._modules.items():
            # 如果模块是 `PackedParameter` 或 `nn.ModuleList` 的实例，则跳过
            if isinstance(module, (PackedParameter, nn.ModuleList)):
                continue
            # 获取模块的字符串表示，并增加两个空格的缩进
            mod_str = repr(module)
            mod_str = nn.modules.module._addindent(mod_str, 2)
            # 将格式化后的模块字符串添加到子模块行列表中
            child_lines.append('(' + key + '): ' + mod_str)

        # 合并额外行和子模块行，得到完整的表示信息列表
        lines = extra_lines + child_lines

        # 初始化主要字符串，表示对象的类名
        main_str = self._get_name() + '('
        # 如果存在行信息
        if lines:
            # 如果额外行只有一行且没有子模块行，则简单地将其作为一行信息添加到主要字符串
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                # 否则，格式化多行信息，每行缩进两个空格，然后添加到主要字符串中
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        # 添加右括号，构成最终的表示信息字符串
        main_str += ')'
        # 返回最终的对象表示字符串
        return main_str

    # 检查输入是否符合预期要求
    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        # 如果 `batch_sizes` 不为 None，则期望输入维度为 2；否则为 3
        expected_input_dim = 2 if batch_sizes is not None else 3
        # 如果输入张量的维度与期望维度不符，则引发运行时错误
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f'input must have {expected_input_dim} dimensions, got {input.dim()}')
        # 如果输入张量的最后一个维度大小与 `input_size` 不相等，则引发运行时错误
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f'input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}')

    # 获取期望的隐藏状态尺寸
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        # 如果 `batch_sizes` 不为 None，则将第一个批次大小转换为整数，否则根据 `batch_first` 选择输入维度
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        # 计算期望的隐藏状态尺寸
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        # 返回期望的隐藏状态尺寸元组
        return expected_hidden_size

    # 检查隐藏状态尺寸是否符合预期要求
    def check_hidden_size(
        self, hx: Tensor, expected_hidden_size: Tuple[int, int, int],
        msg: str = 'Expected hidden size {}, got {}'
    ) -> None:
        # 如果隐藏状态张量 `hx` 的大小与期望的隐藏状态尺寸不符，则引发运行时错误
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(
                expected_hidden_size, list(hx.size())))

    # 检查前向传播参数是否符合预期要求
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
        # 检查输入是否符合预期要求
        self.check_input(input, batch_sizes)
        # 获取期望的隐藏状态尺寸
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        # 检查隐藏状态尺寸是否符合预期要求
        self.check_hidden_size(hidden, expected_hidden_size,
                               msg='Expected hidden size {}, got {}')
    # 对隐藏状态进行置换操作，根据给定的置换 permutation 对隐藏状态 hx 进行重新排列
    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor:
        # 如果置换 permutation 为 None，则直接返回原始隐藏状态 hx
        if permutation is None:
            return hx
        # 否则，使用 _apply_permutation 函数对隐藏状态 hx 按照置换 permutation 进行重新排列
        return _apply_permutation(hx, permutation)

    # 从给定的 state_dict 中加载模型的状态，并处理版本信息，存储在 self.version 中
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 从 local_metadata 中获取版本信息
        version = local_metadata.get('version', None)
        # 将获取的版本信息存储在 self.version 中
        self.version = version
        # 调用父类的 _load_from_state_dict 方法，加载模型状态，且不使用 strict 模式
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)

    # 根据权重和偏置的字典设置模型的权重
    def set_weight_bias(self, weight_bias_dict):
        # 定义生成权重和偏置名称的内部函数
        def weight_bias_name(ihhh, layer, suffix):
            weight_name = f"weight_{ihhh}_l{layer}{suffix}"
            bias_name = f"bias_{ihhh}_l{layer}{suffix}"
            return weight_name, bias_name

        # 确定双向循环神经网络的方向数
        num_directions = 2 if self.bidirectional else 1
        # TODO: 在 RNNBase 的 __init__ 方法中处理重复的部分

        # 存储所有的权重值
        _all_weight_values = []
        # 遍历每一层
        for layer in range(self.num_layers):
            # 遍历每个方向
            for direction in range(num_directions):
                # 根据方向确定后缀
                suffix = "_reverse" if direction == 1 else ""
                # 获取当前层和方向下的权重和偏置名称
                w_ih_name, b_ih_name = weight_bias_name("ih", layer, suffix)
                w_hh_name, b_hh_name = weight_bias_name("hh", layer, suffix)
                # 从权重和偏置的字典中获取对应的值
                w_ih = weight_bias_dict[w_ih_name]
                b_ih = weight_bias_dict[b_ih_name]
                w_hh = weight_bias_dict[w_hh_name]
                b_hh = weight_bias_dict[b_hh_name]
                # 如果权重 w_ih 的数据类型为 torch.qint8
                if w_ih.dtype == torch.qint8:
                    # 使用 quantized 操作预打包 w_ih 和 b_ih
                    packed_ih = torch.ops.quantized.linear_prepack(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack(w_hh, b_hh)
                    # 根据模型版本信息创建动态量化单元参数 cell_params
                    if self.version is None or self.version < 2:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, b_ih, b_hh)
                    else:
                        cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                            packed_ih, packed_hh, b_ih, b_hh, True)
                else:
                    # 否则，使用 quantized 操作预打包 w_ih 和 b_ih（fp16 格式）
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(w_ih, b_ih)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(w_hh, b_hh)
                    # 创建 fp16 格式的量化单元参数 cell_params
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(
                        packed_ih, packed_hh)

                # 将生成的 PackedParameter 对象添加到 _all_weight_values 列表中
                _all_weight_values.append(PackedParameter(cell_params))
        # 使用 torch.nn.ModuleList 将 _all_weight_values 转换为模型的 ModuleList
        self._all_weight_values = torch.nn.ModuleList(_all_weight_values)

    # 声明一个类方法
    @classmethod
    # 返回一个包含权重和偏置的字典
    def _weight_bias(self):
        # 初始化一个空的字典，用于存储权重和偏置
        weight_bias_dict: Dict[str, Dict] = {'weight' : {}, 'bias' : {}}
        # 初始化计数器
        count = 0
        # 确定方向数，双向循环神经网络则为2，否则为1
        num_directions = 2 if self.bidirectional else 1
        # 循环每一层神经网络
        for layer in range(self.num_layers):
            # 遍历每个方向（前向和反向，如果是双向）
            for direction in range(num_directions):
                # 根据方向确定后缀名称
                suffix = '_reverse' if direction == 1 else ''
                # 构建权重的键名和偏置的键名
                key_name1 = f'weight_ih_l{layer}{suffix}'
                key_name2 = f'weight_hh_l{layer}{suffix}'
                # 获取打包的权重和偏置，这些权重和偏置是通过torchbind类CellParamsSerializationType中的packed weights访问的
                packed_weight_bias = self._all_weight_values[count].param.__getstate__()[0][4]
                # 将权重存入字典
                weight_bias_dict['weight'][key_name1] = packed_weight_bias[0].__getstate__()[0][0]
                weight_bias_dict['weight'][key_name2] = packed_weight_bias[1].__getstate__()[0][0]
                # 更新键名为偏置的名称
                key_name1 = f'bias_ih_l{layer}{suffix}'
                key_name2 = f'bias_hh_l{layer}{suffix}'
                # 将偏置存入字典
                weight_bias_dict['bias'][key_name1] = packed_weight_bias[0].__getstate__()[0][1]
                weight_bias_dict['bias'][key_name2] = packed_weight_bias[1].__getstate__()[0][1]
                # 更新计数器
                count = count + 1
        # 返回包含权重和偏置的字典
        return weight_bias_dict

    # 返回所有权重的字典
    def get_weight(self):
        return self._weight_bias()['weight']

    # 返回所有偏置的字典
    def get_bias(self):
        return self._weight_bias()['bias']
class LSTM(RNNBase):
    r"""
    A dynamic quantized LSTM module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.LSTM`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """
    _FLOAT_MODULE = nn.LSTM

    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)
        # 初始化函数，调用父类初始化方法，并设定名称为 'LSTM' 的参数

    def _get_name(self):
        return 'DynamicQuantizedLSTM'
        # 返回当前模型的名称为 'DynamicQuantizedLSTM'

    def forward_impl(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]],
        batch_sizes: Optional[Tensor], max_batch_size: int,
        sorted_indices: Optional[Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
            # 如果隐藏状态 hx 为空，则初始化为零张量，匹配输入数据的大小

        else:
            # 每个隐藏状态的批次应该与用户认为正在传递的输入序列匹配。
            hx = self.permute_hidden(hx, sorted_indices)
            # 否则，通过 permute_hidden 方法调整隐藏状态的顺序以匹配输入数据的顺序

        self.check_forward_args(input, hx, batch_sizes)
        # 检查前向传播参数的合法性

        _all_params = ([m.param for m in self._all_weight_values])
        # 获取所有参数值

        if batch_sizes is None:
            result = torch.quantized_lstm(input, hx, _all_params, self.bias, self.num_layers,
                                          float(self.dropout), self.training, self.bidirectional,
                                          self.batch_first, dtype=self.dtype, use_dynamic=True)
            # 如果没有指定 batch_sizes，调用 torch.quantized_lstm 执行量化 LSTM 前向传播
        else:
            result = torch.quantized_lstm(input, batch_sizes, hx, _all_params, self.bias,
                                          self.num_layers, float(self.dropout), self.training,
                                          self.bidirectional, dtype=self.dtype, use_dynamic=True)
            # 否则，使用指定的 batch_sizes 调用 torch.quantized_lstm 执行量化 LSTM 前向传播

        output = result[0]
        hidden = result[1:]
        # 提取结果的输出和隐藏状态

        return output, hidden
        # 返回输出和隐藏状态的元组

    @torch.jit.export
    def forward_tensor(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_sizes = None
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None
        # 初始化 batch_sizes, max_batch_size, sorted_indices, unsorted_indices

        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices)
        # 调用前向传播的实现方法 forward_impl

        return output, self.permute_hidden(hidden, unsorted_indices)
        # 返回输出和调整后的隐藏状态的元组
    # 接受一个 PackedSequence 类型的输入和一个可选的隐藏状态元组，返回一个元组，包含处理后的 PackedSequence 和更新后的隐藏状态
    def forward_packed(
        self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        # 解包 PackedSequence 输入
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        # 获取批次大小的最大值
        max_batch_size = int(batch_sizes[0])

        # 调用实际的前向传播实现函数
        output_, hidden = self.forward_impl(
            input_, hx, batch_sizes, max_batch_size, sorted_indices
        )

        # 封装输出结果为 PackedSequence 类型
        output = PackedSequence(output_, batch_sizes,
                                sorted_indices, unsorted_indices)
        # 返回处理后的输出和隐藏状态的元组
        return output, self.permute_hidden(hidden, unsorted_indices)

    # "type: ignore" is required due to issue #43072
    # 忽略类型检查是因为问题 #43072
    def permute_hidden(  # type: ignore[override]
        self, hx: Tuple[Tensor, Tensor], permutation: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        # 如果置换张量为空，则直接返回隐藏状态
        if permutation is None:
            return hx
        # 否则，对隐藏状态中的张量进行置换
        return _apply_permutation(hx[0], permutation), _apply_permutation(hx[1], permutation)

    # "type: ignore" is required due to issue #43072
    # 忽略类型检查是因为问题 #43072
    def check_forward_args(  # type: ignore[override]
        self, input: Tensor, hidden: Tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]
    ) -> None:
        # 检查输入和批次大小
        self.check_input(input, batch_sizes)
        # 获取预期的隐藏状态大小
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        # 检查隐藏状态的大小是否符合预期
        self.check_hidden_size(hidden[0], expected_hidden_size,
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], expected_hidden_size,
                               'Expected hidden[1] size {}, got {}')

    @torch.jit.ignore
    # 忽略 Torch JIT 编译
    def forward(self, input, hx=None):
        # 如果输入是 PackedSequence 类型，则调用 forward_packed 方法
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            # 否则调用 forward_tensor 方法
            return self.forward_tensor(input, hx)

    @classmethod
    # 类方法，用于从浮点模型创建量化模型
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    @classmethod
    # 类方法，用于从参考模型创建量化模型
    def from_reference(cls, ref_mod):
        # 断言参考模型有 "weight_ih_l0_dtype" 属性，用于确保 LSTM 中有权重信息
        assert hasattr(ref_mod, "weight_ih_l0_dtype"), "We are assuming weight_ih_l0 " \
        "exists in LSTM, may need to relax the assumption to support the use case"
        
        # 创建一个量化模型对象
        qmod = cls(
            ref_mod.input_size,
            ref_mod.hidden_size,
            ref_mod.num_layers,
            ref_mod.bias,
            ref_mod.batch_first,
            ref_mod.dropout,
            ref_mod.bidirectional,
            # 假设存在层 0，这通常是合理的
            ref_mod.weight_ih_l0_dtype,
        )
        
        # 设置权重和偏置信息到量化模型中
        qmod.set_weight_bias(ref_mod.get_quantized_weight_bias_dict())
        return qmod
# 定义一个多层门控循环单元（GRU）的类，继承自RNNBase基类
class GRU(RNNBase):
    r"""Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.
    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features h_t from the last layer of the GRU,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.

          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

    Shape:
        - Input1: :math:`(L, N, H_{in})` tensor containing input features where
          :math:`H_{in}=\text{input\_size}` and `L` represents a sequence length.
        - Input2: :math:`(S, N, H_{out})` tensor
          containing the initial hidden state for each element in the batch.
          :math:`H_{out}=\text{hidden\_size}`
          Defaults to zero if not provided. where :math:`S=\text{num\_layers} * \text{num\_directions}`
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\text{num\_directions} * \text{hidden\_size}`
        - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`
    # 定义私有变量_FLOAT_MODULE，指定为 nn.GRU
    _FLOAT_MODULE = nn.GRU

    # 定义私有变量__overloads__，包含 'forward' 的两个重载方法：'forward_packed' 和 'forward_tensor'
    __overloads__ = {'forward': ['forward_packed', 'forward_tensor']}

    # 定义初始化方法，继承父类的初始化方法，设定模型类型为 'GRU'
    def __init__(self, *args, **kwargs):
        super().__init__('GRU', *args, **kwargs)

    # 私有方法，返回模型名称 'DynamicQuantizedGRU'
    def _get_name(self):
        return 'DynamicQuantizedGRU'

    # 检查前向传播参数的有效性，验证输入数据和隐藏状态的尺寸匹配性
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None:
        # 检查输入数据和批次大小
        self.check_input(input, batch_sizes)
        # 获取预期的隐藏状态尺寸
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        # 检查隐藏状态的尺寸是否符合预期
        self.check_hidden_size(hidden, expected_hidden_size,
                               'Expected hidden size {}, got {}')

    # 实现具体的前向传播逻辑
    def forward_impl(
        self, input: Tensor, hx: Optional[Tensor],
        batch_sizes: Optional[Tensor], max_batch_size: int,
        sorted_indices: Optional[Tensor]
    ):
    ) -> Tuple[Tensor, Tensor]:
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            # 创建与输入数据相匹配的全零隐藏状态张量
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = zeros
        else:
            # 将传入的隐藏状态按照指定的排列重新排序
            hx = self.permute_hidden(hx, sorted_indices)

        # 检查前向传播参数的合法性
        self.check_forward_args(input, hx, batch_sizes)

        # 获取所有权重值的参数列表
        _all_params = ([m.param for m in self._all_weight_values])
        if batch_sizes is None:
            # 执行量化 GRU 的前向传播计算，不考虑批次大小信息
            result = torch.quantized_gru(input,
                                         hx,
                                         _all_params,
                                         self.bias,
                                         self.num_layers,
                                         self.dropout,
                                         self.training,
                                         self.bidirectional,
                                         self.batch_first)
        else:
            # 执行量化 GRU 的前向传播计算，考虑批次大小信息
            result = torch.quantized_gru(input,
                                         batch_sizes,
                                         hx,
                                         _all_params,
                                         self.bias,
                                         self.num_layers,
                                         self.dropout,
                                         self.training,
                                         self.bidirectional)
        # 提取计算结果的输出和隐藏状态
        output = result[0]
        hidden = result[1]

        return output, hidden


    @torch.jit.export
    def forward_tensor(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch_sizes = None
        # 根据 batch_first 属性确定最大批次大小
        max_batch_size = input.size(0) if self.batch_first else input.size(1)
        sorted_indices = None
        unsorted_indices = None

        # 调用实际的前向传播实现函数
        output, hidden = self.forward_impl(
            input, hx, batch_sizes, max_batch_size, sorted_indices)

        # 返回前向传播的输出和经过重新排列的隐藏状态
        return output, self.permute_hidden(hidden, unsorted_indices)

    @torch.jit.export
    def forward_packed(
        self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> Tuple[PackedSequence, Tensor]:
        # 解包 PackedSequence，获取输入数据、批次大小信息以及排序索引
        input_, batch_sizes, sorted_indices, unsorted_indices = input
        # 确定最大批次大小
        max_batch_size = int(batch_sizes[0])
        # 调用实际的前向传播实现函数
        output_, hidden = self.forward_impl(
            input_, hx, batch_sizes, max_batch_size, sorted_indices
        )

        # 将前向传播的输出重新封装成 PackedSequence 对象
        output = PackedSequence(output_, batch_sizes,
                                sorted_indices, unsorted_indices)
        # 返回前向传播的输出和经过重新排列的隐藏状态
        return output, self.permute_hidden(hidden, unsorted_indices)

    def permute_hidden(
        self, hx: Tensor, permutation: Optional[Tensor]
    # 定义方法签名，指定返回类型为 Tensor
    ) -> Tensor:
        # 如果未提供置换参数，则直接返回输入的张量 hx
        if permutation is None:
            return hx
        # 否则，应用给定的置换 permutation 到输入张量 hx 上并返回结果
        return _apply_permutation(hx, permutation)

    # 忽略该方法，不对其进行 Torch JIT 编译
    @torch.jit.ignore
    def forward(self, input, hx=None):
        # 如果输入是 PackedSequence 类型，则调用 forward_packed 方法处理
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, hx)
        else:
            # 否则，调用 forward_tensor 方法处理输入
            return self.forward_tensor(input, hx)

    # 从浮点模型 mod 创建量化模型的类方法
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 调用父类的 from_float 方法创建量化模型，可以选择是否使用预计算的伪量化参数
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)

    # 从参考模型 ref_mod 创建量化模型的类方法
    @classmethod
    def from_reference(cls, ref_mod):
        # 断言确保参考模型 ref_mod 中存在属性 "weight_ih_l0_dtype"，用于初始化量化模型的权重类型
        assert hasattr(ref_mod, "weight_ih_l0_dtype"), "We are assuming weight_ih_l0 "
        "exists in LSTM, may need to relax the assumption to support the use case"
        # 使用 ref_mod 的属性初始化量化模型 qmod
        qmod = cls(
            ref_mod.input_size,
            ref_mod.hidden_size,
            ref_mod.num_layers,
            ref_mod.bias,
            ref_mod.batch_first,
            ref_mod.dropout,
            ref_mod.bidirectional,
            # 假设至少存在第 0 层权重类型，传递给量化模型的 weight_ih_l0_dtype 属性
            ref_mod.weight_ih_l0_dtype,
        )
        # 设置 qmod 的权重和偏置，从 ref_mod 获取量化后的权重和偏置字典
        qmod.set_weight_bias(ref_mod.get_quantized_weight_bias_dict())
        # 返回初始化后的量化模型 qmod
        return qmod
# 定`
class RNNCellBase(torch.nn.Module):
    # 定义一个常量列表，包含该模块的输入大小、隐藏层大小和偏置信息
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias=True, num_chunks=4, dtype=torch.qint8):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化输入大小
        self.input_size = input_size
        # 初始化隐藏层大小
        self.hidden_size = hidden_size
        # 初始化偏置标志
        self.bias = bias
        # 初始化权重数据类型
        self.weight_dtype = dtype
        # 如果需要偏置，则初始化偏置权重
        if bias:
            self.bias_ih = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
            self.bias_hh = torch.randn(num_chunks * hidden_size).to(dtype=torch.float)
        else:
            # 如果不需要偏置，则注册参数为 None
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        # 初始化输入到隐藏层的权重
        weight_ih = torch.randn(num_chunks * hidden_size, input_size).to(torch.float)
        # 初始化隐藏层到隐藏层的权重
        weight_hh = torch.randn(num_chunks * hidden_size, hidden_size).to(torch.float)
        # 如果权重数据类型是整数8位量化，则进行量化
        if dtype == torch.qint8:
            weight_ih = torch.quantize_per_tensor(weight_ih, scale=1, zero_point=0, dtype=torch.qint8)
            weight_hh = torch.quantize_per_tensor(weight_hh, scale=1, zero_point=0, dtype=torch.qint8)

        # 根据数据类型选择不同的权重打包方法
        if dtype == torch.qint8:
            # 对于每一层和每一个方向，需要量化并打包权重，打包参数的顺序为：
            #   w_ih, w_hh
            packed_weight_ih = torch.ops.quantized.linear_prepack(weight_ih, self.bias_ih)
            packed_weight_hh = torch.ops.quantized.linear_prepack(weight_hh, self.bias_hh)
        else:
            # 对于每一层和每一个方向，需要量化并打包权重，打包参数的顺序为：
            #   packed_ih, packed_hh, b_ih, b_hh
            packed_weight_ih = torch.ops.quantized.linear_prepack_fp16(weight_ih, self.bias_ih)
            packed_weight_hh = torch.ops.quantized.linear_prepack_fp16(weight_hh, self.bias_hh)

        # 将打包的权重存储为实例变量
        self._packed_weight_ih = packed_weight_ih
        self._packed_weight_hh = packed_weight_hh

    # 定义一个方法，返回该模块的名称
    def _get_name(self):
        return 'DynamicQuantizedRNNBase'

    # 定义一个方法，返回模块的额外信息
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        # 如果存在偏置并且偏置不是 True，则添加偏置信息
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        # 如果存在非线性激活函数且非线性激活函数不是 tanh，则添加激活函数信息
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    # 定义一个方法，检查输入的维度是否与输入大小一致
    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                f"input has inconsistent input_size: got {input.size(1)}, expected {self.input_size}")
    # 检查输入张量和隐藏状态张量的批次大小是否匹配
    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        # 如果输入张量的批次大小与隐藏状态张量的批次大小不匹配，则抛出运行时错误
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                f"Input batch size {input.size(0)} doesn't match hidden{hidden_label} batch size {hx.size(0)}")

        # 检查隐藏状态张量的第二维（隐藏单元大小）是否与预期的隐藏单元大小一致
        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                f"hidden{hidden_label} has inconsistent hidden_size: got {hx.size(1)}, expected {self.hidden_size}")

    # 声明一个类方法
    @classmethod
    # 根据浮点模块创建对应的量化 RNN 单元
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 确保模块类型是 nn.LSTMCell、nn.GRUCell 或 nn.RNNCell
        assert type(mod) in {torch.nn.LSTMCell,
                             torch.nn.GRUCell,
                             torch.nn.RNNCell}, 'nn.quantized.dynamic.RNNCellBase.from_float \
                                 only works for nn.LSTMCell, nn.GRUCell and nn.RNNCell'
        # 确保输入的浮点模块定义了 qconfig
        assert hasattr(
            mod, 'qconfig'), 'Input float module must have qconfig defined'

        # 根据模块的 qconfig 来确定权重观察方法
        if mod.qconfig is not None and mod.qconfig.weight is not None:
            weight_observer_method = mod.qconfig.weight
        else:
            # 解决循环导入问题，延迟导入 default_dynamic_qconfig
            from torch.ao.quantization.qconfig import default_dynamic_qconfig
            weight_observer_method = default_dynamic_qconfig.weight

        # 获取权重观察方法的数据类型
        dtype = weight_observer_method().dtype
        # 支持的标量类型包括 torch.qint8 和 torch.float16
        supported_scalar_types = [torch.qint8, torch.float16]
        # 如果数据类型不在支持列表中，则抛出运行时错误
        if dtype not in supported_scalar_types:
            raise RuntimeError(f'Unsupported dtype for dynamic RNN quantization: {dtype}')

        # 根据模块类型实例化相应的量化 RNN 单元
        qRNNCellBase: Union[LSTMCell, GRUCell, RNNCell]

        if type(mod) == torch.nn.LSTMCell:
            qRNNCellBase = LSTMCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.GRUCell:
            qRNNCellBase = GRUCell(mod.input_size, mod.hidden_size, bias=mod.bias, dtype=dtype)
        elif type(mod) == torch.nn.RNNCell:
            qRNNCellBase = RNNCell(mod.input_size, mod.hidden_size, bias=mod.bias, nonlinearity=mod.nonlinearity, dtype=dtype)
        else:
            # 目前只支持 LSTMCell、GRUCell 和 RNNCell
            raise NotImplementedError('Only LSTMCell, GRUCell and RNNCell \
            are supported for QuantizedRNN for now')

        # 确保浮点模块有偏置项
        assert mod.bias

        # 定义观察和量化权重的函数
        def _observe_and_quantize_weight(weight):
            if dtype == torch.qint8:
                weight_observer = weight_observer_method()
                weight_observer(weight)
                # 对权重进行量化
                qweight = _quantize_weight(weight.float(), weight_observer)
                return qweight
            else:
                # 如果数据类型是 float16，则直接返回浮点数权重
                return weight.float()

        # 将量化后的权重和偏置打包到量化 RNN 单元中的输入-隐藏权重中
        qRNNCellBase._packed_weight_ih = pack_weight_bias(_observe_and_quantize_weight(mod.weight_ih), mod.bias_ih, dtype)
        # 将量化后的权重和偏置打包到量化 RNN 单元中的隐藏-隐藏权重中
        qRNNCellBase._packed_weight_hh = pack_weight_bias(_observe_and_quantize_weight(mod.weight_hh), mod.bias_hh, dtype)
        # 返回量化 RNN 单元
        return qRNNCellBase

    @classmethod
    def from_reference(cls, ref_mod):
        # 确保参考模块有"weight_ih_dtype"属性，用于处理权重
        assert hasattr(ref_mod, "weight_ih_dtype"), "We are assuming weight_ih " \
        "exists in reference module, may need to relax the assumption to support the use case"
        
        # 根据参考模块是否有"nonlinearity"属性，创建量化模块实例
        if hasattr(ref_mod, "nonlinearity"):
            qmod = cls(
                ref_mod.input_size,
                ref_mod.hidden_size,
                ref_mod.bias,
                ref_mod.nonlinearity,
                dtype=ref_mod.weight_ih_dtype
            )
        else:
            qmod = cls(
                ref_mod.input_size,
                ref_mod.hidden_size,
                ref_mod.bias,
                dtype=ref_mod.weight_ih_dtype
            )
        
        # 构建包含量化权重和偏置的字典
        weight_bias_dict = {
            "weight": {
                "weight_ih": ref_mod.get_quantized_weight_ih(),
                "weight_hh": ref_mod.get_quantized_weight_hh(),
            },
            "bias": {
                "bias_ih": ref_mod.bias_ih,
                "bias_hh": ref_mod.bias_hh,
            }
        }
        
        # 设置量化模块的权重和偏置
        qmod.set_weight_bias(weight_bias_dict)
        
        # 返回创建的量化模块实例
        return qmod

    def _weight_bias(self):
        # 返回包含权重和偏置的字典
        weight_bias_dict: Dict[str, Dict] = {'weight' : {}, 'bias' : {}}
        
        # 获取并解压缩权重和偏置
        w1, b1 = self._packed_weight_ih.__getstate__()[0]
        w2, b2 = self._packed_weight_hh.__getstate__()[0]
        
        # 将权重和偏置放入字典中
        weight_bias_dict['weight']['weight_ih'] = w1
        weight_bias_dict['weight']['weight_hh'] = w2
        weight_bias_dict['bias']['bias_ih'] = b1
        weight_bias_dict['bias']['bias_hh'] = b2
        
        # 返回构建好的字典
        return weight_bias_dict

    def get_weight(self):
        # 返回权重字典
        return self._weight_bias()['weight']

    def get_bias(self):
        # 返回偏置字典
        return self._weight_bias()['bias']

    def set_weight_bias(self, weight_bias_dict):
        # 使用给定的权重和偏置字典设置压缩后的权重和偏置
        self._packed_weight_ih = pack_weight_bias(
            weight_bias_dict["weight"]["weight_ih"],
            weight_bias_dict["bias"]["bias_ih"],
            self.weight_dtype)
        self._packed_weight_hh = pack_weight_bias(
            weight_bias_dict["weight"]["weight_hh"],
            weight_bias_dict["bias"]["bias_hh"],
            self.weight_dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # 将对象的状态保存到state_dict中，包括压缩后的权重和偏置
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + '_packed_weight_ih'] = self._packed_weight_ih
        destination[prefix + '_packed_weight_hh'] = self._packed_weight_hh
    # 从给定的状态字典中加载模型的权重数据，根据指定的前缀和其他参数进行处理
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 从状态字典中弹出输入门权重数据，并赋值给对象的_packed_weight_ih属性
        self._packed_weight_ih = state_dict.pop(prefix + '_packed_weight_ih')
        # 从状态字典中弹出隐藏状态权重数据，并赋值给对象的_packed_weight_hh属性
        self._packed_weight_hh = state_dict.pop(prefix + '_packed_weight_hh')
        # 调用父类的_load_from_state_dict方法，加载其余的状态数据，且不进行严格的匹配
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)
# 定义一个 RNN 单元的类，继承自 RNNCellBase 类
class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.
    A dynamic quantized RNNCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.RNNCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    # 定义类常量，包括输入大小、隐藏大小、是否包含偏置以及非线性函数类型
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']

    # 初始化方法，设置输入大小、隐藏大小、是否包含偏置、非线性函数类型和数据类型
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", dtype=torch.qint8):
        # 调用父类的初始化方法，设置输入大小、隐藏大小、是否包含偏置、片段数和数据类型
        super().__init__(input_size, hidden_size, bias, num_chunks=1, dtype=dtype)
        # 设置非线性函数类型
        self.nonlinearity = nonlinearity

    # 返回对象名称的方法
    def _get_name(self):
        return 'DynamicQuantizedRNNCell'

    # 前向传播方法，接受输入张量 input 和可选的隐藏状态张量 hx，返回张量
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # 检查输入张量是否有效
        self.check_forward_input(input)
        # 如果隐藏状态 hx 为 None，则初始化为全零张量
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        # 检查前向传播时的隐藏状态和输入是否有效
        self.check_forward_hidden(input, hx, '')
        # 根据非线性函数类型选择对应的动态量化 RNN 单元函数进行前向传播计算
        if self.nonlinearity == "tanh":
            ret = torch.ops.quantized.quantized_rnn_tanh_cell_dynamic(
                input, hx,
                self._packed_weight_ih, self._packed_weight_hh,
                self.bias_ih, self.bias_hh)
        elif self.nonlinearity == "relu":
            ret = torch.ops.quantized.quantized_rnn_relu_cell_dynamic(
                input, hx,
                self._packed_weight_ih, self._packed_weight_hh,
                self.bias_ih, self.bias_hh)
        else:
            # 如果非线性函数类型未知，则引发运行时错误
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                f"Unknown nonlinearity: {self.nonlinearity}")
        # 返回前向传播计算结果张量
        return ret

    # 类方法，从浮点数模型转换为量化模型
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)


# LSTM 单元的类，继承自 RNNCellBase 类
class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    A dynamic quantized LSTMCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.LSTMCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """

    # 初始化方法，继承自父类，支持可变参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，传递参数和片段数为 4
        super().__init__(*args, num_chunks=4, **kwargs)  # type: ignore[misc]
    # 返回固定的字符串名称 'DynamicQuantizedLSTMCell'
    def _get_name(self):
        return 'DynamicQuantizedLSTMCell'

    # 前向传播方法，接受输入张量 input 和隐藏状态 hx（可选），返回输出张量和新的隐藏状态
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        # 检查输入是否符合前向传播的要求
        self.check_forward_input(input)
        # 如果隐藏状态 hx 为空，则创建全零张量作为初始隐藏状态
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        # 检查输入和隐藏状态的有效性，打印日志中标识隐藏状态元组的索引号
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        # 调用量化 LSTM 单元的动态版本实现前向传播
        return torch.ops.quantized.quantized_lstm_cell_dynamic(
            input, hx,
            self._packed_weight_ih, self._packed_weight_hh,
            self.bias_ih, self.bias_hh)

    # 类方法，从浮点数模型 mod 转换为当前量化模型类的实例
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
    r"""A gated recurrent unit (GRU) cell

    A dynamic quantized GRUCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.GRUCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.GRUCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(self, input_size, hidden_size, bias=True, dtype=torch.qint8):
        super().__init__(input_size, hidden_size, bias, num_chunks=3, dtype=dtype)
        # 调用父类的构造函数，初始化动态量化的 GRUCell，设置输入大小、隐藏层大小、是否有偏置，默认数据类型为 qint8

    def _get_name(self):
        return 'DynamicQuantizedGRUCell'
        # 返回当前模块的名称字符串 'DynamicQuantizedGRUCell'

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        # 检查前向传播的输入是否合法

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            # 如果隐藏状态 hx 为 None，则初始化为与输入相同大小的全零张量，数据类型和设备与输入一致

        self.check_forward_hidden(input, hx, '')
        # 检查前向传播的隐藏状态是否合法

        return torch.ops.quantized.quantized_gru_cell_dynamic(
            input, hx,
            self._packed_weight_ih, self._packed_weight_hh,
            self.bias_ih, self.bias_hh,
        )
        # 调用动态量化 GRU 单元的前向计算函数，传入输入 input、隐藏状态 hx、压缩的输入权重、压缩的隐藏状态权重、输入偏置和隐藏状态偏置
        # 返回前向传播的输出张量

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        return super().from_float(mod, use_precomputed_fake_quant=use_precomputed_fake_quant)
        # 从浮点模型 mod 转换得到动态量化的 GRUCell 类，可以选择是否使用预计算的伪量化方法
```