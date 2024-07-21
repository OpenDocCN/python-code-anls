# `.\pytorch\torch\nn\modules\rnn.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和类型
import math
import numbers
import warnings
import weakref
from typing import List, Optional, overload, Tuple
from typing_extensions import deprecated

# 导入 PyTorch 相关模块
import torch
from torch import _VF, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence

# 导入自定义的 Module 类
from .module import Module

# 导出的符号列表
__all__ = [
    "RNNBase",
    "RNN",
    "LSTM",
    "GRU",
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
]

# 定义 RNN 实现的映射关系
_rnn_impls = {
    "RNN_TANH": _VF.rnn_tanh,
    "RNN_RELU": _VF.rnn_relu,
}

# 实现应用置换的函数
def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)

# 废弃的函数，提醒使用 index_select 代替
@deprecated(
    "`apply_permutation` is deprecated, please use `tensor.index_select(dim, permutation)` instead",
    category=FutureWarning,
)
def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return _apply_permutation(tensor, permutation, dim)

# RNNBase 类，继承自 Module 类，用于所有 RNN 模型的基类
class RNNBase(Module):
    r"""Base class for RNN modules (RNN, LSTM, GRU).

    Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization
    and utility methods for parameter storage management.

    .. note::
        The forward method is not implemented by the RNNBase class.

    .. note::
        LSTM and GRU classes override some methods implemented by RNNBase.
    """

    # 定义常量列表
    __constants__ = [
        "mode",
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
    ]
    # 被 JIT 编译器忽略的属性列表
    __jit_unused_properties__ = ["all_weights"]

    # 类的属性定义
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    # 初始化函数
    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ):
        # 父类构造函数
        super().__init__()

    # 初始化扁平化权重的私有方法
    def _init_flat_weights(self):
        # 获取所有扁平化权重的列表
        self._flat_weights = [
            getattr(self, wn) if hasattr(self, wn) else None
            for wn in self._flat_weights_names
        ]
        # 将权重列表转换为弱引用列表
        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]
        # 调用 flatten_parameters 方法
        self.flatten_parameters()

    # 设置属性的特殊方法，用于更新扁平化权重列表
    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # 如果属性在扁平化权重名称列表中，更新对应位置的权重值
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        # 调用父类的属性设置方法
        super().__setattr__(attr, value)
    def flatten_parameters(self) -> None:
        """Reset parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # 如果 _flat_weights 只部分实例化，则立即返回，不执行后续操作
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        # 检查 _flat_weights 中的每个元素是否为 Tensor 类型，如果不是则返回
        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return

        # 获取第一个 _flat_weights 的数据类型，用于后续比较
        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        # 检查 _flat_weights 中的每个元素是否符合 cuDNN 的要求：Tensor 类型、相同数据类型、在 GPU 上、可接受给定条件
        for fw in self._flat_weights:
            if (
                not isinstance(fw, Tensor)
                or not (fw.dtype == dtype)
                or not fw.is_cuda
                or not torch.backends.cudnn.is_acceptable(fw)
            ):
                return

        # 如果 _flat_weights 中存在数据指针重叠，返回以避免使用不唯一的参数指针
        unique_data_ptrs = {p.data_ptr() for p in self._flat_weights}
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        # 在第一个 _flat_weights 的 CUDA 设备上，调用 cudnn.rnn 执行权重扁平化操作
        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # 注意：由于 _cudnn_rnn_flatten_weight 是对 self._flat_weights 的原地操作，因此需要使用 no_grad()
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights,
                        num_weights,
                        self.input_size,
                        rnn.get_cudnn_mode(self.mode),
                        self.hidden_size,
                        self.proj_size,
                        self.num_layers,
                        self.batch_first,
                        bool(self.bidirectional),
                    )

    def _apply(self, fn, recurse=True):
        # 初始化 _flat_weight_refs 为空列表
        self._flat_weight_refs = []
        # 调用父类的 _apply 方法，并返回结果
        ret = super()._apply(fn, recurse)

        # 重新初始化 _flat_weights
        # 注意：在删除此处之前要非常小心，因为第三方设备类型可能依赖此行为来正确处理像 LSTM 这样的模块的 .to() 方法。
        self._init_flat_weights()

        # 返回父类 _apply 方法的返回值
        return ret

    def reset_parameters(self) -> None:
        # 计算参数初始化的标准差
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        # 对模型中的每个参数使用均匀分布进行初始化
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
    # 检查输入张量的类型和维度是否符合预期
    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None:
        # 如果当前不是处于 Torch 脚本模式下
        if not torch.jit.is_scripting():
            # 检查输入张量的数据类型是否与模型中第一个权重张量的数据类型相同，且未启用任何自动类型转换
            if (
                input.dtype != self._flat_weights[0].dtype
                and not torch._C._is_any_autocast_enabled()
            ):
                raise ValueError(
                    f"input must have the type {self._flat_weights[0].dtype}, got type {input.dtype}"
                )
        # 期望的输入张量维度，根据是否提供了 batch_sizes 参数决定是二维还是三维
        expected_input_dim = 2 if batch_sizes is not None else 3
        # 如果输入张量的维度与期望的维度不符，抛出运行时错误
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f"input must have {expected_input_dim} dimensions, got {input.dim()}"
            )
        # 检查输入张量的最后一个维度大小是否与期望的输入大小相同
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}"
            )

    # 获取预期的隐藏状态大小
    def get_expected_hidden_size(
        self, input: Tensor, batch_sizes: Optional[Tensor]
    ) -> Tuple[int, int, int]:
        # 如果提供了 batch_sizes 参数，则使用其第一个元素作为 mini_batch 大小
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            # 否则根据 batch_first 属性确定 mini_batch 大小
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        # 根据模型的属性计算预期的隐藏状态大小
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.proj_size,
            )
        else:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.hidden_size,
            )
        return expected_hidden_size

    # 检查隐藏状态的大小是否符合预期
    def check_hidden_size(
        self,
        hx: Tensor,
        expected_hidden_size: Tuple[int, int, int],
        msg: str = "Expected hidden size {}, got {}",
    ) -> None:
        # 如果隐藏状态的大小与预期的大小不符，抛出运行时错误
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    # 检查权重是否在前向传播过程中发生了变化
    def _weights_have_changed(self):
        # 如果权重张量自上次前向传播以来发生了变化，则返回 True
        weights_changed = False
        for ref, name in zip(self._flat_weight_refs, self._flat_weights_names):
            weight = getattr(self, name) if hasattr(self, name) else None
            if weight is not None and ref is not None and ref() is not weight:
                weights_changed = True
                break
        return weights_changed

    # 检查前向传播参数的合法性
    def check_forward_args(
        self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]
    ):
        # 检查输入张量和 batch_sizes 参数
        self.check_input(input, batch_sizes)
        # 获取预期的隐藏状态大小
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)
        # 检查隐藏状态的大小是否符合预期
        self.check_hidden_size(hidden, expected_hidden_size)

    # 对隐藏状态进行置换操作
    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]):
        # 如果没有提供置换参数，则直接返回原始的隐藏状态张量
        if permutation is None:
            return hx
        # 否则根据置换参数对隐藏状态张量进行置换操作
        return _apply_permutation(hx, permutation)
    # 返回表示对象的额外信息字符串，包括输入大小和隐藏大小
    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        # 如果设置了 proj_size，则添加到信息字符串中
        if self.proj_size != 0:
            s += ", proj_size={proj_size}"
        # 如果层数不为 1，则添加到信息字符串中
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        # 如果没有偏置，则添加到信息字符串中
        if self.bias is not True:
            s += ", bias={bias}"
        # 如果不是 batch_first，则添加到信息字符串中
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        # 如果设置了 dropout，则添加到信息字符串中
        if self.dropout != 0:
            s += ", dropout={dropout}"
        # 如果不是单向的，则添加到信息字符串中
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    # 更新扁平化权重，仅在未进行脚本化时执行
    def _update_flat_weights(self):
        if not torch.jit.is_scripting():
            # 如果权重发生了变化，则重新初始化扁平化权重
            if self._weights_have_changed():
                self._init_flat_weights()

    # 获取对象的序列化状态，用于对象的持久化
    def __getstate__(self):
        # 在序列化之前更新扁平化权重
        self._update_flat_weights()
        # 复制对象的字典状态
        state = self.__dict__.copy()
        # 删除不需要序列化的权重引用
        del state["_flat_weight_refs"]
        return state
    # 定义对象的反序列化方法，接受一个字典 d 作为参数
    def __setstate__(self, d):
        # 调用父类的反序列化方法，恢复对象状态
        super().__setstate__(d)
        
        # 如果字典 d 中包含 "all_weights" 键
        if "all_weights" in d:
            # 将对象的 _all_weights 属性设置为 d["all_weights"] 的值
            self._all_weights = d["all_weights"]
        
        # 在 PyTorch 1.8 版本中添加了 proj_size 成员变量到 LSTM
        # 如果 d 中没有 "proj_size" 键
        if "proj_size" not in d:
            # 设置对象的 proj_size 属性为 0，以保持兼容性
            self.proj_size = 0

        # 如果 self._all_weights[0][0] 不是字符串类型
        if not isinstance(self._all_weights[0][0], str):
            # 获取 LSTM 的层数和方向数
            num_layers = self.num_layers
            num_directions = 2 if self.bidirectional else 1
            
            # 初始化 flat weights 相关属性
            self._flat_weights_names = []
            self._all_weights = []
            
            # 遍历每一层和方向，构建权重名称列表
            for layer in range(num_layers):
                for direction in range(num_directions):
                    suffix = "_reverse" if direction == 1 else ""
                    weights = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "bias_ih_l{}{}",
                        "bias_hh_l{}{}",
                        "weight_hr_l{}{}",
                    ]
                    # 根据层和方向格式化权重名称，并添加到 _all_weights 和 _flat_weights_names
                    weights = [x.format(layer, suffix) for x in weights]
                    
                    # 根据是否有偏置进行不同处理
                    if self.bias:
                        if self.proj_size > 0:
                            self._all_weights += [weights]
                            self._flat_weights_names.extend(weights)
                        else:
                            self._all_weights += [weights[:4]]
                            self._flat_weights_names.extend(weights[:4])
                    else:
                        if self.proj_size > 0:
                            self._all_weights += [weights[:2]] + [weights[-1:]]
                            self._flat_weights_names.extend(
                                weights[:2] + [weights[-1:]]
                            )
                        else:
                            self._all_weights += [weights[:2]]
                            self._flat_weights_names.extend(weights[:2])
            
            # 根据 _flat_weights_names 构建 _flat_weights 属性列表
            self._flat_weights = [
                getattr(self, wn) if hasattr(self, wn) else None
                for wn in self._flat_weights_names
            ]

        # 根据 _flat_weights 构建 _flat_weight_refs 属性列表，弱引用方式
        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]

    # 定义 all_weights 属性的 getter 方法，返回权重参数列表
    @property
    def all_weights(self) -> List[List[Parameter]]:
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]

    # 定义数据并行复制方法
    def _replicate_for_data_parallel(self):
        # 调用父类的复制方法，获取副本 replica
        replica = super()._replicate_for_data_parallel()
        
        # 需要复制这些缓存数据，否则副本将共享相同的 flat weights 列表
        replica._flat_weights = replica._flat_weights[:]
        replica._flat_weights_names = replica._flat_weights_names[:]
        
        # 返回副本对象 replica
        return replica
# 定义一个 RNN 类，继承自 RNNBase 类
class RNN(RNNBase):
    r"""__init__(input_size,hidden_size,num_layers=1,nonlinearity='tanh',bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

    Apply a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity to an input sequence. For each element in the input sequence,
    each layer computes the following function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    .. code-block:: python

        # Efficient implementation equivalent to the following with bidirectional=False
        def forward(x, h_0=None):
            if batch_first:
                x = x.transpose(0, 1)
            seq_len, batch_size, _ = x.size()
            if h_0 is None:
                h_0 = torch.zeros(num_layers, batch_size, hidden_size)
            h_t_minus_1 = h_0
            h_t = h_0
            output = []
            for t in range(seq_len):
                for layer in range(num_layers):
                    # 计算 RNN 的前向传播，使用 tanh 或 relu 非线性激活函数
                    h_t[layer] = torch.tanh(
                        x[t] @ weight_ih[layer].T  # 输入乘以输入到隐藏层的权重转置
                        + bias_ih[layer]           # 加上输入到隐藏层的偏置
                        + h_t_minus_1[layer] @ weight_hh[layer].T  # 加上前一时刻隐藏状态到隐藏状态的权重转置
                        + bias_hh[layer]           # 加上隐藏状态到隐藏状态的偏置
                    )
                output.append(h_t[-1])  # 将当前时刻最后一层的隐藏状态添加到输出列表
                h_t_minus_1 = h_t  # 更新前一时刻的隐藏状态为当前时刻的隐藏状态
            output = torch.stack(output)  # 将输出列表堆叠为张量
            if batch_first:
                output = output.transpose(0, 1)  # 如果 batch_first=True，则将维度重新调整为(batch_size, seq_len, hidden_size)
            return output, h_t  # 返回输出张量和最终的隐藏状态
    """
    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for the input sequence batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}
    """
    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.
    
    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`
    
    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`
    
    .. note::
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.
    
    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.
    
    .. include:: ../cudnn_rnn_determinism.rst
    
    .. include:: ../cudnn_persistent_rnn.rst
    
    Examples::
    
        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """
    
    @overload
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        ...
    
    @overload
    def __init__(self, *args, **kwargs):
        ...
    # 初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 如果关键字参数中包含"proj_size"，则抛出数值错误异常
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        # 如果位置参数的数量大于3，则将第四个位置参数作为非线性函数，并移除它
        if len(args) > 3:
            self.nonlinearity = args[3]
            args = args[:3] + args[4:]
        else:
            # 否则从关键字参数中弹出"nonlinearity"，默认为"tanh"
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        # 根据选择的非线性函数确定模式
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            # 如果非线性函数不是"tanh"或"relu"，则引发数值错误异常
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        # 调用父类的初始化方法，传递模式和其余的位置参数和关键字参数
        super().__init__(mode, *args, **kwargs)

    # 标记为重载方法，接受输入张量和隐藏状态张量，返回张量元组
    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        pass

    # 标记为重载方法，接受压缩序列和隐藏状态张量，返回压缩序列和张量元组
    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> Tuple[PackedSequence, Tensor]:
        pass
# XXX: LSTM and GRU implementation is different from RNNBase, this is because:
# 1. we want to support nn.LSTM and nn.GRU in TorchScript and TorchScript in
#    its current state could not support the python Union Type or Any Type
# 2. TorchScript static typing does not allow a Function or Callable type in
#    Dict values, so we have to separately call _VF instead of using _rnn_impls
# 3. This is temporary only and in the transition state that we want to make it
#    on time for the release
#
# More discussion details in https://github.com/pytorch/pytorch/pull/23266
#
# TODO: remove the overriding implementations for LSTM and GRU when TorchScript
# support expressing these two modules generally.

class LSTM(RNNBase):
    r"""__init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,proj_size=0,device=None,dtype=None)

    Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\delta^{(l-1)}_t` where each :math:`\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.
    Args:
        input_size: 输入 `x` 的特征数
        hidden_size: 隐藏状态 `h` 的特征数
        num_layers: 循环层的数量。例如，设置 ``num_layers=2`` 表示堆叠两个LSTM层以形成 `堆叠LSTM`，第二个LSTM接收第一个LSTM的输出并计算最终结果。默认为 1
        bias: 如果为 ``False``，则该层不使用偏置权重 `b_ih` 和 `b_hh`。默认为 ``True``
        batch_first: 如果为 ``True``，则输入和输出张量的形状为 `(batch, seq, feature)`，否则为 `(seq, batch, feature)`。注意，这不适用于隐藏状态或单元状态。详情请见下面的输入/输出部分。默认为 ``False``
        dropout: 如果非零，则在每个LSTM层的输出上引入一个 `Dropout` 层，且丢弃概率等于 :attr:`dropout`。默认为 0
        bidirectional: 如果为 ``True``，则变成双向LSTM。默认为 ``False``
        proj_size: 如果 ``> 0``，将使用投影大小对应的LSTM。默认为 0

    Inputs: input, (h_0, c_0)
        * **input**: 形状为 :math:`(L, H_{in})` 的张量，对于非批处理输入，
          当 ``batch_first=False`` 时为 :math:`(L, N, H_{in})`，
          当 ``batch_first=True`` 时为 :math:`(N, L, H_{in})`，包含输入序列的特征。输入也可以是打包的变长序列。
          详情请参阅 :func:`torch.nn.utils.rnn.pack_padded_sequence` 或
          :func:`torch.nn.utils.rnn.pack_sequence`。
        * **h_0**: 形状为 :math:`(D * \text{num\_layers}, H_{out})` 的张量，对于非批处理输入，
          当未提供 (h_0, c_0) 时，默认为零。
        * **c_0**: 形状为 :math:`(D * \text{num\_layers}, H_{cell})` 的张量，对于非批处理输入，
          当未提供 (h_0, c_0) 时，默认为零。

        其中：

        .. math::
            \begin{aligned}
                N ={} & \text{批处理大小} \\
                L ={} & \text{序列长度} \\
                D ={} & 2 \text{ 如果 bidirectional=True，否则为 } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size 如果 } \text{proj\_size}>0 \text{，否则为 hidden\_size} \\
            \end{aligned}
    # 返回值描述：
    # - **output**: 形状为 :math:`(L, D * H_{out})` 的张量，对于未批处理的输入，
    #   当 ``batch_first=False`` 时形状为 :math:`(L, N, D * H_{out})`，
    #   当 ``batch_first=True`` 时形状为 :math:`(N, L, D * H_{out})`，包含来自 LSTM 最后一层的输出特征 `(h_t)`，每个 `t` 都有一个。
    #   如果输入是 :class:`torch.nn.utils.rnn.PackedSequence`，输出也将是一个打包的序列。
    #   当 ``bidirectional=True`` 时，`output` 将包含序列中每个时间步的前向和后向隐藏状态的连接。
    # - **h_n**: 形状为 :math:`(D * \text{num\_layers}, H_{out})` 的张量，对于未批处理的输入，
    #   包含每个序列元素的最终隐藏状态。当 ``bidirectional=True`` 时，
    #   `h_n` 将包含前向和后向最终隐藏状态的连接。
    # - **c_n**: 形状为 :math:`(D * \text{num\_layers}, H_{cell})` 的张量，对于未批处理的输入，
    #   包含每个序列元素的最终单元状态。当 ``bidirectional=True`` 时，
    #   `c_n` 将包含前向和后向最终单元状态的连接。
    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If
            ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
            学习的输入到隐藏层权重矩阵，根据层的序号和方向确定形状和具体的权重名称。
    
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, proj_size)`.
            学习的隐藏到隐藏层权重矩阵，根据层的序号和方向确定形状和具体的权重名称。
    
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
            学习的输入到隐藏层的偏置向量，根据层的序号确定具体的偏置名称。
    
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
            学习的隐藏到隐藏层的偏置向量，根据层的序号确定具体的偏置名称。
    
        weight_hr_l[k] : the learnable projection weights of the :math:`\text{k}^{th}` layer
            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was
            specified.
            学习的投影权重矩阵，根据层的序号和是否有投影大小参数确定形状。
    
        weight_ih_l[k]_reverse: Analogous to `weight_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
            反向传播方向的学习的输入到隐藏层权重矩阵，仅在 ``bidirectional=True`` 时存在。
    
        weight_hh_l[k]_reverse:  Analogous to `weight_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
            反向传播方向的学习的隐藏到隐藏层权重矩阵，仅在 ``bidirectional=True`` 时存在。
    
        bias_ih_l[k]_reverse:  Analogous to `bias_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
            反向传播方向的学习的输入到隐藏层的偏置向量，仅在 ``bidirectional=True`` 时存在。
    
        bias_hh_l[k]_reverse:  Analogous to `bias_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
            反向传播方向的学习的隐藏到隐藏层的偏置向量，仅在 ``bidirectional=True`` 时存在。
    
        weight_hr_l[k]_reverse:  Analogous to `weight_hr_l[k]` for the reverse direction.
            Only present when ``bidirectional=True`` and ``proj_size > 0`` was specified.
            反向传播方向的学习的投影权重矩阵，仅在 ``bidirectional=True`` 并且指定了投影大小时存在。
    
    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`
        所有的权重和偏置都是从均匀分布 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 中初始化的，其中 :math:`k = \frac{1}{\text{hidden\_size}}`
    
    .. note::
        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.
        对于双向 LSTM，前向和后向分别对应方向 0 和 1。示例展示了在 ``batch_first=False`` 时如何拆分输出层。
    
    .. note::
        For bidirectional LSTMs, `h_n` is not equivalent to the last element of `output`; the
        former contains the final forward and reverse hidden states, while the latter contains the
        final forward hidden state and the initial reverse hidden state.
        对于双向 LSTM，`h_n` 不等同于 `output` 的最后一个元素；前者包含最终的前向和反向隐藏状态，而后者包含最终的前向隐藏状态和初始的反向隐藏状态。
    
    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.
        对于未批处理的输入，参数 ``batch_first`` 将被忽略。
    
    .. note::
        ``proj_size`` should be smaller than ``hidden_size``.
        ``proj_size`` 应小于 ``hidden_size``。
    
    .. include:: ../cudnn_rnn_determinism.rst
    
    .. include:: ../cudnn_persistent_rnn.rst
    # Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    @overload
    # 初始化函数的重载版本，接受指定参数类型
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        ...

    @overload
    # 初始化函数的重载版本，接受任意参数和关键字参数
    def __init__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法，传递 "LSTM" 和其他参数
        super().__init__("LSTM", *args, **kwargs)

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes: Optional[Tensor]
    ) -> Tuple[int, int, int]:
        # 根据输入和批次大小计算预期的隐藏状态大小
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )
        return expected_hidden_size

    # 在未来，我们应该防止 mypy 在这里应用逆变规则。
    # 参见 torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(
        self,
        input: Tensor,
        hidden: Tuple[Tensor, Tensor],  # type: ignore[override]
        batch_sizes: Optional[Tensor],
    ):
        # 检查输入和隐藏状态大小是否匹配
        self.check_input(input, batch_sizes)
        self.check_hidden_size(
            hidden[0],
            self.get_expected_hidden_size(input, batch_sizes),
            "Expected hidden[0] size {}, got {}",
        )
        self.check_hidden_size(
            hidden[1],
            self.get_expected_cell_size(input, batch_sizes),
            "Expected hidden[1] size {}, got {}",
        )

    # 同上，参见 torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(
        self,
        hx: Tuple[Tensor, Tensor],  # type: ignore[override]
        permutation: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # 对隐藏状态进行排列，如果排列为 None，则返回原始隐藏状态
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(
            hx[1], permutation
        )

    # 同上，参见 torch/nn/modules/module.py::_forward_unimplemented
    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:  # noqa: F811
        pass

    # 同上，参见 torch/nn/modules/module.py::_forward_unimplemented
    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: PackedSequence, hx: Optional[Tuple[Tensor, Tensor]] = None
        ) -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:  # noqa: F811
        pass
    # 声明函数的返回类型注释，此函数返回一个元组，包含一个 PackedSequence 对象和一个元组，
    # 元组包含两个 Tensor 对象。
    def some_function() -> Tuple[PackedSequence, Tuple[Tensor, Tensor]]:
        # 函数体暂时未实现，使用 pass 语句占位
        pass
# 定义一个名为 GRU 的类，继承自 RNNBase
class GRU(RNNBase):
    # 初始化函数，设置 GRU 模型的各种参数和选项
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 device=None, dtype=None):
        # 调用父类的初始化函数
        super(GRU, self).__init__()
        # 定义 GRU 模型的参数和选项
        # 输入的特征数
        self.input_size = input_size
        # 隐藏层状态的特征数
        self.hidden_size = hidden_size
        # 循环层数
        self.num_layers = num_layers
        # 是否使用偏置项
        self.bias = bias
        # 是否将输入的第一维视为批量维度
        self.batch_first = batch_first
        # dropout 概率
        self.dropout = dropout
        # 是否是双向的 GRU
        self.bidirectional = bidirectional
        # 设备类型（GPU/CPU）
        self.device = device
        # 数据类型
        self.dtype = dtype

    # 省略了其它方法的注释，仅注释了初始化函数部分
    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` or
          :math:`(D * \text{num\_layers}, N, H_{out})`
          containing the initial hidden state for the input sequence. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for the input sequence.

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

    .. note::
        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.
    # 根据注释内容来看，这部分代码主要是关于一个 GRU（Gated Recurrent Unit，门控循环单元）模型的实现和说明。
    
    @overload
    # 重载方法注解，用于类型提示和静态分析工具
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        ...
    
    @overload
    # 另一个重载方法注解，支持更灵活的参数形式
    def __init__(self, *args, **kwargs):
        ...
    
    def __init__(self, *args, **kwargs):
        # 检查是否有 proj_size 参数，如果有则抛出异常，因为 GRU 只支持 LSTM 而不支持 proj_size
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        # 调用父类的初始化方法，传递 GRU 类型和其余参数
        super().__init__("GRU", *args, **kwargs)
    
    @overload  # type: ignore[override]
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:  # noqa: F811
        pass
    
    @overload
    @torch._jit_internal._overload_method  # noqa: F811
    def forward(
        self, input: PackedSequence, hx: Optional[Tensor] = None
    ) -> Tuple[PackedSequence, Tensor]:  # noqa: F811
        pass
# 定义 RNNCellBase 类，继承自 Module 类，用于创建基本的循环神经网络单元
class RNNCellBase(Module):
    # 定义类常量，包括输入大小、隐藏状态大小和是否包含偏置项
    __constants__ = ["input_size", "hidden_size", "bias"]

    # 声明类属性：输入大小、隐藏状态大小、是否包含偏置项、输入到隐藏状态的权重和隐藏状态到隐藏状态的权重
    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor  # 权重：输入到隐藏状态的权重矩阵
    weight_hh: Tensor  # 权重：隐藏状态到隐藏状态的权重矩阵

    # 警告：这里故意没有定义 bias_ih 和 bias_hh
    # 参考：https://github.com/pytorch/pytorch/issues/39670

    # 初始化函数，设置 RNN 单元的参数和权重
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
    ) -> None:
        # 设置工厂参数，用于创建张量的设备和数据类型
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # 初始化输入大小、隐藏状态大小和是否包含偏置项
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # 创建参数：输入到隐藏状态的权重矩阵
        self.weight_ih = Parameter(
            torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs)
        )
        # 创建参数：隐藏状态到隐藏状态的权重矩阵
        self.weight_hh = Parameter(
            torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs)
        )
        # 如果包含偏置项，则创建输入到隐藏状态和隐藏状态到隐藏状态的偏置向量
        if bias:
            self.bias_ih = Parameter(
                torch.empty(num_chunks * hidden_size, **factory_kwargs)
            )
            self.bias_hh = Parameter(
                torch.empty(num_chunks * hidden_size, **factory_kwargs)
            )
        # 如果不包含偏置项，则将偏置向量设置为 None
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        # 调用重置参数的方法，初始化权重
        self.reset_parameters()

    # 返回对象的额外信息的字符串表示
    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        # 如果对象有 'bias' 属性且 bias 不为 True，则在字符串中添加 bias 的信息
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        # 如果对象有 'nonlinearity' 属性且 nonlinearity 不是 'tanh'，则在字符串中添加 nonlinearity 的信息
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    # 重置参数的方法，用于初始化权重
    def reset_parameters(self) -> None:
        # 计算标准差，用于初始化权重
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        # 对所有参数进行均匀分布初始化
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


# 定义 RNNCell 类，继承自 RNNCellBase 类，用于创建 Elman 循环神经网络单元
class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: 输入 `x` 中预期的特征数
        hidden_size: 隐藏状态 `h` 中的特征数
        bias: 如果为 ``False``，则该层不使用偏置 `b_ih` 和 `b_hh`。默认为 ``True``
        nonlinearity: 要使用的非线性函数。可以是 ``'tanh'`` 或 ``'relu'``。默认为 ``'tanh'``

    Inputs: input, hidden
        - **input**: 包含输入特征的张量
        - **hidden**: 包含初始隐藏状态的张量，如果未提供，默认为零。

    Outputs: h'
        - **h'** 形状为 `(batch, hidden_size)` 的张量，包含批次中每个元素的下一个隐藏状态
    ```
    """
    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    # 定义常量列表，这些常量表示对象的固有属性
    __constants__ = ["input_size", "hidden_size", "bias", "nonlinearity"]
    # 非线性激活函数的类型，默认为"tanh"
    nonlinearity: str

    # 初始化函数，设置 RNN 单元的各种参数和权重
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        # 准备用于创建张量的关键字参数字典
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的初始化方法，传入输入大小、隐藏层大小、是否有偏置、切块数为1和关键字参数
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        # 设置非线性激活函数的类型
        self.nonlinearity = nonlinearity
    # 定义 RNN 单元的前向传播方法，接受输入和可选的隐藏状态
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # 检查输入的维度是否为 1 或 2
        if input.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        # 如果提供了隐藏状态 hx，则检查其维度是否为 1 或 2
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        # 检查输入是否为批量输入，如果不是，则添加一个批量维度
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        # 如果隐藏状态 hx 未提供，则初始化为零向量
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            # 如果输入是批量输入且隐藏状态不是批量的，则添加一个批量维度
            hx = hx.unsqueeze(0) if not is_batched else hx

        # 根据非线性函数类型选择相应的 RNN 单元计算方法
        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input,
                hx,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input,
                hx,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh,
            )
        else:
            # 如果非线性函数类型未知，则引发运行时错误
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")

        # 如果输入不是批量输入，则移除添加的批量维度
        if not is_batched:
            ret = ret.squeeze(0)

        # 返回计算结果
        return ret
class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f \odot c + i \odot g \\
        h' = o \odot \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
        >>> input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
        >>> hx = torch.randn(3, 20)  # (batch, hidden_size)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(input.size()[0]):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
        >>> output = torch.stack(output, dim=0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Initialize the LSTMCell with specified input and hidden sizes, and optionally with bias.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            bias: If ``False``, the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
            device: Specifies the device where tensors will be allocated (default is None)
            dtype: Specifies the data type for the weights (default is None)

        Returns:
            None
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)
    # 定义 forward 方法，用于执行 LSTM 单元的前向传播
    def forward(
        self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        # 检查输入张量的维度是否为1D或2D，否则抛出数值错误异常
        if input.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        # 如果 hx 不为 None，则检查每个隐藏状态张量的维度是否为1D或2D，否则抛出数值错误异常
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead"
                    )
        # 检查输入张量是否为批处理形式，如果不是，则扩展其维度为批处理形式
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        # 如果 hx 为 None，则初始化全零隐藏状态
        if hx is None:
            zeros = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            # 如果输入已经是批处理形式，则保持 hx 不变，否则将每个隐藏状态张量扩展为批处理形式
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        # 调用底层的 PyTorch LSTM 单元计算函数 _VF.lstm_cell 进行前向传播
        ret = _VF.lstm_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        # 如果输入不是批处理形式，则压缩输出结果的批处理维度
        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        # 返回 LSTM 单元的输出结果
        return ret
# 定义一个 GRU 单元的类，继承自 RNNCellBase 类
class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
        h' = (1 - z) \odot n + z \odot h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, hidden
        - **input** : tensor containing input features
        - **hidden** : tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** : tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)

    """
    
    # 初始化方法，设置 GRU 单元的参数
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # 准备工厂参数字典
        factory_kwargs = {"device": device, "dtype": dtype}
        # 调用父类的初始化方法，传入输入大小、隐藏状态大小、是否使用偏置等参数
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)
    # 定义 GRU 单元的前向传播方法，输入包括输入张量和可选的隐藏状态张量
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # 检查输入张量的维度是否为1D或2D，否则引发值错误异常
        if input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        # 如果提供了隐藏状态张量 hx，则检查其维度是否为1D或2D，否则引发值错误异常
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        
        # 判断输入张量是否为批处理格式
        is_batched = input.dim() == 2
        # 如果输入不是批处理格式，则扩展维度使其变为批处理格式
        if not is_batched:
            input = input.unsqueeze(0)

        # 如果隐藏状态张量 hx 为 None，则初始化为全零张量，与输入张量的大小和设备匹配
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            # 如果隐藏状态张量 hx 不为 None 且输入为非批处理格式，则扩展维度使其变为批处理格式
            hx = hx.unsqueeze(0) if not is_batched else hx

        # 调用 PyTorch 的底层函数 _VF.gru_cell 执行 GRU 单元的计算
        ret = _VF.gru_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
        )

        # 如果输入不是批处理格式，则压缩返回张量的批处理维度
        if not is_batched:
            ret = ret.squeeze(0)

        # 返回 GRU 单元的输出张量 ret
        return ret
```