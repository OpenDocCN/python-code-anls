# `.\pytorch\torch\ao\nn\quantized\reference\modules\rnn.py`

```py

    # 初始化权重量化参数字典，并将其属性赋值给当前对象
    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        # 断言权重量化参数字典不为空
        assert weight_qparams_dict is not None
        # 从权重量化参数字典中获取是否已分解的标志位并设置给当前对象
        self.is_decomposed = weight_qparams_dict["is_decomposed"]
        # 遍历权重量化参数字典中的每个键值对
        for key, weight_qparams in weight_qparams_dict.items():
            # 如果键是"is_decomposed"，则跳过不处理
            if key == "is_decomposed":
                continue
            # TODO: 将重复的代码重构到 utils.py 中
            
            # 获取权重量化参数的量化方案和数据类型
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            # 将权重量化方案和数据类型设置为当前对象的属性
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            # 断言权重量化方案合法性，支持 None、torch.per_tensor_affine、torch.per_channel_affine
            assert weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], \
                Exception(f"qscheme: {weight_qscheme} is not support in {self._get_name()}")
            
            # 如果权重量化方案不为 None，则进行相应处理
            if weight_qscheme is not None:
                # 获取量化参数的缩放因子和零点
                scale = weight_qparams["scale"]
                scale_tensor = scale.clone().detach() \
                    if isinstance(scale, torch.Tensor) else \
                    torch.tensor(scale, dtype=torch.float, device=device)
                self.register_buffer(key + "_scale", scale_tensor)
                
                zp = weight_qparams["zero_point"]
                zp_tensor = zp.clone().detach() \
                    if isinstance(zp, torch.Tensor) else \
                    torch.tensor(zp, dtype=torch.int, device=device)
                self.register_buffer(key + "_zero_point", zp_tensor)
                
                # 如果是按通道量化，则获取通道轴并注册为缓冲区
                if weight_qscheme == torch.per_channel_affine:
                    axis = weight_qparams["axis"]
                    axis_tensor = axis.clone().detach() \
                        if isinstance(axis, torch.Tensor) else \
                        torch.tensor(axis, dtype=torch.int, device=device)
                    self.register_buffer(key + "_axis", axis_tensor)
                else:
                    # 对于 TorchScript 兼容性添加的属性，实际未使用
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device))
                
                # 将通道轴属性的整数值设置为当前对象的属性
                setattr(self, key + "_axis_int", getattr(self, key + "_axis").item())

    # 返回当前对象的名称字符串
    def _get_name(self):
        return "QuantizedRNNCellBase(Reference)"

    # 返回当前对象的输入到隐藏层权重的量化版本
    def get_quantized_weight_ih(self):
        return get_quantized_weight(self, "weight_ih")

    # 返回当前对象的隐藏到隐藏层权重的量化版本
    def get_quantized_weight_hh(self):
        return get_quantized_weight(self, "weight_hh")

    # 返回当前对象的输入到隐藏层权重的非量化版本
    def get_weight_ih(self):
        return _get_quantize_and_dequantized_weight(self, "weight_ih")

    # 返回当前对象的隐藏到隐藏层权重的非量化版本
    def get_weight_hh(self):
        return _get_quantize_and_dequantized_weight(self, "weight_hh")
class RNNCell(RNNCellBase):
    """
    RNNCell 类，继承自 RNNCellBase。

    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    我们将为所有权重（weight_ih 和 weight_hh）存储权重量化参数，需要传入一个 `weight_qparams_dict`，
    该字典映射从权重名称（例如 weight_ih）到该权重的量化参数。
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh",
                 device=None, dtype=None, weight_qparams_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化函数，用于创建一个 RNNCell 对象。

        Args:
            input_size: 输入的特征维度大小
            hidden_size: 隐藏状态的维度大小
            bias: 是否使用偏置项，默认为 True
            nonlinearity: 非线性激活函数类型，默认为 "tanh"
            device: 张量所在的设备
            dtype: 张量的数据类型
            weight_qparams_dict: 权重量化参数的字典

        Returns:
            None
        """
        factory_kwargs = {'device': device, 'dtype': dtype, 'weight_qparams_dict': weight_qparams_dict}
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def _get_name(self):
        """
        获取当前对象的名称。

        Returns:
            str: 当前对象的名称 "QuantizedRNNCell(Reference)"
        """
        return "QuantizedRNNCell(Reference)"

    # TODO: refactor nn.RNNCell to have a _forward that takes weight_ih and weight_hh as input
    # and remove duplicated code, same for the other two Cell modules
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        RNNCell 的前向传播函数，计算给定输入和隐藏状态的输出。

        Args:
            input: 输入张量，可以是 1-D 或 2-D 张量
            hx: 初始隐藏状态张量，默认为 None

        Returns:
            Tensor: 输出张量，与输入形状相同
        """
        assert input.dim() in (1, 2), \
            f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        if self.nonlinearity == "tanh":
            ret = _VF.rnn_tanh_cell(
                input, hx,
                self.get_weight_ih(), self.get_weight_hh(),
                self.bias_ih, self.bias_hh,
            )
        elif self.nonlinearity == "relu":
            ret = _VF.rnn_relu_cell(
                input, hx,
                self.get_weight_ih(), self.get_weight_hh(),
                self.bias_ih, self.bias_hh,
            )
        else:
            ret = input  # TODO: remove when jit supports exception flow
            raise RuntimeError(
                f"Unknown nonlinearity: {self.nonlinearity}")

        if not is_batched:
            ret = ret.squeeze(0)

        return ret

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        """
        从浮点数模型创建一个 RNNCell 对象。

        Args:
            cls: 当前类的引用
            mod: 浮点数模型
            weight_qparams_dict: 权重量化参数的字典

        Returns:
            RNNCell: 创建的 RNNCell 对象
        """
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.nonlinearity,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict)
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        return ref_mod

class LSTMCell(RNNCellBase):
    """
    LSTMCell 类，继承自 RNNCellBase。

    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    我们将为所有权重（weight_ih 和 weight_hh）存储权重量化参数，需要传入一个 `weight_qparams_dict`，
    该字典映射从权重名称（例如 weight_ih）到该权重的量化参数。
    """
    # 初始化方法，设置量化 LSTM 单元的参数
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None, weight_qparams_dict: Optional[Dict[str, Any]] = None) -> None:
        # 准备传递给超类初始化的关键字参数
        factory_kwargs = {'device': device, 'dtype': dtype, 'weight_qparams_dict': weight_qparams_dict}
        # 调用超类初始化方法，传递输入大小、隐藏大小、是否包含偏置项、分块数和其他关键字参数
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    # 获取量化 LSTM 单元的名称
    def _get_name(self):
        return "QuantizedLSTMCell(Reference)"

    # 前向传播方法，接收输入和隐藏状态，返回更新后的隐藏状态
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        # 断言输入张量的维度为1或2
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        # 检查输入是否为批次数据
        is_batched = input.dim() == 2
        # 如果不是批次数据，则扩展维度使其变为批次数据
        if not is_batched:
            input = input.unsqueeze(0)

        # 如果隐藏状态为空，则创建全零张量作为初始隐藏状态
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # 如果隐藏状态不为空，且输入数据不是批次数据，则对隐藏状态进行扩展以匹配输入数据的批次大小
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        # 调用 PyTorch 内部的 LSTM 单元计算函数
        ret = _VF.lstm_cell(
            input, hx,
            self.get_weight_ih(), self.get_weight_hh(),
            self.bias_ih, self.bias_hh,
        )

        # 如果输入不是批次数据，则压缩返回结果的第一维度，以匹配原始的输入形状
        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        # 返回更新后的隐藏状态
        return ret

    # 从浮点模型转换为量化模型的类方法
    @classmethod
    def from_float(cls, mod, weight_qparams_dict, use_precomputed_fake_quant=False):
        # 创建一个新的量化 LSTM 单元对象作为参考模型
        ref_mod = cls(
            mod.input_size,  # 输入大小与浮点模型相同
            mod.hidden_size,  # 隐藏大小与浮点模型相同
            mod.bias,  # 是否包含偏置项与浮点模型相同
            mod.weight_ih.device,  # 使用浮点模型的权重设备
            mod.weight_ih.dtype,  # 使用浮点模型的权重数据类型
            weight_qparams_dict  # 权重量化参数字典
        )
        # 将浮点模型的权重和偏置复制到新的量化模型中
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        # 返回量化模型的参考模型
        return ref_mod
# 定义一个继承自RNNCellBase的GRUCell类
class GRUCell(RNNCellBase):
    
    """
    We'll store weight_qparams for all the weights (weight_ih and weight_hh),
    we need to pass in a `weight_qparams_dict` that maps from weight name,
    e.g. weight_ih, to the weight_qparams for that weight
    """
    
    # 初始化方法，定义GRU单元的参数和权重量化参数字典
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None, weight_qparams_dict: Optional[Dict[str, Any]] = None) -> None:
        # 准备传递给父类构造函数的关键字参数
        factory_kwargs = {'device': device, 'dtype': dtype, 'weight_qparams_dict': weight_qparams_dict}
        # 调用父类的初始化方法，传递必要的参数
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    # 返回当前类的名称字符串
    def _get_name(self):
        return "QuantizedGRUCell(Reference)"

    # 前向传播方法，接收输入和初始隐藏状态，返回计算后的输出
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # 断言输入张量的维度为1或2
        assert input.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        
        # 检查是否输入是批量的
        is_batched = input.dim() == 2
        # 如果不是批量的，将输入张量增加一个维度
        if not is_batched:
            input = input.unsqueeze(0)

        # 如果初始隐藏状态为None，创建与输入大小相同的全零张量作为初始隐藏状态
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            # 如果初始隐藏状态不为None且未批量化，增加一个维度以匹配输入
            hx = hx.unsqueeze(0) if not is_batched else hx

        # 调用底层的_VirtualFunction.gru_cell方法执行GRU单元的计算
        ret = _VF.gru_cell(
            input, hx,
            self.get_weight_ih(), self.get_weight_hh(),
            self.bias_ih, self.bias_hh,
        )

        # 如果输入不是批量的，将输出张量减少一个维度
        if not is_batched:
            ret = ret.squeeze(0)

        # 返回前向传播的结果张量
        return ret

    # 类方法，用于从浮点模型mod和权重量化参数字典创建一个参考的GRUCell实例
    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # 使用浮点模型的参数初始化参考的GRUCell实例
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.bias,
            mod.weight_ih.device,
            mod.weight_ih.dtype,
            weight_qparams_dict)
        # 将浮点模型的权重和偏置复制到参考模型中
        ref_mod.weight_ih = mod.weight_ih
        ref_mod.weight_hh = mod.weight_hh
        ref_mod.bias_ih = mod.bias_ih
        ref_mod.bias_hh = mod.bias_hh
        # 返回初始化后的参考模型实例
        return ref_mod

# 定义一个继承自nn.RNNBase的RNNBase类
class RNNBase(nn.RNNBase):
    # 初始化函数，设置量化 LSTM 层的参数和配置
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0,
                 device=None, dtype=None,
                 weight_qparams_dict: Optional[Dict[str, Any]] = None) -> None:
        # 调用父类构造函数，传入 LSTM 的各项参数
        super().__init__(
            mode, input_size, hidden_size, num_layers, bias, batch_first, dropout,
            bidirectional, proj_size, device, dtype
        )
        # 如果未提供权重量化参数字典，则使用默认的量化参数设置
        if weight_qparams_dict is None:
            weight_qparams = {
                'qscheme': torch.per_tensor_affine,
                'dtype': torch.quint8,
                'scale': 1.0,
                'zero_point': 0
            }
            # 初始化权重量化参数字典，指定是否分解为分量
            weight_qparams_dict = {"is_decomposed": False}  # type: ignore[dict-item]
            # 遍历所有扁平化权重的名称
            for wn in self._flat_weights_names:
                # 如果权重名称以 "weight" 开头，将其对应的量化参数设置为默认值
                if wn.startswith("weight"):
                    weight_qparams_dict[wn] = weight_qparams
        # 调用函数，初始化权重量化参数字典
        self._init_weight_qparams_dict(weight_qparams_dict, device)

    # 初始化权重量化参数字典的私有函数
    def _init_weight_qparams_dict(self, weight_qparams_dict, device):
        # 从权重量化参数字典中提取是否分解为分量的标志位
        self.is_decomposed = weight_qparams_dict["is_decomposed"]
        # 遍历权重量化参数字典的每个键值对
        for key, weight_qparams in weight_qparams_dict.items():
            # 如果键是 "is_decomposed"，跳过处理
            if key == "is_decomposed":
                continue
            # 提取当前权重的量化方案和数据类型
            weight_qscheme = weight_qparams["qscheme"]
            weight_dtype = weight_qparams["dtype"]
            # 设置当前权重的量化方案和数据类型为对象的属性
            setattr(self, key + "_qscheme", weight_qscheme)
            setattr(self, key + "_dtype", weight_dtype)
            # 检查权重量化方案是否为支持的类型之一
            assert weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], \
                Exception(f"qscheme: {weight_qscheme} is not support in {self._get_name()}")
            # 如果权重有量化方案，则注册缓冲区并设置量化参数
            if weight_qscheme is not None:
                self.register_buffer(
                    key + "_scale",
                    torch.tensor(weight_qparams["scale"], dtype=torch.float, device=device))
                self.register_buffer(
                    key + "_zero_point",
                    torch.tensor(weight_qparams["zero_point"], dtype=torch.int, device=device))
                # 如果权重方案是按通道的，还需注册轴缓冲区
                if weight_qscheme == torch.per_channel_affine:
                    self.register_buffer(
                        key + "_axis",
                        torch.tensor(weight_qparams["axis"], dtype=torch.int, device=device))
                else:
                    # 为了 TorchScriptability 添加，但未使用
                    self.register_buffer(
                        key + "_axis", torch.tensor(0, dtype=torch.int, device=device))
                # 设置当前权重轴的整数值属性
                setattr(self, key + "_axis_int", getattr(self, key + "_axis").item())
class LSTM(RNNBase):
    """ Reference Quantized LSTM Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """

    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)

    # Same as above, see torch/nn/modules/module.py::_forward_unimplemented
    def permute_hidden(self,  # type: ignore[override]
                       hx: Tuple[Tensor, Tensor],
                       permutation: Optional[Tensor]
                       ) -> Tuple[Tensor, Tensor]:
        """Permute the hidden state tensor according to the provided permutation tensor.

        Args:
            hx (Tuple[Tensor, Tensor]): Tuple of tensors representing hidden states.
            permutation (Optional[Tensor]): Tensor specifying the permutation indices.

        Returns:
            Tuple[Tensor, Tensor]: Permuted tensors for hidden states.
        """
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(hx[1], permutation)

    def get_expected_cell_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> Tuple[int, int, int]:
        """Calculate the expected size of the LSTM cell based on input and batch sizes.

        Args:
            input (Tensor): Input tensor to the LSTM.
            batch_sizes (Optional[Tensor]): Tensor representing batch sizes.

        Returns:
            Tuple[int, int, int]: Expected size of the LSTM cell.
        """
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.hidden_size)
        return expected_hidden_size

    # In the future, we should prevent mypy from applying contravariance rules here.
    # See torch/nn/modules/module.py::_forward_unimplemented
    def check_forward_args(self,  # type: ignore[override]
                           input: Tensor,
                           hidden: Tuple[Tensor, Tensor],
                           batch_sizes: Optional[Tensor],
                           ):
        """Check arguments for the forward pass of the LSTM.

        Args:
            input (Tensor): Input tensor to the LSTM.
            hidden (Tuple[Tensor, Tensor]): Tuple of tensors representing hidden states.
            batch_sizes (Optional[Tensor]): Tensor representing batch sizes.
        """
        self.check_input(input, batch_sizes)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(input, batch_sizes),
                               'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(input, batch_sizes),
                               'Expected hidden[1] size {}, got {}')

    def get_quantized_weight_bias_dict(self):
        """Return a dictionary mapping flat weight names to their quantized weights or biases.

        Returns:
            Dict[str, Tensor]: Dictionary mapping weight names to quantized weights or biases.
        """
        quantized_weight_bias_dict = {}
        for wn in self._flat_weights_names:
            if hasattr(self, wn):
                if wn.startswith("weight"):
                    weight_or_bias = get_quantized_weight(self, wn)
                else:
                    weight_or_bias = getattr(self, wn)
            else:
                weight_or_bias = None
            quantized_weight_bias_dict[wn] = weight_or_bias
        return quantized_weight_bias_dict
    # 获取平坦化权重列表的方法
    def get_flat_weights(self):
        # 初始化一个空列表用于存储平坦化后的权重
        flat_weights = []
        # 遍历所有平坦化权重的名称
        for wn in self._flat_weights_names:
            # 检查当前对象是否具有属性 wn
            if hasattr(self, wn):
                # 获取当前属性的值（即权重）
                weight = getattr(self, wn)
                # 如果权重名称以 "weight" 开头，则需要进行量化处理
                if wn.startswith("weight"):
                    # 调用辅助函数 _get_weight_and_quantization_params 获取权重及量化参数
                    params = _get_weight_and_quantization_params(self, wn)
                    # 调用 _quantize_and_dequantize_weight 对权重进行量化和反量化
                    weight = _quantize_and_dequantize_weight(*params)
            else:
                # 如果当前对象没有属性 wn，则将权重设为 None
                weight = None
            # 将处理后的权重加入到平坦化权重列表中
            flat_weights.append(weight)
        # 返回所有权重组成的列表
        return flat_weights

    # 返回当前对象的名称
    def _get_name(self):
        return "QuantizedLSTM(Reference)"

    # 类方法：根据浮点数模型创建一个参考模型
    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # 使用类构造函数创建一个新的参考模型 ref_mod
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
            weight_qparams_dict=weight_qparams_dict)
        # 将原始浮点数模型 mod 的所有平坦化权重复制给 ref_mod
        for wn in mod._flat_weights_names:
            setattr(ref_mod, wn, getattr(mod, wn))
        # 返回创建的参考模型 ref_mod
        return ref_mod
class GRU(RNNBase):
    """ Reference Quantized GRU Module
    We'll store weight_qparams for all the weights in _flat_weights, we need to pass in
    a `weight_qparams_dict` that maps from weight name, e.g. weight_ih_l0,
    to the weight_qparams for that weight
    """

    def __init__(self, *args, **kwargs):
        # 如果在初始化时传入了'proj_size'参数，抛出数值错误异常
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        # 调用父类 RNNBase 的初始化方法，指定模型类型为'GRU'
        super().__init__('GRU', *args, **kwargs)

    def get_quantized_weight_bias_dict(self):
        """ dictionary from flat_weight_name to quantized weight or (unquantized) bias
        e.g.
        {
          "weight_ih_l0": quantized_weight,
          "bias_ih_l0": unquantized_bias,
          ...
        }
        """
        # 创建空字典用于存储权重和偏置的量化版本或（未量化的）偏置
        quantized_weight_bias_dict = {}
        # 遍历所有平坦化权重名
        for wn in self._flat_weights_names:
            # 如果模型实例具有当前权重名 wn
            if hasattr(self, wn):
                # 如果权重名以 "weight" 开头，获取量化后的权重
                if wn.startswith("weight"):
                    weight_or_bias = get_quantized_weight(self, wn)
                else:
                    # 否则，获取未量化的偏置
                    weight_or_bias = getattr(self, wn)
            else:
                # 如果模型实例没有当前权重名 wn，则设为 None
                weight_or_bias = None
            # 将权重名及其对应的量化或未量化对象存入字典
            quantized_weight_bias_dict[wn] = weight_or_bias
        return quantized_weight_bias_dict

    def get_flat_weights(self):
        # 创建空列表用于存储所有平坦化权重
        flat_weights = []
        # 遍历所有平坦化权重名
        for wn in self._flat_weights_names:
            # 如果模型实例具有当前权重名 wn
            if hasattr(self, wn):
                # 获取当前权重对象
                weight = getattr(self, wn)
                # 如果权重名以 "weight" 开头，获取权重及其量化参数，并执行量化和去量化操作
                if wn.startswith("weight"):
                    params = _get_weight_and_quantization_params(self, wn)
                    weight = _quantize_and_dequantize_weight(*params)
            else:
                # 如果模型实例没有当前权重名 wn，则设权重为 None
                weight = None
            # 将处理后的权重对象加入列表
            flat_weights.append(weight)
        return flat_weights

    def _get_name(self):
        # 返回当前模型的名称字符串
        return "QuantizedGRU(Reference)"

    @classmethod
    def from_float(cls, mod, weight_qparams_dict):
        # 创建一个 QuantizedGRU(Reference) 的实例 ref_mod，基于浮点数模型 mod 和权重量化参数字典
        ref_mod = cls(
            mod.input_size,
            mod.hidden_size,
            mod.num_layers,
            mod.bias,
            mod.batch_first,
            mod.dropout,
            mod.bidirectional,
            weight_qparams_dict=weight_qparams_dict)
        # 遍历浮点数模型 mod 的所有平坦化权重名
        for wn in mod._flat_weights_names:
            # 将浮点数模型 mod 的对应权重名的权重赋值给 ref_mod 实例
            setattr(ref_mod, wn, getattr(mod, wn))
        return ref_mod
```