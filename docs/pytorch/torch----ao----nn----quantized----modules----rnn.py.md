# `.\pytorch\torch\ao\nn\quantized\modules\rnn.py`

```
# mypy: allow-untyped-defs
# 引入 torch 库
import torch

# 指定外部可见的类名列表，只包含 "LSTM"
__all__ = [
    "LSTM",
]

# 定义 LSTM 类，继承自 torch.ao.nn.quantizable.LSTM
class LSTM(torch.ao.nn.quantizable.LSTM):
    r"""A quantized long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples in :class:`~torch.ao.nn.quantizable.LSTM`

    Examples::
        >>> # xdoctest: +SKIP
        >>> custom_module_config = {
        ...     'float_to_observed_custom_module_class': {
        ...         nn.LSTM: nn.quantizable.LSTM,
        ...     },
        ...     'observed_to_quantized_custom_module_class': {
        ...         nn.quantizable.LSTM: nn.quantized.LSTM,
        ...     }
        ... }
        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)
        >>> tq.convert(model, convert_custom_module_class=custom_module_config)
    """
    # 定义一个类属性 _FLOAT_MODULE，指向 torch.ao.nn.quantizable.LSTM 类
    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM  # type: ignore[assignment]

    # 返回字符串 'QuantizedLSTM'，用于标识当前类的名称
    def _get_name(self):
        return 'QuantizedLSTM'

    # 类方法，从浮点数模型转换得到观察到的模型，不支持这个过程
    @classmethod
    def from_float(cls, *args, **kwargs):
        # 抛出 NotImplementedError 异常，提示用户不支持直接从浮点数模型转换
        raise NotImplementedError("It looks like you are trying to convert a "
                                  "non-observed LSTM module. Please, see "
                                  "the examples on quantizable LSTMs.")

    # 类方法，从观察到的模型转换为量化模型
    @classmethod
    def from_observed(cls, other):
        # 断言 other 的类型必须是 cls._FLOAT_MODULE，即 torch.ao.nn.quantizable.LSTM 类型
        assert type(other) == cls._FLOAT_MODULE  # type: ignore[has-type]
        # 调用 torch.ao.quantization.convert 方法将观察到的模型转换为量化模型
        converted = torch.ao.quantization.convert(other, inplace=False,
                                                  remove_qconfig=True)
        # 将转换后的对象的类修改为当前类（QuantizedLSTM）
        converted.__class__ = cls
        # 返回转换后的量化模型对象
        return converted
```