# `.\pytorch\torch\ao\quantization\utils.py`

```py
# mypy: allow-untyped-defs
"""
Utils shared by different modes of quantization (eager/graph)
"""
# 导入 functools 库，用于函数式编程工具
import functools
# 导入 warnings 库，用于警告处理
import warnings
# 导入 OrderedDict 类，用于有序字典操作
from collections import OrderedDict
# 导入 getfullargspec 和 signature 函数，用于获取函数参数信息
from inspect import getfullargspec, signature
# 导入 Any, Callable, Dict, Optional, Tuple, Union 等类型，用于类型注解
from typing import Any, Callable, Dict, Optional, Tuple, Union

# 导入 torch 库
import torch
# 导入 QuantType 类型，用于量化类型注解
from torch.ao.quantization.quant_type import QuantType
# 导入 Node 类，用于 FX 图中节点的操作
from torch.fx import Node
# 导入 is_parametrized 函数，用于判断是否参数化
from torch.nn.utils.parametrize import is_parametrized

# 定义 NodePattern 类型，可以是 Node 或者 Node 元组
NodePattern = Union[Tuple[Node, Node], Tuple[Node, Tuple[Node, Node]], Any]
# 将 NodePattern 的模块设置为 "torch.ao.quantization.utils"
NodePattern.__module__ = "torch.ao.quantization.utils"

# 定义 QuantizerCls 类型，代表量化器的类实例，来自 torch/quantization/fx/quantize.py
# 为了避免循环导入，此处单独定义
QuantizerCls = Any

# 定义 Pattern 类型，用于融合模式的表示，可以是复杂的结构
# 可以参考 pattern.md 获取更多文档信息
# TODO: 不确定 typing 是否支持递归数据类型
Pattern = Union[
    Callable, Tuple[Callable, Callable], Tuple[Callable, Tuple[Callable, Callable]], Any
]
# 将 Pattern 的模块设置为 "torch.ao.quantization.utils"
Pattern.__module__ = "torch.ao.quantization.utils"

# 定义 MatchAllNode 类，用于表示匹配所有节点的节点模式
# 在 FX 图模式量化中用于定义融合模式
class MatchAllNode:
    """ A node pattern that matches all nodes, used in defining
    fusion patterns in FX Graph Mode Quantization
    """
    pass

# module_type_list 是包含各种 Torch 模块类型的集合
module_type_list = {
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.AdaptiveAvgPool1d,
    torch.nn.AdaptiveAvgPool2d,
    torch.nn.AdaptiveAvgPool3d,
    torch.nn.AvgPool1d,
    torch.nn.AvgPool2d,
    torch.nn.AvgPool3d,
    torch.nn.MaxPool1d,
    torch.nn.MaxPool2d,
    torch.nn.MaxPool3d,
    torch.nn.Identity,
    torch.nn.Hardsigmoid,
    torch.nn.Sigmoid,
    torch.nn.Tanh,
}

# func_list 是包含各种 Torch 函数的集合
func_list = {
    torch.nn.functional.adaptive_avg_pool1d,
    torch.nn.functional.adaptive_avg_pool2d,
    torch.nn.functional.adaptive_avg_pool3d,
    torch.nn.functional.elu,
    torch.nn.functional.hardswish,
    torch.nn.functional.instance_norm,
    torch.nn.functional.layer_norm,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.silu,
    torch.nn.functional.mish,
    torch.nn.functional.dropout,
    torch.nn.functional.max_pool1d,
    torch.nn.functional.max_pool2d,
    torch.nn.functional.max_pool3d,
    torch.nn.functional.relu,
    torch.nn.functional.hardtanh,
    torch.nn.functional.hardtanh_,
    torch.nn.functional.hardsigmoid,
    torch.nn.functional.sigmoid,
    torch.transpose,
    torch.repeat_interleave,
    torch.sigmoid,
    torch.squeeze,
    torch.stack,
    torch.sum,
    torch.tanh,
    torch.unsqueeze,
    torch.cat,
}

# method_list 是包含各种 Torch 方法和字符串的集合
method_list = {
    torch.mean,
    'relu',
    'relu_',
    'contiguous',
    'detach',
    'detach_',
    'hardsigmoid',
    'hardsigmoid_',
    'permute',
    'repeat',
    'repeat_interleave',
    'reshape',
    'resize_',
    'shape',
    'sigmoid',
    'sigmoid_',
    'size',
    'squeeze',
    'squeeze_',
    'tanh',
    'tanh_',
    'transpose',
    'unsqueeze',
    'unsqueeze_',  # Tensor 类的方法，用于在指定维度上增加一个维度
    'view',        # Tensor 类的方法，用于调整张量的形状，返回一个具有相同数据但形状不同的新张量
# TODO: not used now, remove
# 这个函数目前没有使用，可以考虑在代码整理时移除掉
def check_node(node, modules):
    # TODO: reuse is_fixed_qparam_node after we move this function to _lower_to_native_backend.py
    # 检查节点的操作类型和目标是否在预定义的列表中，用于判断节点是否是函数调用
    is_call_function = node.op == "call_function" and node.target in func_list
    # 检查节点的操作类型和目标是否在预定义的列表中，用于判断节点是否是方法调用
    is_call_method = node.op == "call_method" and node.target in method_list
    # 检查节点的操作类型和目标是否在预定义的列表中，用于判断节点是否是模块调用
    is_call_module = node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    # 返回判断结果的元组
    return is_call_function, is_call_method, is_call_module

def get_combined_dict(default_dict, additional_dict):
    """
    Combines two dictionaries.

    This function takes two dictionaries as input and returns a new dictionary
    that contains all the key-value pairs from both input dictionaries.
    If there are any duplicate keys in the `additional_dict`, the values
    from the `additional_dict` will overwrite those in the `default_dict`.
    Args:
        default_dict (dict): The main dictionary that will be used as the base
        additional_dict (dict): The dictionary used to update `default_dict`

    Returns:
        dict: The resulting dictionary
    Example:
        >>> x = dict(a=1, b=1)
        >>> y = dict(b=2, c=3)
        >>> get_combined_dict(x, y)
        {'a': 1, 'b': 2, 'c': 3}
    """
    # 复制默认字典，然后更新为合并后的字典
    d = default_dict.copy()
    d.update(additional_dict)
    return d

def is_per_tensor(qscheme):
    # 检查量化方案是否为每个张量的量化
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def is_per_channel(qscheme):
    # 检查量化方案是否为每个通道的量化
    return qscheme in [torch.per_channel_affine,
                       torch.per_channel_affine_float_qparams,
                       torch.per_channel_symmetric]

def getattr_from_fqn(obj: Any, fqn: str) -> Any:
    """
    Given an obj and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    """
    # 根据完全限定名称从对象中获取属性
    return functools.reduce(getattr, fqn.split("."), obj)

def to_underlying_dtype(qdtype):
    # 将量化类型映射为基础数据类型
    DTYPE_MAPPING = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
        torch.quint4x2: torch.uint8,
        torch.quint2x4: torch.uint8,
        torch.uint8: torch.uint8,
        torch.int8: torch.int8,
        torch.int16: torch.int16,
        torch.int32: torch.int32,
        torch.float8_e5m2: torch.float8_e5m2,
        torch.float8_e4m3fn: torch.float8_e4m3fn,
    }
    # 断言量化类型在映射中，否则抛出异常
    assert qdtype in DTYPE_MAPPING, "Unsupported dtype: " + str(qdtype)
    return DTYPE_MAPPING[qdtype]

def get_qparam_dict(observer_or_fake_quant):
    from torch.ao.quantization.observer import PlaceholderObserver

    qscheme = getattr(observer_or_fake_quant, "qscheme", None)
    dtype = observer_or_fake_quant.dtype
    qparams = {"qscheme": qscheme, "dtype": dtype}

    # 如果没有量化方案或者是占位符观察器，则返回一个默认字典
    if not qscheme or isinstance(observer_or_fake_quant, PlaceholderObserver):
        return {"qscheme": None, "dtype": dtype}

    # 如果是每个张量的量化方案，则统一为 torch.per_tensor_affine
    if is_per_tensor(qscheme):
        qscheme = torch.per_tensor_affine
    elif is_per_channel(qscheme):
        # 如果量化方案是按通道的，因为我们没有对称量化的张量，
        # 所以将其改为仿射量化
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine
        # 设置量化参数中的轴信息为观察器或者仿真量化器的通道轴
        qparams["axis"] = observer_or_fake_quant.ch_axis
    else:
        # 如果量化方案未被识别，则抛出运行时错误并提示未识别的量化方案
        raise RuntimeError(f"Unrecognized qscheme: {qscheme}")

    # 更新量化参数中的量化方案，因为在量化张量中不存在对称量化方案
    qparams["qscheme"] = qscheme

    # 计算量化参数的尺度和零点
    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams["scale"] = scale
    qparams["zero_point"] = zero_point

    # 如果观察器或者仿真量化器具有quant_min属性，则将其设置为量化参数的最小值
    if hasattr(observer_or_fake_quant, "quant_min"):
        qparams["quant_min"] = observer_or_fake_quant.quant_min
    # 如果观察器或者仿真量化器具有quant_max属性，则将其设置为量化参数的最大值
    if hasattr(observer_or_fake_quant, "quant_max"):
        qparams["quant_max"] = observer_or_fake_quant.quant_max

    # 返回计算好的量化参数字典
    return qparams
# 获取需要替换为的观察/量化定制模块类
def get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """ Get the observed/quantized custom module class that we need
    to swap `custom_module` to
    Input:
        custom_module: 输入，可以是浮点数或观察到的定制模块的实例
        custom_module_class_mapping: 浮点数到观察或观察到量化定制模块类的映射
        qconfig: 为定制模块配置的qconfig
    
    Output:
        对应于输入定制模块实例的观察/量化定制模块类
    """
    quant_type = get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    assert type(custom_module) in class_mapping, "did not find corresponding observed " \
        f"module class for {type(custom_module)} in mapping: {class_mapping}"
    return class_mapping[type(custom_module)]

# 获取激活函数的数据类型
def activation_dtype(qconfig):
    assert qconfig is not None
    activation = qconfig.activation()
    return activation.dtype

# 获取权重的数据类型
def weight_dtype(qconfig):
    assert qconfig is not None
    weight = qconfig.weight()
    return weight.dtype

# 判断激活是否静态量化
def activation_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized or not, this includes quantizing to quint8, qint8 and qint32 and float16
    """
    return (
        activation_dtype(qconfig) in [
            torch.quint8,
            torch.qint8,
            torch.qint32,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        ]
        and (not activation_is_dynamically_quantized(qconfig))
    )

# 判断激活是否动态量化
def activation_is_dynamically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    dynamically quantized or not, this includes dynamically quantizing to
    quint8, qint8 and float16
    """
    activation_dtype, _, activation_is_dynamic = \
        get_qconfig_dtypes(qconfig)
    return activation_is_dynamic

# 判断激活是否量化为int8
def activation_is_int8_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int8 or not, this includes quantizing to quint8, qint8
    """
    return activation_dtype(qconfig) in [torch.quint8, torch.qint8, torch.uint8, torch.int8]

# 判断激活是否量化为int32
def activation_is_int32_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int32 or not
    """
    return activation_dtype(qconfig) in [torch.qint32, torch.int32]

# 判断权重是否量化
def weight_is_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be
    quantized or not
    """
    return weight_dtype(qconfig) in [
        torch.quint8,
        torch.qint8,
        torch.float16,
        torch.quint4x2,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ]
def weight_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be statically
    quantized or not
    """
    # 调用 weight_dtype 函数检查 weight 的数据类型是否在静态量化类型列表中
    return weight_dtype(qconfig) in [torch.quint8, torch.qint8, torch.uint8, torch.int8]

def op_is_int8_dynamically_quantized(qconfig) -> bool:
    """ Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
    # 获取 qconfig 的激活、权重数据类型以及激活是否为动态类型
    activation_dtype, weight_dtype, activation_is_dynamic = \
        get_qconfig_dtypes(qconfig)
    # 判断条件：激活数据类型为 uint8 或 quint8，权重数据类型为 qint8 或 int8，并且激活是动态的
    return (
        activation_dtype in [torch.quint8, torch.uint8] and
        weight_dtype in [torch.qint8, torch.int8] and
        activation_is_dynamic
    )

def get_qconfig_dtypes(qconfig):
    r""" returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_is_dynamic)
    """
    assert qconfig is not None
    # 获取 qconfig 的激活和权重数据类型以及激活是否为动态类型的属性值
    activation = qconfig.activation()
    weight = qconfig.weight()
    act_is_dynamic = getattr(activation, "is_dynamic", False)
    return (activation.dtype, weight.dtype, act_is_dynamic)

def get_quant_type(qconfig):
    assert qconfig is not None
    # 获取 qconfig 的激活和权重数据类型
    activation = qconfig.activation()
    weight = qconfig.weight()
    # 定义静态数据类型列表
    static_dtypes = [
        torch.quint8,
        torch.qint8,
        torch.quint4x2,
        torch.qint32,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.float8_e5m2,
        torch.float8_e4m3fn
    ]
    # 检查权重数据类型是否在静态数据类型列表中
    if weight.dtype in static_dtypes:
        # 如果激活具有 'is_dynamic' 属性且为动态类型，则返回动态量化类型
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        # 否则，如果激活数据类型在静态数据类型列表中，则返回静态量化类型
        elif activation.dtype in static_dtypes:
            return QuantType.STATIC
        else:
            return QuantType.WEIGHT_ONLY

    # 如果权重数据类型为 float16
    if weight.dtype == torch.float16:
        # 如果激活具有 'is_dynamic' 属性且为动态类型，则返回动态量化类型
        if hasattr(activation, 'is_dynamic') and activation.is_dynamic:
            return QuantType.DYNAMIC
        # 否则，如果激活数据类型为 float16，则返回静态量化类型
        elif activation.dtype == torch.float16:
            return QuantType.STATIC

    # 抛出异常，表明在 get_quant_type 中遇到了未识别的数据类型组合
    raise Exception(f"Unrecognized dtype combination in get_quant_type: activation({activation.dtype}),"  # noqa: TRY002
                    f"weight({weight.dtype})")

def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    """ Checks if the given minimum and maximum values are valid, meaning that
    they exist and the min value is less than the max value.
    """
    # 检查最小值和最大值是否有效：都存在，并且最小值小于最大值
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(
            "must run observer before calling calculate_qparams. " +
            "Returning default values."
        )
        return False

    # 如果最小值或最大值的维度为 0
    if min_val.dim() == 0 or max_val.dim() == 0:
        # 如果最小值和最大值分别为正无穷和负无穷，则发出警告
        if min_val == float("inf") and max_val == float("-inf"):
            warnings.warn(
                "must run observer before calling calculate_qparams. " +
                "Returning default values."
            )
            return False

        # 否则，断言最小值应小于等于最大值
        assert min_val <= max_val, f"min {min_val} should be less than max {max_val}"
    # 否则情况：如果不满足所有元素的最小值小于等于最大值，则触发断言错误，显示具体的最小值和最大值信息
    else:
        assert torch.all(
            min_val <= max_val
        ), f"min {min_val} should be less than max {max_val}"
    
    # 返回布尔值 True，表示断言通过
    return True
# 根据输入的量化范围和数据类型，计算实际的量化最小值和最大值
def calculate_qmin_qmax(quant_min: int, quant_max: int, has_customized_qrange: bool, dtype: torch.dtype,
                        reduce_range: bool) -> Tuple[int, int]:
    r"""Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
    # 如果有自定义的量化范围
    if has_customized_qrange:
        # 下面的初始化是为了解决 TorchScript 编译问题，并允许使用细化来解耦 initial_qmin 和 initial_qmax
        # 初始的 initial_qmin 和 initial_qmax 的实际值将在下面重置
        if dtype in [torch.qint32, torch.int32]:
            initial_quant_min, initial_quant_max = 0, 2**32 - 1
        else:
            initial_quant_min, initial_quant_max = 0, 255
        
        # 将 self.qmin 和 self.qmax 赋值给本地变量，并根据 TorchScript 的要求对属性进行细化
        custom_quant_min, custom_quant_max = quant_min, quant_max
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (
                custom_quant_min,
                custom_quant_max,
            )

        # 计算量化范围长度
        qrange_len = initial_quant_max - initial_quant_min + 1
        # 如果数据类型为 qint8 或 int8
        if dtype in [torch.qint8, torch.int8]:
            assert (
                0 < qrange_len <= 256
            ), "quantization range should be positive and not exceed the maximum bit range (=256)."
        # 如果数据类型为 qint32 或 int32
        elif dtype in [torch.qint32, torch.int32]:
            assert (
                0 < qrange_len <= 2**32
            ), "quantization range should be positive and not exceed the maximum bit range (=4294967296)."
        
        # 如果需要缩小量化范围
        if reduce_range:
            quant_min, quant_max = quant_min // 2, quant_max // 2
    else:
        # 如果不使用动态范围，则回退到默认的 8 位量化最小值和最大值计算
        if dtype in [torch.qint8, torch.int8]:
            if reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        elif dtype in [torch.quint8, torch.uint8]:
            if reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        elif dtype in [torch.qint32, torch.int32]:
            quant_min, quant_max = -1 * (2 ** 31), (2 ** 31) - 1
        else:
            quant_min, quant_max = 0, 15
    return quant_min, quant_max


# 将目标字符串 'foo.bar' 转换为 ['foo', 'bar']
def _parent_name(target):
    """
    Turn 'foo.bar' into ['foo', 'bar']
    """
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]


# 检查模块是否没有子模块，忽略参数化模块
def has_no_children_ignoring_parametrizations(module):
    """
    Checks if module._modules is empty or
    if module is a parametrization, checks that module._modules only has
    """
    # 检查给定模块是否没有子模块
    """
    如果模块没有子模块，则返回 True
    """
    if len(module._modules) == 0:
        return True
    # 如果模块已经被参数化（具有特定的参数化属性），则检查是否只有一个子模块且其中包含 'parametrizations'
    elif is_parametrized(module):
        return len(module._modules) == 1 and 'parametrizations' in module._modules
    else:
        # 如果模块不是参数化的，则返回 False
        return False
def _get_path_of_module(root: torch.nn.Module, submodule: torch.nn.Module) -> Optional[str]:
    """ Get the path (fully qualified name) of a submodule

    Example::

    >> class M(torch.nn.Module):
           def __init__(self):
               self.linear = torch.nn.Linear(5, 5)
           def forward(self, x):
               return self.linear(x)

    >> m = M()
    >> l = m.linear
    >> _get_path_of_module(m, l)
    "linear"
    """
    # 遍历根模块及其所有子模块，找到与给定子模块匹配的路径名
    for n, p in root.named_modules():
        if submodule is p:
            return n
    return None

def _get_signature_locals(f: Callable, loc: Dict[str, Any]) -> Dict[str, Any]:
    """ Get local keyword arguments

    Example::

    >> def f(self, a, b=9):
           pass
    >> loc = {"a": 6, "c": 7}
    >> _get_signature_locals(f, loc)
    {"a": 6}
    """
    # 返回在函数签名中定义的参数中，与本地关键字参数 loc 中匹配的部分
    return {k: v for k, v in loc.items() if k in signature(f).parameters}

def _get_default_kwargs(f: Callable) -> "OrderedDict[str, Any]":
    """ Get all default keyword arguments from function signature

    Example::

    >> def f(self, a, b=9):
           pass
    >> _get_default_kwargs(f)
    {"b": 9}
    """
    # 获取函数签名中所有具有默认值的关键字参数，并以有序字典形式返回
    kwargs = {}
    for name, param in signature(f).parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default
        elif param.kind is param.VAR_POSITIONAL:
            kwargs[name] = ()
        elif param.kind is param.VAR_KEYWORD:
            kwargs[name] = {}
    return OrderedDict(kwargs)

def _normalize_kwargs(func: Callable, loc: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """ Given a function and local function arguments, normalize the keyword
    arguments by filling in default arguments from function signature

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> loc = {"key2": 6}
    >> _normalize_kwargs(f, loc)
    {"key1": 3, "key2": 6}
    """
    # 获取函数签名中的默认关键字参数和本地关键字参数，用本地参数填充默认参数，返回有序字典
    default_kwargs = _get_default_kwargs(func)
    local_kwargs = _get_signature_locals(func, loc)
    normalized_kwargs = default_kwargs.copy()
    for attr, val in local_kwargs.items():
        if attr in normalized_kwargs:
            # 覆盖默认关键字参数的值
            normalized_kwargs[attr] = val
    return normalized_kwargs

def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    r"""Validates that the user-specified quantization range is properly initialized
    and within the given bound supported by the observer dtype.

    To accommodate lower-bit quantization with respect to the existing torch.qint8 and
    torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
    in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
    values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
    fake quantization. These estimates are compared against parameters learned through backpropagation.
    """
    # 验证用户指定的量化范围是否正确初始化，并且在观察器数据类型支持的给定边界内
    pass
    """
    The related literatures for scale and zero point via backpropagation are as follows:
    
    Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
    Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
    """
    # The variable names are prefixed with "initial" because their values (qmin and qmax) might be adjusted
    # based on whether quantization range is reduced and the datatype (signed/unsigned) used by the observer.
    # Ensure that the user-specified quantization range includes zero.
    assert (
        quant_min <= 0 <= quant_max
    ), "Used-specified quantization range must include 0."
    # Ensure that qmin is strictly less than qmax for the user-specified quantization range.
    assert (
        quant_min < quant_max
    ), "qmin must be strictly less than qmax for user-specified quantization range."
# Functionally equivalent to '_calculate_qparams' in observer.py. Observers must be torchscriptable however and qscheme
# as far as I can tell is not allowed to passed as a parameter in torchscript functions. This makes refactoring observer
# to use this utility a massive pain and very gross. For now Im opting just to duplicate as this code seems unlikey to change
# (last update over 1 year ago) and when torchscript is fully deprecated we can refactor. TODO(jakeszwe, jerryzh168)
def determine_qparams(
        min_val: torch.Tensor, max_val: torch.Tensor, quant_min: int, quant_max: int,
        dtype: torch.dtype, eps: torch.Tensor, has_customized_qrange: bool,
        qscheme: torch.qscheme = torch.per_tensor_affine) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the quantization parameters, given min and max
    value tensors. Works for both per tensor and per channel cases

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel
        quant_min: Minimum quantization value
        quant_max: Maximum quantization value
        dtype: Data type for scale and zero_point tensors
        eps: Small value to avoid division by zero
        has_customized_qrange: Flag indicating if customized quantization range is used
        qscheme: Quantization scheme (default: torch.per_tensor_affine)

    Returns:
        scales: Scales tensor of shape (#channels,)
        zero_points: Zero points tensor of shape (#channels,)
    """

    # Check if min_val and max_val are valid
    if not check_min_max_valid(min_val, max_val):
        # Return default scale and zero_point tensors if min_val or max_val are not valid
        return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

    # Calculate negative part of min_val
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    # Calculate positive part of max_val
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    # Get device from min_val_neg tensor
    device = min_val_neg.device
    # Initialize scale tensor with ones, dtype=torch.double
    scale = torch.ones(min_val_neg.size(), dtype=torch.double, device=device)
    # Initialize zero_point tensor with zeros, dtype=torch.int64
    zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # Handle symmetric quantization schemes
    if (
        qscheme == torch.per_tensor_symmetric
        or qscheme == torch.per_channel_symmetric
    ):
        # Adjust max_val_pos for symmetric quantization
        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        # Calculate scale based on symmetric quantization formula
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        # Ensure scale is not smaller than eps
        scale = torch.max(scale, eps)
        
        # Adjust zero_point based on dtype
        if dtype in [torch.uint8, torch.quint8]:
            if has_customized_qrange:
                # When using customized quantization range, set zero_point to the midpoint
                zero_point = zero_point.new_full(
                    zero_point.size(), (quant_min + quant_max) // 2
                )
            else:
                # Otherwise, set zero_point to 128
                zero_point = zero_point.new_full(zero_point.size(), 128)

    # Handle per-channel affine float quantization scheme
    elif qscheme == torch.per_channel_affine_float_qparams:
        # Calculate scale for per-channel affine float quantization
        scale = (max_val - min_val) / float(quant_max - quant_min)
        # Ensure scale is not smaller than eps
        scale = torch.where(scale > eps, scale, torch.ones_like(scale))
        
        # Calculate zero_point for per-channel affine float quantization
        zero_point = -1 * min_val / scale
    # 如果条件不成立，执行以下计算量化参数的逻辑
    else:
        # 计算量化的缩放因子
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # 确保缩放因子不小于一个极小的正数 eps
        scale = torch.max(scale, eps)
        # 计算量化的零点
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        # 将零点限制在 quant_min 和 quant_max 之间
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

    # 对于标量值，将其转换为大小为 1 的张量，以保持形状与 FakeQuantize 中的默认值一致
    if len(scale.shape) == 0:
        # TODO: 在添加 JIT 支持后切换到 scale.item()
        scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
    if len(zero_point.shape) == 0:
        # TODO: 在添加 JIT 支持后切换到 zero_point.item()
        zero_point = torch.tensor(
            [int(zero_point)], dtype=zero_point.dtype, device=device
        )
        # 如果是按通道进行仿射量化，则将零点转换为浮点数类型
        if qscheme == torch.per_channel_affine_float_qparams:
            zero_point = torch.tensor(
                [float(zero_point)], dtype=zero_point.dtype, device=device
            )

    # 返回双精度浮点类型的缩放因子和64位整数类型的零点
    return scale.to(torch.double), zero_point.to(torch.int64)
# 获取函数的位置参数数量
def _get_num_pos_args(f: Callable) -> int:
    """ Get number of positional args for a function

    Example::

    >> def f(self, key1=3, key2=3):
           pass
    >> _get_num_pos_args(f)
    3
    """
    return len(getfullargspec(f).args)

# 根据模型和其示例输入返回一个字典，键为子模块的完全限定名，值为该子模块的示例输入
def get_fqn_to_example_inputs(
    model: torch.nn.Module,
    example_inputs: Tuple[Any, ...]
) -> Dict[str, Tuple[Any, ...]]:
    """ Given a model and its example inputs, return a dictionary from
    fully qualified name of submodules to example_inputs for that submodule,
    e.g. {"linear1": (tensor1,), "linear2": (tensor2,), "sub": (tensor3,),
          "sub.linear1": (tensor4,), ...}

    Used to make quantizing submodules easier now that FX Graph Mode Quantization requires
    example inputs.

    Also works for keyword arguments with default values, we would flatten keyword
    arguments as positional arguments and fill in the missing keyword args with default
    values, e.g. if we have a forward function:
    def forward(self, x, key1=3, key2=3):
        ...

    and we call it with self.submodule(x, key2=6)
    we'll get example_inputs: (x, 3, 6)

    user can also override `key1` with positional arguments as well:
    for self.submodule(x, 5, key2=6)
    we'll get: (x, 5, 6)

    variable positional arguments and variable positional keyword arguments in forward
    function are not supported currently, so please make sure no submodules is using
    them.
    """
    root = model
    fqn_to_example_inputs = {}

    # 用于替换模块调用方法，记录每个子模块的示例输入
    def _patched_module_call(self, *args, **kwargs):
        submodule_example_inputs = list(args).copy()
        normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
        # 减去 `self` 的数量得到前向方法的位置参数数量
        num_args = _get_num_pos_args(self.forward) - 1
        num_to_pop = num_args - len(submodule_example_inputs)
        while num_to_pop and normalized_kwargs:
            normalized_kwargs.popitem(last=False)
            num_to_pop -= 1
        submodule_example_inputs.extend(normalized_kwargs.values())
        submodule_example_inputs_tuple = tuple(submodule_example_inputs)
        fqn = _get_path_of_module(root, self)
        if fqn is not None:
            fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
        return orig_module_call(self, *args, **kwargs)

    orig_module_call = torch.nn.Module.__call__
    torch.nn.Module.__call__ = _patched_module_call  # type: ignore[method-assign]
    try:
        model(*example_inputs)
    finally:
        # 即使发生异常也要还原模块调用方法
        torch.nn.Module.__call__ = orig_module_call  # type: ignore[method-assign]
    return fqn_to_example_inputs

# 断言并获取模块的唯一设备，如果检测到多个设备则抛出错误
def _assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    devices = {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}
    """
    """
    A temporary workaround for AIMP HHC publish involving CPU check. Remove this later. T163614564
    """
    # 检查是否同时包含 'cpu' 和 'meta' 设备，若是，则发出警告并选择 'cpu'
    if {torch.device("cpu"), torch.device("meta")} == devices:
        warnings.warn("Both 'meta' and 'cpu' are present in the list of devices. Module can have one device. We Select 'cpu'.")
        devices = {torch.device("cpu")}
    
    # 断言设备列表长度不超过1，即只能是cpu或单设备CUDA模块，否则抛出异常
    assert len(devices) <= 1, (
        "prepare only works with cpu or single-device CUDA modules, "
        f"but got devices {devices}"
    )
    
    # 如果设备列表中有设备，则选取第一个设备作为当前设备；否则设备为空
    device = next(iter(devices)) if len(devices) > 0 else None
    # 返回选择的设备
    return device
# 定义一个模块级别的变量 __all__，包含了此模块中可以被外部引用的公共接口名称列表
__all__ = [
    "NodePattern",  # 节点模式类
    "Pattern",  # 模式类
    "MatchAllNode",  # 匹配所有节点
    "check_node",  # 检查节点函数
    "get_combined_dict",  # 获取合并字典函数
    "is_per_tensor",  # 是否逐张量
    "is_per_channel",  # 是否逐通道
    "getattr_from_fqn",  # 根据全限定名称获取属性函数
    "get_qparam_dict",  # 获取量化参数字典函数
    "get_swapped_custom_module_class",  # 获取交换的自定义模块类函数
    "activation_dtype",  # 激活函数的数据类型
    "weight_dtype",  # 权重数据类型
    "activation_is_statically_quantized",  # 激活是否静态量化
    "activation_is_dynamically_quantized",  # 激活是否动态量化
    "activation_is_int8_quantized",  # 激活是否 int8 量化
    "activation_is_int32_quantized",  # 激活是否 int32 量化
    "weight_is_quantized",  # 权重是否量化
    "weight_is_statically_quantized",  # 权重是否静态量化
    "op_is_int8_dynamically_quantized",  # 操作是否 int8 动态量化
    "get_qconfig_dtypes",  # 获取量化配置数据类型函数
    "get_quant_type",  # 获取量化类型函数
    "check_min_max_valid",  # 检查最小值和最大值的有效性函数
    "calculate_qmin_qmax",  # 计算量化的最小值和最大值函数
    "has_no_children_ignoring_parametrizations",  # 忽略参数化的情况下，是否没有子节点函数
    "get_fqn_to_example_inputs",  # 获取全限定名称到示例输入的映射函数
    "to_underlying_dtype",  # 转换为基础数据类型函数
    "determine_qparams",  # 确定量化参数函数
    "validate_qmin_qmax",  # 验证量化的最小值和最大值函数
]
```