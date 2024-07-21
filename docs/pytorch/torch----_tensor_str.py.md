# `.\pytorch\torch\_tensor_str.py`

```
# 声明允许未类型化的函数定义（用于类型检查工具）
# 导入上下文管理、数据类、数学函数、文本包装模块
import contextlib
import dataclasses
import math
import textwrap
# 导入类型提示模块中的任意、字典、可选类型
from typing import Any, Dict, Optional

# 导入PyTorch库
import torch
# 从torch模块中导入无穷大常量
from torch import inf

# 数据类，用于存储打印选项的配置
@dataclasses.dataclass
class __PrinterOptions:
    # 打印浮点数输出的精度（默认为4）
    precision: int = 4
    # 触发摘要而非完整repr的数组元素总数（默认为1000）
    threshold: float = 1000
    # 每个维度开头和结尾摘要中的数组项数（默认为3）
    edgeitems: int = 3
    # 用于插入换行的每行字符数（默认为80）。阈值矩阵将忽略此参数。
    linewidth: int = 80
    # 科学计数法模式，可选（默认为None，由torch._tensor_str._Formatter定义）
    sci_mode: Optional[bool] = None

# 全局变量，存储默认的打印选项配置
PRINT_OPTS = __PrinterOptions()

# 函数：设置打印选项
# 参数：
#   precision: 浮点数输出的数字精度
#   threshold: 触发摘要的数组元素总数
#   edgeitems: 摘要中每个维度开头和结尾的数组项数
#   linewidth: 用于插入换行的每行字符数
#   profile: 用于预设打印选项的模式（默认、短、完整）
#   sci_mode: 启用（True）或禁用（False）科学计数法
def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `torch._tensor_str._Formatter`. This value is automatically chosen
            by the framework.

    Example::

        >>> # Limit the precision of elements
        >>> torch.set_printoptions(precision=2)
        >>> torch.tensor([1.12345])
        tensor([1.12])
        >>> # Limit the number of elements shown
        >>> torch.set_printoptions(threshold=5)
        >>> torch.arange(10)
        tensor([0, 1, 2, ..., 7, 8, 9])
        >>> # Restore defaults
        >>> torch.set_printoptions(profile='default')
        >>> torch.tensor([1.12345])
        tensor([1.1235])
        >>> torch.arange(10)
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    # 根据profile设置默认的打印选项
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    # 设置用户指定的打印选项
    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    # 如果指定了行宽参数，将全局打印选项中的行宽设置为指定值
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    # 将全局打印选项中的科学计数模式设置为指定的模式
    PRINT_OPTS.sci_mode = sci_mode
# 返回当前打印选项作为字典，以便作为 set_printoptions() 的关键字参数传递
def get_printoptions() -> Dict[str, Any]:
    r"""Gets the current options for printing, as a dictionary that
    can be passed as ``**kwargs`` to set_printoptions().
    """
    return dataclasses.asdict(PRINT_OPTS)


# 定义一个上下文管理器，临时修改打印选项。接受的参数与 set_printoptions 函数相同。
@contextlib.contextmanager
def printoptions(**kwargs):
    r"""Context manager that temporarily changes the print options.  Accepted
    arguments are same as :func:`set_printoptions`."""
    # 获取当前的打印选项
    old_kwargs = get_printoptions()
    # 设置新的打印选项
    set_printoptions(**kwargs)
    try:
        # 执行代码块
        yield
    finally:
        # 恢复旧的打印选项
        set_printoptions(**old_kwargs)


# 根据张量的 is_mps 属性选择合适的数据类型（torch.float 或 torch.double），并转换张量类型
def tensor_totype(t):
    dtype = torch.float if t.is_mps else torch.double
    return t.to(dtype=dtype)


class _Formatter:
    def __init__(self, tensor):
        # 初始化函数，接受一个张量 tensor 作为参数

        self.floating_dtype = tensor.dtype.is_floating_point
        # 检查张量 tensor 的数据类型是否为浮点型

        self.int_mode = True
        # 设定整数模式为 True，初始设定为整数模式

        self.sci_mode = False
        # 设定科学计数法模式为 False，初始设定为非科学计数法模式

        self.max_width = 1
        # 设定最大宽度为 1，用于存储最宽的值的字符长度

        with torch.no_grad():
            tensor_view = tensor.reshape(-1)
            # 使用 torch.no_grad() 上下文管理器，将张量 tensor 展开为一维张量 tensor_view

        if not self.floating_dtype:
            # 如果张量 tensor 不是浮点类型
            for value in tensor_view:
                # 遍历 tensor_view 中的每个值
                value_str = f"{value}"
                # 将值转换为字符串
                self.max_width = max(self.max_width, len(value_str))
                # 更新最大宽度为最长值的字符串长度

        else:
            # 如果张量 tensor 是浮点类型
            nonzero_finite_vals = torch.masked_select(
                tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0)
            )
            # 选择 tensor_view 中有限且非零的值

            if nonzero_finite_vals.numel() == 0:
                # 如果没有有效的数值，不进行后续处理
                # no valid number, do nothing
                return

            # 将非零有限值转换为 double 类型以便于计算
            nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
            nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
            nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())

            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    # 如果值不等于其向上取整的结果，表示不是整数模式
                    self.int_mode = False
                    break

            if self.int_mode:
                # 如果是整数模式
                if (
                    nonzero_finite_max / nonzero_finite_min > 1000.0
                    or nonzero_finite_max > 1.0e8
                ):
                    # 如果最大值与最小值的比值大于 1000 或者最大值大于 1.0e8
                    self.sci_mode = True
                    # 设定为科学计数法模式
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                        # 将值格式化为科学计数法字符串
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    # 否则
                    for value in nonzero_finite_vals:
                        value_str = f"{value:.0f}"
                        # 将值格式化为整数字符串
                        self.max_width = max(self.max_width, len(value_str) + 1)
                        # 更新最大宽度

            else:
                # 如果不是整数模式
                if (
                    nonzero_finite_max / nonzero_finite_min > 1000.0
                    or nonzero_finite_max > 1.0e8
                    or nonzero_finite_min < 1.0e-4
                ):
                    # 如果最大值与最小值的比值大于 1000 或者最大值大于 1.0e8 或者最小值小于 1.0e-4
                    self.sci_mode = True
                    # 设定为科学计数法模式
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                        # 将值格式化为科学计数法字符串
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    # 否则
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
                        # 将值格式化为浮点数字符串
                        self.max_width = max(self.max_width, len(value_str))

        if PRINT_OPTS.sci_mode is not None:
            # 如果在打印选项中指定了科学计数法模式
            self.sci_mode = PRINT_OPTS.sci_mode
            # 使用打印选项中指定的科学计数法模式
    # 返回对象的最大宽度属性值
    def width(self):
        return self.max_width

    # 根据属性设定的格式化选项，将给定值转换成字符串并返回
    def format(self, value):
        # 如果属性指示值为浮点数
        if self.floating_dtype:
            # 如果处于科学计数法模式
            if self.sci_mode:
                # 使用科学计数法格式化字符串
                ret = f"{{:{self.max_width}.{PRINT_OPTS.precision}e}}".format(value)
            # 如果处于整数模式
            elif self.int_mode:
                # 格式化为整数字符串
                ret = f"{value:.0f}"
                # 非无穷大或非数字的情况下，在末尾添加小数点
                if not (math.isinf(value) or math.isnan(value)):
                    ret += "."
            else:
                # 使用常规浮点数格式化字符串
                ret = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
        else:
            # 如果不是浮点数，直接转换为字符串
            ret = f"{value}"
        # 返回格式化后的字符串，并在前面补齐空格以填满最大宽度
        return (self.max_width - len(ret)) * " " + ret
# 格式化标量值为字符串，支持复数情况下的双重格式化
def _scalar_str(self, formatter1, formatter2=None):
    if formatter2 is not None:
        # 获取实部的格式化字符串
        real_str = _scalar_str(self.real, formatter1)
        # 获取虚部的格式化字符串，并添加虚数单位 'j'
        imag_str = (_scalar_str(self.imag, formatter2) + "j").lstrip()
        # 处理负数、+0.0、-0.0 的情况，确保虚部前有符号
        if imag_str[0] == "+" or imag_str[0] == "-":
            return real_str + imag_str
        else:
            return real_str + "+" + imag_str
    else:
        # 对单一标量值进行格式化
        return formatter1.format(self.item())

# 格式化向量值为字符串，支持复数情况下的双重格式化
def _vector_str(self, indent, summarize, formatter1, formatter2=None):
    # 计算每个元素的长度，包括元素之间的空格和逗号
    element_length = formatter1.width() + 2
    if formatter2 is not None:
        # 如果存在虚部格式化器，增加其宽度，并加上 'j' 的长度
        element_length += formatter2.width() + 1

    # 每行可以容纳的元素个数，考虑输出宽度和缩进
    elements_per_line = max(
        1, int(math.floor((PRINT_OPTS.linewidth - indent) / (element_length)))
    )

    # 格式化单个元素值的函数
    def _val_formatter(val, formatter1=formatter1, formatter2=formatter2):
        if formatter2 is not None:
            # 获取实部的格式化字符串
            real_str = formatter1.format(val.real)
            # 获取虚部的格式化字符串，并添加虚数单位 'j'
            imag_str = (formatter2.format(val.imag) + "j").lstrip()
            # 处理负数、+0.0、-0.0 的情况，确保虚部前有符号
            if imag_str[0] == "+" or imag_str[0] == "-":
                return real_str + imag_str
            else:
                return real_str + "+" + imag_str
        else:
            # 对单一元素值进行格式化
            return formatter1.format(val)

    if summarize and not PRINT_OPTS.edgeitems:
        # 处理边缘情况，显示省略号
        data = ["..."]
    elif summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        # 处理大尺寸向量，只显示首尾各一部分元素，中间用省略号连接
        data = (
            [_val_formatter(val) for val in self[: PRINT_OPTS.edgeitems].tolist()]
            + [" ..."]
            + [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems :].tolist()]
        )
    else:
        # 格式化整个向量的元素值
        data = [_val_formatter(val) for val in self.tolist()]

    # 按照每行元素个数划分数据，并将每行转换为字符串
    data_lines = [
        data[i : i + elements_per_line] for i in range(0, len(data), elements_per_line)
    ]
    # 将每行连接起来，并在需要的地方添加换行和缩进
    lines = [", ".join(line) for line in data_lines]
    return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"

# 根据张量的维度进行字符串格式化输出，支持复数情况下的双重格式化
def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None):
    dim = self.dim()

    if dim == 0:
        # 对标量张量进行格式化输出
        return _scalar_str(self, formatter1, formatter2)

    if dim == 1:
        # 对一维张量进行格式化输出
        return _vector_str(self, indent, summarize, formatter1, formatter2)
    # 如果需要总结并且张量的第一维大小超过两倍的打印选项的边界长度
    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        # 切片操作，获取张量的部分数据和省略号
        slices = (
            [
                _tensor_str_with_formatter(
                    self[i], indent + 1, summarize, formatter1, formatter2
                )
                for i in range(0, PRINT_OPTS.edgeitems)
            ]
            + ["..."]
            + [
                _tensor_str_with_formatter(
                    self[i], indent + 1, summarize, formatter1, formatter2
                )
                for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))
            ]
        )
    else:
        # 否则获取整个张量的数据
        slices = [
            _tensor_str_with_formatter(
                self[i], indent + 1, summarize, formatter1, formatter2
            )
            for i in range(0, self.size(0))
        ]

    # 将张量切片转换为字符串形式，以逗号和换行分隔，保持正确的缩进
    tensor_str = ("," + "\n" * (dim - 1) + " " * (indent + 1)).join(slices)
    # 返回格式化后的张量字符串，用方括号括起来
    return "[" + tensor_str + "]"
# 根据缩进和函数名称，定义一个方法，用于生成代表张量的字符串表示
def _tensor_str(self, indent):
    # 如果张量元素数量为0，则返回空列表表示的字符串
    if self.numel() == 0:
        return "[]"

    # 如果张量具有命名维度
    if self.has_names():
        # 在张量打印过程中，有两个主要的代码路径：
        # - 张量数据可以轻松地显示在屏幕上
        # - 需要对张量数据进行汇总显示
        # 由于某些代码路径不完全支持命名张量，因此将未命名的张量作为解决方案发送给格式化代码。
        self = self.rename(None)

    # 判断是否需要对张量数据进行汇总显示
    summarize = self.numel() > PRINT_OPTS.threshold

    # 如果张量是零张量
    if self._is_zerotensor():
        self = self.clone()

    # 处理张量是否具有负数部分
    if self.is_neg():
        self = self.resolve_neg()

    # 如果张量数据类型属于浮点数的某些特定类型，转换为浮点数处理
    if self.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ]:
        self = self.float()

    # 如果张量数据类型是 torch.complex32，处理为复数类型
    if self.dtype is torch.complex32:
        self = self.cfloat()

    # 如果张量数据类型是复数，处理共轭部分
    if self.dtype.is_complex:
        # 处理共轭部分
        self = self.resolve_conj()
        # 对实部和虚部分别进行格式化处理
        real_formatter = _Formatter(
            get_summarized_data(self.real) if summarize else self.real
        )
        imag_formatter = _Formatter(
            get_summarized_data(self.imag) if summarize else self.imag
        )
        # 返回带有格式化处理器的张量字符串表示
        return _tensor_str_with_formatter(
            self, indent, summarize, real_formatter, imag_formatter
        )
    else:
        # 对张量进行格式化处理
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        # 返回带有格式化处理器的张量字符串表示
        return _tensor_str_with_formatter(self, indent, summarize, formatter)


# 定义一个方法，用于向张量字符串添加后缀
def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    # 初始化张量字符串列表
    tensor_strs = [tensor_str]
    # 计算最后一行的长度
    last_line_len = len(tensor_str) - tensor_str.rfind("\n") + 1
    # 遍历后缀列表
    for suffix in suffixes:
        suffix_len = len(suffix)
        # 如果需要强制换行或者当前行长度超过设定的打印宽度
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            # 将后缀添加到新行，并根据缩进进行格式化
            tensor_strs.append(",\n" + " " * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            # 将后缀添加到当前行，并更新当前行长度
            tensor_strs.append(", " + suffix)
            last_line_len += suffix_len + 2
    # 结束后缀列表，返回合并后的字符串
    tensor_strs.append(")")
    return "".join(tensor_strs)


# 定义一个方法，用于获取张量的汇总数据
def get_summarized_data(self):
    # 获取张量的维度
    dim = self.dim()
    # 如果张量维度为0，直接返回张量本身
    if dim == 0:
        return self
    # 如果张量维度为1
    if dim == 1:
        # 如果张量大小超过设定的边缘项数，对张量进行分片并连接处理
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return torch.cat(
                (self[: PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems :])
            )
        else:
            return self
    # 如果没有设定边缘项数或者张量大小超过设定的边缘项数
    if not PRINT_OPTS.edgeitems or self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
        end = [self[i] for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))]
        # 对张量的每个分片递归调用汇总数据方法，然后堆叠处理
        return torch.stack([get_summarized_data(x) for x in (start + end)])
    else:
        # 对张量的每个元素递归调用汇总数据方法，然后堆叠处理
        return torch.stack([get_summarized_data(x) for x in self])
# 如果输入是经过Functorch包装的张量，则调用_functorch_wrapper_str_intern函数处理并返回结果
def _str_intern(inp, *, tensor_contents=None):
    # 判断输入张量是否被Functorch包装
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
        return _functorch_wrapper_str_intern(inp, tensor_contents=tensor_contents)

    # 判断输入是否为普通张量或者是torch.nn.Parameter类型的张量
    is_plain_tensor = type(inp) is torch.Tensor or type(inp) is torch.nn.Parameter

    # 如果inp被标记为嵌套张量，则设置前缀为"nested_tensor("
    if inp.is_nested:
        prefix = "nested_tensor("
    # 如果是普通张量，则设置前缀为"tensor("
    elif is_plain_tensor:
        prefix = "tensor("
    # 否则，使用类型名作为前缀
    else:
        prefix = f"{type(inp).__name__}("

    # 缩进量等于前缀的长度
    indent = len(prefix)

    # 初始化后缀列表
    suffixes = []

    # 检查是否提供了自定义的张量内容
    custom_contents_provided = tensor_contents is not None
    if custom_contents_provided:
        tensor_str = tensor_contents

    # 以下代码段用于提取原始值，从而在此函数内禁用前向自动微分（AD）
    # TODO(albanD) 当支持多层级时需要更新此处
    self, tangent = torch.autograd.forward_ad.unpack_dual(inp)

    # 注释 [打印张量设备信息]：
    # 在这里的一般逻辑是只有在设备类型与默认张量类型中指定的设备不匹配时才打印设备信息。
    # 当前torch.set_default_tensor_type()仅支持CPU/CUDA，因此torch._C._get_default_device()只返回cpu或cuda。
    # 在其他情况下，我们尚无法将它们设置为默认值，因此应始终为它们打印出设备信息。
    if (
        self.device.type != torch._C._get_default_device()
        or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        )
        or (self.device.type == "mps")
    ):
        suffixes.append("device='" + str(self.device) + "'")

    # 当张量设备类型为xla、lazy、ipu、mtia时，将张量复制到CPU上以避免打印时的编译操作
    if self.device.type in ["xla", "lazy", "ipu", "mtia"]:
        self = self.to("cpu")

    # TODO: 添加一个API来映射实数到复数dtype

    # 默认复数dtype为torch.cdouble（如果默认dtype为torch.double）或torch.cfloat
    _default_complex_dtype = (
        torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    )
    # 判断张量是否具有默认的dtype
    has_default_dtype = self.dtype in (
        torch.get_default_dtype(),
        _default_complex_dtype,
        torch.int64,
        torch.bool,
    )
    # 如果张量是稀疏的
    if self.is_sparse:
        # 向后缀列表添加张量形状的字符串表示
        suffixes.append("size=" + str(tuple(self.shape)))
        # 导入 FakeTensor 类，这可能会在后续的代码中使用
        from torch._subclasses.fake_tensor import FakeTensor

        # 判断张量是否被认为是元数据，或者是否是 FakeTensor 的实例
        is_meta = self.is_meta or isinstance(self, FakeTensor)
        # 如果不是元数据，向后缀列表添加稀疏张量的非零元素数量字符串表示
        if not is_meta:
            suffixes.append("nnz=" + str(self._nnz()))
        # 如果没有默认的数据类型，则向后缀列表添加张量的数据类型字符串表示
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        # 如果没有提供自定义内容，则处理张量的索引部分
        if not custom_contents_provided:
            indices_prefix = "indices=tensor("
            # 获得张量的索引，并分离它
            indices = self._indices().detach()
            # 如果是元数据，索引字符串会被省略，否则根据给定的缩进调用 _tensor_str 进行字符串表示
            if is_meta:
                indices_str = "..."
            else:
                indices_str = _tensor_str(indices, indent + len(indices_prefix))
            # 如果是元数据或者索引为空，向索引字符串添加张量形状的字符串表示
            if is_meta or indices.numel() == 0:
                indices_str += ", size=" + str(tuple(indices.shape))
            
            values_prefix = "values=tensor("
            # 获得张量的值，并分离它
            values = self._values().detach()
            # 如果是元数据，值字符串会被省略，否则根据给定的缩进调用 _tensor_str 进行字符串表示
            if is_meta:
                values_str = "..."
            else:
                values_str = _tensor_str(values, indent + len(values_prefix))
            # 如果是元数据或者值为空，向值字符串添加张量形状的字符串表示
            if is_meta or values.numel() == 0:
                values_str += ", size=" + str(tuple(values.shape))
            
            # 将索引和值的字符串表示组合成张量的完整字符串表示
            tensor_str = (
                indices_prefix
                + indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    
    # 如果张量的布局在稀疏布局集合中
    elif self.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
        #
        # 导入 FakeTensor 类，用于后续判断是否为 FakeTensor 实例
        from torch._subclasses.fake_tensor import FakeTensor

        # 将张量形状作为后缀添加到列表中
        suffixes.append("size=" + str(tuple(self.shape)))
        
        # 判断是否为元数据张量或 FakeTensor 实例，并添加相应的后缀
        is_meta = self.is_meta or isinstance(self, FakeTensor)
        if not is_meta:
            # 如果不是元数据张量，则添加非零元素数量的后缀
            suffixes.append("nnz=" + str(self._nnz()))
        
        # 如果没有默认数据类型，则添加数据类型的后缀
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        
        # 如果没有提供自定义内容，则根据布局选择压缩索引方法和普通索引方法
        if not custom_contents_provided:
            compressed_indices_method, plain_indices_method = {
                torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
                torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
            }[self.layout]
            
            # 根据布局类型确定压缩索引和普通索引的维度名称前缀
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                cdimname, pdimname = "row", "column"
            else:
                cdimname, pdimname = "column", "row"
            
            # 压缩索引名称前缀，例如 'crow_indices=tensor('
            compressed_indices_prefix = f"c{cdimname[:3]}_indices=tensor("
            # 获取压缩索引的张量，并将其分离出来
            compressed_indices = compressed_indices_method(self).detach()
            
            # 根据是否为元数据，选择是否显示压缩索引的内容
            if is_meta:
                compressed_indices_str = "..."
            else:
                # 调用 _tensor_str 函数，生成压缩索引的字符串表示
                compressed_indices_str = _tensor_str(
                    compressed_indices, indent + len(compressed_indices_prefix)
                )
            
            # 如果压缩索引的元素数为零或是元数据，添加其形状尺寸信息到字符串末尾
            if compressed_indices.numel() == 0 or is_meta:
                compressed_indices_str += ", size=" + str(
                    tuple(compressed_indices.shape)
                )
            
            # 普通索引名称前缀，例如 'col_indices=tensor('
            plain_indices_prefix = f"{pdimname[:3]}_indices=tensor("
            # 获取普通索引的张量，并将其分离出来
            plain_indices = plain_indices_method(self).detach()
            
            # 根据是否为元数据，选择是否显示普通索引的内容
            if is_meta:
                plain_indices_str = "..."
            else:
                # 调用 _tensor_str 函数，生成普通索引的字符串表示
                plain_indices_str = _tensor_str(
                    plain_indices, indent + len(plain_indices_prefix)
                )
            
            # 如果普通索引的元素数为零或是元数据，添加其形状尺寸信息到字符串末尾
            if plain_indices.numel() == 0 or is_meta:
                plain_indices_str += ", size=" + str(tuple(plain_indices.shape))
            
            # 值的名称前缀，例如 'values=tensor('
            values_prefix = "values=tensor("
            # 获取张量的值，并将其分离出来
            values = self.values().detach()
            
            # 根据是否为元数据，选择是否显示值的内容
            if is_meta:
                values_str = "..."
            else:
                # 调用 _tensor_str 函数，生成值的字符串表示
                values_str = _tensor_str(values, indent + len(values_prefix))
            
            # 如果值的元素数为零或是元数据，添加其形状尺寸信息到字符串末尾
            if values.numel() == 0 or is_meta:
                values_str += ", size=" + str(tuple(values.shape))
            
            # 组合所有的张量信息字符串，形成最终的张量字符串表示
            tensor_str = (
                compressed_indices_prefix
                + compressed_indices_str
                + "),\n"
                + " " * indent
                + plain_indices_prefix
                + plain_indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    # 如果张量是量化的情况下执行以下代码块
    elif self.is_quantized:
        # 将张量形状转换为元组形式并加入后缀列表
        suffixes.append("size=" + str(tuple(self.shape)))
        # 如果没有设置默认数据类型，则加入后缀列表中
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        # 将量化方案加入后缀列表
        suffixes.append("quantization_scheme=" + str(self.qscheme()))
        # 如果量化方案为每张量仿射或每张量对称，则加入后缀列表
        if (
            self.qscheme() == torch.per_tensor_affine
            or self.qscheme() == torch.per_tensor_symmetric
        ):
            suffixes.append("scale=" + str(self.q_scale()))
            suffixes.append("zero_point=" + str(self.q_zero_point()))
        # 如果量化方案为每通道仿射、每通道对称或每通道仿射浮点参数，则加入后缀列表
        elif (
            self.qscheme() == torch.per_channel_affine
            or self.qscheme() == torch.per_channel_symmetric
            or self.qscheme() == torch.per_channel_affine_float_qparams
        ):
            suffixes.append("scale=" + str(self.q_per_channel_scales()))
            suffixes.append("zero_point=" + str(self.q_per_channel_zero_points()))
            suffixes.append("axis=" + str(self.q_per_channel_axis()))
        # 如果未提供自定义内容，则生成去量化后的张量的字符串表示形式
        if not custom_contents_provided:
            tensor_str = _tensor_str(self.dequantize(), indent)
    
    # 如果张量是嵌套的情况下执行以下代码块
    elif self.is_nested:
        # 如果未提供自定义内容，则对每个嵌套的张量进行缩进处理并生成字符串表示形式
        if not custom_contents_provided:
            # 定义一个函数，用于对字符串进行缩进处理
            def indented_str(s, indent):
                return "\n".join(f"  {line}" for line in s.split("\n"))
            
            # 生成嵌套张量中每个张量的字符串表示形式，并使用指定缩进进行处理
            strs = ",\n".join(
                indented_str(str(t), indent + 1)
                for t in torch.ops.aten.unbind.int(self, 0)
            )
            # 将嵌套张量的字符串表示形式放入整体张量字符串的格式化输出中
            tensor_str = f"[\n{strs}\n]"
    
    # 如果张量是功能张量的情况下执行以下代码块
    elif torch._is_functional_tensor(self):
        # 设置前缀字符串，并生成功能张量的字符串表示形式
        prefix = "_to_functional_tensor("
        tensor_str = repr(torch._from_functional_tensor(self))
    else:
        # Circular import problem, so we import it here
        # 处理循环导入问题，因此在此处导入 FakeTensor
        from torch._subclasses.fake_tensor import FakeTensor

        if self.is_meta or isinstance(self, FakeTensor):
            # 如果是元数据或者是 FakeTensor 的实例，添加形状信息到后缀列表
            suffixes.append("size=" + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                # 如果数据类型不是默认的 dtype，则添加数据类型信息到后缀列表
                suffixes.append("dtype=" + str(self.dtype))
            # TODO: This implies that ellipses is valid syntax for allocating
            # a meta tensor or FakeTensor, which it could be, but it isn't right now
            # TODO: 这暗示省略号在分配元数据张量或 FakeTensor 时是有效语法，尽管目前不是
            if not custom_contents_provided:
                # 如果没有提供自定义内容，设定张量字符串为省略号
                tensor_str = "..."
        else:
            if self.numel() == 0 and not self.is_sparse:
                # 如果张量元素数为0且不是稀疏张量
                # 显式打印形状，以匹配 NumPy 的行为
                if self.dim() != 1:
                    suffixes.append("size=" + str(tuple(self.shape)))

                # 在空张量中，没有元素可以推断 dtype 是否应该是 int64，因此必须明确显示
                if self.dtype != torch.get_default_dtype():
                    suffixes.append("dtype=" + str(self.dtype))
                if not custom_contents_provided:
                    # 如果没有提供自定义内容，设定张量字符串为 "[]"
                    tensor_str = "[]"
            else:
                if not PRINT_OPTS.edgeitems:
                    # 如果不打印边缘项
                    suffixes.append("size=" + str(tuple(self.shape)))

                if not has_default_dtype:
                    # 如果没有默认的 dtype
                    suffixes.append("dtype=" + str(self.dtype))

                if not custom_contents_provided:
                    if self.layout != torch.strided:
                        # 如果布局不是 strided，则使用 _tensor_str 对张量进行字符串表示
                        tensor_str = _tensor_str(self.to_dense(), indent)
                    else:
                        tensor_str = _tensor_str(self, indent)

    if self.layout != torch.strided:
        # 如果布局不是 strided，添加布局信息到后缀列表
        suffixes.append("layout=" + str(self.layout))

    # Use inp here to get the original grad_fn and not the one generated by the forward grad
    # unpacking.
    # 在这里使用 inp 来获取原始的 grad_fn，而不是由前向梯度解包生成的 grad_fn
    grad_fn_name = None
    try:
        grad_fn = inp.grad_fn
    except RuntimeError:
        # Accessing the grad_fn calls rebasing logic which would cause an error
        # if that tensor is a view created in no-grad mode modified in-place in
        # no-grad mode. See: https://github.com/pytorch/pytorch/issues/99968
        # 访问 grad_fn 会调用重基准逻辑，如果张量是在无梯度模式下创建的视图，并在原地修改，则会引发错误
        grad_fn_name = "Invalid"

    if grad_fn_name is None and grad_fn is not None:  # type: ignore[possibly-undefined]
        # 如果 grad_fn_name 为空且 grad_fn 不为空
        grad_fn_name = type(grad_fn).__name__
        if grad_fn_name == "CppFunction":
            grad_fn_name = grad_fn.name().rsplit("::", 1)[-1]

    if grad_fn_name is not None:
        # 如果 grad_fn_name 不为空，将 grad_fn_name 添加到后缀列表中
        suffixes.append(f"grad_fn=<{grad_fn_name}>")
    elif inp.requires_grad:
        # 否则，如果输入需要梯度，将 "requires_grad=True" 添加到后缀列表中
        suffixes.append("requires_grad=True")

    if self.has_names():
        # 如果张量有命名，将命名信息添加到后缀列表中
        suffixes.append(f"names={self.names}")

    if tangent is not None:
        # 如果有切线，将切线信息添加到后缀列表中
        suffixes.append(f"tangent={tangent}")

    string_repr = _add_suffixes(
        prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse  # type: ignore[possibly-undefined]
        # 使用 _add_suffixes 函数将前缀、张量字符串、后缀列表组合成最终字符串表示
    )

    # 检查当前实例是否标记为参数，并根据需要修改其字符串表示形式。
    # 不幸的是，这个函数必须了解这个细节。
    # 注意：当前为了保持向后兼容，普通张量参数被跳过此操作。未来应该也对它们执行此操作以生成有效的表示形式。
    if isinstance(self, torch.nn.Parameter) and not is_plain_tensor:
        # 如果是 torch.nn.Parameter 的实例且不是普通张量，则在字符串表示形式外层包装为 "Parameter(...)"
        string_repr = f"Parameter({string_repr})"

    # 返回最终的字符串表示形式
    return string_repr
# 定义一个内部函数，用于根据输入的张量返回其字符串表示形式
def _functorch_wrapper_str_intern(tensor, *, tensor_contents=None):
    # 获取张量的功能扩展层级
    level = torch._C._functorch.maybe_get_level(tensor)
    # 断言确保层级不是 -1
    assert level != -1

    # 如果张量是 FunctionalTensor，则需要确保它是最新的
    if torch._C._functorch.is_functionaltensor(tensor):
        # 同步张量确保最新状态
        torch._sync(tensor)

    # 获取张量的原始值
    value = torch._C._functorch.get_unwrapped(tensor)
    # 获取值的字符串表示形式
    value_repr = repr(value)

    # 使用 textwrap 将值的字符串表示形式缩进四个空格
    indented_value_repr = textwrap.indent(value_repr, " " * 4)

    # 如果张量是 BatchedTensor，则返回相应的格式化字符串
    if torch._C._functorch.is_batchedtensor(tensor):
        # 获取批次维度
        bdim = torch._C._functorch.maybe_get_bdim(tensor)
        # 断言确保批次维度不是 -1
        assert bdim != -1
        # 返回 BatchedTensor 的格式化字符串
        return (
            f"BatchedTensor(lvl={level}, bdim={bdim}, value=\n"
            f"{indented_value_repr}\n"
            f")"
        )

    # 如果张量是 GradTrackingTensor，则返回相应的格式化字符串
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        return (
            f"GradTrackingTensor(lvl={level}, value=\n"
            f"{indented_value_repr}\n"
            f")"
        )

    # 如果张量是 FunctionalTensor，则返回相应的格式化字符串
    if torch._C._functorch.is_functionaltensor(tensor):
        return f"FunctionalTensor(lvl={level}, value=\\\n{value_repr})"

    # 如果以上条件均不满足，抛出值错误异常
    raise ValueError("We don't know how to print this, please file us an issue")


# 定义一个方法，用于获取对象的字符串表示形式
def _str(self, *, tensor_contents=None):
    # 使用 torch.no_grad() 禁用梯度计算上下文
    # 使用 _disable_current_modes() 禁用当前模式
    with torch.no_grad(), torch.utils._python_dispatch._disable_current_modes():
        # 创建一个 FuncTorch 禁用对象，保护当前功能扩展状态
        guard = torch._C._DisableFuncTorch()
        # 调用内部函数 _str_intern 返回张量的字符串表示形式
        return _str_intern(self, tensor_contents=tensor_contents)
```