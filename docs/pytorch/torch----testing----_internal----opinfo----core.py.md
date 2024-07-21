# `.\pytorch\torch\testing\_internal\opinfo\core.py`

```py
# mypy: ignore-errors

# 导入必要的模块和库
import collections  # 导入 collections 模块
import collections.abc  # 导入 collections.abc 模块
import math  # 导入 math 模块
import operator  # 导入 operator 模块
import unittest  # 导入 unittest 模块
from dataclasses import asdict, dataclass  # 从 dataclasses 模块导入 asdict 和 dataclass 函数
from enum import Enum  # 导入 Enum 枚举类型
from functools import partial  # 导入 functools 模块中的 partial 函数
from itertools import product  # 导入 itertools 模块中的 product 函数
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union  # 导入 typing 模块中的类型提示

import torch  # 导入 torch 库
from torch.testing import make_tensor  # 从 torch.testing 模块导入 make_tensor 函数
from torch.testing._internal.common_device_type import (  # 导入 torch.testing._internal.common_device_type 模块中的指定函数
    skipCPUIfNoFFT,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import (  # 导入 torch.testing._internal.common_dtype 模块中的指定函数和类型
    _dispatch_dtypes,
    floating_and_complex_types,
    floating_and_complex_types_and,
    floating_types,
    get_all_dtypes,
)
from torch.testing._internal.common_utils import (  # 导入 torch.testing._internal.common_utils 模块中的指定函数和常量
    is_iterable_of_tensors,
    noncontiguous_like,
    TEST_WITH_ROCM,
    torch_to_numpy_dtype_dict,
    TrackedInputIter,
)
from torch.testing._internal.opinfo import utils  # 导入 torch.testing._internal.opinfo 模块中的 utils 函数

from torchgen.utils import dataclass_repr  # 从 torchgen.utils 模块导入 dataclass_repr 函数

# 合理的测试维度大小
L = 20  # 维度 L 的大小为 20
M = 10  # 维度 M 的大小为 10
S = 5   # 维度 S 的大小为 5
XS = 3  # 维度 XS 的大小为 3

# 用于区分默认值和其它值的唯一对象
_NOTHING = object()


# 扩展 getattr 函数以支持限定名称
# 例如 _getattr_qual(torch, 'linalg.norm') -> torch.linalg.norm
def _getattr_qual(obj, name, default=_NOTHING):
    try:
        for path in name.split("."):
            obj = getattr(obj, path)
        return obj
    except AttributeError:
        if default is not _NOTHING:
            return default
        else:
            raise


class DecorateInfo:
    """描述在测试运算符时，应该使用给定装饰器包装的哪些测试或测试类型。
    任何与所有提供的参数匹配的测试都将被装饰。仅当 active_if 参数为 True 时，才会应用装饰器。"""

    __slots__ = [
        "decorators",
        "cls_name",
        "test_name",
        "device_type",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorators,
        cls_name=None,
        test_name=None,
        *,
        device_type=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorators = (
            list(decorators)
            if isinstance(decorators, collections.abc.Sequence)
            else [decorators]
        )  # 将 decorators 转换为列表形式（如果不是的话），并赋值给 self.decorators
        self.cls_name = cls_name  # 初始化类名（如果提供）
        self.test_name = test_name  # 初始化测试名（如果提供）
        self.device_type = device_type  # 初始化设备类型（如果提供）
        self.dtypes = dtypes  # 初始化数据类型列表（如果提供）
        self.active_if = active_if  # 初始化是否激活装饰器的标志

        # 验证数据类型列表的有效性
        if self.dtypes is not None:
            for dtype in self.dtypes:
                assert isinstance(dtype, torch.dtype)  # 断言每个数据类型是 torch 中的有效数据类型
    # 检查当前对象是否处于活动状态，根据传入的参数进行判断
    def is_active(self, cls_name, test_name, device_type, dtype, param_kwargs):
        # 检查是否设置了活动状态条件，并且满足所有条件：
        # - 类名匹配条件或者未指定类名
        # - 测试名称匹配条件或者未指定测试名称
        # - 设备类型匹配条件或者未指定设备类型
        # - 数据类型在指定的数据类型集合中或者未指定数据类型集合
        # - 如果活动状态条件是可调用对象，则通过传入的参数判断活动状态
        return (
            self.active_if
            and (self.cls_name is None or self.cls_name == cls_name)
            and (self.test_name is None or self.test_name == test_name)
            and (self.device_type is None or self.device_type == device_type)
            and (self.dtypes is None or dtype in self.dtypes)
            # 如果 self.active_if 是可调用对象，则使用 param_kwargs 判断活动状态
            # 否则直接返回 self.active_if 的值
            and (
                self.active_if(param_kwargs)
                if isinstance(self.active_if, Callable)
                else self.active_if
            )
        )
# FIXME
# Note: historically the 'input' kwarg had to be a Tensor or TensorList, but we are trying
#   to support scalar inputs, too. Some tests still depend on 'input' being a Tensor
#   or TensorList, however.
# 表示一个样本输入到函数中。

class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "input",                        # 输入的数据，通常是一个张量或张量列表（Sequence[Tensor]）。
        "args",                         # 位置参数的元组。
        "kwargs",                       # 关键字参数的字典。
        "output_process_fn_grad",       # 输出处理函数的梯度函数，通常为 lambda x: x。
        "broadcasts_input",             # 指示输入是否进行了广播，用于验证不支持广播的操作的行为。
        "name",                         # 输入样本的名称。
    ]

    def __init__(
        self,
        input,
        *var_args,
        args=None,
        kwargs=None,
        output_process_fn_grad=None,
        broadcasts_input=None,
        name=None,
        **var_kwargs,
    ):
        # input is the first input to the op and is typically either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        self.input = input          # 将第一个输入作为操作的输入，通常是张量或张量列表。

        # Allow calling either as SampleInput(input, args=args, kwargs=kwargs), or as
        # SampleInput(input, *args, **kwargs) but not to mix the two forms
        if args is not None or kwargs is not None:
            assert (
                not var_args and not var_kwargs
            ), """
A SampleInput can be constructed "naturally" with *args and **kwargs or by
explicitly setting the "args" and "kwargs" parameters, but the two
methods of construction cannot be mixed!"""
        elif len(var_args) or len(var_kwargs):
            assert (
                output_process_fn_grad is None
                and broadcasts_input is None
                and name is None
            ), """
A SampleInput constructed "naturally" with *args and **kwargs
cannot specify additional metadata in keyword arguments"""
        
        # 根据输入参数的不同形式进行初始化
        self.args = args if args is not None else var_args
        assert isinstance(self.args, tuple)
        self.kwargs = kwargs if kwargs is not None else var_kwargs
        assert isinstance(self.kwargs, dict)

        # 设置输出处理函数的梯度函数，默认为恒等函数
        self.output_process_fn_grad = (
            output_process_fn_grad
            if output_process_fn_grad is not None
            else lambda x: x
        )
        
        # 设置输入样本的名称，默认为空字符串
        self.name = name if name is not None else ""

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimeError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = (
            broadcasts_input if broadcasts_input is not None else False
        )

    def with_metadata(
        self, *, output_process_fn_grad=None, broadcasts_input=None, name=None
        # 方法，返回带有指定元数据的新 SampleInput 实例。
    ):
        # 如果给定的参数不为空，则更新对象的输出处理函数梯度
        if output_process_fn_grad is not None:
            self.output_process_fn_grad = output_process_fn_grad
        # 如果给定的参数不为空，则更新对象的广播输入标志
        if broadcasts_input is not None:
            self.broadcasts_input = broadcasts_input
        # 如果给定的参数不为空，则更新对象的名称
        if name is not None:
            self.name = name
        # 返回更新后的对象自身
        return self

    def _repr_helper(self, formatter):
        # 辅助函数，以 `str` 形式返回 SampleInput 的详细信息
        # 它整合了 SampleInput 的所有字段，并允许使用 `formatter` 可调用函数来定制表示方式
        # 可查看 `summary` 方法的示例
        arguments = [
            f"input={formatter(self.input)}",
            f"args={formatter(self.args)}",
            f"kwargs={formatter(self.kwargs)}",
            f"broadcasts_input={self.broadcasts_input}",
            f"name={repr(self.name)}",
        ]

        return f'SampleInput({", ".join(a for a in arguments if a is not None)})'

    def __repr__(self):
        # 返回调用 _repr_helper 方法后的字符串表示形式
        return self._repr_helper(lambda x: x)

    def summary(self):
        # 以更友好的格式返回 SampleInput 的详细信息
        # 它使用 `formatter` 函数来格式化 `Tensor` 和 `TensorList`
        def formatter(arg):
            # 格式化任何 `Tensor` 实例（独立的、列表中的或字典中的）
            # 形式为 Tensor[shape, device, dtype, contiguous]
            # 例如，形状为 (3, 4) 的 Tensor 被格式化为 Tensor[size=(3, 4), device="cuda", dtype=torch.float32]
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape))
                dtype = str(arg.dtype)
                device = str(arg.device)
                contiguity_suffix = ""
                is_sparse = arg.is_sparse or arg.layout == torch.sparse_csr
                if not is_sparse and not arg.is_contiguous():
                    contiguity_suffix = ", contiguous=False"
                return f'Tensor[size={shape}, device="{device}", dtype={dtype}{contiguity_suffix}]'
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return "TensorList[" + ", ".join(map(formatter, arg)) + "]"
            elif isinstance(arg, (list, tuple)):  # 处理列表、元组
                return "(" + ",".join(map(formatter, arg)) + ")"

            return repr(arg)

        # 返回调用 _repr_helper 方法后的字符串表示形式
        return self._repr_helper(formatter)

    # 对 SampleInput 中的每个张量和数据类型应用变换 f(t) -> t
    # 定义一个方法 transform，接受一个函数 f 作为参数
    def transform(self, f):
        # 定义一个嵌套函数 tt，用于递归地转换输入的数据结构 t
        def tt(t):
            # 定义一个内部函数 _tt，使用 torch.no_grad() 上下文管理器执行函数 f(t)
            def _tt(t):
                with torch.no_grad():
                    return f(t)

            # 根据 t 的类型进行不同的处理
            if isinstance(t, torch.Tensor):
                # 如果 t 是 torch.Tensor 类型，则调用 _tt 对其进行处理
                return _tt(t)
            elif isinstance(t, torch.dtype):
                # 如果 t 是 torch.dtype 类型，则调用 _tt 对其进行处理
                return _tt(t)
            elif isinstance(t, list):
                # 如果 t 是 list 类型，则对列表中的每个元素应用 tt 函数
                return list(map(tt, t))
            elif isinstance(t, tuple):
                # 如果 t 是 tuple 类型，则对元组中的每个元素应用 tt 函数
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                # 如果 t 是 dict 类型，则对字典中的每个值应用 tt 函数，并返回新的字典
                return {k: tt(v) for k, v in t.items()}
            else:
                # 其他情况下，直接返回 t
                return t

        # 对输入的 self.input, self.args, self.kwargs 分别应用 tt 函数
        sample_tt_input, tt_args, tt_kwargs = (
            tt(self.input),
            tt(self.args),
            tt(self.kwargs),
        )

        # 返回一个新的 SampleInput 对象，将处理后的数据作为其输入和参数，并保持其他属性不变
        # 注意：转换后的 SampleInput 假设 metadata，如 output_process_fn_grad 仍然有效！
        return SampleInput(
            sample_tt_input,
            args=tt_args,
            kwargs=tt_kwargs,
            output_process_fn_grad=self.output_process_fn_grad,
            broadcasts_input=self.broadcasts_input,
            name=self.name + "_transformed",
        )

    # 返回样本输入对象的 NumPy 版本，以元组形式返回：(input, args, kwargs)
    # 将张量转换为 ndarray，调用 .detach().cpu().numpy() 方法
    # 通过 torch_to_numpy_dtype_dict 将 dtype 转换为对应的 NumPy dtype
    def numpy(self):
        # 定义一个 to_numpy 函数，根据输入 t 的类型进行不同的转换
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                # 如果 t 是 torch.Tensor 类型
                if t.dtype is torch.bfloat16:
                    return t.detach().cpu().to(torch.float32).numpy()
                if t.dtype is torch.chalf:
                    return t.detach().cpu().to(torch.cfloat).numpy()
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                # 如果 t 是 torch.dtype 类型，则使用 torch_to_numpy_dtype_dict 进行转换
                return torch_to_numpy_dtype_dict[t]

            return t

        # 调用 transform 方法，传入 to_numpy 函数，返回转换后的结果
        return self.transform(to_numpy)

    # 返回非连续数据的版本
    def noncontiguous(self):
        # 定义一个 to_noncontiguous 函数，根据输入 t 的类型返回相应的非连续版本
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                # 如果 t 是 torch.Tensor 类型，则调用 noncontiguous_like 函数返回非连续版本
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                # 如果 t 是 torch.dtype 类型，则直接返回 t
                return t

            return t

        # 调用 transform 方法，传入 to_noncontiguous 函数，返回转换后的结果
        return self.transform(to_noncontiguous)
NumericsFilter = collections.namedtuple("NumericsFilter", ["condition", "safe_val"])
# 定义一个命名元组类型 NumericsFilter，用于表示数值过滤器，包含 condition 和 safe_val 两个字段


class ErrorInput:
    """
    A SampleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """
    # 表示会导致操作抛出错误并包含结果错误信息的 SampleInput 类
    __slots__ = ["sample_input", "error_type", "error_regex"]

    def __init__(self, sample_input, *, error_type=RuntimeError, error_regex):
        # 初始化方法，接收 sample_input 作为输入，可以指定 error_type 和 error_regex
        self.sample_input = sample_input
        self.error_type = error_type  # 错误类型，默认为 RuntimeError
        self.error_regex = error_regex  # 错误信息的正则表达式


class AliasInfo:
    """Class holds alias information. For example, torch.abs ->
    torch.absolute, torch.Tensor.absolute, torch.Tensor.absolute_
    """
    # 用于保存别名信息的类，例如 torch.abs 对应 torch.absolute、torch.Tensor.absolute、torch.Tensor.absolute_

    def __init__(self, alias_name):
        self.name = alias_name  # 别名的名称
        self.op = _getattr_qual(torch, alias_name)  # 获取 torch 模块下别名对应的函数或方法
        self.method_variant = getattr(torch.Tensor, alias_name, None)  # 获取 torch.Tensor 类型下别名对应的方法
        self.inplace_variant = getattr(torch.Tensor, alias_name + "_", None)  # 获取 torch.Tensor 类型下别名对应的原地操作方法

    def __call__(self, *args, **kwargs):
        # 实例被调用时执行的方法，调用保存的操作函数 op
        return self.op(*args, **kwargs)


# Note [OpInfos]
# ~~~~~~~~~~~~~~
#
# The majority of this note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# See also: the OpInfo tracker (https://github.com/pytorch/pytorch/issues/54261)
# See also: "Writing Test Templates" in common_device_type.py to learn how to
#   parametrize a test template using OpInfos.
# See also: PyTorch's GitHub wiki on running and writing tests
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
# See also: ModuleInfos, OpInfo's sister class, defined in common_modules.py
#
# An OpInfo is a collection of metadata related to a PyTorch operator. This
#   metadata is used to generate tests that validate properties of the operator,
#   like if it implements the correct gradient formula.
#
# WHY OPINFOS?
# ~~~~~~~~~~~~
#
# OpInfos are principally intended to do three things:
#
#   1) to allow systematic testing over all PyTorch's operators
#   2) to simplify operating testing by autogenerating many tests
#   3) to allow systems (like autograd, torchscript, fx, nnc...) to test
#        against every PyTorch operator
#
# All these goals are still a work in progress. Not every operator has an
#   OpInfo, and some operator tests that could be automatically generated
#   still have to be written manually.
#
# It's helpful to understand that OpInfos are both about test simplification and
#   modularity. PyTorch is a complicated framework with many interrelated systems,
#   too many for any one person to keep track of. An OpInfo can be thought of as the
#   interface between an operator implementer and those other systems. Instead of
#   requiring the implementer of torch.foo understand how to test its forward
#   mode AD or NNC support that's typically handled automatically just by
#   defining an OpInfo.
#
# It's often surprising to OpInfo writers that just implementing an OpInfo
#   doesn't test any code by itself. This is because the tests that OpInfos
#   generate are often parametrized over a wide variety of inputs. In the case
#   of torch.foo, we might want to test its forward and backward modes using
#   floats, doubles, and possibly bfloat16s and halfs. An OpInfo is about setting
#   up that machinery, not about testing any one specific case. Thus, while
#   OpInfos are typically defined in terms of a specific test (or set of tests),
#   they're not tests themselves.
#
# You might also notice the parallel with ModuleInfos here: they're both about
#   defining the interface to some testing tool that may be used in many
#   different places.
#   typically can't verify an operator is actually implemented correctly:
#   通常情况下无法验证运算符是否真正正确实现：

# "If an OpInfo doesn't validate my op works as expected, what's the point
#     of it?"
# 如果 OpInfo 不能验证我的运算符是否按预期工作，那有什么意义呢？

# But the point of is the above. OpInfos are intended to let you focus on testing
#   the operator logic you're familiar with instead of having to write tests for
#   how the operator interacts with each of PyTorch's many systems.
# 但是上述的重点在于，OpInfos 旨在让您专注于测试您熟悉的运算符逻辑，而不是为运算符如何与 PyTorch 的众多系统交互编写测试。

# And, OK, it turns out that SOMETIMES just writing an OpInfo DOES
#   validate your op works as expected, but that's only in special
#   cases. See below for details.
# 好吧，事实证明，有时仅编写 OpInfo 就能验证您的运算符是否按预期工作，但这只在特殊情况下才是如此。详情见下文。

# WHAT'S AN OPINFO?
# ~~~~~~~~~~~~~~~~~
# OPINFO 是什么？

# So what is an OpInfo? It's a Python class that describes an operator's properties,
#   like which dtypes it supports on the CPU and whether it has any aliases.
#   These properties can be divided into three categories:
# OpInfo 是什么？它是一个描述运算符属性的 Python 类，例如它在 CPU 上支持哪些数据类型以及是否有任何别名。
# 这些属性可以分为三类：

#   1) Metadata describing the operator, like the operator's name and if it
#     "supports" the out kwarg.
#   1）描述运算符的元数据，例如运算符的名称以及它是否支持 out 关键字参数。

#   2) Test directives, like "skips" that tell the test suite to skip some
#     tests.
#   2）测试指令，例如告诉测试套件跳过某些测试的“skips”。

#   3) A "sample inputs" function that generates valid inputs for the operator.
#   3）生成运算符有效输入的“sample inputs”函数。

# OpInfo attributes are described in more detail below.
# OpInfo 属性将在下文详细描述。

# THE SAMPLE INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# “sample inputs” 函数

# The "sample inputs" function merits special elaboration. This function is
#   crucial to testing with OpInfos. A typical OpInfo test has to treat the operator
#   as a black box. There's no structure for the test to understand or exploit.
#   Without "sample inputs" it wouldn't even know how to call the OpInfo's
#   operator. The sample input function saves the day by providing different
#   "SampleInputs" that can be used to call the operator. A sample input
#   function should have the following signature:
# “sample inputs” 函数值得特别详细说明。这个函数对于使用 OpInfos 进行测试至关重要。典型的 OpInfo 测试必须将运算符视为黑盒子。
# 测试没有结构可以理解或利用。如果没有“sample inputs”，它甚至不知道如何调用 OpInfo 的运算符。
# 通过提供不同的“SampleInputs”，样本输入函数拯救了情况，这些可以用来调用运算符。样本输入函数应具有以下签名：

#   def sample_inputs_foo(op_info, device, dtype, requires_grad, **kwargs):
#   并且应返回一个 SampleInputs 的可迭代对象（参见上述类描述）。每个 SampleInput 定义一个“input”、一个“args”、一个“kwargs”、一个“output_process_fn_grad”函数、一个“broadcasts_input”布尔值和一个“name”。

#   All the "sample_inputs" functions are invoked within a `torch.no_grad()`
#   environment for efficiency and correctness. As such remember to set the
#   "requires_grad" flag on the inputs **after** performing any transformations
#   on them.
# 所有的“sample_inputs”函数在 `torch.no_grad()` 环境中调用，以提高效率和正确性。因此，请在对输入进行任何转换后设置“requires_grad”标志。

# The "input" is the first argument to the operator, or the tensor that
#   the method or inplace variants of the operator should be called on, and
#   should be on the requested device, of the requested dtype, and its
#   requires_grad attribute should be set to the requires_grad argument.
# “input”是运算符的第一个参数，或者应该在其上调用运算符的方法或原位变体的张量，并且应位于请求的设备上，请求的数据类型，并且其“requires_grad”属性应设置为“requires_grad”参数。

# "args" should contain positional arguments, and "kwargs" keyword arguments.
# “args”应包含位置参数，“kwargs”关键字参数。

# "output_process_fn_grad" has an interesting name. It's a function that maps
#   the operator's output (when given the input, args, and kwargs) to the
#   portion of the output to gradcheck. For example, consider an operator
#   like torch.linalg.slogdet
# “output_process_fn_grad”有一个有趣的名字。它是一个函数，将运算符的输出（当给定输入、args 和 kwargs 时）映射到 gradcheck 的输出部分。例如，考虑像 torch.linalg.slogdet 这样的运算符。
#   (https://pytorch.org/docs/main/generated/torch.linalg.slogdet.html).
#   This operator returns a tuple of two tensors, but the first tensor
#   cannot be backwarded through. Its "output_process_fn_grad" filters
#   this output tuple to just the second argument, which we can call backward
#   on. Functions that produce a single tensor can ignore this argument.
#
# "broadcasts_input" is a bool indicated if the SampleInput causes the operator
#   to broadcast the "input" argument. This is important for tests to understand
#   because inplace variants of operations throw a runtime error if they
#   would broadcast their input arguments, so tests that work with inplace
#   variants filter SampleInputs that broadcast their input.
#
# "name" is a string that's just used for debugging. It appears when printing
#   the SampleInput.
#
# Sample inputs are designed to be used with many tests, some
#   that are very time consuming, so they should be a small
#   set with small tensors. An elaborated set of sample inputs
#   can be specified using the "reference_inputs_func" attribute.
#   The "reference inputs" for an operation are an extended
#   set of sample inputs that can more exhaustively test an
#   operator. They are used by only a few tests that are careful
#   not to take too long to run. Adding reference inputs
#   is highly encouraged!
#
# THE (OPTIONAL) ERROR INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# OpInfos may optionally specify "error inputs" through an error function. If
#   specified test_errors in test_ops.py will call the op with these inputs
#   and validate that the desired error is thrown.
#
# Error inputs automate a common testing pattern where multiple inputs are
#   passed to an operation and the errors they thrown are reviewed. Tests
#   written in this style should be ported to the new OpInfo pattern.
#
# Error inputs are specified using the ErrorInputs class, which contains
#   a SampleInput (see above) and data about the expected error.
#
# OPINFO FILE ORGANIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# All OpInfos are currently defined in this file. Most OpInfo tests are defined
#   in test_ops.py, but some system-specific tests are defined in those
#   systems' test files, and subclass-specific tests are defined in the test
#   file that corresponds to that subclass (see the below).
#   Expect a reorganization in the future.
#
# WHAT'S TESTED?
# ~~~~~~~~~~~~~~
#
# Every OpInfo in the op_db sequence has the following properties validated in
# test_ops.py:
#
#   - that its supported dtypes are specified correctly
#   - that the operation produces the same results when called with noncontiguous inputs
#   - that it supports the out= argument properly (if it allows out=),
#       see https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
#   - that it works with the conjugate view bit properly
#   - that its function, method, and inplace variants perform the same operation
# 以下是一系列关于测试操作符功能的注释，涵盖了多种测试方面和注意事项，以确保操作符的正确性和一致性。
# 这些测试包括但不限于：
#   - 确保 torch.add, torch.Tensor.add 和 torch.Tensor.add_ 三者执行相同的操作
#   - 原地操作保留输入张量的存储
#   - 梯度公式的正确实现，以及正确支持 op 的梯度计算和复杂情况下的自动微分
#   - 使用 JIT 进行追踪或脚本化时，操作执行相同的操作
#   - JIT 能够按预期进行自动微分
#   - 确保操作符的别名（如果有）执行相同的操作，并且 JIT 能够理解别名
#   - 确保操作符在错误输入时抛出正确的错误（如果定义了 error_inputs）
#   - 确保操作符与 NumPy 参考实现的结果一致（如果定义了 ref）
#   - 在扩展的“参考输入集”上，确保操作符与 NumPy 参考实现的结果一致（如果 ref 和 reference_inputs_func 均已定义）
#   - 确保操作符能在不同的 CUDA 设备上正常工作
#
# 针对性能，上述某些测试可能仅对 OpInfo 样本输入函数返回的第一个样本输入运行。
#
# 除了这些测试外，还有一些 OpInfo 的子类（在下一节中讨论），定义了额外的测试。
#
# 需要强调的是，并非所有操作符的功能都会经过测试。在实现 OpInfo 时，工程师通常仍需编写一个或多个测试来验证操作符的行为。
# 唯一的例外是如果参考测试已经足够，或者操作属于一个拥有更详尽操作符测试的 OpInfo 子类。特别是，元素级别的一元和二元操作符通常不需要额外的测试，
# 只需编写 OpInfo 即可。
#
# 还有一些 OpInfo 的子类，如 UnaryUfuncInfo，用于一元元素级操作，这些操作具有共同的结构，test_unary_ufuncs.py 中的自动化测试可以充分利用这一结构，
# 将操作符与大量数值的 NumPy 参考函数进行比较。对于一元元素级操作，实现 OpInfo 通常已经足够进行测试。
#
# ForeachFuncInfo 是另一个 OpInfo 子类，非常专门化，适用于一类非常特殊的操作。这些 OpInfo 不包括在内
# 定义一个数据类，用于存储操作符的信息及获取相关信息的辅助函数
@dataclass
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    # 操作符函数的名称，为字符串类型
    name: str

    # 可选的参考函数，接受 ndarrays（即“NumPy 数组”）
    # 该参考函数用于比较操作符的实现结果与预期结果
    # 如果提供了，操作将在其每个样本输入上与其参考进行比较。
    ref: Optional[Callable] = None

    # 下面的元数据描述了操作符及其变体和别名（如果有）。
    
    # 别名的可迭代集合，例如对于 torch.abs 是 ("absolute",)
    aliases: Iterable = None

    # 在测试名称中包含的额外字符串
    # 当一个操作需要多个 OpInfos 时很有用，
    # 比如 divide 经常需要，通常因为它实际上是幕后的几个不同操作
    variant_test_name: str = ""

    # 操作的函数变体，如果为 None，则默认为 torch.<name>
    op: Callable = None

    # 允许指定此操作的方法变体：
    # - 如果为 _NOTHING（默认），则 OpInfo 尝试使用其名称来发现变体
    # - 如果为 None，则 OpInfo 明确指定没有相关方法
    # - 如果为 Callable，则该可调用对象应为与此操作关联的方法
    method_variant: Callable = _NOTHING

    # 允许指定此操作的就地变体：
    # - 如果为 _NOTHING（默认），则 OpInfo 尝试使用其名称来发现变体
    # - 如果为 None，则 OpInfo 明确指定没有相关就地变体
    # - 如果为 Callable，则该可调用对象应为与此操作关联的就地变体
    inplace_variant: Callable = _NOTHING

    # 允许指定此操作的操作符变体：
    # - 如果为 _NOTHING（默认），则 OpInfo 尝试使用其名称来发现变体
    # - 如果为 None，则 OpInfo 明确指定没有相关操作符
    # - 如果为 Callable，则该可调用对象应为与此操作关联的操作符
    operator_variant: Callable = _NOTHING

    # 允许指定此操作的就地操作符变体：
    # - 如果为 _NOTHING（默认），则 OpInfo 尝试使用其名称来发现变体
    # - 如果为 None，则 OpInfo 明确指定没有相关就地操作符
    # - 如果为 Callable，则该可调用对象应为与此操作关联的就地操作符
    inplace_operator_variant: Callable = _NOTHING

    # 下面的元数据是用于跳过或修改测试的测试指令
    
    # 关于要跳过哪些测试的信息
    skips: Tuple = tuple()

    # 应用于生成的测试的装饰器
    decorators: Tuple = tuple()

    # 下面是生成特定类型输入的函数的指针
    
    # 生成带有分步布局的样本输入的函数
    sample_inputs_func: Callable = None

    # 生成更彻底的带有分步布局的样本输入的函数
    reference_inputs_func: Callable = None

    # 生成将引发错误的输入的函数
    error_inputs_func: Callable = None
    # 定义生成稀疏输入数据的函数，会引发错误
    error_inputs_sparse_func: Callable = None

    # 定义生成使用稀疏 COO 布局的示例输入数据的函数
    sample_inputs_sparse_coo_func: Callable = None

    # 定义生成使用稀疏 CSR 布局的示例输入数据的函数
    sample_inputs_sparse_csr_func: Callable = None

    # 定义生成使用稀疏 CSC 布局的示例输入数据的函数
    sample_inputs_sparse_csc_func: Callable = None

    # 定义生成使用稀疏 BSR 布局的示例输入数据的函数
    sample_inputs_sparse_bsr_func: Callable = None

    # 定义生成使用稀疏 BSC 布局的示例输入数据的函数
    sample_inputs_sparse_bsc_func: Callable = None

    # 下面的元数据与数据类型支持有关，在 test_ops.py 中进行正确性测试

    # 此函数在 CPU 上支持的数据类型，会被其他设备类型继承，除非它们指定了自己的数据类型
    dtypes: _dispatch_dtypes = None

    # 在 CUDA 上预期此函数能够处理的数据类型
    dtypesIfCUDA: _dispatch_dtypes = None

    # 在 ROCM 上预期此函数能够处理的数据类型
    dtypesIfROCM: _dispatch_dtypes = None

    # 在 XPU 上预期此函数能够处理的数据类型
    dtypesIfXPU: _dispatch_dtypes = None

    # 此函数预期支持的反向传播用的数据类型
    backward_dtypes: _dispatch_dtypes = None

    # 在 CUDA 上预期此函数能够处理的反向传播用的数据类型
    backward_dtypesIfCUDA: _dispatch_dtypes = None

    # 在 ROCM 上预期此函数能够处理的反向传播用的数据类型
    backward_dtypesIfROCM: _dispatch_dtypes = None

    # 下面的元数据描述了操作符支持的 out= 参数

    # 此操作是否支持 out 关键字参数，默认为 True
    # 如果操作不允许或不正确支持 out 关键字参数，则 test_ops.py 中的 test_out 测试应失败
    supports_out: bool = True

    # 下面的元数据与自动微分支持有关

    # 是否支持反向模式自动微分
    # 如果为 True，则在 test_ops.py 中使用操作的示例输入数据来测试梯度的正确性
    supports_autograd: bool = True

    # 是否支持二阶梯度
    # 如果为 True，则在 test_ops.py 中测试 gradgrad 的正确性，默认与 supports_autograd 的值相同
    # TODO: 重命名为 supports_bwgrad_bwgrad 以与下面保持一致
    supports_gradgrad: bool = None

    # 是否支持通过 forward-over-reverse 实现的二阶梯度
    # 如果为 True，则测试 forward-over-reverse 的 gradgrad 正确性
    # 如果为 False，则测试前向梯度未实现
    # 默认为 False
    supports_fwgrad_bwgrad: bool = False

    # 是否支持原地自动微分
    # 如果为 True，则在 test_ops.py 中进行测试，默认与 supports_autograd 的值相同
    supports_inplace_autograd: bool = None
    # 是否支持前向模式的自动微分
    # 如果为 True，则检查梯度是否正确
    # 如果为 False，则测试前向梯度未实现
    supports_forward_ad: bool = False

    # 是否支持可变参数的变体
    # （例如像 ones、zeros 这样的函数，像 view、permute 这样的方法）
    supports_varargs: bool = False

    # 是否前向操作避免材料化 COW 张量输入
    supports_cow_input_no_materialize_forward: bool = True

    # 是否反向操作避免材料化 COW 张量输入
    supports_cow_input_no_materialize_backward: bool = True

    # 是否跳过 COW 张量输入测试的反向部分
    skip_cow_input_backward: bool = False

    # 如果 `supports_cow_input_no_materialize_forward == True`，则此列表包含
    # 预期材料化的输入的参数索引或关键字参数名称
    allow_cow_input_materialize_forward: List[Union[int, str]] = None

    # 如果 `supports_cow_input_no_materialize_backward == True`，则此列表包含
    # 预期材料化的输入的参数索引或关键字参数名称
    allow_cow_input_materialize_backward: List[Union[int, str]] = None

    # gradcheck 的包装函数
    gradcheck_wrapper: Callable = lambda op, *args, **kwargs: op(*args, **kwargs)

    # 在进行 gradcheck 时是否检查批处理梯度
    # 默认为 support_autograd 的值
    check_batched_grad: bool = None

    # 在进行 gradgradcheck 时是否检查批处理梯度梯度
    # 默认为 support_gradgrad 的值
    check_batched_gradgrad: bool = None

    # 在进行 gradcheck 时是否检查批处理前向梯度
    # 默认为 supports_forward_ad 的值
    check_batched_forward_grad: bool = None

    # 在进行 gradcheck 时是否检查原地批处理前向梯度
    # 默认为 check_batched_forward_grad 的值
    check_inplace_batched_forward_grad: bool = None

    # 在执行 gradcheck 时的非确定性容差
    gradcheck_nondet_tol: float = 0.0

    # 是否使用快速模式进行 gradcheck/gradgradcheck 的实现
    # 当设置为 None 时，延迟到 gradcheck 的包装函数提供的默认值
    gradcheck_fast_mode: bool = None

    # 以下元数据与 JIT 支持相关，在 test_ops.py 中测试其正确性

    # 对应的 aten:: 运算符的名称
    aten_name: str = None

    # 如果这是一个复合的隐式自动微分操作，则是解耦的操作
    decomp_aten_name: Optional[str] = None

    # 用于反向的对应的 aten:: 运算符的名称
    aten_backward_name: Optional[str] = None

    # 如果一个操作的 aten::node 预期被符号自动微分
    assert_autodiffed: bool = False

    # 包含在 DifferentiableGraph 中预期自动微分的节点名称列表
    # 例如：['aten::add', 'aten::mm']
    # 默认情况下，用于非融合节点的列表，初始化为 None
    autodiff_nonfusible_nodes: List[str] = None

    # 期望在 DifferentiableGraphs 的 FusionGroups 中出现的节点名称列表，用于自动微分操作
    # 例如：['aten::add', 'aten::mm']，默认为空列表
    # 注意：当前没有操作使用可融合节点
    autodiff_fusible_nodes: List[str] = None

    # 下列元数据与稀疏支持相关，用于 test_sparse.py 测试

    # 操作是否支持稀疏 COO 输入，默认为 False
    # TODO: 将 supports_sparse 重命名为 supports_sparse_coo
    supports_sparse: bool = None

    # 仅运行追踪测试
    supports_scripting: bool = True

    # 操作是否支持追踪
    supports_tracing: bool = True

    # 下列元数据与稀疏压缩支持相关，用于 test_sparse_csr.py 和 test_sparse.py 测试

    # 操作是否支持稀疏 CSR 输入，默认为 False
    supports_sparse_csr: bool = None
    # 操作是否支持稀疏 CSC 输入，默认为 False
    supports_sparse_csc: bool = None
    # 操作是否支持稀疏 BSR 输入，默认为 False
    supports_sparse_bsr: bool = None
    # 操作是否支持稀疏 BSC 输入，默认为 False
    supports_sparse_bsc: bool = None

    # 操作是否将整数输入提升为浮点数
    promotes_int_to_float: bool = False

    # 下列元数据与复杂数支持相关，用于 test_ops.py 测试

    test_conjugated_samples: bool = True

    test_neg_view: bool = True

    # 断言 JIT 形状分析是否完全传播形状
    assert_jit_shape_analysis: bool = False

    # 下列元数据与 ExpandedWeights 支持相关，用于 test_expanded_weights.py 测试

    supports_expanded_weight: bool = False

    is_factory_function: bool = False

    def __call__(self, *args, **kwargs):
        """调用操作符的函数变体。"""
        return self.op(*args, **kwargs)

    def __str__(self):
        """返回数据类的字符串表示形式。"""
        return dataclass_repr(self)

    def get_op(self):
        """返回操作符的函数变体，如 torch.<op_name>。"""
        return self.op

    def get_method(self):
        """返回操作符的方法变体，如 torch.Tensor.<op_name>。
        如果操作符没有方法变体，则返回 None。
        """
        return self.method_variant

    def get_inplace(self):
        """返回操作符的原位变体，如 torch.Tensor.<op_name>_。
        如果操作符没有原位变体，则返回 None。
        """
        return self.inplace_variant

    def get_operator(self):
        """返回操作符的运算符变体，例如 operator.neg。
        如果操作符没有运算符变体，则返回 None。
        """
        return self.operator_variant
    # 返回操作符的 inplace 变体，例如 operator.iadd
    # 如果操作符没有 inplace 变体则返回 None
    def get_inplace_operator(self):
        return self.inplace_operator_variant

    # 返回带有共轭的 SampleInputs 的可迭代对象，其中张量输入或序列输入的第一个张量被共轭
    def conjugate_sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        # 使用 sample_inputs_func 获取样本输入
        samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        # 创建 samples 的副本
        conj_samples = list(samples)

        # 定义共轭函数
        def conjugate(tensor):
            _requires_grad = tensor.requires_grad
            tensor = tensor.conj()
            return tensor.requires_grad_(_requires_grad)

        # 遍历样本并共轭张量输入
        for i, sample in enumerate(samples):
            sample = conj_samples[i]
            # 注意：假设此处的输入要么是张量要么是张量列表
            if isinstance(sample.input, torch.Tensor):
                sample.input = conjugate(sample.input)
            else:
                sample.input[0] = conjugate(sample.input[0])

        # 返回追踪的共轭样本输入的可迭代对象
        return TrackedInputIter(iter(conj_samples), "conjugate sample input")

    # 返回 SampleInputs 的可迭代对象
    # 这些样本足以测试函数在 autograd、TorchScript 等中的正确性
    def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)

        # 如果 include_conjugated_inputs 参数为 True，则包含共轭输入的样本
        if kwargs.get("include_conjugated_inputs", False):
            conj_samples = self.conjugate_sample_inputs(
                device, dtype, requires_grad, **kwargs
            )
            samples_list = list(samples)
            samples_list.extend(conj_samples)
            samples = tuple(samples_list)

        # 返回追踪的样本输入的可迭代对象
        return TrackedInputIter(iter(samples), "sample input")

    # 返回 ReferenceInputs 的可迭代对象
    # 当 reference_inputs_func 定义时，返回扩展的输入集；未定义时返回样本输入
    def reference_inputs(self, device, dtype, requires_grad=False, **kwargs):
        # 如果未定义 reference_inputs_func，则返回 sample_inputs_func 获取的样本输入
        if self.reference_inputs_func is None:
            samples = self.sample_inputs_func(
                self, device, dtype, requires_grad, **kwargs
            )
            return TrackedInputIter(iter(samples), "sample input")

        # 如果 include_conjugated_inputs 参数为 True，则抛出 NotImplementedError
        if kwargs.get("include_conjugated_inputs", False):
            raise NotImplementedError

        # 使用 reference_inputs_func 获取参考输入
        references = self.reference_inputs_func(
            self, device, dtype, requires_grad, **kwargs
        )
        # 返回追踪的参考输入的可迭代对象
        return TrackedInputIter(iter(references), "reference input")
    def error_inputs(self, device, **kwargs):
        """
        Returns an iterable of ErrorInputs.
        """
        # 使用 error_inputs_func 方法获取 ErrorInputs 的可迭代对象 errs
        errs = self.error_inputs_func(self, device, **kwargs)
        # 将 errs 包装成 TrackedInputIter 对象，并指定迭代器的描述和回调函数
        return TrackedInputIter(
            iter(errs), "error input", callback=lambda e: e.sample_input
        )

    def error_inputs_sparse(self, device, layout, **kwargs):
        """
        Returns an iterable of ErrorInputs that contain sparse sample
        inputs with a specified layout.
        """
        # 检查 OpInfo 是否支持指定的稀疏布局，如果不支持则抛出 SkipTest 异常
        if not self.supports_sparse_layout(layout):
            raise unittest.SkipTest("unsupported sparse layout")
        # 使用 error_inputs_sparse_func 方法获取包含稀疏样本输入的 ErrorInputs 可迭代对象
        return self.error_inputs_sparse_func(self, device, layout, **kwargs)

    def supports_sparse_layout(self, layout):
        """Return True if OpInfo supports the specified sparse layout."""
        # 提取稀疏布局的名称，将 torch.sparse_coo 映射到 OpInfo.supports_sparse
        layout_name = str(layout).split(".")[-1]
        layout_name = layout_name.replace("_coo", "")
        # 返回 OpInfo 是否支持对应的稀疏布局
        return getattr(self, f"supports_{layout_name}")

    def sample_inputs_sparse(
        self, layout, device, dtype, requires_grad=False, **kwargs
    ):
        """Returns an iterable of SampleInputs that contain inputs with a
        specified sparse layout.
        """
        # 提取稀疏布局的名称
        layout_name = str(layout).split(".")[-1]
        # 获取对应稀疏布局的样本输入生成函数
        sample_inputs_mth = getattr(self, "sample_inputs_" + layout_name)

        def non_empty_sampler(op, generator):
            # 非空样本生成器，用于确保至少存在一个样本
            found_sample = False
            for sample in generator:
                found_sample = True
                yield sample
            if not found_sample:
                raise unittest.SkipTest("NO SAMPLES!")

        # 使用非空样本生成器获取稀疏布局的 SampleInputs 可迭代对象
        return non_empty_sampler(
            self,
            sample_inputs_mth(device, dtype, requires_grad=requires_grad, **kwargs),
        )

    def _sample_inputs_unspecified(self, *args, **kwargs):
        """Raises an NotImplemented exception in a OpInfo instance creation
        that specifies supports_sparse(|_csr|_csc|_bsr|_bsc)=True
        without specifying the corresponding sample function as
        sample_inputs_sparse_(coo|csr|csc|bsr|bsc)_func.

        To avoid this, either define the corresponding sample function,
        or re-map unsupported samples to error inputs in an appropiate

          opinfo/definitions/sparse.py:_validate_sample_input_sparse_<op>

        function.
        """
        # 抛出未实现异常，用于 OpInfo 实例创建时指定 supports_sparse=True，
        # 但没有相应的 sample 函数定义时使用
        raise NotImplementedError("no sample function specified")

    def sample_inputs_sparse_coo(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        coo layout.
        """
        # 使用 sample_inputs_sparse_coo_func 方法获取包含 COO 稀疏布局输入的 SampleInputs 可迭代对象
        return self.sample_inputs_sparse_coo_func(
            self, device, dtype, requires_grad, **kwargs
        )
    # 返回一个包含稀疏 csr 布局输入样本的可迭代对象 SampleInputs
    def sample_inputs_sparse_csr(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        csr layout.
        """
        # 调用特定函数获取稀疏 csr 布局的输入样本
        return self.sample_inputs_sparse_csr_func(
            self, device, dtype, requires_grad, **kwargs
        )

    # 返回一个包含稀疏 csc 布局输入样本的可迭代对象 SampleInputs
    def sample_inputs_sparse_csc(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        csc layout.
        """
        # 调用特定函数获取稀疏 csc 布局的输入样本
        return self.sample_inputs_sparse_csc_func(
            self, device, dtype, requires_grad, **kwargs
        )

    # 返回一个包含稀疏 bsr 布局输入样本的可迭代对象 SampleInputs
    def sample_inputs_sparse_bsr(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        bsr layout.
        """
        # 调用特定函数获取稀疏 bsr 布局的输入样本
        return self.sample_inputs_sparse_bsr_func(
            self, device, dtype, requires_grad, **kwargs
        )

    # 返回一个包含稀疏 bsc 布局输入样本的可迭代对象 SampleInputs
    def sample_inputs_sparse_bsc(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs that contain inputs with sparse
        bsc layout.
        """
        # 调用特定函数获取稀疏 bsc 布局的输入样本
        return self.sample_inputs_sparse_bsc_func(
            self, device, dtype, requires_grad, **kwargs
        )

    # 返回给定测试的修饰器列表
    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        """Returns the decorators targeting the given test."""
        # 初始化结果列表
        result = []
        # 遍历每个修饰器
        for decorator in self.decorators:
            # 如果修饰器是 DecorateInfo 类型
            if isinstance(decorator, DecorateInfo):
                # 检查修饰器是否激活，如果是则将其修饰器列表加入结果中
                if decorator.is_active(
                    test_class, test_name, device, dtype, param_kwargs
                ):
                    result.extend(decorator.decorators)
            else:
                # 如果修饰器不是 DecorateInfo 类型，则直接加入结果中
                result.append(decorator)
        # 返回最终的修饰器列表
        return result

    # 返回支持给定设备类型的数据类型集合
    def supported_dtypes(self, device_type):
        # 如果设备类型是 "privateuse1"，则获取其对应的后端名称
        if device_type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        # 将设备类型转换为 Torch 设备对象，然后获取其类型
        device_type = torch.device(device_type).type
        # 如果设备类型是 "cuda"
        if device_type == "cuda":
            # 根据测试环境选择返回不同的数据类型集合
            return self.dtypesIfROCM if TEST_WITH_ROCM else self.dtypesIfCUDA
        # 如果设备类型是 "xpu"，返回专门的数据类型集合
        if device_type == "xpu":
            return self.dtypesIfXPU
        # 对于其他设备类型，返回默认的数据类型集合
        return self.dtypes

    # 返回支持给定设备类型的反向传播数据类型集合
    def supported_backward_dtypes(self, device_type):
        # 如果不支持自动求导，则返回空集合
        if not self.supports_autograd:
            return set()

        # 如果设备类型是 "privateuse1"，则获取其对应的后端名称
        if device_type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        # 将设备类型转换为 Torch 设备对象，然后获取其类型
        device_type = torch.device(device_type).type
        backward_dtypes = None
        # 如果设备类型是 "cuda"
        if device_type == "cuda":
            # 根据测试环境选择返回不同的反向传播数据类型集合
            backward_dtypes = (
                self.backward_dtypesIfROCM
                if TEST_WITH_ROCM
                else self.backward_dtypesIfCUDA
            )
        else:
            # 对于其他设备类型，返回默认的反向传播数据类型集合
            backward_dtypes = self.backward_dtypes

        # 允许的反向传播数据类型集合为浮点数和复数类型及其他指定类型的交集
        allowed_backward_dtypes = floating_and_complex_types_and(
            torch.bfloat16, torch.float16, torch.complex32
        )
        # 返回允许的反向传播数据类型集合与实际支持的交集
        return set(allowed_backward_dtypes).intersection(backward_dtypes)
    # 检查指定的数据类型是否在当前设备类型支持的数据类型列表中
    def supports_dtype(self, dtype, device_type) -> bool:
        return dtype in self.supported_dtypes(device_type)

    # 返回格式化后的操作名称，用于测试名称中的显示
    @property
    def formatted_name(self):
        """Returns a formatted full name for this OpInfo that can be used in test names."""
        # 如果存在变体测试名称，则将其替换为下划线形式，并添加到操作名称中
        variant = (
            "_" + self.variant_test_name.replace(".", "_")
            if self.variant_test_name
            else ""
        )
        # 返回操作名称（点号替换为下划线）加上变体名称（如果存在）
        return f"{self.name.replace('.', '_')}{variant}"
# 生成用于测试缩减操作的输入张量生成器函数
def _generate_reduction_inputs(device, dtype, requires_grad, **kwargs):
    """Generates input tensors for testing reduction operators"""
    yield make_tensor([], dtype=dtype, device=device, requires_grad=requires_grad)
    yield make_tensor([2], dtype=dtype, device=device, requires_grad=requires_grad)
    yield make_tensor([3, 5], dtype=dtype, device=device, requires_grad=requires_grad)
    yield make_tensor(
        [3, 2, 1, 2], dtype=dtype, device=device, requires_grad=requires_grad
    )


# 生成适用于测试缩减操作的部分有效的 dim 和 keepdim 参数的生成器函数
def _generate_reduction_kwargs(ndim, supports_multiple_dims=True):
    """Generates a subset of all valid dim and keepdim kwargs given ndim that
    is appropriate for testing reduction operators.
    """

    # 测试默认的 dim 和 keepdim 参数组合
    yield {}

    # 测试对最内层和最外层维度进行缩减
    yield {"dim": 0, "keepdim": True}
    yield {"dim": -1, "keepdim": False}

    # 测试对中间维度进行缩减
    if ndim > 2:
        yield {"dim": ndim // 2, "keepdim": True}

    if supports_multiple_dims:
        # 测试对所有维度进行缩减
        yield {"dim": tuple(range(ndim)), "keepdim": False}

        # 测试同时对第一个和最后一个维度进行缩减
        if ndim > 1:
            yield {"dim": (0, -1), "keepdim": True}

        # 测试从第二个维度开始每隔一个维度进行缩减
        if ndim > 3:
            yield {"dim": tuple(range(1, ndim, 2)), "keepdim": False}


# 为缩减操作提供样本输入数据
def sample_inputs_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for reduction operators."""

    # 从 kwargs 中获取是否支持多个维度缩减的参数，默认为 True
    supports_multiple_dims: bool = kwargs.get("supports_multiple_dims", True)

    # 从 kwargs 中获取生成参数和关键字参数的函数，默认返回空元组和空字典
    generate_args_kwargs = kwargs.get(
        "generate_args_kwargs", lambda *args, **kwargs: (yield tuple(), {})
    )

    for t in _generate_reduction_inputs(device, dtype, requires_grad):
        for reduction_kwargs in _generate_reduction_kwargs(
            t.ndim, supports_multiple_dims
        ):
            for args, kwargs in generate_args_kwargs(t, **reduction_kwargs):
                kwargs.update(reduction_kwargs)
                yield SampleInput(
                    t.detach().requires_grad_(requires_grad), args=args, kwargs=kwargs
                )


# NOTE [Reductions]:
#
# 出于测试目的，我们放宽了以下缩减操作的定义，如下面文档字符串中所定义。
# 我们这样做是为了捕捉具有类似 API 的操作，以便可以进行自动化测试。然而...
#
# 严格来说，一个缩减操作是一个能够将数组缩减为单个标量值并且能够从缩减子数组的部分结果计算得到的操作符。
# 通常这意味着缩减操作可以从部分结果中逐步计算得到最终结果。
# 定义了一个继承自OpInfo的类，用于描述一个归约操作符的信息
#
# 这个类主要用于说明一个操作符是否是一个归约操作符，即将输入张量的一个或多个维度归约到单个值。
# 归约操作符必须实现以下签名：
# - `op(input, *args, *, dim=None, keepdim=False, **kwargs) -> Tensor`
#
# ReductionOpInfo测试确保归约操作符实现了一致的API。
# ReductionOpInfo构造函数中的可选参数捕获了多维度归约等可选特性。
#
# 如果一个归约操作符尚未实现归约操作符的完整API，应该通过xfail标记失败的测试来记录，
# 而不是向ReductionOpInfo添加可选参数。
#
# 注意：
# 归约操作符的API尚未最终确定，某些要求可能会更改。
#
# 详细的测试见test/test_reductions.py

class ReductionOpInfo(OpInfo):
    """Reduction operator information.

    An operator is a reduction operator if it reduces one or more dimensions of
    the input tensor to a single value. Reduction operators must implement the
    following signature:

    - `op(input, *args, *, dim=None, keepdim=False, **kwargs) -> Tensor`

    ReductionOpInfo tests that reduction operators implement a consistent API.
    Optional features such as reducing over multiple dimensions are captured in
    the optional keyword parameters of the ReductionOpInfo constructor.

    If a reduction operator does not yet implement the full required API of
    reduction operators, this should be documented by xfailing the failing
    tests rather than adding optional parameters to ReductionOpInfo.

    NOTE
    The API for reduction operators has not yet been finalized and some
    requirements may change.

    See tests in test/test_reductions.py
    """

    def __init__(
        self,
        name,
        *,
        # The identity value for the operator if it has one.
        identity: Optional[Any] = None,
        # The nan policy for the operator if it implements one.
        # - propagate: NaN values are propagated to the output
        # - omit: NaN values are discarded during the reduction
        nan_policy: Optional[str] = None,
        # Whether the operator supports reducing multiple dimensions.
        supports_multiple_dims: bool = True,
        # Whether the operator promotes integral to floating point dtypes.
        promotes_int_to_float: bool = False,
        # Whether the operator promotes all integral dtypes to int64.
        promotes_int_to_int64: bool = False,
        # If a specific dtype is given, then the operator always returns that
        # dtype irrespective of the input dtype. If None, the operator returns
        # the dtype according to the type promotion rules above.
        result_dtype: Optional[torch.dtype] = None,
        # Casts complex results to real (e.g. linalg.norm or torch.var)
        complex_to_real: bool = False,
        # ReductionOpInfo tests generate their own input, dim and keepdim
        # arguments and call this function to generate tuples of extra args and
        # kwargs to use when calling the op. This is required for operators that
        # have other required parameters besides the input tensor.
        generate_args_kwargs: Callable = lambda t, dim=None, keepdim=False: (
            yield tuple(),
            {},
        ),
        # Options from the OpInfo base class
        **kwargs,
        # 复制当前函数的所有局部变量，保存为实例对象的原始缩减参数
        self._original_reduction_args = locals().copy()
        # 断言 NaN 策略只能是 None, "propagate", "omit" 中的一个
        assert nan_policy in (None, "propagate", "omit")

        # 以下断言确保这些选项是互斥的
        assert not (result_dtype and promotes_int_to_float)
        assert not (result_dtype and promotes_int_to_int64)
        assert not (result_dtype and complex_to_real)
        assert not (promotes_int_to_float and promotes_int_to_int64)

        # 默认的 sample_inputs_func 函数用于 ReductionOpInfo，用来扩展从 sample_inputs_reduction 得到的输入
        # 可以使用 generate_args_kwargs 的 args 和 kwargs。只有在 sample_inputs_func 为 None 时使用。
        def sample_inputs_func(*args, **kwargs):
            kwargs["supports_multiple_dims"] = supports_multiple_dims
            kwargs["generate_args_kwargs"] = generate_args_kwargs
            yield from sample_inputs_reduction(*args, **kwargs)

        # 设置 OpInfo 的默认值，并调用基类的 __init__ 方法
        kwargs.setdefault("inplace_variant", None)
        kwargs.setdefault("sample_inputs_func", sample_inputs_func)
        super().__init__(name, promotes_int_to_float=promotes_int_to_float, **kwargs)

        # 设置实例对象的属性
        self.identity = identity
        self.nan_policy = nan_policy
        self.supports_multiple_dims = supports_multiple_dims
        self.promotes_int_to_int64 = promotes_int_to_int64
        self.complex_to_real = complex_to_real
        self.result_dtype = result_dtype
        self.generate_args_kwargs = generate_args_kwargs
# 定义一个函数用于生成元素级二元操作的基础参考输入
def _reference_inputs_elementwise_binary(
    op, device, dtype, requires_grad, exclude_zero, **kwargs
):
    # 从操作的样本输入函数中生成输入
    yield from op.sample_inputs_func(op, device, dtype, requires_grad, **kwargs)
    
    # 生成元素级二元张量
    yield from generate_elementwise_binary_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    
    # 如果数据类型不是布尔型，生成小值的元素级二元张量
    if dtype is not torch.bool:
        yield from generate_elementwise_binary_small_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad
        )
    
    # 如果数据类型不是布尔型、无符号字节或有符号字节，生成大值的元素级二元张量
    if dtype not in (torch.bool, torch.uint8, torch.int8):
        yield from generate_elementwise_binary_large_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad
        )
    
    # 生成广播的元素级二元张量
    yield from generate_elementwise_binary_broadcasting_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    
    # 生成包含标量样本的元素级二元张量
    yield from generate_elementwise_binary_with_scalar_samples(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # 生成包含标量和类型提升样本的元素级二元张量
    yield from generate_elementwise_binary_with_scalar_and_type_promotion_samples(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # 如果数据类型是浮点型或复数型，生成极值的元素级二元张量
    if dtype.is_floating_point or dtype.is_complex:
        yield from generate_elementwise_binary_extremal_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad
        )


# 注意，这些参考输入使用标量作为SampleInput.input的值，
#   许多测试要求SampleInput.input是张量或张量列表
def reference_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs):
    # 如果操作对象具有rhs_make_tensor_kwargs属性，则获取exclude_zero参数值
    if hasattr(op, "rhs_make_tensor_kwargs"):
        exclude_zero = op.rhs_make_tensor_kwargs.get("exclude_zero", False)

    # 创建_partial函数，用于生成_reference_inputs_elementwise_binary函数的偏函数
    gen = partial(
        _reference_inputs_elementwise_binary,
        op,
        device,
        dtype,
        requires_grad,
        exclude_zero,
        **kwargs,
    )

    # 生成“正常”的样本
    yield from gen()

    # 生成非连续的样本
    for sample in gen():
        yield sample.noncontiguous()

    # 生成非连续的元素级二元张量
    yield from generate_elementwise_binary_noncontiguous_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )

    # 生成任意步幅的元素级二元张量
    yield from generate_elementwise_binary_arbitrarily_strided_tensors(
        op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )


# 一个函数，用于将元素级二元操作符的专用错误输入扩展为该类操作的通用错误输入
def make_error_inputs_elementwise_binary(error_inputs_func):
    # 这里可以添加进一步的实现细节或文档
    # 定义一个函数，用于生成错误输入的生成器函数
    def error_inputs_func_wrapper(op, device, **kwargs):
        # 如果提供了错误输入函数，则生成其产生的所有错误输入
        if error_inputs_func is not None:
            yield from error_inputs_func(op, device, **kwargs)
    
        # 如果操作不支持右手边的Python标量，则生成一个SampleInput对象作为错误输入
        if not op.supports_rhs_python_scalar:
            si = SampleInput(torch.tensor((1, 2, 3), device=device), args=(2,))
            yield ErrorInput(si, error_type=Exception, error_regex="")
    
        # 如果操作不支持一个Python标量，则生成一个SampleInput对象作为错误输入
        if not op.supports_one_python_scalar:
            si = SampleInput(2, args=(torch.tensor((1, 2, 3), device=device),))
            yield ErrorInput(si, error_type=Exception, error_regex="")
    
        # 如果没有设置"skip_two_python_scalars"为True，并且操作不支持两个Python标量，则生成一个SampleInput对象作为错误输入
        if (
            not kwargs.get("skip_two_python_scalars", False)
            and not op.supports_two_python_scalars
        ):
            si = SampleInput(2, args=(3,))
            yield ErrorInput(si, error_type=Exception, error_regex="")
    
    # 返回错误输入生成器函数
    return error_inputs_func_wrapper
# 用于测试逐元素二进制运算符的函数和类。

# 返回一个生成器，生成请求设备上的连续张量对，具有请求的数据类型。
# 这个函数旨在测试逐元素二进制函数的非矢量化和矢量化代码路径，
#   以及它们对奇怪张量大小（如零维张量和元素为零的张量）的处理。
#
# 每个可迭代项将包括一个没有元素的张量，
#   零维（标量）张量，小型1D张量，中型1D张量，
#   和一个大型2D张量。
def generate_elementwise_binary_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    shapes = (
        # 没有元素的张量
        (0,),
        (1, 0, 3),
        # 零维（标量）张量
        (),
        # 小型1D张量
        (20,),
        # 中型1D张量
        (812,),
        # 大型2D张量
        (1029, 917),
    )

    # partial函数，用于创建张量
    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
        yield SampleInput(lhs, args=(rhs,))


# 返回一个生成器，生成请求设备上的任意步幅张量对，具有请求的数据类型。
def generate_elementwise_binary_arbitrarily_strided_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    # 形状，步幅，偏移
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    # partial函数，用于创建张量
    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    for shape, strides, offset in strided_cases:
        # 创建一个特定步幅的张量a，作为输入对
        a = make_arg(500).as_strided(shape, strides, offset)
        b = make_arg(shape)
        yield SampleInput(a, args=(b,))


# 返回一个生成器，生成请求设备上的连续张量对，具有请求的数据类型。
#
# 不同于前一个函数，这些张量中的值是手动指定的。
def generate_elementwise_binary_small_value_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=None
):
    if exclude_zero is None:
        if hasattr(op, "rhs_make_tensor_kwargs"):
            exclude_zero = op.rhs_make_tensor_kwargs.get("exclude_zero", False)

    # 定义有趣的值
    _unsigned_int_vals = (0, 1, 55, 127, 128, 190, 210, 220, 254)
    _int_vals = (0, -1, 1, -55, 55, -127, 127, -128)
    # 定义一组浮点数值，包括正负零和一些常见的数值
    _float_vals = (
        0.0,
        -0.0,
        -0.001,
        0.001,
        -0.25,
        0.25,
        -1.0,
        1.0,
        -math.pi / 2,
        math.pi / 2,
        -math.pi + 0.00001,
        math.pi - 0.00001,
        -math.pi,
        math.pi,
        -math.pi - 0.00001,
        math.pi + 0.00001,
    )

    # 初始化空列表，用于存放左侧和右侧的值
    l_vals = []
    r_vals = []

    # 根据数据类型的属性进行不同的处理
    if dtype.is_floating_point:
        # 如果数据类型是浮点数，则生成浮点数值的笛卡尔积
        prod = product(_float_vals, _float_vals)
    elif dtype.is_complex:
        # 如果数据类型是复数，则生成复数值的笛卡尔积
        complex_vals = product(_float_vals, _float_vals)
        # 注意需要将生成器转换为列表，以防止在下一步生成笛卡尔积时被清空
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        # 如果数据类型是有符号整数，则生成有符号整数值的笛卡尔积
        prod = product(_int_vals, _int_vals)
    elif dtype is torch.uint8:
        # 如果数据类型是无符号整数，则生成无符号整数值的笛卡尔积
        prod = product(_unsigned_int_vals, _unsigned_int_vals)
    else:
        # 如果数据类型不受支持，则抛出错误
        raise ValueError("Unsupported dtype!")

    # 遍历笛卡尔积生成的元组，将左侧和右侧的值分别添加到列表中
    for l, r in prod:
        l_vals.append(l)
        # 如果右侧的值为零且排除零的选项为真，则将右侧的值设为1，否则保留原始值
        if r == 0 and exclude_zero:
            r_vals.append(1)
        else:
            r_vals.append(r)

    # 使用生成的值列表创建 Torch 张量，设备、数据类型和梯度是否需要根据输入进行设置
    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)

    # 返回一个 SampleInput 对象，其中左侧为 lhs，右侧为 rhs
    yield SampleInput(lhs, args=(rhs,))
# 定义一个生成器函数，用于生成逐元素操作的大值张量对
def generate_elementwise_binary_large_value_tensors(
    op, *, device, dtype, requires_grad=False
):
    # 定义各种数据类型的大整数值和大浮点数值
    _large_int_vals = (-1113, 1113, -10701, 10701)
    _large_float16_vals = (-501, 501, -1001.2, 1001.2, -13437.7, 13437.7)
    _large_float_vals = _large_float16_vals + (-4988429.2, 4988429.2, -1e20, 1e20)

    # 初始化左右操作数的空列表
    l_vals = []
    r_vals = []

    # 根据数据类型选择要生成的操作数对
    if dtype == torch.float16:
        prod = product(_large_float16_vals, _large_float16_vals)
    elif dtype.is_floating_point:
        prod = product(_large_float_vals, _large_float_vals)
    elif dtype.is_complex:
        # 生成复数值对，并确保使用列表保存以避免迭代器被清空
        complex_vals = product(_large_float_vals, _large_float_vals)
        complex_vals = [complex(*x) for x in complex_vals]
        prod = product(complex_vals, complex_vals)
    elif dtype in (torch.int16, torch.int32, torch.int64):
        prod = product(_large_int_vals, _large_int_vals)
    else:
        raise ValueError("Unsupported dtype!")

    # 遍历生成的操作数对，将左右操作数分别添加到列表中
    for l, r in prod:
        l_vals.append(l)
        r_vals.append(r)

    # 创建左右操作数的张量，并作为生成器的输出
    lhs = torch.tensor(l_vals, device=device, dtype=dtype, requires_grad=requires_grad)
    rhs = torch.tensor(r_vals, device=device, dtype=dtype, requires_grad=requires_grad)

    yield SampleInput(lhs, args=(rhs,))


# Returns a generator of pairs of contiguous and noncontiguous tensors that
#   require broadcasting
def generate_elementwise_binary_broadcasting_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False


# 定义一个函数参数列表，包含多个参数：
# - op: 操作符
# - device: 设备类型
# - dtype: 数据类型
# - requires_grad: 是否需要梯度，默认为 False
# - exclude_zero: 是否排除零值，默认为 False
# 定义了一系列元组，每个元组包含两个元素，分别表示左操作数和右操作数的形状
shapes = (
    ((1,), ()),          # 单个元素和标量
    ((2,), ()),          # 两个元素和标量
    ((1,), (2,)),        # 单个元素和两个元素的元组
    ((2, 1), (2,)),      # 两行一列和两个元素的元组
    ((1, 2), (2,)),      # 一行两列和两个元素的元组
    ((3, 2), (2,)),      # 三行两列和两个元素的元组
    ((1, 3, 2), (2,)),   # 一行三列两行和两个元素的元组
    ((1, 3, 2), (3, 2)), # 一行三列两行和三行两列的元组
    ((3, 1, 2), (3, 2)), # 三行一列两行和三行两列的元组
    ((2, 3, 2), ()),     # 两行三列两行和标量
    ((3, 1, 2), (1, 3, 2)),  # 三行一列两行和一行三列两行的元组
)

# 使用偏函数创建一个make_arg函数，固定了一些参数，用于创建张量
make_arg = partial(
    make_tensor,
    device=device,
    dtype=dtype,
    requires_grad=requires_grad,
    exclude_zero=exclude_zero,
)

# 遍历所有形状和是否非连续的组合
for shape, noncontiguous in product(shapes, [True, False]):
    shape_lhs, shape_rhs = shape
    # 创建左操作数张量
    lhs = make_arg(
        shape_lhs, noncontiguous=noncontiguous, **op.lhs_make_tensor_kwargs
    )
    # 创建右操作数张量
    rhs = make_arg(
        shape_rhs, noncontiguous=noncontiguous, **op.rhs_make_tensor_kwargs
    )

    # 生成SampleInput对象，表示一组操作数对，用于二元操作
    yield SampleInput(lhs, args=(rhs,), broadcasts_input=True)



# 返回一个生成器，生成包含连续张量和标量的样本对
def generate_elementwise_binary_with_scalar_samples(
    op, *, device, dtype, requires_grad=False
):
    # 使用偏函数创建一个make_arg函数，固定了一些参数，用于创建张量
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    # 定义不同形状的样本
    shapes = ((), (3,), (5, 3), (0, 1, 3), (1, 5))

    # 如果操作支持右操作数为Python标量
    if op.supports_rhs_python_scalar:
        for shape in shapes:
            # 创建左操作数张量
            lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
            # 创建右操作数张量
            rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
            # 创建左操作数的标量值
            lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
            # 创建右操作数的标量值
            rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()

            # 生成SampleInput对象，表示一组操作数对，其中右操作数为标量
            yield SampleInput(lhs, args=(rhs_scalar,))

        # 如果操作支持左操作数为Python标量，扩展生成样本
        if op.supports_one_python_scalar:
            yield SampleInput(lhs_scalar, args=(rhs,))

    # 如果操作支持两个Python标量的情况，生成样本
    if op.supports_two_python_scalars:
        lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
        rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()

        yield SampleInput(lhs_scalar, args=(rhs_scalar,))



# 返回一个生成器，生成包含连续张量、零维张量、标量及类型提升的样本对
def generate_elementwise_binary_with_scalar_and_type_promotion_samples(
    op, *, device, dtype, requires_grad=False
):
    # 仅对逻辑和比较操作添加这些样本，算术操作不支持极端标量
    if op.name in (
        "eq",
        "ne",
        "gt",
        "ge",
        "lt",
        "le",
        "logical_and",
        "logical_or",
        "logical_xor",
    ):
        # 使用偏函数创建一个make_arg函数，固定了一些参数，用于创建张量
        make_arg = partial(
            make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
        )

        # 如果操作支持右操作数为Python标量
        if op.supports_rhs_python_scalar:
            for shape in shapes:
                # 创建左操作数张量
                lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
                # 创建右操作数张量
                rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
                # 创建左操作数的标量值
                lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
                # 创建右操作数的标量值
                rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()

                # 生成SampleInput对象，表示一组操作数对，其中右操作数为标量
                yield SampleInput(lhs, args=(rhs_scalar,))

            # 如果操作支持左操作数为Python标量，扩展生成样本
            if op.supports_one_python_scalar:
                yield SampleInput(lhs_scalar, args=(rhs,))

        # 如果操作支持两个Python标量的情况，生成样本
        if op.supports_two_python_scalars:
            lhs_scalar = make_arg((), **op.lhs_make_tensor_kwargs).item()
            rhs_scalar = make_arg((), **op.rhs_make_tensor_kwargs).item()

            yield SampleInput(lhs_scalar, args=(rhs_scalar,))
    ):
        # 创建部分应用了 make_tensor 的偏函数 make_arg，用于创建张量
        make_arg = partial(
            make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
        )
        # 定义张量的形状为 (23,)，这个形状足够大以触发向量化，并且有非向量化的尾部
        shape = (
            23,
        )
        # 定义一些特殊的数值，包括 NaN、正无穷、负无穷
        values = (float("nan"), float("inf"), -float("inf"))
        # 创建标量张量的元组，用于测试
        scalar_tensors = tuple(torch.tensor(val) for val in values)
        # 如果操作支持右操作数为 Python 标量
        if op.supports_rhs_python_scalar:
            # 创建左操作数和右操作数的张量
            lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
            rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)
            # 遍历 values 和 scalar_tensors 中的每个标量值
            for scalar in values + scalar_tensors:
                # 生成 SampleInput 对象，以左操作数和标量为参数
                yield SampleInput(lhs, args=(scalar,))
                # 如果操作支持单个 Python 标量
                if op.supports_one_python_scalar:
                    # 生成 SampleInput 对象，以标量和右操作数为参数
                    yield SampleInput(scalar, args=(rhs,))
# 返回一对非连续张量的生成器
def generate_elementwise_binary_noncontiguous_tensors(
    op, *, device, dtype, requires_grad=False, exclude_zero=False
):
    # 使用偏函数创建张量的辅助函数，指定设备、数据类型、是否需要梯度和是否排除零值
    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )

    # 生成通用的非连续张量
    lhs = make_arg((1026,), noncontiguous=True, **op.lhs_make_tensor_kwargs)
    rhs = make_arg((1026,), noncontiguous=True, **op.rhs_make_tensor_kwargs)
    # 产生样本输入，使用克隆的张量作为参数
    yield SampleInput(lhs.clone(), args=(rhs.clone(),))
    # 产生样本输入，其中 lhs 是连续的，rhs 是非连续的
    yield SampleInput(lhs.contiguous(), args=(rhs,))

    # 转置操作
    lhs = make_arg((789, 357), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((789, 357), **op.rhs_make_tensor_kwargs)
    # 产生样本输入，其中 lhs 和 rhs 都进行了转置
    yield SampleInput(lhs.T, args=(rhs.T,))

    # 更多的非连续性示例
    shapes = ((5, 7), (1024,))

    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)

        # 创建非连续的张量，并使用 copy_ 方法复制数据
        lhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
        lhs_non_contig.copy_(lhs)

        rhs_non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
        rhs_non_contig.copy_(rhs)

        # 产生样本输入，其中 lhs_non_contig 是克隆的非连续张量，rhs_non_contig 是非连续的
        yield SampleInput(lhs_non_contig.clone(), args=(rhs_non_contig.clone(),))
        # 产生样本输入，其中 lhs_non_contig 是连续的，rhs_non_contig 是非连续的
        yield SampleInput(lhs_non_contig.contiguous(), args=(rhs_non_contig,))

    # 非连续索引示例
    shape = (2, 2, 1, 2)
    lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
    rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)

    # 使用非连续索引获取部分数据
    lhs_non_contig = lhs[:, 1, ...]
    rhs_non_contig = rhs[:, 1, ...]

    # 产生样本输入，其中 lhs_non_contig 是克隆的非连续张量，rhs_non_contig 是非连续的
    yield SampleInput(lhs_non_contig.clone(), args=(rhs_non_contig.clone(),))
    # 产生样本输入，其中 lhs_non_contig 是连续的，rhs_non_contig 是非连续的
    yield SampleInput(lhs_non_contig.contiguous(), args=(rhs_non_contig,))

    # 扩展张量示例
    shapes = ((1, 3), (1, 7), (5, 7))

    for shape in shapes:
        lhs = make_arg(shape, **op.lhs_make_tensor_kwargs)
        rhs = make_arg(shape, **op.rhs_make_tensor_kwargs)

        # 使用 expand 方法扩展张量维度
        lhs_non_contig = lhs.expand(3, -1, -1)
        rhs_non_contig = rhs.expand(3, -1, -1)

        # 产生样本输入，其中 lhs_non_contig 和 rhs_non_contig 都是非连续的
        yield SampleInput(lhs_non_contig, args=(rhs_non_contig,))
    shapes = (
        ((), ()),                           # 空元组和空元组作为形状
        ((_S,), ()),                        # 包含一个元素的元组和空元组作为形状
        ((_S, 1), (_S,)),                   # 包含一个元素和一个元素的元组作为形状
        ((_M, _S), ()),                     # 包含两个元素的元组和空元组作为形状
        ((_S, _M, _S), (_M, _S)),           # 包含三个元素的元组和两个元素的元组作为形状
        ((_S, _M, _S), (_S, _M, _S)),        # 包含三个元素的元组和三个元素的元组作为形状
        ((_M, 1, _S), (_M, _S)),            # 包含三个元素的元组和两个元素的元组作为形状
        ((_M, 1, _S), (1, _M, _S)),         # 包含三个元素的元组和三个元素的元组作为形状
        ((0, 1, XS), (0, _M, XS)),          # 包含三个元素的元组和三个元素的元组作为形状
    )

    sample_kwargs = kwargs.get("sample_kwargs", {})  # 获取名为 "sample_kwargs" 的关键字参数，若不存在则设置为空字典

    for shape_lhs, shape_rhs in shapes:  # 遍历 shapes 中的每对形状
        lhs = make_arg(shape_lhs, **op.lhs_make_tensor_kwargs)  # 根据 shape_lhs 和操作的左操作数关键字参数生成左操作数
        rhs = make_arg(shape_rhs, **op.rhs_make_tensor_kwargs)  # 根据 shape_rhs 和操作的右操作数关键字参数生成右操作数
        broadcasts_input = shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs)  # 判断是否需要广播输入形状

        yield SampleInput(  # 生成一个 SampleInput 实例
            lhs, args=(rhs,), kwargs=sample_kwargs, broadcasts_input=broadcasts_input  # 设置 SampleInput 实例的参数和关键字参数
        )
# 为二元通用函数（ufuncs）提供元数据的类
# 继承自OpInfo类，用于描述接受两个张量并具有共同属性的二进制ufuncs
class BinaryUfuncInfo(OpInfo):
    """Operator information for 'universal binary functions (binary ufuncs).'
    这些是对两个张量执行的元素级函数，具有以下共同特性：
      - 它们是元素级函数
      - 输出形状由输入形状确定
      - 通常有方法和原位变体
      - 通常支持out参数
      - 通常具有NumPy或SciPy的参考实现
    有关ufunc概念的更多详细信息，请参阅NumPy的通用函数文档
    (https://numpy.org/doc/stable/reference/ufuncs.html)。
    """

    def __init__(
        self,
        name,
        *,
        sample_inputs_func=sample_inputs_elementwise_binary,  # 函数：生成二元元素级示例输入的方法
        reference_inputs_func=reference_inputs_elementwise_binary,  # 函数：生成二元元素级参考输入的方法
        error_inputs_func=None,  # 函数：生成二元元素级错误输入的方法（可选）
        lhs_make_tensor_kwargs=None,  # 左侧张量构造参数的字典（可选）
        rhs_make_tensor_kwargs=None,  # 右侧张量构造参数的字典（可选）
        always_returns_bool=False,  # 如果操作始终返回布尔张量，则设置为True
        supports_rhs_python_scalar=True,  # 操作是否允许张量 x 标量的输入
        supports_one_python_scalar=False,  # 操作是否允许标量 x 张量和张量 x 标量的输入
        supports_two_python_scalars=False,  # 操作是否允许标量 x 标量的输入
        **kwargs,
        ):
            self._original_binary_ufunc_args = locals().copy()

# 将当前作用域的局部变量复制给 `_original_binary_ufunc_args`，用于后续参考原始的二元通用函数参数。

        # Elementwise binary operations perform the equivalent of test_numpy_refs
        #   in test_binary_ufuncs, but with additional test granularity. So the
        #   generic test_ops.py test is skipped because it's redundant.
        common_skips = (
            DecorateInfo(
                unittest.skip("Skipping redundant test."),
                "TestCommon",
                "test_numpy_refs",
            ),
        )

# 创建一个常用的跳过测试信息的元组 `common_skips`，其中包含一个用于跳过重复测试的装饰器信息。

        kwargs["skips"] = kwargs.get("skips", tuple()) + common_skips

# 更新 `kwargs` 字典中的 "skips" 键，添加 `common_skips` 到已有的跳过信息中。

        super().__init__(
            name,
            sample_inputs_func=sample_inputs_func,
            reference_inputs_func=reference_inputs_func,
            error_inputs_func=make_error_inputs_elementwise_binary(error_inputs_func),
            **kwargs,
        )

# 调用父类的构造函数初始化对象，传递名称 `name` 和各种函数参数，包括样本输入函数、参考输入函数、错误输入函数和其他关键字参数 `kwargs`。

        # [lr]hs_make_tensor_kwargs are part of the OpInfo to be able to dynamically generate valid samples later on.
        if lhs_make_tensor_kwargs is None:
            lhs_make_tensor_kwargs = {}
        self.lhs_make_tensor_kwargs = lhs_make_tensor_kwargs

# 如果 `lhs_make_tensor_kwargs` 为空，则将其设为一个空字典，并将其赋值给 `self.lhs_make_tensor_kwargs`。

        if rhs_make_tensor_kwargs is None:
            rhs_make_tensor_kwargs = {}
        self.rhs_make_tensor_kwargs = rhs_make_tensor_kwargs

# 如果 `rhs_make_tensor_kwargs` 为空，则将其设为一个空字典，并将其赋值给 `self.rhs_make_tensor_kwargs`。

        self.always_returns_bool = always_returns_bool
        self.supports_rhs_python_scalar = supports_rhs_python_scalar
        self.supports_one_python_scalar = supports_one_python_scalar
        self.supports_two_python_scalars = supports_two_python_scalars

# 分别将 `always_returns_bool`、`supports_rhs_python_scalar`、`supports_one_python_scalar`、`supports_two_python_scalars` 分配给对象的相应属性。

        if self.supports_two_python_scalars:
            self.supports_one_python_scalar = True

# 如果对象支持两个 Python 标量，则自动支持一个 Python 标量。

        if self.supports_one_python_scalar:
            assert (
                supports_rhs_python_scalar
            ), "Can't support lhs and rhs Python scalars but not rhs scalars!"

# 如果对象支持一个 Python 标量，则断言必须同时支持右手边的 Python 标量，否则抛出异常信息。
# 测试元素级一元运算符的函数和类定义。

# 生成用于元素级一元运算的示例输入。
def sample_inputs_elementwise_unary(
    op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
):
    if not op_kwargs:
        op_kwargs = {}

    # 如果没有指定 "small_inputs_only" 参数，使用常规的输入大小 L，否则使用 S。
    _L = S if kwargs.get("small_inputs_only", False) else L

    # 获取操作符的定义域的下界和上界。
    low, high = op_info.domain
    # 确定数据类型是否为浮点数或复数。
    is_floating = dtype.is_floating_point or dtype.is_complex
    # 调整下界，确保在浮点数情况下不会低于定义域的上限。
    low = low if low is None or not is_floating else low + op_info._domain_eps
    # 调整上界，确保在浮点数情况下不会超过定义域的下限。
    high = high if high is None or not is_floating else high - op_info._domain_eps

    # 检查是否支持稀疏格式的测试，如 CSR、CSC、BSR、BSC。
    if (
        op_info.supports_sparse_csr
        or op_info.supports_sparse_csc
        or op_info.supports_sparse_bsr
        or op_info.supports_sparse_bsc
    ):
        # 对于稀疏压缩测试，创建二维张量。
        yield SampleInput(
            make_tensor(
                (_L, _L),
                device=device,
                dtype=dtype,
                low=low,
                high=high,
                requires_grad=requires_grad,
            ),
            kwargs=op_kwargs,
        )
    else:
        # 创建一维、空、标量张量的示例输入。
        for shape in ((_L,), (1, 0, 3), ()):
            yield SampleInput(
                make_tensor(
                    shape,
                    device=device,
                    dtype=dtype,
                    low=low,
                    high=high,
                    requires_grad=requires_grad,
                ),
                kwargs=op_kwargs,
            )


# 替换满足条件的张量值为安全值。用于阻止可能导致奇异性（如 tan(pi/2)）的值。
def _replace_values_in_tensor(tensor, condition, safe_value):
    # 根据条件生成掩码。
    mask = condition(tensor)
    # 使用安全值替换满足条件的值。
    tensor.masked_fill_(mask, safe_value)


# 辅助函数，创建一个具有有效输入的一元元素级张量。
def _make_unary_elementwise_tensor(shape, *, op, dtype, **kwargs):
    # 获取操作符的定义域的下界和上界。
    low, high = op.domain
    # 确定数据类型是否为浮点数或复数。
    is_floating = dtype.is_floating_point or dtype.is_complex
    # 调整下界，确保在浮点数情况下不会低于定义域的上限。
    low = low if low is None or not is_floating else low + op._domain_eps
    # 调整上界，确保在浮点数情况下不会超过定义域的下限。
    high = high if high is None or not is_floating else high - op._domain_eps

    # 创建具有指定形状的张量。
    a = make_tensor(shape, low=low, high=high, dtype=dtype, **kwargs)

    # 如果定义了参考数值过滤器并且数据类型不是布尔型，则应用数值替换。
    if op.reference_numerics_filter is not None and dtype is not torch.bool:
        condition, safe_value = op.reference_numerics_filter
        _replace_values_in_tensor(a, condition, safe_value)

    return a


# 将张量中的值限制在给定一元元素级操作符的定义域内。
def _filter_unary_elementwise_tensor(a, *, op):
    # 对于布尔张量，直接返回，不进行范围过滤。
    if a.dtype is torch.bool:
        return a

    # 获取操作符的定义域的下界和上界。
    low, high = op.domain
    # 确定数据类型是否为浮点数或复数。
    is_floating = a.dtype.is_floating_point or a.dtype.is_complex
    # 调整下界，确保在浮点数情况下不会低于定义域的上限。
    low = low if low is None or not is_floating else low + op._domain_eps
    # 调整上界，确保在浮点数情况下不会超过定义域的下限。
    high = high if high is None or not is_floating else high - op._domain_eps

    # 对于无符号8位整型张量，如果下界不为None，则确保不低于0。
    if a.dtype is torch.uint8 and low is not None:
        low = max(low, 0)
    # 检查数组 `a` 的数据类型是否既不是浮点型也不是复数型
    if not a.dtype.is_floating_point and not a.dtype.is_complex:
        # 如果指定了 `low`，向上取整；否则保持为 None
        low = math.ceil(low) if low is not None else None
        # 如果指定了 `high`，向下取整；否则保持为 None
        high = math.floor(high) if high is not None else None

    # 如果操作对象 `op` 包含参考数值过滤器
    if op.reference_numerics_filter is not None:
        # 解包条件和安全值
        condition, safe_value = op.reference_numerics_filter
        # 在张量 `a` 中替换符合条件的数值为安全值
        _replace_values_in_tensor(a, condition, safe_value)

    # 如果指定了 `low` 或 `high`
    if low is not None or high is not None:
        # 如果数组 `a` 的数据类型是复数型
        if a.dtype.is_complex:
            # 对复数数组 `a` 的实部进行范围限制
            a.real.clamp_(low, high)
            # 对复数数组 `a` 的虚部进行范围限制
            a.imag.clamp_(low, high)
        else:
            # 对非复数数组 `a` 进行整体范围限制
            a.clamp_(min=low, max=high)

    # 返回处理后的数组 `a`
    return a
# 生成按元素操作的一元张量的生成器函数，根据给定的操作符和参数生成不同类型的张量
def generate_elementwise_unary_tensors(op, *, device, dtype, requires_grad, **kwargs):
    # 对布尔类型进行特殊处理
    if dtype is torch.bool:
        # 定义包含不同布尔类型张量的元组
        tensors = (
            torch.empty(0, device=device, dtype=torch.bool),
            torch.tensor(True, device=device),
            torch.tensor(False, device=device),
            torch.tensor((True, False), device=device),
            make_tensor((812,), device=device, dtype=dtype),
            make_tensor((1029, 917), device=device, dtype=dtype),
        )
        # 遍历并生成每个张量的样本输入
        for a in tensors:
            yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])

    # 定义不同形状的张量
    shapes = (
        (1029, 917),
        (812,),
        # 空尺寸
        (0,),
        (0, 3, 3),
        (1, 0, 5),
        (6, 0, 0, 0),
        (3, 0, 1, 0),
    )

    # 部分应用生成一元元素操作的张量函数
    make_arg = partial(
        _make_unary_elementwise_tensor,
        op=op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    # 遍历不同形状，生成并返回每个张量的样本输入
    for shape in shapes:
        a = make_arg(shape)
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])


# 生成小值一元元素操作张量的生成器函数
def generate_elementwise_unary_small_value_tensors(op, *, device, dtype, requires_grad=False):
    # 调用生成二元小值元素操作张量的生成器函数，过滤得到一元元素操作的输入样本
    for sample in generate_elementwise_binary_small_value_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    ):
        a = _filter_unary_elementwise_tensor(sample.input, op=op)
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])


# 生成大值一元元素操作张量的生成器函数
def generate_elementwise_unary_large_value_tensors(op, *, device, dtype, requires_grad=False):
    # 调用生成二元大值元素操作张量的生成器函数，过滤得到一元元素操作的输入样本
    for sample in generate_elementwise_binary_large_value_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    ):
        a = _filter_unary_elementwise_tensor(sample.input, op=op)
        yield SampleInput(sample.input, kwargs=op.sample_kwargs(device, dtype, a)[0])


# 生成极端值一元元素操作张量的生成器函数
def generate_elementwise_unary_extremal_value_tensors(op, *, device, dtype, requires_grad=False):
    # 调用生成二元极端值元素操作张量的生成器函数，直接返回一元元素操作的输入样本
    for sample in generate_elementwise_binary_extremal_value_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad
    ):
        yield SampleInput(
            sample.input, kwargs=op.sample_kwargs(device, dtype, sample.input)[0]
        )


# 生成非连续一元元素操作张量的生成器函数
def generate_elementwise_unary_noncontiguous_tensors(op, *, device, dtype, requires_grad=False):
    # 部分应用生成一元元素操作的张量函数，标记为非连续
    make_arg = partial(
        _make_unary_elementwise_tensor,
        op=op,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    # 生成具有非连续性的通用张量
    t = make_arg((1026,), noncontiguous=True)
    yield SampleInput(t, kwargs=op.sample_kwargs(device, dtype, t)[0])

    # 生成转置后的张量
    t = make_arg((1024, 1024)).T
    yield SampleInput(t, kwargs=op.sample_kwargs(device, dtype, t)[0])

    # 扩展张量的不同形状
    shapes = ((1, 3), (1, 7), (5, 7))
    # 对于 shapes 列表中的每个元素 shape，执行以下操作：
    for shape in shapes:
        # 调用 make_arg 函数，生成一个新的张量 t，该张量基于 shape 参数
        t = make_arg(shape)
        # 对张量 t 进行非连续扩展，将其扩展为三维，第一维度不变，后两维度为原始张量的大小
        t_non_contig = t.expand(3, -1, -1)
        # 使用 op.sample_kwargs 函数为当前操作生成样本参数 kwargs，该函数返回一个元组，取第一个元素
        # 传递 device、dtype 和 t_non_contig 作为参数
        yield SampleInput(
            t_non_contig, kwargs=op.sample_kwargs(device, dtype, t_non_contig)[0]
        )
# 生成任意步幅的元素级一元张量
def generate_elementwise_unary_arbitrarily_strided_tensors(
    op, *, device, dtype, requires_grad=False
):
    # 定义不同的步幅情况
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),   # 形状为 (5, 6, 2)，步幅为 (1, 1, 7)，偏移为 2
        ((5, 5, 4), (1, 1, 7), 2),   # 形状为 (5, 5, 4)，步幅为 (1, 1, 7)，偏移为 2
        ((5, 5, 2), (4, 5, 7), 3),   # 形状为 (5, 5, 2)，步幅为 (4, 5, 7)，偏移为 3
        ((5, 5, 2), (5, 5, 7), 3),   # 形状为 (5, 5, 2)，步幅为 (5, 5, 7)，偏移为 3
        ((5, 5, 2), (5, 5, 5), 3),   # 形状为 (5, 5, 2)，步幅为 (5, 5, 5)，偏移为 3
        ((9, 5, 2), (0, 1, 7), 3),   # 形状为 (9, 5, 2)，步幅为 (0, 1, 7)，偏移为 3
    )

    # 部分函数应用，创建张量的辅助函数
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    
    # 遍历不同步幅的情况
    for shape, strides, offset in strided_cases:
        # 创建张量并设置为给定步幅和偏移量的视图
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        # 生成一个样本输入对象，包括张量和操作的关键字参数
        yield SampleInput(a, kwargs=op.sample_kwargs(device, dtype, a)[0])


# 保持元素级二元生成器的一致性重用
# TODO: 将来泛化参考生成器以处理 n 元元素级操作
def _reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs):
    # 从操作中获取样本输入的生成器
    yield from op.sample_inputs_func(op, device, dtype, requires_grad, **kwargs)

    # 生成元素级一元张量的样本输入
    yield from generate_elementwise_unary_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )

    # 如果数据类型不是 torch.bool，则生成小数值的元素级一元张量
    if dtype is not torch.bool:
        yield from generate_elementwise_unary_small_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )
    
    # 如果数据类型不是 torch.bool、torch.uint8 或 torch.int8，并且操作处理大浮点数，
    # 或者数据类型既不是浮点数也不是复数，则生成大数值的元素级一元张量
    if dtype not in (torch.bool, torch.uint8, torch.int8) and (
        op.handles_large_floats
        or (not dtype.is_floating_point and not dtype.is_complex)
    ):
        yield from generate_elementwise_unary_large_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )

    # 如果数据类型是浮点数或操作处理复数极端值，则生成极端值的元素级一元张量
    if dtype.is_floating_point or (
        op.handles_complex_extremal_values and dtype.is_complex
    ):
        yield from generate_elementwise_unary_extremal_value_tensors(
            op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )


# 生成元素级一元操作的参考输入
def reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs):
    # 部分函数应用，创建元素级一元参考输入的辅助函数
    gen = partial(
        _reference_inputs_elementwise_unary, op, device, dtype, requires_grad, **kwargs
    )

    # 生成“正常”样本
    yield from gen()

    # 生成非连续样本
    for sample in gen():
        yield sample.noncontiguous()

    # 生成非连续元素级一元张量
    yield from generate_elementwise_unary_noncontiguous_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )

    # 生成任意步幅的元素级一元张量
    yield from generate_elementwise_unary_arbitrarily_strided_tensors(
        op, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
    )


# 用于单个张量接受通用一元函数（ufuncs）的元数据类，具有共同的属性：
# - UnaryUfuncInfo 继承自 OpInfo 类
class UnaryUfuncInfo(OpInfo):
    """用于通用一元函数（unary ufuncs）的操作符信息。"""
    """
    These are functions of a single tensor with common properties like:
      - they are elementwise functions
      - the input shape is the output shape
      - they typically have method and inplace variants
      - they typically support the out kwarg
      - they typically have NumPy or SciPy references
    See NumPy's universal function documentation
    (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
    about the concept of ufuncs.
    """
    
    # 初始化函数，用于创建一个单元素张量函数对象
    def __init__(
        self,
        name,  # 函数的字符串名称
        *,
        dtypes=floating_types(),  # 支持的数据类型，默认为浮点数类型
        domain=(None, None),  # 函数的定义域 [low, high)
        handles_complex_extremal_values=True,  # 是否正确处理复数极值（如 nan/inf）
        handles_large_floats=True,  # 是否正确处理大浮点数值（如 1e20）
        supports_complex_to_float=False,  # 是否支持从复数输入安全地转换为实数输出，例如 angle
        sample_inputs_func=sample_inputs_elementwise_unary,  # 获取示例输入的函数
        reference_inputs_func=reference_inputs_elementwise_unary,  # 获取参考输入的函数
        sample_kwargs=lambda device, dtype, input: ({}, {}),  # 提供输入的关键字参数的实用函数
        reference_numerics_filter=None,  # 过滤器，排除在定义域范围内但不应进行测试的值
        **kwargs,  # 其他可选关键字参数
    ):
        # 将所有传入参数保存在字典中，用于后续参考
        self._original_unary_ufunc_args = locals().copy()
    
        # 调用父类的初始化方法
        super().__init__(
            name,
            dtypes=dtypes,
            sample_inputs_func=sample_inputs_func,
            reference_inputs_func=reference_inputs_func,
            **kwargs,
        )
    
        # 设置函数的定义域
        self.domain = domain
        # 是否正确处理复数的极值
        self.handles_complex_extremal_values = handles_complex_extremal_values
        # 是否正确处理大浮点数值
        self.handles_large_floats = handles_large_floats
        # 是否支持从复数输入安全地转换为实数输出
        self.supports_complex_to_float = supports_complex_to_float
        # 参考数值过滤器
        self.reference_numerics_filter = reference_numerics_filter
    
        # test_unary_ufuncs.py 生成自己的输入以测试操作符在切片张量、非连续张量等上的一致性。
        # `sample_kwargs` 是一个实用函数，用于在需要时为 torch 操作符和参考 NumPy 操作符提供关键字参数。
        # 它应该返回两个字典，第一个包含 torch 操作符的关键字参数，第二个包含参考 NumPy 操作符的关键字参数。
        self.sample_kwargs = sample_kwargs
    
        # Epsilon 用于确保 grad 和 gradgrad 检查不测试函数域之外的值。
        self._domain_eps = 1e-5
def sample_inputs_spectral_ops(self, device, dtype, requires_grad=False, **kwargs):
    # 检查 dtype 是否为 torch.complex32 或 torch.half，确定是否为 fp16 或 chalf 类型
    is_fp16_or_chalf = dtype == torch.complex32 or dtype == torch.half
    if not is_fp16_or_chalf:
        # 如果不是 fp16 或 chalf 类型，则创建 nd_tensor 和 oned_tensor
        nd_tensor = partial(
            make_tensor,
            (S, S + 1, S + 2),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        oned_tensor = partial(
            make_tensor, (31,), device=device, dtype=dtype, requires_grad=requires_grad
        )
    else:
        # 如果是 fp16 或 chalf 类型，则根据 self.name 设置 shapes 和可能的 low、high 值
        if self.name in ["fft.hfft", "fft.irfft", "_refs.fft.hfft", "_refs.fft.irfft"]:
            shapes = ((2, 9, 9), (33,))
        elif self.name in [
            "fft.hfft2",
            "fft.irfft2",
            "_refs.fft.hfft2",
            "_refs.fft.irfft2",
        ]:
            shapes = ((2, 8, 9), (33,))
        elif self.name in [
            "fft.hfftn",
            "fft.irfftn",
            "_refs.fft.hfftn",
            "_refs.fft.irfftn",
        ]:
            shapes = ((2, 2, 33), (33,))
            # 由于 float16 过饱和，调整限制以避免测试不稳定
            # 参考：https://github.com/pytorch/pytorch/pull/81416
            low = -1.0
            high = 1.0
        else:
            shapes = ((2, 8, 16), (32,))
        # 根据 shapes、device、low、high 创建 nd_tensor 和 oned_tensor
        nd_tensor = partial(
            make_tensor,
            shapes[0],
            device=device,
            low=low,
            high=high,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        oned_tensor = partial(
            make_tensor,
            shapes[1],
            device=device,
            low=low,
            high=high,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    # 如果是 ND 类型的操作，生成不同的 SampleInput 对象
    if self.ndimensional == SpectralFuncType.ND:
        yield SampleInput(
            nd_tensor(),
            s=(3, 10) if not is_fp16_or_chalf else (4, 8),
            dim=(1, 2),
            norm="ortho",
        )
        yield SampleInput(nd_tensor(), norm="ortho")
        yield SampleInput(nd_tensor(), s=(8,))
        yield SampleInput(oned_tensor())
        # 生成多个 SampleInput 对象，针对不同的维度配置
        yield from (SampleInput(nd_tensor(), dim=dim) for dim in [-1, -2, -3, (0, -1)])
    # 如果数据是二维光谱函数类型，则生成以下示例输入
    elif self.ndimensional == SpectralFuncType.TwoD:
        # 生成一个样本输入，使用默认的多维张量，指定形状和维度
        yield SampleInput(
            nd_tensor(),
            s=(3, 10) if not is_fp16_or_chalf else (4, 8),  # 如果不是 FP16 或 CHalf，则设置特定形状；否则设置另一种形状
            dim=(1, 2),  # 指定维度
            norm="ortho",  # 使用正交归一化
        )
        # 生成一个样本输入，使用默认的多维张量，应用正交归一化
        yield SampleInput(nd_tensor(), norm="ortho")
        # 生成一个样本输入，使用默认的多维张量，指定特定形状
        yield SampleInput(nd_tensor(), s=(6, 8) if not is_fp16_or_chalf else (4, 8))
        # 生成一个样本输入，使用默认的多维张量，指定维度为0
        yield SampleInput(nd_tensor(), dim=0)
        # 生成一个样本输入，使用默认的多维张量，指定负数维度
        yield SampleInput(nd_tensor(), dim=(0, -1))
        # 生成一个样本输入，使用默认的多维张量，指定负数维度
        yield SampleInput(nd_tensor(), dim=(-3, -2, -1))
    else:
        # 如果数据不是二维光谱函数类型，则生成以下示例输入
        yield SampleInput(
            nd_tensor(),
            n=10 if not is_fp16_or_chalf else 8,  # 如果不是 FP16 或 CHalf，则设置特定数量；否则设置另一种数量
            dim=1,  # 指定维度为1
            norm="ortho",  # 使用正交归一化
        )
        # 生成一个样本输入，使用默认的多维张量，应用正交归一化
        yield SampleInput(nd_tensor(), norm="ortho")
        # 生成一个样本输入，使用默认的多维张量，指定特定数量
        yield SampleInput(nd_tensor(), n=7 if not is_fp16_or_chalf else 8)
        # 生成一个样本输入，使用默认的一维张量
        yield SampleInput(oned_tensor())
        # 生成一系列样本输入，每个输入使用默认的多维张量，并指定不同的负数维度
        yield from (SampleInput(nd_tensor(), dim=dim) for dim in [-1, -2, -3])
# 定义枚举类型 SpectralFuncType，包含 OneD、TwoD、ND 三种谱函数类型
SpectralFuncType = Enum("SpectralFuncType", ("OneD", "TwoD", "ND"))


# 用于描述 torch.fft 中快速傅里叶变换的元数据类
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms."""

    def __init__(
        self,
        name,  # 函数的字符串名称
        *,
        ref=None,  # 参考实现（通常在 np.fft 命名空间中）
        dtypes=floating_and_complex_types(),  # 支持的数据类型，默认为浮点数和复数类型
        ndimensional: SpectralFuncType,  # 谱函数的维度类型，可以是 OneD、TwoD 或 ND
        sample_inputs_func=sample_inputs_spectral_ops,  # 生成样本输入的函数
        decorators=None,  # 装饰器列表，用于修饰操作符信息
        **kwargs,  # 其他参数
    ):
        self._original_spectral_func_args = dict(locals()).copy()
        self._original_spectral_func_args.update(kwargs)

        # 默认装饰器列表，包括 skipCPUIfNoFFT 和特定的测试装饰器
        decorators = list(decorators) if decorators is not None else []
        decorators += [
            skipCPUIfNoFFT,
            DecorateInfo(
                toleranceOverride({torch.chalf: tol(4e-2, 4e-2)}),
                "TestCommon",
                "test_complex_half_reference_testing",
            ),
        ]

        super().__init__(
            name=name,
            dtypes=dtypes,
            decorators=decorators,
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.ref = ref  # 设置参考实现
        self.ndimensional = ndimensional  # 设置谱函数的维度类型


# 用于形状操作（如 tile 和 roll）的专门化 OpInfo 类的早期版本
class ShapeFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for Shape manipulating operations like tile and roll"""

    def __init__(
        self,
        name,  # 函数的字符串名称
        *,
        ref,  # 参考函数
        dtypes=floating_types(),  # 支持的数据类型，默认为浮点数类型
        dtypesIfCUDA=None,  # 如果在 CUDA 下支持的数据类型
        dtypesIfROCM=None,  # 如果在 ROCM 下支持的数据类型
        dtypesIfXPU=None,  # 如果在 XPU 下支持的数据类型
        sample_inputs_func=None,  # 生成样本输入的函数
        **kwargs,  # 其他参数
    ):
        super().__init__(
            name,
            dtypes=dtypes,
            dtypesIfCUDA=dtypesIfCUDA,
            dtypesIfROCM=dtypesIfROCM,
            dtypesIfXPU=dtypesIfXPU,
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.ref = ref  # 设置参考函数


# 用于 foreach 操作的样本输入生成函数
def sample_inputs_foreach(
    self,
    device,
    dtype,
    N,
    *,
    noncontiguous=False,  # 是否生成非连续的张量
    same_size=False,  # 是否生成相同大小的张量
    low=None,  # 随机数范围的下界
    high=None,  # 随机数范围的上界
    zero_size: bool,  # 是否生成大小为零的张量
    requires_grad: bool,  # 是否需要梯度
    # 互斥的选项，与 same_size 和 zero_size 冲突，要么都是 True，要么都是 False
    intersperse_empty_tensors: bool = False,
):
    if zero_size:
        return [torch.empty(0, dtype=dtype, device=device) for _ in range(N)]  # 返回 N 个大小为零的张量列表
    if same_size:
        return [
            make_tensor(
                (N, N),  # 张量的形状为 (N, N)
                dtype=dtype,
                device=device,
                noncontiguous=noncontiguous,
                low=low,
                high=high,
                requires_grad=requires_grad,
            )
            for _ in range(N)
        ]  # 返回 N 个相同大小的张量列表
    # 如果不满足前述条件，则返回一个列表，其中包含一些空张量和最后两个张量为空（参见 issue #100701）
    else:
        return [
            # 如果满足条件：索引能被 3 整除或者索引大于等于 N-2，并且 intersperse_empty_tensors 为真，则返回一个空张量
            torch.empty(0, dtype=dtype, device=device, requires_grad=requires_grad)
            # 否则，调用 make_tensor 函数创建一个具有指定属性的张量
            if (i % 3 == 0 or i >= N - 2) and intersperse_empty_tensors
            else make_tensor(
                (N - i, N - i),  # 创建一个形状为 (N-i, N-i) 的张量
                dtype=dtype,  # 张量的数据类型
                device=device,  # 张量的设备
                noncontiguous=noncontiguous,  # 是否是非连续的张量
                low=low,  # 随机数的下界
                high=high,  # 随机数的上界
                requires_grad=requires_grad,  # 是否需要梯度
            )
            for i in range(N)  # 对于范围在 0 到 N-1 的每一个索引 i
        ]
# 定义函数 get_foreach_method_names，用于获取 torch 库中特定方法的引用
def get_foreach_method_names(name):
    # 构造 inplace 方法的名称，例如将输入名称加上 "_foreach_" 前缀
    op_name = "_foreach_" + name
    # 构造 inplace 方法的完整名称，加上 "_" 后缀
    inplace_op_name = op_name + "_"

    # 获取 torch 模块中名称为 op_name 的函数的引用，如果不存在则为 None
    op = getattr(torch, op_name, None)
    # 获取 torch 模块中名称为 inplace_op_name 的函数的引用，如果不存在则为 None
    inplace_op = getattr(torch, inplace_op_name, None)

    # 获取 torch 模块中名称为 name 的函数的引用，如果不存在则为 None
    ref = getattr(torch, name, None)
    # 获取 torch.Tensor 类中名称为 name + "_" 的方法的引用，如果不存在则为 None
    ref_inplace = getattr(torch.Tensor, name + "_", None)

    # 返回获取到的四个引用
    return op, inplace_op, ref, ref_inplace


# 定义一个数据类 ForeachFuncInfo，它继承自 OpInfo 类
@dataclass
class ForeachFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for foreach functions

    The main differences from the parent class are (a) `dtypes`, `dtypesIfCUDA`, and `dtypesIfROCM`
    are set to `get_all_dtypes(include_qint=False)`, and (b) the following arguments.

    ``supports_alpha_param=True`` means that the function supports a python scalar (``numbers.Number``)
    as the last keyword argument such as `_foreach_add`.
    ``supports_scalar_self_arg=True`` means that the function can take a python scalar as its first argument.
    Currently only `_foreach_pow` supports this.
    ``backward_requires_result=True``, which could sound self-explanatory, means that the function uses
    the forward result for its backward computation.
    """

    # 表示该类特化于 foreach 函数的 OpInfo 的早期版本

    # 指定支持的 alpha 参数为 True，表示该函数支持将 Python 标量作为最后一个关键字参数
    supports_alpha_param: bool = False
    # 指定支持的 scalar self 参数为 False，表示该函数不支持将 Python 标量作为第一个参数
    supports_scalar_self_arg: bool = False
    # 指定 backward 需要 result 参数为 False，表示该函数在反向计算时不使用前向结果
    backward_requires_result: bool = False
    def __post_init__(self):
        (
            foreach_method,
            foreach_method_inplace,
            torch_ref_method,
            torch_ref_inplace,
        ) = get_foreach_method_names(self.name)
        # 获取当前操作的迭代方法名称和相关的 Torch 方法名称

        if not self.supports_out:
            # 如果当前操作不支持输出结果（out 参数）
            # 注意：对于 "zero" 操作，`foreach_method` 为 `None`，但此时调用 `_getattr_qual` 
            # 在 `OpInfo.__post_init__` 中会失败，因为此时 `_foreach_zero` 未定义。
            # 为了跳过这种限定，设置一个类似的 Torch 函数。
            assert foreach_method is None
            assert torch_ref_method is None
            foreach_method = foreach_method_inplace
            torch_ref_method = torch_ref_inplace

        self.dtypes = _dispatch_dtypes(get_all_dtypes(include_qint=False))
        # 设置当前对象的数据类型，调用 `_dispatch_dtypes` 处理所有数据类型（排除 qint）。

        self.op = foreach_method
        self.method_variant = foreach_method
        self.ref = torch_ref_method
        self.inplace_variant = foreach_method_inplace
        self.ref_inplace = torch_ref_inplace
        self.has_no_in_place = self.inplace_variant is None
        # 设置当前对象的操作方法、方法变体、参考方法以及相关的就地操作方法和就地参考方法。
        # 设置是否具有就地操作的标志。

        name = self.name
        self.name = f"_foreach_{name}"
        # 修改对象的名称，添加 "_foreach_" 前缀。

        if name == "norm":
            self.ref = torch.linalg.vector_norm
        elif name == "minimum":
            # 如果操作是 "minimum"，因为最小值操作不支持就地操作或标量
            self.ref = torch.clamp_max
            self.ref_inplace = torch.Tensor.clamp_max_
        elif name == "maximum":
            # 如果操作是 "maximum"，因为最大值操作不支持就地操作或标量
            self.ref = torch.clamp_min
            self.ref_inplace = torch.Tensor.clamp_min_
        # 根据操作名称调整参考方法和就地参考方法。

        # 以下设置 `dtypesIfCUDA` 和 `dtypesIfROCM`。
        super().__post_init__()

    def sample_zero_size_inputs(self, device, dtype, requires_grad=False, **kwargs):
        if not hasattr(self.sample_inputs_func, "sample_zero_size_tensor_inputs"):
            return []
        # 如果 `sample_inputs_func` 对象没有属性 "sample_zero_size_tensor_inputs"，返回空列表。
        
        return self.sample_inputs_func.sample_zero_size_tensor_inputs(
            self, device, dtype, requires_grad, **kwargs
        )
    # 使用 `sample_inputs_func` 中的方法生成零大小的输入样本。
# 用于包装梯度检查的函数，针对接收埃尔米特矩阵作为输入的函数进行适配
def gradcheck_wrapper_hermitian_input(op, input, *args, **kwargs):
    """Gradcheck wrapper for functions that take Hermitian matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the Hermitian property of the input.
    """
    # 将输入矩阵与其共轭转置相加，作为修改后函数的输入，然后调用原始操作函数
    return op(input + input.mH, *args, **kwargs)


# 用于包装梯度检查的函数，针对接收下三角或上三角矩阵作为输入的函数进行适配
def gradcheck_wrapper_triangular_input(op, *args, upper=False, idx=0, **kwargs):
    """Gradcheck wrapper for functions that take lower or upper triangular matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the triangular property of the input.
    `idx` is used to specific which `args[idx]` is to be triangularized.
    """
    # 根据参数中的 `upper` 标志位，获取需要转换为三角矩阵的参数
    triangular_arg = args[idx].triu() if upper else args[idx].tril()
    # 调用原始操作函数，将转换后的三角矩阵作为参数之一传递
    return op(*args[:idx], triangular_arg, *args[idx + 1 :], upper, **kwargs)


# 用于包装梯度检查的函数，针对接收具有实数和正对角线的下三角或上三角矩阵作为输入的函数进行适配
def gradcheck_wrapper_triangular_input_real_positive_diagonal(
    op, *args, upper=False, idx=0, **kwargs
):
    """Gradcheck wrapper for functions that take lower/upper triangular matrices
    with real and positive diagonals, for example, cholesky-like operations.
    """
    # 提取参数中指定索引处矩阵的对角线元素
    arg = args[idx]
    arg_diag = arg.diagonal(0, -2, -1)
    # 构建对角线元素的对角矩阵
    arg_diag_embed = torch.diag_embed(arg_diag)
    # 创建与参数矩阵相同形状的全一张量，然后构建全一张量的对角矩阵
    id_diag_tensor = torch.ones_like(arg_diag)
    id_tensor = torch.diag_embed(id_diag_tensor)
    # 计算新的参数矩阵，用原始矩阵减去其对角线元素对角矩阵，再加上全一张量对角矩阵
    new_arg = arg - arg_diag_embed + id_tensor
    # 调用处理三角矩阵输入的包装函数，将新的参数矩阵作为修改后的参数传递
    return gradcheck_wrapper_triangular_input(
        op, *args[:idx], new_arg, *args[idx + 1 :], upper=upper, idx=idx, **kwargs
    )


# 用于包装梯度检查的函数，针对带有掩码操作的函数进行适配
def gradcheck_wrapper_masked_operation(op, input, *args, **kwargs):
    """Gradcheck wrapper for masked operations.

    When mask is specified, replaces masked-out elements with zeros.

    Use for operations that produce non-finite masked-out elements,
    for instance, for minimum and maximum reductions.
    """
    # 调用原始操作函数，获取基本输出结果
    output = op(input, *args, **kwargs)
    # 获取掩码参数
    mask = kwargs.get("mask")
    # 如果存在掩码，调用 torch.masked._output_mask 获取掩码输出，并用零替换掩码输出
    if mask is not None:
        output_mask = torch.masked._output_mask(op, input, *args, **kwargs)
        output = torch.where(output_mask, output, output.new_zeros([]))
    return output


# 用于包装梯度检查的函数，针对带有掩码点对点操作的函数进行适配
def gradcheck_wrapper_masked_pointwise_operation(op, input, *args, **kwargs):
    """Gradcheck wrapper for masked pointwise operations. Assumes that the result
    will be masked iff both tensors are masked at a specific index

    When mask is specified, replaces masked-out elements with zeros.

    Use for operations that produce non-finite masked-out elements,
    for instance, for minimum and maximum reductions.
    """
    # 调用原始操作函数，获取基本输出结果
    output = op(input, *args, **kwargs)
    # 获取输入掩码和其他掩码参数
    input_mask = kwargs.get("input_mask")
    other_mask = kwargs.get("other_mask")
    # 检查输入掩码和其他掩码是否都不为 None
    if input_mask is not None and other_mask is not None:
        # 使用逻辑与操作符合并输入掩码和其他掩码
        combined_mask = torch.logical_and(input_mask, other_mask)
        # 创建一个新的关键字参数字典，包括合并后的掩码和其他所有的关键字参数
        new_kwargs = dict(mask=combined_mask, **kwargs)
        # 调用 torch.masked._input_mask 函数，传入输入张量 input，位置参数 *args 和新的关键字参数字典 new_kwargs
        output_mask = torch.masked._input_mask(input, *args, **new_kwargs)
        # 使用 torch.where 函数根据 output_mask 来选择输出值或零张量的对应元素
        output = torch.where(output_mask, output, output.new_zeros([]))
    # 返回处理后的输出张量
    return output
def clone_sample(sample, **kwargs):
    """
    Given a SampleInput, this function analyzes its input, args and kwargs,
    and produces a copy with each non-Tensor entry being copied by reference,
    and with each Tensor entry cloned with `t.clone().requires_grad_(t.requires_grad)`
    """

    # 定义一个内部函数，用于克隆 Tensor 对象
    def clone_tensor(t):
        if isinstance(t, torch.Tensor):
            # 如果是 Tensor 对象，则克隆该 Tensor，并保留其梯度信息
            return t.detach().clone().requires_grad_(t.requires_grad)
        else:
            # 如果不是 Tensor 对象，则直接返回原对象（引用复制）
            return t

    # 如果传入了关键字参数 kwargs，则使用传入的 kwargs；否则使用 sample 对象的 kwargs 属性
    sample_kwargs = kwargs if kwargs else sample.kwargs

    # 返回一个新的 SampleInput 对象，其中包括：
    # - sample.input 的克隆对象（可能是 Tensor 或者非 Tensor）
    # - sample.args 的每个元素经过 clone_tensor 函数处理后的元组
    # - sample_kwargs 字典的每个值经过 clone_tensor 函数处理后的新字典
    return SampleInput(
        clone_tensor(sample.input),
        args=tuple(map(clone_tensor, sample.args)),
        kwargs={k: clone_tensor(v) for k, v in sample_kwargs.items()},
    )
```