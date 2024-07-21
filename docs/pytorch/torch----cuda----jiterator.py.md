# `.\pytorch\torch\cuda\jiterator.py`

```py
# mypy: allow-untyped-defs
# 引入正则表达式库和类型定义库
import re
from typing import Callable, List

import torch
from torch import Tensor

__all__: List[str] = []  # 导出变量列表，初始化为空列表


class _CodeParser:
    def __init__(self, code_string: str):
        optional_ws = r"\s*"  # 匹配零或多个空白字符
        required_ws = r"\s+"  # 匹配至少一个空白字符
        template_params = r"(?P<template_params>\<.+\>)"  # 匹配模板参数
        return_type = r"(?P<return_type>\w+)"  # 匹配返回类型
        function_name = r"(?P<function_name>\w+)"  # 匹配函数名
        function_params = r"(?P<function_params>\(.+\))"  # 匹配函数参数列表
        function_body = r"(?P<function_body>\{.+\})"  # 匹配函数体

        # 构建完整的正则表达式模式
        pattern = (
            optional_ws
            + "template"
            + optional_ws
            + template_params
            + optional_ws
            + return_type
            + required_ws
            + function_name
            + optional_ws
            + function_params
            + optional_ws
            + function_body
            + optional_ws
        )

        # 使用正则表达式匹配给定的代码字符串，支持多行匹配
        result = re.match(
            pattern, code_string, re.DOTALL
        )  # DOTALL 标记用于匹配多行

        if result is None:
            raise Exception(  # noqa: TRY002
                f"Couldn't parse code, please check correctness:\n {code_string}"
            )

        # 提取匹配结果中的各个部分作为对象的属性
        self.template_params = result["template_params"]
        self.return_type = result["return_type"]
        self.function_name = result["function_name"]
        self.function_params = result["function_params"]
        self.function_body = result["function_body"]


class _JittedFunction:
    def __init__(
        self, code_string: str, return_by_ref: bool, num_outputs: int, **kwargs
    ):
        self.code_string = code_string

        # 断言：如果按引用返回，则只支持单一输出；否则需要指定单一输出的数量
        assert (
            return_by_ref or num_outputs == 1
        ), "Return by value only works for single output. "
        self.return_by_ref = return_by_ref
        self.num_outputs = num_outputs

        # 解析给定的代码字符串
        parsed_code = _CodeParser(code_string)
        self.kernel_name = parsed_code.function_name  # 获取函数名作为内核名称

        self.kwargs_dict = kwargs  # 复制附加参数字典
        self.is_cuda_available = torch.cuda.is_available()  # 检查CUDA是否可用

    def __call__(self, *tensors: Tensor, **kwargs):
        # Jiterator 遵循 torch.cuda 的惰性初始化行为
        # 延迟在函数调用时检查 CUDA 的可用性
        assert (
            self.is_cuda_available
        ), "Jiterator is only supported on CUDA and ROCm GPUs, none are available."

        assert len(tensors) <= 8, "jiterator only supports up to 8 tensor inputs."

        expanded_kwargs = self.kwargs_dict.copy()
        for key, value in kwargs.items():
            if key in self.kwargs_dict:
                expanded_kwargs[key] = value
            else:
                raise KeyError(f"{key} is not declared in function definition")

        # 调用底层 Torch CUDA 函数编译和启动内核
        return torch._C._cuda_jiterator_compile_and_launch_kernel(
            self.code_string,
            self.kernel_name,
            self.return_by_ref,
            self.num_outputs,
            tensors,
            expanded_kwargs,
        )
# 创建一个用于生成基于 jiterator 的 CUDA 内核的函数，用于支持逐元素操作
def _create_jit_fn(code_string: str, **kwargs) -> Callable:
    """
    Create a jiterator-generated cuda kernel for an elementwise op.

    The code string has to be a valid CUDA function that describes the computation for a single element. The code
    string has to follow the c++ template pattern, as shown in the example below. This function will be inlined
    into elementwise kernel template, and compiled on the fly. Compiled kernel will be cached in memory, as well as
    local temp dir.

    Jiterator-generated kernels accepts noncontiguous tensors, and supports broadcasting and type promotion.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return by value.
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> T my_kernel(T x, T y, T alpha) { return -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device='cuda')
        b = torch.rand(3, device='cuda')
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    code_string also allows multiple function definitions, and the last function will be treated as the entry function.

    Example::

        code_string = "template <typename T> T util_fn(T x, T y) { return ::sin(x) + ::cos(y); }"
        code_string += "template <typename T> T my_kernel(T x, T y, T val) { return ::min(val, util_fn(x, y)); }"
        jitted_fn = create_jit_fn(code_string, val=0.0)
        a = torch.rand(3, device='cuda')
        b = torch.rand(3, device='cuda')
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b)  # using default val=0.0

    Jiterator can be used together with python registration to override an operator's cuda kernel.
    Following example is overriding gelu's cuda kernel with relu.

    Example::

        code_string = "template <typename T> T my_gelu(T a) { return a > 0 ? a : 0; }"
        my_gelu = create_jit_fn(code_string)
        my_lib = torch.library.Library("aten", "IMPL")
        my_lib.impl('aten::gelu', my_gelu, "CUDA")
        # torch.nn.GELU and torch.nn.functional.gelu are now overridden
        a = torch.rand(3, device='cuda')
        torch.allclose(torch.nn.functional.gelu(a), torch.nn.functional.relu(a))

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 1 output

    .. warning::
        All input tensors must live in CUDA device
    """
    # 返回一个 _JittedFunction 对象，用提供的 CUDA 代码字符串和其他参数初始化
    return _JittedFunction(code_string, return_by_ref=False, num_outputs=1, **kwargs)


# 创建一个用于生成支持返回一个或多个输出的逐元素操作的 jiterator 生成的 CUDA 内核
def _create_multi_output_jit_fn(
    code_string: str, num_outputs: int, **kwargs
) -> Callable:
    """
    Create a jiterator-generated cuda kernel for an elementwise op that supports returning one or more outputs.
    """
    # 返回一个 _JittedFunction 对象，用提供的 CUDA 代码字符串和其他参数初始化，指定返回多个输出
    return _JittedFunction(code_string, return_by_ref=False, num_outputs=num_outputs, **kwargs)
    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return value by reference.
            CUDA代码字符串，将由jiterator编译。入口函数必须通过引用返回值。
        num_outputs(int): number of outputs return by the kernel
            内核返回的输出数量
        kwargs (Dict, optional): Keyword arguments for generated function
            生成函数的可选关键字参数

    Example::

        code_string = "template <typename T> void my_kernel(T x, T y, T alpha, T& out) { out = -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device='cuda')
        b = torch.rand(3, device='cuda')
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)
            # 像普通Python函数一样调用已编译的函数

    .. warning::
        This API is in beta and may change in future releases.
            此API处于测试阶段，可能会在未来版本中更改。

    .. warning::
        This API only supports up to 8 inputs and 8 outputs
            此API仅支持最多8个输入和8个输出
    """
    return _JittedFunction(
        code_string, return_by_ref=True, num_outputs=num_outputs, **kwargs
    )
```