# `.\pytorch\torch\jit\_decompositions.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库及相关模块
import torch
from torch import Tensor

# 导入 inspect 模块和警告模块
import inspect
import warnings
# 导入类型提示相关的类和模块
from typing import Dict, List, Optional, Set

# 从 torch.types 模块导入 Number 类型
from torch.types import Number

# 创建空字典 decomposition_table，用于存储函数名到 torch.jit.ScriptFunction 的映射
decomposition_table: Dict[str, torch.jit.ScriptFunction] = {}
# 创建空集合 function_name_set，用于存储注册过的函数名
function_name_set: Set[str] = set()

# 定义函数 check_decomposition_has_type_annotations，用于检查函数参数和返回值是否有类型注解
def check_decomposition_has_type_annotations(f):
    # 获取 inspect 模块中的空值定义
    inspect_empty = inspect._empty  # type: ignore[attr-defined]
    # 获取函数的签名信息
    sig = inspect.signature(f)
    # 检查每个参数是否有注解
    for param in sig.parameters.values():
        assert (
            param.annotation != inspect_empty
        ), f"No signature on param {param.name} for function {f.name}"
    # 检查返回值是否有注解
    assert (
        sig.return_annotation != inspect_empty
    ), f"No return annotation for function {f.name}"

# 定义函数 signatures_match，用于比较两个函数的签名是否匹配
def signatures_match(decomposition_sig, torch_op_sig):
    # 获取分解函数和 Torch 操作函数的参数列表
    decomp_params = decomposition_sig.parameters
    op_params = torch_op_sig.parameters

    # 如果参数数量不一致，返回 False
    if len(decomp_params) != len(op_params):
        return False

    # 逐个比较参数的名称和注解
    for decomp_param, op_param in zip(decomp_params.values(), op_params.values()):
        # 检查参数的名称和注解是否相同
        inspect_empty = inspect._empty  # type: ignore[attr-defined]
        for field in ["name", "annotation"]:
            if field == "name" and decomp_param.name == "self":
                warnings.warn("PyTorch uses 'input' instead of 'self' on public api")

            if getattr(decomp_param, field) != getattr(op_param, field):
                return False

        # 检查参数的默认值是否相同
        decomp_default = decomp_param.default
        op_default = op_param.default
        if decomp_default != inspect_empty and op_default != inspect_empty:
            if decomp_default != op_default:
                return False

    # 检查返回值的注解是否相同
    return decomposition_sig.return_annotation == torch_op_sig.return_annotation

# 定义函数 register_decomposition，用于注册函数的分解器
def register_decomposition(aten_op, registry=None):
    # 定义内部装饰器函数 decomposition_decorator
    def decomposition_decorator(f):
        nonlocal registry
        # 如果 registry 为空，则使用全局的 decomposition_table
        if registry is None:
            registry = decomposition_table

        # 确保 aten_op 是 torch._ops.OpOverload 类型
        assert isinstance(aten_op, torch._ops.OpOverload)

        # 确保函数名未注册过
        assert (
            f.__name__ not in function_name_set
        ), f"Duplicated function name {f.__name__}"
        # 将函数名添加到已注册的集合中
        function_name_set.add(f.__name__)

        # 对函数进行脚本化
        scripted_func = torch.jit.script(f)
        # 执行 JIT Passes 以优化脚本化函数的图形
        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        # 将分解函数注册到 registry 中，键为 aten_op._schema 的字符串表示
        registry[str(aten_op._schema)] = scripted_func
        # 返回原始函数
        return f

    # 返回内部装饰器函数 decomposition_decorator
    return decomposition_decorator

# TODO: replace torch.sigmoid -> aten.sigmoid
# 待完成：替换 torch.sigmoid 为 aten.sigmoid
# 将 `aten.var.correction` 注册为 `var_decomposition` 的装饰器函数
@register_decomposition(aten.var.correction)
# 方差分解函数，计算输入张量在指定维度上的方差
def var_decomposition(
    input: Tensor,
    dim: Optional[List[int]] = None,
    correction: Optional[Number] = None,
    keepdim: bool = False,
) -> Tensor:
    # 如果未指定维度，则将 dim_i 初始化为空列表
    if dim is None:
        dim_i: List[int] = []
        dim = dim_i

    # 如果 dim 是元组或列表且长度为 0，则计算输入张量元素总数 n
    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        n = input.numel()
    else:
        n = 1
        # 计算指定维度上的元素个数乘积 n
        for dim_i in dim:  # type: ignore[assignment]
            n *= input.shape[dim_i]  # type: ignore[call-overload]

    # 计算输入张量在指定维度上的均值
    mean = aten.mean(input, dim, True)
    # 计算输入张量与均值的差
    sub = input - mean
    # 计算差的平方
    sq = sub * sub
    # 沿指定维度对平方值求和
    sum = aten.sum(sq, dim, keepdim)

    # 计算修正后的分母 denom
    if correction is None:
        denom = float(n - 1)
    else:
        if isinstance(correction, int):
            denom = float(n - correction)
        elif isinstance(correction, float):
            denom = float(n) - correction
        else:
            raise RuntimeError("correction must be int or float")

    # 返回方差值，防止分母为零
    return sum / max(0, denom)


# 将 `aten.var.default` 注册为 `var` 的装饰器函数
@register_decomposition(aten.var.default)
# 方差计算函数，调用方差分解函数 `var_decomposition` 来计算方差
def var(input: Tensor, unbiased: bool = True) -> Tensor:
    # 调用方差分解函数，并根据 unbiased 参数选择是否修正方差计算
    return var_decomposition(input, correction=(1 if unbiased else 0))
```