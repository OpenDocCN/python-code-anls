# `.\pytorch\torch\_dynamo\testing.py`

```py
# mypy: allow-untyped-defs
# 导入所需模块
import contextlib  # 上下文管理模块
import dis  # 字节码分析模块
import functools  # 函数工具模块
import logging  # 日志记录模块
import os.path  # 路径操作模块
import random  # 随机数生成模块
import re  # 正则表达式模块
import sys  # 系统相关模块
import types  # 类型操作模块
import unittest  # 单元测试模块
from typing import List, Optional, Sequence, Union  # 类型注解支持
from unittest.mock import patch  # 单元测试模块中的模拟模块

np: Optional[types.ModuleType] = None
try:
    import numpy as np  # 尝试导入 NumPy 模块
except ModuleNotFoundError:
    np = None  # 若导入失败，设置为 None

import torch  # 导入 PyTorch 模块
from torch import fx  # PyTorch 中的 FX 模块
from torch._dynamo.output_graph import OutputGraph  # 输出图形模块

from . import config, eval_frame, optimize_assert, reset  # 导入当前包中的模块
from .bytecode_transformation import (
    create_instruction,  # 导入指令创建函数
    debug_checks,  # 导入调试检查函数
    is_generator,  # 导入生成器判断函数
    transform_code_object,  # 导入代码对象转换函数
)
from .guards import CheckFunctionManager, GuardedCode  # 导入函数检查管理器和受保护代码
from .utils import same  # 导入同等函数

unsupported = eval_frame.unsupported  # 设置 unsupported 变量为 eval_frame 模块中的 unsupported
three = 3  # 设置 three 变量为整数 3

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def clone_me(x):
    # 如果输入为 None，则返回 None
    if x is None:
        return None
    # 对输入进行分离、克隆并设置梯度跟踪
    return x.detach().clone().requires_grad_(x.requires_grad)


def remove_optimized_module_prefix(name) -> str:
    # 移除优化模块的前缀
    return re.sub(r"^_orig_mod[.]", "", name)


def collect_results(model, prediction, loss, example_inputs):
    results = []  # 创建空列表 results
    results.append(prediction)  # 将预测结果添加到 results 中
    results.append(loss)  # 将损失值添加到 results 中
    # 若损失值为 torch.Tensor 且其值大于 1，则发出警告
    # log.warning(
    #     f"High loss value alert - {loss:.2f}. Can result in unstable gradients."
    # )

    grads = dict()  # 创建空字典 grads
    params = dict()  # 创建空字典 params
    for name, param in model.named_parameters():
        if isinstance(model, eval_frame.OptimizedModule):
            # 如果模型是 OptimizedModule，则移除优化模块前缀
            name = remove_optimized_module_prefix(name)
        param_copy = param  # 复制参数
        grad = param.grad  # 获取参数的梯度
        # 如果参数的梯度为 None，则设置为与参数相同形状的零张量
        if param.grad is None:
            grad = torch.zeros_like(param)
        grads[name + ".grad"] = grad  # 将参数梯度添加到 grads 字典中
        params[name] = param_copy  # 将参数添加到 params 字典中
    results.append(grads)  # 将 grads 添加到 results 中
    results.append(params)  # 将 params 添加到 results 中

    buffers = dict()  # 创建空字典 buffers
    for name, buffer in model.named_buffers():
        if isinstance(model, eval_frame.OptimizedModule):
            # 如果模型是 OptimizedModule，则移除优化模块前缀
            name = remove_optimized_module_prefix(name)
        buffers[name] = buffer  # 将缓冲区添加到 buffers 字典中
    results.append(buffers)  # 将 buffers 添加到 results 中

    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            for inp in example:
                if isinstance(inp, torch.Tensor):
                    results.append(inp.grad)  # 将张量的梯度添加到 results 中
        else:
            if isinstance(example, torch.Tensor):
                results.append(example.grad)  # 将张量的梯度添加到 results 中

    return results  # 返回结果列表


def requires_bwd_pass(out):
    if isinstance(out, torch.Tensor):
        return out.requires_grad  # 返回张量是否需要梯度的布尔值
    elif isinstance(out, (list, tuple)):
        return any(requires_bwd_pass(x) for x in out)  # 返回列表或元组中是否有任何元素需要梯度
    elif out is None:
        return False  # 如果输出为 None，则返回 False
    elif isinstance(out, int):
        return False  # 如果输出为整数，则返回 False
    raise NotImplementedError("Don't know how to reduce", type(out))  # 抛出未实现错误


def reduce_to_scalar_loss(out):
    """Reduce the output of a model to get scalar loss"""
    if isinstance(out, torch.Tensor):
        # 对模型的输出进行缩减，得到标量损失
        # 平均值不适用于整数张量
        return out.sum() / out.numel()
    # 如果 `out` 是列表或元组类型，对其中每个元素调用 `reduce_to_scalar_loss` 函数，然后计算平均值作为返回值
    elif isinstance(out, (list, tuple)):
        return sum(reduce_to_scalar_loss(x) for x in out) / len(out)
    # 如果 `out` 的类型名称属于以下任意一种，调用 `reduce_to_scalar_loss` 函数处理 `out.logits` 并返回结果
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
        return reduce_to_scalar_loss(out.logits)
    # 如果 `out` 的类型名称为 "SquashedNormal"，计算其均值并返回
    elif type(out).__name__ == "SquashedNormal":
        return out.mean.sum()
    # 如果 `out` 是字典类型，对字典中每个值调用 `reduce_to_scalar_loss` 函数，然后计算平均值作为返回值
    elif isinstance(out, dict):
        return sum(reduce_to_scalar_loss(value) for value in out.values()) / len(
            out.keys()
        )
    # 如果以上情况都不满足，则抛出未实现的错误，显示无法处理 `out` 的类型信息
    raise NotImplementedError("Don't know how to reduce", type(out))
# 定义一个函数，用于返回调试目录的路径
def debug_dir() -> str:
    # 拼接调试目录的路径，相对于当前文件所在目录的上级目录
    path = os.path.join(os.path.dirname(__file__), "../debug")
    # 如果路径不存在，则创建该路径
    if not os.path.exists(path):
        os.mkdir(path)
    # 返回调试目录的路径
    return path


# 定义一个函数，用于将给定的代码对象以指定名称写入调试文件
def debug_dump(name, code: types.CodeType, extra="") -> None:
    # 打开调试文件，将指定的代码对象信息、字节码和额外信息写入文件
    with open(os.path.join(debug_dir(), name), "w") as fd:
        fd.write(
            f"{dis.Bytecode(code).info()}\n\n{dis.Bytecode(code).dis()}\n\n{extra}\n"
        )


# 定义一个函数，用于在代码帧中插入空操作指令来调试跳转更新
def debug_insert_nops(
    frame, cache_size, hooks, _, *, skip: int = 0
) -> Optional[GuardedCode]:
    """used to debug jump updates"""

    # 定义一个函数，用于在指令列表的开头插入空操作指令
    def insert_nops(instructions, code_options):
        instructions.insert(0, create_instruction("NOP"))
        instructions.insert(0, create_instruction("NOP"))

    # 如果是生成器函数，则返回空值
    if is_generator(frame.f_code):
        return None

    # 执行调试检查
    debug_checks(frame.f_code)
    # 对代码对象进行转换，并在其中插入空操作指令
    code = transform_code_object(frame.f_code, insert_nops)
    # 创建输出图对象，用于存储编译后的代码和相关信息
    graph = OutputGraph(
        code_options={},
        compiler_fn=None,
        root_tx=None,
        export=False,
        export_constraints=None,
        frame_state={"_id": 0},
        # TODO: shouldn't this be f_locals/f_globals from frame?
        local_scope=locals(),
        global_scope=globals(),
        f_code=frame.f_code,
    )

    # 返回包装后的代码对象及其相关检查函数管理器的实例
    return GuardedCode(code, CheckFunctionManager(graph).check_fn)


# 定义一个用于统计编译次数和操作次数的类
class CompileCounter:
    def __init__(self):
        self.frame_count = 0
        self.op_count = 0

    # 当实例被调用时，对图模块中的节点进行统计
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        # 返回图模块的前向函数
        return gm.forward

    # 清空计数器
    def clear(self):
        self.frame_count = 0
        self.op_count = 0


# 定义一个带后端参数的编译计数器类
class CompileCounterWithBackend:
    def __init__(self, backend):
        self.frame_count = 0
        self.op_count = 0
        self.backend = backend
        self.graphs = []

    # 当实例被调用时，从注册表中查找后端，并对图模块中的节点进行统计
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        from .backends.registry import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        # 使用查找后端函数处理图模块，并返回结果
        return lookup_backend(self.backend)(gm, example_inputs)


# 定义一个类，功能类似于 backend="eager"，但还记录了图模块，供断言使用
class EagerAndRecordGraphs:
    def __init__(self):
        self.graphs = []

    # 当实例被调用时，将图模块添加到记录的图列表中，并返回其前向函数
    def __call__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        self.graphs.append(gm)
        return gm.forward


# 定义一个函数，用于去除代码中的注释
def strip_comment(code) -> str:
    code = str(code)
    # 使用正则表达式去除多行注释
    return re.sub(r"(?m)^ *#.*\n?", "", code)


# 定义一个函数，用于去除代码每行末尾的空格
def remove_trailing_space(code) -> str:
    # 对代码按行分割，去除每行末尾的空格后重新连接为字符串
    return "\n".join([line.rstrip() for line in code.split("\n")])


# 定义一个函数，用于规范化图模块字符串表示形式
def normalize_gm(gm_str) -> str:
    # 去除字符串中的注释和末尾空格，以便于系统间的比较
    # 注释可能包含文件路径，会因系统不同而异
    return remove_trailing_space(strip_comment(gm_str))


# 定义一个标准测试函数
def standard_test(
    self,
    fn,
    nargs,
    expected_ops=None,
    expected_ops_dynamic=None,
    expected_frame_count=1,
# 定义一个函数，用于生成带有补丁的函数
def _make_fn_with_patches(fn, *patches):
    # 使用functools库的wraps装饰器，保留原始函数的元数据
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        # 使用contextlib.ExitStack创建一个上下文管理器，用于管理多个上下文
        with contextlib.ExitStack() as stack:
            # 遍历补丁列表，每个补丁包括模块、属性、值
            for module, attr, val in patches:
                # 将patch.object上下文加入栈中
                stack.enter_context(patch.object(module, attr, val))

            # 调用原始函数，并返回其结果
            return fn(*args, **kwargs)

    # 返回生成的带补丁的函数
    return _fn
    # 遍历给定类(cls)的所有属性名
    for name in dir(cls):
        # 检查属性名是否以 "test_" 开头
        if name.startswith("test_"):
            # 获取属性对应的对象或方法
            fn = getattr(cls, name)
            # 如果属性不可调用（即不是方法），直接将其设置到 DummyTestClass 中
            if not callable(fn):
                setattr(DummyTestClass, name, getattr(cls, name))
                continue
            # 为原始方法名添加后缀 fn_suffix
            new_name = f"{name}{fn_suffix}"
            # 使用 _make_fn_with_patches 函数对方法 fn 进行修饰，并生成新的方法 new_fn
            new_fn = _make_fn_with_patches(fn, *patches)
            # 设置新方法的名称为带有后缀的新名称
            new_fn.__name__ = new_name
            # 如果设置了 xfail_prop 并且原始方法有该属性，将新方法标记为预期失败的测试
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            # 使用 decorator 函数修饰新方法，并将其设置到 DummyTestClass 中
            setattr(DummyTestClass, new_name, decorator(new_fn))
        # NB: Doesn't handle slots correctly, but whatever
        # 如果 DummyTestClass 中没有名为 name 的属性，则将 cls 中的该属性复制到 DummyTestClass 中
        elif not hasattr(DummyTestClass, name):
            setattr(DummyTestClass, name, getattr(cls, name))

    # 返回已经装配好的 DummyTestClass 类
    return DummyTestClass
# 测试 Python 3.11+ 特定功能

# 如果当前 Python 版本大于等于 3.11，则不跳过该测试函数
def skipIfNotPy311(fn):
    if sys.version_info >= (3, 11):
        return fn
    # 如果 Python 版本小于 3.11，则使用 unittest 跳过该测试函数
    return unittest.skip(fn)


# 如果当前 Python 版本大于等于 3.12，则不跳过该测试函数
def skipIfNotPy312(fn):
    if sys.version_info >= (3, 12):
        return fn
    # 如果 Python 版本小于 3.12，则使用 unittest 跳过该测试函数
    return unittest.skip(fn)


# 如果当前 Python 版本大于等于 3.12，则使用 unittest.expectedFailure 标记该测试函数为预期失败
def xfailIfPy312(fn):
    if sys.version_info >= (3, 12):
        return unittest.expectedFailure(fn)
    return fn


# 如果当前 Python 版本大于等于 3.12，则使用 unittest 跳过该测试函数
def skipIfPy312(fn):
    if sys.version_info >= (3, 12):
        return unittest.skip(fn)
    return fn


# 控制在 test/inductor/test_torchinductor_dynamic_shapes.py 和 test/dynamo/test_dynamic_shapes.py 生成的测试
def expectedFailureDynamic(fn):
    # 将函数标记为在动态情况下预期失败
    fn._expected_failure_dynamic = True
    return fn


# 控制在 test/inductor/test_torchinductor_codegen_dynamic_shapes.py 生成的测试
def expectedFailureCodegenDynamic(fn):
    # 将函数标记为在动态代码生成情况下预期失败
    fn._expected_failure_codegen_dynamic = True
    return fn


# 控制在 test/inductor/test_cpp_wrapper.py 生成的测试
def expectedFailureDynamicWrapper(fn):
    # 将函数标记为在动态包装器情况下预期失败
    fn._expected_failure_dynamic_wrapper = True
    return fn


# 重置随机数生成器状态，可选择是否使用 XLA
def reset_rng_state(use_xla=False):
    # 设置 torch 的随机种子
    torch.manual_seed(1337)
    # 设置 Python 内置的随机数生成器的种子
    random.seed(1337)
    # 如果 NumPy 可用，则设置 NumPy 的随机种子
    if np:
        np.random.seed(1337)
    # 如果使用了 XLA，则设置 XLA 设备的随机数生成器状态
    if use_xla:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(1337, str(xm.xla_device()))
```