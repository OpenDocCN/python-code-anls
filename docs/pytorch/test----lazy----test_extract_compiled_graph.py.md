# `.\pytorch\test\lazy\test_extract_compiled_graph.py`

```py
# Owner(s): ["oncall: jit"]

# 导入单元测试模块
import unittest

# 初始化 Torch 时间序列后端
from torch._lazy.ts_backend import init as init_ts_backend

# 调用初始化 Torch 时间序列后端函数
init_ts_backend()

# 导入 copy 模块，用于对象拷贝
import copy

# 导入 dis 模块，用于解析 Python 字节码
import dis

# 导入 inspect 模块，用于检查对象
import inspect

# 导入 re 模块，用于正则表达式操作
import re

# 导入 contextmanager 模块中的 contextmanager 装饰器
from contextlib import contextmanager

# 导入 Torch 模块
import torch

# 从 Torch 模块导入 fx 和 nn 模块
from torch import fx, nn

# 导入 Torch _lazy 模块中的 config 模块
from torch._lazy import config

# 从 Torch _lazy 模块导入 extract_compiled_graph 函数
from torch._lazy.extract_compiled_graph import extract_compiled_graph


# 定义一个继承自 nn.Module 的 ModuleConstScale 类
class ModuleConstScale(nn.Module):
    # 重写 forward 方法，实现乘法运算
    def forward(self, a):
        return a * 2


# 定义一个继承自 nn.Module 的 ModuleSub 类
class ModuleSub(nn.Module):
    # 重写 forward 方法，实现减法运算
    def forward(self, a, b):
        return a - b


# 定义一个继承自 nn.Module 的 ModuleAddcmul 类
class ModuleAddcmul(nn.Module):
    """
    addcmul function takes a at::Scalar which results in a special TSData containing a Scalar rather than a Tensor.
    """
    # 重写 forward 方法，调用 torch.addcmul 函数进行元素级乘加运算
    def forward(self, a, b, c):
        return torch.addcmul(a, b, c, value=5)


# 定义一个继承自 nn.Module 的 ModuleReturnMulti 类
class ModuleReturnMulti(nn.Module):
    # 重写 forward 方法，返回元组中的两个计算结果
    def forward(self, a, b):
        return (b + 1, a - 1)


# 定义一个继承自 nn.Module 的 ModuleReturnDupTensor 类
class ModuleReturnDupTensor(nn.Module):
    """
    Handle the corner case that the same tensor appears multiple times in the
    returned tuple. torchbench like drq will hit this corner case when running
    thru torchdynamo..
    """
    # 重写 forward 方法，处理返回值中同一张量多次出现的情况
    def forward(self, a, b):
        c = a + b
        return a - b, c, a + 1, c


# 定义一个继承自 nn.Module 的 ModuleInplaceUpdate 类
class ModuleInplaceUpdate(nn.Module):
    # 重写 forward 方法，执行就地更新操作并返回两个计算结果
    def forward(self, a, b):
        a.sub_(b)
        return b - 1, b + 1


# 定义一个上下文管理器，用于强制切换到指定后端的上下文环境
@contextmanager
def force_fallback_ctx_mgr(fallback_op):
    oldconfig = config.get_force_fallback()
    config.set_force_fallback(fallback_op)
    try:
        yield None
    finally:
        config.set_force_fallback(oldconfig)


# 定义一个空操作的上下文管理器
@contextmanager
def nop_ctx_mgr():
    try:
        yield None
    finally:
        pass


# 定义一个函数，生成随机参数以供模型测试使用
def gen_rand_args(mod):
    args = []
    # 循环生成模型 forward 方法所需的随机参数
    for _ in range(len(inspect.signature(mod.forward).parameters)):
        args.append(torch.randn(2, 3))
    return args


# 定义一个函数，用于比较两个值是否近似相等
def allclose(expected, actual):
    # 定义一个函数，用于展开包含单个元素的列表或元组
    def unwrap(cont):
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont
    
    # 对期望值和实际值进行展开处理
    expected = unwrap(expected)
    actual = unwrap(actual)
    # 检查 expected 和 actual 是否都是 torch.Tensor 类型，并且其内容是否全部近似相等
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    # 如果 expected 和 actual 都是元组或列表类型，则检查它们的长度是否相等，并逐元素比较是否全部近似相等
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(
            torch.allclose(a, b) for a, b in zip(expected, actual)
        )
    else:
        # 如果类型不符合预期，抛出运行时异常
        raise RuntimeError("Unexpected types")
# 定义一个函数，用于验证重复使用编译图的效果
def verify_reusing_compiled_graph(mod, exception_msg_pattern, ncase=10):
    # 生成随机参数
    args = gen_rand_args(mod)
    # 调用模型的前向传播方法
    out = mod(*args)

    # 打印模型前向传播方法的字节码指令序列
    dis.dis(mod.forward)

    try:
        # 提取模型的编译图并优化
        optimized_mod = extract_compiled_graph(fx.symbolic_trace(mod), args)
    except RuntimeError as e:
        if exception_msg_pattern is None:
            raise e  # 重新引发异常
        exception_message = str(e)
        # 如果异常消息不符合指定的模式，则抛出异常
        if not re.search(exception_msg_pattern, exception_message):
            raise RuntimeError(
                f"Exception message does not match the required pattern: {exception_message}"
            ) from e
        else:
            # 对于期望异常的测试用例，完成测试
            return

    if exception_msg_pattern is not None:
        # 如果期望捕获异常但未捕获到，则抛出异常
        raise RuntimeError(
            f"Expect an exception matching pattern {exception_msg_pattern}"
        )
    
    # 打印优化后模型的返回值
    print("return value of optimized_mod", optimized_mod(*args))

    # 检查正确性
    failed_index = []
    for i in range(ncase):
        # 生成随机参数
        rand_args = gen_rand_args(mod)
        rand_args_copy = copy.deepcopy(rand_args)
        # 计算期望结果
        expected = mod(*rand_args)
        # 计算优化后模型的实际结果
        actual = optimized_mod(*rand_args_copy)

        # 如果期望结果与实际结果不一致，则打印错误信息
        if not allclose(expected, actual):
            print(f"Incorrect results. expected {expected}, actual {actual}")
            failed_index.append(i)
            continue

        # 确保调用模型前向传播方法后参数仍然一致，以处理原地更新
        if not allclose(rand_args, rand_args_copy):
            print(
                f"Incorrect updated arguments. expected {rand_args}, actual {rand_args_copy}"
            )
            failed_index.append(i)
            continue

    # 如果有测试用例失败，则抛出异常
    if len(failed_index) > 0:
        raise RuntimeError(f"Failed {len(failed_index)}/{ncase} cases")


# 定义一个测试生成器函数，用于创建测试函数，并调用 verify_reusing_compiled_graph 进行测试
def maketest(module_cls, exception_msg_pattern=None, ctxmgr=None):
    def wrapper(self):
        nonlocal ctxmgr
        # 如果未提供上下文管理器，则使用默认的空操作上下文管理器
        if not ctxmgr:
            ctxmgr = nop_ctx_mgr()
        # 使用上下文管理器执行测试
        with ctxmgr:
            verify_reusing_compiled_graph(module_cls(), exception_msg_pattern)

    return wrapper


# 定义一个测试类 OptimizeTest，继承自 unittest.TestCase
class OptimizeTest(unittest.TestCase):
    # 创建测试函数 test_sub，测试 ModuleSub 模块的优化效果
    test_sub = maketest(ModuleSub)
    
    # 创建测试函数 test_ltc_fallback，测试 ModuleSub 模块在强制使用 LTC 回退时是否捕获预期异常
    test_ltc_fallback = maketest(
        ModuleSub,
        exception_msg_pattern="fallback.*aten::sub",
        ctxmgr=force_fallback_ctx_mgr("aten::sub"),
    )
    
    # 创建其他测试函数，分别测试不同模块的优化效果
    test_const_scale = maketest(ModuleConstScale)
    test_addcmul = maketest(ModuleAddcmul)
    test_return_multi = maketest(ModuleReturnMulti)
    test_return_dup_tensor = maketest(ModuleReturnDupTensor)
    test_inplace_update = maketest(ModuleInplaceUpdate)
```