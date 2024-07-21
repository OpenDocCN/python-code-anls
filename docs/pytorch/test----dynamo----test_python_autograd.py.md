# `.\pytorch\test\dynamo\test_python_autograd.py`

```
# Owner(s): ["module: dynamo"]
from typing import Callable, Dict, List, NamedTuple, Optional

import torch

import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter, same

"""
This is an example of a pure-python version of autograd implemented by
@zdevito.  It represents a rather challenging test case for TorchDynamo
to push the limits of what it can do.
"""


_name: int = 0


def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


class Variable:
    def __init__(self, value: torch.Tensor, name: str = None):
        self.value = value
        self.name = name or fresh_name()

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value: torch.Tensor, name: str = None):
        return Variable(value, name)

    def __repr__(self):
        return repr(self.value)

    # This performs a pointwise multiplication of a Variable, tracking gradients
    def __mul__(self, rhs: "Variable") -> "Variable":
        # defined later in the notebook
        return operator_mul(self, rhs)

    def __add__(self, rhs: "Variable") -> "Variable":
        return operator_add(self, rhs)

    def sum(self, name: Optional[str] = None) -> "Variable":
        return operator_sum(self, name)

    def expand(self, sizes: List[int]) -> "Variable":
        return operator_expand(self, sizes)


class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs: List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: "Callable[List[Variable], List[Variable]]"


gradient_tape: List[TapeEntry] = []


def reset_tape():
    gradient_tape.clear()
    global _name
    _name = 0


def grad(L, desired_results: List[Variable]) -> List[Variable]:
    # this map holds dL/dX for all values X
    dL_d: Dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable(torch.ones(()))
    # print(f'd{L.name} ------------------------')

    # look up dL_dentries. If a variable is never used to compute the loss,
    # we consider its gradient None, see the note below about zeros for more information.
    def gather_grad(entries: List[str]):
        return [dL_d[entry] if entry in dL_d else None for entry in entries]

    # propagate the gradient information backward
    # 对梯度记录进行反向遍历，处理每一个记录条目
    for entry in reversed(gradient_tape):
        # 提取当前记录条目的输出梯度
        dL_doutputs = gather_grad(entry.outputs)
        
        # 如果所有输出梯度均为 None，则跳过当前记录条目的处理
        if all(dL_doutput is None for dL_doutput in dL_doutputs):
            # 优化处理梯度路径中某些路径为零的情况。详见下面的注释说明。
            continue

        # 根据链式法则，传播特定于每个计算的梯度
        dL_dinputs = entry.propagate(dL_doutputs)

        # 累积每个输入产生的梯度
        # 每个变量的使用产生对应的梯度 dL_dinput。多元链式法则告诉我们可以安全地将所有贡献值相加。
        for input, dL_dinput in zip(entry.inputs, dL_dinputs):
            if input not in dL_d:
                dL_d[input] = dL_dinput
            else:
                dL_d[input].value += dL_dinput.value

    # 打印一些信息，以了解每个中间变量的值
    # for name, value in dL_d.items():
    #    print(f'd{L.name}_d{name} = {value.name}')
    # print(f'------------------------')

    # 返回所有期望结果的梯度汇总
    return gather_grad(desired.name for desired in desired_results)
# 定义乘法操作符重载方法，用于变量自乘
def operator_mul(self: Variable, rhs: Variable) -> Variable:
    # 如果右操作数是浮点数且等于1.0，进行短路优化
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # 定义前向传播计算
    r = Variable(self.value * rhs.value)
    # 记录操作符的输入和输出
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # 定义反向传播函数
    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs

        # 计算对自身和右操作数的偏导数
        dr_dself = rhs  # r = self * rhs 的偏导数
        dr_drhs = self  # r = self * rhs 的偏导数

        # 使用链式法则从输出到输入传播梯度
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs]
        return dL_dinputs

    # 将计算记录到梯度记录带中
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r


# 定义加法操作符重载方法
def operator_add(self: Variable, rhs: Variable) -> Variable:
    # 加法操作与乘法类似，但不涉及变量捕获
    r = Variable(self.value + rhs.value)

    # 定义加法操作的反向传播函数
    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]

    # 将计算记录到梯度记录带中
    gradient_tape.append(
        TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate)
    )
    return r


# 定义求和操作符重载方法
def operator_sum(self: Variable, name: Optional[str]) -> "Variable":
    # 对变量值进行求和，可选指定结果名称
    r = Variable(torch.sum(self.value), name=name)

    # 定义求和操作的反向传播函数
    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        size = self.value.size()
        return [dL_dr.expand(*size)]

    # 将计算记录到梯度记录带中
    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


# 定义扩展操作符重载方法
def operator_expand(self: Variable, sizes: List[int]) -> "Variable":
    # 断言变量值维度为0（仅适用于标量）
    assert self.value.dim() == 0
    # 对变量值进行扩展
    r = Variable(self.value.expand(sizes))

    # 定义扩展操作的反向传播函数
    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        return [dL_dr.sum()]

    # 将计算记录到梯度记录带中
    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


# 定义一个简单的函数，进行加法和乘法操作
def simple(a, b):
    t = a + b  # 执行加法
    return t * b  # 执行乘法


# 测试用例类的声明
class TestPythonAutograd(TestCase):
    # 定义一个名为 `_common` 的方法，用于执行通用的测试操作，接受一个函数 `fn` 和预期操作数 `expected_ops` 作为参数
    def _common(self, fn, expected_ops):
        # 创建两个包含随机数据的张量列表
        args1 = [torch.randn(10), torch.randn(10)]
        args2 = [torch.randn(10), torch.randn(10)]
        # 创建一个编译计数器对象
        cnt = CompileCounter()
        # 使用 `_dynamo.optimize_assert` 优化传入的函数 `fn`，返回优化后的函数 `fn_dynamo`
        fn_dynamo = torch._dynamo.optimize_assert(cnt)(fn)
        # 重置计算图
        reset_tape()
        # 调用优化后的函数 `fn_dynamo`，传入 `args1`，记录结果到 `res1`
        res1 = fn_dynamo(*args1)
        # 重置计算图
        reset_tape()
        # 调用优化后的函数 `fn_dynamo`，传入 `args2`，记录结果到 `res2`
        res2 = fn_dynamo(*args2)
        # 重置计算图
        reset_tape()
        # 断言 `res1` 与原始函数 `fn` 在 `args1` 上的结果一致
        self.assertTrue(same(res1, fn(*args1)))
        # 重置计算图
        reset_tape()
        # 断言 `res2` 与原始函数 `fn` 在 `args2` 上的结果一致
        self.assertTrue(same(res2, fn(*args2)))
        # 重置计算图
        reset_tape()
        # 断言编译计数器的帧计数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言编译计数器的操作数等于预期操作数 `expected_ops`
        self.assertEqual(cnt.op_count, expected_ops)

    # 定义一个测试方法 `test_forwards1`
    def test_forwards1(self):
        # 定义一个函数 `fn`，接受两个参数 `a` 和 `b`，将它们设为常量并计算损失值，返回损失值
        def fn(a, b):
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            return loss

        # 调用 `_common` 方法，传入函数 `fn` 和预期的操作数 3
        self._common(fn, 3)

    # 定义一个测试方法 `test_forwards2`
    def test_forwards2(self):
        # 定义一个函数 `fn`，与 `test_forwards1` 类似，但在计算损失值前后调用 `reset_tape` 重置计算图
        def fn(a, b):
            reset_tape()
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            reset_tape()
            return loss

        # 调用 `_common` 方法，传入函数 `fn` 和预期的操作数 3
        self._common(fn, 3)

    # 定义一个测试方法 `test_backwards1`
    def test_backwards1(self):
        # 定义一个函数 `fn`，计算变量 `a` 和 `b` 的损失函数，并返回它们的梯度
        def fn(a, b):
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            return grad(loss, [a, b])

        # 调用 `_common` 方法，传入函数 `fn` 和预期的操作数 8
        self._common(fn, 8)

    # 定义一个测试方法 `test_backwards2`
    def test_backwards2(self):
        # 定义一个函数 `fn`，与 `test_backwards1` 类似，但在计算梯度前后调用 `reset_tape` 重置计算图
        def fn(a, b):
            reset_tape()
            a = Variable.constant(a, name="a")
            b = Variable.constant(b, name="b")
            loss = simple(a, b).sum()
            res = grad(loss, [a, b])
            reset_tape()
            return res

        # 调用 `_common` 方法，传入函数 `fn` 和预期的操作数 8
        self._common(fn, 8)

    # 定义一个测试方法 `test_split`
    def test_split(self):
        # 创建两个常量变量 `v1` 和 `v2`
        v1 = Variable.constant(torch.randn(10), name="a")
        v2 = Variable.constant(torch.randn(10), name="b")
        # 创建一个编译计数器对象
        cnt = CompileCounter()

        # 定义一个前向传播函数 `forward`，计算 `v1` 和 `v2` 的简单操作的损失函数并返回
        def forward(a, b):
            return simple(a, b).sum()

        # 重置计算图
        reset_tape()
        # 计算未优化的前向传播的损失 `loss1`，并计算其梯度 `grad1`
        loss1 = forward(v1, v2)
        grad1 = grad(loss1, [v1, v2])

        # 重置计算图
        reset_tape()
        # 使用 `_dynamo.optimize_assert` 优化前向传播函数 `forward`，并返回优化后的版本 `opt_forward`
        opt_forward = torch._dynamo.optimize_assert(cnt)(forward)
        # 使用 `_dynamo.optimize_assert` 优化梯度函数 `grad`，并返回优化后的版本 `opt_grad`
        opt_grad = torch._dynamo.optimize_assert(cnt)(grad)
        # 计算优化后的前向传播损失 `loss2`
        loss2 = opt_forward(v1, v2)
        # 强制两个帧
        grad2 = opt_grad(loss2, [v1, v2])

        # 断言两种计算方式得到的损失值 `loss1` 和 `loss2` 相同
        self.assertTrue(same(loss1, loss2))
        # 断言两种计算方式得到的梯度 `grad1` 和 `grad2` 相同
        self.assertTrue(same(grad1, grad2))
        # 断言编译计数器的帧计数为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言编译计数器的操作数为 8
        self.assertEqual(cnt.op_count, 8)
# 如果这个脚本被直接运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```