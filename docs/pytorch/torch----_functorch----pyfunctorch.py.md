# `.\pytorch\torch\_functorch\pyfunctorch.py`

```
"""
This file contains the integration of functorch with PyDispatcher.

PyDispatcher cannot directly utilize functorch's C++ implemented DynamicLayerStack
dispatching logic for FuncTorchDynamicLayer{Front, Back}Mode due to the absence
of C++ boxed fallbacks for those dispatch keys. Therefore, this file implements
the stack peeking logic and interpreter selection in Python instead of trying
to adapt PyDispatcher to handle C++ implementations.

The main difference from C++ functorch is that PyDispatcher's logic retrieves
an interpreter from the stack and executes its associated rule, whereas C++
functorch manually adjusts dispatch keys to manage DynamicLayerFrontMode and
DynamicLayerBackMode.

This Python implementation avoids the need for ping-ponging dispatch keys like
in C++ functorch, by directly registering batching rules to transforms for
interpreters to invoke.

"""

# mypy: allow-untyped-defs
import contextlib
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    CFunctionalizeInterpreterPtr,
    CGradInterpreterPtr,
    CInterpreter,
    CJvpInterpreterPtr,
    CVmapInterpreterPtr,
    pop_dynamic_layer_stack,
    push_dynamic_layer_stack,
    RandomnessType,
    TransformType,
)
from torch.autograd.forward_ad import _set_fwd_grad_enabled

# FuncTorchInterpreter is a Python abstraction of the Interpreter class from
# C++. It wraps around the actual C++ Interpreter object stored in self._cptr.
# Methods in this class mirror those in aten/src/ATen/functorch/Interpreter.h
class FuncTorchInterpreter(ABC):
    def __init__(self, cptr: Any):
        self._cptr = cptr

    # Process an operation, e.g., invoking a batching rule for vmap.
    # Analogous to Interpreter::process in C++.
    @abstractmethod
    def process(self, op, args, kwargs):
        pass

    # Lower an operation to the next Interpreter on the stack by temporarily
    # popping the current Interpreter.
    # Analogous to Interpreter::sendToNextInterpreter in C++.
    def lower(self):
        return temporarily_pop_interpreter_stack()

    # Get the level of the interpreter in the stack.
    def level(self):
        return self._cptr.level()

    # Get the dispatch key associated with the interpreter.
    def key(self):
        return self._cptr.key()

    # Get the state of the interpreter.
    def get_state(self):
        raise NotImplementedError

    # Check if the provided state matches the current interpreter's state.
    def check_state(self, state):
        return state == self.get_state()


@contextlib.contextmanager
def temporarily_pop_interpreter_stack():
    try:
        # Temporarily pop an interpreter from the dynamic layer stack.
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        # 无论如何都会执行的代码块，用于清理或执行必要的收尾工作
        push_dynamic_layer_stack(saved)
@contextlib.contextmanager
def temporarily_clear_interpreter_stack():
    # 创建一个空列表来存储被清空的解释器栈
    stack = []
    try:
        # 当存在动态图层时，不断将其弹出并添加到stack列表中
        while torch._C._functorch.peek_interpreter_stack() is not None:
            stack.append(pop_dynamic_layer_stack())
        # 使用stack列表作为上下文管理器的返回值
        yield list(stack)
    finally:
        # 在最终步骤中，将stack列表中的内容逐个压回解释器栈中
        while stack:
            push_dynamic_layer_stack(stack.pop())


@contextlib.contextmanager
def temporarily_restore_interpreter_stack(stack):
    # 创建一个空列表来存储已压回的动态图层
    pushed = []
    try:
        # 反向遍历给定的stack列表，并逐个将动态图层压回解释器栈中
        for s in reversed(stack):
            push_dynamic_layer_stack(s)
            pushed.append(s)
        # 使用空yield语句表示此上下文管理器不返回任何值
        yield
    finally:
        # 在最终步骤中，逆向遍历已压回的动态图层列表，并逐个将其弹出
        for s in reversed(pushed):
            # TODO: would be nice to assert that the layers are the same, but
            # Python object identity is not preserved
            pop_dynamic_layer_stack()


class VmapInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter):
        # 断言传入的CInterpreter的类型是TransformType.Vmap
        assert cdata.key() == TransformType.Vmap
        # NOTE: [Interpreter cdata vs cptr]
        # 将通用的CInterpreter包装在CVmapInterpreterPtr中，
        # 以便能够访问特定于vmap解释器的方法
        self._cdata = cdata
        self._cptr = CVmapInterpreterPtr(cdata)

    def process(self, op, args, kwargs):
        # 获取TransformType.Vmap对应的内核函数
        kernel = op.functorch_table[TransformType.Vmap]
        # 调用内核函数，并返回其结果
        return kernel(self, *args, **kwargs)

    def batch_size(self):
        # 返回Vmap解释器实例的批处理大小
        return self._cptr.batchSize()

    def randomness(self):
        # 获取Vmap解释器实例的随机性类型，并返回相应的字符串表示
        typ = self._cptr.randomness()
        if typ == RandomnessType.Error:
            return "error"
        elif typ == RandomnessType.Same:
            return "same"
        elif typ == RandomnessType.Different:
            return "different"
        # 如果随机性类型未知，则抛出运行时错误
        raise RuntimeError(f"Unknown RandomnessType: {typ}")

    def get_state(self):
        # 返回Vmap解释器实例的状态，包括其名称、级别和随机性类型
        return (self.key().name, self.level(), self.randomness())


@contextlib.contextmanager
def nested(*contexts):
    # 使用ExitStack来管理多个上下文管理器，确保它们都能被正确处理
    with contextlib.ExitStack() as stack:
        # 逐个进入给定的contexts中的上下文管理器
        for ctx in contexts:
            stack.enter_context(ctx)
        # 使用yield语句将contexts作为上下文管理器的返回值
        yield contexts


class GradInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter):
        # 断言传入的CInterpreter的类型是TransformType.Grad
        assert cdata.key() == TransformType.Grad
        # See NOTE: [Interpreter cdata vs cptr]
        # 将通用的CInterpreter包装在CGradInterpreterPtr中，
        # 以便能够访问特定于Grad解释器的方法
        self._cdata = cdata
        self._cptr = CGradInterpreterPtr(cdata)

    def lift(self, args, kwargs):
        # 将args和kwargs中的所有torch.Tensor对象映射到Grad解释器特定的方法上
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        # 返回映射后的args和kwargs
        return args, kwargs

    def process(self, op, args, kwargs):
        # 获取TransformType.Grad对应的内核函数
        kernel = op.functorch_table[TransformType.Grad]
        # 调用lift方法处理args和kwargs，并传递给内核函数，返回其结果
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # GradInterpreter因为与no_grad的交互而具有自定义的lower方法
    # See NOTE [grad and vjp interaction with no_grad]
    # 此逻辑从C++ GradInterpreterPtr::sendToNextInterpreter中镜像出来
    # 定义一个方法，用于降低当前对象的梯度模式
    def lower(self):
        # 获取当前对象的前一个梯度模式状态
        prev_grad_mode = self.prev_grad_mode()
        # 如果前一个梯度模式不存在，则设定当前梯度模式为不计算梯度，并调用父类的降低方法
        if not prev_grad_mode:
            return nested(torch.no_grad(), super().lower())
        # 否则，调用父类的降低方法
        return super().lower()

    # 定义一个方法，返回当前对象的前一个梯度模式状态
    def prev_grad_mode(self):
        return self._cptr.prevGradMode()

    # 定义一个方法，返回当前对象的状态信息，包括键名、级别和前一个梯度模式状态
    def get_state(self):
        return (self.key().name, self.level(), self.prev_grad_mode())
class JvpInterpreter(FuncTorchInterpreter):
    # JvpInterpreter 类，继承自 FuncTorchInterpreter 类

    def __init__(self, cdata: CInterpreter):
        # 初始化方法，接受一个 cdata 参数，类型为 CInterpreter 实例
        assert cdata.key() == TransformType.Jvp
        # 断言 cdata 的类型为 TransformType.Jvp

        # See NOTE: [Interpreter cdata vs cptr]
        # 查看注意事项: [Interpreter cdata vs cptr]
        self._cdata = cdata
        # 将 cdata 存储在实例变量 _cdata 中
        self._cptr = CJvpInterpreterPtr(cdata)
        # 使用 cdata 创建 CJvpInterpreterPtr 对象，并将其存储在 _cptr 中

    def lift(self, args, kwargs):
        # lift 方法，接受 args 和 kwargs 作为参数
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        # 使用 pytree.tree_map_only 方法将 args 和 kwargs 中的所有元素转换为 torch.Tensor 类型，调用 self._cptr.lift 方法
        return args, kwargs
        # 返回转换后的 args 和 kwargs

    def process(self, op, args, kwargs):
        # process 方法，接受 op、args 和 kwargs 作为参数
        kernel = op.functorch_table[TransformType.Jvp]
        # 从 op 的 functorch_table 属性中获取 TransformType.Jvp 对应的 kernel 函数
        args, kwargs = self.lift(args, kwargs)
        # 调用 self.lift 方法，对 args 和 kwargs 进行处理
        return kernel(self, *args, **kwargs)
        # 调用 kernel 函数，传入 self、args 和 kwargs，并返回结果

    # Jvp has custom lower because of the no_fwd_grad interaction
    # See NOTE [grad and vjp interaction with no_grad] for related info.
    # This logic is mirrored from C++ JvpInterpreterPtr::sendToNextInterpreter
    # Jvp 类有自定义的 lower 方法，因为与 no_fwd_grad 交互的特殊情况
    # 查看注意事项 [grad and vjp interaction with no_grad] 获取相关信息
    # 这个逻辑与 C++ 中 JvpInterpreterPtr::sendToNextInterpreter 方法中的逻辑是相同的
    def lower(self):
        # lower 方法
        prev_fwd_grad_mode = self.prev_fwd_grad_mode()
        # 调用 self.prev_fwd_grad_mode 方法，获取前向梯度模式
        if not prev_fwd_grad_mode:
            # 如果前向梯度模式为 False
            return nested(_set_fwd_grad_enabled(False), super().lower())
            # 调用 _set_fwd_grad_enabled(False) 设置前向梯度为 False，然后调用 super().lower() 返回结果
        return super().lower()
        # 否则直接调用 super().lower() 返回结果

    def prev_fwd_grad_mode(self):
        # prev_fwd_grad_mode 方法，获取前向梯度模式
        return self._cptr.prevFwdGradMode()
        # 调用 self._cptr.prevFwdGradMode() 方法，返回前向梯度模式

    def get_state(self):
        # get_state 方法，获取当前状态
        return (self.key().name, self.level(), self.prev_fwd_grad_mode())
        # 返回一个包含名称、级别和前向梯度模式的元组


class FunctionalizeInterpreter(FuncTorchInterpreter):
    # FunctionalizeInterpreter 类，继承自 FuncTorchInterpreter 类

    def __init__(self, cdata: CInterpreter):
        # 初始化方法，接受一个 cdata 参数，类型为 CInterpreter 实例
        assert cdata.key() == TransformType.Functionalize
        # 断言 cdata 的类型为 TransformType.Functionalize
        self._cdata = cdata
        # 将 cdata 存储在实例变量 _cdata 中
        self._cptr = CFunctionalizeInterpreterPtr(cdata)
        # 使用 cdata 创建 CFunctionalizeInterpreterPtr 对象，并将其存储在 _cptr 中

    def process(self, op, args, kwargs):
        # process 方法，接受 op、args 和 kwargs 作为参数
        kernel = op.functorch_table[TransformType.Functionalize]
        # 从 op 的 functorch_table 属性中获取 TransformType.Functionalize 对应的 kernel 函数
        return kernel(self, *args, **kwargs)
        # 调用 kernel 函数，传入 self、args 和 kwargs，并返回结果

    def functionalize_add_back_views(self):
        # functionalize_add_back_views 方法，添加回视图功能
        return self._cptr.functionalizeAddBackViews()
        # 调用 self._cptr.functionalizeAddBackViews() 方法，返回结果

    def get_state(self):
        # get_state 方法，获取当前状态
        return (self.key().name, self.level())
        # 返回一个包含名称和级别的元组


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    # coerce_cinterpreter 函数，将 CInterpreter 实例转换为 FuncTorchInterpreter 实例
    key = cinterpreter.key()
    # 获取 cinterpreter 的 key
    if key == TransformType.Grad:
        # 如果 key 为 TransformType.Grad
        return GradInterpreter(cinterpreter)
        # 返回 GradInterpreter 实例，使用 cinterpreter 初始化
    if key == TransformType.Vmap:
        # 如果 key 为 TransformType.Vmap
        return VmapInterpreter(cinterpreter)
        # 返回 VmapInterpreter 实例，使用 cinterpreter 初始化
    if key == TransformType.Jvp:
        # 如果 key 为 TransformType.Jvp
        return JvpInterpreter(cinterpreter)
        # 返回 JvpInterpreter 实例，使用 cinterpreter 初始化
    if key == TransformType.Functionalize:
        # 如果 key 为 TransformType.Functionalize
        return FunctionalizeInterpreter(cinterpreter)
        # 返回 FunctionalizeInterpreter 实例，使用 cinterpreter 初始化
    raise RuntimeError(f"NYI: PyDispatcher has not implemented support for {key}")
    # 抛出运行时错误，表示暂未实现对 key 类型的支持


def retrieve_current_functorch_interpreter() -> FuncTorchInterpreter:
    # retrieve_current_functorch_interpreter 函数，获取当前的 FuncTorchInterpreter 实例
    interpreter = torch._C._functorch.peek_interpreter_stack()
    # 获取顶部的 interpreter
    assert interpreter is not None
    # 断言 interpreter 不为空
    return coerce_cinterpreter(interpreter)
    # 返回转换后的 interpreter


def retrieve_all_functorch_interpreters() -> List[FuncTorchInterpreter]:
    # retrieve_all_functorch_interpreters 函数，获取所有的 FuncTorchInterpreter 实例列表
    cis = torch._C._functorch.get_interpreter_stack()
    # 获取所有的 interpreter 栈
    if cis is None:
        # 如果 cis 为空
        return []
        # 返回空列表
    return [coerce_cinterpreter(ci) for ci in cis]
    # 对于每个 ci，调用 coerce_cinterpreter 方法并返回结果列表


def compare_functorch_state(states: List[Tuple[Any, ...]]) -> bool:
    # compare_functorch_state 函数，比较 functorch 的状态
    # 这里涵盖了四种可能的情况：
    # 1. 当前堆栈为空，并且生成时堆栈不为空 -> 使其无效
    # 获取当前的 Functorch 解释器栈顶状态
    peek = torch._C._functorch.peek_interpreter_stack()
    # 检查条件：
    # 2. 当前解释器栈不为空且生成时为空 -> 无效
    # 3. 当前解释器栈和生成时均为空 -> 有效的 FX 图
    # 4. 当前解释器栈和生成时均不为空 -> 如果两个状态匹配，则有效
    if (peek is None and len(states) != 0) or (peek is not None and len(states) == 0):
        return False

    # 获取所有的 Functorch 解释器状态列表
    cis = retrieve_all_functorch_interpreters()
    # 返回 True 如果 Functorch 解释器数量与状态数量相同，并且所有状态都匹配相应的解释器
    return len(cis) == len(states) and all(
        ci.check_state(state) for ci, state in zip(cis, states)
    )
def dispatch_functorch(op, args, kwargs):
    # 检索当前的Functorch解释器
    interpreter = retrieve_current_functorch_interpreter()
    
    # 在传统的PyTorch操作符中，DispatchKey::FuncTorchTensorWrapper的
    # unwrap_dead_tensors回退处理了死亡张量包装器的解包。
    # PyDispatcher在处理functorch变换时绕过了PyTorch分发器，
    # 因此我们在这里手动解包死亡张量。
    # 当我们完全过渡到仅有模式的functorch时，这段逻辑将不再需要存在。
    args, kwargs = pytree.tree_map_only(
        torch.Tensor, torch._C._functorch.unwrap_if_dead, (args, kwargs)
    )
    
    # 调用解释器处理操作符op及其参数args和kwargs
    return interpreter.process(op, args, kwargs)
```