# `.\pytorch\torch\utils\module_tracker.py`

```
# mypy: allow-untyped-defs
# 导入 weakref 模块，用于创建弱引用对象
import weakref

# 导入 Set 类型，用于定义集合类型
from typing import Set

# 导入 torch 库，用于神经网络相关操作
import torch

# 从 torch.autograd.graph 模块导入 register_multi_grad_hook 函数
from torch.autograd.graph import register_multi_grad_hook

# 从 torch.nn.modules.module 模块导入以下函数
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

# 从 torch.utils._pytree 模块导入 tree_flatten 函数
from torch.utils._pytree import tree_flatten

# 定义模块可导出的内容列表
__all__ = ["ModuleTracker"]


class ModuleTracker:
    """
    ``ModuleTracker`` 是一个上下文管理器，用于跟踪 nn.Module 在执行过程中的层次结构，
    以便其他系统可以查询当前正在执行的 Module（或其 backward 正在执行）。

    通过访问此上下文管理器的 ``parents`` 属性，可以获取当前正在执行的所有 Module 的集合，
    使用它们的完全限定名称（fqn，也用作 state_dict 中的键）。
    可以通过访问 ``is_bw`` 属性来判断当前是否在执行 backward。

    注意，``parents`` 永远不会为空，并且始终包含 "Global" 键。
    ``is_bw`` 标志在 forward 之后仍然保持为 ``True``，直到另一个 Module 被执行。
    如果需要更精确的控制，请提交一个问题请求。将 fqn 映射到模块实例是可能的但尚未完成的，
    如果需要，请提交一个问题请求。

    示例用法：

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            # 在 forward 过程中访问任何内容
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias
            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    """

    parents: Set[str]
    """
    包含当前正在执行 forward 的每个模块的 fqn 的集合
    """

    def __init__(self):
        # 初始化 parents 属性为包含 "Global" 的集合
        self.parents = {"Global"}

        # 初始化 _known_modules 为弱引用字典，用于存储模块实例的弱引用
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

        # 初始化 _seen_modules 为弱引用集合，用于跟踪已经观察到的模块实例
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()

        # 初始化 _has_callback 标志为 False，用于跟踪是否已经设置了回调函数
        self._has_callback = False

    def _maybe_set_engine_callback(self):
        # 如果已经设置了回调函数，则直接返回
        if self._has_callback:
            return

        # 定义回调函数，重置 parents 为只包含 "Global"
        def callback():
            self.parents = {"Global"}
            self._has_callback = False

        # 将回调函数加入执行引擎的回调队列中
        torch.autograd.Variable._execution_engine.queue_callback(callback)

        # 标记已经设置了回调函数
        self._has_callback = True

    @property
    def is_bw(self):
        """
        布尔值属性，标记当前是否在执行 backward
        """
        return torch._C._current_graph_task_id() != -1
    # 获取模块的名称并缓存，如果未知则添加到已知模块列表
    def _get_mod_name(self, mod):
        if mod not in self._known_modules:
            self._known_modules[mod] = type(mod).__name__
        mod_name = self._known_modules[mod]
        # 如果模块尚未遍历过，则遍历其子模块并记录名称
        if mod not in self._seen_modules:
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
                self._get_mod_name(submod)
            self._seen_modules.add(mod)
        return mod_name

    # 返回一个函数，用于向父模块列表添加当前模块的名称
    def _get_append_fn(self, name, is_bw):
        def fn(*args):
            if is_bw:
                self._maybe_set_engine_callback()
            # 检查当前模块是否已经在父模块列表中，如果是，则打印错误信息
            if name in self.parents:
                print(
                    "The module hierarchy tracking seems to be messed up."
                    "Please file a bug to PyTorch."
                )
            self.parents.add(name)

        return fn

    # 返回一个函数，用于从父模块列表中移除当前模块的名称
    def _get_pop_fn(self, name, is_bw):
        def fn(*args):
            # 如果当前模块在父模块列表中，则移除
            if name in self.parents:
                self.parents.remove(name)
            # 如果不是反向传播，并且当前模块不在父模块列表中，则抛出运行时错误
            elif not is_bw:
                raise RuntimeError(
                    "The Module hierarchy tracking is wrong. Report a bug to PyTorch"
                )

        return fn

    # 前向传播前钩子函数，用于注册前向传播前的处理函数
    def _fw_pre_hook(self, mod, input):
        # 获取模块名称并添加到父模块列表
        name = self._get_mod_name(mod)
        self._get_append_fn(name, False)()

        # 将输入扁平化并找出需要梯度的张量，注册梯度钩子
        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            register_multi_grad_hook(tensors, self._get_pop_fn(name, True))

    # 前向传播后钩子函数，用于注册前向传播后的处理函数
    def _fw_post_hook(self, mod, input, output):
        # 获取模块名称并从父模块列表中移除
        name = self._get_mod_name(mod)
        self._get_pop_fn(name, False)()

        # 将输出扁平化并找出需要梯度的张量，注册梯度钩子
        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            register_multi_grad_hook(tensors, self._get_append_fn(name, True))

    # 进入上下文管理器时调用，注册前向传播前后钩子函数，并返回实例本身
    def __enter__(self):
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(self._fw_post_hook)
        return self

    # 退出上下文管理器时调用，移除前向传播前后钩子函数的注册
    def __exit__(self, *args):
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
```