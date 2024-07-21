# `.\pytorch\torch\distributed\_tools\mod_tracker.py`

```py
# mypy: allow-untyped-defs
# 引入警告模块，用于处理警告信息
import warnings
# 引入弱引用模块，用于管理弱引用对象
import weakref
# 引入类型提示模块
from typing import Callable, Optional, Set

# 引入PyTorch模块
import torch
# 引入用于注册多重梯度钩子的函数
from torch.autograd.graph import register_multi_grad_hook
# 引入用于注册模块前向钩子的函数
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
# 引入PyTorch内部的树展平函数
from torch.utils._pytree import tree_flatten

# 导出的模块列表
__all__ = ["ModTracker"]

# 定义ModTracker类
class ModTracker:
    """
    ``ModTracker``是一个上下文管理器，用于跟踪执行过程中的nn.Module层次结构，
    以便其他系统可以查询当前正在执行的Module（或其反向传播正在执行的Module）。

    通过访问此上下文管理器的``parents``属性，可以获取当前所有正在执行的Module的集合，
    通过其完全限定名（也用作state_dict中的键）。
    可以访问``is_bw``属性来判断当前是否在执行反向传播。

    注意，``parents``集合从不为空，并始终包含“Global”键。
    ``is_bw``标志在前向传播结束后仍保持为``True``，直到执行另一个Module。
    如果需要更精确的控制，请提交问题请求。
    将fqn映射到模块实例是可能的，但尚未实现，请提交问题请求。

    示例用法：

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModTracker() as tracker:
            # 在前向传播期间访问任何内容
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias
            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    """

    # 包含当前运行其前向传播的每个模块的fqn的集合
    parents: Set[str]

    def __init__(self):
        # 初始化parents集合，包含全局键"Global"
        self.parents = {"Global"}
        # 活跃模块计数器
        self._active_module_cnt = {}
        # 已知模块的弱引用字典
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # 已见模块的弱引用集合
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()
        # 是否存在回调标志
        self._has_callback = False
        # 用户定义的前向传播钩子
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        # 用户定义的反向传播钩子
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _maybe_set_engine_callback(self):
        """
        如果没有设置回调函数，则设置引擎回调函数。
        这假设没有并发调用backward。
        """
        if self._has_callback:
            return

        def callback():
            # 重置parents为包含"Global"的集合
            self.parents = {"Global"}
            self._has_callback = False

        # 将回调函数添加到执行引擎的队列中
        torch.autograd.Variable._execution_engine.queue_callback(callback)
        self._has_callback = True

    @property
    def is_bw(self):
        """
        一个布尔值，标记当前是否在执行反向传播。
        """
        # 判断当前是否在执行反向传播
        return torch._C._current_graph_task_id() != -1
    def get_known_fqn(self, mod):
        """
        Return the fqn for the given module if it is known to the ``ModTracker``, otherwise ``None``.
        """
        # 返回给定模块的全限定名称（fqn），如果模块在ModTracker中已知，否则返回None
        return self._known_modules.get(mod, None)

    def register_user_hooks(
        self,
        pre_fw_hook: Optional[Callable] = None,
        post_fw_hook: Optional[Callable] = None,
        pre_bw_hook: Optional[Callable] = None,
        post_bw_hook: Optional[Callable] = None,
    ):
        """
        Registers user-specified hooks to be called before/after the forward/backward pass for each
        module tracked by the ``ModTracker``. One or more can be ``None``.
        Args:
            pre_fw_hook (Callable, optional): A hook to be called before the forward pass for the
                module. It should have the following signature:
                pre_fw_hook (module, input) -> None
            post_fw_hook (Callable, optional): A hook to be called after the forward pass for the
                module. It should have the following signature:
                post_fw_hook (module, input, output) -> None
            pre_bw_hook (Callable, optional): A multi-grad hook to be called on all the outputs of
                the module that require gradients. It should have the following signature:
                pre_bw_hook (module, grad_output) -> None
            post_bw_hook (Callable, optional): A multi-grad hook to be called on all the inputs of
                the module that require gradients. It should have the following signature:
                post_bw_hook (module, grad_input) -> None
        Raises:
            AssertionError: If a new hook is provided when one is already registered.
        Note:
            If the module is not alive during the backward pass, the pre_bw_hook and post_bw_hook will
            will receive None as the module argument.
            The module fqn will be present in the ``parents`` attribute when each of the hooks is called.
            Hooks are intended to be used as markers only not to modify the inputs/outputs.
        """

        def set_hook(hook, user_hook, hook_name):
            if hook is not None and user_hook is not None:
                raise AssertionError(
                    f"Only one {hook_name} can be registered at a time"
                    f" Clear the existing hook by calling ``clear_user_hooks`` before registering a new one"
                )
            return hook

        # 设置用户定义的钩子函数，检查是否已经存在注册的钩子函数，防止重复注册
        self._user_pre_fw_hook = set_hook(
            pre_fw_hook, self._user_pre_fw_hook, "pre_fw_hook"
        )
        self._user_post_fw_hook = set_hook(
            post_fw_hook, self._user_post_fw_hook, "post_fw_hook"
        )
        self._user_pre_bw_hook = set_hook(
            pre_bw_hook, self._user_pre_bw_hook, "pre_bw_hook"
        )
        self._user_post_bw_hook = set_hook(
            post_bw_hook, self._user_post_bw_hook, "post_bw_hook"
        )
    def clear_user_hooks(self):
        """
        Clears the user specified hooks registered with ``register_user_hooks``
        """
        # 清空用户注册的钩子函数
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _get_mod_name(self, mod):
        if mod not in self._known_modules:
            # 如果模块不在已知模块中，则将其添加，并记录模块类型名称
            self._known_modules[mod] = type(mod).__name__
        # 获取模块的名称
        mod_name = self._known_modules[mod]
        if mod not in self._seen_modules:
            # 如果模块未被记录过，则遍历其子模块，记录子模块名称
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
                self._get_mod_name(submod)
            # 将当前模块标记为已记录
            self._seen_modules.add(mod)
        return mod_name

    def _get_append_fn(self, w_mod, name, is_bw):
        def fn(*args):
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents and not self.is_bw:
                # 定义自定义的警告消息格式
                def custom_formatwarning(msg, category, filename, lineno, line=None):
                    return f"{filename}:{lineno}: {category.__name__}: {msg} \n"
                # 设置警告输出格式为自定义格式
                warnings.formatwarning = custom_formatwarning
                # 发出警告消息
                warnings.warn(
                    "The module hierarchy tracking maybe be messed up."
                    " Please file a bug to PyTorch, if it is the case."
                )
            if name not in self.parents:
                # 如果模块名称不在父模块集合中，则添加，并记录模块活跃计数为1
                self._active_module_cnt[name] = 1
                self.parents.add(name)
            else:
                # 否则，增加模块活跃计数
                self._active_module_cnt[name] += 1

            if self._user_pre_bw_hook is not None and is_bw:
                # 如果存在用户定义的反向钩子函数，并且当前为反向传播阶段，则调用它
                self._user_pre_bw_hook(w_mod(), args)

        return fn

    def _get_pop_fn(self, w_mod, name, is_bw):
        def fn(*args):
            if self._user_post_bw_hook is not None and is_bw:
                # 如果存在用户定义的反向后钩子函数，并且当前为反向传播阶段，则调用它
                self._user_post_bw_hook(w_mod(), args)
            if name in self.parents:
                # 如果模块名称在父模块集合中，则减少模块活跃计数
                self._active_module_cnt[name] -= 1
                if self._active_module_cnt[name] == 0:
                    # 如果模块活跃计数为0，则从父模块集合中移除
                    self.parents.remove(name)
            elif not self.is_bw:
                # 如果不是反向传播阶段且模块名称不在父模块集合中，抛出运行时错误
                raise RuntimeError(
                    "The Module hierarchy tracking is wrong. Report a bug to PyTorch"
                )

        return fn

    def _fw_pre_hook(self, mod, input):
        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        self._get_append_fn(w_mod, name, False)()
        if self._user_pre_fw_hook is not None:
            # 如果存在用户定义的前向预钩子函数，则调用它
            self._user_pre_fw_hook(mod, input)
        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw and tensors:
            # 如果不是反向传播阶段且输入包含需要梯度的张量，则注册多重梯度钩子
            register_multi_grad_hook(tensors, self._get_pop_fn(w_mod, name, True))
    # 定义一个私有方法 `_fw_post_hook`，用于处理模型前向传播后的钩子函数
    def _fw_post_hook(self, mod, input, output):
        # 获取模块名称
        name = self._get_mod_name(mod)
        # 使用弱引用封装模块对象
        w_mod = weakref.ref(mod)
        # 如果用户定义了后向传播钩子函数，则调用该函数
        if self._user_post_fw_hook is not None:
            self._user_post_fw_hook(mod, input, output)
        # 获取并调用弹出函数，用于处理模块后向传播时的清理工作
        self._get_pop_fn(w_mod, name, False)()
        # 将输出扁平化，并仅保留包含梯度信息的张量
        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        # 如果不是反向传播阶段且存在需要梯度的张量，则注册多梯度钩子函数
        if not self.is_bw and tensors:
            register_multi_grad_hook(tensors, self._get_append_fn(w_mod, name, True))

    # 进入上下文管理器时调用，注册模块前向传播前钩子和后向传播钩子
    def __enter__(self):
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(
            self._fw_post_hook, always_call=True
        )
        return self

    # 退出上下文管理器时调用，移除注册的模块前向传播前钩子和后向传播钩子
    def __exit__(self, *args):
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
```