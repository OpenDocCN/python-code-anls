# `.\pytorch\torch\utils\hooks.py`

```
# 引入mypy模块，并允许未类型化的定义
import torch
# 引入OrderedDict用于有序字典操作
from collections import OrderedDict
# 引入weakref用于弱引用的支持
import weakref
# 引入warnings模块用于警告处理
import warnings
# 引入Any和Tuple用于类型提示
from typing import Any, Tuple

# 定义公开接口列表
__all__ = ["RemovableHandle", "unserializable_hook", "warn_if_has_hooks", "BackwardHook"]

class RemovableHandle:
    """
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """

    # 类级别的属性，用于生成唯一id
    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        # 使用弱引用来持有hooks_dict，避免循环引用
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        # 设置当前对象的id，并更新下一个可用的id
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        # 处理额外的字典引用，使用弱引用来持有extra_dict中的字典对象
        self.extra_dict_ref: Tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        # 从hooks_dict中移除当前id对应的hook
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        # 从extra_dict中移除当前id对应的hook
        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        # 序列化对象状态，包括hooks_dict_ref、id及extra_dict_ref的引用状态
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (self.hooks_dict_ref(), self.id, tuple(ref() for ref in self.extra_dict_ref))

    def __setstate__(self, state) -> None:
        # 反序列化对象状态，恢复hooks_dict_ref、id及extra_dict_ref的引用状态
        if state[0] is None:
            # 创建一个失效的引用，即空的OrderedDict引用
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        # 更新next_id，确保它大于当前的id值
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        # 支持上下文管理协议，返回当前对象本身
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        # 退出上下文管理器时执行的操作，调用remove方法移除对应的hook
        self.remove()


def unserializable_hook(f):
    """
    Mark a function as an unserializable hook with this decorator.

    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.
    """
    # 使用装饰器标记函数为不可序列化的hook
    f.__torch_unserializable__ = True
    return f


def warn_if_has_hooks(tensor):
    # 函数用途待补充
    pass
    # 检查张量是否有反向传播钩子（hook）
    if tensor._backward_hooks:
        # 遍历张量的所有反向传播钩子
        for k in tensor._backward_hooks:
            # 获取具体的反向传播钩子对象
            hook = tensor._backward_hooks[k]
            # 检查该钩子对象是否没有属性 "__torch_unserializable__"
            if not hasattr(hook, "__torch_unserializable__"):
                # 如果没有 "__torch_unserializable__" 属性，则发出警告
                warnings.warn(f"backward hook {repr(hook)} on tensor will not be "
                              "serialized.  If this is expected, you can "
                              "decorate the function with @torch.utils.hooks.unserializable_hook "
                              "to suppress this warning")
class BackwardHook:
    """
    A wrapper class to implement nn.Module backward hooks.

    It handles:
      - Ignoring non-Tensor inputs and replacing them by None before calling the user hook
      - Generating the proper Node to capture a set of Tensor's gradients
      - Linking the gradients captures for the outputs with the gradients captured for the input
      - Calling the user hook once both output and input gradients are available
    """

    def __init__(self, module, user_hooks, user_pre_hooks):
        # 初始化反向钩子对象
        self.user_hooks = user_hooks
        # 用户定义的反向钩子函数列表
        self.user_pre_hooks = user_pre_hooks
        # 用户定义的预处理反向钩子函数列表
        self.module = module
        # 所属的 nn.Module 对象

        self.grad_outputs = None
        # 梯度输出初始化为 None
        self.n_outputs = -1
        # 输出数量初始化为 -1
        self.output_tensors_index = None
        # 输出张量索引初始化为 None
        self.n_inputs = -1
        # 输入数量初始化为 -1
        self.input_tensors_index = None
        # 输入张量索引初始化为 None

    def _pack_with_none(self, indices, values, size):
        # 将指定索引处的值填充到指定大小的列表中，其余位置用 None 填充
        res = [None] * size
        for idx, val in zip(indices, values):
            res[idx] = val

        return tuple(res)

    def _unpack_none(self, indices, values):
        # 解包给定索引处的值，返回为元组
        res = []
        for idx in indices:
            res.append(values[idx])

        return tuple(res)

    def _set_user_hook(self, grad_fn):
        # 设置用户定义的反向钩子函数
        def hook(grad_input, _):
            if self.grad_outputs is None:
                # 如果梯度输出为空，则直接返回（通常在双重反向传播时发生）
                return
            res = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)

            for hook in self.user_hooks:
                out = hook(self.module, res, self.grad_outputs)

                if out is None:
                    continue

                if len(out) != len(res):
                    raise RuntimeError("Backward hook returned an invalid number of grad_input, "
                                       f"got {len(out)}, but expected {len(res)}")

                res = out

            self.grad_outputs = None
            # 清空梯度输出

            return self._unpack_none(self.input_tensors_index, res)

        grad_fn.register_hook(hook)
        # 注册钩子函数到梯度函数
    # 将给定函数应用于参数中包含的张量。返回更新后的参数和张量索引。
    def _apply_on_tensors(self, fn, args):
        # 用于存储张量的索引和实际张量
        tensors_idx = []
        tensors = []

        # 检查每个参数，如果是张量则记录其索引和内容
        requires_grad = False
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensors_idx.append(i)
                tensors.append(arg)
                requires_grad |= arg.requires_grad

        # 如果没有张量需要梯度或者梯度已禁用，则直接返回原始参数和空张量索引
        if not (requires_grad and torch.is_grad_enabled()):
            return args, None

        # 调用 PyTorch 内部函数 BackwardHookFunction.apply 处理张量
        new_tensors = torch.nn.modules._functions.BackwardHookFunction.apply(*tensors)
        # 如果处理后的张量为空，则抛出运行时错误
        if len(new_tensors) == 0:
            raise RuntimeError("Cannot set Module backward hook for a Module with no input Tensors.")

        # 提取每个新张量的梯度函数，并确保至少有一个有效的梯度函数存在
        grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == "BackwardHookFunctionBackward"]
        if len(grad_fns) == 0:
            raise RuntimeError("Error while setting up backward hooks. Please open "
                               "an issue with a code sample to reproduce this.")

        # 将处理后的梯度函数应用于指定的函数 fn
        fn(grad_fns[0])

        # 更新参数列表中的张量部分
        arg_list = list(args)
        for idx, val in zip(tensors_idx, new_tensors):
            arg_list[idx] = val

        # 根据参数类型返回更新后的结果
        if type(args) is tuple:
            out = tuple(arg_list)
        else:
            out = type(args)(*arg_list)
        return out, tensors_idx

    # 设置输入钩子函数，用于处理输入参数中的张量
    def setup_input_hook(self, args):
        # 内部函数，将梯度函数设置为用户定义的钩子
        def fn(grad_fn):
            self._set_user_hook(grad_fn)

        # 调用 _apply_on_tensors 函数，处理输入参数中的张量，并返回处理结果和张量索引
        res, input_idx = self._apply_on_tensors(fn, args)
        
        # 记录输入参数的数量和张量索引
        self.n_inputs = len(args)
        self.input_tensors_index = input_idx
        
        # 返回处理后的结果
        return res
    # 设置输出hook的函数，接受参数args作为输入
    def setup_output_hook(self, args):
        # 定义内部函数fn，接受grad_fn作为参数
        def fn(grad_fn):
            # 定义hook函数，接受_和grad_output作为参数
            def hook(_, grad_output):
                # 使用_pack_with_none方法将输出张量索引、grad_output和输出数量打包成self.grad_outputs
                self.grad_outputs = self._pack_with_none(self.output_tensors_index,
                                                         grad_output,
                                                         self.n_outputs)

                # 如果存在用户预处理钩子
                if self.user_pre_hooks:
                    # 计算预期的self.grad_outputs长度
                    expected_len = len(self.grad_outputs)
                    # 遍历每个用户预处理钩子
                    for user_pre_hook in self.user_pre_hooks:
                        # 调用用户预处理钩子，处理self.module和self.grad_outputs
                        hook_grad_outputs = user_pre_hook(self.module, self.grad_outputs)
                        # 如果hook_grad_outputs为None，则继续下一个循环
                        if hook_grad_outputs is None:
                            continue

                        # 计算实际的hook_grad_outputs长度
                        actual_len = len(hook_grad_outputs)
                        # 如果实际长度与预期长度不匹配，则抛出运行时错误
                        if actual_len != expected_len:
                            raise RuntimeError("Backward pre hook returned an invalid number of grad_output, "
                                               f"got {actual_len}, but expected {expected_len}")
                        # 更新self.grad_outputs为hook处理后的结果
                        self.grad_outputs = hook_grad_outputs

                # 需要能够清除self.grad_outputs但也需要返回它
                local_grad_outputs = self.grad_outputs

                # 如果不需要输入梯度，则此hook应直接调用用户hook
                if self.input_tensors_index is None:
                    # 使用_pack_with_none方法将空列表作为输入梯度，和空列表作为grad_outputs打包成grad_inputs
                    grad_inputs = self._pack_with_none([], [], self.n_inputs)
                    # 遍历每个用户hook
                    for user_hook in self.user_hooks:
                        # 调用用户hook，处理self.module、grad_inputs和self.grad_outputs
                        res = user_hook(self.module, grad_inputs, self.grad_outputs)
                        # 如果res不为None且不是全为None的元组，则抛出运行时错误
                        if res is not None and not (isinstance(res, tuple) and all(el is None for el in res)):
                            raise RuntimeError("Backward hook for Modules where no input requires "
                                               "gradient should always return None or None for all gradients.")
                    # 将self.grad_outputs设为None，表示没有梯度输出
                    self.grad_outputs = None

                # 如果local_grad_outputs不为None，则断言self.output_tensors_index不为None（类型检查）
                if local_grad_outputs is not None:
                    assert self.output_tensors_index is not None  # mypy
                    # 返回self.output_tensors_index指定的local_grad_outputs中的元素构成的元组
                    return tuple(local_grad_outputs[i] for i in self.output_tensors_index)

            # 将hook函数注册到grad_fn中
            grad_fn.register_hook(hook)

        # 初始化is_tuple为True
        is_tuple = True
        # 如果args不是元组，则将其转换为包含args的元组，同时将is_tuple设为False
        if not isinstance(args, tuple):
            args = (args,)
            is_tuple = False

        # 调用_apply_on_tensors方法，传入fn函数和args，获取返回结果res和输出索引output_idx
        res, output_idx = self._apply_on_tensors(fn, args)
        # 将self.n_outputs设为args的长度
        self.n_outputs = len(args)
        # 将self.output_tensors_index设为output_idx
        self.output_tensors_index = output_idx

        # 如果原来args不是元组，则将res设为res的第一个元素
        if not is_tuple:
            res = res[0]
        # 返回res作为结果
        return res
```