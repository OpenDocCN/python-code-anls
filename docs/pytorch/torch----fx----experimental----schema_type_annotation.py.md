# `.\pytorch\torch\fx\experimental\schema_type_annotation.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 相关模块
import torch
import torch.fx
import inspect
from typing import Any, Dict, Optional, Tuple
from torch.fx.node import Argument, Target
from torch._jit_internal import boolean_dispatched
from torch.fx.operator_schemas import _torchscript_type_to_python_type

from torch.fx import Transformer

class AnnotateTypesWithSchema(Transformer):
    """
    使用 Python 函数签名来为 FX 图中的 `Nodes` 注释类型。
    这会提取出以下内容的 Python 函数签名：

        1. 标准的 `torch.nn` 模块调用
        2. `torch.nn.functional` 调用
        3. 通过 `get_attr` 获取的属性

    示例用法：

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = AnnotateTypesWithSchema(traced).transform()

    """
    def __init__(self, module : torch.nn.Module, annotate_functionals : bool = True,
                 annotate_modules : bool = True, annotate_get_attrs : bool = True):
        super().__init__(module)
        # 初始化属性
        self.annotate_functionals = annotate_functionals
        self.annotate_modules = annotate_modules
        self.annotate_get_attrs = annotate_get_attrs

    def call_function(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        # 初始化 Python 返回类型为 None
        python_ret_type = None
        # 如果允许注释 functionals 并且目标位于 'torch.nn.functional' 模块下
        if self.annotate_functionals and target.__module__ == 'torch.nn.functional':
            target_for_analysis = target
            # 如果目标函数在 boolean_dispatched 中
            if target in boolean_dispatched:
                # HACK: `boolean_dispatch` 在 `torch.nn.functional` 中的使用使得我们可以基于布尔值进行双向分发。
                # 这里我们检查分发的 `true` 和 `false` 分支是否具有完全相同的签名。
                # 如果是，则使用 `true` 分支的签名进行分析，否则保持未归一化状态。
                assert not isinstance(target, str)
                dispatched = boolean_dispatched[target]
                if_true, if_false = dispatched['if_true'], dispatched['if_false']
                # TODO: 我们是否可以发出这些的联合？这对 TorchScript 编译有何影响？
                if inspect.signature(if_true).return_annotation != inspect.signature(if_false).return_annotation:
                    return super().call_function(target, args, kwargs)
                target_for_analysis = if_true

            # 提取目标函数的 Python 返回类型
            python_ret_type = self._extract_python_return_type(target_for_analysis)

        # 调用父类的 call_function 方法
        return_proxy = super().call_function(target, args, kwargs)
        # 如果返回的代理节点类型为空，则使用 python_ret_type
        return_proxy.node.type = return_proxy.node.type if return_proxy.node.type else python_ret_type
        return return_proxy
    # 调用模块的方法，传入目标对象、位置参数元组、关键字参数字典
    def call_module(self, target : Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        # 初始化 Python 返回类型为 None
        python_ret_type = None
        # 断言目标对象是字符串类型
        assert isinstance(target, str)
        # 获取目标对象的属性
        submod = self.fetch_attr(target)
        # 如果启用模块注解并且子模块有类名属性
        if self.annotate_modules and hasattr(submod.__class__, '__name__'):
            # 获取子模块的类名
            classname = submod.__class__.__name__
            # 如果子模块是 torch.nn 模块的实例，则尝试提取其 forward 方法的 Python 返回类型
            if getattr(torch.nn, classname, None) == submod.__class__:
                python_ret_type = self._extract_python_return_type(submod.forward)
        # 调用父类的 call_module 方法，获取返回的代理对象
        return_proxy = super().call_module(target, args, kwargs)
        # 如果返回的代理对象的节点类型为空，则设置为之前提取的 Python 返回类型
        return_proxy.node.type = return_proxy.node.type if return_proxy.node.type else python_ret_type
        # 返回更新后的代理对象
        return return_proxy

    # 获取属性的方法，传入目标对象、位置参数元组、关键字参数字典
    def get_attr(self, target : torch.fx.node.Target, args : Tuple[Argument, ...], kwargs : Dict[str, Any]):
        # 调用父类的 get_attr 方法，获取返回的属性代理对象
        attr_proxy = super().get_attr(target, args, kwargs)

        # 如果启用了获取属性的注解
        if self.annotate_get_attrs:
            # 从当前模块开始迭代查找属性
            module_itr = self.module
            # 断言目标对象是字符串类型
            assert isinstance(target, str)
            # 按点号分割目标字符串，迭代查找属性
            atoms = target.split('.')
            for i, atom in enumerate(atoms):
                # 如果当前模块没有对应属性，则抛出运行时错误
                if not hasattr(module_itr, atom):
                    raise RuntimeError(f'Node referenced nonextent target {".".join(atoms[:i])}!')
                module_itr = getattr(module_itr, atom)

            # 尝试推断当前模块的 TorchScript 类型
            maybe_inferred_ts_type = torch._C._jit_try_infer_type(module_itr)
            # 如果推断成功，转换为 Python 类型，并设置属性代理对象的节点类型
            if maybe_inferred_ts_type.success():
                python_type = _torchscript_type_to_python_type(maybe_inferred_ts_type.type())
                attr_proxy.node.type = python_type if not attr_proxy.node.type else attr_proxy.node.type

        # 返回更新后的属性代理对象
        return attr_proxy

    # 提取 Python 返回类型的方法，传入目标对象作为可调用对象，返回可选的类型注解
    def _extract_python_return_type(self, target : Target) -> Optional[Any]:
        """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
        # 断言目标对象是可调用的
        assert callable(target)
        try:
            # 获取目标对象的签名信息
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            # 如果获取签名信息失败，返回 None
            return None

        # 返回目标对象的返回注解，如果不存在则返回 None
        return sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else None
```