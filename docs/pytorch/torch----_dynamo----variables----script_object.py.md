# `.\pytorch\torch\_dynamo\variables\script_object.py`

```py
# mypy: allow-untyped-defs
# 引入 functools 模块，用于装饰器功能
import functools
# 引入 Dict 类型提示，用于声明字典类型的变量
from typing import Dict

# 引入 torch 模块
import torch
# 引入相关异常和错误类
from ..exc import unimplemented, UnsafeScriptObjectError, Unsupported

# 引入 VariableTracker 类
from .base import VariableTracker
# 引入 UserDefinedObjectVariable 类
from .user_defined import UserDefinedObjectVariable


# 装饰器函数，用于捕获 Unsupported 异常并抛出 UnsafeScriptObjectError 异常
def _raise_hard_error_if_graph_break(reason):
    def deco(fn):
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                raise UnsafeScriptObjectError(e.msg) from e

        return graph_break_as_hard_error

    return deco


# 继承自 UserDefinedObjectVariable 类，用于表示 TorchScript 对象的变量
class TorchScriptObjectVariable(UserDefinedObjectVariable):
    # 缓存假脚本对象的字典
    _fake_script_object_cache: Dict[int, "TorchScriptObjectVariable"] = {}

    # 类方法，检查给定的用户类是否是 torch.ScriptObject 的子类
    @classmethod
    def is_matching_cls(cls, user_cls: type):
        return issubclass(user_cls, torch.ScriptObject)

    # 静态方法，用于创建 TorchScriptObjectVariable 对象
    @staticmethod
    def create(proxy, value, **options):
        return TorchScriptObjectVariable(proxy, value, **options)

    # 初始化方法，初始化 TorchScriptObjectVariable 对象
    def __init__(self, proxy, value, source, **kwargs):
        super().__init__(value, **kwargs)
        self.proxy = proxy
        self.proxy.node.meta["example_value"] = value
        self.source = source

    # 返回对象的代理
    def as_proxy(self):
        return self.proxy

    # 方法装饰器，捕获 Unsupported 异常并抛出 UnsafeScriptObjectError 异常
    # 用于获取属性时的方法，返回 TorchHigherOrderOperatorVariable 对象
    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def var_getattr(self, tx, name: str) -> VariableTracker:
        from torch._higher_order_ops.torchbind import call_torchbind
        from ..source import AttrSource
        from .higher_order_ops import TorchHigherOrderOperatorVariable

        # 获取属性名对应的方法
        method = getattr(self.value, name, None)
        if method is None:
            # 抛出未实现异常，提示未定义该方法
            unimplemented(
                f"FakeScriptObject doesn't define method {name}. Did you forget to implement it in the fake class?"
            )

        # 如果属性不可调用，抛出未实现异常
        if not callable(method):
            unimplemented(
                "Only method calls on TorchScript objects can be supported safely."
                " Please use method calls instead of attribute access."
            )

        # 创建并返回 TorchHigherOrderOperatorVariable 对象
        return TorchHigherOrderOperatorVariable.make(
            call_torchbind,
            source=AttrSource(self.source, name),
            script_obj_var=self,
            method_name=name,
        )

    # 方法装饰器，捕获 Unsupported 异常并抛出 UnsafeScriptObjectError 异常
    # 用于调用方法时的方法，抛出未实现异常
    @_raise_hard_error_if_graph_break(
        "Dynamo cannot safely trace script object due to graph break."
    )
    def call_method(self, tx, name, args, kwargs):
        unimplemented(f"call method {name} on script object is not safe.")
```