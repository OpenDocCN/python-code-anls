# `.\pytorch\torch\_dynamo\variables\sdpa.py`

```
# mypy: ignore-errors

# 从 inspect 模块中导入 getattr_static 函数
from inspect import getattr_static

# 导入自定义模块和类
from ..bytecode_transformation import create_call_function
from ..exc import Unsupported
from .base import VariableTracker


class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for scaled dot product attention.
    This is a read-only container."""

    @staticmethod
    def create(tx, value, source):
        # 导入 torch.backends.cuda 模块中的 SDPAParams 类
        from torch.backends.cuda import SDPAParams
        # 导入相关模块和类
        from ..source import AttrSource
        from .builder import VariableBuilder
        from .torch import TorchInGraphFunctionVariable

        # 使用 VariableBuilder 创建各种参数变量
        query_var = VariableBuilder(tx, AttrSource(source, "query"))(value.query)
        key_var = VariableBuilder(tx, AttrSource(source, "key"))(value.key)
        value_var = VariableBuilder(tx, AttrSource(source, "value"))(value.value)
        attn_mask_var = VariableBuilder(tx, AttrSource(source, "attn_mask"))(
            value.attn_mask
        )
        dropout_var = VariableBuilder(tx, AttrSource(source, "dropout"))(value.dropout)
        is_causal_var = VariableBuilder(tx, AttrSource(source, "is_causal"))(
            value.is_causal
        )
        # 将所有参数变量存入列表
        param_vars = [
            query_var,
            key_var,
            value_var,
            attn_mask_var,
            dropout_var,
            is_causal_var,
        ]
        # 调用 TorchInGraphFunctionVariable 类的 call_function 方法
        return TorchInGraphFunctionVariable(SDPAParams).call_function(
            tx, param_vars, {}
        )

    def __init__(self, proxy, param_vars, **kwargs):
        # 初始化 SDPAParamsVariable 对象，设置代理和参数变量
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def reconstruct(self, codegen):
        # 在重建过程中，确保来源为空，参数变量不为空
        assert self.source is None
        assert self.param_vars is not None
        # 向 codegen 中添加推送空值操作，加载 torch._C 模块中的 _SDPAParams 符号
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._C", "_SDPAParams")
        )
        # 遍历参数变量列表并扩展输出
        codegen.foreach(self.param_vars)
        codegen.extend_output(create_call_function(len(self.param_vars), False))

    def as_proxy(self):
        # 返回代理对象
        return self.proxy

    def var_getattr(self, tx, name: str) -> VariableTracker:
        # 导入 torch._C 模块
        import torch._C
        from ..source import AttrSource
        from .builder import wrap_fx_proxy
        from .misc import GetAttrVariable

        try:
            # 尝试获取静态属性
            getattr_static(torch._C._SDPAParams, name)
        except AttributeError:
            # 如果属性不存在，则抛出 Unsupported 异常
            raise Unsupported(
                f"Unsupported torch._C._SDPAParams attribute {name}"
            ) from None

        # 创建属性访问代理
        proxy = GetAttrVariable.create_getattr_proxy(self.as_proxy(), name)
        if self.source is not None:
            # 如果存在来源，则用来源包装代理对象并返回
            return wrap_fx_proxy(
                tx=tx, proxy=proxy, source=AttrSource(self.source, name)
            )
        else:
            # 否则直接返回代理对象
            return wrap_fx_proxy(tx=tx, proxy=proxy)

    @staticmethod
    def is_sdpa_params(value):
        # 判断给定值是否为 SDPAParams 类型
        from torch.backends.cuda import SDPAParams

        return value is SDPAParams
```