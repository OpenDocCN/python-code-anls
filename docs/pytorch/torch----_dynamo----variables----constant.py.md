# `.\pytorch\torch\_dynamo\variables\constant.py`

```py
# mypy: ignore-errors  # 忽略类型检查错误

import operator  # 导入操作符模块
from typing import Dict, List  # 导入类型提示中的字典和列表类型

import torch  # 导入 PyTorch 库
from torch._dynamo.source import GetItemSource  # 从 torch._dynamo.source 中导入 GetItemSource

from .. import variables  # 从父级包中导入 variables 模块
from ..exc import unimplemented, UserError, UserErrorType  # 从父级包中导入异常和错误相关模块
from ..guards import GuardBuilder, install_guard  # 从父级包中导入 GuardBuilder 和 install_guard 函数
from ..utils import common_constant_types, istype, np  # 从父级包中导入常用的常量类型、istype 和 np 函数
from .base import typestr, VariableTracker  # 从当前包中的 base 模块导入 typestr 和 VariableTracker 类

# 不同类型到断言失败原因的映射字典
_type_to_assert_reason = {
    list: "List types must use ListVariable.",  # 列表类型必须使用 ListVariable
    dict: "Dict types must use ConstDictVariable.",  # 字典类型必须使用 ConstDictVariable
    torch.Tensor: "Tensor types must use TensorVariable.",  # Tensor 类型必须使用 TensorVariable
    torch.SymInt: "SymInts must use SymNodeVariable. "  # SymInt 类型必须使用 SymNodeVariable
    "If the underlying value is static, we will create a ConstantVariable and specialize.",  # 如果底层值是静态的，我们将创建一个 ConstantVariable 并进行特化
    torch.SymFloat: "SymInts must use SymNodeVariable",  # SymFloat 类型必须使用 SymNodeVariable
}


class ConstantVariable(VariableTracker):
    @staticmethod
    def create(value, **kwargs) -> VariableTracker:
        source = kwargs.get("source", None)  # 获取关键字参数中的 source，若不存在则为 None
        is_literal = ConstantVariable.is_literal(value)  # 检查是否为字面量
        if not is_literal:
            # 对于非字面量的情况，检查是否属于不允许的类型，并抛出相应的断言失败原因
            for disallowed_type, reason in _type_to_assert_reason.items():
                assert not isinstance(value, disallowed_type), reason

        # 处理列表和元组字面量的情况
        if is_literal and isinstance(value, (list, tuple, set, frozenset)):
            items = []
            for i, x in enumerate(value):
                item_source = GetItemSource(source, i) if source else None
                if item_source:
                    # 如果存在项源，安装相应的保护
                    install_guard(item_source.make_guard(GuardBuilder.CONSTANT_MATCH))
                items.append(
                    ConstantVariable.create(
                        x,
                        source=item_source,
                    )
                )
            if isinstance(value, (list, tuple)):
                # 若为列表或元组，则返回相应类型的列表变量对象
                return variables.BaseListVariable.cls_for(type(value))(items, **kwargs)
            else:
                assert isinstance(value, (set, frozenset)), type(value)
                # 若为集合或冻结集合，则返回集合变量对象
                return variables.SetVariable(items)

        return ConstantVariable(value, **kwargs)  # 返回常量变量对象
    # 初始化方法，接受一个值和任意关键字参数
    def __init__(self, value, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        
        # 如果 value 不是字面常量，则进行类型检查
        if not ConstantVariable.is_literal(value):
            # 遍历不允许的类型列表，并抛出断言错误
            for disallowed_type, reason in _type_to_assert_reason.items():
                assert not isinstance(value, disallowed_type), reason

        # 禁止 ConstantVariable 类型为列表或元组
        assert not isinstance(
            value, (list, tuple)
        ), "ConstantVariable(list) is banned - please create a ListVariable(items)"
        
        # 如果有 numpy 并且 value 是 numpy 数字类型，则取其标量值
        if np is not None and isinstance(value, np.number):
            self.value = value.item()
        else:
            self.value = value

    # 返回值的代理
    def as_proxy(self):
        return self.value

    # 返回值的字符串表示形式，包含类型和值的信息
    def __str__(self):
        return f"ConstantVariable({type(self.value).__name__}: {repr(self.value)})"

    # 返回值的 Python 类型
    def python_type(self):
        return type(self.value)

    # 返回值作为 Python 常量的形式
    def as_python_constant(self):
        return self.value

    # 判断值是否为 Python 常量
    def is_python_constant(self):
        return True

    # 返回属性 items 的文档字符串，用于添加 BaseListVariable 和 ConstantVariable 的情况
    @property
    def items(self):
        """
        Need this when adding a BaseListVariable and a ConstantVariable together.
        Happens in detectron2.
        """
        return self.unpack_var_sequence(tx=None)

    # 获取常量值的指定索引处的变量追踪器
    def getitem_const(self, arg: VariableTracker):
        return ConstantVariable.create(
            self.value[arg.as_python_constant()],
        )

    # 静态方法，判断对象是否为字面常量
    @staticmethod
    def is_literal(obj):
        # 如果对象类型在常见的常量类型列表中，则为字面常量
        if type(obj) in common_constant_types:
            return True
        # 如果对象类型是列表、元组、集合、冻结集或者 torch.Size，则其中所有元素必须为字面常量
        if type(obj) in (list, tuple, set, frozenset, torch.Size):
            return all(ConstantVariable.is_literal(x) for x in obj)
        return False

    # 解包变量序列的方法，返回 ConstantVariable 对象的列表
    def unpack_var_sequence(self, tx):
        try:
            return [ConstantVariable.create(x) for x in self.as_python_constant()]
        except TypeError as e:
            raise NotImplementedError from e

    # 获取常量的指定属性值
    def const_getattr(self, tx, name):
        # 如果值的类型是类，则抛出用户错误，提示不支持访问类成员
        if isinstance(self.value, type):
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                "Can't access members of type(obj) for a generated custom object. "
                "Please use __class__ instead",
                case_name="type_reflection_method",
            )
        # 否则获取值的指定属性
        member = getattr(self.value, name)
        # 如果属性是可调用的，则抛出未实现错误
        if callable(member):
            raise NotImplementedError
        return member

    # 调用对象的方法
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
        ) -> "VariableTracker":
        # 导入 SymNodeVariable 类，用于后续的类型检查
        from .tensor import SymNodeVariable

        # 如果 name 为 "format" 并且 self.value 的类型为 str，则调用内置变量 str.format
        if name == "format" and istype(self.value, str):
            return variables.BuiltinVariable(str.format).call_function(
                tx, [self, *args], kwargs
            )

        # 如果 args 中有任何一个元素的类型为 SymNodeVariable，则提升为 SymNodeVariable 类型，用于处理涉及动态形状的操作
        if any(isinstance(x, SymNodeVariable) for x in args):
            return variables.SymNodeVariable(self.as_proxy(), self.value).call_method(
                tx, name, args, kwargs
            )

        # 尝试将 args 和 kwargs 转换为其 Python 常量形式
        try:
            const_args = [a.as_python_constant() for a in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
        except NotImplementedError:
            # 如果转换失败，则调用父类的 call_method 方法处理
            return super().call_method(tx, name, args, kwargs)

        # 定义一个函数判断是否存在算术二元操作
        def has_arith_binop(num_ty):
            return (
                isinstance(self.value, num_ty)
                and hasattr(operator, name)
                and len(args) == 1
                and args[0].is_python_constant()
            )

        # 如果 self.value 的类型为 str 并且 name 在 str 类的方法中
        if isinstance(self.value, str) and name in str.__dict__.keys():
            # 获取 str 类中的对应方法，并创建 ConstantVariable 对象
            method = getattr(self.value, name)
            return ConstantVariable.create(method(*const_args, **const_kwargs))
        # 如果是整数或浮点数的算术二元操作
        elif has_arith_binop(int) or has_arith_binop(float):
            # 获取对应的运算符函数，并创建 ConstantVariable 对象
            op = getattr(operator, name)
            add_target = const_args[0]
            if isinstance(add_target, (torch.SymInt, torch.SymFloat)):
                from .tensor import SymNodeVariable

                # 创建 SymNodeVariable 对象处理符号化计算
                proxy = tx.output.create_proxy(
                    "call_function", op, (self.value, add_target), {}
                )
                return SymNodeVariable.create(tx, proxy, add_target)
            return ConstantVariable.create(op(self.value, add_target))
        # 如果是获取字符串长度的方法调用
        elif name == "__len__" and not (args or kwargs):
            # 返回字符串长度的 ConstantVariable 对象
            return ConstantVariable.create(len(self.value))
        # 如果是检查是否包含某元素的方法调用
        elif name == "__contains__" and len(args) == 1 and args[0].is_python_constant():
            assert not kwargs
            search = args[0].as_python_constant()
            result = search in self.value
            return ConstantVariable.create(result)

        # 报告未实现的常量方法调用
        unimplemented(f"const method call {typestr(self.value)}.{name}")

    # 调用 hasattr 函数检查对象是否具有指定属性或方法
    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        result = hasattr(self.value, name)
        return variables.ConstantVariable.create(result)
# 定义一个继承自 VariableTracker 的枚举变量类
class EnumVariable(VariableTracker):
    # 初始化方法，接受一个值和其他可选参数
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        # 将传入的值赋给实例变量 self.value
        self.value = value

    # 类方法，用于创建枚举变量实例
    @classmethod
    def create(cls, cls_type, value_vt, options):
        # 如果 value_vt 是 ConstantVariable 类的实例
        if isinstance(value_vt, variables.ConstantVariable):
            # 遍历枚举类型 cls_type 的成员
            for member in list(cls_type):
                # 如果成员的值等于 value_vt 的 Python 常量表示
                if member.value == value_vt.as_python_constant():
                    # 返回一个 EnumVariable 实例，使用该成员值和 options 参数
                    return cls(member, **options)
        # 如果不满足上述条件，抛出未实现异常
        unimplemented("Enum variable is constructed with non constant values")

    # 返回枚举值 self.value 作为代理对象
    def as_proxy(self):
        return self.value

    # 返回描述枚举变量的字符串表示
    def __str__(self):
        return f"EnumVariable({type(self.value)})"

    # 返回枚举值 self.value 的 Python 类型
    def python_type(self):
        return type(self.value)

    # 返回枚举值 self.value 作为 Python 常量
    def as_python_constant(self):
        return self.value

    # 获取枚举值 self.value 的属性，如果是可调用的成员，则抛出未实现异常
    def const_getattr(self, tx, name):
        member = getattr(self.value, name)
        if callable(member):
            raise NotImplementedError
        return member
```