# `.\pytorch\torch\_dynamo\variables\base.py`

```
# mypy: ignore-errors

# 导入必要的模块和库
import collections
from enum import Enum
from typing import Any, Callable, Dict, List

# 从上级目录导入特定模块
from .. import variables
from ..current_scope_id import current_scope_id
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import istype

# 定义枚举 MutableLocalSource，用于标识可变局部变量的来源类型
class MutableLocalSource(Enum):
    """
    If the VariableTracker.mutable_local represents a Variable that:
    - already existed that Dynamo began tracking while introspection (Existing)
    - is a new variable that is created during Dynamo introspection (Local)
    """
    Existing = 0
    Local = 1

# 定义基类 MutableLocalBase，用于表示可变局部变量的基础属性和行为
class MutableLocalBase:
    """
    Base class for Variable.mutable_local
    """
    def __init__(self, typ: MutableLocalSource):
        # 根据传入的 typ 参数确定当前对象的 scope 属性，用于标识对象的作用域
        # 在 HigherOrderOperator 追踪中，我们需要区分位于 HigherOrderOperator 内部和外部的 MutableLocals
        # 例如，在以下示例中，不能安全地修改 `a`，因为它是在不同的作用域中构造的。
        #
        # def f(x):
        #     a = 1
        #     def g(x):
        #         nonlocal a
        #         a = 2
        #         return x
        #     return wrap(g, x) + a
        #
        # 我们使用 self.scope 区分这一点。
        # scope == 0: 对象是一个已存在的变量
        # scope == 1: 对象是在 Dynamo 内省函数时创建的
        # scope >= 2: 对象是通过 Dynamo 内省高阶操作创建的，确切的数字对应嵌套高阶操作的层级
        if typ is MutableLocalSource.Existing:
            self.scope = 0
        elif typ is MutableLocalSource.Local:
            self.scope = current_scope_id()
        else:
            unimplemented(f"Unsupported MutableLocalSource: {typ}")

# 定义 MutableLocal 类，继承自 MutableLocalBase，表示可变局部变量的特性
class MutableLocal(MutableLocalBase):
    """
    Marker used to indicate this (list, iter, etc) was constructed in
    local scope and can be mutated safely in analysis without leaking
    state.
    """
    def __init__(self):
        super().__init__(MutableLocalSource.Local)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

# 判断作用域标识是否为顶层作用域
def _is_top_level_scope(scope_id):
    return scope_id == 1

# 判断是否允许对 MutableLocalBase 类型的对象进行副作用安全的修改
def is_side_effect_safe(m: MutableLocalBase):
    scope_id = current_scope_id()

    # 在顶层作用域（如果没有涉及到 HigherOrderOperators），允许修改本作用域内创建的变量以及已存在的变量
    if _is_top_level_scope(scope_id):
        return True
    # 否则，仅允许对当前作用域中创建的变量进行局部修改
    return m.scope == scope_id

# 定义 VariableTrackerMeta 元类
class VariableTrackerMeta(type):
    all_subclasses = []
    # 实现 __instancecheck__ 方法，用于判断是否为指定类型的实例
    def __instancecheck__(cls, instance) -> bool:
        """Make isinstance work with LazyVariableTracker"""
        # 检查 instance 是否为 LazyVariableTracker 的实例，并且 cls 不是 VariableTracker 或 LazyVariableTracker
        if type.__instancecheck__(
            variables.LazyVariableTracker, instance
        ) and cls not in (
            VariableTracker,
            variables.LazyVariableTracker,
        ):
            # 若满足条件，调用 instance 的 realize 方法
            instance = instance.realize()
        # 调用 type 类的 __instancecheck__ 方法进行实例检查，并返回结果
        return type.__instancecheck__(cls, instance)

    # 初始化方法，当创建新类时自动调用
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # 将当前类 cls 添加到 VariableTrackerMeta 类的 all_subclasses 列表中
        VariableTrackerMeta.all_subclasses.append(cls)
# 定义变量追踪器类，使用 VariableTrackerMeta 作为元类
class VariableTracker(metaclass=VariableTrackerMeta):
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.
    """

    # 不应在 apply() 中修改的字段集合
    _nonvar_fields = {
        "value",
        "guards",
        "source",
        "mutable_local",
        "parents_tracker",
        "user_code_variable_name",
    }

    # 浅复制方法，可选择性地进行一些更改
    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def visit(
        cls,
        fn: Callable[["VariableTracker"], None],
        value,
        cache=None,
    ):
        """
        Walk value and call fn on all the VariableTracker instances
        """
        if cache is None:
            cache = dict()

        idx = id(value)
        if idx in cache:
            return
        # 保存 value，确保 id() 不会被重用
        cache[idx] = value

        if isinstance(value, VariableTracker):
            value = value.unwrap()
            fn(value)
            value = value.unwrap()  # 调用 fn() 可能会实现它
            nonvars = value._nonvar_fields
            # 遍历 value 对象的 __dict__ 属性，处理不属于非变量字段的子值
            for key, subvalue in value.__dict__.items():
                if key not in nonvars:
                    cls.visit(fn, subvalue, cache)
        elif istype(value, (list, tuple)):
            # 对于列表或元组类型的 value，递归处理每个子值
            for subvalue in value:
                cls.visit(fn, subvalue, cache)
        elif istype(value, (dict, collections.OrderedDict)):
            # 对于字典类型的 value，递归处理每个值
            for subvalue in value.values():
                cls.visit(fn, subvalue, cache)

    def __repr__(self):
        # 返回变量追踪器类的字符串表示形式
        return f"{self.__class__.__name__}()"

    def debug_repr(self):
        # 应当被子类重写以提供更多信息的调试表示
        try:
            # 调用 as_python_constant() 方法获取 Python 常量的表示形式并返回其字符串表示
            return repr(self.as_python_constant())
        except NotImplementedError:
            # 如果无法获取常量表示，返回实例的默认字符串表示
            return repr(self)

    def python_type(self):
        """
        Abstract method to be implemented by subclasses of VariableTracker.

        This method should return the type represented by the instance of the subclass.
        The purpose is to provide a standardized way to retrieve the Python type information
        of the variable being tracked.

        Returns:
            type: The Python type (such as int, str, list, etc.) of the variable tracked by
                the subclass. If the type cannot be determined or is not relevant,
                leaving it undefined or invoking super() is always sound.

        Note:
            This is an abstract method and may be overridden in subclasses.

        Example:
            class SetVariable(VariableTracker):
                def python_type(self):
                    return set

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        # 抽象方法，应当在 VariableTracker 的子类中实现，返回变量追踪的 Python 类型
        raise NotImplementedError(f"{self} has no type")
    # 当对象表示一个常量时，将其转换为 Python 常量
    def as_python_constant(self):
        """For constants"""
        # 抛出未实现错误，显示对象不是常量
        raise NotImplementedError(f"{self} is not a constant")

    # 类似于 as_python_constant()，但在尝试强制对象变成常量时添加 ID_MATCH 保护
    def guard_as_python_constant(self):
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        try:
            # 调用 as_python_constant() 方法
            return self.as_python_constant()
        except NotImplementedError as e:
            # 如果未实现错误被抛出，则调用 unimplemented() 函数
            unimplemented(str(e))

    # 检查对象是否表示一个 Python 常量
    def is_python_constant(self):
        try:
            # 尝试调用 as_python_constant() 方法
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    # 创建一个保护对象，使其能够重新生成 Python 对象
    def make_guard(self, fn):
        if self.source:
            # 如果有源对象，则调用源对象的 make_guard() 方法
            return self.source.make_guard(fn)
        # 否则抛出未实现错误
        raise NotImplementedError

    # 获取对象的属性，并确保返回的是一个 Python 常量
    def const_getattr(self, tx, name: str) -> Any:
        """getattr(self, name) returning a python constant"""
        # 抛出未实现错误
        raise NotImplementedError

    # 获取对象的属性，并返回一个新的变量对象
    def var_getattr(self, tx, name: str) -> "VariableTracker":
        """getattr(self, name) returning a new variable"""
        # 获取属性值
        value = self.const_getattr(tx, name)
        # 如果值不是字面常量，则抛出未实现错误
        if not variables.ConstantVariable.is_literal(value):
            raise NotImplementedError
        # 如果有源对象，则创建一个带有源的常量变量对象
        source = None
        if self.source:
            source = AttrSource(self.source, name)
        return variables.ConstantVariable.create(value, source=source)

    # 检查对象是否是代理对象
    def is_proxy(self):
        try:
            # 尝试调用 as_proxy() 方法
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    # 将对象转换为代理对象
    def as_proxy(self):
        # 抛出未实现错误，显示对象不是代理对象
        raise NotImplementedError(str(self))

    # 尝试将对象转换为 Torch 的 FX 节点对象
    def maybe_fx_node(self):
        try:
            # 尝试调用 as_proxy() 方法
            proxy = self.as_proxy()
            import torch.fx

            # 如果代理对象是 Torch 的 Proxy 类型，则返回其节点对象
            if isinstance(proxy, torch.fx.Proxy):
                return proxy.node
            return None
        except NotImplementedError:
            return None

    # 重新生成对象的表示代码
    def reconstruct(self, codegen):
        # 抛出未实现错误
        raise NotImplementedError

    # 检查是否可以重新生成该对象所代表的 Python 对象
    def can_reconstruct(self, tx):
        """If it is possible to reconstruct the Python object this
        VariableTracker represents."""
        # 断言只有根事务可以重新生成
        assert tx is tx.output.root_tx, "Only root tx can reconstruct"
        try:
            # 尝试使用 PyCodegen 类来重新生成对象
            from ..codegen import PyCodegen

            cg = PyCodegen(tx)
            self.reconstruct(cg)
            return True
        except NotImplementedError:
            return False

    # 解包变量序列
    def unpack_var_sequence(self, tx) -> List["VariableTracker"]:
        # 抛出未实现错误
        raise NotImplementedError

    # 检查是否有解包的变量序列
    def has_unpack_var_sequence(self, tx) -> bool:
        try:
            # 尝试解包变量序列
            self.unpack_var_sequence(tx)
            return True
        except NotImplementedError:
            return False

    # 返回参数名称列表
    def inspect_parameter_names(self) -> List[str]:
        # 抛出未实现错误，显示调用了未实现的方法
        unimplemented(f"inspect_parameter_names: {self}")

    # 调用对象的 hasattr 方法
    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        # 抛出未实现错误，显示调用了未实现的方法
        unimplemented(f"hasattr {self.__class__.__name__} {name}")

    # 调用函数方法
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 当前方法声明，返回类型为 "VariableTracker"
        unimplemented(f"call_function {self} {args} {kwargs}")

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # 如果方法名为 "__len__" 并且 self 拥有可解包的变量序列
        if name == "__len__" and self.has_unpack_var_sequence(tx):
            assert not (args or kwargs)
            # 返回一个表示长度的常量变量
            return variables.ConstantVariable.create(len(self.unpack_var_sequence(tx)))
        elif (
            # 如果方法名为 "__getattr__" 并且只有一个参数，参数是 Python 常量，并且没有关键字参数
            name == "__getattr__"
            and len(args) == 1
            and args[0].is_python_constant()
            and not kwargs
        ):
            # 调用 var_getattr 方法处理获取属性的逻辑
            return self.var_getattr(tx, args[0].as_python_constant())
        # 对于未实现的其它方法调用，抛出未实现异常
        unimplemented(f"call_method {self} {name} {args} {kwargs}")

    def set_name_hint(self, name):
        # 设置名称提示，但是没有实际操作
        pass

    def realize(self) -> "VariableTracker":
        """Used by LazyVariableTracker to build the real VariableTracker"""
        # 用于 LazyVariableTracker 创建真实的 VariableTracker，返回自身
        return self

    def unwrap(self) -> "VariableTracker":
        """Used by LazyVariableTracker to return the real VariableTracker if it already exists"""
        # 用于 LazyVariableTracker 返回真实的 VariableTracker（如果已存在），返回自身
        return self

    def is_realized(self):
        """Used by LazyVariableTracker to indicate an unrealized node"""
        # 用于 LazyVariableTracker 表示一个未实现的节点，总是返回 True
        return True

    def next_variable(self, tx):
        # 未实现的方法调用，表示获取下一个变量，抛出未实现异常
        unimplemented(f"next({self})")

    def is_strict_mode(self, tx):
        # 检查是否为严格模式，依赖于 tx.strict_checks_fn 是否存在和其返回值
        return tx.strict_checks_fn and tx.strict_checks_fn(self)

    def __init__(
        self,
        *,
        source: Source = None,
        mutable_local: MutableLocal = None,
    ):
        # 初始化方法，接受两个可选参数：source 和 mutable_local
        super().__init__()
        # 设置实例变量 source 和 mutable_local
        self.source = source
        self.mutable_local = mutable_local
# 定义一个函数 typestr，用于返回参数 objs 的类型字符串表示
def typestr(*objs):
    # 检查参数个数是否为1
    if len(objs) == 1:
        # 解包 objs 中的唯一对象
        (obj,) = objs
        # 如果 obj 是 VariableTracker 类型的实例，则返回其字符串表示形式
        if isinstance(obj, VariableTracker):
            return str(obj)
        # 否则返回 obj 的类型名称
        else:
            return type(obj).__name__
    else:
        # 如果 objs 中有多个对象，则递归调用 typestr 函数，并用空格连接结果
        return " ".join(map(typestr, objs))
```