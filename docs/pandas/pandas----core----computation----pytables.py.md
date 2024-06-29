# `D:\src\scipysrc\pandas\pandas\core\computation\pytables.py`

```
"""manage PyTables query interface via Expressions"""

# 导入未来的注释语法以支持类型提示
from __future__ import annotations

import ast  # 导入抽象语法树模块，用于处理Python代码的语法分析
from decimal import (  # 导入decimal模块，支持高精度算术运算
    Decimal,  # Decimal类型，用于精确表示十进制数
    InvalidOperation,  # 无效操作异常类
)
from functools import partial  # 导入偏函数模块，用于部分应用函数
from typing import (  # 导入类型提示模块
    TYPE_CHECKING,  # 类型检查标志
    Any,  # 任意类型
    ClassVar,  # 类变量类型
)

import numpy as np  # 导入NumPy库，用于科学计算

from pandas._libs.tslibs import (  # 导入Pandas时间序列相关模块
    Timedelta,  # 时间增量类型
    Timestamp,  # 时间戳类型
)
from pandas.errors import UndefinedVariableError  # 导入Pandas错误处理模块中的未定义变量错误

from pandas.core.dtypes.common import is_list_like  # 导入Pandas中用于判断是否为列表样式数据的函数

import pandas.core.common as com  # 导入Pandas核心共用模块
from pandas.core.computation import (  # 导入Pandas计算模块
    expr,  # 表达式处理函数
    ops,  # 操作函数
    scope as _scope,  # 作用域类并重命名为_scope
)
from pandas.core.computation.common import ensure_decoded  # 导入确保解码的函数
from pandas.core.computation.expr import BaseExprVisitor  # 导入表达式访问者基类
from pandas.core.computation.ops import is_term  # 导入用于判断是否为项的函数
from pandas.core.construction import extract_array  # 导入从结构中提取数组的函数
from pandas.core.indexes.base import Index  # 导入Pandas索引基类

from pandas.io.formats.printing import (  # 导入Pandas打印格式模块
    pprint_thing,  # 打印通用对象
    pprint_thing_encoded,  # 打印编码后的对象
)

if TYPE_CHECKING:
    from pandas._typing import (  # 如果是类型检查，则导入相关类型
        Self,  # 类型自身
        npt,  # NumPy类型
    )


class PyTablesScope(_scope.Scope):
    """PyTablesScope类继承自_scope.Scope，表示PyTables查询作用域。"""

    __slots__ = ("queryables",)  # 限定只能有queryables属性

    queryables: dict[str, Any]  # queryables属性是一个字符串到任意类型值的字典

    def __init__(
        self,
        level: int,
        global_dict=None,
        local_dict=None,
        queryables: dict[str, Any] | None = None,
    ) -> None:
        """初始化PyTablesScope对象。

        Args:
            level (int): 作用域的层级
            global_dict (dict, optional): 全局字典，默认为None
            local_dict (dict, optional): 本地字典，默认为None
            queryables (dict[str, Any] | None, optional): 可查询对象的字典，默认为None
        """
        super().__init__(level + 1, global_dict=global_dict, local_dict=local_dict)
        self.queryables = queryables or {}


class Term(ops.Term):
    """Term类继承自ops.Term，表示表达式中的术语。"""

    env: PyTablesScope  # env属性是PyTablesScope类型

    def __new__(cls, name, env, side=None, encoding=None):
        """创建新的Term对象。

        Args:
            name (str): 术语的名称
            env (PyTablesScope): PyTablesScope环境
            side (str, optional): 侧面信息，默认为None
            encoding (str, optional): 编码方式，默认为None

        Returns:
            object: 返回新创建的对象
        """
        if isinstance(name, str):
            klass = cls
        else:
            klass = Constant
        return object.__new__(klass)

    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        """初始化Term对象。

        Args:
            name (str): 术语的名称
            env (PyTablesScope): PyTablesScope环境
            side (str, optional): 侧面信息，默认为None
            encoding (str, optional): 编码方式，默认为None
        """
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        """解析术语的名称。

        Returns:
            str: 返回解析后的名称
        """
        # 必须是可查询的
        if self.side == "left":
            # 注意：__new__方法的行为保证了这里的self.name是一个字符串
            if self.name not in self.env.queryables:
                raise NameError(f"name {self.name!r} is not defined")
            return self.name

        # 解析右侧表达式（允许它为None）
        try:
            return self.env.resolve(self.name, is_local=False)
        except UndefinedVariableError:
            return self.name

    # 只读属性覆盖读/写属性
    @property  # type: ignore[misc]
    def value(self):
        """值属性，返回术语的值。"""
        return self._value


class Constant(Term):
    """Constant类继承自Term，表示常量术语。"""

    def __init__(self, name, env: PyTablesScope, side=None, encoding=None) -> None:
        """初始化Constant对象。

        Args:
            name (str): 术语的名称
            env (PyTablesScope): PyTablesScope环境
            side (str, optional): 侧面信息，默认为None
            encoding (str, optional): 编码方式，默认为None
        """
        assert isinstance(env, PyTablesScope), type(env)
        super().__init__(name, env, side=side, encoding=encoding)

    def _resolve_name(self):
        """解析术语的名称。

        Returns:
            str: 返回解析后的名称
        """
        return self._name


class BinOp(ops.BinOp):
    """BinOp类继承自ops.BinOp，表示二元操作。"""

    _max_selectors = 31  # 最大选择器数量

    op: str  # 操作符字符串
    queryables: dict[str, Any]  # queryables属性是一个字符串到任意类型值的字典
    condition: str | None  # 条件字符串或None
    # 初始化函数，接受操作符、左操作数、右操作数、可查询对象字典和编码方式作为参数
    # 调用父类的初始化方法，传递操作符、左右操作数
    def __init__(self, op: str, lhs, rhs, queryables: dict[str, Any], encoding) -> None:
        super().__init__(op, lhs, rhs)
        # 将传入的可查询对象字典和编码方式保存到实例变量中
        self.queryables = queryables
        self.encoding = encoding
        # 初始化条件为None
        self.condition = None

    # 禁止只有标量的布尔运算
    def _disallow_scalar_only_bool_ops(self) -> None:
        pass  # 该方法暂未实现，留空

    # 修剪方法，返回经过精简处理后的结果
    def prune(self, klass):
        # 定义内部函数pr，用于创建并返回从当前实例生成的新特定二元操作
        def pr(left, right):
            """create and return a new specialized BinOp from myself"""
            # 如果左操作数为空，则返回右操作数
            if left is None:
                return right
            # 如果右操作数为空，则返回左操作数
            elif right is None:
                return left

            # 默认使用传入的类
            k = klass
            # 如果左操作数是ConditionBinOp的实例
            if isinstance(left, ConditionBinOp):
                # 如果右操作数也是ConditionBinOp的实例，则使用JointConditionBinOp
                if isinstance(right, ConditionBinOp):
                    k = JointConditionBinOp
                # 如果左操作数是传入的类的实例，则返回左操作数
                elif isinstance(left, k):
                    return left
                # 如果右操作数是传入的类的实例，则返回右操作数
                elif isinstance(right, k):
                    return right

            # 如果左操作数是FilterBinOp的实例
            elif isinstance(left, FilterBinOp):
                # 如果右操作数也是FilterBinOp的实例，则使用JointFilterBinOp
                if isinstance(right, FilterBinOp):
                    k = JointFilterBinOp
                # 如果左操作数是传入的类的实例，则返回左操作数
                elif isinstance(left, k):
                    return left
                # 如果右操作数是传入的类的实例，则返回右操作数
                elif isinstance(right, k):
                    return right

            # 返回用传入的类生成的新操作，并计算其结果
            return k(
                self.op, left, right, queryables=self.queryables, encoding=self.encoding
            ).evaluate()

        # 获取左右操作数
        left, right = self.lhs, self.rhs

        # 如果左右操作数都是术语（term）
        if is_term(left) and is_term(right):
            # 调用pr函数，传入左右操作数的值，得到结果
            res = pr(left.value, right.value)
        # 如果左操作数不是术语而右操作数是术语
        elif not is_term(left) and is_term(right):
            # 调用pr函数，传入左操作数修剪后的结果和右操作数的值，得到结果
            res = pr(left.prune(klass), right.value)
        # 如果左操作数是术语而右操作数不是术语
        elif is_term(left) and not is_term(right):
            # 调用pr函数，传入左操作数的值和右操作数修剪后的结果，得到结果
            res = pr(left.value, right.prune(klass))
        # 如果左右操作数都不是术语
        elif not (is_term(left) or is_term(right)):
            # 调用pr函数，传入左右操作数修剪后的结果，得到结果
            res = pr(left.prune(klass), right.prune(klass))

        # 返回最终结果
        return res

    # 将rhs适配为符合要求的类型
    def conform(self, rhs):
        """inplace conform rhs"""
        # 如果rhs不是列表样式的对象，则转换为列表
        if not is_list_like(rhs):
            rhs = [rhs]
        # 如果rhs是numpy数组，则将其扁平化处理
        if isinstance(rhs, np.ndarray):
            rhs = rhs.ravel()
        # 返回适配后的rhs
        return rhs

    # 返回当前字段是否有效的布尔值
    @property
    def is_valid(self) -> bool:
        """return True if this is a valid field"""
        # 检查当前字段是否存在于可查询对象字典中
        return self.lhs in self.queryables

    # 返回当前字段是否为表中有效的列名
    @property
    def is_in_table(self) -> bool:
        """
        return True if this is a valid column name for generation (e.g. an
        actual column in the table)
        """
        # 检查当前字段在可查询对象字典中的值是否不为None
        return self.queryables.get(self.lhs) is not None

    # 返回当前字段的类型
    @property
    def kind(self):
        """the kind of my field"""
        # 返回当前字段在可查询对象字典中关联对象的kind属性，若不存在则返回None
        return getattr(self.queryables.get(self.lhs), "kind", None)

    # 返回当前字段的meta信息
    @property
    def meta(self):
        """the meta of my field"""
        # 返回当前字段在可查询对象字典中关联对象的meta属性，若不存在则返回None
        return getattr(self.queryables.get(self.lhs), "meta", None)

    # 返回当前字段的metadata信息
    @property
    def metadata(self):
        """the metadata of my field"""
        # 返回当前字段在可查询对象字典中关联对象的metadata属性，若不存在则返回None
        return getattr(self.queryables.get(self.lhs), "metadata", None)

    # 根据给定的值v生成并返回操作字符串
    def generate(self, v) -> str:
        """create and return the op string for this TermValue"""
        # 将给定的值v转换为特定编码的字符串
        val = v.tostring(self.encoding)
        # 返回生成的操作字符串，形式为"(lhs op val)"
        return f"({self.lhs} {self.op} {val})"
    def convert_value(self, v) -> TermValue:
        """
        convert the expression that is in the term to something that is
        accepted by pytables
        """

        # 将值转换为字符串形式，根据编码进行编码输出
        def stringify(value):
            if self.encoding is not None:
                return pprint_thing_encoded(value, encoding=self.encoding)
            return pprint_thing(value)

        # 确保种类名称已解码
        kind = ensure_decoded(self.kind)
        # 确保元信息已解码
        meta = ensure_decoded(self.meta)

        # 处理 datetime 和 datetime64 类型
        if kind == "datetime" or (kind and kind.startswith("datetime64")):
            if isinstance(v, (int, float)):
                v = stringify(v)
            v = ensure_decoded(v)
            # 转换为 Timestamp 对象，并设置单位为纳秒
            v = Timestamp(v).as_unit("ns")
            if v.tz is not None:
                v = v.tz_convert("UTC")
            return TermValue(v, v._value, kind)

        # 处理 timedelta64 和 timedelta 类型
        elif kind in ("timedelta64", "timedelta"):
            if isinstance(v, str):
                v = Timedelta(v)
            else:
                v = Timedelta(v, unit="s")
            v = v.as_unit("ns")._value
            return TermValue(int(v), v, kind)

        # 处理分类数据
        elif meta == "category":
            # 提取 metadata 数组，用于后续搜索和比较
            metadata = extract_array(self.metadata, extract_numpy=True)
            result: npt.NDArray[np.intp] | np.intp | int
            if v not in metadata:
                result = -1
            else:
                result = metadata.searchsorted(v, side="left")
            return TermValue(result, result, "integer")

        # 处理整数类型
        elif kind == "integer":
            try:
                v_dec = Decimal(v)
            except InvalidOperation:
                # 当无法转换为 Decimal 时，尝试转换为浮点数以引发 ValueError
                float(v)
            else:
                # 将 Decimal 转换为最接近的整数
                v = int(v_dec.to_integral_exact(rounding="ROUND_HALF_EVEN"))
            return TermValue(v, v, kind)

        # 处理浮点数类型
        elif kind == "float":
            v = float(v)
            return TermValue(v, v, kind)

        # 处理布尔类型
        elif kind == "bool":
            if isinstance(v, str):
                # 将字符串转换为布尔值
                v = v.strip().lower() not in [
                    "false",
                    "f",
                    "no",
                    "n",
                    "none",
                    "0",
                    "[]",
                    "{}",
                    "",
                ]
            else:
                v = bool(v)
            return TermValue(v, v, kind)

        # 处理字符串类型
        elif isinstance(v, str):
            # 对字符串进行引用处理
            return TermValue(v, stringify(v), "string")

        # 若未匹配到任何处理类型，则抛出类型错误异常
        else:
            raise TypeError(f"Cannot compare {v} of type {type(v)} to {kind} column")
class FilterBinOp(BinOp):
    # 定义了一个名为 FilterBinOp 的类，继承自 BinOp 类

    filter: tuple[Any, Any, Index] | None = None
    # 类变量 filter，用于存储过滤条件的元组或者空值

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式
        if self.filter is None:
            return "Filter: Not Initialized"
            # 如果 filter 为空，则返回未初始化的提示字符串
        return pprint_thing(f"[Filter : [{self.filter[0]}] -> [{self.filter[1]}]")
        # 否则返回格式化后的包含过滤条件的字符串表示

    def invert(self) -> Self:
        """invert the filter"""
        # 反转过滤条件的方法
        if self.filter is not None:
            self.filter = (
                self.filter[0],
                self.generate_filter_op(invert=True),
                self.filter[2],
            )
            # 如果 filter 不为空，则将其反转并更新
        return self
        # 返回当前对象的引用

    def format(self):
        """return the actual filter format"""
        # 返回实际过滤条件的格式方法
        return [self.filter]
        # 返回存储在 filter 中的过滤条件的列表形式

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        # 评估方法，返回自身或空值
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")
            # 如果当前查询条件无效，则抛出值错误

        rhs = self.conform(self.rhs)
        # 将右手边的值进行格式化（或调整）以适应条件

        values = list(rhs)
        # 将右手边的值转换为列表形式

        if self.is_in_table:
            # 如果条件在表中存在
            if self.op in ["==", "!="] and len(values) > self._max_selectors:
                # 如果操作符为 == 或 !=，并且值的数量超过最大选择器的限制
                filter_op = self.generate_filter_op()
                # 生成过滤操作符
                self.filter = (self.lhs, filter_op, Index(values))
                # 将左手边、过滤操作符和值的索引组成的元组存储在 filter 中

                return self
                # 返回当前对象
            return None
            # 否则返回空值

        # 对等条件
        if self.op in ["==", "!="]:
            filter_op = self.generate_filter_op()
            # 生成过滤操作符
            self.filter = (self.lhs, filter_op, Index(values))
            # 将左手边、过滤操作符和值的索引组成的元组存储在 filter 中

        else:
            raise TypeError(
                f"passing a filterable condition to a non-table indexer [{self}]"
            )
            # 否则抛出类型错误，传递了一个可过滤条件给非表索引器

        return self
        # 返回当前对象

    def generate_filter_op(self, invert: bool = False):
        # 生成过滤操作符的方法，可以选择是否反转
        if (self.op == "!=" and not invert) or (self.op == "==" and invert):
            return lambda axis, vals: ~axis.isin(vals)
            # 如果操作符为 != 且不反转，或者操作符为 == 且反转，则返回一个 lambda 函数
        else:
            return lambda axis, vals: axis.isin(vals)
            # 否则返回另一个 lambda 函数


class JointFilterBinOp(FilterBinOp):
    # 继承自 FilterBinOp 类的 JointFilterBinOp 类

    def format(self):
        # 格式化方法
        raise NotImplementedError("unable to collapse Joint Filters")
        # 抛出未实现错误，无法折叠联合过滤器

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self:  # type: ignore[override]
        # 评估方法，返回自身
        return self
        # 返回当前对象


class ConditionBinOp(BinOp):
    # 继承自 BinOp 类的 ConditionBinOp 类

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式
        return pprint_thing(f"[Condition : [{self.condition}]]")
        # 返回包含条件的格式化字符串表示

    def invert(self):
        # 反转条件的方法
        # if self.condition is not None:
        #    self.condition = "~(%s)" % self.condition
        # return self
        raise NotImplementedError(
            "cannot use an invert condition when passing to numexpr"
        )
        # 抛出未实现错误，当传递给 numexpr 时无法使用反转条件

    def format(self):
        # 格式化方法
        """return the actual ne format"""
        return self.condition
        # 返回实际的条件格式

    # error: Signature of "evaluate" incompatible with supertype "BinOp"
    def evaluate(self) -> Self | None:  # type: ignore[override]
        # 如果查询条件无效，则抛出值错误异常
        if not self.is_valid:
            raise ValueError(f"query term is not valid [{self}]")

        # 如果不在表格中，返回 None
        if not self.is_in_table:
            return None

        # 将右手边的值转换为规范形式
        rhs = self.conform(self.rhs)
        
        # 将每个值转换为适当的类型
        values = [self.convert_value(v) for v in rhs]

        # 处理等式条件
        if self.op in ["==", "!="]:
            # 如果值的数量不超过最大选择器数量，则创建表达式
            if len(values) <= self._max_selectors:
                # 生成每个值的表达式
                vs = [self.generate(v) for v in values]
                # 将表达式用括号和逻辑或连接起来
                self.condition = f"({' | '.join(vs)})"

            # 如果值的数量超过最大选择器数量，则返回 None
            else:
                return None
        else:
            # 对于其他操作符，直接生成单个值的条件
            self.condition = self.generate(values[0])

        # 返回自身对象
        return self
class JointConditionBinOp(ConditionBinOp):
    # 继承自 ConditionBinOp 的联合条件操作类

    def evaluate(self) -> Self:  # type: ignore[override]
        # 评估操作：构建条件字符串并返回自身对象
        self.condition = f"({self.lhs.condition} {self.op} {self.rhs.condition})"
        return self


class UnaryOp(ops.UnaryOp):
    # 继承自 ops.UnaryOp 的一元操作类

    def prune(self, klass):
        # 对操作进行修剪

        if self.op != "~":
            # 如果操作不是按位取反，抛出未实现的错误
            raise NotImplementedError("UnaryOp only support invert type ops")

        operand = self.operand
        operand = operand.prune(klass)

        if operand is not None and (
            issubclass(klass, ConditionBinOp)
            and operand.condition is not None
            or not issubclass(klass, ConditionBinOp)
            and issubclass(klass, FilterBinOp)
            and operand.filter is not None
        ):
            # 如果操作数不为空，并且符合一定的条件类别，执行反转操作
            return operand.invert()
        return None


class PyTablesExprVisitor(BaseExprVisitor):
    # 继承自 BaseExprVisitor 的 PyTables 表达式访问者类
    const_type: ClassVar[type[ops.Term]] = Constant
    term_type: ClassVar[type[Term]] = Term

    def __init__(self, env, engine, parser, **kwargs) -> None:
        # 初始化方法

        super().__init__(env, engine, parser)

        for bin_op in self.binary_ops:
            bin_node = self.binary_op_nodes_map[bin_op]
            setattr(
                self,
                f"visit_{bin_node}",
                lambda node, bin_op=bin_op: partial(BinOp, bin_op, **kwargs),
            )
            # 动态设置访问二元操作节点的方法

    def visit_UnaryOp(self, node, **kwargs) -> ops.Term | UnaryOp | None:
        # 访问一元操作节点的方法

        if isinstance(node.op, (ast.Not, ast.Invert)):
            # 如果操作是逻辑非或按位取反，返回对应的 UnaryOp 对象
            return UnaryOp("~", self.visit(node.operand))
        elif isinstance(node.op, ast.USub):
            # 如果操作是一元负号，返回负数的 Constant 对象
            return self.const_type(-self.visit(node.operand).value, self.env)
        elif isinstance(node.op, ast.UAdd):
            # 一元加法操作未支持，抛出未实现的错误
            raise NotImplementedError("Unary addition not supported")
        # TODO: return None might never be reached
        return None

    def visit_Index(self, node, **kwargs):
        # 访问索引节点的方法

        return self.visit(node.value).value

    def visit_Assign(self, node, **kwargs):
        # 访问赋值节点的方法

        cmpr = ast.Compare(
            ops=[ast.Eq()], left=node.targets[0], comparators=[node.value]
        )
        return self.visit(cmpr)

    def visit_Subscript(self, node, **kwargs) -> ops.Term:
        # 访问下标操作节点的方法，返回 Term 对象

        # 只允许简单的下标操作

        value = self.visit(node.value)
        slobj = self.visit(node.slice)
        try:
            value = value.value
        except AttributeError:
            pass

        if isinstance(slobj, Term):
            # 在 Python 3.9 中，使用包含整数的 Term 查找 np.ndarray 会抛出异常
            slobj = slobj.value

        try:
            return self.const_type(value[slobj], self.env)
        except TypeError as err:
            raise ValueError(f"cannot subscript {value!r} with {slobj!r}") from err
    # 处理访问对象属性的情况
    def visit_Attribute(self, node, **kwargs):
        # 获取属性名和属性值
        attr = node.attr
        value = node.value

        # 获取节点上下文类型
        ctx = type(node.ctx)
        # 如果上下文为加载（ast.Load），则解析值
        if ctx == ast.Load:
            # 解析属性值
            resolved = self.visit(value)

            # 尝试获取属性值，以确定是否为另一个表达式
            try:
                resolved = resolved.value
            except AttributeError:
                pass

            # 尝试返回属性值，如果无法获取则抛出异常
            try:
                return self.term_type(getattr(resolved, attr), self.env)
            except AttributeError:
                # 处理类似 datetime.datetime 这样作用域被重写的情况
                if isinstance(value, ast.Name) and value.id == attr:
                    return resolved

        # 如果上下文不是加载，则抛出值错误异常
        raise ValueError(f"Invalid Attribute context {ctx.__name__}")

    # 将 In 操作符翻译成相应的比较操作符（例如 ast.Eq()）
    def translate_In(self, op):
        return ast.Eq() if isinstance(op, ast.In) else op

    # 重写成员操作符（membership operator）的方法，访问操作符和操作数
    def _rewrite_membership_op(self, node, left, right):
        return self.visit(node.op), node.op, left, right
def _validate_where(w):
    """
    Validate that the where statement is of the right type.

    The type may either be String, Expr, or list-like of Exprs.

    Parameters
    ----------
    w : String term expression, Expr, or list-like of Exprs.

    Returns
    -------
    where : The original where clause if the check was successful.

    Raises
    ------
    TypeError : An invalid data type was passed in for w (e.g. dict).
    """
    # 检查传入的 where 参数是否是有效的类型
    if not (isinstance(w, (PyTablesExpr, str)) or is_list_like(w)):
        # 如果不是 PyTablesExpr、str 或者类似列表中的一种，则引发类型错误
        raise TypeError(
            "where must be passed as a string, PyTablesExpr, "
            "or list-like of PyTablesExpr"
        )

    # 返回原始的 where 子句，如果检查成功
    return w


class PyTablesExpr(expr.Expr):
    """
    Hold a pytables-like expression, comprised of possibly multiple 'terms'.

    Parameters
    ----------
    where : string term expression, PyTablesExpr, or list-like of PyTablesExprs
    queryables : a "kinds" map (dict of column name -> kind), or None if column
        is non-indexable
    encoding : an encoding that will encode the query terms

    Returns
    -------
    a PyTablesExpr object

    Examples
    --------
    'index>=date'
    "columns=['A', 'D']"
    'columns=A'
    'columns==A'
    "~(columns=['A','B'])"
    'index>df.index[3] & string="bar"'
    '(index>df.index[3] & index<=df.index[6]) | string="bar"'
    "ts>=Timestamp('2012-02-01')"
    "major_axis>=20130101"
    """

    _visitor: PyTablesExprVisitor | None
    env: PyTablesScope
    expr: str

    def __init__(
        self,
        where,
        queryables: dict[str, Any] | None = None,
        encoding=None,
        scope_level: int = 0,
    ):
        # 初始化 PyTablesExpr 对象，接收 where 表达式、可查询项、编码方式和作用域级别参数
        pass
    ) -> None:
        where = _validate_where(where)  # 调用 _validate_where 函数验证并返回 where 参数

        self.encoding = encoding  # 设置对象的编码属性
        self.condition = None  # 初始化条件属性为 None
        self.filter = None  # 初始化过滤器属性为 None
        self.terms = None  # 初始化术语属性为 None
        self._visitor = None  # 初始化访问者属性为 None

        # 如果 where 参数是 PyTablesExpr 类型，则捕获其环境的本地字典
        local_dict: _scope.DeepChainMap[Any, Any] | None = None

        if isinstance(where, PyTablesExpr):
            local_dict = where.env.scope  # 将 where 的环境字典赋给 local_dict
            _where = where.expr  # 将 where 的表达式赋给 _where

        elif is_list_like(where):
            where = list(where)  # 将 where 转换为列表形式
            for idx, w in enumerate(where):
                if isinstance(w, PyTablesExpr):
                    local_dict = w.env.scope  # 如果列表项是 PyTablesExpr 类型，则赋给 local_dict
                else:
                    where[idx] = _validate_where(w)  # 否则验证列表项并替换为验证后的值
            _where = " & ".join([f"({w})" for w in com.flatten(where)])  # 将列表中的表达式用 "&" 连接起来形成新的 _where 字符串
        else:
            # _validate_where 确保此处 where 是字符串类型
            _where = where  # 将 where 直接赋给 _where

        self.expr = _where  # 将最终确定的 _where 赋给对象的表达式属性
        self.env = PyTablesScope(scope_level + 1, local_dict=local_dict)  # 创建 PyTablesScope 环境对象，并设置环境字典

        if queryables is not None and isinstance(self.expr, str):
            self.env.queryables.update(queryables)  # 更新环境中的 queryables 属性
            self._visitor = PyTablesExprVisitor(
                self.env,
                queryables=queryables,
                parser="pytables",
                engine="pytables",
                encoding=encoding,
            )  # 创建 PyTablesExprVisitor 访问者对象，并设置相关属性
            self.terms = self.parse()  # 解析表达式并将结果赋给对象的 terms 属性

    def __repr__(self) -> str:
        if self.terms is not None:
            return pprint_thing(self.terms)  # 如果 terms 属性不为 None，则返回 terms 的可打印字符串表示
        return pprint_thing(self.expr)  # 否则返回 expr 的可打印字符串表示

    def evaluate(self):
        """create and return the numexpr condition and filter"""
        try:
            self.condition = self.terms.prune(ConditionBinOp)  # 尝试使用 terms 的 prune 方法生成条件对象
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid condition"
            ) from err
        try:
            self.filter = self.terms.prune(FilterBinOp)  # 尝试使用 terms 的 prune 方法生成过滤器对象
        except AttributeError as err:
            raise ValueError(
                f"cannot process expression [{self.expr}], [{self}] "
                "is not a valid filter"
            ) from err

        return self.condition, self.filter  # 返回生成的条件和过滤器对象
class TermValue:
    """hold a term value the we use to construct a condition/filter"""

    def __init__(self, value, converted, kind: str) -> None:
        """Initialize TermValue object with value, converted value, and kind."""
        assert isinstance(kind, str), kind
        self.value = value          # Assign the original value
        self.converted = converted  # Assign the converted value
        self.kind = kind            # Assign the type of value (string, float, etc.)

    def tostring(self, encoding) -> str:
        """Return a string representation of the value based on its type."""
        if self.kind == "string":
            if encoding is not None:
                return str(self.converted)   # Return the converted string as-is
            return f'"{self.converted}"'     # Return the converted string within quotes
        elif self.kind == "float":
            # For float values, ensure repr() is used due to potential precision issues
            return repr(self.converted)     # Return the float representation using repr()
        return str(self.converted)           # Return the converted value as a string


def maybe_expression(s) -> bool:
    """loose checking if s is a pytables-acceptable expression"""
    if not isinstance(s, str):
        return False   # Return False if s is not a string

    operations = PyTablesExprVisitor.binary_ops + PyTablesExprVisitor.unary_ops + ("=",)

    # Check if any of the acceptable operations are present in the string s
    return any(op in s for op in operations)
```