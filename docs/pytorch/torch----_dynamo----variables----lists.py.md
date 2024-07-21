# `.\pytorch\torch\_dynamo\variables\lists.py`

```
# mypy: ignore-errors
# 导入必要的模块和函数
import collections  # 导入 collections 模块，用于处理各种集合类型
import functools  # 导入 functools 模块，用于高阶函数的操作
import inspect  # 导入 inspect 模块，用于解析源代码的工具
import operator  # 导入 operator 模块，提供了一组对应Python内部操作的函数
import types  # 导入 types 模块，定义了Python中的类型对象
from typing import Dict, List, Optional  # 导入类型提示相关的声明

import torch  # 导入 PyTorch 库
import torch.fx  # 导入 PyTorch FX 模块，用于函数级别的模型表示

from ..._guards import Source  # 导入 Source 类型定义

from .. import polyfill, variables  # 导入当前包的 polyfill 和 variables 模块
from ..bytecode_transformation import create_call_function, create_instruction  # 导入字节码转换相关的函数
from ..exc import unimplemented  # 导入未实现异常相关的定义
from ..source import AttrSource, GetItemSource  # 导入属性来源和索引来源相关的定义
from ..utils import (  # 导入一系列实用工具函数
    get_fake_value,
    guard_if_dyn,
    is_namedtuple,
    istype,
    iter_contains,
    Lit,
    namedtuple_fields,
    odict_values,
    set_example_value,
)
from .base import MutableLocal, VariableTracker  # 导入基类 MutableLocal 和 VariableTracker
from .constant import ConstantVariable  # 导入常量变量类
from .functions import UserFunctionVariable, UserMethodVariable  # 导入用户自定义函数和方法变量类


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for_instance(obj):
        # 根据对象类型判断返回相应的变量类
        if is_namedtuple(obj):
            return functools.partial(NamedTupleVariable, tuple_cls=type(obj))
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj):
        # 根据对象类型返回相应的变量类
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
            odict_values: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
            collections.deque: DequeVariable,
        }[obj]

    def __init__(
        self,
        items: List[VariableTracker],  # 列表变量，每个元素是 VariableTracker 类型
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(items, list)  # 断言 items 是列表类型
        assert all(isinstance(x, VariableTracker) for x in items)  # 断言 items 列表中所有元素均为 VariableTracker 类型
        self.items: List[VariableTracker] = items  # 初始化实例变量 items

    def _as_proxy(self):
        # 返回代理对象列表，每个元素是对应 items 列表元素的代理对象
        return [x.as_proxy() for x in self.items]

    def modified(self, items, **kwargs):
        # 返回一个修改后的实例，传入新的 items 列表
        return type(self)(items, **kwargs)

    @property
    def value(self):
        # 返回当前对象的 Python 常量表示
        return self.as_python_constant()

    def debug_repr_helper(self, prefix, suffix):
        # 返回调试时的字符串表示，前缀和后缀包裹 items 列表元素的调试表示
        return prefix + ", ".join(i.debug_repr() for i in self.items) + suffix

    def as_python_constant(self):
        # 返回当前对象的 Python 常量表示，转换每个元素为 Python 常量
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        # 断言当前对象不是 SizeVariable 类型，返回对象的代理表示
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())
    # 获取常量项的方法，根据变量追踪器的类型确定索引值
    def getitem_const(self, arg: VariableTracker):
        # 导入 SymNodeVariable 类
        from .tensor import SymNodeVariable

        # 如果参数是 SymNodeVariable 类型，则使用其符号编号作为索引
        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            # 否则将参数转换为 Python 常量作为索引
            index = arg.as_python_constant()

        # 如果索引是切片类型
        if isinstance(index, slice):
            # 如果存在原始数据源，返回一个克隆对象，只包含切片后的项和更新的源
            if self.source is not None:
                return self.clone(
                    items=self.items[index],
                    source=GetItemSource(self.source, index),
                    mutable_local=MutableLocal() if self.mutable_local else None,
                )
            else:
                # 否则返回一个克隆对象，只包含切片后的项
                return self.clone(
                    items=self.items[index],
                    mutable_local=MutableLocal() if self.mutable_local else None,
                )
        else:
            # 否则索引应为整数或 torch.SymInt 类型，直接返回索引处的项
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    # 将列表项解包为列表
    def unpack_var_sequence(self, tx):
        return list(self.items)

    # 调用对象的方法
    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        # 如果方法名是 "__getitem__"
        if name == "__getitem__":
            # 导入 TensorVariable 类
            from .tensor import TensorVariable

            # 断言参数没有关键字参数，且只有一个位置参数
            assert not kwargs and len(args) == 1
            # 如果参数是 TensorVariable 类型
            if isinstance(args[0], TensorVariable):
                # 获取代理节点的假值
                value = get_fake_value(args[0].as_proxy().node, tx)
                # 如果假值是常量且只包含一个元素，创建一个常量变量
                if value.constant is not None and value.constant.numel() == 1:
                    value = variables.ConstantVariable.create(value.constant.item())
                else:
                    # 否则抛出未实现的异常
                    unimplemented("__getitem__ with non-constant tensor")
            else:
                # 否则直接使用参数作为值
                value = args[0]
            # 调用 getitem_const 方法获取结果
            return self.getitem_const(value)
        # 如果方法名是 "__contains__"
        elif name == "__contains__":
            # 断言只有一个位置参数且没有关键字参数
            assert len(args) == 1
            assert not kwargs
            # 调用 iter_contains 函数检查对象中是否包含指定元素
            return iter_contains(self.unpack_var_sequence(tx), args[0], tx)
        # 如果方法名是 "index"
        elif name == "index":
            # 导入 SourcelessBuilder 类
            from .builder import SourcelessBuilder

            # 使用 tx.inline_user_function_return 方法内联用户函数返回值
            return tx.inline_user_function_return(
                SourcelessBuilder.create(tx, polyfill.index),
                [self] + list(args),
                kwargs,
            )

        # 否则调用父类的 call_method 方法处理
        return super().call_method(tx, name, args, kwargs)

    # 静态方法：比较列表项
    @staticmethod
    def list_compare(tx, op, left, right):
        return variables.UserFunctionVariable(polyfill.list_cmp).call_function(
            tx, [variables.BuiltinVariable(op), left, right], {}
        )
class RangeVariable(BaseListVariable):
    # 定义 RangeVariable 类，继承自 BaseListVariable 类

    def __init__(self, items, **kwargs):
        # RangeVariable 类的初始化方法，接受 items 和其他关键字参数

        items_to_map = items
        # 将传入的 items 赋值给 items_to_map

        start = variables.ConstantVariable.create(0)
        # 创建一个初始值为 0 的常量变量 start

        stop = None
        # 初始化 stop 变量为 None

        step = variables.ConstantVariable.create(1)
        # 创建一个初始值为 1 的常量变量 step

        if len(items_to_map) == 1:
            (stop,) = items_to_map
            # 如果 items_to_map 的长度为 1，则将其解包赋值给 stop
        elif len(items_to_map) == 2:
            start, stop = items_to_map
            # 如果 items_to_map 的长度为 2，则将其解包分别赋值给 start 和 stop
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
            # 如果 items_to_map 的长度为 3，则将其解包分别赋值给 start、stop 和 step
        else:
            raise AssertionError
            # 如果 items_to_map 的长度不在 1 到 3 之间，抛出断言错误

        assert stop is not None
        # 断言 stop 不为 None

        super().__init__([start, stop, step], **kwargs)
        # 调用父类 BaseListVariable 的初始化方法，传入包含 start、stop、step 的列表和其他关键字参数

    def debug_repr(self):
        # 返回用于调试的字符串表示，格式为 "range(...)"
        return self.debug_repr_helper("range(", ")")

    def python_type(self):
        # 返回 Python 类型 range
        return range

    def start(self):
        # 返回起始值 start 的 Python 常量表示
        return self.items[0].as_python_constant()

    def stop(self):
        # 返回终止值 stop 的 Python 常量表示
        return self.items[1].as_python_constant()

    def step(self):
        # 返回步长 step 的 Python 常量表示
        return self.items[2].as_python_constant()

    def range_length(self):
        # 计算 Range 变量的长度

        lo = self.start()
        # 获取起始值 lo

        hi = self.stop()
        # 获取终止值 hi

        step = self.step()
        # 获取步长 step

        assert step != 0
        # 断言步长不为 0

        if step > 0 and lo < hi:
            return 1 + (hi - 1 - lo) // step
            # 如果步长大于 0 且起始值小于终止值，则返回计算出的长度
        elif step < 0 and lo > hi:
            return 1 + (lo - 1 - hi) // (0 - step)
            # 如果步长小于 0 且起始值大于终止值，则返回计算出的长度
        else:
            return 0
            # 否则返回长度为 0

    def _get_slice_indices(self, length, slice):
        # 获取切片的索引范围

        step_is_negative = 0
        # 初始化步长是否为负数的标志

        if slice.step is None:
            step = 1
            step_is_negative = False
            # 如果切片的步长为 None，则默认步长为 1，步长为负数标志为 False
        else:
            step = slice.step
            step_is_negative = slice.step < 0
            # 否则使用切片的步长，并根据步长判断是否为负数

        # 计算开始和结束的下标范围
        if step_is_negative:
            lower = -1
            upper = length + lower
            # 如果步长为负数，则 lower 为 -1，upper 为 length - 1
        else:
            lower = 0
            upper = length
            # 否则 lower 为 0，upper 为 length

        # 计算开始位置
        if slice.start is None:
            start = upper if step_is_negative else lower
            # 如果切片的开始位置为 None，则根据步长的正负确定开始位置
        else:
            start = slice.start
            # 否则使用切片指定的开始位置

        if start < 0:
            start += length
            if start < lower:
                start = lower
            # 如果开始位置小于 0，则加上 length，并确保不小于 lower
        else:
            if start > upper:
                start = upper
            # 否则确保开始位置不大于 upper

        # 计算结束位置
        if slice.stop is None:
            stop = lower if step_is_negative else upper
            # 如果切片的结束位置为 None，则根据步长的正负确定结束位置
        else:
            stop = slice.stop
            # 否则使用切片指定的结束位置

        if stop < 0:
            stop += length
            if stop < lower:
                stop = lower
            # 如果结束位置小于 0，则加上 length，并确保不小于 lower
        else:
            if stop > upper:
                stop = upper
            # 否则确保结束位置不大于 upper

        return [start, stop, step]
        # 返回开始、结束和步长组成的列表

    def apply_index(self, index):
        # 根据索引应用 Range 变量

        length = self.range_length()
        # 获取 Range 变量的长度

        if index < 0:
            index = length + index
            # 如果索引为负数，则将其转换为正数索引

        if index < 0 or index >= length:
            raise IndexError(f"index {index} is out of range")
            # 如果索引小于 0 或大于等于长度，则抛出索引错误异常

        return variables.ConstantVariable.create(self.start() + (index * self.step()))
        # 返回以索引计算得出的常量变量
    # 定义一个方法，用于应用切片操作到当前对象上
    def apply_slice(self, slice):
        # 从切片对象中获取起始、终止和步长信息，并转换为索引
        (slice_start, slice_stop, slice_step) = self._get_slice_indices(
            self.range_length(), slice
        )

        # 定义一个计算单个项的函数，根据索引计算对应的值
        def compute_item(index):
            return self.start() + (index * self.step())

        # 计算切片后的步长、起始和终止值
        sub_step = self.step() * slice_step
        sub_start = compute_item(slice_start)
        sub_stop = compute_item(slice_stop)

        # 创建一个新的 RangeVariable 对象，代表切片后的范围
        result = RangeVariable(
            [
                variables.ConstantVariable.create(x)
                for x in [sub_start, sub_stop, sub_step]
            ],
            mutable_local=MutableLocal() if self.mutable_local else None,
        )
        return result

    # 将当前对象转换为其 Python 常量表示
    def as_python_constant(self):
        return range(*[x.as_python_constant() for x in self.items])

    # 获取指定索引处的元素
    def getitem_const(self, arg: VariableTracker):
        # 模仿 Python C 源码中的实现方式
        index = arg.as_python_constant()

        # 如果索引是切片对象，则应用切片操作
        if isinstance(index, slice):
            return self.apply_slice(index)
        else:
            # 否则应用索引操作
            return self.apply_index(index)

    # 将当前对象转换为其代理表示
    def as_proxy(self):
        return self.python_type()(*self._as_proxy())

    # 将变量序列解包，并转换为常量变量列表
    def unpack_var_sequence(self, tx=None):
        return [variables.ConstantVariable.create(x) for x in self.as_python_constant()]

    # 使用代码生成器重新构造当前对象
    def reconstruct(self, codegen):
        # 确保在代码生成环境中不存在 "range" 变量
        assert "range" not in codegen.tx.f_globals
        # 将代码生成的空值推送到堆栈，加载 Python 模块 "range" 并追加到输出中
        codegen.add_push_null(
            lambda: codegen.append_output(codegen.create_load_python_module(range))
        )
        # 遍历当前对象的项，并扩展输出以创建函数调用
        codegen.foreach(self.items)
        codegen.extend_output(create_call_function(3, False))

    # 获取对象变量的属性
    def var_getattr(self, tx, name):
        # 可用属性列表
        fields = ["start", "stop", "step"]
        # 如果请求的属性不在列表中，则报告未实现该属性
        if name not in fields:
            unimplemented(f"range.{name}")
        # 返回对应属性的值
        return self.items[fields.index(name)]
# 定义一个继承自 BaseListVariable 的类 CommonListMethodsVariable，实现了一些适用于 List 和类似 List 的对象的共同方法。
class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    # 调用特定方法的函数，根据方法名执行相应操作并返回结果
    def call_method(
        self,
        tx,  # 事务对象，用于处理事务相关的操作
        name,  # 方法名
        args: List["VariableTracker"],  # 参数列表，包含 VariableTracker 对象的列表
        kwargs: Dict[str, "VariableTracker"],  # 关键字参数字典，键为字符串，值为 VariableTracker 对象
    ) -> "VariableTracker":  # 返回值为 VariableTracker 对象
        # 如果方法名为 "append"，并且对象为可变类型
        if name == "append" and self.mutable_local:
            assert not kwargs  # 确保没有关键字参数
            (arg,) = args  # 取出第一个参数
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            self.items.append(arg)  # 将参数添加到列表中
            return ConstantVariable.create(None)  # 返回一个表示常量 None 的 ConstantVariable 对象
        # 如果方法名为 "extend"，并且对象为可变类型，并且参数不为空且第一个参数具有可展开的变量序列
        elif (
            name == "extend"
            and self.mutable_local
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs  # 确保没有关键字参数
            (arg,) = args  # 取出第一个参数
            seq = arg.unpack_var_sequence(tx)  # 获取参数的可展开变量序列
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            self.items.extend(seq)  # 将序列中的元素扩展到列表中
            return ConstantVariable.create(None)  # 返回一个表示常量 None 的 ConstantVariable 对象
        # 如果方法名为 "insert"，并且对象为可变类型
        elif name == "insert" and self.mutable_local:
            assert not kwargs  # 确保没有关键字参数
            idx, value = args  # 获取参数中的索引和值
            const_idx = idx.as_python_constant()  # 获取索引的 Python 常量值
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            self.items.insert(const_idx, value)  # 在指定索引处插入值
            return ConstantVariable.create(None)  # 返回一个表示常量 None 的 ConstantVariable 对象
        # 如果方法名为 "pop"，并且对象为可变类型
        elif name == "pop" and self.mutable_local:
            assert not kwargs  # 确保没有关键字参数
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            return self.items.pop(*[a.as_python_constant() for a in args])  # 弹出并返回指定索引处的元素
        # 如果方法名为 "clear"，并且对象为可变类型
        elif name == "clear" and self.mutable_local:
            assert not kwargs and not args  # 确保既没有关键字参数也没有位置参数
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            self.items.clear()  # 清空列表
            return ConstantVariable.create(None)  # 返回一个表示常量 None 的 ConstantVariable 对象
        # 如果方法名为 "__setitem__"，并且对象为可变类型，并且参数不为空且第一个参数为 Python 常量
        elif (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs  # 确保没有关键字参数
            key, value = args  # 获取参数中的键和值
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            # 如果键是 SliceVariable 类型，则将值的项目列表赋给指定索引处的切片
            if isinstance(key, SliceVariable):
                self.items[key.as_python_constant()] = list(value.items)
            else:
                self.items[key.as_python_constant()] = value  # 否则直接赋值给指定索引处
            return ConstantVariable.create(None)  # 返回一个表示常量 None 的 ConstantVariable 对象
        # 如果方法名为 "copy"
        elif name == "copy":
            # List 的 copy() 方法没有参数
            assert not kwargs  # 确保没有关键字参数
            assert not args  # 确保没有位置参数
            items = list(self.items)  # 复制列表中的元素
            return self.modified(items, mutable_local=MutableLocal())  # 返回经修改的副本对象
        # 如果方法名为 "reverse"，并且对象为可变类型
        elif name == "reverse" and self.mutable_local:
            assert not kwargs  # 确保没有关键字参数
            assert not args  # 确保没有位置参数
            self.items.reverse()  # 反转列表中的元素顺序
            tx.output.side_effects.mutation(self)  # 输出侧效应：对自身进行变异操作
            return ConstantVariable.create(None)  # 返回一个表示常量 None 的 ConstantVariable 对象
        else:
            return super().call_method(tx, name, args, kwargs)  # 其他情况调用父类的 call_method 方法处理

# 定义一个类 ListVariable，继承自 CommonListMethodsVariable
class ListVariable(CommonListMethodsVariable):
    
    # 返回 Python 中的列表类型
    def python_type(self):
        return list

    # 返回对象的字符串表示形式，包含列表长度信息
    def __repr__(self):
        return f"{self.__class__.__name__}(length={len(self.items)})"
    # 调试用的对象表示方法，返回对象的调试表示字符串
    def debug_repr(self):
        return self.debug_repr_helper("[", "]")

    # 根据给定的代码生成器重建对象，用于序列化操作
    def reconstruct(self, codegen):
        # 遍历对象的元素并生成相应的指令
        codegen.foreach(self.items)
        # 在生成的指令中添加构建列表的操作
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))

    # 调用对象的方法
    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        # 如果调用的是 "__setitem__" 方法，并且对象是可变的本地对象，并且有参数，并且第一个参数是Python常量
        if (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            # 断言不应该有关键字参数
            assert not kwargs
            # 获取键和值
            key, value = args
            # 在事务输出中标记这次操作是一个突变
            tx.output.side_effects.mutation(self)
            # 如果键是 SliceVariable 类型
            if isinstance(key, SliceVariable):
                # 如果值不能展开为变量序列，则报告未实现的错误
                if not value.has_unpack_var_sequence(tx):
                    unimplemented(
                        f"Missing dynamo support for expanding {value} into a list for slice assignment."
                    )
                # 将值展开为变量序列并赋给对应的切片键
                self.items[key.as_python_constant()] = value.unpack_var_sequence(tx)
            else:
                # 否则直接将值赋给对应的键
                self.items[key.as_python_constant()] = value
            # 返回一个表示无返回值的常量变量
            return ConstantVariable.create(None)
        else:
            # 如果不满足上述条件，则调用父类的方法处理
            return super().call_method(tx, name, args, kwargs)

    # 调用对象的 hasattr 方法
    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        # 如果对象的 Python 类型不是 list，则调用父类的 hasattr 方法处理
        if self.python_type() is not list:
            return super().call_hasattr(tx, name)
        # 否则创建一个表示是否存在指定属性的常量变量
        return variables.ConstantVariable.create(hasattr([], name))
class DequeVariable(CommonListMethodsVariable):
    # 返回 Python deque 类型
    def python_type(self):
        return collections.deque

    # 返回调试表示形式，用于输出调试信息
    def debug_repr(self):
        return self.debug_repr_helper("deque([", "])")

    # 根据代码生成器重构对象
    def reconstruct(self, codegen):
        # 确保不会在全局命名空间中找到 "deque" 对象
        assert "deque" not in codegen.tx.f_globals
        # 向代码生成器添加推送空值操作
        codegen.add_push_null(
            # 在代码生成器中附加加载 Python 模块 collections.deque 的操作
            lambda: codegen.append_output(
                codegen.create_load_python_module(collections.deque)
            )
        )
        # 对每个元素执行迭代
        codegen.foreach(self.items)
        # 扩展输出以创建函数调用
        codegen.extend_output(create_call_function(len(self.items), False))

    # 调用方法处理
    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        # 处理 "__setitem__" 方法调用
        if (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            # 断言键是 Python 常量且为整数类型
            assert key.is_python_constant() and isinstance(
                key.as_python_constant(), int
            )
            # 在事务输出中进行变异
            tx.output.side_effects.mutation(self)
            # 更新对象的元素
            self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)
        # 处理 "extendleft" 方法调用
        elif name == "extendleft" and self.mutable_local:
            assert not kwargs

            (arg,) = args
            # 解包变量序列并逆转顺序
            prefix = arg.unpack_var_sequence(tx)
            prefix.reverse()
            # 在事务输出中进行变异
            tx.output.side_effects.mutation(self)
            # 更新对象的元素
            self.items = prefix + list(self.items)
            return ConstantVariable.create(None)
        # 处理 "popleft" 方法调用
        elif name == "popleft" and self.mutable_local:
            assert not args
            assert not kwargs
            # 获取第一个元素
            item = self.items[0]
            # 在事务输出中进行变异
            tx.output.side_effects.mutation(self)
            # 更新对象的元素
            self.items = self.items[1:]
            return item
        # 处理 "appendleft" 方法调用
        elif name == "appendleft" and self.mutable_local:
            assert not kwargs
            # 在事务输出中进行变异
            tx.output.side_effects.mutation(self)
            # 更新对象的元素
            self.items = [args[0]] + list(self.items)
            return ConstantVariable.create(None)
        else:
            # 默认调用父类的方法
            return super().call_method(tx, name, args, kwargs)


class TupleVariable(BaseListVariable):
    # 返回 Python tuple 类型
    def python_type(self):
        return tuple

    # 返回调试表示形式，用于输出调试信息
    def debug_repr(self):
        return self.debug_repr_helper("(", ")")

    # 根据代码生成器重构对象
    def reconstruct(self, codegen):
        # 对每个元素执行迭代
        codegen.foreach(self.items)
        # 扩展输出以创建指令 "BUILD_TUPLE"
        codegen.append_output(create_instruction("BUILD_TUPLE", arg=len(self.items)))

    # 调用方法处理
    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        # 默认调用父类的方法
        return super().call_method(tx, name, args, kwargs)

    # 调用 hasattr 方法处理
    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        # 如果 Python 类型不是 tuple，则调用父类的方法
        if self.python_type() is not tuple:
            return super().call_hasattr(tx, name)
        # 否则，返回常量变量，表示是否具有指定属性名
        return variables.ConstantVariable.create(hasattr((), name))


class SizeVariable(TupleVariable):
    """torch.Size(...)"""
    # 非变量字段集合，包含字符串 "proxy" 和 TupleVariable 类的非变量字段
    _nonvar_fields = {
        "proxy",
        *TupleVariable._nonvar_fields,
    }
    
    # 初始化方法，接受一个 items 参数作为 VariableTracker 对象的列表，
    # 可选的 proxy 参数作为 torch.fx.Proxy 对象，默认为 None。
    # **kwargs 是其他关键字参数的集合。
    def __init__(
        self,
        items: List[VariableTracker],
        proxy: Optional[torch.fx.Proxy] = None,
        **kwargs,
    ):
        # 将 proxy 参数赋值给实例变量 self.proxy
        self.proxy = proxy
        # 调用父类的初始化方法，将 items 参数和其他关键字参数传递给父类
        super().__init__(items, **kwargs)
    
    # 返回一个调试用的字符串表示，格式为 "torch.Size([ ... ])"
    def debug_repr(self):
        return self.debug_repr_helper("torch.Size([", "])")
    
    # 返回 torch.Size 类型
    def python_type(self):
        return torch.Size
    
    # 返回代理对象 self.proxy；如果 self.proxy 为 None，则处理 torch.Size 特殊情况的逻辑。
    # torch.Size 是一个特殊的情况，它虽然是 tuple 的子类，但不允许包含非 int-like 成员（如 Proxy 和 Node）。
    # 因此，需要特殊处理 torch.Size 的代理情况。
    def as_proxy(self):
        if self.proxy is not None:
            return self.proxy
    
        # 查找代理对象的方法。如果没有找到代理对象，则执行遗留的行为
        tracer = None
        proxies = self._as_proxy()
        for proxy in proxies:
            if isinstance(proxy, torch.fx.Proxy):
                tracer = proxy.tracer
                break
    
        # 如果没有找到 tracer，则直接返回 torch.Size 对象
        if tracer is None:
            return torch.Size(proxies)
    
        # 使用 tracer 创建一个代理对象，以表示 torch.Size 的代理
        proxy = tracer.create_proxy("call_function", torch.Size, (proxies,), {})
        # 设置代理节点的示例值
        set_example_value(
            proxy.node,
            torch.Size(
                [
                    p.node.meta["example_value"] if not isinstance(p, int) else p
                    for p in proxies
                ]
            ),
        )
        return proxy
    
    # 重构方法，接受一个 codegen 参数，用于生成代码。
    # 在 codegen 中添加指令以创建一个 torch.Size 对象。
    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen.load_import_from("torch", "Size"))
        codegen.foreach(self.items)
        build_torch_size = [
            create_instruction("BUILD_TUPLE", arg=len(self.items)),
        ] + create_call_function(1, False)
        codegen.extend_output(build_torch_size)
    
    # 解包变量序列的方法，返回一个包含 self.items 中所有元素的列表。
    def unpack_var_sequence(self, tx):
        return list(self.items)
    def numel(self, tx):
        # 导入必要的模块和类
        from .builtin import BuiltinVariable
        from .tensor import SymNodeVariable

        # 初始化常量结果为1，符号大小列表为空
        const_result = 1
        sym_sizes = []

        # 遍历self.items中的每个变量v
        for v in self.items:
            if isinstance(v, ConstantVariable):
                # 如果v是常量变量，则将其值乘到const_result中
                const_result *= v.value
            else:
                # 如果v是符号节点变量，则断言其为SymNodeVariable类型，并将其添加到sym_sizes中
                assert isinstance(v, SymNodeVariable), type(v)
                # 延迟代理调用，直到需要确保其必要性
                sym_sizes.append(v)

        # 创建一个包含const_result值的常量变量结果
        result = ConstantVariable.create(const_result)

        # 如果存在符号大小列表并且const_result为1，则跳过乘以1的操作
        if sym_sizes and const_result == 1:
            result, *sym_sizes = sym_sizes

        # 如果没有符号大小列表或者const_result为0，则直接返回结果
        if not sym_sizes or const_result == 0:
            return result

        # 创建一个乘法操作的内置变量
        mul = BuiltinVariable(operator.mul)
        
        # 对sym_sizes中的每个符号大小变量执行乘法操作，累积到result中
        for v in sym_sizes:
            result = mul.call_function(tx, [result, v], {})

        # 返回最终计算结果
        return result

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        # 如果调用的方法名为"__getitem__"
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            # 调用get_item_dyn方法获取动态项目
            out = self.get_item_dyn(tx, args[0])
            return out
        # 如果调用的方法名为"numel"
        elif name == "numel":
            assert not args and not kwargs
            # 调用本对象的numel方法进行计算
            return self.numel(tx)

        # 如果以上条件都不符合，则调用父类的call_method方法
        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(self, tx, arg: VariableTracker):
        # 导入必要的模块和类
        from .tensor import SymNodeVariable

        # 如果参数arg是SymNodeVariable类型，则将其索引设为sym_num
        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            # 否则将其转换为Python常量
            index = arg.as_python_constant()
        
        # 如果索引是一个切片对象，则返回对应的SizeVariable对象
        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            # 否则断言索引为int或torch.SymInt类型，并返回对应的项目
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        # 返回torch.Size类是否具有指定属性名的常量变量
        return variables.ConstantVariable.create(hasattr(torch.Size, name))
class NamedTupleVariable(TupleVariable):
    # 定义类级变量 _nonvar_fields，包含字符串 "tuple_cls" 和 TupleVariable 类的 _nonvar_fields
    _nonvar_fields = {
        "tuple_cls",
        *TupleVariable._nonvar_fields,
    }

    # 初始化方法，接受 items、tuple_cls 和 kwargs 作为参数
    def __init__(self, items, tuple_cls, **kwargs):
        # 调用父类的初始化方法
        super().__init__(items, **kwargs)
        # 设置实例变量 tuple_cls
        self.tuple_cls = tuple_cls

    # 调试用表示方法，返回该对象的调试表示
    def debug_repr(self):
        # 生成包含调试表示的元组，并返回其表示形式
        return repr(self.tuple_cls(*(Lit(x.debug_repr()) for x in self.items)))

    # 返回该对象的 Python 类型
    def python_type(self):
        return self.tuple_cls

    # 返回该对象的 Python 常量表示
    def as_python_constant(self):
        # 使用对象的 Python 类型创建一个新实例，并返回其常量表示
        return self.python_type()(*[x.as_python_constant() for x in self.items])

    # 返回该对象的代理表示
    def as_proxy(self):
        # 断言该对象的 Python 类型不是 SizeVariable
        assert self.python_type() is not SizeVariable
        # 使用对象的 Python 类型创建一个新实例，并返回其代理表示
        return self.python_type()(*self._as_proxy())

    # 根据代码生成器重构该对象
    def reconstruct(self, codegen):
        # 获取创建元组的函数，如果不存在则使用 self.tuple_cls
        create_fn = getattr(self.tuple_cls, "_make", self.tuple_cls)
        # 添加将空值推送到堆栈的操作
        codegen.add_push_null(
            lambda: codegen.append_output(codegen._create_load_const(create_fn))
        )
        # 遍历对象的 items
        codegen.foreach(self.items)
        # 扩展输出，包括创建元组的指令和调用函数的指令
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(self.items)),
            ]
            + create_call_function(1, False)
        )

    # 获取对象的属性，如果不存在则调用父类的方法
    def var_getattr(self, tx, name):
        # 内部函数，检查并创建方法对象
        def check_and_create_method():
            # 获取指定名称的方法
            method = inspect.getattr_static(self.tuple_cls, name, None)
            if isinstance(method, classmethod):
                # 如果是类方法，则返回一个用户方法变量
                return UserMethodVariable(
                    method.__func__,
                    variables.UserDefinedClassVariable(self.tuple_cls),
                )
            elif isinstance(method, staticmethod):
                # 如果是静态方法，则返回一个用户函数变量
                return UserFunctionVariable(method.__func__)
            elif inspect.isfunction(method):
                # 如果是普通函数，则返回一个用户方法变量
                return UserMethodVariable(method, self)
            else:
                return None

        # 获取命名元组的字段列表
        fields = namedtuple_fields(self.tuple_cls)
        # 如果名称不在字段列表中，则尝试创建方法对象，否则返回字段对应的元素
        if name not in fields:
            method = check_and_create_method()
            if not method:
                super().var_getattr(tx, name)
            return method
        return self.items[fields.index(name)]

    # 调用 hasattr 方法，检查对象是否具有指定名称的属性
    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        return variables.ConstantVariable.create(hasattr(self.tuple_cls, name))


class SliceVariable(BaseListVariable):
    # 初始化方法，接受 items 和 kwargs 作为参数
    def __init__(self, items, **kwargs):
        # 将 items 赋值给局部变量 items_to_map
        items_to_map = items
        # 初始化 start、stop、step 为 None 的常量变量
        start, stop, step = [variables.ConstantVariable.create(None)] * 3

        # 根据 items_to_map 的长度确定 start、stop、step 的值
        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        # 如果 start 或 stop 是 TensorVariable，则报错
        if isinstance(start, variables.TensorVariable) or isinstance(
            stop, variables.TensorVariable
        ):
            unimplemented("Dynamic slicing on data-dependent value is not supported")

        # 调用父类的初始化方法，传入 [start, stop, step] 和 kwargs
        super().__init__([start, stop, step], **kwargs)
    # 返回一个调试用的字符串表示，格式为 "slice(...)"
    def debug_repr(self):
        return self.debug_repr_helper("slice(", ")")

    # 将当前对象转换为一个切片对象
    def as_proxy(self):
        return slice(*self._as_proxy())

    # 返回切片对象的 Python 类型
    def python_type(self):
        return slice

    # 将当前对象转换为一个 Python 的常量表示的切片对象
    def as_python_constant(self):
        # 对 self.items 中的每个元素应用 guard_if_dyn 函数，然后作为参数传递给 slice 函数
        return slice(*[guard_if_dyn(x) for x in self.items])

    # 使用 codegen 对象对 self.items 中的每个元素执行迭代操作
    # 然后将生成的指令添加到 codegen 的输出中，指令为 "BUILD_SLICE"，参数为 self.items 的长度
    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_SLICE", arg=len(self.items)))

    # 获取对象的属性值，其中 tx 是一个变量，name 是属性名
    # 如果属性名不是 "start", "stop", "step" 中的一个，将会调用未实现函数 unimplemented，并传递错误信息
    # 否则返回 self.items 中对应属性名的元素值
    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"slice.{name}")
        return self.items[fields.index(name)]
class ListIteratorVariable(VariableTracker):
    # ListIteratorVariable 类，继承自 VariableTracker
    
    _nonvar_fields = {
        "index",
        *VariableTracker._nonvar_fields,
    }
    # _nonvar_fields 定义了类的非变量字段，包括 "index" 和 VariableTracker 类的 _nonvar_fields
    
    def __init__(self, items, index: int = 0, **kwargs):
        super().__init__(**kwargs)
        # 调用父类的构造函数，传递所有关键字参数
        
        assert isinstance(items, list)
        # 断言 items 是一个列表
        
        # Removing this check as it slows things down too much
        # https://github.com/pytorch/pytorch/pull/87533#issuecomment-1287574492
        # 移除这个检查，因为它导致速度变慢
        
        # assert all(isinstance(x, VariableTracker) for x in items)
        # 断言 items 中的所有元素都是 VariableTracker 的实例
        
        self.items = items
        self.index = index
        # 初始化实例变量 items 和 index
        
    def __repr__(self):
        return f"{self.__class__.__name__}(length={len(self.items)}, index={repr(self.index)})"
        # 返回实例的字符串表示，包括 items 的长度和 index 的表示
    
    def next_variable(self, tx):
        assert self.mutable_local
        # 断言 self.mutable_local 为真
        
        old_index = self.index
        # 记录旧的索引值
        
        if old_index >= len(self.items):
            raise StopIteration
        # 如果旧的索引超过了 items 的长度，抛出 StopIteration 异常
        
        tx.output.side_effects.mutation(self)
        # 在 tx.output 上执行副作用的 mutation 操作
        
        self.index += 1
        # 索引值加一
        
        return self.items[old_index]
        # 返回旧索引处的 items 元素

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ):
        if name == "__contains__":
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.items[self.index :], args[0], tx)
        # 如果调用的方法名为 "__contains__"，则调用 iter_contains 函数

        return super().call_method(tx, name, args, kwargs)
        # 否则调用父类的 call_method 方法

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError
        # 如果索引大于 0，抛出 NotImplementedError
        
        return iter([x.as_python_constant() for x in self.items])
        # 返回 items 中每个元素的 as_python_constant() 方法调用结果的迭代器

    def unpack_var_sequence(self, tx):
        return list(self.items[self.index :])
        # 返回从当前索引开始到末尾的 items 列表

    def reconstruct(self, codegen):
        remaining_items = self.items[self.index :]
        # 获取剩余的 items 元素
        
        codegen.foreach(remaining_items)
        # 使用 codegen 执行 foreach 操作遍历 remaining_items
        
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(remaining_items)),
                create_instruction("GET_ITER"),
            ]
        )
        # 将指令序列扩展到输出中，构建一个元组并获取其迭代器


class TupleIteratorVariable(ListIteratorVariable):
    pass
    # TupleIteratorVariable 类，继承自 ListIteratorVariable


class RestrictedListSubclassVariable(ListVariable):
    """
    This is a special case of UserDefinedObjectVariable where:
        1) The user subclasses list
        2) None of the list methods are overriden, merely some new methods are added

    In these cases, we can prevent graph breaks by not using the general
    UserDefinedObjectVariable machinery and instead treating it like
    a ListVariable.
    """
    # RestrictedListSubclassVariable 类，继承自 ListVariable
    # 用于描述一种特殊情况，用户定义的对象类别变量，其子类化了列表，并添加了一些新方法

    _nonvar_fields = {"user_cls", "user_cls_source", *ListVariable._nonvar_fields}
    # _nonvar_fields 定义了类的非变量字段，包括 "user_cls", "user_cls_source" 和 ListVariable 类的 _nonvar_fields
    
    _allowed_names = {
        "__call__",
        "__module__",
        "__dict__",
        "__doc__",
        "__name__",
        "__qualname__",
    }
    # _allowed_names 定义了允许访问的方法名集合
    
    _disallowed_names = {
        "__getattribute__",
        "__getattr__",
        "__setattr__",
    }
    # _disallowed_names 定义了禁止访问的方法名集合

    @classmethod
    def _is_non_conflicting_subclass(
        cls,
        user_cls: type,
        python_cls: type,
        ...
    ):
        """
        确保 user_cls 继承自 python_cls（例如 list），并且不覆盖 python_cls 的任何方法
        """
        if (
            not istype(user_cls, type)  # 检查 user_cls 是否为 type 类型
            or user_cls.__bases__ != (python_cls,)  # 检查 user_cls 的基类是否为 python_cls
            or user_cls.__mro__ != (user_cls, python_cls, object)  # 检查 user_cls 的方法解析顺序
        ):
            return False  # 如果不满足上述条件，则不是子类
        return not any(
            hasattr(python_cls, name) or name in cls._disallowed_names
            for name in set(user_cls.__dict__.keys()) - cls._allowed_names
            # 检查 user_cls 的方法是否与 python_cls 的方法名冲突，或者方法名是否在禁止的名称列表中
        )

    @classmethod
    def is_matching_cls(cls, user_cls: type):
        """
        检查 user_cls 是否与 list 类型兼容，即是否可以作为 RestrictedListSubclassVariable 的 user_cls
        """
        return cls._is_non_conflicting_subclass(user_cls, list)

    def __init__(self, items, *, user_cls: type, user_cls_source: Source, **kwargs):
        """
        初始化方法，接受 items、user_cls 和 user_cls_source 作为参数
        """
        super().__init__(items=items, **kwargs)
        self.user_cls = user_cls  # 设置实例变量 user_cls
        self.user_cls_source = user_cls_source  # 设置实例变量 user_cls_source
        assert istype(user_cls, type)  # 断言 user_cls 是一个类型
        assert isinstance(user_cls_source, Source)  # 断言 user_cls_source 是 Source 类型的实例

    def debug_repr(self):
        """
        返回一个字符串表示，用于调试输出
        """
        # 构造一个 user_cls 的实例，包含 items 列表中每个元素的 debug_repr 结果
        return repr(self.user_cls([Lit(x.debug_repr()) for x in self.items]))

    def python_type(self):
        """
        返回 user_cls 类型
        """
        return self.user_cls

    def as_proxy(self):
        """
        返回 items 列表中每个元素的 as_proxy() 方法结果的列表
        """
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        """
        抛出 NotImplementedError 异常，表示不支持将对象转换为 Python 常量
        """
        raise NotImplementedError

    def is_python_constant(self):
        """
        返回 False，表示当前对象不是 Python 常量
        """
        return False

    @property
    def value(self):
        """
        抛出 AttributeError 异常，表示对象没有名为 value 的属性
        """
        raise AttributeError("value")

    def modified(self, items, **kwargs):
        """
        返回一个新的对象，用指定的 items 和其他参数修改当前对象的副本
        """
        return type(self)(
            items,
            user_cls=self.user_cls,
            user_cls_source=self.user_cls_source,
            **kwargs,
        )

    def reconstruct(self, codegen):
        """
        用于在代码生成器中重构对象，添加一个 null 值和 user_cls_source 的生成代码
        """
        codegen.add_push_null(lambda: codegen(self.user_cls_source))
        super().reconstruct(codegen)
        codegen.extend_output(create_call_function(1, False))

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        """
        调用对象的方法，如果方法存在于 user_cls 中，则执行；否则，调用父类的同名方法
        """
        if name in self.user_cls.__dict__:  # 检查方法名是否在 user_cls 的字典中
            method = self.user_cls.__dict__[name]  # 获取方法对象
            if isinstance(method, types.FunctionType):  # 检查方法是否为函数类型
                # 内联方法调用
                source = AttrSource(self.user_cls_source, name)
                return UserMethodVariable(method, self, source=source).call_function(
                    tx, args, kwargs
                )
            unimplemented(
                f"RestrictedListSubclassVariable method {self.user_cls.__name__}.{name}"
            )
        return super().call_method(tx, name, args, kwargs)  # 调用父类的方法
    # 定义一个方法 `call_function`，它接受以下参数：
    # - `tx`: 事务对象，用于处理函数调用
    # - `args`: 一个变量追踪器对象的列表，表示函数的位置参数
    # - `kwargs`: 一个字典，键为字符串表示函数的关键字参数，值为变量追踪器对象
    # 方法返回一个变量追踪器对象，表示函数调用的结果
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 调用对象的 `call_method` 方法来执行函数调用，传入以下参数：
        # - `tx`: 事务对象，用于处理函数调用
        # - `"__call__"`: 字符串表示要调用的方法名，这里是调用对象自身的 `__call__` 方法
        # - `args`: 位置参数的变量追踪器列表
        # - `kwargs`: 关键字参数的变量追踪器字典
        return self.call_method(tx, "__call__", args, kwargs)
```