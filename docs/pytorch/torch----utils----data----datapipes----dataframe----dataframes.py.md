# `.\pytorch\torch\utils\data\datapipes\dataframe\dataframes.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型定义
from typing import Any, Dict, List, Optional

# 从特定模块中导入功能性数据管道装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 从数据框架结构模块中导入数据块数据帧
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
# 从数据管道模块中导入数据块迭代数据管道和迭代数据管道
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


# TODO(VitalyFedyunin): 当合并两个不同的跟踪时添加错误处理

# 公开的符号列表
__all__ = [
    "Capture",
    "CaptureA",
    "CaptureAdd",
    "CaptureCall",
    "CaptureControl",
    "CaptureDataFrame",
    "CaptureDataFrameWithDataPipeOps",
    "CaptureF",
    "CaptureGetAttr",
    "CaptureGetItem",
    "CaptureInitial",
    "CaptureLikeMock",
    "CaptureMul",
    "CaptureSetItem",
    "CaptureSub",
    "CaptureVariable",
    "CaptureVariableAssign",
    "DataFrameTracer",
    "DataFrameTracedOps",
    "disable_capture",
    "get_val",
]


# 禁用捕获功能的函数
def disable_capture():
    CaptureControl.disabled = True


# 捕获控制类，用于控制是否禁用捕获功能
class CaptureControl:
    disabled = False


# 数据帧追踪操作类，继承自数据块迭代数据管道
class DataFrameTracedOps(DFIterDataPipe):
    def __init__(self, source_datapipe, output_var):
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        # 迭代源数据管道的每个项目，应用输出变量的操作
        for item in self.source_datapipe:
            yield self.output_var.apply_ops(item)


# 待实现的数据管道操作列表
DATAPIPES_OPS = [
    "_dataframes_as_tuples",
    "groupby",
    "_dataframes_filter",
    "map",
    "to_datapipe",
    "shuffle",
    "concat",
    "batch",
    "_dataframes_per_row",
    "_dataframes_concat",
    "_dataframes_shuffle",
]

# 未实现的属性列表
UNIMPLEMENTED_ATTR = ["__deepcopy__", "__setstate__", "is_shardable", "apply_sharding"]


# 初始捕获类
class Capture:
    # TODO: 所有操作都在整个InitialCapture中共享，需要弄清楚如果我们合并两个捕获时会发生什么

    def __init__(self, schema_df=None):
        # 初始化上下文，包含操作列表、变量列表和模式数据帧
        self.ctx = {"operations": [], "variables": [], "schema_df": schema_df}

    def __str__(self):
        # 返回操作列表的字符串表示
        return self._ops_str()

    def _ops_str(self):
        # 获取操作列表的字符串表示
        res = ""
        for op in self.ctx["operations"]:
            if len(res) > 0:
                res += "\n"
            res += str(op)
        return res

    def __getstate__(self):
        # 获取对象的状态以便序列化
        # TODO(VitalyFedyunin): 目前无法 pickle（为什么？）
        self.ctx["schema_df"] = None
        for var in self.ctx["variables"]:
            var.calculated_value = None
        state = {}
        for item in self.__dict__:
            state[item] = getattr(self, item)
        return state

    def __setstate__(self, state):
        # 设置对象的状态以便反序列化
        for k, v in state.items():
            setattr(self, k, v)

    def __getattr__(self, attrname):
        # 获取属性时的处理逻辑，特定属性引发异常
        if attrname == "kwarg" or attrname == "kwargs":
            raise RuntimeError("no kwargs!")
        if attrname in ["__deepcopy__"]:
            raise AttributeError
        # 返回捕获属性获取对象
        result = CaptureGetAttr(self, attrname, ctx=self.ctx)
        return result

    def __getitem__(self, key):
        # 获取索引时的处理逻辑
        return CaptureGetItem(self, key, ctx=self.ctx)
    # 当对象的键值对被设置时，将操作记录到上下文的操作列表中
    def __setitem__(self, key, value):
        self.ctx["operations"].append(CaptureSetItem(self, key, value, ctx=self.ctx))

    # 当对象与另一个值相加时，创建一个捕获加法操作的结果，并记录到上下文的操作列表中
    def __add__(self, add_val):
        res = CaptureAdd(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx["operations"].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        )
        return var

    # 当对象与另一个值相减时，创建一个捕获减法操作的结果，并记录到上下文的操作列表中
    def __sub__(self, add_val):
        res = CaptureSub(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx["operations"].append(
            CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        )
        return var

    # 当对象与另一个值相乘时，创建一个捕获乘法操作的结果，并记录到上下文的操作列表中
    def __mul__(self, add_val):
        res = CaptureMul(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        t = CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        self.ctx["operations"].append(t)
        return var

    # 检查上下文是否为空，即操作列表和变量列表都为空
    def _is_context_empty(self):
        return len(self.ctx["operations"]) == 0 and len(self.ctx["variables"]) == 0

    # 将当前对象应用于给定的数据框架，执行上下文中记录的操作
    def apply_ops_2(self, dataframe):
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        # 更新上下文中第一个变量的计算值为数据框架
        self.ctx["variables"][0].calculated_value = dataframe
        # 对上下文中记录的每个操作执行执行函数
        for op in self.ctx["operations"]:
            op.execute()

    # 获取列属性时，先将上下文应用于架构数据框架，然后执行操作并返回执行结果的列
    @property
    def columns(self):
        self.apply_ops_2(self.ctx["schema_df"])
        value = self.execute()
        return value.columns

    # 对象被调用时的行为，根据传入的参数和关键字参数来更新上下文
    def __call__(self, *args, **kwargs):
        # TODO: Check if args or kwargs have more than one different context
        # 如果当前上下文为空，则从传入参数中的第一个非空上下文中获取上下文
        if self._is_context_empty():
            # TODO: Allow CaptureA to take context from mock
            for arg in args:
                if isinstance(arg, Capture) and not arg._is_context_empty():
                    self.ctx = arg.ctx
                    break
            if self._is_context_empty():
                for k, v in kwargs.items():
                    if isinstance(k, Capture) and not k._is_context_empty():
                        self.ctx = k.ctx
                        break
                    if isinstance(v, Capture) and not v._is_context_empty():
                        self.ctx = v.ctx
                        break

        # 创建一个捕获调用操作的结果，并将其记录到上下文的操作列表中
        res = CaptureCall(self, ctx=self.ctx, args=args, kwargs=kwargs)
        var = CaptureVariable(None, ctx=self.ctx)
        t = CaptureVariableAssign(ctx=self.ctx, variable=var, value=res)
        self.ctx["operations"].append(t)
        return var
class CaptureF(Capture):
    # 定义一个名为 CaptureF 的类，继承自 Capture 类
    def __init__(self, ctx=None, **kwargs):
        # 初始化方法，接受上下文和关键字参数
        if ctx is None:
            self.ctx = {"operations": [], "variables": []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs
        # 将传入的 kwargs 参数保存到实例变量 self.kwargs 中


class CaptureA(CaptureF):
    # 定义一个名为 CaptureA 的类，继承自 CaptureF 类
    def __str__(self):
        # 返回实例的字符串表示形式，包含参数 name 对应的值
        return f"{self.kwargs['name']}"

    def execute(self):
        # 执行方法，获取参数 real_attribute 对应的值，并返回
        value = self.kwargs["real_attribute"]
        return value


class CaptureLikeMock:
    # 定义一个名为 CaptureLikeMock 的类
    def __init__(self, name):
        # 初始化方法，接受参数 name
        import unittest.mock as mock

        # 使用 unittest.mock 模块的私有函数 _get_target 获取目标和属性
        # TODO(VitalyFedyunin): Do not use provate function here, copy own implementation instead.
        get_target, attribute = mock._get_target(name)  # type: ignore[attr-defined]
        self.get_target = get_target
        self.attribute = attribute
        self.name = name

    def __enter__(self):
        # 进入上下文时执行的方法
        # 保存当前目标对象的属性值
        self.save = getattr(self.get_target(), self.attribute)
        # 创建 CaptureA 实例 capt，并将其赋值给目标对象的属性
        capt = CaptureA(name=self.name, real_attribute=self.save)
        setattr(self.get_target(), self.attribute, capt)

    def __exit__(self, *exc_info):
        # 退出上下文时执行的方法
        # 恢复目标对象的属性为之前保存的值
        setattr(self.get_target(), self.attribute, self.save)


class CaptureCall(Capture):
    # 定义一个名为 CaptureCall 的类，继承自 Capture 类
    def __init__(self, callable, ctx=None, **kwargs):
        # 初始化方法，接受可调用对象 callable，上下文和关键字参数
        if ctx is None:
            self.ctx = {"operations": [], "variables": []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs
        self.callable = callable

    def __str__(self):
        # 返回实例的字符串表示形式，包含 callable 和 kwargs 的格式化字符串
        return "{callable}({args},{kwargs})".format(
            callable=self.callable, **self.kwargs
        )

    def execute(self):
        # 执行方法，执行 kwargs 中的参数，可能包含嵌套结构
        executed_args = []
        for arg in self.kwargs["args"]:
            if isinstance(arg, Capture):
                executed_args.append(arg.execute())
            else:
                executed_args.append(arg)
        # 调用 callable，传入执行后的参数列表和 kwargs 中的关键字参数，并返回结果
        left = get_val(self.callable)
        return left(*executed_args, **self.kwargs["kwargs"])


class CaptureVariableAssign(CaptureF):
    # 定义一个名为 CaptureVariableAssign 的类，继承自 CaptureF 类
    def __str__(self):
        # 返回实例的字符串表示形式，包含变量和值的赋值语句
        variable = self.kwargs["variable"]
        value = self.kwargs["value"]
        return f"{variable} = {value}"

    def execute(self):
        # 执行方法，将值的执行结果赋给变量的 calculated_value 属性
        self.kwargs["variable"].calculated_value = self.kwargs["value"].execute()


class CaptureVariable(Capture):
    # 定义一个名为 CaptureVariable 的类，继承自 Capture 类
    # TODO(VitalyFedyunin): This should be atomic and thread safe
    names_idx = 0

    def __init__(self, value, ctx):
        # 初始化方法，接受值和上下文作为参数
        if CaptureControl.disabled:
            raise RuntimeError("Attempting to create capture variable with capture off")
        self.ctx = ctx
        self.value = value
        # 创建唯一的变量名
        self.name = f"var_{CaptureVariable.names_idx}"
        CaptureVariable.names_idx += 1
        # 将当前实例添加到上下文的 variables 列表中
        self.ctx["variables"].append(self)

    def __str__(self):
        # 返回实例的名称
        return self.name

    def execute(self):
        # 执行方法，返回计算后的值
        return self.calculated_value
    def apply_ops(self, dataframe):
        # TODO(VitalyFedyunin): Make this calculation thread safe (as currently it updates pointer)
        # 将传入的 dataframe 赋值给变量列表中的第一个变量的 calculated_value 属性
        self.ctx["variables"][0].calculated_value = dataframe
        # 遍历操作列表中的每个操作对象，并执行其 execute 方法
        for op in self.ctx["operations"]:
            op.execute()
        # 返回对象自身的 calculated_value 属性
        return self.calculated_value
class CaptureGetItem(Capture):
    def __init__(self, left, key, ctx):
        self.ctx = ctx  # 初始化上下文变量
        self.left = left  # 设置左操作数
        self.key = key  # 设置键值

    def __str__(self):
        return f"{self.left}[{get_val(self.key)}]"  # 返回左操作数和键值对应的值的字符串表示

    def execute(self):
        left = self.left.execute()  # 执行左操作数并获取结果
        return left[self.key]  # 返回左操作数中键值对应的值


class CaptureSetItem(Capture):
    def __init__(self, left, key, value, ctx):
        self.ctx = ctx  # 初始化上下文变量
        self.left = left  # 设置左操作数
        self.key = key  # 设置键值
        self.value = value  # 设置要设置的值

    def __str__(self):
        return f"{self.left}[{get_val(self.key)}] = {self.value}"  # 返回设置左操作数中键值的表达式字符串

    def execute(self):
        left = self.left.execute()  # 执行左操作数并获取结果
        value = self.value.execute()  # 执行要设置的值并获取结果
        left[self.key] = value  # 在左操作数中设置键值对应的值


class CaptureAdd(Capture):
    def __init__(self, left, right, ctx):
        self.ctx = ctx  # 初始化上下文变量
        self.left = left  # 设置左操作数
        self.right = right  # 设置右操作数

    def __str__(self):
        return f"{self.left} + {self.right}"  # 返回左右操作数的加法表达式字符串

    def execute(self):
        return get_val(self.left) + get_val(self.right)  # 返回左右操作数的值的和


class CaptureMul(Capture):
    def __init__(self, left, right, ctx):
        self.ctx = ctx  # 初始化上下文变量
        self.left = left  # 设置左操作数
        self.right = right  # 设置右操作数

    def __str__(self):
        return f"{self.left} * {self.right}"  # 返回左右操作数的乘法表达式字符串

    def execute(self):
        return get_val(self.left) * get_val(self.right)  # 返回左右操作数的值的积


class CaptureSub(Capture):
    def __init__(self, left, right, ctx):
        self.ctx = ctx  # 初始化上下文变量
        self.left = left  # 设置左操作数
        self.right = right  # 设置右操作数

    def __str__(self):
        return f"{self.left} - {self.right}"  # 返回左右操作数的减法表达式字符串

    def execute(self):
        return get_val(self.left) - get_val(self.right)  # 返回左操作数减去右操作数的值


class CaptureGetAttr(Capture):
    def __init__(self, src, name, ctx):
        self.ctx = ctx  # 初始化上下文变量
        self.src = src  # 设置源对象
        self.name = name  # 设置属性名

    def __str__(self):
        return f"{self.src}.{self.name}"  # 返回获取属性的表达式字符串

    def execute(self):
        val = get_val(self.src)  # 获取源对象的值
        return getattr(val, self.name)  # 返回源对象的属性值


def get_val(capture):
    if isinstance(capture, Capture):
        return capture.execute()  # 如果是捕获对象，则执行其execute方法并返回结果
    elif isinstance(capture, str):
        return f'"{capture}"'  # 如果是字符串，则返回带引号的字符串
    else:
        return capture  # 否则直接返回捕获对象本身


class CaptureInitial(CaptureVariable):
    def __init__(self, schema_df=None):
        new_ctx: Dict[str, List[Any]] = {
            "operations": [],  # 初始化操作列表为空
            "variables": [],  # 初始化变量列表为空
            "schema_df": schema_df,  # 设置模式数据框
        }
        super().__init__(None, new_ctx)  # 调用父类构造函数初始化

        # 设置新的变量名
        self.name = f"input_{self.name}"  


class CaptureDataFrame(CaptureInitial):
    pass  # 捕获数据框类继承初始化捕获类


class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):
    def as_datapipe(self):
        return DataFrameTracedOps(self.ctx["variables"][0].source_datapipe, self)

    def raw_iterator(self):
        return self.as_datapipe().__iter__()

    def __iter__(self):
        return iter(self._dataframes_as_tuples())  # 返回数据框的迭代器
    # 定义一个方法 `batch`，用于批处理数据帧流
    def batch(self, batch_size=10, drop_last: bool = False, wrapper_class=DataChunkDF):
        # 获取每行数据帧并进行拼接，得到一个新的数据帧流 `dp`
        dp = self._dataframes_per_row()._dataframes_concat(batch_size)
        # 将 `dp` 转换为数据管道，并调用其 `batch` 方法进行分批处理
        dp = dp.as_datapipe().batch(1, drop_last=drop_last, wrapper_class=wrapper_class)
        # 标记 `dp` 包含数据帧
        dp._dp_contains_dataframe = True
        # 返回处理后的数据帧流 `dp`
        return dp

    # 定义一个方法 `groupby`，用于按照指定条件对数据帧进行分组
    def groupby(
        self,
        group_key_fn,
        *,
        buffer_size=10000,
        group_size=None,
        guaranteed_group_size=None,
        drop_remaining=False,
    ):
        # 获取每行数据帧，并转换为数据管道，调用其 `groupby` 方法进行分组
        dp = self._dataframes_per_row().as_datapipe().groupby(
            group_key_fn,
            buffer_size=buffer_size,
            group_size=group_size,
            guaranteed_group_size=guaranteed_group_size,
            drop_remaining=drop_remaining,
        )
        # 返回分组后的数据帧流 `dp`
        return dp

    # 定义一个方法 `shuffle`，用于对数据帧进行随机重排
    def shuffle(self, *args, **kwargs):
        # 调用内部方法 `_dataframes_shuffle` 对数据帧进行随机重排
        return self._dataframes_shuffle(*args, **kwargs)

    # 定义一个方法 `filter`，用于对数据帧进行筛选
    def filter(self, *args, **kwargs):
        # 调用内部方法 `_dataframes_filter` 对数据帧进行筛选
        return self._dataframes_filter(*args, **kwargs)

    # 定义一个方法 `collate`，抛出运行时异常，表示无法对未分批的数据帧流进行整合操作
    def collate(self, *args, **kwargs):
        raise RuntimeError("Can't collate unbatched DataFrames stream")

    # 定义 `__getattr__` 方法，用于在未实现的属性调用时进行处理
    def __getattr__(self, attrname):  # ?
        # 如果属性名在未实现的属性列表中，则抛出属性错误异常
        if attrname in UNIMPLEMENTED_ATTR:
            raise AttributeError("Attempting to get ", attrname)
        # 如果属性名在数据管道操作列表中，则返回数据管道对象的对应属性
        if attrname in DATAPIPES_OPS:
            return (self.as_datapipe()).__getattr__(attrname)
        # 否则，调用父类的属性获取方法
        return super().__getattr__(attrname)
# 使用装饰器 @functional_datapipe 将类注册为数据管道的一部分，用于跟踪数据框架操作的函数
@functional_datapipe("trace_as_dataframe")
# 定义一个类 DataFrameTracer，该类同时继承自 CaptureDataFrameWithDataPipeOps 和 IterDataPipe 类（忽略杂项类型检查）
class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):
    # 可选的属性，用于存储数据管道的原始数据来源
    source_datapipe: Optional[Any] = None

    # TODO(VitalyFedyunin): 必须实现数据管道的所有特殊函数

    # 定义一个函数 set_shuffle_settings，接受任意参数但不执行具体操作
    def set_shuffle_settings(self, *args, **kwargs):
        pass

    # 定义一个函数 is_shardable，返回 False，表示数据不可分片
    def is_shardable(self):
        return False

    # 定义类的初始化函数 __init__，接受 source_datapipe 和 schema_df（可选参数，默认为 None）
    def __init__(self, source_datapipe, schema_df=None):
        # 将 source_datapipe 存储到实例属性中
        self.source_datapipe = source_datapipe
        # 如果 schema_df 为 None，则从 source_datapipe 中获取一个数据框架
        if schema_df is None:
            schema_df = next(iter(self.source_datapipe))
        # 调用父类 CaptureDataFrameWithDataPipeOps 的初始化函数，传入 schema_df
        super().__init__(schema_df=schema_df)
```