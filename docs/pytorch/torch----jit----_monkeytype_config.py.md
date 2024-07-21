# `.\pytorch\torch\jit\_monkeytype_config.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和库
import inspect  # 导入 inspect 模块，用于获取对象信息
import sys  # 导入 sys 模块，提供与 Python 解释器交互的功能
import typing  # 导入 typing 模块，支持类型提示
from collections import defaultdict  # 导入 defaultdict 类，实现默认值为零的字典
from pathlib import Path  # 导入 Path 类，用于操作文件和目录路径
from types import CodeType  # 导入 CodeType 类，表示 Python 字节码对象
from typing import Dict, Iterable, List, Optional  # 导入多个类型提示，用于静态类型检查

import torch  # 导入 torch 库，用于深度学习任务

# 检查是否安装了 MonkeyType
_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # 尝试导入 monkeytype 模块，用于运行时类型跟踪
    from monkeytype import trace as monkeytype_trace  # 导入 monkeytype_trace 函数，用于跟踪函数调用
    from monkeytype.config import _startswith, LIB_PATHS  # 导入 _startswith 和 LIB_PATHS 常量
    from monkeytype.db.base import (  # 导入数据库相关类
        CallTraceStore,  # 调用跟踪存储类
        CallTraceStoreLogger,  # 调用跟踪存储日志类
        CallTraceThunk,  # 调用跟踪 thunk 类
    )
    from monkeytype.tracing import CallTrace, CodeFilter  # 导入调用跟踪和代码过滤器类
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False  # 如果导入失败，设置 MonkeyType 未安装标志为 False


# 检查类是否定义在 torch.* 模块中
def is_torch_native_class(cls):
    if not hasattr(cls, "__module__"):  # 如果类没有 __module__ 属性，返回 False
        return False

    parent_modules = cls.__module__.split(".")  # 获取类的模块路径，并按 . 分割为列表
    if not parent_modules:  # 如果列表为空，返回 False
        return False

    root_module = sys.modules.get(parent_modules[0])  # 获取顶级模块
    return root_module is torch  # 返回顶级模块是否为 torch 的比较结果


def get_type(type):
    """将给定类型转换为 TorchScript 可接受的格式。"""
    if isinstance(type, str):  # 如果类型是字符串，直接返回
        return type
    elif inspect.getmodule(type) == typing:
        # 如果类型来自 typing 模块，则移除 `typing.` 前缀，因为 TorchScript 不接受带前缀的 typing 类型
        type_to_string = str(type)
        return type_to_string.replace(type.__module__ + ".", "")
    elif is_torch_native_class(type):
        # 如果类型是 torch 模块的子类型，TorchScript 需要完全限定名称
        return type.__module__ + "." + type.__name__
    else:
        # 对于其他类型，使用类型的名称
        return type.__name__


def get_optional_of_element_type(types):
    """从合并的类型中提取元素类型，并返回 `Optional[element type]`。"""
    elem_type = types[1] if type(None) == types[0] else types[0]  # 获取元素类型
    elem_type = get_type(elem_type)  # 转换元素类型为 TorchScript 可接受格式

    # Optional 类型在 TorchScript 中内部转换为 Union[type, NoneType]，目前不支持，因此以字符串表示可选类型
    return "Optional[" + elem_type + "]"


def get_qualified_name(func):
    return func.__qualname__  # 返回函数的限定名


if _IS_MONKEYTYPE_INSTALLED:
    # 如果安装了 MonkeyType，则执行以下内容
    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""

        def __init__(self, store: CallTraceStore):
            super().__init__(store)

        def log(self, trace: CallTrace) -> None:
            self.traces.append(trace)



    class JitTypeTraceStore(CallTraceStore):
        def __init__(self):
            super().__init__()
            # A dictionary keeping all collected CallTrace
            # key is fully qualified name of called function
            # value is list of all CallTrace
            self.trace_records: Dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]):
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: Optional[str] = None,
            limit: int = 2000,
        ) -> List[CallTraceThunk]:
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> Dict:
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments.
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> Dict:
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            for arg, types in all_args.items():
                types = list(types)
                type_length = len(types)
                if type_length == 2 and type(None) in types:
                    # TODO: To remove this check once Union suppport in TorchScript lands.
                    all_args[arg] = get_optional_of_element_type(types)
                elif type_length > 1:
                    all_args[arg] = "Any"
                elif type_length == 1:
                    all_args[arg] = get_type(types[0])
            return all_args

        def get_args_types(self, qualified_name: str) -> Dict:
            return self.consolidate_types(qualified_name)
    # 定义 JitTypeTraceConfig 类，继承自 monkeytype.config.Config 类
    class JitTypeTraceConfig(monkeytype.config.Config):
        
        # 初始化方法，接受一个 JitTypeTraceStore 类型的参数 s，并调用父类的初始化方法
        def __init__(self, s: JitTypeTraceStore):
            super().__init__()
            # 将参数 s 赋值给实例变量 self.s

        # 返回一个 JitCallTraceStoreLogger 对象，该对象用于将日志记录到配置的追踪存储中
        def trace_logger(self) -> JitTypeTraceStoreLogger:
            """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
            return JitTypeTraceStoreLogger(self.trace_store())

        # 返回配置的追踪存储对象，类型为 CallTraceStore
        def trace_store(self) -> CallTraceStore:
            return self.s

        # 返回一个可选的代码过滤器对象，类型为 Optional[CodeFilter]
        def code_filter(self) -> Optional[CodeFilter]:
            return jit_code_filter
else:
    # 当未安装 MonkeyType 时，提供以下类的虚拟定义
    # 用于下列类。
    class JitTypeTraceStoreLogger:  # type: ignore[no-redef]
        def __init__(self):
            pass

    class JitTypeTraceStore:  # type: ignore[no-redef]
        def __init__(self):
            self.trace_records = None

    class JitTypeTraceConfig:  # type: ignore[no-redef]
        def __init__(self):
            pass

    monkeytype_trace = None  # type: ignore[assignment]  # noqa: F811


def jit_code_filter(code: CodeType) -> bool:
    """Codefilter for Torchscript to trace forward calls.

    Torchscript 用于跟踪前向调用的代码过滤器。

    自定义的 CodeFilter 在创建 FX Traced 前向调用时是必需的。
    FX Traced 前向调用具有 `code.co_filename` 以 '<' 开头的特点，
    默认代码过滤器排除了 stdlib 和 site-packages 的跟踪。
    由于我们需要跟踪所有前向调用，这个自定义代码过滤器检查 `code.co_name` 是否为 'forward'，
    并启用所有这类调用的跟踪。
    该代码过滤器类似于 MonkeyType 的默认代码过滤器，并排除了 stdlib 和 site-packages 的跟踪。
    """
    # 过滤掉没有源文件的代码，并排除这种检查对于 'forward' 调用。
    if code.co_name != "forward" and (
        not code.co_filename or code.co_filename[0] == "<"
    ):
        return False

    filename = Path(code.co_filename).resolve()
    return not any(_startswith(filename, lib_path) for lib_path in LIB_PATHS)
```