# `.\pytorch\torch\testing\_internal\logging_tensor.py`

```py
# mypy: ignore-errors

# 导入PyTorch库
import torch
# 导入tree_map函数，该函数似乎来自私有模块utils._pytree
from torch.utils._pytree import tree_map
# 导入类型提示
from typing import Iterator, List, Optional
# 导入日志记录模块
import logging
# 导入上下文管理模块
import contextlib
# 导入迭代工具模块
import itertools
# 导入Torch调度模块
from torch.utils._python_dispatch import TorchDispatchMode
# 导入弱引用Tensor字典模块
from torch.utils.weak import WeakTensorKeyDictionary
# 导入函数工具模块
import functools
# 导入性能分析工具模块
from torch._C._profiler import gather_traceback, symbolize_tracebacks

# 创建名为LoggingTensor的日志记录器
logger = logging.getLogger("LoggingTensor")

# 定义数据类型缩写字典
_dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

# LoggingTensor的调用链工作方式的注释
# 1. 调用torch.sin
# 2. 尝试__torch_function__，在LoggingTensor中禁用了torch函数，因此完全绕过它
# 3. 进入调度程序，通过Autograd逐步执行
# 4. 到达Python调度键，调用__torch_dispatch__

# 这个Tensor可以与Autograd一起以两种方式工作：
# - 封装的Tensor不需要梯度。在这种情况下，LoggingTensor可以在构造函数kwarg中要求梯度。
# - 封装的Tensor可以需要梯度。在这种情况下，对封装的Tensor进行Autograd跟踪，LoggingTensor本身不能要求梯度。
# 警告：我们允许这两种可能性用于测试目的。您绝对不应该在单个测试中同时使用两者，否则可能会得到意外的行为。

# TODO: TensorBase应该起作用
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    # 定义上下文为nullcontext
    context = contextlib.nullcontext

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # 封装的张量（LoggingTensor）不应该为目标类保留任何内存，
        # 但是应该仍然声明与之前相同的设备
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: 克隆存储别名
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=kwargs.get("requires_grad", False)
        )
        # ...真正的张量作为tensor的元素来保存
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    # 返回LoggingTensor的字符串表示形式
    def __repr__(self):
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    # 定义一个特殊方法 __torch_dispatch__，用于处理特定的 Torch 类的调度
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 定义一个内部函数 unwrap，用于解包参数中的元素（如果是 Torch 类则解包，否则保持不变）
        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        # 定义一个内部函数 wrap，用于包装参数中的 Torch 张量（如果是 torch.Tensor 则包装成 cls 类，否则保持不变）
        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e

        # 在特定 Torch 类的上下文环境中执行以下代码块
        with cls.context():
            # 使用 tree_map 函数将参数 args 和 kwargs 中的元素分别进行 unwrap 解包，并传递给 func 函数进行处理
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        # 记录日志信息，记录调用的函数模块和名称，以及参数 args、kwargs 和处理后的结果 rs
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)  # noqa: G004
        # 返回处理后的结果 rs
        return rs
class LoggingTensorMode(TorchDispatchMode):
    # 定义一个自定义的 Torch 调度模式，继承自 TorchDispatchMode
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 如果未提供 kwargs，则初始化为空字典
        if kwargs is None:
            kwargs = {}
        # 调用传入的函数 func，记录日志信息，并返回其结果
        rs = func(*args, **kwargs)
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)  # noqa: G004
        return rs

class LoggingTensorReentrant(LoggingTensor):
    # 定义一个支持可重入调度的 LoggingTensor 类，继承自 LoggingTensor

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):
    # 日志处理器类，继承自 logging.Handler
    def __init__(
            self, log_list: List[str], use_shortid_for_all_tensors: bool,
            with_type: bool, tracebacks_list: Optional[List]) -> None:
        # 初始化日志处理器对象
        logging.Handler.__init__(self)
        # 初始化日志列表、是否对所有张量使用短标识符、是否需要类型信息、追踪信息列表
        self.log_list = log_list
        self.use_shortid_for_all_tensors = use_shortid_for_all_tensors
        self.tracebacks_list = tracebacks_list
        self.memo = WeakTensorKeyDictionary()
        self.next_id = 0
        self.with_type = with_type

    def _shortid(self, t: torch.Tensor) -> int:
        # 根据张量 t 返回一个短标识符
        if t not in self.memo:
            self.memo[t] = self.next_id
            self.next_id += 1
        return self.memo[t]

    def _fmt(self, a: object, with_type: bool = False) -> str:
        # 格式化日志信息中的参数 a，根据是否需要类型信息进行处理
        cond_cls = torch.Tensor if self.use_shortid_for_all_tensors else LoggingTensor
        if isinstance(a, cond_cls):
            maybe_type = ""
            if with_type and self.with_type:
                maybe_type = f": {_dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]"
            x = f"${self._shortid(a)}{maybe_type}"
            return x
        else:
            return repr(a)

    def emit(self, record):
        # 发送日志记录
        fmt_args = ", ".join(
            itertools.chain(
                (str(tree_map(self._fmt, a)) for a in record.args[0]),
                (f"{k}={str(tree_map(self._fmt, v))}" for k, v in record.args[1].items()),
            )
        )
        fmt_rets = tree_map(functools.partial(self._fmt, with_type=True), record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')
        if self.tracebacks_list is not None:
            self.tracebacks_list.append(record.traceback)

def log_input(name: str, var: object) -> None:
    # 记录输入变量的日志信息
    logger.info("input", (name,), {}, var)  # noqa: PLE1205

class GatherTraceback(logging.Filter):
    # 日志过滤器类，用于收集异常的回溯信息
    def __init__(self, python=True, script=True, cpp=False):
        self.python = python
        self.script = script
        self.cpp = cpp

    def filter(self, record):
        # 过滤日志记录，并收集异常的回溯信息
        record.traceback = gather_traceback(python=self.python, script=self.script, cpp=self.cpp)
        return True

@contextlib.contextmanager
def capture_logs(is_mode=False, python_tb=False, script_tb=False, cpp_tb=False) -> Iterator[List[str]]:
    # 上下文管理器，用于捕获日志信息
    collect_traceback = python_tb or script_tb or cpp_tb
    log_list: List[str] = []
    tracebacks_list: List[str] = []
    # 创建一个 LoggingTensorHandler 对象，用于将日志记录到 log_list 中，同时记录张量的类型信息
    handler = LoggingTensorHandler(
        log_list,
        with_type=True,
        use_shortid_for_all_tensors=is_mode,
        tracebacks_list=tracebacks_list if collect_traceback else None
    )
    
    # 将 handler 添加到 logger 对象中，使得日志可以通过 handler 处理
    logger.addHandler(handler)
    
    # 设置 logger 的日志级别为 INFO，只输出 INFO 级别及以上的日志消息
    logger.setLevel(logging.INFO)
    
    # 禁止 logger 的日志消息传播给父级 logger
    logger.propagate = False
    
    # 如果需要收集 traceback 信息，则添加 GatherTraceback 过滤器到 logger
    if collect_traceback:
        logger.addFilter(GatherTraceback(python=python_tb, script=script_tb, cpp=cpp_tb))
    
    try:
        # 在 try 块中执行以下逻辑
        if collect_traceback:
            # 如果需要收集 traceback 信息，则在 yield 前返回 log_list 和 tracebacks_list
            yield log_list, tracebacks_list
        else:
            # 否则，只返回 log_list
            yield log_list
    finally:
        # 最终块，无论如何都会执行的清理逻辑
    
        # 将 tracebacks_list 中的 traceback 信息符号化处理
        symbolized_tracebacks = symbolize_tracebacks(tracebacks_list)
    
        # 清空 tracebacks_list 中的内容
        tracebacks_list.clear()
    
        # 将符号化后的 traceback 信息重新加入 tracebacks_list 中
        tracebacks_list.extend(symbolized_tracebacks)
    
        # 从 logger 中移除之前添加的 handler 对象
        logger.removeHandler(handler)
# 使用 @contextlib.contextmanager 装饰器定义一个上下文管理器函数，用于捕获日志并启用 LoggingTensorMode
@contextlib.contextmanager
def capture_logs_with_logging_tensor_mode(python_tb=False, script_tb=False, cpp_tb=False):
    # 在 LoggingTensorMode 的上下文中，同时使用 capture_logs 函数捕获日志
    with LoggingTensorMode(), capture_logs(True, python_tb, script_tb, cpp_tb) as logs:
        # yield 语句用于将 logs 对象返回给调用者
        yield logs
```