# `.\pytorch\torch\utils\benchmark\utils\valgrind_wrapper\timer_interface.py`

```
"""Intermediate layer between `Timer` and `valgrind`."""
# 引入必要的库和模块
import collections  # 提供了额外的数据结构
import enum  # 支持定义枚举类型的标准库
import dataclasses  # 提供了数据类的支持
import itertools as it  # 提供了高效的迭代工具
import os  # 提供了与操作系统交互的功能
import pickle  # 提供了对象序列化和反序列化的功能
import re  # 提供了正则表达式的功能
import shutil  # 提供了文件和目录操作的高级函数
import subprocess  # 提供了创建和管理子进程的功能
import sys  # 提供了与 Python 解释器进行交互的功能
import textwrap  # 提供了文本格式化的功能
from typing import (
    cast, Any, Callable, DefaultDict, Dict, Iterator, List, NamedTuple,
    Optional, Tuple, Union, TYPE_CHECKING
)

import torch  # 引入了 PyTorch 深度学习库
from torch.utils.benchmark.utils import common, cpp_jit  # 引入了 PyTorch 的基准测试工具相关模块
from torch.utils.benchmark.utils._stubs import CallgrindModuleType  # 引入了 Callgrind 模块类型的存根
import operator  # 提供了 Python 内置的运算符函数

__all__ = ["FunctionCount", "FunctionCounts", "CallgrindStats", "CopyIfCallgrind"]

# 定义 CompletedProcessType 类型，根据是否是类型检查阶段分别赋值
if TYPE_CHECKING:
    CompletedProcessType = subprocess.CompletedProcess[str]
else:
    CompletedProcessType = subprocess.CompletedProcess

# 定义 NamedTuple FunctionCount，包含计数和函数名两个字段
class FunctionCount(NamedTuple):
    # TODO(#105471): Rename the count field
    count: int  # type: ignore[assignment]  # 记录函数调用次数的计数字段
    function: str  # 记录函数名的字段

# 定义数据类 FunctionCounts，用于操作 Callgrind 结果
@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class FunctionCounts:
    """Container for manipulating Callgrind results.

    It supports:
        1) Addition and subtraction to combine or diff results.
        2) Tuple-like indexing.
        3) A `denoise` function which strips CPython calls which are known to
           be non-deterministic and quite noisy.
        4) Two higher order methods (`filter` and `transform`) for custom
           manipulation.
    """
    _data: Tuple[FunctionCount, ...]  # 存储 FunctionCount 元组的私有字段
    inclusive: bool  # 指示结果是否包含所有信息的标志
    truncate_rows: bool = True  # 是否截断输出行的标志，默认为 True

    # For normal use, torch._tensor_str.PRINT_OPTS.linewidth determines
    # the print settings. This is simply to allow hermetic unit tests.
    _linewidth: Optional[int] = None  # 控制输出宽度的可选参数，默认为 None

    # 实现迭代器方法，遍历 _data 中的 FunctionCount
    def __iter__(self) -> Iterator[FunctionCount]:
        yield from self._data

    # 返回 _data 中 FunctionCount 的数量
    def __len__(self) -> int:
        return len(self._data)

    # 实现索引访问方法，返回指定位置的 FunctionCount 或 FunctionCounts 对象
    def __getitem__(self, item: Any) -> Union[FunctionCount, "FunctionCounts"]:
        data: Union[FunctionCount, Tuple[FunctionCount, ...]] = self._data[item]
        return (
            FunctionCounts(cast(Tuple[FunctionCount, ...], data), self.inclusive, truncate_rows=False)
            if isinstance(data, tuple) else data
        )

    # 返回对象的字符串表示形式，包括格式化后的计数和函数名
    def __repr__(self) -> str:
        count_len = 0
        for c, _ in self:
            # Account for sign in string length.
            count_len = max(count_len, len(str(c)) + int(c < 0))

        lines = []
        linewidth = self._linewidth or torch._tensor_str.PRINT_OPTS.linewidth
        fn_str_len = max(linewidth - count_len - 4, 40)
        for c, fn in self:
            if len(fn) > fn_str_len:
                left_len = int((fn_str_len - 5) // 2)
                fn = fn[:left_len] + " ... " + fn[-(fn_str_len - left_len - 5):]
            lines.append(f"  {c:>{count_len}}  {fn}")

        if self.truncate_rows and len(lines) > 18:
            lines = lines[:9] + ["...".rjust(count_len + 2)] + lines[-9:]

        if not self.inclusive:
            lines.extend(["", f"Total: {self.sum()}"])

        return "\n".join([super().__repr__()] + lines)
    # 定义特殊方法 __add__，用于两个 FunctionCounts 对象的加法操作
    def __add__(
        self,
        other: "FunctionCounts",
    ) -> "FunctionCounts":
        # 调用 _merge 方法，传入 other 和一个 lambda 函数，lambda 函数为恒等函数，即返回原值
        return self._merge(other, lambda c: c)

    # 定义特殊方法 __sub__，用于两个 FunctionCounts 对象的减法操作
    def __sub__(
        self,
        other: "FunctionCounts",
    ) -> "FunctionCounts":
        # 调用 _merge 方法，传入 other 和 operator.neg 函数，用于对 c 取反
        return self._merge(other, operator.neg)

    # 定义特殊方法 __mul__，用于 FunctionCounts 对象与整数或浮点数的乘法操作
    def __mul__(self, other: Union[int, float]) -> "FunctionCounts":
        # 使用字典推导式，对每个函数名 fn 对应的计数值 c 进行乘法运算，生成新的字典
        return self._from_dict({
            fn: int(c * other) for c, fn in self._data
        }, self.inclusive)

    # 定义 transform 方法，将给定的 map_fn 应用于所有函数名
    def transform(self, map_fn: Callable[[str], str]) -> "FunctionCounts":
        """Apply `map_fn` to all of the function names.

        This can be used to regularize function names (e.g. stripping irrelevant
        parts of the file path), coalesce entries by mapping multiple functions
        to the same name (in which case the counts are added together), etc.
        """
        # 使用 defaultdict 初始化一个 counts 字典，存储映射后的函数名及其对应的计数
        counts: DefaultDict[str, int] = collections.defaultdict(int)
        # 遍历 self._data 中的每个计数 c 和函数名 fn
        for c, fn in self._data:
            # 将 map_fn 应用于函数名 fn，更新 counts 中相应函数名的计数
            counts[map_fn(fn)] += c

        # 使用 _from_dict 方法，基于 counts 字典创建新的 FunctionCounts 对象并返回
        return self._from_dict(counts, self.inclusive)

    # 定义 filter 方法，根据给定的 filter_fn 过滤函数名
    def filter(self, filter_fn: Callable[[str], bool]) -> "FunctionCounts":
        """Keep only the elements where `filter_fn` applied to function name returns True."""
        # 使用列表推导式过滤出满足 filter_fn 条件的元素，创建新的 FunctionCounts 对象并返回
        return FunctionCounts(tuple(i for i in self if filter_fn(i.function)), self.inclusive)

    # 定义 sum 方法，计算所有函数计数的总和并返回
    def sum(self) -> int:
        return sum(c for c, _ in self)

    # 定义 denoise 方法，移除已知的噪声指令对应的函数名及其计数
    def denoise(self) -> "FunctionCounts":
        """Remove known noisy instructions.

        Several instructions in the CPython interpreter are rather noisy. These
        instructions involve unicode to dictionary lookups which Python uses to
        map variable names. FunctionCounts is generally a content agnostic
        container, however this is sufficiently important for obtaining
        reliable results to warrant an exception."""
        # 使用 filter 方法，排除包含 "dictobject.c:lookdict_unicode" 的函数名，并返回新的 FunctionCounts 对象
        return self.filter(lambda fn: "dictobject.c:lookdict_unicode" not in fn)

    # 定义 _merge 方法，用于合并两个 FunctionCounts 对象的计数信息
    def _merge(
        self,
        second: "FunctionCounts",
        merge_fn: Callable[[int], int]
    ) -> "FunctionCounts":
        # 断言 self 和 second 的 inclusive 属性相同
        assert self.inclusive == second.inclusive, "Cannot merge inclusive and exclusive counts."
        # 使用 defaultdict 初始化一个 counts 字典，存储合并后的函数名及其计数
        counts: DefaultDict[str, int] = collections.defaultdict(int)
        # 遍历 self 中的每个计数 c 和函数名 fn，将计数值加入 counts
        for c, fn in self:
            counts[fn] += c

        # 遍历 second 中的每个计数 c 和函数名 fn，根据 merge_fn 对 c 进行处理后加入 counts
        for c, fn in second:
            counts[fn] += merge_fn(c)

        # 使用 _from_dict 方法，基于 counts 字典创建新的 FunctionCounts 对象并返回
        return self._from_dict(counts, self.inclusive)

    # 定义 _from_dict 静态方法，根据给定的 counts 字典和 inclusive 属性创建 FunctionCounts 对象
    @staticmethod
    def _from_dict(counts: Dict[str, int], inclusive: bool) -> "FunctionCounts":
        # 使用生成器表达式创建 flat_counts 元组，每个元素是 FunctionCount 对象，按计数值逆序排序
        flat_counts = (FunctionCount(c, fn) for fn, c in counts.items() if c)
        # 使用 tuple 将 flat_counts 转换为元组，并传入 FunctionCounts 构造函数，返回新的 FunctionCounts 对象
        return FunctionCounts(tuple(sorted(flat_counts, reverse=True)), inclusive)
@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class CallgrindStats:
    """Top level container for Callgrind results collected by Timer.
    
    Manipulation is generally done using the FunctionCounts class, which is
    obtained by calling `CallgrindStats.stats(...)`. Several convenience
    methods are provided as well; the most significant is
    `CallgrindStats.as_standardized()`.
    """
    task_spec: common.TaskSpec  # Task specification associated with the CallgrindStats instance
    number_per_run: int  # Number of runs per measurement
    built_with_debug_symbols: bool  # Indicates if PyTorch was built with debug symbols
    baseline_inclusive_stats: FunctionCounts  # Inclusive function counts for baseline measurements
    baseline_exclusive_stats: FunctionCounts  # Exclusive function counts for baseline measurements
    stmt_inclusive_stats: FunctionCounts  # Inclusive function counts for statement measurements
    stmt_exclusive_stats: FunctionCounts  # Exclusive function counts for statement measurements
    stmt_callgrind_out: Optional[str]  # Optional output path for Callgrind statement results
    
    def __repr__(self) -> str:
        newline = "\n"  # `\` cannot appear in fstring code section.
        base_stats = self.baseline_exclusive_stats
        output = f"""
{super().__repr__()}
{self.task_spec.summarize()}
  {'':>25}All{'':>10}Noisy symbols removed
    Instructions: {self.counts(denoise=False):>12}{'':>15}{self.counts(denoise=True):>12}
    Baseline:     {base_stats.sum():>12}{'':>15}{base_stats.denoise().sum():>12}
{self.number_per_run} runs per measurement, {self.task_spec.num_threads} thread{'s' if self.task_spec.num_threads > 1 else ''}
""".strip()
        if not self.built_with_debug_symbols:
            output += textwrap.dedent("""
            Warning: PyTorch was not built with debug symbols.
                     Source information may be limited. Rebuild with
                     REL_WITH_DEB_INFO=1 for more detailed results.""")
        return output

    def stats(self, inclusive: bool = False) -> FunctionCounts:
        """Returns detailed function counts.

        Conceptually, the FunctionCounts returned can be thought of as a tuple
        of (count, path_and_function_name) tuples.

        `inclusive` matches the semantics of callgrind. If True, the counts
        include instructions executed by children. `inclusive=True` is useful
        for identifying hot spots in code; `inclusive=False` is useful for
        reducing noise when diffing counts from two different runs. (See
        CallgrindStats.delta(...) for more details)
        """
        return self.stmt_inclusive_stats if inclusive else self.stmt_exclusive_stats

    def counts(self, *, denoise: bool = False) -> int:
        """Returns the total number of instructions executed.

        See `FunctionCounts.denoise()` for an explanation of the `denoise` arg.
        """
        stats = self.stmt_exclusive_stats
        return (stats.denoise() if denoise else stats).sum()

    # FIXME: Once 3.7 is the minimum version, type annotate `other` per PEP 563
    def delta(
        self,
        other: "CallgrindStats",
        inclusive: bool = False,
    ) -> FunctionCounts:
        """
        返回一个 FunctionCounts 对象，表示两组计数的差异。

        收集指令计数的一个常见原因是确定特定更改对执行某个工作单元所需指令数量的影响。
        如果更改增加了该数量，下一个逻辑问题是“为什么”。这通常涉及查看代码的哪一部分增加了指令计数。
        此函数自动化此过程，以便可以轻松地在包含和不包含的基础上对计数进行差异化分析。
        """
        # 返回当前对象的统计信息和另一个对象统计信息的差异，可以选择返回包含和不包含的计数
        return self.stats(inclusive=inclusive) - other.stats(inclusive=inclusive)
    def`
    def as_standardized(self) -> "CallgrindStats":
        """Strip library names and some prefixes from function strings.

        When comparing two different sets of instruction counts, one stumbling
        block can be path prefixes. Callgrind includes the full filepath
        when reporting a function (as it should). However, this can cause
        issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, which
        can result in something resembling::

            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

        Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivalent call sites
        when diffing.
        """
        def strip(stats: FunctionCounts) -> FunctionCounts:
            transforms = (
                # PyTorch may have been built in different locations.
                (r"^.+build/\.\./", "build/../"),
                (r"^.+/" + re.escape("build/aten/"), "build/aten/"),

                # "Python" and "Objects" come from CPython.
                (r"^.+/" + re.escape("Python/"), "Python/"),
                (r"^.+/" + re.escape("Objects/"), "Objects/"),

                # Strip library name. e.g. `libtorch.so`
                (r"\s\[.+\]$", ""),
            )

            for before, after in transforms:
                stats = stats.transform(lambda fn: re.sub(before, after, fn))

            return stats

        return CallgrindStats(
            task_spec=self.task_spec,
            number_per_run=self.number_per_run,
            built_with_debug_symbols=self.built_with_debug_symbols,
            baseline_inclusive_stats=strip(self.baseline_inclusive_stats),
            baseline_exclusive_stats=strip(self.baseline_exclusive_stats),
            stmt_inclusive_stats=strip(self.stmt_inclusive_stats),
            stmt_exclusive_stats=strip(self.stmt_exclusive_stats),

            # `as_standardized` will change symbol names, so the contents will
            # no longer map directly to `callgrind.out`
            stmt_callgrind_out=None,
        )
class Serialization(enum.Enum):
    # 定义枚举类型 Serialization，包括 PICKLE、TORCH 和 TORCH_JIT 三种序列化方式
    PICKLE = 0
    TORCH = 1
    TORCH_JIT = 2


_GLOBALS_ALLOWED_TYPES: Dict[Serialization, Tuple[Any, ...]] = {
    Serialization.PICKLE: (str, bytes, bool, int, float, complex),
    Serialization.TORCH_JIT: (torch.jit.ScriptFunction, torch.jit.ScriptModule),
    Serialization.TORCH: (torch.nn.Module,),
    # 全局变量 _GLOBALS_ALLOWED_TYPES 定义了不同序列化方式支持的数据类型
}


class CopyIfCallgrind:
    """Signal that a global may be replaced with a deserialized copy.

    See `GlobalsBridge` for why this matters.
    """
    def __init__(self, value: Any, *, setup: Optional[str] = None):
        for method, supported_types in _GLOBALS_ALLOWED_TYPES.items():
            # 遍历 _GLOBALS_ALLOWED_TYPES 中的每个序列化方式和支持的数据类型
            if any(isinstance(value, t) for t in supported_types):
                self._value: Any = value
                self._setup: Optional[str] = setup
                self._serialization: Serialization = method
                break
        else:
            # 如果传入的值的类型不在支持的类型列表中，抛出 ValueError 异常
            supported_str = "\n".join([
                getattr(t, "__name__", repr(t))
                for t in it.chain(_GLOBALS_ALLOWED_TYPES.values())])

            raise ValueError(
                f"Unsupported type: {type(value)}\n"
                f"`collect_callgrind` restricts globals to the following types:\n"
                f"{textwrap.indent(supported_str, '  ')}"
            )

    @property
    def value(self) -> Any:
        return self._value

    @property
    def setup(self) -> Optional[str]:
        return self._setup

    @property
    def serialization(self) -> Serialization:
        return self._serialization

    @staticmethod
    def unwrap_all(globals: Dict[str, Any]) -> Dict[str, Any]:
        # 静态方法 unwrap_all 将 globals 中的 CopyIfCallgrind 实例解包成其原始值
        return {
            k: (v.value if isinstance(v, CopyIfCallgrind) else v)
            for k, v in globals.items()
        }


class GlobalsBridge:
    """Handle the transfer of (certain) globals when collecting Callgrind statistics.

    Key takeaway: Any globals passed must be wrapped in `CopyIfCallgrind` to
                  work with `Timer.collect_callgrind`.

    Consider the following code snippet:
    ```
        import pickle
        import timeit

        class Counter:
            value = 0

            def __call__(self):
                self.value += 1

        counter = Counter()
        timeit.Timer("counter()", globals={"counter": counter}).timeit(10)
        print(counter.value)  # 10

        timeit.Timer(
            "counter()",
            globals={"counter": pickle.loads(pickle.dumps(counter))}
        ).timeit(20)
        print(counter.value)  # Still 10
    ```

    In the first case, `stmt` is executed using the objects in `globals`;
    however, the addition of serialization and deserialization changes the
    semantics and may meaningfully change behavior.

    This is a practical consideration when collecting Callgrind statistics.
    Unlike `exec` based execution (which `timeit` uses under the hood) which
    can share in-memory data structures with the caller, Callgrind collection
    # GlobalsBridge 类负责处理在收集 Callgrind 统计数据时（某些）全局变量的传递
    """
    requires an entirely new process in order to run under Valgrind. This means
    that any data structures used for statement execution will have to be
    serialized and deserialized in the subprocess.

    In order to avoid surprising semantics from (user invisible) process
    boundaries, what can be passed through `globals` is severely restricted
    for `Timer.collect_callgrind`. It is expected that most setup should be
    achievable (albeit perhaps less ergonomically) by passing a `setup`
    string.

    There are, however, exceptions. One such class are TorchScripted functions.
    Because they require a concrete file with source code it is not possible
    to define them using a `setup` string. Another group are torch.nn.Modules,
    whose construction can be complex and prohibitively cumbersome to coerce
    into a `setup` string. Finally, most builtin types are sufficiently well
    behaved and sufficiently common to warrant allowing as well. (e.g.
    `globals={"n": 1}` is very convenient.)

    Fortunately, all have well defined serialization semantics. This class
    is responsible for enabling the Valgrind subprocess to use elements in
    `globals` so long as they are an allowed type.

    Caveats:
        The user is required to acknowledge this serialization by wrapping
        elements in `globals` with `CopyIfCallgrind`.

        While ScriptFunction and ScriptModule are expected to save and load
        quite robustly, it is up to the user to ensure that an nn.Module can
        un-pickle successfully.

        `torch.Tensor` and `np.ndarray` are deliberately excluded. The
        serialization/deserialization process perturbs the representation of a
        tensor in ways that could result in incorrect measurements. For example,
        if a tensor lives in pinned CPU memory, this fact would not be preserved
        by a dump, and that will in turn change the performance of certain CUDA
        operations.
    """

    # 初始化方法，接受全局变量字典和数据目录
    def __init__(self, globals: Dict[str, Any], data_dir: str) -> None:
        # 初始化全局变量字典为一个空字典，存储需要传递给Valgrind子进程的元素
        self._globals: Dict[str, CopyIfCallgrind] = {}
        # 设置数据目录属性
        self._data_dir = data_dir
        # 如果数据目录不存在，则创建之
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        # 检查是否尝试替换torch模块，这是不支持的
        if globals.get("torch", torch) is not torch:
            raise ValueError("`collect_callgrind` does not support mocking out `torch`.")

        # 遍历传入的全局变量字典
        for name, value in globals.items():
            # 跳过torch和__builtins__，因为它们有特殊处理或被添加
            if name in ("torch", "__builtins__"):
                continue

            # 要求所有非基本类型的全局变量必须使用CopyIfCallgrind进行封装
            if not isinstance(value, CopyIfCallgrind):
                raise ValueError(
                    "`collect_callgrind` requires that globals be wrapped in "
                    "`CopyIfCallgrind` so that serialization is explicit."
                )

            # 将合法的全局变量存储在实例的_globals字典中
            self._globals[name] = value
    # 定义一个方法，返回一个字符串
    def construct(self) -> str:
        # 创建一个空列表，用于存储加载数据的代码行
        load_lines = []
        # 遍历全局变量字典中的每个项
        for name, wrapped_value in self._globals.items():
            # 如果包装值的设置不为None，则添加设置代码到加载行列表中
            if wrapped_value.setup is not None:
                load_lines.append(textwrap.dedent(wrapped_value.setup))

            # 如果包装值的序列化方式是PICKLE
            if wrapped_value.serialization == Serialization.PICKLE:
                # 构造pickle文件的路径
                path = os.path.join(self._data_dir, f"{name}.pkl")
                # 添加打开pickle文件并加载数据的代码行到加载行列表中
                load_lines.append(
                    f"with open({repr(path)}, 'rb') as f:\n    {name} = pickle.load(f)")
                # 使用pickle将值保存到文件中
                with open(path, "wb") as f:
                    pickle.dump(wrapped_value.value, f)

            # 如果包装值的序列化方式是TORCH
            elif wrapped_value.serialization == Serialization.TORCH:
                # 构造torch文件的路径
                path = os.path.join(self._data_dir, f"{name}.pt")
                # 添加加载torch文件数据的代码行到加载行列表中
                load_lines.append(f"{name} = torch.load({repr(path)})")
                # 使用torch将值保存到文件中
                torch.save(wrapped_value.value, path)

            # 如果包装值的序列化方式是TORCH_JIT
            elif wrapped_value.serialization == Serialization.TORCH_JIT:
                # 构造torch jit文件的路径
                path = os.path.join(self._data_dir, f"{name}.pt")
                # 添加加载torch jit文件数据的代码行到加载行列表中
                load_lines.append(f"{name} = torch.jit.load({repr(path)})")
                # 使用torch jit将值保存到文件中
                with open(path, "wb") as f:
                    torch.jit.save(wrapped_value.value, f)  # type: ignore[no-untyped-call]

            # 如果序列化方式未知，则抛出未实现错误
            else:
                raise NotImplementedError(
                    f"Unknown serialization method: {wrapped_value.serialization}")

        # 将所有加载行连接成一个字符串并返回
        return "\n".join(load_lines)
class _ValgrindWrapper:
    # ValgrindWrapper 类的构造函数，初始化成员变量
    def __init__(self) -> None:
        # _bindings_module 属性用于存储 Callgrind 模块的绑定对象，初始为 None
        self._bindings_module: Optional[CallgrindModuleType] = None
        # 定义 Valgrind 所需的符号列表
        valgrind_symbols = (
            "_valgrind_supported_platform",
            "_valgrind_toggle",
            "_valgrind_toggle_and_dump_stats",
        )
        # 检查 torch._C 中是否存在所有的 Valgrind 符号
        if all(hasattr(torch._C, symbol) for symbol in valgrind_symbols):
            # 如果存在，则设置 _supported_platform 为 True
            self._supported_platform: bool = torch._C._valgrind_supported_platform()

        else:
            # 如果不存在，打印警告信息并尝试 JIT 绑定
            print("Callgrind bindings are not present in `torch._C`. JIT-ing bindings.")
            # 获取兼容的 JIT 绑定模块
            self._bindings_module = cpp_jit.get_compat_bindings()
            # 断言检查是否所有 Valgrind 符号都存在于绑定模块中
            assert all(hasattr(self._bindings_module, symbol) for symbol in valgrind_symbols)
            # 设置 _supported_platform 为 JIT 绑定模块的结果
            self._supported_platform = self._bindings_module._valgrind_supported_platform()

        # _commands_available 属性用于存储可用的 Valgrind 命令的状态字典
        self._commands_available: Dict[str, bool] = {}
        if self._supported_platform:
            # 只有在支持的平台上进行检查
            for cmd in ("valgrind", "callgrind_control", "callgrind_annotate"):
                # 检查命令是否可用，存储结果到 _commands_available 字典中
                self._commands_available[cmd] = not subprocess.run(
                    ["which", cmd],
                    capture_output=True,
                    check=False,
                ).returncode

        # _build_type 属性用于存储构建类型信息，默认为 None
        self._build_type: Optional[str] = None
        # 从 torch 的配置信息中查找 BUILD_TYPE 的值，并存储到 _build_type 中
        build_search = re.search("BUILD_TYPE=(.+),", torch.__config__.show())  # type: ignore[no-untyped-call]
        if build_search is not None:
            self._build_type = build_search.groups()[0].split(",")[0]

    # _validate 方法用于验证 Valgrind 支持及相关命令是否可用
    def _validate(self) -> None:
        # 如果当前平台不支持 Valgrind，则抛出 OSError
        if not self._supported_platform:
            raise OSError("Valgrind is not supported on this platform.")

        # 检查是否有缺失的 Valgrind 命令，如果有则抛出 OSError
        missing_cmds = [cmd for cmd, available in self._commands_available.items() if not available]
        if missing_cmds:
            raise OSError("Missing: " + ", ".join(missing_cmds))

    # collect_callgrind 方法用于收集 Callgrind 数据
    def collect_callgrind(
        self,
        task_spec: common.TaskSpec,
        globals: Dict[str, Any],
        *,
        number: int,
        repeats: int,
        collect_baseline: bool,
        is_python: bool,
        retain_out_file: bool,
        # 方法未完全列出，需要继续补充
    ) -> Tuple[CallgrindStats, ...]:
        """Collect stats, and attach a reference run which can be used to filter interpreter overhead."""
        # 确保对象状态有效
        self._validate()
        # 如果是 Python 解释器或者不收集基准数据，则断言为真
        assert is_python or not collect_baseline

        # 调用内部方法 `_invoke`，并将结果解包
        *task_stats, baseline_stats = self._invoke(
            task_spec=task_spec,
            globals=globals,
            number=number,
            repeats=repeats,
            collect_baseline=collect_baseline,
            is_python=is_python,
            retain_out_file=retain_out_file,
        )
        # 断言任务统计数据的长度等于重复次数
        assert len(task_stats) == repeats

        # 返回一个元组，每个元素为一个 `CallgrindStats` 对象
        return tuple(
            CallgrindStats(
                task_spec=task_spec,
                number_per_run=number,
                built_with_debug_symbols=self._build_type == "RelWithDebInfo",
                baseline_inclusive_stats=baseline_stats[0],
                baseline_exclusive_stats=baseline_stats[1],
                stmt_inclusive_stats=stmt_inclusive_stats,
                stmt_exclusive_stats=stmt_exclusive_stats,
                stmt_callgrind_out=out_contents,
            )
            # 对每个任务统计数据进行迭代，生成对应的 `CallgrindStats` 对象
            for stmt_inclusive_stats, stmt_exclusive_stats, out_contents in task_stats
        )

    def _invoke(
        self,
        *,
        task_spec: common.TaskSpec,
        globals: Dict[str, Any],
        number: int,
        repeats: int,
        collect_baseline: bool,
        is_python: bool,
        retain_out_file: bool,
    ):
        """Invoke a task for profiling."""
        # 省略 `_invoke` 方法的实现细节
        pass

    @staticmethod
    def _construct_script(
        task_spec: common.TaskSpec,
        globals: GlobalsBridge,
        *,
        number: int,
        repeats: int,
        collect_baseline: bool,
        error_log: str,
        stat_log: str,
        bindings: Optional[CallgrindModuleType],
    ):
        """Constructs a script to run for profiling."""
        # 省略 `_construct_script` 方法的实现细节
        pass
# 定义一个全局变量 CALLGRIND_SINGLETON，用于存储 Valgrind 包装器的单例对象，初始值为 None
CALLGRIND_SINGLETON: Optional[_ValgrindWrapper] = None

# 定义一个函数 wrapper_singleton，用于获取 Valgrind 包装器的单例对象
def wrapper_singleton() -> _ValgrindWrapper:
    # 声明使用全局变量 CALLGRIND_SINGLETON
    global CALLGRIND_SINGLETON
    # 如果 CALLGRIND_SINGLETON 尚未初始化（即为 None），则创建一个 _ValgrindWrapper 对象赋值给它
    if CALLGRIND_SINGLETON is None:
        CALLGRIND_SINGLETON = _ValgrindWrapper()
    # 返回已经存在或新创建的 CALLGRIND_SINGLETON 对象作为单例
    return CALLGRIND_SINGLETON
```