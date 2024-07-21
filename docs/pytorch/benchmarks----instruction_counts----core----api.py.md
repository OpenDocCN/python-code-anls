# `.\pytorch\benchmarks\instruction_counts\core\api.py`

```
"""Key enums and structs used to handle data flow within the benchmark."""

# 引入必要的模块和类
import dataclasses  # 用于定义数据类
import enum  # 用于定义枚举类型
import itertools as it  # 用于生成迭代器
import re  # 用于正则表达式匹配
import textwrap  # 用于文本包装
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union  # 导入类型提示相关的模块

from worker.main import WorkerTimerArgs  # 从 worker.main 导入 WorkerTimerArgs 类

# 如果在类型检查模式下，导入特定的 Language 类，否则导入 torch.utils.benchmark 下的 Language 类
if TYPE_CHECKING:
    # Benchmark utils are only partially strict compliant, so MyPy won't follow
    # imports using the public namespace. (Due to an exclusion rule in
    # mypy-strict.ini)
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language

# Note:
#   WorkerTimerArgs is defined in worker.main so that the worker does not
#   depend on any files, including core.api. We mirror it with a public symbol
#   `TimerArgs` for API consistency.
# TimerArgs 类别名，用于保持 API 的一致性，与 WorkerTimerArgs 对象相对应
TimerArgs = WorkerTimerArgs


# 定义运行模式的枚举
class RuntimeMode(enum.Enum):
    EAGER = "Eager"
    JIT = "TorchScript"
    EXPLICIT = ""


# 定义自动微分模式的枚举
class AutogradMode(enum.Enum):
    FORWARD = "Forward"
    FORWARD_BACKWARD = "Forward + Backward"
    EXPLICIT = ""


# dataclass，用于存储 TimerArgs 实例的标签信息
@dataclasses.dataclass(frozen=True)
class AutoLabels:
    """Labels for a TimerArgs instance which are inferred during unpacking."""
    
    runtime: RuntimeMode  # 运行时模式
    autograd: AutogradMode  # 自动微分模式
    language: Language  # 使用的编程语言

    @property
    def as_dict(self) -> Dict[str, str]:
        """Dict representation for CI reporting."""
        # 返回一个用于 CI 报告的字典表示
        return {
            "runtime": self.runtime.value,
            "autograd": self.autograd.value,
            "language": "Python" if self.language == Language.PYTHON else "C++",
        }


# dataclass，用于存储组合的设置信息
@dataclasses.dataclass(frozen=True)
class GroupedSetup:
    """Class for grouping setup configurations."""
    
    py_setup: str = ""  # Python 设置
    cpp_setup: str = ""  # C++ 设置
    global_setup: str = ""  # 全局设置

    def __post_init__(self) -> None:
        """Post initialization method."""
        # 确保各字段都是字符串类型，并去除文本块的缩进
        for field in dataclasses.fields(self):
            assert field.type == str
            value: str = getattr(self, field.name)
            object.__setattr__(self, field.name, textwrap.dedent(value))


# dataclass，用于定义基准测试的组合信息
@dataclasses.dataclass(frozen=True)
class GroupedBenchmark:
    """Base class for defining groups of benchmarks."""
    
    # PyTorch 性能测量的多维度定义
    # 具体接口有：
    #  - `core.api.GroupedStmts`     (init_from_stmts)
    #  - `core.api.GroupedModules`   (init_from_model)
    #  - `core.api.GroupedVariants`  (init_from_variants)
    #
    # PyTorch 性能测量可以在多个维度上进行，包括：
    #  - Python, C++
    #  - Eager, TorchScript
    #  - 单线程, 多线程
    #  - 训练, 推断
    #
    # 定义它们一起，不仅可以清晰、简洁地定义基准测试，还能进行更智能的后处理和分析。
    #
    # 在 PyTorch 中有两种编程习惯。一种是自由形式的代码（所谓的“带梯度的 NumPy”），
    # 另一种是使用 `torch.nn.Module` 组织代码。（这是通过 PyTorch API 公开的常见神经网络层。）
    # 为了支持简单的定义，提供了两种初始化方法：
    #  - `init_from_stmts`
    #  - `init_from_model`
    """
    These attributes define parameters and setup details for benchmarking.

    py_fwd_stmt: Optional[str]
        Python forward statement executed by Timer. Used in benchmarks if defined.

    cpp_fwd_stmt: Optional[str]
        C++ forward statement executed by Timer. Used in benchmarks if defined.

    py_model_setup: Optional[str]
        Python model setup code. Generated if `torchscript` is True using `torch.jit.script`.

    cpp_model_setup: Optional[str]
        C++ model setup code. Generated indirectly via benchmarks.

    inferred_model_setup: bool
        Indicates whether model setup was inferred (`init_from_stmts`).

    setup: GroupedSetup
        Defines initialization parameters for benchmarks in both Python and C++.

    signature_args: Optional[Tuple[str, ...]]
        Arguments of the function signature in benchmarks.

    signature_output: Optional[str]
        Output of the function signature in benchmarks.

    torchscript: bool
        True if TorchScript JIT compilation is required for benchmarks.

    autograd: bool
        True if benchmarks require automatic differentiation (backward pass).

    num_threads: Tuple[int, ...]
        Specifies the number of threads to be used in benchmarks.

    @classmethod
    """
    # 类方法：从自由形式的语句集创建一组基准测试
    # 这种基准测试定义方式类似于使用 Timer 的方式，我们直接执行提供的语句。
    def init_from_stmts(
        cls,
        py_stmt: Optional[str] = None,  # Python 语句字符串，可选
        cpp_stmt: Optional[str] = None,  # C++ 语句字符串，可选
        # 通用的构造函数参数
        setup: GroupedSetup = GroupedSetup(),  # 初始化设置对象，默认为空的分组设置
        signature: Optional[str] = None,  # 函数签名字符串，可选
        torchscript: bool = False,  # 是否使用 TorchScript，默认为 False
        autograd: bool = False,  # 是否使用自动求导，默认为 False
        num_threads: Union[int, Tuple[int, ...]] = 1,  # 线程数，可以是整数或整数元组，默认为 1
    ) -> "GroupedBenchmark":
        """Create a set of benchmarks from free-form statements.

        This method of benchmark definition is analogous to Timer use, where
        we simply execute the provided stmts.
        """
        if py_stmt is not None:
            py_stmt = textwrap.dedent(py_stmt)  # 去除 Python 语句的缩进

        if cpp_stmt is not None:
            cpp_stmt = textwrap.dedent(cpp_stmt)  # 去除 C++ 语句的缩进

        # 解析函数签名，获取参数和输出
        signature_args, signature_output = cls._parse_signature(signature)

        # 如果 torchscript 为真，则从 Python 语句创建模型设置
        py_model_setup = (
            cls._model_from_py_stmt(
                py_stmt=py_stmt,
                signature_args=signature_args,
                signature_output=signature_output,
            )
            if torchscript
            else None
        )

        # 返回 GroupedBenchmark 类的实例，初始化参数如下
        return cls(
            py_fwd_stmt=py_stmt,  # Python 前向语句
            cpp_fwd_stmt=cpp_stmt,  # C++ 前向语句
            py_model_setup=py_model_setup,  # Python 模型设置
            cpp_model_setup=None,  # C++ 模型设置为空
            inferred_model_setup=True,  # 推断模型设置为真
            setup=setup,  # 初始化设置对象
            signature_args=signature_args,  # 函数签名参数
            signature_output=signature_output,  # 函数签名输出
            torchscript=torchscript,  # 是否使用 TorchScript
            autograd=autograd,  # 是否使用自动求导
            num_threads=(num_threads,) if isinstance(num_threads, int) else num_threads,  # 线程数
        )

    @classmethod
    def init_from_model(
        cls,
        py_model_setup: Optional[str] = None,  # Python 模型设置字符串，可选
        cpp_model_setup: Optional[str] = None,  # C++ 模型设置字符串，可选
        # 通用的构造函数参数
        setup: GroupedSetup = GroupedSetup(),  # 初始化设置对象，默认为空的分组设置
        signature: Optional[str] = None,  # 函数签名字符串，可选
        torchscript: bool = False,  # 是否使用 TorchScript，默认为 False
        autograd: bool = False,  # 是否使用自动求导，默认为 False
        num_threads: Union[int, Tuple[int, ...]] = 1,  # 线程数，可以是整数或整数元组，默认为 1
    ) -> "GroupedBenchmark":
        """Create a set of benchmarks using torch.nn Modules.

        This method of benchmark creation takes setup code, and then calls
        a model rather than a free form block of code. As a result, there are
        two additional requirements compared to `init_from_stmts`:
          - `signature` must be provided.
          - A model (named "model") must be defined, either with `model = ...`
            or `def model(...): ...` in Python or `auto model = ...` in C++.
        """
        # 解析给定的签名，获取签名参数和输出
        signature_args, signature_output = cls._parse_signature(signature)
        # 如果签名参数为空，则抛出数值错误异常
        if signature_args is None:
            raise ValueError(
                "signature is needed when initializing from model definitions."
            )

        # 使用 `_make_model_invocation` 方法创建模型调用所需的参数，并返回这些参数的元组
        return cls(
            *cls._make_model_invocation(
                signature_args, signature_output, RuntimeMode.EAGER
            ),
            py_model_setup=py_model_setup,  # 设置 Python 模型的初始化代码块
            cpp_model_setup=cpp_model_setup,  # 设置 C++ 模型的初始化代码块
            inferred_model_setup=False,  # 不推断模型的初始化设置
            setup=setup,  # 设置基准测试的额外设置代码块
            signature_args=signature_args,  # 保存签名参数以备后用
            signature_output=signature_output,  # 保存签名输出以备后用
            torchscript=torchscript,  # 是否使用 TorchScript 运行模型
            autograd=autograd,  # 是否启用自动求导
            num_threads=(num_threads,) if isinstance(num_threads, int) else num_threads,  # 设置并行线程数
        )

    @classmethod
    def init_from_variants(
        cls,
        py_block: str = "",
        cpp_block: str = "",
        num_threads: Union[int, Tuple[int, ...]] = 1,
    ) -> Dict[Union[Tuple[str, ...], Optional[str]], "GroupedBenchmark"]:
        # 解析 Python 和 C++ 代码块的测试案例、设置和全局设置
        py_cases, py_setup, py_global_setup = cls._parse_variants(
            py_block, Language.PYTHON
        )
        cpp_cases, cpp_setup, cpp_global_setup = cls._parse_variants(
            cpp_block, Language.CPP
        )

        # 断言确保 Python 全局设置为空
        assert not py_global_setup
        # 创建组合的设置对象，包括 Python 设置、C++ 设置和全局设置（C++）
        setup = GroupedSetup(
            py_setup=py_setup,
            cpp_setup=cpp_setup,
            global_setup=cpp_global_setup,
        )

        # NB: 实际键类型为 `Tuple[str, ...]`，但 MyPy 可能会混淆，
        #     使用 `Union[Tuple[str, ...], Optional[str]` 作为更大的集合以匹配预期的签名。
        # 创建空字典，用于存储不同标签及其对应的测试数据
        variants: Dict[Union[Tuple[str, ...], Optional[str]], GroupedBenchmark] = {}

        # 记录已经使用的标签，以避免重复
        seen_labels: Set[str] = set()
        # 遍历 Python 和 C++ 测试案例的标签
        for label in it.chain(py_cases.keys(), cpp_cases.keys()):
            # 如果标签已经处理过，则跳过
            if label in seen_labels:
                continue
            seen_labels.add(label)

            # 获取当前标签对应的 Python 和 C++ 代码行列表
            py_lines = py_cases.get(label, [])
            cpp_lines = cpp_cases.get(label, [])

            # 确定需要处理的最大行数，并填充到相同长度
            n_lines = max(len(py_lines), len(cpp_lines))
            py_lines += [""] * (n_lines - len(py_lines))
            cpp_lines += [""] * (n_lines - len(cpp_lines))
            # 创建包含匹配的 Python 和 C++ 代码行的元组列表
            lines = [
                (py_stmt, cpp_stmt)
                for py_stmt, cpp_stmt in zip(py_lines, cpp_lines)
                if py_stmt or cpp_stmt
            ]

            # 遍历处理每一对代码行
            for i, (py_stmt, cpp_stmt) in enumerate(lines):
                # 根据代码行数确定案例标签
                case = (f"Case: {i:>2}",) if len(lines) > 1 else ()
                # 使用 GroupedBenchmark 类初始化每个测试案例
                variants[(label,) + case] = GroupedBenchmark.init_from_stmts(
                    py_stmt=py_stmt or None,
                    cpp_stmt=cpp_stmt or None,
                    setup=setup,
                    num_threads=num_threads,
                )

        # 返回所有测试案例的字典
        return variants

    def __post_init__(self) -> None:
        # 如果开启自动求导但未指定输出变量，抛出数值错误
        if self.autograd and self.signature_output is None:
            raise ValueError(
                "An output variable must be specified when `autograd=True`."
            )

        # 如果 Python 模型设置中未包含模型定义，抛出数值错误
        if self.py_model_setup and "model" not in self.py_model_setup:
            raise ValueError(
                "`py_model_setup` appears to be missing `model` definition."
            )

        # 如果 C++ 模型设置中未包含模型定义，抛出数值错误
        if self.cpp_model_setup and "model" not in self.cpp_model_setup:
            raise ValueError(
                "`cpp_model_setup` appears to be missing `model` definition."
            )

    # =========================================================================
    # == 字符串处理方法 =======================================================
    # =========================================================================

    @staticmethod
    def _parse_signature(
        signature: Optional[str],
    ) -> Tuple[Optional[Tuple[str, ...]], Optional[str]]:
        # 如果函数签名为 None，则返回 None
        if signature is None:
            return None, None

        # 使用正则表达式匹配函数签名，解析出参数和返回类型
        match = re.search(r"^f\((.*)\) -> (.*)$", signature)
        if match is None:
            raise ValueError(f"Invalid signature: `{signature}`")

        # 将参数和返回类型解析为元组和字符串，并去除首尾空格
        args: Tuple[str, ...] = tuple(match.groups()[0].split(", "))
        output: str = match.groups()[1].strip()

        # 如果返回类型包含逗号，则抛出异常，因为当前不支持多返回值
        if "," in output:
            raise ValueError(
                f"Multiple return values are not currently allowed: `{output}`"
            )

        # 如果返回类型为 "None"，则返回参数元组和 None
        if output == "None":
            return args, None

        # 否则返回参数元组和返回类型字符串
        return args, output

    @staticmethod
    def _model_from_py_stmt(
        py_stmt: Optional[str],
        signature_args: Optional[Tuple[str, ...]],
        signature_output: Optional[str],
    ) -> str:
        # 如果 py_stmt 为 None，则抛出异常，必须定义 py_stmt 才能派生模型
        if py_stmt is None:
            raise ValueError("`py_stmt` must be defined in order to derive a model.")

        # 如果函数签名参数为 None，则抛出异常，必须有签名才能派生模型
        if signature_args is None:
            raise ValueError("signature is needed in order to derive a model.")

        # 使用 textwrap.dedent 处理代码块，生成模型函数的字符串表示
        return textwrap.dedent(
            f"""\
            def model({', '.join(signature_args)}):
                {stmt_str}
                return {signature_output}
        """
        ).format(stmt_str=textwrap.indent(py_stmt, " " * 4))

    @staticmethod
    def _make_model_invocation(
        signature_args: Tuple[str, ...],
        signature_output: Optional[str],
        runtime: RuntimeMode,
    ) -> Tuple[str, str]:
        # 初始化 Python 和 C++ 的前缀为空字符串
        py_prefix, cpp_prefix = "", ""

        # 如果存在返回类型，则设置对应的 Python 和 C++ 前缀
        if signature_output is not None:
            py_prefix = f"{signature_output} = "
            cpp_prefix = f"auto {signature_output} = "

        # 根据运行时模式选择模型名称和 C++ 调用方式
        if runtime == RuntimeMode.EAGER:
            model_name = "model"
            cpp_invocation = (
                f"{cpp_prefix}{model_name}->forward({', '.join(signature_args)});"
            )

        else:
            assert runtime == RuntimeMode.JIT
            model_name = "jit_model"
            cpp_invocation = textwrap.dedent(
                f"""\
                std::vector<torch::jit::IValue> ivalue_inputs({{
                    {', '.join([f'torch::jit::IValue({a})' for a in signature_args])}
                }});
                {cpp_prefix}{model_name}.forward(ivalue_inputs);
            """
            )

        # Python 调用字符串表示
        py_invocation = f"{py_prefix}{model_name}({', '.join(signature_args)})"

        # 返回 Python 和 C++ 的调用字符串
        return py_invocation, cpp_invocation

    @staticmethod
    def _parse_variants(
        block: str, language: Language
        # 解析变体方法，接受块和语言参数
    # 定义函数签名和返回类型注解，函数接收一个字符串块和语言类型，返回一个元组
    -> Tuple[Dict[str, List[str]], str, str]:
        # 去除字符串块的首尾空白并去除首部缩进，以便处理
        block = textwrap.dedent(block).strip()
        # 根据语言类型选择注释符号
        comment = "#" if language == Language.PYTHON else "//"
        # 定义用于匹配标签的正则表达式模式
        label_pattern = f"{comment} @(.+)$"
        # 初始化标签为空字符串
        label = ""

        # 初始化存储不同标签下行的字典，包括默认的"SETUP"和"GLOBAL_SETUP"
        lines_by_label: Dict[str, List[str]] = {"SETUP": [], "GLOBAL_SETUP": []}
        # 遍历字符串块中的每一行
        for line in block.splitlines(keepends=False):
            # 在每行中查找匹配标签的内容
            match = re.search(label_pattern, line.strip())
            if match:
                # 若找到匹配项，则提取标签内容并处理空格和大小写
                label = match.groups()[0]
                if label.replace(" ", "_").upper() in ("SETUP", "GLOBAL_SETUP"):
                    label = label.replace(" ", "_").upper()
                continue

            # 将当前行添加到对应标签下的列表中
            lines_by_label.setdefault(label, [])
            # 如果行以注释符号开头，则将行内容设为空字符串
            if line.startswith(comment):
                line = ""
            lines_by_label[label].append(line)

        # 将"SETUP"和"GLOBAL_SETUP"标签下的行内容连接成字符串
        setup = "\n".join(lines_by_label.pop("SETUP"))
        global_setup = "\n".join(lines_by_label.pop("GLOBAL_SETUP"))

        # 返回整理后的行字典以及单独的"SETUP"和"GLOBAL_SETUP"内容字符串
        return lines_by_label, setup, global_setup
# 这些是面向用户的 API。
GroupedStmts = GroupedBenchmark.init_from_stmts
GroupedModules = GroupedBenchmark.init_from_model
GroupedVariants = GroupedBenchmark.init_from_variants
```