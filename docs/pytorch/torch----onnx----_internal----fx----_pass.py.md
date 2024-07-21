# `.\pytorch\torch\onnx\_internal\fx\_pass.py`

```py
# 添加类型提示的兼容性标记
# mypy: allow-untyped-defs
# 导入 Python 未来版本的注解特性
from __future__ import annotations

# 导入抽象基类模块
import abc

# 导入上下文管理模块
import contextlib
# 导入数据类支持模块
import dataclasses
# 导入文本差异比较模块
import difflib

# 导入 IO 操作模块
import io
# 导入日志记录模块
import logging
# 导入系统相关模块
import sys

# 导入类型提示模块
from typing import Any, Callable, Optional, Tuple, Union

# 导入 PyTorch 主模块
import torch
# 导入 PyTorch FX 模块
import torch.fx
# 导入 PyTorch 下级子类虚拟张量
from torch._subclasses import fake_tensor
# 导入 PyTorch FX 实验性代理张量
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
# 导入 PyTorch ONNX 内部模块
from torch.onnx._internal import _beartype
# 导入 PyTorch ONNX 内部 FX 模块
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher


# 数据类：包信息
@dataclasses.dataclass
class PackageInfo:
    package_name: str  # 包名
    version: Optional[str]  # 版本号
    commit_hash: Optional[str]  # 提交哈希值

    # 转换为 ONNX 域字符串
    def to_onnx_domain_string(self) -> str:
        return ".".join(
            filter(None, ("pkg", self.package_name, self.version, self.commit_hash))
        )

    # 从 Python 类名获取包信息
    @classmethod
    def from_python_class(cls, python_class_name: Union[type, str]) -> PackageInfo:
        if isinstance(python_class_name, type):
            python_class_name = python_class_name.__module__
        package_name = python_class_name.split(".")[0]
        package = __import__(package_name)
        version = getattr(package, "__version__", None)
        # TODO: Figure out how to retrieve commit hash.
        commit_hash = None
        return cls(package_name, version, commit_hash)


# 数据类：图模块 ONNX 元信息
@dataclasses.dataclass
class GraphModuleOnnxMeta:
    package_info: PackageInfo  # 包信息对象


# 上下文管理器：修补 difflib.SequenceMatcher 初始化
@contextlib.contextmanager
def _patch_difflib_sequence_matcher_init():
    """Context patching `difflib.SequenceMatcher` for fx readable graph.

    Under this context, the `autojunk` argument of `difflib.SequenceMatcher` will always
    be considered as `False`. This is to prevent `difflib.SequenceMatcher` recognizing
    stacktrace messages in fx readable graph as junk, as these messages tend to be long (>200)
    and repeat multiple times, which falls under the junk filter criteria.

    `difflib.SequenceMatcher` is used underneath by all sorts of diffing functions
    in `difflib`, including `difflib.unified_diff`, `difflib.ndiff`, `difflib.context_diff`.
    Unfortunately, there is no way to pass `autojunk` argument to these functions, and
    they all default to `True`. This context patching will affect all of them.

    `Reference: Automatic junk heuristic <https://docs.python.org/3/library/difflib.html>`_
    """
    original_init = difflib.SequenceMatcher.__init__

    def patched_init(self, isjunk=None, a="", b="", autojunk=True):
        original_init(self, isjunk, a, b, autojunk=False)

    difflib.SequenceMatcher.__init__ = patched_init  # type: ignore[assignment]
    try:
        yield
    finally:
        difflib.SequenceMatcher.__init__ = original_init  # type: ignore[assignment]


# 函数：生成两个字符串的统一差异字符串
def _unified_diff(a: str, b: str) -> str:
    """Return a string containing the unified diff of two strings.

    This function calls a patched version of `difflib.unified_diff` with `autojunk` set
    to `False` for `difflib.SequenceMatcher` class. More details can be found in
    """
    # 定义 `_patch_difflib_sequence_matcher_init` 函数，用于计算两个字符串的统一 diff。

    Args:
        a: 第一个字符串。
        b: 第二个字符串。

    Returns:
        两个字符串的统一 diff。如果没有差异，则返回 "<no diff>"。

    Example::

        >>> a = '''class GraphModule(torch.nn.Module):
        ...     def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view = input_ids.view(-1, 3);  input_ids = None
        ... '''
        >>> b = '''class <lambda>(torch.nn.Module):
        ...     def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
        ... '''
        >>> print(_unified_diff(a, b))
        ---
        +++
        @@ -1,4 +1,4 @@
        -class GraphModule(torch.nn.Module):
        -    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        +class <lambda>(torch.nn.Module):
        +    def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
                # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        -        view = input_ids.view(-1, 3);  input_ids = None
        +        view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
    """

    # 将字符串 `a` 和 `b` 按行拆分成列表，保留行尾符号。
    a_list = a.splitlines(keepends=True)
    b_list = b.splitlines(keepends=True)

    # 使用 `_patch_difflib_sequence_matcher_init` 上下文管理器，计算 `a_list` 和 `b_list` 的统一 diff。
    with _patch_difflib_sequence_matcher_init():
        # 将统一 diff 结果连接成一个字符串。
        diff = "".join(difflib.unified_diff(a_list, b_list, n=sys.maxsize))

    # 如果没有差异，返回 "<no diff>"。
    if not diff:
        return "<no diff>"
    # 否则，返回差异结果。
    return diff
@_beartype.beartype
# 使用 @_beartype 装饰器来对函数进行类型检查和类型注解
def _transform_diagnose_call_message_formatter(
    run: Callable,
    self: Transform,
    *args: Any,
    **kwargs: Any,
) -> str:
    # 返回一个格式化的字符串，指示正在运行的转换器类名
    return f"Running {self.__class__.__name__} pass. "


def maybe_fx_graph_tabular(graph: torch.fx.Graph) -> Optional[str]:
    """Return the Graph nodes in tabular format. Equivalent to stdout of `graph.print_tabular()`.
    If `tabulate` is not installed, return `None`.

    Args:
        graph: The Graph to print.

    Returns:
        The Graph printed in a tabular format. None if `tabulate` is not installed.
    """
    # 创建一个字符串流对象
    f = io.StringIO()
    # 重定向标准输出到字符串流 f
    with contextlib.redirect_stdout(f):
        try:
            # 尝试打印图的节点信息到标准输出，通过 tabular 格式
            graph.print_tabular()
        except ImportError:
            # 如果导入错误（tabulate 模块未安装），返回 None
            return None
    # 获取字符串流 f 中的全部内容并返回
    return f.getvalue()


class Transform(abc.ABC):
    """Base class for FX graph transformations to be used by FX-ONNX exporter.

    Similar to `FX Interpreter <https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter>`_,
    specializations of this class execute the FX graph Node-by-Node.
    Methods in the `Transform` class can be overridden to customize the behavior of the model.
    This pattern can be useful for many things, including writing code transformations as well as analysis passes.

    The following methods can be overridden::

        _run()
            +-- run_node()
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    One important aspect to note is that if the transformation modifies the model input and/or output signature,
    (e.g. additional inputs/outputs are added to the model), :class:`InputAdaptStep` and/or :class:`OutputAdaptStep`
    are needed to reconcile :attr:`ONNXProgram.model_signature` and :attr:`ONNXProgram.model_proto`.
    That is, the model signature and the model representation must match.

    As an additional feature, this class provides builtin support for transformation recording using the diagnostics.
    The granularity of overriding is up to the user. And it affects the granularity of
    the diagnostics information. For example, if `_run()` is overridden, the
    diagnostics information will only contain graph level transformation. Instead,
    if `call_function()` is overridden, the diagnostics information will additionally
    contain the node level information of `call_function()`.

    TODO(bowbao): Add more overridable methods in call hierarchy
    TODO(bowbao): Create an example once more overridable methods are added.
    """

    diagnostic_context: diagnostics.DiagnosticContext
    """The diagnostic context for recording diagnostics."""

    module: torch.fx.GraphModule
    """The module to be transformed."""

    fake_mode: Optional[fake_tensor.FakeTensorMode]
    """The existing fake mode detected from `self.module`."""
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
    ):
        """
        Initialize the transform.

        Args:
            diagnostic_context: The diagnostic context for recording diagnostics.
            module: The module to be transformed.
        """
        self.diagnostic_context = diagnostic_context  # 将传入的诊断上下文存储在实例变量中
        self.module = module  # 将传入的模块存储在实例变量中
        self.fake_mode = self._detect_fake_mode()  # 调用内部方法检测并存储模块的伪张量模式

    def _detect_fake_mode(self) -> Optional[fake_tensor.FakeTensorMode]:
        """
        Detect fake mode from the graph.

        Scan through all nodes in graph and their meta['val'] to detect fake mode.
        """
        fake_tensors = [node.meta.get("val") for node in self.module.graph.nodes]  # 获取模块图中每个节点的'meta['val']'值组成列表
        with maybe_disable_fake_tensor_mode():  # 可能禁用伪张量模式的上下文管理器
            return torch._dynamo.utils.detect_fake_mode(fake_tensors)  # 使用工具函数检测伪张量模式并返回结果

    def _maybe_fakefy_args(
        self, fake_mode: Optional[fake_tensor.FakeTensorMode], *args: Any
    ) -> Tuple[Any, ...]:
        if fake_mode is None:
            return args  # 如果伪张量模式为None，则直接返回参数
        # NB: This should hit the cache if tensors were fakefied before.
        # E.g., when the fx graph is produced by Dynamo.
        return tuple(
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args
        )  # 对参数进行可能的伪张量转换，如果参数是张量，则使用伪张量模式进行转换

    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        """
        Abstract method to be implemented by subclasses.

        This method defines the transformation logic that subclasses must implement.
        """
        ...

    @diagnostics.diagnose_call(
        diagnostics.rules.fx_pass,
        diagnostic_message_formatter=_transform_diagnose_call_message_formatter,
    )
    # 使用诊断装饰器来处理函数调用，指定诊断规则和消息格式化函数
    def run(self, *args, **kwargs) -> torch.fx.GraphModule:
        """Run the transform on `self.module`.

        Note that this method may or may not mutate `self.module`, and the returned
        `GraphModule` could be either `self.module` or a new `GraphModule`.

        Args:
            *args: Positional arguments for `self.module` to run.
            **kwargs: Keyword arguments for `self.module` to run.
        """
        # 创建诊断对象来记录执行的诊断信息
        diagnostic = self.diagnostic_context.inflight_diagnostic(
            rule=diagnostics.rules.fx_pass
        )
        # 输出信息，指导如何开启详细的图形修改日志记录
        diagnostic.info(
            "For detailed logging of graph modifications by this pass, either set "
            "`DiagnosticOptions.verbosity_level` to `logging.DEBUG` or use the environment variable "
            "`TORCH_LOGS='onnx_diagnostics'`."
        )

        # 在应用转换之前收集图形信息
        graph_diff_log_level = logging.DEBUG
        if diagnostic.logger.isEnabledFor(graph_diff_log_level):
            # 如果启用了调试日志级别，获取转换前的可读图形表示
            old_readable_graph = self.module.print_readable(print_output=False)
            # 获取可能存在的 FX 图表格化信息
            old_tabular = maybe_fx_graph_tabular(self.module.graph)
        else:
            # 如果未启用调试日志级别，则设置为空字符串，避免未绑定警告
            old_readable_graph = ""
            old_tabular = ""

        # 执行实际的转换操作
        module = self._run(*args, **kwargs)

        # 在应用转换之后收集图形信息
        if diagnostic.logger.isEnabledFor(graph_diff_log_level):
            # 获取转换后的可读图形表示
            new_readable_graph = module.print_readable(print_output=False)
            # 获取可能存在的新的 FX 图表格化信息
            new_tabular = maybe_fx_graph_tabular(module.graph)

            # 使用诊断记录图形的差异
            with diagnostic.log_section(graph_diff_log_level, "Graph diff:"):
                diagnostic.log(
                    graph_diff_log_level,
                    "```\n%s\n```py",
                    diagnostics.LazyString(
                        _unified_diff, old_readable_graph, new_readable_graph
                    ),
                )

            # 使用诊断记录表格化信息的差异
            with diagnostic.log_section(graph_diff_log_level, "Tabular diff:"):
                if old_tabular is None or new_tabular is None:
                    diagnostic.log(
                        graph_diff_log_level,
                        "Tabular diff is not available because `tabulate` is not installed.",
                    )
                else:
                    diagnostic.log(
                        graph_diff_log_level,
                        "```\n%s\n```py",
                        diagnostics.LazyString(_unified_diff, old_tabular, new_tabular),
                    )

        # 返回经过转换后的模块对象
        return module
#`
class AnalysisResult(abc.ABC):  # noqa: B024
    # 定义一个抽象基类 AnalysisResult，继承自 abc.ABC
    ...

class Analysis(abc.ABC):
    @_beartype.beartype
    # 定义一个抽象基类 Analysis，继承自 abc.ABC，使用 @beartype.beartype 装饰器进行类型检查
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,  # 初始化时接收一个诊断上下文对象
        module: torch.fx.GraphModule,  # 初始化时接收一个 torch.fx.GraphModule 对象
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,  # 初始化时接收一个 OnnxFunctionDispatcher 对象
    ):
        self.diagnostic_context = diagnostic_context  # 将诊断上下文对象赋值给实例变量
        self.module = module  # 将 torch.fx.GraphModule 对象赋值给实例变量
        self.onnxfunction_dispatcher = onnxfunction_dispatcher  # 将 OnnxFunctionDispatcher 对象赋值给实例变量

    @abc.abstractmethod
    # 定义一个抽象方法 analyze，接收一个诊断级别参数，并返回一个 AnalysisResult 对象
    def analyze(self, diagnostic_level: diagnostics.infra.Level) -> AnalysisResult:
        ...
```