# `.\pytorch\torch\onnx\_internal\diagnostics\infra\context.py`

```
# mypy: allow-untyped-defs
"""A diagnostic context based on SARIF."""

# 引入上下文管理模块
import contextlib

# 引入数据类支持
import dataclasses

# 引入 gzip 压缩模块
import gzip

# 引入日志记录模块
import logging

# 引入类型提示相关模块
from typing import (
    Callable,           # 用于类型提示的可调用对象
    Generator,          # 生成器类型
    Generic,            # 泛型类型支持
    List,               # 列表类型
    Literal,            # 字面常量类型
    Mapping,            # 映射类型
    Optional,           # 可选类型
    Type,               # 类型对象
    TypeVar,            # 类型变量
)

# 引入 typing_extensions 中的 Self 类型
from typing_extensions import Self

# 引入 torch.onnx._internal.diagnostics 中的子模块和相关类
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version

# 日志记录器对象，默认使用当前模块名称
diagnostic_logger: logging.Logger = logging.getLogger(__name__)

# 数据类，表示一个诊断对象
@dataclasses.dataclass
class Diagnostic:
    rule: infra.Rule                                    # 诊断规则对象
    level: infra.Level                                  # 诊断级别对象
    message: Optional[str] = None                       # 可选的诊断消息
    locations: List[infra.Location] = dataclasses.field(default_factory=list)  # 位置列表
    stacks: List[infra.Stack] = dataclasses.field(default_factory=list)        # 堆栈列表
    graphs: List[infra.Graph] = dataclasses.field(default_factory=list)        # 图表列表
    thread_flow_locations: List[infra.ThreadFlowLocation] = dataclasses.field(
        default_factory=list
    )  # 线程流位置列表
    additional_messages: List[str] = dataclasses.field(default_factory=list)   # 额外消息列表
    tags: List[infra.Tag] = dataclasses.field(default_factory=list)             # 标签列表
    source_exception: Optional[Exception] = None        # 导致诊断生成的异常对象
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    """The logger for this diagnostic. Defaults to 'diagnostic_logger' which has the same
    log level setting with `DiagnosticOptions.verbosity_level`."""
    _current_log_section_depth: int = 0                 # 当前日志段深度

    def __post_init__(self) -> None:
        pass
    # 返回当前诊断信息的 SARIF 格式表示
    def sarif(self) -> sarif.Result:
        """Returns the SARIF Result representation of this diagnostic."""
        # 使用诊断消息或默认模板创建消息
        message = self.message or self.rule.message_default_template
        # 如果有额外消息，则将它们连接成一个 Markdown 格式的字符串
        if self.additional_messages:
            additional_message = "\n".join(self.additional_messages)
            message_markdown = (
                f"{message}\n\n## Additional Message:\n\n{additional_message}"
            )
        else:
            message_markdown = message

        # 确定 SARIF 结果的级别，根据诊断级别确定信息类型
        kind: Literal["informational", "fail"] = (
            "informational" if self.level == infra.Level.NONE else "fail"
        )

        # 创建 SARIF 结果对象
        sarif_result = sarif.Result(
            message=sarif.Message(text=message, markdown=message_markdown),
            level=self.level.name.lower(),  # 指定 SARIF 结果的级别
            rule_id=self.rule.id,
            kind=kind,  # 指定 SARIF 结果的类型
        )
        
        # 将位置信息转换为 SARIF 格式并添加到 SARIF 结果中
        sarif_result.locations = [location.sarif() for location in self.locations]
        # 将堆栈信息转换为 SARIF 格式并添加到 SARIF 结果中
        sarif_result.stacks = [stack.sarif() for stack in self.stacks]
        # 将图形信息转换为 SARIF 格式并添加到 SARIF 结果中
        sarif_result.graphs = [graph.sarif() for graph in self.graphs]
        # 将代码流信息转换为 SARIF 格式并添加到 SARIF 结果中
        sarif_result.code_flows = [
            sarif.CodeFlow(
                thread_flows=[
                    sarif.ThreadFlow(
                        locations=[loc.sarif() for loc in self.thread_flow_locations]
                    )
                ]
            )
        ]
        # 添加属性标签到 SARIF 结果的属性信息中
        sarif_result.properties = sarif.PropertyBag(
            tags=[tag.value for tag in self.tags]
        )
        # 返回 SARIF 结果对象
        return sarif_result

    # 向诊断信息添加一个位置
    def with_location(self: Self, location: infra.Location) -> Self:
        """Adds a location to the diagnostic."""
        self.locations.append(location)
        return self

    # 向诊断信息添加一个线程流位置
    def with_thread_flow_location(
        self: Self, location: infra.ThreadFlowLocation
    ) -> Self:
        """Adds a thread flow location to the diagnostic."""
        self.thread_flow_locations.append(location)
        return self

    # 向诊断信息添加一个堆栈信息
    def with_stack(self: Self, stack: infra.Stack) -> Self:
        """Adds a stack to the diagnostic."""
        self.stacks.append(stack)
        return self

    # 向诊断信息添加一个图形信息
    def with_graph(self: Self, graph: infra.Graph) -> Self:
        """Adds a graph to the diagnostic."""
        self.graphs.append(graph)
        return self

    # 定义一个上下文管理器，用于记录特定段落的日志信息
    @contextlib.contextmanager
    def log_section(
        self, level: int, message: str, *args, **kwargs
    ) -> Generator[None, None, None]:
        """
        Context manager for a section of log messages, denoted by a title message and increased indentation.

        Same api as `logging.Logger.log`.

        This context manager logs the given title at the specified log level, increases the current
        section depth for subsequent log messages, and ensures that the section depth is decreased
        again when exiting the context.

        Args:
            level: The log level.
                The severity level of the log message.
            message: The title message to log.
                The main message that describes the section.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.
                Additional parameters to customize the logging behavior.

        Yields:
            None: This context manager does not yield any value.

        Example:
            >>> with DiagnosticContext("DummyContext", "1.0"):
            ...     rule = infra.Rule("RuleID", "DummyRule", "Rule message")
            ...     diagnostic = Diagnostic(rule, infra.Level.WARNING)
            ...     with diagnostic.log_section(logging.INFO, "My Section"):
            ...         diagnostic.log(logging.INFO, "My Message")
            ...         with diagnostic.log_section(logging.INFO, "My Subsection"):
            ...             diagnostic.log(logging.INFO, "My Submessage")
            ...     diagnostic.additional_messages
            ['## My Section', 'My Message', '### My Subsection', 'My Submessage']
        """
        # Check if the logger is enabled for the specified log level
        if self.logger.isEnabledFor(level):
            # Generate the formatted title message with the current log section depth
            indented_format_message = (
                f"##{'#' * self._current_log_section_depth } {message}"
            )
            # Log the formatted message at the specified log level
            self.log(
                level,
                indented_format_message,
                *args,
                **kwargs,
            )
        # Increase the current log section depth for nested sections
        self._current_log_section_depth += 1
        try:
            # Yield to allow execution of the code within the context
            yield
        finally:
            # Decrease the log section depth when exiting the context
            self._current_log_section_depth -= 1
    def log(self, level: int, message: str, *args, **kwargs) -> None:
        """Logs a message within the diagnostic. Same api as `logging.Logger.log`.

        If logger is not enabled for the given level, the message will not be logged.
        Otherwise, the message will be logged and also added to the diagnostic's additional_messages.

        The default setting for `DiagnosticOptions.verbosity_level` is `logging.INFO`. Based on this default,
        the log level recommendations are as follows. If you've set a different default verbosity level in your
        application, please adjust accordingly:

        - logging.ERROR: Log any events leading to application failure.
        - logging.WARNING: Log events that might result in application issues or failures, although not guaranteed.
        - logging.INFO: Log general useful information, ensuring minimal performance overhead.
        - logging.DEBUG: Log detailed debug information, which might affect performance when logged.

        Args:
            level: The log level.
            message: The message to log.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.
        """
        # 检查给定级别的日志记录器是否启用
        if self.logger.isEnabledFor(level):
            # 格式化消息，并延迟评估参数的昂贵计算
            formatted_message = message % args
            # 使用给定级别记录消息，并传递关键字参数
            self.logger.log(level, formatted_message, **kwargs)
            # 将格式化的消息添加到附加消息列表中
            self.additional_messages.append(formatted_message)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Logs a debug message within the diagnostic. Same api as logging.Logger.debug.

        Checkout `log` for more details.
        """
        # 调用 log 方法记录调试级别的消息
        self.log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Logs an info message within the diagnostic. Same api as logging.Logger.info.

        Checkout `log` for more details.
        """
        # 调用 log 方法记录信息级别的消息
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Logs a warning message within the diagnostic. Same api as logging.Logger.warning.

        Checkout `log` for more details.
        """
        # 调用 log 方法记录警告级别的消息
        self.log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Logs an error message within the diagnostic. Same api as logging.Logger.error.

        Checkout `log` for more details.
        """
        # 调用 log 方法记录错误级别的消息
        self.log(logging.ERROR, message, *args, **kwargs)
    # 记录源异常并进行日志记录
    def log_source_exception(self, level: int, exception: Exception) -> None:
        """Logs a source exception within the diagnostic.

        Invokes `log_section` and `log` to log the exception in markdown section format.
        """
        # 将异常保存在对象中以备后续处理
        self.source_exception = exception
        # 使用 log_section 方法创建一个日志段落，并记录异常
        with self.log_section(level, "Exception log"):
            # 使用 log 方法记录异常的格式化输出
            self.log(level, "%s", formatter.lazy_format_exception(exception))

    # 记录当前 Python 调用堆栈
    def record_python_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current Python call stack."""
        # 调整需要跳过的帧数，包括本函数自身
        frames_to_skip += 1  # Skip this function.
        # 使用 utils 模块记录 Python 调用堆栈
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip)
        # 将记录的堆栈信息关联到当前对象
        self.with_stack(stack)
        # 如果堆栈中有帧数大于 0，则关联第一个帧的位置信息到当前对象
        if len(stack.frames) > 0:
            self.with_location(stack.frames[0].location)
        # 返回记录的堆栈信息
        return stack

    # 记录 Python 函数调用作为线程流程的一步
    def record_python_call(
        self,
        fn: Callable,
        state: Mapping[str, str],
        message: Optional[str] = None,
        frames_to_skip: int = 0,
    ) -> infra.ThreadFlowLocation:
        """Records a python call as one thread flow step."""
        # 调整需要跳过的帧数，包括本函数自身
        frames_to_skip += 1  # Skip this function.
        # 使用 utils 模块记录 Python 调用堆栈，并限制堆栈帧数为 5
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip, frames_to_log=5)
        # 获取函数 fn 的位置信息
        location = utils.function_location(fn)
        # 将可选的消息关联到位置信息
        location.message = message
        # 将函数的位置信息作为堆栈的顶部帧添加到记录的堆栈中
        stack.frames.insert(0, infra.StackFrame(location=location))
        # 创建线程流程位置对象，包括位置信息、状态信息、索引和堆栈
        thread_flow_location = infra.ThreadFlowLocation(
            location=location,
            state=state,
            index=len(self.thread_flow_locations),
            stack=stack,
        )
        # 将线程流程位置对象添加到当前对象的线程流程位置列表中
        self.with_thread_flow_location(thread_flow_location)
        # 返回记录的线程流程位置对象
        return thread_flow_location
class RuntimeErrorWithDiagnostic(RuntimeError):
    """Runtime error with enclosed diagnostic information."""

    def __init__(self, diagnostic: Diagnostic):
        # 调用父类的初始化方法，传入诊断信息的消息作为错误信息
        super().__init__(diagnostic.message)
        # 存储诊断信息对象
        self.diagnostic = diagnostic


@dataclasses.dataclass
class DiagnosticContext(Generic[_Diagnostic]):
    name: str
    version: str
    options: infra.DiagnosticOptions = dataclasses.field(
        default_factory=infra.DiagnosticOptions
    )
    diagnostics: List[_Diagnostic] = dataclasses.field(init=False, default_factory=list)
    # TODO(bowbao): Implement this.
    # _invocation: infra.Invocation = dataclasses.field(init=False)
    _inflight_diagnostics: List[_Diagnostic] = dataclasses.field(
        init=False, default_factory=list
    )
    _previous_log_level: int = dataclasses.field(init=False, default=logging.WARNING)
    logger: logging.Logger = dataclasses.field(init=False, default=diagnostic_logger)
    _bound_diagnostic_type: Type = dataclasses.field(init=False, default=Diagnostic)

    def __enter__(self):
        # 保存当前日志级别，并设置为诊断上下文中的详细级别
        self._previous_log_level = self.logger.level
        self.logger.setLevel(self.options.verbosity_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复之前的日志级别
        self.logger.setLevel(self._previous_log_level)
        return None

    def sarif(self) -> sarif.Run:
        """Returns the SARIF Run object."""
        # 生成 SARIF 格式的运行对象，包括工具信息和诊断结果
        unique_rules = {diagnostic.rule for diagnostic in self.diagnostics}
        return sarif.Run(
            sarif.Tool(
                driver=sarif.ToolComponent(
                    name=self.name,
                    version=self.version,
                    rules=[rule.sarif() for rule in unique_rules],
                )
            ),
            results=[diagnostic.sarif() for diagnostic in self.diagnostics],
        )

    def sarif_log(self) -> sarif.SarifLog:  # type: ignore[name-defined]
        """Returns the SARIF Log object."""
        # 生成 SARIF 格式的日志对象
        return sarif.SarifLog(
            version=sarif_version.SARIF_VERSION,
            schema_uri=sarif_version.SARIF_SCHEMA_LINK,
            runs=[self.sarif()],
        )

    def to_json(self) -> str:
        # 将 SARIF 日志对象转换为 JSON 字符串
        return formatter.sarif_to_json(self.sarif_log())

    def dump(self, file_path: str, compress: bool = False) -> None:
        """Dumps the SARIF log to a file."""
        # 将 SARIF 日志写入指定文件，支持可选的压缩功能
        if compress:
            with gzip.open(file_path, "wt") as f:
                f.write(self.to_json())
        else:
            with open(file_path, "w") as f:
                f.write(self.to_json())
    # 记录诊断信息到日志中。
    def log(self, diagnostic: _Diagnostic) -> None:
        """Logs a diagnostic.

        This method should be used only after all the necessary information for the diagnostic
        has been collected.

        Args:
            diagnostic: The diagnostic to add.
        """
        # 检查诊断是否符合预期的类型
        if not isinstance(diagnostic, self._bound_diagnostic_type):
            raise TypeError(
                f"Expected diagnostic of type {self._bound_diagnostic_type}, got {type(diagnostic)}"
            )
        # 如果设置了警告作为错误处理，并且诊断级别为警告，则将其提升为错误级别
        if self.options.warnings_as_errors and diagnostic.level == infra.Level.WARNING:
            diagnostic.level = infra.Level.ERROR
        # 将诊断信息添加到诊断列表中
        self.diagnostics.append(diagnostic)

    # 记录诊断信息到日志中，并在出现错误时抛出异常。
    def log_and_raise_if_error(self, diagnostic: _Diagnostic) -> None:
        """Logs a diagnostic and raises an exception if it is an error.

        Use this method for logging non inflight diagnostics where diagnostic level is not known or
        lower than ERROR. If it is always expected raise, use `log` and explicit
        `raise` instead. Otherwise there is no way to convey the message that it always
        raises to Python intellisense and type checking tools.

        This method should be used only after all the necessary information for the diagnostic
        has been collected.

        Args:
            diagnostic: The diagnostic to add.
        """
        # 调用 log 方法记录诊断信息
        self.log(diagnostic)
        # 如果诊断级别为错误，则根据情况抛出异常
        if diagnostic.level == infra.Level.ERROR:
            # 如果诊断包含源异常信息，则抛出源异常
            if diagnostic.source_exception is not None:
                raise diagnostic.source_exception
            # 否则抛出包含诊断信息的运行时异常
            raise RuntimeErrorWithDiagnostic(diagnostic)

    # 添加一个正在进行中的诊断到上下文中。
    @contextlib.contextmanager
    def add_inflight_diagnostic(
        self, diagnostic: _Diagnostic
    ) -> Generator[_Diagnostic, None, None]:
        """Adds a diagnostic to the context.

        Use this method to add diagnostics that are not created by the context.
        Args:
            diagnostic: The diagnostic to add.
        """
        # 将诊断信息添加到正在进行中的诊断列表中
        self._inflight_diagnostics.append(diagnostic)
        try:
            # 在上下文中生成诊断信息
            yield diagnostic
        finally:
            # 在生成器退出后移除正在进行中的诊断信息
            self._inflight_diagnostics.pop()

    # 将诊断信息推送到正在进行中的诊断栈中。
    def push_inflight_diagnostic(self, diagnostic: _Diagnostic) -> None:
        """Pushes a diagnostic to the inflight diagnostics stack.

        Args:
            diagnostic: The diagnostic to push.

        Raises:
            ValueError: If the rule is not supported by the tool.
        """
        # 将诊断信息推送到正在进行中的诊断栈中
        self._inflight_diagnostics.append(diagnostic)

    # 从正在进行中的诊断栈中弹出最后一个诊断信息。
    def pop_inflight_diagnostic(self) -> _Diagnostic:
        """Pops the last diagnostic from the inflight diagnostics stack.

        Returns:
            The popped diagnostic.
        """
        # 从正在进行中的诊断栈中弹出最后一个诊断信息并返回
        return self._inflight_diagnostics.pop()
    # 定义一个方法用于返回正在进行中的诊断信息，可以根据规则来筛选
    def inflight_diagnostic(self, rule: Optional[infra.Rule] = None) -> _Diagnostic:
        if rule is None:
            # 如果没有指定规则，则返回最近的一个正在进行中的诊断信息
            if len(self._inflight_diagnostics) <= 0:
                # 如果没有正在进行中的诊断信息，则抛出断言错误
                raise AssertionError("No inflight diagnostics")

            return self._inflight_diagnostics[-1]
        else:
            # 如果指定了规则，则从最近的到最远的诊断信息中查找匹配该规则的诊断信息
            for diagnostic in reversed(self._inflight_diagnostics):
                if diagnostic.rule == rule:
                    return diagnostic
            # 如果找不到匹配指定规则的诊断信息，则抛出断言错误
            raise AssertionError(f"No inflight diagnostic for rule {rule.name}")
```