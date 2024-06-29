# `D:\src\scipysrc\pandas\pandas\_testing\_warnings.py`

```
# 导入必要的模块和类
from __future__ import annotations

from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
)
import inspect  # 导入用于检查对象的模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关功能的模块
from typing import (
    TYPE_CHECKING,
    Literal,
    Union,
    cast,
)
import warnings  # 导入警告处理模块

from pandas.compat import PY311  # 从 pandas 兼容模块导入 PY311

if TYPE_CHECKING:
    from collections.abc import (
        Generator,
        Sequence,
    )

# 定义一个上下文管理器，用于验证代码是否产生了预期的警告
@contextmanager
def assert_produces_warning(
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None = Warning,
    filter_level: Literal[
        "error", "ignore", "always", "default", "module", "once"
    ] = "always",
    check_stacklevel: bool = True,
    raise_on_extra_warnings: bool = True,
    match: str | tuple[str | None, ...] | None = None,
    must_find_all_warnings: bool = True,
) -> Generator[list[warnings.WarningMessage], None, None]:
    """
    上下文管理器，用于运行预期会引发特定警告、多个特定警告或不引发任何警告的代码。
    验证代码是否引发了预期的警告，并确保不会引发任何其他未预期的警告。本质上是 ``warnings.catch_warnings`` 的包装器。

    参数
    ----------
    expected_warning : {Warning, False, tuple[Warning, ...], None}, 默认为 Warning
        引发的警告类型。``exception.Warning`` 是所有警告的基类。
        要引发多种类型的异常，请将它们作为元组传递。要检查是否未返回任何警告，请指定 ``False`` 或 ``None``。
    filter_level : str 或 None, 默认为 "always"
        指定警告是被忽略、显示还是转换成错误。
        有效值包括：

        * "error" - 将匹配的警告转换为异常
        * "ignore" - 忽略该警告
        * "always" - 总是发出警告
        * "default" - 首次从每个位置生成警告时打印警告
        * "module" - 首次从每个模块生成警告时打印警告
        * "once" - 首次生成警告时打印警告

    check_stacklevel : bool, 默认为 True
        如果为 True，显示调用包含警告的函数的行以显示函数的调用位置。
        否则，显示实现函数的行。
    raise_on_extra_warnings : bool, 默认为 True
        是否应该使不属于 `expected_warning` 类型的额外警告导致测试失败。
    match : {str, tuple[str, ...]}, 可选
        匹配警告消息。如果是元组，则必须与 `expected_warning` 的大小相同。
        如果此外 `must_find_all_warnings` 为 True，则每个预期警告的消息都将与相应的匹配项匹配。
        否则，多个值将被视为备选项。

    must_find_all_warnings : bool, 默认为 True
        是否必须找到所有预期的警告。
    """
    # 控制是否要求捕获所有预期警告
    must_find_all_warnings : bool, default True
        # 如果为 True 并且 `expected_warning` 是一个元组，则必须遇到每种预期警告类型。
        # 否则，即使遇到一个预期警告也算成功。

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    >>> with assert_produces_warning(False):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning):
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'.

    ..warn:: This is *not* thread-safe.
    """
    # 隐藏此段代码的追溯信息
    __tracebackhide__ = True

    # 使用 `warnings.catch_warnings` 上下文管理器捕获警告
    with warnings.catch_warnings(record=True) as w:
        # 设置警告的过滤级别
        warnings.simplefilter(filter_level)
        try:
            # 执行 yield 语句，允许在上下文中使用 `w` 来引用捕获的警告列表
            yield w
        finally:
            # 如果有预期的警告
            if expected_warning:
                # 如果 `expected_warning` 是一个元组，并且要求捕获所有预期警告
                if isinstance(expected_warning, tuple) and must_find_all_warnings:
                    # 匹配警告的类型和内容
                    match = (
                        match
                        if isinstance(match, tuple)
                        else (match,) * len(expected_warning)
                    )
                    # 遍历预期的警告类型和匹配项，检查是否捕获到了预期的警告
                    for warning_type, warning_match in zip(expected_warning, match):
                        _assert_caught_expected_warnings(
                            caught_warnings=w,
                            expected_warning=warning_type,
                            match=warning_match,
                            check_stacklevel=check_stacklevel,
                        )
                else:
                    # 将 `expected_warning` 转换为正确的类型
                    expected_warning = cast(
                        Union[type[Warning], tuple[type[Warning], ...]],
                        expected_warning,
                    )
                    # 处理匹配内容的格式
                    match = (
                        "|".join(m for m in match if m)
                        if isinstance(match, tuple)
                        else match
                    )
                    # 检查是否捕获到了预期的警告
                    _assert_caught_expected_warnings(
                        caught_warnings=w,
                        expected_warning=expected_warning,
                        match=match,
                        check_stacklevel=check_stacklevel,
                    )
            # 如果设置了 `raise_on_extra_warnings`，则检查是否有额外的未预期警告
            if raise_on_extra_warnings:
                _assert_caught_no_extra_warnings(
                    caught_warnings=w,
                    expected_warning=expected_warning,
                )
# 返回一个上下文管理器，可能根据条件检查警告
def maybe_produces_warning(
    warning: type[Warning], condition: bool, **kwargs
) -> AbstractContextManager:
    if condition:
        # 如果条件为真，调用 assert_produces_warning 函数并返回其结果
        return assert_produces_warning(warning, **kwargs)
    else:
        # 如果条件为假，返回一个空的上下文管理器 nullcontext
        return nullcontext()


# 断言捕获的警告中包含了预期的警告
def _assert_caught_expected_warnings(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning] | tuple[type[Warning], ...],
    match: str | None,
    check_stacklevel: bool,
) -> None:
    """Assert that there was the expected warning among the caught warnings."""
    saw_warning = False  # 标记是否看到了预期的警告
    matched_message = False  # 标记是否匹配了预期的警告消息
    unmatched_messages = []  # 存储未匹配的警告消息列表
    warning_name = (
        tuple(x.__name__ for x in expected_warning)
        if isinstance(expected_warning, tuple)
        else expected_warning.__name__
    )

    # 遍历捕获到的警告列表
    for actual_warning in caught_warnings:
        # 检查捕获到的警告类是否是预期警告类的子类
        if issubclass(actual_warning.category, expected_warning):
            saw_warning = True  # 标记为看到了预期的警告

            # 如果需要检查 stacklevel，则调用 _assert_raised_with_correct_stacklevel 函数
            if check_stacklevel:
                _assert_raised_with_correct_stacklevel(actual_warning)

            # 如果有指定匹配的消息，则检查警告消息是否匹配
            if match is not None:
                if re.search(match, str(actual_warning.message)):
                    matched_message = True  # 标记为消息匹配
                else:
                    unmatched_messages.append(actual_warning.message)  # 添加未匹配的消息至列表

    # 如果未看到预期的警告，则抛出 AssertionError
    if not saw_warning:
        raise AssertionError(f"Did not see expected warning of class {warning_name!r}")

    # 如果有指定匹配消息，并且没有匹配成功，则抛出 AssertionError
    if match and not matched_message:
        raise AssertionError(
            f"Did not see warning {warning_name!r} "
            f"matching '{match}'. The emitted warning messages are "
            f"{unmatched_messages}"
        )


# 断言没有捕获到额外的警告，除了预期的那些
def _assert_caught_no_extra_warnings(
    *,
    caught_warnings: Sequence[warnings.WarningMessage],
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None,
) -> None:
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings = []  # 存储额外捕获的警告列表
    # 遍历捕获到的警告列表中的每一个警告对象
    for actual_warning in caught_warnings:
        # 检查当前警告是否为预期之外的警告
        if _is_unexpected_warning(actual_warning, expected_warning):
            # 如果警告类别为 ResourceWarning
            if actual_warning.category == ResourceWarning:
                # GH 44732: 在依赖项中，避免因 SSL 相关的 ResourceWarning 导致 CI 失败
                if "unclosed <ssl.SSLSocket" in str(actual_warning.message):
                    # 继续下一个警告，不处理这类未关闭的 SSL Socket 警告
                    continue
                # GH 44844: Matplotlib 在整个进程中保持字体文件打开状态，避免因此类警告导致 CI 失败
                if any("matplotlib" in mod for mod in sys.modules):
                    # 继续下一个警告，不处理与 Matplotlib 相关的 ResourceWarning
                    continue
            # 如果配置为 PY311 并且警告类别为 EncodingWarning
            if PY311 and actual_warning.category == EncodingWarning:
                # 在 CI 中会检查 EncodingWarning
                # pyproject.toml 在 pandas 的 EncodingWarning 上会报错
                # 忽略其他库引发的 EncodingWarning
                continue
            # 将未预期的警告信息添加到 extra_warnings 列表中
            extra_warnings.append(
                (
                    actual_warning.category.__name__,
                    actual_warning.message,
                    actual_warning.filename,
                    actual_warning.lineno,
                )
            )

    # 如果 extra_warnings 列表不为空
    if extra_warnings:
        # 抛出断言错误，指示引发了未预期的警告
        raise AssertionError(f"Caused unexpected warning(s): {extra_warnings!r}")
# 判断实际警告是否为预期外的函数
def _is_unexpected_warning(
    actual_warning: warnings.WarningMessage,
    expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None,
) -> bool:
    """Check if the actual warning issued is unexpected."""
    # 如果存在实际警告但预期警告为假，则返回True，表示实际警告是意外的
    if actual_warning and not expected_warning:
        return True
    # 将预期警告转换为Warning类型
    expected_warning = cast(type[Warning], expected_warning)
    # 判断实际警告的类别是否不是预期警告的子类，返回判断结果
    return bool(not issubclass(actual_warning.category, expected_warning))


# 检查是否以正确的stacklevel引发警告
def _assert_raised_with_correct_stacklevel(
    actual_warning: warnings.WarningMessage,
) -> None:
    # 获取当前帧对象
    frame = inspect.currentframe()
    # 向上遍历4层帧
    for _ in range(4):
        frame = frame.f_back  # type: ignore[union-attr]
    try:
        # 获取调用者的文件名
        caller_filename = inspect.getfile(frame)  # type: ignore[arg-type]
    finally:
        # 清理帧对象，参考说明
        del frame
    # 构建警告信息
    msg = (
        "Warning not set with correct stacklevel. "
        f"File where warning is raised: {actual_warning.filename} != "
        f"{caller_filename}. Warning message: {actual_warning.message}"
    )
    # 断言实际警告的文件名与调用者的文件名相同，否则抛出消息msg
    assert actual_warning.filename == caller_filename, msg
```