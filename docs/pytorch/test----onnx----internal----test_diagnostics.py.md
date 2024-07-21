# `.\pytorch\test\onnx\internal\test_diagnostics.py`

```py
# Owner(s): ["module: onnx"]
# 引入未来版本的注解语法支持
from __future__ import annotations

# 引入上下文管理模块
import contextlib
# 引入数据类支持
import dataclasses
# 引入输入输出流模块
import io
# 引入日志记录模块
import logging
# 引入类型提示模块
import typing
# 从类型提示模块中引入特定类型
from typing import AbstractSet, Protocol, Tuple

# 引入PyTorch模块
import torch
# 从torch.onnx模块中引入错误处理相关模块
from torch.onnx import errors
# 从torch.onnx._internal中引入诊断信息相关模块
from torch.onnx._internal import diagnostics
# 从torch.onnx._internal.diagnostics.infra中引入格式化和SARIF相关模块
from torch.onnx._internal.diagnostics.infra import formatter, sarif
# 从torch.onnx._internal.fx.diagnostics中引入诊断信息相关模块
from torch.onnx._internal.fx import diagnostics as fx_diagnostics
# 从torch.testing._internal中引入通用工具和日志工具模块
from torch.testing._internal import common_utils, logging_utils

# 如果是类型检查模式，则引入unittest模块
if typing.TYPE_CHECKING:
    import unittest


# 定义一个协议，用于描述SARIF日志构建器应具备的方法签名
class _SarifLogBuilder(Protocol):
    def sarif_log(self) -> sarif.SarifLog:
        ...


# 断言SARIF日志构建器生成的日志中包含一组规则和级别的诊断信息
def _assert_has_diagnostics(
    sarif_log_builder: _SarifLogBuilder,
    rule_level_pairs: AbstractSet[Tuple[infra.Rule, infra.Level]],
):
    # 获取SARIF日志对象
    sarif_log = sarif_log_builder.sarif_log()
    # 创建未见过的规则ID和级别的集合
    unseen_pairs = {(rule.id, level.name.lower()) for rule, level in rule_level_pairs}
    # 存储实际结果的列表
    actual_results = []
    # 遍历SARIF日志中的每一个运行结果
    for run in sarif_log.runs:
        if run.results is None:
            continue
        for result in run.results:
            # 获取结果的规则ID和级别
            id_level_pair = (result.rule_id, result.level)
            # 从未见过的集合中移除当前结果
            unseen_pairs.discard(id_level_pair)
            # 将结果添加到实际结果列表中
            actual_results.append(id_level_pair)

    # 如果仍有未见过的规则ID和级别，则抛出断言错误
    if unseen_pairs:
        raise AssertionError(
            f"Expected diagnostic results of rule id and level pair {unseen_pairs} not found. "
            f"Actual diagnostic results: {actual_results}"
        )


# 数据类，用于测试目的的规则集合，继承自infra.RuleCollection
@dataclasses.dataclass
class _RuleCollectionForTest(infra.RuleCollection):
    # 默认规则，没有消息参数
    rule_without_message_args: infra.Rule = dataclasses.field(
        default=infra.Rule(
            "1",
            "rule-without-message-args",
            message_default_template="rule message",
        )
    )


# 上下文管理器，用于断言所有诊断信息都已生成
@contextlib.contextmanager
def assert_all_diagnostics(
    test_suite: unittest.TestCase,
    sarif_log_builder: _SarifLogBuilder,
    rule_level_pairs: AbstractSet[Tuple[infra.Rule, infra.Level]],
):
    """Context manager to assert that all diagnostics are emitted.

    Usage:
        with assert_all_diagnostics(
            self,
            diagnostics.engine,
            {(rule, infra.Level.Error)},
        ):
            torch.onnx.export(...)

    Args:
        test_suite: The test suite instance.
        sarif_log_builder: The SARIF log builder.
        rule_level_pairs: A set of rule and level pairs to assert.

    Returns:
        A context manager.

    Raises:
        AssertionError: If not all diagnostics are emitted.
    """

    try:
        yield
    except errors.OnnxExporterError:
        # 断言错误中包含至少一个错误级别的诊断信息
        test_suite.assertIn(infra.Level.ERROR, {level for _, level in rule_level_pairs})
    finally:
        # 断言SARIF日志生成了所有规则和级别的诊断信息
        _assert_has_diagnostics(sarif_log_builder, rule_level_pairs)


# 上下文管理器，用于断言特定规则和级别的诊断信息已生成
def assert_diagnostic(
    test_suite: unittest.TestCase,
    sarif_log_builder: _SarifLogBuilder,
    rule: infra.Rule,
    level: infra.Level,
):
    """Context manager to assert that a diagnostic is emitted.
    # 使用说明文档：
    # 使用assert_diagnostic上下文管理器来验证特定条件下的诊断信息是否符合预期
    # 参数：
    # - self: 测试类实例本身
    # - diagnostics.engine: 诊断引擎实例
    # - rule: 要断言的规则
    # - infra.Level.Error: 断言的错误级别
    # 功能：
    # 在torch.onnx.export(...)期间执行一些诊断检查
    with assert_diagnostic(
        self,
        diagnostics.engine,
        rule,
        infra.Level.Error,
    ):
        torch.onnx.export(...)

    # 返回：
    # 返回一个上下文管理器，用于执行特定测试条件下的诊断验证
    return assert_all_diagnostics(test_suite, sarif_log_builder, {(rule, level)})
class TestDynamoOnnxDiagnostics(common_utils.TestCase):
    """Test cases for diagnostics emitted by the Dynamo ONNX export code."""

    def setUp(self):
        # 创建一个诊断上下文对象，用于记录诊断信息，命名为"dynamo_export"
        self.diagnostic_context = fx_diagnostics.DiagnosticContext("dynamo_export", "")
        # 创建一个用于测试的规则集合对象
        self.rules = _RuleCollectionForTest()
        # 调用父类的setUp方法进行初始化设置
        return super().setUp()

    def test_log_is_recorded_in_sarif_additional_messages_according_to_diagnostic_options_verbosity_level(
        self,
    ):
        # 定义不同的日志级别列表
        logging_levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ]
        # 遍历每个日志级别
        for verbosity_level in logging_levels:
            # 设置诊断上下文的verbosity_level属性为当前遍历到的日志级别
            self.diagnostic_context.options.verbosity_level = verbosity_level
            # 进入诊断上下文，记录诊断信息
            with self.diagnostic_context:
                # 创建一个诊断对象，使用一个没有消息参数的规则进行初始化
                diagnostic = fx_diagnostics.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.NONE
                )
                # 记录当前附加消息的数量
                additional_messages_count = len(diagnostic.additional_messages)
                # 对每个日志级别进行记录日志消息
                for log_level in logging_levels:
                    diagnostic.log(level=log_level, message="log message")
                    # 如果当前日志级别大于或等于当前设置的verbosity_level
                    if log_level >= verbosity_level:
                        # 断言附加消息数量增加
                        self.assertGreater(
                            len(diagnostic.additional_messages),
                            additional_messages_count,
                            f"Additional message should be recorded when log level is {log_level} "
                            f"and verbosity level is {verbosity_level}",
                        )
                    else:
                        # 断言附加消息数量不变
                        self.assertEqual(
                            len(diagnostic.additional_messages),
                            additional_messages_count,
                            f"Additional message should not be recorded when log level is "
                            f"{log_level} and verbosity level is {verbosity_level}",
                        )

    def test_torch_logs_environment_variable_precedes_diagnostic_options_verbosity_level(
        self,
    ):
        # 设置诊断上下文的verbosity_level属性为ERROR级别
        self.diagnostic_context.options.verbosity_level = logging.ERROR
        # 在"onnx_diagnostics"日志设置下，进入诊断上下文，记录诊断信息
        with logging_utils.log_settings("onnx_diagnostics"), self.diagnostic_context:
            # 创建一个诊断对象，使用一个没有消息参数的规则进行初始化
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NONE
            )
            # 记录当前附加消息的数量
            additional_messages_count = len(diagnostic.additional_messages)
            # 记录DEBUG级别的消息
            diagnostic.debug("message")
            # 断言附加消息数量增加
            self.assertGreater(
                len(diagnostic.additional_messages), additional_messages_count
            )
    # 测试日志在未启用日志工件时不会输出到终端
    def test_log_is_not_emitted_to_terminal_when_log_artifact_is_not_enabled(self):
        # 设置诊断上下文的日志详细级别为 INFO
        self.diagnostic_context.options.verbosity_level = logging.INFO
        # 进入诊断上下文
        with self.diagnostic_context:
            # 创建一个 Diagnostic 对象，指定规则和级别为 NONE
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NONE
            )

            # 使用 assertLogs 断言，检查在 INFO 级别是否记录了日志
            with self.assertLogs(
                diagnostic.logger, level=logging.INFO
            ) as assert_log_context:
                # 记录一条 INFO 级别的日志消息
                diagnostic.info("message")
                # 注意：self.assertNoLogs 只在 Python 3.10 及以上版本支持
                # 添加这条虚拟日志以便通过 self.assertLogs，并检查 assert_log_context.records，
                # 确保我们不希望的日志没有被输出。
                diagnostic.logger.log(logging.ERROR, "dummy message")

            # 断言确保只有一条日志被记录
            self.assertEqual(len(assert_log_context.records), 1)

    # 测试日志在启用日志工件时输出到终端
    def test_log_is_emitted_to_terminal_when_log_artifact_is_enabled(self):
        # 设置诊断上下文的日志详细级别为 INFO
        self.diagnostic_context.options.verbosity_level = logging.INFO

        # 设置日志记录配置为 "onnx_diagnostics"，并进入诊断上下文
        with logging_utils.log_settings("onnx_diagnostics"), self.diagnostic_context:
            # 创建一个 Diagnostic 对象，指定规则和级别为 NONE
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NONE
            )

            # 使用 assertLogs 断言，检查在 INFO 级别是否记录了日志
            with self.assertLogs(diagnostic.logger, level=logging.INFO):
                # 记录一条 INFO 级别的日志消息
                diagnostic.info("message")

    # 测试诊断日志以正确格式化的字符串输出
    def test_diagnostic_log_emit_correctly_formatted_string(self):
        # 设置日志详细级别为 INFO
        verbosity_level = logging.INFO
        self.diagnostic_context.options.verbosity_level = verbosity_level

        # 进入诊断上下文
        with self.diagnostic_context:
            # 创建一个 Diagnostic 对象，指定规则和级别为 NOTE
            diagnostic = fx_diagnostics.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 记录一条日志，使用 LazyString 格式化消息
            diagnostic.log(
                logging.INFO,
                "%s",
                formatter.LazyString(lambda x, y: f"{x} {y}", "hello", "world"),
            )
            # 断言确保 "hello world" 在 diagnostic.additional_messages 中
            self.assertIn("hello world", diagnostic.additional_messages)

    # 测试当诊断类型错误时，将诊断日志记录到诊断上下文会引发异常
    def test_log_diagnostic_to_diagnostic_context_raises_when_diagnostic_type_is_wrong(
        self,
    ):
        # 进入诊断上下文
        with self.diagnostic_context:
            # 创建一个 base infra.Diagnostic 对象，而不是预期的 fx_diagnostics.Diagnostic
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )
            # 使用 assertRaises 断言捕获 TypeError 异常
            with self.assertRaises(TypeError):
                # 尝试将错误类型的诊断记录到诊断上下文
                self.diagnostic_context.log(diagnostic)
    def setUp(self):
        # 获取诊断引擎实例
        engine = diagnostics.engine
        # 清空引擎中的诊断信息
        engine.clear()
        # 设置一个示例规则为缺失自定义符号函数的诊断规则
        self._sample_rule = diagnostics.rules.missing_custom_symbolic_function
        # 调用父类的setUp方法进行初始化
        super().setUp()

    def _trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp(
        self,
    ) -> diagnostics.TorchScriptOnnxExportDiagnostic:
        # 定义一个自定义的 TorchScript 函数 CustomAdd
        class CustomAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def symbolic(g, x, y):
                # 使用自定义操作符 "custom::CustomAdd" 对 x 和 y 执行符号操作
                return g.op("custom::CustomAdd", x, y)

        # 定义一个简单的 Module M，其 forward 方法中使用了 CustomAdd 函数
        class M(torch.nn.Module):
            def forward(self, x):
                return CustomAdd.apply(x, x)

        # 触发一个缺失 ONNX 形状推断警告的导出
        rule = diagnostics.rules.node_missing_onnx_shape_inference
        torch.onnx.export(M(), torch.randn(3, 4), io.BytesIO())

        # 获取最后一个上下文中的诊断信息
        context = diagnostics.engine.contexts[-1]
        # 遍历该上下文中的诊断信息
        for diagnostic in context.diagnostics:
            # 如果找到了符合规则并且级别为警告的诊断信息，则返回该诊断信息
            if (
                diagnostic.rule == rule
                and diagnostic.level == diagnostics.levels.WARNING
            ):
                return typing.cast(
                    diagnostics.TorchScriptOnnxExportDiagnostic, diagnostic
                )
        # 如果没有找到符合条件的诊断信息，则抛出断言错误
        raise AssertionError("No diagnostic found.")

    def test_assert_diagnostic_raises_when_diagnostic_not_found(self):
        # 测试断言：当未找到预期的诊断信息时应该抛出断言错误
        with self.assertRaises(AssertionError):
            with assert_diagnostic(
                self,
                diagnostics.engine,
                diagnostics.rules.node_missing_onnx_shape_inference,
                diagnostics.levels.WARNING,
            ):
                pass

    def test_cpp_diagnose_emits_warning(self):
        # 测试：CPP 诊断应该触发警告
        with assert_diagnostic(
            self,
            diagnostics.engine,
            diagnostics.rules.node_missing_onnx_shape_inference,
            diagnostics.levels.WARNING,
        ):
            # 触发一个缺失 ONNX 形状推断警告的诊断
            self._trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp()

    def test_py_diagnose_emits_error(self):
        # 测试：Python 诊断应该触发错误
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.diagonal(x)

        with assert_diagnostic(
            self,
            diagnostics.engine,
            diagnostics.rules.operator_supported_in_newer_opset_version,
            diagnostics.levels.ERROR,
        ):
            # 触发一个在较新的 opset 版本中不支持的操作符导出的错误诊断
            torch.onnx.export(
                M(),
                torch.randn(3, 4),
                io.BytesIO(),
                opset_version=9,
            )

    def test_diagnostics_engine_records_diagnosis_reported_outside_of_export(
        self,
    ):
        # 测试：诊断引擎应记录在导出之外报告的诊断
        # （该方法的具体实现在这里省略）
        pass
    ):
        # 设置样本级别为错误级别
        sample_level = diagnostics.levels.ERROR
        # 使用断言来验证诊断信息
        with assert_diagnostic(
            self,
            diagnostics.engine,
            self._sample_rule,
            sample_level,
        ):
            # 创建诊断对象
            diagnostic = infra.Diagnostic(self._sample_rule, sample_level)
            # 将诊断信息记录到诊断输出中
            diagnostics.export_context().log(diagnostic)

    def test_diagnostics_records_python_call_stack(self):
        # 创建 TorchScriptOnnxExportDiagnostic 对象用于记录 Python 调用栈的诊断信息
        diagnostic = diagnostics.TorchScriptOnnxExportDiagnostic(self._sample_rule, diagnostics.levels.NOTE)  # fmt: skip
        # 确保 Python 调用栈不为空
        stack = diagnostic.python_call_stack
        assert stack is not None  # for mypy
        # 断言调用栈帧数大于 0
        self.assertGreater(len(stack.frames), 0)
        # 获取第一个帧对象
        frame = stack.frames[0]
        # 确保帧对象中的代码片段不为空，并包含 "self._sample_rule"
        assert frame.location.snippet is not None  # for mypy
        self.assertIn("self._sample_rule", frame.location.snippet)
        # 确保帧对象中的 URI 不为空，并包含 "test_diagnostics.py"
        assert frame.location.uri is not None  # for mypy
        self.assertIn("test_diagnostics.py", frame.location.uri)

    def test_diagnostics_records_cpp_call_stack(self):
        # 触发从 C++ 端生成的节点缺失 ONNX 形状推断警告的诊断记录
        diagnostic = (
            self._trigger_node_missing_onnx_shape_inference_warning_diagnostic_from_cpp()
        )
        # 获取 C++ 调用栈
        stack = diagnostic.cpp_call_stack
        assert stack is not None  # for mypy
        # 断言 C++ 调用栈帧数大于 0
        self.assertGreater(len(stack.frames), 0)
        # 提取帧对象中的消息列表
        frame_messages = [frame.location.message for frame in stack.frames]
        # 断言至少有一条消息包含 "torch::jit::NodeToONNX"
        self.assertTrue(
            any(
                isinstance(message, str) and "torch::jit::NodeToONNX" in message
                for message in frame_messages
            )
        )
@common_utils.instantiate_parametrized_tests
class TestDiagnosticsInfra(common_utils.TestCase):
    """Test cases for diagnostics infra."""

    def setUp(self):
        # 使用自定义的测试规则集初始化 self.rules
        self.rules = _RuleCollectionForTest()
        # 使用 ExitStack 确保资源的正确释放
        with contextlib.ExitStack() as stack:
            # 创建 DiagnosticContext 对象并赋给 self.context，用于记录诊断信息
            self.context: infra.DiagnosticContext[
                infra.Diagnostic
            ] = stack.enter_context(infra.DiagnosticContext("test", "1.0.0"))
            # 添加清理操作，确保在测试结束时关闭所有已打开的资源
            self.addCleanup(stack.pop_all().close)
        return super().setUp()

    def test_diagnostics_engine_records_diagnosis_with_custom_rules(self):
        # 创建自定义的规则集 custom_rules
        custom_rules = infra.RuleCollection.custom_collection_from_list(
            "CustomRuleCollection",
            [
                infra.Rule(
                    "1",
                    "custom-rule",
                    message_default_template="custom rule message",
                ),
                infra.Rule(
                    "2",
                    "custom-rule-2",
                    message_default_template="custom rule message 2",
                ),
            ],
        )

        # 使用 assert_all_diagnostics 上下文管理器，确保所有的诊断都被正确记录和检查
        with assert_all_diagnostics(
            self,
            self.context,
            {
                (custom_rules.custom_rule, infra.Level.WARNING),  # type: ignore[attr-defined]
                (custom_rules.custom_rule_2, infra.Level.ERROR),  # type: ignore[attr-defined]
            },
        ):
            # 创建并记录第一个诊断
            diagnostic1 = infra.Diagnostic(
                custom_rules.custom_rule, infra.Level.WARNING  # type: ignore[attr-defined]
            )
            self.context.log(diagnostic1)

            # 创建并记录第二个诊断
            diagnostic2 = infra.Diagnostic(
                custom_rules.custom_rule_2, infra.Level.ERROR  # type: ignore[attr-defined]
            )
            self.context.log(diagnostic2)

    def test_diagnostic_log_is_not_emitted_when_level_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        # 设置日志输出的详细级别为 INFO
        verbosity_level = logging.INFO
        self.context.options.verbosity_level = verbosity_level
        # 进入 DiagnosticContext 上下文管理器，确保所有的日志在适当的上下文中记录
        with self.context:
            # 创建一个 DIAGNOSTIC 级别的诊断
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 使用 assertLogs 上下文管理器来断言日志的记录情况
            with self.assertLogs(
                diagnostic.logger, level=verbosity_level
            ) as assert_log_context:
                # 记录一个 DEBUG 级别的信息，用于检查是否能正确忽略
                diagnostic.log(logging.DEBUG, "debug message")
                # 添加一个 INFO 级别的虚拟日志以便通过 assertLogs 检查记录的级别是否正确
                diagnostic.log(logging.INFO, "info message")

        # 检查所有记录的日志级别是否都大于或等于 INFO
        for record in assert_log_context.records:
            self.assertGreaterEqual(record.levelno, logging.INFO)
        
        # 检查是否不包含特定的 DEBUG 级别的附加消息
        self.assertFalse(
            any(
                message.find("debug message") >= 0
                for message in diagnostic.additional_messages
            )
        )
    def test_diagnostic_log_is_emitted_when_level_not_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        # 设定日志输出级别为 INFO
        verbosity_level = logging.INFO
        # 设置上下文的详细输出级别
        self.context.options.verbosity_level = verbosity_level
        # 在上下文中执行以下操作
        with self.context:
            # 创建一个 Diagnostic 对象，指定一个没有消息参数的规则和 NOTE 级别
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 定义不同级别的日志消息对
            level_message_pairs = [
                (logging.INFO, "info message"),
                (logging.WARNING, "warning message"),
                (logging.ERROR, "error message"),
            ]

            # 遍历日志级别和消息对
            for level, message in level_message_pairs:
                # 使用 assertLogs 断言日志输出，指定日志记录器和日志级别
                with self.assertLogs(diagnostic.logger, level=verbosity_level):
                    # 记录特定级别的消息
                    diagnostic.log(level, message)

            # 断言附加消息列表中至少包含一个匹配的消息
            self.assertTrue(
                any(
                    message.find(message) >= 0
                    for message in diagnostic.additional_messages
                )
            )

    @common_utils.parametrize(
        "log_api, log_level",
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
        ],
    )
    def test_diagnostic_log_is_emitted_according_to_api_level_and_diagnostic_options_verbosity_level(
        self, log_api: str, log_level: int
    ):
        # 设定日志输出级别为 INFO
        verbosity_level = logging.INFO
        # 设置上下文的详细输出级别
        self.context.options.verbosity_level = verbosity_level
        # 在上下文中执行以下操作
        with self.context:
            # 创建一个 Diagnostic 对象，指定一个没有消息参数的规则和 NOTE 级别
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 定义日志消息
            message = "log message"
            # 使用 assertLogs 断言日志输出，指定日志记录器和日志级别
            with self.assertLogs(
                diagnostic.logger, level=verbosity_level
            ) as assert_log_context:
                # 调用 Diagnostic 对象的特定日志 API，记录消息
                getattr(diagnostic, log_api)(message)
                # 添加一个虚拟日志，以便通过 self.assertLogs 通过检查 assert_log_context.records 来验证日志级别是否正确
                diagnostic.log(logging.ERROR, "dummy message")

            # 遍历 assert_log_context.records 中的记录
            for record in assert_log_context.records:
                # 断言记录的级别不小于设定的 INFO 级别
                self.assertGreaterEqual(record.levelno, logging.INFO)

            # 如果 log_level 大于等于 verbosity_level，则断言消息在附加消息列表中
            if log_level >= verbosity_level:
                self.assertIn(message, diagnostic.additional_messages)
            # 否则断言消息不在附加消息列表中
            else:
                self.assertNotIn(message, diagnostic.additional_messages)

    def test_diagnostic_log_lazy_string_is_not_evaluated_when_level_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        # 设定日志详细级别为 INFO
        verbosity_level = logging.INFO
        # 将上下文的详细级别设定为 verbosity_level
        self.context.options.verbosity_level = verbosity_level
        # 进入上下文环境
        with self.context:
            # 创建一个诊断对象，使用指定的规则和级别 NOTE
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 设定初始参考值为 0
            reference_val = 0

            # 定义一个昂贵的格式化函数
            def expensive_formatting_function() -> str:
                # 使用 nonlocal 修改 reference_val，以反映函数已被评估
                nonlocal reference_val
                reference_val += 1
                return f"expensive formatting {reference_val}"

            # 使用 LazyString 包装的方式调用 expensive_formatting_function，不应该被评估
            diagnostic.debug("%s", formatter.LazyString(expensive_formatting_function))
            # 断言 reference_val 仍为 0，因为 LazyString 应延迟评估
            self.assertEqual(
                reference_val,
                0,
                "expensive_formatting_function should not be evaluated after being wrapped under LazyString",
            )

    def test_diagnostic_log_lazy_string_is_evaluated_once_when_level_not_less_than_diagnostic_options_verbosity_level(
        self,
    ):
        # 设定日志详细级别为 INFO
        verbosity_level = logging.INFO
        # 将上下文的详细级别设定为 verbosity_level
        self.context.options.verbosity_level = verbosity_level
        # 进入上下文环境
        with self.context:
            # 创建一个诊断对象，使用指定的规则和级别 NOTE
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 设定初始参考值为 0
            reference_val = 0

            # 定义一个昂贵的格式化函数
            def expensive_formatting_function() -> str:
                # 使用 nonlocal 修改 reference_val，以反映函数已被评估
                nonlocal reference_val
                reference_val += 1
                return f"expensive formatting {reference_val}"

            # 使用 LazyString 包装的方式调用 expensive_formatting_function，应该被评估一次
            diagnostic.info("%s", formatter.LazyString(expensive_formatting_function))
            # 断言 reference_val 现在为 1，因为 LazyString 应在调用时评估
            self.assertEqual(
                reference_val,
                1,
                "expensive_formatting_function should only be evaluated once after being wrapped under LazyString",
            )

    def test_diagnostic_log_emit_correctly_formatted_string(self):
        # 设定日志详细级别为 INFO
        verbosity_level = logging.INFO
        # 将上下文的详细级别设定为 verbosity_level
        self.context.options.verbosity_level = verbosity_level
        # 进入上下文环境
        with self.context:
            # 创建一个诊断对象，使用指定的规则和级别 NOTE
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )
            # 使用 lambda 表达式创建一个 LazyString，以 "hello" 和 "world" 作为参数
            diagnostic.log(
                logging.INFO,
                "%s",
                formatter.LazyString(lambda x, y: f"{x} {y}", "hello", "world"),
            )
            # 断言 "hello world" 存在于 diagnostic 的附加消息中
            self.assertIn("hello world", diagnostic.additional_messages)

    def test_diagnostic_nested_log_section_emits_messages_with_correct_section_title_indentation(
        self,
    ):
        # 设置日志的详细程度为 INFO
        verbosity_level = logging.INFO
        # 将日志的详细程度设置到上下文选项中
        self.context.options.verbosity_level = verbosity_level
        # 进入上下文环境
        with self.context:
            # 创建一个诊断对象，用于处理没有消息参数的规则
            diagnostic = infra.Diagnostic(
                self.rules.rule_without_message_args, infra.Level.NOTE
            )

            # 在诊断对象中记录一个名为 "My Section" 的日志段落
            with diagnostic.log_section(logging.INFO, "My Section"):
                # 在 "My Section" 日志段落中记录一条 INFO 级别的消息
                diagnostic.log(logging.INFO, "My Message")
                # 在 "My Section" 下创建一个名为 "My Subsection" 的子段落
                with diagnostic.log_section(logging.INFO, "My Subsection"):
                    # 在 "My Subsection" 中记录一条 INFO 级别的子消息
                    diagnostic.log(logging.INFO, "My Submessage")

            # 在诊断对象中记录一个名为 "My Section 2" 的另一个日志段落
            with diagnostic.log_section(logging.INFO, "My Section 2"):
                # 在 "My Section 2" 日志段落中记录一条 INFO 级别的消息
                diagnostic.log(logging.INFO, "My Message 2")

            # 确认 "My Section"、"My Subsection" 和 "My Section 2" 这些消息存在于附加消息中
            self.assertIn("## My Section", diagnostic.additional_messages)
            self.assertIn("### My Subsection", diagnostic.additional_messages)
            self.assertIn("## My Section 2", diagnostic.additional_messages)

    # 测试在诊断日志中记录源异常时是否会输出异常的追踪和错误消息
    def test_diagnostic_log_source_exception_emits_exception_traceback_and_error_message(
        self,
    ):
        # 设置日志的详细程度为 INFO
        verbosity_level = logging.INFO
        # 将日志的详细程度设置到上下文选项中
        self.context.options.verbosity_level = verbosity_level
        # 进入上下文环境
        with self.context:
            try:
                # 抛出一个值错误异常
                raise ValueError("original exception")
            except ValueError as e:
                # 创建一个诊断对象，用于处理没有消息参数的规则
                diagnostic = infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.NOTE
                )
                # 记录源异常的追踪和错误消息到诊断对象中
                diagnostic.log_source_exception(logging.ERROR, e)

            # 将诊断对象中的所有附加消息连接成字符串
            diagnostic_message = "\n".join(diagnostic.additional_messages)

            # 确认异常的错误消息和追踪信息在诊断消息中
            self.assertIn("ValueError: original exception", diagnostic_message)
            self.assertIn("Traceback (most recent call last):", diagnostic_message)

    # 测试在诊断上下文中记录日志时，如果日志类型错误会触发 TypeError 异常
    def test_log_diagnostic_to_diagnostic_context_raises_when_diagnostic_type_is_wrong(
        self,
    ):
        # 进入上下文环境
        with self.context:
            # 确认抛出 TypeError 异常，因为日志方法要求传入 'Diagnostic' 类或其子类的对象
            with self.assertRaises(TypeError):
                self.context.log("This is a str message.")

    # 测试在诊断上下文中，如果诊断为错误时会触发 infra.RuntimeErrorWithDiagnostic 异常
    def test_diagnostic_context_raises_if_diagnostic_is_error(self):
        # 确认抛出 infra.RuntimeErrorWithDiagnostic 异常，因为尝试记录错误级别的诊断
        with self.assertRaises(infra.RuntimeErrorWithDiagnostic):
            self.context.log_and_raise_if_error(
                infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.ERROR
                )
            )

    # 测试在诊断上下文中，如果从异常创建诊断对象会抛出原始异常
    def test_diagnostic_context_raises_original_exception_from_diagnostic_created_from_it(
        self,
    ):
        # 确认抛出值错误异常，因为从异常创建了诊断对象并尝试记录错误级别的日志
        with self.assertRaises(ValueError):
            try:
                # 抛出一个值错误异常
                raise ValueError("original exception")
            except ValueError as e:
                # 创建一个诊断对象，用于处理没有消息参数的规则
                diagnostic = infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.ERROR
                )
                # 记录源异常的追踪和错误消息到诊断对象中
                diagnostic.log_source_exception(logging.ERROR, e)
                # 将诊断对象传递给上下文的方法，如果有错误则抛出
                self.context.log_and_raise_if_error(diagnostic)
    # 定义一个测试方法，验证在诊断为警告且将警告视为错误时，抛出 infra.RuntimeErrorWithDiagnostic 异常
    def test_diagnostic_context_raises_if_diagnostic_is_warning_and_warnings_as_errors_is_true(
        self,
    ):
        # 使用 self.assertRaises 断言，期望抛出 infra.RuntimeErrorWithDiagnostic 异常
        with self.assertRaises(infra.RuntimeErrorWithDiagnostic):
            # 设置 self.context.options.warnings_as_errors 为 True，将警告视为错误
            self.context.options.warnings_as_errors = True
            # 调用 self.context.log_and_raise_if_error 方法，传入一个 infra.Diagnostic 对象
            # 该 Diagnostic 对象使用 self.rules.rule_without_message_args 和 infra.Level.WARNING 初始化
            self.context.log_and_raise_if_error(
                infra.Diagnostic(
                    self.rules.rule_without_message_args, infra.Level.WARNING
                )
            )
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，通常用于执行单元测试
    common_utils.run_tests()
```