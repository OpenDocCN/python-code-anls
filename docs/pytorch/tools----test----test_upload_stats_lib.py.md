# `.\pytorch\tools\test\test_upload_stats_lib.py`

```
# 从未来版本中导入注解支持
from __future__ import annotations

# 导入decimal模块，用于精确计算
import decimal
# 导入inspect模块，用于检查对象
import inspect
# 导入sys模块，提供对Python解释器的访问
import sys
# 导入unittest模块，用于编写和运行单元测试
import unittest
# 从pathlib模块中导入Path类，用于处理文件路径
from pathlib import Path
# 导入Any类型，用于表示任意类型
from typing import Any
# 导入mock类，用于模拟对象行为
from unittest import mock

# 获取当前文件的父目录的父目录的父目录，作为项目根目录
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# 将项目根目录添加到sys.path中，以便在此后导入的模块中可以访问到项目的其他模块
sys.path.insert(0, str(REPO_ROOT))

# 从tools.stats.upload_metrics模块导入add_global_metric和emit_metric函数
from tools.stats.upload_metrics import add_global_metric, emit_metric
# 从tools.stats.upload_stats_lib模块导入BATCH_SIZE和upload_to_rockset常量或函数

# 从sys.path中移除项目根目录，恢复环境设置
sys.path.remove(str(REPO_ROOT))

# 默认数值设置
REPO = "some/repo"
BUILD_ENV = "cuda-10.2"
TEST_CONFIG = "test-config"
WORKFLOW = "some-workflow"
JOB = "some-job"
RUN_ID = 56
RUN_NUMBER = 123
RUN_ATTEMPT = 3
PR_NUMBER = 6789
JOB_ID = 234
JOB_NAME = "some-job-name"

# 定义一个单元测试类TestUploadStats，继承自unittest.TestCase
class TestUploadStats(unittest.TestCase):
    
    # 在每个测试方法执行前调用，设置环境变量为默认值
    def setUp(self) -> None:
        # 使用mock.patch.dict方法，模拟os.environ字典
        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
                "TEST_CONFIG": TEST_CONFIG,
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_WORKFLOW": WORKFLOW,
                "GITHUB_JOB": JOB,
                "GITHUB_RUN_ID": str(RUN_ID),
                "GITHUB_RUN_NUMBER": str(RUN_NUMBER),
                "GITHUB_RUN_ATTEMPT": str(RUN_ATTEMPT),
                "JOB_ID": str(JOB_ID),
                "JOB_NAME": str(JOB_NAME),
            },
            clear=True,  # 清除任何预设的环境变量
        ).start()

    # 使用mock.patch装饰器，模拟boto3.Session.resource方法
    @mock.patch("boto3.Session.resource")
    def test_emits_default_and_given_metrics(self, mock_resource: Any) -> None:
        metric = {
            "some_number": 123,
            "float_number": 32.34,
        }

        # 查询当前模块的名称，而不是硬编码，因为根据运行环境不同可能会变化
        current_module = inspect.getmodule(inspect.currentframe()).__name__  # type: ignore[union-attr]

        # 准备应该包含在度量输出中的信息
        emit_should_include = {
            "metric_name": "metric_name",
            "calling_file": "test_upload_stats_lib.py",
            "calling_module": current_module,
            "calling_function": "test_emits_default_and_given_metrics",
            "repo": REPO,
            "workflow": WORKFLOW,
            "build_environment": BUILD_ENV,
            "job": JOB,
            "test_config": TEST_CONFIG,
            "run_id": RUN_ID,
            "run_number": RUN_NUMBER,
            "run_attempt": RUN_ATTEMPT,
            "some_number": 123,
            "float_number": decimal.Decimal(str(32.34)),
            "job_id": JOB_ID,
            "job_name": JOB_NAME,
        }

        # 保存被发出的度量
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        # 模拟资源的行为
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 发出度量
        emit_metric("metric_name", metric)

        # 断言发出的度量与预期的包含信息一致
        self.assertEqual(
            emitted_metric,
            {**emit_should_include, **emitted_metric},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_global_metric_specified_then_it_emits_it(
        self, mock_resource: Any
    ) -> None:
        metric = {
            "some_number": 123,
        }

        global_metric_name = "global_metric"
        global_metric_value = "global_value"

        # 添加全局度量
        add_global_metric(global_metric_name, global_metric_value)

        # 准备应该包含在度量输出中的信息，包括全局度量
        emit_should_include = {
            **metric,
            global_metric_name: global_metric_value,
        }

        # 保存被发出的度量
        emitted_metric: dict[str, Any] = {}

        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        # 模拟资源的行为
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 发出度量
        emit_metric("metric_name", metric)

        # 断言发出的度量与预期的包含信息一致
        self.assertEqual(
            emitted_metric,
            {**emitted_metric, **emit_should_include},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_local_and_global_metric_specified_then_global_is_overridden(
        self, mock_resource: Any
    ) -> None:


这些注释解释了每行代码的具体作用，包括变量的定义、函数的调用、模拟资源的设置以及预期的度量输出。
    ) -> None:
        # 定义全局指标名称和值
        global_metric_name = "global_metric"
        global_metric_value = "global_value"
        local_override = "local_override"

        # 调用函数将全局指标添加到指标集合中
        add_global_metric(global_metric_name, global_metric_value)

        # 创建一个包含指标的字典
        metric = {
            "some_number": 123,
            global_metric_name: local_override,
        }

        # 创建一个应包含的指标字典，包括全局指标和本地覆盖值
        emit_should_include = {
            **metric,
            global_metric_name: local_override,
        }

        # 保留发出的指标
        emitted_metric: dict[str, Any] = {}

        # 定义模拟的 put_item 函数，用于保存发出的指标
        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        # 设置模拟资源的 Table 的 put_item 方法为 mock_put_item 函数
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 调用 emit_metric 函数发出指标
        emit_metric("metric_name", metric)

        # 断言发出的指标与预期的 emit_should_include 相同
        self.assertEqual(
            emitted_metric,
            {**emitted_metric, **emit_should_include},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_optional_envvar_set_to_actual_value_then_emit_vars_emits_it(
        self, mock_resource: Any
    ) -> None:
        # 定义指标字典，其中包含一个数字和可能的环境变量
        metric = {
            "some_number": 123,
        }

        # 定义应包含的指标字典，包括之前定义的指标和环境变量 PR_NUMBER 的值
        emit_should_include = {
            **metric,
            "pr_number": PR_NUMBER,
        }

        # 模拟设置环境变量 PR_NUMBER 的值
        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": str(PR_NUMBER),
            },
        ).start()

        # 保留发出的指标
        emitted_metric: dict[str, Any] = {}

        # 定义模拟的 put_item 函数，用于保存发出的指标
        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        # 设置模拟资源的 Table 的 put_item 方法为 mock_put_item 函数
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 调用 emit_metric 函数发出指标
        emit_metric("metric_name", metric)

        # 断言发出的指标与预期的 emit_should_include 相同
        self.assertEqual(
            emitted_metric,
            {**emit_should_include, **emitted_metric},
        )

    @mock.patch("boto3.Session.resource")
    def test_when_optional_envvar_set_to_a_empty_str_then_emit_vars_ignores_it(
        self, mock_resource: Any
    ) -> None:
        #
    ) -> None:
        # 定义一个名为 metric 的字典，包含一个键值对 "some_number": 123
        metric = {"some_number": 123}

        # 创建 emit_should_include 字典，复制 metric 的内容
        emit_should_include: dict[str, Any] = metric.copy()

        # 设置 default_val 为一个空字符串，用于模拟环境变量
        default_val = ""
        # 使用 mock.patch.dict 来模拟设置环境变量 "PR_NUMBER" 为 default_val
        mock.patch.dict(
            "os.environ",
            {
                "PR_NUMBER": default_val,
            },
        ).start()

        # 初始化 emitted_metric 字典，用于存储发出的度量信息
        emitted_metric: dict[str, Any] = {}

        # 定义 mock_put_item 函数，用于捕获传入的参数 Item，并存储到 emitted_metric
        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal emitted_metric
            emitted_metric = Item

        # 将 mock_put_item 函数绑定到 mock_resource 返回的 Table 的 put_item 方法上
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 调用 emit_metric 函数来发出度量信息 "metric_name", metric
        emit_metric("metric_name", metric)

        # 断言发出的度量信息等于 emit_should_include 和 emitted_metric 的合并结果
        self.assertEqual(
            emitted_metric,
            {**emit_should_include, **emitted_metric},
            f"Metrics should be emitted when an option parameter is set to '{default_val}'",
        )

        # 断言发出的度量信息中不包含键 "pr_number"
        self.assertFalse(
            emitted_metric.get("pr_number"),
            f"Metrics should not include optional item 'pr_number' when it's envvar is set to '{default_val}'",
        )

    @mock.patch("boto3.Session.resource")
    def test_blocks_emission_if_reserved_keyword_used(self, mock_resource: Any) -> None:
        # 定义一个名为 metric 的字典，包含一个键值对 "repo": "awesome/repo"
        metric = {"repo": "awesome/repo"}

        # 使用 self.assertRaises 来断言 emit_metric 函数在给定 metric 时会引发 ValueError 异常
        with self.assertRaises(ValueError):
            emit_metric("metric_name", metric)

    @mock.patch("boto3.Session.resource")
    def test_no_metrics_emitted_if_required_env_var_not_set(
        self, mock_resource: Any
    ) -> None:
        # 定义一个名为 metric 的字典，包含一个键值对 "some_number": 123
        metric = {"some_number": 123}

        # 设置 CI 和 BUILD_ENVIRONMENT 环境变量
        mock.patch.dict(
            "os.environ",
            {
                "CI": "true",
                "BUILD_ENVIRONMENT": BUILD_ENV,
            },
            clear=True,
        ).start()

        # 初始化 put_item_invoked 为 False，用于跟踪是否调用了 put_item 方法
        put_item_invoked = False

        # 定义 mock_put_item 函数，用于捕获传入的参数 Item，并将 put_item_invoked 设置为 True
        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal put_item_invoked
            put_item_invoked = True

        # 将 mock_put_item 函数绑定到 mock_resource 返回的 Table 的 put_item 方法上
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 调用 emit_metric 函数来发出度量信息 "metric_name", metric
        emit_metric("metric_name", metric)

        # 断言 put_item_invoked 应为 False，即不应调用 put_item 方法
        self.assertFalse(put_item_invoked)

    @mock.patch("boto3.Session.resource")
    def test_no_metrics_emitted_if_required_env_var_set_to_empty_string(
        self, mock_resource: Any
    ) -> None:
        # 定义一个名为 metric 的字典，包含一个键值对 "some_number": 123
        metric = {"some_number": 123}

        # 设置 GITHUB_JOB 环境变量为空字符串
        mock.patch.dict(
            "os.environ",
            {
                "GITHUB_JOB": "",
            },
        ).start()

        # 初始化 put_item_invoked 为 False，用于跟踪是否调用了 put_item 方法
        put_item_invoked = False

        # 定义 mock_put_item 函数，用于捕获传入的参数 Item，并将 put_item_invoked 设置为 True
        def mock_put_item(Item: dict[str, Any]) -> None:
            nonlocal put_item_invoked
            put_item_invoked = True

        # 将 mock_put_item 函数绑定到 mock_resource 返回的 Table 的 put_item 方法上
        mock_resource.return_value.Table.return_value.put_item = mock_put_item

        # 调用 emit_metric 函数来发出度量信息 "metric_name", metric
        emit_metric("metric_name", metric)

        # 断言 put_item_invoked 应为 False，即不应调用 put_item 方法
        self.assertFalse(put_item_invoked)
    # 定义一个测试方法，用于测试在不同批处理大小下上传到Rockset的行为
    def test_upload_to_rockset_batch_size(self) -> None:
        # 定义不同的测试用例列表，每个用例包括批处理大小和预期请求次数
        cases = [
            {
                "batch_size": BATCH_SIZE - 1,  # 批处理大小减一，预期只有一次请求
                "expected_number_of_requests": 1,
            },
            {
                "batch_size": BATCH_SIZE,      # 标准批处理大小，预期只有一次请求
                "expected_number_of_requests": 1,
            },
            {
                "batch_size": BATCH_SIZE + 1,  # 批处理大小加一，预期有两次请求
                "expected_number_of_requests": 2,
            },
        ]

        # 遍历每个测试用例
        for case in cases:
            # 创建一个模拟的客户端对象
            mock_client = mock.Mock()
            mock_client.Documents.add_documents.return_value = "OK"

            # 获取当前测试用例的批处理大小和预期请求次数
            batch_size = case["batch_size"]
            expected_number_of_requests = case["expected_number_of_requests"]

            # 创建一个包含指定大小范围的文档列表
            docs = list(range(batch_size))
            # 调用上传到Rockset的函数，传入集合名称、文档列表、工作空间和模拟客户端
            upload_to_rockset(
                collection="test", docs=docs, workspace="commons", client=mock_client
            )
            # 断言模拟客户端的文档添加方法被调用的次数符合预期请求次数
            self.assertEqual(
                mock_client.Documents.add_documents.call_count,
                expected_number_of_requests,
            )
# 如果这个脚本被直接执行（而不是被导入作为模块），则执行单元测试框架的主程序入口
if __name__ == "__main__":
    # 运行 Python 单元测试框架的主程序入口，这通常会执行脚本中定义的测试用例
    unittest.main()
```