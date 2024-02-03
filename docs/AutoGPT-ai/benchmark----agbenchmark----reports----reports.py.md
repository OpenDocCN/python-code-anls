# `.\AutoGPT\benchmark\agbenchmark\reports\reports.py`

```py
# 导入所需的模块
import json
import logging
import os
from pathlib import Path
import pytest

# 导入自定义模块
from agbenchmark.challenges import ChallengeInfo
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types import Test, TestMetrics, TestResult
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.data_types import DifficultyLevel

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 获取并更新测试成功历史记录
def get_and_update_success_history(
    test_name: str, success: bool | None
) -> list[bool | None]:
    # 检查是否设置了环境变量 IS_MOCK
    mock = os.getenv("IS_MOCK")

    # 获取先前测试结果
    prev_test_results = SingletonReportManager().SUCCESS_RATE_TRACKER.tests.get(
        test_name, []
    )

    if not mock:
        # 只有在实际测试时才添加结果
        prev_test_results.append(success)
        SingletonReportManager().SUCCESS_RATE_TRACKER.update(
            test_name, prev_test_results
        )

    return prev_test_results

# 更新回归测试
def update_regression_tests(
    prev_test_results: list[bool | None],
    test_report: Test,
    test_name: str,
) -> None:
    if len(prev_test_results) >= 3 and prev_test_results[-3:] == [True, True, True]:
        # 如果最近的三次测试都成功，则将其添加到回归测试中
        test_report.metrics.is_regression = True
        SingletonReportManager().REGRESSION_MANAGER.add_test(
            test_name, test_report.dict(include={"difficulty", "data_path"})
        )

# 创建空的测试报告
def make_empty_test_report(
    challenge_info: ChallengeInfo,
) -> Test:
    difficulty = challenge_info.difficulty
    if isinstance(difficulty, DifficultyLevel):
        difficulty = difficulty.value
    # 返回一个Test对象，包含以下属性：
    # - category: 挑战信息中的类别列表
    # - difficulty: 挑战信息中的难度
    # - data_path: 挑战信息中的数据路径
    # - description: 挑战信息中的描述，如果为空则使用空字符串
    # - task: 挑战信息中的任务
    # - answer: 挑战信息中的参考答案，如果为空则使用空字符串
    # - metrics: 包含尝试状态和是否为回归测试的TestMetrics对象
    # - results: 空列表
    return Test(
        category=[c.value for c in challenge_info.category],
        difficulty=difficulty,
        data_path=challenge_info.source_uri,
        description=challenge_info.description or "",
        task=challenge_info.task,
        answer=challenge_info.reference_answer or "",
        metrics=TestMetrics(attempted=False, is_regression=False),
        results=[],
    )
# 将测试结果添加到测试报告中
def add_test_result_to_report(
    test_report: Test,  # 测试报告对象
    item: pytest.Item,  # pytest 测试项
    call: pytest.CallInfo,  # pytest 调用信息
    config: AgentBenchmarkConfig,  # 代理基准配置
) -> None:
    # 将测试项的用户属性转换为字典
    user_properties: dict = dict(item.user_properties)
    # 获取测试名称
    test_name: str = user_properties.get("test_name", "")

    # 检查是否设置了环境变量 IS_MOCK
    mock = os.getenv("IS_MOCK")

    # 如果有异常信息
    if call.excinfo:
        # 如果不是模拟模式
        if not mock:
            # 从回归管理器中移除测试
            SingletonReportManager().REGRESSION_MANAGER.remove_test(test_name)

        # 设置测试报告的尝试状态
        test_report.metrics.attempted = call.excinfo.typename != "Skipped"
    else:
        # 设置测试报告的尝试状态为 True
        test_report.metrics.attempted = True

    # 将测试结果添加到测试报告中
    test_report.results.append(
        TestResult(
            success=call.excinfo is None,
            run_time=f"{str(round(call.duration, 3))} seconds",
            fail_reason=str(call.excinfo.value) if call.excinfo else None,
            reached_cutoff=user_properties.get("timed_out", False),
        )
    )
    # 计算测试报告的成功百分比
    test_report.metrics.success_percentage = (
        sum(r.success or False for r in test_report.results)
        / len(test_report.results)
        * 100
    )

    # 获取并更新测试历史结果
    prev_test_results: list[bool | None] = get_and_update_success_history(
        test_name, test_report.results[-1].success
    )

    # 更新回归测试
    update_regression_tests(prev_test_results, test_report, test_name)

    # 如果存在测试报告和测试名称
    if test_report and test_name:
        # 如果不是模拟模式
        if not mock:
            # 更新已击败的挑战
            update_challenges_already_beaten(
                config.challenges_already_beaten_file, test_report, test_name
            )

        # 将测试报告添加到信息管理器中
        SingletonReportManager().INFO_MANAGER.add_test_report(test_name, test_report)


# 更新已击败的挑战
def update_challenges_already_beaten(
    challenges_already_beaten_file: Path, test_report: Test, test_name: str
) -> None:
    # 检查测试报告中是否有任何一个测试成功
    current_run_successful = any(r.success for r in test_report.results)
    
    # 尝试打开已经通过的挑战文件，如果文件不存在则创建一个空字典
    try:
        with open(challenges_already_beaten_file, "r") as f:
            challenges_beaten_before = json.load(f)
    except FileNotFoundError:
        challenges_beaten_before = {}
    
    # 获取测试名称对应的挑战是否曾经被通过过
    has_ever_been_beaten = challenges_beaten_before.get(test_name)
    
    # 更新挑战是否曾经被通过过的信息
    challenges_beaten_before[test_name] = has_ever_been_beaten or current_run_successful
    
    # 将更新后的挑战信息写入已通过的挑战文件
    with open(challenges_already_beaten_file, "w") as f:
        json.dump(challenges_beaten_before, f, indent=4)
# 结束会话的函数，接受一个AgentBenchmarkConfig类型的参数，不返回任何结果
def session_finish(agbenchmark_config: AgentBenchmarkConfig) -> None:
    # 获取单例的报告管理器，调用INFO_MANAGER对象的finalize_session_report方法，传入agbenchmark_config参数
    SingletonReportManager().INFO_MANAGER.finalize_session_report(agbenchmark_config)
    # 获取单例的报告管理器，调用REGRESSION_MANAGER对象的save方法
    SingletonReportManager().REGRESSION_MANAGER.save()
    # 获取单例的报告管理器，调用SUCCESS_RATE_TRACKER对象的save方法
    SingletonReportManager().SUCCESS_RATE_TRACKER.save()
```