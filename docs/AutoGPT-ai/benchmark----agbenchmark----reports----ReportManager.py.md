# `.\AutoGPT\benchmark\agbenchmark\reports\ReportManager.py`

```py
# 导入所需的模块
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 从自定义的模块中导入相关类和函数
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.graphs import save_single_radar_chart
from agbenchmark.reports.processing.process_report import (
    get_highest_achieved_difficulty_per_category,
)
from agbenchmark.reports.processing.report_types import MetricsOverall, Report, Test
from agbenchmark.utils.utils import get_highest_success_difficulty

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义一个单例模式的报告管理器类
class SingletonReportManager:
    instance = None

    INFO_MANAGER: "SessionReportManager"
    REGRESSION_MANAGER: "RegressionTestsTracker"
    SUCCESS_RATE_TRACKER: "SuccessRatesTracker"

    # 实现单例模式
    def __new__(cls):
        if not cls.instance:
            cls.instance = super(SingletonReportManager, cls).__new__(cls)

            # 加载配置文件
            agent_benchmark_config = AgentBenchmarkConfig.load()
            # 获取当前时间
            benchmark_start_time_dt = datetime.now(
                timezone.utc
            )  # or any logic to fetch the datetime

            # 创建 SessionReportManager 实例
            cls.INFO_MANAGER = SessionReportManager(
                agent_benchmark_config.get_report_dir(benchmark_start_time_dt)
                / "report.json",
                benchmark_start_time_dt,
            )
            # 创建 RegressionTestsTracker 实例
            cls.REGRESSION_MANAGER = RegressionTestsTracker(
                agent_benchmark_config.regression_tests_file
            )
            # 创建 SuccessRatesTracker 实例
            cls.SUCCESS_RATE_TRACKER = SuccessRatesTracker(
                agent_benchmark_config.success_rate_file
            )

        return cls.instance

    # 清除单例实例
    @classmethod
    def clear_instance(cls):
        cls.instance = None
        cls.INFO_MANAGER = None
        cls.REGRESSION_MANAGER = None
        cls.SUCCESS_RATE_TRACKER = None

# 定义一个抽象基类，用于处理回归测试文件
class BaseReportManager:
    """Abstracts interaction with the regression tests file"""

    tests: dict[str, Any]
    # 初始化方法，接受一个报告文件路径作为参数
    def __init__(self, report_file: Path):
        # 将报告文件路径保存在实例变量中
        self.report_file = report_file

        # 调用 load 方法加载报告文件内容
        self.load()

    # 加载报告文件内容的方法
    def load(self) -> None:
        # 如果报告文件不存在，则创建其父目录
        if not self.report_file.exists():
            self.report_file.parent.mkdir(exist_ok=True)

        try:
            # 尝试以只读方式打开报告文件
            with self.report_file.open("r") as f:
                # 读取文件内容并解析为 JSON 格式，保存在 self.tests 中
                data = json.load(f)
                self.tests = {k: data[k] for k in sorted(data)}
        except FileNotFoundError:
            # 如果文件不存在，则初始化 self.tests 为空字典
            self.tests = {}
        except json.decoder.JSONDecodeError as e:
            # 如果解析 JSON 出错，则记录警告信息，并初始化 self.tests 为空字典
            logger.warning(f"Could not parse {self.report_file}: {e}")
            self.tests = {}

    # 将测试结果保存到报告文件中的方法
    def save(self) -> None:
        # 以写入方式打开报告文件
        with self.report_file.open("w") as f:
            # 将 self.tests 中的内容以缩进格式写入文件
            json.dump(self.tests, f, indent=4)

    # 删除指定测试结果的方法
    def remove_test(self, test_name: str) -> None:
        # 如果测试结果存在于 self.tests 中，则删除该测试结果并保存
        if test_name in self.tests:
            del self.tests[test_name]
            self.save()

    # 重置测试结果的方法
    def reset(self) -> None:
        # 将 self.tests 重置为空字典，并保存
        self.tests = {}
        self.save()
class SessionReportManager(BaseReportManager):
    """Abstracts interaction with the regression tests file"""

    # 定义一个属性 tests，可以是字典类型的字符串到 Test 对象或 Report 对象的映射
    tests: dict[str, Test] | Report

    # 初始化方法，接受报告文件路径和基准开始时间作为参数
    def __init__(self, report_file: Path, benchmark_start_time: datetime):
        # 调用父类的初始化方法
        super().__init__(report_file)

        # 记录当前时间
        self.start_time = time.time()
        # 记录基准开始时间
        self.benchmark_start_time = benchmark_start_time

    # 保存方法，将测试结果保存到报告文件中
    def save(self) -> None:
        # 打开报告文件进行写操作
        with self.report_file.open("w") as f:
            # 如果 tests 是 Report 对象，则将其转换为 JSON 格式写入文件
            if isinstance(self.tests, Report):
                f.write(self.tests.json(indent=4))
            # 否则，将 tests 中的数据转换为字典，再将字典转换为 JSON 格式写入文件
            else:
                json.dump({k: v.dict() for k, v in self.tests.items()}, f, indent=4)

    # 加载方法，从报告文件中加载测试结果
    def load(self) -> None:
        # 调用父类的加载方法
        super().load()
        # 如果 tests 中包含 "tests" 键
        if "tests" in self.tests:  # type: ignore
            # 将 "tests" 对应的值解析为 Report 对象
            self.tests = Report.parse_obj(self.tests)
        else:
            # 否则，将 tests 中的数据解析为字典，其中键为测试名称，值为 Test 对象
            self.tests = {n: Test.parse_obj(d) for n, d in self.tests.items()}

    # 添加测试报告方法，将测试名称和测试报告添加到 tests 中
    def add_test_report(self, test_name: str, test_report: Test) -> None:
        # 如果 tests 是 Report 对象，则抛出运行时错误
        if isinstance(self.tests, Report):
            raise RuntimeError("Session report already finalized")

        # 如果测试名称以 "Test" 开头，则去掉前缀
        if test_name.startswith("Test"):
            test_name = test_name[4:]
        # 将测试名称和测试报告添加到 tests 中
        self.tests[test_name] = test_report

        # 保存更新后的测试结果
        self.save()
    # 定义一个方法，用于最终化会话报告
    def finalize_session_report(self, config: AgentBenchmarkConfig) -> None:
        # 获取当前命令行参数并拼接成字符串
        command = " ".join(sys.argv)

        # 如果 self.tests 是 Report 类型，则抛出运行时错误
        if isinstance(self.tests, Report):
            raise RuntimeError("Session report already finalized")

        # 创建一个新的 Report 对象，包含命令、提交 SHA、完成时间、开始时间、总运行时间、最高难度、总成本等信息
        self.tests = Report(
            command=command.split(os.sep)[-1],
            benchmark_git_commit_sha="---",
            agent_git_commit_sha="---",
            completion_time=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            benchmark_start_time=self.benchmark_start_time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            metrics=MetricsOverall(
                run_time=str(round(time.time() - self.start_time, 2)) + " seconds",
                highest_difficulty=get_highest_success_difficulty(self.tests),
                total_cost=self.get_total_costs(),
            ),
            tests=copy.copy(self.tests),
            config=config.dict(exclude_none=True),
        )

        # 获取每个类别中最高难度的代理
        agent_categories = get_highest_achieved_difficulty_per_category(self.tests)
        # 如果代理类别数量大于1，则保存雷达图
        if len(agent_categories) > 1:
            save_single_radar_chart(
                agent_categories,
                config.get_report_dir(self.benchmark_start_time) / "radar_chart.png",
            )

        # 保存最终化的报告
        self.save()

    # 计算总成本
    def get_total_costs(self):
        # 如果 self.tests 是 Report 类型，则获取其 tests 属性，否则直接使用 self.tests
        if isinstance(self.tests, Report):
            tests = self.tests.tests
        else:
            tests = self.tests

        # 初始化总成本和是否所有成本都为 None 的标志
        total_cost = 0
        all_costs_none = True
        # 遍历所有测试数据，计算总成本
        for test_data in tests.values():
            cost = sum(r.cost or 0 for r in test_data.results)

            # 如果成本不为 None，则更新标志并累加到总成本中
            if cost is not None:  # check if cost is not None
                all_costs_none = False
                total_cost += cost  # add cost to total
        # 如果所有成本都为 None，则将总成本设为 None
        if all_costs_none:
            total_cost = None
        return total_cost
class RegressionTestsTracker(BaseReportManager):
    """Abstracts interaction with the regression tests file"""

    # 定义一个类 RegressionTestsTracker，继承自 BaseReportManager，用于处理回归测试文件的交互

    tests: dict[str, dict]
    # 定义一个属性 tests，类型为字典，键为字符串，值为字典

    def add_test(self, test_name: str, test_details: dict) -> None:
        # 定义一个方法 add_test，接受测试名称和测试详情作为参数

        if test_name.startswith("Test"):
            # 如果测试名称以 "Test" 开头

            test_name = test_name[4:]
            # 则将测试名称的前四个字符去掉

        self.tests[test_name] = test_details
        # 将测试名称和测试详情添加到 tests 字典中

        self.save()
        # 调用 save 方法保存数据

    def has_regression_test(self, test_name: str) -> bool:
        # 定义一个方法 has_regression_test，接受测试名称作为参数，返回布尔值

        return self.tests.get(test_name) is not None
        # 返回 tests 字典中是否存在指定测试名称的数据


class SuccessRatesTracker(BaseReportManager):
    """Abstracts interaction with the regression tests file"""

    # 定义一个类 SuccessRatesTracker，继承自 BaseReportManager，用于处理回归测试文件的交互

    tests: dict[str, list[bool | None]]
    # 定义一个属性 tests，类型为字典，键为字符串，值为布尔值或空值的列表

    def update(self, test_name: str, success_history: list[bool | None]) -> None:
        # 定义一个方法 update，接受测试名称和成功历史记录作为参数

        if test_name.startswith("Test"):
            # 如果测试名称以 "Test" 开头

            test_name = test_name[4:]
            # 则将测试名称的前四个字符去掉

        self.tests[test_name] = success_history
        # 将测试名称和成功历史记录添加到 tests 字典中

        self.save()
        # 调用 save 方法保存数据
```