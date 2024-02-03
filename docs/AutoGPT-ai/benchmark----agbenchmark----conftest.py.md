# `.\AutoGPT\benchmark\agbenchmark\conftest.py`

```py
# 导入所需的模块
import contextlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Generator

import pytest

# 导入自定义模块
from agbenchmark.challenges import OPTIONAL_CATEGORIES, BaseChallenge
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types import Test
from agbenchmark.reports.ReportManager import RegressionTestsTracker
from agbenchmark.reports.reports import (
    add_test_result_to_report,
    make_empty_test_report,
    session_finish,
)
from agbenchmark.utils.data_types import Category

# 设置全局超时时间
GLOBAL_TIMEOUT = (
    1500  # The tests will stop after 25 minutes so we can send the reports.
)

# 加载 AgentBenchmarkConfig 配置
agbenchmark_config = AgentBenchmarkConfig.load()
# 获取 logger
logger = logging.getLogger(__name__)

# 定义 pytest 插件和忽略的文件夹
pytest_plugins = ["agbenchmark.utils.dependencies"]
collect_ignore = ["challenges"]

# 定义 fixture，返回 AgentBenchmarkConfig 对象
@pytest.fixture(scope="module")
def config() -> AgentBenchmarkConfig:
    return agbenchmark_config

# 定义 fixture，用于设置和清理临时文件夹
@pytest.fixture(autouse=True)
def temp_folder() -> Generator[Path, None, None]:
    """
    Pytest fixture that sets up and tears down the temporary folder for each test.
    It is automatically used in every test due to the 'autouse=True' parameter.
    """

    # 如果输出目录不存在，则创建
    if not os.path.exists(agbenchmark_config.temp_folder):
        os.makedirs(agbenchmark_config.temp_folder, exist_ok=True)

    # 生成临时文件夹路径
    yield agbenchmark_config.temp_folder
    # 测试函数完成后进行清理
    # 如果环境变量中没有设置 KEEP_TEMP_FOLDER_FILES，则执行以下操作
    if not os.getenv("KEEP_TEMP_FOLDER_FILES"):
        # 遍历临时文件夹中的所有文件
        for filename in os.listdir(agbenchmark_config.temp_folder):
            # 获取文件的完整路径
            file_path = os.path.join(agbenchmark_config.temp_folder, filename)
            try:
                # 如果是文件或符号链接，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 如果是文件夹，则递归删除
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                # 如果删除失败，则记录警告信息
                logger.warning(f"Failed to delete {file_path}. Reason: {e}")
# 定义 pytest_addoption 函数，用于向 pytest 命令添加命令行选项
def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Pytest hook that adds command-line options to the `pytest` command.
    The added options are specific to agbenchmark and control its behavior:
    * `--mock` is used to run the tests in mock mode.
    * `--host` is used to specify the host for the tests.
    * `--category` is used to run only tests of a specific category.
    * `--nc` is used to run the tests without caching.
    * `--cutoff` is used to specify a cutoff time for the tests.
    * `--improve` is used to run only the tests that are marked for improvement.
    * `--maintain` is used to run only the tests that are marked for maintenance.
    * `--explore` is used to run the tests in exploration mode.
    * `--test` is used to run a specific test.
    * `--no-dep` is used to run the tests without dependencies.
    * `--keep-answers` is used to keep the answers of the tests.

    Args:
        parser: The Pytest CLI parser to which the command-line options are added.
    """
    # 添加命令行选项
    parser.addoption("-N", "--attempts", action="store")
    parser.addoption("--no-dep", action="store_true")
    parser.addoption("--mock", action="store_true")
    parser.addoption("--host", default=None)
    parser.addoption("--nc", action="store_true")
    parser.addoption("--cutoff", action="store")
    parser.addoption("--category", action="append")
    parser.addoption("--test", action="append")
    parser.addoption("--improve", action="store_true")
    parser.addoption("--maintain", action="store_true")
    parser.addoption("--explore", action="store_true")
    parser.addoption("--keep-answers", action="store_true")


# 定义 pytest_configure 函数，用于配置 pytest
def pytest_configure(config: pytest.Config) -> None:
    # Register category markers to prevent "unknown marker" warnings
    # 注册类别标记以防止“未知标记”警告
    for category in Category:
        config.addinivalue_line("markers", f"{category.value}: {category}")


# 定义 check_regression fixture，用于检查回归
@pytest.fixture(autouse=True)
def check_regression(request: pytest.FixtureRequest) -> None:
    """
    # 检查每个测试是否应被视为回归测试，并根据此进行跳过
    # 从 `request` 对象中获取测试名称。回归报告从基准配置中指定的路径加载。

    # 效果：
    # * 如果使用了 `--improve` 选项并且当前测试被视为回归测试，则跳过该测试。
    # * 如果使用了 `--maintain` 选项并且当前测试不被视为回归测试，则也跳过该测试。

    # 参数：
    # request: 从中检索测试名称和基准配置的请求对象。
    """
    # 使用上下文管理器忽略文件未找到的错误
    with contextlib.suppress(FileNotFoundError):
        # 创建 RegressionTestsTracker 对象，从基准配置中获取回归测试文件路径
        rt_tracker = RegressionTestsTracker(agbenchmark_config.regression_tests_file)

        # 获取测试名称
        test_name = request.node.parent.name
        # 获取挑战位置
        challenge_location = getattr(request.node.parent.cls, "CHALLENGE_LOCATION", "")
        # 构建跳过信息字符串
        skip_string = f"Skipping {test_name} at {challenge_location}"

        # 检查测试名称是否存在于回归测试中
        is_regression_test = rt_tracker.has_regression_test(test_name)
        # 如果使用了 `--improve` 选项并且当前测试被视为回归测试，则跳过该测试
        if request.config.getoption("--improve") and is_regression_test:
            pytest.skip(f"{skip_string} because it's a regression test")
        # 如果使用了 `--maintain` 选项并且当前测试不被视为回归测试，则跳过该测试
        elif request.config.getoption("--maintain") and not is_regression_test:
            pytest.skip(f"{skip_string} because it's not a regression test")
# 定义一个 pytest fixture，用于自动使用并在 session 范围内运行
@pytest.fixture(autouse=True, scope="session")
def mock(request: pytest.FixtureRequest) -> bool:
    """
    Pytest fixture that retrieves the value of the `--mock` command-line option.
    The `--mock` option is used to run the tests in mock mode.

    Args:
        request: The `pytest.FixtureRequest` from which the `--mock` option value
            is retrieved.

    Returns:
        bool: Whether `--mock` is set for this session.
    """
    return request.config.getoption("--mock")


# 定义一个字典，用于存储测试报告
test_reports: dict[str, Test] = {}


# 定义 pytest hook，在生成测试报告时调用
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    """
    Pytest hook that is called when a test report is being generated.
    It is used to generate and finalize reports for each test.

    Args:
        item: The test item for which the report is being generated.
        call: The call object from which the test result is retrieved.
    """
    challenge: type[BaseChallenge] = item.cls  # type: ignore
    challenge_id = challenge.info.eval_id

    if challenge_id not in test_reports:
        test_reports[challenge_id] = make_empty_test_report(challenge.info)

    if call.when == "setup":
        test_name = item.nodeid.split("::")[1]
        item.user_properties.append(("test_name", test_name))

    if call.when == "call":
        add_test_result_to_report(
            test_reports[challenge_id], item, call, agbenchmark_config
        )


# 定义一个超时监控函数，限制测试套件的总执行时间
def timeout_monitor(start_time: int) -> None:
    """
    Function that limits the total execution time of the test suite.
    This function is supposed to be run in a separate thread and calls `pytest.exit`
    if the total execution time has exceeded the global timeout.

    Args:
        start_time (int): The start time of the test suite.
    """
    while time.time() - start_time < GLOBAL_TIMEOUT:
        time.sleep(1)  # check every second

    pytest.exit("Test suite exceeded the global timeout", returncode=1)
# 定义 pytest 会话开始时的钩子函数，设置并运行一个独立线程的超时监视器
def pytest_sessionstart(session: pytest.Session) -> None:
    # 记录当前时间
    start_time = time.time()
    # 创建一个线程，目标函数为 timeout_monitor，传入参数为开始时间
    t = threading.Thread(target=timeout_monitor, args=(start_time,))
    # 将线程设置为守护线程，守护线程在关闭时会被强制停止
    t.daemon = True
    # 启动线程
    t.start()

# 定义 pytest 会话结束时的钩子函数，完成并保存测试报告
def pytest_sessionfinish(session: pytest.Session) -> None:
    session_finish(agbenchmark_config)

# 定义 pytest 生成测试用例时的钩子函数
def pytest_generate_tests(metafunc: pytest.Metafunc):
    # 如果传入的参数 -N 是字符串类型
    if type(n := metafunc.config.getoption("-N")) is str:
        # 根据参数值生成测试用例
        metafunc.parametrize("i_attempt", range(int(n)))

# 定义 pytest 收集测试用例后的修改钩子函数
def pytest_collection_modifyitems(
    items: list[pytest.Item], config: pytest.Config
) -> None:
    # 创建 RegressionTestsTracker 对象
    rt_tracker = RegressionTestsTracker(agbenchmark_config.regression_tests_file)

    try:
        # 尝试加载已经通过的挑战数据
        challenges_beaten_in_the_past = json.loads(
            agbenchmark_config.challenges_already_beaten_file.read_bytes()
        )
    except FileNotFoundError:
        challenges_beaten_in_the_past = {}

    # 获取命令行参数 --test 和 --category 的值
    selected_tests: tuple[str] = config.getoption("--test")
    selected_categories: tuple[str] = config.getoption("--category")

    # 无法使用 for 循环直接删除元素，使用索引 i 进行删除
    i = 0
```