# `.\yolov8\tests\conftest.py`

```py
# 导入 shutil 和 Path 类
import shutil
from pathlib import Path

# 导入 tests 模块中的 TMP 目录
from tests import TMP


def pytest_addoption(parser):
    """
    向 pytest 添加自定义命令行选项。

    Args:
        parser (pytest.config.Parser): pytest 解析器对象，用于添加自定义命令行选项。

    Returns:
        (None)
    """
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests")


def pytest_collection_modifyitems(config, items):
    """
    修改测试项列表，如果未指定 --slow 选项，则排除标记为 slow 的测试。

    Args:
        config (pytest.config.Config): pytest 配置对象，提供访问命令行选项的功能。
        items (list): 已收集的 pytest 项目对象列表，根据 --slow 选项的存在进行修改。

    Returns:
        (None) 函数会直接在原地修改 'items' 列表，不返回任何值。
    """
    if not config.getoption("--slow"):
        # 如果未指定 --slow 选项，则从测试项列表中完全移除标记为 'slow' 的项目
        items[:] = [item for item in items if "slow" not in item.keywords]


def pytest_sessionstart(session):
    """
    初始化 pytest 的会话配置。

    此函数由 pytest 在创建 'Session' 对象后但在执行测试收集之前自动调用。它设置初始种子并准备测试会话的临时目录。

    Args:
        session (pytest.Session): pytest 会话对象。

    Returns:
        (None)
    """
    from ultralytics.utils.torch_utils import init_seeds

    # 初始化随机种子
    init_seeds()
    # 删除现有的 tests/tmp 目录（如果存在）
    shutil.rmtree(TMP, ignore_errors=True)
    # 创建一个新的空目录
    TMP.mkdir(parents=True, exist_ok=True)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    在 pytest 会话结束后进行清理操作。

    此函数由 pytest 在整个测试会话结束时自动调用。它删除测试过程中使用的特定文件和目录。

    Args:
        terminalreporter (pytest.terminal.TerminalReporter): 终端报告对象，用于终端输出。
        exitstatus (int): 测试运行的退出状态。
        config (pytest.config.Config): pytest 配置对象。

    Returns:
        (None)
    """
    from ultralytics.utils import WEIGHTS_DIR

    # 删除文件
    models = [path for x in ["*.onnx", "*.torchscript"] for path in WEIGHTS_DIR.rglob(x)]
    for file in ["bus.jpg", "yolov8n.onnx", "yolov8n.torchscript"] + models:
        Path(file).unlink(missing_ok=True)

    # 删除目录
    models = [path for x in ["*.mlpackage", "*_openvino_model"] for path in WEIGHTS_DIR.rglob(x)]
    for directory in [TMP.parents[1] / ".pytest_cache", TMP] + models:
        shutil.rmtree(directory, ignore_errors=True)
```