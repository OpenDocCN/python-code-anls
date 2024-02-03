# `.\AutoGPT\autogpts\autogpt\tests\unit\test_spinner.py`

```py
# 导入时间模块
import time

# 从 autogpt.app.spinner 模块中导入 Spinner 类
from autogpt.app.spinner import Spinner

# 定义几个常量
ALMOST_DONE_MESSAGE = "Almost done..."
PLEASE_WAIT = "Please wait..."


# 测试 Spinner 类以默认值初始化
def test_spinner_initializes_with_default_values():
    """Tests that the spinner initializes with default values."""
    # 使用 Spinner 类创建实例 spinner
    with Spinner() as spinner:
        # 断言消息为默认值 "Loading..."
        assert spinner.message == "Loading..."
        # 断言延迟时间为默认值 0.1
        assert spinner.delay == 0.1


# 测试 Spinner 类以自定义值初始化
def test_spinner_initializes_with_custom_values():
    """Tests that the spinner initializes with custom message and delay values."""
    # 使用自定义消息和延迟时间创建 Spinner 实例 spinner
    with Spinner(message=PLEASE_WAIT, delay=0.2) as spinner:
        # 断言消息为自定义消息 PLEASE_WAIT
        assert spinner.message == PLEASE_WAIT
        # 断言延迟时间为自定义值 0.2
        assert spinner.delay == 0.2


# 测试 Spinner 类开始旋转并停止旋转
def test_spinner_stops_spinning():
    """Tests that the spinner starts spinning and stops spinning without errors."""
    # 创建 Spinner 实例 spinner
    with Spinner() as spinner:
        # 等待1秒
        time.sleep(1)
    # 断言 spinner 不在运行
    assert not spinner.running


# 测试 Spinner 类作为上下文管理器使用
def test_spinner_can_be_used_as_context_manager():
    """Tests that the spinner can be used as a context manager."""
    # 创建 Spinner 实例 spinner
    with Spinner() as spinner:
        # 断言 spinner 在运行
        assert spinner.running
    # 断言 spinner 不在运行
    assert not spinner.running
```