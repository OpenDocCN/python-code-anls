# `kubehunter\tests\conf\test_logging.py`

```
# 导入 logging 模块
import logging
# 从 kube_hunter.conf.logging 模块中导入 setup_logger 函数
from kube_hunter.conf.logging import setup_logger

# 测试设置日志级别的函数
def test_setup_logger_level():
    # 定义测试用例，包括日志级别和预期结果
    test_cases = [
        ("INFO", logging.INFO),
        ("Debug", logging.DEBUG),
        ("critical", logging.CRITICAL),
        ("NOTEXISTS", logging.INFO),
        ("BASIC_FORMAT", logging.INFO),
    ]
    # 遍历测试用例
    for level, expected in test_cases:
        # 设置日志级别
        setup_logger(level)
        # 获取实际的日志级别
        actual = logging.getLogger().getEffectiveLevel()
        # 断言实际结果与预期结果相等
        assert actual == expected, f"{level} level should be {expected} (got {actual})"

# 测试设置日志级别为 NONE 的函数
def test_setup_logger_none():
    # 设置日志级别为 NONE
    setup_logger("NONE")
    # 断言日志管理器的禁用级别为 CRITICAL
    assert logging.getLogger().manager.disable == logging.CRITICAL
```