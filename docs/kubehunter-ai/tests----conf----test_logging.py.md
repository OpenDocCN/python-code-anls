# `kubehunter\tests\conf\test_logging.py`

```
# 导入 logging 模块
import logging
# 从 kube_hunter.conf.logging 模块中导入 setup_logger 函数
from kube_hunter.conf.logging import setup_logger

# 定义测试函数 test_setup_logger_level
def test_setup_logger_level():
    # 定义测试用例列表
    test_cases = [
        ("INFO", logging.INFO),  # 测试 INFO 级别日志
        ("Debug", logging.DEBUG),  # 测试 DEBUG 级别日志
        ("critical", logging.CRITICAL),  # 测试 CRITICAL 级别日志
        ("NOTEXISTS", logging.INFO),  # 测试不存在的日志级别，默认为 INFO
        ("BASIC_FORMAT", logging.INFO),  # 测试 BASIC_FORMAT 日志级别，默认为 INFO
    ]
    # 遍历测试用例
    for level, expected in test_cases:
        # 调用 setup_logger 函数设置日志级别
        setup_logger(level)
        # 获取实际的日志级别
        actual = logging.getLogger().getEffectiveLevel()
        # 断言实际日志级别与期望日志级别相等
        assert actual == expected, f"{level} level should be {expected} (got {actual})"

# 定义测试函数 test_setup_logger_none
def test_setup_logger_none():
# 设置日志记录器的级别为"NONE"
setup_logger("NONE")
# 断言日志记录器的管理器的禁用级别为CRITICAL
assert logging.getLogger().manager.disable == logging.CRITICAL
```