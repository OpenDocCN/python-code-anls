# `kubehunter\tests\modules\test_reports.py`

```
# 从 kube_hunter.modules.report 模块中导入 get_reporter 和 get_dispatcher 函数
from kube_hunter.modules.report import get_reporter, get_dispatcher
# 从 kube_hunter.modules.report.factory 模块中导入 YAMLReporter, JSONReporter, PlainReporter, HTTPDispatcher, STDOUTDispatcher 类
from kube_hunter.modules.report.factory import (
    YAMLReporter,
    JSONReporter,
    PlainReporter,
    HTTPDispatcher,
    STDOUTDispatcher,
)

# 定义一个名为 test_reporters 的函数
def test_reporters():
    # 定义测试用例列表，每个测试用例包括报告类型和预期的报告类
    test_cases = [
        ("plain", PlainReporter),
        ("json", JSONReporter),
        ("yaml", YAMLReporter),
        ("notexists", PlainReporter),
    ]

    # 遍历测试用例列表
    for report_type, expected in test_cases:
        # 调用 get_reporter 函数，根据报告类型获取实际的报告类
        actual = get_reporter(report_type)
# 断言实际值的类型与期望值相同
assert type(actual) is expected

# 测试调度器函数
def test_dispatchers():
    # 定义测试用例，包括调度器类型和期望的调度器类
    test_cases = [
        ("stdout", STDOUTDispatcher),  # 标准输出调度器
        ("http", HTTPDispatcher),  # HTTP调度器
        ("notexists", STDOUTDispatcher),  # 不存在的调度器，默认为标准输出调度器
    ]

    # 遍历测试用例
    for dispatcher_type, expected in test_cases:
        # 获取实际的调度器类
        actual = get_dispatcher(dispatcher_type)
        # 断言实际值的类型与期望值相同
        assert type(actual) is expected
```