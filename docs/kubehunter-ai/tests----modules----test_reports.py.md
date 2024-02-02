# `kubehunter\tests\modules\test_reports.py`

```py
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

# 定义测试报告生成器的函数
def test_reporters():
    # 定义测试用例列表，每个元素包含报告类型和预期的报告生成器类
    test_cases = [
        ("plain", PlainReporter),
        ("json", JSONReporter),
        ("yaml", YAMLReporter),
        ("notexists", PlainReporter),
    ]

    # 遍历测试用例列表
    for report_type, expected in test_cases:
        # 调用 get_reporter 函数获取实际的报告生成器类
        actual = get_reporter(report_type)
        # 断言实际的报告生成器类与预期的报告生成器类相同
        assert type(actual) is expected

# 定义测试分发器的函数
def test_dispatchers():
    # 定义测试用例列表，每个元素包含分发器类型和预期的分发器类
    test_cases = [
        ("stdout", STDOUTDispatcher),
        ("http", HTTPDispatcher),
        ("notexists", STDOUTDispatcher),
    ]

    # 遍历测试用例列表
    for dispatcher_type, expected in test_cases:
        # 调用 get_dispatcher 函数获取实际的分发器类
        actual = get_dispatcher(dispatcher_type)
        # 断言实际的分发器类与预期的分发器类相同
        assert type(actual) is expected
```