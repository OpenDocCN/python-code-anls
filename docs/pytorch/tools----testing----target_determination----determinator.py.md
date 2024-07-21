# `.\pytorch\tools\testing\target_determination\determinator.py`

```
# 导入未来的注解功能，以支持类型提示中的字符串列表和任意类型
from __future__ import annotations

# 导入系统相关模块
import sys
# 导入任意类型
from typing import Any

# 从工具包中导入测试目标确定模块的启发式算法相关内容
from tools.testing.target_determination.heuristics import (
    AggregatedHeuristics as AggregatedHeuristics,
    HEURISTICS,
    TestPrioritizations as TestPrioritizations,
)


# 定义函数，用于获取测试优先级信息
def get_test_prioritizations(
    tests: list[str], file: Any = sys.stdout
) -> AggregatedHeuristics:
    # 创建聚合启发式算法对象，传入测试列表
    aggregated_results = AggregatedHeuristics(tests)
    # 打印接收到的测试数量信息
    print(f"Received {len(tests)} tests to prioritize", file=file)
    # 遍历打印每个测试名称
    for test in tests:
        print(f"  {test}", file=file)

    # 遍历已定义的启发式算法列表
    for heuristic in HEURISTICS:
        # 获取当前启发式算法对测试的预测置信度
        new_rankings: TestPrioritizations = heuristic.get_prediction_confidence(tests)
        # 将当前启发式算法的结果添加到聚合结果中
        aggregated_results.add_heuristic_results(heuristic, new_rankings)

        # 打印当前启发式算法的结果信息
        print(f"Results from {heuristic.__class__.__name__}")
        print(new_rankings.get_info_str(verbose=False), file=file)

    # 返回聚合的测试优先级结果对象
    return aggregated_results
```