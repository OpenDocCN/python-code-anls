# `MetaGPT\tests\metagpt\utils\test_cost_manager.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/27
@Author  : mashenquan
@File    : test_cost_manager.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.utils.cost_manager 模块中导入 CostManager 类
from metagpt.utils.cost_manager import CostManager

# 定义测试函数 test_cost_manager
def test_cost_manager():
    # 创建 CostManager 对象，设置总预算为 20
    cm = CostManager(total_budget=20)
    # 更新成本信息，包括提示标记数、完成标记数和模型名称
    cm.update_cost(prompt_tokens=1000, completion_tokens=100, model="gpt-4-1106-preview")
    # 断言总提示标记数为 1000
    assert cm.get_total_prompt_tokens() == 1000
    # 断言总完成标记数为 100
    assert cm.get_total_completion_tokens() == 100
    # 断言总成本为 0.013
    assert cm.get_total_cost() == 0.013
    # 再次更新成本信息
    cm.update_cost(prompt_tokens=100, completion_tokens=10, model="gpt-4-1106-preview")
    # 断言总提示标记数为 1100
    assert cm.get_total_prompt_tokens() == 1100
    # 断言总完成标记数为 110
    assert cm.get_total_completion_tokens() == 110
    # 断言总成本为 0.0143
    assert cm.get_total_cost() == 0.0143
    # 获取成本信息对象
    cost = cm.get_costs()
    # 断言成本信息对象存在
    assert cost
    # 断言成本信息对象的总成本与 CostManager 对象的总成本相等
    assert cost.total_cost == cm.get_total_cost()
    # 断言成本信息对象的总提示标记数与 CostManager 对象的总提示标记数相等
    assert cost.total_prompt_tokens == cm.get_total_prompt_tokens()
    # 断言成本信息对象的总完成标记数与 CostManager 对象的总完成标记数相等
    assert cost.total_completion_tokens == cm.get_total_completion_tokens()
    # 断言成本信息对象的总预算与 CostManager 对象的总预算相等
    assert cost.total_budget == 20

# 如果当前文件被直接运行，则执行测试
if __name__ == "__main__":
    # 使用 pytest 执行当前文件，并打印输出
    pytest.main([__file__, "-s"])

```