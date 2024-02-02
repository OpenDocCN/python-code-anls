# `MetaGPT\tests\metagpt\actions\test_write_review.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/20 15:01
@Author  : alexanderwu
@File    : test_write_review.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.actions.write_review 模块中导入 WriteReview 类
from metagpt.actions.write_review import WriteReview

# 定义测试上下文
CONTEXT = """
{
    "Language": "zh_cn",
    "Programming Language": "Python",
    "Original Requirements": "写一个简单的2048",
    "Project Name": "game_2048",
    "Product Goals": [
        "创建一个引人入胜的用户体验",
        "确保高性能",
        "提供可定制的功能"
    ],
    "User Stories": [
        "作为用户，我希望能够选择不同的难度级别",
        "作为玩家，我希望在每局游戏结束后能看到我的得分"
    ],
    "Competitive Analysis": [
        "Python Snake Game: 界面简单，缺乏高级功能"
    ],
    "Competitive Quadrant Chart": "quadrantChart\n    title \"Reach and engagement of campaigns\"\n    x-axis \"Low Reach\" --> \"High Reach\"\n    y-axis \"Low Engagement\" --> \"High Engagement\"\n    quadrant-1 \"我们应该扩展\"\n    quadrant-2 \"需要推广\"\n    quadrant-3 \"重新评估\"\n    quadrant-4 \"可能需要改进\"\n    \"Campaign A\": [0.3, 0.6]\n    \"Campaign B\": [0.45, 0.23]\n    \"Campaign C\": [0.57, 0.69]\n    \"Campaign D\": [0.78, 0.34]\n    \"Campaign E\": [0.40, 0.34]\n    \"Campaign F\": [0.35, 0.78]\n    \"Our Target Product\": [0.5, 0.6]",
    "Requirement Analysis": "产品应该用户友好。",
    "Requirement Pool": [
        [
            "P0",
            "主要代码..."
        ],
        [
            "P0",
            "游戏算法..."
        ]
    ],
    "UI Design draft": "基本功能描述，简单的风格和布局。",
    "Anything UNCLEAR": "..."
}
"""

# 异步测试函数
@pytest.mark.asyncio
async def test_write_review():
    # 创建 WriteReview 实例
    write_review = WriteReview()
    # 运行 WriteReview 实例的 run 方法，传入测试上下文，获取 review 对象
    review = await write_review.run(CONTEXT)
    # 断言 review 对象的 instruct_content 属性存在
    assert review.instruct_content
    # 断言 review 对象的 "LGTM" 属性值在指定列表中
    assert review.get("LGTM") in ["LGTM", "LBTM"]

```