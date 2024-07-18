# `.\graphrag\graphrag\index\verbs\graph\report\strategies\graph_intelligence\defaults.py`

```py
# 版权声明及许可信息，版权归 Microsoft Corporation 所有，基于 MIT 许可协议
# 导入 JSON 模块，用于处理 JSON 数据
import json

# 默认的数据块大小常量，设定为 3000
DEFAULT_CHUNK_SIZE = 3000

# 模拟的响应数据列表，包含一个 JSON 对象的序列化字符串
MOCK_RESPONSES = [
    json.dumps({
        "title": "<report_title>",  # 报告标题，占位符
        "summary": "<executive_summary>",  # 执行摘要，占位符
        "rating": 2,  # 评分等级，此处为示例值
        "rating_explanation": "<rating_explanation>",  # 评分解释，占位符
        "findings": [
            {
                "summary": "<insight_1_summary>",  # 第一个发现的摘要，占位符
                "explanation": "<insight_1_explanation",  # 第一个发现的解释，占位符
            },
            {
                "summary": "<farts insight_2_summary>",  # 第二个发现的摘要，有误的占位符示例
                "explanation": "<insight_2_explanation",  # 第二个发现的解释，占位符
            },
        ],
    })
]
```