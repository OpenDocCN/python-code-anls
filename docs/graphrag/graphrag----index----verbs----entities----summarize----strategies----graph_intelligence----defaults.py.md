# `.\graphrag\graphrag\index\verbs\entities\summarize\strategies\graph_intelligence\defaults.py`

```py
# 引入所需的模块和类
"""A file containing some default responses."""
# 定义一个包含默认响应的模拟LLM响应列表
MOCK_LLM_RESPONSES = [
    """
    This is a MOCK response for the LLM. It is summarized!
    """.strip()
]
# 定义默认的LLM配置字典，指定LLM类型为静态响应，包含上面定义的模拟LLM响应列表
DEFAULT_LLM_CONFIG = {
    "type": LLMType.StaticResponse,
    "responses": MOCK_LLM_RESPONSES,
}
```