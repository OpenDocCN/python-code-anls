# `MetaGPT\tests\metagpt\tools\test_summarize.py`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 17:46
@Author  : alexanderwu
@File    : test_summarize.py
"""

# 导入 pytest 模块
import pytest

# 定义测试用例列表
CASES = [
    """# 上下文
# 用户搜索请求
屈臣氏有什么产品可以去痘？

# 要求
你是专业管家团队的一员，会给出有帮助的建议
1. 请根据上下文，对用户搜索请求进行总结性回答，不要包括与请求无关的文本
2. 以 [正文](引用链接) markdown形式在正文中**自然标注**~5个文本（如商品词或类似文本段），以便跳转
3. 回复优雅、清晰，**绝不重复文本**，行文流畅，长度居中""",
    """# 上下文
# 用户搜索请求
厦门有什么好吃的？

# 要求
你是专业管家团队的一员，会给出有帮助的建议
1. 请根据上下文，对用户搜索请求进行总结性回答，不要包括与请求无关的文本
2. 以 [正文](引用链接) markdown形式在正文中**自然标注**3-5个文本（如商品词或类似文本段），以便跳转
3. 回复优雅、清晰，**绝不重复文本**，行文流畅，长度居中""",
]

# 使用 pytest 的 usefixtures 装饰器，指定 llm_api 作为测试用例的前置条件
@pytest.mark.usefixtures("llm_api")
# 定义测试函数 test_summarize
def test_summarize(llm_api):
    # 测试函数暂时不包含任何代码，pass 保持函数结构完整
    pass
```