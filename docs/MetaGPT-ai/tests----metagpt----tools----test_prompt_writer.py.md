# `MetaGPT\tests\metagpt\tools\test_prompt_writer.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 17:46
@Author  : alexanderwu
@File    : test_prompt_writer.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger
# 从 metagpt.tools.prompt_writer 模块中导入 BEAGECTemplate, EnronTemplate, GPTPromptGenerator, WikiHowTemplate 类
from metagpt.tools.prompt_writer import (
    BEAGECTemplate,
    EnronTemplate,
    GPTPromptGenerator,
    WikiHowTemplate,
)

# 使用 pytest.mark.asyncio 装饰器标记异步测试
# 使用 pytest.mark.usefixtures 装饰器标记使用 llm_api fixture
@pytest.mark.asyncio
@pytest.mark.usefixtures("llm_api")
# 定义测试函数 test_gpt_prompt_generator，传入 llm_api 参数
async def test_gpt_prompt_generator(llm_api):
    # 创建 GPTPromptGenerator 对象
    generator = GPTPromptGenerator()
    # 定义示例字符串
    example = (
        "商品名称:WonderLab 新肌果味代餐奶昔 小胖瓶 胶原蛋白升级版 饱腹代餐粉6瓶 75g/瓶(6瓶/盒) 店铺名称:金力宁食品专营店 " "品牌:WonderLab 保质期:1年 产地:中国 净含量:450g"
    )
    # 调用 llm_api.aask_batch 方法，传入 generator.gen(example) 作为参数，获取结果
    results = await llm_api.aask_batch(generator.gen(example))
    # 记录结果
    logger.info(results)
    # 断言结果长度大于 0
    assert len(results) > 0

# 使用 pytest.mark.usefixtures 装饰器标记使用 llm_api fixture
def test_wikihow_template(llm_api):
    # 创建 WikiHowTemplate 对象
    template = WikiHowTemplate()
    # 定义问题字符串和步骤数
    question = "learn Python"
    step = 5
    # 调用 template.gen 方法，传入 question 和 step 作为参数，获取结果
    results = template.gen(question, step)
    # 断言结果长度大于 0，并且结果中包含特定字符串
    assert len(results) > 0
    assert any("Give me 5 steps to learn Python." in r for r in results)

# 使用 pytest.mark.usefixtures 装饰器标记使用 llm_api fixture
def test_enron_template(llm_api):
    # 创建 EnronTemplate 对象
    template = EnronTemplate()
    # 定义主题字符串
    subj = "Meeting Agenda"
    # 调用 template.gen 方法，传入 subj 作为参数，获取结果
    results = template.gen(subj)
    # 断言结果长度大于 0，并且结果中包含特定字符串
    assert len(results) > 0
    assert any('Write an email with the subject "Meeting Agenda".' in r for r in results)

# 定义测试函数 test_beagec_template
def test_beagec_template():
    # 创建 BEAGECTemplate 对象
    template = BEAGECTemplate()
    # 调用 template.gen 方法，获取结果
    results = template.gen()
    # 断言结果长度大于 0，并且结果中包含特定字符串
    assert len(results) > 0
    assert any(
        "Edit and revise this document to improve its grammar, vocabulary, spelling, and style." in r for r in results
    )

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行 pytest 测试，传入当前文件名和 "-s" 参数
    pytest.main([__file__, "-s"])

```