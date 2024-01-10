# `MetaGPT\tests\metagpt\test_gpt.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 19:47
@Author  : alexanderwu
@File    : test_gpt.py
"""
# 导入所需的模块
import openai  # 导入openai模块
import pytest  # 导入pytest模块

from metagpt.config import CONFIG  # 从metagpt.config模块导入CONFIG变量
from metagpt.logs import logger  # 从metagpt.logs模块导入logger变量

# 使用pytest的usefixtures装饰器，指定llm_api为测试用例的fixture
@pytest.mark.usefixtures("llm_api")
class TestGPT:
    # 使用pytest的mark装饰器，指定为异步测试用例
    @pytest.mark.asyncio
    async def test_llm_api_aask(self, llm_api):
        # 调用llm_api的aask方法，传入参数"hello chatgpt"，stream设置为False
        answer = await llm_api.aask("hello chatgpt", stream=False)
        logger.info(answer)  # 记录日志
        assert len(answer) > 0  # 断言返回的answer长度大于0

        # 再次调用llm_api的aask方法，传入参数"hello chatgpt"，stream设置为True
        answer = await llm_api.aask("hello chatgpt", stream=True)
        logger.info(answer)  # 记录日志
        assert len(answer) > 0  # 断言返回的answer长度大于0

    @pytest.mark.asyncio
    async def test_llm_api_aask_code(self, llm_api):
        try:
            # 调用llm_api的aask_code方法，传入参数["请扮演一个Google Python专家工程师，如果理解，回复明白", "写一个hello world"]，设置超时时间为60秒
            answer = await llm_api.aask_code(["请扮演一个Google Python专家工程师，如果理解，回复明白", "写一个hello world"], timeout=60)
            logger.info(answer)  # 记录日志
            assert len(answer) > 0  # 断言返回的answer长度大于0
        except openai.BadRequestError:  # 捕获openai模块的BadRequestError异常
            assert CONFIG.OPENAI_API_TYPE == "azure"  # 断言CONFIG.OPENAI_API_TYPE等于"azure"

    @pytest.mark.asyncio
    async def test_llm_api_costs(self, llm_api):
        # 调用llm_api的aask方法，传入参数"hello chatgpt"，stream设置为False
        await llm_api.aask("hello chatgpt", stream=False)
        costs = llm_api.get_costs()  # 获取llm_api的费用信息
        logger.info(costs)  # 记录日志
        assert costs.total_cost > 0  # 断言总费用大于0

# 如果当前文件被直接运行，则执行pytest测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```