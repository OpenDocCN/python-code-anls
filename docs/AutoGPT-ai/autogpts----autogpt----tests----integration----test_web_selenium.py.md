# `.\AutoGPT\autogpts\autogpt\tests\integration\test_web_selenium.py`

```py
# 导入 pytest 模块
import pytest

# 从 autogpt.agents.agent 模块中导入 Agent 类
# 从 autogpt.commands.web_selenium 模块中导入 BrowsingError, read_webpage 函数
from autogpt.agents.agent import Agent
from autogpt.commands.web_selenium import BrowsingError, read_webpage

# 使用 pytest.mark.vcr 装饰器标记测试用例
# 使用 pytest.mark.requires_openai_api_key 装饰器标记测试用例
# 使用 pytest.mark.asyncio 装饰器标记异步测试用例
async def test_browse_website_nonexistent_url(agent: Agent, cached_openai_client: None):
    # 定义网站 URL
    url = "https://auto-gpt-thinks-this-website-does-not-exist.com"
    # 定义问题
    question = "How to execute a barrel roll"

    # 使用 pytest.raises 检查是否抛出 BrowsingError 异常，并匹配异常信息为 "NAME_NOT_RESOLVED"
    with pytest.raises(BrowsingError, match="NAME_NOT_RESOLVED") as raised:
        # 异步调用 read_webpage 函数，传入 URL、问题和 agent 参数
        await read_webpage(url=url, question=question, agent=agent)

        # 对抛出的异常信息进行长度检查，确保不超过 200 个字符
        assert len(raised.exconly()) < 200
```