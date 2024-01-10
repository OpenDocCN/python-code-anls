# `MetaGPT\tests\metagpt\roles\test_researcher.py`

```

# 导入所需的模块
from pathlib import Path
from random import random
from tempfile import TemporaryDirectory
import pytest
from metagpt.roles import researcher

# 定义一个异步函数，模拟LLM提问的情况
async def mock_llm_ask(self, prompt: str, system_msgs):
    # 根据提示返回相应的关键词
    if "Please provide up to 2 necessary keywords" in prompt:
        return '["dataiku", "datarobot"]'
    # 根据提示返回相关的查询
    elif "Provide up to 4 queries related to your research topic" in prompt:
        return (
            '["Dataiku machine learning platform", "DataRobot AI platform comparison", '
            '"Dataiku vs DataRobot features", "Dataiku and DataRobot use cases"]'
        )
    # 根据提示返回排序结果
    elif "sort the remaining search results" in prompt:
        return "[1,2]"
    # 根据提示返回是否相关
    elif "Not relevant." in prompt:
        return "Not relevant" if random() > 0.5 else prompt[-100:]
    # 根据提示返回研究报告
    elif "provide a detailed research report" in prompt:
        return f"# Research Report\n## Introduction\n{prompt}"
    return ""

# 定义一个异步测试函数
@pytest.mark.asyncio
async def test_researcher(mocker):
    # 使用临时目录进行测试
    with TemporaryDirectory() as dirname:
        topic = "dataiku vs. datarobot"
        # 模拟LLM提问的情况
        mocker.patch("metagpt.provider.base_llm.BaseLLM.aask", mock_llm_ask)
        # 设置研究路径
        researcher.RESEARCH_PATH = Path(dirname)
        # 运行研究者的功能
        await researcher.Researcher().run(topic)
        # 断言研究报告是否以特定字符开头
        assert (researcher.RESEARCH_PATH / f"{topic}.md").read_text().startswith("# Research Report")

# 定义一个测试写报告的函数
def test_write_report(mocker):
    # 使用临时目录进行测试
    with TemporaryDirectory() as dirname:
        for i, topic in enumerate(
            [
                ("1./metagpt"),
                ('2.:"metagpt'),
                ("3.*?<>|metagpt"),
                ("4. metagpt\n"),
            ]
        ):
            # 设置研究路径
            researcher.RESEARCH_PATH = Path(dirname)
            content = "# Research Report"
            # 写入报告
            researcher.Researcher().write_report(topic, content)
            # 断言报告是否以特定字符开头
            assert (researcher.RESEARCH_PATH / f"{i+1}. metagpt.md").read_text().startswith("# Research Report")

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```