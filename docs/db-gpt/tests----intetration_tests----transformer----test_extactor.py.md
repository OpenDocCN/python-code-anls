# `.\DB-GPT-src\tests\intetration_tests\transformer\test_extactor.py`

```py
import json  # 导入处理 JSON 数据的模块
import pytest  # 导入 pytest 测试框架

from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient  # 导入 OpenAI 的语言模型客户端
from dbgpt.rag.transformer.keyword_extractor import KeywordExtractor  # 导入关键词提取器
from dbgpt.rag.transformer.triplet_extractor import TripletExtractor  # 导入三元组提取器

model_name = "gpt-3.5-turbo"  # 定义模型名称为 "gpt-3.5-turbo"

@pytest.fixture
def llm():
    yield OpenAILLMClient()  # 返回一个 OpenAI 语言模型客户端实例

@pytest.fixture
def triplet_extractor(llm):
    yield TripletExtractor(llm, model_name)  # 返回一个三元组提取器实例，依赖于 llm 和 model_name 参数

@pytest.fixture
def keyword_extractor(llm):
    yield KeywordExtractor(llm, model_name)  # 返回一个关键词提取器实例，依赖于 llm 和 model_name 参数

@pytest.mark.asyncio
async def test_extract_triplet(triplet_extractor):
    triplets = await triplet_extractor.extract(
        "Alice is Bob and Cherry's mother and lives in New York.", 10
    )
    print(json.dumps(triplets))  # 打印以 JSON 格式输出的三元组结果
    assert len(triplets) == 3  # 断言三元组的数量为 3

@pytest.mark.asyncio
async def test_extract_keyword(keyword_extractor):
    keywords = await keyword_extractor.extract(
        "Alice is Bob and Cherry's mother and lives in New York.",
    )
    print(json.dumps(keywords))  # 打印以 JSON 格式输出的关键词结果
    assert len(keywords) > 0  # 断言关键词数量大于 0
```