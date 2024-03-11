# `.\Langchain-Chatchat\server\chat\search_engine_chat.py`

```
# 导入必要的模块和类
from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from configs import (BING_SEARCH_URL, BING_SUBSCRIPTION_KEY, METAPHOR_API_KEY,
                     LLM_MODELS, SEARCH_ENGINE_TOP_K, TEMPERATURE, OVERLAP_SIZE)
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler

from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from sse_starlette import EventSourceResponse
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from server.chat.utils import History
from typing import AsyncIterable
import asyncio
import json
from typing import List, Optional, Dict
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from markdownify import markdownify

# 定义一个函数，用于执行必应搜索
def bing_search(text, result_len=SEARCH_ENGINE_TOP_K, **kwargs):
    # 如果缺少必应搜索的 URL 和订阅密钥，则返回提示信息
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [{"snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                 "title": "env info is not found",
                 "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html"}]
    # 创建必应搜索API包装器对象
    search = BingSearchAPIWrapper(bing_subscription_key=BING_SUBSCRIPTION_KEY,
                                  bing_search_url=BING_SEARCH_URL)
    # 返回搜索结果
    return search.results(text, result_len)

# 定义一个函数，用于执行DuckDuckGo搜索
def duckduckgo_search(text, result_len=SEARCH_ENGINE_TOP_K, **kwargs):
    # 创建DuckDuckGo搜索API包装器对象
    search = DuckDuckGoSearchAPIWrapper()
    # 返回搜索结果
    return search.results(text, result_len)

# 定义一个函数，用于执行隐喻搜索
def metaphor_search(
        text: str,
        result_len: int = SEARCH_ENGINE_TOP_K,
        split_result: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = OVERLAP_SIZE,
) -> List[Dict]:
    # 从 metaphor_python 模块中导入 Metaphor 类
    from metaphor_python import Metaphor

    # 如果没有设置 METAPHOR_API_KEY，则返回空列表
    if not METAPHOR_API_KEY:
        return []

    # 使用 METAPHOR_API_KEY 创建 Metaphor 客户端
    client = Metaphor(METAPHOR_API_KEY)
    # 使用客户端搜索文本，获取搜索结果
    search = client.search(text, num_results=result_len, use_autoprompt=True)
    # 从搜索结果中获取内容列表
    contents = search.get_contents().contents
    # 对每个内容进行 markdown 转换
    for x in contents:
        x.extract = markdownify(x.extract)

    # 如果需要对结果进行分词处理
    if split_result:
        # 将内容转换为 Document 对象
        docs = [Document(page_content=x.extract,
                         metadata={"link": x.url, "title": x.title})
                for x in contents]
        # 使用 RecursiveCharacterTextSplitter 对文档进行分词
        text_splitter = RecursiveCharacterTextSplitter(["\n\n", "\n", ".", " "],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        splitted_docs = text_splitter.split_documents(docs)

        # 如果切分后的文档数量大于要求的结果数量
        if len(splitted_docs) > result_len:
            # 使用 NormalizedLevenshtein 计算相似度，重新排序文档
            normal = NormalizedLevenshtein()
            for x in splitted_docs:
                x.metadata["score"] = normal.similarity(text, x.page_content)
            splitted_docs.sort(key=lambda x: x.metadata["score"], reverse=True)
            # 保留 TOP_K 个文档
            splitted_docs = splitted_docs[:result_len]

        # 将切分后的文档转换为指定格式
        docs = [{"snippet": x.page_content,
                 "link": x.metadata["link"],
                 "title": x.metadata["title"]}
                for x in splitted_docs]
    else:
        # 如果不需要分词处理，则直接使用原始内容
        docs = [{"snippet": x.extract,
                 "link": x.url,
                 "title": x.title}
                for x in contents]

    # 返回处理后的文档列表
    return docs
# 定义一个字典，将搜索引擎名称映射到对应的搜索函数
SEARCH_ENGINES = {"bing": bing_search,
                  "duckduckgo": duckduckgo_search,
                  "metaphor": metaphor_search,
                  }

# 将搜索结果转换为文档对象列表
def search_result2docs(search_results):
    docs = []
    for result in search_results:
        # 创建文档对象，提取页面内容、来源链接和文件名
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs

# 异步函数，根据查询、搜索引擎名称和参数获取搜索结果文档
async def lookup_search_engine(
        query: str,
        search_engine_name: str,
        top_k: int = SEARCH_ENGINE_TOP_K,
        split_result: bool = False,
):
    # 根据搜索引擎名称获取对应的搜索函数
    search_engine = SEARCH_ENGINES[search_engine_name]
    # 在线程池中运行搜索函数，获取搜索结果
    results = await run_in_threadpool(search_engine, query, result_len=top_k, split_result=split_result)
    # 将搜索结果转换为文档对象列表
    docs = search_result2docs(results)
    return docs
# 定义一个异步函数，用于搜索引擎对话
async def search_engine_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                             # 用户输入的查询内容
                             search_engine_name: str = Body(..., description="搜索引擎名称", examples=["duckduckgo"]),
                             # 搜索引擎的名称
                             top_k: int = Body(SEARCH_ENGINE_TOP_K, description="检索结果数量"),
                             # 检索结果的数量
                             history: List[History] = Body([],
                                                           description="历史对话",
                                                           examples=[[
                                                               {"role": "user",
                                                                "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                               {"role": "assistant",
                                                                "content": "虎头虎脑"}]]
                                                           ),
                             # 历史对话记录
                             stream: bool = Body(False, description="流式输出"),
                             # 是否流式输出
                             model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                             # LLM 模型的名称
                             temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                             # LLM 采样温度
                             max_tokens: Optional[int] = Body(None,
                                                              description="限制LLM生成Token数量，默认None代表模型最大值"),
                             # 限制LLM生成Token数量
                             prompt_name: str = Body("default",
                                                     description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                             # 使用的prompt模板名称
                             split_result: bool = Body(False,
                                                       description="是否对搜索结果进行拆分（主要用于metaphor搜索引擎）")
                             ):
    # 如果搜索引擎名称不在已支持的搜索引擎列表中，则返回错误信息
    if search_engine_name not in SEARCH_ENGINES.keys():
        return BaseResponse(code=404, msg=f"未支持搜索引擎 {search_engine_name}")
    # 如果搜索引擎名称为"bing"且未设置BING_SUBSCRIPTION_KEY，则返回错误响应
    if search_engine_name == "bing" and not BING_SUBSCRIPTION_KEY:
        return BaseResponse(code=404, msg=f"要使用Bing搜索引擎，需要设置 `BING_SUBSCRIPTION_KEY`")

    # 将历史记录数据转换为History对象列表
    history = [History.from_data(h) for h in history]

    # 返回事件源响应，调用search_engine_chat_iterator函数生成搜索引擎聊天迭代器
    return EventSourceResponse(search_engine_chat_iterator(query=query,
                                                           search_engine_name=search_engine_name,
                                                           top_k=top_k,
                                                           history=history,
                                                           model_name=model_name,
                                                           prompt_name=prompt_name),
                               )
```