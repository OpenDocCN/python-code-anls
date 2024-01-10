# `MetaGPT\metagpt\actions\research.py`

```

#!/usr/bin/env python
# 指定脚本解释器为python

from __future__ import annotations
# 导入未来的注解特性

import asyncio
from typing import Callable, Optional, Union
# 导入异步IO库和类型提示相关的模块

from pydantic import Field, parse_obj_as
# 导入pydantic库中的Field和parse_obj_as函数

from metagpt.actions import Action
from metagpt.config import CONFIG
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.provider.base_llm import BaseLLM
from metagpt.tools.search_engine import SearchEngine
from metagpt.tools.web_browser_engine import WebBrowserEngine, WebBrowserEngineType
from metagpt.utils.common import OutputParser
from metagpt.utils.text import generate_prompt_chunk, reduce_message_length
# 导入自定义模块

LANG_PROMPT = "Please respond in {language}."
# 定义语言提示常量

RESEARCH_BASE_SYSTEM = """You are an AI critical thinker research assistant. Your sole purpose is to write well \
written, critically acclaimed, objective and structured reports on the given text."""
# 定义研究基础系统文本常量

RESEARCH_TOPIC_SYSTEM = "You are an AI researcher assistant, and your research topic is:\n#TOPIC#\n{topic}"
# 定义研究主题系统文本常量

SEARCH_TOPIC_PROMPT = """Please provide up to 2 necessary keywords related to your research topic for Google search. \
Your response must be in JSON format, for example: ["keyword1", "keyword2"]."""
# 定义搜索主题提示常量

SUMMARIZE_SEARCH_PROMPT = """### Requirements
1. The keywords related to your research topic and the search results are shown in the "Search Result Information" section.
2. Provide up to {decomposition_nums} queries related to your research topic base on the search results.
3. Please respond in the following JSON format: ["query1", "query2", "query3", ...].

### Search Result Information
{search_results}
"""
# 定义总结搜索提示常量

COLLECT_AND_RANKURLS_PROMPT = """### Topic
{topic}
### Query
{query}

### The online search results
{results}

### Requirements
Please remove irrelevant search results that are not related to the query or topic. Then, sort the remaining search results \
based on the link credibility. If two results have equal credibility, prioritize them based on the relevance. Provide the
ranked results' indices in JSON format, like [0, 1, 3, 4, ...], without including other words.
"""
# 定义收集和排名URL提示常量

WEB_BROWSE_AND_SUMMARIZE_PROMPT = """### Requirements
1. Utilize the text in the "Reference Information" section to respond to the question "{query}".
2. If the question cannot be directly answered using the text, but the text is related to the research topic, please provide \
a comprehensive summary of the text.
3. If the text is entirely unrelated to the research topic, please reply with a simple text "Not relevant."
4. Include all relevant factual information, numbers, statistics, etc., if available.

### Reference Information
{content}
"""
# 定义浏览和总结网页提示常量

CONDUCT_RESEARCH_PROMPT = """### Reference Information
{content}

### Requirements
Please provide a detailed research report in response to the following topic: "{topic}", using the information provided \
above. The report must meet the following requirements:

- Focus on directly addressing the chosen topic.
- Ensure a well-structured and in-depth presentation, incorporating relevant facts and figures where available.
- Present data and findings in an intuitive manner, utilizing feature comparative tables, if applicable.
- The report should have a minimum word count of 2,000 and be formatted with Markdown syntax following APA style guidelines.
- Include all source URLs in APA format at the end of the report.
"""
# 定义进行研究提示常量

class CollectLinks(Action):
    """Action class to collect links from a search engine."""
    # 收集搜索引擎链接的动作类

    name: str = "CollectLinks"
    context: Optional[str] = None
    desc: str = "Collect links from a search engine."

    search_engine: SearchEngine = Field(default_factory=SearchEngine)
    rank_func: Optional[Callable[[list[str]], None]] = None
    # 定义属性

    async def run(
        self,
        topic: str,
        decomposition_nums: int = 4,
        url_per_query: int = 4,
        system_text: str | None = None,
    ) -> dict[str, list[str]]:
        """Run the action to collect links.

        Args:
            topic: The research topic.
            decomposition_nums: The number of search questions to generate.
            url_per_query: The number of URLs to collect per search question.
            system_text: The system text.

        Returns:
            A dictionary containing the search questions as keys and the collected URLs as values.
        """
        # 运行收集链接的动作

    async def _search_and_rank_urls(self, topic: str, query: str, num_results: int = 4) -> list[str]:
        """Search and rank URLs based on a query.

        Args:
            topic: The research topic.
            query: The search query.
            num_results: The number of URLs to collect.

        Returns:
            A list of ranked URLs.
        """
        # 根据查询搜索和排名URL

class WebBrowseAndSummarize(Action):
    """Action class to explore the web and provide summaries of articles and webpages."""
    # 探索网络并提供文章和网页摘要的动作类

    name: str = "WebBrowseAndSummarize"
    context: Optional[str] = None
    llm: BaseLLM = Field(default_factory=LLM)
    desc: str = "Explore the web and provide summaries of articles and webpages."
    browse_func: Union[Callable[[list[str]], None], None] = None
    web_browser_engine: Optional[WebBrowserEngine] = None
    # 定义属性

    async def run(
        self,
        url: str,
        *urls: str,
        query: str,
        system_text: str = RESEARCH_BASE_SYSTEM,
    ) -> dict[str, str]:
        """Run the action to browse the web and provide summaries.

        Args:
            url: The main URL to browse.
            urls: Additional URLs to browse.
            query: The research question.
            system_text: The system text.

        Returns:
            A dictionary containing the URLs as keys and their summaries as values.
        """
        # 运行浏览网络和提供摘要的动作

class ConductResearch(Action):
    """Action class to conduct research and generate a research report."""
    # 进行研究和生成研究报告的动作类

    name: str = "ConductResearch"
    context: Optional[str] = None
    llm: BaseLLM = Field(default_factory=LLM)
    # 定义属性

def get_research_system_text(topic: str, language: str):
    """Get the system text for conducting research.

    Args:
        topic: The research topic.
        language: The language for the system text.

    Returns:
        The system text for conducting research.
    """
    # 获取进行研究的系统文本

```