# `.\DB-GPT-src\dbgpt\rag\transformer\keyword_extractor.py`

```py
"""
KeywordExtractor class.
"""
# 导入日志模块
import logging
# 导入类型提示模块
from typing import List, Optional

# 导入调试模块的核心功能
from dbgpt.core import LLMClient
# 导入LLMExtractor类
from dbgpt.rag.transformer.llm_extractor import LLMExtractor

# 关键词抽取的提示文本模板
KEYWORD_EXTRACT_PT = (
    "A question is provided below. Given the question, extract up to "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question.\n"
    "Generate as more as possible synonyms or alias of the keywords "
    "considering possible cases of capitalization, pluralization, "
    "common expressions, etc.\n"
    "Avoid stopwords.\n"
    "Provide the keywords and synonyms in comma-separated format."
    "Formatted keywords and synonyms text should be separated by a semicolon.\n"
    "---------------------\n"
    "Example:\n"
    "Text: Alice is Bob's mother.\n"
    "Keywords:\nAlice,mother,Bob;mummy\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Keywords:\nPhilz,coffee shop,Berkeley,1982;coffee bar,coffee house\n"
    "---------------------\n"
    "Text: {text}\n"
    "Keywords:\n"
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class KeywordExtractor(LLMExtractor):
    """
    KeywordExtractor class.

    继承自LLMExtractor类，用于从文本中提取关键词。
    """

    def __init__(self, llm_client: LLMClient, model_name: str):
        """
        Initialize the KeywordExtractor.

        初始化关键词提取器，设置语言模型客户端和模型名称。
        """
        super().__init__(llm_client, model_name, KEYWORD_EXTRACT_PT)

    def _parse_response(self, text: str, limit: Optional[int] = None) -> List[str]:
        """
        Parse the response text to extract keywords.

        解析响应文本以提取关键词。

        Args:
        - text (str): 输入的文本
        - limit (Optional[int]): 最大返回关键词数量限制

        Returns:
        - List[str]: 提取的关键词列表
        """
        keywords = set()

        # 按分号分隔文本，处理每个分段
        for part in text.split(";"):
            # 按逗号分隔每个分段中的关键词
            for s in part.strip().split(","):
                keyword = s.strip()
                if keyword:
                    keywords.add(keyword)
                    # 如果达到了关键词数量限制，则返回结果
                    if limit and len(keywords) >= limit:
                        return list(keywords)

        return list(keywords)
```