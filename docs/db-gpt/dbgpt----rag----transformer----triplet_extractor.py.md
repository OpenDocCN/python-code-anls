# `.\DB-GPT-src\dbgpt\rag\transformer\triplet_extractor.py`

```py
"""TripletExtractor class."""
import logging  # 导入日志模块
import re  # 导入正则表达式模块
from typing import Any, List, Optional, Tuple  # 导入类型提示模块

from dbgpt.core import LLMClient  # 导入LLMClient类
from dbgpt.rag.transformer.llm_extractor import LLMExtractor  # 导入LLMExtractor类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

TRIPLET_EXTRACT_PT = (
    "Some text is provided below. Given the text, "
    "extract up to knowledge triplets as more as possible "
    "in the form of (subject, predicate, object).\n"
    "Avoid stopwords.\n"
    "---------------------\n"
    "Example:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Alice has 2 apples.\n"
    "Triplets:\n(Alice, has 2, apple)\n"
    "Text: Alice was given 1 apple by Bob.\n"
    "Triplets:(Bob, gives 1 apple, Bob)\n"
    "Text: Alice was pushed by Bob.\n"
    "Triplets:(Bob, pushes, Alice)\n"
    "Text: Bob's mother Alice has 2 apples.\n"
    "Triplets:\n(Alice, is mother of, Bob)\n(Alice, has 2, apple)\n"
    "Text: A Big monkey climbed up the tall fruit tree and picked 3 peaches.\n"
    "Triplets:\n(monkey, climbed up, fruit tree)\n(monkey, picked 3, peach)\n"
    "Text: Alice has 2 apples, she gives 1 to Bob.\n"
    "Triplets:\n"
    "(Alice, has 2, apple)\n(Alice, gives 1 apple, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

class TripletExtractor(LLMExtractor):
    """TripletExtractor class."""

    def __init__(self, llm_client: LLMClient, model_name: str):
        """Initialize the TripletExtractor."""
        super().__init__(llm_client, model_name, TRIPLET_EXTRACT_PT)  # 调用父类的初始化方法

    def _parse_response(
        self, text: str, limit: Optional[int] = None
    ) -> List[Tuple[Any, ...]]:
        triplets = []  # 初始化一个空列表，用于存储三元组

        for line in text.split("\n"):  # 遍历文本中的每一行
            for match in re.findall(r"\((.*?)\)", line):  # 使用正则表达式找到括号内的内容
                splits = match.split(",")  # 将匹配到的内容按逗号分割
                parts = [split.strip() for split in splits if split.strip()]  # 去除空格后生成部分
                if len(parts) == 3:  # 如果部分长度为3
                    parts = [
                        p.strip(
                            "`~!@#$%^&*()-=+[]\\{}|;':\",./<>?"
                            "·！￥&*（）—【】、「」；‘’：“”，。、《》？"
                        )
                        for p in parts
                    ]  # 去除特殊字符后生成最终部分
                    triplets.append(tuple(parts))  # 将处理后的部分生成三元组并添加到列表中
                    if limit and len(triplets) >= limit:  # 如果有限制并且已达到限制数量
                        return triplets  # 返回已生成的三元组列表

        return triplets  # 返回所有生成的三元组列表
```