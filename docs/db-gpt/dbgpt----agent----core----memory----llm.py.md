# `.\DB-GPT-src\dbgpt\agent\core\memory\llm.py`

```py
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings.

        1. First, split by newline
        2. Remove whitespace from each line
        """
        # Split the input text by newline characters
        lines = re.split(r"\n", text.strip())
        # Remove any lines that are empty after stripping whitespace
        lines = [line for line in lines if line.strip()]
        # Remove leading numbers followed by a dot (e.g., "1. ", "2. ") from each line
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]
    def _parse_number(text: str, importance_weight: Optional[float] = None) -> float:
        """Parse a number from a string."""
        # 使用正则表达式寻找字符串开头的数字序列
        match = re.search(r"^\D*(\d+)", text)
        if match:
            # 如果找到匹配的数字序列，将其转换为浮点数
            score = float(match.group(1))
            # 如果提供了重要性权重参数，则根据权重进行调整
            if importance_weight is not None:
                score = (score / 10) * importance_weight
            # 返回解析得到的分数
            return score
        else:
            # 如果未找到匹配的数字序列，则返回默认值 0.0
            return 0.0
class LLMInsightExtractor(BaseLLMCaller, InsightExtractor[T]):
    """LLM Insight Extractor.

    Get high-level insights from memories.
    """

    # 提示信息，用于生成高层次洞察
    prompt: str = (
        "There are some memories: {content}\nCan you infer from the "
        "above memories the high-level insight for this person's character? The insight"
        " needs to be significantly different from the content and structure of the "
        "original memories.Respond in one sentence.\n\n"
        "Results:"
    )

    async def extract_insights(
        self,
        memory_fragment: T,
        llm_client: Optional[LLMClient] = None,
    ) -> InsightMemoryFragment[T]:
        """Extract insights from memory fragments.

        Args:
            memory_fragment(T): Memory fragment
            llm_client(Optional[LLMClient]): LLM client

        Returns:
            InsightMemoryFragment: The insights of the memory fragment.
        """
        # 调用语言模型生成洞察信息
        insights_str: str = await self.call_llm(
            self.prompt, llm_client, content=memory_fragment.raw_observation
        )
        # 解析生成的洞察信息为列表
        insights_list = self._parse_list(insights_str)
        # 返回包含洞察信息的 InsightMemoryFragment 对象
        return InsightMemoryFragment(memory_fragment, insights_list)


class LLMImportanceScorer(BaseLLMCaller, ImportanceScorer[T]):
    """LLM Importance Scorer.

    Score the importance of memories.
    """

    # 提示信息，用于生成记忆重要性评分
    prompt: str = (
        "Please give an importance score between 1 to 10 for the following "
        "observation. Higher score indicates the observation is more important. More "
        "rules that should be followed are:"
        "\n(1): Learning experience of a certain skill is important"
        "\n(2): The occurrence of a particular event is important"
        "\n(3): User thoughts and emotions matter"
        "\n(4): More informative indicates more important."
        "Please respond with a single integer."
        "\nObservation:{content}"
        "\nRating:"
    )

    async def score_importance(
        self,
        memory_fragment: T,
        llm_client: Optional[LLMClient] = None,
    ) -> float:
        """Score the importance of memory fragments.

        Args:
            memory_fragment(T): Memory fragment
            llm_client(Optional[LLMClient]): LLM client

        Returns:
            float: The importance score of the memory fragment.
        """
        # 调用语言模型生成记忆重要性评分
        score: str = await self.call_llm(
            self.prompt, llm_client, content=memory_fragment.raw_observation
        )
        # 解析生成的评分字符串为数值
        return self._parse_number(score)
```