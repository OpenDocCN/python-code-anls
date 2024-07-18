# `.\graphrag\graphrag\index\graph\extractors\summarize\description_summary_extractor.py`

```py
        ) -> SummarizationResult:
        """异步方法，接受项目及其描述列表作为输入，返回SummarizationResult对象。

        Args:
            items (str | tuple[str, str]): 项目标识或元组形式的项目标识。
            descriptions (list[str]): 描述项目的字符串列表。

        Returns:
            SummarizationResult: 包含项目及其描述总结的结果对象。
        """
        result = ""
        # 如果描述列表为空，结果为空字符串
        if len(descriptions) == 0:
            result = ""
        # 如果描述列表只有一个元素，结果为该元素
        if len(descriptions) == 1:
            result = descriptions[0]
        else:
            # 否则，调用内部方法_summarize_descriptions进行多描述的总结
            result = await self._summarize_descriptions(items, descriptions)

        # 返回SummarizationResult对象，包含项目标识及其描述的总结结果
        return SummarizationResult(
            items=items,
            description=result or "",
        )
    async def summarize_descriptions(
        self, items: list[str] | tuple[str, ...], descriptions: list[str]
    ) -> str:
        """Summarize descriptions into a single description."""
        # 如果 items 是列表，则按顺序排列；否则保持原样
        sorted_items = sorted(items) if isinstance(items, list) else items

        # 安全检查，确保 descriptions 总是一个列表
        if not isinstance(descriptions, list):
            descriptions = [descriptions]

        # 计算可用的输入令牌数，减去总结提示的令牌数
        usable_tokens = self._max_input_tokens - num_tokens_from_string(
            self._summarization_prompt
        )
        descriptions_collected = []
        result = ""

        # 遍历所有描述
        for i, description in enumerate(descriptions):
            usable_tokens -= num_tokens_from_string(description)
            descriptions_collected.append(description)

            # 如果缓冲区已满或所有描述都已添加，则进行总结
            if (usable_tokens < 0 and len(descriptions_collected) > 1) or (
                i == len(descriptions) - 1
            ):
                # 调用 _summarize_descriptions_with_llm 方法进行总结
                result = await self._summarize_descriptions_with_llm(
                    sorted_items, descriptions_collected
                )

                # 如果需要再进行循环，则重置值
                if i != len(descriptions) - 1:
                    descriptions_collected = [result]
                    usable_tokens = (
                        self._max_input_tokens
                        - num_tokens_from_string(self._summarization_prompt)
                        - num_tokens_from_string(result)
                    )

        return result

    async def _summarize_descriptions_with_llm(
        self, items: str | tuple[str, str] | list[str], descriptions: list[str]
    ):
        """Summarize descriptions using the LLM."""
        # 调用语言模型进行总结，返回响应
        response = await self._llm(
            self._summarization_prompt,
            name="summarize",
            variables={
                self._entity_name_key: json.dumps(items),
                self._input_descriptions_key: json.dumps(sorted(descriptions)),
            },
            model_parameters={"max_tokens": self._max_summary_length},
        )
        # 计算最终结果
        return str(response.output)
```