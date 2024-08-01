# `.\DB-GPT-src\examples\rag\rewrite_rag_example.py`

```py
"""Query rewrite example.

    pre-requirements:
        1. install openai python sdk
        ```
            pip install openai
        ```py
        2. set openai key and base
        ```
            export OPENAI_API_KEY={your_openai_key}
            export OPENAI_API_BASE={your_openai_base}
        ```py
        or
        ```
            import os
            os.environ["OPENAI_API_KEY"] = {your_openai_key}
            os.environ["OPENAI_API_BASE"] = {your_openai_base}
        ```py
    Examples:
        ..code-block:: shell
            python examples/rag/rewrite_rag_example.py
"""

import asyncio  # 引入异步IO库

from dbgpt.model.proxy import OpenAILLMClient  # 从dbgpt.model.proxy模块导入OpenAILLMClient类
from dbgpt.rag.retriever import QueryRewrite  # 从dbgpt.rag.retriever模块导入QueryRewrite类


async def main():
    # 定义查询字符串
    query = "compare steve curry and lebron james"
    # 创建OpenAI语言模型客户端
    llm_client = OpenAILLMClient()
    # 创建查询重写对象，使用gpt-3.5-turbo模型
    reinforce = QueryRewrite(
        llm_client=llm_client,
        model_name="gpt-3.5-turbo",
    )
    # 调用重写方法异步获取重写后的结果
    return await reinforce.rewrite(origin_query=query, nums=1)


if __name__ == "__main__":
    # 运行主函数并获取输出结果
    output = asyncio.run(main())
    # 打印输出结果
    print(f"output: \n\n{output}")
```