# `.\DB-GPT-src\examples\rag\summary_extractor_example.py`

```py
"""Summary extractor example.
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
            python examples/rag/summary_extractor_example.py
"""

# 引入 asyncio 库，用于异步编程
import asyncio
# 引入 os 模块，用于操作系统相关功能
import os

# 从 dbgpt.configs.model_config 中导入 ROOT_PATH 常量
from dbgpt.configs.model_config import ROOT_PATH
# 从 dbgpt.model.proxy 中导入 OpenAILLMClient 类
from dbgpt.model.proxy import OpenAILLMClient
# 从 dbgpt.rag 中导入 ChunkParameters 类
from dbgpt.rag import ChunkParameters
# 从 dbgpt.rag.assembler 中导入 SummaryAssembler 类
from dbgpt.rag.assembler import SummaryAssembler
# 从 dbgpt.rag.knowledge 中导入 KnowledgeFactory 类
from dbgpt.rag.knowledge import KnowledgeFactory

# 定义异步函数 main()
async def main():
    # 构建文件路径，连接 ROOT_PATH 和特定文件路径
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")
    # 创建 OpenAILLMClient 实例
    llm_client = OpenAILLMClient()
    # 从文件路径创建知识库对象
    knowledge = KnowledgeFactory.from_file_path(file_path)
    # 定义 ChunkParameters 实例，使用 CHUNK_BY_SIZE 策略
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
    # 从知识库、ChunkParameters 和 llm_client 创建 SummaryAssembler 实例
    assembler = SummaryAssembler.load_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        llm_client=llm_client,
        model_name="gpt-3.5-turbo",
    )
    # 生成摘要并返回结果
    return await assembler.generate_summary()

# 当文件作为脚本直接执行时
if __name__ == "__main__":
    # 运行异步函数 main()，获取生成的摘要
    output = asyncio.run(main())
    # 打印输出摘要结果
    print(f"output: \n\n{output}")
```