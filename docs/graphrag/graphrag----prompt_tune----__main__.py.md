# `.\graphrag\graphrag\prompt_tune\__main__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Prompt auto templating package root."""

# 引入命令行参数解析模块
import argparse
# 引入异步事件循环模块
import asyncio
# 引入枚举类型模块
from enum import Enum

# 从 graphrag.prompt_tune.generator 中导入 MAX_TOKEN_COUNT 常量
from graphrag.prompt_tune.generator import MAX_TOKEN_COUNT
# 从 graphrag.prompt_tune.loader 中导入 MIN_CHUNK_SIZE 常量
from graphrag.prompt_tune.loader import MIN_CHUNK_SIZE

# 从当前包的 cli 模块中导入 fine_tune 函数
from .cli import fine_tune

# 定义文档选择类型的枚举类
class DocSelectionType(Enum):
    """The type of document selection to use."""
    # 全部选择
    ALL = "all"
    # 随机选择
    RANDOM = "random"
    # 顶部选择
    TOP = "top"

    def __str__(self):
        """Return the string representation of the enum value."""
        return self.value

# 当作为主程序运行时
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加参数：数据项目根目录，包括配置文件 yml、json 或 .env
    parser.add_argument(
        "--root",
        help="The data project root. Including the config yml, json or .env",
        required=False,
        type=str,
        default=".",
    )

    # 添加参数：输入数据相关的领域，例如 'space science'、'microbiology'、'environmental news'。如果为空，则从输入数据推断领域。
    parser.add_argument(
        "--domain",
        help="The domain your input data is related to. For example 'space science', 'microbiology', 'environmental news'. If left empty, the domain will be inferred from the input data.",
        required=False,
        default="",
        type=str,
    )

    # 添加参数：选择文档的方法，可以是 all、random 或 top 之一
    parser.add_argument(
        "--method",
        help="The method to select documents, one of: all, random or top",
        required=False,
        type=DocSelectionType,
        choices=list(DocSelectionType),
        default=DocSelectionType.RANDOM,
    )

    # 添加参数：在进行随机或顶部选择时加载文件的限制数
    parser.add_argument(
        "--limit",
        help="The limit of files to load when doing random or top selection",
        type=int,
        required=False,
        default=15,
    )

    # 添加参数：用于提示生成的最大 token 数量
    parser.add_argument(
        "--max-tokens",
        help="Max token count for prompt generation",
        type=int,
        required=False,
        default=MAX_TOKEN_COUNT,
    )

    # 添加参数：用于提示生成的 chunk 大小
    parser.add_argument(
        "--chunk-size",
        help="Max token count for prompt generation",
        type=int,
        required=False,
        default=MIN_CHUNK_SIZE,
    )

    # 添加参数：在 GraphRAG 上使用的主要语言
    parser.add_argument(
        "--language",
        help="Primary language used for inputs and outputs on GraphRAG",
        type=str,
        required=False,
        default="",
    )

    # 添加参数：使用无类型实体提取生成
    parser.add_argument(
        "--no-entity-types",
        help="Use untyped entity extraction generation",
        action="store_true",
        required=False,
        default=False,
    )

    # 添加参数：保存生成提示的文件夹
    parser.add_argument(
        "--output",
        help="Folder to save the generated prompts to",
        type=str,
        required=False,
        default="prompts",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取异步事件循环对象
    loop = asyncio.get_event_loop()

    # 运行 fine_tune 函数，进行模型微调
    loop.run_until_complete(
        fine_tune(
            args.root,  # 数据项目根目录
            args.domain,  # 数据相关领域
            str(args.method),  # 文档选择方法
            args.limit,  # 加载文件限制数
            args.max_tokens,  # 最大 token 数量
            args.chunk_size,  # chunk 大小
            args.language,  # 主要语言
            args.no_entity_types,  # 是否使用无类型实体生成
            args.output,  # 保存文件夹
        )
    )
```