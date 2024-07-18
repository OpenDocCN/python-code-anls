# `.\graphrag\graphrag\query\__main__.py`

```py
"""The Query Engine package root."""

import argparse  # 导入处理命令行参数的模块
from enum import Enum  # 导入枚举类型的支持

from .cli import run_global_search, run_local_search  # 导入本地和全局搜索函数

INVALID_METHOD_ERROR = "Invalid method"  # 定义无效方法错误消息

class SearchType(Enum):
    """定义要运行的搜索类型枚举。"""

    LOCAL = "local"   # 本地搜索类型
    GLOBAL = "global"  # 全局搜索类型

    def __str__(self):
        """返回枚举值的字符串表示形式。"""
        return self.value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象

    parser.add_argument(
        "--data",
        help="从管道输出数据的路径",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--root",
        help="数据项目根目录。默认值为当前目录",
        required=False,
        default=".",
        type=str,
    )

    parser.add_argument(
        "--method",
        help="要运行的方法，可以是 local 或 global 之一",
        required=True,
        type=SearchType,
        choices=list(SearchType),
    )

    parser.add_argument(
        "--community_level",
        help="Leiden社区层次结构中的社区级别，从中加载社区报告。较高的值意味着我们使用较小社区的报告。",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--response_type",
        help="自由格式描述响应类型和格式，可以是任何内容，如多段落、单段落、单句子、3-7点列表、单页、多页报告等",
        type=str,
        default="Multiple Paragraphs",
    )

    parser.add_argument(
        "query",
        nargs=1,
        help="要运行的查询",
        type=str,
    )

    args = parser.parse_args()  # 解析命令行参数

    match args.method:  # 根据方法类型选择相应的搜索函数
        case SearchType.LOCAL:
            run_local_search(
                args.data,
                args.root,
                args.community_level,
                args.response_type,
                args.query[0],
            )
        case SearchType.GLOBAL:
            run_global_search(
                args.data,
                args.root,
                args.community_level,
                args.response_type,
                args.query[0],
            )
        case _:  # 如果方法类型不合法，则抛出值错误
            raise ValueError(INVALID_METHOD_ERROR)
```