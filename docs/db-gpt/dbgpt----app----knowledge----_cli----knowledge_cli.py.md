# `.\DB-GPT-src\dbgpt\app\knowledge\_cli\knowledge_cli.py`

```py
# 导入必要的模块：functools、logging、os 和 click
import functools
import logging
import os

import click

# 从 dbgpt.configs.model_config 导入 DATASETS_DIR 变量
from dbgpt.configs.model_config import DATASETS_DIR

# 定义默认的 API 地址常量
_DEFAULT_API_ADDRESS: str = "http://127.0.0.1:5670"
# 设置 API_ADDRESS 变量初始值为默认的 API 地址
API_ADDRESS: str = _DEFAULT_API_ADDRESS

# 创建名为 "dbgpt_cli" 的 logger 对象
logger = logging.getLogger("dbgpt_cli")

# 定义命令行工具组 "knowledge"，接受 --address 选项
@click.group("knowledge")
@click.option(
    "--address",
    type=str,
    default=API_ADDRESS,
    required=False,
    show_default=True,
    help=(
        "Address of the Api server(If not set, try to read from environment variable: API_ADDRESS)."
    ),
)
def knowledge_cli_group(address: str):
    """Knowledge command line tool"""
    # 在函数作用域内修改全局的 API_ADDRESS 变量
    global API_ADDRESS
    # 如果 address 参数与默认 API 地址相同，则尝试从环境变量 API_ADDRESS 中获取值
    if address == _DEFAULT_API_ADDRESS:
        address = os.getenv("API_ADDRESS", _DEFAULT_API_ADDRESS)
    # 将最终确定的地址赋给全局的 API_ADDRESS 变量
    API_ADDRESS = address

# 装饰器函数，用于为命令行函数添加多个选项
def add_knowledge_options(func):
    # 添加名为 "space_name" 的 --space_name 选项
    @click.option(
        "--space_name",
        required=False,
        type=str,
        default="default",
        show_default=True,
        help="Your knowledge space name",
    )
    # 使用 functools 模块的 wraps 装饰器，确保函数被正确包装
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 调用被装饰的函数，并返回其结果
        return func(*args, **kwargs)

    # 返回装饰后的函数对象
    return wrapper

# 在 knowledge_cli_group 命令组上创建新的命令 "load"，并添加多个选项
@knowledge_cli_group.command()
@add_knowledge_options
@click.option(
    "--vector_store_type",
    required=False,
    type=str,
    default="Chroma",
    show_default=True,
    help="Vector store type.",
)
@click.option(
    "--local_doc_path",
    required=False,
    type=str,
    default=DATASETS_DIR,
    show_default=True,
    help="Your document directory or document file path.",
)
@click.option(
    "--skip_wrong_doc",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Skip wrong document.",
)
@click.option(
    "--overwrite",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Overwrite existing document(they has same name).",
)
@click.option(
    "--max_workers",
    required=False,
    type=int,
    default=None,
    help="The maximum number of threads that can be used to upload document.",
)
@click.option(
    "--pre_separator",
    required=False,
    type=str,
    default=None,
    help="Preseparator, this separator is used for pre-splitting before the document is "
    "actually split by the text splitter. Preseparator are not included in the vectorized text. ",
)
@click.option(
    "--separator",
    required=False,
    type=str,
    default=None,
    help="This is the document separator. Currently, only one separator is supported.",
)
@click.option(
    "--chunk_size",
    required=False,
    type=int,
    default=None,
    help="Maximum size of chunks to split.",
)
@click.option(
    "--chunk_overlap",
    required=False,
    type=int,
    default=None,
    help="Overlap in characters between chunks.",
)
def load(
    space_name: str,
    vector_store_type: str,
    local_doc_path: str,
    skip_wrong_doc: bool,
    overwrite: bool,
    max_workers: int,
    pre_separator: str,
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Load your local documents to DB-GPT"""
    # 从指定路径导入知识初始化函数
    from dbgpt.app.knowledge._cli.knowledge_client import knowledge_init
    
    # 调用知识初始化函数，配置和启动知识服务
    knowledge_init(
        API_ADDRESS,         # API 地址，指定知识服务的接口位置
        space_name,          # 知识空间名称，用于标识不同的知识库
        vector_store_type,   # 向量存储类型，指定如何存储和索引文档向量
        local_doc_path,      # 本地文档路径，指定从哪里加载待索引的文档
        skip_wrong_doc,      # 是否跳过无效文档的标志，用于文档索引过程中的错误处理
        overwrite,           # 是否覆盖已存在的索引，影响文档索引的行为
        max_workers,         # 最大工作线程数，控制并行处理文档的数量
        pre_separator,       # 前置分隔符，用于处理文档内容的分隔逻辑
        separator,           # 分隔符，用于划分文档内容中的各个部分
        chunk_size,          # 分块大小，用于分割大文档以便并行处理
        chunk_overlap        # 分块重叠量，控制分块时的重叠区域大小
    )
# 定义一个命令组，用于处理知识空间相关操作的命令
@knowledge_cli_group.command()
# 添加知识相关选项
@add_knowledge_options
# 点击命令行选项，用于指定要删除的文档名称或整个空间的名称
@click.option(
    "--doc_name",
    required=False,
    type=str,
    default=None,
    help="The document name you want to delete. If doc_name is None, this command will delete the whole space.",
)
# 点击命令行选项，用于确认操作的选择
@click.option(
    "-y",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Confirm your choice",
)
# 删除知识空间或其中文档的函数定义
def delete(space_name: str, doc_name: str, y: bool):
    """Delete your knowledge space or document in space"""
    # 导入知识删除函数
    from dbgpt.app.knowledge._cli.knowledge_client import knowledge_delete
    # 调用知识删除函数，传递API地址、空间名称、文档名称以及确认标志
    knowledge_delete(API_ADDRESS, space_name, doc_name, confirm=y)


@knowledge_cli_group.command()
# 点击命令行选项，用于指定要操作的知识空间的名称
@click.option(
    "--space_name",
    required=False,
    type=str,
    default=None,
    show_default=True,
    help="Your knowledge space name. If None, list all spaces",
)
# 点击命令行选项，用于指定要操作的知识文档的ID
@click.option(
    "--doc_id",
    required=False,
    type=int,
    default=None,
    show_default=True,
    help="Your document id in knowledge space. If Not None, list all chunks in current document",
)
# 点击命令行选项，用于指定查询的页码
@click.option(
    "--page",
    required=False,
    type=int,
    default=1,
    show_default=True,
    help="The page for every query",
)
# 点击命令行选项，用于指定每页返回的条目数
@click.option(
    "--page_size",
    required=False,
    type=int,
    default=20,
    show_default=True,
    help="The page size for every query",
)
# 点击命令行选项，用于指定是否显示文档内容的标志
@click.option(
    "--show_content",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Query the document content of chunks",
)
# 点击命令行选项，用于指定输出的格式
@click.option(
    "--output",
    required=False,
    type=click.Choice(["text", "html", "csv", "latex", "json"]),
    default="text",
    help="The output format",
)
# 列出知识空间的函数定义
def list(
    space_name: str,
    doc_id: int,
    page: int,
    page_size: int,
    show_content: bool,
    output: str,
):
    """List knowledge space"""
    # 导入知识列表函数
    from dbgpt.app.knowledge._cli.knowledge_client import knowledge_list
    # 调用知识列表函数，传递API地址、空间名称、页码、页大小、文档ID、是否显示内容以及输出格式
    knowledge_list(
        API_ADDRESS, space_name, page, page_size, doc_id, show_content, output
    )
```