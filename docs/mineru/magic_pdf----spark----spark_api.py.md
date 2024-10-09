# `.\MinerU\magic_pdf\spark\spark_api.py`

```
# 从 loguru 模块导入 logger 用于日志记录
from loguru import logger

# 从 magic_pdf.libs.drop_reason 导入 DropReason，用于处理丢弃原因
from magic_pdf.libs.drop_reason import DropReason


# 从给定的 JSON 字典中获取数据源
def get_data_source(jso: dict):
    # 尝试从 JSON 中获取 "data_source"
    data_source = jso.get("data_source")
    # 如果 "data_source" 不存在，尝试获取 "file_source"
    if data_source is None:
        data_source = jso.get("file_source")
    # 返回数据源
    return data_source


# 从给定的 JSON 字典中获取数据类型
def get_data_type(jso: dict):
    # 尝试从 JSON 中获取 "data_type"
    data_type = jso.get("data_type")
    # 如果 "data_type" 不存在，尝试获取 "file_type"
    if data_type is None:
        data_type = jso.get("file_type")
    # 返回数据类型
    return data_type


# 从给定的 JSON 字典中获取书籍 ID
def get_bookid(jso: dict):
    # 尝试从 JSON 中获取 "bookid"
    book_id = jso.get("bookid")
    # 如果 "bookid" 不存在，尝试获取 "original_file_id"
    if book_id is None:
        book_id = jso.get("original_file_id")
    # 返回书籍 ID
    return book_id


# 处理异常并更新 JSON 字典
def exception_handler(jso: dict, e):
    # 记录异常信息
    logger.exception(e)
    # 标记需要丢弃
    jso["_need_drop"] = True
    # 记录丢弃原因
    jso["_drop_reason"] = DropReason.Exception
    # 记录异常信息
    jso["_exception"] = f"ERROR: {e}"
    # 返回更新后的 JSON
    return jso


# 从给定的 JSON 字典中生成书籍名称
def get_bookname(jso: dict):
    # 获取数据源
    data_source = get_data_source(jso)
    # 从 JSON 中获取文件 ID
    file_id = jso.get("file_id")
    # 生成书籍名称
    book_name = f"{data_source}/{file_id}"
    # 返回书籍名称
    return book_name


# 从 JSON 中提取数据并返回一个字典
def spark_json_extractor(jso: dict) -> dict:

    """
    从json中提取数据，返回一个dict
    """

    # 返回包含 PDF 类型和文档布局结果的字典
    return {
        "_pdf_type": jso["_pdf_type"],
        "model_list": jso["doc_layout_result"],
    }
```