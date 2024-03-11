# `.\Langchain-Chatchat\server\knowledge_base\migrate.py`

```py
# 从configs模块中导入所需的变量和函数
from configs import (
    EMBEDDING_MODEL, DEFAULT_VS_TYPE, ZH_TITLE_ENHANCE,
    CHUNK_SIZE, OVERLAP_SIZE,
    logger, log_verbose
)
# 从knowledge_base.utils模块中导入所需的函数和类
from server.knowledge_base.utils import (
    get_file_path, list_kbs_from_folder,
    list_files_from_folder, files2docs_in_thread,
    KnowledgeFile
)
# 从kb_service.base模块中导入KBServiceFactory类
from server.knowledge_base.kb_service.base import KBServiceFactory
# 从conversation_model模块中导入ConversationModel类
from server.db.models.conversation_model import ConversationModel
# 从message_model模块中导入MessageModel类
from server.db.models.message_model import MessageModel
# 从knowledge_file_repository模块中导入add_file_to_db函数
from server.db.repository.knowledge_file_repository import add_file_to_db # ensure Models are imported
# 从knowledge_metadata_repository模块中导入add_summary_to_db函数
from server.db.repository.knowledge_metadata_repository import add_summary_to_db

# 从db.base模块中导入Base类和engine对象
from server.db.base import Base, engine
# 从db.session模块中导入session_scope函数
from server.db.session import session_scope
# 导入os模块
import os
# 从dateutil.parser模块中导入parse函数
from dateutil.parser import parse
# 从typing模块中导入Literal和List类型
from typing import Literal, List

# 定义函数create_tables，用于创建数据库表
def create_tables():
    # 调用Base.metadata.create_all方法创建所有表
    Base.metadata.create_all(bind=engine)

# 定义函数reset_tables，用于重置数据库表
def reset_tables():
    # 调用Base.metadata.drop_all方法删除所有表
    Base.metadata.drop_all(bind=engine)
    # 调用create_tables函数重新创建表
    create_tables()

# 定义函数import_from_db，用于从备份数据库中导入数据到info.db
def import_from_db(
        sqlite_path: str = None,
        # csv_path: str = None,
) -> bool:
    """
    在知识库与向量库无变化的情况下，从备份数据库中导入数据到 info.db。
    适用于版本升级时，info.db 结构变化，但无需重新向量化的情况。
    请确保两边数据库表名一致，需要导入的字段名一致
    当前仅支持 sqlite
    """
    # 导入sqlite3模块并重命名为sql
    import sqlite3 as sql
    # 导入pprint函数
    from pprint import pprint

    # 获取Base.registry.mappers中的所有模型
    models = list(Base.registry.mappers)
    # 尝试连接 SQLite 数据库
    try:
        con = sql.connect(sqlite_path)
        # 设置返回的行为字典类型
        con.row_factory = sql.Row
        cur = con.cursor()
        # 获取数据库中所有表的名称
        tables = [x["name"] for x in cur.execute("select name from sqlite_master where type='table'").fetchall()]
        # 遍历模型列表
        for model in models:
            # 获取模型对应的表名
            table = model.local_table.fullname
            # 如果表名不在数据库中，则跳过当前模型
            if table not in tables:
                continue
            # 打印正在处理的表名
            print(f"processing table: {table}")
            # 使用 session_scope 上下文管理器
            with session_scope() as session:
                # 遍历数据库表中的每一行数据
                for row in cur.execute(f"select * from {table}").fetchall():
                    # 将行数据转换为字典，只包含模型中定义的列
                    data = {k: row[k] for k in row.keys() if k in model.columns}
                    # 如果数据中包含"create_time"字段，则解析为时间对象
                    if "create_time" in data:
                        data["create_time"] = parse(data["create_time"])
                    # 打印数据
                    pprint(data)
                    # 向会话中添加模型对象
                    session.add(model.class_(**data))
        # 关闭数据库连接
        con.close()
        # 返回 True 表示成功
        return True
    # 捕获异常并打印错误信息
    except Exception as e:
        print(f"无法读取备份数据库：{sqlite_path}。错误信息：{e}")
        # 返回 False 表示失败
        return False
# 将文件名列表转换为知识文件对象列表
def file_to_kbfile(kb_name: str, files: List[str]) -> List[KnowledgeFile]:
    # 初始化知识文件对象列表
    kb_files = []
    # 遍历文件列表
    for file in files:
        try:
            # 创建知识文件对象并添加到列表中
            kb_file = KnowledgeFile(filename=file, knowledge_base_name=kb_name)
            kb_files.append(kb_file)
        except Exception as e:
            # 如果出现异常，记录错误信息并跳过当前文件
            msg = f"{e}，已跳过"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
    # 返回知识文件对象列表
    return kb_files


# 将本地文件夹中的文件信息填充到数据库和/或向量存储中
def folder2db(
        kb_names: List[str],
        mode: Literal["recreate_vs", "update_in_db", "increment"],
        vs_type: Literal["faiss", "milvus", "pg", "chromadb"] = DEFAULT_VS_TYPE,
        embed_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
):
    """
    使用本地文件夹中的现有文件填充数据库和/或向量存储。
    将参数 `mode` 设置为:
        recreate_vs: 重新创建所有向量存储，并使用本地文件中的现有文件填充数据库信息
        fill_info_only(disabled): 不创建向量存储，仅使用现有文件填充数据库信息
        update_in_db: 使用仅存在于数据库中的本地文件更新向量存储和数据库信息
        increment: 仅为数据库中不存在的本地文件创建向量存储和数据库信息
    """
    # 将文件转换为文档向量并添加到向量库中
    def files2vs(kb_name: str, kb_files: List[KnowledgeFile]):
        # 遍历文件并将其转换为文档向量
        for success, result in files2docs_in_thread(kb_files,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    zh_title_enhance=zh_title_enhance):
            # 如果成功转换文件为文档向量
            if success:
                # 解构结果元组
                _, filename, docs = result
                # 打印正在处理的文件信息
                print(f"正在将 {kb_name}/{filename} 添加到向量库，共包含{len(docs)}条文档")
                # 创建知识文件对象
                kb_file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
                kb_file.splited_docs = docs
                # 将文档向量添加到向量库中
                kb.add_doc(kb_file=kb_file, not_refresh_vs_cache=True)
            # 如果转换文件为文档向量失败
            else:
                # 打印失败信息
                print(result)

    # 如果没有指定知识库名称，则从文件夹中获取知识库名称列表
    kb_names = kb_names or list_kbs_from_folder()
    # 遍历知识库名称列表
    for kb_name in kb_names:
        # 根据知识库名称、向量类型和嵌入模型获取知识库服务
        kb = KBServiceFactory.get_service(kb_name, vs_type, embed_model)
        # 如果知识库不存在，则创建知识库
        if not kb.exists():
            kb.create_kb()

        # 如果模式为"recreate_vs"，则清除向量库并从本地文件重新构建
        if mode == "recreate_vs":
            kb.clear_vs()
            kb.create_kb()
            # 获取知识库文件列表，并将文件转换为知识库文件对象
            kb_files = file_to_kbfile(kb_name, list_files_from_folder(kb_name))
            # 将文件转换为向量并保存向量库
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # 如果模式为"update_in_db"，则利用数据库中文件列表更新向量库
        elif mode == "update_in_db":
            files = kb.list_files()
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # 如果模式为"increment"，则进行增量向量化
        elif mode == "increment":
            db_files = kb.list_files()
            folder_files = list_files_from_folder(kb_name)
            # 获取本地目录与数据库中文件列表的差集作为需要增量向量化的文件列表
            files = list(set(folder_files) - set(db_files))
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        else:
            # 如果模式不支持，则打印提示信息
            print(f"unsupported migrate mode: {mode}")
# 删除数据库中不存在的文档，用于在用户在文件浏览器中删除文档文件后删除数据库中的文档
def prune_db_docs(kb_names: List[str]):
    # 遍历知识库名称列表
    for kb_name in kb_names:
        # 根据知识库名称获取知识库服务
        kb = KBServiceFactory.get_service_by_name(kb_name)
        # 如果知识库服务存在
        if kb is not None:
            # 获取数据库中的文件列表
            files_in_db = kb.list_files()
            # 获取文件夹中的文件列表
            files_in_folder = list_files_from_folder(kb_name)
            # 找到数据库中存在但文件夹中不存在的文件
            files = list(set(files_in_db) - set(files_in_folder))
            # 将文件转换为知识库文件对象
            kb_files = file_to_kbfile(kb_name, files)
            # 遍历需要删除的知识库文件
            for kb_file in kb_files:
                # 删除文档
                kb.delete_doc(kb_file, not_refresh_vs_cache=True)
                # 打印删除成功的信息
                print(f"success to delete docs for file: {kb_name}/{kb_file.filename}")
            # 保存向量存储
            kb.save_vector_store()

# 删除本地文件夹中不存在的文档文件，用于通过删除未使用的文档文件释放本地磁盘空间
def prune_folder_files(kb_names: List[str]):
    # 遍历知识库名称列表
    for kb_name in kb_names:
        # 根据知识库名称获取知识库服务
        kb = KBServiceFactory.get_service_by_name(kb_name)
        # 如果知识库服务存在
        if kb is not None:
            # 获取数据库中的文件列表
            files_in_db = kb.list_files()
            # 获取文件夹中的文件列表
            files_in_folder = list_files_from_folder(kb_name)
            # 找到文件夹中存在但数据库中不存在的文件
            files = list(set(files_in_folder) - set(files_in_db))
            # 遍历需要删除的文件
            for file in files:
                # 删除文件
                os.remove(get_file_path(kb_name, file))
                # 打印删除成功的信息
                print(f"success to delete file: {kb_name}/{file}")
```