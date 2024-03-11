# `.\Langchain-Chatchat\init_database.py`

```
# 导入 sys 模块
import sys
# 将当前目录添加到 sys.path 中
sys.path.append(".")
# 从指定路径导入模块
from server.knowledge_base.migrate import (create_tables, reset_tables, import_from_db,
                                           folder2db, prune_db_docs, prune_folder_files)
# 从配置文件中导入 NLTK_DATA_PATH 和 EMBEDDING_MODEL
from configs.model_config import NLTK_DATA_PATH, EMBEDDING_MODEL
# 导入 nltk 模块
import nltk
# 设置 nltk.data.path 为指定路径
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
# 导入 datetime 模块
from datetime import datetime

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 导入 argparse 模块
    import argparse
    
    # 创建 ArgumentParser 对象，设置描述信息
    parser = argparse.ArgumentParser(description="please specify only one operate method once time.")

    # 添加参数选项 -r/--recreate-vs，设置 action 为 store_true，帮助信息为重新创建向量存储
    parser.add_argument(
        "-r",
        "--recreate-vs",
        action="store_true",
        help=('''
            recreate vector store.
            use this option if you have copied document files to the content folder, but vector store has not been populated or DEFAUL_VS_TYPE/EMBEDDING_MODEL changed.
            '''
        )
    )
    # 添加参数选项 --create-tables，设置 action 为 store_true，帮助信息为如果不存在则创建空表
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help=("create empty tables if not existed")
    )
    # 添加参数选项 --clear-tables，设置 action 为 store_true，帮助信息为如果不存在则创建空表，或在重新创建向量存储之前删除数据库表
    parser.add_argument(
        "--clear-tables",
        action="store_true",
        help=("create empty tables, or drop the database tables before recreate vector stores")
    )
    # 添加参数选项 --import-db，帮助信息为从指定的 SQLite 数据库导入表
    parser.add_argument(
        "--import-db",
        help="import tables from specified sqlite database"
    )
    # 添加参数选项 -u/--update-in-db，设置 action 为 store_true，帮助信息为为数据库中存在的文件更新向量存储
    parser.add_argument(
        "-u",
        "--update-in-db",
        action="store_true",
        help=('''
            update vector store for files exist in database.
            use this option if you want to recreate vectors for files exist in db and skip files exist in local folder only.
            '''
        )
    )
    # 添加参数选项 -i/--increment，设置 action 为 store_true，帮助信息为为本地文件中存在但数据库中不存在的文件更新向量存储
    parser.add_argument(
        "-i",
        "--increment",
        action="store_true",
        help=('''
            update vector store for files exist in local folder and not exist in database.
            use this option if you want to create vectors incrementally.
            '''
        )
    )
    # 添加一个命令行参数，用于删除数据库中不存在于本地文件夹中的文档
    parser.add_argument(
        "--prune-db",
        action="store_true",
        help=('''
            delete docs in database that not existed in local folder.
            it is used to delete database docs after user deleted some doc files in file browser
            '''
        )
    )
    
    # 添加一个命令行参数，用于删除本地文件夹中不存在于数据库中的文档文件
    parser.add_argument(
        "--prune-folder",
        action="store_true",
        help=('''
            delete doc files in local folder that not existed in database.
            is is used to free local disk space by delete unused doc files.
            '''
        )
    )
    
    # 添加一个命令行参数，用于指定要操作的知识库名称
    parser.add_argument(
        "-n",
        "--kb-name",
        type=str,
        nargs="+",
        default=[],
        help=("specify knowledge base names to operate on. default is all folders exist in KB_ROOT_PATH.")
    )
    
    # 添加一个命令行参数，用于指定嵌入模型
    parser.add_argument(
        "-e",
        "--embed-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=("specify embeddings model.")
    )

    # 解析命令行参数
    args = parser.parse_args()
    start_time = datetime.now()

    # 如果指定了创建表格的参数，则确认表格存在
    if args.create_tables:
        create_tables() # confirm tables exist

    # 如果指定了清空表格的参数，则重置表格
    if args.clear_tables:
        reset_tables()
        print("database tables reset")

    # 如果指定了重新创建向量存储的参数
    if args.recreate_vs:
        create_tables()
        print("recreating all vector stores")
        folder2db(kb_names=args.kb_name, mode="recreate_vs", embed_model=args.embed_model)
    # 如果指定了从数据库导入的参数
    elif args.import_db:
        import_from_db(args.import_db)
    # 如果指定了在数据库中更新的参数
    elif args.update_in_db:
        folder2db(kb_names=args.kb_name, mode="update_in_db", embed_model=args.embed_model)
    # 如果指定了增量更新的参数
    elif args.increment:
        folder2db(kb_names=args.kb_name, mode="increment", embed_model=args.embed_model)
    # 如果指定了删除数据库中文档的参数
    elif args.prune_db:
        prune_db_docs(args.kb_name)
    # 如果指定了删除本地文件夹中文件的参数
    elif args.prune_folder:
        prune_folder_files(args.kb_name)

    end_time = datetime.now()
    print(f"总计用时： {end_time-start_time}")
```