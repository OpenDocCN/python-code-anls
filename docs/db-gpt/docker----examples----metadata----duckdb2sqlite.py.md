# `.\DB-GPT-src\docker\examples\metadata\duckdb2sqlite.py`

```py
# 导入sqlite3模块，用于操作SQLite数据库
import sqlite3

# 导入duckdb模块，用于操作DuckDB数据库
import duckdb

# 定义一个字典，将DuckDB文件与表名映射起来
duckdb_files_to_tables = {
    "pilot/message/chat_history.db": "chat_history",
    "pilot/message/connect_config.db": "connect_config",
}

# SQLite数据库文件的路径
sqlite_db_path = "pilot/meta_data/dbgpt.db"

# 连接到SQLite数据库
conn_sqlite = sqlite3.connect(sqlite_db_path)


# 定义函数，用于迁移数据从DuckDB到SQLite
def migrate_table(duckdb_file_path, source_table, destination_table, conn_sqlite):
    # 连接到DuckDB数据库
    conn_duckdb = duckdb.connect(duckdb_file_path)
    try:
        # 获取DuckDB游标
        cursor_duckdb = conn_duckdb.cursor()
        
        # 查询源表的所有数据
        cursor_duckdb.execute(f"SELECT * FROM {source_table}")
        
        # 获取列名列表，忽略名称为"id"的列
        column_names = [
            desc[0] for desc in cursor_duckdb.description if desc[0].lower() != "id"
        ]
        
        # 构建SELECT语句中的列列表
        select_columns = ", ".join(column_names)

        # 执行带有指定列的SELECT查询
        cursor_duckdb.execute(f"SELECT {select_columns} FROM {source_table}")
        
        # 获取所有查询结果
        results = cursor_duckdb.fetchall()

        # 获取SQLite数据库游标
        cursor_sqlite = conn_sqlite.cursor()
        
        # 遍历结果集，逐行插入到SQLite目标表中
        for row in results:
            # 创建占位符列表，用于构建INSERT语句
            placeholders = ", ".join(["?"] * len(row))
            insert_query = f"INSERT INTO {destination_table} ({', '.join(column_names)}) VALUES ({placeholders})"
            # 执行INSERT语句，插入当前行数据
            cursor_sqlite.execute(insert_query, row)
        
        # 提交事务
        conn_sqlite.commit()
        
        # 关闭SQLite游标
        cursor_sqlite.close()
    
    finally:
        # 关闭DuckDB连接
        conn_duckdb.close()


try:
    # 遍历需要迁移的DuckDB文件和对应的表名
    for duckdb_file, table in duckdb_files_to_tables.items():
        # 打印迁移信息：正在从DuckDB迁移表到SQLite
        print(f"Migrating table {table} from {duckdb_file} to SQLite...")
        # 调用迁移函数，将数据从DuckDB迁移到SQLite
        migrate_table(duckdb_file, table, table, conn_sqlite)
        # 打印迁移成功信息
        print(f"Table {table} migrated to SQLite successfully.")

finally:
    # 最终关闭SQLite连接
    conn_sqlite.close()

# 打印迁移完成信息
print("Migration to SQLite completed.")
```