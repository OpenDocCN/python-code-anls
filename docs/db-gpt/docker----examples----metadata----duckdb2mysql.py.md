# `.\DB-GPT-src\docker\examples\metadata\duckdb2mysql.py`

```py
import duckdb
import pymysql

""" migrate duckdb to mysql"""

# MySQL数据库配置信息
mysql_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "your_password",
    "db": "dbgpt",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

# 指定DuckDB文件到MySQL表的映射关系
duckdb_files_to_tables = {
    "pilot/message/chat_history.db": "chat_history",
    "pilot/message/connect_config.db": "connect_config",
}

# 连接MySQL数据库
conn_mysql = pymysql.connect(**mysql_config)


def migrate_table(duckdb_file_path, source_table, destination_table, conn_mysql):
    # 连接到指定的DuckDB文件
    conn_duckdb = duckdb.connect(duckdb_file_path)
    try:
        cursor = conn_duckdb.cursor()
        # 执行SQL查询，获取源表的列名，去除ID列（如果存在）
        cursor.execute(f"SELECT * FROM {source_table}")
        column_names = [
            desc[0] for desc in cursor.description if desc[0].lower() != "id"
        ]
        select_columns = ", ".join(column_names)

        # 执行SQL查询，从源表中获取所有数据行
        cursor.execute(f"SELECT {select_columns} FROM {source_table}")
        results = cursor.fetchall()

        # 使用MySQL连接的游标，逐行将数据插入目标表中
        with conn_mysql.cursor() as cursor_mysql:
            for row in results:
                placeholders = ", ".join(["%s"] * len(row))
                insert_query = f"INSERT INTO {destination_table} ({', '.join(column_names)}) VALUES ({placeholders})"
                cursor_mysql.execute(insert_query, row)
        # 提交事务
        conn_mysql.commit()
    finally:
        # 关闭DuckDB连接
        conn_duckdb.close()


try:
    # 遍历每个DuckDB文件及其对应的目标表名，进行迁移操作
    for duckdb_file, table in duckdb_files_to_tables.items():
        print(f"Migrating table {table} from {duckdb_file}...")
        migrate_table(duckdb_file, table, table, conn_mysql)
        print(f"Table {table} migrated successfully.")
finally:
    # 关闭MySQL连接
    conn_mysql.close()

# 迁移过程结束
print("Migration completed.")
```