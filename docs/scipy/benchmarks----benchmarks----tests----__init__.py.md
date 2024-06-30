# `D:\src\scipysrc\scipy\benchmarks\benchmarks\tests\__init__.py`

```
# 导入Python标准库中的sqlite3模块，用于访问SQLite数据库
import sqlite3

# 创建或连接到名为example.db的SQLite数据库，如果不存在则会被创建
conn = sqlite3.connect('example.db')

# 创建一个用于执行SQL命令的游标对象
cursor = conn.cursor()

# 使用游标对象执行SQL命令，创建一个名为users的表，该表包含id（整数）和name（文本）两列
cursor.execute('''CREATE TABLE IF NOT EXISTS users
                (id INT PRIMARY KEY     NOT NULL,
                 name           TEXT    NOT NULL);''')

# 在users表中插入一条数据，将id设为1，name设为'John'
cursor.execute("INSERT INTO users (id, name) VALUES (1, 'John')")

# 提交对数据库的修改，确保数据持久化到数据库文件
conn.commit()

# 关闭游标对象
cursor.close()

# 关闭数据库连接
conn.close()
```