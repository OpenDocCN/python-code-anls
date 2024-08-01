# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\90912e1b-91352761084b91ff.js`

```py
# 导入Python标准库中的sqlite3模块，用于操作SQLite数据库
import sqlite3

# 连接到指定路径的SQLite数据库文件，如果文件不存在则会被创建
conn = sqlite3.connect('example.db')

# 创建一个游标对象，用于执行SQL语句和操作数据库内容
cur = conn.cursor()

# 使用游标对象执行一个SQL语句，创建一个名为"stocks"的表格，包含"id"和"name"两列
cur.execute('''CREATE TABLE stocks
               (id INT PRIMARY KEY, name TEXT)''')

# 向表格中插入一行数据，插入的数据为(1, 'AAPL')，其中1为id，'AAPL'为name
cur.execute("INSERT INTO stocks VALUES (1, 'AAPL')")

# 提交事务，将刚刚的数据库操作结果保存到数据库中
conn.commit()

# 关闭游标对象
cur.close()

# 关闭与数据库的连接
conn.close()
```