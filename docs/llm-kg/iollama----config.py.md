# `.\iollama\config.py`

```
# 导入操作系统模块
import os

# 定义初始化索引的布尔值，根据环境变量 'INIT_INDEX' 是否为 'true' 来决定
INIT_INDEX = os.getenv('INIT_INDEX', 'false').lower() == 'true'

# 索引持久化目录的路径，如果环境变量 'INDEX_PERSIST_DIRECTORY' 不存在，则使用默认路径 "./data/chromadb"
INDEX_PERSIST_DIRECTORY = os.getenv('INDEX_PERSIST_DIRECTORY', "./data/chromadb")

# 要抓取的目标 URL 地址，如果环境变量 'TARGET_URL' 不存在，则使用默认地址 "https://open5gs.org/open5gs/docs/"
TARGET_URL =  os.getenv('TARGET_URL', "https://open5gs.org/open5gs/docs/")

# HTTP API 的端口号，如果环境变量 'HTTP_PORT' 不存在，则使用默认端口 7654
HTTP_PORT = os.getenv('HTTP_PORT', 7654)

# MongoDB 数据库的主机地址，如果环境变量 'MONGO_HOST' 不存在，则使用默认地址 'localhost'
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')

# MongoDB 数据库的端口号，如果环境变量 'MONGO_PORT' 不存在，则使用默认端口 27017
MONGO_PORT = os.getenv('MONGO_PORT', 27017)

# MongoDB 数据库的用户名，如果环境变量 'MONGO_USER' 不存在，则使用默认用户名 'testuser'
MONGO_USER = os.getenv('MONGO_USER', 'testuser')

# MongoDB 数据库的密码，如果环境变量 'MONGO_PASS' 不存在，则使用默认密码 'testpass'
MONGO_PASS = os.getenv('MONGO_PASS', 'testpass')
```