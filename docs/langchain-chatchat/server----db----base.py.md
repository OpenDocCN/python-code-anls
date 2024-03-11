# `.\Langchain-Chatchat\server\db\base.py`

```py
# 导入所需的模块
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker

# 从 configs 模块中导入 SQLALCHEMY_DATABASE_URI 变量
from configs import SQLALCHEMY_DATABASE_URI
import json

# 创建数据库引擎，使用 SQLALCHEMY_DATABASE_URI 连接数据库，并设置 json 序列化函数
engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

# 创建一个会话工厂，用于创建数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建一个基类，用于声明数据库模型
Base: DeclarativeMeta = declarative_base()
```