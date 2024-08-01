# `.\DB-GPT-src\dbgpt\datasource\db_conn_info.py`

```py
"""Configuration for database connection."""
# 引入需要的库
from dbgpt._private.pydantic import BaseModel, Field

# 定义数据库连接配置的模型
class DBConfig(BaseModel):
    """Database connection configuration."""

    # 数据库类型，例如 sqlite、mysql 等
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")
    # 数据库名称
    db_name: str = Field(..., description="Database name.")
    # 文件型数据库的文件路径
    file_path: str = Field("", description="File path for file-based database.")
    # 数据库主机
    db_host: str = Field("", description="Database host.")
    # 数据库端口
    db_port: int = Field(0, description="Database port.")
    # 数据库用户名
    db_user: str = Field("", description="Database user.")
    # 数据库密码
    db_pwd: str = Field("", description="Database password.")
    # 数据库的注释信息
    comment: str = Field("", description="Comment for the database.")


# 定义数据库类型信息的模型
class DbTypeInfo(BaseModel):
    """Database type information."""

    # 数据库类型
    db_type: str = Field(..., description="Database type.")
    # 是否为文件型数据库
    is_file_db: bool = Field(False, description="Whether the database is file-based.")
```