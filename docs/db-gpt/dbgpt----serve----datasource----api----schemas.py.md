# `.\DB-GPT-src\dbgpt\serve\datasource\api\schemas.py`

```py
from typing import Optional

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field

from ..config import SERVE_APP_NAME_HUMP


class DatasourceServeRequest(BaseModel):
    """数据源服务请求模型"""

    """name: knowledge space name"""  # 数据源名称，知识空间名称

    """vector_type: vector type"""  # 向量类型，矢量类型
    id: Optional[int] = Field(None, description="The datasource id")  # 数据源ID，可选整数类型
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")  # 数据库类型，例如 sqlite, mysql 等
    db_name: str = Field(..., description="Database name.")  # 数据库名称
    db_path: str = Field("", description="File path for file-based database.")  # 基于文件的数据库文件路径
    db_host: str = Field("", description="Database host.")  # 数据库主机
    db_port: int = Field(0, description="Database port.")  # 数据库端口
    db_user: str = Field("", description="Database user.")  # 数据库用户名
    db_pwd: str = Field("", description="Database password.")  # 数据库密码
    comment: str = Field("", description="Comment for the database.")  # 数据库备注信息


class DatasourceServeResponse(BaseModel):
    """数据源服务响应模型"""

    model_config = ConfigDict(title=f"ServeResponse for {SERVE_APP_NAME_HUMP}")

    """name: knowledge space name"""  # 数据源名称，知识空间名称

    """vector_type: vector type"""  # 向量类型，矢量类型
    id: int = Field(None, description="The datasource id")  # 数据源ID，整数类型
    db_type: str = Field(..., description="Database type, e.g. sqlite, mysql, etc.")  # 数据库类型，例如 sqlite, mysql 等
    db_name: str = Field(..., description="Database name.")  # 数据库名称
    db_path: str = Field("", description="File path for file-based database.")  # 基于文件的数据库文件路径
    db_host: str = Field("", description="Database host.")  # 数据库主机
    db_port: int = Field(0, description="Database port.")  # 数据库端口
    db_user: str = Field("", description="Database user.")  # 数据库用户名
    db_pwd: str = Field("", description="Database password.")  # 数据库密码
    comment: str = Field("", description="Comment for the database.")  # 数据库备注信息
```