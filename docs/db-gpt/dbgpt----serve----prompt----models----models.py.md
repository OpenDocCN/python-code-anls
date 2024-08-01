# `.\DB-GPT-src\dbgpt\serve\prompt\models\models.py`

```py
"""This is an auto-generated model file
You can define your own models and DAOs here
"""

# 引入必要的模块和类
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
from typing import Any, Dict, Union  # 导入类型相关的模块

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, UniqueConstraint
# 从 sqlalchemy 中导入必要的列类型和约束

from dbgpt._private.pydantic import model_to_dict  # 导入模型转字典函数
from dbgpt.storage.metadata import BaseDao, Model, db  # 导入基础 DAO 类、模型类和数据库连接

from ..api.schemas import ServeRequest, ServerResponse  # 导入服务请求和响应的模型
from ..config import SERVER_APP_TABLE_NAME, ServeConfig  # 导入服务器配置和表名

# 定义 ServeEntity 类，继承自 Model 类，表示具体的服务实体
class ServeEntity(Model):
    __tablename__ = "prompt_manage"  # 数据库表名为 prompt_manage
    __table_args__ = (
        UniqueConstraint(
            "prompt_name",
            "sys_code",
            "prompt_language",
            "model",
            name="uk_prompt_name_sys_code",  # 定义唯一约束的名称
        ),
    )
    id = Column(Integer, primary_key=True, comment="Auto increment id")  # 主键 id，自增

    chat_scene = Column(String(100), comment="Chat scene")  # 聊天场景
    sub_chat_scene = Column(String(100), comment="Sub chat scene")  # 子聊天场景
    prompt_type = Column(String(100), comment="Prompt type(eg: common, private)")  # 提示类型
    prompt_name = Column(String(256), comment="Prompt name")  # 提示名称
    content = Column(Text, comment="Prompt content")  # 提示内容
    input_variables = Column(
        String(1024), nullable=True, comment="Prompt input variables(split by comma))"  # 输入变量，以逗号分隔
    )
    model = Column(
        String(128),
        nullable=True,
        comment="Prompt model name(we can use different models for different prompt",  # 提示模型名称
    )
    prompt_language = Column(
        String(32), index=True, nullable=True, comment="Prompt language(eg:en, zh-cn)"  # 提示语言，支持索引
    )
    prompt_format = Column(
        String(32),
        index=True,
        nullable=True,
        default="f-string",
        comment="Prompt format(eg: f-string, jinja2)",  # 提示格式，默认为 f-string 或 jinja2
    )
    prompt_desc = Column(String(512), nullable=True, comment="Prompt description")  # 提示描述
    user_name = Column(String(128), index=True, nullable=True, comment="User name")  # 用户名，支持索引
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")  # 系统代码，支持索引
    gmt_created = Column(DateTime, default=datetime.now, comment="Record creation time")  # 记录创建时间
    gmt_modified = Column(DateTime, default=datetime.now, comment="Record update time")  # 记录更新时间

    def __repr__(self):
        return (
            f"ServeEntity(id={self.id}, chat_scene='{self.chat_scene}', sub_chat_scene='{self.sub_chat_scene}', "
            f"prompt_type='{self.prompt_type}', prompt_name='{self.prompt_name}', content='{self.content}',"
            f"user_name='{self.user_name}', gmt_created='{self.gmt_created}', gmt_modified='{self.gmt_modified}')"
        )
        # 返回对象的字符串表示形式，用于调试和日志记录


class ServeDao(BaseDao[ServeEntity, ServeRequest, ServerResponse]):
    """The DAO class for Prompt"""
    # Prompt 的 DAO 类，用于处理 Prompt 相关的数据库操作

    def __init__(self, serve_config: ServeConfig):
        super().__init__()  # 调用父类的构造函数
        self._serve_config = serve_config  # 保存服务配置信息
    def from_request(self, request: Union[ServeRequest, Dict[str, Any]]) -> ServeEntity:
        """将请求转换为实体对象

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): 请求对象

        Returns:
            ServeEntity: 转换后的实体对象
        """
        # 如果请求是 ServeRequest 类型，则将其转换为字典
        request_dict = (
            model_to_dict(request) if isinstance(request, ServeRequest) else request
        )
        # 使用字典创建 ServeEntity 对象
        entity = ServeEntity(**request_dict)
        # 返回转换后的实体对象
        return entity

    def to_request(self, entity: ServeEntity) -> ServeRequest:
        """将实体对象转换为请求对象

        Args:
            entity (ServeEntity): 实体对象

        Returns:
            ServeRequest: 转换后的请求对象
        """
        return ServeRequest(
            chat_scene=entity.chat_scene,
            sub_chat_scene=entity.sub_chat_scene,
            prompt_type=entity.prompt_type,
            prompt_name=entity.prompt_name,
            content=entity.content,
            prompt_desc=entity.prompt_desc,
            user_name=entity.user_name,
            sys_code=entity.sys_code,
        )

    def to_response(self, entity: ServeEntity) -> ServerResponse:
        """将实体对象转换为响应对象

        Args:
            entity (ServeEntity): 实体对象

        Returns:
            ServerResponse: 转换后的响应对象
        """
        # TODO implement your own logic here, transfer the entity to a response
        # 将实体对象中的时间字段转换为字符串格式
        gmt_created_str = entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S")
        gmt_modified_str = entity.gmt_modified.strftime("%Y-%m-%d %H:%M:%S")
        # 使用转换后的时间字符串创建 ServerResponse 对象
        return ServerResponse(
            id=entity.id,
            chat_scene=entity.chat_scene,
            sub_chat_scene=entity.sub_chat_scene,
            prompt_type=entity.prompt_type,
            prompt_name=entity.prompt_name,
            content=entity.content,
            prompt_desc=entity.prompt_desc,
            user_name=entity.user_name,
            sys_code=entity.sys_code,
            gmt_created=gmt_created_str,
            gmt_modified=gmt_modified_str,
        )
```