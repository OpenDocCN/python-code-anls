# `.\DB-GPT-src\dbgpt\serve\flow\models\models.py`

```py
"""
This is an auto-generated model file
You can define your own models and DAOs here
"""

# 导入必要的库
import json  # 导入用于处理 JSON 格式的模块
from datetime import datetime  # 导入处理日期时间的模块
from typing import Any, Dict, Union  # 导入用于类型提示的模块

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint  # 导入 SQLAlchemy 中的表和列定义
from dbgpt._private.pydantic import model_to_dict  # 导入用于将模型转换为字典的函数
from dbgpt.core.awel.flow.flow_factory import State  # 导入流程工厂中的状态类
from dbgpt.storage.metadata import BaseDao, Model  # 导入基础 DAO 和模型类
from dbgpt.storage.metadata._base_dao import QUERY_SPEC  # 导入查询规范

from ..api.schemas import ServeRequest, ServerResponse  # 导入服务请求和服务响应的数据模型
from ..config import SERVER_APP_TABLE_NAME, ServeConfig  # 导入服务器应用表名称和服务配置


class ServeEntity(Model):
    __tablename__ = SERVER_APP_TABLE_NAME  # 定义数据库表名
    __table_args__ = (UniqueConstraint("uid", name="uk_uid"),)  # 定义表的唯一约束条件

    id = Column(Integer, primary_key=True, comment="Auto increment id")  # 定义自增主键 id
    uid = Column(String(128), index=True, nullable=False, comment="Unique id")  # 定义唯一标识 uid
    dag_id = Column(String(128), index=True, nullable=True, comment="DAG id")  # 定义 DAG id
    label = Column(String(128), nullable=True, comment="Flow label")  # 定义流程标签
    name = Column(String(128), index=True, nullable=True, comment="Flow name")  # 定义流程名称
    flow_category = Column(String(64), nullable=True, comment="Flow category")  # 定义流程类别
    flow_data = Column(Text, nullable=True, comment="Flow data, JSON format")  # 定义流程数据，JSON 格式
    description = Column(String(512), nullable=True, comment="Flow description")  # 定义流程描述
    state = Column(String(32), nullable=True, comment="Flow state")  # 定义流程状态
    error_message = Column(String(512), nullable=True, comment="Error message")  # 定义错误消息
    source = Column(String(64), nullable=True, comment="Flow source")  # 定义流程来源
    source_url = Column(String(512), nullable=True, comment="Flow source url")  # 定义流程来源 URL
    version = Column(String(32), nullable=True, comment="Flow version")  # 定义流程版本
    define_type = Column(
        String(32),
        default="json",
        nullable=True,
        comment="Flow define type(json or python)",
    )  # 定义流程定义类型，默认为 JSON 或 Python
    editable = Column(
        Integer, nullable=True, comment="Editable, 0: editable, 1: not editable"
    )  # 定义是否可编辑，0 表示可编辑，1 表示不可编辑
    user_name = Column(String(128), index=True, nullable=True, comment="User name")  # 定义用户名
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")  # 定义系统代码
    gmt_created = Column(DateTime, default=datetime.now, comment="Record creation time")  # 定义记录创建时间
    gmt_modified = Column(DateTime, default=datetime.now, comment="Record update time")  # 定义记录更新时间

    def __repr__(self):
        return (
            f"ServeEntity(id={self.id}, uid={self.uid}, dag_id={self.dag_id}, name={self.name}, "
            f"flow_data={self.flow_data}, user_name={self.user_name}, "
            f"sys_code={self.sys_code}, gmt_created={self.gmt_created}, "
            f"gmt_modified={self.gmt_modified})"
        )  # 定义模型对象的字符串表示形式

    @classmethod
    # 解析可编辑属性，将不同类型的输入转换为整数表示
    def parse_editable(cls, editable: Any) -> int:
        """Parse editable"""
        # 如果 editable 是 None，则返回 0
        if editable is None:
            return 0
        # 如果 editable 是布尔类型，返回 0（True）或 1（False）
        if isinstance(editable, bool):
            return 0 if editable else 1
        # 如果 editable 是整数类型，返回 0（True）或 1（False），取决于其是否为 0
        elif isinstance(editable, int):
            return 0 if editable == 0 else 1
        else:
            # 如果 editable 类型不是上述类型，则引发 ValueError 异常
            raise ValueError(f"Invalid editable: {editable}")

    @classmethod
    # 将整数表示的 editable 转换为布尔类型
    def to_bool_editable(cls, editable: int) -> bool:
        """Convert editable to bool"""
        # 如果 editable 是 None 或者等于 0，则返回 True，否则返回 False
        return editable is None or editable == 0
class ServeDao(BaseDao[ServeEntity, ServeRequest, ServerResponse]):
    """The DAO class for Flow"""

    def __init__(self, serve_config: ServeConfig):
        super().__init__()
        self._serve_config = serve_config

    def from_request(self, request: Union[ServeRequest, Dict[str, Any]]) -> ServeEntity:
        """Convert the request to an entity

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): The request

        Returns:
            T: The entity
        """
        # 将请求转换为字典形式
        request_dict = (
            model_to_dict(request) if isinstance(request, ServeRequest) else request
        )
        # 将流数据部分转换为 JSON 字符串
        flow_data = json.dumps(request_dict.get("flow_data"), ensure_ascii=False)
        # 获取状态，默认为初始化中
        state = request_dict.get("state", State.INITIALIZING.value)
        # 获取错误消息，并限制在500字符以内
        error_message = request_dict.get("error_message")
        if error_message:
            error_message = error_message[:500]
        # 构建新的字典对象，包含请求的各个字段
        new_dict = {
            "uid": request_dict.get("uid"),
            "dag_id": request_dict.get("dag_id"),
            "label": request_dict.get("label"),
            "name": request_dict.get("name"),
            "flow_category": request_dict.get("flow_category"),
            "flow_data": flow_data,
            "state": state,
            "error_message": error_message,
            "source": request_dict.get("source"),
            "source_url": request_dict.get("source_url"),
            "version": request_dict.get("version"),
            "define_type": request_dict.get("define_type"),
            "editable": ServeEntity.parse_editable(request_dict.get("editable")),
            "description": request_dict.get("description"),
            "user_name": request_dict.get("user_name"),
            "sys_code": request_dict.get("sys_code"),
        }
        # 使用新字典对象创建 ServeEntity 实例
        entity = ServeEntity(**new_dict)
        return entity

    def to_request(self, entity: ServeEntity) -> ServeRequest:
        """Convert the entity to a request

        Args:
            entity (T): The entity

        Returns:
            REQ: The request
        """
        # 将实体的流数据字段解析为 JSON 对象
        flow_data = json.loads(entity.flow_data)
        # 构建并返回 ServeRequest 实例，包含实体的各个字段
        return ServeRequest(
            uid=entity.uid,
            dag_id=entity.dag_id,
            label=entity.label,
            name=entity.name,
            flow_category=entity.flow_category,
            flow_data=flow_data,
            state=State.value_of(entity.state),
            error_message=entity.error_message,
            source=entity.source,
            source_url=entity.source_url,
            version=entity.version,
            define_type=entity.define_type,
            editable=ServeEntity.to_bool_editable(entity.editable),
            description=entity.description,
            user_name=entity.user_name,
            sys_code=entity.sys_code,
        )
    def to_response(self, entity: ServeEntity) -> ServerResponse:
        """Convert the entity to a response

        Args:
            entity (T): The entity

        Returns:
            RES: The response
        """
        # 将实体的流数据解析为 JSON 格式
        flow_data = json.loads(entity.flow_data)
        # 将实体的创建时间格式化为字符串
        gmt_created_str = entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S")
        # 将实体的修改时间格式化为字符串
        gmt_modified_str = entity.gmt_modified.strftime("%Y-%m-%d %H:%M:%S")
        # 返回一个 ServerResponse 对象，包含从实体属性映射得到的信息
        return ServerResponse(
            uid=entity.uid,
            dag_id=entity.dag_id,
            label=entity.label,
            name=entity.name,
            flow_category=entity.flow_category,
            flow_data=flow_data,
            description=entity.description,
            state=State.value_of(entity.state),
            error_message=entity.error_message,
            source=entity.source,
            source_url=entity.source_url,
            version=entity.version,
            editable=ServeEntity.to_bool_editable(entity.editable),
            define_type=entity.define_type,
            user_name=entity.user_name,
            sys_code=entity.sys_code,
            gmt_created=gmt_created_str,
            gmt_modified=gmt_modified_str,
        )

    def update(
        self, query_request: QUERY_SPEC, update_request: ServeRequest
        # 下面的代码在这里继续
    ) -> ServerResponse:
        # 使用自身的会话管理器，确保在操作中可以回滚事务
        with self.session(commit=False) as session:
            # 根据查询请求创建查询对象
            query = self._create_query_object(session, query_request)
            # 查询并返回第一个结果实体
            entry: ServeEntity = query.first()
            # 如果查询结果为空，则抛出异常
            if entry is None:
                raise Exception("Invalid request")
            # 如果有更新请求中的标签，则更新实体的标签属性
            if update_request.label:
                entry.label = update_request.label
            # 如果有更新请求中的名称，则更新实体的名称属性
            if update_request.name:
                entry.name = update_request.name
            # 如果有更新请求中的流程类别，则更新实体的流程类别属性
            if update_request.flow_category:
                entry.flow_category = update_request.flow_category
            # 如果有更新请求中的流程数据，则将其转换为 JSON 字符串并更新实体的流程数据属性
            if update_request.flow_data:
                entry.flow_data = json.dumps(
                    model_to_dict(update_request.flow_data), ensure_ascii=False
                )
            # 如果有更新请求中的描述，则更新实体的描述属性
            if update_request.description:
                entry.description = update_request.description
            # 如果有更新请求中的状态，则更新实体的状态属性
            if update_request.state:
                entry.state = update_request.state.value
            # 如果更新请求中的错误消息不为空，则只保留前 500 个字符并更新实体的错误消息属性
            if update_request.error_message is not None:
                # 保留前 500 个字符
                entry.error_message = update_request.error_message[:500]
            # 如果有更新请求中的来源，则更新实体的来源属性
            if update_request.source:
                entry.source = update_request.source
            # 如果有更新请求中的来源 URL，则更新实体的来源 URL 属性
            if update_request.source_url:
                entry.source_url = update_request.source_url
            # 如果有更新请求中的版本号，则更新实体的版本号属性
            if update_request.version:
                entry.version = update_request.version
            # 解析更新请求中的可编辑属性并更新实体的可编辑状态
            entry.editable = ServeEntity.parse_editable(update_request.editable)
            # 如果有更新请求中的定义类型，则更新实体的定义类型属性
            if update_request.define_type:
                entry.define_type = update_request.define_type
            # 如果有更新请求中的用户名，则更新实体的用户名属性
            if update_request.user_name:
                entry.user_name = update_request.user_name
            # 如果有更新请求中的系统代码，则更新实体的系统代码属性
            if update_request.sys_code:
                entry.sys_code = update_request.sys_code
            # 合并更新后的实体到会话中
            session.merge(entry)
            # 提交会话中的所有更改
            session.commit()
            # 返回更新后的查询结果
            return self.get_one(query_request)
```