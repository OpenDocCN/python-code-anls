# `.\DB-GPT-src\dbgpt\serve\agent\db\gpts_app.py`

```py
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import uuid  # 导入生成唯一标识符的模块
from datetime import datetime  # 导入处理日期时间的模块
from itertools import groupby  # 导入用于迭代操作的模块中的 groupby 函数
from typing import Any, Dict, List, Optional, Union  # 导入用于类型注解的模块

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint  # 导入 SQLAlchemy 中的数据库列类型

from dbgpt._private.pydantic import (  # 导入用于数据验证和序列化的 Pydantic 相关模块
    BaseModel,
    ConfigDict,
    Field,
    model_to_json,
    model_validator,
)
from dbgpt.agent.core.plan import AWELTeamContext  # 导入用于代理计划的上下文类
from dbgpt.agent.resource.base import AgentResource  # 导入代理资源基类
from dbgpt.serve.agent.team.base import TeamMode  # 导入团队模式相关基类
from dbgpt.storage.metadata import BaseDao, Model  # 导入存储元数据相关类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class GptsAppDetail(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义模型配置字典，允许任意类型

    app_code: Optional[str] = None  # 应用程序代码，可选字符串，默认为 None
    app_name: Optional[str] = None  # 应用程序名称，可选字符串，默认为 None
    agent_name: Optional[str] = None  # 代理名称，可选字符串，默认为 None
    node_id: Optional[str] = None  # 节点 ID，可选字符串，默认为 None
    resources: Optional[list[AgentResource]] = None  # 资源列表，可选的代理资源对象列表，默认为 None
    prompt_template: Optional[str] = None  # 提示模板，可选字符串，默认为 None
    llm_strategy: Optional[str] = None  # LLM 策略，可选字符串，默认为 None
    llm_strategy_value: Optional[str] = None  # LLM 策略值，可选字符串，默认为 None
    created_at: datetime = datetime.now()  # 创建时间，默认为当前时间
    updated_at: datetime = datetime.now()  # 更新时间，默认为当前时间

    def to_dict(self):
        return {k: self._serialize(v) for k, v in self.__dict__.items()}  # 将对象转换为字典形式

    def _serialize(self, value):
        if isinstance(value, BaseModel):  # 如果值是 BaseModel 类型
            return value.to_dict()  # 递归调用 to_dict 方法
        elif isinstance(value, list):  # 如果值是列表类型
            return [self._serialize(item) for item in value]  # 递归调用 _serialize 方法
        elif isinstance(value, dict):  # 如果值是字典类型
            return {k: self._serialize(v) for k, v in value.items()}  # 递归调用 _serialize 方法
        else:
            return value  # 直接返回原始值

    @classmethod
    def from_dict(cls, d: Dict[str, Any], parse_llm_strategy: bool = False):
        lsv = d.get("llm_strategy_value")  # 获取字典中的 llm_strategy_value 值
        if parse_llm_strategy and lsv:  # 如果需要解析 LLM 策略且存在 llm_strategy_value
            strategies = json.loads(lsv)  # 将 llm_strategy_value 解析为列表
            llm_strategy_value = ",".join(strategies)  # 使用逗号连接列表中的策略
        else:
            llm_strategy_value = d.get("llm_strategy_value", None)  # 否则直接获取 llm_strategy_value 值

        return cls(
            app_code=d["app_code"],  # 设置应用程序代码
            app_name=d["app_name"],  # 设置应用程序名称
            agent_name=d["agent_name"],  # 设置代理名称
            node_id=d["node_id"],  # 设置节点 ID
            resources=AgentResource.from_json_list_str(d.get("resources", None)),  # 解析并设置资源列表
            prompt_template=d.get("prompt_template", None),  # 设置提示模板
            llm_strategy=d.get("llm_strategy", None),  # 设置LLM策略
            llm_strategy_value=llm_strategy_value,  # 设置LLM策略值
            created_at=d.get("created_at", None),  # 设置创建时间
            updated_at=d.get("updated_at", None),  # 设置更新时间
        )
    # 类方法，用于从实体对象中创建代理对象
    def from_entity(cls, entity):
        # 调用AgentResource类的类方法，从JSON列表字符串中创建资源对象列表
        resources = AgentResource.from_json_list_str(entity.resources)
        # 返回使用从实体中提取的属性创建的代理对象
        return cls(
            app_code=entity.app_code,  # 设置代理对象的应用代码
            app_name=entity.app_name,  # 设置代理对象的应用名称
            agent_name=entity.agent_name,  # 设置代理对象的代理名称
            node_id=entity.node_id,  # 设置代理对象的节点ID
            resources=resources,  # 设置代理对象的资源列表
            prompt_template=entity.prompt_template,  # 设置代理对象的提示模板
            llm_strategy=entity.llm_strategy,  # 设置代理对象的LLM策略
            llm_strategy_value=entity.llm_strategy_value,  # 设置代理对象的LLM策略值
            created_at=entity.created_at,  # 设置代理对象的创建时间
            updated_at=entity.updated_at,  # 设置代理对象的更新时间
        )
class GptsApp(BaseModel):
    # 定义一个名为 GptsApp 的类，继承自 BaseModel

    model_config = ConfigDict(arbitrary_types_allowed=True)
    # 定义一个名为 model_config 的类属性，类型为 ConfigDict，允许任意类型的值

    app_code: Optional[str] = None
    # 定义一个可选的实例属性 app_code，类型为 str，默认为 None

    app_name: Optional[str] = None
    # 定义一个可选的实例属性 app_name，类型为 str，默认为 None

    app_describe: Optional[str] = None
    # 定义一个可选的实例属性 app_describe，类型为 str，默认为 None

    team_mode: Optional[str] = None
    # 定义一个可选的实例属性 team_mode，类型为 str，默认为 None

    language: Optional[str] = None
    # 定义一个可选的实例属性 language，类型为 str，默认为 None

    team_context: Optional[Union[str, AWELTeamContext]] = None
    # 定义一个可选的实例属性 team_context，类型为 str 或 AWELTeamContext，默认为 None

    user_code: Optional[str] = None
    # 定义一个可选的实例属性 user_code，类型为 str，默认为 None

    sys_code: Optional[str] = None
    # 定义一个可选的实例属性 sys_code，类型为 str，默认为 None

    is_collected: Optional[str] = None
    # 定义一个可选的实例属性 is_collected，类型为 str，默认为 None

    icon: Optional[str] = None
    # 定义一个可选的实例属性 icon，类型为 str，默认为 None

    created_at: datetime = datetime.now()
    # 定义一个实例属性 created_at，类型为 datetime，初始值为当前时间

    updated_at: datetime = datetime.now()
    # 定义一个实例属性 updated_at，类型为 datetime，初始值为当前时间

    details: List[GptsAppDetail] = []
    # 定义一个实例属性 details，类型为 GptsAppDetail 的列表，初始为空列表

    def to_dict(self):
        # 将实例转换为字典的方法
        return {k: self._serialize(v) for k, v in self.__dict__.items()}

    def _serialize(self, value):
        # 序列化方法，根据值的类型进行递归序列化
        if isinstance(value, BaseModel):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._serialize(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        else:
            return value

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        # 从字典构造实例的类方法
        return cls(
            app_code=d.get("app_code", None),
            app_name=d["app_name"],
            language=d["language"],
            app_describe=d["app_describe"],
            team_mode=d["team_mode"],
            team_context=d.get("team_context", None),
            user_code=d.get("user_code", None),
            sys_code=d.get("sys_code", None),
            is_collected=d.get("is_collected", None),
            created_at=d.get("created_at", None),
            updated_at=d.get("updated_at", None),
            details=d.get("details", None),
        )

    @model_validator(mode="before")
    @classmethod
    def pre_fill(cls, values):
        # 数据填充前的验证方法
        if not isinstance(values, dict):
            return values
        is_collected = values.get("is_collected")
        if is_collected is not None and isinstance(is_collected, bool):
            values["is_collected"] = "true" if is_collected else "false"
        return values


class GptsAppQuery(GptsApp):
    # 定义一个名为 GptsAppQuery 的类，继承自 GptsApp 类

    page_size: int = 100
    # 定义一个实例属性 page_size，类型为 int，默认值为 100

    page_no: int = 1
    # 定义一个实例属性 page_no，类型为 int，默认值为 1


class GptsAppResponse(BaseModel):
    # 定义一个名为 GptsAppResponse 的类，继承自 BaseModel

    total_count: Optional[int] = 0
    # 定义一个可选的实例属性 total_count，类型为 int，默认值为 0

    total_page: Optional[int] = 0
    # 定义一个可选的实例属性 total_page，类型为 int，默认值为 0

    current_page: Optional[int] = 0
    # 定义一个可选的实例属性 current_page，类型为 int，默认值为 0

    app_list: Optional[List[GptsApp]] = Field(
        default_factory=list, description="app list"
    )
    # 定义一个可选的实例属性 app_list，类型为 GptsApp 的列表，默认工厂为 list，描述为 "app list"


class GptsAppCollection(BaseModel):
    # 定义一个名为 GptsAppCollection 的类，继承自 BaseModel

    app_code: Optional[str] = None
    # 定义一个可选的实例属性 app_code，类型为 str，默认值为 None

    user_code: Optional[str] = None
    # 定义一个可选的实例属性 user_code，类型为 str，默认值为 None

    sys_code: Optional[str] = None
    # 定义一个可选的实例属性 sys_code，类型为 str，默认值为 None

    def to_dict(self):
        # 将实例转换为字典的方法
        return {k: self._serialize(v) for k, v in self.__dict__.items()}

    def _serialize(self, value):
        # 序列化方法，根据值的类型进行递归序列化
        if isinstance(value, BaseModel):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._serialize(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}
        else:
            return value

    @classmethod
    # 从字典 d 中构建一个类实例的类方法
    def from_dict(cls, d: Dict[str, Any]):
        # 使用字典 d 中的键值对作为参数创建类 cls 的实例，并返回
        return cls(
            app_code=d.get("app_code", None),     # 从字典 d 中获取键为 "app_code" 的值，如果不存在则默认为 None
            user_code=d.get("user_code", None),   # 从字典 d 中获取键为 "user_code" 的值，如果不存在则默认为 None
            sys_code=d.get("sys_code", None),     # 从字典 d 中获取键为 "sys_code" 的值，如果不存在则默认为 None
            created_at=d.get("created_at", None), # 从字典 d 中获取键为 "created_at" 的值，如果不存在则默认为 None
            updated_at=d.get("updated_at", None), # 从字典 d 中获取键为 "updated_at" 的值，如果不存在则默认为 None
        )
class GptsAppCollectionEntity(Model):
    # 数据库表名
    __tablename__ = "gpts_app_collection"
    # 主键，自增长 ID
    id = Column(Integer, primary_key=True, comment="autoincrement id")
    # 当前 AI 助手代码，不能为空
    app_code = Column(String(255), nullable=False, comment="Current AI assistant code")
    # 用户代码，可为空
    user_code = Column(String(255), nullable=True, comment="user code")
    # 系统应用代码，可为空
    sys_code = Column(String(255), nullable=True, comment="system app code")
    # 创建时间，默认为当前时间
    created_at = Column(DateTime, default=datetime.utcnow, comment="create time")
    # 最后更新时间，默认为当前时间，在更新时自动更新
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="last update time",
    )


class GptsAppEntity(Model):
    # 数据库表名
    __tablename__ = "gpts_app"
    # 主键，自增长 ID
    id = Column(Integer, primary_key=True, comment="autoincrement id")
    # 当前 AI 助手代码，不能为空
    app_code = Column(String(255), nullable=False, comment="Current AI assistant code")
    # 当前 AI 助手名称，不能为空
    app_name = Column(String(255), nullable=False, comment="Current AI assistant name")
    # 应用图标的 URL，可为空
    icon = Column(String(1024), nullable=True, comment="app icon, url")
    # 当前 AI 助手的描述信息，不能为空
    app_describe = Column(
        String(2255), nullable=False, comment="Current AI assistant describe"
    )
    # 使用的语言，不能为空
    language = Column(String(100), nullable=False, comment="gpts language")
    # 团队工作模式，不能为空
    team_mode = Column(String(255), nullable=False, comment="Team work mode")
    # 团队上下文，依赖不同工作模式的执行逻辑和团队成员内容，可为空
    team_context = Column(
        Text,
        nullable=True,
        comment="The execution logic and team member content that teams with different "
        "working modes rely on",
    )
    # 用户代码，可为空
    user_code = Column(String(255), nullable=True, comment="user code")
    # 系统应用代码，可为空
    sys_code = Column(String(255), nullable=True, comment="system app code")
    # 创建时间，默认为当前时间
    created_at = Column(DateTime, default=datetime.utcnow, comment="create time")
    # 最后更新时间，默认为当前时间，在更新时自动更新
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="last update time",
    )
    # 设置表约束，确保应用名称唯一
    __table_args__ = (UniqueConstraint("app_name", name="uk_gpts_app"),)


class GptsAppDetailEntity(Model):
    # 数据库表名
    __tablename__ = "gpts_app_detail"
    # 主键，自增长 ID
    id = Column(Integer, primary_key=True, comment="autoincrement id")
    # 当前 AI 助手代码，不能为空
    app_code = Column(String(255), nullable=False, comment="Current AI assistant code")
    # 当前 AI 助手名称，不能为空
    app_name = Column(String(255), nullable=False, comment="Current AI assistant name")
    # 代理人名称，不能为空
    agent_name = Column(String(255), nullable=False, comment=" Agent name")
    # 当前 AI 助手代理节点 ID，不能为空
    node_id = Column(
        String(255), nullable=False, comment="Current AI assistant Agent Node id"
    )
    # 代理绑定资源，可为空
    resources = Column(Text, nullable=True, comment="Agent bind  resource")
    # 代理绑定模板，可为空
    prompt_template = Column(Text, nullable=True, comment="Agent bind  template")
    # 使用的语言模型策略，可为空
    llm_strategy = Column(String(25), nullable=True, comment="Agent use llm strategy")
    # 使用的语言模型策略值，可为空
    llm_strategy_value = Column(
        Text, nullable=True, comment="Agent use llm strategy value"
    )
    # 创建时间，默认为当前时间
    created_at = Column(DateTime, default=datetime.utcnow, comment="create time")
    # 定义一个名为 updated_at 的列，类型为 DateTime，其默认值为当前的 UTC 时间，
    # 在更新时也会更新为当前的 UTC 时间，用于记录最后更新时间
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="last update time",
    )

    # 定义表的特殊参数，包含一个名为 "uk_gpts_app_agent_node" 的唯一约束，
    # 约束条件为 ("app_name", "agent_name", "node_id")，用于确保这三列的组合值唯一
    __table_args__ = (
        UniqueConstraint(
            "app_name", "agent_name", "node_id", name="uk_gpts_app_agent_node"
        ),
    )

    # 将对象转换为字典格式的方法，用于序列化对象的属性
    def to_dict(self):
        return {k: self._serialize(v) for k, v in self.__dict__.items()}

    # 递归序列化属性值的内部方法
    def _serialize(self, value):
        if isinstance(value, BaseModel):
            return value.to_dict()  # 如果属性值是 BaseModel 对象，则递归调用 to_dict 方法进行序列化
        elif isinstance(value, list):
            return [self._serialize(item) for item in value]  # 如果属性值是列表，则递归调用 _serialize 方法处理列表中的每个元素
        elif isinstance(value, dict):
            return {k: self._serialize(v) for k, v in value.items()}  # 如果属性值是字典，则递归调用 _serialize 方法处理字典中的每个值
        else:
            return value  # 其他情况下直接返回属性值本身，如基本数据类型
class GptsAppCollectionDao(BaseDao):
    # GptsAppCollectionDao 类，继承自 BaseDao

    def collect(
        self,
        app_code: str,
        user_code: Optional[str] = None,
        sys_code: Optional[str] = None,
    ):
        # collect 方法，用于收藏应用信息

        with self.session() as session:
            # 使用 session() 进行上下文管理

            app_qry = session.query(GptsAppCollectionEntity)
            # 查询 GptsAppCollectionEntity 对象

            if user_code:
                app_qry = app_qry.filter(GptsAppCollectionEntity.user_code == user_code)
                # 如果指定了 user_code，则筛选出指定 user_code 的记录

            if sys_code:
                app_qry = app_qry.filter(GptsAppCollectionEntity.sys_code == sys_code)
                # 如果指定了 sys_code，则筛选出指定 sys_code 的记录

            if app_code:
                app_qry = app_qry.filter(GptsAppCollectionEntity.app_code == app_code)
                # 如果指定了 app_code，则筛选出指定 app_code 的记录

            app_entity = app_qry.one_or_none()
            # 获取匹配条件的唯一记录或者空

            if app_entity is not None:
                raise f"current app has been collected!"
                # 如果已存在匹配的应用实体，抛出已经收藏的异常

            app_entity = GptsAppCollectionEntity(
                app_code=app_code,
                user_code=user_code,
                sys_code=sys_code,
            )
            # 创建新的 GptsAppCollectionEntity 实例

            session.add(app_entity)
            # 将新实例添加到数据库会话中

    def uncollect(
        self,
        app_code: str,
        user_code: Optional[str] = None,
        sys_code: Optional[str] = None,
    ):
        # uncollect 方法，用于取消收藏应用信息

        with self.session() as session:
            # 使用 session() 进行上下文管理

            app_qry = session.query(GptsAppCollectionEntity)
            # 查询 GptsAppCollectionEntity 对象

            if user_code:
                app_qry = app_qry.filter(GptsAppCollectionEntity.user_code == user_code)
                # 如果指定了 user_code，则筛选出指定 user_code 的记录

            if sys_code:
                app_qry = app_qry.filter(GptsAppCollectionEntity.sys_code == sys_code)
                # 如果指定了 sys_code，则筛选出指定 sys_code 的记录

            if app_code:
                app_qry = app_qry.filter(GptsAppCollectionEntity.app_code == app_code)
                # 如果指定了 app_code，则筛选出指定 app_code 的记录

            app_entity = app_qry.one_or_none()
            # 获取匹配条件的唯一记录或者空

            if app_entity:
                session.delete(app_entity)
                # 如果找到匹配的应用实体，则从数据库会话中删除

                session.commit()
                # 提交数据库会话，执行删除操作

    def list(self, query: GptsAppCollection):
        # list 方法，用于查询应用收藏列表

        with self.session() as session:
            # 使用 session() 进行上下文管理

            app_qry = session.query(GptsAppCollectionEntity)
            # 查询 GptsAppCollectionEntity 对象

            if query.user_code:
                app_qry = app_qry.filter(
                    GptsAppCollectionEntity.user_code == query.user_code
                )
                # 如果查询对象中指定了 user_code，则筛选出指定 user_code 的记录

            if query.sys_code:
                app_qry = app_qry.filter(
                    GptsAppCollectionEntity.sys_code == query.sys_code
                )
                # 如果查询对象中指定了 sys_code，则筛选出指定 sys_code 的记录

            if query.app_code:
                app_qry = app_qry.filter(
                    GptsAppCollectionEntity.app_code == query.app_code
                )
                # 如果查询对象中指定了 app_code，则筛选出指定 app_code 的记录

            res = app_qry.all()
            # 获取所有匹配条件的记录列表

            session.close()
            # 关闭数据库会话

            return res
            # 返回查询结果列表


class GptsAppDao(BaseDao):
    # GptsAppDao 类，继承自 BaseDao

    def _group_app_details(self, app_codes, session):
        # _group_app_details 方法，用于分组应用详细信息

        app_detail_qry = session.query(GptsAppDetailEntity).filter(
            GptsAppDetailEntity.app_code.in_(app_codes)
        )
        # 查询指定 app_codes 中的应用详细信息

        app_details = app_detail_qry.all()
        # 获取所有匹配条件的应用详细信息列表

        app_details.sort(key=lambda x: x.app_code)
        # 按照 app_code 对应用详细信息列表进行排序

        app_details_group = {
            key: list(group)
            for key, group in groupby(app_details, key=lambda x: x.app_code)
        }
        # 根据 app_code 对应用详细信息列表进行分组

        return app_details_group
        # 返回分组后的应用详细信息字典
    # 定义一个方法，用于获取特定应用程序的详细信息
    def app_detail(self, app_code: str):
        # 使用数据库会话，查询包含指定应用程序代码的应用程序实体
        with self.session() as session:
            app_qry = session.query(GptsAppEntity).filter(
                GptsAppEntity.app_code == app_code
            )

            # 获取第一个匹配的应用程序实体
            app_info = app_qry.first()

            # 查询包含指定应用程序代码的应用程序详细信息实体
            app_detail_qry = session.query(GptsAppDetailEntity).filter(
                GptsAppDetailEntity.app_code == app_code
            )
            # 获取所有匹配的应用程序详细信息实体
            app_details = app_detail_qry.all()

            # 使用从数据库中检索的数据创建一个 GptsApp 对象
            app = GptsApp.from_dict(
                {
                    "app_code": app_info.app_code,
                    "app_name": app_info.app_name,
                    "language": app_info.language,
                    "app_describe": app_info.app_describe,
                    "team_mode": app_info.team_mode,
                    "team_context": _load_team_context(
                        app_info.team_mode, app_info.team_context
                    ),
                    "user_code": app_info.user_code,
                    "sys_code": app_info.sys_code,
                    "created_at": app_info.created_at,
                    "updated_at": app_info.updated_at,
                    "details": [
                        GptsAppDetail.from_dict(item.to_dict()) for item in app_details
                    ],
                }
            )

            # 返回创建的应用程序对象
            return app

    # 定义一个方法，用于删除特定应用程序及其相关插件和集合
    def delete(
        self,
        app_code: str,
        user_code: Optional[str] = None,
        sys_code: Optional[str] = None,
    ):
        """
        To delete the application, you also need to delete the corresponding plug-ins and collections.
        """
        # 如果应用程序代码为空，则抛出异常
        if app_code is None:
            raise f"cannot delete app when app_code is None"

        # 使用数据库会话，开始删除操作
        with self.session() as session:
            # 查询包含指定应用程序代码的应用程序实体
            app_qry = session.query(GptsAppEntity)
            app_qry = app_qry.filter(GptsAppEntity.app_code == app_code)
            # 删除匹配的应用程序实体
            app_qry.delete()

            # 查询包含指定应用程序代码的应用程序详细信息实体
            app_detail_qry = session.query(GptsAppDetailEntity).filter(
                GptsAppDetailEntity.app_code == app_code
            )
            # 删除匹配的应用程序详细信息实体
            app_detail_qry.delete()

            # 查询包含指定应用程序代码的应用程序集合实体
            app_collect_qry = session.query(GptsAppCollectionEntity).filter(
                GptsAppCollectionEntity.app_code == app_code
            )
            # 删除匹配的应用程序集合实体
            app_collect_qry.delete()
    # 定义一个方法用于创建一个新的 GptsApp 实体
    def create(self, gpts_app: GptsApp):
        # 使用 session() 方法获取一个数据库会话对象，确保操作的原子性
        with self.session() as session:
            # 创建一个新的 GptsAppEntity 对象，用于表示应用程序的基本信息
            app_entity = GptsAppEntity(
                app_code=str(uuid.uuid1()),  # 使用 UUID v1 生成一个唯一的应用程序代码
                app_name=gpts_app.app_name,  # 设置应用程序的名称
                app_describe=gpts_app.app_describe,  # 设置应用程序的描述
                team_mode=gpts_app.team_mode,  # 设置应用程序的团队模式
                team_context=_parse_team_context(gpts_app.team_context),  # 解析并设置团队上下文
                language=gpts_app.language,  # 设置应用程序的语言
                user_code=gpts_app.user_code,  # 设置用户代码
                sys_code=gpts_app.sys_code,  # 设置系统代码
                created_at=gpts_app.created_at,  # 设置创建时间
                updated_at=gpts_app.updated_at,  # 设置更新时间
                icon=gpts_app.icon,  # 设置应用程序的图标
            )
            # 将新创建的应用程序实体添加到会话中，以便保存到数据库
            session.add(app_entity)

            # 创建一个空列表，用于存储应用程序的详细信息
            app_details = []
            # 遍历给定的 gpts_app.details 列表
            for item in gpts_app.details:
                # 将每个 detail 中的资源转换为字典格式存储
                resource_dicts = [resource.to_dict() for resource in item.resources]
                # 如果 agent_name 为 None，则抛出异常
                if item.agent_name is None:
                    raise f"agent name cannot be None"

                # 创建一个新的 GptsAppDetailEntity 对象，用于表示应用程序的详细信息
                app_details.append(
                    GptsAppDetailEntity(
                        app_code=app_entity.app_code,  # 关联到对应的应用程序代码
                        app_name=app_entity.app_name,  # 关联到对应的应用程序名称
                        agent_name=item.agent_name,  # 设置代理名称
                        node_id=str(uuid.uuid1()),  # 使用 UUID v1 生成一个唯一的节点 ID
                        resources=json.dumps(resource_dicts, ensure_ascii=False),  # 将资源字典转换为 JSON 字符串
                        prompt_template=item.prompt_template,  # 设置提示模板
                        llm_strategy=item.llm_strategy,  # 设置LLM策略
                        llm_strategy_value=(
                            None
                            if item.llm_strategy_value is None
                            else json.dumps(tuple(item.llm_strategy_value.split(",")))  # 将LLM策略值转换为JSON字符串
                        ),
                        created_at=item.created_at,  # 设置创建时间
                        updated_at=item.updated_at,  # 设置更新时间
                    )
                )
            # 将所有的应用程序详细信息实体添加到会话中，以便保存到数据库
            session.add_all(app_details)

            # 将 gpts_app 的 app_code 更新为刚创建的 app_entity 的 app_code
            gpts_app.app_code = app_entity.app_code
            # 返回更新后的 gpts_app 对象
            return gpts_app
    # 定义一个方法，用于编辑给定的 GptsApp 对象
    def edit(self, gpts_app: GptsApp):
        # 开始数据库会话
        with self.session() as session:
            # 查询 GptsAppEntity 表中的数据
            app_qry = session.query(GptsAppEntity)
            
            # 如果 gpts_app 的 app_code 为 None，则抛出异常
            if gpts_app.app_code is None:
                raise f"app_code is None, don't allow to edit!"
            
            # 根据 app_code 过滤查询结果
            app_qry = app_qry.filter(GptsAppEntity.app_code == gpts_app.app_code)
            
            # 从查询结果中获取唯一的 app_entity 对象
            app_entity = app_qry.one()
            
            # 更新 app_entity 对象的属性值
            app_entity.app_name = gpts_app.app_name
            app_entity.app_describe = gpts_app.app_describe
            app_entity.language = gpts_app.language
            app_entity.team_mode = gpts_app.team_mode
            app_entity.icon = gpts_app.icon
            app_entity.team_context = _parse_team_context(gpts_app.team_context)
            
            # 合并更新后的 app_entity 对象到数据库会话中
            session.merge(app_entity)

            # 查询并删除与 gpts_app.app_code 相关的旧的 GptsAppDetailEntity 记录
            old_details = session.query(GptsAppDetailEntity).filter(
                GptsAppDetailEntity.app_code == gpts_app.app_code
            )
            old_details.delete()
            
            # 提交数据库会话中的所有操作
            session.commit()

            # 准备将 gpts_app.details 中的数据添加到数据库中
            app_details = []
            for item in gpts_app.details:
                # 将 item 中的资源转换为字典形式
                resource_dicts = [resource.to_dict() for resource in item.resources]
                
                # 创建新的 GptsAppDetailEntity 对象，并添加到 app_details 列表中
                app_details.append(
                    GptsAppDetailEntity(
                        app_code=gpts_app.app_code,
                        app_name=gpts_app.app_name,
                        agent_name=item.agent_name,
                        node_id=str(uuid.uuid1()),
                        resources=json.dumps(resource_dicts, ensure_ascii=False),
                        prompt_template=item.prompt_template,
                        llm_strategy=item.llm_strategy,
                        llm_strategy_value=(
                            None
                            if item.llm_strategy_value is None
                            else json.dumps(tuple(item.llm_strategy_value.split(",")))
                        ),
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                    )
                )
            
            # 将所有新创建的 app_details 记录添加到数据库会话中
            session.add_all(app_details)
            
            # 方法执行成功，返回 True
            return True
# 将 team_context 解析为字符串
def _parse_team_context(team_context: Optional[Union[str, AWELTeamContext]] = None):
    """
    parse team_context to str
    """
    # 如果 team_context 是 AWELTeamContext 类型，则将其转换为 JSON 字符串
    if isinstance(team_context, AWELTeamContext):
        return model_to_json(team_context)
    # 否则直接返回 team_context
    return team_context


# 加载 team_context 为字符串或 AWELTeamContext 对象
def _load_team_context(
    team_mode: str = None, team_context: str = None
) -> Union[str, AWELTeamContext]:
    """
    load team_context to str or AWELTeamContext
    """
    # 如果 team_mode 不为 None
    if team_mode is not None:
        # 匹配 team_mode 的值
        match team_mode:
            # 当 team_mode 为 TeamMode.AWEL_LAYOUT.value 时
            case TeamMode.AWEL_LAYOUT.value:
                try:
                    # 尝试将 team_context 解析为 AWELTeamContext 对象
                    awel_team_ctx = AWELTeamContext(**json.loads(team_context))
                    return awel_team_ctx
                except Exception as ex:
                    # 捕获异常并记录日志
                    logger.info(
                        f"_load_team_context error, team_mode={team_mode}, team_context={team_context}, {ex}"
                    )
    # 返回 team_context
    return team_context
```