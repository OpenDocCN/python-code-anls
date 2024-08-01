# `.\DB-GPT-src\dbgpt\serve\agent\db\gpts_manage_db.py`

```py
# 导入 datetime 模块中的 datetime 类，用于处理日期和时间
from datetime import datetime

# 导入 SQLAlchemy 中的各个列类型和约束
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

# 导入自定义的模型和基础 DAO 类
from dbgpt.storage.metadata import BaseDao, Model


# 定义 GptsInstanceEntity 类，继承自 Model 类，映射到数据库表 'gpts_instance'
class GptsInstanceEntity(Model):
    # 指定数据库表名
    __tablename__ = "gpts_instance"

    # 主键字段 id，自增长，用于唯一标识实例
    id = Column(Integer, primary_key=True, comment="autoincrement id")

    # 字段 gpts_name，存储当前 AI 助手的名称，不允许为空
    gpts_name = Column(String(255), nullable=False, comment="Current AI assistant name")

    # 字段 gpts_describe，存储当前 AI 助手的描述，不允许为空
    gpts_describe = Column(
        String(2255), nullable=False, comment="Current AI assistant describe"
    )

    # 字段 team_mode，存储团队工作模式，不允许为空
    team_mode = Column(String(255), nullable=False, comment="Team work mode")

    # 字段 is_sustainable，存储是否应用于可持续对话，布尔类型，默认为 False
    is_sustainable = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Applications for sustainable dialogue",
    )

    # 字段 resource_db，存储当前 GPTs 包含的结构化数据库名称列表，可以为空
    resource_db = Column(
        Text,
        nullable=True,
        comment="List of structured database names contained in the current gpts",
    )

    # 字段 resource_internet，存储是否可以从互联网检索信息，可以为空
    resource_internet = Column(
        Text,
        nullable=True,
        comment="Is it possible to retrieve information from the internet",
    )

    # 字段 resource_knowledge，存储当前 GPTs 包含的非结构化数据库名称列表，可以为空
    resource_knowledge = Column(
        Text,
        nullable=True,
        comment="List of unstructured database names contained in the current gpts",
    )

    # 字段 gpts_agents，存储当前 GPTs 包含的代理人名称列表，可以为空
    gpts_agents = Column(
        String(1000),
        nullable=True,
        comment="List of agents names contained in the current gpts",
    )

    # 字段 gpts_models，存储当前 GPTs 包含的语言模型名称列表，可以为空
    gpts_models = Column(
        String(1000),
        nullable=True,
        comment="List of llm model names contained in the current gpts",
    )

    # 字段 language，存储当前 GPTs 使用的语言，可以为空
    language = Column(String(100), nullable=True, comment="gpts language")

    # 字段 user_code，存储用户代码，不允许为空
    user_code = Column(String(255), nullable=False, comment="user code")

    # 字段 sys_code，存储系统应用代码，可以为空
    sys_code = Column(String(255), nullable=True, comment="system app code")

    # 字段 created_at，存储创建时间，默认为当前 UTC 时间
    created_at = Column(DateTime, default=datetime.utcnow, comment="create time")

    # 字段 updated_at，存储最后更新时间，每次更新时自动更新为当前 UTC 时间
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        comment="last update time",
    )

    # 定义表级别的约束条件，确保 gpts_name 的唯一性
    __table_args__ = (UniqueConstraint("gpts_name", name="uk_gpts"),)


# 定义 GptsInstanceDao 类，继承自 BaseDao 类，用于实现对 GptsInstanceEntity 的数据库操作
class GptsInstanceDao(BaseDao):

    # 添加新的 GptsInstanceEntity 实例到数据库
    def add(self, engity: GptsInstanceEntity):
        # 获取原始的数据库会话
        session = self.get_raw_session()
        # 添加实体对象到会话中
        session.add(engity)
        # 提交会话中的改动到数据库
        session.commit()
        # 获取添加后的实体对象的 id
        id = engity.id
        # 关闭会话
        session.close()
        # 返回添加实体的 id
        return id

    # 根据名称查询 GptsInstanceEntity 实例
    def get_by_name(self, name: str) -> GptsInstanceEntity:
        # 获取原始的数据库会话
        session = self.get_raw_session()
        # 创建查询对象，查询 GptsInstanceEntity 表
        gpts_instance = session.query(GptsInstanceEntity)
        # 如果提供了名称，添加名称过滤条件到查询对象中
        if name:
            gpts_instance = gpts_instance.filter(GptsInstanceEntity.gpts_name == name)
        # 执行查询，返回第一个匹配的结果对象
        result = gpts_instance.first()
        # 关闭会话
        session.close()
        # 返回查询结果
        return result
    # 定义一个方法，用于根据用户代码和系统代码查询数据库中的记录
    def get_by_user(self, user_code: str = None, sys_code: str = None):
        # 获取原始数据库会话对象
        session = self.get_raw_session()
        
        # 从数据库中查询GptsInstanceEntity表的实例
        gpts_instance = session.query(GptsInstanceEntity)
        
        # 如果提供了用户代码，则按用户代码过滤查询结果
        if user_code:
            gpts_instance = gpts_instance.filter(
                GptsInstanceEntity.user_code == user_code
            )
        
        # 如果提供了系统代码，则按系统代码过滤查询结果
        if sys_code:
            gpts_instance = gpts_instance.filter(
                GptsInstanceEntity.sys_code == sys_code
            )
        
        # 获取所有符合条件的查询结果
        result = gpts_instance.all()
        
        # 关闭数据库会话
        session.close()
        
        # 返回查询结果
        return result
```