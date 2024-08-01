# `.\DB-GPT-src\dbgpt\serve\agent\db\gpts_conversations_db.py`

```py
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

from sqlalchemy import (  # 导入 SQLAlchemy 中需要的多个类和函数
    Column,  # 定义数据库表的列
    DateTime,  # 定义日期时间类型的列
    Index,  # 定义数据库索引
    Integer,  # 定义整数类型的列
    String,  # 定义字符串类型的列
    Text,  # 定义文本类型的列
    UniqueConstraint,  # 定义唯一约束
    desc,  # 定义降序排序
    func,  # 定义 SQL 函数
)

from dbgpt.storage.metadata import BaseDao, Model  # 导入自定义的数据库基类和模型基类


class GptsConversationsEntity(Model):
    __tablename__ = "gpts_conversations"  # 设置数据库表名

    id = Column(Integer, primary_key=True, comment="autoincrement id")  # 定义主键 id 列

    conv_id = Column(  # 定义 conv_id 列
        String(255), nullable=False, comment="The unique id of the conversation record"
    )
    user_goal = Column(  # 定义 user_goal 列
        Text, nullable=False, comment="User's goals content"
    )

    gpts_name = Column(  # 定义 gpts_name 列
        String(255), nullable=False, comment="The gpts name"
    )
    team_mode = Column(  # 定义 team_mode 列
        String(255), nullable=False, comment="The conversation team mode"
    )
    state = Column(  # 定义 state 列
        String(255), nullable=True, comment="The gpts state"
    )

    max_auto_reply_round = Column(  # 定义 max_auto_reply_round 列
        Integer, nullable=False, comment="max auto reply round"
    )
    auto_reply_count = Column(  # 定义 auto_reply_count 列
        Integer, nullable=False, comment="auto reply count"
    )

    user_code = Column(  # 定义 user_code 列
        String(255), nullable=True, comment="user code"
    )
    sys_code = Column(  # 定义 sys_code 列
        String(255), nullable=True, comment="system app"
    )

    created_at = Column(  # 定义 created_at 列
        DateTime, default=datetime.utcnow, comment="create time"
    )
    updated_at = Column(  # 定义 updated_at 列
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="last update time"
    )

    __table_args__ = (  # 定义表级参数
        UniqueConstraint("conv_id", name="uk_gpts_conversations"),  # 定义唯一约束
        Index("idx_gpts_name", "gpts_name"),  # 定义索引
    )


class GptsConversationsDao(BaseDao):
    def add(self, engity: GptsConversationsEntity):
        session = self.get_raw_session()  # 获取数据库会话
        session.add(engity)  # 将实体对象添加到会话中
        session.commit()  # 提交会话，将变更写入数据库
        id = engity.id  # 获取新增记录的主键 id
        session.close()  # 关闭会话
        return id  # 返回新增记录的主键 id

    def get_by_conv_id(self, conv_id: str):
        session = self.get_raw_session()  # 获取数据库会话
        gpts_conv = session.query(GptsConversationsEntity)  # 查询 GptsConversationsEntity 对象
        if conv_id:
            gpts_conv = gpts_conv.filter(GptsConversationsEntity.conv_id == conv_id)  # 根据 conv_id 过滤查询结果
        result = gpts_conv.first()  # 获取查询结果的第一个对象
        session.close()  # 关闭会话
        return result  # 返回查询结果

    def get_convs(self, user_code: str = None, system_app: str = None):
        session = self.get_raw_session()  # 获取数据库会话
        gpts_conversations = session.query(GptsConversationsEntity)  # 查询 GptsConversationsEntity 对象
        if user_code:
            gpts_conversations = gpts_conversations.filter(
                GptsConversationsEntity.user_code == user_code  # 根据 user_code 过滤查询结果
            )
        if system_app:
            gpts_conversations = gpts_conversations.filter(
                GptsConversationsEntity.system_app == system_app  # 根据 system_app 过滤查询结果
            )

        result = (
            gpts_conversations.limit(20)  # 限制查询结果的数量为 20 条
            .order_by(desc(GptsConversationsEntity.id))  # 根据 id 降序排序
            .all()  # 获取所有符合条件的查询结果
        )
        session.close()  # 关闭会话
        return result  # 返回查询结果
    # 定义一个方法，用于更新会话状态
    def update(self, conv_id: str, state: str):
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 从数据库中查询 GptsConversationsEntity 对象
        gpts_convs = session.query(GptsConversationsEntity)
        # 根据 conv_id 筛选出特定的会话对象
        gpts_convs = gpts_convs.filter(GptsConversationsEntity.conv_id == conv_id)
        # 更新选定会话对象的状态字段为指定的 state 值
        gpts_convs.update(
            {GptsConversationsEntity.state: state}, synchronize_session="fetch"
        )
        # 提交事务，保存更新到数据库
        session.commit()
        # 关闭数据库会话
        session.close()
```