# `.\DB-GPT-src\dbgpt\serve\rag\models\models.py`

```py
# 从 datetime 模块导入 datetime 类型
from datetime import datetime
# 从 typing 模块导入 Any、Dict、List、Union 类型
from typing import Any, Dict, List, Union

# 从 sqlalchemy 库中导入 Column、DateTime、Integer、String、Text 类型
from sqlalchemy import Column, DateTime, Integer, String, Text

# 从 dbgpt._private.pydantic 模块导入 model_to_dict 函数
from dbgpt._private.pydantic import model_to_dict
# 从 dbgpt.serve.rag.api.schemas 模块导入 SpaceServeRequest、SpaceServeResponse 类型
from dbgpt.serve.rag.api.schemas import SpaceServeRequest, SpaceServeResponse
# 从 dbgpt.storage.metadata 模块导入 BaseDao、Model 类型
from dbgpt.storage.metadata import BaseDao, Model


# 定义 KnowledgeSpaceEntity 类，继承自 Model 类
class KnowledgeSpaceEntity(Model):
    # 数据库表名设置为 "knowledge_space"
    __tablename__ = "knowledge_space"
    # 定义 id 列，主键，整数类型
    id = Column(Integer, primary_key=True)
    # 定义 name 列，字符串类型，最大长度 100
    name = Column(String(100))
    # 定义 vector_type 列，字符串类型，最大长度 100
    vector_type = Column(String(100))
    # 定义 domain_type 列，字符串类型，最大长度 100
    domain_type = Column(String(100))
    # 定义 desc 列，字符串类型，最大长度 100
    desc = Column(String(100))
    # 定义 owner 列，字符串类型，最大长度 100
    owner = Column(String(100))
    # 定义 context 列，文本类型
    context = Column(Text)
    # 定义 gmt_created 列，日期时间类型
    gmt_created = Column(DateTime)
    # 定义 gmt_modified 列，日期时间类型
    gmt_modified = Column(DateTime)

    # 定义 __repr__ 方法，返回对象的可打印字符串表示
    def __repr__(self):
        return f"KnowledgeSpaceEntity(id={self.id}, name='{self.name}', vector_type='{self.vector_type}', domain_type='{self.domain_type}', desc='{self.desc}', owner='{self.owner}' context='{self.context}', gmt_created='{self.gmt_created}', gmt_modified='{self.gmt_modified}')"


# 定义 KnowledgeSpaceDao 类，继承自 BaseDao 类
class KnowledgeSpaceDao(BaseDao):
    # 定义 create_knowledge_space 方法，用于创建知识空间
    def create_knowledge_space(self, space: SpaceServeRequest):
        """Create knowledge space"""
        # 获取数据库会话
        session = self.get_raw_session()
        # 创建 KnowledgeSpaceEntity 实例
        knowledge_space = KnowledgeSpaceEntity(
            name=space.name,
            vector_type=space.vector_type,
            domain_type=space.domain_type,
            desc=space.desc,
            owner=space.owner,
            gmt_created=datetime.now(),  # 设置创建时间为当前时间
            gmt_modified=datetime.now(),  # 设置修改时间为当前时间
        )
        # 将实例添加到数据库会话中
        session.add(knowledge_space)
        # 提交会话中的所有操作
        session.commit()
        # 获取创建的知识空间的 ID
        space_id = knowledge_space.id
        # 关闭数据库会话
        session.close()
        # 将创建的知识空间转换为响应对象并返回
        return self.to_response(knowledge_space)
    def get_knowledge_space(self, query: KnowledgeSpaceEntity):
        """Get knowledge space by query"""
        # 获取原始数据库会话
        session = self.get_raw_session()
        # 从数据库中查询所有 KnowledgeSpaceEntity 对象
        knowledge_spaces = session.query(KnowledgeSpaceEntity)

        # 根据查询对象的属性逐一过滤查询结果集
        if query.id is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.id == query.id
            )
        if query.name is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.name == query.name
            )
        if query.vector_type is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.vector_type == query.vector_type
            )
        if query.domain_type is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.domain_type == query.domain_type
            )
        if query.desc is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.desc == query.desc
            )
        if query.owner is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.owner == query.owner
            )
        if query.gmt_created is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.gmt_created == query.gmt_created
            )
        if query.gmt_modified is not None:
            knowledge_spaces = knowledge_spaces.filter(
                KnowledgeSpaceEntity.gmt_modified == query.gmt_modified
            )
        
        # 根据 gmt_created 属性降序排序查询结果
        knowledge_spaces = knowledge_spaces.order_by(
            KnowledgeSpaceEntity.gmt_created.desc()
        )
        
        # 获取所有查询结果
        result = knowledge_spaces.all()
        
        # 关闭数据库会话
        session.close()
        
        # 返回查询结果
        return result

    def update_knowledge_space(self, space: KnowledgeSpaceEntity):
        """Update knowledge space"""
        # 获取原始数据库会话
        session = self.get_raw_session()
        
        # 创建 SpaceServeRequest 对象，用于查询
        request = SpaceServeRequest(id=space.id)
        
        # 将 space 对象转换为更新请求对象
        update_request = self.to_request(space)
        
        # 根据查询对象创建查询
        query = self._create_query_object(session, request)
        
        # 查询第一个匹配的结果对象
        entry = query.first()
        
        # 如果未找到匹配的结果对象，抛出异常
        if entry is None:
            raise Exception("Invalid request")
        
        # 将 update_request 中的非空字段更新到 entry 对象中
        for key, value in model_to_dict(update_request).items():  # type: ignore
            if value is not None:
                setattr(entry, key, value)
        
        # 将更新后的 entry 对象合并到数据库会话中
        session.merge(entry)
        
        # 提交会话中的所有更改
        session.commit()
        
        # 关闭数据库会话
        session.close()
        
        # 返回更新后的 space 对象的响应
        return self.to_response(space)

    def delete_knowledge_space(self, space: KnowledgeSpaceEntity):
        """Delete knowledge space"""
        # 获取原始数据库会话
        session = self.get_raw_session()
        
        # 如果 space 对象存在，则从数据库中删除
        if space:
            session.delete(space)
            session.commit()
        
        # 关闭数据库会话
        session.close()

    def from_request(
        self, request: Union[SpaceServeRequest, Dict[str, Any]]
    ) -> KnowledgeSpaceEntity:
        """
        将请求转换为实体对象

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): 请求对象或字典表示的请求

        Returns:
            KnowledgeSpaceEntity: 转换后的实体对象
        """
        # 如果请求是 SpaceServeRequest 类型，则将其转换为字典
        request_dict = (
            model_to_dict(request)
            if isinstance(request, SpaceServeRequest)
            else request
        )
        # 使用请求字典创建 KnowledgeSpaceEntity 实例
        entity = KnowledgeSpaceEntity(**request_dict)
        return entity

    def to_request(self, entity: KnowledgeSpaceEntity) -> SpaceServeRequest:
        """
        将实体对象转换为请求对象

        Args:
            entity (KnowledgeSpaceEntity): 实体对象

        Returns:
            SpaceServeRequest: 转换后的请求对象
        """
        # 使用实体对象的属性创建 SpaceServeRequest 实例
        return SpaceServeRequest(
            id=entity.id,
            name=entity.name,
            vector_type=entity.vector_type,
            desc=entity.desc,
            owner=entity.owner,
            context=entity.context,
        )

    def to_response(self, entity: KnowledgeSpaceEntity) -> SpaceServeResponse:
        """
        将实体对象转换为响应对象

        Args:
            entity (KnowledgeSpaceEntity): 实体对象

        Returns:
            SpaceServeResponse: 转换后的响应对象
        """
        # 使用实体对象的属性创建 SpaceServeResponse 实例
        return SpaceServeResponse(
            id=entity.id,
            name=entity.name,
            vector_type=entity.vector_type,
            desc=entity.desc,
            owner=entity.owner,
            context=entity.context,
            domain_type=entity.domain_type,
        )
```