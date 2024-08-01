# `.\DB-GPT-src\dbgpt\serve\agent\db\plugin_hub_db.py`

```py
# 导入需要的模块
from datetime import datetime
from typing import List

import pytz
from sqlalchemy import (
    DDL,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)

# 导入自定义模块
from dbgpt.storage.metadata import BaseDao, Model
from ..model import PluginHubVO

# 定义一个DDL语句，用于修改表的字符集
char_set_sql = DDL("ALTER TABLE plugin_hub CONVERT TO CHARACTER SET utf8mb4")

# 定义插件中心实体类
class PluginHubEntity(Model):
    # 指定表名
    __tablename__ = "plugin_hub"
    # 定义自增主键id
    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="autoincrement id"
    )
    # 定义插件名称字段，唯一且不为空
    name = Column(String(255), unique=True, nullable=False, comment="plugin name")
    # 定义插件描述字段，不为空
    description = Column(String(255), nullable=False, comment="plugin description")
    # 定义插件作者字段
    author = Column(String(255), nullable=True, comment="plugin author")
    # 定义插件作者邮箱字段
    email = Column(String(255), nullable=True, comment="plugin author email")
    # 定义插件类型字段
    type = Column(String(255), comment="plugin type")
    # 定义插件版本字段
    version = Column(String(255), comment="plugin version")
    # 定义插件存储通道字段
    storage_channel = Column(String(255), comment="plugin storage channel")
    # 定义插件下载链接字段
    storage_url = Column(String(255), comment="plugin download url")
    # 定义插件下载参数字段
    download_param = Column(String(255), comment="plugin download param")
    # 定义插件上传时间字段，默认为当前时间
    gmt_created = Column(
        DateTime, default=datetime.utcnow, comment="plugin upload time"
    )
    # 定义插件安装数量字段，默认为0
    installed = Column(Integer, default=False, comment="plugin already installed count")

    # 定义name字段的唯一约束
    UniqueConstraint("name", name="uk_name")
    # 创建type字段的索引
    Index("idx_q_type", "type")

    # 将实体类列表转换为VO列表的方法
    @classmethod
    def to_vo(cls, entities: List["PluginHubEntity"]) -> List[PluginHubVO]:
        results = []
        for entity in entities:
            vo = PluginHubVO(
                id=entity.id,
                name=entity.name,
                description=entity.description,
                author=entity.author,
                email=entity.email,
                type=entity.type,
                version=entity.version,
                storage_channel=entity.storage_channel,
                storage_url=entity.storage_url,
                download_param=entity.download_param,
                installed=entity.installed,
                gmt_created=entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S"),
            )
            results.append(vo)
        return results

# 定义插件中心数据访问对象
class PluginHubDao(BaseDao):
    # 添加插件信息的方法
    def add(self, engity: PluginHubEntity):
        # 获取数据库会话
        session = self.get_raw_session()
        # 设置时区为亚洲/上海
        timezone = pytz.timezone("Asia/Shanghai")
        # 创建插件中心实体对象
        plugin_hub = PluginHubEntity(
            name=engity.name,
            author=engity.author,
            email=engity.email,
            type=engity.type,
            version=engity.version,
            storage_channel=engity.storage_channel,
            storage_url=engity.storage_url,
            gmt_created=timezone.localize(datetime.now()),
        )
        # 将实体对象添加到会话中
        session.add(plugin_hub)
        # 提交事务
        session.commit()
        # 获取插件id
        id = plugin_hub.id
        # 关闭会话
        session.close()
        # 返回插件id
        return id
    # 更新数据库中的实体对象
    def raw_update(self, entity: PluginHubEntity):
        # 获取原始数据库会话
        session = self.get_raw_session()
        try:
            # 合并（更新或插入）给定的实体对象，并提交事务
            updated = session.merge(entity)
            session.commit()
            # 返回更新后实体对象的 ID
            return updated.id
        finally:
            # 无论如何都关闭数据库会话
            session.close()

    # 根据查询条件获取插件中心实体对象的列表
    def list(
        self, query: PluginHubEntity, page=1, page_size=20
    ) -> list[PluginHubEntity]:
        # 获取原始数据库会话
        session = self.get_raw_session()
        plugin_hubs = session.query(PluginHubEntity)

        # 获取总记录数
        all_count = plugin_hubs.count()

        # 根据查询条件过滤数据
        if query.id is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.id == query.id)
        if query.name is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.name == query.name)
        if query.type is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.type == query.type)
        if query.author is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.author == query.author)
        if query.storage_channel is not None:
            plugin_hubs = plugin_hubs.filter(
                PluginHubEntity.storage_channel == query.storage_channel
            )

        # 根据 ID 降序排列数据
        plugin_hubs = plugin_hubs.order_by(PluginHubEntity.id.desc())
        # 分页获取数据
        plugin_hubs = plugin_hubs.offset((page - 1) * page_size).limit(page_size)
        # 获取所有结果
        result = plugin_hubs.all()
        # 关闭数据库会话
        session.close()

        # 计算总页数
        total_pages = all_count // page_size
        if all_count % page_size != 0:
            total_pages += 1

        # 返回结果列表、总页数和总记录数
        return result, total_pages, all_count

    # 根据存储 URL 获取插件中心实体对象
    def get_by_storage_url(self, storage_url):
        # 获取原始数据库会话
        session = self.get_raw_session()
        plugin_hubs = session.query(PluginHubEntity)
        # 根据存储 URL 过滤数据
        plugin_hubs = plugin_hubs.filter(PluginHubEntity.storage_url == storage_url)
        # 获取所有结果
        result = plugin_hubs.all()
        # 关闭数据库会话
        session.close()
        # 返回结果列表
        return result

    # 根据名称获取单个插件中心实体对象
    def get_by_name(self, name: str) -> PluginHubEntity:
        # 获取原始数据库会话
        session = self.get_raw_session()
        plugin_hubs = session.query(PluginHubEntity)
        # 根据名称过滤数据，获取第一个结果
        result = plugin_hubs.filter(PluginHubEntity.name == name).first()
        # 关闭数据库会话
        session.close()
        # 返回单个实体对象结果
        return result
    # 定义一个方法用于计算符合查询条件的插件中心实体的数量
    def count(self, query: PluginHubEntity):
        # 获取一个原始的数据库会话
        session = self.get_raw_session()
        
        # 创建一个查询以统计 PluginHubEntity 表中的记录数量
        plugin_hubs = session.query(func.count(PluginHubEntity.id))
        
        # 如果查询对象的 id 属性不为空，则添加 id 条件过滤
        if query.id is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.id == query.id)
        
        # 如果查询对象的 name 属性不为空，则添加 name 条件过滤
        if query.name is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.name == query.name)
        
        # 如果查询对象的 type 属性不为空，则添加 type 条件过滤
        if query.type is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.type == query.type)
        
        # 如果查询对象的 author 属性不为空，则添加 author 条件过滤
        if query.author is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.author == query.author)
        
        # 如果查询对象的 storage_channel 属性不为空，则添加 storage_channel 条件过滤
        if query.storage_channel is not None:
            plugin_hubs = plugin_hubs.filter(
                PluginHubEntity.storage_channel == query.storage_channel
            )
        
        # 执行查询并获取结果数量
        count = plugin_hubs.scalar()
        
        # 关闭数据库会话
        session.close()
        
        # 返回符合条件的记录数量
        return count

    # 定义一个方法用于删除指定插件 ID 的插件中心实体
    def raw_delete(self, plugin_id: int):
        # 获取一个原始的数据库会话
        session = self.get_raw_session()
        
        # 如果插件 ID 为空，则抛出异常
        if plugin_id is None:
            raise Exception("plugin_id is None")
        
        # 创建一个查询以获取指定 ID 的插件中心实体
        plugin_hubs = session.query(PluginHubEntity)
        
        # 根据插件 ID 添加过滤条件
        if plugin_id is not None:
            plugin_hubs = plugin_hubs.filter(PluginHubEntity.id == plugin_id)
        
        # 执行删除操作
        plugin_hubs.delete()
        
        # 提交会话中的事务
        session.commit()
        
        # 关闭数据库会话
        session.close()
```