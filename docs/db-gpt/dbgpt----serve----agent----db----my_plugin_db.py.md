# `.\DB-GPT-src\dbgpt\serve\agent\db\my_plugin_db.py`

```py
# 从 datetime 模块中导入 datetime 类
from datetime import datetime
# 从 typing 模块中导入 List 类型
from typing import List
# 导入 SQLAlchemy 中的 Column、DateTime、Integer、String 和 UniqueConstraint 类
from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint, func
# 从 dbgpt.storage.metadata 模块中导入 BaseDao 和 Model 类
from dbgpt.storage.metadata import BaseDao, Model
# 从当前包中的 model 模块导入 MyPluginVO 类
from ..model import MyPluginVO

# 定义 MyPluginEntity 类，继承自 Model 类
class MyPluginEntity(Model):
    # 指定数据库表名为 my_plugin
    __tablename__ = "my_plugin"
    # 定义 id 字段，为整型主键，自增长，注释为 "autoincrement id"
    id = Column(Integer, primary_key=True, comment="autoincrement id")
    # 定义 tenant 字段，为最大长度为 255 的字符串，可为空，注释为 "user's tenant"
    tenant = Column(String(255), nullable=True, comment="user's tenant")
    # 定义 user_code 字段，为最大长度为 255 的字符串，不可为空，注释为 "user code"
    user_code = Column(String(255), nullable=False, comment="user code")
    # 定义 user_name 字段，为最大长度为 255 的字符串，可为空，注释为 "user name"
    user_name = Column(String(255), nullable=True, comment="user name")
    # 定义 name 字段，为最大长度为 255 的字符串，必须唯一，不可为空，注释为 "plugin name"
    name = Column(String(255), unique=True, nullable=False, comment="plugin name")
    # 定义 file_name 字段，为最大长度为 255 的字符串，不可为空，注释为 "plugin package file name"
    file_name = Column(String(255), nullable=False, comment="plugin package file name")
    # 定义 type 字段，为最大长度为 255 的字符串，可为空，注释为 "plugin type"
    type = Column(String(255), comment="plugin type")
    # 定义 version 字段，为最大长度为 255 的字符串，可为空，注释为 "plugin version"
    version = Column(String(255), comment="plugin version")
    # 定义 use_count 字段，为整型，可为空，默认为 0，注释为 "plugin total use count"
    use_count = Column(Integer, nullable=True, default=0, comment="plugin total use count")
    # 定义 succ_count 字段，为整型，可为空，默认为 0，注释为 "plugin total success count"
    succ_count = Column(Integer, nullable=True, default=0, comment="plugin total success count")
    # 定义 sys_code 字段，为最大长度为 128 的字符串，可为空，加索引，注释为 "System code"
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")
    # 定义 gmt_created 字段，为日期时间类型，默认为当前 UTC 时间，注释为 "plugin install time"
    gmt_created = Column(DateTime, default=datetime.utcnow, comment="plugin install time")
    # 唯一约束，保证 (user_code, name) 唯一
    UniqueConstraint("user_code", "name", name="uk_name")

    # 类方法，将 MyPluginEntity 对象列表转换为 MyPluginVO 对象列表
    @classmethod
    def to_vo(cls, entities: List["MyPluginEntity"]) -> List[MyPluginVO]:
        # 初始化结果列表
        results = []
        # 遍历实体列表
        for entity in entities:
            # 将实体转换为 MyPluginVO 对象，并加入结果列表
            results.append(
                MyPluginVO(
                    id=entity.id,
                    tenant=entity.tenant,
                    user_code=entity.user_code,
                    user_name=entity.user_name,
                    sys_code=entity.sys_code,
                    name=entity.name,
                    file_name=entity.file_name,
                    type=entity.type,
                    version=entity.version,
                    use_count=entity.use_count,
                    succ_count=entity.succ_count,
                    gmt_created=entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        # 返回转换后的结果列表
        return results


# 定义 MyPluginDao 类，继承自 BaseDao 类
class MyPluginDao(BaseDao):
    # 添加方法，向数据库中添加 MyPluginEntity 对象
    def add(self, entity: MyPluginEntity):
        # 获取数据库会话
        session = self.get_raw_session()
        # 创建 MyPluginEntity 对象
        my_plugin = MyPluginEntity(
            tenant=entity.tenant,
            user_code=entity.user_code,
            user_name=entity.user_name,
            name=entity.name,
            type=entity.type,
            version=entity.version,
            use_count=entity.use_count or 0,
            succ_count=entity.succ_count or 0,
            sys_code=entity.sys_code,
            gmt_created=datetime.now(),
        )
        # 将对象添加到会话中
        session.add(my_plugin)
        # 提交会话
        session.commit()
        # 获取插入后的 id
        id = my_plugin.id
        # 关闭会话
        session.close()
        # 返回插入的 id
        return id

    # 更新方法，更新数据库中的 MyPluginEntity 对象
    def raw_update(self, entity: MyPluginEntity):
        # 获取数据库会话
        session = self.get_raw_session()
        # 合并并更新实体对象
        updated = session.merge(entity)
        # 提交会话
        session.commit()
        # 返回更新后的对象 id
        return updated.id
    # 根据指定用户获取对应的插件列表
    def get_by_user(self, user: str) -> list[MyPluginEntity]:
        # 获取原始数据库会话对象
        session = self.get_raw_session()
        # 从数据库中查询所有的 MyPluginEntity 对象
        my_plugins = session.query(MyPluginEntity)
        # 如果提供了用户参数，则过滤出特定用户的插件
        if user:
            my_plugins = my_plugins.filter(MyPluginEntity.user_code == user)
        # 将查询结果转换为列表并返回
        result = my_plugins.all()
        # 关闭数据库会话
        session.close()
        return result

    # 根据用户和插件名称获取特定的插件实体
    def get_by_user_and_plugin(self, user: str, plugin: str) -> MyPluginEntity:
        # 获取原始数据库会话对象
        session = self.get_raw_session()
        # 从数据库中查询所有的 MyPluginEntity 对象
        my_plugins = session.query(MyPluginEntity)
        # 如果提供了用户参数，则过滤出特定用户的插件
        if user:
            my_plugins = my_plugins.filter(MyPluginEntity.user_code == user)
        # 进一步根据插件名称过滤出特定插件
        my_plugins = my_plugins.filter(MyPluginEntity.name == plugin)
        # 返回符合条件的第一个插件实体
        result = my_plugins.first()
        # 关闭数据库会话
        session.close()
        return result

    # 查询符合给定查询条件的插件列表，并分页返回结果
    def list(self, query: MyPluginEntity, page=1, page_size=20) -> list[MyPluginEntity]:
        # 获取原始数据库会话对象
        session = self.get_raw_session()
        # 从数据库中查询所有的 MyPluginEntity 对象
        my_plugins = session.query(MyPluginEntity)
        
        # 获取符合查询条件的总记录数
        all_count = my_plugins.count()
        
        # 根据传入的查询参数逐一过滤插件列表
        if query.id is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.id == query.id)
        if query.name is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.name == query.name)
        if query.tenant is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.tenant == query.tenant)
        if query.type is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.type == query.type)
        if query.user_code is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.user_code == query.user_code)
        if query.user_name is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.user_name == query.user_name)
        if query.sys_code is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.sys_code == query.sys_code)

        # 根据插件 ID 降序排列
        my_plugins = my_plugins.order_by(MyPluginEntity.id.desc())
        # 根据分页参数计算偏移量和限制结果数量
        my_plugins = my_plugins.offset((page - 1) * page_size).limit(page_size)
        # 将查询结果转换为列表并返回
        result = my_plugins.all()
        # 关闭数据库会话
        session.close()
        
        # 计算总页数
        total_pages = all_count // page_size
        if all_count % page_size != 0:
            total_pages += 1
        
        # 返回查询结果列表、总页数和总记录数
        return result, total_pages, all_count
    # 计算满足给定查询条件的插件数量
    def count(self, query: MyPluginEntity):
        # 获取数据库会话
        session = self.get_raw_session()
        # 创建查询对象，计算 MyPluginEntity 表中符合条件的记录数量
        my_plugins = session.query(func.count(MyPluginEntity.id))
        # 根据查询对象的属性逐步添加过滤条件
        if query.id is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.id == query.id)
        if query.name is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.name == query.name)
        if query.type is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.type == query.type)
        if query.tenant is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.tenant == query.tenant)
        if query.user_code is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.user_code == query.user_code)
        if query.user_name is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.user_name == query.user_name)
        if query.sys_code is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.sys_code == query.sys_code)
        # 执行查询，获取符合条件的记录数量
        count = my_plugins.scalar()
        # 关闭数据库会话
        session.close()
        # 返回符合条件的记录数量
        return count

    # 删除指定插件 ID 的记录
    def raw_delete(self, plugin_id: int):
        # 获取数据库会话
        session = self.get_raw_session()
        # 如果插件 ID 为 None，抛出异常
        if plugin_id is None:
            raise Exception("plugin_id is None")
        # 创建插件实体对象并设置其 ID
        query = MyPluginEntity(id=plugin_id)
        # 创建插件实体查询对象
        my_plugins = session.query(MyPluginEntity)
        # 添加过滤条件，仅保留指定 ID 的插件记录
        if query.id is not None:
            my_plugins = my_plugins.filter(MyPluginEntity.id == query.id)
        # 执行删除操作
        my_plugins.delete()
        # 提交事务
        session.commit()
        # 关闭数据库会话
        session.close()
```