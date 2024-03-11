# `.\Langchain-Chatchat\server\db\models\base.py`

```
# 导入 datetime 模块中的 datetime 类
# 从 sqlalchemy 模块中导入 Column、DateTime、String、Integer 类
class BaseModel:
    """
    基础模型
    """
    # 定义 id 字段，为整数类型，作为主键，可索引，注释为"主键ID"
    id = Column(Integer, primary_key=True, index=True, comment="主键ID")
    # 定义 create_time 字段，为日期时间类型，默认为当前时间，注释为"创建时间"
    create_time = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    # 定义 update_time 字段，为日期时间类型，默认为 None，更新时为当前时间，注释为"更新时间"
    update_time = Column(DateTime, default=None, onupdate=datetime.utcnow, comment="更新时间")
    # 定义 create_by 字段，为字符串类型，默认为 None，注释为"创建者"
    create_by = Column(String, default=None, comment="创建者")
    # 定义 update_by 字段，为字符串类型，默认为 None，注释为"更新者"
    update_by = Column(String, default=None, comment="更新者")
```