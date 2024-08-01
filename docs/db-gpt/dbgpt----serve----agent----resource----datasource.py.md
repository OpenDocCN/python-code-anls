# `.\DB-GPT-src\dbgpt\serve\agent\resource\datasource.py`

```py
import dataclasses  # 导入用于创建数据类的模块
import logging  # 导入日志记录模块
from typing import Any, List, Optional, Type, Union, cast  # 导入类型提示相关的模块

from dbgpt._private.config import Config  # 从私有模块导入Config类
from dbgpt.agent.resource.database import DBParameters, RDBMSConnectorResource  # 导入数据库参数和资源连接器
from dbgpt.util import ParameterDescription  # 导入参数描述工具类

CFG = Config()  # 创建Config类的实例并赋值给CFG

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclasses.dataclass
class DatasourceDBParameters(DBParameters):
    """The DB parameters for the datasource."""

    db_name: str = dataclasses.field(metadata={"help": "DB name"})  # 数据类字段：数据库名称

    @classmethod
    def _resource_version(cls) -> str:
        """Return the resource version."""
        return "v1"  # 返回资源版本号为v1

    @classmethod
    def to_configurations(
        cls,
        parameters: Type["DatasourceDBParameters"],
        version: Optional[str] = None,
    ) -> Any:
        """Convert the parameters to configurations."""
        conf: List[ParameterDescription] = cast(
            List[ParameterDescription], super().to_configurations(parameters)
        )  # 转换参数为配置项列表
        version = version or cls._resource_version()  # 如果版本号为None，则使用类属性中的版本号
        if version != "v1":
            return conf  # 如果版本不是v1，则直接返回配置项列表
        # Compatible with old version
        for param in conf:
            if param.param_name == "db_name":
                return param.valid_values or []  # 返回参数的有效值列表或空列表
        return []  # 如果没有找到匹配的参数，返回空列表作为兜底

    @classmethod
    def from_dict(
        cls, data: dict, ignore_extra_fields: bool = True
    ) -> "DatasourceDBParameters":
        """Create a new instance from a dictionary."""
        copied_data = data.copy()  # 复制输入的字典数据
        if "db_name" not in copied_data and "value" in copied_data:
            copied_data["db_name"] = copied_data.pop("value")  # 如果字典中不包含db_name但包含value，则使用value替换为db_name
        return super().from_dict(copied_data, ignore_extra_fields=ignore_extra_fields)  # 调用父类方法创建新实例


class DatasourceResource(RDBMSConnectorResource):
    def __init__(self, name: str, db_name: Optional[str] = None, **kwargs):
        conn = CFG.local_db_manager.get_connector(db_name)  # 使用Config实例的本地数据库管理器获取连接器
        super().__init__(name, connector=conn, db_name=db_name, **kwargs)  # 调用父类构造函数初始化

    @classmethod
    def resource_parameters_class(cls) -> Type[DatasourceDBParameters]:
        dbs = CFG.local_db_manager.get_db_list()  # 获取本地数据库列表
        results = [db["db_name"] for db in dbs]  # 提取数据库名称列表

        @dataclasses.dataclass
        class _DynDBParameters(DatasourceDBParameters):
            db_name: str = dataclasses.field(
                metadata={"help": "DB name", "valid_values": results}
            )  # 数据类字段：数据库名称，包含有效值列表

        return _DynDBParameters  # 返回动态生成的数据库参数类

    def get_schema_link(
        self, db: str, question: Optional[str] = None
    ):  # 方法定义：获取模式链接，包含数据库名称和可选问题参数
    ) -> Union[str, List[str]]:
        """声明函数的返回类型为字符串或字符串列表"""
        """Return the schema link of the database."""
        try:
            # 尝试导入数据库摘要客户端模块
            from dbgpt.rag.summary.db_summary_client import DBSummaryClient
        except ImportError:
            # 如果导入失败，抛出值错误异常
            raise ValueError("Could not import DBSummaryClient. ")
        # 使用系统应用配置创建数据库摘要客户端对象
        client = DBSummaryClient(system_app=CFG.SYSTEM_APP)
        # 初始化表信息为空
        table_infos = None
        try:
            # 尝试获取数据库摘要信息
            table_infos = client.get_db_summary(
                db,
                question,
                CFG.KNOWLEDGE_SEARCH_TOP_SIZE,
            )
        except Exception as e:
            # 捕获异常并记录日志
            logger.warning(f"db summary find error!{str(e)}")
        # 如果未能获取到表信息
        if not table_infos:
            # 获取本地数据库连接器
            conn = CFG.local_db_manager.get_connector(db)
            # 获取简单的表信息
            table_infos = conn.table_simple_info()

        # 返回表信息
        return table_infos
```