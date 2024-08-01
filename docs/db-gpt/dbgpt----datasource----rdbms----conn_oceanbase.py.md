# `.\DB-GPT-src\dbgpt\datasource\rdbms\conn_oceanbase.py`

```py
"""Ocean base connect."""

# 导入日志模块
import logging

# 导入 RDBMSConnector 类
from dbgpt.datasource.rdbms.base import RDBMSConnector

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class OceanBaseConnect(RDBMSConnector):
    """Connect Oceanbase Database fetch MetaData.

    Args:
    Usage:
    """

    # 数据库类型
    db_type: str = "oceanbase"
    # 数据库方言
    db_dialect: str = "mysql"
    # 数据库驱动程序
    driver: str = "mysql+ob"

    # 默认数据库列表
    default_db = ["information_schema", "performance_schema", "sys", "mysql"]

    def get_users(self):
        """Get_users."""
        # 返回空列表，表示暂无用户信息
        return []

    def get_grants(self):
        """Get_grants."""
        # 返回空列表，表示暂无权限信息
        return []

    def get_collation(self):
        """Get collation."""
        # 返回数据库字符集校对规则
        return "UTF-8"

    def get_charset(self):
        """Get_charset."""
        # 返回数据库默认字符集
        return "UTF-8"
```