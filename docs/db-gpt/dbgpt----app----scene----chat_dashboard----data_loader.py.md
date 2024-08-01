# `.\DB-GPT-src\dbgpt\app\scene\chat_dashboard\data_loader.py`

```py
# 导入日期时间模块
import datetime
# 导入日志记录模块
import logging
# 导入 Decimal 类型
from decimal import Decimal
# 导入 List 类型
from typing import List

# 导入配置对象 Config
from dbgpt._private.config import Config
# 导入 ValueItem 类型，用于数据准备
from dbgpt.app.scene.chat_dashboard.data_preparation.report_schma import ValueItem

# 创建 Config 实例
CFG = Config()
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


class DashboardDataLoader:
    def get_sql_value(self, db_conn, chart_sql: str):
        # 调用 db_conn 对象的 query_ex 方法执行 SQL 查询，并返回结果
        return db_conn.query_ex(chart_sql)

    def get_chart_values_by_conn(self, db_conn, chart_sql: str):
        # 调用 db_conn 对象的 query_ex 方法执行 SQL 查询，获取字段名和数据
        field_names, datas = db_conn.query_ex(chart_sql)
        # 调用本类的 get_chart_values_by_data 方法处理字段名和数据，并传入原始 SQL 查询语句
        return self.get_chart_values_by_data(field_names, datas, chart_sql)

    def get_chart_values_by_db(self, db_name: str, chart_sql: str):
        # 记录日志，标记正在获取数据库 db_name 中的 chart_sql 数据
        logger.info(f"get_chart_values_by_db:{db_name},{chart_sql}")
        # 获取本地数据库管理器 CFG 的连接器对象，并连接到指定数据库 db_name
        db_conn = CFG.local_db_manager.get_connector(db_name)
        # 调用 get_chart_values_by_conn 方法获取指定数据库中 chart_sql 的图表数据
        return self.get_chart_values_by_conn(db_conn, chart_sql)
```