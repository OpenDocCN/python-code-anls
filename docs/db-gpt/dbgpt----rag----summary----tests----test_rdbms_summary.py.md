# `.\DB-GPT-src\dbgpt\rag\summary\tests\test_rdbms_summary.py`

```py
# 导入单元测试相关的模块
import unittest
# 导入列表类型的类型提示
from typing import List
# 导入 Mock 类和 patch 函数
from unittest.mock import Mock, patch

# 导入需要测试的模块或类
from dbgpt.rag.summary.rdbms_db_summary import RdbmsSummary


# 定义一个模拟的 RDBMS 连接器类，用于测试
class MockRDBMSConnector(object):
    # 模拟获取用户信息的方法
    def get_users(self):
        return "user1, user2"

    # 模拟获取授权信息的方法
    def get_grants(self):
        return "grant1, grant2"

    # 模拟获取字符集信息的方法
    def get_charset(self):
        return "utf8"

    # 模拟获取校对规则信息的方法
    def get_collation(self):
        return "utf8_general_ci"

    # 模拟获取表名列表的方法
    def get_table_names(self):
        return ["table1", "table2"]

    # 模拟获取表列信息的方法
    def get_columns(self, table_name):
        if table_name == "table1":
            return [{"name": "column1", "comment": "first column"}, {"name": "column2"}]
        return [{"name": "column1"}]

    # 模拟获取表索引信息的方法
    def get_indexes(self, table_name):
        return [{"name": "index1", "column_names": ["column1"]}]

    # 模拟获取表注释信息的方法
    def get_table_comment(self, table_name):
        return {"text": f"{table_name} comment"}


# 定义测试 RdbmsSummary 类的单元测试
class TestRdbmsSummary(unittest.TestCase):
    # 在每个测试方法执行前设置环境
    def setUp(self):
        # 创建 Mock 对象来替代本地数据库管理对象
        self.mock_local_db_manage = Mock()
        # 设置 Mock 对象的返回值为 MockRDBMSConnector 的实例
        self.mock_local_db_manage.get_connector.return_value = MockRDBMSConnector()

    # 测试 RdbmsSummary 对象的初始化
    def test_rdbms_summary_initialization(self):
        # 创建 RdbmsSummary 对象进行测试
        rdbms_summary = RdbmsSummary(
            name="test_db", type="test_type", manager=self.mock_local_db_manage
        )
        # 断言数据库名称
        self.assertEqual(rdbms_summary.name, "test_db")
        # 断言数据库类型
        self.assertEqual(rdbms_summary.type, "test_type")
        # 断言元数据中包含用户信息
        self.assertTrue("user info :user1, user2" in rdbms_summary.metadata)
        # 断言元数据中包含授权信息
        self.assertTrue("grant info:grant1, grant2" in rdbms_summary.metadata)
        # 断言元数据中包含字符集信息
        self.assertTrue("charset:utf8" in rdbms_summary.metadata)
        # 断言元数据中包含校对规则信息
        self.assertTrue("collation:utf8_general_ci" in rdbms_summary.metadata)

    # 测试表格摘要方法
    def test_table_summaries(self):
        # 创建 RdbmsSummary 对象进行测试
        rdbms_summary = RdbmsSummary(
            name="test_db", type="test_type", manager=self.mock_local_db_manage
        )
        # 获取表格摘要列表
        summaries = rdbms_summary.table_summaries()
        # 断言表格1的摘要信息
        self.assertTrue(
            "table1(column1 (first column), column2), and index keys: index1(`column1`) , and table comment: table1 comment"
            in summaries
        )
        # 断言表格2的摘要信息
        self.assertTrue(
            "table2(column1), and index keys: index1(`column1`) , and table comment: table2 comment"
            in summaries
        )


# 如果当前脚本作为主程序执行，则运行单元测试
if __name__ == "__main__":
    unittest.main()
```