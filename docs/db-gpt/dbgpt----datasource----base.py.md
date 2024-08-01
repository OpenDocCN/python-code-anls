# `.\DB-GPT-src\dbgpt\datasource\base.py`

```py
"""Base class for all connectors."""
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Dict, Iterable, List, Optional, Tuple  # 导入类型提示


class BaseConnector(ABC):  # 定义一个抽象基类BaseConnector，继承自ABC（抽象基类）

    db_type: str = "__abstract__db_type__"  # 类属性，数据库类型，默认为抽象类型
    driver: str = ""  # 类属性，数据库驱动，默认为空字符串

    @abstractmethod
    def get_table_names(self) -> Iterable[str]:
        """Get all table names."""
        raise NotImplementedError("Current connector does not support get_table_names")

    @abstractmethod
    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        r"""Get table info about specified table.

        Returns:
            str: Table information joined by "\n\n"
        """
        raise NotImplementedError("Current connector does not support get_table_info")

    @abstractmethod
    def get_index_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get index info about specified table.

        Args:
             table_names (Optional[List[str]]): table names
        """
        raise NotImplementedError("Current connector does not support get_index_info")

    @abstractmethod
    def get_example_data(self, table: str, count: int = 3):
        """Get example data about specified table.

        Not used now.

        Args:
            table (str): table name
            count (int): example data count
        """
        raise NotImplementedError("Current connector does not support get_example_data")

    @abstractmethod
    def get_database_names(self) -> List[str]:
        """Return database names.

        Examples:
            .. code-block:: python

                print(conn.get_database_names())
                # ['db1', 'db2']

        Returns:
            List[str]: database list
        """
        raise NotImplementedError(
            "Current connector does not support get_database_names"
        )

    @abstractmethod
    def get_table_comments(self, db_name: str) -> List[Tuple[str, str]]:
        """Get table comments.

        Args:
            db_name (str): database name

        Returns:
            List[Tuple[str, str]]: Table comments, first element is table name, second
                element is comment
        """
        raise NotImplementedError(
            "Current connector does not support get_table_comments"
        )

    @abstractmethod
    def get_table_comment(self, table_name: str) -> Dict:
        """Return table comment with table name.

        Args:
            table_name (str): table name

        Returns:
            comment: Dict, which contains text: Optional[str], eg:["text": "comment"]
        """
        raise NotImplementedError(
            "Current connector does not support get_table_comment"
        )
    @abstractmethod
    def get_columns(self, table_name: str) -> List[Dict]:
        """Return columns with table name.

        Args:
            table_name (str): table name

        Returns:
            List[Dict]: columns of table, which contains name: str, type: str,
                default_expression: str, is_in_primary_key: bool, comment: str
                eg: [{'name': 'id', 'type': 'int', 'default_expression': '',
                'is_in_primary_key': True, 'comment': 'id'}, ...]
        """
        # 抛出未实现错误，因为当前连接器不支持获取表的列信息
        raise NotImplementedError("Current connector does not support get_columns")

    def get_column_comments(self, db_name: str, table_name: str):
        """Return column comments with db name and table name.

        Args:
            db_name (str): database name
            table_name (_type_): _description_
        """
        # 抛出未实现错误，因为当前连接器不支持获取列的注释信息
        raise NotImplementedError(
            "Current connector does not support get_column_comments"
        )

    @abstractmethod
    def run(self, command: str, fetch: str = "all") -> List:
        """Execute sql command.

        Args:
            command (str): sql command
            fetch (str): fetch type

        Returns:
            List: result list
        """
        # 抽象方法，子类需要实现具体的 SQL 命令执行逻辑
        pass

    def run_to_df(self, command: str, fetch: str = "all"):
        """Execute sql command and return result as dataframe.

        Args:
            command (str): sql command
            fetch (str): fetch type

        Returns:
            DataFrame: result dataframe
        """
        # 抛出未实现错误，因为当前连接器不支持将 SQL 命令执行结果转换为 DataFrame
        raise NotImplementedError("Current connector does not support run_to_df")

    def get_users(self) -> List[Tuple[str, str]]:
        """Return user information.

        Returns:
            List[Tuple[str, str]]: user list, which contains username and host
        """
        # 返回空列表，因为当前连接器不支持获取用户信息
        return []

    def get_grants(self) -> List[Tuple]:
        """Return grant information.

        Examples:
            .. code-block:: python

                print(conn.get_grants())
                # [(('GRANT SELECT, INSERT, UPDATE, DROP ROLE ON *.* TO `root`@`%`
                # WITH GRANT OPTION',)]

        Returns:
            List[Tuple]: grant list, which contains grant information
        """
        # 返回空列表，因为当前连接器不支持获取授权信息
        return []

    def get_collation(self) -> Optional[str]:
        """Return collation.

        Returns:
            Optional[str]: collation
        """
        # 返回 None，因为当前连接器不支持获取排序规则信息
        return None

    def get_charset(self) -> str:
        """Get character_set of current database."""
        # 返回字符集信息，当前连接器默认使用 UTF-8
        return "utf-8"

    def get_fields(self, table_name: str) -> List[Tuple]:
        """Get column fields about specified table.

        Args:
            table_name (str): table name

        Returns:
            List[Tuple]: column fields, which contains column name, column type,
                column default, is nullable, column comment
        """
        # 抛出未实现错误，因为当前连接器不支持获取表的字段信息
        raise NotImplementedError("Current connector does not support get_fields")
    def get_simple_fields(self, table_name: str) -> List[Tuple]:
        """Return simple fields about specified table.

        Args:
            table_name (str): table name

        Returns:
            List[Tuple]: simple fields, which contains column name, column type,
                is nullable, column key, default value, extra.
        """
        # 调用父类方法获取指定表的简单字段信息并返回
        return self.get_fields(table_name)

    def get_show_create_table(self, table_name: str) -> str:
        """Return show create table about specified table.

        Returns:
            str: show create table
        """
        # 抛出未实现异常，因为当前连接器不支持获取表的创建语句
        raise NotImplementedError(
            "Current connector does not support get_show_create_table"
        )

    def get_indexes(self, table_name: str) -> List[Dict]:
        """Return indexes about specified table.

        Args:
            table_name (str): table name

        Returns:
            List[Dict], eg:[{'name': 'idx_key', 'column_names': ['id']}]
        """
        # 抛出未实现异常，因为当前连接器不支持获取表的索引信息
        raise NotImplementedError("Current connector does not support get_indexes")

    @classmethod
    def is_normal_type(cls) -> bool:
        """Return whether the connector is a normal type."""
        # 返回连接器是否为普通类型，这里始终返回 True
        return True

    @classmethod
    def is_graph_type(cls) -> bool:
        """Return whether the connector is a graph database connector."""
        # 返回连接器是否为图数据库连接器，这里始终返回 False
        return False
```