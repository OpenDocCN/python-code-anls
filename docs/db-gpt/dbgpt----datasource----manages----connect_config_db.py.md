# `.\DB-GPT-src\dbgpt\datasource\manages\connect_config_db.py`

```py
"""DB Model for connect_config."""

import logging  # 导入日志模块
from typing import Any, Dict, Optional, Union  # 导入类型提示相关模块

from sqlalchemy import Column, Index, Integer, String, Text, UniqueConstraint, text  # 导入 SQLAlchemy 相关模块

from dbgpt.serve.datasource.api.schemas import (
    DatasourceServeRequest,
    DatasourceServeResponse,
)  # 导入数据源服务请求和响应模块

from dbgpt.storage.metadata import BaseDao, Model  # 导入基础 DAO 和模型类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ConnectConfigEntity(Model):
    """DB connector config entity."""

    __tablename__ = "connect_config"  # 定义数据库表名

    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="autoincrement id"
    )  # 定义自增的主键 id

    db_type = Column(String(255), nullable=False, comment="db type")  # 数据库类型字段
    db_name = Column(String(255), nullable=False, comment="db name")  # 数据库名称字段
    db_path = Column(String(255), nullable=True, comment="file db path")  # 文件型数据库路径字段
    db_host = Column(String(255), nullable=True, comment="db connect host(not file db)")  # 数据库连接主机字段（非文件型数据库）
    db_port = Column(String(255), nullable=True, comment="db connect port(not file db)")  # 数据库连接端口字段（非文件型数据库）
    db_user = Column(String(255), nullable=True, comment="db user")  # 数据库用户名字段
    db_pwd = Column(String(255), nullable=True, comment="db password")  # 数据库密码字段
    comment = Column(Text, nullable=True, comment="db comment")  # 数据库备注字段
    sys_code = Column(String(128), index=True, nullable=True, comment="System code")  # 系统代码字段，建立索引加快查询速度

    __table_args__ = (
        UniqueConstraint("db_name", name="uk_db"),  # 添加唯一约束，保证数据库名称的唯一性
        Index("idx_q_db_type", "db_type"),  # 创建索引以优化按数据库类型查询的性能
    )


class ConnectConfigDao(BaseDao):
    """DB connector config dao."""

    def get_by_names(self, db_name: str) -> Optional[ConnectConfigEntity]:
        """Get db connect info by name."""
        session = self.get_raw_session()  # 获取原始数据库会话
        db_connect = session.query(ConnectConfigEntity)  # 创建查询 ConnectConfigEntity 的查询对象
        db_connect = db_connect.filter(ConnectConfigEntity.db_name == db_name)  # 根据数据库名称筛选结果
        result = db_connect.first()  # 获取第一个匹配的结果
        session.close()  # 关闭数据库会话
        return result  # 返回查询结果的第一个对象或 None

    def add_url_db(
        self,
        db_name,
        db_type,
        db_host: str,
        db_port: int,
        db_user: str,
        db_pwd: str,
        comment: str = "",
    ):
        """
        Add db connect info.

        Args:
            db_name: db name
            db_type: db type
            db_host: db host
            db_port: db port
            db_user: db user
            db_pwd: db password
            comment: comment
        """
        try:
            # 获取原始数据库会话
            session = self.get_raw_session()

            # 导入 SQLAlchemy 的 text 模块
            from sqlalchemy import text

            # 构建 SQL 插入语句
            insert_statement = text(
                """
                INSERT INTO connect_config (
                    db_name, db_type, db_path, db_host, db_port, db_user, db_pwd,
                    comment) VALUES (:db_name, :db_type, :db_path, :db_host, :db_port
                    , :db_user, :db_pwd, :comment
                )
            """
            )

            # 设置参数字典
            params = {
                "db_name": db_name,
                "db_type": db_type,
                "db_path": "",
                "db_host": db_host,
                "db_port": db_port,
                "db_user": db_user,
                "db_pwd": db_pwd,
                "comment": comment,
            }

            # 执行 SQL 插入操作
            session.execute(insert_statement, params)
            # 提交事务
            session.commit()
            # 关闭数据库会话
            session.close()
        except Exception as e:
            # 捕获异常并记录警告日志
            logger.warning("add db connect info error！" + str(e))

    def update_db_info(
        self,
        db_name,
        db_type,
        db_path: str = "",
        db_host: str = "",
        db_port: int = 0,
        db_user: str = "",
        db_pwd: str = "",
        comment: str = "",
    ):
        """
        Update db connect info.
        """
        # 获取旧的数据库配置信息
        old_db_conf = self.get_db_config(db_name)
        if old_db_conf:
            try:
                # 获取原始数据库会话
                session = self.get_raw_session()

                # 根据是否提供 db_path 构建不同的更新 SQL 语句
                if not db_path:
                    update_statement = text(
                        f"UPDATE connect_config set db_type='{db_type}', "
                        f"db_host='{db_host}', db_port={db_port}, db_user='{db_user}', "
                        f"db_pwd='{db_pwd}', comment='{comment}' where "
                        f"db_name='{db_name}'"
                    )
                else:
                    update_statement = text(
                        f"UPDATE connect_config set db_type='{db_type}', "
                        f"db_path='{db_path}', comment='{comment}' where "
                        f"db_name='{db_name}'"
                    )

                # 执行 SQL 更新操作
                session.execute(update_statement)
                # 提交事务
                session.commit()
                # 关闭数据库会话
                session.close()
            except Exception as e:
                # 捕获异常并记录警告日志
                logger.warning("edit db connect info error！" + str(e))
            return True
        # 如果没有找到旧的数据库配置信息，抛出 ValueError 异常
        raise ValueError(f"{db_name} not have config info!")
    def add_file_db(self, db_name, db_type, db_path: str, comment: str = ""):
        """Add file db connect info."""
        try:
            # 获取一个数据库会话对象
            session = self.get_raw_session()
            # 准备插入数据库的 SQL 语句，使用 text 对象包装多行 SQL 语句
            insert_statement = text(
                """
                INSERT INTO connect_config(
                    db_name, db_type, db_path, db_host, db_port, db_user, db_pwd,
                    comment) VALUES (
                    :db_name, :db_type, :db_path, :db_host, :db_port, :db_user, :db_pwd
                    , :comment
                )
            """
            )
            # 定义插入 SQL 语句中的参数
            params = {
                "db_name": db_name,
                "db_type": db_type,
                "db_path": db_path,
                "db_host": "",
                "db_port": 0,
                "db_user": "",
                "db_pwd": "",
                "comment": comment,
            }

            # 执行 SQL 插入操作
            session.execute(insert_statement, params)

            # 提交事务
            session.commit()
            # 关闭数据库会话
            session.close()
        except Exception as e:
            # 捕获异常并记录警告日志
            logger.warning("add db connect info error！" + str(e))

    def get_db_config(self, db_name):
        """Return db connect info by name."""
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        if db_name:
            # 准备查询数据库的 SQL 语句，使用 text 对象包装多行 SQL 语句
            select_statement = text(
                """
                SELECT
                    *
                FROM
                    connect_config
                WHERE
                    db_name = :db_name
            """
            )
            # 定义查询 SQL 语句中的参数
            params = {"db_name": db_name}
            # 执行 SQL 查询操作
            result = session.execute(select_statement, params)

        else:
            # 如果 db_name 为空，抛出数值错误异常
            raise ValueError("Cannot get database by name" + db_name)

        # 记录信息日志，显示查询结果
        logger.info(f"Result: {result}")
        # 获取查询结果中的字段名列表
        fields = [field[0] for field in result.cursor.description]
        # 初始化一个空字典用于存储查询结果的第一行数据
        row_dict = {}
        # 获取查询结果的第一行数据
        row_1 = list(result.cursor.fetchall()[0])
        # 将查询结果中的字段名和对应的数据整合成字典
        for i, field in enumerate(fields):
            row_dict[field] = row_1[i]
        # 返回包含查询结果的字典
        return row_dict

    def get_db_list(self):
        """Get db list."""
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 执行 SQL 查询操作，获取所有连接配置的数据
        result = session.execute(text("SELECT *  FROM connect_config"))

        # 获取查询结果中的字段名列表
        fields = [field[0] for field in result.cursor.description]
        # 初始化一个空列表，用于存储查询结果的所有行数据
        data = []
        # 遍历查询结果的每一行数据
        for row in result.cursor.fetchall():
            # 初始化一个空字典，用于存储每一行数据的字段名和对应的数据
            row_dict = {}
            # 将每一行数据的字段名和对应的数据整合成字典
            for i, field in enumerate(fields):
                row_dict[field] = row[i]
            # 将整合好的行数据字典添加到 data 列表中
            data.append(row_dict)
        # 返回包含所有查询结果的列表
        return data

    def delete_db(self, db_name):
        """Delete db connect info."""
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        # 准备执行删除操作的 SQL 语句，使用 text 对象包装多行 SQL 语句
        delete_statement = text("""DELETE FROM connect_config where db_name=:db_name""")
        # 定义删除 SQL 语句中的参数
        params = {"db_name": db_name}
        # 执行 SQL 删除操作
        session.execute(delete_statement, params)
        # 提交事务
        session.commit()
        # 关闭数据库会话
        session.close()
        # 返回删除成功的标志
        return True

    def from_request(
        self, request: Union[DatasourceServeRequest, Dict[str, Any]]
    ) -> ConnectConfigEntity:
        """Convert the request to an entity.

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): The request

        Returns:
            T: The entity
        """
        # 如果请求是 DatasourceServeRequest 类型，则转换为字典
        request_dict = (
            request.dict() if isinstance(request, DatasourceServeRequest) else request
        )
        # 使用转换后的字典创建 ConnectConfigEntity 实例
        entity = ConnectConfigEntity(**request_dict)
        return entity

    def to_request(self, entity: ConnectConfigEntity) -> DatasourceServeRequest:
        """Convert the entity to a request.

        Args:
            entity (T): The entity

        Returns:
            REQ: The request
        """
        # 使用 ConnectConfigEntity 实例的属性创建 DatasourceServeRequest 实例
        return DatasourceServeRequest(
            id=entity.id,
            db_type=entity.db_type,
            db_name=entity.db_name,
            db_path=entity.db_path,
            db_host=entity.db_host,
            db_port=entity.db_port,
            db_user=entity.db_user,
            db_pwd=entity.db_pwd,
            comment=entity.comment,
        )

    def to_response(self, entity: ConnectConfigEntity) -> DatasourceServeResponse:
        """Convert the entity to a response.

        Args:
            entity (T): The entity

        Returns:
            REQ: The request
        """
        # 使用 ConnectConfigEntity 实例的属性创建 DatasourceServeResponse 实例
        return DatasourceServeResponse(
            id=entity.id,
            db_type=entity.db_type,
            db_name=entity.db_name,
            db_path=entity.db_path,
            db_host=entity.db_host,
            db_port=entity.db_port,
            db_user=entity.db_user,
            db_pwd=entity.db_pwd,
            comment=entity.comment,
        )
```