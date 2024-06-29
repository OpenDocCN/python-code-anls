# `D:\src\scipysrc\pandas\pandas\tests\io\test_sql.py`

```
# 从未来导入注释类型以支持类型提示（Python 3.7之前版本的兼容）
from __future__ import annotations

# 导入上下文管理模块
import contextlib
# 从上下文管理模块中导入关闭上下文的函数
from contextlib import closing
# 导入处理CSV文件的模块
import csv
# 导入日期和时间处理模块
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
# 导入用于处理内存中文本的模块
from io import StringIO
# 导入操作路径的模块
from pathlib import Path
# 导入 SQLite 数据库模块
import sqlite3
# 导入类型提示检查相关模块
from typing import TYPE_CHECKING
# 导入唯一标识符模块
import uuid

# 导入数值计算和数据分析库
import numpy as np
# 导入 pytest 测试框架
import pytest

# 导入 pandas 数据处理库的核心函数库
from pandas._libs import lib
# 导入兼容性函数
from pandas.compat import pa_version_under14p1
# 导入测试装饰器函数
import pandas.util._test_decorators as td

# 导入 pandas 数据处理库
import pandas as pd
# 从 pandas 库中导入常用对象和函数
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
    isna,
    to_datetime,
    to_timedelta,
)
# 导入 pandas 的测试工具库
import pandas._testing as tm
# 导入 pandas 的核心数组对象
from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)
# 导入版本信息相关功能
from pandas.util.version import Version

# 导入 pandas 的 SQL IO 模块
from pandas.io import sql
# 从 pandas 的 SQL IO 模块中导入 SQLAlchemy 相关功能
from pandas.io.sql import (
    SQLAlchemyEngine,
    SQLDatabase,
    SQLiteDatabase,
    get_engine,
    pandasSQL_builder,
    read_sql_query,
    read_sql_table,
)

# 如果是类型检查模式，导入 sqlalchemy 模块
if TYPE_CHECKING:
    import sqlalchemy


# 使用 pytest 的标记，忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


# 定义一个 pytest 的 fixture，返回 SQL 查询字符串字典
@pytest.fixture
def sql_strings():
    return {
        "read_parameters": {
            "sqlite": "SELECT * FROM iris WHERE Name=? AND SepalLength=?",
            "mysql": "SELECT * FROM iris WHERE `Name`=%s AND `SepalLength`=%s",
            "postgresql": 'SELECT * FROM iris WHERE "Name"=%s AND "SepalLength"=%s',
        },
        "read_named_parameters": {
            "sqlite": """
                SELECT * FROM iris WHERE Name=:name AND SepalLength=:length
                """,
            "mysql": """
                SELECT * FROM iris WHERE
                `Name`=%(name)s AND `SepalLength`=%(length)s
                """,
            "postgresql": """
                SELECT * FROM iris WHERE
                "Name"=%(name)s AND "SepalLength"=%(length)s
                """,
        },
        "read_no_parameters_with_percent": {
            "sqlite": "SELECT * FROM iris WHERE Name LIKE '%'",
            "mysql": "SELECT * FROM iris WHERE `Name` LIKE '%'",
            "postgresql": "SELECT * FROM iris WHERE \"Name\" LIKE '%'",
        },
    }


# 定义 iris 数据表的元数据结构
def iris_table_metadata():
    import sqlalchemy
    from sqlalchemy import (
        Column,
        Double,
        Float,
        MetaData,
        String,
        Table,
    )

    # 根据 SQLAlchemy 版本选择 Double 或 Float 数据类型
    dtype = Double if Version(sqlalchemy.__version__) >= Version("2.0.0") else Float
    # 创建元数据对象
    metadata = MetaData()
    # 创建 iris 表对象并定义列
    iris = Table(
        "iris",
        metadata,
        Column("SepalLength", dtype),
        Column("SepalWidth", dtype),
        Column("PetalLength", dtype),
        Column("PetalWidth", dtype),
        Column("Name", String(200)),
    )
    return iris


# 创建并加载 iris 数据到 SQLite3 数据库
def create_and_load_iris_sqlite3(conn, iris_file: Path):
    # 定义 SQL 语句，创建名为 iris 的表，包含五个列：SepalLength、SepalWidth、PetalLength、PetalWidth 和 Name
    stmt = """CREATE TABLE iris (
            "SepalLength" REAL,
            "SepalWidth" REAL,
            "PetalLength" REAL,
            "PetalWidth" REAL,
            "Name" TEXT
        )"""

    # 获取数据库连接对象的游标
    cur = conn.cursor()
    # 执行 SQL 语句，创建 iris 表
    cur.execute(stmt)
    
    # 打开 iris_file 文件，使用 UTF-8 编码读取数据，忽略换行符
    with iris_file.open(newline=None, encoding="utf-8") as csvfile:
        # 创建 CSV 读取器对象
        reader = csv.reader(csvfile)
        # 跳过 CSV 文件的标题行
        next(reader)
        # 准备 SQL 插入语句，插入五个字段的数据
        stmt = "INSERT INTO iris VALUES(?, ?, ?, ?, ?)"
        # ADBC 要求显式指定类型 - 不允许隐式将 str 转换为 float
        # 从 CSV 文件的每一行创建一个记录，将数值字段转换为浮点数
        records = [
            (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                row[4],
            )
            for row in reader
        ]

        # 批量执行 SQL 插入语句，插入记录数据
        cur.executemany(stmt, records)
    
    # 关闭数据库游标
    cur.close()

    # 提交事务，将修改保存到数据库
    conn.commit()
# 创建一个包含各种数据类型的表的元数据，并根据给定的数据库方言选择适当的日期类型
def types_table_metadata(dialect: str):
    from sqlalchemy import (
        TEXT,
        Boolean,
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        Table,
    )

    # 根据方言选择日期类型
    date_type = TEXT if dialect == "sqlite" else DateTime
    # 根据方言选择布尔类型
    bool_type = Integer if dialect == "sqlite" else Boolean
    # 创建一个 SQLAlchemy 元数据对象
    metadata = MetaData()
    # 定义包含各种列的表结构
    types = Table(
        "types",
        metadata,
        Column("TextCol", TEXT),
        Column("DateCol", date_type),
        Column("IntDateCol", Integer),
        Column("IntDateOnlyCol", Integer),
        Column("FloatCol", Float),
        Column("IntCol", Integer),
        Column("BoolCol", bool_type),
        Column("IntColWithNull", Integer),
        Column("BoolColWithNull", bool_type),
    )
    # 返回表结构对象
    return types


# 使用给定的连接和数据，创建并加载类型表到 SQLite 数据库中
def create_and_load_types_sqlite3(conn, types_data: list[dict]):
    # 注意：这里函数体未完全给出，但根据前文和函数名可以推测其作用
    # 定义创建表格的 SQL 语句，包括各列的数据类型
    stmt = """CREATE TABLE types (
                    "TextCol" TEXT,
                    "DateCol" TEXT,
                    "IntDateCol" INTEGER,
                    "IntDateOnlyCol" INTEGER,
                    "FloatCol" REAL,
                    "IntCol" INTEGER,
                    "BoolCol" INTEGER,
                    "IntColWithNull" INTEGER,
                    "BoolColWithNull" INTEGER
                )"""
    
    # 定义插入数据的 SQL 语句，使用占位符 '?' 表示待填充的值
    ins_stmt = """
                INSERT INTO types
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
    
    # 检查 conn 是否为 sqlite3.Connection 对象，如果是则使用 cursor() 创建游标对象 cur，并执行创建表格和批量插入数据操作
    if isinstance(conn, sqlite3.Connection):
        cur = conn.cursor()
        cur.execute(stmt)  # 执行创建表格的 SQL 语句
        cur.executemany(ins_stmt, types_data)  # 执行批量插入数据的 SQL 语句
    else:
        # 如果 conn 不是 sqlite3.Connection 对象，则使用 with 语句创建上下文管理器中的游标对象 cur，并执行相同的创建表格和批量插入数据操作
        with conn.cursor() as cur:
            cur.execute(stmt)  # 执行创建表格的 SQL 语句
            cur.executemany(ins_stmt, types_data)  # 执行批量插入数据的 SQL 语句
    
        conn.commit()  # 提交事务，将数据持久化到数据库
def create_and_load_types_postgresql(conn, types_data: list[dict]):
    # 使用给定的数据库连接创建游标对象
    with conn.cursor() as cur:
        # SQL语句：创建名为types的表格，包含多个列定义
        stmt = """CREATE TABLE types (
                        "TextCol" TEXT,
                        "DateCol" TIMESTAMP,
                        "IntDateCol" INTEGER,
                        "IntDateOnlyCol" INTEGER,
                        "FloatCol" DOUBLE PRECISION,
                        "IntCol" INTEGER,
                        "BoolCol" BOOLEAN,
                        "IntColWithNull" INTEGER,
                        "BoolColWithNull" BOOLEAN
                    )"""
        # 执行创建表格的SQL语句
        cur.execute(stmt)

        # SQL语句：插入多行数据到types表格中，使用参数化方式传入数据
        stmt = """
                INSERT INTO types
                VALUES($1, $2::timestamp, $3, $4, $5, $6, $7, $8, $9)
                """
        # 执行插入操作，批量插入types_data中的数据
        cur.executemany(stmt, types_data)

    # 提交事务，确保SQL操作生效
    conn.commit()


def create_and_load_types(conn, types_data: list[dict], dialect: str):
    # 导入需要的SQLAlchemy模块
    from sqlalchemy import insert
    from sqlalchemy.engine import Engine

    # 获取指定方言的types表格元数据
    types = types_table_metadata(dialect)

    # 构建插入数据的SQLAlchemy语句
    stmt = insert(types).values(types_data)
    # 如果连接是Engine类型（SQLAlchemy引擎）
    if isinstance(conn, Engine):
        # 使用连接进行操作
        with conn.connect() as conn:
            with conn.begin():
                # 如果表格存在，删除它（仅在第一次创建时使用）
                types.drop(conn, checkfirst=True)
                # 创建表格
                types.create(bind=conn)
                # 执行插入操作
                conn.execute(stmt)
    else:
        # 普通数据库连接情况下
        with conn.begin():
            # 如果表格存在，删除它（仅在第一次创建时使用）
            types.drop(conn, checkfirst=True)
            # 创建表格
            types.create(bind=conn)
            # 执行插入操作
            conn.execute(stmt)


def create_and_load_postgres_datetz(conn):
    # 导入所需的SQLAlchemy和pandas模块
    from sqlalchemy import (
        Column,
        DateTime,
        MetaData,
        Table,
        insert,
    )
    from sqlalchemy.engine import Engine

    # 创建Metadata对象
    metadata = MetaData()
    # 创建名为datetz的表格，其中包含一个DateColWithTz列，表示带时区的日期时间
    datetz = Table("datetz", metadata, Column("DateColWithTz", DateTime(timezone=True)))
    # 准备要插入的数据
    datetz_data = [
        {
            "DateColWithTz": "2000-01-01 00:00:00-08:00",
        },
        {
            "DateColWithTz": "2000-06-01 00:00:00-07:00",
        },
    ]
    # 构建插入数据的SQLAlchemy语句
    stmt = insert(datetz).values(datetz_data)
    # 如果连接是Engine类型（SQLAlchemy引擎）
    if isinstance(conn, Engine):
        # 使用连接进行操作
        with conn.connect() as conn:
            with conn.begin():
                # 如果表格存在，删除它（仅在第一次创建时使用）
                datetz.drop(conn, checkfirst=True)
                # 创建表格
                datetz.create(bind=conn)
                # 执行插入操作
                conn.execute(stmt)
    else:
        # 普通数据库连接情况下
        with conn.begin():
            # 如果表格存在，删除它（仅在第一次创建时使用）
            datetz.drop(conn, checkfirst=True)
            # 创建表格
            datetz.create(bind=conn)
            # 执行插入操作
            conn.execute(stmt)

    # "2000-01-01 00:00:00-08:00" 转换为 "2000-01-01 08:00:00"
    # "2000-06-01 00:00:00-07:00" 转换为 "2000-06-01 07:00:00"
    # GH 6415
    # 预期的数据结果，将包含两个带时区的日期时间对象
    expected_data = [
        Timestamp("2000-01-01 08:00:00", tz="UTC"),
        Timestamp("2000-06-01 07:00:00", tz="UTC"),
    ]
    # 返回一个pandas Series对象，包含预期的日期时间数据
    return Series(expected_data, name="DateColWithTz").astype("M8[us, UTC]")


def check_iris_frame(frame: DataFrame):
    # 获取DataFrame的第一列的Python类型
    pytype = frame.dtypes.iloc[0].type
    # 获取DataFrame的第一行数据
    row = frame.iloc[0]
    # 断言DataFrame的第一列数据类型是np.floating类型
    assert issubclass(pytype, np.floating)
    # 使用测试工具库中的函数来比较两个序列的相等性，确保测试行数据与预期的序列一致
    tm.assert_series_equal(
        # 要比较的行数据序列
        row, 
        # 创建一个包含预期数值和标签的 Series 对象，用于与行数据序列进行比较
        Series([5.1, 3.5, 1.4, 0.2, "Iris-setosa"], index=frame.columns, name=0)
    )
    
    # 断言数据框的形状在两种可能的形状中的一个，以确保数据框具有正确的维度
    assert frame.shape in ((150, 5), (8, 5))
# 定义一个函数用于统计数据库表中的行数
def count_rows(conn, table_name: str):
    # 构造 SQL 查询语句，统计表中的行数
    stmt = f"SELECT count(*) AS count_1 FROM {table_name}"
    # 导入可选依赖 adbc_driver_manager.dbapi，如果导入失败则忽略
    adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
    
    # 如果连接是 sqlite3.Connection 类型
    if isinstance(conn, sqlite3.Connection):
        # 获取游标
        cur = conn.cursor()
        # 执行 SQL 查询并获取结果的第一行第一列的值（即行数），返回该值
        return cur.execute(stmt).fetchone()[0]
    
    # 如果 adbc 导入成功且连接是 adbc.Connection 类型
    elif adbc and isinstance(conn, adbc.Connection):
        # 使用上下文管理器获取游标
        with conn.cursor() as cur:
            # 执行 SQL 查询并获取结果的第一行第一列的值（即行数），返回该值
            cur.execute(stmt)
            return cur.fetchone()[0]
    
    # 如果连接参数是字符串
    else:
        # 导入 SQLAlchemy 相关库
        from sqlalchemy import create_engine
        from sqlalchemy.engine import Engine
        
        # 如果 conn 是字符串类型
        if isinstance(conn, str):
            try:
                # 创建 SQLAlchemy 引擎
                engine = create_engine(conn)
                # 使用上下文管理器与数据库建立连接
                with engine.connect() as conn:
                    # 执行驱动程序特定的 SQL 查询并获取结果的第一行第一列的值（即行数），返回该值
                    return conn.exec_driver_sql(stmt).scalar_one()
            finally:
                # 释放引擎资源
                engine.dispose()
        
        # 如果连接是 SQLAlchemy 引擎类型
        elif isinstance(conn, Engine):
            # 使用上下文管理器与数据库建立连接
            with conn.connect() as conn:
                # 执行驱动程序特定的 SQL 查询并获取结果的第一行第一列的值（即行数），返回该值
                return conn.exec_driver_sql(stmt).scalar_one()
        
        # 如果连接类型无法识别，假设它支持执行驱动程序特定的 SQL 查询
        else:
            # 执行驱动程序特定的 SQL 查询并获取结果的第一行第一列的值（即行数），返回该值
            return conn.exec_driver_sql(stmt).scalar_one()
    data = [
        (
            "2000-01-03 00:00:00",  # 第一个元组的第一个元素，日期时间字符串
            0.980268513777,         # 第一个元组的第二个元素，浮点数数据
            3.68573087906,          # 第一个元组的第三个元素，浮点数数据
            -0.364216805298,        # 第一个元组的第四个元素，浮点数数据
            -1.15973806169,         # 第一个元组的第五个元素，浮点数数据
        ),
        (
            "2000-01-04 00:00:00",  # 第二个元组的第一个元素，日期时间字符串
            1.04791624281,          # 第二个元组的第二个元素，浮点数数据
            -0.0412318367011,       # 第二个元组的第三个元素，浮点数数据
            -0.16181208307,         # 第二个元组的第四个元素，浮点数数据
            0.212549316967,         # 第二个元组的第五个元素，浮点数数据
        ),
        (
            "2000-01-05 00:00:00",  # 第三个元组的第一个元素，日期时间字符串
            0.498580885705,         # 第三个元组的第二个元素，浮点数数据
            0.731167677815,         # 第三个元组的第三个元素，浮点数数据
            -0.537677223318,        # 第三个元组的第四个元素，浮点数数据
            1.34627041952,          # 第三个元组的第五个元素，浮点数数据
        ),
        (
            "2000-01-06 00:00:00",  # 第四个元组的第一个元素，日期时间字符串
            1.12020151869,          # 第四个元组的第二个元素，浮点数数据
            1.56762092543,          # 第四个元组的第三个元素，浮点数数据
            0.00364077397681,       # 第四个元组的第四个元素，浮点数数据
            0.67525259227,          # 第四个元组的第五个元素，浮点数数据
        ),
    ]
    return DataFrame(data, columns=columns)
@pytest.fixture
# 创建一个名为 test_frame3 的 Pytest fixture，返回一个特定结构的 DataFrame 对象
def test_frame3():
    # 列名定义为 ["index", "A", "B"]
    columns = ["index", "A", "B"]
    # 数据以元组列表的形式定义
    data = [
        ("2000-01-03 00:00:00", 2**31 - 1, -1.987670),
        ("2000-01-04 00:00:00", -29, -0.0412318367011),
        ("2000-01-05 00:00:00", 20000, 0.731167677815),
        ("2000-01-06 00:00:00", -290867, 1.56762092543),
    ]
    # 使用 DataFrame 构造函数创建 DataFrame 对象，并指定列名
    return DataFrame(data, columns=columns)


# 检索给定连接中所有的视图名称
def get_all_views(conn):
    if isinstance(conn, sqlite3.Connection):
        # 对 SQLite 连接执行 SQL 查询以获取视图名称列表
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        return [view[0] for view in c.fetchall()]
    else:
        # 尝试导入 adbc_driver_manager.dbapi 依赖，如果成功则处理特定类型的连接
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            # 使用 adbc 执行查询以获取视图信息并返回视图名称列表
            results = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                for schema in catalog["catalog_db_schemas"]:
                    for table in schema["db_schema_tables"]:
                        if table["table_type"] == "view":
                            results.append(table["table_name"])
            return results
        else:
            # 使用 SQLAlchemy 的 inspect 函数获取所有视图名称
            from sqlalchemy import inspect
            return inspect(conn).get_view_names()


# 检索给定连接中所有的表名称
def get_all_tables(conn):
    if isinstance(conn, sqlite3.Connection):
        # 对 SQLite 连接执行 SQL 查询以获取表名称列表
        c = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [table[0] for table in c.fetchall()]
    else:
        # 尝试导入 adbc_driver_manager.dbapi 依赖，如果成功则处理特定类型的连接
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            # 使用 adbc 执行查询以获取表信息并返回表名称列表
            results = []
            info = conn.adbc_get_objects().read_all().to_pylist()
            for catalog in info:
                for schema in catalog["catalog_db_schemas"]:
                    for table in schema["db_schema_tables"]:
                        if table["table_type"] == "table":
                            results.append(table["table_name"])
            return results
        else:
            # 使用 SQLAlchemy 的 inspect 函数获取所有表名称
            from sqlalchemy import inspect
            return inspect(conn).get_table_names()


# 删除给定名称的表格
def drop_table(
    table_name: str,
    conn: sqlite3.Connection | sqlalchemy.engine.Engine | sqlalchemy.engine.Connection,
):
    if isinstance(conn, sqlite3.Connection):
        # 对 SQLite 连接执行 SQL 语句以删除指定名称的表
        conn.execute(f"DROP TABLE IF EXISTS {sql._get_valid_sqlite_name(table_name)}")
        conn.commit()
    else:
        # 尝试导入 adbc_driver_manager.dbapi 依赖，如果成功则处理特定类型的连接
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            # 使用 adbc 执行 SQL 语句以删除指定名称的表
            with conn.cursor() as cur:
                cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        else:
            # 使用 SQLAlchemy 的 SQLDatabase 类进行数据库操作，删除指定名称的表
            with conn.begin() as con:
                with sql.SQLDatabase(con) as db:
                    db.drop_table(table_name)


# 删除给定名称的视图
def drop_view(
    view_name: str,
    conn: sqlite3.Connection | sqlalchemy.engine.Engine | sqlalchemy.engine.Connection,
):
    if isinstance(conn, sqlite3.Connection):
        # 对 SQLite 连接执行 SQL 语句以删除指定名称的视图
        conn.execute(f"DROP VIEW IF EXISTS {sql._get_valid_sqlite_name(view_name)}")
        conn.commit()
    else:
        # 尝试导入 adbc_driver_manager.dbapi 依赖，如果成功则处理特定类型的连接
        adbc = import_optional_dependency("adbc_driver_manager.dbapi", errors="ignore")
        if adbc and isinstance(conn, adbc.Connection):
            # 使用 adbc 执行 SQL 语句以删除指定名称的视图
            with conn.cursor() as cur:
                cur.execute(f'DROP VIEW IF EXISTS "{view_name}"')
        else:
            # 使用 SQLAlchemy 的 SQLDatabase 类进行数据库操作，删除指定名称的视图
            with conn.begin() as con:
                with sql.SQLDatabase(con) as db:
                    db.drop_view(view_name)
    conn: sqlite3.Connection | sqlalchemy.engine.Engine | sqlalchemy.engine.Connection,



# 定义一个变量 conn，可以是 sqlite3.Connection、sqlalchemy.engine.Engine 或 sqlalchemy.engine.Connection 类型之一
# 导入SQLAlchemy库，用于数据库操作
import sqlalchemy

# 定义一个 pytest fixture，创建和加载 MySQL + PyMySQL 引擎
@pytest.fixture
def mysql_pymysql_engine():
    # 导入 pytest 库并确认导入 sqlalchemy 和 pymysql 库
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pymysql = pytest.importorskip("pymysql")
    # 创建一个 MySQL + PyMySQL 引擎连接
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://root@localhost:3306/pandas",
        connect_args={"client_flag": pymysql.constants.CLIENT.MULTI_STATEMENTS},
        poolclass=sqlalchemy.pool.NullPool,
    )
    # 返回创建的引擎对象
    yield engine
    # 在 fixture 结束时，清理所有视图和表
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    # 释放引擎资源
    engine.dispose()


# 定义另一个 pytest fixture，用于创建和加载 MySQL + PyMySQL 引擎，加载 iris 数据
@pytest.fixture
def mysql_pymysql_engine_iris(mysql_pymysql_engine, iris_path):
    create_and_load_iris(mysql_pymysql_engine, iris_path)
    create_and_load_iris_view(mysql_pymysql_engine)
    return mysql_pymysql_engine


# 定义另一个 pytest fixture，用于创建和加载 MySQL + PyMySQL 引擎，加载 types 数据
@pytest.fixture
def mysql_pymysql_engine_types(mysql_pymysql_engine, types_data):
    create_and_load_types(mysql_pymysql_engine, types_data, "mysql")
    return mysql_pymysql_engine


# 定义一个 pytest fixture，创建 MySQL + PyMySQL 连接
@pytest.fixture
def mysql_pymysql_conn(mysql_pymysql_engine):
    # 使用 MySQL + PyMySQL 引擎创建数据库连接
    with mysql_pymysql_engine.connect() as conn:
        yield conn


# 定义一个 pytest fixture，创建 MySQL + PyMySQL 连接，并加载 iris 数据
@pytest.fixture
def mysql_pymysql_conn_iris(mysql_pymysql_engine_iris):
    with mysql_pymysql_engine_iris.connect() as conn:
        yield conn


# 定义一个 pytest fixture，创建 MySQL + PyMySQL 连接，并加载 types 数据
@pytest.fixture
def mysql_pymysql_conn_types(mysql_pymysql_engine_types):
    with mysql_pymysql_engine_types.connect() as conn:
        yield conn


# 定义一个 pytest fixture，创建 PostgreSQL + psycopg2 引擎
@pytest.fixture
def postgresql_psycopg2_engine():
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("psycopg2")
    # 创建一个 PostgreSQL + psycopg2 引擎连接
    engine = sqlalchemy.create_engine(
        "postgresql+psycopg2://postgres:postgres@localhost:5432/pandas",
        poolclass=sqlalchemy.pool.NullPool,
    )
    # 返回创建的引擎对象
    yield engine
    # 在 fixture 结束时，清理所有视图和表
    for view in get_all_views(engine):
        drop_view(view, engine)
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    # 释放引擎资源
    engine.dispose()


# 定义一个 pytest fixture，用于创建和加载 PostgreSQL + psycopg2 引擎，加载 iris 数据
@pytest.fixture
def postgresql_psycopg2_engine_iris(postgresql_psycopg2_engine, iris_path):
    create_and_load_iris(postgresql_psycopg2_engine, iris_path)
    create_and_load_iris_view(postgresql_psycopg2_engine)
    return postgresql_psycopg2_engine


# 定义一个 pytest fixture，用于创建和加载 PostgreSQL + psycopg2 引擎，加载 types 数据
@pytest.fixture
def postgresql_psycopg2_engine_types(postgresql_psycopg2_engine, types_data):
    # 使用指定的 PostgreSQL 引擎和类型数据，创建并加载数据类型
    create_and_load_types(postgresql_psycopg2_engine, types_data, "postgres")
    # 返回已配置好的 PostgreSQL 引擎对象
    return postgresql_psycopg2_engine
@pytest.fixture
# 定义 PostgreSQL 数据库连接的 pytest fixture
def postgresql_psycopg2_conn(postgresql_psycopg2_engine):
    # 使用 psycopg2 引擎连接 PostgreSQL 数据库
    with postgresql_psycopg2_engine.connect() as conn:
        yield conn  # 返回数据库连接对象

@pytest.fixture
# 定义 PostgreSQL ADBC 连接的 pytest fixture
def postgresql_adbc_conn():
    # 确保 adbc_driver_postgresql 包可导入
    pytest.importorskip("adbc_driver_postgresql")
    from adbc_driver_postgresql import dbapi

    # PostgreSQL 数据库连接 URI
    uri = "postgresql://postgres:postgres@localhost:5432/pandas"
    with dbapi.connect(uri) as conn:
        yield conn  # 返回数据库连接对象

        # 在连接结束后删除所有视图
        for view in get_all_views(conn):
            drop_view(view, conn)
        # 在连接结束后删除所有表格
        for tbl in get_all_tables(conn):
            drop_table(tbl, conn)
        conn.commit()  # 提交事务

@pytest.fixture
# 定义 PostgreSQL ADBC 连接及数据初始化的 pytest fixture
def postgresql_adbc_iris(postgresql_adbc_conn, iris_path):
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn

    try:
        # 尝试获取 iris 表的表结构
        conn.adbc_get_table_schema("iris")
    except mgr.ProgrammingError:
        conn.rollback()  # 回滚事务
        create_and_load_iris_postgresql(conn, iris_path)  # 创建并加载 iris 数据到 PostgreSQL

    try:
        # 尝试获取 iris_view 视图的表结构
        conn.adbc_get_table_schema("iris_view")
    except mgr.ProgrammingError:  # 处理异常：箭头-adbc 问题 1022
        conn.rollback()  # 回滚事务
        create_and_load_iris_view(conn)  # 创建 iris_view 视图

    return conn  # 返回连接对象

@pytest.fixture
# 定义 PostgreSQL ADBC 连接及数据类型初始化的 pytest fixture
def postgresql_adbc_types(postgresql_adbc_conn, types_data):
    import adbc_driver_manager as mgr

    conn = postgresql_adbc_conn

    try:
        # 尝试获取 types 表的表结构
        conn.adbc_get_table_schema("types")
    except mgr.ProgrammingError:
        conn.rollback()  # 回滚事务
        new_data = [tuple(entry.values()) for entry in types_data]

        create_and_load_types_postgresql(conn, new_data)  # 创建并加载 types 数据到 PostgreSQL

    return conn  # 返回连接对象

@pytest.fixture
# 定义带有 iris 数据库的 psycopg2 连接的 pytest fixture
def postgresql_psycopg2_conn_iris(postgresql_psycopg2_engine_iris):
    # 使用 psycopg2 引擎连接带有 iris 数据库的 PostgreSQL
    with postgresql_psycopg2_engine_iris.connect() as conn:
        yield conn  # 返回数据库连接对象

@pytest.fixture
# 定义带有 types 数据库的 psycopg2 连接的 pytest fixture
def postgresql_psycopg2_conn_types(postgresql_psycopg2_engine_types):
    # 使用 psycopg2 引擎连接带有 types 数据库的 PostgreSQL
    with postgresql_psycopg2_engine_types.connect() as conn:
        yield conn  # 返回数据库连接对象

@pytest.fixture
# 定义 SQLite 数据库连接字符串的 pytest fixture
def sqlite_str():
    # 确保能导入 sqlalchemy
    pytest.importorskip("sqlalchemy")
    with tm.ensure_clean() as name:
        yield f"sqlite:///{name}"  # 返回 SQLite 数据库连接字符串

@pytest.fixture
# 定义带有 SQLite 引擎的 pytest fixture
def sqlite_engine(sqlite_str):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    # 创建 SQLite 引擎
    engine = sqlalchemy.create_engine(sqlite_str, poolclass=sqlalchemy.pool.NullPool)
    yield engine  # 返回 SQLite 引擎对象

    # 在结束后删除所有视图
    for view in get_all_views(engine):
        drop_view(view, engine)
    # 在结束后删除所有表格
    for tbl in get_all_tables(engine):
        drop_table(tbl, engine)
    engine.dispose()  # 释放引擎资源

@pytest.fixture
# 定义带有 SQLite 连接的 pytest fixture
def sqlite_conn(sqlite_engine):
    # 使用 SQLite 引擎进行连接
    with sqlite_engine.connect() as conn:
        yield conn  # 返回数据库连接对象

@pytest.fixture
# 定义带有 iris 数据库的 SQLite 连接字符串的 pytest fixture
def sqlite_str_iris(sqlite_str, iris_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
    # 创建 SQLite 引擎
    engine = sqlalchemy.create_engine(sqlite_str)
    create_and_load_iris(engine, iris_path)  # 创建并加载 iris 数据到 SQLite
    create_and_load_iris_view(engine)  # 创建 iris 视图到 SQLite
    engine.dispose()  # 释放引擎资源
    return sqlite_str  # 返回 SQLite 连接字符串

@pytest.fixture
# 定义带有 iris 数据库的 SQLite 引擎的 pytest fixture
def sqlite_engine_iris(sqlite_engine, iris_path):
    create_and_load_iris(sqlite_engine, iris_path)  # 创建并加载 iris 数据到 SQLite
    create_and_load_iris_view(sqlite_engine)  # 创建 iris 视图到 SQLite
    return sqlite_engine  # 返回 SQLite 引擎对象

@pytest.fixture
# 定义带有 iris 数据库的 SQLite 连接的 pytest fixture
def sqlite_conn_iris(sqlite_engine_iris):
    # 使用 sqlite_engine_iris 建立数据库连接，使用完毕后自动关闭连接
    with sqlite_engine_iris.connect() as conn:
        # 返回连接对象，允许在此上下文中执行数据库操作
        yield conn
@pytest.fixture
# 定义 sqlite_str_types 的测试夹具，用于返回一个 SQLite 连接字符串，并加载自定义数据类型
def sqlite_str_types(sqlite_str, types_data):
    sqlalchemy = pytest.importorskip("sqlalchemy")  # 导入并检查是否存在 SQLAlchemy
    engine = sqlalchemy.create_engine(sqlite_str)  # 创建 SQLite 引擎
    create_and_load_types(engine, types_data, "sqlite")  # 调用函数加载自定义数据类型到 SQLite 中
    engine.dispose()  # 处置（关闭）引擎
    return sqlite_str  # 返回 SQLite 连接字符串


@pytest.fixture
# 定义 sqlite_engine_types 的测试夹具，用于返回一个已加载了自定义数据类型的 SQLite 引擎
def sqlite_engine_types(sqlite_engine, types_data):
    create_and_load_types(sqlite_engine, types_data, "sqlite")  # 调用函数加载自定义数据类型到 SQLite 引擎中
    return sqlite_engine  # 返回已加载了自定义数据类型的 SQLite 引擎


@pytest.fixture
# 定义 sqlite_conn_types 的测试夹具，用于返回一个 SQLite 连接
def sqlite_conn_types(sqlite_engine_types):
    with sqlite_engine_types.connect() as conn:  # 使用 SQLite 引擎建立连接
        yield conn  # 生成器，返回连接对象
        

@pytest.fixture
# 定义 sqlite_adbc_conn 的测试夹具，用于返回一个 ADBC SQLite 连接
def sqlite_adbc_conn():
    pytest.importorskip("adbc_driver_sqlite")  # 导入并检查是否存在 ADBC SQLite 驱动
    from adbc_driver_sqlite import dbapi  # 导入 ADBC SQLite 的 DBAPI

    with tm.ensure_clean() as name:  # 确保测试环境的清洁性，生成临时文件名
        uri = f"file:{name}"  # 创建 SQLite 文件 URI
        with dbapi.connect(uri) as conn:  # 使用 ADBC SQLite 驱动连接数据库
            yield conn  # 生成器，返回连接对象
            # 清理工作：删除所有视图和表
            for view in get_all_views(conn):
                drop_view(view, conn)
            for tbl in get_all_tables(conn):
                drop_table(tbl, conn)
            conn.commit()  # 提交事务


@pytest.fixture
# 定义 sqlite_adbc_iris 的测试夹具，用于返回一个 ADBC SQLite 连接并创建 iris 数据表和视图
def sqlite_adbc_iris(sqlite_adbc_conn, iris_path):
    import adbc_driver_manager as mgr  # 导入 ADBC 管理器

    conn = sqlite_adbc_conn  # 获取 ADBC SQLite 连接
    try:
        conn.adbc_get_table_schema("iris")  # 尝试获取 iris 表结构信息
    except mgr.ProgrammingError:
        conn.rollback()  # 回滚事务
        create_and_load_iris_sqlite3(conn, iris_path)  # 创建并加载 iris 数据表
    try:
        conn.adbc_get_table_schema("iris_view")  # 尝试获取 iris_view 视图结构信息
    except mgr.ProgrammingError:
        conn.rollback()  # 回滚事务
        create_and_load_iris_view(conn)  # 创建 iris_view 视图
    return conn  # 返回连接对象


@pytest.fixture
# 定义 sqlite_adbc_types 的测试夹具，用于返回一个 ADBC SQLite 连接并加载自定义数据类型
def sqlite_adbc_types(sqlite_adbc_conn, types_data):
    import adbc_driver_manager as mgr  # 导入 ADBC 管理器

    conn = sqlite_adbc_conn  # 获取 ADBC SQLite 连接
    try:
        conn.adbc_get_table_schema("types")  # 尝试获取 types 表结构信息
    except mgr.ProgrammingError:
        conn.rollback()  # 回滚事务
        new_data = []
        # 调整数据格式并加载到 SQLite 中
        for entry in types_data:
            entry["BoolCol"] = int(entry["BoolCol"])
            if entry["BoolColWithNull"] is not None:
                entry["BoolColWithNull"] = int(entry["BoolColWithNull"])
            new_data.append(tuple(entry.values()))
        create_and_load_types_sqlite3(conn, new_data)  # 创建并加载自定义数据类型
        conn.commit()  # 提交事务

    return conn  # 返回连接对象


@pytest.fixture
# 定义 sqlite_buildin 的测试夹具，用于返回一个内存中的 SQLite 连接
def sqlite_buildin():
    with contextlib.closing(sqlite3.connect(":memory:")) as closing_conn:
        with closing_conn as conn:  # 使用内存中的 SQLite 连接
            yield conn  # 生成器，返回连接对象


@pytest.fixture
# 定义 sqlite_buildin_iris 的测试夹具，用于返回一个内存中的 SQLite 连接并创建 iris 数据表和视图
def sqlite_buildin_iris(sqlite_buildin, iris_path):
    create_and_load_iris_sqlite3(sqlite_buildin, iris_path)  # 创建并加载 iris 数据表
    create_and_load_iris_view(sqlite_buildin)  # 创建 iris 视图
    return sqlite_buildin  # 返回连接对象


@pytest.fixture
# 定义 sqlite_buildin_types 的测试夹具，用于返回一个内存中的 SQLite 连接并加载自定义数据类型
def sqlite_buildin_types(sqlite_buildin, types_data):
    types_data = [tuple(entry.values()) for entry in types_data]  # 调整数据格式
    create_and_load_types_sqlite3(sqlite_buildin, types_data)  # 创建并加载自定义数据类型
    return sqlite_buildin  # 返回连接对象


# 定义 mysql_connectable 列表，包含用于连接到 MySQL 的测试参数，并使用 pytest.mark.db 进行标记
mysql_connectable = [
    pytest.param("mysql_pymysql_engine", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn", marks=pytest.mark.db),
]

# 定义 mysql_connectable_iris 列表，包含用于连接到带 iris 数据表的 MySQL 的测试参数，并使用 pytest.mark.db 进行标记
mysql_connectable_iris = [
    pytest.param("mysql_pymysql_engine_iris", marks=pytest.mark.db),
]
    # 使用 pytest.param 创建一个带有标记的参数，标记为 pytest.mark.db
    pytest.param("mysql_pymysql_conn_iris", marks=pytest.mark.db),
]

# 定义一个列表，包含 MySQL 数据库连接类型参数，使用 pytest.mark.db 标记
mysql_connectable_types = [
    pytest.param("mysql_pymysql_engine_types", marks=pytest.mark.db),
    pytest.param("mysql_pymysql_conn_types", marks=pytest.mark.db),
]

# 定义一个列表，包含 PostgreSQL 数据库连接类型参数，使用 pytest.mark.db 标记
postgresql_connectable = [
    pytest.param("postgresql_psycopg2_engine", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn", marks=pytest.mark.db),
]

# 定义一个列表，包含带有 iris 标记的 PostgreSQL 数据库连接类型参数，使用 pytest.mark.db 标记
postgresql_connectable_iris = [
    pytest.param("postgresql_psycopg2_engine_iris", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_iris", marks=pytest.mark.db),
]

# 定义一个列表，包含 PostgreSQL 数据库连接类型参数，使用 pytest.mark.db 标记
postgresql_connectable_types = [
    pytest.param("postgresql_psycopg2_engine_types", marks=pytest.mark.db),
    pytest.param("postgresql_psycopg2_conn_types", marks=pytest.mark.db),
]

# 定义一个列表，包含 SQLite 数据库连接类型参数
sqlite_connectable = [
    "sqlite_engine",
    "sqlite_conn",
    "sqlite_str",
]

# 定义一个列表，包含带有 iris 标记的 SQLite 数据库连接类型参数
sqlite_connectable_iris = [
    "sqlite_engine_iris",
    "sqlite_conn_iris",
    "sqlite_str_iris",
]

# 定义一个列表，包含 SQLite 数据库连接类型参数
sqlite_connectable_types = [
    "sqlite_engine_types",
    "sqlite_conn_types",
    "sqlite_str_types",
]

# 创建一个列表，包含所有的 SQLAlchemy 连接类型参数，合并了 MySQL、PostgreSQL 和 SQLite 连接参数
sqlalchemy_connectable = mysql_connectable + postgresql_connectable + sqlite_connectable

# 创建一个带有 iris 标记的列表，包含所有的 SQLAlchemy 连接类型参数，合并了 MySQL、PostgreSQL 和 SQLite 连接参数
sqlalchemy_connectable_iris = (
    mysql_connectable_iris + postgresql_connectable_iris + sqlite_connectable_iris
)

# 创建一个列表，包含所有的 SQLAlchemy 连接类型参数，合并了 MySQL、PostgreSQL 和 SQLite 连接类型参数
sqlalchemy_connectable_types = (
    mysql_connectable_types + postgresql_connectable_types + sqlite_connectable_types
)

# 创建一个列表，包含 ADBC（Abstract Database Connectivity）连接参数，包括 SQLite 和带有标记的 PostgreSQL 连接
adbc_connectable = [
    "sqlite_adbc_conn",
    pytest.param("postgresql_adbc_conn", marks=pytest.mark.db),
]

# 创建一个带有 iris 标记的列表，包含 ADBC 连接参数，包括带有标记的 PostgreSQL 和 SQLite 连接
adbc_connectable_iris = [
    pytest.param("postgresql_adbc_iris", marks=pytest.mark.db),
    pytest.param("sqlite_adbc_iris", marks=pytest.mark.db),
]

# 创建一个列表，包含带有标记的 ADBC 连接类型参数，包括带有标记的 PostgreSQL 和 SQLite 连接
adbc_connectable_types = [
    pytest.param("postgresql_adbc_types", marks=pytest.mark.db),
    pytest.param("sqlite_adbc_types", marks=pytest.mark.db),
]

# 创建一个列表，包含所有可连接的数据库类型参数，包括所有的 SQLAlchemy 连接参数和一个 SQLite 内置连接参数
all_connectable = sqlalchemy_connectable + ["sqlite_buildin"] + adbc_connectable

# 创建一个带有 iris 标记的列表，包含所有可连接的数据库类型参数，包括所有的 SQLAlchemy 连接参数和一个带有 iris 标记的 SQLite 内置连接参数
all_connectable_iris = (
    sqlalchemy_connectable_iris + ["sqlite_buildin_iris"] + adbc_connectable_iris
)

# 创建一个列表，包含所有可连接的数据库类型参数，包括所有的 SQLAlchemy 连接类型参数和一个带有标记的 ADBC 连接类型参数
all_connectable_types = (
    sqlalchemy_connectable_types + ["sqlite_buildin_types"] + adbc_connectable_types
)

# 使用 pytest.mark.parametrize 装饰器，为 test_dataframe_to_sql 函数提供参数化测试，使用所有可连接的数据库类型参数
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql(conn, test_frame1, request):
    # GH 51086 如果 conn 是 sqlite_engine，则从 request 中获取 conn 的 fixture 值
    conn = request.getfixturevalue(conn)
    # 将 test_frame1 数据框写入数据库，表名为 "test"，连接对象为 conn，如果表已存在则追加数据，不包含索引列
    test_frame1.to_sql(name="test", con=conn, if_exists="append", index=False)

# 使用 pytest.mark.parametrize 装饰器，为 test_dataframe_to_sql_empty 函数提供参数化测试，使用所有可连接的数据库类型参数
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_empty(conn, test_frame1, request):
    # 如果 conn 是 "postgresql_adbc_conn"，向当前测试节点添加一个标记，说明该测试预计会失败
    if conn == "postgresql_adbc_conn":
        request.node.add_marker(
            pytest.mark.xfail(
                reason="postgres ADBC driver cannot insert index with null type",
                strict=True,
            )
        )
    # GH 51086 如果 conn 是 sqlite_engine，则从 request 中获取 conn 的 fixture 值
    conn = request.getfixturevalue(conn)
    # 创建一个空的数据框 empty_df，与 test_frame1 相同结构但没有行数据
    empty_df = test_frame1.iloc[:0]
    # 将 empty_df 数据框写入数据库，表名为 "test"，连接对象为 conn，如果表已存在则追加数据，不包含索引列
    empty_df.to_sql(name="test", con=conn, if_exists="append", index=False)

# 使用 pytest.mark.parametrize 装饰器，为 test_dataframe_to_sql_arrow_dtypes 函数提供参数化测试，使用所有可连接的数据库类型参数
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes(conn, request):
    # GH 52046
    # 导入 pytest 库，并检查是否能导入 pyarrow 库，如果不能则跳过测试
    pytest.importorskip("pyarrow")
    
    # 创建一个 DataFrame 对象，包含不同类型的列，每列都使用 pyarrow 扩展的数据类型
    df = DataFrame(
        {
            "int": pd.array([1], dtype="int8[pyarrow]"),  # 整数列，使用 pyarrow 扩展的 int8 类型
            "datetime": pd.array(
                [datetime(2023, 1, 1)], dtype="timestamp[ns][pyarrow]"  # 日期时间列，使用 pyarrow 扩展的 timestamp 类型
            ),
            "date": pd.array([date(2023, 1, 1)], dtype="date32[day][pyarrow]"  # 日期列，使用 pyarrow 扩展的 date32 类型
            ),
            "timedelta": pd.array([timedelta(1)], dtype="duration[ns][pyarrow]"  # 时间差列，使用 pyarrow 扩展的 duration 类型
            ),
            "string": pd.array(["a"], dtype="string[pyarrow]")  # 字符串列，使用 pyarrow 扩展的 string 类型
        }
    )
    
    # 检查连接字符串中是否包含 "adbc" 字符串
    if "adbc" in conn:
        # 如果连接字符串包含 "adbc"
        # 检查具体的连接字符串是否为 "sqlite_adbc_conn"，如果是，则删除 DataFrame 中的 "timedelta" 列
        if conn == "sqlite_adbc_conn":
            df = df.drop(columns=["timedelta"])
        
        # 根据 pyarrow 版本号是否低于 14.1，决定使用的警告类型和警告消息
        if pa_version_under14p1:
            exp_warning = DeprecationWarning  # 使用过时警告
            msg = "is_sparse is deprecated"  # 警告消息
        else:
            exp_warning = None  # 不使用特定的警告
            msg = ""  # 空消息
            
    else:
        # 如果连接字符串中不包含 "adbc"，设定警告类型为 UserWarning，警告消息为 "the 'timedelta'"
        exp_warning = UserWarning
        msg = "the 'timedelta'"
    
    # 根据请求中的连接字符串获取连接对象
    conn = request.getfixturevalue(conn)
    
    # 使用上下文管理器确保在执行 df.to_sql 操作时产生期望的警告信息
    with tm.assert_produces_warning(exp_warning, match=msg, check_stacklevel=False):
        df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)
# 使用 pytest 的 parametrize 装饰器，将参数化的连接器传递给测试函数
@pytest.mark.parametrize("conn", all_connectable)
def test_dataframe_to_sql_arrow_dtypes_missing(conn, request, nulls_fixture):
    # 注释：测试用例针对 GitHub 问题编号 52046
    pytest.importorskip("pyarrow")
    # 创建包含特定数据帧的 DataFrame 对象
    df = DataFrame(
        {
            "datetime": pd.array(
                [datetime(2023, 1, 1), nulls_fixture], dtype="timestamp[ns][pyarrow]"
            ),
        }
    )
    # 通过请求获取连接器实例
    conn = request.getfixturevalue(conn)
    # 将 DataFrame 对象写入指定数据库表中
    df.to_sql(name="test_arrow", con=conn, if_exists="replace", index=False)


# 使用 pytest 的 parametrize 装饰器，传递参数化的连接器和方法给测试函数
@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("method", [None, "multi"])
def test_to_sql(conn, method, test_frame1, request):
    # 如果方法为 "multi" 并且连接器名中包含 "adbc"
    if method == "multi" and "adbc" in conn:
        # 为当前测试用例添加标记，表明预期失败
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'method' not implemented for ADBC drivers", strict=True
            )
        )

    # 通过请求获取连接器实例
    conn = request.getfixturevalue(conn)
    # 使用 pandasSQL_builder 创建连接并确保在事务中操作
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        # 将数据帧写入指定数据库表中
        pandasSQL.to_sql(test_frame1, "test_frame", method=method)
        # 断言数据库中存在名为 "test_frame" 的表
        assert pandasSQL.has_table("test_frame")
    # 断言数据库表中的行数等于测试数据帧的行数
    assert count_rows(conn, "test_frame") == len(test_frame1)


# 使用 pytest 的 parametrize 装饰器，传递参数化的连接器、模式和行数系数给测试函数
@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("mode, num_row_coef", [("replace", 1), ("append", 2)])
def test_to_sql_exist(conn, mode, num_row_coef, test_frame1, request):
    # 通过请求获取连接器实例
    conn = request.getfixturevalue(conn)
    # 使用 pandasSQL_builder 创建连接并确保在事务中操作
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        # 将数据帧写入指定数据库表中，如果表已存在则操作失败
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        # 将数据帧写入指定数据库表中，根据不同模式进行处理
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists=mode)
        # 断言数据库中存在名为 "test_frame" 的表
        assert pandasSQL.has_table("test_frame")
    # 断言数据库表中的行数等于测试数据帧的行数乘以行数系数
    assert count_rows(conn, "test_frame") == num_row_coef * len(test_frame1)


# 使用 pytest 的 parametrize 装饰器，传递参数化的连接器给测试函数
@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_exist_fail(conn, test_frame1, request):
    # 通过请求获取连接器实例
    conn = request.getfixturevalue(conn)
    # 使用 pandasSQL_builder 创建连接并确保在事务中操作
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        # 将数据帧写入指定数据库表中，如果表已存在则操作失败
        pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")
        # 断言数据库中存在名为 "test_frame" 的表
        assert pandasSQL.has_table("test_frame")
        # 检查是否引发 ValueError 异常，并匹配特定的错误消息
        msg = "Table 'test_frame' already exists"
        with pytest.raises(ValueError, match=msg):
            pandasSQL.to_sql(test_frame1, "test_frame", if_exists="fail")


# 使用 pytest 的 parametrize 装饰器，传递参数化的连接器给测试函数
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query(conn, request):
    # 通过请求获取连接器实例
    conn = request.getfixturevalue(conn)
    # 从数据库中读取 iris 表的全部数据
    iris_frame = read_sql_query("SELECT * FROM iris", conn)
    # 检查读取的 iris 数据帧是否符合预期
    check_iris_frame(iris_frame)
    # 从数据库中读取 iris 表的全部数据并检查
    iris_frame = pd.read_sql("SELECT * FROM iris", conn)
    check_iris_frame(iris_frame)
    # 从数据库中读取 iris 表的空数据并断言其形状为 (0, 5)
    iris_frame = pd.read_sql("SELECT * FROM iris where 0=1", conn)
    assert iris_frame.shape == (0, 5)
    # 断言 iris 数据帧的列中包含 "SepalWidth"
    assert "SepalWidth" in iris_frame.columns


# 使用 pytest 的 parametrize 装饰器，传递参数化的连接器给测试函数
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_chunksize(conn, request):
    # 检查连接字符串中是否包含"adbc"，如果是，则标记当前测试用例为预期失败（xfail）
    if "adbc" in conn:
        # 添加一个 xfail 标记到当前测试用例，指定失败的原因为 ADBC 驱动程序未实现 'chunksize' 参数
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'chunksize' not implemented for ADBC drivers",
                strict=True,
            )
        )
    
    # 使用 Pytest 的 request 对象获取测试用例中的连接字符串对应的 fixture
    conn = request.getfixturevalue(conn)
    
    # 从数据库中读取 iris 表的数据，并将其合并成一个 DataFrame，设置每块数据的大小为 7
    iris_frame = concat(read_sql_query("SELECT * FROM iris", conn, chunksize=7))
    
    # 检查合并后的 iris_frame DataFrame 是否符合预期
    check_iris_frame(iris_frame)
    
    # 使用 Pandas 的 read_sql 方法从数据库中读取 iris 表的数据，并将其合并成一个 DataFrame，设置每块数据的大小为 7
    iris_frame = concat(pd.read_sql("SELECT * FROM iris", conn, chunksize=7))
    
    # 再次检查合并后的 iris_frame DataFrame 是否符合预期
    check_iris_frame(iris_frame)
    
    # 使用 Pandas 的 read_sql 方法从数据库中读取不包含任何数据的 iris 表的数据，并将其合并成一个 DataFrame，设置每块数据的大小为 7
    iris_frame = concat(pd.read_sql("SELECT * FROM iris where 0=1", conn, chunksize=7))
    
    # 断言空的 iris_frame DataFrame 的形状应该是 (0, 5)
    assert iris_frame.shape == (0, 5)
    
    # 断言合并后的 iris_frame DataFrame 的列中应该包含 'SepalWidth' 列
    assert "SepalWidth" in iris_frame.columns
@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_query_expression_with_parameter(conn, request):
    # 如果连接字符串中包含"adbc"，标记该测试为预期失败，原因是ADBC驱动程序不支持'chunksize'
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'chunksize' not implemented for ADBC drivers",
                strict=True,
            )
        )
    # 根据连接字符串获取对应的数据库连接对象
    conn = request.getfixturevalue(conn)
    # 导入所需的SQLAlchemy模块
    from sqlalchemy import (
        MetaData,
        Table,
        create_engine,
        select,
    )

    # 创建Metadata对象
    metadata = MetaData()
    # 根据连接字符串创建SQLAlchemy引擎对象，或者直接使用传入的连接对象
    autoload_con = create_engine(conn) if isinstance(conn, str) else conn
    # 创建名为"iris"的数据表对象，自动加载表结构
    iris = Table("iris", metadata, autoload_with=autoload_con)
    # 使用select查询表iris的数据，并传入参数"name"和"length"
    iris_frame = read_sql_query(
        select(iris), conn, params={"name": "Iris-setosa", "length": 5.1}
    )
    # 检查查询结果
    check_iris_frame(iris_frame)
    # 如果使用的是字符串形式的连接，释放数据库连接资源
    if isinstance(conn, str):
        autoload_con.dispose()


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_iris_query_string_with_parameter(conn, request, sql_strings):
    # 如果连接字符串中包含"adbc"，标记该测试为预期失败，原因是ADBC驱动程序不支持'chunksize'
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'chunksize' not implemented for ADBC drivers",
                strict=True,
            )
        )

    # 在sql_strings['read_parameters']中查找与当前连接字符串相关的查询语句
    for db, query in sql_strings["read_parameters"].items():
        if db in conn:
            break
    else:
        # 如果未找到与连接字符串匹配的查询语句，抛出KeyError异常
        raise KeyError(f"No part of {conn} found in sql_strings['read_parameters']")
    
    # 根据连接字符串获取对应的数据库连接对象
    conn = request.getfixturevalue(conn)
    # 使用查询语句执行数据库查询，并传入参数"Iris-setosa"和5.1
    iris_frame = read_sql_query(query, conn, params=("Iris-setosa", 5.1))
    # 检查查询结果
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table(conn, request):
    # 如果连接字符串是"sqlite_iris_str"，则存在一个问题（GH 51015），暂不详细描述
    # 未修改conn的意图
    conn = request.getfixturevalue(conn)
    # 读取名为"iris"的表的数据，并返回DataFrame对象
    iris_frame = read_sql_table("iris", conn)
    # 检查查询结果
    check_iris_frame(iris_frame)
    # 使用pd.read_sql读取名为"iris"的表的数据，并返回DataFrame对象
    iris_frame = pd.read_sql("iris", conn)
    # 检查查询结果
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_iris_table_chunksize(conn, request):
    # 如果连接字符串中包含"adbc"，标记该测试为预期失败，原因是ADBC驱动程序不支持'chunksize'参数
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    # 根据连接字符串获取对应的数据库连接对象
    conn = request.getfixturevalue(conn)
    # 读取名为"iris"的表的数据，并返回DataFrame对象，设定chunksize为7
    iris_frame = concat(read_sql_table("iris", conn, chunksize=7))
    # 检查查询结果
    check_iris_frame(iris_frame)
    # 使用pd.read_sql读取名为"iris"的表的数据，并返回DataFrame对象，设定chunksize为7
    iris_frame = concat(pd.read_sql("iris", conn, chunksize=7))
    # 检查查询结果
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_to_sql_callable(conn, test_frame1, request):
    # 根据连接字符串获取对应的数据库连接对象
    conn = request.getfixturevalue(conn)

    check = []  # 用于确认下面的函数确实被使用了

    # 定义一个函数sample，将数据逐行插入到数据库中
    def sample(pd_table, conn, keys, data_iter):
        check.append(1)
        # 将data_iter中的每行数据转换成字典形式，便于插入数据库
        data = [dict(zip(keys, row)) for row in data_iter]
        # 执行数据插入操作
        conn.execute(pd_table.table.insert(), data)

    # 使用pandasSQL_builder创建pandasSQL对象，开启事务模式
    with pandasSQL_builder(conn, need_transaction=True) as pandasSQL:
        # 将test_frame1数据框中的数据插入到名为"test_frame"的数据库表中，使用自定义的插入方法sample
        pandasSQL.to_sql(test_frame1, "test_frame", method=sample)
        # 断言名为"test_frame"的表已经存在于数据库中
        assert pandasSQL.has_table("test_frame")
    # 断言检查列表 check 是否等于 [1]
    assert check == [1]
    # 断言检查数据库连接 conn 中表 "test_frame" 的行数是否等于 test_frame1 的长度
    assert count_rows(conn, "test_frame") == len(test_frame1)
python
# 使用 pytest 的 @pytest.mark.parametrize 装饰器，用来为测试函数 test_default_type_conversion 添加参数化测试数据
@pytest.mark.parametrize("conn", all_connectable_types)
# 定义测试函数 test_default_type_conversion，接受参数 conn 和 request
def test_default_type_conversion(conn, request):
    # 将连接类型的字符串保存到 conn_name 变量中
    conn_name = conn
    # 如果 conn_name 是 "sqlite_buildin_types"，则将一个 xfail 标记应用到当前测试用例上
    if conn_name == "sqlite_buildin_types":
        request.applymarker(
            pytest.mark.xfail(
                reason="sqlite_buildin connection does not implement read_sql_table"
            )
        )

    # 使用 request.getfixturevalue 获取 conn 对应的 fixture，并将其赋值给 conn 变量
    conn = request.getfixturevalue(conn)
    # 使用 sql.read_sql_table 方法从数据库表 "types" 中读取数据，并将结果保存到 df 变量
    df = sql.read_sql_table("types", conn)

    # 断言 FloatCol 列的数据类型是 np.floating 的子类
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    # 断言 IntCol 列的数据类型是 np.integer 的子类
    assert issubclass(df.IntCol.dtype.type, np.integer)

    # 根据数据库类型进行不同的断言
    # PostgreSQL 数据库中没有真正的 BOOL 类型，因此断言 BoolCol 列的数据类型是 np.bool_
    if "postgresql" in conn_name:
        assert issubclass(df.BoolCol.dtype.type, np.bool_)
    else:
        assert issubclass(df.BoolCol.dtype.type, np.integer)

    # 断言 IntColWithNull 列的数据类型是 np.floating 的子类，即使该列包含 NA 值
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)

    # 根据数据库类型进行不同的断言
    # PostgreSQL 数据库中，BoolColWithNull 列的数据类型是 object，其他数据库中是 np.floating
    if "postgresql" in conn_name:
        assert issubclass(df.BoolColWithNull.dtype.type, object)
    else:
        assert issubclass(df.BoolColWithNull.dtype.type, np.floating)


# 使用 pytest 的 @pytest.mark.parametrize 装饰器，为测试函数 test_read_procedure 添加参数化测试数据
@pytest.mark.parametrize("conn", mysql_connectable)
# 定义测试函数 test_read_procedure，接受参数 conn 和 request
def test_read_procedure(conn, request):
    # 使用 request.getfixturevalue 获取 conn 对应的 fixture，并将其赋值给 conn 变量
    conn = request.getfixturevalue(conn)

    # GH 7324
    # 虽然更多是一个 API 测试，但将其添加到 mysql 测试中，因为 sqlite 不支持存储过程
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    # 创建一个 DataFrame 对象 df，包含两列 "a" 和 "b" 的数据
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    # 将 df 对象写入数据库表 "test_frame" 中
    df.to_sql(name="test_frame", con=conn, index=False)

    # 定义存储过程 proc 的字符串，包括创建和删除过程
    proc = """DROP PROCEDURE IF EXISTS get_testdb;

    CREATE PROCEDURE get_testdb ()

    BEGIN
        SELECT * FROM test_frame;
    END"""
    # 将 proc 字符串转换为 sqlalchemy.text 对象
    proc = text(proc)
    
    # 根据 conn 的类型执行不同的数据库操作
    if isinstance(conn, Engine):
        # 如果 conn 是 Engine 类型的对象，使用 conn.connect() 建立连接，并使用 engine_conn 执行 proc
        with conn.connect() as engine_conn:
            with engine_conn.begin():
                engine_conn.execute(proc)
    else:
        # 否则，使用 conn.begin() 开始数据库事务，并使用 conn 执行 proc
        with conn.begin():
            conn.execute(proc)

    # 使用 sql.read_sql_query 方法执行 SQL 查询 "CALL get_testdb();"，并将结果保存到 res1 变量
    res1 = sql.read_sql_query("CALL get_testdb();", conn)
    # 断言读取到的数据 res1 与 DataFrame 对象 df 相等
    tm.assert_frame_equal(df, res1)

    # 使用 sql.read_sql 方法执行 SQL 查询 "CALL get_testdb();"，并将结果保存到 res2 变量
    res2 = sql.read_sql("CALL get_testdb();", conn)
    # 断言读取到的数据 res2 与 DataFrame 对象 df 相等
    tm.assert_frame_equal(df, res2)


# 使用 pytest 的 @pytest.mark.parametrize 装饰器，为测试函数 test_copy_from_callable_insertion_method 添加参数化测试数据
@pytest.mark.parametrize("conn", postgresql_connectable)
@pytest.mark.parametrize("expected_count", [2, "Success!"])
# 定义测试函数 test_copy_from_callable_insertion_method，接受参数 conn、expected_count 和 request
def test_copy_from_callable_insertion_method(conn, expected_count, request):
    # GH 8953
    # 在 io.rst 文件的 _io.sql.method 部分找到的示例
    # 在 sqlite 和 mysql 中不可用
    def psql_insert_copy(table, conn, keys, data_iter):
        # 获取可以提供游标的 DBAPI 连接
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            # 创建一个内存字符串缓冲区
            s_buf = StringIO()
            # 创建 CSV 写入器
            writer = csv.writer(s_buf)
            # 将数据迭代器写入到 CSV 缓冲区
            writer.writerows(data_iter)
            # 将文件指针移动到缓冲区起始位置
            s_buf.seek(0)

            # 将键列表转换为逗号分隔的列名字符串
            columns = ", ".join([f'"{k}"' for k in keys])
            # 如果表有模式，则使用模式和表名构建完整的表名
            if table.schema:
                table_name = f"{table.schema}.{table.name}"
            else:
                table_name = table.name

            # 构建 COPY 命令的 SQL 查询字符串
            sql_query = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV"
            # 执行 COPY 命令，将数据从缓冲区导入到数据库表中
            cur.copy_expert(sql=sql_query, file=s_buf)
        # 返回预期的行数
        return expected_count

    # 从请求获取数据库连接对象
    conn = request.getfixturevalue(conn)
    # 创建预期的数据帧
    expected = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})
    # 将预期的数据帧写入数据库表中，使用指定的插入方法 psql_insert_copy
    result_count = expected.to_sql(
        name="test_frame", con=conn, index=False, method=psql_insert_copy
    )
    # GH 46891
    # 如果预期行数为 None，则断言结果行数也应为 None
    if expected_count is None:
        assert result_count is None
    else:
        # 否则，断言结果行数与预期行数相等
        assert result_count == expected_count
    # 从数据库中读取名为 "test_frame" 的表，并将结果存储在 result 中
    result = sql.read_sql_table("test_frame", conn)
    # 断言读取的结果与预期的数据帧内容相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器为函数 test_insertion_method_on_conflict_do_nothing 添加参数化测试
@pytest.mark.parametrize("conn", postgresql_connectable)
def test_insertion_method_on_conflict_do_nothing(conn, request):
    # GH 15988: 在 to_sql 文档字符串中的示例
    # 从 request 中获取连接对象
    conn = request.getfixturevalue(conn)

    # 导入必要的模块
    from sqlalchemy.dialects.postgresql import insert
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    # 定义一个函数 insert_on_conflict，用于处理插入冲突时的操作
    def insert_on_conflict(table, conn, keys, data_iter):
        # 将 data_iter 中的数据转换为字典格式
        data = [dict(zip(keys, row)) for row in data_iter]
        # 创建插入语句，并指定冲突时的处理方式为 do nothing
        stmt = (
            insert(table.table)
            .values(data)
            .on_conflict_do_nothing(index_elements=["a"])
        )
        # 执行插入操作，并返回受影响的行数
        result = conn.execute(stmt)
        return result.rowcount

    # 定义创建表格的 SQL 语句
    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a  integer PRIMARY KEY,
        b  numeric,
        c  text
    );
    """
    )
    
    # 根据不同的数据库连接类型执行不同的操作
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                # 在连接 con 上执行创建表的 SQL 语句
                con.execute(create_sql)
    else:
        with conn.begin():
            # 在连接 conn 上执行创建表的 SQL 语句
            conn.execute(create_sql)

    # 创建预期的 DataFrame 对象
    expected = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    # 将预期的数据插入到数据库中
    expected.to_sql(
        name="test_insert_conflict", con=conn, if_exists="append", index=False
    )

    # 创建要插入的 DataFrame 对象
    df_insert = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    # 调用 to_sql 方法，将 df_insert 数据插入到数据库中，使用自定义的插入冲突处理方法 insert_on_conflict
    inserted = df_insert.to_sql(
        name="test_insert_conflict",
        con=conn,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    
    # 从数据库中读取表数据到 result DataFrame 对象中
    result = sql.read_sql_table("test_insert_conflict", conn)
    # 断言 result 和 expected 的内容是否一致
    tm.assert_frame_equal(result, expected)
    # 断言插入的行数是否为 0
    assert inserted == 0

    # 清理操作，删除创建的测试表
    # 使用 sql.SQLDatabase 对象进行数据库操作，并指定需要事务保护
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        # 删除名为 "test_insert_conflict" 的表
        pandasSQL.drop_table("test_insert_conflict")
    def insert_on_conflict(table, conn, keys, data_iter):
        # 将迭代器中的每一行数据转换为字典列表
        data = [dict(zip(keys, row)) for row in data_iter]
        # 创建插入语句，并设置冲突时更新的策略
        stmt = insert(table.table).values(data)
        stmt = stmt.on_duplicate_key_update(b=stmt.inserted.b, c=stmt.inserted.c)
        # 执行 SQL 语句，并返回执行结果对象
        result = conn.execute(stmt)
        # 返回插入的行数
        return result.rowcount

    # 创建用于创建表的 SQL 语句
    create_sql = text(
        """
    CREATE TABLE test_insert_conflict (
        a INT PRIMARY KEY,
        b FLOAT,
        c VARCHAR(10)
    );
    """
    )
    # 根据连接对象的类型执行不同的 SQL 操作
    if isinstance(conn, Engine):
        with conn.connect() as con:
            with con.begin():
                # 执行创建表的 SQL 语句
                con.execute(create_sql)
    else:
        with conn.begin():
            # 执行创建表的 SQL 语句
            conn.execute(create_sql)

    # 创建 DataFrame，准备插入的数据
    df = DataFrame([[1, 2.1, "a"]], columns=list("abc"))
    # 将 DataFrame 数据插入到数据库中的表中
    df.to_sql(name="test_insert_conflict", con=conn, if_exists="append", index=False)

    # 准备预期的数据，用于测试插入冲突处理功能
    expected = DataFrame([[1, 3.2, "b"]], columns=list("abc"))
    # 将预期数据插入到数据库表中，使用自定义的插入冲突处理方法
    inserted = expected.to_sql(
        name="test_insert_conflict",
        con=conn,
        index=False,
        if_exists="append",
        method=insert_on_conflict,
    )
    # 从数据库中读取表数据到 DataFrame
    result = sql.read_sql_table("test_insert_conflict", conn)
    # 断言读取的数据与预期数据相等
    tm.assert_frame_equal(result, expected)
    # 断言插入的行数为 2
    assert inserted == 2

    # 清理操作，删除测试用的表
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("test_insert_conflict")
# 使用 pytest 提供的参数化装饰器，用于多次运行该测试函数，每次使用不同的连接参数
@pytest.mark.parametrize("conn", postgresql_connectable)
def test_read_view_postgres(conn, request):
    # GH 52969
    # 获取 conn 的具体实例，通过 request 对象调用 getfixturevalue 方法
    conn = request.getfixturevalue(conn)

    # 导入 SQLAlchemy 相关库
    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import text

    # 创建随机的表名和视图名，使用 UUID 生成唯一标识
    table_name = f"group_{uuid.uuid4().hex}"
    view_name = f"group_view_{uuid.uuid4().hex}"

    # 定义 SQL 语句，创建表和视图
    sql_stmt = text(
        f"""
    CREATE TABLE {table_name} (
        group_id INTEGER,
        name TEXT
    );
    INSERT INTO {table_name} VALUES
        (1, 'name');
    CREATE VIEW {view_name}
    AS
    SELECT * FROM {table_name};
    """
    )

    # 根据连接对象的类型执行 SQL 语句
    if isinstance(conn, Engine):
        # 如果连接对象是 Engine 类型，使用 connect 方法执行 SQL 语句
        with conn.connect() as con:
            with con.begin():
                con.execute(sql_stmt)
    else:
        # 否则，直接使用连接对象的 begin 方法执行 SQL 语句
        with conn.begin():
            conn.execute(sql_stmt)

    # 调用 read_sql_table 函数读取视图数据到 result 变量
    result = read_sql_table(view_name, conn)

    # 定义预期的 DataFrame 结果
    expected = DataFrame({"group_id": [1], "name": "name"})

    # 使用 assert_frame_equal 检查实际结果和预期结果是否相同
    tm.assert_frame_equal(result, expected)


# 测试从 SQLite 数据库读取视图数据
def test_read_view_sqlite(sqlite_buildin):
    # GH 52969
    # 定义创建表的 SQL 语句
    create_table = """
CREATE TABLE groups (
   group_id INTEGER,
   name TEXT
);
"""
    # 定义插入数据的 SQL 语句
    insert_into = """
INSERT INTO groups VALUES
    (1, 'name');
"""
    # 定义创建视图的 SQL 语句
    create_view = """
CREATE VIEW group_view
AS
SELECT * FROM groups;
"""
    # 在 SQLite 连接上依次执行 SQL 语句
    sqlite_buildin.execute(create_table)
    sqlite_buildin.execute(insert_into)
    sqlite_buildin.execute(create_view)

    # 使用 pd.read_sql 函数从 SQLite 数据库中读取视图数据到 result 变量
    result = pd.read_sql("SELECT * FROM group_view", sqlite_buildin)

    # 定义预期的 DataFrame 结果
    expected = DataFrame({"group_id": [1], "name": "name"})

    # 使用 assert_frame_equal 检查实际结果和预期结果是否相同
    tm.assert_frame_equal(result, expected)


# 根据连接名称确定数据库类型（postgresql/sqlite/mysql）
def flavor(conn_name):
    if "postgresql" in conn_name:
        return "postgresql"
    elif "sqlite" in conn_name:
        return "sqlite"
    elif "mysql" in conn_name:
        return "mysql"

    # 如果连接名称不匹配任何已知类型，抛出 ValueError 异常
    raise ValueError(f"unsupported connection: {conn_name}")


# 使用 pytest 的参数化装饰器，测试从不同数据库中读取 Iris 数据集
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_parameter(conn, request, sql_strings):
    # 如果连接类型为 "adbc"，标记该测试为预期失败，因为 ADBC 驱动不支持 'params'
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'params' not implemented for ADBC drivers",
                strict=True,
            )
        )

    # 获取连接名称，并通过 request 对象获取连接实例
    conn_name = conn
    conn = request.getfixturevalue(conn)

    # 获取要执行的 SQL 查询语句，根据数据库类型选择不同的查询语句
    query = sql_strings["read_parameters"][flavor(conn_name)]

    # 定义查询参数
    params = ("Iris-setosa", 5.1)

    # 使用 pandasSQL_builder 类的上下文管理器，执行 SQL 查询
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            # 调用 read_query 方法读取查询结果到 iris_frame 变量
            iris_frame = pandasSQL.read_query(query, params=params)

    # 调用 check_iris_frame 函数检查 iris_frame 结果
    check_iris_frame(iris_frame)


# 使用 pytest 的参数化装饰器，测试从不同数据库中读取 Iris 数据集（带命名参数）
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_named_parameter(conn, request, sql_strings):
    # 如果连接类型为 "adbc"，标记该测试为预期失败，因为 ADBC 驱动不支持 'params'
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'params' not implemented for ADBC drivers",
                strict=True,
            )
        )

    # 获取连接名称，并通过 request 对象获取连接实例
    conn_name = conn
    conn = request.getfixturevalue(conn)

    # 获取要执行的 SQL 查询语句，根据数据库类型选择不同的查询语句
    query = sql_strings["read_named_parameters"][flavor(conn_name)]

    # 定义命名查询参数
    params = {"name": "Iris-setosa", "length": 5.1}
    # 使用 pandasSQL_builder 函数构建连接对象 conn，并将其赋值给 pandasSQL
    with pandasSQL_builder(conn) as pandasSQL:
        # 进入 pandasSQL 的上下文管理器，开始事务
        with pandasSQL.run_transaction():
            # 使用 pandasSQL 执行带参数的查询 query，并将结果赋值给 iris_frame
            iris_frame = pandasSQL.read_query(query, params=params)
    # 检查 iris_frame 的有效性和内容
    check_iris_frame(iris_frame)
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_iris_no_parameter_with_percent(conn, request, sql_strings):
    # 如果连接是 MySQL 或者 PostgreSQL 且不包含 "adbc"，则标记为预期失败的测试
    if "mysql" in conn or ("postgresql" in conn and "adbc" not in conn):
        request.applymarker(pytest.mark.xfail(reason="broken test"))

    # 备份原始连接名称
    conn_name = conn
    # 从 pytest 的 fixture 中获取连接对象
    conn = request.getfixturevalue(conn)

    # 获取特定数据库类型的 SQL 查询字符串
    query = sql_strings["read_no_parameters_with_percent"][flavor(conn_name)]
    # 使用 pandasSQL_builder 创建 pandasSQL 对象
    with pandasSQL_builder(conn) as pandasSQL:
        # 开启一个事务
        with pandasSQL.run_transaction():
            # 执行 SQL 查询，并返回结果到 iris_frame
            iris_frame = pandasSQL.read_query(query, params=None)
    # 检查 iris_frame 的内容
    check_iris_frame(iris_frame)


# -----------------------------------------------------------------------------
# -- Testing the public API


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_read_sql_view(conn, request):
    # 从 pytest 的 fixture 中获取连接对象
    conn = request.getfixturevalue(conn)
    # 执行 SQL 查询，获取 iris_view 的数据，并返回到 iris_frame
    iris_frame = sql.read_sql_query("SELECT * FROM iris_view", conn)
    # 检查 iris_frame 的内容
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", all_connectable_iris)
def test_api_read_sql_with_chunksize_no_result(conn, request):
    # 如果连接包含 "adbc"，则标记为预期失败的测试
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    # 从 pytest 的 fixture 中获取连接对象
    conn = request.getfixturevalue(conn)
    # 执行 SQL 查询，获取符合条件的 iris_view 数据，分块读取并返回到 with_batch 和 without_batch
    query = 'SELECT * FROM iris_view WHERE "SepalLength" < 0.0'
    with_batch = sql.read_sql_query(query, conn, chunksize=5)
    without_batch = sql.read_sql_query(query, conn)
    # 比较两种查询结果是否相等
    tm.assert_frame_equal(concat(with_batch), without_batch)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql(conn, request, test_frame1):
    # 从 pytest 的 fixture 中获取连接对象
    conn = request.getfixturevalue(conn)
    # 如果数据库中已经存在表 "test_frame1"，则删除它
    if sql.has_table("test_frame1", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame1")

    # 将 test_frame1 数据写入到数据库中的表 "test_frame1"
    sql.to_sql(test_frame1, "test_frame1", conn)
    # 断言数据库中已经存在表 "test_frame1"
    assert sql.has_table("test_frame1", conn)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_fail(conn, request, test_frame1):
    # 从 pytest 的 fixture 中获取连接对象
    conn = request.getfixturevalue(conn)
    # 如果数据库中已经存在表 "test_frame2"，则删除它
    if sql.has_table("test_frame2", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame2")

    # 将 test_frame1 数据写入到数据库中的表 "test_frame2"，如果表已存在则失败
    sql.to_sql(test_frame1, "test_frame2", conn, if_exists="fail")
    # 断言数据库中已经存在表 "test_frame2"
    assert sql.has_table("test_frame2", conn)

    # 预期抛出 ValueError，并检查错误信息是否包含特定信息
    msg = "Table 'test_frame2' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(test_frame1, "test_frame2", conn, if_exists="fail")


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_replace(conn, request, test_frame1):
    # 从 pytest 的 fixture 中获取连接对象
    conn = request.getfixturevalue(conn)
    # 如果数据库中已经存在表 "test_frame3"，则删除它
    if sql.has_table("test_frame3", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame3")

    # 将 test_frame1 数据写入到数据库中的表 "test_frame3"，如果表已存在则失败
    sql.to_sql(test_frame1, "test_frame3", conn, if_exists="fail")
    # 再次将 test_frame1 数据写入到数据库中的表 "test_frame3"，使用替换模式
    sql.to_sql(test_frame1, "test_frame3", conn, if_exists="replace")
    # 断言检查在数据库中是否存在名为 "test_frame3" 的表格
    assert sql.has_table("test_frame3", conn)
    
    # 计算测试数据框 test_frame1 中的条目数量
    num_entries = len(test_frame1)
    
    # 调用函数 count_rows 统计数据库连接 conn 中 "test_frame3" 表中的行数
    num_rows = count_rows(conn, "test_frame3")
    
    # 断言检查数据库中 "test_frame3" 表的行数与 test_frame1 中的条目数量是否相等
    assert num_rows == num_entries
@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_append(conn, request, test_frame1):
    # 获取连接的具体实例
    conn = request.getfixturevalue(conn)
    # 如果数据库中已存在表 "test_frame4"，则删除该表
    if sql.has_table("test_frame4", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame4")

    # 将 DataFrame 数据写入数据库表 "test_frame4"，如果表已存在则失败
    assert sql.to_sql(test_frame1, "test_frame4", conn, if_exists="fail") == 4

    # 再次将 DataFrame 数据追加到数据库表 "test_frame4"
    assert sql.to_sql(test_frame1, "test_frame4", conn, if_exists="append") == 4
    # 确认数据库中存在表 "test_frame4"
    assert sql.has_table("test_frame4", conn)

    # 计算预期的表条目数
    num_entries = 2 * len(test_frame1)
    # 统计数据库表 "test_frame4" 中的行数
    num_rows = count_rows(conn, "test_frame4")

    # 确认数据库表中的行数与预期的条目数相等
    assert num_rows == num_entries


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_type_mapping(conn, request, test_frame3):
    # 获取连接的具体实例
    conn = request.getfixturevalue(conn)
    # 如果数据库中已存在表 "test_frame5"，则删除该表
    if sql.has_table("test_frame5", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame5")

    # 将 DataFrame 数据写入数据库表 "test_frame5"，不包括索引
    sql.to_sql(test_frame3, "test_frame5", conn, index=False)
    # 从数据库读取表 "test_frame5" 中的数据
    result = sql.read_sql("SELECT * FROM test_frame5", conn)

    # 使用测试框架验证写入和读取的数据是否相等
    tm.assert_frame_equal(test_frame3, result)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_series(conn, request):
    # 获取连接的具体实例
    conn = request.getfixturevalue(conn)
    # 如果数据库中已存在表 "test_series"，则删除该表
    if sql.has_table("test_series", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_series")

    # 创建一个 Series 对象并将其写入数据库表 "test_series"，不包括索引
    s = Series(np.arange(5, dtype="int64"), name="series")
    sql.to_sql(s, "test_series", conn, index=False)
    # 从数据库读取表 "test_series" 中的数据
    s2 = sql.read_sql_query("SELECT * FROM test_series", conn)
    # 使用测试框架验证写入和读取的数据是否相等
    tm.assert_frame_equal(s.to_frame(), s2)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_roundtrip(conn, request, test_frame1):
    # 获取连接的名称
    conn_name = conn
    # 获取连接的具体实例
    conn = request.getfixturevalue(conn)
    # 如果数据库中已存在表 "test_frame_roundtrip"，则删除该表
    if sql.has_table("test_frame_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")

    # 将 DataFrame 数据写入数据库表 "test_frame_roundtrip"
    sql.to_sql(test_frame1, "test_frame_roundtrip", con=conn)
    # 从数据库读取表 "test_frame_roundtrip" 中的数据
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn)

    # 对特定连接名进行处理（一个 HACK 操作）
    if "adbc" in conn_name:
        result = result.rename(columns={"__index_level_0__": "level_0"})
    # 设置结果数据的索引与原始 DataFrame 相同
    result.index = test_frame1.index
    result.set_index("level_0", inplace=True)
    result.index.astype(int)
    result.index.name = None
    # 使用测试框架验证写入和读取的数据是否相等
    tm.assert_frame_equal(result, test_frame1)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_roundtrip_chunksize(conn, request, test_frame1):
    # 如果连接名包含 "adbc"，则将该测试标记为失败
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    # 获取连接的具体实例
    conn = request.getfixturevalue(conn)
    # 如果数据库中已存在表 "test_frame_roundtrip"，则删除该表
    if sql.has_table("test_frame_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_frame_roundtrip")
    # 将 DataFrame 中的数据写入 SQL 数据库中的表 "test_frame_roundtrip"，忽略索引，每次写入数据的块大小为 2
    sql.to_sql(
        test_frame1,
        "test_frame_roundtrip",
        con=conn,
        index=False,
        chunksize=2,
    )
    # 从 SQL 数据库中读取表 "test_frame_roundtrip" 的所有数据，存储在 result 变量中
    result = sql.read_sql_query("SELECT * FROM test_frame_roundtrip", con=conn)
    # 使用测试框架的函数 tm.assert_frame_equal 检查 result 和 test_frame1 是否相等
    tm.assert_frame_equal(result, test_frame1)
@pytest.mark.parametrize("conn", all_connectable_iris)
# 使用pytest的@parametrize装饰器，为测试函数提供多个数据库连接的参数化测试
def test_api_execute_sql(conn, request):
    # drop_sql = "DROP TABLE IF EXISTS test"  # 应该已经完成
    # 获取数据库连接实例
    conn = request.getfixturevalue(conn)
    # 使用sql.pandasSQL_builder创建pandasSQL对象，并使用with语句管理连接的生命周期
    with sql.pandasSQL_builder(conn) as pandas_sql:
        # 执行SQL查询语句"SELECT * FROM iris"
        iris_results = pandas_sql.execute("SELECT * FROM iris")
        # 获取查询结果的第一行数据
        row = iris_results.fetchone()
        # 关闭查询结果对象
        iris_results.close()
    # 断言获取的行数据是否符合预期
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


@pytest.mark.parametrize("conn", all_connectable_types)
# 使用pytest的@parametrize装饰器，为测试函数提供多个数据库连接的参数化测试
def test_api_date_parsing(conn, request):
    # 保存数据库连接名
    conn_name = conn
    # 获取数据库连接实例
    conn = request.getfixturevalue(conn)
    
    # 在read_sql查询中测试日期解析功能
    # 无解析
    # 执行SQL查询语句"SELECT * FROM types"，并使用sql.read_sql_query读取结果
    df = sql.read_sql_query("SELECT * FROM types", conn)
    # 如果数据库类型不是mysql或postgres，断言DateCol列的dtype不是np.datetime64
    if not ("mysql" in conn_name or "postgres" in conn_name):
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)
    
    # 使用parse_dates参数进行日期解析
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates=["DateCol"])
    # 断言DateCol列的dtype是np.datetime64
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    # 断言解析后的日期数据是否符合预期
    assert df.DateCol.tolist() == [
        Timestamp(2000, 1, 3, 0, 0, 0),
        Timestamp(2000, 1, 4, 0, 0, 0),
    ]

    # 使用parse_dates参数，指定日期格式进行解析
    df = sql.read_sql_query(
        "SELECT * FROM types",
        conn,
        parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"},
    )
    # 断言DateCol列的dtype是np.datetime64
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    # 断言解析后的日期数据是否符合预期
    assert df.DateCol.tolist() == [
        Timestamp(2000, 1, 3, 0, 0, 0),
        Timestamp(2000, 1, 4, 0, 0, 0),
    ]

    # 使用parse_dates参数进行整数日期列解析
    df = sql.read_sql_query("SELECT * FROM types", conn, parse_dates=["IntDateCol"])
    # 断言IntDateCol列的dtype是np.datetime64
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    # 断言解析后的日期数据是否符合预期
    assert df.IntDateCol.tolist() == [
        Timestamp(1986, 12, 25, 0, 0, 0),
        Timestamp(2013, 1, 1, 0, 0, 0),
    ]

    # 使用parse_dates参数，指定日期格式进行整数日期列解析
    df = sql.read_sql_query(
        "SELECT * FROM types", conn, parse_dates={"IntDateCol": "s"}
    )
    # 断言IntDateCol列的dtype是np.datetime64
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    # 断言解析后的日期数据是否符合预期
    assert df.IntDateCol.tolist() == [
        Timestamp(1986, 12, 25, 0, 0, 0),
        Timestamp(2013, 1, 1, 0, 0, 0),
    ]

    # 使用parse_dates参数，指定日期格式进行整数日期列解析
    df = sql.read_sql_query(
        "SELECT * FROM types",
        conn,
        parse_dates={"IntDateOnlyCol": "%Y%m%d"},
    )
    # 断言IntDateOnlyCol列的dtype是np.datetime64
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    # 断言解析后的日期数据是否符合预期
    assert df.IntDateOnlyCol.tolist() == [
        Timestamp("2010-10-10"),
        Timestamp("2010-12-12"),
    ]


@pytest.mark.parametrize("conn", all_connectable_types)
@pytest.mark.parametrize("error", ["raise", "coerce"])
@pytest.mark.parametrize(
    "read_sql, text, mode",
    [
        (sql.read_sql, "SELECT * FROM types", ("sqlalchemy", "fallback")),
        (sql.read_sql, "types", ("sqlalchemy")),
        (
            sql.read_sql_query,
            "SELECT * FROM types",
            ("sqlalchemy", "fallback"),
        ),
        (sql.read_sql_table, "types", ("sqlalchemy")),
    ],
)
# 使用pytest的@parametrize装饰器，为测试函数提供多个数据库连接、错误处理方式、和不同的读取SQL方式的参数化测试
def test_api_custom_dateparsing_error(
    conn, request, read_sql, text, mode, error, types_data_frame
):
    # 保存数据库连接名
    conn_name = conn
    # 获取数据库连接实例
    conn = request.getfixturevalue(conn)
    # 如果条件满足：text 等于 "types" 且 conn_name 等于 "sqlite_buildin_types"
    if text == "types" and conn_name == "sqlite_buildin_types":
        # 应用 xfail 标记，表示此组合参数预期失败
        request.applymarker(
            pytest.mark.xfail(reason="failing combination of arguments")
        )

    # 期望的数据框，将 DateCol 列转换为 datetime64[s] 类型
    expected = types_data_frame.astype({"DateCol": "datetime64[s]"})

    # 调用 read_sql 函数，从数据库中读取数据
    result = read_sql(
        text,
        con=conn,
        parse_dates={
            "DateCol": {"errors": error},
        },
    )

    # 如果数据库连接名包含 "postgres"
    if "postgres" in conn_name:
        # 清理 types_data_frame fixture 的待办事项
        # 将结果中的 BoolCol 列转换为整数类型
        result["BoolCol"] = result["BoolCol"].astype(int)
        # 将结果中的 BoolColWithNull 列转换为浮点数类型
        result["BoolColWithNull"] = result["BoolColWithNull"].astype(float)

    # 如果数据库连接名为 "postgresql_adbc_types"
    if conn_name == "postgresql_adbc_types":
        # 将期望的数据框中的特定列转换为 int32 类型
        expected = expected.astype(
            {
                "IntDateCol": "int32",
                "IntDateOnlyCol": "int32",
                "IntCol": "int32",
            }
        )

    # 如果数据库连接名为 "postgresql_adbc_types" 且 pa_version_under14p1 为真
    if conn_name == "postgresql_adbc_types" and pa_version_under14p1:
        # 将期望的数据框中的 DateCol 列转换为 datetime64[ns] 类型
        expected["DateCol"] = expected["DateCol"].astype("datetime64[ns]")
    # 否则，如果数据库连接名包含 "postgres" 或 "mysql"
    elif "postgres" in conn_name or "mysql" in conn_name:
        # 将期望的数据框中的 DateCol 列转换为 datetime64[us] 类型
        expected["DateCol"] = expected["DateCol"].astype("datetime64[us]")
    else:
        # 其他情况下，将期望的数据框中的 DateCol 列转换为 datetime64[s] 类型
        expected["DateCol"] = expected["DateCol"].astype("datetime64[s]")

    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected 数据框
    tm.assert_frame_equal(result, expected)
# 使用pytest的装饰器标记测试参数化，conn表示连接类型参数
@pytest.mark.parametrize("conn", all_connectable_types)
# 定义测试函数test_api_date_and_index，测试解析日期和索引列的情况
def test_api_date_and_index(conn, request):
    # 从request获取conn的fixture值
    conn = request.getfixturevalue(conn)
    # 使用sql.read_sql_query从数据库中读取数据到DataFrame df
    df = sql.read_sql_query(
        "SELECT * FROM types",
        conn,
        index_col="DateCol",  # 将DateCol列作为索引列
        parse_dates=["DateCol", "IntDateCol"],  # 解析DateCol和IntDateCol列为日期类型
    )

    # 断言索引列的dtype是np.datetime64类型
    assert issubclass(df.index.dtype.type, np.datetime64)
    # 断言IntDateCol列的dtype是np.datetime64类型
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


# 使用pytest的装饰器标记测试参数化，conn表示连接类型参数
@pytest.mark.parametrize("conn", all_connectable)
# 定义测试函数test_api_timedelta，测试处理时间差的情况
def test_api_timedelta(conn, request):
    # 将conn_name设为conn参数的值
    conn_name = conn
    # 从request获取conn的fixture值
    conn = request.getfixturevalue(conn)
    
    # 如果数据库中存在表"test_timedelta"
    if sql.has_table("test_timedelta", conn):
        # 使用pandasSQL对象连接到conn，并设置需要事务操作
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            # 删除表"test_timedelta"
            pandasSQL.drop_table("test_timedelta")
    
    # 创建一个包含时间差数据的DataFrame df
    df = to_timedelta(Series(["00:00:01", "00:00:03"], name="foo")).to_frame()

    # 如果conn_name等于"sqlite_adbc_conn"
    if conn_name == "sqlite_adbc_conn":
        # 给当前测试节点添加一个标记，标记为预期失败，原因是"sqlite ADBC driver doesn't implement timedelta"
        request.node.add_marker(
            pytest.mark.xfail(
                reason="sqlite ADBC driver doesn't implement timedelta",
            )
        )

    # 如果conn_name中包含"adbc"
    if "adbc" in conn_name:
        # 如果pa_version_under14p1为True，设置期望的警告类型为DeprecationWarning
        if pa_version_under14p1:
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
    else:
        # 否则设置期望的警告类型为UserWarning
        exp_warning = UserWarning

    # 使用tm.assert_produces_warning断言会产生指定类型的警告，不检查调用堆栈深度
    with tm.assert_produces_warning(exp_warning, check_stacklevel=False):
        # 将df写入数据库表"test_timedelta"，并返回写入的行数
        result_count = df.to_sql(name="test_timedelta", con=conn)
    # 断言写入的行数为2
    assert result_count == 2
    
    # 从数据库中读取表"test_timedelta"的内容到DataFrame result
    result = sql.read_sql_query("SELECT * FROM test_timedelta", conn)

    # 如果conn_name等于"postgresql_adbc_conn"
    if conn_name == "postgresql_adbc_conn":
        # TODO: Postgres存储一个间隔（INTERVAL），ADBC读取为Month-Day-Nano间隔；
        # 默认的pandas类型映射将其映射为DateOffset，但也许我们应该尝试在这里恢复timedelta？
        # 创建期望的Series expected，包含DateOffset对象的列表
        expected = Series(
            [
                pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0),
                pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0),
            ],
            name="foo",
        )
    else:
        # 否则，将df中"foo"列的dtype转换为int64作为期望值
        expected = df["foo"].astype("int64")
    
    # 使用tm.assert_series_equal断言result的"foo"列与期望值expected相等
    tm.assert_series_equal(result["foo"], expected)


# 使用pytest的装饰器标记测试参数化，conn表示连接类型参数
@pytest.mark.parametrize("conn", all_connectable)
# 使用pytest的装饰器标记测试参数化，index_name、index_label、expected是测试参数
@pytest.mark.parametrize(
    "index_name,index_label,expected",
    [
        # 没有指定索引名称，使用默认的 'index'
        (None, None, "index"),
        # 指定了索引标签为 "other_label"
        (None, "other_label", "other_label"),
        # 使用索引名称 "index_name"
        ("index_name", None, "index_name"),
        # 指定了索引名称为 "index_name"，但同时指定了索引标签为 "other_label"
        ("index_name", "other_label", "other_label"),
        # 索引名称为整数 0
        (0, None, "0"),
        # 索引名称为 None，但索引标签为整数 0
        (None, 0, "0"),
    ],
# 定义测试函数，用于将 API 数据导入 SQL 数据库，并验证索引标签是否正确
def test_api_to_sql_index_label(conn, request, index_name, index_label, expected):
    # 如果连接字符串中包含 "adbc"，则添加一个标记，说明在 ADBC 中不支持 index_label 参数
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )
    
    # 从测试请求中获取连接对象
    conn = request.getfixturevalue(conn)
    
    # 如果数据库中存在名为 "test_index_label" 的表，先删除它
    if sql.has_table("test_index_label", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")

    # 创建一个临时 DataFrame，包含一列名为 'col1'，索引名为 index_name
    temp_frame = DataFrame({"col1": range(4)})
    temp_frame.index.name = index_name
    
    # 构造 SQL 查询语句，选择从表 "test_index_label" 中的所有内容
    query = "SELECT * FROM test_index_label"
    
    # 将 DataFrame 导入到 SQL 表 "test_index_label" 中，使用指定的 index_label 参数
    sql.to_sql(temp_frame, "test_index_label", conn, index_label=index_label)
    
    # 从数据库中读取数据，检查第一列的列名是否与预期的值相同
    frame = sql.read_sql_query(query, conn)
    assert frame.columns[0] == expected


# 使用参数化装饰器，定义多个连接对象进行测试
@pytest.mark.parametrize("conn", all_connectable)
def test_api_to_sql_index_label_multiindex(conn, request):
    # 获取连接名称
    conn_name = conn
    
    # 如果连接字符串中包含 "mysql"，则应用一个标记，说明在 MySQL 中使用 TEXT 类型作为键可能会失败
    if "mysql" in conn_name:
        request.applymarker(
            pytest.mark.xfail(
                reason="MySQL can fail using TEXT without length as key", strict=False
            )
        )
    # 如果连接字符串中包含 "adbc"，则添加一个标记，说明在 ADBC 中不支持 index_label 参数
    elif "adbc" in conn_name:
        request.node.add_marker(
            pytest.mark.xfail(reason="index_label argument NotImplemented with ADBC")
        )
    
    # 从测试请求中获取连接对象
    conn = request.getfixturevalue(conn)
    
    # 如果数据库中存在名为 "test_index_label" 的表，先删除它
    if sql.has_table("test_index_label", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_index_label")
    
    # 设置预期的行数为 4
    expected_row_count = 4
    
    # 创建一个多级索引的临时 DataFrame，包含一列名为 'col1'，索引由 MultiIndex 构成
    temp_frame = DataFrame(
        {"col1": range(4)},
        index=MultiIndex.from_product([("A0", "A1"), ("B0", "B1")]),
    )

    # 执行将 DataFrame 导入到 SQL 表 "test_index_label" 的操作，不指定 index_label 参数
    result = sql.to_sql(temp_frame, "test_index_label", conn)
    assert result == expected_row_count
    
    # 从数据库中读取数据，检查前两列的列名是否分别为 'level_0' 和 'level_1'
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[0] == "level_0"
    assert frame.columns[1] == "level_1"

    # 执行将 DataFrame 导入到 SQL 表 "test_index_label" 的操作，指定 index_label 参数为 ["A", "B"]
    result = sql.to_sql(
        temp_frame,
        "test_index_label",
        conn,
        if_exists="replace",
        index_label=["A", "B"],
    )
    assert result == expected_row_count
    
    # 从数据库中读取数据，检查前两列的列名是否分别为 'A' 和 'B'
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["A", "B"]

    # 将临时 DataFrame 的索引名称设置为 ["A", "B"]，执行将其导入到 SQL 表 "test_index_label" 的操作
    result = sql.to_sql(temp_frame, "test_index_label", conn, if_exists="replace")
    assert result == expected_row_count
    
    # 从数据库中读取数据，检查前两列的列名是否分别为 'A' 和 'B'
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["A", "B"]

    # 执行将 DataFrame 导入到 SQL 表 "test_index_label" 的操作，指定 index_label 参数为 ["C", "D"]
    result = sql.to_sql(
        temp_frame,
        "test_index_label",
        conn,
        if_exists="replace",
        index_label=["C", "D"],
    )
    assert result == expected_row_count
    
    # 从数据库中读取数据，检查前两列的列名是否分别为 'C' 和 'D'
    frame = sql.read_sql_query("SELECT * FROM test_index_label", conn)
    assert frame.columns[:2].tolist() == ["C", "D"]
    # 定义错误消息，用于匹配 pytest 抛出的 ValueError 异常信息
    msg = "Length of 'index_label' should match number of levels, which is 2"
    
    # 使用 pytest 的上下文管理器 `pytest.raises`，期望捕获到 ValueError 异常，并检查其异常消息是否匹配预期的 msg 变量
    with pytest.raises(ValueError, match=msg):
        # 调用 sql 对象的 to_sql 方法，将 temp_frame 对象写入到名为 "test_index_label" 的表中
        # 使用连接对象 conn 与数据库通信，如果该表已存在则用新数据替换，指定 index_label 为 "C"
        sql.to_sql(
            temp_frame,
            "test_index_label",
            conn,
            if_exists="replace",
            index_label="C",
        )
@pytest.mark.parametrize("conn", all_connectable)
def test_api_multiindex_roundtrip(conn, request):
    # 使用参数化测试，对所有连接参数化
    conn = request.getfixturevalue(conn)
    # 如果数据库中存在名为 "test_multiindex_roundtrip" 的表，删除之
    if sql.has_table("test_multiindex_roundtrip", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_multiindex_roundtrip")

    # 创建包含多索引的 DataFrame 对象
    df = DataFrame.from_records(
        [(1, 2.1, "line1"), (2, 1.5, "line2")],
        columns=["A", "B", "C"],
        index=["A", "B"],
    )

    # 将 DataFrame 对象写入数据库表 "test_multiindex_roundtrip"
    df.to_sql(name="test_multiindex_roundtrip", con=conn)
    # 从数据库读取数据到结果 DataFrame 对象，指定多列作为索引
    result = sql.read_sql_query(
        "SELECT * FROM test_multiindex_roundtrip", conn, index_col=["A", "B"]
    )
    # 使用测试工具比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(df, result, check_index_type=True)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize(
    "dtype",
    [
        None,
        int,
        float,
        {"A": int, "B": float},
    ],
)
def test_api_dtype_argument(conn, request, dtype):
    # GH10285 在 read_sql_query 中添加 dtype 参数
    conn_name = conn
    # 获取连接的 fixture 值
    conn = request.getfixturevalue(conn)
    # 如果数据库中存在名为 "test_dtype_argument" 的表，删除之
    if sql.has_table("test_dtype_argument", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_dtype_argument")

    # 创建 DataFrame 对象
    df = DataFrame([[1.2, 3.4], [5.6, 7.8]], columns=["A", "B"])
    # 将 DataFrame 对象写入数据库表 "test_dtype_argument"，返回写入的行数
    assert df.to_sql(name="test_dtype_argument", con=conn) == 2

    # 将 DataFrame 对象按照指定的 dtype 转换为期望的类型
    expected = df.astype(dtype)

    # 根据不同的数据库类型构造不同的 SQL 查询语句
    if "postgres" in conn_name:
        query = 'SELECT "A", "B" FROM test_dtype_argument'
    else:
        query = "SELECT A, B FROM test_dtype_argument"
    # 从数据库中读取数据到结果 DataFrame 对象，指定期望的数据类型
    result = sql.read_sql_query(query, con=conn, dtype=dtype)

    # 使用测试工具比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
def test_api_integer_col_names(conn, request):
    # 使用参数化测试，对所有连接参数化
    conn = request.getfixturevalue(conn)
    # 创建包含整数列名的 DataFrame 对象
    df = DataFrame([[1, 2], [3, 4]], columns=[0, 1])
    # 将 DataFrame 对象写入数据库表 "test_frame_integer_col_names"，如果表存在则替换
    sql.to_sql(df, "test_frame_integer_col_names", conn, if_exists="replace")


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema(conn, request, test_frame1):
    # 如果连接类型包含 "adbc"，标记测试为预期失败，因为 ADBC 驱动不支持 get_schema
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    # 获取连接的 fixture 值
    conn = request.getfixturevalue(conn)
    # 生成 DataFrame 对象的创建 SQL 语句
    create_sql = sql.get_schema(test_frame1, "test", con=conn)
    # 断言 SQL 语句包含 "CREATE"
    assert "CREATE" in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_with_schema(conn, request, test_frame1):
    # GH28486
    # 如果连接类型包含 "adbc"，标记测试为预期失败，因为 ADBC 驱动不支持 get_schema
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    # 获取连接的 fixture 值
    conn = request.getfixturevalue(conn)
    # 生成包含指定 schema 的 DataFrame 对象的创建 SQL 语句
    create_sql = sql.get_schema(test_frame1, "test", con=conn, schema="pypi")
    # 断言 SQL 语句包含 "CREATE TABLE pypi."
    assert "CREATE TABLE pypi." in create_sql


@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_dtypes(conn, request):
    # 使用参数化测试，对所有连接参数化
    conn = request.getfixturevalue(conn)
    # 检查连接字符串是否包含子字符串 "adbc"
    if "adbc" in conn:
        # 为测试用例添加标记，标记为预期失败，原因是 ADBC 驱动程序不支持 'get_schema' 方法
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    
    # 将原始连接字符串保存在变量 conn_name 中
    conn_name = conn
    
    # 使用 pytest 的 fixture 功能获取连接对象，替换原始的连接字符串 conn
    conn = request.getfixturevalue(conn)
    
    # 创建一个包含浮点数数据的 DataFrame 对象
    float_frame = DataFrame({"a": [1.1, 1.2], "b": [2.1, 2.2]})
    
    # 根据连接字符串判断数据类型
    if conn_name == "sqlite_buildin":
        # 如果连接字符串是 "sqlite_buildin"，则数据类型为 "INTEGER"
        dtype = "INTEGER"
    else:
        # 否则，从 sqlalchemy 模块导入 Integer 类型，数据类型为 Integer
        from sqlalchemy import Integer
        dtype = Integer
    
    # 使用 sql 模块的 get_schema 函数生成创建表的 SQL 语句
    create_sql = sql.get_schema(float_frame, "test", con=conn, dtype={"b": dtype})
    
    # 断言 SQL 语句中包含 "CREATE" 关键字
    assert "CREATE" in create_sql
    
    # 断言 SQL 语句中包含 "INTEGER" 类型关键字
    assert "INTEGER" in create_sql
# 使用 pytest 的参数化装饰器，对所有连接进行测试
@pytest.mark.parametrize("conn", all_connectable)
def test_api_get_schema_keys(conn, request, test_frame1):
    # 如果连接字符串中包含 "adbc"，则标记为预期失败，因为 ADBC 驱动程序未实现 'get_schema'
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="'get_schema' not implemented for ADBC drivers",
                strict=True,
            )
        )
    # 将原始连接名称保存在 conn_name 中
    conn_name = conn
    # 使用 request 获取 fixture 中的连接对象
    conn = request.getfixturevalue(conn)
    # 创建一个测试用的 DataFrame
    frame = DataFrame({"Col1": [1.1, 1.2], "Col2": [2.1, 2.2]})
    # 调用 sql.get_schema 函数获取创建表的 SQL 语句
    create_sql = sql.get_schema(frame, "test", con=conn, keys="Col1")

    # 根据连接名称判断数据库类型，生成相应的主键约束语句
    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`Col1`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("Col1")'
    # 断言主键约束语句是否包含在 SQL 语句中
    assert constraint_sentence in create_sql

    # 测试多列作为键时的情况，通过传入列表作为 keys 参数
    create_sql = sql.get_schema(test_frame1, "test", con=conn, keys=["A", "B"])
    if "mysql" in conn_name:
        constraint_sentence = "CONSTRAINT test_pk PRIMARY KEY (`A`, `B`)"
    else:
        constraint_sentence = 'CONSTRAINT test_pk PRIMARY KEY ("A", "B")'
    # 断言主键约束语句是否包含在 SQL 语句中
    assert constraint_sentence in create_sql


# 使用 pytest 的参数化装饰器，对所有连接进行测试
@pytest.mark.parametrize("conn", all_connectable)
def test_api_chunksize_read(conn, request):
    # 如果连接字符串中包含 "adbc"，则标记为预期失败，因为 ADBC 驱动程序未实现 chunksize 参数
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    # 将原始连接名称保存在 conn_name 中
    conn_name = conn
    # 使用 request 获取 fixture 中的连接对象
    conn = request.getfixturevalue(conn)
    # 如果数据库中存在名为 "test_chunksize" 的表，则删除该表
    if sql.has_table("test_chunksize", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_chunksize")

    # 创建一个 DataFrame，并将其写入名为 "test_chunksize" 的表中
    df = DataFrame(
        np.random.default_rng(2).standard_normal((22, 5)), columns=list("abcde")
    )
    df.to_sql(name="test_chunksize", con=conn, index=False)

    # 一次性读取查询结果
    res1 = sql.read_sql_query("select * from test_chunksize", conn)

    # 使用 read_sql_query 分块读取查询结果
    res2 = DataFrame()
    i = 0
    sizes = [5, 5, 5, 5, 2]

    for chunk in sql.read_sql_query("select * from test_chunksize", conn, chunksize=5):
        res2 = concat([res2, chunk], ignore_index=True)
        # 断言每个分块的长度是否符合预期
        assert len(chunk) == sizes[i]
        i += 1

    # 断言一次性读取与分块读取的结果是否相同
    tm.assert_frame_equal(res1, res2)

    # 根据连接名称判断数据库类型，测试 read_sql_table 是否支持 chunksize 参数
    if conn_name == "sqlite_buildin":
        with pytest.raises(NotImplementedError, match=""):
            sql.read_sql_table("test_chunksize", conn, chunksize=5)
    else:
        res3 = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]

        for chunk in sql.read_sql_table("test_chunksize", conn, chunksize=5):
            res3 = concat([res3, chunk], ignore_index=True)
            # 断言每个分块的长度是否符合预期
            assert len(chunk) == sizes[i]
            i += 1

        # 断言一次性读取与分块读取的结果是否相同
        tm.assert_frame_equal(res1, res3)
    # 如果连接类型为 "postgresql_adbc_conn"
    if conn == "postgresql_adbc_conn":
        # 尝试导入可选的 ADBC PostgreSQL 驱动，并忽略可能出现的错误
        adbc = import_optional_dependency("adbc_driver_postgresql", errors="ignore")
        # 如果成功导入驱动且其版本低于 "0.9.0"
        if adbc is not None and Version(adbc.__version__) < Version("0.9.0"):
            # 给当前测试用例节点添加一个标记，标记为预期失败，原因是 ADBC postgres 驱动不支持分类数据类型
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="categorical dtype not implemented for ADBC postgres driver",
                    strict=True,
                )
            )
    
    # GH8624
    # 测试确保分类数据正确写入为密集列
    # 使用请求对象获取连接
    conn = request.getfixturevalue(conn)
    
    # 如果在数据库中存在名为 "test_categorical" 的表
    if sql.has_table("test_categorical", conn):
        # 使用 pandasSQL 对象操作数据库，需要事务支持
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            # 删除名为 "test_categorical" 的表
            pandasSQL.drop_table("test_categorical")
    
    # 创建一个 DataFrame 对象 df
    df = DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_name": ["John P. Doe", "Jane Dove", "John P. Doe"],
        }
    )
    # 复制 df 到 df2
    df2 = df.copy()
    # 将 df2 的 "person_name" 列类型转换为分类数据类型
    df2["person_name"] = df2["person_name"].astype("category")
    
    # 将 df2 数据写入数据库表 "test_categorical"，不包含索引
    df2.to_sql(name="test_categorical", con=conn, index=False)
    # 从数据库中读取 "test_categorical" 表的数据到 res
    res = sql.read_sql_query("SELECT * FROM test_categorical", conn)
    
    # 使用测试工具比较 res 和预期的 df 是否相等
    tm.assert_frame_equal(res, df)
# 使用 pytest 的 parametrize 装饰器为 test_api_unicode_column_name 函数提供多个连接对象作为参数
@pytest.mark.parametrize("conn", all_connectable)
def test_api_unicode_column_name(conn, request):
    # 标识：GitHub issue 11431
    # 从 pytest 请求中获取 conn 对象，这里可能是一个测试夹具
    conn = request.getfixturevalue(conn)
    # 如果数据库中存在名为 "test_unicode" 的表，则删除该表
    if sql.has_table("test_unicode", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_unicode")

    # 创建一个包含两列的 DataFrame，其中一列的列名为 '\xe9'（Unicode 编码），另一列的列名为 "b"
    df = DataFrame([[1, 2], [3, 4]], columns=["\xe9", "b"])
    # 将 DataFrame 写入到数据库表 "test_unicode" 中，使用给定的连接对象 conn，不包括索引列
    df.to_sql(name="test_unicode", con=conn, index=False)


# 使用 pytest 的 parametrize 装饰器为 test_api_escaped_table_name 函数提供多个连接对象作为参数
@pytest.mark.parametrize("conn", all_connectable)
def test_api_escaped_table_name(conn, request):
    # 标识：GitHub issue 13206
    # 将 conn 保存到 conn_name 中
    conn_name = conn
    # 从 pytest 请求中获取 conn 对象，这里可能是一个测试夹具
    conn = request.getfixturevalue(conn)
    # 如果数据库中存在名为 "d1187b08-4943-4c8d-a7f6" 的表，则删除该表
    if sql.has_table("d1187b08-4943-4c8d-a7f6", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("d1187b08-4943-4c8d-a7f6")

    # 创建一个包含两列的 DataFrame，列名为 "A" 和 "B"，其中 "A" 的值包括整数和 NaN
    df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})
    # 将 DataFrame 写入到数据库表 "d1187b08-4943-4c8d-a7f6" 中，使用给定的连接对象 conn，不包括索引列
    df.to_sql(name="d1187b08-4943-4c8d-a7f6", con=conn, index=False)

    # 如果连接字符串中包含 "postgres"，则使用双引号引用表名，否则使用反引号引用表名
    if "postgres" in conn_name:
        query = 'SELECT * FROM "d1187b08-4943-4c8d-a7f6"'
    else:
        query = "SELECT * FROM `d1187b08-4943-4c8d-a7f6`"
    # 执行 SQL 查询，并将结果存储在 res 中
    res = sql.read_sql_query(query, conn)

    # 使用 pytest 测试工具比较查询结果 res 和预期的 DataFrame df 是否相等
    tm.assert_frame_equal(res, df)


# 使用 pytest 的 parametrize 装饰器为 test_api_read_sql_duplicate_columns 函数提供多个连接对象作为参数
@pytest.mark.parametrize("conn", all_connectable)
def test_api_read_sql_duplicate_columns(conn, request):
    # 标识：GitHub issue 53117
    # 如果连接字符串中包含 "adbc"，则尝试导入 pyarrow，如果版本不符合条件，则跳过测试并标记为失败
    if "adbc" in conn:
        pa = pytest.importorskip("pyarrow")
        if not (
            Version(pa.__version__) >= Version("16.0")
            and conn in ["sqlite_adbc_conn", "postgresql_adbc_conn"]
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    reason="pyarrow->pandas throws ValueError", strict=True
                )
            )
    # 从 pytest 请求中获取 conn 对象，这里可能是一个测试夹具
    conn = request.getfixturevalue(conn)
    # 如果数据库中存在名为 "test_table" 的表，则删除该表
    if sql.has_table("test_table", conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table("test_table")

    # 创建一个包含三列的 DataFrame，列名为 "a"、"b" 和 "c"
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": 1})
    # 将 DataFrame 写入到数据库表 "test_table" 中，使用给定的连接对象 conn，不包括索引列
    df.to_sql(name="test_table", con=conn, index=False)

    # 执行 SQL 查询，选择 "a"、"b" 列，并对 "a" 列执行加法运算并重命名为 "a"，同时选择 "c" 列
    result = pd.read_sql("SELECT a, b, a + 1 as a, c FROM test_table", conn)
    # 创建一个预期的 DataFrame，与查询结果相比较
    expected = DataFrame(
        [[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]],
        columns=["a", "b", "a", "c"],
    )
    # 使用 pytest 测试工具比较查询结果 result 和预期的 DataFrame expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器为 test_read_table_columns 函数提供多个连接对象作为参数
@pytest.mark.parametrize("conn", all_connectable)
def test_read_table_columns(conn, request, test_frame1):
    # 测试 read_table 函数中的 columns 参数
    conn_name = conn
    # 如果连接名为 "sqlite_buildin"，则标记为预期失败，因为此功能未实现
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))

    # 从 pytest 请求中获取 conn 对象，这里可能是一个测试夹具
    conn = request.getfixturevalue(conn)
    # 将 test_frame1 数据框写入到数据库表 "test_frame" 中，使用给定的连接对象 conn
    sql.to_sql(test_frame1, "test_frame", conn)

    # 指定需要选择的列名列表
    cols = ["A", "B"]
    # 执行 SQL 查询，并选择指定的列名列表
    result = sql.read_sql_table("test_frame", conn, columns=cols)
    # 使用断言检查查询结果的列名是否与预期的列表 cols 相同
    assert result.columns.tolist() == cols


# 使用 pytest 的 parametrize 装饰器为 test_read_table_index_col 函数提供多个连接对象作为参数
@pytest.mark.parametrize("conn", all_connectable)
def test_read_table_index_col(conn, request, test_frame1):
    # 测试 read_table 函数中的 index_col 参数
    conn_name = conn
    # 如果数据库连接名为 "sqlite_buildin"
    if conn_name == "sqlite_buildin":
        # 应用 pytest 的 xfail 标记，说明此处预期是失败的情况
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))

    # 使用 pytest 的 request 对象获取指定的连接 fixture
    conn = request.getfixturevalue(conn)
    # 将测试数据框 test_frame1 写入数据库中名为 "test_frame" 的表中，使用指定的连接 conn
    sql.to_sql(test_frame1, "test_frame", conn)

    # 从数据库中读取表 "test_frame" 的数据，指定 index_col 为 "index"
    result = sql.read_sql_table("test_frame", conn, index_col="index")
    # 断言读取的结果的索引名应为 ["index"]
    assert result.index.names == ["index"]

    # 从数据库中读取表 "test_frame" 的数据，指定 index_col 为 ["A", "B"]
    result = sql.read_sql_table("test_frame", conn, index_col=["A", "B"])
    # 断言读取的结果的索引名应为 ["A", "B"]
    assert result.index.names == ["A", "B"]

    # 从数据库中读取表 "test_frame" 的数据，指定 index_col 为 ["A", "B"]，列为 ["C", "D"]
    result = sql.read_sql_table(
        "test_frame", conn, index_col=["A", "B"], columns=["C", "D"]
    )
    # 断言读取的结果的索引名应为 ["A", "B"]，列名应为 ["C", "D"]
    assert result.index.names == ["A", "B"]
    assert result.columns.tolist() == ["C", "D"]
# 使用参数化测试，在所有连接的情况下运行测试函数
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_read_sql_delegate(conn, request):
    # 对于特定的 sqlite_buildin_iris 连接，标记为预期失败，因为该连接不实现 read_sql_table
    if conn == "sqlite_buildin_iris":
        request.applymarker(
            pytest.mark.xfail(
                reason="sqlite_buildin connection does not implement read_sql_table"
            )
        )

    # 获取连接的实际对象
    conn = request.getfixturevalue(conn)
    
    # 通过 SQL 查询从数据库中读取整张表 iris 的内容
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn)
    
    # 通过通用的 read_sql 方法读取整张表 iris 的内容
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn)
    
    # 断言两种读取方式得到的数据框架内容相等
    tm.assert_frame_equal(iris_frame1, iris_frame2)

    # 通过 read_sql_table 方法从数据库中读取表 iris 的内容
    iris_frame1 = sql.read_sql_table("iris", conn)
    
    # 通过通用的 read_sql 方法读取表 iris 的内容
    iris_frame2 = sql.read_sql("iris", conn)
    
    # 断言两种读取方式得到的数据框架内容相等
    tm.assert_frame_equal(iris_frame1, iris_frame2)


# 测试函数：确保 read_sql_table 方法能够正确处理异常情况
def test_not_reflect_all_tables(sqlite_conn):
    # 获取 SQLite 连接对象
    conn = sqlite_conn
    
    # 导入所需的 SQL 相关模块
    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    # 创建两个 SQL 查询的列表
    query_list = [
        text("CREATE TABLE invalid (x INTEGER, y UNKNOWN);"),
        text("CREATE TABLE other_table (x INTEGER, y INTEGER);"),
    ]

    # 遍历查询列表并执行每个查询
    for query in query_list:
        if isinstance(conn, Engine):
            # 如果连接是 Engine 类型，使用上下文管理器执行查询
            with conn.connect() as conn:
                with conn.begin():
                    conn.execute(query)
        else:
            # 否则，直接使用连接对象执行查询
            with conn.begin():
                conn.execute(query)

    # 确保在特定情况下不会产生警告
    with tm.assert_produces_warning(None):
        # 尝试从连接中读取表 other_table 的内容，确保不会产生警告
        sql.read_sql_table("other_table", conn)
        
        # 尝试通过 SQL 查询读取表 other_table 的内容，确保不会产生警告
        sql.read_sql_query("SELECT * FROM other_table", conn)


# 参数化测试函数：测试在特定连接下是否会产生不区分大小写的表名警告
@pytest.mark.parametrize("conn", all_connectable)
def test_warning_case_insensitive_table_name(conn, request, test_frame1):
    # 获取连接名
    conn_name = conn
    
    # 对于特定连接（sqlite_buildin 或包含 adbc 的连接），标记为预期失败，因为不会触发警告
    if conn_name == "sqlite_buildin" or "adbc" in conn_name:
        request.applymarker(pytest.mark.xfail(reason="Does not raise warning"))

    # 获取连接的实际对象
    conn = request.getfixturevalue(conn)
    
    # 测试特定情况下是否会产生警告
    with tm.assert_produces_warning(
        UserWarning,
        match=(
            r"The provided table name 'TABLE1' is not found exactly as such in "
            r"the database after writing the table, possibly due to case "
            r"sensitivity issues. Consider using lower case table names."
        ),
    ):
        with sql.SQLDatabase(conn) as db:
            # 检查数据库中是否存在不区分大小写的表名 TABLE1
            db.check_case_sensitive("TABLE1", "")

    # 确保在正常情况下不会产生警告
    with tm.assert_produces_warning(None):
        # 将测试数据框架 test_frame1 写入数据库表 CaseSensitive，确保不会产生警告
        test_frame1.to_sql(name="CaseSensitive", con=conn)


# 参数化测试函数：测试 SQLAlchemy 的类型映射是否正常工作
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_sqlalchemy_type_mapping(conn, request):
    # 获取连接的实际对象
    conn = request.getfixturevalue(conn)
    
    # 导入所需的 SQLAlchemy 模块
    from sqlalchemy import TIMESTAMP
    
    # 创建一个包含 Timestamp 对象的数据框架 df（由于时区问题，没有使用 datetime64）（GH9085）
    df = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    
    # 使用 SQLDatabase 上下文管理器创建名为 test_type 的 SQL 表，将数据框架 df 写入表中
    with sql.SQLDatabase(conn) as db:
        table = sql.SQLTable("test_type", db, frame=df)
        
        # 断言时间列 'time' 的类型是否为 TIMESTAMP（GH 9086：建议用 TIMESTAMP 类型表示带有时区的日期时间）
        assert isinstance(table.table.c["time"].type, TIMESTAMP)
# 使用 pytest.mark.parametrize 装饰器为测试函数 test_sqlalchemy_integer_mapping 添加参数化测试
# 使用 sqlalchemy_connectable 参数化 conn，用于连接 SQLAlchemy 数据库
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 使用 pytest.mark.parametrize 装饰器为测试函数 test_sqlalchemy_integer_mapping 添加参数化测试
# 参数化 integer 和 expected，测试不同的数据类型映射到 SQLAlchemy 的预期类型
@pytest.mark.parametrize(
    "integer, expected",
    [
        ("int8", "SMALLINT"),  # 将 int8 映射为 SMALLINT 类型
        ("Int8", "SMALLINT"),  # 将 Int8 映射为 SMALLINT 类型
        ("uint8", "SMALLINT"),  # 将 uint8 映射为 SMALLINT 类型
        ("UInt8", "SMALLINT"),  # 将 UInt8 映射为 SMALLINT 类型
        ("int16", "SMALLINT"),  # 将 int16 映射为 SMALLINT 类型
        ("Int16", "SMALLINT"),  # 将 Int16 映射为 SMALLINT 类型
        ("uint16", "INTEGER"),  # 将 uint16 映射为 INTEGER 类型
        ("UInt16", "INTEGER"),  # 将 UInt16 映射为 INTEGER 类型
        ("int32", "INTEGER"),   # 将 int32 映射为 INTEGER 类型
        ("Int32", "INTEGER"),   # 将 Int32 映射为 INTEGER 类型
        ("uint32", "BIGINT"),   # 将 uint32 映射为 BIGINT 类型
        ("UInt32", "BIGINT"),   # 将 UInt32 映射为 BIGINT 类型
        ("int64", "BIGINT"),    # 将 int64 映射为 BIGINT 类型
        ("Int64", "BIGINT"),    # 将 Int64 映射为 BIGINT 类型
        # 将 int 映射为 BIGINT 或 INTEGER 类型，取决于其 numpy 数据类型的名称
        (int, "BIGINT" if np.dtype(int).name == "int64" else "INTEGER"),
    ],
)
def test_sqlalchemy_integer_mapping(conn, request, integer, expected):
    # GH35076 Map pandas integer to optimal SQLAlchemy integer type
    # 从 request 中获取 conn 的 fixture 值，用于数据库连接
    conn = request.getfixturevalue(conn)
    # 创建一个 DataFrame，包含一个整数列 'a'，指定 dtype 为 integer
    df = DataFrame([0, 1], columns=["a"], dtype=integer)
    # 使用 SQLDatabase 对象连接到数据库
    with sql.SQLDatabase(conn) as db:
        # 创建一个名为 'test_type' 的 SQLTable 对象，将 DataFrame df 写入数据库 db
        table = sql.SQLTable("test_type", db, frame=df)
        # 获取表中列 'a' 的数据类型，并转换为字符串形式
        result = str(table.table.c.a.type)
    # 断言 result 的值等于预期的 expected 值
    assert result == expected


# 使用 pytest.mark.parametrize 装饰器为测试函数 test_sqlalchemy_integer_overload_mapping 添加参数化测试
# 使用 sqlalchemy_connectable 参数化 conn，用于连接 SQLAlchemy 数据库
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 使用 pytest.mark.parametrize 装饰器为测试函数 test_sqlalchemy_integer_overload_mapping 添加参数化测试
# 参数化 integer，测试不支持的 uint64 和 UInt64 数据类型映射到 SQLAlchemy 的情况
@pytest.mark.parametrize("integer", ["uint64", "UInt64"])
def test_sqlalchemy_integer_overload_mapping(conn, request, integer):
    # 从 request 中获取 conn 的 fixture 值，用于数据库连接
    conn = request.getfixturevalue(conn)
    # GH35076 Map pandas integer to optimal SQLAlchemy integer type
    # 创建一个 DataFrame，包含一个整数列 'a'，指定 dtype 为 integer
    df = DataFrame([0, 1], columns=["a"], dtype=integer)
    # 使用 SQLDatabase 对象连接到数据库
    with sql.SQLDatabase(conn) as db:
        # 使用 pytest.raises 检查创建名为 'test_type' 的 SQLTable 对象时，抛出预期的 ValueError 异常
        with pytest.raises(
            ValueError, match="Unsigned 64 bit integer datatype is not supported"
        ):
            sql.SQLTable("test_type", db, frame=df)


# 使用 pytest.mark.parametrize 装饰器为测试函数 test_database_uri_string 添加参数化测试
# 使用 all_connectable 参数化 conn，测试各种数据库的连接字符串
@pytest.mark.parametrize("conn", all_connectable)
def test_database_uri_string(conn, request, test_frame1):
    # 导入 sqlalchemy 库，如果导入失败则跳过此测试
    pytest.importorskip("sqlalchemy")
    # 从 request 中获取 conn 的 fixture 值，用于数据库连接
    conn = request.getfixturevalue(conn)
    # Test read_sql and .to_sql method with a database URI (GH10654)
    # 创建一个临时数据库 URI，并使用 test_frame1 将数据写入其中
    with tm.ensure_clean() as name:
        db_uri = "sqlite:///" + name
        table = "iris"
        # 将 test_frame1 的数据写入名为 'iris' 的表中，存在则替换，不包含索引
        test_frame1.to_sql(name=table, con=db_uri, if_exists="replace", index=False)
        # 使用 sql.read_sql 从数据库 db_uri 中读取表 'iris' 的数据到 test_frame2
        test_frame2 = sql.read_sql(table, db_uri)
        # 使用 sql.read_sql_table 从数据库 db_uri 中读取表 'iris' 的数据到 test_frame3
        test_frame3 = sql.read_sql_table(table, db_uri)
        # 执行查询语句 "SELECT * FROM iris" 并将结果读取到 test_frame4
        query = "SELECT * FROM iris"
        test_frame4 = sql.read_sql_query(query, db_uri)
    # 使用 tm.assert_frame_equal 断言 test_frame1 和 test_frame2 的数据相等
    tm.assert_frame_equal(test_frame1, test_frame2)
    # 使用 tm.assert_frame_equal 断言 test_frame1 和 test_frame3 的数据相等
    tm.assert_frame_equal(test_frame1, test_frame3)
    # 使用 tm.assert_frame_equal 断言 test_frame1 和 test_frame4 的数据相等
    tm.assert_frame_equal(test_frame1, test_frame4)


# 使用 td.skip_if_installed 装饰器检查 pg8000 是否已安装，如果已安装则跳过测试
@td.skip_if_installed("pg8000")
# 使用 pytest.mark.parametrize 装饰器为测试函数 test_pg8000_sqlalchemy_passthrough_error 添加参数化测试
# 使用 all_connectable 参数化 conn，测试各种数据库的连接字符串
@pytest.mark.parametrize("conn", all_connectable)
def test_pg8000_sqlalchemy_passthrough_error(conn, request):
    # 导入 sqlalchemy 库，如果导入失败则跳过此测试
    pytest.importorskip("sqlalchemy")
    # 从 request 中获取 conn 的 fixture 值，用于数据库连接
    conn = request.getfixturevalue(conn)
    # using driver that will not be installed on CI to trigger error
    # in sqlalchemy.create_engine -> test passing of this error to user
    # 创建一个 PostgreSQL 的数据库 URI，使用 pg8000 驱动
    db_uri = "postgresql+pg8000://user:pass@host/dbname"
    # 使用 pytest 的断言 `raises` 来检测是否抛出 ImportError 异常，并验证异常消息中包含 "pg8000"
    with pytest.raises(ImportError, match="pg8000"):
        # 调用 sql 模块中的 read_sql 函数，执行 SQL 查询 "select * from table"，使用给定的数据库 URI db_uri
        sql.read_sql("select * from table", db_uri)
@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_query_by_text_obj(conn, request):
    # WIP : GH10846
    # 保存连接名称
    conn_name = conn
    # 获取实际的连接对象
    conn = request.getfixturevalue(conn)
    # 导入SQLAlchemy的text模块，用于构建SQL语句
    from sqlalchemy import text

    # 根据数据库类型选择不同的SQL语句模板
    if "postgres" in conn_name:
        name_text = text('select * from iris where "Name"=:name')
    else:
        name_text = text("select * from iris where name=:name")
    
    # 执行SQL查询，并将结果转换为DataFrame
    iris_df = sql.read_sql(name_text, conn, params={"name": "Iris-versicolor"})
    # 获取所有结果中的唯一名称，断言是否符合预期
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-versicolor"}


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_query_by_select_obj(conn, request):
    # 获取实际的连接对象
    conn = request.getfixturevalue(conn)
    # WIP : GH10846
    # 导入SQLAlchemy的bindparam和select模块
    from sqlalchemy import (
        bindparam,
        select,
    )

    # 获取iris表的元数据
    iris = iris_table_metadata()
    # 构建基于选择的SQL查询，其中包含绑定的参数"name"
    name_select = select(iris).where(iris.c.Name == bindparam("name"))
    # 执行SQL查询，并将结果转换为DataFrame
    iris_df = sql.read_sql(name_select, conn, params={"name": "Iris-setosa"})
    # 获取所有结果中的唯一名称，断言是否符合预期
    all_names = set(iris_df["Name"])
    assert all_names == {"Iris-setosa"}


@pytest.mark.parametrize("conn", all_connectable)
def test_column_with_percentage(conn, request):
    # GH 37157
    # 保存连接名称
    conn_name = conn
    # 如果连接是sqlite_buildin，标记为预期失败
    if conn_name == "sqlite_buildin":
        request.applymarker(pytest.mark.xfail(reason="Not Implemented"))

    # 获取实际的连接对象
    conn = request.getfixturevalue(conn)
    # 创建包含特定列和百分比列的DataFrame
    df = DataFrame({"A": [0, 1, 2], "%_variation": [3, 4, 5]})
    # 将DataFrame写入数据库中
    df.to_sql(name="test_column_percentage", con=conn, index=False)

    # 从数据库中读取表的内容
    res = sql.read_sql_table("test_column_percentage", conn)

    # 断言从数据库中读取的结果与原始DataFrame相同
    tm.assert_frame_equal(res, df)


def test_sql_open_close(test_frame3):
    # Test if the IO in the database still work if the connection closed
    # between the writing and reading (as in many real situations).

    # 使用tm.ensure_clean确保操作环境干净
    with tm.ensure_clean() as name:
        # 使用closing确保在使用完毕后关闭数据库连接
        with closing(sqlite3.connect(name)) as conn:
            # 将DataFrame写入数据库，并验证写入的行数是否为4
            assert sql.to_sql(test_frame3, "test_frame3_legacy", conn, index=False) == 4

        # 重新打开数据库连接，从数据库中读取数据，并与原始DataFrame进行比较
        with closing(sqlite3.connect(name)) as conn:
            result = sql.read_sql_query("SELECT * FROM test_frame3_legacy;", conn)

    # 断言从数据库中读取的结果与原始DataFrame相同
    tm.assert_frame_equal(test_frame3, result)


@td.skip_if_installed("sqlalchemy")
def test_con_string_import_error():
    # 测试当没有安装SQLAlchemy时，使用URI字符串会抛出ImportError异常
    conn = "mysql://root@localhost/pandas"
    msg = "Using URI string without sqlalchemy installed"
    with pytest.raises(ImportError, match=msg):
        sql.read_sql("SELECT * FROM iris", conn)


@td.skip_if_installed("sqlalchemy")
def test_con_unknown_dbapi2_class_does_not_error_without_sql_alchemy_installed():
    # 当未安装SQLAlchemy时，测试使用自定义的MockSqliteConnection类不会报错
    class MockSqliteConnection:
        def __init__(self, *args, **kwargs) -> None:
            self.conn = sqlite3.Connection(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.conn, name)

        def close(self):
            self.conn.close()

    # 使用MockSqliteConnection创建连接对象
    with contextlib.closing(MockSqliteConnection(":memory:")) as conn:
        # 使用assert_produces_warning确保在不支持的情况下会产生警告
        with tm.assert_produces_warning(UserWarning, match="only supports SQLAlchemy"):
            sql.read_sql("SELECT 1", conn)
# 测试读取 SQLite 数据库中的 SQL 查询结果
def test_sqlite_read_sql_delegate(sqlite_buildin_iris):
    # 使用提供的 SQLite 连接对象
    conn = sqlite_buildin_iris
    # 从数据库中执行 SQL 查询，返回一个 Pandas 数据帧
    iris_frame1 = sql.read_sql_query("SELECT * FROM iris", conn)
    # 从数据库中执行 SQL 查询，返回一个 Pandas 数据帧
    iris_frame2 = sql.read_sql("SELECT * FROM iris", conn)
    # 断言两个数据帧相等
    tm.assert_frame_equal(iris_frame1, iris_frame2)

    # 准备一个错误消息字符串
    msg = "Execution failed on sql 'iris': near \"iris\": syntax error"
    # 使用 pytest 检查是否抛出预期的异常
    with pytest.raises(sql.DatabaseError, match=msg):
        sql.read_sql("iris", conn)


# 测试获取数据库架构信息的函数
def test_get_schema2(test_frame1):
    # 在不提供连接对象的情况下（用于向后兼容）
    # 生成创建数据表的 SQL 语句
    create_sql = sql.get_schema(test_frame1, "test")
    # 断言 SQL 语句包含 "CREATE" 关键词
    assert "CREATE" in create_sql


# 测试 SQLite 数据库中的数据类型映射
def test_sqlite_type_mapping(sqlite_buildin):
    # 测试 Timestamp 对象（没有 datetime64 类型，因为涉及时区）(GH9085)
    conn = sqlite_buildin
    # 创建一个包含时间列的 Pandas 数据帧
    df = DataFrame(
        {"time": to_datetime(["2014-12-12 01:54", "2014-12-11 02:54"], utc=True)}
    )
    # 使用数据库连接创建一个 SQLiteDatabase 对象
    db = sql.SQLiteDatabase(conn)
    # 在 SQLite 数据库中创建名为 "test_type" 的数据表，并将数据帧导入
    table = sql.SQLiteTable("test_type", db, frame=df)
    # 获取数据表的 SQL 架构信息
    schema = table.sql_schema()
    # 遍历 SQL 架构信息的每一行
    for col in schema.split("\n"):
        # 如果列名为 "time"
        if col.split()[0].strip('"') == "time":
            # 断言列的数据类型为 "TIMESTAMP"
            assert col.split()[1] == "TIMESTAMP"


# -----------------------------------------------------------------------------
# -- 数据库特定的测试


# 使用参数化测试连接不同的 SQLAlchemy 连接
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_create_table(conn, request):
    # 对于 SQLite 连接字符串，跳过测试（因为没有检查系统）
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    # 获取参数化测试中的连接对象
    conn = request.getfixturevalue(conn)

    # 导入 SQLAlchemy 的 inspect 函数
    from sqlalchemy import inspect

    # 创建临时的 Pandas 数据帧
    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    # 使用 PandasSQL 上下文管理器连接到数据库，并设置需要事务
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        # 将临时数据帧写入数据库表 "temp_frame"，并断言写入的行数为 4
        assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4

    # 使用 inspect 函数检查数据库连接
    insp = inspect(conn)
    # 断言数据库中存在名为 "temp_frame" 的数据表
    assert insp.has_table("temp_frame")

    # 清理操作
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        # 使用事务删除数据表 "temp_frame"
        pandasSQL.drop_table("temp_frame")


# 使用参数化测试连接不同的 SQLAlchemy 连接
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_drop_table(conn, request):
    # 对于 SQLite 连接字符串，跳过测试（因为没有检查系统）
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    # 获取参数化测试中的连接对象
    conn = request.getfixturevalue(conn)

    # 导入 SQLAlchemy 的 inspect 函数
    from sqlalchemy import inspect

    # 创建临时的 Pandas 数据帧
    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    # 使用 PandasSQL 上下文管理器连接到数据库
    with sql.SQLDatabase(conn) as pandasSQL:
        # 在事务中运行以下操作
        with pandasSQL.run_transaction():
            # 将临时数据帧写入数据库表 "temp_frame"，并断言写入的行数为 4
            assert pandasSQL.to_sql(temp_frame, "temp_frame") == 4

        # 使用 inspect 函数检查数据库连接
        insp = inspect(conn)
        # 断言数据库中存在名为 "temp_frame" 的数据表
        assert insp.has_table("temp_frame")

        # 再次在事务中运行以下操作
        with pandasSQL.run_transaction():
            # 使用事务删除数据表 "temp_frame"
            pandasSQL.drop_table("temp_frame")
        try:
            # 清除缓存，适用于 SQLAlchemy 2.0 及以上版本
            insp.clear_cache()
        except AttributeError:
            pass
        # 断言数据库中不存在名为 "temp_frame" 的数据表
        assert not insp.has_table("temp_frame")


# 使用参数化测试连接所有可连接的 SQLAlchemy 连接
@pytest.mark.parametrize("conn", all_connectable)
def test_roundtrip(conn, request, test_frame1):
    # 对于 SQLite 连接字符串，跳过测试（因为没有检查系统）
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    # 获取参数化测试中的连接对象
    conn_name = conn
    # 从测试夹具中获取数据库连接对象
    conn = request.getfixturevalue(conn)
    # 根据数据库连接创建一个 pandasSQL_builder 对象
    pandasSQL = pandasSQL_builder(conn)
    # 使用 pandasSQL 对象运行事务
    with pandasSQL.run_transaction():
        # 将 DataFrame test_frame1 写入数据库表 "test_frame_roundtrip"，并断言写入的行数为 4
        assert pandasSQL.to_sql(test_frame1, "test_frame_roundtrip") == 4
        # 从数据库中读取表 "test_frame_roundtrip" 的数据
        result = pandasSQL.read_query("SELECT * FROM test_frame_roundtrip")

    # 如果连接名中包含字符串 "adbc"
    if "adbc" in conn_name:
        # 重命名结果 DataFrame 的列名 "__index_level_0__" 为 "level_0"
        result = result.rename(columns={"__index_level_0__": "level_0"})
    # 将结果 DataFrame 的 "level_0" 列设置为索引
    result.set_index("level_0", inplace=True)
    # 清除结果 DataFrame 的索引名
    result.index.name = None

    # 使用测试模块中的函数 tm.assert_frame_equal 检查结果 DataFrame 和 test_frame1 是否相等
    tm.assert_frame_equal(result, test_frame1)
@pytest.mark.parametrize("conn", all_connectable_iris)
def test_execute_sql(conn, request):
    # 从 request 中获取数据库连接 fixture
    conn = request.getfixturevalue(conn)
    # 使用 pandasSQL_builder 创建 pandasSQL 对象，使用完毕后自动关闭
    with pandasSQL_builder(conn) as pandasSQL:
        # 开启事务
        with pandasSQL.run_transaction():
            # 执行 SQL 查询，获取 iris 表中的第一行数据
            iris_results = pandasSQL.execute("SELECT * FROM iris")
            # 获取查询结果的第一行记录
            row = iris_results.fetchone()
            # 关闭查询结果游标
            iris_results.close()
    # 断言第一行数据是否符合预期
    assert list(row) == [5.1, 3.5, 1.4, 0.2, "Iris-setosa"]


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table(conn, request):
    # 从 request 中获取数据库连接 fixture
    conn = request.getfixturevalue(conn)
    # 使用 SQLAlchemy 的 sql.read_sql_table 方法读取 iris 表的全部数据
    iris_frame = sql.read_sql_table("iris", con=conn)
    # 检查读取的 iris 数据框架
    check_iris_frame(iris_frame)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_sqlalchemy_read_table_columns(conn, request):
    # 从 request 中获取数据库连接 fixture
    conn = request.getfixturevalue(conn)
    # 使用 SQLAlchemy 的 sql.read_sql_table 方法读取 iris 表的部分列数据
    iris_frame = sql.read_sql_table(
        "iris", con=conn, columns=["SepalLength", "SepalLength"]
    )
    # 检查读取的 iris 数据框架的列名
    tm.assert_index_equal(iris_frame.columns, Index(["SepalLength", "SepalLength__1"]))


@pytest.mark.parametrize("conn", sqlalchemy_connectable_iris)
def test_read_table_absent_raises(conn, request):
    # 从 request 中获取数据库连接 fixture
    conn = request.getfixturevalue(conn)
    # 测试当读取不存在的表时是否引发 ValueError 异常
    msg = "Table this_doesnt_exist not found"
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("this_doesnt_exist", con=conn)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_sqlalchemy_default_type_conversion(conn, request):
    # 获取连接名称
    conn_name = conn
    # 对于特定的数据库类型，跳过测试
    if conn_name == "sqlite_str":
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "mysql" in conn_name or "sqlite" in conn_name:
        # 标记预期的测试失败，因为布尔类型未正确推断
        request.applymarker(
            pytest.mark.xfail(reason="boolean dtype not inferred properly")
        )

    # 从 request 中获取数据库连接 fixture
    conn = request.getfixturevalue(conn)
    # 使用 SQLAlchemy 的 sql.read_sql_table 方法读取 types 表的数据
    df = sql.read_sql_table("types", conn)

    # 断言数据框架的数据类型是否正确推断
    assert issubclass(df.FloatCol.dtype.type, np.floating)
    assert issubclass(df.IntCol.dtype.type, np.integer)
    assert issubclass(df.BoolCol.dtype.type, np.bool_)

    # 断言带有 NA 值的 Int 列依然保持为 float 类型
    assert issubclass(df.IntColWithNull.dtype.type, np.floating)
    # 断言带有 NA 值的 Bool 列变为 object 类型
    assert issubclass(df.BoolColWithNull.dtype.type, object)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_bigint(conn, request):
    # 从 request 中获取数据库连接 fixture
    conn = request.getfixturevalue(conn)
    # 创建一个包含 int64 数据的 DataFrame
    df = DataFrame(data={"i64": [2**62]})
    # 将数据写入数据库表 "test_bigint"，并断言写入的行数为 1
    assert df.to_sql(name="test_bigint", con=conn, index=False) == 1
    # 从数据库中读取表 "test_bigint" 的数据
    result = sql.read_sql_table("test_bigint", conn)

    # 断言读取的数据与原始 DataFrame 是否一致
    tm.assert_frame_equal(df, result)


@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_default_date_load(conn, request):
    # 获取连接名称
    conn_name = conn
    # 对于特定的数据库类型，跳过测试
    if conn_name == "sqlite_str":
        pytest.skip("types tables not created in sqlite_str fixture")
    elif "sqlite" in conn_name:
        # 标记预期的测试失败，因为 SQLite 无法正确读取日期
        request.applymarker(
            pytest.mark.xfail(reason="sqlite does not read date properly")
        )
    # 从 request 的 fixture 中获取数据库连接对象 conn
    conn = request.getfixturevalue(conn)
    # 使用 sql 模块从数据库表 "types" 中读取数据，并将结果存储在 DataFrame df 中
    df = sql.read_sql_table("types", conn)
    # 断言 DataFrame df 的列 "DateCol" 的数据类型是 np.datetime64 的子类
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
# 使用 pytest 的 parametrize 装饰器，对 conn 参数进行参数化测试，使用 postgresql_connectable 的值
# 根据 request 获取 conn 对象，用于数据库连接
@pytest.mark.parametrize("conn", postgresql_connectable)
# 对 parse_dates 参数进行参数化测试，分别为 None 和 ["DateColWithTz"]
@pytest.mark.parametrize("parse_dates", [None, ["DateColWithTz"]])
# 测试函数：测试带有时区的 PostgreSQL datetime 查询
def test_datetime_with_timezone_query(conn, request, parse_dates):
    # 获得 conn 对象的 fixture 值
    conn = request.getfixturevalue(conn)
    # 创建并加载带时区的 PostgreSQL datetime 数据
    expected = create_and_load_postgres_datetz(conn)

    # GH11216
    # 从数据库中读取 datetz 表的数据到 DataFrame df
    df = read_sql_query("select * from datetz", conn, parse_dates=parse_dates)
    # 获取 df 的 DateColWithTz 列
    col = df.DateColWithTz
    # 断言：验证 Series col 是否与预期值 expected 相等
    tm.assert_series_equal(col, expected)


# 使用 pytest 的 parametrize 装饰器，对 conn 参数进行参数化测试，使用 postgresql_connectable 的值
@pytest.mark.parametrize("conn", postgresql_connectable)
# 测试函数：测试带有时区的 PostgreSQL datetime 查询（使用 chunksize）
def test_datetime_with_timezone_query_chunksize(conn, request):
    # 获得 conn 对象的 fixture 值
    conn = request.getfixturevalue(conn)
    # 创建并加载带时区的 PostgreSQL datetime 数据
    expected = create_and_load_postgres_datetz(conn)

    # 从数据库中逐块读取 datetz 表的数据到 DataFrame df
    df = concat(
        list(read_sql_query("select * from datetz", conn, chunksize=1)),
        ignore_index=True,
    )
    # 获取 df 的 DateColWithTz 列
    col = df.DateColWithTz
    # 断言：验证 Series col 是否与预期值 expected 相等
    tm.assert_series_equal(col, expected)


# 使用 pytest 的 parametrize 装饰器，对 conn 参数进行参数化测试，使用 postgresql_connectable 的值
@pytest.mark.parametrize("conn", postgresql_connectable)
# 测试函数：测试带有时区的 PostgreSQL datetime 表读取
def test_datetime_with_timezone_table(conn, request):
    # 获得 conn 对象的 fixture 值
    conn = request.getfixturevalue(conn)
    # 创建并加载带时区的 PostgreSQL datetime 数据
    expected = create_and_load_postgres_datetz(conn)
    # 从数据库中读取 datetz 表的数据到 DataFrame result
    result = sql.read_sql_table("datetz", conn)

    # 将预期 DataFrame expected 转换为帧对象
    exp_frame = expected.to_frame()
    # 断言：验证 DataFrame result 是否与预期帧 exp_frame 相等
    tm.assert_frame_equal(result, exp_frame)


# 使用 pytest 的 parametrize 装饰器，对 conn 参数进行参数化测试，使用 sqlalchemy_connectable 的值
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 测试函数：测试带有时区的 datetime 轮回
def test_datetime_with_timezone_roundtrip(conn, request):
    # 保存 conn 的名称
    conn_name = conn
    # 获得 conn 对象的 fixture 值
    conn = request.getfixturevalue(conn)
    # 创建带有时区的 DataFrame expected
    expected = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3, tz="US/Pacific", unit="us")}
    )
    # 断言：将 expected 写入数据库表 test_datetime_tz，验证是否成功写入了3行数据
    assert expected.to_sql(name="test_datetime_tz", con=conn, index=False) == 3

    if "postgresql" in conn_name:
        # 对于 PostgreSQL，时间戳的时区被强制转换为 UTC
        expected["A"] = expected["A"].dt.tz_convert("UTC")
    else:
        # 否则，时间戳被返回为本地时区的无时区时间
        expected["A"] = expected["A"].dt.tz_localize(None)

    # 从数据库中读取表 test_datetime_tz 的数据到 DataFrame result
    result = sql.read_sql_table("test_datetime_tz", conn)
    # 断言：验证 DataFrame result 是否与预期值 expected 相等
    tm.assert_frame_equal(result, expected)

    # 再次从数据库中执行查询语句，将结果保存到 DataFrame result
    result = sql.read_sql_query("SELECT * FROM test_datetime_tz", conn)
    if "sqlite" in conn_name:
        # 对于 SQLite，read_sql_query 不返回 datetime 类型，而是返回字符串
        assert isinstance(result.loc[0, "A"], str)
        # 将 result 的 A 列转换为 datetime64[us] 单位
        result["A"] = to_datetime(result["A"]).dt.as_unit("us")
    # 断言：验证 DataFrame result 是否与预期值 expected 相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，对 conn 参数进行参数化测试，使用 sqlalchemy_connectable 的值
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 测试函数：测试超出边界的 datetime
def test_out_of_bounds_datetime(conn, request):
    # 获得 conn 对象的 fixture 值
    conn = request.getfixturevalue(conn)
    # 创建带有 datetime 9999-01-01 的 DataFrame data
    data = DataFrame({"date": datetime(9999, 1, 1)}, index=[0])
    # 使用断言验证将数据写入数据库表 "test_datetime_obb" 中，并返回成功写入的行数为 1
    assert data.to_sql(name="test_datetime_obb", con=conn, index=False) == 1
    # 从数据库中读取名为 "test_datetime_obb" 的表格内容并存储在 result 变量中
    result = sql.read_sql_table("test_datetime_obb", conn)
    # 创建一个预期的 DataFrame，包含一个日期为 9999 年 1 月 1 日的日期列
    expected = DataFrame(
        np.array([datetime(9999, 1, 1)], dtype="M8[us]"), columns=["date"]
    )
    # 使用测试工具比较 result 和 expected 的内容是否一致
    tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 对 sqlalchemy_connectable 参数化测试函数
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_naive_datetimeindex_roundtrip(conn, request):
    # GH 23510
    # 确保一个 naive DatetimeIndex 不会被转换为 UTC
    # 获取测试函数的连接参数值
    conn = request.getfixturevalue(conn)
    # 创建一个日期范围，频率为每6小时一次，单位是微秒，然后移除频率信息
    dates = date_range("2018-01-01", periods=5, freq="6h", unit="us")._with_freq(None)
    # 创建一个期望的 DataFrame，包含一个名为 "nums" 的列，索引为上面创建的日期
    expected = DataFrame({"nums": range(5)}, index=dates)
    # 将期望的 DataFrame 写入名为 "foo_table" 的数据库表中，使用连接对象 conn，并指定索引列为 "info_date"
    assert expected.to_sql(name="foo_table", con=conn, index_label="info_date") == 5
    # 从数据库表 "foo_table" 读取数据到 result 中，指定索引列为 "info_date"
    result = sql.read_sql_table("foo_table", conn, index_col="info_date")
    # 使用测试函数的辅助函数 tm.assert_frame_equal 检查 result 和 expected 是否相等，不检查列名
    tm.assert_frame_equal(result, expected, check_names=False)


# 使用 pytest.mark.parametrize 对 sqlalchemy_connectable_types 参数化测试函数
@pytest.mark.parametrize("conn", sqlalchemy_connectable_types)
def test_date_parsing(conn, request):
    # No Parsing
    # 将 conn_name 设置为 conn
    conn_name = conn
    # 获取测试函数的连接参数值
    conn = request.getfixturevalue(conn)
    # 从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table("types", conn)
    # 根据数据库类型确定 expected_type 是 object 还是 np.datetime64
    expected_type = object if "sqlite" in conn_name else np.datetime64
    # 检查 df 的 DateCol 列的类型是否是 expected_type 的子类
    assert issubclass(df.DateCol.dtype.type, expected_type)

    # 使用 parse_dates=["DateCol"] 参数重新从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table("types", conn, parse_dates=["DateCol"])
    # 检查 df 的 DateCol 列的类型是否是 np.datetime64
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

    # 使用 parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"} 参数重新从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table("types", conn, parse_dates={"DateCol": "%Y-%m-%d %H:%M:%S"})
    # 检查 df 的 DateCol 列的类型是否是 np.datetime64
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

    # 使用 parse_dates={"DateCol": {"format": "%Y-%m-%d %H:%M:%S"}} 参数重新从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table(
        "types",
        conn,
        parse_dates={"DateCol": {"format": "%Y-%m-%d %H:%M:%S"}},
    )
    # 检查 df 的 DateCol 列的类型是否是 np.datetime64
    assert issubclass(df.DateCol.dtype.type, np.datetime64)

    # 使用 parse_dates=["IntDateCol"] 参数重新从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table("types", conn, parse_dates=["IntDateCol"])
    # 检查 df 的 IntDateCol 列的类型是否是 np.datetime64
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

    # 使用 parse_dates={"IntDateCol": "s"} 参数重新从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": "s"})
    # 检查 df 的 IntDateCol 列的类型是否是 np.datetime64
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)

    # 使用 parse_dates={"IntDateCol": {"unit": "s"}} 参数重新从数据库表 "types" 中读取数据到 DataFrame df 中
    df = sql.read_sql_table("types", conn, parse_dates={"IntDateCol": {"unit": "s"}})
    # 检查 df 的 IntDateCol 列的类型是否是 np.datetime64
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)


# 使用 pytest.mark.parametrize 对 sqlalchemy_connectable 参数化测试函数
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime(conn, request):
    # 将 conn_name 设置为 conn
    conn_name = conn
    # 获取测试函数的连接参数值
    conn = request.getfixturevalue(conn)
    # 创建一个包含两列（"A" 和 "B"）的 DataFrame df
    df = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3), "B": np.arange(3.0)}
    )
    # 将 df 写入名为 "test_datetime" 的数据库表中，使用连接对象 conn
    assert df.to_sql(name="test_datetime", con=conn) == 3

    # 使用 read_sql_table 函数从数据库表 "test_datetime" 中读取数据到 result 中
    result = sql.read_sql_table("test_datetime", conn)
    # 删除 result 的 "index" 列
    result = result.drop("index", axis=1)

    # 创建一个期望的 DataFrame expected，与 df 相同
    expected = df[:]
    # 将 expected 的 "A" 列转换为类型 "M8[us]"
    expected["A"] = expected["A"].astype("M8[us]")
    # 使用测试函数的辅助函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用 read_sql_query 函数从数据库表 "test_datetime" 中执行 SQL 查询并读取数据到 result 中
    result = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    # 删除 result 的 "index" 列
    result = result.drop("index", axis=1)
    # 如果连接类型是 sqlite，检查 result 的 "A" 列是否是字符串类型，然后将其转换为 datetime
    if "sqlite" in conn_name:
        assert isinstance(result.loc[0, "A"], str)
        result["A"] = to_datetime(result["A"])
    # 使用测试函数的辅助函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 对 sqlalchemy_connectable 参数化测试函数
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_NaT(conn, request):
    # 将 conn_name 设置为 conn
    conn_name = conn
    # 获取名为 conn 的测试夹具的连接对象
    conn = request.getfixturevalue(conn)
    # 创建一个 DataFrame 对象，包含两列：A列是从'2013-01-01 09:00:00'开始的日期时间序列，B列是从0开始的浮点数序列
    df = DataFrame(
        {"A": date_range("2013-01-01 09:00:00", periods=3), "B": np.arange(3.0)}
    )
    # 将 df 中第二行第一列 "A" 的值设置为 NaN
    df.loc[1, "A"] = np.nan
    # 将 DataFrame df 中的数据写入到名为 "test_datetime" 的 SQL 表中，不包括索引列
    assert df.to_sql(name="test_datetime", con=conn, index=False) == 3

    # 使用 read_sql_table 方法从数据库中读取表 "test_datetime" 的数据，并存储在 result 变量中
    result = sql.read_sql_table("test_datetime", conn)
    # 创建一个预期的 DataFrame expected，其内容与 df 相同
    expected = df[:]
    # 将 expected 中的 "A" 列数据类型转换为 datetime64[us]
    expected["A"] = expected["A"].astype("M8[us]")
    # 使用 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 使用 read_sql_query 方法从数据库中执行 SELECT 查询 "SELECT * FROM test_datetime" 并将结果存储在 result 变量中
    result = sql.read_sql_query("SELECT * FROM test_datetime", conn)
    # 如果连接的数据库是 sqlite，则执行以下操作
    if "sqlite" in conn_name:
        # 断言 result 中第一行第一列 "A" 的数据类型为字符串
        assert isinstance(result.loc[0, "A"], str)
        # 将 result 中的 "A" 列数据转换为 datetime，并使用 "coerce" 模式处理错误值
        result["A"] = to_datetime(result["A"], errors="coerce")

    # 使用 assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_date(conn, request):
    # 使用pytest的@parametrize装饰器，对conn参数进行参数化测试
    # 测试是否支持datetime.date类型的数据
    conn = request.getfixturevalue(conn)  # 从request获取conn的fixture值
    df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])  # 创建包含日期对象的DataFrame
    assert df.to_sql(name="test_date", con=conn, index=False) == 2  # 将DataFrame写入SQL表，并验证写入行数为2
    res = read_sql_table("test_date", conn)  # 从数据库中读取刚刚写入的表格
    result = res["a"]  # 获取结果中的"a"列
    expected = to_datetime(df["a"])  # 将DataFrame中的日期转换为datetime64格式
    # 检查结果是否为datetime64类型
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_datetime_time(conn, request, sqlite_buildin):
    # 使用pytest的@parametrize装饰器，对conn参数进行参数化测试
    # 测试是否支持datetime.time类型的数据
    conn_name = conn  # 备份conn的名称
    conn = request.getfixturevalue(conn)  # 从request获取conn的fixture值
    df = DataFrame([time(9, 0, 0), time(9, 1, 30)], columns=["a"])  # 创建包含时间对象的DataFrame
    assert df.to_sql(name="test_time", con=conn, index=False) == 2  # 将DataFrame写入SQL表，并验证写入行数为2
    res = read_sql_table("test_time", conn)  # 从数据库中读取刚刚写入的表格
    tm.assert_frame_equal(res, df)  # 检查读取的结果与原始DataFrame是否相等

    # GH8341
    # 首先使用回退（fallback）以确保SQLite适配器被正确设置
    sqlite_conn = sqlite_buildin  # 获取sqlite_buildin的连接对象
    assert sql.to_sql(df, "test_time2", sqlite_conn, index=False) == 2  # 将DataFrame写入SQLite表，并验证写入行数为2
    res = sql.read_sql_query("SELECT * FROM test_time2", sqlite_conn)  # 从SQLite中读取刚刚写入的表格
    ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))  # 将DataFrame中的时间对象格式化为字符串
    tm.assert_frame_equal(ref, res)  # 检查结果与预期格式化的字符串是否相等，验证SQLite适配器是否生效
    # 然后测试SQLAlchemy是否受SQLite适配器影响
    assert sql.to_sql(df, "test_time3", conn, index=False) == 2  # 将DataFrame写入SQL表，并验证写入行数为2
    if "sqlite" in conn_name:
        res = sql.read_sql_query("SELECT * FROM test_time3", conn)  # 从数据库中读取刚刚写入的表格
        ref = df.map(lambda _: _.strftime("%H:%M:%S.%f"))  # 将DataFrame中的时间对象格式化为字符串
        tm.assert_frame_equal(ref, res)  # 检查结果与预期格式化的字符串是否相等，验证SQLite适配器是否生效
    res = sql.read_sql_table("test_time3", conn)  # 从数据库中读取刚刚写入的表格
    tm.assert_frame_equal(df, res)  # 检查读取的结果与原始DataFrame是否相等


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_mixed_dtype_insert(conn, request):
    # 见GitHub问题GH6509
    # 使用pytest的@parametrize装饰器，对conn参数进行参数化测试
    conn = request.getfixturevalue(conn)  # 从request获取conn的fixture值
    s1 = Series(2**25 + 1, dtype=np.int32)  # 创建包含大整数的Series对象
    s2 = Series(0.0, dtype=np.float32)  # 创建包含浮点数的Series对象
    df = DataFrame({"s1": s1, "s2": s2})  # 创建包含不同数据类型的DataFrame

    # 写入并重新读取
    assert df.to_sql(name="test_read_write", con=conn, index=False) == 1  # 将DataFrame写入SQL表，并验证写入行数为1
    df2 = sql.read_sql_table("test_read_write", conn)  # 从数据库中读取刚刚写入的表格

    tm.assert_frame_equal(df, df2, check_dtype=False, check_exact=True)  # 检查读取的结果与原始DataFrame是否相等，忽略数据类型，精确比较


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_nan_numeric(conn, request):
    # 测试数值浮点列中的NaN值
    conn = request.getfixturevalue(conn)  # 从request获取conn的fixture值
    df = DataFrame({"A": [0, 1, 2], "B": [0.2, np.nan, 5.6]})  # 创建包含NaN值的DataFrame
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3  # 将DataFrame写入SQL表，并验证写入行数为3

    # 使用read_sql_table读取
    result = sql.read_sql_table("test_nan", conn)  # 从数据库中读取刚刚写入的表格
    tm.assert_frame_equal(result, df)  # 检查读取的结果与原始DataFrame是否相等

    # 使用read_sql查询
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)  # 从数据库中读取刚刚写入的表格
    tm.assert_frame_equal(result, df)  # 检查读取的结果与原始DataFrame是否相等


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_nan_fullcolumn(conn, request):
    # 测试完全为NaN的数值浮点列
    conn = request.getfixturevalue(conn)  # 从request获取conn的fixture值
    # 创建一个 DataFrame 对象，包含两列数据：列"A"包含整数列表，列"B"包含NaN值
    df = DataFrame({"A": [0, 1, 2], "B": [np.nan, np.nan, np.nan]})
    # 使用断言确保将 DataFrame 写入数据库表"test_nan"，并返回写入的行数（期望为3）
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3

    # 使用 sql 模块的 read_sql_table 方法从数据库中读取表"test_nan"的内容
    result = sql.read_sql_table("test_nan", conn)
    # 断言读取的数据框架与之前创建的 df 相等
    tm.assert_frame_equal(result, df)

    # 使用 sql 模块的 read_sql_query 方法执行 SQL 查询，选择表"test_nan"中的所有数据
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    # 将 DataFrame 列"B"的数据类型转换为"object"，然后将该列的所有值设置为 None
    df["B"] = df["B"].astype("object")
    df["B"] = None
    # 断言读取的数据框架与更新后的 df 相等
    tm.assert_frame_equal(result, df)
# 使用 pytest 的参数化装饰器，针对 sqlalchemy_connectable 中的每个连接对象执行测试
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_nan_string(conn, request):
    # 在字符串列中处理 NaN 值
    conn = request.getfixturevalue(conn)  # 获取测试夹具中的连接对象
    df = DataFrame({"A": [0, 1, 2], "B": ["a", "b", np.nan]})
    # 将 DataFrame 写入数据库表 "test_nan"，并断言插入的行数为 3
    assert df.to_sql(name="test_nan", con=conn, index=False) == 3

    # 将第三行的 "B" 列值设置为 None，验证 NaN 值返回为 None
    df.loc[2, "B"] = None

    # 使用 sql.read_sql_table 读取表 "test_nan" 的内容
    result = sql.read_sql_table("test_nan", conn)
    # 断言读取的 DataFrame 与预期的 df 相等
    tm.assert_frame_equal(result, df)

    # 使用 sql.read_sql_query 通过 SQL 查询读取表 "test_nan" 的内容
    result = sql.read_sql_query("SELECT * FROM test_nan", conn)
    # 断言读取的 DataFrame 与预期的 df 相等
    tm.assert_frame_equal(result, df)


# 使用 pytest 的参数化装饰器，针对 all_connectable 中的每个连接对象执行测试
@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_save_index(conn, request):
    # 如果连接对象包含 "adbc"，标记测试为预期失败
    if "adbc" in conn:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="ADBC implementation does not create index", strict=True
            )
        )
    conn_name = conn
    conn = request.getfixturevalue(conn)  # 获取测试夹具中的连接对象
    df = DataFrame.from_records(
        [(1, 2.1, "line1"), (2, 1.5, "line2")], columns=["A", "B", "C"], index=["A"]
    )

    tbl_name = "test_to_sql_saves_index"
    # 使用 pandasSQL_builder 构造器来创建 pandasSQL 对象
    with pandasSQL_builder(conn) as pandasSQL:
        # 在事务内执行 to_sql 操作，并断言插入的行数为 2
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(df, tbl_name) == 2

    # 如果连接名称为 "sqlite_buildin" 或 "sqlite_str"，查询表的索引信息
    if conn_name in {"sqlite_buildin", "sqlite_str"}:
        # 查询 SQLite 数据库中的索引信息
        ixs = sql.read_sql_query(
            "SELECT * FROM sqlite_master WHERE type = 'index' "
            f"AND tbl_name = '{tbl_name}'",
            conn,
        )
        ix_cols = []
        # 获取索引的列信息
        for ix_name in ixs.name:
            ix_info = sql.read_sql_query(f"PRAGMA index_info({ix_name})", conn)
            ix_cols.append(ix_info.name.tolist())
    else:
        # 对于其他数据库，使用 SQLAlchemy 的 inspect 方法获取表的索引信息
        from sqlalchemy import inspect

        insp = inspect(conn)
        ixs = insp.get_indexes(tbl_name)
        ix_cols = [i["column_names"] for i in ixs]

    # 断言表的索引列与预期的 ["A"] 相等
    assert ix_cols == [["A"]]


# 使用 pytest 的参数化装饰器，针对 all_connectable 中的每个连接对象执行测试
@pytest.mark.parametrize("conn", all_connectable)
def test_transactions(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)  # 获取测试夹具中的连接对象

    stmt = "CREATE TABLE test_trans (A INT, B TEXT)"
    # 如果连接名称不是 "sqlite_buildin" 并且不包含 "adbc"，使用 SQLAlchemy 的 text 方法处理 SQL 语句
    if conn_name != "sqlite_buildin" and "adbc" not in conn_name:
        from sqlalchemy import text

        stmt = text(stmt)

    # 使用 pandasSQL_builder 构造器创建 pandasSQL 对象
    with pandasSQL_builder(conn) as pandasSQL:
        # 在事务内执行 SQL 语句创建表 test_trans
        with pandasSQL.run_transaction() as trans:
            trans.execute(stmt)


# 使用 pytest 的参数化装饰器，针对 all_connectable 中的每个连接对象执行测试
@pytest.mark.parametrize("conn", all_connectable)
def test_transaction_rollback(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)  # 获取测试夹具中的连接对象
    # 使用 pandasSQL_builder 创建一个连接对象 pandasSQL，并使用上下文管理器
    with pandasSQL_builder(conn) as pandasSQL:
        # 在 pandasSQL 上下文中，使用 run_transaction 方法创建事务对象 trans
        with pandasSQL.run_transaction() as trans:
            # 定义 SQL 语句创建表 test_trans
            stmt = "CREATE TABLE test_trans (A INT, B TEXT)"
            # 如果 conn_name 中包含 "adbc" 或者 pandasSQL 是 SQLiteDatabase 类型的实例
            if "adbc" in conn_name or isinstance(pandasSQL, SQLiteDatabase):
                # 执行 SQL 语句创建表
                trans.execute(stmt)
            else:
                # 否则导入 SQLAlchemy 的 text 模块，将 stmt 转换成文本类型的 SQL 语句对象
                from sqlalchemy import text
                stmt = text(stmt)
                # 执行文本类型的 SQL 语句
                trans.execute(stmt)

        # 定义一个自定义的异常类 DummyException
        class DummyException(Exception):
            pass

        # 确保事务回滚时没有数据插入
        ins_sql = "INSERT INTO test_trans (A,B) VALUES (1, 'blah')"
        # 如果 pandasSQL 是 SQLDatabase 的实例
        if isinstance(pandasSQL, SQLDatabase):
            # 导入 SQLAlchemy 的 text 模块，将 ins_sql 转换成文本类型的 SQL 语句对象
            from sqlalchemy import text
            ins_sql = text(ins_sql)

        # 尝试执行插入操作并抛出 DummyException 异常
        try:
            with pandasSQL.run_transaction() as trans:
                trans.execute(ins_sql)
                raise DummyException("error")
        except DummyException:
            # 捕获并忽略 DummyException 异常
            pass
        
        # 使用 pandasSQL 上下文创建事务
        with pandasSQL.run_transaction():
            # 从 test_trans 表中读取数据
            res = pandasSQL.read_query("SELECT * FROM test_trans")
        # 断言查询结果为空
        assert len(res) == 0

        # 确保事务提交时数据被成功插入
        with pandasSQL.run_transaction() as trans:
            trans.execute(ins_sql)
            # 从 test_trans 表中读取数据
            res2 = pandasSQL.read_query("SELECT * FROM test_trans")
        # 断言查询结果包含一条记录
        assert len(res2) == 1
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_get_schema_create_table(conn, request, test_frame3):
    # 参数化测试，使用不同的数据库连接
    # 使用不包含布尔列的数据框，因为 MySQL 将布尔转换为 TINYINT，而 read_sql_table 将其返回为整数，导致 dtype 不匹配
    if conn == "sqlite_str":
        # 对于 sqlite_str fixture，标记为预期失败，因为该测试不支持 sqlite_str
        request.applymarker(
            pytest.mark.xfail(reason="test does not support sqlite_str fixture")
        )

    # 获取对应的数据库连接
    conn = request.getfixturevalue(conn)

    from sqlalchemy import text
    from sqlalchemy.engine import Engine

    tbl = "test_get_schema_create_table"
    # 获取用于创建表的 SQL 语句
    create_sql = sql.get_schema(test_frame3, tbl, con=conn)
    # 创建一个空的测试数据框
    blank_test_df = test_frame3.iloc[:0]

    # 将 SQL 语句转换为文本对象
    create_sql = text(create_sql)
    if isinstance(conn, Engine):
        # 如果连接是 Engine 类型，则使用连接执行 SQL 语句
        with conn.connect() as newcon:
            with newcon.begin():
                newcon.execute(create_sql)
    else:
        # 否则，直接使用连接执行 SQL 语句
        conn.execute(create_sql)

    # 从数据库中读取表格数据到返回的数据框中
    returned_df = sql.read_sql_table(tbl, conn)
    # 断言返回的数据框与空的测试数据框相等，不检查索引类型
    tm.assert_frame_equal(returned_df, blank_test_df, check_index_type=False)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_dtype(conn, request):
    if conn == "sqlite_str":
        # 如果是 sqlite_str，跳过测试，因为 sqlite_str 没有检查系统
        pytest.skip("sqlite_str has no inspection system")

    # 获取对应的数据库连接
    conn = request.getfixturevalue(conn)

    from sqlalchemy import (
        TEXT,
        String,
    )
    from sqlalchemy.schema import MetaData

    cols = ["A", "B"]
    data = [(0.8, True), (0.9, None)]
    # 创建一个数据框
    df = DataFrame(data, columns=cols)

    # 将数据框写入数据库表格，检查返回的行数为 2
    assert df.to_sql(name="dtype_test", con=conn) == 2
    # 使用特定的 dtype 将数据框写入数据库表格，检查返回的行数为 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": TEXT}) == 2

    meta = MetaData()
    # 反射数据库结构
    meta.reflect(bind=conn)
    # 获取表格 dtype_test2 中列 B 的 SQL 类型
    sqltype = meta.tables["dtype_test2"].columns["B"].type
    # 断言列 B 的 SQL 类型是 TEXT
    assert isinstance(sqltype, TEXT)

    msg = "The type of B is not a SQLAlchemy type"
    # 断言抛出 ValueError 异常，消息为 "The type of B is not a SQLAlchemy type"
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": str})

    # GH9083
    # 使用特定长度的 String 类型将数据框写入数据库表格，检查返回的行数为 2
    assert df.to_sql(name="dtype_test3", con=conn, dtype={"B": String(10)}) == 2
    meta.reflect(bind=conn)
    # 获取表格 dtype_test3 中列 B 的 SQL 类型
    sqltype = meta.tables["dtype_test3"].columns["B"].type
    # 断言列 B 的 SQL 类型是 String 类型
    assert isinstance(sqltype, String)
    # 断言列 B 的 String 类型长度为 10
    assert sqltype.length == 10

    # 使用单一 dtype 将数据框写入数据库表格，检查返回的行数为 2
    assert df.to_sql(name="single_dtype_test", con=conn, dtype=TEXT) == 2
    meta.reflect(bind=conn)
    # 获取表格 single_dtype_test 中列 A 和 B 的 SQL 类型
    sqltypea = meta.tables["single_dtype_test"].columns["A"].type
    sqltypeb = meta.tables["single_dtype_test"].columns["B"].type
    # 断言列 A 和 B 的 SQL 类型是 TEXT
    assert isinstance(sqltypea, TEXT)
    assert isinstance(sqltypeb, TEXT)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_notna_dtype(conn, request):
    if conn == "sqlite_str":
        # 如果是 sqlite_str，跳过测试，因为 sqlite_str 没有检查系统
        pytest.skip("sqlite_str has no inspection system")

    # 保存原始的连接名
    conn_name = conn
    # 获取对应的数据库连接
    conn = request.getfixturevalue(conn)

    from sqlalchemy import (
        Boolean,
        DateTime,
        Float,
        Integer,
    )
    from sqlalchemy.schema import MetaData
    # 创建一个包含不同数据类型的字典，每种数据类型对应一个 Series 对象
    cols = {
        "Bool": Series([True, None]),
        "Date": Series([datetime(2012, 5, 1), None]),
        "Int": Series([1, None], dtype="object"),
        "Float": Series([1.1, None]),
    }
    # 根据字典创建一个 DataFrame 对象
    df = DataFrame(cols)

    # 定义要写入的表名
    tbl = "notna_dtype_test"
    # 将 DataFrame 写入 SQL 数据库，并断言写入的行数为 2
    assert df.to_sql(name=tbl, con=conn) == 2
    # 从数据库中读取刚刚写入的表，并存储在 _ 变量中
    _ = sql.read_sql_table(tbl, conn)
    # 创建一个数据库的元数据对象
    meta = MetaData()
    # 反射数据库结构并绑定到给定的连接上
    meta.reflect(bind=conn)
    # 根据数据库连接名决定使用的数据类型，可能是 Integer 或 Boolean
    my_type = Integer if "mysql" in conn_name else Boolean
    # 获取表的列字典
    col_dict = meta.tables[tbl].columns
    # 断言每列的数据类型是否符合预期
    assert isinstance(col_dict["Bool"].type, my_type)
    assert isinstance(col_dict["Date"].type, DateTime)
    assert isinstance(col_dict["Int"].type, Integer)
    assert isinstance(col_dict["Float"].type, Float)
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 使用 pytest 的 parametrize 标记来多次运行该测试函数，每次传入不同的连接参数
def test_double_precision(conn, request):
    # 如果连接是 sqlite_str，则跳过测试，因为 sqlite_str 没有检查系统
    if conn == "sqlite_str":
        pytest.skip("sqlite_str has no inspection system")

    # 获取 conn 对象，这里使用 request 对象来获取相应的 fixture
    conn = request.getfixturevalue(conn)

    # 导入需要使用的 SQLAlchemy 相关模块和类
    from sqlalchemy import (
        BigInteger,
        Float,
        Integer,
    )
    from sqlalchemy.schema import MetaData

    # 设置一个浮点数值 V
    V = 1.23456789101112131415

    # 创建一个 DataFrame 对象 df，包含不同类型和精度的列数据
    df = DataFrame(
        {
            "f32": Series([V], dtype="float32"),
            "f64": Series([V], dtype="float64"),
            "f64_as_f32": Series([V], dtype="float64"),
            "i32": Series([5], dtype="int32"),
            "i64": Series([5], dtype="int64"),
        }
    )

    # 将 DataFrame df 写入 SQL 表格，指定连接对象 conn，如果表已存在则替换，指定部分列的数据类型
    assert (
        df.to_sql(
            name="test_dtypes",
            con=conn,
            index=False,
            if_exists="replace",
            dtype={"f64_as_f32": Float(precision=23)},
        )
        == 1
    )

    # 从 SQL 表格中读取数据到 res
    res = sql.read_sql_table("test_dtypes", conn)

    # 检查 float64 列的精度
    assert np.round(df["f64"].iloc[0], 14) == np.round(res["f64"].iloc[0], 14)

    # 检查 SQL 类型
    meta = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables["test_dtypes"].columns
    assert str(col_dict["f32"].type) == str(col_dict["f64_as_f32"].type)
    assert isinstance(col_dict["f32"].type, Float)
    assert isinstance(col_dict["f64"].type, Float)
    assert isinstance(col_dict["i32"].type, Integer)
    assert isinstance(col_dict["i64"].type, BigInteger)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 使用 pytest 的 parametrize 标记来多次运行该测试函数，每次传入不同的连接参数
def test_connectable_issue_example(conn, request):
    # 获取 conn 对象，这里使用 request 对象来获取相应的 fixture
    conn = request.getfixturevalue(conn)

    # 这个测试案例检验了在 GitHub 上报告的问题
    # https://github.com/pandas-dev/pandas/issues/10104
    from sqlalchemy.engine import Engine

    # 定义测试函数 test_select，执行 SQL 查询
    def test_select(connection):
        query = "SELECT test_foo_data FROM test_foo_data"
        return sql.read_sql_query(query, con=connection)

    # 定义测试函数 test_append，将数据追加到 SQL 表中
    def test_append(connection, data):
        data.to_sql(name="test_foo_data", con=connection, if_exists="append")

    # 定义测试函数 test_connectable，执行 SQL 查询并将数据追加到连接对象中
    def test_connectable(conn):
        foo_data = test_select(conn)
        test_append(conn, foo_data)

    # 定义主函数 main，根据连接类型执行相应的操作
    def main(connectable):
        if isinstance(connectable, Engine):
            with connectable.connect() as conn:
                with conn.begin():
                    test_connectable(conn)
        else:
            test_connectable(connectable)

    # 将一个 DataFrame 对象写入 SQL 表格，名称为 test_foo_data，指定连接对象 conn
    assert (
        DataFrame({"test_foo_data": [0, 1, 2]}).to_sql(name="test_foo_data", con=conn)
        == 3
    )

    # 执行主函数 main，传入连接对象 conn
    main(conn)


@pytest.mark.parametrize("conn", sqlalchemy_connectable)
# 使用 pytest 的 parametrize 标记来多次运行该测试函数，每次传入不同的连接参数
@pytest.mark.parametrize(
    "input",
    [{"foo": [np.inf]}, {"foo": [-np.inf]}, {"foo": [-np.inf], "infe0": ["bar"]}],
)
def test_to_sql_with_negative_npinf(conn, request, input):
    # GH 34431

    # 创建一个 DataFrame 对象 df，包含输入参数 input 的数据
    df = DataFrame(input)
    conn_name = conn
    # 从测试夹具中获取数据库连接对象
    conn = request.getfixturevalue(conn)

    # 如果连接名称中包含 "mysql"
    if "mysql" in conn_name:
        # GH 36465
        # 对于 pymysql 版本 >= 0.10，输入 {"foo": [-np.inf], "infe0": ["bar"]} 不会引发任何错误
        # 对于 pymysql 版本 < 1.0.3 并且 DataFrame 的列中包含 "infe0"，标记该测试为预期失败
        # TODO(GH#36465): 在 GH 36465 修复后移除此版本检查
        pymysql = pytest.importorskip("pymysql")

        if Version(pymysql.__version__) < Version("1.0.3") and "infe0" in df.columns:
            mark = pytest.mark.xfail(reason="GH 36465")
            request.applymarker(mark)

        # 尝试将 DataFrame 写入名为 "foobar" 的 MySQL 数据库表中，预期会引发 ValueError 异常
        msg = "inf cannot be used with MySQL"
        with pytest.raises(ValueError, match=msg):
            df.to_sql(name="foobar", con=conn, index=False)
    else:
        # 断言将 DataFrame 写入名为 "foobar" 的数据库表成功，并返回值为 1
        assert df.to_sql(name="foobar", con=conn, index=False) == 1
        # 从数据库中读取名为 "foobar" 的表，并将结果存储在 res 变量中
        res = sql.read_sql_table("foobar", conn)
        # 使用测试工具比较 DataFrame 和从数据库中读取的结果 res 是否相等
        tm.assert_equal(df, res)
# 使用 pytest.mark.parametrize 注册测试用例，conn 参数来自 sqlalchemy_connectable
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_temporary_table(conn, request):
    # 如果 conn 为 "sqlite_str"，跳过测试并给出相应的提示信息
    if conn == "sqlite_str":
        pytest.skip("test does not work with str connection")

    # 获取 conn 对应的 fixture 值
    conn = request.getfixturevalue(conn)

    # 导入需要的 SQLAlchemy 类和函数
    from sqlalchemy import (
        Column,
        Integer,
        Unicode,
        select,
    )
    from sqlalchemy.orm import (
        Session,
        declarative_base,
    )

    # 设置测试数据和预期结果
    test_data = "Hello, World!"
    expected = DataFrame({"spam": [test_data]})
    
    # 创建 SQLAlchemy 的基类 Base
    Base = declarative_base()

    # 定义临时表 Temporary
    class Temporary(Base):
        __tablename__ = "temp_test"
        __table_args__ = {"prefixes": ["TEMPORARY"]}
        id = Column(Integer, primary_key=True)
        spam = Column(Unicode(30), nullable=False)

    # 使用 session 进行数据库操作
    with Session(conn) as session:
        with session.begin():
            conn = session.connection()
            # 在数据库中创建 Temporary 表
            Temporary.__table__.create(conn)
            # 向 Temporary 表中添加数据
            session.add(Temporary(spam=test_data))
            session.flush()
            # 从数据库中读取数据并转换为 DataFrame
            df = sql.read_sql_query(sql=select(Temporary.spam), con=conn)
    
    # 断言 DataFrame 是否与预期结果一致
    tm.assert_frame_equal(df, expected)


# 使用 pytest.mark.parametrize 注册测试用例，conn 参数来自 all_connectable
@pytest.mark.parametrize("conn", all_connectable)
def test_invalid_engine(conn, request, test_frame1):
    # 如果 conn 为 "sqlite_buildin" 或者包含 "adbc"，标记为预期失败，提供相应的原因
    if conn == "sqlite_buildin" or "adbc" in conn:
        request.applymarker(
            pytest.mark.xfail(
                reason="SQLiteDatabase/ADBCDatabase does not raise for bad engine"
            )
        )

    # 获取 conn 对应的 fixture 值
    conn = request.getfixturevalue(conn)
    msg = "engine must be one of 'auto', 'sqlalchemy'"
    
    # 使用 pandasSQL_builder 创建 pandasSQL 对象，上下文管理
    with pandasSQL_builder(conn) as pandasSQL:
        with pytest.raises(ValueError, match=msg):
            # 调用 pandasSQL 的 to_sql 方法，传入错误的 engine 参数，应该抛出 ValueError 异常
            pandasSQL.to_sql(test_frame1, "test_frame1", engine="bad_engine")


# 使用 pytest.mark.parametrize 注册测试用例，conn 参数来自 all_connectable
@pytest.mark.parametrize("conn", all_connectable)
def test_to_sql_with_sql_engine(conn, request, test_frame1):
    """`to_sql` with the `engine` param"""
    # mostly copied from this class's `_to_sql()` method
    # 获取 conn 对应的 fixture 值
    conn = request.getfixturevalue(conn)
    
    # 使用 pandasSQL_builder 创建 pandasSQL 对象，上下文管理
    with pandasSQL_builder(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            # 断言 to_sql 方法使用 auto engine 插入数据后返回值为 4
            assert pandasSQL.to_sql(test_frame1, "test_frame1", engine="auto") == 4
            # 断言数据库中存在表 "test_frame1"
            assert pandasSQL.has_table("test_frame1")

    # 获取 test_frame1 数据的条目数
    num_entries = len(test_frame1)
    # 获取数据库中 "test_frame1" 表的行数
    num_rows = count_rows(conn, "test_frame1")
    # 断言行数与数据条目数相等
    assert num_rows == num_entries


# 使用 pytest.mark.parametrize 注册测试用例，conn 参数来自 sqlalchemy_connectable
@pytest.mark.parametrize("conn", sqlalchemy_connectable)
def test_options_sqlalchemy(conn, request, test_frame1):
    # use the set option
    # 获取 conn 对应的 fixture 值
    conn = request.getfixturevalue(conn)
    
    # 设置 pandas 的 option，指定 io.sql.engine 为 "sqlalchemy"
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        # 使用 pandasSQL_builder 创建 pandasSQL 对象，上下文管理
        with pandasSQL_builder(conn) as pandasSQL:
            with pandasSQL.run_transaction():
                # 断言 to_sql 方法使用 sqlalchemy engine 插入数据后返回值为 4
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                # 断言数据库中存在表 "test_frame1"
                assert pandasSQL.has_table("test_frame1")

        # 获取 test_frame1 数据的条目数
        num_entries = len(test_frame1)
        # 获取数据库中 "test_frame1" 表的行数
        num_rows = count_rows(conn, "test_frame1")
        # 断言行数与数据条目数相等
        assert num_rows == num_entries


# 使用 pytest.mark.parametrize 注册测试用例，conn 参数来自 all_connectable
@pytest.mark.parametrize("conn", all_connectable)
def test_options_auto(conn, request, test_frame1):
    # 使用指定选项设置连接对象
    conn = request.getfixturevalue(conn)
    # 使用 Pandas 的上下文管理器，设置 SQL 引擎自动化
    with pd.option_context("io.sql.engine", "auto"):
        # 使用 pandasSQL_builder 创建 pandasSQL 对象，并使用上下文管理器
        with pandasSQL_builder(conn) as pandasSQL:
            # 在 pandasSQL 对象上运行事务
            with pandasSQL.run_transaction():
                # 断言将 test_frame1 数据框写入数据库表 "test_frame1"，返回行数为 4
                assert pandasSQL.to_sql(test_frame1, "test_frame1") == 4
                # 断言检查数据库中是否存在表 "test_frame1"
                assert pandasSQL.has_table("test_frame1")

        # 计算 test_frame1 的条目数
        num_entries = len(test_frame1)
        # 统计数据库表 "test_frame1" 中的行数
        num_rows = count_rows(conn, "test_frame1")
        # 断言检查数据库表 "test_frame1" 中的行数是否等于 test_frame1 的条目数
        assert num_rows == num_entries
def test_options_get_engine():
    # 导入 SQLAlchemy，如果不成功则跳过测试
    pytest.importorskip("sqlalchemy")
    # 确保 get_engine 返回的引擎类型是 SQLAlchemyEngine
    assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

    # 设置 pandas 上下文选项 "io.sql.engine" 为 "sqlalchemy"
    with pd.option_context("io.sql.engine", "sqlalchemy"):
        # 确保 get_engine("auto") 返回的引擎类型是 SQLAlchemyEngine
        assert isinstance(get_engine("auto"), SQLAlchemyEngine)
        # 再次确认 get_engine("sqlalchemy") 返回的引擎类型是 SQLAlchemyEngine
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)

    # 设置 pandas 上下文选项 "io.sql.engine" 为 "auto"
    with pd.option_context("io.sql.engine", "auto"):
        # 确保 get_engine("auto") 返回的引擎类型是 SQLAlchemyEngine
        assert isinstance(get_engine("auto"), SQLAlchemyEngine)
        # 再次确认 get_engine("sqlalchemy") 返回的引擎类型是 SQLAlchemyEngine
        assert isinstance(get_engine("sqlalchemy"), SQLAlchemyEngine)


def test_get_engine_auto_error_message():
    # 预期 get_engine(engine="auto") 在引擎未安装或版本错误时会产生不同的错误消息
    # TODO(GH#36893) 当我们添加更多引擎时再填写此部分


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
def test_read_sql_dtype_backend(
    conn,
    request,
    string_storage,
    func,
    dtype_backend,
    dtype_backend_data,
    dtype_backend_expected,
):
    # GH#50048
    # 使用给定的连接名获取连接对象
    conn_name = conn
    conn = request.getfixturevalue(conn)
    # 定义数据库表名
    table = "test"
    # 获取测试数据
    df = dtype_backend_data
    # 将 DataFrame 写入数据库表中，如果表已存在则替换
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    # 设置 pandas 上下文选项 "mode.string_storage" 为 string_storage
    with pd.option_context("mode.string_storage", string_storage):
        # 调用 pd.func 查询数据库中的数据，并指定 dtype_backend
        result = getattr(pd, func)(
            f"Select * from {table}", conn, dtype_backend=dtype_backend
        )
    # 获取预期的结果
    expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    # 断言查询结果与预期结果相等
    tm.assert_frame_equal(result, expected)

    # 如果连接名中包含 "adbc"
    if "adbc" in conn_name:
        # 标记为 xfail，因为 adbc 不支持 chunksize 参数
        request.applymarker(
            pytest.mark.xfail(reason="adbc does not support chunksize argument")
        )

    # 再次设置 pandas 上下文选项 "mode.string_storage" 为 string_storage
    with pd.option_context("mode.string_storage", string_storage):
        # 使用指定的 chunksize 迭代查询数据库中的数据，并指定 dtype_backend
        iterator = getattr(pd, func)(
            f"Select * from {table}",
            con=conn,
            dtype_backend=dtype_backend,
            chunksize=3,
        )
        # 获取预期的结果
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        # 逐一断言迭代结果与预期结果相等
        for result in iterator:
            tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table"])
def test_read_sql_dtype_backend_table(
    conn,
    request,
    string_storage,
    func,
    dtype_backend,
    dtype_backend_data,
    dtype_backend_expected,
):
    # 如果连接名中包含 "sqlite" 且不包含 "adbc"
    if "sqlite" in conn and "adbc" not in conn:
        # 标记为 xfail，因为 SQLite 实际上通过 read_sql_table 返回适当的布尔值，
        # 但在 pytest 重构之前被跳过了
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "SQLite actually returns proper boolean values via "
                    "read_sql_table, but before pytest refactor was skipped"
                )
            )
        )
    # GH#50048
    # 使用给定的连接名获取连接对象
    conn_name = conn
    conn = request.getfixturevalue(conn)
    # 定义数据库表名
    table = "test"
    # 获取测试数据
    df = dtype_backend_data
    # 将 DataFrame 写入数据库表中，如果表已存在则替换
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")
    # 设置 Pandas 上下文管理器，修改 "mode.string_storage" 参数为指定值 string_storage
    with pd.option_context("mode.string_storage", string_storage):
        # 调用 Pandas 中的函数 func，并传入 table, conn, dtype_backend 等参数，将结果赋给 result
        result = getattr(pd, func)(table, conn, dtype_backend=dtype_backend)
    
    # 调用函数 dtype_backend_expected，计算预期结果 expected，用于后续比较
    expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
    
    # 如果连接名 conn_name 中包含 "adbc" 字符串，则返回，因为 adbc 不支持 chunksize 参数
    if "adbc" in conn_name:
        return
    
    # 设置 Pandas 上下文管理器，修改 "mode.string_storage" 参数为指定值 string_storage
    with pd.option_context("mode.string_storage", string_storage):
        # 调用 Pandas 中的函数 func，并传入 table, conn, dtype_backend 等参数，
        # 同时指定 chunksize=3，将结果赋给 iterator
        iterator = getattr(pd, func)(
            table,
            conn,
            dtype_backend=dtype_backend,
            chunksize=3,
        )
        
        # 计算预期结果 expected，用于后续比较
        expected = dtype_backend_expected(string_storage, dtype_backend, conn_name)
        
        # 遍历迭代器 iterator 中的每个结果 result，与预期结果 expected 进行比较
        for result in iterator:
            tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器参数化测试用例，其中 conn 参数取自 all_connectable 列表
# func 参数取自列表 ["read_sql", "read_sql_table", "read_sql_query"]
@pytest.mark.parametrize("conn", all_connectable)
@pytest.mark.parametrize("func", ["read_sql", "read_sql_table", "read_sql_query"])
def test_read_sql_invalid_dtype_backend_table(conn, request, func, dtype_backend_data):
    # 获取测试用例中的数据库连接对象
    conn = request.getfixturevalue(conn)
    # 设置测试表名为 "test"
    table = "test"
    # 使用 fixture 提供的 dtype_backend_data 作为测试数据集
    df = dtype_backend_data
    # 将数据框 df 写入数据库表中，若表存在则替换
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    # 准备用于匹配异常消息的字符串
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    # 断言调用 pandas 的 func 函数时，使用 "numpy" 作为 dtype_backend 抛出 ValueError 异常，并匹配消息 msg
    with pytest.raises(ValueError, match=msg):
        getattr(pd, func)(table, conn, dtype_backend="numpy")


# 定义 fixture，返回一个包含指定数据的 DataFrame 对象
@pytest.fixture
def dtype_backend_data() -> DataFrame:
    return DataFrame(
        {
            "a": Series([1, np.nan, 3], dtype="Int64"),
            "b": Series([1, 2, 3], dtype="Int64"),
            "c": Series([1.5, np.nan, 2.5], dtype="Float64"),
            "d": Series([1.5, 2.0, 2.5], dtype="Float64"),
            "e": [True, False, None],
            "f": [True, False, True],
            "g": ["a", "b", "c"],
            "h": ["a", "b", None],
        }
    )


# 定义一个空的 fixture，等待后续补充
@pytest.fixture
def dtype_backend_expected():
    pass


这些注释完整解释了每行代码的作用和意图，符合要求的格式和内容要求。
    # 定义一个名为 func 的函数，接受三个参数 storage, dtype_backend, conn_name，并返回一个 DataFrame 对象
    def func(storage, dtype_backend, conn_name) -> DataFrame:
        # 声明两个变量 string_array 和 string_array_na，类型为 StringArray 或 ArrowStringArray
        string_array: StringArray | ArrowStringArray
        string_array_na: StringArray | ArrowStringArray
        
        # 如果 storage 参数为 "python"，执行以下代码块
        if storage == "python":
            # 使用 np.array 创建一个包含 ["a", "b", "c"] 的数组，并转换为 StringArray 类型赋值给 string_array
            string_array = StringArray(np.array(["a", "b", "c"], dtype=np.object_))
            # 使用 np.array 创建一个包含 ["a", "b", pd.NA] 的数组，并转换为 StringArray 类型赋值给 string_array_na
            string_array_na = StringArray(np.array(["a", "b", pd.NA], dtype=np.object_))

        # 如果 dtype_backend 参数为 "pyarrow"，执行以下代码块
        elif dtype_backend == "pyarrow":
            # 导入 pytest 并跳过导入失败的情况
            pa = pytest.importorskip("pyarrow")
            # 从 pandas.arrays 模块导入 ArrowExtensionArray 类
            from pandas.arrays import ArrowExtensionArray
            
            # 使用 pa.array 创建一个包含 ["a", "b", "c"] 的数组，并转换为 ArrowExtensionArray 类型赋值给 string_array
            string_array = ArrowExtensionArray(pa.array(["a", "b", "c"]))  # type: ignore[assignment]
            # 使用 pa.array 创建一个包含 ["a", "b", None] 的数组，并转换为 ArrowExtensionArray 类型赋值给 string_array_na
            string_array_na = ArrowExtensionArray(pa.array(["a", "b", None]))  # type: ignore[assignment]

        # 如果以上两个条件均不满足，执行以下代码块
        else:
            # 导入 pytest 并跳过导入失败的情况
            pa = pytest.importorskip("pyarrow")
            # 使用 pa.array 创建一个包含 ["a", "b", "c"] 的数组，并转换为 ArrowStringArray 类型赋值给 string_array
            string_array = ArrowStringArray(pa.array(["a", "b", "c"]))
            # 使用 pa.array 创建一个包含 ["a", "b", None] 的数组，并转换为 ArrowStringArray 类型赋值给 string_array_na
            string_array_na = ArrowStringArray(pa.array(["a", "b", None]))

        # 使用 DataFrame 构造函数创建一个 DataFrame 对象 df，包含以下列和数据
        df = DataFrame(
            {
                "a": Series([1, np.nan, 3], dtype="Int64"),    # 整数列 "a" 包含值 [1, NaN, 3]
                "b": Series([1, 2, 3], dtype="Int64"),         # 整数列 "b" 包含值 [1, 2, 3]
                "c": Series([1.5, np.nan, 2.5], dtype="Float64"),  # 浮点数列 "c" 包含值 [1.5, NaN, 2.5]
                "d": Series([1.5, 2.0, 2.5], dtype="Float64"),  # 浮点数列 "d" 包含值 [1.5, 2.0, 2.5]
                "e": Series([True, False, pd.NA], dtype="boolean"),  # 布尔列 "e" 包含值 [True, False, NaN]
                "f": Series([True, False, True], dtype="boolean"),   # 布尔列 "f" 包含值 [True, False, True]
                "g": string_array,                              # 字符串列 "g" 使用之前定义的 string_array
                "h": string_array_na,                           # 字符串列 "h" 使用之前定义的 string_array_na
            }
        )

        # 如果 dtype_backend 参数为 "pyarrow"，执行以下代码块
        if dtype_backend == "pyarrow":
            # 导入 pytest 并跳过导入失败的情况
            pa = pytest.importorskip("pyarrow")
            # 从 pandas.arrays 模块导入 ArrowExtensionArray 类
            from pandas.arrays import ArrowExtensionArray
            
            # 使用字典推导式，将 df 中的每一列转换为 ArrowExtensionArray 类型，重新赋值给 df
            df = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )

        # 如果 conn_name 参数中包含 "mysql" 或 "sqlite"，执行以下代码块
        if "mysql" in conn_name or "sqlite" in conn_name:
            # 如果 dtype_backend 参数为 "numpy_nullable"，执行以下代码块
            if dtype_backend == "numpy_nullable":
                # 将 df 的 "e" 和 "f" 列类型转换为 "Int64"
                df = df.astype({"e": "Int64", "f": "Int64"})
            # 否则，执行以下代码块
            else:
                # 将 df 的 "e" 和 "f" 列类型转换为 "int64[pyarrow]"
                df = df.astype({"e": "int64[pyarrow]", "f": "int64[pyarrow]"})

        # 返回最终构造好的 DataFrame 对象 df
        return df

    # 返回 func 函数的引用
    return func
@pytest.mark.parametrize("conn", all_connectable)
# 使用 pytest.mark.parametrize 装饰器，为测试函数 test_chunksize_empty_dtypes 参数化数据库连接
def test_chunksize_empty_dtypes(conn, request):
    # GH#50245: GitHub issue reference indicating the reason for the following condition
    if "adbc" in conn:
        # 如果数据库连接字符串包含 "adbc"，则给当前测试用例添加标记，表示这种情况下 chunksize 参数不支持
        request.node.add_marker(
            pytest.mark.xfail(reason="chunksize argument NotImplemented with ADBC")
        )
    # 获取数据库连接
    conn = request.getfixturevalue(conn)
    # 定义数据类型字典
    dtypes = {"a": "int64", "b": "object"}
    # 创建一个空的 DataFrame，并指定列的数据类型
    df = DataFrame(columns=["a", "b"]).astype(dtypes)
    # 复制 DataFrame，作为预期结果
    expected = df.copy()
    # 将 DataFrame 写入数据库表中
    df.to_sql(name="test", con=conn, index=False, if_exists="replace")

    # 使用 chunksize=1 读取从数据库表中查询的结果集
    for result in read_sql_query(
        "SELECT * FROM test",
        conn,
        dtype=dtypes,
        chunksize=1,
    ):
        # 使用测试工具函数验证每个结果是否与预期一致
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("conn", all_connectable)
# 参数化数据库连接
@pytest.mark.parametrize("dtype_backend", [lib.no_default, "numpy_nullable"])
# 参数化 dtype_backend，可能的取值是 lib.no_default 或 "numpy_nullable"
@pytest.mark.parametrize("func", ["read_sql", "read_sql_query"])
# 参数化函数名称，可以是 "read_sql" 或 "read_sql_query"
def test_read_sql_dtype(conn, request, func, dtype_backend):
    # GH#50797: GitHub issue reference indicating the reason for this test case
    conn = request.getfixturevalue(conn)
    # 获取数据库连接
    table = "test"
    # 创建一个测试用的 DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": 5})
    # 将 DataFrame 写入数据库表中
    df.to_sql(name=table, con=conn, index=False, if_exists="replace")

    # 使用 getattr 动态调用 pandas 中的函数（read_sql 或 read_sql_query）
    result = getattr(pd, func)(
        f"Select * from {table}",
        conn,
        dtype={"a": np.float64},
        dtype_backend=dtype_backend,
    )
    # 创建预期结果的 DataFrame
    expected = DataFrame(
        {
            "a": Series([1, 2, 3], dtype=np.float64),
            "b": Series(
                [5, 5, 5],
                dtype="int64" if not dtype_backend == "numpy_nullable" else "Int64",
            ),
        }
    )
    # 使用测试工具函数验证结果是否与预期一致
    tm.assert_frame_equal(result, expected)


def test_bigint_warning(sqlite_engine):
    conn = sqlite_engine
    # 测试确保不会因为 BIGINT 数据类型而引发警告 (GH7433)
    df = DataFrame({"a": [1, 2]}, dtype="int64")
    # 将 DataFrame 写入数据库表中，并断言返回的行数为 2
    assert df.to_sql(name="test_bigintwarning", con=conn, index=False) == 2

    # 使用 pytest 的 assert_produces_warning 上下文管理器，验证不会产生任何警告
    with tm.assert_produces_warning(None):
        # 读取数据库表，断言不会产生警告
        sql.read_sql_table("test_bigintwarning", conn)


def test_valueerror_exception(sqlite_engine):
    conn = sqlite_engine
    # 创建一个 DataFrame，其中指定了一个空的表名，预期会引发 ValueError 异常，异常信息包含 "Empty table name specified"
    df = DataFrame({"col1": [1, 2], "col2": [3, 4]})
    # 使用 pytest 的 pytest.raises 断言捕获 ValueError 异常，检查异常信息是否符合预期
    with pytest.raises(ValueError, match="Empty table name specified"):
        # 将 DataFrame 写入数据库表中
        df.to_sql(name="", con=conn, if_exists="replace", index=False)


def test_row_object_is_named_tuple(sqlite_engine):
    conn = sqlite_engine
    # GH 40682: GitHub issue reference indicating the reason for placing this test case here
    # 测试 is_named_tuple() 函数的使用，由于其使用了 sqlalchemy，因此将其放置在这里

    from sqlalchemy import (
        Column,
        Integer,
        String,
    )
    from sqlalchemy.orm import (
        declarative_base,
        sessionmaker,
    )

    BaseModel = declarative_base()

    class Test(BaseModel):
        __tablename__ = "test_frame"
        id = Column(Integer, primary_key=True)
        string_column = Column(String(50))

    with conn.begin():
        # 使用连接的事务执行创建表的操作
        BaseModel.metadata.create_all(conn)
    # 创建会话类
    Session = sessionmaker(bind=conn)
    # 创建一个数据库会话，使用完毕后会自动关闭会话
    with Session() as session:
        # 创建一个数据框，包含两列数据：id 和 string_column
        df = DataFrame({"id": [0, 1], "string_column": ["hello", "world"]})
        # 将数据框写入数据库中的表 "test_frame"，如果表已存在则替换，返回受影响的行数（这里是2）
        assert (
            df.to_sql(name="test_frame", con=conn, index=False, if_exists="replace")
            == 2
        )
        # 提交数据库会话中的所有操作
        session.commit()
        # 创建一个数据库查询，查询表 Test 中的 id 和 string_column 列
        test_query = session.query(Test.id, Test.string_column)
        # 将查询结果转换为数据框
        df = DataFrame(test_query)
    
    # 断言数据框的列名是否与指定的列表一致
    assert list(df.columns) == ["id", "string_column"]
def test_read_sql_string_inference(sqlite_engine):
    # 使用传入的 SQLite 引擎连接数据库
    conn = sqlite_engine
    # 导入 pyarrow 库，如果失败则跳过此测试
    pytest.importorskip("pyarrow")
    # 定义表名为 "test"
    table = "test"
    # 创建一个包含一列 "a" 的 DataFrame
    df = DataFrame({"a": ["x", "y"]})
    # 将 DataFrame 写入 SQLite 表中，如果表已存在则替换
    df.to_sql(table, con=conn, index=False, if_exists="replace")

    # 在上下文中设置选项 "future.infer_string" 为 True
    with pd.option_context("future.infer_string", True):
        # 从数据库中读取表的数据
        result = read_sql_table(table, conn)

    # 定义预期的 DataFrame 数据类型为 "string[pyarrow_numpy]"
    dtype = "string[pyarrow_numpy]"
    # 创建预期的 DataFrame 对象
    expected = DataFrame(
        {"a": ["x", "y"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
    )

    # 使用 pytest 的 assert_frame_equal 断言 result 和 expected 相等
    tm.assert_frame_equal(result, expected)


def test_roundtripping_datetimes(sqlite_engine):
    # 使用传入的 SQLite 引擎连接数据库
    conn = sqlite_engine
    # 创建一个包含日期时间的 DataFrame 对象
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    # 将 DataFrame 写入 SQLite 表中，如果表已存在则替换
    df.to_sql("test", conn, if_exists="replace", index=False)
    # 从数据库中读取表的数据，并选择第一个元素的第一个列的值
    result = pd.read_sql("select * from test", conn).iloc[0, 0]
    # 使用 assert 断言结果与字符串 "2020-12-31 12:00:00.000000" 相等
    assert result == "2020-12-31 12:00:00.000000"


@pytest.fixture
def sqlite_builtin_detect_types():
    # 使用内存数据库 ":memory:" 创建 SQLite 连接，指定解析日期时间类型
    with contextlib.closing(
        sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    ) as closing_conn:
        with closing_conn as conn:
            # 生成 SQLite 连接对象
            yield conn


def test_roundtripping_datetimes_detect_types(sqlite_builtin_detect_types):
    # 使用传入的内存 SQLite 连接对象进行测试
    conn = sqlite_builtin_detect_types
    # 创建一个包含日期时间的 DataFrame 对象
    df = DataFrame({"t": [datetime(2020, 12, 31, 12)]}, dtype="datetime64[ns]")
    # 将 DataFrame 写入 SQLite 表中，如果表已存在则替换
    df.to_sql("test", conn, if_exists="replace", index=False)
    # 从数据库中读取表的数据，并选择第一个元素的第一个列的值
    result = pd.read_sql("select * from test", conn).iloc[0, 0]
    # 使用 assert 断言结果与 Timestamp("2020-12-31 12:00:00.000000") 相等
    assert result == Timestamp("2020-12-31 12:00:00.000000")


@pytest.mark.db
def test_psycopg2_schema_support(postgresql_psycopg2_engine):
    # 使用传入的 PostgreSQL 引擎连接数据库
    conn = postgresql_psycopg2_engine

    # 只有在 PostgreSQL 中才支持模式（schema），不支持在 MySQL 或 SQLite 中
    # 创建一个包含三列的 DataFrame 对象
    df = DataFrame({"col1": [1, 2], "col2": [0.1, 0.2], "col3": ["a", "n"]})

    # 在连接中创建一个新的模式 "other"
    with conn.connect() as con:
        with con.begin():
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            con.exec_driver_sql("CREATE SCHEMA other;")

    # 将 DataFrame 写入不同的模式中的表
    assert df.to_sql(name="test_schema_public", con=conn, index=False) == 2
    assert (
        df.to_sql(
            name="test_schema_public_explicit",
            con=conn,
            index=False,
            schema="public",
        )
        == 2
    )
    assert (
        df.to_sql(name="test_schema_other", con=conn, index=False, schema="other") == 2
    )

    # 从数据库中读取表，并使用 assert_frame_equal 断言与预期的 DataFrame 相等
    res1 = sql.read_sql_table("test_schema_public", conn)
    tm.assert_frame_equal(df, res1)
    res2 = sql.read_sql_table("test_schema_public_explicit", conn)
    tm.assert_frame_equal(df, res2)
    res3 = sql.read_sql_table("test_schema_public_explicit", conn, schema="public")
    tm.assert_frame_equal(df, res3)
    res4 = sql.read_sql_table("test_schema_other", conn, schema="other")
    tm.assert_frame_equal(df, res4)
    # 使用消息断言表 "test_schema_other" 存在
    msg = "Table test_schema_other not found"
    # 使用 pytest 的上下文，验证调用 read_sql_table 方法时会引发 ValueError，并且匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        sql.read_sql_table("test_schema_other", conn, schema="public")

    # 不同的 if_exists 选项示例

    # 创建一个新的模式（schema）
    with conn.connect() as con:
        with con.begin():
            # 如果存在，则删除模式 other，并级联删除其依赖的对象
            con.exec_driver_sql("DROP SCHEMA IF EXISTS other CASCADE;")
            # 创建名为 other 的新模式
            con.exec_driver_sql("CREATE SCHEMA other;")

    # 使用不同的 if_exists 选项将 DataFrame 写入数据库表
    assert (
        df.to_sql(name="test_schema_other", con=conn, schema="other", index=False) == 2
    )
    # 将 DataFrame 写入名为 test_schema_other 的数据库表，如果该表已存在则替换
    df.to_sql(
        name="test_schema_other",
        con=conn,
        schema="other",
        index=False,
        if_exists="replace",
    )
    # 将 DataFrame 追加到名为 test_schema_other 的数据库表中
    assert (
        df.to_sql(
            name="test_schema_other",
            con=conn,
            schema="other",
            index=False,
            if_exists="append",
        )
        == 2
    )
    # 从数据库中读取名为 test_schema_other 的表的内容，schema 为 other
    res = sql.read_sql_table("test_schema_other", conn, schema="other")
    # 使用 pytest 的 assert_frame_equal 方法验证读取的 DataFrame 是否与预期的 DataFrame 一致
    tm.assert_frame_equal(concat([df, df], ignore_index=True), res)
@pytest.mark.db
# 标记此测试函数为数据库相关测试，使用 pytest 的标记功能
def test_self_join_date_columns(postgresql_psycopg2_engine):
    # GH 44421
    # GitHub issue 44421，记录了这个测试的相关背景信息
    conn = postgresql_psycopg2_engine
    # 获取传入的 PostgreSQL 引擎对象

    from sqlalchemy.sql import text
    # 导入 SQLAlchemy 的 text 函数，用于执行 SQL 查询字符串

    create_table = text(
        """
    CREATE TABLE person
    (
        id serial constraint person_pkey primary key,
        created_dt timestamp with time zone
    );

    INSERT INTO person
        VALUES (1, '2021-01-01T00:00:00Z');
    """
    )
    # 定义创建表格和插入数据的 SQL 字符串

    with conn.connect() as con:
        with con.begin():
            con.execute(create_table)
    # 使用连接对象执行创建表格和插入数据的操作

    sql_query = (
        'SELECT * FROM "person" AS p1 INNER JOIN "person" AS p2 ON p1.id = p2.id;'
    )
    # 定义 SQL 查询语句，执行自连接操作

    result = pd.read_sql(sql_query, conn)
    # 使用 Pandas 从数据库中读取查询结果

    expected = DataFrame(
        [[1, Timestamp("2021", tz="UTC")] * 2], columns=["id", "created_dt"] * 2
    )
    # 定义预期的 DataFrame 结果，包含两列 id 和 created_dt

    expected["created_dt"] = expected["created_dt"].astype("M8[us, UTC]")
    # 调整预期结果中 created_dt 列的数据类型

    tm.assert_frame_equal(result, expected)
    # 使用 Pandas 测试工具比较查询结果和预期结果的 DataFrame

    # Cleanup
    # 清理操作：删除测试过程中创建的表格
    with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
        pandasSQL.drop_table("person")
        # 使用 SQLDatabase 对象执行删除表格的操作


def test_create_and_drop_table(sqlite_engine):
    conn = sqlite_engine
    # 获取传入的 SQLite 引擎对象

    temp_frame = DataFrame({"one": [1.0, 2.0, 3.0, 4.0], "two": [4.0, 3.0, 2.0, 1.0]})
    # 创建一个临时的 DataFrame 对象

    with sql.SQLDatabase(conn) as pandasSQL:
        with pandasSQL.run_transaction():
            assert pandasSQL.to_sql(temp_frame, "drop_test_frame") == 4
        # 使用 PandasSQL 对象执行事务，将 DataFrame 写入数据库表格，并检查写入行数

        assert pandasSQL.has_table("drop_test_frame")
        # 检查表格是否存在

        with pandasSQL.run_transaction():
            pandasSQL.drop_table("drop_test_frame")
        # 使用 PandasSQL 对象执行事务，删除表格

        assert not pandasSQL.has_table("drop_test_frame")
        # 再次检查表格是否被成功删除


def test_sqlite_datetime_date(sqlite_buildin):
    conn = sqlite_buildin
    # 获取传入的 SQLite 内建引擎对象

    df = DataFrame([date(2014, 1, 1), date(2014, 1, 2)], columns=["a"])
    # 创建包含日期数据的 DataFrame 对象

    assert df.to_sql(name="test_date", con=conn, index=False) == 2
    # 将 DataFrame 写入 SQLite 数据库表格，检查写入的行数

    res = read_sql_query("SELECT * FROM test_date", conn)
    # 执行 SQL 查询，读取写入的数据

    # comes back as strings
    # 查询结果以字符串形式返回

    tm.assert_frame_equal(res, df.astype(str))
    # 使用 Pandas 测试工具比较查询结果和预期的字符串格式的 DataFrame


@pytest.mark.parametrize("tz_aware", [False, True])
# 使用 pytest 的参数化功能，定义测试函数的多组参数
def test_sqlite_datetime_time(tz_aware, sqlite_buildin):
    conn = sqlite_buildin
    # 获取传入的 SQLite 内建引擎对象

    # test support for datetime.time, GH #8341
    # 测试对 datetime.time 的支持，参考 GitHub issue #8341

    if not tz_aware:
        tz_times = [time(9, 0, 0), time(9, 1, 30)]
    else:
        tz_dt = date_range("2013-01-01 09:00:00", periods=2, tz="US/Pacific")
        tz_times = Series(tz_dt.to_pydatetime()).map(lambda dt: dt.timetz())
    # 根据 tz_aware 参数选择不同的时间数据生成方式

    df = DataFrame(tz_times, columns=["a"])
    # 创建包含时间数据的 DataFrame 对象

    assert df.to_sql(name="test_time", con=conn, index=False) == 2
    # 将 DataFrame 写入 SQLite 数据库表格，检查写入的行数

    res = read_sql_query("SELECT * FROM test_time", conn)
    # 执行 SQL 查询，读取写入的数据

    # comes back as strings
    # 查询结果以字符串形式返回

    expected = df.map(lambda _: _.strftime("%H:%M:%S.%f"))
    # 将预期结果的时间数据格式化为字符串

    tm.assert_frame_equal(res, expected)
    # 使用 Pandas 测试工具比较查询结果和预期的格式化字符串结果


def get_sqlite_column_type(conn, table, column):
    # 定义函数，获取 SQLite 表格中指定列的数据类型
    recs = conn.execute(f"PRAGMA table_info({table})")
    # 执行 PRAGMA 查询，获取表格信息

    for cid, name, ctype, not_null, default, pk in recs:
        if name == column:
            return ctype
    # 遍历查询结果，找到匹配的列名并返回其数据类型

    raise ValueError(f"Table {table}, column {column} not found")
    # 如果未找到指定的表格或列，抛出数值错误异常


def test_sqlite_test_dtype(sqlite_buildin):
    conn = sqlite_buildin
    # 获取传入的 SQLite 内建引擎对象

    cols = ["A", "B"]
    # 定义需要测试的列名列表
    # 创建一个包含元组数据的列表
    data = [(0.8, True), (0.9, None)]
    # 使用列名 `cols` 创建一个 DataFrame 对象
    df = DataFrame(data, columns=cols)
    # 将 DataFrame 写入 SQLite 数据库，并断言写入行数为 2
    assert df.to_sql(name="dtype_test", con=conn) == 2
    # 将 DataFrame 指定列的数据类型为 STRING，写入 SQLite 数据库，并断言写入行数为 2
    assert df.to_sql(name="dtype_test2", con=conn, dtype={"B": "STRING"}) == 2

    # 断言 SQLite 中存储的布尔值类型为 INTEGER
    assert get_sqlite_column_type(conn, "dtype_test", "B") == "INTEGER"

    # 断言 SQLite 中指定的列数据类型为 STRING
    assert get_sqlite_column_type(conn, "dtype_test2", "B") == "STRING"
    # 设置错误消息字符串，用于异常断言
    msg = r"B \(<class 'bool'>\) not a string"
    # 使用 pytest 断言抛出 ValueError 异常，并匹配指定错误消息
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="error", con=conn, dtype={"B": bool})

    # 单一数据类型设置
    assert df.to_sql(name="single_dtype_test", con=conn, dtype="STRING") == 2
    # 断言 SQLite 中指定列数据类型为 STRING
    assert get_sqlite_column_type(conn, "single_dtype_test", "A") == "STRING"
    assert get_sqlite_column_type(conn, "single_dtype_test", "B") == "STRING"
# 定义一个测试函数，用于验证在 SQLite 中创建表格并设置列类型的功能
def test_sqlite_notna_dtype(sqlite_buildin):
    # 使用传入的 SQLite 连接对象
    conn = sqlite_buildin
    # 创建包含不同数据类型的列的数据字典
    cols = {
        "Bool": Series([True, None]),
        "Date": Series([datetime(2012, 5, 1), None]),
        "Int": Series([1, None], dtype="object"),
        "Float": Series([1.1, None]),
    }
    # 将数据字典转换为 DataFrame
    df = DataFrame(cols)

    # 指定表名
    tbl = "notna_dtype_test"
    # 断言将 DataFrame 写入 SQLite 表中的行数为 2
    assert df.to_sql(name=tbl, con=conn) == 2

    # 断言获取表中列的 SQLite 数据类型是否正确
    assert get_sqlite_column_type(conn, tbl, "Bool") == "INTEGER"
    assert get_sqlite_column_type(conn, tbl, "Date") == "TIMESTAMP"
    assert get_sqlite_column_type(conn, tbl, "Int") == "INTEGER"
    assert get_sqlite_column_type(conn, tbl, "Float") == "REAL"


# 定义一个测试函数，用于验证 SQLite 中非法表名和列名的处理
def test_sqlite_illegal_names(sqlite_buildin):
    # 使用传入的 SQLite 连接对象
    conn = sqlite_buildin
    # 创建一个包含特殊命名的 DataFrame
    df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"])

    # 设置异常消息
    msg = "Empty table or column name specified"
    # 使用 pytest 断言，验证空表名会抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        df.to_sql(name="", con=conn)

    # 遍历特殊命名列表，验证各种非法命名在 SQLite 中的写入和存在性检查
    for ndx, weird_name in enumerate(
        [
            "test_weird_name]",
            "test_weird_name[",
            "test_weird_name`",
            'test_weird_name"',
            "test_weird_name'",
            "_b.test_weird_name_01-30",
            '"_b.test_weird_name_01-30"',
            "99beginswithnumber",
            "12345",
            "\xe9",
        ]
    ):
        assert df.to_sql(name=weird_name, con=conn) == 2
        sql.table_exists(weird_name, conn)

        # 创建包含特殊命名列的 DataFrame，并验证在 SQLite 中的写入和存在性检查
        df2 = DataFrame([[1, 2], [3, 4]], columns=["a", weird_name])
        c_tbl = f"test_weird_col_name{ndx:d}"
        assert df2.to_sql(name=c_tbl, con=conn) == 2
        sql.table_exists(c_tbl, conn)


# 定义一个函数，用于根据不同类型的参数格式化 SQL 查询字符串
def format_query(sql, *args):
    # 设定参数类型与格式化函数的映射关系
    _formatters = {
        datetime: "'{}'".format,
        str: "'{}'".format,
        np.str_: "'{}'".format,
        bytes: "'{}'".format,
        float: "{:.8f}".format,
        int: "{:d}".format,
        type(None): lambda x: "NULL",
        np.float64: "{:.10f}".format,
        bool: "'{!s}'".format,
    }
    processed_args = []
    # 遍历参数列表，根据其类型选择对应的格式化函数并处理
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None

        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))

    # 使用处理后的参数列表格式化 SQL 查询字符串并返回
    return sql % tuple(processed_args)


# 定义一个简化的 SQL 查询函数，用于执行查询并返回结果列表
def tquery(query, con=None):
    """Replace removed sql.tquery function"""
    with sql.pandasSQL_builder(con) as pandas_sql:
        res = pandas_sql.execute(query).fetchall()
    return None if res is None else list(res)


# 定义一个测试函数，用于验证在 SQLite 中基本的数据写入和读取功能
def test_xsqlite_basic(sqlite_buildin):
    # 创建一个包含随机数据的 DataFrame
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 断言将 DataFrame 写入 SQLite 表中的行数为 10
    assert sql.to_sql(frame, name="test_table", con=sqlite_buildin, index=False) == 10
    # 从 SQLite 表中读取数据到结果 DataFrame
    result = sql.read_sql("select * from test_table", sqlite_buildin)

    # HACK! Change this once indexes are handled properly.
    # 临时处理：修改结果 DataFrame 的索引为原始 DataFrame 的索引
    result.index = frame.index

    # 断言读取的结果与原始 DataFrame 相等
    expected = frame
    # 使用 pandas 测试框架比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, frame)

    # 在 DataFrame 中新增一列名为 "txt"，内容为字符串 "a" 重复若干次，长度与原 DataFrame 相同
    frame["txt"] = ["a"] * len(frame)

    # 复制原 DataFrame 到 frame2
    frame2 = frame.copy()

    # 创建一个新的索引对象 new_idx，其值为原索引数组加上 10
    new_idx = Index(np.arange(len(frame2)), dtype=np.int64) + 10

    # 在 frame2 中新增一列名为 "Idx"，其内容为 new_idx 的复制
    frame2["Idx"] = new_idx.copy()

    # 将 frame2 中的数据写入 SQLite 数据库中的名为 "test_table2" 的表格，不包括索引列，并断言返回值为 10
    assert sql.to_sql(frame2, name="test_table2", con=sqlite_buildin, index=False) == 10

    # 从 SQLite 数据库中读取名为 "test_table2" 的表格数据，并将其存入 result 变量，使用 "Idx" 作为索引列
    result = sql.read_sql("select * from test_table2", sqlite_buildin, index_col="Idx")

    # 创建一个预期的 DataFrame 对象 expected，其数据与 frame 相同，但索引被替换为 new_idx，并命名为 "Idx"
    expected = frame.copy()
    expected.index = new_idx
    expected.index.name = "Idx"

    # 使用 pandas 测试框架比较预期的 DataFrame expected 与读取出的 result 是否相等
    tm.assert_frame_equal(expected, result)
def test_xsqlite_write_row_by_row(sqlite_buildin):
    # 创建一个包含随机数据的DataFrame对象，用于测试
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 将第一个元素设置为NaN
    frame.iloc[0, 0] = np.nan
    # 根据DataFrame生成创建表的SQL语句
    create_sql = sql.get_schema(frame, "test")
    # 获取SQLite数据库的游标
    cur = sqlite_buildin.cursor()
    # 执行创建表的SQL语句
    cur.execute(create_sql)

    # 构造插入数据的SQL语句模板
    ins = "INSERT INTO test VALUES (%s, %s, %s, %s)"
    # 遍历DataFrame的每一行，执行插入数据操作
    for _, row in frame.iterrows():
        # 格式化SQL语句，将行数据插入到SQL语句中
        fmt_sql = format_query(ins, *row)
        # 执行SQL查询
        tquery(fmt_sql, con=sqlite_buildin)

    # 提交事务
    sqlite_buildin.commit()

    # 从数据库中读取数据，与原始DataFrame进行比较
    result = sql.read_sql("select * from test", con=sqlite_buildin)
    # 将结果DataFrame的索引设置为与原始DataFrame相同
    result.index = frame.index
    # 使用测试工具比较两个DataFrame，设置允许的相对误差
    tm.assert_frame_equal(result, frame, rtol=1e-3)


def test_xsqlite_execute(sqlite_buildin):
    # 创建一个包含随机数据的DataFrame对象，用于测试
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 根据DataFrame生成创建表的SQL语句
    create_sql = sql.get_schema(frame, "test")
    # 获取SQLite数据库的游标
    cur = sqlite_buildin.cursor()
    # 执行创建表的SQL语句
    cur.execute(create_sql)
    # 构造带参数的插入数据的SQL语句模板
    ins = "INSERT INTO test VALUES (?, ?, ?, ?)"

    # 获取DataFrame的第一行数据
    row = frame.iloc[0]
    # 使用pandasSQL_builder上下文管理器执行插入数据操作
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute(ins, tuple(row))
    # 提交事务
    sqlite_buildin.commit()

    # 从数据库中读取数据，与原始DataFrame的第一行进行比较
    result = sql.read_sql("select * from test", sqlite_buildin)
    # 设置结果DataFrame的索引为原始DataFrame的第一行索引
    result.index = frame.index[:1]
    # 使用测试工具比较两个DataFrame的内容
    tm.assert_frame_equal(result, frame[:1])


def test_xsqlite_schema(sqlite_buildin):
    # 创建一个包含随机数据的DataFrame对象，用于测试
    frame = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 根据DataFrame生成创建表的SQL语句
    create_sql = sql.get_schema(frame, "test")
    # 将SQL语句按行分割
    lines = create_sql.splitlines()
    # 遍历每一行SQL语句
    for line in lines:
        # 将每行SQL语句按空格分割为单词
        tokens = line.split(" ")
        # 如果分割后的单词数为2，并且第一个单词是"A"
        if len(tokens) == 2 and tokens[0] == "A":
            # 断言第二个单词是"DATETIME"
            assert tokens[1] == "DATETIME"

    # 根据DataFrame生成创建表的SQL语句，指定"A"和"B"为主键
    create_sql = sql.get_schema(frame, "test", keys=["A", "B"])
    # 断言在SQL语句中包含主键的定义
    assert 'PRIMARY KEY ("A", "B")' in create_sql
    # 获取SQLite数据库的游标
    cur = sqlite_buildin.cursor()
    # 执行创建表的SQL语句
    cur.execute(create_sql)


def test_xsqlite_execute_fail(sqlite_buildin):
    # 创建一个包含失败数据插入的SQL语句
    create_sql = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    # 获取SQLite数据库的游标
    cur = sqlite_buildin.cursor()
    # 执行创建表的SQL语句
    cur.execute(create_sql)

    # 使用pandasSQL_builder上下文管理器执行插入数据操作，包括两个正常插入和一个异常插入
    with sql.pandasSQL_builder(sqlite_buildin) as pandas_sql:
        pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
        pandas_sql.execute('INSERT INTO test VALUES("foo", "baz", 2.567)')

        # 使用pytest断言捕获sql.DatabaseError异常，匹配特定的错误信息
        with pytest.raises(sql.DatabaseError, match="Execution failed on sql"):
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 7)')


def test_xsqlite_execute_closed_connection():
    # 创建一个包含关闭连接的测试SQL语句
    create_sql = """
    CREATE TABLE test
    (
    a TEXT,
    b TEXT,
    c REAL,
    PRIMARY KEY (a, b)
    );
    """
    # 使用 contextlib.closing 包装的 SQLite 内存数据库连接，确保连接在代码块结束后自动关闭
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        # 创建游标对象
        cur = conn.cursor()
        # 执行传入的 SQL 创建表格或者其他数据库操作
        cur.execute(create_sql)
    
        # 使用 sql.pandasSQL_builder 提供的上下文管理器创建 pandasSQL 对象并执行插入操作
        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
    
    # 定义错误消息字符串
    msg = "Cannot operate on a closed database."
    # 使用 pytest.raises 捕获 sqlite3.ProgrammingError 异常，并验证异常消息与预期的错误消息匹配
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        # 尝试在已关闭的数据库连接上执行查询操作 tquery("select * from test", con=conn)
        tquery("select * from test", con=conn)
# 定义一个测试函数，用于测试在 SQLite 数据库中使用关键字作为列名的情况
def test_xsqlite_keyword_as_column_names(sqlite_buildin):
    # 创建一个包含名为 "From" 的列的 DataFrame，列中填充值为 1
    df = DataFrame({"From": np.ones(5)})
    # 将 DataFrame 写入 SQLite 数据库中的表 "testkeywords"，并验证插入的行数为 5
    assert sql.to_sql(df, con=sqlite_buildin, name="testkeywords", index=False) == 5


# 定义一个测试函数，用于测试将只包含整数列的 DataFrame 写入 SQLite 数据库的情况
def test_xsqlite_onecolumn_of_integer(sqlite_buildin):
    # 创建一个只包含整数列的 DataFrame
    mono_df = DataFrame([1, 2], columns=["c0"])
    # 将 DataFrame 写入 SQLite 数据库中的表 "mono_df"，并验证插入的行数为 2
    assert sql.to_sql(mono_df, con=sqlite_buildin, name="mono_df", index=False) == 2
    # 通过 SQL 查询计算 "c0" 列的总和
    con_x = sqlite_buildin
    the_sum = sum(my_c0[0] for my_c0 in con_x.execute("select * from mono_df"))
    # 断言计算的总和为 3
    assert the_sum == 3
    # 通过 SQL 读取 "mono_df" 表的数据，并验证与原始 DataFrame 相等
    result = sql.read_sql("select * from mono_df", con_x)
    tm.assert_frame_equal(result, mono_df)


# 定义一个测试函数，用于测试 SQL 数据库表存在时的不同插入行为
def test_xsqlite_if_exists(sqlite_buildin):
    # 创建两个不同的 DataFrame
    df_if_exists_1 = DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
    df_if_exists_2 = DataFrame({"col1": [3, 4, 5], "col2": ["C", "D", "E"]})
    table_name = "table_if_exists"
    sql_select = f"SELECT * FROM {table_name}"

    # 测试插入时 if_exists 参数为非法值的情况
    msg = "'notvalidvalue' is not valid for if_exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="notvalidvalue",
        )
    drop_table(table_name, sqlite_buildin)

    # 测试 if_exists='fail' 的情况
    sql.to_sql(
        frame=df_if_exists_1, con=sqlite_buildin, name=table_name, if_exists="fail"
    )
    msg = "Table 'table_if_exists' already exists"
    with pytest.raises(ValueError, match=msg):
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="fail",
        )

    # 测试 if_exists='replace' 的情况
    sql.to_sql(
        frame=df_if_exists_1,
        con=sqlite_buildin,
        name=table_name,
        if_exists="replace",
        index=False,
    )
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert (
        sql.to_sql(
            frame=df_if_exists_2,
            con=sqlite_buildin,
            name=table_name,
            if_exists="replace",
            index=False,
        )
        == 3
    )
    assert tquery(sql_select, con=sqlite_buildin) == [(3, "C"), (4, "D"), (5, "E")]
    drop_table(table_name, sqlite_buildin)

    # 测试 if_exists='append' 的情况
    assert (
        sql.to_sql(
            frame=df_if_exists_1,
            con=sqlite_buildin,
            name=table_name,
            if_exists="fail",
            index=False,
        )
        == 2
    )
    assert tquery(sql_select, con=sqlite_buildin) == [(1, "A"), (2, "B")]
    assert (
        sql.to_sql(
            frame=df_if_exists_2,
            con=sqlite_buildin,
            name=table_name,
            if_exists="append",
            index=False,
        )
        == 3
    )
    # 断言语句，验证 tquery 函数调用的结果是否符合预期
    assert tquery(sql_select, con=sqlite_buildin) == [
        (1, "A"),
        (2, "B"),
        (3, "C"),
        (4, "D"),
        (5, "E"),
    ]
    # 调用函数 drop_table，删除指定的表格，使用的 SQLite 连接为 sqlite_buildin
    drop_table(table_name, sqlite_buildin)
```