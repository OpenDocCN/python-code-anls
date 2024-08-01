# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_reader.py`

```py
import logging  # 导入日志模块
import os  # 导入操作系统功能模块

import chardet  # 导入字符编码检测模块
import duckdb  # 导入DuckDB数据库模块
import numpy as np  # 导入NumPy科学计算库
import pandas as pd  # 导入Pandas数据分析库
import sqlparse  # 导入SQL解析器
from pyparsing import (  # 导入pyparsing库的若干功能
    CaselessKeyword,
    Forward,
    Literal,
    Optional,
    Regex,
    Word,
    alphanums,
    delimitedList,
)

from dbgpt.util.pd_utils import csv_colunm_foramt  # 导入自定义的Pandas工具函数
from dbgpt.util.string_utils import is_chinese_include_number  # 导入自定义的字符串处理函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def excel_colunm_format(old_name: str) -> str:
    new_column = old_name.strip()  # 去除字符串首尾空格
    new_column = new_column.replace(" ", "_")  # 将空格替换为下划线
    return new_column  # 返回格式化后的列名


def detect_encoding(file_path):
    # 读取文件的二进制数据
    with open(file_path, "rb") as f:
        data = f.read()
    # 使用 chardet 来检测文件编码
    result = chardet.detect(data)
    encoding = result["encoding"]  # 获取检测到的编码格式
    confidence = result["confidence"]  # 获取检测到的编码置信度
    return encoding, confidence  # 返回编码格式和置信度


def add_quotes_ex(sql: str, column_names):
    sql = sql.replace("`", '"')  # 将反引号替换为双引号
    for column_name in column_names:
        if sql.find(column_name) != -1 and sql.find(f'"{column_name}"') == -1:
            sql = sql.replace(column_name, f'"{column_name}"')  # 在 SQL 语句中添加双引号
    return sql  # 返回添加双引号后的 SQL 语句


def parse_sql(sql):
    # 定义关键字和标识符
    select_stmt = Forward()
    column = Regex(r"[\w一-龥]*")  # 匹配字母、数字和中文字符
    table = Word(alphanums)  # 匹配表名
    join_expr = Forward()
    where_expr = Forward()
    group_by_expr = Forward()
    order_by_expr = Forward()

    select_keyword = CaselessKeyword("SELECT")  # 匹配 SELECT 关键字
    from_keyword = CaselessKeyword("FROM")  # 匹配 FROM 关键字
    join_keyword = CaselessKeyword("JOIN")  # 匹配 JOIN 关键字
    on_keyword = CaselessKeyword("ON")  # 匹配 ON 关键字
    where_keyword = CaselessKeyword("WHERE")  # 匹配 WHERE 关键字
    group_by_keyword = CaselessKeyword("GROUP BY")  # 匹配 GROUP BY 关键字
    order_by_keyword = CaselessKeyword("ORDER BY")  # 匹配 ORDER BY 关键字
    and_keyword = CaselessKeyword("AND")  # 匹配 AND 关键字
    or_keyword = CaselessKeyword("OR")  # 匹配 OR 关键字
    in_keyword = CaselessKeyword("IN")  # 匹配 IN 关键字
    not_in_keyword = CaselessKeyword("NOT IN")  # 匹配 NOT IN 关键字

    # 定义语法规则
    select_stmt <<= (
        select_keyword
        + delimitedList(column)  # 使用逗号分隔的列名列表
        + from_keyword
        + delimitedList(table)  # 使用逗号分隔的表名列表
        + Optional(join_expr)
        + Optional(where_keyword + where_expr)
        + Optional(group_by_keyword + group_by_expr)
        + Optional(order_by_keyword + order_by_expr)
    )

    join_expr <<= join_keyword + table + on_keyword + column + Literal("=") + column

    where_expr <<= (
        column + Literal("=") + Word(alphanums) + Optional(and_keyword + where_expr)
        | column + Literal(">") + Word(alphanums) + Optional(and_keyword + where_expr)
        | column + Literal("<") + Word(alphanums) + Optional(and_keyword + where_expr)
    )

    group_by_expr <<= delimitedList(column)  # 使用逗号分隔的分组列名列表

    order_by_expr <<= column + Optional(Literal("ASC") | Literal("DESC"))  # 匹配列名及可选的升序或降序

    # 解析 SQL 语句
    parsed_result = select_stmt.parseString(sql)

    return parsed_result.asList()  # 返回解析后的 SQL 语句列表形式


def add_quotes(sql, column_names=[]):
    sql = sql.replace("`", "")  # 删除 SQL 语句中的反引号
    sql = sql.replace("'", "")  # 删除 SQL 语句中的单引号
    parsed = sqlparse.parse(sql)  # 解析 SQL 语句
    for stmt in parsed:
        for token in stmt.tokens:
            deep_quotes(token, column_names)  # 对 SQL 语句中的标记进行深度添加引号处理
    return str(parsed[0])


# 将解析结果的第一个元素转换为字符串并返回
def deep_quotes(token, column_names=[]):
    # 如果 token 对象具有 "tokens" 属性，说明它包含子 token
    if hasattr(token, "tokens"):
        # 遍历 token 的子 token
        for token_child in token.tokens:
            # 递归调用 deep_quotes() 处理子 token
            deep_quotes(token_child, column_names)
    else:
        # 如果 token 是中文（包含数字），则添加双引号，并更新 token 的值
        if is_chinese_include_number(token.value):
            new_value = token.value.replace("`", "").replace("'", "")
            token.value = f'"{new_value}"'


def get_select_clause(sql):
    parsed = sqlparse.parse(sql)[0]  # 解析 SQL 语句，获取第一个语句块

    select_tokens = []
    is_select = False

    for token in parsed.tokens:
        # 如果当前 token 是关键字且值为 SELECT（不区分大小写），则标记为 SELECT 子句开始
        if token.is_keyword and token.value.upper() == "SELECT":
            is_select = True
        elif is_select:
            # 如果当前 token 是关键字且值为 FROM，结束 SELECT 子句解析
            if token.is_keyword and token.value.upper() == "FROM":
                break
            # 将 SELECT 子句中的 token 添加到 select_tokens 列表中
            select_tokens.append(token)
    # 将 select_tokens 中的 token 转换为字符串并返回
    return "".join(str(token) for token in select_tokens)


def parse_select_fields(sql):
    parsed = sqlparse.parse(sql)[0]  # 解析 SQL 语句，获取第一个语句块
    fields = []

    for token in parsed.tokens:
        # 如果 token 是单引号包裹的字符串，将其扁平化处理
        if token.match(sqlparse.tokens.Literal.String.Single):
            token.flatten()
        # 如果 token 是标识符（字段名），获取其真实名称并添加到 fields 列表中
        if isinstance(token, sqlparse.sql.Identifier):
            fields.append(token.get_real_name())

    # 对于 fields 中的中文字段名，添加双引号
    fields = [field.replace(f"field", f'"{field}"') for field in fields]

    return fields


def add_quotes_to_chinese_columns(sql, column_names=[]):
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        # 对 SQL 语句的每个语句块进行处理，添加双引号到中文字段名
        process_statement(stmt, column_names)
    # 将处理后的 SQL 语句转换为字符串并返回（假设 parsed 至少有一个语句块）
    return str(parsed[0])


def process_statement(statement, column_names=[]):
    if isinstance(statement, sqlparse.sql.IdentifierList):
        # 如果 statement 是标识符列表，遍历其中的每个标识符
        for identifier in statement.get_identifiers():
            # 处理每个标识符
            process_identifier(identifier)
    elif isinstance(statement, sqlparse.sql.Identifier):
        # 如果 statement 是单个标识符，处理该标识符
        process_identifier(statement, column_names)
    elif isinstance(statement, sqlparse.sql.TokenList):
        # 如果 statement 是 TokenList，递归处理其中的每个 token
        for item in statement.tokens:
            process_statement(item)


def process_identifier(identifier, column_names=[]):
    # 如果 identifier 具有别名，获取其别名并修改 token 值（已注释部分）
    # 如果 identifier 的值在 column_names 中且是中文，处理其值
    if hasattr(identifier, "tokens") and identifier.value in column_names:
        if is_chinese(identifier.value):
            new_value = get_new_value(identifier.value)
            # 更新 identifier 的值和相关属性
            identifier.value = new_value
            identifier.normalized = new_value
            identifier.tokens = [sqlparse.sql.Token(sqlparse.tokens.Name, new_value)]
    # 如果条件不满足，则执行以下代码块
    else:
        # 如果 identifier 对象具有 "tokens" 属性，则进入循环遍历其 tokens
        if hasattr(identifier, "tokens"):
            # 遍历 identifier 的 tokens
            for token in identifier.tokens:
                # 如果 token 是一个 SQL 函数对象，则调用 process_function 函数处理它
                if isinstance(token, sqlparse.sql.Function):
                    process_function(token)
                # 如果 token 的类型属于 SQL 的名称类型
                elif token.ttype in sqlparse.tokens.Name:
                    # 获取 token 的新值
                    new_value = get_new_value(token.value)
                    # 更新 token 的值和规范化后的值
                    token.value = new_value
                    token.normalized = new_value
                # 如果 token 的值在 column_names 中
                elif token.value in column_names:
                    # 获取 token 的新值
                    new_value = get_new_value(token.value)
                    # 更新 token 的值和规范化后的值
                    token.value = new_value
                    token.normalized = new_value
                    # 更新 token 的 tokens 属性，将其重置为包含新值的 Name 类型的 token
                    token.tokens = [sqlparse.sql.Token(sqlparse.tokens.Name, new_value)]
def get_new_value(value):
    # 返回处理后的新字符串，将所有的反引号、单引号和双引号替换为空字符串，并在两侧加上双引号
    return f""" "{value.replace("`", "").replace("'", "").replace('"', "")}" """


def process_function(function):
    # 获取函数的所有参数列表
    function_params = list(function.get_parameters())
    # 遍历函数参数列表的索引
    for i in range(len(function_params)):
        # 获取当前参数对象
        param = function_params[i]
        # 如果参数是一个标识符（字段名）
        if isinstance(param, sqlparse.sql.Identifier):
            # 判断参数值是否包含中文字符
            # if is_chinese(param.value):
            # 调用函数获取处理后的新字段值
            new_value = get_new_value(param.value)
            # 替换原参数的值为新值的 tokens
            function_params[i].tokens = [
                sqlparse.sql.Token(sqlparse.tokens.Name, new_value)
            ]
    # 打印处理后的函数字符串表示
    print(str(function))


def is_chinese(text):
    # 检查文本中是否包含任何中文字符
    for char in text:
        if "一" <= char <= "鿿":
            return True
    return False


class ExcelReader:
    # 空白，未添加注释部分
    # 构造函数初始化，接受文件路径作为参数
    def __init__(self, file_path):
        # 获取文件名（带扩展名）
        file_name = os.path.basename(file_path)
        # 获取不带扩展名的文件名
        self.file_name_without_extension = os.path.splitext(file_name)[0]
        # 检测文件编码
        encoding, confidence = detect_encoding(file_path)
        # 记录日志，显示检测到的编码和置信度
        logger.info(f"Detected Encoding: {encoding} (Confidence: {confidence})")
        # 记录文件名和文件扩展名
        self.excel_file_name = file_name
        self.extension = os.path.splitext(file_name)[1]

        # 读取 Excel 文件
        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            # 读取 Excel 文件内容到 DataFrame，不指定索引列
            df_tmp = pd.read_excel(file_path, index_col=False)
            # 使用指定的转换器读取 Excel 文件内容到 DataFrame
            self.df = pd.read_excel(
                file_path,
                index_col=False,
                converters={i: csv_colunm_foramt for i in range(df_tmp.shape[1])},
            )
        elif file_path.endswith(".csv"):
            # 读取 CSV 文件内容到 DataFrame，指定文件编码
            df_tmp = pd.read_csv(file_path, index_col=False, encoding=encoding)
            # 使用指定的转换器读取 CSV 文件内容到 DataFrame
            self.df = pd.read_csv(
                file_path,
                index_col=False,
                encoding=encoding,
                # csv_colunm_foramt 可以修改更多，只是针对美元人民币符号，假如是“你好¥¥¥”则会报错！
                converters={i: csv_colunm_foramt for i in range(df_tmp.shape[1])},
            )
        else:
            # 抛出异常，不支持的文件格式
            raise ValueError("Unsupported file format.")

        # 将 DataFrame 中的空字符串替换为 NaN
        self.df.replace("", np.nan, inplace=True)

        # 删除 DataFrame 中以 "Unnamed" 开头且全为空值的列
        unnamed_columns_tmp = [
            col
            for col in df_tmp.columns
            if col.startswith("Unnamed") and df_tmp[col].isnull().all()
        ]
        df_tmp.drop(columns=unnamed_columns_tmp, inplace=True)

        # 更新 self.df，只保留与 df_tmp 列名相同的列
        self.df = self.df[df_tmp.columns.values]

        # 初始化列名映射字典
        self.columns_map = {}
        # 遍历 DataFrame 的列名
        for column_name in df_tmp.columns:
            # 将 DataFrame 列转换为字符串类型
            self.df[column_name] = self.df[column_name].astype(str)
            # 使用 excel_colunm_format 函数更新列名映射字典
            self.columns_map.update({column_name: excel_colunm_format(column_name)})
            try:
                # 尝试将列内容转换为日期字符串（格式：%Y-%m-%d）
                self.df[column_name] = pd.to_datetime(self.df[column_name]).dt.strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                try:
                    # 尝试将列内容转换为数值类型
                    self.df[column_name] = pd.to_numeric(self.df[column_name])
                except ValueError:
                    try:
                        # 若转换失败，保持列内容为字符串类型
                        self.df[column_name] = self.df[column_name].astype(str)
                    except Exception:
                        # 捕获并打印无法转换的列名
                        print("Can't transform column: " + column_name)

        # 重命名 DataFrame 列名，去除空格并替换为下划线
        self.df = self.df.rename(columns=lambda x: x.strip().replace(" ", "_"))

        # 连接 DuckDB 内存数据库
        self.db = duckdb.connect(database=":memory:", read_only=False)

        # 指定表名
        self.table_name = "excel_data"

        # 在 DuckDB 中注册 DataFrame
        self.db.register(self.table_name, self.df)

        # 执行 SQL 查询获取表结构信息并打印
        result = self.db.execute(f"DESCRIBE {self.table_name}")
        columns = result.fetchall()
        for column in columns:
            print(column)
    # 定义一个方法 `run`，用于执行传入的 SQL 查询语句并返回结果
    def run(self, sql):
        try:
            # 如果 SQL 语句中包含表名的双引号，则将其替换为不带引号的表名
            if f'"{self.table_name}"' in sql:
                sql = sql.replace(f'"{self.table_name}"', self.table_name)
            # 对 SQL 语句中的中文列名添加引号
            sql = add_quotes_to_chinese_columns(sql)
            # 打印执行的 SQL 语句
            print(f"excute sql:{sql}")
            # 使用数据库连接对象执行 SQL 语句，并获取查询结果
            results = self.db.execute(sql)
            # 提取查询结果的列名列表
            colunms = []
            for descrip in results.description:
                colunms.append(descrip[0])
            # 返回列名列表和查询结果的所有数据行
            return colunms, results.fetchall()
        except Exception as e:
            # 如果发生异常，记录错误日志并抛出带有详细错误信息的 ValueError 异常
            logger.error(f"excel sql run error!, {str(e)}")
            raise ValueError(f"Data Query Exception!\\nSQL[{sql}].\\nError:{str(e)}")

    # 定义一个方法 `get_df_by_sql_ex`，根据传入的 SQL 查询语句获取并返回一个 Pandas DataFrame
    def get_df_by_sql_ex(self, sql):
        # 调用 `run` 方法执行 SQL 查询，获取列名和数据值
        colunms, values = self.run(sql)
        # 使用查询结果创建一个 Pandas DataFrame，并指定列名
        return pd.DataFrame(values, columns=colunms)

    # 定义一个方法 `get_sample_data`，用于获取表中的前5行数据作为示例
    def get_sample_data(self):
        # 调用 `run` 方法执行指定的 SQL 查询，查询前5行数据并返回
        return self.run(f"SELECT * FROM {self.table_name} LIMIT 5;")
```