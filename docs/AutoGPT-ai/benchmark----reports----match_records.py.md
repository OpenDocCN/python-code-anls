# `.\AutoGPT\benchmark\reports\match_records.py`

```py
# 导入所需的库
import glob
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from pydantic import BaseModel, Field

# 定义 Metrics 类，包含一些指标字段
class Metrics(BaseModel):
    difficulty: str
    success: bool
    success_percent: float = Field(..., alias="success_%")
    run_time: Optional[str] = None
    fail_reason: Optional[str] = None
    attempted: Optional[bool] = None

# 定义 MetricsOverall 类，包含一些总体指标字段
class MetricsOverall(BaseModel):
    run_time: str
    highest_difficulty: str
    percentage: Optional[float] = None

# 定义 Test 类，包含一些测试相关字段
class Test(BaseModel):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None

# 定义 SuiteTest 类，包含一些测试套件相关字段
class SuiteTest(BaseModel):
    data_path: str
    metrics: MetricsOverall
    tests: Dict[str, Test]
    category: Optional[List[str]] = None
    task: Optional[str] = None
    reached_cutoff: Optional[bool] = None

# 定义 Report 类，包含一些报告相关字段
class Report(BaseModel):
    command: str
    completion_time: str
    benchmark_start_time: str
    metrics: MetricsOverall
    tests: Dict[str, Union[Test, SuiteTest]]
    config: Dict[str, str | dict[str, str]]

# 定义函数 get_reports，用于获取报告数据
def get_reports():
    # 初始化一个空列表来存储报告数据
    report_data = []

    # 获取当前工作目录
    current_dir = os.getcwd()

    # 检查当前目录是否以'reports'结尾
    if current_dir.endswith("reports"):
        reports_dir = "/"
    else:
        reports_dir = "reports"

    # 遍历报告目录中的所有代理目录
    return pd.DataFrame(report_data)

# 定义函数 get_helicone_data，用于获取 Helicone 数据
def get_helicone_data():
    # 获取 Helicone API 密钥
    helicone_api_key = os.getenv("HELICONE_API_KEY")

    # 定义 Helicone API 的 URL
    url = "https://www.helicone.ai/api/graphql"
    # 请用您的个人访问密钥替换<KEY>
    # 创建基于AIOHTTP的传输对象，指定URL和授权头信息
    transport = AIOHTTPTransport(
        url=url, headers={"authorization": f"Bearer {helicone_api_key}"}
    )

    # 创建GraphQL客户端对象，使用指定的传输对象，并从传输对象中获取模式
    client = Client(transport=transport, fetch_schema_from_transport=True)

    # 设置每次请求获取的数据量
    SIZE = 250

    # 初始化数据索引
    i = 0

    # 初始化数据列表
    data = []

    # 打印提示信息
    print("Fetching data from Helicone")
    # 无限循环，用于不断查询数据
    while True:
        # 定义 GraphQL 查询语句
        query = gql(
            """
            query ExampleQuery($limit: Int, $offset: Int){
                heliconeRequest(
                    limit: $limit
                    offset: $offset
                ) {
                    costUSD
                    prompt
                    properties{
                        name
                        value
                    }
                    
                    requestBody
                    response
                    createdAt

                }

                }
        """
        )
        # 打印当前查询的记录范围
        print(f"Fetching {i * SIZE} to {(i + 1) * SIZE} records")
        try:
            # 执行查询并获取结果
            result = client.execute(
                query, variable_values={"limit": SIZE, "offset": i * SIZE}
            )
        except Exception as e:
            # 捕获异常并打印错误信息
            print(f"Error occurred: {e}")
            result = None

        # 更新查询偏移量
        i += 1

        # 如果有结果，则处理每个查询结果
        if result:
            for item in result["heliconeRequest"]:
                # 从每个查询结果中提取属性信息
                properties = {
                    prop["name"]: prop["value"] for prop in item["properties"]
                }
                # 将提取的数据添加到列表中
                data.append(
                    {
                        "createdAt": item["createdAt"],
                        "agent": properties.get("agent"),
                        "costUSD": item["costUSD"],
                        "job_id": properties.get("job_id"),
                        "challenge": properties.get("challenge"),
                        "benchmark_start_time": properties.get("benchmark_start_time"),
                        "prompt": item["prompt"],
                        "response": item["response"],
                        "model": item["requestBody"].get("model"),
                        "request": item["requestBody"].get("messages"),
                    }
                )

        # 如果没有结果或者查询结果为空，则结束循环
        if not result or (len(result["heliconeRequest"]) == 0):
            print("No more results")
            break

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    # 删除 agent 为 None 的行
    # 删除 agent 列中包含缺失值的行
    df = df.dropna(subset=["agent"])

    # 将 agent 列中的所有值转换为小写
    df["agent"] = df["agent"].str.lower()

    # 返回处理后的数据框
    return df
# 检查是否存在名为"raw_reports.pkl"和"raw_helicone.pkl"的文件，如果存在则读取数据，否则调用相应函数生成数据并保存到文件
if os.path.exists("raw_reports.pkl") and os.path.exists("raw_helicone.pkl"):
    reports_df = pd.read_pickle("raw_reports.pkl")
    helicone_df = pd.read_pickle("raw_helicone.pkl")
else:
    reports_df = get_reports()
    reports_df.to_pickle("raw_reports.pkl")
    helicone_df = get_helicone_data()
    helicone_df.to_pickle("raw_helicone.pkl")

# 定义一个函数用于尝试不同的日期格式转换
def try_formats(date_str):
    formats = ["%Y-%m-%d-%H:%M", "%Y-%m-%dT%H:%M:%S%z"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            pass
    return None

# 尝试将"benchmark_start_time"列的值转换为日期时间格式，并设定为UTC时区
helicone_df["benchmark_start_time"] = pd.to_datetime(
    helicone_df["benchmark_start_time"].apply(try_formats), utc=True
)
# 删除包含缺失值的行
helicone_df = helicone_df.dropna(subset=["benchmark_start_time"])
# 将"createdAt"列的值转换为日期时间格式，单位为毫秒，起始时间为UNIX纪元
helicone_df["createdAt"] = pd.to_datetime(
    helicone_df["createdAt"], unit="ms", origin="unix"
)
# 尝试将"benchmark_start_time"列的值转换为日期时间格式，并设定为UTC时区
reports_df["benchmark_start_time"] = pd.to_datetime(
    reports_df["benchmark_start_time"].apply(try_formats), utc=True
)
# 删除包含缺失值的行
reports_df = reports_df.dropna(subset=["benchmark_start_time"])

# 断言"benchmark_start_time"列的数据类型为日期时间格式，否则抛出异常
assert pd.api.types.is_datetime64_any_dtype(
    helicone_df["benchmark_start_time"]
), "benchmark_start_time in helicone_df is not datetime"
assert pd.api.types.is_datetime64_any_dtype(
    reports_df["benchmark_start_time"]
), "benchmark_start_time in reports_df is not datetime"

# 将"report_time"列的值设定为"benchmark_start_time"列的值
reports_df["report_time"] = reports_df["benchmark_start_time"]

# 合并两个数据框，根据"benchmark_start_time"、"agent"和"challenge"列进行内连接
df = pd.merge(
    helicone_df,
    reports_df,
    on=["benchmark_start_time", "agent", "challenge"],
    how="inner",
)

# 将合并后的数据框保存为"df.pkl"文件
df.to_pickle("df.pkl")
# 打印数据框的信息
print(df.info())
# 打印提示信息
print("Data saved to df.pkl")
print("To load the data use: df = pd.read_pickle('df.pkl')")
```