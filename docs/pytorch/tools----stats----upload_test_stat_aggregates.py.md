# `.\pytorch\tools\stats\upload_test_stat_aggregates.py`

```
# 引入未来版本兼容性，允许在旧版本 Python 中使用新特性注解
from __future__ import annotations

# 引入解析命令行参数的模块
import argparse
# 引入 AST（抽象语法树）模块，用于分析和操作 Python 代码的语法树
import ast
# 引入处理日期和时间的模块
import datetime
# 引入处理 JSON 数据的模块
import json
# 引入操作系统相关的功能模块
import os
# 引入正则表达式模块
import re
# 引入 Any 类型，表示任意类型
from typing import Any

# 引入 Rockset 数据库客户端库
import rockset  # type: ignore[import]

# 从本地工具包导入上传统计信息到 S3 的函数
from tools.stats.upload_stats_lib import upload_to_s3


# 定义从测试文件中获取责任人列表的函数
def get_oncall_from_testfile(testfile: str) -> list[str] | None:
    # 构建测试文件的路径
    path = f"test/{testfile}"
    # 如果路径不是以 .py 结尾，则加上 .py 后缀
    if not path.endswith(".py"):
        path += ".py"
    # 尝试打开测试文件并逐行读取
    try:
        with open(path) as f:
            for line in f:
                # 如果某行以 "# Owner(s): " 开头
                if line.startswith("# Owner(s): "):
                    # 使用正则表达式找到括号内的列表形式的字符串
                    possible_lists = re.findall(r"\[.*\]", line)
                    # 如果找到多于一个列表，则抛出异常
                    if len(possible_lists) > 1:
                        raise Exception("More than one list found")  # noqa: TRY002
                    # 如果找到的列表数为零，则抛出异常
                    elif len(possible_lists) == 0:
                        raise Exception(  # noqa: TRY002
                            "No oncalls found or file is badly formatted"
                        )  # noqa: TRY002
                    # 将找到的字符串形式的列表转换为实际列表对象
                    oncalls = ast.literal_eval(possible_lists[0])
                    # 返回责任人列表
                    return list(oncalls)
    # 捕获任何异常，处理文件名中带有点号的情况，返回模块名作为责任人列表
    except Exception as e:
        if "." in testfile:
            return [f"module: {testfile.split('.')[0]}"]
        else:
            return ["module: unmarked"]
    # 如果以上尝试失败，则返回空值
    return None


# 定义获取测试统计聚合数据的函数
def get_test_stat_aggregates(date: datetime.date) -> Any:
    # 使用环境变量中的 Rockset API 密钥初始化 Rockset 客户端
    rockset_api_key = os.environ["ROCKSET_API_KEY"]
    rockset_api_server = "api.rs2.usw2.rockset.com"
    iso_date = date.isoformat()
    # 使用指定的 API 密钥和服务器地址创建 Rockset 客户端
    rs = rockset.RocksetClient(host="api.usw2a1.rockset.com", api_key=rockset_api_key)

    # 定义 Rockset 集合和 Lambda 函数的名称
    collection_name = "commons"
    lambda_function_name = "test_insights_per_daily_upload"
    # 定义查询参数，设置查询 Lambda 函数所需的开始时间参数
    query_parameters = [
        rockset.models.QueryParameter(name="startTime", type="string", value=iso_date)
    ]
    # 执行查询 Lambda 函数，获取 API 响应
    api_response = rs.QueryLambdas.execute_query_lambda(
        query_lambda=lambda_function_name,
        version="692684fa5b37177f",
        parameters=query_parameters,
    )
    
    # 遍历 API 响应中的结果列表
    for i in range(len(api_response["results"])):
        # 获取测试文件名对应的责任人列表
        oncalls = get_oncall_from_testfile(api_response["results"][i]["test_file"])
        # 将责任人列表添加到 API 响应中的结果对象中
        api_response["results"][i]["oncalls"] = oncalls
    
    # 将 API 响应结果转换为 JSON 格式，并返回
    return json.loads(
        json.dumps(api_response["results"], indent=4, sort_keys=True, default=str)
    )


# 主程序入口，解析命令行参数并调用获取测试统计聚合数据的函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload test stat aggregates to Rockset."
    )
    # 添加日期参数的命令行选项
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        help="Date to upload test stat aggregates for (YYYY-MM-DD). Must be in the last 30 days",
        required=True,
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 检查日期参数是否在最近 30 天内，否则抛出数值错误异常
    if args.date < datetime.datetime.now().date() - datetime.timedelta(days=30):
        raise ValueError("date must be in the last 30 days")
    # 调用获取测试统计聚合数据的函数，并传入日期参数
    data = get_test_stat_aggregates(date=args.date)
    # 将数据上传到指定的 S3 存储桶中
    upload_to_s3(
        # 指定要上传到的 S3 存储桶的名称
        bucket_name="torchci-aggregated-stats",
        # 指定在 S3 存储桶中的对象键（路径），使用日期参数作为一部分
        key=f"test_data_aggregates/{str(args.date)}",
        # 上传的数据内容
        docs=data,
    )
```