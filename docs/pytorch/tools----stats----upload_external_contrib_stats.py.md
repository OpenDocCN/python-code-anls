# `.\pytorch\tools\stats\upload_external_contrib_stats.py`

```
# 从未来导入标注，允许使用函数中的类型提示
from __future__ import annotations

# 导入必要的库
import argparse  # 用于命令行参数解析
import datetime  # 用于日期和时间操作
import json  # 用于 JSON 数据处理
import os  # 提供操作系统相关的功能
import time  # 提供时间相关的功能
import urllib.parse  # 用于 URL 编码和解码
from typing import Any, Callable, cast, Dict, List  # 引入类型提示相关的类和函数
from urllib.error import HTTPError  # 处理 URL 相关的错误
from urllib.request import Request, urlopen  # 发送 HTTP 请求

# 导入自定义模块中的函数
from tools.stats.upload_stats_lib import upload_to_s3

# 不包含在统计数据中的用户列表
FILTER_OUT_USERS = {
    "pytorchmergebot",
    "facebook-github-bot",
    "pytorch-bot[bot]",
    "pytorchbot",
    "pytorchupdatebot",
    "dependabot[bot]",
}


def _fetch_url(
    url: str,
    headers: dict[str, str],  # HTTP 请求头信息
    data: dict[str, Any] | None = None,  # HTTP 请求的数据，可选
    method: str | None = None,  # HTTP 请求方法，可选
    reader: Callable[[Any], Any] = lambda x: x.read(),  # 处理 HTTP 响应的函数，默认读取全部内容
) -> Any:
    # 从环境变量中获取 GitHub Token
    token = os.environ.get("GITHUB_TOKEN")
    # 如果存在 token 并且请求地址是 GitHub API
    if token is not None and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"token {token}"
    # 如果有数据，则将其转换为 JSON 字符串并编码成字节流
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        # 使用 urllib 发送 HTTP 请求
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return reader(conn)  # 使用提供的读取函数处理 HTTP 响应
    except HTTPError as err:
        print(err.reason)  # 打印 HTTP 错误原因
        print(err.headers)  # 打印 HTTP 响应头
        # 如果是因为速率限制而失败，打印速率限制信息
        if err.code == 403 and all(
            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]
        ):
            print(
                f"Rate limit exceeded: {err.headers['X-RateLimit-Used']}/{err.headers['X-RateLimit-Limit']}"
            )
        raise  # 抛出异常，终止程序


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,  # HTTP 请求的查询参数，可选
    data: dict[str, Any] | None = None,  # HTTP 请求的数据，可选
) -> list[dict[str, Any]]:
    headers = {"Accept": "application/vnd.github.v3+json"}  # GitHub API v3 版本的 JSON 请求头
    # 如果有查询参数，则将其转换为 URL 查询字符串格式
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={urllib.parse.quote(str(val))}" for name, val in params.items()
        )
    # 调用 _fetch_url 函数获取 JSON 数据并返回
    return cast(
        List[Dict[str, Any]],
        _fetch_url(url, headers=headers, data=data, reader=json.load),
    )


def get_external_pr_data(
    start_date: datetime.date,  # 外部 PR 数据的起始日期
    end_date: datetime.date,  # 外部 PR 数据的结束日期
    period_length: int = 1  # 外部 PR 统计的周期长度，默认为 1 天
) -> list[dict[str, Any]]:
    pr_info = []  # 初始化 PR 数据列表为空
    period_begin_date = start_date  # 初始化周期起始日期为开始日期

    pr_count = 0  # 初始化 PR 计数为 0
    users: set[str] = set()  # 初始化用户集合为空
    # 当周期开始日期小于结束日期时执行循环
    while period_begin_date < end_date:
        # 计算周期结束日期，当前周期开始日期加上周期长度减一天
        period_end_date = period_begin_date + datetime.timedelta(days=period_length - 1)
        
        # 初始化页面数为1和响应列表为空列表
        page = 1
        responses: list[dict[str, Any]] = []
        
        # 在响应列表非空或者页面为1时执行循环
        while len(responses) > 0 or page == 1:
            # 获取 GitHub API 中的问题搜索结果
            response = cast(
                Dict[str, Any],
                fetch_json(
                    "https://api.github.com/search/issues",
                    params={
                        "q": f'repo:pytorch/pytorch is:pr is:closed \
                            label:"open source" label:Merged -label:Reverted closed:{period_begin_date}..{period_end_date}',
                        "per_page": "100",
                        "page": str(page),
                    },
                ),
            )
            # 提取 API 响应中的项目列表
            items = response["items"]
            
            # 遍历每个项目
            for item in items:
                # 提取用户登录名
                u = item["user"]["login"]
                
                # 如果用户不在过滤列表中，则增加 Pull Request 计数和用户列表
                if u not in FILTER_OUT_USERS:
                    pr_count += 1
                    users.add(u)
            
            # 增加页面数，准备获取下一页的数据
            page += 1

        # 将当前周期的统计信息添加到 PR 信息列表中
        pr_info.append(
            {
                "date": str(period_begin_date),
                "pr_count": pr_count,
                "user_count": len(users),
                "users": list(users),
            }
        )
        
        # 更新周期开始日期为当前周期结束日期的下一天
        period_begin_date = period_end_date + datetime.timedelta(days=1)
    
    # 返回所有周期的 PR 统计信息列表
    return pr_info
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则开始执行以下代码块

    parser = argparse.ArgumentParser(
        description="Upload external contribution stats to Rockset"
    )
    # 创建参数解析器，用于处理命令行参数，描述为将外部贡献统计数据上传到Rockset

    parser.add_argument(
        "--startDate",
        type=datetime.date.fromisoformat,
        required=True,
        help="the first date to upload data for in any valid ISO 8601 format format (eg. YYYY-MM-DD).",
    )
    # 添加命令行参数 --startDate，要求必须提供，类型为日期对象，用于指定上传数据的起始日期

    parser.add_argument(
        "--length",
        type=int,
        required=False,
        help="the number of days to upload data for. Default is 1.",
        default=1,
    )
    # 添加命令行参数 --length，可选参数，类型为整数，默认为1，表示上传数据的天数长度

    parser.add_argument(
        "--period-length",
        type=int,
        required=False,
        help="the number of days to group data for. Default is 1.",
        default=1,
    )
    # 添加命令行参数 --period-length，可选参数，类型为整数，默认为1，表示数据分组的天数长度

    args = parser.parse_args()
    # 解析命令行参数，将结果存储在args对象中

    for i in range(args.length):
        # 循环，从0到args.length-1

        tries = 0
        # 尝试次数初始化为0

        startdate = args.startDate + datetime.timedelta(days=i)
        # 根据参数传入的起始日期和当前循环次数i，计算出实际起始日期

        data = get_external_pr_data(
            startdate,
            startdate + datetime.timedelta(days=args.period_length),
            period_length=args.period_length,
        )
        # 调用函数get_external_pr_data，获取外部Pull Request数据，传入起始日期、结束日期和周期长度作为参数

        for pr_info in data:
            # 遍历data中的每个Pull Request信息

            # sometimes users does not get added, so we check it got uploaded
            assert "users" in pr_info
            assert isinstance(pr_info["users"], list)
            # 断言确保pr_info中包含键"users"并且其值是一个列表

        print(f"uploading the following data: \n {data}")
        # 打印上传的数据信息

        upload_to_s3(
            bucket_name="torchci-contribution-data",
            key=f"external_contribution_counts/{str(startdate)}",
            docs=data,
        )
        # 调用上传函数upload_to_s3，将数据上传到指定的S3存储桶，指定键为外部贡献计数的日期字符串形式

        # get around rate limiting
        time.sleep(10)
        # 等待10秒，以避免请求速率限制
```