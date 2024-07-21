# `.\pytorch\benchmarks\upload_scribe.py`

```
"""Scribe Uploader for Pytorch Benchmark Data

Currently supports data in pytest-benchmark format but can be extended.

New fields can be added just by modifying the schema in this file, schema
checking is only here to encourage reusing existing fields and avoiding typos.
"""

# 导入必要的库
import argparse  # 解析命令行参数的库
import json  # 处理 JSON 格式数据的库
import os  # 提供操作系统相关的功能的库
import subprocess  # 执行外部命令的库
import time  # 提供时间相关操作的库
from collections import defaultdict  # 默认字典，可以设置默认值的字典

import requests  # 发送 HTTP 请求的库


class ScribeUploader:
    def __init__(self, category):
        self.category = category  # 初始化时设置上传类别

    def format_message(self, field_dict):
        assert "time" in field_dict, "Missing required Scribe field 'time'"
        message = defaultdict(dict)  # 创建默认字典 message
        for field, value in field_dict.items():
            if field in self.schema["normal"]:
                message["normal"][field] = str(value)  # 将字段值转换为字符串并存入 message 中
            elif field in self.schema["int"]:
                message["int"][field] = int(value)  # 将字段值转换为整数并存入 message 中
            elif field in self.schema["float"]:
                message["float"][field] = float(value)  # 将字段值转换为浮点数并存入 message 中
            else:
                raise ValueError(
                    f"Field {field} is not currently used, be intentional about adding new fields"
                )  # 如果字段不在预定义的类型中，则抛出值错误异常
        return message  # 返回处理后的消息字典

    def _upload_intern(self, messages):
        for m in messages:
            json_str = json.dumps(m)  # 将消息字典转换为 JSON 格式的字符串
            cmd = ["scribe_cat", self.category, json_str]  # 构建上传命令
            subprocess.run(cmd)  # 执行上传命令

    def upload(self, messages):
        if os.environ.get("SCRIBE_INTERN"):  # 检查是否使用内部上传
            return self._upload_intern(messages)  # 调用内部上传方法
        access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN")
        if not access_token:
            raise ValueError("Can't find access token from environment variable")
        url = "https://graph.facebook.com/scribe_logs"  # Scribe 日志服务的 URL
        r = requests.post(
            url,
            data={
                "access_token": access_token,  # 设置访问令牌
                "logs": json.dumps(
                    [
                        {
                            "category": self.category,  # 设置日志类别
                            "message": json.dumps(message),  # 转换消息字典为 JSON 格式
                            "line_escape": False,  # 禁用行转义
                        }
                        for message in messages  # 遍历所有消息
                    ]
                ),
            },
        )
        print(r.text)  # 输出响应内容
        r.raise_for_status()  # 抛出异常如果请求不成功


class PytorchBenchmarkUploader(ScribeUploader):
    # 这个类继承自 ScribeUploader 类，用于上传 PyTorch Benchmark 数据到 Scribe
    # 初始化方法，继承父类并设定特定名称
    def __init__(self):
        super().__init__("perfpipe_pytorch_benchmarks")
        # 定义数据模式的字典，包含不同类型的数据和相应的字段列表
        self.schema = {
            "int": [
                "time",
                "rounds",
            ],
            "normal": [
                "benchmark_group",
                "benchmark_name",
                "benchmark_executor",
                "benchmark_fuser",
                "benchmark_class",
                "benchmark_time",
                "pytorch_commit_id",
                "pytorch_branch",
                "pytorch_commit_time",
                "pytorch_version",
                "pytorch_git_dirty",
                "machine_kernel",
                "machine_processor",
                "machine_hostname",
                "circle_build_num",
                "circle_project_reponame",
            ],
            "float": [
                "stddev",
                "min",
                "median",
                "max",
                "mean",
            ],
        }

    # 发布 Pytest 基准测试结果的方法，接受 Pytest JSON 对象作为输入
    def post_pytest_benchmarks(self, pytest_json):
        # 获取测试机器信息和提交信息
        machine_info = pytest_json["machine_info"]
        commit_info = pytest_json["commit_info"]
        # 记录上传时间戳
        upload_time = int(time.time())
        # 初始化消息列表
        messages = []
        # 遍历每个基准测试结果
        for b in pytest_json["benchmarks"]:
            # 解析测试名称和网络名称
            test = b["name"].split("[")[0]
            net_name = b["params"]["net_name"]
            benchmark_name = f"{test}[{net_name}]"
            # 获取执行者和融合器信息
            executor = b["params"]["executor"]
            fuser = b["params"]["fuser"]
            # 格式化消息数据
            m = self.format_message(
                {
                    "time": upload_time,
                    "benchmark_group": b["group"],
                    "benchmark_name": benchmark_name,
                    "benchmark_executor": executor,
                    "benchmark_fuser": fuser,
                    "benchmark_class": b["fullname"],
                    "benchmark_time": pytest_json["datetime"],
                    "pytorch_commit_id": commit_info["id"],
                    "pytorch_branch": commit_info["branch"],
                    "pytorch_commit_time": commit_info["time"],
                    "pytorch_version": None,  # 这里待填写具体的 PyTorch 版本信息
                    "pytorch_git_dirty": commit_info["dirty"],
                    "machine_kernel": machine_info["release"],
                    "machine_processor": machine_info["processor"],
                    "machine_hostname": machine_info["node"],
                    "circle_build_num": os.environ.get("CIRCLE_BUILD_NUM"),
                    "circle_project_reponame": os.environ.get(
                        "CIRCLE_PROJECT_REPONAME"
                    ),
                    "stddev": b["stats"]["stddev"],
                    "rounds": b["stats"]["rounds"],
                    "min": b["stats"]["min"],
                    "median": b["stats"]["median"],
                    "max": b["stats"]["max"],
                    "mean": b["stats"]["mean"],
                }
            )
            # 将格式化后的消息添加到消息列表
            messages.append(m)
        # 调用上传方法，将所有消息上传
        self.upload(messages)
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而非被导入为模块），则执行以下代码块

    # 创建一个参数解析器对象，用于处理命令行参数
    parser = argparse.ArgumentParser(description=__doc__)
    
    # 添加一个命令行参数 `--pytest-bench-json`，支持参数别名 `--pytest_bench_json`
    # 参数类型为打开文件对象，用于读取上传的 JSON 数据文件
    parser.add_argument(
        "--pytest-bench-json",
        "--pytest_bench_json",
        type=argparse.FileType("r"),
        help="Upload json data formatted by pytest-benchmark module",
    )
    
    # 解析命令行参数，并将其存储在 `args` 对象中
    args = parser.parse_args()
    
    # 如果存在参数 `--pytest_bench_json`，则执行以下代码块
    if args.pytest_bench_json:
        # 创建 PytorchBenchmarkUploader 的实例对象
        benchmark_uploader = PytorchBenchmarkUploader()
        
        # 从命令行参数指定的 JSON 文件中加载数据
        json_data = json.load(args.pytest_bench_json)
        
        # 调用 benchmark_uploader 对象的方法，将加载的 JSON 数据上传
        benchmark_uploader.post_pytest_benchmarks(json_data)
```