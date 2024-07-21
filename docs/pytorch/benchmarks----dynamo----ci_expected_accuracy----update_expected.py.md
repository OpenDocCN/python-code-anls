# `.\pytorch\benchmarks\dynamo\ci_expected_accuracy\update_expected.py`

```py
"""
Update commited CSV files used as reference points by dynamo/inductor CI.

Currently only cares about graph breaks, so only saves those columns.

Hardcodes a list of job names and artifacts per job, but builds the lookup
by querying github sha and finding associated github actions workflow ID and CI jobs,
downloading artifact zips, extracting CSVs and filtering them.

Usage:

python benchmarks/dynamo/ci_expected_accuracy.py <sha of pytorch commit that has completed inductor benchmark jobs>

Known limitations:
- doesn't handle 'retry' jobs in CI, if the same hash has more than one set of artifacts, gets the first one
"""

import argparse  # 用于解析命令行参数的库
import json  # 用于处理 JSON 数据的库
import os  # 用于与操作系统进行交互的库
import subprocess  # 用于运行外部命令的库
import sys  # 用于与 Python 解释器进行交互的库
import urllib  # 用于处理 URL 的库
from io import BytesIO  # 用于创建字节流的类
from itertools import product  # 用于生成迭代器的库
from pathlib import Path  # 用于处理文件路径的库
from urllib.request import urlopen  # 用于打开 URL 的函数
from zipfile import ZipFile  # 用于处理 ZIP 文件的类

import pandas as pd  # 导入 pandas 库用于数据处理
import requests  # 用于发送 HTTP 请求的库

# Note: the public query url targets this rockset lambda:
# https://console.rockset.com/lambdas/details/commons.artifacts
ARTIFACTS_QUERY_URL = "https://api.usw2a1.rockset.com/v1/public/shared_lambdas/4ca0033e-0117-41f5-b043-59cde19eff35"
# 设置查询工件的 URL，指向 Rockset Lambda 函数

CSV_LINTER = str(
    Path(__file__).absolute().parent.parent.parent.parent
    / "tools/linter/adapters/no_merge_conflict_csv_linter.py"
)
# 设置 CSV 文件的验证工具路径

def query_job_sha(repo, sha):
    # 构造查询参数
    params = {
        "parameters": [
            {"name": "sha", "type": "string", "value": sha},
            {"name": "repo", "type": "string", "value": repo},
        ]
    }

    # 发送 POST 请求到 Rockset Lambda API
    r = requests.post(url=ARTIFACTS_QUERY_URL, json=params)
    # 获取返回的 JSON 数据
    data = r.json()
    return data["results"]
    # 返回查询结果中的数据项列表

def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))
    # 解析作业名称，返回去除空白后的分隔部分迭代器

def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(")").split(","))
    # 解析测试字符串，返回去除空白后的分隔部分迭代器

S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"
# 设置 S3 存储桶的基础 URL

def get_artifacts_urls(results, suites):
    urls = {}
    for r in results:
        if (
            r["workflowName"] in ("inductor", "inductor-periodic")
            and "test" in r["jobName"]
        ):
            # 解析作业名称和测试字符串
            config_str, test_str = parse_job_name(r["jobName"])
            suite, shard_id, num_shards, machine, *_ = parse_test_str(test_str)
            workflowId = r["workflowId"]
            id = r["id"]
            runAttempt = r["runAttempt"]

            if suite in suites:
                # 构造 S3 URL
                artifact_filename = f"test-reports-test-{suite}-{shard_id}-{num_shards}-{machine}_{id}.zip"
                s3_url = f"{S3_BASE_URL}/{repo}/{workflowId}/{runAttempt}/artifact/{artifact_filename}"
                urls[(suite, int(shard_id))] = s3_url
                print(f"{suite} {shard_id}, {num_shards}: {s3_url}")
    return urls
    # 返回包含 suite 和 S3 URL 的字典

def normalize_suite_filename(suite_name):
    strs = suite_name.split("_")
    subsuite = strs[-1]
    if "timm" in subsuite:
        subsuite = subsuite.replace("timm", "timm_models")

    return subsuite
    # 标准化套件文件名并返回

def download_artifacts_and_extract_csvs(urls):
    dataframes = {}
    # 遍历包含 URL 的字典 `urls`，其中键为元组 `(suite, shard)`，值为 URL `url`
    for (suite, shard), url in urls.items():
        # 尝试打开指定的 URL
        try:
            # 使用 urlopen 打开 URL，获取响应对象 `resp`
            resp = urlopen(url)
            # 根据 suite 文件名进行规范化处理，得到 `subsuite`
            subsuite = normalize_suite_filename(suite)
            # 将响应内容封装成字节流，然后创建 ZipFile 对象 `artifact`
            artifact = ZipFile(BytesIO(resp.read()))
            # 遍历需要处理的两个阶段：`training` 和 `inference`
            for phase in ("training", "inference"):
                # 构造 CSV 文件名 `name`
                name = f"test/test-reports/{phase}_{subsuite}.csv"
                try:
                    # 尝试从 `artifact` 中读取名为 `name` 的 CSV 文件，并创建 DataFrame `df`
                    df = pd.read_csv(artifact.open(name))
                    # 将 DataFrame 中的 "graph_breaks" 列的 NaN 值填充为 0，并转换为整数类型
                    df["graph_breaks"] = df["graph_breaks"].fillna(0).astype(int)
                    # 获取之前存储的同一 suite 和 phase 的 DataFrame `prev_df`
                    prev_df = dataframes.get((suite, phase), None)
                    # 将新读取的 DataFrame `df` 与之前的 `prev_df` 合并，存储到 `dataframes` 中
                    dataframes[(suite, phase)] = (
                        pd.concat([prev_df, df]) if prev_df is not None else df
                    )
                except KeyError:
                    # 如果在 `artifact` 中找不到指定的 `name`，打印警告信息
                    print(
                        f"Warning: Unable to find {name} in artifacts file from {url}, continuing"
                    )
        except urllib.error.HTTPError:
            # 如果无法下载 URL，打印相应的错误信息
            print(f"Unable to download {url}, perhaps the CI job isn't finished?")
    
    # 返回存储所有处理过的 DataFrame 的 `dataframes` 字典
    return dataframes
# 在指定的根路径下，为每个数据框写入过滤后的 CSV 文件
def write_filtered_csvs(root_path, dataframes):
    # 遍历 dataframes 字典中的每个项，其中 key 是 (suite, phase)，value 是对应的 DataFrame df
    for (suite, phase), df in dataframes.items():
        # 构造输出文件名，格式为 "{suite}_{phase}.csv"，并将路径拼接到根路径下
        out_fn = os.path.join(root_path, f"{suite}_{phase}.csv")
        # 将 DataFrame 写入 CSV 文件，不包括索引，仅包括指定列 ["name", "accuracy", "graph_breaks"]
        df.to_csv(out_fn, index=False, columns=["name", "accuracy", "graph_breaks"])
        # 对刚写入的 CSV 文件应用 lint 检查
        apply_lints(out_fn)


# 应用 lint 规则来修正指定文件中的问题
def apply_lints(filename):
    # 使用 subprocess 执行 CSV lint 工具，并解析输出的 JSON 补丁
    patch = json.loads(subprocess.check_output([sys.executable, CSV_LINTER, filename]))
    # 如果补丁中包含 "replacement" 字段
    if patch.get("replacement"):
        # 打开文件并读取所有内容
        with open(filename) as fd:
            data = fd.read().replace(patch["original"], patch["replacement"])
        # 将修正后的内容写回到文件中
        with open(filename, "w") as fd:
            fd.write(data)


# 当直接运行脚本时执行的主程序入口
if __name__ == "__main__":
    # 创建解析器对象，用于处理命令行参数
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # 添加命令行参数 "sha"，用于接收提交的 SHA 标识符
    parser.add_argument("sha")
    # 解析命令行参数
    args = parser.parse_args()

    # 定义 GitHub 仓库名
    repo = "pytorch/pytorch"

    # 定义测试套件组合集合
    suites = {
        f"{a}_{b}"
        for a, b in product(
            [
                "aot_eager",
                "aot_inductor",
                "cpu_inductor",
                "dynamic_aot_eager",
                "dynamic_cpu_inductor",
                "dynamic_inductor",
                "dynamo_eager",
                "inductor",
            ],
            ["huggingface", "timm", "torchbench"],
        )
    }

    # 定义根路径，指向预期准确性的基准测试数据目录
    root_path = "benchmarks/dynamo/ci_expected_accuracy/"
    # 断言根路径存在，否则输出指定信息并终止程序
    assert os.path.exists(root_path), f"cd <pytorch root> and ensure {root_path} exists"

    # 查询指定提交 SHA 标识符的作业结果
    results = query_job_sha(repo, args.sha)
    # 获取结果中与测试套件相关的所有构件的 URL
    urls = get_artifacts_urls(results, suites)
    # 下载构件并提取其中的 CSV 文件数据
    dataframes = download_artifacts_and_extract_csvs(urls)
    # 将下载的数据写入过滤后的 CSV 文件
    write_filtered_csvs(root_path, dataframes)
    # 输出成功信息，提示用户确认对 .csv 文件的更改并使用 `git add` 将其加入版本控制
    print("Success. Now, confirm the changes to .csvs and `git add` them if satisfied.")
```