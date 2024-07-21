# `.\pytorch\scripts\compile_tests\download_reports.py`

```py
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统相关功能的模块
import pprint  # 导入用于漂亮打印数据结构的模块
import re  # 导入正则表达式模块
import subprocess  # 导入子进程管理模块

import requests  # 导入发送 HTTP 请求的模块


CONFIGS = {
    "dynamo38": {
        "linux-focal-py3.8-clang10 / test (dynamo, 1, 3, linux.2xlarge)",
        "linux-focal-py3.8-clang10 / test (dynamo, 2, 3, linux.2xlarge)",
        "linux-focal-py3.8-clang10 / test (dynamo, 3, 3, linux.2xlarge)",
    },
    "dynamo311": {
        "linux-focal-py3.11-clang10 / test (dynamo, 1, 3, linux.2xlarge)",
        "linux-focal-py3.11-clang10 / test (dynamo, 2, 3, linux.2xlarge)",
        "linux-focal-py3.11-clang10 / test (dynamo, 3, 3, linux.2xlarge)",
    },
    "eager311": {
        "linux-focal-py3.11-clang10 / test (default, 1, 3, linux.2xlarge)",
        "linux-focal-py3.11-clang10 / test (default, 2, 3, linux.2xlarge)",
        "linux-focal-py3.11-clang10 / test (default, 3, 3, linux.2xlarge)",
    },
}


def download_reports(commit_sha, configs=("dynamo38", "dynamo311", "eager311")):
    log_dir = "tmp_test_reports_" + commit_sha  # 设置存放日志的临时目录名

    def subdir_path(config):
        return f"{log_dir}/{config}"  # 生成指定配置的子目录路径

    for config in configs:
        assert config in CONFIGS.keys(), config  # 确保所有配置在 CONFIGS 中存在
    subdir_paths = [subdir_path(config) for config in configs]  # 生成所有配置的子目录路径列表

    # 检查哪些配置的日志尚未下载
    missing_configs = []
    for config, path in zip(configs, subdir_paths):
        if os.path.exists(path):
            continue
        missing_configs.append(config)
    if len(missing_configs) == 0:
        print(
            f"All required logs appear to exist, not downloading again. Run `rm -rf {log_dir}` if this is not the case"
        )
        return subdir_paths  # 如果所有日志都已存在，则直接返回子目录路径列表

    # 获取 GitHub Workflow 运行的 ID
    output = subprocess.check_output(
        ["gh", "run", "list", "-c", commit_sha, "-w", "pull", "--json", "databaseId"]
    ).decode()
    workflow_run_id = str(json.loads(output)[0]["databaseId"])

    # 查看特定 Workflow 运行的详细信息
    output = subprocess.check_output(["gh", "run", "view", workflow_run_id])
    workflow_jobs = parse_workflow_jobs(output)
    print("found the following workflow jobs:")
    pprint.pprint(workflow_jobs)

    # 确定需要下载日志的作业列表
    required_jobs = []
    for config in configs:
        required_jobs.extend(list(CONFIGS[config]))
    for job in required_jobs:
        assert (
            job in workflow_jobs
        ), f"{job} not found, is the commit_sha correct? has the job finished running? The GitHub API may take a couple minutes to update."

    # 请求获取特定 Workflow 运行的所有构件列表信息
    listings = requests.get(
        f"https://hud.pytorch.org/api/artifacts/s3/{workflow_run_id}"
    ).json()
    def download_report(job_name, subdir):
        # 从 workflow_jobs 字典中获取 job_name 对应的 job_id
        job_id = workflow_jobs[job_name]
        
        # 遍历 listings 列表中的每个 listing
        for listing in listings:
            # 获取 listing 的名称
            name = listing["name"]
            
            # 如果名称不以 "test-reports-" 开头，则跳过当前循环
            if not name.startswith("test-reports-"):
                continue
            
            # 如果名称以 "_{job_id}.zip" 结尾，则获取对应的 url
            if name.endswith(f"_{job_id}.zip"):
                url = listing["url"]
                
                # 使用 subprocess 调用 wget 命令下载文件到 subdir 目录
                subprocess.run(["wget", "-P", subdir, url], check=True)
                
                # 构建下载的 ZIP 文件路径
                path_to_zip = f"{subdir}/{name}"
                
                # 构建解压后的目录名
                dir_name = path_to_zip[:-4]
                
                # 使用 subprocess 调用 unzip 命令解压 ZIP 文件到 dir_name 目录
                subprocess.run(["unzip", path_to_zip, "-d", dir_name], check=True)
                
                # 下载完成后立即返回
                return
        
        # 如果没有找到符合条件的 ZIP 文件，则抛出断言错误
        raise AssertionError("should not be hit")

    # 如果 log_dir 不存在，则创建该目录
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 遍历 configs 集合中存在但是 missing_configs 集合中不存在的每个 config
    for config in set(configs) - set(missing_configs):
        # 打印日志，说明该 config 的日志已经存在，无需重新下载
        print(
            f"Logs for {config} already exist, not downloading again. Run `rm -rf {subdir_path(config)}` if this is not the case."
        )
    
    # 遍历 missing_configs 集合中的每个 config
    for config in missing_configs:
        # 获取 config 对应的子目录路径
        subdir = subdir_path(config)
        
        # 创建该子目录
        os.mkdir(subdir)
        
        # 获取该 config 对应的 job_names 列表
        job_names = CONFIGS[config]
        
        # 遍历 job_names 列表中的每个 job_name，调用 download_report 函数下载报告
        for job_name in job_names:
            download_report(job_name, subdir)

    # 返回所有子目录路径的列表
    return subdir_paths
# 解析工作流输出，将其转换为字典格式返回
def parse_workflow_jobs(output):
    # 初始化一个空字典用于存储解析结果
    result = {}
    # 将输出按行分割，并解码为字符串，形成行的列表
    lines = output.decode().split("\n")
    # 遍历每一行
    for line in lines:
        # 使用正则表达式匹配行中特定格式的内容
        match = re.search(r"(\S+ / .*) in .* \(ID (\d+)\)", line)
        # 如果没有匹配到任何内容，继续下一次循环
        if match is None:
            continue
        # 将匹配到的结果存入字典，文件名为键，ID为值
        result[match.group(1)] = match.group(2)
    # 返回最终解析得到的字典
    return result
```