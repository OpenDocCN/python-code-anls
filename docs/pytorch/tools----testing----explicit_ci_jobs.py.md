# `.\pytorch\tools\testing\explicit_ci_jobs.py`

```
#!/usr/bin/env python3

from __future__ import annotations

import argparse  # 导入命令行参数解析模块
import fnmatch  # 导入文件名匹配模块
import subprocess  # 导入子进程管理模块
import textwrap  # 导入文本包装模块
from pathlib import Path  # 导入路径处理模块
from typing import Any  # 导入类型提示模块

import yaml  # 导入 YAML 解析模块


REPO_ROOT = Path(__file__).parent.parent.parent  # 获取当前脚本所在目录的上级目录的上级目录
CONFIG_YML = REPO_ROOT / ".circleci" / "config.yml"  # 拼接配置文件路径
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"  # 拼接工作流目录路径


WORKFLOWS_TO_CHECK = [
    "binary_builds",
    "build",
    "master_build",
    # These are formatted slightly differently, skip them
    # "scheduled-ci",
    # "debuggable-scheduled-ci",
    # "slow-gradcheck-scheduled-ci",
    # "promote",
]


def add_job(
    workflows: dict[str, Any],
    workflow_name: str,
    type: str,
    job: dict[str, Any],
    past_jobs: dict[str, Any],
) -> None:
    """
    Add job 'job' under 'type' and 'workflow_name' to 'workflow' in place. Also
    add any dependencies (they must already be in 'past_jobs')
    """
    if workflow_name not in workflows:
        workflows[workflow_name] = {"when": "always", "jobs": []}

    requires = job.get("requires", None)
    if requires is not None:
        for requirement in requires:
            dependency = past_jobs[requirement]
            add_job(
                workflows,
                dependency["workflow_name"],
                dependency["type"],
                dependency["job"],
                past_jobs,
            )

    workflows[workflow_name]["jobs"].append({type: job})


def get_filtered_circleci_config(
    workflows: dict[str, Any], relevant_jobs: list[str]
) -> dict[str, Any]:
    """
    Given an existing CircleCI config, remove every job that's not listed in
    'relevant_jobs'
    """
    new_workflows: dict[str, Any] = {}
    past_jobs: dict[str, Any] = {}
    for workflow_name, workflow in workflows.items():
        if workflow_name not in WORKFLOWS_TO_CHECK:
            # Don't care about this workflow, skip it entirely
            continue

        for job_dict in workflow["jobs"]:
            for type, job in job_dict.items():
                if "name" not in job:
                    # Job doesn't have a name so it can't be handled
                    print("Skipping", type)
                else:
                    if job["name"] in relevant_jobs:
                        # Found a job that was specified at the CLI, add it to
                        # the new result
                        add_job(new_workflows, workflow_name, type, job, past_jobs)

                    # Record the job in case it's needed as a dependency later
                    past_jobs[job["name"]] = {
                        "workflow_name": workflow_name,
                        "type": type,
                        "job": job,
                    }

    return new_workflows


def commit_ci(files: list[str], message: str) -> None:
    # Check that there are no other modified files than the ones edited by this
    # tool
    stdout = subprocess.run(
        ["git", "status", "--porcelain"], stdout=subprocess.PIPE
    ).stdout.decode()
    # 解码Git命令的标准输出结果为字符串，并获取其解码后的内容
    
    for line in stdout.split("\n"):
        # 遍历标准输出字符串按行分割后的列表
        if line == "":
            # 如果行为空则跳过当前循环
            continue
        if line[0] != " ":
            # 如果行的第一个字符不是空格，则抛出运行时错误，拒绝提交，显示错误信息
            raise RuntimeError(
                f"Refusing to commit while other changes are already staged: {line}"
            )

    # 将文件添加到Git暂存区
    subprocess.run(["git", "add"] + files)
    
    # 使用Git提交更改，并包含提交信息
    subprocess.run(["git", "commit", "-m", message])
# 如果这个脚本作为主程序被执行
if __name__ == "__main__":
    # 创建一个参数解析器对象，用于解析命令行参数
    parser = argparse.ArgumentParser(
        description="make .circleci/config.yml only have a specific set of jobs and delete GitHub actions"
    )
    # 添加一个命令行参数 --job，用于指定作业名称，可以重复使用
    parser.add_argument("--job", action="append", help="job name", default=[])
    # 添加一个命令行参数 --filter-gha，用于指定要保留的 GitHub Actions，支持通配符匹配
    parser.add_argument(
        "--filter-gha", help="keep only these github actions (glob match)", default=""
    )
    # 添加一个命令行参数 --make-commit，如果使用该选项，则将更改提交到 Git 并标记为不可合并
    parser.add_argument(
        "--make-commit",
        action="store_true",
        help="add change to git with to a do-not-merge commit",
    )
    # 解析命令行参数，并将其保存在 args 对象中
    args = parser.parse_args()

    # 初始化一个列表，包含要修改的文件列表，初始包含 CONFIG_YML 文件
    touched_files = [CONFIG_YML]
    # 打开 CONFIG_YML 文件并读取内容，使用 yaml.safe_load 将其解析为 Python 对象
    with open(CONFIG_YML) as f:
        config_yml = yaml.safe_load(f.read())

    # 调用自定义函数 get_filtered_circleci_config，根据传入的作业名称 args.job 过滤 config_yml["workflows"] 中的工作流配置
    config_yml["workflows"] = get_filtered_circleci_config(
        config_yml["workflows"], args.job
    )

    # 将更新后的 config_yml 对象重新写入 CONFIG_YML 文件
    with open(CONFIG_YML, "w") as f:
        yaml.dump(config_yml, f)

    # 如果指定了 args.filter_gha 参数
    if args.filter_gha:
        # 遍历 WORKFLOWS_DIR 目录中的文件
        for relative_file in WORKFLOWS_DIR.iterdir():
            # 构建文件的绝对路径
            path = REPO_ROOT.joinpath(relative_file)
            # 如果文件名不匹配 args.filter_gha 的通配符规则
            if not fnmatch.fnmatch(path.name, args.filter_gha):
                # 将该文件添加到 touched_files 列表中，并从文件系统中删除
                touched_files.append(path)
                path.resolve().unlink()

    # 如果指定了 args.make_commit 参数
    if args.make_commit:
        # 生成一个包含 args.job 中作业名称的列表字符串
        jobs_str = "\n".join([f" * {job}" for job in args.job])
        # 创建提交消息，包含特定的格式化文本
        message = textwrap.dedent(
            f"""
        [skip ci][do not merge] Edit config.yml to filter specific jobs

        Filter CircleCI to only run:
        {jobs_str}

        See [Run Specific CI Jobs](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md#run-specific-ci-jobs) for details.
        """
        ).strip()
        # 调用 commit_ci 函数，将 touched_files 中的文件提交到 Git 并附带提交消息
        commit_ci([str(f.relative_to(REPO_ROOT)) for f in touched_files], message)
```