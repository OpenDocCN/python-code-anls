# `D:\src\scipysrc\scikit-learn\maint_tools\update_tracking_issue.py`

```
"""Creates or updates an issue if the CI fails. This is useful to keep track of
scheduled jobs that are failing repeatedly.

This script depends on:
- `defusedxml` for safer parsing for xml
- `PyGithub` for interacting with GitHub

The GitHub token only requires the `repo:public_repo` scope are described in
https://docs.github.com/en/developers/apps/building-oauth-apps/scopes-for-oauth-apps#available-scopes.
This scope allows the bot to create and edit its own issues. It is best to use a
github account that does **not** have commit access to the public repo.
"""

import argparse   # 导入用于解析命令行参数的模块
import sys   # 导入系统相关的模块
from datetime import datetime, timezone   # 从 datetime 模块导入日期时间和时区相关的类
from pathlib import Path   # 导入用于处理文件路径的模块

import defusedxml.ElementTree as ET   # 导入安全解析 XML 的模块
from github import Github   # 从 PyGithub 库中导入与 GitHub 交互的类和方法

parser = argparse.ArgumentParser(
    description="Create or update issue from JUnit test results from pytest"
)
parser.add_argument(
    "bot_github_token", help="Github token for creating or updating an issue"
)
parser.add_argument("ci_name", help="Name of CI run instance")
parser.add_argument("issue_repo", help="Repo to track issues")
parser.add_argument("link_to_ci_run", help="URL to link to")
parser.add_argument("--junit-file", help="JUnit file to determine if tests passed")
parser.add_argument(
    "--tests-passed",
    help=(
        "If --tests-passed is true, then the original issue is closed if the issue "
        "exists. If tests-passed is false, then the an issue is updated or created."
    ),
)
parser.add_argument(
    "--auto-close",
    help=(
        "If --auto-close is false, then issues will not auto close even if the tests"
        " pass."
    ),
    default="true",
)

args = parser.parse_args()   # 解析命令行参数并存储到 args 变量中

if args.junit_file is not None and args.tests_passed is not None:
    print("--junit-file and --test-passed can not be set together")
    sys.exit(1)   # 如果同时设置了 --junit-file 和 --test-passed，则打印错误信息并退出程序

if args.junit_file is None and args.tests_passed is None:
    print("Either --junit-file or --test-passed must be passed in")
    sys.exit(1)   # 如果未同时设置 --junit-file 和 --test-passed，则打印错误信息并退出程序

gh = Github(args.bot_github_token)   # 使用传入的 GitHub token 创建 Github 对象
issue_repo = gh.get_repo(args.issue_repo)   # 获取指定仓库的 GitHub 仓库对象
dt_now = datetime.now(tz=timezone.utc)   # 获取当前的 UTC 时间
date_str = dt_now.strftime("%b %d, %Y")   # 将当前时间格式化为指定的字符串格式
title_query = f"CI failed on {args.ci_name}"
title = f"⚠️ {title_query} (last failure: {date_str}) ⚠️"


def get_issue():
    login = gh.get_user().login   # 获取当前 GitHub 用户的登录名
    issues = gh.search_issues(
        f"repo:{args.issue_repo} {title_query} in:title state:open author:{login}"
    )   # 使用 GitHub API 搜索符合条件的问题（issue）
    first_page = issues.get_page(0)
    # Return issue if it exist
    return first_page[0] if first_page else None   # 如果存在符合条件的问题，则返回第一个；否则返回 None


def create_or_update_issue(body=""):
    # Interact with GitHub API to create issue
    link = f"[{args.ci_name}]({args.link_to_ci_run})"   # 创建包含 CI 运行实例链接的 Markdown 链接
    issue = get_issue()   # 获取与当前 CI 实例相关的问题（如果存在的话）

    max_body_length = 60_000   # GitHub API 对问题正文的最大长度限制
    original_body_length = len(body)
    # Avoid "body is too long (maximum is 65536 characters)" error from github REST API
    # 如果原始消息体长度超过了最大长度限制，则进行截断处理并生成新的消息体
    if original_body_length > max_body_length:
        body = (
            f"{body[:max_body_length]}\n...\n"
            f"Body was too long ({original_body_length} characters) and was shortened"
        )

    # 如果issue为None，则创建新的issue
    if issue is None:
        # 构建新issue的标题
        header = f"**CI failed on {link}** ({date_str})"
        # 创建issue并提交到issue仓库
        issue = issue_repo.create_issue(title=title, body=f"{header}\n{body}")
        # 打印创建的issue信息
        print(f"Created issue in {args.issue_repo}#{issue.number}")
        # 退出程序
        sys.exit()
    else:
        # 如果issue不为None，则更新现有的issue
        header = f"**CI is still failing on {link}** ({date_str})"
        # 编辑issue的标题和内容
        issue.edit(title=title, body=f"{header}\n{body}")
        # 打印更新issue的信息
        print(f"Commented on issue: {args.issue_repo}#{issue.number}")
        # 退出程序
        sys.exit()
# 如果测试通过，则关闭问题（issue）并退出程序
if args.tests_passed is not None:
    if args.tests_passed.lower() == "true":
        close_issue_if_opened()
    else:
        create_or_update_issue()

# 将 JUnit 文件路径转为 Path 对象
junit_path = Path(args.junit_file)

# 如果 JUnit 文件不存在，创建或更新一个问题（issue），并退出程序
if not junit_path.exists():
    body = "Unable to find junit file. Please see link for details."
    create_or_update_issue(body)

# 解析 JUnit 文件的 XML 树结构
tree = ET.parse(args.junit_file)
failure_cases = []

# 检查是否有测试集错误（error）
error = tree.find("./testsuite/testcase/error")
if error is not None:
    # 添加测试集错误信息到失败案例列表中
    failure_cases.append("Test Collection Failure")

# 遍历每个测试案例（testcase）
for item in tree.iter("testcase"):
    failure = item.find("failure")
    if failure is None:
        continue

    # 如果测试案例有失败信息，将其名称添加到失败案例列表中
    failure_cases.append(item.attrib["name"])

# 如果没有失败案例，关闭问题（issue）并退出程序
if not failure_cases:
    close_issue_if_opened()

# 为问题（issue）创建内容
body_list = [f"- {case}" for case in failure_cases]
body = "\n".join(body_list)
create_or_update_issue(body)
```