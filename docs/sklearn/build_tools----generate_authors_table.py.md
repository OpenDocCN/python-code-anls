# `D:\src\scipysrc\scikit-learn\build_tools\generate_authors_table.py`

```
"""
This script generates an html table of contributors, with names and avatars.
The list is generated from scikit-learn's teams on GitHub, plus a small number
of hard-coded contributors.

The table should be updated for each new inclusion in the teams.
Generating the table requires admin rights.
"""

# 导入必要的模块
import getpass  # 导入用于隐藏输入密码的模块
import sys  # 导入用于标准输出流和错误流的模块
import time  # 导入时间处理模块
from os import path  # 从 os 模块中导入 path 模块
from pathlib import Path  # 导入 Path 类，用于处理文件和目录路径

import requests  # 导入处理 HTTP 请求的模块

# 输出用户提示信息到标准错误流
print("user:", file=sys.stderr)

# 获取用户输入的用户名
user = input()

# 获取用户输入的访问令牌，隐藏输入内容
token = getpass.getpass("access token:\n")

# 设置认证信息，使用用户输入的用户名和访问令牌
auth = (user, token)

# 定义 GitHub 组织 Logo 的 URL
LOGO_URL = "https://avatars2.githubusercontent.com/u/365630?v=4"

# 获取脚本所在目录的上级目录路径
REPO_FOLDER = Path(path.abspath(__file__)).parent.parent


def get(url):
    """Send HTTP GET request to the given URL with authentication."""
    # 多次尝试获取 URL 的内容，直到成功或者超过 API 限制
    for sleep_time in [10, 30, 0]:
        reply = requests.get(url, auth=auth)
        # 检查响应内容是否包含 API 速率限制的信息
        api_limit = (
            "message" in reply.json()
            and "API rate limit exceeded" in reply.json()["message"]
        )
        if not api_limit:
            break
        # 输出提示信息，并等待一段时间后重试
        print("API rate limit exceeded, waiting..")
        time.sleep(sleep_time)

    # 检查是否有错误状态码，并抛出异常
    reply.raise_for_status()
    return reply


def get_contributors():
    """Get the list of contributor profiles. Require admin rights."""
    # 获取核心开发人员和贡献者体验团队的列表
    core_devs = []
    documentation_team = []
    contributor_experience_team = []
    comm_team = []
    core_devs_slug = "core-devs"
    contributor_experience_team_slug = "contributor-experience-team"
    comm_team_slug = "communication-team"
    documentation_team_slug = "documentation-team"

    # 设置 GitHub API 入口点
    entry_point = "https://api.github.com/orgs/scikit-learn/"

    # 获取各团队成员的详细信息
    for team_slug, lst in zip(
        (
            core_devs_slug,
            contributor_experience_team_slug,
            comm_team_slug,
            documentation_team_slug,
        ),
        (core_devs, contributor_experience_team, comm_team, documentation_team),
    ):
        for page in [1, 2]:  # 每页30个成员
            # 发送 GET 请求获取团队成员信息
            reply = get(f"{entry_point}teams/{team_slug}/members?page={page}")
            lst.extend(reply.json())

    # 获取 scikit-learn GitHub 组织的成员列表
    members = []
    for page in [1, 2, 3]:  # 每页30个成员
        # 发送 GET 请求获取组织成员信息
        reply = get(f"{entry_point}members?page={page}")
        members.extend(reply.json())

    # 仅保留登录名信息
    core_devs = set(c["login"] for c in core_devs)
    documentation_team = set(c["login"] for c in documentation_team)
    contributor_experience_team = set(c["login"] for c in contributor_experience_team)
    comm_team = set(c["login"] for c in comm_team)
    members = set(c["login"] for c in members)

    # 添加缺失的具有 GitHub 账户的贡献者
    members |= {"dubourg", "mbrucher", "thouis", "jarrodmillman"}

    # 添加缺失的没有 GitHub 账户的贡献者
    members |= {"Angel Soler Gollonet"}

    # 移除 CI 机器人
    members -= {"sklearn-ci", "sklearn-wheels", "sklearn-lgtm"}

    # 从贡献者体验团队中移除核心开发人员 ogrisel
    contributor_experience_team -= core_devs
    # 计算 emeritus 成员，即不属于核心开发、贡献者体验团队、社区团队和文档团队的成员
    emeritus = (
        members
        - core_devs
        - contributor_experience_team
        - comm_team
        - documentation_team
    )

    # 固定列表
    emeritus_contributor_experience_team = {
        "cmarmo",
    }
    # 固定列表
    emeritus_comm_team = {"reshamas"}

    # 更新 emeritus 成员列表，排除贡献者体验团队和社区团队的成员
    emeritus -= emeritus_contributor_experience_team | emeritus_comm_team

    # 更新社区团队列表，从中排除 "reshamas"
    comm_team -= {"reshamas"}  # 在社区团队中但不在网页上

    # 从 GitHub 获取核心开发者的个人资料
    core_devs = [get_profile(login) for login in core_devs]
    # 从 GitHub 获取 emeritus 成员的个人资料
    emeritus = [get_profile(login) for login in emeritus]
    # 从 GitHub 获取贡献者体验团队成员的个人资料
    contributor_experience_team = [
        get_profile(login) for login in contributor_experience_team
    ]
    # 从 GitHub 获取 emeritus 贡献者体验团队成员的个人资料
    emeritus_contributor_experience_team = [
        get_profile(login) for login in emeritus_contributor_experience_team
    ]
    # 从 GitHub 获取社区团队成员的个人资料
    comm_team = [get_profile(login) for login in comm_team]
    # 从 GitHub 获取 emeritus 社区团队成员的个人资料
    emeritus_comm_team = [get_profile(login) for login in emeritus_comm_team]
    # 从 GitHub 获取文档团队成员的个人资料
    documentation_team = [get_profile(login) for login in documentation_team]

    # 根据姓氏排序核心开发者列表
    core_devs = sorted(core_devs, key=key)
    # 根据姓氏排序 emeritus 成员列表
    emeritus = sorted(emeritus, key=key)
    # 根据姓氏排序贡献者体验团队列表
    contributor_experience_team = sorted(contributor_experience_team, key=key)
    # 根据姓氏排序 emeritus 贡献者体验团队列表
    emeritus_contributor_experience_team = sorted(
        emeritus_contributor_experience_team, key=key
    )
    # 根据姓氏排序文档团队列表
    documentation_team = sorted(documentation_team, key=key)
    # 根据姓氏排序社区团队列表
    comm_team = sorted(comm_team, key=key)
    # 根据姓氏排序 emeritus 社区团队列表
    emeritus_comm_team = sorted(emeritus_comm_team, key=key)

    # 返回所有团队成员的列表
    return (
        core_devs,
        emeritus,
        contributor_experience_team,
        emeritus_contributor_experience_team,
        comm_team,
        emeritus_comm_team,
        documentation_team,
    )
def get_profile(login):
    """根据登录名获取 GitHub 用户的个人资料"""
    # 打印正在获取某个登录名的个人资料
    print("get profile for %s" % (login,))
    try:
        # 尝试从 GitHub API 获取用户资料并将其解析为 JSON 格式
        profile = get("https://api.github.com/users/%s" % login).json()
    except requests.exceptions.HTTPError:
        # 如果请求失败，则返回一个空的字典，包含登录名、LOGO_URL（未定义的变量）、空的 HTML URL
        return dict(name=login, avatar_url=LOGO_URL, html_url="")

    # 如果用户没有填写姓名，则将登录名作为姓名
    if profile["name"] is None:
        profile["name"] = profile["login"]

    # 修正缺失的姓名
    missing_names = {
        "bthirion": "Bertrand Thirion",
        "dubourg": "Vincent Dubourg",
        "Duchesnay": "Edouard Duchesnay",
        "Lars": "Lars Buitinck",
        "MechCoder": "Manoj Kumar",
    }
    if profile["name"] in missing_names:
        profile["name"] = missing_names[profile["name"]]

    # 返回用户的资料字典
    return profile


def key(profile):
    """根据姓和名生成排序关键字"""
    # 将用户姓名转换为小写，并按空格分割成组件
    components = profile["name"].lower().split(" ")
    # 返回姓在前名在后的排序关键字
    return " ".join([components[-1]] + components[:-1])


def generate_table(contributors):
    """生成作者信息表格的 HTML"""
    lines = [
        ".. raw :: html\n",
        "    <!-- Generated by generate_authors_table.py -->",
        '    <div class="sk-authors-container">',
        "    <style>",
        "      img.avatar {border-radius: 10px;}",
        "    </style>",
    ]
    for contributor in contributors:
        lines.append("    <div>")
        lines.append(
            "    <a href='%s'><img src='%s' class='avatar' /></a> <br />"
            % (contributor["html_url"], contributor["avatar_url"])
        )
        lines.append("    <p>%s</p>" % (contributor["name"],))
        lines.append("    </div>")
    lines.append("    </div>")
    # 返回生成的 HTML 表格内容
    return "\n".join(lines) + "\n"


def generate_list(contributors):
    """生成作者姓名列表的文本"""
    lines = []
    for contributor in contributors:
        lines.append("- %s" % (contributor["name"],))
    # 返回生成的姓名列表文本内容
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    (
        core_devs,
        emeritus,
        contributor_experience_team,
        emeritus_contributor_experience_team,
        comm_team,
        emeritus_comm_team,
        documentation_team,
    ) = get_contributors()

    with open(
        REPO_FOLDER / "doc" / "maintainers.rst", "w+", encoding="utf-8"
    ) as rst_file:
        rst_file.write(generate_table(core_devs))

    with open(
        REPO_FOLDER / "doc" / "maintainers_emeritus.rst", "w+", encoding="utf-8"
    ) as rst_file:
        rst_file.write(generate_list(emeritus))

    with open(
        REPO_FOLDER / "doc" / "contributor_experience_team.rst", "w+", encoding="utf-8"
    ) as rst_file:
        rst_file.write(generate_table(contributor_experience_team))

    with open(
        REPO_FOLDER / "doc" / "contributor_experience_team_emeritus.rst",
        "w+",
        encoding="utf-8",
    ) as rst_file:
        rst_file.write(generate_list(emeritus_contributor_experience_team))

    with open(
        REPO_FOLDER / "doc" / "communication_team.rst", "w+", encoding="utf-8"
    ) as rst_file:
        rst_file.write(generate_table(comm_team))
    # 使用指定路径打开文件 "communication_team_emeritus.rst"，如果不存在则创建，使用 UTF-8 编码
    with open(
        REPO_FOLDER / "doc" / "communication_team_emeritus.rst", "w+", encoding="utf-8"
    ) as rst_file:
        # 调用函数生成 Emeritus Communication Team 的成员列表，并将结果写入文件
        rst_file.write(generate_list(emeritus_comm_team))
    
    # 使用指定路径打开文件 "documentation_team.rst"，如果不存在则创建，使用 UTF-8 编码
    with open(
        REPO_FOLDER / "doc" / "documentation_team.rst", "w+", encoding="utf-8"
    ) as rst_file:
        # 调用函数生成 Documentation Team 的成员表格，并将结果写入文件
        rst_file.write(generate_table(documentation_team))
```