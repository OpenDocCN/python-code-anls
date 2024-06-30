# `D:\src\scipysrc\scikit-learn\build_tools\get_comment.py`

```
# This script is used to generate a comment for a PR when linting issues are
# detected. It is used by the `Comment on failed linting` GitHub Action.
# This script fails if there are not comments to be posted.

import os  # 导入操作系统相关模块

import requests  # 导入处理HTTP请求的模块


def get_versions(versions_file):
    """Get the versions of the packages used in the linter job.

    Parameters
    ----------
    versions_file : str
        The path to the file that contains the versions of the packages.

    Returns
    -------
    versions : dict
        A dictionary with the versions of the packages.
    """
    with open("versions.txt", "r") as f:
        return dict(line.strip().split("=") for line in f)  # 从文件中读取每行内容，按等号分割为键值对，返回字典


def get_step_message(log, start, end, title, message, details):
    """Get the message for a specific test.

    Parameters
    ----------
    log : str
        The log of the linting job.

    start : str
        The string that marks the start of the test.

    end : str
        The string that marks the end of the test.

    title : str
        The title for this section.

    message : str
        The message to be added at the beginning of the section.

    details : bool
        Whether to add the details of each step.

    Returns
    -------
    message : str
        The message to be added to the comment.
    """
    if end not in log:
        return ""  # 如果结束标记不在日志中，则返回空字符串
    res = (
        "-----------------------------------------------\n"
        + f"### {title}\n\n"
        + message
        + "\n\n"
    )
    if details:
        res += (
            "<details>\n\n```\n"
            + log[log.find(start) + len(start) + 1 : log.find(end) - 1]
            + "\n```\n\n</details>\n\n"
        )  # 如果需要详细信息，将详细信息添加到结果中
    return res


def get_message(log_file, repo, pr_number, sha, run_id, details, versions):
    with open(log_file, "r") as f:
        log = f.read()  # 读取日志文件内容

    sub_text = (
        "\n\n<sub> _Generated for commit:"
        f" [{sha[:7]}](https://github.com/{repo}/pull/{pr_number}/commits/{sha}). "
        "Link to the linter CI: [here]"
        f"(https://github.com/{repo}/actions/runs/{run_id})_ </sub>"
    )  # 生成附加的提交信息文本

    if "### Linting completed ###" not in log:
        return (
            "## ❌ Linting issues\n\n"
            "There was an issue running the linter job. Please update with "
            "`upstream/main` ([link]("
            "https://scikit-learn.org/dev/developers/contributing.html"
            "#how-to-contribute)) and push the changes. If you already have done "
            "that, please send an empty commit with `git commit --allow-empty` "
            "and push the changes to trigger the CI.\n\n" + sub_text
        )  # 如果日志中不存在完成标记，返回linting出现问题的消息和提交信息

    message = ""

    # black
    # 将运行 `black` 的步骤信息添加到消息中
    message += get_step_message(
        log,
        start="### Running black ###",  # 开始运行 black 的提示
        end="Problems detected by black",  # black 检测到问题的结束提示
        title="`black`",  # 标题为 `black`
        message=(
            "`black` detected issues. Please run `black .` locally and push "
            "the changes. Here you can see the detected issues. Note that "
            "running black might also fix some of the issues which might be "
            "detected by `ruff`. Note that the installed `black` version is "
            f"`black={versions['black']}`."  # 提示关于 black 检测到问题的消息，包括安装版本信息
        ),
        details=details,  # 细节信息参数
    )

    # 将运行 `ruff` 的步骤信息添加到消息中
    message += get_step_message(
        log,
        start="### Running ruff ###",  # 开始运行 ruff 的提示
        end="Problems detected by ruff",  # ruff 检测到问题的结束提示
        title="`ruff`",  # 标题为 `ruff`
        message=(
            "`ruff` detected issues. Please run "
            "`ruff check --fix --output-format=full .` locally, fix the remaining "
            "issues, and push the changes. Here you can see the detected issues. Note "
            f"that the installed `ruff` version is `ruff={versions['ruff']}`."  # 提示关于 ruff 检测到问题的消息，包括安装版本信息
        ),
        details=details,  # 细节信息参数
    )

    # 将运行 `mypy` 的步骤信息添加到消息中
    message += get_step_message(
        log,
        start="### Running mypy ###",  # 开始运行 mypy 的提示
        end="Problems detected by mypy",  # mypy 检测到问题的结束提示
        title="`mypy`",  # 标题为 `mypy`
        message=(
            "`mypy` detected issues. Please fix them locally and push the changes. "
            "Here you can see the detected issues. Note that the installed `mypy` "
            f"version is `mypy={versions['mypy']}`."  # 提示关于 mypy 检测到问题的消息，包括安装版本信息
        ),
        details=details,  # 细节信息参数
    )

    # 将运行 `cython-lint` 的步骤信息添加到消息中
    message += get_step_message(
        log,
        start="### Running cython-lint ###",  # 开始运行 cython-lint 的提示
        end="Problems detected by cython-lint",  # cython-lint 检测到问题的结束提示
        title="`cython-lint`",  # 标题为 `cython-lint`
        message=(
            "`cython-lint` detected issues. Please fix them locally and push "
            "the changes. Here you can see the detected issues. Note that the "
            "installed `cython-lint` version is "
            f"`cython-lint={versions['cython-lint']}`."  # 提示关于 cython-lint 检测到问题的消息，包括安装版本信息
        ),
        details=details,  # 细节信息参数
    )

    # 检查坏掉的弃用顺序的步骤信息添加到消息中
    message += get_step_message(
        log,
        start="### Checking for bad deprecation order ###",  # 开始检查坏掉的弃用顺序的提示
        end="Problems detected by deprecation order check",  # 弃用顺序检查检测到问题的结束提示
        title="Deprecation Order",  # 标题为 Deprecation Order
        message=(
            "Deprecation order check detected issues. Please fix them locally and "
            "push the changes. Here you can see the detected issues."  # 提示关于弃用顺序检查检测到问题的消息
        ),
        details=details,  # 细节信息参数
    )

    # 检查默认 doctest 指令的步骤信息添加到消息中
    message += get_step_message(
        log,
        start="### Checking for default doctest directives ###",  # 开始检查默认 doctest 指令的提示
        end="Problems detected by doctest directive check",  # doctest 指令检查检测到问题的结束提示
        title="Doctest Directives",  # 标题为 Doctest Directives
        message=(
            "doctest directive check detected issues. Please fix them locally and "
            "push the changes. Here you can see the detected issues."  # 提示关于 doctest 指令检查检测到问题的消息
        ),
        details=details,  # 细节信息参数
    )

    # joblib 导入
    # 将消息扩展为包含 lint 检查的详细信息
    message += get_step_message(
        log,
        start="### Checking for joblib imports ###",  # 设置日志检查的起始标记
        end="Problems detected by joblib import check",  # 设置日志检查的结束标记
        title="Joblib Imports",  # 设置日志检查的标题为 "Joblib Imports"
        message=(
            "`joblib` import check detected issues. Please fix them locally and "
            "push the changes. Here you can see the detected issues."  # 提供有关 joblib 检查问题的信息
        ),
        details=details,  # 将详细信息传递给消息函数
    )

    if not message:
        # 如果未检测到任何问题，则返回通过 lint 检查的消息
        return (
            "## ✔️ Linting Passed\n"
            "All linting checks passed. Your pull request is in excellent shape! ☀️"
            + sub_text  # 添加附加文本到 lint 通过的消息中
        )

    if not details:
        # 如果未提供详细信息，则说明日志发布失败，通常是因为日志过长或者 PR 分支未及时更新导致的
        branch_not_updated = (
            "_Merging with `upstream/main` might fix / improve the issues if you "
            "haven't done that since 21.06.2023._\n\n"
        )
    else:
        branch_not_updated = ""  # 如果提供了详细信息，则将未更新的分支信息设为空字符串

    message = (
        "## ❌ Linting issues\n\n"  # 设置 lint 问题的标题
        + branch_not_updated  # 添加分支未更新的信息（如果有）
        + "This PR is introducing linting issues. Here's a summary of the issues. "  # 提供 lint 问题的概要信息
        + "Note that you can avoid having linting issues by enabling `pre-commit` "  # 提醒可以通过启用 pre-commit hooks 避免 lint 问题
        + "hooks. Instructions to enable them can be found [here]("  # 提供启用 pre-commit hooks 的链接
        + "https://scikit-learn.org/dev/developers/contributing.html#how-to-contribute)"
        + ".\n\n"
        + "You can see the details of the linting issues under the `lint` job [here]"  # 提供 lint 问题详细信息的链接
        + f"(https://github.com/{repo}/actions/runs/{run_id})\n\n"
        + message  # 将 lint 检查消息的详细信息添加到总消息中
        + sub_text  # 添加附加文本到总消息中
    )

    return message  # 返回总消息
# 获取用于 GitHub API 的请求头信息
def get_headers(token):
    return {
        "Accept": "application/vnd.github+json",  # 接受 JSON 格式的响应
        "Authorization": f"Bearer {token}",  # 使用给定的 token 进行身份验证
        "X-GitHub-Api-Version": "2022-11-28",  # 指定 GitHub API 的版本日期
    }


# 查找 linting bot 的评论
def find_lint_bot_comments(repo, token, pr_number):
    """Get the comment from the linting bot."""
    # repo 的格式为 "org/repo"
    # API 文档: https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#list-issue-comments  # noqa
    response = requests.get(
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
        headers=get_headers(token),
    )
    response.raise_for_status()  # 如果请求失败，抛出异常
    all_comments = response.json()  # 解析 JSON 格式的响应

    failed_comment = "❌ Linting issues"  # 标识 linting 未通过的评论内容
    success_comment = "✔️ Linting Passed"  # 标识 linting 已通过的评论内容

    # 查找与 linting bot 相关的所有评论，并返回第一个匹配的评论
    # 应该只会有一个这样的评论，或者如果 PR 刚创建则可能一个也没有
    comments = [
        comment
        for comment in all_comments
        if comment["user"]["login"] == "github-actions[bot]"  # 评论由特定的 bot 用户创建
        and (failed_comment in comment["body"] or success_comment in comment["body"])  # 包含指定的评论内容
    ]

    if len(all_comments) > 25 and not comments:
        # 默认情况下 API 返回前 30 条评论。如果在这些评论中找不到 bot 的评论，则引发异常，
        # 从而跳过创建评论的步骤。
        raise RuntimeError("Comment not found in the first 30 comments.")

    return comments[0] if comments else None  # 返回找到的第一个评论，如果没有则返回 None


# 创建或更新评论
def create_or_update_comment(comment, message, repo, pr_number, token):
    """Create a new comment or update existing one."""
    # repo 的格式为 "org/repo"
    if comment is not None:
        print("updating existing comment")  # 更新现有评论的提示信息
        # API 文档: https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#update-an-issue-comment  # noqa
        response = requests.patch(
            f"https://api.github.com/repos/{repo}/issues/comments/{comment['id']}",
            headers=get_headers(token),  # 使用给定的 token 进行身份验证
            json={"body": message},  # 更新评论的内容
        )
    else:
        print("creating new comment")  # 创建新评论的提示信息
        # API 文档: https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#create-an-issue-comment  # noqa
        response = requests.post(
            f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
            headers=get_headers(token),  # 使用给定的 token 进行身份验证
            json={"body": message},  # 创建评论的内容
        )

    response.raise_for_status()  # 如果请求失败，抛出异常


if __name__ == "__main__":
    repo = os.environ["GITHUB_REPOSITORY"]  # 从环境变量中获取 GitHub 仓库信息
    token = os.environ["GITHUB_TOKEN"]  # 从环境变量中获取 GitHub API 的访问 token
    pr_number = os.environ["PR_NUMBER"]  # 从环境变量中获取 Pull Request 编号
    sha = os.environ["BRANCH_SHA"]  # 从环境变量中获取分支的 SHA 值
    log_file = os.environ["LOG_FILE"]  # 从环境变量中获取日志文件路径
    run_id = os.environ["RUN_ID"]  # 从环境变量中获取运行 ID
    versions_file = os.environ["VERSIONS_FILE"]  # 从环境变量中获取版本文件路径

    versions = get_versions(versions_file)  # 调用 get_versions 函数获取版本信息
    # 检查所需环境变量是否存在，若有任何一个缺失则抛出数值错误异常
    if not repo or not token or not pr_number or not log_file or not run_id:
        raise ValueError(
            "One of the following environment variables is not set: "
            "GITHUB_REPOSITORY, GITHUB_TOKEN, PR_NUMBER, LOG_FILE, RUN_ID"
        )

    try:
        # 尝试查找Lint Bot的评论
        comment = find_lint_bot_comments(repo, token, pr_number)
    except RuntimeError:
        # 如果在前30个评论中未找到评论，则输出跳过信息并退出程序
        print("Comment not found in the first 30 comments. Skipping!")
        exit(0)

    try:
        # 获取日志文件中的消息，包括详细信息和版本信息
        message = get_message(
            log_file,
            repo=repo,
            pr_number=pr_number,
            sha=sha,
            run_id=run_id,
            details=True,
            versions=versions,
        )
        # 创建或更新评论，将消息发送到GitHub上的PR页面
        create_or_update_comment(
            comment=comment,
            message=message,
            repo=repo,
            pr_number=pr_number,
            token=token,
        )
        # 打印消息到控制台
        print(message)
    except requests.HTTPError:
        # 如果消息过长导致请求失败，则尝试以不包含详细信息的方式再次获取消息
        message = get_message(
            log_file,
            repo=repo,
            pr_number=pr_number,
            sha=sha,
            run_id=run_id,
            details=False,
            versions=versions,
        )
        # 创建或更新评论，将简化后的消息发送到GitHub上的PR页面
        create_or_update_comment(
            comment=comment,
            message=message,
            repo=repo,
            pr_number=pr_number,
            token=token,
        )
        # 打印消息到控制台
        print(message)
```