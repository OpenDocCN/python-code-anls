# `D:\src\scipysrc\scikit-learn\build_tools\azure\get_commit_message.py`

```
# 导入必要的模块 argparse, os, subprocess
import argparse
import os
import subprocess

# 从环境变量中获取构建源版本的消息作为提交消息
def get_commit_message():
    build_source_version_message = os.environ["BUILD_SOURCEVERSIONMESSAGE"]

    # 如果构建原因是 PullRequest
    if os.environ["BUILD_REASON"] == "PullRequest":
        # 默认情况下，拉取请求使用 refs/pull/PULL_ID/merge 作为源分支，
        # 其中包含 "Merge ID into ID" 作为提交消息。最新的提交消息是倒数第二个提交。
        commit_id = build_source_version_message.split()[1]
        git_cmd = ["git", "log", commit_id, "-1", "--pretty=%B"]
        # 运行 git 命令获取提交消息
        commit_message = subprocess.run(
            git_cmd, capture_output=True, text=True
        ).stdout.strip()
    else:
        commit_message = build_source_version_message

    # 清理提交消息以避免引入漏洞：PR 提交者可以在提交消息中包含 "##vso" 特殊标记，
    # 试图混淆注入到 Azure 流水线中的任意命令。
    # 在受保护的分支上，Azure 已经对 `BUILD_SOURCEVERSIONMESSAGE` 进行了清理，
    # 但出于预防，这里仍然对消息进行了清理。
    commit_message = commit_message.replace("##vso", "..vso")

    return commit_message


# 解析命令行参数
def parsed_args():
    parser = argparse.ArgumentParser(
        description=(
            "Show commit message that triggered the build in Azure DevOps pipeline"
        )
    )
    parser.add_argument(
        "--only-show-message",
        action="store_true",
        default=False,
        help=(
            "Only print commit message. Useful for direct use in scripts rather than"
            " setting output variable of the Azure job"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parsed_args()
    commit_message = get_commit_message()

    if args.only_show_message:
        # 如果只需要显示提交消息，则打印提交消息
        print(commit_message)
    else:
        # 设置要传播到其他步骤的环境变量
        print(f"##vso[task.setvariable variable=message;isOutput=true]{commit_message}")
        # 输出提交消息，用于调试
        print(f"commit message: {commit_message}")
```