# `.\commands\user.py`

```py
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from argparse import ArgumentParser
from typing import List, Union

from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError

from . import BaseTransformersCLICommand


class UserCommands(BaseTransformersCLICommand):
    @staticmethod
    # 静态方法：注册子命令到给定的参数解析器
    def register_subcommand(parser: ArgumentParser):
        # 添加登录子命令解析器，用于登录到huggingface.co
        login_parser = parser.add_parser("login", help="Log in using the same credentials as on huggingface.co")
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        
        # 添加whoami子命令解析器，用于查看当前登录的huggingface.co账户
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        
        # 添加登出子命令解析器，用于退出登录
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))

        # 新系统：基于git的存储库系统
        # 添加repo子命令解析器，与huggingface.co存储库交互的命令集（已弃用）
        repo_parser = parser.add_parser(
            "repo",
            help="Deprecated: use `huggingface-cli` instead. Commands to interact with your huggingface.co repos.",
        )
        # repo子命令的子解析器集合，与huggingface.co存储库相关的命令（已弃用）
        repo_subparsers = repo_parser.add_subparsers(
            help="Deprecated: use `huggingface-cli` instead. huggingface.co repos related commands"
        )
        
        # 添加create子命令解析器，创建一个新的huggingface.co存储库（已弃用）
        repo_create_parser = repo_subparsers.add_parser(
            "create", help="Deprecated: use `huggingface-cli` instead. Create a new repo on huggingface.co"
        )
        # create子命令解析器的参数设置：存储库的名称，将被命名空间化在您的用户名下以构建模型ID
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your model's repo. Will be namespaced under your username to build the model id.",
        )
        # 可选参数：组织命名空间
        repo_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        # 可选参数：对提示回答是
        repo_create_parser.add_argument("-y", "--yes", action="store_true", help="Optional: answer Yes to the prompt")
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _red = "\u001b[31m"
    _gray = "\u001b[90m"
    _reset = "\u001b[0m"

    @classmethod
    # 类方法：使文本加粗
    def bold(cls, s):
        return f"{cls._bold}{s}{cls._reset}"

    @classmethod
    # 类方法：使文本显示为红色
    def red(cls, s):
        return f"{cls._bold}{cls._red}{s}{cls._reset}"
    # 类方法 `gray`，接受一个字符串 `s`，返回一个包含灰色文本的字符串
    def gray(cls, s):
        return f"{cls._gray}{s}{cls._reset}"
# 定义一个函数 tabulate，用于将二维列表按表格格式输出为字符串
def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    # 计算每列的最大宽度，包括表头
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    # 根据列宽度创建格式化字符串，用于格式化每行数据
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    # 初始化输出行列表
    lines = []
    # 添加表头行
    lines.append(row_format.format(*headers))
    # 添加分隔线行
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    # 遍历每行数据，格式化后添加到输出行列表
    for row in rows:
        lines.append(row_format.format(*row))
    # 将所有行拼接成一个字符串，用换行符连接
    return "\n".join(lines)


class BaseUserCommand:
    def __init__(self, args):
        self.args = args


class LoginCommand(BaseUserCommand):
    def run(self):
        # 打印红色警告信息，指出登录命令已过时
        print(
            ANSI.red(
                "ERROR! `huggingface-cli login` uses an outdated login mechanism "
                "that is not compatible with the Hugging Face Hub backend anymore. "
                "Please use `huggingface-cli login instead."
            )
        )


class WhoamiCommand(BaseUserCommand):
    def run(self):
        # 打印红色警告信息，指出 whoami 命令已过时
        print(
            ANSI.red(
                "WARNING! `transformers-cli whoami` is deprecated and will be removed in v5. Please use "
                "`huggingface-cli whoami` instead."
            )
        )
        # 获取用户 token
        token = HfFolder.get_token()
        # 如果 token 为空，则打印未登录并退出程序
        if token is None:
            print("Not logged in")
            exit()
        try:
            # 调用 whoami 函数获取用户和组织信息
            user, orgs = whoami(token)
            # 打印用户信息
            print(user)
            # 如果有组织信息，则打印组织信息
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))
        except HTTPError as e:
            # 捕获 HTTPError 异常，打印异常信息和响应内容，并退出程序
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)


class LogoutCommand(BaseUserCommand):
    def run(self):
        # 打印红色警告信息，指出注销命令已过时
        print(
            ANSI.red(
                "ERROR! `transformers-cli logout` uses an outdated logout mechanism "
                "that is not compatible with the Hugging Face Hub backend anymore. "
                "Please use `huggingface-cli logout instead."
            )
        )


class RepoCreateCommand(BaseUserCommand):
    def run(self):
        # 打印警告信息，提示通过 transformers-cli 管理仓库已被弃用，建议使用 `huggingface-cli`
        print(
            ANSI.red(
                "WARNING! Managing repositories through transformers-cli is deprecated. "
                "Please use `huggingface-cli` instead."
            )
        )
        # 获取用户的令牌
        token = HfFolder.get_token()
        # 如果未获取到令牌，打印未登录信息，并退出程序
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            # 检查并获取当前安装的 git 版本信息
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            # 如果未找到 git 命令，提示用户未安装 git
            print("Looks like you do not have git installed, please install.")

        try:
            # 检查并获取当前安装的 git-lfs 版本信息
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            # 如果未找到 git-lfs 命令，提示用户未安装 git-lfs，并提供安装指南
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

        # 获取当前登录用户信息
        user, _ = whoami(token)
        # 确定命名空间，可以是命令行参数中指定的组织，也可以是当前用户
        namespace = self.args.organization if self.args.organization is not None else user
        # 组装完整的仓库名称
        full_name = f"{namespace}/{self.args.name}"
        # 打印即将创建的仓库名称，使用 ANSI 加粗样式
        print(f"You are about to create {ANSI.bold(full_name)}")

        # 如果不是自动确认模式，询问用户是否继续
        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                # 如果用户选择不继续，打印中止信息并退出程序
                print("Abort")
                exit()
        try:
            # 创建仓库，并获取返回的仓库 URL
            url = create_repo(token, name=self.args.name, organization=self.args.organization)
        except HTTPError as e:
            # 如果发生 HTTP 错误，打印错误信息和响应内容，并退出程序
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        # 打印创建成功后的仓库 URL
        print("\nYour repo now lives at:")
        print(f"  {ANSI.bold(url)}")
        # 提示用户可以通过克隆命令将仓库克隆到本地，并正常进行提交和推送操作
        print("\nYou can clone it locally with the command below, and commit/push as usual.")
        print(f"\n  git clone {url}")
        print("")
```