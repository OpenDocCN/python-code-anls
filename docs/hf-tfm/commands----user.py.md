# `.\transformers\commands\user.py`

```
# 引入 subprocess 模块，用于执行外部命令
import subprocess
# 从 argparse 模块中引入 ArgumentParser 类，用于解析命令行参数
from argparse import ArgumentParser
# 从 typing 模块中引入 List 和 Union 类型，用于类型提示
from typing import List, Union
# 从 huggingface_hub.hf_api 模块中引入 HfFolder、create_repo 和 whoami 函数
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
# 从 requests.exceptions 模块中引入 HTTPError 异常类
from requests.exceptions import HTTPError
# 从当前目录中引入 BaseTransformersCLICommand 类
from . import BaseTransformersCLICommand

# 定义 UserCommands 类，继承自 BaseTransformersCLICommand 类
class UserCommands(BaseTransformersCLICommand):
    # 静态方法：注册子命令到命令行解析器
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # 添加登录子命令到命令行解析器
        login_parser = parser.add_parser("login", help="Log in using the same credentials as on huggingface.co")
        # 设置登录子命令的默认操作为创建 LoginCommand 实例并执行
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        # 添加 whoami 子命令到命令行解析器
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        # 设置 whoami 子命令的默认操作为创建 WhoamiCommand 实例并执行
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        # 添加注销子命令到命令行解析器
        logout_parser = parser.add_parser("logout", help="Log out")
        # 设置注销子命令的默认操作为创建 LogoutCommand 实例并执行
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))

        # 创建 repo 子命令，用于与 huggingface.co 仓库进行交互
        repo_parser = parser.add_parser(
            "repo",
            help="Deprecated: use `huggingface-cli` instead. Commands to interact with your huggingface.co repos.",
        )
        # 添加 repo 子命令的子解析器
        repo_subparsers = repo_parser.add_subparsers(
            help="Deprecated: use `huggingface-cli` instead. huggingface.co repos related commands"
        )
        # 添加创建仓库子命令到 repo 子命令的子解析器中
        repo_create_parser = repo_subparsers.add_parser(
            "create", help="Deprecated: use `huggingface-cli` instead. Create a new repo on huggingface.co"
        )
        # 添加创建仓库子命令的参数
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your model's repo. Will be namespaced under your username to build the model id.",
        )
        repo_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_create_parser.add_argument("-y", "--yes", action="store_true", help="Optional: answer Yes to the prompt")
        # 设置创建仓库子命令的默认操作为创建 RepoCreateCommand 实例并执行
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))

# 定义 ANSI 类，用于处理 ANSI 转义码
class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    # ANSI 转义码：加粗
    _bold = "\u001b[1m"
    # ANSI 转义码：红色
    _red = "\u001b[31m"
    # ANSI 转义码：灰色
    _gray = "\u001b[90m"
    # ANSI 转义码：重置
    _reset = "\u001b[0m"

    # 类方法：返回文本加粗格式的字符串
    @classmethod
    def bold(cls, s):
        return f"{cls._bold}{s}{cls._reset}"

    # 类方法：返回文本红色加粗格式的字符串
    @classmethod
    def red(cls, s):
        return f"{cls._bold}{cls._red}{s}{cls._reset}"
    # 定义一个静态方法，用于将输入的字符串添加灰色效果
    def gray(cls, s):
        # 返回带有灰色效果的字符串
        return f"{cls._gray}{s}{cls._reset}"
# 定义一个函数用于将给定的列表数据以表格形式输出为字符串
def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    # 计算每列的最大宽度
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    # 根据列宽度创建格式化字符串
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    # 存储表格每一行的字符串
    lines = []
    # 添加表头行
    lines.append(row_format.format(*headers))
    # 添加表格分隔行
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    # 添加每一行数据
    for row in rows:
        lines.append(row_format.format(*row))
    # 将所有行组合成一个字符串，以换行符连接
    return "\n".join(lines)


# 定义一个基础用户命令类
class BaseUserCommand:
    # 初始化方法，接收参数 args
    def __init__(self, args):
        self.args = args


# 定义登录命令类，继承自基础用户命令类
class LoginCommand(BaseUserCommand):
    # 定义运行方法
    def run(self):
        # 打印错误信息，使用 ANSI 控制字符设置颜色
        print(
            ANSI.red(
                "ERROR! `huggingface-cli login` uses an outdated login mechanism "
                "that is not compatible with the Hugging Face Hub backend anymore. "
                "Please use `huggingface-cli login instead."
            )
        )


# 定义查询当前用户信息命令类，继承自基础用户命令类
class WhoamiCommand(BaseUserCommand):
    # 定义运行方法
    def run(self):
        # 打印警告信息，使用 ANSI 控制字符设置颜色
        print(
            ANSI.red(
                "WARNING! `transformers-cli whoami` is deprecated and will be removed in v5. Please use "
                "`huggingface-cli whoami` instead."
            )
        )
        # 获取用户 token
        token = HfFolder.get_token()
        # 如果 token 为空
        if token is None:
            # 打印未登录信息
            print("Not logged in")
            # 退出程序
            exit()
        try:
            # 尝试获取用户信息和组织信息
            user, orgs = whoami(token)
            # 打印用户信息
            print(user)
            # 如果存在组织信息
            if orgs:
                # 打印组织信息
                print(ANSI.bold("orgs: "), ",".join(orgs))
        # 捕获 HTTP 错误
        except HTTPError as e:
            # 打印错误信息
            print(e)
            # 打印错误响应文本，使用 ANSI 控制字符设置颜色
            print(ANSI.red(e.response.text))
            # 退出程序，返回错误状态码
            exit(1)


# 定义登出命令类，继承自基础用户命令类
class LogoutCommand(BaseUserCommand):
    # 定义运行方法
    def run(self):
        # 打印错误信息，使用 ANSI 控制字符设置颜色
        print(
            ANSI.red(
                "ERROR! `transformers-cli logout` uses an outdated logout mechanism "
                "that is not compatible with the Hugging Face Hub backend anymore. "
                "Please use `huggingface-cli logout instead."
            )
        )


# 定义创建仓库命令类，继承自基础用户命令类
class RepoCreateCommand(BaseUserCommand):
    # 定义一个方法，用于执行特定操作
    def run(self):
        # 打印警告信息，指示通过transformers-cli管理仓库已经被弃用，建议使用`huggingface-cli`
        print(
            ANSI.red(
                "WARNING! Managing repositories through transformers-cli is deprecated. "
                "Please use `huggingface-cli` instead."
            )
        )
        # 获取用户的token
        token = HfFolder.get_token()
        # 如果token为None，打印未登录信息，然后退出
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            # 检查系统是否安装了git，并获取其版本信息
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            # 打印git版本信息
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            # 如果未找到git命令，则打印未安装git的信息
            print("Looks like you do not have git installed, please install.")

        try:
            # 检查系统是否安装了git-lfs，并获取其版本信息
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            # 打印git-lfs版本信息
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            # 如果未找到git-lfs命令，则打印未安装git-lfs的信息
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

        # 获取当前用户信息
        user, _ = whoami(token)
        # 如果指定了组织名，则使用组织名，否则使用用户自身名字
        namespace = self.args.organization if self.args.organization is not None else user
        # 构建仓库的完整名称
        full_name = f"{namespace}/{self.args.name}"
        # 打印即将创建的仓库名称
        print(f"You are about to create {ANSI.bold(full_name)}")

        # 如果没有指定确认标志，则询问用户是否继续
        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            # 如果用户选择不继续，则打印终止信息并退出
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            # 创建仓库，并获取其URL
            url = create_repo(token, name=self.args.name, organization=self.args.organization)
        except HTTPError as e:
            # 如果创建失败，则打印错误信息并退出
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        # 打印仓库URL信息
        print("\nYour repo now lives at:")
        print(f"  {ANSI.bold(url)}")
        # 打印克隆仓库的命令，以及提交/推送的提示信息
        print("\nYou can clone it locally with the command below, and commit/push as usual.")
        print(f"\n  git clone {url}")
        print("")
```