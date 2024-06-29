# `.\numpy\tools\linter.py`

```
import os
import sys
import subprocess
from argparse import ArgumentParser
from git import Repo, exc

CONFIG = os.path.join(
         os.path.abspath(os.path.dirname(__file__)),
         'lint_diff.ini',
)
# 配置文件的路径，用于指定 pycodestyle 的配置

# NOTE: The `diff` and `exclude` options of pycodestyle seem to be
# incompatible, so instead just exclude the necessary files when
# computing the diff itself.
EXCLUDE = (
    "numpy/typing/tests/data/",
    "numpy/typing/_char_codes.py",
    "numpy/__config__.py",
    "numpy/f2py",
)
# 需要在差异计算时排除的文件列表

class DiffLinter:
    def __init__(self, branch):
        self.branch = branch
        self.repo = Repo('.')
        self.head = self.repo.head.commit

    def get_branch_diff(self, uncommitted = False):
        """
            Determine the first common ancestor commit.
            Find diff between branch and FCA commit.
            Note: if `uncommitted` is set, check only
                  uncommitted changes
        """
        try:
            commit = self.repo.merge_base(self.branch, self.head)[0]
        except exc.GitCommandError:
            print(f"Branch with name `{self.branch}` does not exist")
            sys.exit(1)
        
        # 构建排除文件列表，以便在差异计算时使用
        exclude = [f':(exclude){i}' for i in EXCLUDE]
        if uncommitted:
            # 如果只检查未提交的更改，则使用当前头部与排除文件列表来计算差异
            diff = self.repo.git.diff(
                self.head, '--unified=0', '***.py', *exclude
            )
        else:
            # 否则，使用合并基础与当前头部以及排除文件列表来计算差异
            diff = self.repo.git.diff(
                commit, self.head, '--unified=0', '***.py', *exclude
            )
        return diff

    def run_pycodestyle(self, diff):
        """
            Original Author: Josh Wilson (@person142)
            Source:
              https://github.com/scipy/scipy/blob/main/tools/lint_diff.py
            Run pycodestyle on the given diff.
        """
        # 运行 pycodestyle 来检查给定的差异内容
        res = subprocess.run(
            ['pycodestyle', '--diff', '--config', CONFIG],
            input=diff,
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        return res.returncode, res.stdout

    def run_lint(self, uncommitted):
        # 获取差异内容
        diff = self.get_branch_diff(uncommitted)
        # 运行 pycodestyle 来检查差异
        retcode, errors = self.run_pycodestyle(diff)

        # 如果有错误则打印出来
        errors and print(errors)

        # 根据返回码退出程序
        sys.exit(retcode)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--branch", type=str, default='main',
                        help="The branch to diff against")
    parser.add_argument("--uncommitted", action='store_true',
                        help="Check only uncommitted changes")
    args = parser.parse_args()

    # 创建 DiffLinter 实例并运行 lint 检查
    DiffLinter(args.branch).run_lint(args.uncommitted)
```