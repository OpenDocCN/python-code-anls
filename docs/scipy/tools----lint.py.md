# `D:\src\scipysrc\scipy\tools\lint.py`

```
#!/usr/bin/env python
import os  # 导入操作系统相关的功能
import sys  # 导入系统相关的功能
import subprocess  # 导入子进程管理相关的功能
import packaging.version  # 导入版本管理相关的功能
from argparse import ArgumentParser  # 从argparse模块中导入ArgumentParser类


CONFIG = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'lint.toml',  # 设置配置文件路径为当前脚本所在目录下的lint.toml
)


def rev_list(branch, num_commits):
    """List commits in reverse chronological order.

    Only the first `num_commits` are shown.

    """
    # 运行git命令获取指定分支的最近num_commits个提交的哈希值列表
    res = subprocess.run(
        [
            'git',
            'rev-list',
            '--max-count',
            f'{num_commits}',
            '--first-parent',
            branch
        ],
        stdout=subprocess.PIPE,  # 将标准输出捕获到PIPE中
        encoding='utf-8',  # 指定输出的编码格式为utf-8
    )
    res.check_returncode()  # 检查命令执行的返回码，若不为0则抛出异常
    return res.stdout.rstrip('\n').split('\n')  # 返回去掉末尾换行符后的输出，转换为列表形式


def find_branch_point(branch):
    """Find when the current branch split off from the given branch.

    It is based off of this Stackoverflow post:

    https://stackoverflow.com/questions/1527234/finding-a-branch-point-with-git#4991675

    """
    branch_commits = rev_list('HEAD', 1000)  # 获取当前分支最近1000个提交的哈希值列表
    main_commits = set(rev_list(branch, 1000))  # 获取指定分支最近1000个提交的哈希值集合
    for branch_commit in branch_commits:
        if branch_commit in main_commits:  # 遍历当前分支的提交，找到与指定分支有共同提交的分支点
            return branch_commit

    # 如果当前分支与指定分支在最近的1000个提交中没有共同的提交点，则抛出异常
    raise RuntimeError(
        'Failed to find a common ancestor in the last 1000 commits'
    )


def diff_files(sha):
    """Find the diff since the given SHA."""
    # 运行git命令获取自指定SHA以来修改的文件列表
    res = subprocess.run(
        ['git', 'diff', '--name-only', '--diff-filter=ACMR', '-z', sha, '--',
         '*.py', '*.pyx', '*.pxd', '*.pxi'],
        stdout=subprocess.PIPE,  # 将标准输出捕获到PIPE中
        encoding='utf-8'  # 指定输出的编码格式为utf-8
    )
    res.check_returncode()  # 检查命令执行的返回码，若不为0则抛出异常
    return [f for f in res.stdout.split('\0') if f]  # 返回去掉空字符串的文件列表


def run_ruff(files, fix):
    if not files:
        return 0, ""
    args = ['--fix', '--exit-non-zero-on-fix'] if fix else []  # 根据fix参数设置命令行参数列表
    # 运行ruff命令进行代码检查，指定配置文件和其他参数
    res = subprocess.run(
        ['ruff', 'check', f'--config={CONFIG}'] + args + list(files),
        stdout=subprocess.PIPE,  # 将标准输出捕获到PIPE中
        encoding='utf-8'  # 指定输出的编码格式为utf-8
    )
    return res.returncode, res.stdout  # 返回命令执行的返回码和标准输出内容


def run_cython_lint(files):
    if not files:
        return 0, ""
    # 运行cython-lint命令进行Cython代码检查，不包含pycodestyle检查
    res = subprocess.run(
        ['cython-lint', '--no-pycodestyle'] + list(files),
        stdout=subprocess.PIPE,  # 将标准输出捕获到PIPE中
        encoding='utf-8'  # 指定输出的编码格式为utf-8
    )
    return res.returncode, res.stdout  # 返回命令执行的返回码和标准输出内容


def check_ruff_version():
    min_version = packaging.version.parse('0.0.292')  # 设置最低兼容版本号
    # 运行ruff命令获取当前版本号
    res = subprocess.run(
        ['ruff', '--version'],
        stdout=subprocess.PIPE,  # 将标准输出捕获到PIPE中
        encoding='utf-8'  # 指定输出的编码格式为utf-8
    )
    version = res.stdout.replace('ruff ', '')  # 从输出中提取版本号
    if packaging.version.parse(version) < min_version:  # 检查版本号是否低于最低兼容版本
        raise RuntimeError("Linting requires `ruff>=0.0.292`. Please upgrade `ruff`.")


def main():
    check_ruff_version()  # 检查ruff的版本要求
    parser = ArgumentParser(description="Also see `pre-commit-hook.py` which "
                                        "lints all files staged in git.")
    # In Python 3.9, can use: argparse.BooleanOptionalAction
    # 添加一个名为 '--fix' 的命令行参数，如果指定则尝试修复代码中的 linting 违规
    parser.add_argument("--fix", action='store_true',
                        help='Attempt to fix linting violations')
    
    # 添加一个名为 '--diff-against' 的命令行参数，用于指定与之比较差异并对修改的文件进行 lint 检查
    parser.add_argument("--diff-against", dest='branch',
                        type=str, default=None,
                        help="Diff against "
                             "this branch and lint modified files. Use either "
                             "`--diff-against` or `--files`, but not both.")
    
    # 添加一个名为 '--files' 的命令行参数，指定需要进行 lint 检查的文件或目录
    parser.add_argument("--files", nargs='*',
                        help="Lint these files or directories; "
                             "use **/*.py to lint all files")
    
    # 解析命令行参数并将其存储到 args 变量中
    args = parser.parse_args()
    
    # 检查 '--diff-against' 和 '--files' 是否同时指定，若同时指定则报错并退出
    if not ((args.files is None) ^ (args.branch is None)):
        print('Specify either `--diff-against` or `--files`. Aborting.')
        sys.exit(1)
    
    # 如果指定了 '--diff-against' 参数，则查找该分支点并获取修改的文件列表
    if args.branch:
        branch_point = find_branch_point(args.branch)
        files = diff_files(branch_point)
    else:
        # 否则直接使用 '--files' 参数指定的文件列表
        files = args.files
    
    # 定义 Cython 文件的扩展名集合
    cython_exts = ('.pyx', '.pxd', '.pxi')
    # 从文件列表中筛选出所有属于 Cython 文件的文件名，并存储到集合 cython_files 中
    cython_files = {f for f in files if any(f.endswith(ext) for ext in cython_exts)}
    # 计算非 Cython 文件的文件名集合
    other_files = set(files) - cython_files
    
    # 运行 Cython lint 检查，返回 lint 结果代码 rc_cy 和可能的错误信息列表 errors
    rc_cy, errors = run_cython_lint(cython_files)
    # 如果有错误信息则打印出来
    if errors:
        print(errors)
    
    # 运行 Ruff lint 检查，返回 lint 结果代码 rc 和可能的错误信息列表 errors，同时尝试修复 lint 违规根据参数 '--fix'
    rc, errors = run_ruff(other_files, fix=args.fix)
    # 如果有错误信息则打印出来
    if errors:
        print(errors)
    
    # 如果 Ruff lint 的返回代码 rc 是成功的（0），但 Cython lint 的返回代码 rc_cy 不是成功的（非0），则将 rc 更新为 rc_cy
    if rc == 0 and rc_cy != 0:
        rc = rc_cy
    
    # 退出程序，返回 lint 检查的最终结果代码 rc
    sys.exit(rc)
# 如果当前模块被直接运行（而不是被导入到其他模块），则执行以下代码块
if __name__ == '__main__':
    # 调用主函数 main()
    main()
```