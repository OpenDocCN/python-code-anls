# `.\numpy\tools\ci\push_docs_to_repo.py`

```py
#!/usr/bin/env python3

import argparse  # 导入用于解析命令行参数的模块
import subprocess  # 导入用于执行外部命令的模块
import tempfile  # 导入用于创建临时文件和目录的模块
import os  # 导入操作系统相关功能的模块
import sys  # 导入系统相关功能的模块
import shutil  # 导入高级文件操作模块，用于文件复制和删除


parser = argparse.ArgumentParser(
    description='Upload files to a remote repo, replacing existing content'
)
parser.add_argument('dir', help='directory of which content will be uploaded')  # 解析要上传内容的目录参数
parser.add_argument('remote', help='remote to which content will be pushed')  # 解析远程仓库的参数
parser.add_argument('--message', default='Commit bot upload',
                    help='commit message to use')  # 解析提交信息的参数，默认为"Commit bot upload"
parser.add_argument('--committer', default='numpy-commit-bot',
                    help='Name of the git committer')  # 解析提交者姓名的参数，默认为"numpy-commit-bot"
parser.add_argument('--email', default='numpy-commit-bot@nomail',
                    help='Email of the git committer')  # 解析提交者邮箱的参数，默认为"numpy-commit-bot@nomail"
parser.add_argument('--count', default=1, type=int,
                    help="minimum number of expected files, defaults to 1")  # 解析期望上传的文件最小数量，默认为1

parser.add_argument(
    '--force', action='store_true',
    help='hereby acknowledge that remote repo content will be overwritten'
)  # 解析是否强制覆盖远程仓库内容的参数
args = parser.parse_args()
args.dir = os.path.abspath(args.dir)  # 获取绝对路径以确保目录存在

if not os.path.exists(args.dir):  # 检查目录是否存在
    print('Content directory does not exist')  # 输出提示信息
    sys.exit(1)  # 如果目录不存在，退出程序并返回错误码1

count = len([name for name in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, name))])  # 统计目录下的文件数量

if count < args.count:  # 检查文件数量是否达到预期值
    print(f"Expected {args.count} top-directory files to upload, got {count}")  # 输出文件数量不符的提示信息
    sys.exit(1)  # 如果文件数量不符，退出程序并返回错误码1

def run(cmd, stdout=True):  # 定义运行外部命令的函数
    pipe = None if stdout else subprocess.DEVNULL  # 根据stdout参数确定是否需要输出命令结果
    try:
        subprocess.check_call(cmd, stdout=pipe, stderr=pipe)  # 执行命令
    except subprocess.CalledProcessError:
        print("\n! Error executing: `%s;` aborting" % ' '.join(cmd))  # 输出命令执行错误的信息
        sys.exit(1)  # 如果命令执行错误，退出程序并返回错误码1

workdir = tempfile.mkdtemp()  # 创建临时工作目录
os.chdir(workdir)  # 切换工作目录到临时目录

run(['git', 'init'])  # 初始化 Git 仓库
# 确保工作分支命名为 "main"
# （在旧版本的 Git 上，`--initial-branch=main` 可能会失败）:
run(['git', 'checkout', '-b', 'main'])  # 创建并切换到名为 "main" 的分支
run(['git', 'remote', 'add', 'origin',  args.remote])  # 添加远程仓库地址
run(['git', 'config', '--local', 'user.name', args.committer])  # 设置本地 Git 用户名
run(['git', 'config', '--local', 'user.email', args.email])  # 设置本地 Git 用户邮箱

print('- committing new content: "%s"' % args.message)  # 输出正在提交新内容的信息
run(['cp', '-R', os.path.join(args.dir, '.'), '.'])  # 复制要上传的内容到当前工作目录
run(['git', 'add', '.'], stdout=False)  # 添加所有修改到 Git 暂存区
run(['git', 'commit', '--allow-empty', '-m', args.message], stdout=False)  # 提交更改到 Git 仓库

print('- uploading as %s <%s>' % (args.committer, args.email))  # 输出正在以提交者身份上传的信息
if args.force:
    run(['git', 'push', 'origin', 'main', '--force'])  # 强制推送更改到远程仓库
else:
    print('\n!! No `--force` argument specified; aborting')  # 输出没有指定 `--force` 参数的警告信息
    print('!! Before enabling that flag, make sure you know what it does\n')  # 输出在启用该标志之前，请确保了解其功能的建议
    sys.exit(1)  # 如果没有指定 `--force` 参数，退出程序并返回错误码1

shutil.rmtree(workdir)  # 删除临时工作目录
```