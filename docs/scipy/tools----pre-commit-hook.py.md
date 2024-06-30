# `D:\src\scipysrc\scipy\tools\pre-commit-hook.py`

```
#!/usr/bin/env python
#
# Pre-commit linting hook.
#
# Install from root of repository with:
#
#   cp tools/pre-commit-hook.py .git/hooks/pre-commit

# 导入必要的模块
import subprocess
import sys
import os


# Run lint.py from the scipy source tree
# 定义lint.py文件的路径列表，优先级从高到低寻找存在的文件
linters = [
    '../../tools/lint.py',
    'tools/lint.py',
    'lint.py'  # in case pre-commit hook is run from tools dir
]

# 选择第一个存在的lint.py文件作为linter的路径
linter = [f for f in linters if os.path.exists(f)][0]

# Run unicode-check.py for checking unicode issues
# 定义unicode-check.py文件的路径列表，优先级从高到低寻找存在的文件
unicode_checks = [
    '../../tools/unicode-check.py',
    'tools/unicode-check.py',
    'unicode-check.py'  # in case pre-commit hook is run from tools dir
]

# 选择第一个存在的unicode-check.py文件作为unicode_check的路径
unicode_check = [f for f in unicode_checks if os.path.exists(f)][0]


# names of files that were staged
# 获取已暂存文件的文件名列表，限定文件类型为*.py和*.pyx
# 注意使用了subprocess.run执行git命令来获取输出结果
p = subprocess.run(['git', 'diff',
                    '--cached', '--name-only', '-z',
                    '--diff-filter=ACMR',
                    '--', '*.py', '*.pyx'],
                   capture_output=True, check=True)

# 解析git diff命令的输出，将文件名存入files列表
files = p.stdout.decode(sys.getfilesystemencoding()).split('\0')
files = [f for f in files if f]

# create a temporary copy of what would get committed, without unstaged
# modifications (e.g., only certain changes in a file may have been committed)
# 创建预提交时的工作目录的临时副本，仅包含将要提交的更改
# 使用git write-tree和git commit-tree命令来生成树对象和虚拟提交对象
git_dir = os.environ.get('GIT_DIR', '.git')
work_dir = os.path.join(git_dir, '.pre-commit-work_dir')

p = subprocess.run(['git', 'write-tree'], capture_output=True, check=True)
tree_hash = p.stdout.decode('ascii').split('\n')[0]

p = subprocess.run(['git', 'commit-tree', '-p', 'HEAD',
                    tree_hash, '-m', '...'], capture_output=True, check=True)
fake_commit = p.stdout.decode('ascii').split('\n')[0]

# 如果预提交工作目录不存在，则从本地克隆一个
if not os.path.isdir(work_dir):
    subprocess.run(['git', 'clone', '-qns', git_dir, work_dir])

# 重设工作目录到HEAD状态，并checkout到虚拟提交对象
subprocess.run(['git', 'reset', '--quiet', '--hard', 'HEAD'],
               env={}, cwd=work_dir, check=True)
subprocess.run(['git', 'checkout', '-q', fake_commit],
               env={}, cwd=work_dir, check=True)
subprocess.run(['git', 'reset', '--quiet', '--hard', fake_commit],
               env={}, cwd=work_dir, check=True)


if '--fix' in sys.argv:
    # 如果命令行参数包含--fix，则运行linter修复错误
    print('Running linter to fix errors...')
    p = subprocess.run([linter, '--fix', '--files'] + files)

    # Discover which files were modified
    # 检查哪些文件被修改了，并打印出来
    p = subprocess.run([linter, '--fix', '--files'] + files, cwd=work_dir)
    p = subprocess.run(['git', 'diff', '--name-only', '--', '*.py', '*.pyx'],
                       capture_output=True, check=True, cwd=work_dir)
    files = p.stdout.decode(sys.getfilesystemencoding()).split('\0')
    files = [f for f in files if f]
    if files:
        print('The following files were modified:')
        print()
        print('\n'.join(files))
    else:
        print('No files were modified.\n')

    # 提示用户记得使用git add添加修改后的文件
    print('Please remember to `git add` modified files.')
    sys.exit(p.returncode)


# 如果没有--fix参数，则运行linter检查文件
p = subprocess.run([linter, '--files'] + files, cwd=work_dir)

if p.returncode != 0:
    # 如果linter返回非零值，提示linting失败，需要修复错误后重新提交
    print('!! Linting failed; please fix errors, `git add` files, and re-commit.')
    print()
    # 打印一条消息，指出可能可以通过运行特定的命令来自动修复一些错误
    print('Some errors may be fixable automatically by running:')
    
    # 打印空行
    print()
    
    # 打印另一条消息，指出可以运行哪个命令来修复错误
    print('  ./tools/pre-commit-hook.py --fix')
    
    # 使用 subprocess 模块中的 sys.exit 函数退出程序，并返回子进程的返回码作为退出码
    sys.exit(p.returncode)
# 运行外部进程来执行 unicode_check 命令，当前工作目录设置为 work_dir
p = subprocess.run(unicode_check, cwd=work_dir)

# 检查子进程的返回码，如果返回码不为 0，表示命令执行失败，直接退出程序并返回子进程的返回码
if p.returncode != 0:
    sys.exit(p.returncode)
```