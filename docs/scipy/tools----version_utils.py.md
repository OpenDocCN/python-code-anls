# `D:\src\scipysrc\scipy\tools\version_utils.py`

```
#!/usr/bin/env python3
# 导入必要的库
import os                 # 导入操作系统接口模块
import subprocess         # 导入子进程管理模块
import argparse           # 导入命令行参数解析模块


# 定义版本号相关的常量
MAJOR = 1
MINOR = 15
MICRO = 0
ISRELEASED = False
IS_RELEASE_BRANCH = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# 获取版本信息函数
def get_version_info(source_root):
    # 设置基本版本号
    FULLVERSION = VERSION
    # 如果存在.git目录，则获取Git版本信息
    if os.path.exists(os.path.join(source_root, '.git')):
        GIT_REVISION, COMMIT_COUNT = git_version(source_root)
    # 否则，如果存在scipy/version.py文件，则加载现有版本信息
    elif os.path.exists('scipy/version.py'):
        import runpy
        ns = runpy.run_path('scipy/version.py')
        GIT_REVISION = ns['git_revision']
        COMMIT_COUNT = ns['git_revision']
    else:
        GIT_REVISION = "Unknown"
        COMMIT_COUNT = "Unknown"

    # 如果未发布，将开发版本信息添加到完整版本号中
    if not ISRELEASED:
        FULLVERSION += '.dev0+' + COMMIT_COUNT + '.' + GIT_REVISION

    return FULLVERSION, GIT_REVISION, COMMIT_COUNT


# 生成版本信息文件函数
def write_version_py(source_root, filename='scipy/version.py'):
    cnt = """\
# THIS FILE IS GENERATED DURING THE SCIPY BUILD
# See tools/version_utils.py for details

short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
commit_count = '%(commit_count)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # 获取版本信息
    FULLVERSION, GIT_REVISION, COMMIT_COUNT = get_version_info(source_root)

    # 写入文件
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'commit_count': COMMIT_COUNT,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


# 获取Git版本信息函数
def git_version(cwd):
    # 构建最小化的外部命令执行函数
    def _minimal_ext_cmd(cmd):
        # 构建最小的环境变量
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        # 执行命令并获取输出
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env, cwd=cwd).communicate()[0]
        return out
    # 尝试获取当前脚本文件的父目录的绝对路径，然后加上 ".git" 目录，以得到 Git 仓库的路径
    git_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    git_dir = os.path.join(git_dir, ".git")
    # 调用 _minimal_ext_cmd 函数执行 Git 命令，获取当前 HEAD 的短版本哈希值
    out = _minimal_ext_cmd(['git',
                            '--git-dir',
                            git_dir,
                            'rev-parse',
                            'HEAD'])
    # 将获取的输出去除空白字符并转换为 ASCII 编码，截取前7个字符作为 Git 版本的简短标识
    GIT_REVISION = out.strip().decode('ascii')[:7]

    # 生成一个用于标识版本号的字符串，以确保在 nightly 构建中排序正确
    # 这个版本号应该能够随着新提交而增加，但又要能复现，不依赖于日期时间而是基于提交历史
    # 它表示当前分支相对于前一个分支点的提交数量（假设是完整的 `git clone`，若用了 `--depth` 则可能较少）
    prev_version_tag = f'^v{MAJOR}.{MINOR - 2}.0'
    out = _minimal_ext_cmd(['git', '--git-dir', git_dir,
                            'rev-list', 'HEAD', prev_version_tag,
                            '--count'])
    # 获取并存储从前一个版本标签到当前 HEAD 的提交数量
    COMMIT_COUNT = out.strip().decode('ascii')
    COMMIT_COUNT = '0' if not COMMIT_COUNT else COMMIT_COUNT
# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个名为 "--source-root" 的命令行参数，类型为字符串，默认值为当前目录'.'
    parser.add_argument("--source-root", type=str, default='.',
                        help="Relative path to the root of the source directory")
    # 解析命令行参数，并将其存储在args对象中
    args = parser.parse_args()

    # 调用函数write_version_py，传入命令行参数中指定的源根目录
    write_version_py(args.source_root)
```