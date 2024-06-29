# `.\numpy\numpy\_build_utils\gitversion.py`

```
#!/usr/bin/env python3
import os  # 导入操作系统相关功能的模块
import textwrap  # 导入文本包装模块


def init_version():
    # 初始化版本信息，从相对路径中读取 pyproject.toml 文件
    init = os.path.join(os.path.dirname(__file__), '../../pyproject.toml')
    with open(init) as fid:
        data = fid.readlines()  # 读取文件所有行的内容存入列表

    # 找到包含版本信息的行
    version_line = next(
        line for line in data if line.startswith('version =')
    )

    # 解析版本号，去除引号和空格
    version = version_line.strip().split(' = ')[1]
    version = version.replace('"', '').replace("'", '')

    return version


def git_version(version):
    # 为开发版本信息添加最后提交的日期和哈希值

    import subprocess  # 导入子进程模块
    import os.path  # 导入路径操作模块

    git_hash = ''
    try:
        # 运行 git 命令获取最后一次提交的哈希和日期
        p = subprocess.Popen(
            ['git', 'log', '-1', '--format="%H %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),  # 设置工作目录为当前文件所在目录
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode('utf-8')
                .strip()
                .replace('"', '')
                .split('T')[0]
                .replace('-', '')
                .split()
            )

            # 只有在开发版本中附加 git 标签信息
            if 'dev' in version:
                version += f'+git{git_date}.{git_hash[:7]}'

    return version, git_hash


if __name__ == "__main__":
    import argparse  # 导入命令行参数解析模块

    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--write', help="Save version to this file")  # 添加保存版本信息的文件路径参数
    parser.add_argument(
        '--meson-dist',
        help='Output path is relative to MESON_DIST_ROOT',  # 相对于 MESON_DIST_ROOT 的输出路径参数说明
        action='store_true'  # 设置参数为布尔类型
    )
    args = parser.parse_args()  # 解析命令行参数

    version, git_hash = git_version(init_version())  # 获取版本信息和 git 哈希值

    # 为 NumPy 2.0 提供更详细的版本信息模块
    template = textwrap.dedent(f'''
        """
        Module to expose more detailed version info for the installed `numpy`
        """
        version = "{version}"  # 设置版本号
        __version__ = version  # 设置模块版本号
        full_version = version  # 设置完整版本号

        git_revision = "{git_hash}"  # 设置 git 哈希值
        release = 'dev' not in version and '+' not in version  # 检查是否为正式发布版本
        short_version = version.split("+")[0]  # 获取短版本号
    ''')

    if args.write:
        outfile = args.write
        if args.meson_dist:
            outfile = os.path.join(
                os.environ.get('MESON_DIST_ROOT', ''),
                outfile
            )

        # 打印人类可读的输出路径
        relpath = os.path.relpath(outfile)
        if relpath.startswith('.'):
            relpath = outfile

        with open(outfile, 'w') as f:
            print(f'Saving version to {relpath}')
            f.write(template)  # 将模板写入文件
    else:
        print(version)  # 打印版本号
```