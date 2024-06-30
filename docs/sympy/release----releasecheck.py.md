# `D:\src\scipysrc\sympy\release\releasecheck.py`

```
#!/usr/bin/env python3

# 导入必要的模块和函数
from os.path import join, basename, normpath
from subprocess import check_call

# 主函数，用于执行发布过程的各个阶段
def main(version, prevversion, outdir):
    # 检查版本号和输出目录
    check_version(version, outdir)
    # 运行阶段：更新作者信息
    run_stage(['bin/mailmap_check.py', '--update-authors'])
    # 运行阶段：创建输出目录
    run_stage(['mkdir', '-p', outdir])
    # 构建发布文件：wheel 文件
    build_release_files('--wheel', 'sympy-%s-py3-none-any.whl', outdir, version)
    # 构建发布文件：sdist 文件
    build_release_files('--sdist', 'sympy-%s.tar.gz', outdir, version)
    # 运行阶段：比较生成的 tar 文件与 git 仓库
    run_stage(['release/compare_tar_against_git.py', join(outdir, 'sympy-%s.tar.gz' % (version,)), '.'])
    # 运行阶段：构建文档
    run_stage(['release/build_docs.py', version, outdir])
    # 运行阶段：计算 SHA256
    run_stage(['release/sha256.py', version, outdir])
    # 运行阶段：更新作者信息
    run_stage(['release/authors.py', version, prevversion, outdir])

# 用于返回带有绿色 ANSI 标记的文本
def green(text):
    return "\033[32m%s\033[0m" % text

# 用于返回带有红色 ANSI 标记的文本
def red(text):
    return "\033[31m%s\033[0m" % text

# 打印带有指定颜色 ANSI 标记的标题和消息
def print_header(color, *msgs):
    newlines = '\n'
    vline = '-' * 80
    print(color(newlines + vline))
    for msg in msgs:
        print(color(msg))
    print(color(vline + newlines))

# 运行指定的命令，打印运行状态，并检查是否出错
def run_stage(cmd):
    # 构建命令行字符串
    cmdline = '    $ %s' % (' '.join(cmd),)

    # 打印绿色标题：正在运行指定命令
    print_header(green, 'running:', cmdline)
    try:
        # 执行命令
        check_call(cmd)
    except Exception as e:
        # 打印红色标题：运行失败
        print_header(red, 'failed:', cmdline)
        # 抛出异常，但从当前堆栈中删除内部异常信息
        raise e from None
    else:
        # 打印绿色标题：运行成功
        print_header(green, 'completed:', cmdline)

# 构建发布文件，执行指定的构建命令并移动生成的文件
def build_release_files(cmd, fname, outdir, version):
    # 根据版本号格式化文件名
    fname = fname % (version,)
    # 运行阶段：使用 build 模块创建发布文件
    run_stage(['python', '-m', 'build', cmd])
    # 源文件和目标文件路径
    src = join('dist', fname)
    dst = join(outdir, fname)
    # 移动文件
    run_stage(['mv', src, dst])

# 检查给定版本号和输出目录是否匹配预期值
def check_version(version, outdir):
    # 导入 sympy.release 模块中的版本信息
    from sympy.release import __version__ as checked_out_version
    # 检查版本号是否匹配
    if version != checked_out_version:
        msg = "version %s does not match checkout %s"
        raise AssertionError(msg % (version, checked_out_version))
    # 检查输出目录的基本名称是否符合预期格式
    if basename(normpath(outdir)) != 'release-%s' % (version,):
        msg = "version %s does not match output directory %s"
        raise AssertionError(msg % (version, outdir))

# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
```