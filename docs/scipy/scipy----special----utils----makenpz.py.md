# `D:\src\scipysrc\scipy\scipy\special\utils\makenpz.py`

```
"""
python makenpz.py DIRECTORY

Build a npz containing all data files in the directory.

"""

# 导入必要的库
import os               # 导入操作系统相关的功能
import numpy as np      # 导入NumPy库
import argparse         # 导入命令行参数解析库
from stat import ST_MTIME  # 从stat模块导入ST_MTIME常量，用于获取文件修改时间


def newer(source, target):
    """
    Return true if 'source' exists and is more recently modified than
    'target', or if 'source' exists and 'target' doesn't.  Return false if
    both exist and 'target' is the same age or younger than 'source'.
    """
    # 如果'source'文件不存在，则抛出异常
    if not os.path.exists(source):
        raise ValueError("file '%s' does not exist" % os.path.abspath(source))
    # 如果'target'文件不存在，返回True，表示需要更新
    if not os.path.exists(target):
        return 1

    # 获取'source'和'target'文件的修改时间
    mtime1 = os.stat(source)[ST_MTIME]
    mtime2 = os.stat(target)[ST_MTIME]

    # 比较两个文件的修改时间，判断是否需要更新
    return mtime1 > mtime2


def main():
    # 创建命令行参数解析器
    p = argparse.ArgumentParser(usage=(__doc__ or '').strip())
    # 添加命令行选项：是否使用时间戳进行更新检查
    p.add_argument('--use-timestamp', action='store_true', default=False,
                   help="don't rewrite npz file if it is newer than sources")
    # 添加位置参数：目录名称
    p.add_argument('dirname')  # for Meson: 'boost' or 'gsl'
    # 添加可选参数：输出目录的相对路径
    p.add_argument("-o", "--outdir", type=str,
                   help="Relative path to the output directory")
    # 解析命令行参数
    args = p.parse_args()

    # 检查是否提供了输出目录参数
    if not args.outdir:
        raise ValueError("Missing `--outdir` argument to makenpz.py")
    else:
        # 构建输入目录路径
        inp = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           '..', 'tests', 'data', args.dirname)
        # 构建输出目录的绝对路径
        outdir_abs = os.path.join(os.getcwd(), args.outdir)
        # 构建输出文件路径
        outp = os.path.join(outdir_abs, args.dirname + ".npz")

    # 如果输出文件已存在且输入目录不存在，则直接返回，无需重新构建
    if os.path.isfile(outp) and not os.path.isdir(inp):
        return

    # 查找源文件
    files = []
    for dirpath, dirnames, filenames in os.walk(inp):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            # 只选择以'.txt'结尾的文件作为数据源
            if fn.endswith('.txt'):
                # 构建数据在npz文件中的键
                key = dirpath[len(inp)+1:] + '-' + fn[:-4]
                key = key.strip('-')
                # 将文件路径添加到files列表中
                files.append((key, os.path.join(dirpath, fn)))

    # 检查是否需要重新构建npz文件
    if args.use_timestamp and os.path.isfile(outp):
        try:
            # 尝试加载旧的npz文件数据
            old_data = np.load(outp)
            try:
                # 检查数据是否发生变化
                changed = set(old_data.keys()) != {key for key, _ in files}
            finally:
                # 关闭旧npz文件
                old_data.close()
        except OSError:
            # 如果旧文件损坏，则需要重新构建
            changed = True

        # 检查源文件是否有更新
        changed = changed or any(newer(fn, outp) for key, fn in files)
        # 检查当前脚本文件是否有更新
        changed = changed or newer(__file__, outp)
        # 如果没有变化，则直接返回，无需重新构建
        if not changed:
            return

    # 构建数据字典，加载所有数据文件到内存中
    data = {}
    for key, fn in files:
        data[key] = np.loadtxt(fn)

    # 使用压缩方式保存数据到npz文件
    np.savez_compressed(outp, **data)


if __name__ == "__main__":
    main()
```