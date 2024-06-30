# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\data\openml\id_42074\__init__.py`

```
# 导入所需的模块：os 模块用于与操作系统进行交互，shutil 模块提供高级文件操作功能
import os
import shutil

# 定义函数 copytree，用于复制整个目录树结构
def copytree(src, dst, symlinks=False, ignore=None):
    # 如果目标目录已经存在，则引发异常
    if os.path.exists(dst):
        raise OSError(f"目标目录 '{dst}' 已经存在。")
    # 如果目标路径是文件而不是目录，则引发异常
    if os.path.isfile(dst):
        raise OSError(f"无法复制到文件 '{dst}'，目标必须是目录。")
    # 创建目标目录
    os.makedirs(dst)
    # 遍历源目录下的所有文件和目录
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        # 如果是符号链接，并且 symlinks 参数为 True，则创建符号链接
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except Exception:
                pass  # 忽略权限错误
        # 如果是目录，则递归调用 copytree 复制子目录
        elif os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        # 如果是普通文件，则直接复制文件
        else:
            shutil.copy2(s, d)
```