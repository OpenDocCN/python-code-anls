# `D:\src\scipysrc\seaborn\doc\tools\set_nb_kernels.py`

```
"""Recursively set the kernel name for all jupyter notebook files."""
# 导入系统模块和文件路径模式匹配模块
import sys
from glob import glob

# 导入 Jupyter Notebook 格式处理模块
import nbformat

# 如果作为主程序运行
if __name__ == "__main__":
    # 从命令行参数中获取内核名称
    _, kernel_name = sys.argv

    # 获取当前目录及子目录中所有的 IPython Notebook 文件路径列表
    nb_paths = glob("./**/*.ipynb", recursive=True)
    
    # 遍历每一个 Notebook 文件路径
    for path in nb_paths:

        # 打开 Notebook 文件
        with open(path) as f:
            # 读取 Notebook 文件内容，并将其解析为 nbformat 版本 4
            nb = nbformat.read(f, as_version=4)

        # 设置 Notebook 元数据中的内核名称
        nb["metadata"]["kernelspec"]["name"] = kernel_name
        # 设置 Notebook 元数据中的内核显示名称
        nb["metadata"]["kernelspec"]["display_name"] = kernel_name

        # 将更新后的 Notebook 对象写回到原文件
        with open(path, "w") as f:
            nbformat.write(nb, f)
```