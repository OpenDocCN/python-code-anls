# `D:\src\scipysrc\seaborn\ci\check_gallery.py`

```
"""执行在线文档中示例展示的脚本。"""
# 导入 glob 模块，用于匹配文件路径模式
from glob import glob
# 导入 matplotlib.pyplot 模块，用于绘制图表
import matplotlib.pyplot as plt

# 如果当前脚本是主程序
if __name__ == "__main__":

    # 获取 examples 文件夹下所有以 .py 结尾的文件路径，并按字母顺序排序
    fnames = sorted(glob("examples/*.py"))

    # 遍历每个文件路径
    for fname in fnames:

        # 打印当前处理的文件名
        print(f"- {fname}")

        # 打开文件，并执行文件内容（将文件内容作为代码执行）
        with open(fname) as fid:
            exec(fid.read())

        # 关闭所有已打开的图表，确保每个示例运行后图表被清除
        plt.close("all")
```