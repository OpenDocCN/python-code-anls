# `D:\src\scipysrc\matplotlib\galleries\examples\misc\image_thumbnail_sgskip.py`

```py
"""
===============
Image Thumbnail
===============

You can use Matplotlib to generate thumbnails from existing images.
Matplotlib relies on Pillow_ for reading images, and thus supports all formats
supported by Pillow.

.. _Pillow: https://python-pillow.org/
"""

# 从 argparse 模块导入 ArgumentParser 类
from argparse import ArgumentParser
# 从 pathlib 模块导入 Path 类
from pathlib import Path
# 导入 sys 模块
import sys
# 导入 matplotlib.image 模块并重命名为 image
import matplotlib.image as image

# 创建 ArgumentParser 对象，设置描述信息
parser = ArgumentParser(
    description="Build thumbnails of all images in a directory.")
# 添加一个位置参数 imagedir，类型为 Path
parser.add_argument("imagedir", type=Path)
# 解析命令行参数并将其存储在 args 变量中
args = parser.parse_args()

# 如果指定的输入目录不存在，则输出错误信息并退出程序
if not args.imagedir.is_dir():
    sys.exit(f"Could not find input directory {args.imagedir}")

# 创建一个名为 thumbs 的路径对象，如果不存在则递归创建
outdir = Path("thumbs")
outdir.mkdir(parents=True, exist_ok=True)

# 遍历输入目录中所有扩展名为 .png 的文件
for path in args.imagedir.glob("*.png"):
    # 构造输出路径，将 thumbs 目录和文件名组合而成
    outpath = outdir / path.name
    # 使用 matplotlib.image.thumbnail 函数生成缩略图，并指定缩放比例为 0.15
    fig = image.thumbnail(path, outpath, scale=0.15)
    # 打印生成的缩略图的保存信息，包括原始文件路径和缩略图输出路径
    print(f"saved thumbnail of {path} to {outpath}")
```