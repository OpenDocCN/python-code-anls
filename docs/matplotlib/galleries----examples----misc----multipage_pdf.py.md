# `D:\src\scipysrc\matplotlib\galleries\examples\misc\multipage_pdf.py`

```
"""
=============
Multipage PDF
=============

This is a demo of creating a pdf file with several pages,
as well as adding metadata and annotations to pdf files.

If you want to use a multipage pdf file using LaTeX, you need
to use ``from matplotlib.backends.backend_pgf import PdfPages``.
This version however does not support `.attach_note`.
"""

import datetime  # 导入 datetime 模块，用于处理日期时间信息

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算

from matplotlib.backends.backend_pdf import PdfPages  # 导入 PdfPages 类，用于操作 PDF 文件

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
# 创建 PdfPages 对象，用于保存 PDF 页面
with PdfPages('multipage_pdf.pdf') as pdf:
    
    plt.figure(figsize=(3, 3))  # 创建一个 3x3 英寸大小的图像窗口
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')  # 绘制红色线条图，并添加圆点标记
    plt.title('Page One')  # 设置图像标题
    pdf.savefig()  # 将当前图像保存为 PDF 页面
    plt.close()  # 关闭图像窗口

    # if LaTeX is not installed or error caught, change to `False`
    plt.rcParams['text.usetex'] = True  # 设置使用 LaTeX 渲染文本
    plt.figure(figsize=(8, 6))  # 创建一个 8x6 英寸大小的图像窗口
    x = np.arange(0, 5, 0.1)  # 生成一个从 0 到 5 的间隔为 0.1 的数组
    plt.plot(x, np.sin(x), 'b-')  # 绘制蓝色正弦曲线
    plt.title('Page Two')  # 设置图像标题
    pdf.attach_note("plot of sin(x)")  # 将元数据（PDF 注释）附加到页面
    pdf.savefig()  # 将当前图像保存为 PDF 页面
    plt.close()  # 关闭图像窗口

    plt.rcParams['text.usetex'] = False  # 关闭使用 LaTeX 渲染文本的设置
    fig = plt.figure(figsize=(4, 5))  # 创建一个 4x5 英寸大小的图像窗口
    plt.plot(x, x ** 2, 'ko')  # 绘制黑色圆点图
    plt.title('Page Three')  # 设置图像标题
    pdf.savefig(fig)  # 将指定的 Figure 对象保存为 PDF 页面
    plt.close()  # 关闭图像窗口

    # We can also set the file's metadata via the PdfPages object:
    # 通过 PdfPages 对象设置 PDF 文件的元数据
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'  # 设置 PDF 文件的标题
    d['Author'] = 'Jouni K. Sepp\xe4nen'  # 设置 PDF 文件的作者
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'  # 设置 PDF 文件的主题
    d['Keywords'] = 'PdfPages multipage keywords author title subject'  # 设置 PDF 文件的关键词
    d['CreationDate'] = datetime.datetime(2009, 11, 13)  # 设置 PDF 文件的创建日期
    d['ModDate'] = datetime.datetime.today()  # 设置 PDF 文件的修改日期
```