# `D:\src\scipysrc\sympy\sympy\printing\tests\test_preview.py`

```
# -*- coding: utf-8 -*-  # 声明文件编码格式为 UTF-8

# 从 sympy 库导入必要的类和函数
from sympy.core.relational import Eq  # 导入 Eq 类，用于表示等式关系
from sympy.core.symbol import Symbol  # 导入 Symbol 类，用于创建符号变量
from sympy.functions.elementary.piecewise import Piecewise  # 导入 Piecewise 类，用于创建分段函数
from sympy.printing.preview import preview  # 导入 preview 函数，用于生成数学表达式的预览图像

# 从 io 库导入 BytesIO 类
from io import BytesIO


# 定义测试函数 test_preview，用于生成符号 x 的预览图像并将结果输出到 BytesIO 对象
def test_preview():
    x = Symbol('x')  # 创建符号变量 x
    obj = BytesIO()  # 创建一个 BytesIO 对象来存储预览图像
    try:
        preview(x, output='png', viewer='BytesIO', outputbuffer=obj)  # 生成符号 x 的 PNG 格式预览图像
    except RuntimeError:
        pass  # 捕获 RuntimeError 异常，通常表示在 CI 服务器上未安装 LaTeX


# 定义测试函数 test_preview_unicode_symbol，用于生成带有希腊字母 α 的预览图像并将结果输出到 BytesIO 对象
def test_preview_unicode_symbol():
    # issue 9107
    a = Symbol('α')  # 创建带有希腊字母 α 的符号变量 a
    obj = BytesIO()  # 创建一个 BytesIO 对象来存储预览图像
    try:
        preview(a, output='png', viewer='BytesIO', outputbuffer=obj)  # 生成带有希腊字母 α 的 PNG 格式预览图像
    except RuntimeError:
        pass  # 捕获 RuntimeError 异常，通常表示在 CI 服务器上未安装 LaTeX


# 定义测试函数 test_preview_latex_construct_in_expr，用于生成包含 LaTeX 构造的表达式的预览图像并将结果输出到 BytesIO 对象
def test_preview_latex_construct_in_expr():
    # see PR 9801
    x = Symbol('x')  # 创建符号变量 x
    pw = Piecewise((1, Eq(x, 0)), (0, True))  # 创建一个 Piecewise 对象 pw，表示分段函数
    obj = BytesIO()  # 创建一个 BytesIO 对象来存储预览图像
    try:
        preview(pw, output='png', viewer='BytesIO', outputbuffer=obj)  # 生成包含 LaTeX 构造的表达式的 PNG 格式预览图像
    except RuntimeError:
        pass  # 捕获 RuntimeError 异常，通常表示在 CI 服务器上未安装 LaTeX
```