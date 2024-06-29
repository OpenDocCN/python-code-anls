# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_texmanager.py`

```py
# 导入标准库和第三方模块
import os
from pathlib import Path
import re
import sys

# 导入 pytest 测试框架
import pytest

# 导入 matplotlib 的 pyplot 子模块
import matplotlib.pyplot as plt
# 导入 matplotlib 测试相关的模块
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing._markers import needs_usetex
from matplotlib.texmanager import TexManager


def test_fontconfig_preamble():
    """测试文档中包含 LaTeX 的导言部分。"""
    # 设置使用 LaTeX 渲染文本
    plt.rcParams['text.usetex'] = True

    # 获取不同配置下的 LaTeX 源码
    src1 = TexManager()._get_tex_source("", fontsize=12)
    # 设置另一种 LaTeX 导言
    plt.rcParams['text.latex.preamble'] = '\\usepackage{txfonts}'
    src2 = TexManager()._get_tex_source("", fontsize=12)

    # 断言两次获取的 LaTeX 源码不同
    assert src1 != src2


@pytest.mark.parametrize(
    "rc, preamble, family", [
        # 参数化测试不同字体配置
        ({"font.family": "sans-serif", "font.sans-serif": "helvetica"},
         r"\usepackage{helvet}", r"\sffamily"),
        ({"font.family": "serif", "font.serif": "palatino"},
         r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "cursive", "font.cursive": "zapf chancery"},
         r"\usepackage{chancery}", r"\rmfamily"),
        ({"font.family": "monospace", "font.monospace": "courier"},
         r"\usepackage{courier}", r"\ttfamily"),
        ({"font.family": "helvetica"}, r"\usepackage{helvet}", r"\sffamily"),
        ({"font.family": "palatino"}, r"\usepackage{mathpazo}", r"\rmfamily"),
        ({"font.family": "zapf chancery"},
         r"\usepackage{chancery}", r"\rmfamily"),
        ({"font.family": "courier"}, r"\usepackage{courier}", r"\ttfamily")
    ])
def test_font_selection(rc, preamble, family):
    """测试不同字体配置下生成的 LaTeX 源码是否符合预期。"""
    # 更新图表的配置参数
    plt.rcParams.update(rc)
    # 创建 TexManager 实例
    tm = TexManager()
    # 读取生成的 LaTeX 源文件内容
    src = Path(tm.make_tex("hello, world", fontsize=12)).read_text()
    # 断言生成的源码中包含指定的 LaTeX 导言
    assert preamble in src
    # 断言生成的源码中的字体系列符合预期
    assert [*re.findall(r"\\\w+family", src)] == [family]


@needs_usetex
def test_unicode_characters():
    """测试 Unicode 字符在使用 LaTeX 渲染时不会引发问题。"""
    # 设置使用 LaTeX 渲染文本
    plt.rcParams['text.usetex'] = True
    # 创建图表和坐标轴对象
    fig, ax = plt.subplots()
    # 设置坐标轴标签包含特定的 Unicode 字符
    ax.set_ylabel('\\textit{Velocity (\N{DEGREE SIGN}/sec)}')
    ax.set_xlabel('\N{VULGAR FRACTION ONE QUARTER}Öøæ')
    fig.canvas.draw()

    # 断言不是所有 Unicode 字符都能正确渲染，应该引发 RuntimeError
    with pytest.raises(RuntimeError):
        ax.set_title('\N{SNOWMAN}')
        fig.canvas.draw()


@needs_usetex
def test_openin_any_paranoid():
    """测试在 'openin_any=p' 模式下是否能正常显示图表。"""
    # 使用 subprocess 运行测试命令
    completed = subprocess_run_for_testing(
        [sys.executable, "-c",
         'import matplotlib.pyplot as plt;'
         'plt.rcParams.update({"text.usetex": True});'
         'plt.title("paranoid");'
         'plt.show(block=False);'],
        env={**os.environ, 'openin_any': 'p'}, check=True, capture_output=True)
    # 断言运行结果的标准错误输出为空
    assert completed.stderr == ""
```