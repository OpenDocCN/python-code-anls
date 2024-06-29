# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_sphinxext.py`

```
"""Tests for tinypages build using sphinx extensions."""

import filecmp  # 导入文件比较模块
import os  # 导入操作系统功能模块
from pathlib import Path  # 导入路径处理模块
import shutil  # 导入文件和目录操作模块
import sys  # 导入系统模块

from matplotlib.testing import subprocess_run_for_testing  # 导入matplotlib的子进程测试模块
import pytest  # 导入pytest测试框架


pytest.importorskip('sphinx',
                    minversion=None if sys.version_info < (3, 10) else '4.1.3')  # 确保导入sphinx模块，版本至少为3.10或者确切为4.1.3以上


def build_sphinx_html(source_dir, doctree_dir, html_dir, extra_args=None):
    # Build the pages with warnings turned into errors
    extra_args = [] if extra_args is None else extra_args  # 如果extra_args为None，则设置为空列表
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html',
           '-d', str(doctree_dir), str(source_dir), str(html_dir), *extra_args]  # 构建sphinx命令行参数列表
    proc = subprocess_run_for_testing(
        cmd, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": ""})  # 运行sphinx命令作为测试，并捕获输出，设置环境变量MPLBACKEND为空字符串
    out = proc.stdout  # 捕获的标准输出
    err = proc.stderr  # 捕获的标准错误

    assert proc.returncode == 0, \
        f"sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n"  # 断言sphinx构建成功，否则输出标准输出和标准错误信息
    if err:
        pytest.fail(f"sphinx build emitted the following warnings:\n{err}")  # 如果有错误信息，使用pytest的失败断言输出警告信息

    assert html_dir.is_dir()  # 断言HTML目录存在且为目录


def test_tinypages(tmp_path):
    shutil.copytree(Path(__file__).parent / 'tinypages', tmp_path,
                    dirs_exist_ok=True)  # 复制'tinypages'目录到临时目录，允许目录存在
    html_dir = tmp_path / '_build' / 'html'  # HTML输出目录路径
    img_dir = html_dir / '_images'  # 图片目录路径
    doctree_dir = tmp_path / 'doctrees'  # doctree目录路径
    # Build the pages with warnings turned into errors
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html',
           '-d', str(doctree_dir),
           str(Path(__file__).parent / 'tinypages'), str(html_dir)]  # 构建sphinx命令行参数列表
    # On CI, gcov emits warnings (due to agg headers being included with the
    # same name in multiple extension modules -- but we don't care about their
    # coverage anyways); hide them using GCOV_ERROR_FILE.
    proc = subprocess_run_for_testing(
        cmd, capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "", "GCOV_ERROR_FILE": os.devnull}
    )  # 在CI环境中，gcov会发出警告（由于agg头文件与多个扩展模块中的相同名称包含在内），忽略这些警告
    out = proc.stdout  # 捕获的标准输出
    err = proc.stderr  # 捕获的标准错误

    # Build the pages with warnings turned into errors
    build_sphinx_html(tmp_path, doctree_dir, html_dir)  # 调用build_sphinx_html函数构建HTML页面

    def plot_file(num):
        return img_dir / f'some_plots-{num}.png'  # 返回指定编号的图像文件路径

    def plot_directive_file(num):
        # This is always next to the doctree dir.
        return doctree_dir.parent / 'plot_directive' / f'some_plots-{num}.png'  # 返回指定编号的图像文件路径

    range_10, range_6, range_4 = [plot_file(i) for i in range(1, 4)]  # 获取范围为1到3的图像文件路径列表
    # Plot 5 is range(6) plot
    assert filecmp.cmp(range_6, plot_file(5))  # 断言文件比较结果为真
    # Plot 7 is range(4) plot
    assert filecmp.cmp(range_4, plot_file(7))  # 断言文件比较结果为真
    # Plot 11 is range(10) plot
    assert filecmp.cmp(range_10, plot_file(11))  # 断言文件比较结果为真
    # Plot 12 uses the old range(10) figure and the new range(6) figure
    assert filecmp.cmp(range_10, plot_file('12_00'))  # 断言文件比较结果为真
    assert filecmp.cmp(range_6, plot_file('12_01'))  # 断言文件比较结果为真
    # Plot 13 shows close-figs in action
    assert filecmp.cmp(range_4, plot_file(13))  # 断言文件比较结果为真
    # Plot 14 has included source
    html_contents = (html_dir / 'some_plots.html').read_bytes()  # 读取HTML文件的字节内容
    # 确保 HTML 内容中包含特定的注释
    assert b'# Only a comment' in html_contents
    # 检查是否文件比较通过，验证范围为4的图像文件是否与指定目录中的range4.png一致
    assert filecmp.cmp(range_4, img_dir / 'range4.png')
    # 检查是否文件比较通过，验证范围为6的图像文件是否与指定目录中的range6_range6.png一致
    assert filecmp.cmp(range_6, img_dir / 'range6_range6.png')
    # 确保 HTML 文件中包含特定的图像标题注释
    assert b'This is the caption for plot 15.' in html_contents
    # 确保 HTML 文件中包含使用 :caption: 指定的图像标题注释
    assert b'Plot 17 uses the caption option.' in html_contents
    # 确保 HTML 文件中包含特定的图像标题注释
    assert b'This is the caption for plot 18.' in html_contents
    # 确保 HTML 文件中包含自定义类的指定内容
    assert b'plot-directive my-class my-other-class' in html_contents
    # 确保两处应用了相同的多图像标题
    assert html_contents.count(b'This caption applies to both plots.') == 2
    # 确保范围为6的图像文件与使用指定参数的绘图文件相同
    assert filecmp.cmp(range_6, plot_file(17))
    # 确保范围为10的图像文件与指定目录中的range6_range10.png一致
    assert filecmp.cmp(range_10, img_dir / 'range6_range10.png')

    # 修改包含的绘图文件
    contents = (tmp_path / 'included_plot_21.rst').read_bytes()
    contents = contents.replace(b'plt.plot(range(6))', b'plt.plot(range(4))')
    (tmp_path / 'included_plot_21.rst').write_bytes(contents)
    # 重新构建页面并检查修改后的文件是否更新
    modification_times = [plot_directive_file(i).stat().st_mtime
                          for i in (1, 2, 3, 5)]
    build_sphinx_html(tmp_path, doctree_dir, html_dir)
    assert filecmp.cmp(range_4, plot_file(17))
    # 确保 plot_directive 文件夹中的绘图文件未更改
    assert plot_directive_file(1).stat().st_mtime == modification_times[0]
    assert plot_directive_file(2).stat().st_mtime == modification_times[1]
    assert plot_directive_file(3).stat().st_mtime == modification_times[2]
    assert filecmp.cmp(range_10, plot_file(1))
    assert filecmp.cmp(range_6, plot_file(2))
    assert filecmp.cmp(range_4, plot_file(3))
    # 确保标记为上下文的图形重新创建（但内容相同）
    assert plot_directive_file(5).stat().st_mtime > modification_times[3]
    assert filecmp.cmp(range_6, plot_file(5))
def test_plot_html_show_source_link(tmp_path):
    # 获取当前文件所在的目录路径
    parent = Path(__file__).parent
    # 复制配置文件到临时路径下
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    # 复制静态文件夹到临时路径下
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    # 设置 doctree 目录路径
    doctree_dir = tmp_path / 'doctrees'
    # 写入 index.rst 文件内容
    (tmp_path / 'index.rst').write_text("""
.. plot::

    plt.plot(range(2))
""")
    # 确保默认情况下生成源代码脚本
    html_dir1 = tmp_path / '_build' / 'html1'
    build_sphinx_html(tmp_path, doctree_dir, html_dir1)
    # 断言生成了一个 index-1.py 文件
    assert len(list(html_dir1.glob("**/index-1.py"))) == 1
    # 确保在 plot_html_show_source_link 设置为 False 时不生成源代码脚本
    html_dir2 = tmp_path / '_build' / 'html2'
    build_sphinx_html(tmp_path, doctree_dir, html_dir2,
                      extra_args=['-D', 'plot_html_show_source_link=0'])
    # 断言未生成 index-1.py 文件
    assert len(list(html_dir2.glob("**/index-1.py"))) == 0


@pytest.mark.parametrize('plot_html_show_source_link', [0, 1])
def test_show_source_link_true(tmp_path, plot_html_show_source_link):
    # 测试当 :show-source-link: 为 true 时生成源代码链接，
    # 不论 plot_html_show_source_link 是否为 true。
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text("""
.. plot::
    :show-source-link: true

    plt.plot(range(2))
""")
    html_dir = tmp_path / '_build' / 'html'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=[
        '-D', f'plot_html_show_source_link={plot_html_show_source_link}'])
    # 断言生成了一个 index-1.py 文件
    assert len(list(html_dir.glob("**/index-1.py"))) == 1


@pytest.mark.parametrize('plot_html_show_source_link', [0, 1])
def test_show_source_link_false(tmp_path, plot_html_show_source_link):
    # 测试当 :show-source-link: 为 false 时不生成源代码链接，
    # 不论 plot_html_show_source_link 是否为 true。
    parent = Path(__file__).parent
    shutil.copyfile(parent / 'tinypages/conf.py', tmp_path / 'conf.py')
    shutil.copytree(parent / 'tinypages/_static', tmp_path / '_static')
    doctree_dir = tmp_path / 'doctrees'
    (tmp_path / 'index.rst').write_text("""
.. plot::
    :show-source-link: false

    plt.plot(range(2))
""")
    html_dir = tmp_path / '_build' / 'html'
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=[
        '-D', f'plot_html_show_source_link={plot_html_show_source_link}'])
    # 断言未生成 index-1.py 文件
    assert len(list(html_dir.glob("**/index-1.py"))) == 0


def test_srcset_version(tmp_path):
    # 复制整个 'tinypages' 文件夹到临时路径下，允许目标文件夹存在
    shutil.copytree(Path(__file__).parent / 'tinypages', tmp_path,
                    dirs_exist_ok=True)
    # 设置 html 和图片目录路径
    html_dir = tmp_path / '_build' / 'html'
    img_dir = html_dir / '_images'
    # 设置 doctree 目录路径
    doctree_dir = tmp_path / 'doctrees'
    # 执行构建 HTML 的函数，使用 2x 的 srcset 版本
    build_sphinx_html(tmp_path, doctree_dir, html_dir, extra_args=[
        '-D', 'plot_srcset=2x'])
    # 定义一个函数，根据给定的编号和后缀返回对应的文件路径
    def plot_file(num, suff=''):
        return img_dir / f'some_plots-{num}{suff}.png'

    # 检查一系列的文件是否存在，包括默认和带后缀的文件
    for ind in [1, 2, 3, 5, 7, 11, 13, 15, 17]:
        assert plot_file(ind).exists()  # 检查默认后缀的文件是否存在
        assert plot_file(ind, suff='.2x').exists()  # 检查带 '.2x' 后缀的文件是否存在

    # 检查特定的图片文件是否存在
    assert (img_dir / 'nestedpage-index-1.png').exists()
    assert (img_dir / 'nestedpage-index-1.2x.png').exists()
    assert (img_dir / 'nestedpage-index-2.png').exists()
    assert (img_dir / 'nestedpage-index-2.2x.png').exists()
    assert (img_dir / 'nestedpage2-index-1.png').exists()
    assert (img_dir / 'nestedpage2-index-1.2x.png').exists()
    assert (img_dir / 'nestedpage2-index-2.png').exists()
    assert (img_dir / 'nestedpage2-index-2.2x.png').exists()

    # 检查 HTML 文件中的图片 srcset 是否包含特定的文件路径和后缀信息
    assert ('srcset="_images/some_plots-1.png, _images/some_plots-1.2x.png 2.00x"'
            in (html_dir / 'some_plots.html').read_text(encoding='utf-8'))

    # 检查 HTML 文件中的图片 srcset 是否包含特定的文件路径和后缀信息，针对嵌套页面 'nestedpage'
    st = ('srcset="../_images/nestedpage-index-1.png, '
          '../_images/nestedpage-index-1.2x.png 2.00x"')
    assert st in (html_dir / 'nestedpage/index.html').read_text(encoding='utf-8')

    # 检查 HTML 文件中的图片 srcset 是否包含特定的文件路径和后缀信息，针对嵌套页面 'nestedpage2'
    st = ('srcset="../_images/nestedpage2-index-2.png, '
          '../_images/nestedpage2-index-2.2x.png 2.00x"')
    assert st in (html_dir / 'nestedpage2/index.html').read_text(encoding='utf-8')
```