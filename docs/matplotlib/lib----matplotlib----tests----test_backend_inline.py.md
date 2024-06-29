# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_inline.py`

```
import os  # 导入操作系统相关的功能模块
from pathlib import Path  # 导入路径操作相关的功能模块
from tempfile import TemporaryDirectory  # 导入临时目录操作相关的功能模块
import sys  # 导入系统相关的功能模块

import pytest  # 导入 pytest 测试框架

from matplotlib.testing import subprocess_run_for_testing  # 导入用于测试的子进程运行函数

nbformat = pytest.importorskip('nbformat')  # 导入并检查 nbformat 库是否可用
pytest.importorskip('nbconvert')  # 导入并检查 nbconvert 库是否可用
pytest.importorskip('ipykernel')  # 导入并检查 ipykernel 库是否可用
pytest.importorskip('matplotlib_inline')  # 导入并检查 matplotlib_inline 库是否可用


@pytest.mark.skipif(sys.version_info[:2] <= (3, 9), reason="Requires Python 3.10+")
def test_ipynb():
    nb_path = Path(__file__).parent / 'test_inline_01.ipynb'  # 获取当前脚本所在目录下的 test_inline_01.ipynb 文件路径

    with TemporaryDirectory() as tmpdir:  # 创建一个临时目录 tmpdir，并在离开 with 语句时自动清理
        out_path = Path(tmpdir, "out.ipynb")  # 设置输出文件的路径为临时目录下的 out.ipynb

        subprocess_run_for_testing(
            ["jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--ExecutePreprocessor.timeout=500",
             "--output", str(out_path), str(nb_path)],  # 执行子进程命令，将 nb_path 的内容转换为 notebook 格式并执行，输出到 out_path
            env={**os.environ, "IPYTHONDIR": tmpdir},  # 设置子进程的环境变量，包括当前系统环境变量和 IPYTHONDIR 变量
            check=True)  # 检查子进程执行是否成功

        with out_path.open() as out:  # 打开输出的 notebook 文件
            nb = nbformat.read(out, nbformat.current_nbformat)  # 使用 nbformat 读取输出的 notebook 文件内容

    # 获取所有错误输出的列表
    errors = [output for cell in nb.cells for output in cell.get("outputs", [])
              if output.output_type == "error"]
    assert not errors  # 断言没有错误输出

    import IPython
    if IPython.version_info[:2] >= (8, 24):
        expected_backend = "inline"
    else:
        # 当 IPython 版本低于 8.24 时，使用指定的后端
        expected_backend = "module://matplotlib_inline.backend_inline"
    
    # 检查第三个单元格的输出，确认使用的后端与预期相符
    backend_outputs = nb.cells[2]["outputs"]
    assert backend_outputs[0]["data"]["text/plain"] == f"'{expected_backend}'"

    # 检查第二个单元格的输出，确认包含特定的图像信息
    image = nb.cells[1]["outputs"][1]["data"]
    assert image["text/plain"] == "<Figure size 300x200 with 1 Axes>"
    assert "image/png" in image
```