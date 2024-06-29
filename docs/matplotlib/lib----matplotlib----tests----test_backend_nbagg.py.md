# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_nbagg.py`

```
# 导入所需的库和模块
import os  # 导入操作系统相关功能的模块
from pathlib import Path  # 导入用于处理路径的模块
from tempfile import TemporaryDirectory  # 导入临时目录的模块

import pytest  # 导入 pytest 测试框架

from matplotlib.testing import subprocess_run_for_testing  # 导入用于测试的子进程运行功能

# 导入 nbformat 库，如果导入失败则跳过测试
nbformat = pytest.importorskip('nbformat')
# 导入 nbconvert 库，如果导入失败则跳过测试
pytest.importorskip('nbconvert')
# 导入 ipykernel 库，如果导入失败则跳过测试
pytest.importorskip('ipykernel')

# 从指定的 URL 地址获取 Jupyter 笔记本测试的函数
# 参考来源：https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/
def test_ipynb():
    # 获取当前文件所在目录的父目录，并拼接测试用的笔记本文件路径
    nb_path = Path(__file__).parent / 'test_nbagg_01.ipynb'

    # 使用临时目录作为输出路径，执行 nbconvert 命令进行笔记本执行和转换
    with TemporaryDirectory() as tmpdir:
        # 设置输出文件路径
        out_path = Path(tmpdir, "out.ipynb")
        # 调用 subprocess_run_for_testing 函数执行 nbconvert 命令
        subprocess_run_for_testing(
            ["jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--ExecutePreprocessor.timeout=500",
             "--output", str(out_path), str(nb_path)],
            # 设置环境变量，包括当前环境变量和 IPYTHONDIR 变量
            env={**os.environ, "IPYTHONDIR": tmpdir},
            # 检查执行结果
            check=True)
        
        # 打开生成的输出文件，并使用 nbformat 读取其中的内容
        with out_path.open() as out:
            nb = nbformat.read(out, nbformat.current_nbformat)

    # 检查执行过程中是否有错误输出，将错误信息收集到 errors 列表中
    errors = [output for cell in nb.cells for output in cell.get("outputs", [])
              if output.output_type == "error"]
    # 断言没有错误输出
    assert not errors

    # 导入 IPython 库，检查版本信息并设置预期的后端输出
    import IPython
    if IPython.version_info[:2] >= (8, 24):
        expected_backend = "notebook"
    else:
        # 当 Python 3.12 达到支持终止生命周期时，可以移除此代码段
        expected_backend = "nbAgg"
    
    # 获取第三个单元格的输出，确保其文本输出与预期的后端一致
    backend_outputs = nb.cells[2]["outputs"]
    assert backend_outputs[0]["data"]["text/plain"] == f"'{expected_backend}'"
```