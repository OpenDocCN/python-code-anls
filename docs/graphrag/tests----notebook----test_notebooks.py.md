# `.\graphrag\tests\notebook\test_notebooks.py`

```py
# 导入必要的库和模块
import subprocess  # 用于执行外部命令
import tempfile  # 用于创建临时文件
from pathlib import Path  # 用于处理路径操作

import nbformat  # 用于处理 Jupyter 笔记本文件
import pytest  # 用于编写和运行测试

# 定义文档路径
DOCS_PATH = Path("../../docsite")

# 获取所有文档路径下的 Jupyter 笔记本文件列表
notebooks_list = list(DOCS_PATH.rglob("*.ipynb"))


def _notebook_run(filepath: Path):
    """执行一个笔记本文件，通过 nbconvert 收集输出。
    :returns 执行过程中的错误输出
    """
    # 使用临时文件保存执行后的笔记本文件
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_file:
        # 构建执行命令的参数列表
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "-y",
            "--no-prompt",
            "--output",
            temp_file.name,
            filepath.absolute().as_posix(),
        ]
        # 执行命令，将笔记本文件转换并执行
        subprocess.check_call(args)

        # 将临时文件内容读取为 nbformat 格式的笔记本对象
        temp_file.seek(0)
        nb = nbformat.read(temp_file, nbformat.current_nbformat)

    # 收集执行过程中产生的错误输出
    return [
        output
        for cell in nb.cells
        if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]


@pytest.mark.parametrize("notebook_path", notebooks_list)
def test_notebook(notebook_path: Path):
    # 断言执行笔记本文件后没有产生任何错误输出
    assert _notebook_run(notebook_path) == []
```