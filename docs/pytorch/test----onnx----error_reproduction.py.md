# `.\pytorch\test\onnx\error_reproduction.py`

```
"""Error reproduction utilities for op consistency tests."""

# 引入未来版本的注释允许，确保代码在不同 Python 版本中的兼容性
from __future__ import annotations

# 引入标准库模块
import difflib       # 提供比较文本差异的工具
import pathlib       # 提供操作路径的工具
import platform      # 提供获取操作系统平台信息的工具
import sys           # 提供访问与 Python 解释器相关的变量和函数
import time          # 提供处理时间的工具
import traceback     # 提供访问和格式化异常堆栈跟踪信息的工具

# 引入第三方库
import numpy as np   # 提供多维数组和矩阵运算支持

import onnx          # 提供处理和加载 ONNX 模型的工具
import onnxruntime as ort   # 提供运行 ONNX 模型的工具
import onnxscript    # 提供 ONNX 脚本的相关支持

import torch         # 提供深度学习框架 PyTorch 的支持

# 定义用于生成 Markdown 报告的模板字符串
_MISMATCH_MARKDOWN_TEMPLATE = """\
### Summary

The output of ONNX Runtime does not match that of PyTorch when executing test
`{test_name}`, `sample {sample_num}` in ONNX Script `TorchLib`.

To recreate this report, use


CREATE_REPRODUCTION_REPORT=1 python -m pytest onnxscript/tests/function_libs/torch_lib/ops_test.py -k {short_test_name}


### ONNX Model


{onnx_model_text}


### Inputs

Shapes: `{input_shapes}`

<details><summary>Details</summary>
<p>


kwargs = {kwargs}
inputs = {inputs}


</p>
</details>

### Expected output

Shape: `{expected_shape}`

<details><summary>Details</summary>
<p>


expected = {expected}


</p>
</details>

### Actual output

Shape: `{actual_shape}`

<details><summary>Details</summary>
<p>


actual = {actual}


</p>
</details>

### Difference

<details><summary>Details</summary>
<p>


{diff}


</p>
</details>

### Full error stack


{error_stack}


### Environment


{sys_info}


"""


def create_mismatch_report(
    test_name: str,
    sample_num: int,
    onnx_model: onnx.ModelProto,
    inputs,
    kwargs,
    actual,
    expected,
    error: Exception,
) -> None:
    # 设置打印选项以便打印大量数据
    torch.set_printoptions(threshold=sys.maxsize)

    # 获取异常的文本描述和堆栈信息
    error_text = str(error)
    error_stack = error_text + "\n" + "".join(traceback.format_tb(error.__traceback__))

    # 从完整的测试名称中提取出短名称
    short_test_name = test_name.split(".")[-1]

    # 生成实际输出和期望输出之间的差异信息
    diff = difflib.unified_diff(
        str(actual).splitlines(),
        str(expected).splitlines(),
        fromfile="actual",
        tofile="expected",
        lineterm="",
    )

    # 将 ONNX 模型转换为可读文本格式
    onnx_model_text = onnx.printer.to_text(onnx_model)

    # 构建输入数据的形状信息的字符串表示
    input_shapes = repr(
        [
            f"Tensor<{inp.shape}, dtype={inp.dtype}>"
            if isinstance(inp, torch.Tensor)
            else inp
            for inp in inputs
        ]
    )

    # 构建当前运行环境的系统信息
    sys_info = f"""\
OS: {platform.platform()}
Python version: {sys.version}
onnx=={onnx.__version__}
onnxruntime=={ort.__version__}
onnxscript=={onnxscript.__version__}
numpy=={np.__version__}
torch=={torch.__version__}"""

    # 使用模板字符串填充 Markdown 报告的内容
    markdown = _MISMATCH_MARKDOWN_TEMPLATE.format(
        test_name=test_name,
        short_test_name=short_test_name,
        sample_num=sample_num,
        input_shapes=input_shapes,
        inputs=inputs,
        kwargs=kwargs,
        expected=expected,
        expected_shape=expected.shape if isinstance(expected, torch.Tensor) else None,
        actual=actual,
        actual_shape=actual.shape if isinstance(actual, torch.Tensor) else None,
        diff="\n".join(diff),
        error_stack=error_stack,
        sys_info=sys_info,
        onnx_model_text=onnx_model_text,
    )
    # 根据给定的测试名称和时间戳创建一个 Markdown 文件名
    markdown_file_name = f'mismatch-{short_test_name.replace("/", "-").replace(":", "-")}-{str(time.time()).replace(".", "_")}.md'
    # 调用保存错误报告的函数，将 Markdown 内容保存到指定路径，并返回文件路径
    markdown_file_path = save_error_report(markdown_file_name, markdown)
    # 打印出创建的错误重现报告的路径
    print(f"Created reproduction report at {markdown_file_path}")
# 定义一个函数，用于保存错误报告到指定文件
def save_error_report(file_name: str, text: str):
    # 设置错误报告保存的目录为 'error_reports'
    reports_dir = pathlib.Path("error_reports")
    # 如果目录不存在，则创建它，包括所有必要的父目录
    reports_dir.mkdir(parents=True, exist_ok=True)
    # 构建错误报告文件的完整路径
    file_path = reports_dir / file_name
    # 打开文件，以写入模式写入文本内容，使用 UTF-8 编码
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 返回保存的错误报告文件的完整路径
    return file_path
```