# `.\pytorch\tools\update_masked_docs.py`

```py
"""
This script updates the file torch/masked/_docs.py that contains
the generated doc-strings for various masked operations. The update
should be triggered whenever a new masked operation is introduced to
torch.masked package. Running the script requires that torch package
is functional.
"""

# 导入操作系统接口模块
import os

# 定义主函数
def main() -> None:
    # 目标文件路径
    target = os.path.join("torch", "masked", "_docs.py")

    try:
        # 尝试导入 torch 库
        import torch
    except ImportError as msg:
        # 如果导入失败，打印错误信息并返回
        print(f"Failed to import torch required to build {target}: {msg}")
        return

    # 检查目标文件是否存在，如果存在则读取其内容
    if os.path.isfile(target):
        with open(target) as _f:
            current_content = _f.read()
    else:
        current_content = ""

    # 初始化新内容列表，并添加文件头注释
    _new_content = []
    _new_content.append(
        """\
# -*- coding: utf-8 -*-
# This file is generated, do not modify it!
#
# To update this file, run the update masked docs script as follows:
#
#   python tools/update_masked_docs.py
#
# The script must be called from an environment where the development
# version of torch package can be imported and is functional.
#
"""
    )

    # 遍历所有在 torch.masked._ops.__all__ 中定义的函数名
    for func_name in sorted(torch.masked._ops.__all__):
        # 获取函数对象
        func = getattr(torch.masked._ops, func_name)
        # 生成函数文档字符串
        func_doc = torch.masked._generate_docstring(func)  # type: ignore[no-untyped-call, attr-defined]
        # 将函数文档字符串添加到新内容列表中
        _new_content.append(f'{func_name}_docstring = """{func_doc}"""\n')

    # 将新内容列表转换为字符串形式
    new_content = "\n".join(_new_content)

    # 如果新内容与当前文件内容相同，则无需更新，打印消息并返回
    if new_content == current_content:
        print(f"Nothing to update in {target}")
        return

    # 将新内容写入目标文件
    with open(target, "w") as _f:
        _f.write(new_content)

    # 打印更新成功的消息
    print(f"Successfully updated {target}")


# 如果该脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```