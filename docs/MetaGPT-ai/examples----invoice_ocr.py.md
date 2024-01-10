# `MetaGPT\examples\invoice_ocr.py`

```

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 21:40:57
@Author  : Stitch-z
@File    : invoice_ocr.py
"""

# 导入 asyncio 模块
import asyncio
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 从 metagpt.roles.invoice_ocr_assistant 模块中导入 InvoiceOCRAssistant 类和 InvoicePath 类
from metagpt.roles.invoice_ocr_assistant import InvoiceOCRAssistant, InvoicePath
# 从 metagpt.schema 模块中导入 Message 类
from metagpt.schema import Message

# 定义异步函数 main
async def main():
    # 定义相对路径列表
    relative_paths = [
        Path("../tests/data/invoices/invoice-1.pdf"),
        Path("../tests/data/invoices/invoice-2.png"),
        Path("../tests/data/invoices/invoice-3.jpg"),
        Path("../tests/data/invoices/invoice-4.zip"),
    ]
    # 获取文件的绝对路径
    absolute_file_paths = [Path.cwd() / path for path in relative_paths]

    # 遍历绝对路径列表
    for path in absolute_file_paths:
        # 创建 InvoiceOCRAssistant 对象
        role = InvoiceOCRAssistant()
        # 运行 InvoiceOCRAssistant 对象的 run 方法
        await role.run(Message(content="Invoicing date", instruct_content=InvoicePath(file_path=path)))

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行异步函数 main
    asyncio.run(main())

```