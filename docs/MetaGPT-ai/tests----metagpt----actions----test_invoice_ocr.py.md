# `MetaGPT\tests\metagpt\actions\test_invoice_ocr.py`

```py

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/10/09 18:40:34
@Author  : Stitch-z
@File    : test_invoice_ocr.py
"""

# 导入模块
from pathlib import Path
import pytest
# 导入自定义模块
from metagpt.actions.invoice_ocr import GenerateTable, InvoiceOCR, ReplyQuestion
from metagpt.const import TEST_DATA_PATH

# 异步测试函数，参数化测试
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoice_path",
    [
        Path("invoices/invoice-3.jpg"),
        Path("invoices/invoice-4.zip"),
    ],
)
async def test_invoice_ocr(invoice_path: Path):
    # 拼接测试数据路径
    invoice_path = TEST_DATA_PATH / invoice_path
    # 运行 InvoiceOCR 模块，获取结果
    resp = await InvoiceOCR().run(file_path=Path(invoice_path))
    # 断言结果类型为列表
    assert isinstance(resp, list)

# 异步测试函数，参数化测试
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invoice_path", "expected_result"),
    [
        (Path("invoices/invoice-1.pdf"), {"收款人": "小明", "城市": "深圳", "总费用/元": 412.00, "开票日期": "2023年02月03日"}),
    ],
)
async def test_generate_table(invoice_path: Path, expected_result: dict):
    # 拼接测试数据路径
    invoice_path = TEST_DATA_PATH / invoice_path
    filename = invoice_path.name
    # 运行 InvoiceOCR 模块，获取结果
    ocr_result = await InvoiceOCR().run(file_path=Path(invoice_path))
    # 运行 GenerateTable 模块，获取结果
    table_data = await GenerateTable().run(ocr_results=ocr_result, filename=filename)
    # 断言结果类型为列表
    assert isinstance(table_data, list)
    table_data = table_data[0]
    # 断言各字段值与预期结果一致
    assert expected_result["收款人"] == table_data["收款人"]
    assert expected_result["城市"] in table_data["城市"]
    assert float(expected_result["总费用/元"]) == float(table_data["总费用/元"])
    assert expected_result["开票日期"] == table_data["开票日期"]

# 异步测试函数，参数化测试
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("invoice_path", "query", "expected_result"),
    [(Path("invoices/invoice-1.pdf"), "Invoicing date", "2023年02月03日")],
)
async def test_reply_question(invoice_path: Path, query: dict, expected_result: str):
    # 拼接测试数据路径
    invoice_path = TEST_DATA_PATH / invoice_path
    # 运行 InvoiceOCR 模块，获取结果
    ocr_result = await InvoiceOCR().run(file_path=Path(invoice_path))
    # 运行 ReplyQuestion 模块，获取结果
    result = await ReplyQuestion().run(query=query, ocr_result=ocr_result)
    # 断言预期结果在返回结果中
    assert expected_result in result

```