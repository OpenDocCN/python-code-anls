# `MetaGPT\metagpt\prompts\invoice_ocr.py`

```

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 16:30:25
@Author  : Stitch-z
@File    : invoice_ocr.py
@Describe : Prompts of the invoice ocr assistant.
"""

# 设置文件编码和解释器路径

COMMON_PROMPT = "Now I will provide you with the OCR text recognition results for the invoice."

# 提示信息：提供发票的OCR文本识别结果

EXTRACT_OCR_MAIN_INFO_PROMPT = (
    COMMON_PROMPT
    + """
Please extract the payee, city, total cost, and invoicing date of the invoice.

The OCR data of the invoice are as follows:
{ocr_result}

Mandatory restrictions are returned according to the following requirements:
1. The total cost refers to the total price and tax. Do not include `¥`.
2. The city must be the recipient's city.
2. The returned JSON dictionary must be returned in {language}
3. Mandatory requirement to output in JSON format: {{"收款人":"x","城市":"x","总费用/元":"","开票日期":""}}.
"""
)

# 提取OCR主要信息的提示信息

REPLY_OCR_QUESTION_PROMPT = (
    COMMON_PROMPT
    + """
Please answer the question: {query}

The OCR data of the invoice are as follows:
{ocr_result}

Mandatory restrictions are returned according to the following requirements:
1. Answer in {language} language.
2. Enforce restrictions on not returning OCR data sent to you.
3. Return with markdown syntax layout.
"""
)

# 回复OCR问题的提示信息

INVOICE_OCR_SUCCESS = "Successfully completed OCR text recognition invoice."

# 发票OCR文本识别成功的提示信息

```