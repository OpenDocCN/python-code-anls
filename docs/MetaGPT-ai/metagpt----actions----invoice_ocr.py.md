# `MetaGPT\metagpt\actions\invoice_ocr.py`

```py

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/21 18:10:20
@Author  : Stitch-z
@File    : invoice_ocr.py
@Describe : Actions of the invoice ocr assistant.
"""

# 导入所需的库
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from paddleocr import PaddleOCR
from pydantic import Field

from metagpt.actions import Action
from metagpt.const import INVOICE_OCR_TABLE_PATH
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.prompts.invoice_ocr import (
    EXTRACT_OCR_MAIN_INFO_PROMPT,
    REPLY_OCR_QUESTION_PROMPT,
)
from metagpt.provider.base_llm import BaseLLM
from metagpt.utils.common import OutputParser
from metagpt.utils.file import File

# 定义一个名为InvoiceOCR的类，继承自Action类
class InvoiceOCR(Action):
    """Action class for performing OCR on invoice files, including zip, PDF, png, and jpg files.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language for OCR output. Defaults to "ch" (Chinese).

    """

    # 初始化类的属性
    name: str = "InvoiceOCR"
    context: Optional[str] = None

    # 定义一个静态方法，用于检查文件类型
    @staticmethod
    async def _check_file_type(file_path: Path) -> str:
        """Check the file type of the given filename.

        Args:
            file_path: The path of the file.

        Returns:
            The file type based on FileExtensionType enum.

        Raises:
            Exception: If the file format is not zip, pdf, png, or jpg.
        """
        # 获取文件后缀
        ext = file_path.suffix
        # 如果文件后缀不是.zip, .pdf, .png, .jpg中的一种，则抛出异常
        if ext not in [".zip", ".pdf", ".png", ".jpg"]:
            raise Exception("The invoice format is not zip, pdf, png, or jpg")

        return ext

    # 定义一个静态方法，用于解压文件
    @staticmethod
    async def _unzip(file_path: Path) -> Path:
        """Unzip a file and return the path to the unzipped directory.

        Args:
            file_path: The path to the zip file.

        Returns:
            The path to the unzipped directory.
        """
        # 创建解压后的文件夹路径
        file_directory = file_path.parent / "unzip_invoices" / datetime.now().strftime("%Y%m%d%H%M%S")
        # 使用zipfile库解压文件
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                # 使用CP437编码文件名，然后使用GBK解码以防止中文乱码
                relative_name = Path(zip_info.filename.encode("cp437").decode("gbk"))
                if relative_name.suffix:
                    full_filename = file_directory / relative_name
                    await File.write(full_filename.parent, relative_name.name, zip_ref.read(zip_info.filename))

        logger.info(f"unzip_path: {file_directory}")
        return file_directory

    # 定义一个静态方法，用于进行OCR识别
    @staticmethod
    async def _ocr(invoice_file_path: Path):
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=1)
        ocr_result = ocr.ocr(str(invoice_file_path), cls=True)
        for result in ocr_result[0]:
            result[1] = (result[1][0], round(result[1][1], 2))  # round long confidence scores to reduce token costs
        return ocr_result

    # 定义一个异步方法，用于执行OCR识别
    async def run(self, file_path: Path, *args, **kwargs) -> list:
        """Execute the action to identify invoice files through OCR.

        Args:
            file_path: The path to the input file.

        Returns:
            A list of OCR results.
        """
        # 检查文件类型
        file_ext = await self._check_file_type(file_path)

        if file_ext == ".zip":
            # 如果文件类型是.zip，则进行批量OCR识别
            unzip_path = await self._unzip(file_path)
            ocr_list = []
            for root, _, files in os.walk(unzip_path):
                for filename in files:
                    invoice_file_path = Path(root) / Path(filename)
                    # 识别匹配类型的文件
                    if Path(filename).suffix in [".zip", ".pdf", ".png", ".jpg"]:
                        ocr_result = await self._ocr(str(invoice_file_path))
                        ocr_list.append(ocr_result)
            return ocr_list

        else:
            # 识别单个文件
            ocr_result = await self._ocr(file_path)
            return [ocr_result]


# 定义一个名为GenerateTable的类，继承自Action类
class GenerateTable(Action):
    """Action class for generating tables from OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for the generated table. Defaults to "ch" (Chinese).

    """

    # 初始化类的属性
    name: str = "GenerateTable"
    context: Optional[str] = None
    llm: BaseLLM = Field(default_factory=LLM)
    language: str = "ch"

    # 定义一个异步方法，用于执行生成表格的操作
    async def run(self, ocr_results: list, filename: str, *args, **kwargs) -> dict[str, str]:
        """Processes OCR results, extracts invoice information, generates a table, and saves it as an Excel file.

        Args:
            ocr_results: A list of OCR results obtained from invoice processing.
            filename: The name of the output Excel file.

        Returns:
            A dictionary containing the invoice information.

        """
        table_data = []
        pathname = INVOICE_OCR_TABLE_PATH
        pathname.mkdir(parents=True, exist_ok=True)

        for ocr_result in ocr_results:
            # 提取发票OCR主要信息
            prompt = EXTRACT_OCR_MAIN_INFO_PROMPT.format(ocr_result=ocr_result, language=self.language)
            ocr_info = await self._aask(prompt=prompt)
            invoice_data = OutputParser.extract_struct(ocr_info, dict)
            if invoice_data:
                table_data.append(invoice_data)

        # 生成Excel文件
        filename = f"{filename.split('.')[0]}.xlsx"
        full_filename = f"{pathname}/{filename}"
        df = pd.DataFrame(table_data)
        df.to_excel(full_filename, index=False)
        return table_data


# 定义一个名为ReplyQuestion的类，继承自Action类
class ReplyQuestion(Action):
    """Action class for generating replies to questions based on OCR results.

    Args:
        name: The name of the action. Defaults to an empty string.
        language: The language used for generating the reply. Defaults to "ch" (Chinese).

    """

    # 初始化类的属性
    name: str = "ReplyQuestion"
    context: Optional[str] = None
    llm: BaseLLM = Field(default_factory=LLM)
    language: str = "ch"

    # 定义一个异步方法，用于生成基于OCR结果的回复
    async def run(self, query: str, ocr_result: list, *args, **kwargs) -> str:
        """Reply to questions based on ocr results.

        Args:
            query: The question for which a reply is generated.
            ocr_result: A list of OCR results.

        Returns:
            A reply result of string type.
        """
        prompt = REPLY_OCR_QUESTION_PROMPT.format(query=query, ocr_result=ocr_result, language=self.language)
        resp = await self._aask(prompt=prompt)
        return resp

```