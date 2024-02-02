# `MetaGPT\metagpt\provider\postprocess\base_postprocess_plugin.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : base llm postprocess plugin to do the operations like repair the raw llm output

from typing import Union

from metagpt.utils.repair_llm_raw_output import (
    RepairType,  # 导入 RepairType 枚举类型
    extract_content_from_output,  # 导入从输出中提取内容的函数
    repair_llm_raw_output,  # 导入修复原始 llm 输出的函数
    retry_parse_json_text,  # 导入重试解析 JSON 文本的函数
)


class BasePostProcessPlugin(object):
    model = None  # 插件的 `model` 属性，用于在 `llm_postprocess` 中判断

    def run_repair_llm_output(self, output: str, schema: dict, req_key: str = "[/CONTENT]") -> Union[dict, list]:
        """
        修复步骤
            1. 使用 schema 的字段修复大小写问题
            2. 从 req_key 对中提取内容（xx[REQ_KEY]xxx[/REQ_KEY]xx）
            3. 修复内容中的无效 JSON 文本
            4. 解析 JSON 文本并根据异常进行修复，使用重试循环
        """
        output_class_fields = list(schema["properties"].keys())  # Custom ActionOutput 的字段

        content = self.run_repair_llm_raw_output(output, req_keys=output_class_fields + [req_key])
        content = self.run_extract_content_from_output(content, right_key=req_key)
        # # req_keys mocked
        content = self.run_repair_llm_raw_output(content, req_keys=[None], repair_type=RepairType.JSON)
        parsed_data = self.run_retry_parse_json_text(content)

        return parsed_data

    def run_repair_llm_raw_output(self, content: str, req_keys: list[str], repair_type: str = None) -> str:
        """继承类可以重新实现该函数"""
        return repair_llm_raw_output(content, req_keys=req_keys, repair_type=repair_type)

    def run_extract_content_from_output(self, content: str, right_key: str) -> str:
        """继承类可以重新实现该函数"""
        return extract_content_from_output(content, right_key=right_key)

    def run_retry_parse_json_text(self, content: str) -> Union[dict, list]:
        """继承类可以重新实现该函数"""
        # logger.info(f"extracted json CONTENT from output:\n{content}")
        parsed_data = retry_parse_json_text(output=content)  # 应该使用 output=content
        return parsed_data

    def run(self, output: str, schema: dict, req_key: str = "[/CONTENT]") -> Union[dict, list]:
        """
        用于具有 JSON 格式输出要求和外部对键的提示，例如
            [REQ_KEY]
                {
                    "Key": "value"
                }
            [/REQ_KEY]

        Args
            outer (str): llm 原始输出
            schema: 输出 JSON 模式
            req_key: 外部对右键，通常以 `[/REQ_KEY]` 格式

        """
        assert len(schema.get("properties")) > 0
        assert "/" in req_key

        # 当前，后处理仅处理 repair_llm_raw_output
        new_output = self.run_repair_llm_output(output=output, schema=schema, req_key=req_key)
        return new_output

```