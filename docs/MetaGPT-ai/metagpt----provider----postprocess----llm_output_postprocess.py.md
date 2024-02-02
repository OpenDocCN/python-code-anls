# `MetaGPT\metagpt\provider\postprocess\llm_output_postprocess.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the entry of choosing which PostProcessPlugin to deal particular LLM model's output

from typing import Union

from metagpt.provider.postprocess.base_postprocess_plugin import BasePostProcessPlugin


def llm_output_postprocess(
    output: str, schema: dict, req_key: str = "[/CONTENT]", model_name: str = None
) -> Union[dict, str]:
    """
    default use BasePostProcessPlugin if there is not matched plugin.
    """
    # TODO choose different model's plugin according to the model_name
    # 创建一个 BasePostProcessPlugin 的实例作为后处理插件
    postprocess_plugin = BasePostProcessPlugin()

    # 调用后处理插件的 run 方法处理模型输出
    result = postprocess_plugin.run(output=output, schema=schema, req_key=req_key)
    # 返回处理结果
    return result

```