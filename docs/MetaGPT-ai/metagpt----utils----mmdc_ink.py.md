# `MetaGPT\metagpt\utils\mmdc_ink.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/4 16:12
@Author  : alitrack
@File    : mermaid.py
"""
# 导入base64模块
import base64
# 导入ClientError和ClientSession类
from aiohttp import ClientError, ClientSession
# 导入日志模块中的logger对象
from metagpt.logs import logger

# 定义一个异步函数，将mermaid代码转换为文件
async def mermaid_to_file(mermaid_code, output_file_without_suffix):
    """suffix: png/svg
    :param mermaid_code: mermaid code
    :param output_file_without_suffix: output filename without suffix
    :return: 0 if succeed, -1 if failed
    """
    # 将mermaid代码进行base64编码
    encoded_string = base64.b64encode(mermaid_code.encode()).decode()

    # 遍历后缀列表，分别生成svg和png文件
    for suffix in ["svg", "png"]:
        # 根据后缀生成输出文件名
        output_file = f"{output_file_without_suffix}.{suffix}"
        # 根据后缀确定路径类型
        path_type = "svg" if suffix == "svg" else "img"
        # 构建请求URL
        url = f"https://mermaid.ink/{path_type}/{encoded_string}"
        # 创建异步会话
        async with ClientSession() as session:
            try:
                # 发起GET请求
                async with session.get(url) as response:
                    # 判断响应状态码
                    if response.status == 200:
                        # 读取响应内容
                        text = await response.content.read()
                        # 将内容写入文件
                        with open(output_file, "wb") as f:
                            f.write(text)
                        # 记录日志
                        logger.info(f"Generating {output_file}..")
                    else:
                        # 记录错误日志
                        logger.error(f"Failed to generate {output_file}")
                        return -1
            except ClientError as e:
                # 记录网络错误日志
                logger.error(f"network error: {e}")
                return -1
    return 0

```