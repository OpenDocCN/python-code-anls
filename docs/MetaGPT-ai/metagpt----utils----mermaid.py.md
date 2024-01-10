# `MetaGPT\metagpt\utils\mermaid.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/4 10:53
@Author  : alexanderwu alitrack
@File    : mermaid.py
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""
import asyncio  # 引入异步IO库
import os  # 引入操作系统相关库
from pathlib import Path  # 引入路径相关库

import aiofiles  # 引入异步文件操作库

from metagpt.config import CONFIG  # 从metagpt.config模块中引入CONFIG配置
from metagpt.logs import logger  # 从metagpt.logs模块中引入logger日志记录器
from metagpt.utils.common import check_cmd_exists  # 从metagpt.utils.common模块中引入check_cmd_exists函数


async def mermaid_to_file(mermaid_code, output_file_without_suffix, width=2048, height=2048) -> int:
    """suffix: png/svg/pdf

    :param mermaid_code: mermaid code
    :param output_file_without_suffix: output filename
    :param width:
    :param height:
    :return: 0 if succeed, -1 if failed
    """
    # Write the Mermaid code to a temporary file
    dir_name = os.path.dirname(output_file_without_suffix)  # 获取输出文件路径的目录名
    if dir_name and not os.path.exists(dir_name):  # 如果目录名存在且目录不存在
        os.makedirs(dir_name)  # 创建目录
    tmp = Path(f"{output_file_without_suffix}.mmd")  # 创建临时文件路径
    async with aiofiles.open(tmp, "w", encoding="utf-8") as f:  # 异步打开临时文件
        await f.write(mermaid_code)  # 写入mermaid代码
    # tmp.write_text(mermaid_code, encoding="utf-8")

    engine = CONFIG.mermaid_engine.lower()  # 获取mermaid引擎配置并转换为小写
    if engine == "nodejs":  # 如果引擎为nodejs
        if check_cmd_exists(CONFIG.mmdc) != 0:  # 检查mmdc命令是否存在
            logger.warning(
                "RUN `npm install -g @mermaid-js/mermaid-cli` to install mmdc,"
                "or consider changing MERMAID_ENGINE to `playwright`, `pyppeteer`, or `ink`."
            )  # 记录警告日志
            return -1  # 返回失败状态

        for suffix in ["pdf", "svg", "png"]:  # 遍历后缀列表
            output_file = f"{output_file_without_suffix}.{suffix}"  # 构建输出文件名
            # Call the `mmdc` command to convert the Mermaid code to a PNG
            logger.info(f"Generating {output_file}..")  # 记录信息日志

            if CONFIG.puppeteer_config:  # 如果存在puppeteer配置
                commands = [
                    CONFIG.mmdc,
                    "-p",
                    CONFIG.puppeteer_config,
                    "-i",
                    str(tmp),
                    "-o",
                    output_file,
                    "-w",
                    str(width),
                    "-H",
                    str(height),
                ]  # 构建命令列表
            else:
                commands = [CONFIG.mmdc, "-i", str(tmp), "-o", output_file, "-w", str(width), "-H", str(height)]  # 构建命令列表
            process = await asyncio.create_subprocess_shell(  # 创建子进程
                " ".join(commands), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()  # 获取子进程的标准输出和标准错误
            if stdout:  # 如果存在标准输出
                logger.info(stdout.decode())  # 记录信息日志
            if stderr:  # 如果存在标准错误
                logger.warning(stderr.decode())  # 记录警告日志
    else:  # 如果引擎不是nodejs
        if engine == "playwright":  # 如果引擎是playwright
            from metagpt.utils.mmdc_playwright import mermaid_to_file  # 从metagpt.utils.mmdc_playwright模块中引入mermaid_to_file函数

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, width, height)  # 调用mermaid_to_file函数
        elif engine == "pyppeteer":  # 如果引擎是pyppeteer
            from metagpt.utils.mmdc_pyppeteer import mermaid_to_file  # 从metagpt.utils.mmdc_pyppeteer模块中引入mermaid_to_file函数

            return await mermaid_to_file(mermaid_code, output_file_without_suffix, width, height)  # 调用mermaid_to_file函数
        elif engine == "ink":  # 如果引擎是ink
            from metagpt.utils.mmdc_ink import mermaid_to_file  # 从metagpt.utils.mmdc_ink模块中引入mermaid_to_file函数

            return await mermaid_to_file(mermaid_code, output_file_without_suffix)  # 调用mermaid_to_file函数
        else:  # 如果引擎不是以上任何一种
            logger.warning(f"Unsupported mermaid engine: {engine}")  # 记录警告日志
    return 0  # 返回成功状态


MMC1 = """
classDiagram
    class Main {
        -SearchEngine search_engine
        +main() str
    }
    class SearchEngine {
        -Index index
        -Ranking ranking
        -Summary summary
        +search(query: str) str
    }
    class Index {
        -KnowledgeBase knowledge_base
        +create_index(data: dict)
        +query_index(query: str) list
    }
    class Ranking {
        +rank_results(results: list) list
    }
    class Summary {
        +summarize_results(results: list) str
    }
    class KnowledgeBase {
        +update(data: dict)
        +fetch_data(query: str) dict
    }
    Main --> SearchEngine
    SearchEngine --> Index
    SearchEngine --> Ranking
    SearchEngine --> Summary
    Index --> KnowledgeBase
"""

MMC2 = """
sequenceDiagram
    participant M as Main
    participant SE as SearchEngine
    participant I as Index
    participant R as Ranking
    participant S as Summary
    participant KB as KnowledgeBase
    M->>SE: search(query)
    SE->>I: query_index(query)
    I->>KB: fetch_data(query)
    KB-->>I: return data
    I-->>SE: return results
    SE->>R: rank_results(results)
    R-->>SE: return ranked_results
    SE->>S: summarize_results(ranked_results)
    S-->>SE: return summary
    SE-->>M: return summary
"""

```