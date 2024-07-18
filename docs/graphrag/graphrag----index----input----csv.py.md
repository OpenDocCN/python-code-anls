# `.\graphrag\graphrag\index\input\csv.py`

```py
# 版权声明和许可证声明，指明版权归属和许可协议
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""定义了加载方法的模块。"""

# 导入日志记录模块
import logging
# 导入正则表达式模块
import re
# 导入字节流模块
from io import BytesIO
# 导入类型转换相关的工具
from typing import cast

# 导入 pandas 库
import pandas as pd

# 导入自定义模块
from graphrag.index.config import PipelineCSVInputConfig, PipelineInputConfig
from graphrag.index.progress import ProgressReporter
from graphrag.index.storage import PipelineStorage
from graphrag.index.utils import gen_md5_hash

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 默认的文件名模式正则表达式，匹配以 .csv 结尾的文件名
DEFAULT_FILE_PATTERN = re.compile(r"(?P<filename>[^\\/]).csv$")

# 输入类型，默认为 "csv"
input_type = "csv"


async def load(
    config: PipelineInputConfig,
    progress: ProgressReporter | None,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """从目录加载 CSV 输入文件。"""
    # 将 config 强制转换为 PipelineCSVInputConfig 类型
    csv_config = cast(PipelineCSVInputConfig, config)
    # 记录日志，指明正在加载的 CSV 文件所在目录
    log.info("Loading csv files from %s", csv_config.base_dir)

    # 设置文件名匹配模式，如果未指定则使用默认的文件名模式
    file_pattern = (
        re.compile(config.file_pattern)
        if config.file_pattern is not None
        else DEFAULT_FILE_PATTERN
    )
    # 在存储中查找符合条件的文件列表
    files = list(
        storage.find(
            file_pattern,
            progress=progress,
            file_filter=config.file_filter,
        )
    )

    # 如果找不到符合条件的文件，则抛出异常
    if len(files) == 0:
        msg = f"No CSV files found in {config.base_dir}"
        raise ValueError(msg)

    # 用于存储已加载的文件数据列表
    files_loaded = []

    # 遍历符合条件的文件列表
    for file, group in files:
        try:
            # 异步加载文件数据，并添加到 files_loaded 列表中
            files_loaded.append(await load_file(file, group))
        except Exception:  # noqa: BLE001 (catching Exception is fine here)
            # 捕获任何异常并记录警告日志，跳过当前文件加载
            log.warning("Warning! Error loading csv file %s. Skipping...", file)

    # 记录日志，指明找到的 CSV 文件数量和成功加载的文件数量
    log.info("Found %d csv files, loading %d", len(files), len(files_loaded))
    # 合并所有已加载文件的数据，生成结果数据帧
    result = pd.concat(files_loaded)
    # 记录日志，指明未过滤的 CSV 行总数
    total_files_log = f"Total number of unfiltered csv rows: {len(result)}"
    log.info(total_files_log)
    # 返回合并后的数据帧作为加载结果
    return result
```