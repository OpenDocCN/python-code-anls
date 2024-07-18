# `.\graphrag\graphrag\index\input\text.py`

```py
# 著作权声明和许可证声明
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""包含加载方法定义的模块。"""
# 导入日志记录模块
import logging
# 导入正则表达式模块
import re
# 导入路径操作模块
from pathlib import Path
# 导入类型提示模块
from typing import Any

# 导入 pandas 库并使用 pd 别名
import pandas as pd

# 导入自定义模块
from graphrag.index.config import PipelineInputConfig
from graphrag.index.progress import ProgressReporter
from graphrag.index.storage import PipelineStorage
from graphrag.index.utils import gen_md5_hash

# 定义默认的文件名模式正则表达式
DEFAULT_FILE_PATTERN = re.compile(
    r".*[\\/](?P<source>[^\\/]+)[\\/](?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_(?P<author>[^_]+)_\d+\.txt"
)
# 设置输入类型为文本
input_type = "text"
# 获取日志记录器对象
log = logging.getLogger(__name__)


async def load(
    config: PipelineInputConfig,
    progress: ProgressReporter | None,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """从目录中加载文本输入。"""

    async def load_file(
        path: str, group: dict | None = None, _encoding: str = "utf-8"
    ) -> dict[str, Any]:
        """加载单个文件的内容并返回字典形式的数据。"""
        # 如果未提供组字典，则创建一个空字典
        if group is None:
            group = {}
        # 使用存储对象从路径中获取文本内容
        text = await storage.get(path, encoding="utf-8")
        # 创建新的条目字典，包括原始组信息和文本内容
        new_item = {**group, "text": text}
        # 根据新条目内容生成唯一标识符
        new_item["id"] = gen_md5_hash(new_item, new_item.keys())
        # 将文件路径的最后部分作为标题存储在新条目中
        new_item["title"] = str(Path(path).name)
        return new_item

    # 使用存储对象查找与配置文件模式匹配的文件列表
    files = list(
        storage.find(
            re.compile(config.file_pattern),
            progress=progress,
            file_filter=config.file_filter,
        )
    )
    # 如果没有找到任何文件，抛出 ValueError 异常
    if len(files) == 0:
        msg = f"No text files found in {config.base_dir}"
        raise ValueError(msg)
    # 记录找到的文件信息
    found_files = f"found text files from {config.base_dir}, found {files}"
    log.info(found_files)

    # 存储加载的文件数据的列表
    files_loaded = []

    # 遍历每个文件及其对应的组信息
    for file, group in files:
        try:
            # 调用异步加载单个文件的函数，并将结果添加到 files_loaded 列表中
            files_loaded.append(await load_file(file, group))
        except Exception:  # noqa: BLE001 (catching Exception is fine here)
            # 捕获任何异常情况并记录警告信息，继续下一个文件的处理
            log.warning("Warning! Error loading file %s. Skipping...", file)

    # 记录加载完成的文件数和总文件数
    log.info("Found %d files, loading %d", len(files), len(files_loaded))

    # 将加载的文件数据转换为 pandas 的 DataFrame 并返回
    return pd.DataFrame(files_loaded)
```