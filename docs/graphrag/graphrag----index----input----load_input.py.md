# `.\graphrag\graphrag\index\input\load_input.py`

```py
# 版权声明，声明使用 MIT 许可证
"""A module containing load_input method definition."""

# 导入所需的库
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pandas as pd

# 导入自定义模块和类
from graphrag.config import InputConfig, InputType
from graphrag.index.config import PipelineInputConfig
from graphrag.index.progress import NullProgressReporter, ProgressReporter
from graphrag.index.storage import (
    BlobPipelineStorage,
    FilePipelineStorage,
)

# 导入子模块和函数
from .csv import input_type as csv
from .csv import load as load_csv
from .text import input_type as text
from .text import load as load_text

# 获取日志记录器
log = logging.getLogger(__name__)
# 创建加载器的字典，用于根据类型选择对应的加载方法
loaders: dict[str, Callable[..., Awaitable[pd.DataFrame]]] = {
    text: load_text,
    csv: load_csv,
}

# 异步加载输入数据的方法
async def load_input(
    config: PipelineInputConfig | InputConfig,
    progress_reporter: ProgressReporter | None = None,
    root_dir: str | None = None,
) -> pd.DataFrame:
    """Load the input data for a pipeline."""
    # 如果未指定根目录，则设置为空字符串
    root_dir = root_dir or ""
    # 记录加载的信息
    log.info("loading input from root_dir=%s", config.base_dir)
    progress_reporter = progress_reporter or NullProgressReporter()

    # 如果未提供输入配置，则引发 ValueError
    if config is None:
        msg = "No input specified!"
        raise ValueError(msg)

    # 根据输入类型选择不同的存储方式
    match config.type:
        case InputType.blob:
            log.info("using blob storage input")
            # 检查配置是否完整，如果不完整则引发 ValueError
            if config.container_name is None:
                msg = "Container name required for blob storage"
                raise ValueError(msg)
            if (
                config.connection_string is None
                and config.storage_account_blob_url is None
            ):
                msg = "Connection string or storage account blob url required for blob storage"
                raise ValueError(msg)
            # 使用 BlobPipelineStorage 存储方式
            storage = BlobPipelineStorage(
                connection_string=config.connection_string,
                storage_account_blob_url=config.storage_account_blob_url,
                container_name=config.container_name,
                path_prefix=config.base_dir,
            )
        case InputType.file:
            log.info("using file storage for input")
            # 使用 FilePipelineStorage 存储方式
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )
        case _:
            log.info("using file storage for input")
            # 使用默认的 FilePipelineStorage 存储方式
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )

    # 若加载器中有针对指定文件类型的加载方法，则使用加载器加载数据
    if config.file_type in loaders:
        # 创建进度报告器
        progress = progress_reporter.child(
            f"Loading Input ({config.file_type})", transient=False
        )
        loader = loaders[config.file_type]
        results = await loader(config, progress, storage)
        # 返回加载的数据结果
        return cast(pd.DataFrame, results)

    # 若加载器中未包含指定文件类型的加载方法，则引发 ValueError
    msg = f"Unknown input type {config.file_type}"
    raise ValueError(msg)
```