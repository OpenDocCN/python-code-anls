# `.\graphrag\graphrag\index\storage\__init__.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The Indexing Engine storage package root."""

# 从当前包中导入 BlobPipelineStorage 类和 create_blob_storage 函数
from .blob_pipeline_storage import BlobPipelineStorage, create_blob_storage
# 从当前包中导入 FilePipelineStorage 类
from .file_pipeline_storage import FilePipelineStorage
# 从当前包中导入 load_storage 函数
from .load_storage import load_storage
# 从当前包中导入 MemoryPipelineStorage 类
from .memory_pipeline_storage import MemoryPipelineStorage
# 从当前包中导入 PipelineStorage 类
from .typing import PipelineStorage

# 定义一个列表，包含所有在当前模块中希望公开的名称
__all__ = [
    "BlobPipelineStorage",       # BlobPipelineStorage 类
    "FilePipelineStorage",       # FilePipelineStorage 类
    "MemoryPipelineStorage",     # MemoryPipelineStorage 类
    "PipelineStorage",           # PipelineStorage 类
    "create_blob_storage",       # create_blob_storage 函数
    "load_storage",              # load_storage 函数
]
```