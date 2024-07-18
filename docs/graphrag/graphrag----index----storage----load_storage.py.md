# `.\graphrag\graphrag\index\storage\load_storage.py`

```py
# 定义了 load_storage 函数，用于加载一个流水线的存储配置
def load_storage(config: PipelineStorageConfig):
    """Load the storage for a pipeline."""
    # 使用匹配语句根据存储配置的类型进行不同的处理
    match config.type:
        # 如果存储类型是内存类型
        case StorageType.memory:
            # 调用 create_memory_storage 函数创建内存存储并返回
            return create_memory_storage()
        # 如果存储类型是 blob 类型
        case StorageType.blob:
            # 将 config 强制转换为 PipelineBlobStorageConfig 类型
            config = cast(PipelineBlobStorageConfig, config)
            # 调用 create_blob_storage 函数创建 blob 存储并返回
            return create_blob_storage(
                config.connection_string,
                config.storage_account_blob_url,
                config.container_name,
                config.base_dir,
            )
        # 如果存储类型是文件类型
        case StorageType.file:
            # 将 config 强制转换为 PipelineFileStorageConfig 类型
            config = cast(PipelineFileStorageConfig, config)
            # 调用 create_file_storage 函数创建文件存储并返回
            return create_file_storage(config.base_dir)
        # 如果存储类型是未知的
        case _:
            # 抛出 ValueError 异常，指示未知的存储类型
            msg = f"Unknown storage type: {config.type}"
            raise ValueError(msg)
```