# `.\graphrag\graphrag\index\storage\file_pipeline_storage.py`

```py
    ) -> Any:
        """Retrieve a file from storage asynchronously."""
        file_path = Path(self._root_dir) / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {key}")

        if as_bytes:
            async with aiofiles.open(file_path, mode="rb") as file:
                return await file.read()

        encoding_to_use = encoding or self._encoding
        async with aiofiles.open(file_path, mode="r", encoding=encoding_to_use) as file:
            return await file.read()

    async def put(
        self, key: str, data: Any, encoding: str | None = None
    ) -> None:
        """Store data into a file asynchronously."""
        file_path = Path(self._root_dir) / key
        async with aiofiles.open(file_path, mode="w", encoding=encoding or self._encoding) as file:
            await file.write(data)

    async def delete(self, key: str) -> None:
        """Delete a file from storage asynchronously."""
        file_path = Path(self._root_dir) / key
        if file_path.exists():
            await remove(file_path)

    async def move(self, src_key: str, dest_key: str) -> None:
        """Move a file within the storage asynchronously."""
        src_path = Path(self._root_dir) / src_key
        dest_path = Path(self._root_dir) / dest_key
        await aiofiles.os.rename(src_path, dest_path)

    async def copy(self, src_key: str, dest_key: str) -> None:
        """Copy a file within the storage asynchronously."""
        src_path = Path(self._root_dir) / src_key
        dest_path = Path(self._root_dir) / dest_key
        async with aiofiles.open(src_path, 'rb') as src_file:
            async with aiofiles.open(dest_path, 'wb') as dest_file:
                while True:
                    chunk = await src_file.read(1024)
                    if not chunk:
                        break
                    await dest_file.write(chunk)

    async def exists(self, key: str) -> bool:
        """Check if a file exists in storage asynchronously."""
        file_path = Path(self._root_dir) / key
        return await exists(file_path)

    async def cleanup(self) -> None:
        """Cleanup resources asynchronously."""
        try:
            shutil.rmtree(self._root_dir)
        except FileNotFoundError:
            pass
    async def get(self, key: str) -> Any:
        """获取方法的定义。"""
        # 构建文件路径
        file_path = join_path(self._root_dir, key)

        # 如果键存在，读取文件内容并返回
        if await self.has(key):
            return await self._read_file(file_path, as_bytes, encoding)
        
        # 如果键不存在但是文件存在（预加载的新文件），也读取文件内容并返回
        if await exists(key):
            return await self._read_file(key, as_bytes, encoding)

        # 键不存在，返回 None
        return None

    async def _read_file(
        self,
        path: str | Path,
        as_bytes: bool | None = False,
        encoding: str | None = None,
    ) -> Any:
        """读取文件内容。"""
        # 根据 as_bytes 决定读取方式
        read_type = "rb" if as_bytes else "r"
        # 如果是文本文件，设置编码；如果是二进制文件，则不需要编码
        encoding = None if as_bytes else (encoding or self._encoding)

        # 使用 aiofiles 异步打开文件
        async with aiofiles.open(
            path,
            cast(Any, read_type),
            encoding=encoding,
        ) as f:
            return await f.read()

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """设置方法的定义。"""
        # 判断值是否为字节流
        is_bytes = isinstance(value, bytes)
        # 根据值的类型确定写入方式
        write_type = "wb" if is_bytes else "w"
        # 如果是文本文件，设置编码；如果是字节流，则不需要编码
        encoding = None if is_bytes else encoding or self._encoding

        # 使用 aiofiles 异步打开文件并写入内容
        async with aiofiles.open(
            join_path(self._root_dir, key), cast(Any, write_type), encoding=encoding
        ) as f:
            await f.write(value)

    async def has(self, key: str) -> bool:
        """判断键是否存在的方法。"""
        # 判断键对应的文件是否存在
        return await exists(join_path(self._root_dir, key))

    async def delete(self, key: str) -> None:
        """删除方法的定义。"""
        # 如果键存在，删除对应的文件
        if await self.has(key):
            await remove(join_path(self._root_dir, key))

    async def clear(self) -> None:
        """清除方法的定义。"""
        # 遍历根目录下的所有文件和文件夹，并逐一删除
        for file in Path(self._root_dir).glob("*"):
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

    def child(self, name: str | None) -> "PipelineStorage":
        """创建子存储实例的方法。"""
        # 如果子存储的名称为 None，则返回当前实例本身
        if name is None:
            return self
        # 否则返回一个新的子存储实例，其根目录为当前根目录下的子目录
        return FilePipelineStorage(str(Path(self._root_dir) / Path(name)))
# 将文件路径和文件名合并成一个完整的路径。不受操作系统影响。
def join_path(file_path: str, file_name: str) -> Path:
    """Join a path and a file. Independent of the OS."""
    # 使用 Path 类来合并文件路径和文件名的父目录，再加上文件名本身，形成完整的路径
    return Path(file_path) / Path(file_name).parent / Path(file_name).name


# 创建基于文件的存储对象。
def create_file_storage(out_dir: str | None) -> PipelineStorage:
    """Create a file based storage."""
    # 记录日志，指示在 out_dir 处创建文件存储
    log.info("Creating file storage at %s", out_dir)
    # 返回一个 FilePipelineStorage 类的实例，该实例在指定的 out_dir 中存储文件
    return FilePipelineStorage(out_dir)


# 创建进度状态对象，描述加载和过滤的文件数量。
def _create_progress_status(
    num_loaded: int, num_filtered: int, num_total: int
) -> Progress:
    # 返回一个 Progress 类的实例，描述加载的总文件数、已完成的文件数和简要说明
    return Progress(
        total_items=num_total,
        completed_items=num_loaded + num_filtered,
        description=f"{num_loaded} files loaded ({num_filtered} filtered)",
    )
```