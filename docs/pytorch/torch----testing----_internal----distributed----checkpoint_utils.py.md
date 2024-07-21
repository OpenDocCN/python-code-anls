# `.\pytorch\torch\testing\_internal\distributed\checkpoint_utils.py`

```
# mypy: ignore-errors

# 导入必要的模块
import os  # 操作系统接口
import shutil  # 文件操作工具
import tempfile  # 创建临时文件和目录
from functools import wraps  # 提供装饰器用于函数包装
from typing import Any, Callable, Dict, Optional, Tuple  # 类型提示支持

import torch.distributed as dist  # 导入分布式处理模块


def with_temp_dir(
    func: Optional[Callable] = None,
) -> Optional[Callable]:
    """
    用于初始化临时目录以进行分布式检查点的装饰器。
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        """
        包装器函数，用于实际执行装饰的函数，管理临时目录的创建和清理。
        """
        if dist.is_initialized():
            # 仅在 rank 为 0 时创建临时目录
            if dist.get_rank() == 0:
                temp_dir = tempfile.mkdtemp()  # 创建临时目录
                print(f"Using temp directory: {temp_dir}")  # 打印临时目录路径
            else:
                temp_dir = ""  # 其他进程使用空字符串
            object_list = [temp_dir]

            # 广播临时目录到所有其他进程
            os.sync()  # 同步文件系统
            dist.broadcast_object_list(object_list)  # 分布式广播列表
            self.temp_dir = object_list[0]  # 设置实例的临时目录属性
            os.sync()  # 再次同步文件系统
        else:
            temp_dir = tempfile.mkdtemp()  # 创建临时目录
            print(f"No process group initialized, using temp directory: {temp_dir}")  # 打印未初始化分组的消息
            self.temp_dir = temp_dir  # 设置实例的临时目录属性

        try:
            func(self, *args, **kwargs)  # 调用被装饰的函数
        finally:
            # 清理临时目录
            if dist.is_initialized() and dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)  # 删除临时目录及其内容
            else:
                shutil.rmtree(self.temp_dir, ignore_errors=True)  # 删除临时目录及其内容

    return wrapper
```