# `.\pytorch\torch\utils\file_baton.py`

```
# mypy: allow-untyped-defs
# 引入操作系统和时间模块
import os
import time

# 文件锁同步工具类
class FileBaton:
    """A primitive, file-based synchronization utility."""

    def __init__(self, lock_file_path, wait_seconds=0.1):
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
        """
        # 初始化文件锁的路径和等待时间
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None  # 文件描述符初始化为空

    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
        try:
            # 尝试以原子操作创建一个具有独占访问权限的文件
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL)
            return True
        except FileExistsError:
            # 如果文件已经存在，则获取锁失败
            return False

    def wait(self):
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        # 当锁文件存在时，周期性地休眠一段时间
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)

    def release(self):
        """Release the baton and removes its file."""
        # 如果文件描述符存在，则关闭文件描述符
        if self.fd is not None:
            os.close(self.fd)

        # 移除锁文件
        os.remove(self.lock_file_path)
```