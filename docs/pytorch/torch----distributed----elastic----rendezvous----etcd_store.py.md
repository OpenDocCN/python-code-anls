# `.\pytorch\torch\distributed\elastic\rendezvous\etcd_store.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入 datetime, random, time 模块
import datetime
import random
import time
# 导入 base64 模块中的 b64decode, b64encode 函数
from base64 import b64decode, b64encode
# 导入 Optional 类型提示
from typing import Optional

# 导入 etcd 模块，忽略类型检查
import etcd  # type: ignore[import]

# pyre-ignore[21]: Could not find name `Store` in `torch.distributed`.
# 从 torch.distributed 模块导入 Store 类

from torch.distributed import Store


# 延迟一小段随机时间，以减少 CAS 失败
# 这不影响正确性，但会减少对 etcd 服务器的请求
def cas_delay():
    time.sleep(random.uniform(0, 0.1))


# pyre-fixme[11]: Annotation `Store` is not defined as a type.
# 定义 EtcdStore 类，继承自 Store 类，实现了 c10 Store 接口
class EtcdStore(Store):
    """
    Implement a c10 Store interface by piggybacking on the rendezvous etcd instance.

    This is the store object returned by ``EtcdRendezvous``.
    """

    def __init__(
        self,
        etcd_client,
        etcd_store_prefix,
        # Default timeout same as in c10d/Store.hpp
        timeout: Optional[datetime.timedelta] = None,
    ):
        # 调用父类 Store 的初始化方法，必要用于 pybind 的 trampoline。
        super().__init__()

        # 设置 etcd 客户端和存储前缀
        self.client = etcd_client
        self.prefix = etcd_store_prefix

        # 如果设置了超时时间，调用 set_timeout 方法设置超时时间
        if timeout is not None:
            self.set_timeout(timeout)

        # 确保前缀以斜杠结尾
        if not self.prefix.endswith("/"):
            self.prefix += "/"

    def set(self, key, value):
        """
        Write a key/value pair into ``EtcdStore``.

        Both key and value may be either Python ``str`` or ``bytes``.
        """
        # 将键值对写入到 EtcdStore 中，键值可以是 Python 的 str 或 bytes 类型
        self.client.set(key=self.prefix + self._encode(key), value=self._encode(value))

    def get(self, key) -> bytes:
        """
        Get a value by key, possibly doing a blocking wait.

        If key is not immediately present, will do a blocking wait
        for at most ``timeout`` duration or until the key is published.


        Returns:
            value ``(bytes)``

        Raises:
            LookupError - If key still not published after timeout
        """
        # 尝试获取指定键的值，可能会进行阻塞等待
        b64_key = self.prefix + self._encode(key)
        kvs = self._try_wait_get([b64_key])

        # 如果获取到的键值对为 None，则抛出 LookupError
        if kvs is None:
            raise LookupError(f"Key {key} not found in EtcdStore")

        # 返回解码后的值（bytes 类型）
        return self._decode(kvs[b64_key])
    def add(self, key, num: int) -> int:
        """
        Atomically increment a value by an integer amount.

        The integer is represented as a string using base 10. If key is not present,
        a default value of ``0`` will be assumed.

        Returns:
             the new (incremented) value
        """
        # 将键编码为 base64 格式
        b64_key = self._encode(key)
        # c10d Store 假设值是表示为十进制字符串的整数
        try:
            # 假设默认值为 "0"，如果该键尚不存在：
            node = self.client.write(
                key=self.prefix + b64_key,
                value=self._encode(str(num)),  # 即 0 + num
                prevExist=False,
            )
            return int(self._decode(node.value))
        except etcd.EtcdAlreadyExist:
            pass

        while True:
            # 注意：c10d Store 没有删除键的方法，所以我们可以确保它仍然存在。
            node = self.client.get(key=self.prefix + b64_key)
            new_value = self._encode(str(int(self._decode(node.value)) + num))
            try:
                node = self.client.test_and_set(
                    key=node.key, value=new_value, prev_value=node.value
                )
                return int(self._decode(node.value))
            except etcd.EtcdCompareFailed:
                cas_delay()

    def wait(self, keys, override_timeout: Optional[datetime.timedelta] = None):
        """
        Wait until all of the keys are published, or until timeout.

        Raises:
            LookupError - if timeout occurs
        """
        # 将键转换为 base64 格式
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError("Timeout while waiting for keys in EtcdStore")
        # 成功时没有返回值

    def check(self, keys) -> bool:
        """Check if all of the keys are immediately present (without waiting)."""
        # 将键转换为 base64 格式
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(
            b64_keys,
            override_timeout=datetime.timedelta(microseconds=1),  # 如无需等待
        )
        return kvs is not None

    #
    # 以 base64 编码键/值数据，这样我们可以在 EtcdStore 中存储任意二进制数据。
    # 输入可以是 `str` 或 `bytes`。
    # 对于 `str`，假设使用 utf-8 编码。
    #
    def _encode(self, value) -> str:
        if type(value) == bytes:
            return b64encode(value).decode()
        elif type(value) == str:
            return b64encode(value.encode()).decode()
        raise ValueError("Value must be of type str or bytes")

    #
    # 解码 base64 字符串（类型为 `str` 或 `bytes`）。
    # 返回类型为 `bytes`，这在处理存储接口时更方便。
    #
    # 将输入的 value 解码为 bytes 类型，支持输入为 bytes 或 str 类型
    def _decode(self, value) -> bytes:
        if type(value) == bytes:
            return b64decode(value)
        elif type(value) == str:
            return b64decode(value.encode())
        # 如果输入既不是 bytes 类型也不是 str 类型，则抛出 ValueError 异常
        raise ValueError("Value must be of type str or bytes")

    #
    # 一次获取所有（base64 编码的）etcd键，或者等待直到所有键都被发布或超时。
    # 这是公共接口方法的辅助方法。
    #
    # 成功时返回一个字典，键为 etcd 键，值为 etcd 值。
    # 超时时返回 None。
    #
    def _try_wait_get(self, b64_keys, override_timeout=None):
        # 获取超时时间，优先使用传入的 override_timeout，否则使用默认的 self.timeout
        timeout = self.timeout if override_timeout is None else override_timeout  # type: ignore[attr-defined]
        # 计算截止时间点
        deadline = time.time() + timeout.total_seconds()

        while True:
            # 读取整个目录（所有键），筛选出等待的键
            all_nodes = self.client.get(key=self.prefix)
            # 创建一个字典，包含所需的键值对，其中键在 b64_keys 中
            req_nodes = {
                node.key: node.value
                for node in all_nodes.children
                if node.key in b64_keys
            }

            # 如果获取到的键值对数量等于 b64_keys 的长度，则表示所有键都已经可用
            if len(req_nodes) == len(b64_keys):
                # 返回所需的键值对字典
                return req_nodes

            # 计算监听超时时间
            watch_timeout = deadline - time.time()
            # 如果监听超时时间小于等于 0，则返回 None
            if watch_timeout <= 0:
                return None

            try:
                # 发起 etcd 的 watch 操作，监听目录变化
                self.client.watch(
                    key=self.prefix,
                    recursive=True,
                    timeout=watch_timeout,
                    index=all_nodes.etcd_index + 1,
                )
            except etcd.EtcdWatchTimedOut:
                # 如果 watch 操作超时，则判断是否已经超过截止时间
                if time.time() >= deadline:
                    return None
                else:
                    continue
            except etcd.EtcdEventIndexCleared:
                continue
```