# `MetaGPT\metagpt\utils\redis.py`

```py

# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/27
@Author  : mashenquan
@File    : redis.py
"""
# 导入必要的模块
from __future__ import annotations  # 导入未来版本的注解特性

import traceback  # 导入异常追踪模块
from datetime import timedelta  # 导入时间间隔模块

import aioredis  # 导入异步 Redis 客户端模块
# 导入自定义模块
from metagpt.config import CONFIG  # 导入配置模块
from metagpt.logs import logger  # 导入日志模块

# 定义 Redis 类
class Redis:
    def __init__(self):
        self._client = None  # 初始化 Redis 客户端为 None

    # 异步连接 Redis
    async def _connect(self, force=False):
        # 如果已经连接且不需要强制连接，则直接返回 True
        if self._client and not force:
            return True
        # 如果未配置 Redis，则返回 False
        if not self.is_configured:
            return False

        try:
            # 使用配置的信息连接 Redis
            self._client = await aioredis.from_url(
                f"redis://{CONFIG.REDIS_HOST}:{CONFIG.REDIS_PORT}",
                username=CONFIG.REDIS_USER,
                password=CONFIG.REDIS_PASSWORD,
                db=CONFIG.REDIS_DB,
            )
            return True
        except Exception as e:
            logger.warning(f"Redis initialization has failed:{e}")  # 记录连接失败的日志
        return False

    # 异步获取 Redis 中指定 key 的值
    async def get(self, key: str) -> bytes | None:
        # 如果未连接或者 key 为空，则返回 None
        if not await self._connect() or not key:
            return None
        try:
            v = await self._client.get(key)  # 获取指定 key 的值
            return v
        except Exception as e:
            logger.exception(f"{e}, stack:{traceback.format_exc()}")  # 记录异常日志
            return None

    # 异步设置 Redis 中指定 key 的值
    async def set(self, key: str, data: str, timeout_sec: int = None):
        # 如果未连接或者 key 为空，则直接返回
        if not await self._connect() or not key:
            return
        try:
            ex = None if not timeout_sec else timedelta(seconds=timeout_sec)  # 根据超时时间设置过期时间
            await self._client.set(key, data, ex=ex)  # 设置指定 key 的值
        except Exception as e:
            logger.exception(f"{e}, stack:{traceback.format_exc()}")  # 记录异常日志

    # 异步关闭 Redis 连接
    async def close(self):
        if not self._client:
            return
        await self._client.close()  # 关闭 Redis 连接
        self._client = None  # 将 Redis 客户端置为 None

    # 判断 Redis 连接是否有效
    @property
    def is_valid(self) -> bool:
        return self._client is not None  # 返回 Redis 客户端是否有效的布尔值

    # 判断 Redis 是否已配置
    @property
    def is_configured(self) -> bool:
        return bool(
            CONFIG.REDIS_HOST
            and CONFIG.REDIS_HOST != "YOUR_REDIS_HOST"
            and CONFIG.REDIS_PORT
            and CONFIG.REDIS_PORT != "YOUR_REDIS_PORT"
            and CONFIG.REDIS_DB is not None
            and CONFIG.REDIS_PASSWORD is not None
        )  # 返回 Redis 是否已配置的布尔值

```