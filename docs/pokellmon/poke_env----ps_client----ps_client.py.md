# `.\PokeLLMon\poke_env\ps_client\ps_client.py`

```
"""
这个模块定义了一个与 Showdown 服务器通信的基类。
"""
# 导入必要的库
import asyncio
import json
import logging
from asyncio import CancelledError, Event, Lock, create_task, sleep
from logging import Logger
from time import perf_counter
from typing import Any, List, Optional, Set

import requests
import websockets.client as ws
from websockets.exceptions import ConnectionClosedOK

from poke_env.concurrency import (
    POKE_LOOP,
    create_in_poke_loop,
    handle_threaded_coroutines,
)
from poke_env.exceptions import ShowdownException
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

# 定义 Pokemon Showdown 客户端类
class PSClient:
    """
    Pokemon Showdown 客户端。

    负责与 Showdown 服务器通信。还实现了一些用于基本任务的高级方法，如更改头像和低级消息处理。
    """

    def __init__(
        self,
        account_configuration: AccountConfiguration,
        *,
        avatar: Optional[int] = None,
        log_level: Optional[int] = None,
        server_configuration: ServerConfiguration,
        start_listening: bool = True,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        """
        :param account_configuration: Account configuration.
        :type account_configuration: AccountConfiguration
        :param avatar: Player avatar id. Optional.
        :type avatar: int, optional
        :param log_level: The player's logger level.
        :type log_level: int. Defaults to logging's default level.
        :param server_configuration: Server configuration.
        :type server_configuration: ServerConfiguration
        :param start_listening: Whether to start listening to the server. Defaults to
            True.
        :type start_listening: bool
        :param ping_interval: How long between keepalive pings (Important for backend
            websockets). If None, disables keepalive entirely.
        :type ping_interval: float, optional
        :param ping_timeout: How long to wait for a timeout of a specific ping
            (important for backend websockets.
            Increase only if timeouts occur during runtime).
            If None pings will never time out.
        :type ping_timeout: float, optional
        """
        # 初始化活动任务集合
        self._active_tasks: Set[Any] = set()
        # 设置 ping_interval 和 ping_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        # 设置服务器配置和账户配置
        self._server_configuration = server_configuration
        self._account_configuration = account_configuration

        # 设置玩家头像
        self._avatar = avatar

        # 创建登录事件和发送锁
        self._logged_in: Event = create_in_poke_loop(Event)
        self._sending_lock = create_in_poke_loop(Lock)

        # 初始化 websocket 和日志记录器
        self.websocket: ws.WebSocketClientProtocol
        self._logger: Logger = self._create_logger(log_level)

        # 如果需要开始监听服务器，则在 POKE_LOOP 线程安全地运行监听协程
        if start_listening:
            self._listening_coroutine = asyncio.run_coroutine_threadsafe(
                self.listen(), POKE_LOOP
            )
    # 异步方法，接受挑战，发送接受挑战的消息给指定用户名
    async def accept_challenge(self, username: str, packed_team: Optional[str]):
        # 断言当前用户已登录
        assert (
            self.logged_in.is_set()
        ), f"Expected player {self.username} to be logged in."
        # 设置队伍
        await self.set_team(packed_team)
        # 发送消息给指定用户名，接受挑战
        await self.send_message("/accept %s" % username)

    # 异步方法，发起挑战，发送挑战消息给指定用户名和格式
    async def challenge(self, username: str, format_: str, packed_team: Optional[str]):
        # 断言当前用户已登录
        assert (
            self.logged_in.is_set()
        ), f"Expected player {self.username} to be logged in."
        # 设置队伍
        await self.set_team(packed_team)
        # 发送挑战消息给指定用户名和格式
        await self.send_message(f"/challenge {username}, {format_}")

    # 创建日志记录器
    def _create_logger(self, log_level: Optional[int]) -> Logger:
        """Creates a logger for the client.

        Returns a Logger displaying asctime and the account's username before messages.

        :param log_level: The logger's level.
        :type log_level: int
        :return: The logger.
        :rtype: Logger
        """
        # 创建以用户名为名称的日志记录器
        logger = logging.getLogger(self.username)

        # 创建流处理器
        stream_handler = logging.StreamHandler()
        # 如果有指定日志级别，设置日志级别
        if log_level is not None:
            logger.setLevel(log_level)

        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)

        # 添加流处理器到日志记录器
        logger.addHandler(stream_handler)
        return logger

    # 异步方法，停止监听
    async def _stop_listening(self):
        # 关闭 WebSocket 连接
        await self.websocket.close()

    # 异步方法，更改玩家的头像
    async def change_avatar(self, avatar_id: Optional[int]):
        """Changes the player's avatar.

        :param avatar_id: The new avatar id. If None, nothing happens.
        :type avatar_id: int
        """
        # 等待用户登录
        await self.wait_for_login()
        # 如果有指定头像 id，发送更改头像的消息
        if avatar_id is not None:
            await self.send_message(f"/avatar {avatar_id}")
    # 异步方法，用于监听 showdown websocket 并分发消息以进行处理
    async def listen(self):
        # 记录日志，表示开始监听 showdown websocket
        self.logger.info("Starting listening to showdown websocket")
        try:
            # 使用 async with 连接到 websocket
            async with ws.connect(
                self.websocket_url,
                max_queue=None,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            ) as websocket:
                # 将 websocket 赋值给实例变量
                self.websocket = websocket
                # 遍历 websocket 接收到的消息
                async for message in websocket:
                    # 记录接收到的消息
                    self.logger.info("\033[92m\033[1m<<<\033[0m %s", message)
                    # 创建任务来处理接收到的消息
                    task = create_task(self._handle_message(str(message)))
                    # 将任务添加到活动任务集合中
                    self._active_tasks.add(task)
                    # 添加任务完成时的回调函数，从活动任务集合中移除任务
                    task.add_done_callback(self._active_tasks.discard)

        except ConnectionClosedOK:
            # 记录警告日志，表示 websocket 连接已关闭
            self.logger.warning(
                "Websocket connection with %s closed", self.websocket_url
            )
        except (CancelledError, RuntimeError) as e:
            # 记录严重错误日志，表示监听被中断
            self.logger.critical("Listen interrupted by %s", e)
        except Exception as e:
            # 记录异常日志
            self.logger.exception(e)
    # 异步方法，用于登录玩家，需要传入分割后的消息列表
    async def log_in(self, split_message: List[str]):
        """Log the player with specified username and password.

        Split message contains information sent by the server. This information is
        necessary to log in.

        :param split_message: Message received from the server that triggers logging in.
        :type split_message: List[str]
        """
        # 如果存在账户密码
        if self.account_configuration.password:
            # 发送登录请求，包括用户名、密码和服务器信息
            log_in_request = requests.post(
                self.server_configuration.authentication_url,
                data={
                    "act": "login",
                    "name": self.account_configuration.username,
                    "pass": self.account_configuration.password,
                    "challstr": split_message[2] + "%7C" + split_message[3],
                },
            )
            # 记录发送认证请求的信息
            self.logger.info("Sending authentication request")
            # 从返回的数据中获取认证信息
            assertion = json.loads(log_in_request.text[1:])["assertion"]
        else:
            # 如果不存在账户密码，则跳过认证请求
            self.logger.info("Bypassing authentication request")
            assertion = ""

        # 发送消息，包括用户名和认证信息
        await self.send_message(f"/trn {self.username},0,{assertion}")

        # 更改头像
        await self.change_avatar(self._avatar)

    # 异步方法，用于搜索排位赛游戏，需要传入比赛格式和打包的队伍信息
    async def search_ladder_game(self, format_: str, packed_team: Optional[str]):
        # 设置队伍信息
        await self.set_team(packed_team)
        # 发送搜索游戏消息，包括比赛格式
        await self.send_message(f"/search {format_}")

    # 异步方法，用于发送消息，可以指定房间和第二条消息
    async def send_message(
        self, message: str, room: str = "", message_2: Optional[str] = None
    ):
        """Sends a message to the specified room.

        `message_2` can be used to send a sequence of length 2.

        :param message: The message to send.
        :type message: str
        :param room: The room to which the message should be sent.
        :type room: str
        :param message_2: Second element of the sequence to be sent. Optional.
        :type message_2: str, optional
        """
        # 如果存在第二个消息，将消息和房间名以及第二个消息用竖线连接起来
        if message_2:
            to_send = "|".join([room, message, message_2])
        else:
            to_send = "|".join([room, message])
        # 发送消息
        await self.websocket.send(to_send)

    async def set_team(self, packed_team: Optional[str]):
        # 如果存在打包的团队信息，发送消息 "/utm {packed_team}"
        if packed_team:
            await self.send_message(f"/utm {packed_team}")
        else:
            # 否则发送消息 "/utm null"
            await self.send_message("/utm null")

    async def stop_listening(self):
        # 停止监听
        await handle_threaded_coroutines(self._stop_listening())

    async def wait_for_login(self, checking_interval: float = 0.001, wait_for: int = 5):
        start = perf_counter()
        # 在指定时间内等待登录
        while perf_counter() - start < wait_for:
            await sleep(checking_interval)
            if self.logged_in:
                return
        # 如果超时仍未登录，则抛出异常
        assert self.logged_in, f"Expected player {self.username} to be logged in."

    @property
    def account_configuration(self) -> AccountConfiguration:
        """The client's account configuration.

        :return: The client's account configuration.
        :rtype: AccountConfiguration
        """
        # 返回客户端的账户配置
        return self._account_configuration

    @property
    def logged_in(self) -> Event:
        """Event object associated with user login.

        :return: The logged-in event
        :rtype: Event
        """
        # 返回与用户登录相关的事件对象
        return self._logged_in

    @property
    def logger(self) -> Logger:
        """Logger associated with the player.

        :return: The logger.
        :rtype: Logger
        """
        # 返回与玩家相关的日志记录器
        return self._logger

    @property
    def server_configuration(self) -> ServerConfiguration:
        """获取客户端的服务器配置信息。

        :return: 客户端的服务器配置信息。
        :rtype: ServerConfiguration
        """
        return self._server_configuration

    @property
    def username(self) -> str:
        """玩家的用户名。

        :return: 玩家的用户名。
        :rtype: str
        """
        return self.account_configuration.username

    @property
    def websocket_url(self) -> str:
        """WebSocket 的 URL。

        它是从服务器 URL 派生而来。

        :return: WebSocket 的 URL。
        :rtype: str
        """
        return f"ws://{self.server_configuration.server_url}/showdown/websocket"
```