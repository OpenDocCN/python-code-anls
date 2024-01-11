# `arknights-mower\arknights_mower\utils\device\scrcpy\core.py`

```
# 从未来版本中导入注解功能
from __future__ import annotations

# 导入 functools 模块
import functools
# 导入 socket 模块
import socket
# 导入 struct 模块
import struct
# 导入 threading 模块
import threading
# 导入 time 模块
import time
# 导入 traceback 模块
import traceback
# 从 typing 模块中导入 Optional 和 Tuple 类型注解
from typing import Optional, Tuple

# 导入 numpy 模块，并重命名为 np
import numpy as np

# 从指定路径中导入 __rootdir__ 模块
from .... import __rootdir__
# 从当前目录中的 log 模块中导入 logger 对象
from ...log import logger
# 从当前目录中的 adb_client 模块中导入 ADBClient 类
from ..adb_client import ADBClient
# 从当前目录中的 adb_client.socket 模块中导入 Socket 类
from ..adb_client.socket import Socket
# 从当前目录中的 control 模块中导入 ControlSender 类
from .control import ControlSender

# 定义常量 SCR_PATH，表示路径为 '/data/local/tmp/minitouch'
SCR_PATH = '/data/local/tmp/minitouch'

# 定义 Client 类
class Client:
    # 初始化方法
    def __init__(
        self,
        client: ADBClient,
        max_width: int = 0,
        bitrate: int = 8000000,
        max_fps: int = 0,
        flip: bool = False,
        block_frame: bool = False,
        stay_awake: bool = False,
        lock_screen_orientation: int = const.LOCK_SCREEN_ORIENTATION_UNLOCKED,
        displayid: Optional[int] = None,
        connection_timeout: int = 3000,
    ):
        """
        Create a scrcpy client, this client won't be started until you call the start function
        Args:
            client: ADB client
            max_width: frame width that will be broadcast from android server
            bitrate: bitrate
            max_fps: maximum fps, 0 means not limited (supported after android 10)
            flip: flip the video
            block_frame: only return nonempty frames, may block cv2 render thread
            stay_awake: keep Android device awake
            lock_screen_orientation: lock screen orientation, LOCK_SCREEN_ORIENTATION_*
            connection_timeout: timeout for connection, unit is ms
        """

        # User accessible
        self.client = client  # 设置客户端
        self.last_frame: Optional[np.ndarray] = None  # 最后一帧图像
        self.resolution: Optional[Tuple[int, int]] = None  # 分辨率
        self.device_name: Optional[str] = None  # 设备名称
        self.control = ControlSender(self)  # 控制发送器

        # Params
        self.flip = flip  # 翻转视频
        self.max_width = max_width  # 最大宽度
        self.bitrate = bitrate  # 比特率
        self.max_fps = max_fps  # 最大帧率
        self.block_frame = block_frame  # 只返回非空帧，可能会阻塞 cv2 渲染线程
        self.stay_awake = stay_awake  # 保持 Android 设备唤醒
        self.lock_screen_orientation = lock_screen_orientation  # 锁定屏幕方向
        self.connection_timeout = connection_timeout  # 连接超时时间
        self.displayid = displayid  # 显示 ID

        # Need to destroy
        self.__server_stream: Optional[Socket] = None  # 服务器流
        self.__video_socket: Optional[Socket] = None  # 视频套接字
        self.control_socket: Optional[Socket] = None  # 控制套接字
        self.control_socket_lock = threading.Lock()  # 控制套接字锁

        self.start()  # 启动

    def __del__(self) -> None:
        self.stop()  # 停止
    # 启动服务器并获取连接
    def __start_server(self) -> None:
        # 构建启动服务器的命令行
        cmdline = f'CLASSPATH={SCR_PATH} app_process /data/local/tmp com.genymobile.scrcpy.Server 1.21 log_level=verbose control=true tunnel_forward=true'
        # 如果显示ID不为空，则添加到命令行中
        if self.displayid is not None:
            cmdline += f' display_id={self.displayid}'
        # 使用客户端的流式shell执行命令行，获取服务器流
        self.__server_stream: Socket = self.client.stream_shell(cmdline)
        # 等待服务器启动
        response = self.__server_stream.recv(100)
        logger.debug(response)
        # 如果响应中不包含'[server]'，则抛出连接错误
        if b'[server]' not in response:
            raise ConnectionError(
                'Failed to start scrcpy-server: ' + response.decode('utf-8', 'ignore'))

    # 部署服务器到安卓设备
    def __deploy_server(self) -> None:
        # 获取服务器文件路径
        server_file_path = __rootdir__ / 'vendor' / 'scrcpy-server-novideo' / 'scrcpy-server-novideo.jar'
        # 读取服务器文件的字节流
        server_buf = server_file_path.read_bytes()
        # 将服务器文件的字节流推送到设备上的指定路径
        self.client.push(SCR_PATH, server_buf)
        # 启动服务器
        self.__start_server()
    # 初始化服务器连接，连接到安卓服务器，包括视频和控制两个套接字
    # 这个方法会设置：video_socket, control_socket, resolution 变量
    def __init_server_connection(self) -> None:
        try:
            # 通过客户端连接到本地抽象套接字'scrcpy'，获取视频套接字
            self.__video_socket = self.client.stream('localabstract:scrcpy')
        except socket.timeout:
            # 如果连接超时，则抛出连接错误
            raise ConnectionError('Failed to connect scrcpy-server')

        # 接收一个字节的数据，用于确认连接是否成功
        dummy_byte = self.__video_socket.recv(1)
        # 如果没有接收到数据或者接收到的数据不是 b'\x00'，则抛出连接错误
        if not len(dummy_byte) or dummy_byte != b'\x00':
            raise ConnectionError('Did not receive Dummy Byte!')

        try:
            # 通过客户端连接到本地抽象套接字'scrcpy'，获取控制套接字
            self.control_socket = self.client.stream('localabstract:scrcpy')
        except socket.timeout:
            # 如果连接超时，则抛出连接错误
            raise ConnectionError('Failed to connect scrcpy-server')

        # 从视频套接字接收设备名称，解码成 utf-8 格式
        self.device_name = self.__video_socket.recv(64).decode('utf-8')
        # 去除设备名称末尾的空字符
        self.device_name = self.device_name.rstrip('\x00')
        # 如果设备名称为空，则抛出连接错误
        if not len(self.device_name):
            raise ConnectionError('Did not receive Device Name!')

        # 从视频套接字接收分辨率数据，并解包成大端模式的两个无符号短整型数
        res = self.__video_socket.recv(4)
        self.resolution = struct.unpack('>HH', res)
        # 设置视频套接字为非阻塞模式（注释掉的代码）
        # self.__video_socket.setblocking(False)

    # 开始监听视频流
    def start(self) -> None:
        try_count = 0
        # 最多尝试 3 次连接
        while try_count < 3:
            try:
                # 部署服务器并初始化服务器连接
                self.__deploy_server()
                time.sleep(0.5)
                self.__init_server_connection()
                break
            except ConnectionError:
                # 如果连接错误，则记录日志并停止连接
                logger.debug(traceback.format_exc())
                logger.warning('Failed to connect scrcpy-server.')
                self.stop()
                logger.warning('Try again in 10 seconds...')
                time.sleep(10)
                try_count += 1
        else:
            # 如果尝试连接 3 次仍然失败，则抛出运行时错误
            raise RuntimeError('Failed to connect scrcpy-server.')
    # 停止监听（包括线程和阻塞）
    def stop(self) -> None:
        # 如果服务器流存在，则关闭服务器流并置为None
        if self.__server_stream is not None:
            self.__server_stream.close()
            self.__server_stream = None
        # 如果控制套接字存在，则关闭控制套接字并置为None
        if self.control_socket is not None:
            self.control_socket.close()
            self.control_socket = None
        # 如果视频套接字存在，则关闭视频套接字并置为None
        if self.__video_socket is not None:
            self.__video_socket.close()
            self.__video_socket = None

    # 检查 adb 服务器是否存活
    def check_adb_alive(self) -> bool:
        """ check if adb server alive """
        return self.client.check_server_alive()

    # 装饰器函数，用于处理稳定性问题
    def stable(f):
        @functools.wraps(f)
        def inner(self: Client, *args, **kwargs):
            try_count = 0
            # 最多尝试3次
            while try_count < 3:
                try:
                    # 调用被装饰的函数
                    f(self, *args, **kwargs)
                    break
                except (ConnectionResetError, BrokenPipeError):
                    # 发生连接重置或管道中断错误时，停止当前操作
                    self.stop()
                    # 等待1秒
                    time.sleep(1)
                    # 检查 adb 服务器是否存活
                    self.check_adb_alive()
                    # 重新启动
                    self.start()
                    try_count += 1
            else:
                # 如果尝试次数超过3次，则抛出运行时错误
                raise RuntimeError('Failed to start scrcpy-server.')
        return inner

    # 使用装饰器处理稳定性问题的 tap 函数
    @stable
    def tap(self, x: int, y: int) -> None:
        # 调用控制对象的 tap 方法
        self.control.tap(x, y)

    @stable
    # 定义一个滑动操作的方法，接受起始点坐标、终点坐标、滑动持续时间、释放前的停留时间、是否允许滑动结束时松手、是否允许滑动结束时抬手
    def swipe(self, x0, y0, x1, y1, move_duraion: float = 1, hold_before_release: float = 0, fall: bool = True, lift: bool = True):
        # 定义每帧的时间间隔
        frame_time = 1 / 60

        # 获取当前时间作为起始时间
        start_time = time.perf_counter()
        # 计算滑动结束时间
        end_time = start_time + move_duraion
        # 如果允许滑动结束时松手，则按下起始点
        fall and self.control.touch(x0, y0, const.ACTION_DOWN)
        # 获取当前时间
        t1 = time.perf_counter()
        # 计算从开始到当前的时间间隔
        step_time = t1 - start_time
        # 如果时间间隔小于每帧时间间隔，则休眠直到满足每帧时间间隔
        if step_time < frame_time:
            time.sleep(frame_time - step_time)
        # 循环直到滑动结束时间
        while True:
            t0 = time.perf_counter()
            if t0 > end_time:
                break
            # 计算当前时间相对于起始时间的进度
            time_progress = (t0 - start_time) / move_duraion
            # 计算滑动路径的进度
            path_progress = time_progress
            # 模拟滑动操作
            self.control.touch(int(x0 + (x1 - x0) * path_progress),
                               int(y0 + (y1 - y0) * path_progress), const.ACTION_MOVE)
            t1 = time.perf_counter()
            # 计算从上一次循环到当前的时间间隔
            step_time = t1 - t0
            # 如果时间间隔小于每帧时间间隔，则休眠直到满足每帧时间间隔
            if step_time < frame_time:
                time.sleep(frame_time - step_time)
        # 模拟滑动结束时的操作
        self.control.touch(x1, y1, const.ACTION_MOVE)
        # 如果释放前的停留时间大于0，则休眠指定时间
        if hold_before_release > 0:
            time.sleep(hold_before_release)
        # 如果允许滑动结束时抬手，则抬手
        lift and self.control.touch(x1, y1, const.ACTION_UP)
```