# `arknights-mower\arknights_mower\utils\device\minitouch\core.py`

```
# 导入必要的模块
from __future__ import annotations
import os
import time
# import random
from typing import Union
from ... import config
from ...log import log_sync, logger
from ..adb_client import ADBClient
from ..utils import download_file
from .command import CommandBuilder
from .session import Session

# minitouch 的预构建 URL
MNT_PREBUILT_URL = 'https://oss.nano.ac/arknights_mower/minitouch'
# minitouch 的路径
MNT_PATH = '/data/local/tmp/minitouch'

# 定义一个名为 Client 的类，用于方便地控制 Android 设备
class Client(object):
    """ Use minitouch to control Android devices easily """

    # 初始化方法，接受 ADBClient 对象和触摸设备的名称作为参数
    def __init__(self, client: ADBClient, touch_device: str = config.MNT_TOUCH_DEVICE) -> None:
        self.client = client
        self.touch_device = touch_device
        self.process = None
        self.start()

    # 启动方法，调用安装和启动 minitouch 的私有方法
    def start(self) -> None:
        self.__install()
        self.__server()

    # 析构方法，在对象被销毁时调用停止方法
    def __del__(self) -> None:
        self.stop()

    # 停止方法，调用停止 minitouch 服务器的私有方法
    def stop(self) -> None:
        self.__server_stop()

    # 安装 minitouch 的私有方法
    def __install(self) -> None:
        """ install minitouch for android devices """
        # 获取设备的 ABI
        self.abi = self.__get_abi()
        # 检查设备上是否已经存在 minitouch
        if self.__is_mnt_existed():
            logger.debug(
                f'minitouch already existed in {self.client.device_id}')
        else:
            self.__download_mnt()

    # 获取设备的 ABI 的私有方法
    def __get_abi(self) -> str:
        """ query device ABI """
        # 通过 adb 命令获取设备的 ABI
        abi = self.client.cmd_shell('getprop ro.product.cpu.abi', True).strip()
        logger.debug(f'device_abi: {abi}')
        return abi

    # 检查设备上是否已经存在 minitouch 的私有方法
    def __is_mnt_existed(self) -> bool:
        """ check if minitouch is existed in the device """
        # 通过 adb 命令列出 /data/local/tmp 目录下的文件，检查是否存在 minitouch
        file_list = self.client.cmd_shell('ls /data/local/tmp', True)
        return 'minitouch' in file_list
    # 下载 minitouch
    def __download_mnt(self) -> None:
        """ download minitouch """
        # 构建 minitouch 的下载链接
        url = f'{MNT_PREBUILT_URL}/{self.abi}/bin/minitouch'
        logger.info(f'minitouch url: {url}')
        # 下载 minitouch 文件到本地
        mnt_path = download_file(url)

        # 推送并授权文件
        self.client.cmd_push(mnt_path, MNT_PATH)
        self.client.cmd_shell(f'chmod 777 {MNT_PATH}')
        logger.info('minitouch already installed in {MNT_PATH}')

        # 删除临时文件
        os.remove(mnt_path)

    # 执行 minitouch
    def __server(self) -> None:
        """ execute minitouch with adb shell """
        # 设置端口号
        self.port = config.MNT_PORT
        # 将端口号转发到设备
        self.__forward_port()
        self.process = None
        # 创建管道，用于接收 minitouch 的日志
        r, self.stderr = os.pipe()
        log_sync('minitouch', r).start()
        self.__start_mnt()

        # 确保 minitouch 已经启动
        time.sleep(1)
        if not self.check_mnt_alive(False):
            raise RuntimeError('minitouch did not work. see https://github.com/Konano/arknights-mower/issues/82')

    # 停止 minitouch
    def __server_stop(self) -> None:
        """ stop minitouch """
        # 如果进程存在，则杀死进程
        self.process and self.process.kill()

    # 允许 PC 访问 minitouch 的端口
    def __forward_port(self) -> None:
        """ allow pc access minitouch with port """
        # 执行 adb 命令，将端口转发到 minitouch
        output = self.client.cmd(
            f'forward tcp:{self.port} localabstract:minitouch')
        logger.debug(f'output: {output}')

    # 在 Android 设备上启动 minitouch 进程
    def __start_mnt(self) -> None:
        """ fork a process to start minitouch on android """
        # 如果触摸设备为空，则启动 minitouch 进程
        if self.touch_device is None:
            self.process = self.client.process('/data/local/tmp/minitouch', [], self.stderr)
        # 否则，使用指定的触摸设备启动 minitouch 进程
        else:
            self.process = self.client.process('/data/local/tmp/minitouch', ['-d', self.touch_device], self.stderr)
    # 检查 minitouch 进程是否存活，可选择是否重启
    def check_mnt_alive(self, restart: bool = True) -> bool:
        """ check if minitouch process alive """
        # 如果 minitouch 进程存在且未结束，则返回 True
        if self.process and self.process.poll() is None:
            return True
        # 如果需要重启 minitouch
        elif restart:
            # 停止 minitouch 服务
            self.__server_stop()
            # 重新转发端口
            self.__forward_port()
            # 启动 minitouch 服务
            self.__start_mnt()
            # 等待 1 秒
            time.sleep(1)
            # 如果 minitouch 进程不存在或已结束，则抛出 RuntimeError
            if not (self.process and self.process.poll() is None):
                raise RuntimeError('minitouch did not work. see https://github.com/Konano/arknights-mower/issues/82')
            return True
        # 如果不需要重启 minitouch，则返回 False
        return False

    # 检查 adb 服务器是否存活
    def check_adb_alive(self) -> bool:
        """ check if adb server alive """
        return self.client.check_server_alive()

    # 转换坐标，根据显示帧信息和最大坐标值进行转换
    def convert_coordinate(self, point: tuple[int, int], display_frames: tuple[int, int, int], max_x: int, max_y: int) -> tuple[int, int]:
        """
        check compatibility mode and convert coordinate
        see details: https://github.com/Konano/arknights-mower/issues/85
        """
        # 如果不是兼容模式，则直接返回原始坐标
        if not config.MNT_COMPATIBILITY_MODE:
            return point
        x, y = point
        w, h, r = display_frames
        # 根据显示帧的旋转参数进行坐标转换
        if r == 1:
            return [(h - y) * max_x // h, x * max_y // w]
        if r == 3:
            return [y * max_x // h, (w - x) * max_y // w]
        # 如果旋转参数不是 1 或 3，则记录警告信息并返回原始坐标
        logger.debug(f'warning: unexpected rotation parameter: display_frames({w}, {h}, {r})')
        return point
    # 定义一个方法，用于在屏幕上施加压力和持续时间进行点击
    def tap(self, points: list[tuple[int, int]], display_frames: tuple[int, int, int], pressure: int = 100, duration: int = None, lift: bool = True) -> None:
        """
        tap on screen with pressure and duration

        :param points: list[int], look like [(x1, y1), (x2, y2), ...]  # 点击的坐标点列表，格式为[(x1, y1), (x2, y2), ...]
        :param display_frames: tuple[int, int, int], which means [weight, high, rotation] by "adb shell dumpsys window | grep DisplayFrames"  # 显示框架的元组，表示[宽度，高度，旋转]，通过"adb shell dumpsys window | grep DisplayFrames"获取
        :param pressure: default to 100  # 默认压力为100
        :param duration: in milliseconds  # 持续时间，单位为毫秒
        :param lift: if True, "lift" the touch point  # 如果为True，则“抬起”触摸点
        """
        self.check_adb_alive()  # 检查adb是否存活
        self.check_mnt_alive()  # 检查mnt是否存活

        builder = CommandBuilder()  # 创建命令构建器对象
        points = [list(map(int, point)) for point in points]  # 将坐标点列表中的坐标转换为整数
        with Session(self.port) as conn:  # 创建会话对象
            for id, point in enumerate(points):  # 遍历坐标点列表
                x, y = self.convert_coordinate(point, display_frames, int(conn.max_x), int(conn.max_y))  # 将坐标点转换为屏幕坐标
                builder.down(id, x, y, pressure)  # 在指定坐标按下触摸，施加压力
            builder.commit()  # 提交命令

            if duration:  # 如果有持续时间
                builder.wait(duration)  # 等待指定的持续时间
                builder.commit()  # 提交命令

            if lift:  # 如果lift为True
                for id in range(len(points)):  # 遍历坐标点列表
                    builder.up(id)  # 抬起触摸点

            builder.publish(conn)  # 发布命令到连接
    # 定义一个私有方法，用于模拟手指在屏幕上滑动的操作
    def __swipe(self, points: list[tuple[int, int]], display_frames: tuple[int, int, int], pressure: int = 100, duration: Union[list[int], int] = None, up_wait: int = 0, fall: bool = True, lift: bool = True) -> None:
        """
        swipe between points one by one, with pressure and duration

        :param points: list, look like [(x1, y1), (x2, y2), ...]，表示滑动的路径点坐标
        :param display_frames: tuple[int, int, int], which means [weight, high, rotation] by "adb shell dumpsys window | grep DisplayFrames"，表示屏幕的尺寸和旋转信息
        :param pressure: default to 100，表示滑动的压力
        :param duration: in milliseconds，表示每段滑动的持续时间
        :param up_wait: in milliseconds，表示抬起手指后的等待时间
        :param fall: if True, "fall" the first touch point，如果为True，则模拟第一个触摸点按下
        :param lift: if True, "lift" the last touch point，如果为True，则模拟最后一个触摸点抬起
        """
        # 检查ADB连接是否正常
        self.check_adb_alive()
        # 检查设备挂载状态是否正常
        self.check_mnt_alive()

        # 将路径点坐标转换为整数类型
        points = [list(map(int, point)) for point in points]
        # 如果持续时间不是列表，则将其转换为与路径点数量相同的列表
        if not isinstance(duration, list):
            duration = [duration] * (len(points) - 1)
        # 断言持续时间列表长度为路径点数量减一
        assert len(duration) + 1 == len(points)

        # 创建命令构建器对象
        builder = CommandBuilder()
        # 创建会话对象
        with Session(self.port) as conn:
            # 如果需要模拟按下第一个触摸点
            if fall:
                x, y = self.convert_coordinate(points[0], display_frames, int(conn.max_x), int(conn.max_y))
                builder.down(0, x, y, pressure)
                builder.publish(conn)

            # 遍历路径点，模拟手指在屏幕上滑动
            for idx, point in enumerate(points[1:]):
                x, y = self.convert_coordinate(point, display_frames, int(conn.max_x), int(conn.max_y))
                builder.move(0, x, y, pressure)
                if duration[idx-1]:
                    builder.wait(duration[idx-1])
                builder.commit()
            builder.publish(conn)

            # 如果需要模拟抬起最后一个触摸点
            if lift:
                builder.up(0)
                if up_wait:
                    builder.wait(up_wait)
                builder.publish(conn)
    # 定义一个方法用于在屏幕上进行滑动操作
    def swipe(self, points: list[tuple[int, int]], display_frames: tuple[int, int, int], pressure: int = 100, duration: Union[list[int], int] = None, up_wait: int = 0, part: int = 10, fall: bool = True, lift: bool = True) -> None:
        """
        swipe between points one by one, with pressure and duration
        it will split distance between points into pieces

        :param points: list, look like [(x1, y1), (x2, y2), ...]  # 滑动的起始点和终点坐标列表
        :param display_frames: tuple[int, int, int], which means [weight, high, rotation] by "adb shell dumpsys window | grep DisplayFrames"  # 屏幕的尺寸和旋转信息
        :param pressure: default to 100  # 默认的滑动压力
        :param duration: in milliseconds  # 滑动持续时间，单位为毫秒
        :param up_wait: in milliseconds  # 抬起手指后的等待时间，单位为毫秒
        :param part: default to 10  # 将每个点之间的距离分成多少份
        :param fall: if True, "fall" the first touch point  # 如果为True，则"落下"第一个触摸点
        :param lift: if True, "lift" the last touch point  # 如果为True，则"抬起"最后一个触摸点
        """
        # 将坐标点转换为整数类型的列表
        points = [list(map(int, point)) for point in points]
        # 如果持续时间不是列表，则将其转换为与点数相匹配的列表
        if not isinstance(duration, list):
            duration = [duration] * (len(points) - 1)
        # 断言持续时间的长度等于点数减一
        assert len(duration) + 1 == len(points)
        
        # 初始化新的点和持续时间列表
        new_points = [points[0]]
        new_duration = []
        # 遍历每个点，计算新的点和持续时间
        for id in range(1, len(points)):
            pre_point = points[id-1]
            cur_point = points[id]
            offset = (
                (cur_point[0] - pre_point[0]) // part,  # 计算x方向的偏移量
                (cur_point[1] - pre_point[1]) // part,  # 计算y方向的偏移量
            )
            new_points += [
                (pre_point[0] + i * offset[0], pre_point[1] + i * offset[1])  # 计算新的点的坐标
                for i in range(1, part+1)
            ]
            if duration[id-1] is None:
                new_duration += [None] * part  # 如果持续时间为None，则添加相应数量的None
            else:
                new_duration += [duration[id-1] // part] * part  # 否则添加相应数量的持续时间
        # 调用私有方法进行滑动操作
        self.__swipe(new_points, display_frames, pressure, new_duration, up_wait, fall, lift)
```