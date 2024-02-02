# `arknights-mower\arknights_mower\utils\device\adb_client\core.py`

```py
# 导入必要的模块和类型提示
from __future__ import annotations
import socket
import subprocess
import time
from typing import Optional, Union
# 导入自定义模块和配置
from ... import config
from ...log import logger
from .session import Session
from .socket import Socket
from .utils import adb_buildin, run_cmd

# 定义 ADB 客户端类
class Client(object):
    """ ADB Client """

    # 初始化方法，接受设备 ID、连接方式和 ADB 路径作为参数
    def __init__(self, device_id: str = None, connect: str = None, adb_bin: str = None) -> None:
        # 初始化设备 ID、连接方式和 ADB 路径
        self.device_id = device_id
        self.connect = connect
        self.adb_bin = adb_bin
        # 设置错误限制为 3
        self.error_limit = 3
        # 初始化 ADB
        self.__init_adb()
        # 初始化设备
        self.__init_device()

    # 初始化 ADB 方法
    def __init_adb(self) -> None:
        # 如果已经指定了 ADB 路径，则直接返回
        if self.adb_bin is not None:
            return
        # 遍历配置中的 ADB 路径
        for adb_bin in config.ADB_BINARY:
            # 输出调试信息
            logger.debug(f'try adb binary: {adb_bin}')
            # 检查 ADB 路径是否可用
            if self.__check_adb(adb_bin):
                # 如果可用，则设置 ADB 路径并返回
                self.adb_bin = adb_bin
                return
        # 如果配置中没有指定 ADB 路径，则尝试使用内置的 ADB
        if config.ADB_BUILDIN is None:
            adb_buildin()
        # 检查内置 ADB 是否可用
        if self.__check_adb(config.ADB_BUILDIN):
            # 如果可用，则设置 ADB 路径并返回
            self.adb_bin = config.ADB_BUILDIN
            return
        # 如果都不可用，则抛出运行时错误
        raise RuntimeError("Can't start adb server")
    # 初始化设备连接
    def __init_device(self) -> None:
        # 等待新启动的 ADB 服务器探测模拟器
        time.sleep(1)
        # 如果设备 ID 为空或不在配置文件中的设备列表中，则选择设备
        if self.device_id is None or self.device_id not in config.ADB_DEVICE:
            self.device_id = self.__choose_devices()
        # 如果设备 ID 为空
        if self.device_id is None :
            # 如果连接为空
            if self.connect is None:
                # 如果配置文件中的第一个设备不为空，则尝试连接配置文件中的设备
                if config.ADB_DEVICE[0] != '':
                    for connect in config.ADB_CONNECT:
                        Session().connect(connect)
            else:
                # 否则，连接指定的设备
                Session().connect(self.connect)
            # 重新选择设备
            self.device_id = self.__choose_devices()
        # 如果连接为空
        elif self.connect is None:
            # 连接设备
            Session().connect(self.device_id)

        # 记录可用设备信息
        logger.info(self.__available_devices())
        # 如果设备不在可用设备列表中，则记录错误并抛出异常
        if self.device_id not in self.__available_devices():
            logger.error('未检测到相应设备。请运行 `adb devices` 确认列表中列出了目标模拟器或设备。')
            raise RuntimeError('Device connection failure')

    # 选择可用设备
    def __choose_devices(self) -> Optional[str]:
        """ choose available devices """
        devices = self.__available_devices()
        for device in config.ADB_DEVICE:
            if device in devices:
                return device
        # 如果有可用设备且配置文件中的第一个设备为空，则选择第一个可用设备
        if len(devices) > 0 and config.ADB_DEVICE[0] == '':
            logger.debug(devices[0])
            return devices[0]

    # 返回可用设备列表
    def __available_devices(self) -> list[str]:
        """ return available devices """
        return [x[0] for x in Session().devices_list() if x[1] != 'offline']
    # 使用adb_bin执行命令
    def __exec(self, cmd: str, adb_bin: str = None) -> None:
        """ exec command with adb_bin """
        # 打印调试信息
        logger.debug(f'client.__exec: {cmd}')
        # 如果adb_bin为None，则使用默认的adb_bin
        if adb_bin is None:
            adb_bin = self.adb_bin
        # 使用subprocess模块运行命令
        subprocess.run([adb_bin, cmd], check=True)

    # 使用Session运行命令
    def __run(self, cmd: str, restart: bool = True) -> Optional[bytes]:
        """ run command with Session """
        # 设置错误限制为3次
        error_limit = 3
        # 循环执行命令
        while True:
            try:
                # 使用Session运行命令
                return Session().run(cmd)
            except (socket.timeout, ConnectionRefusedError, RuntimeError):
                # 如果需要重启并且错误限制大于0，则执行重启操作
                if restart and error_limit > 0:
                    error_limit -= 1
                    self.__exec('kill-server')
                    self.__exec('start-server')
                    time.sleep(10)
                    continue
                return

    # 检查adb服务器是否正常工作
    def check_server_alive(self, restart: bool = True) -> bool:
        """ check adb server if it works """
        # 检查adb服务器是否正常工作
        return self.__run('host:version', restart) is not None

    # 检查adb_bin是否正常工作
    def __check_adb(self, adb_bin: str) -> bool:
        """ check adb_bin if it works """
        try:
            # 启动adb服务器
            self.__exec('start-server', adb_bin)
            # 如果adb服务器正常工作，则返回True
            if self.check_server_alive(False):
                return True
            # 关闭adb服务器
            self.__exec('kill-server', adb_bin)
            # 重新启动adb服务器
            self.__exec('start-server', adb_bin)
            time.sleep(10)
            # 如果adb服务器正常工作，则返回True
            if self.check_server_alive(False):
                return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
        else:
            return False

    # 获取adb客户端和adb服务器之间的会话
    def session(self) -> Session:
        """ get a session between adb client and adb server """
        # 如果adb服务器不正常工作，则抛出RuntimeError异常
        if not self.check_server_alive():
            raise RuntimeError('ADB server is not working')
        # 返回一个会话对象
        return Session().device(self.device_id)
    # 运行 adb 执行命令
    def run(self, cmd: str) -> Optional[bytes]:
        """ run adb exec command """
        # 记录调试信息，输出执行的命令
        logger.debug(f'command: {cmd}')
        # 设置错误次数限制
        error_limit = 3
        # 循环执行命令
        while True:
            try:
                # 执行命令并获取响应
                resp = self.session().exec(cmd)
                # 执行成功则跳出循环
                break
            except (socket.timeout, ConnectionRefusedError, RuntimeError) as e:
                # 如果还有错误次数，则减少错误次数，重启 adb 服务，等待一段时间，重新初始化设备，继续循环
                if error_limit > 0:
                    error_limit -= 1
                    self.__exec('kill-server')
                    self.__exec('start-server')
                    time.sleep(10)
                    self.__init_device()
                    continue
                # 如果错误次数用完，则抛出异常
                raise e
        # 如果响应数据长度小于等于256，则记录调试信息，输出响应数据
        if len(resp) <= 256:
            logger.debug(f'response: {repr(resp)}')
        # 返回响应数据
        return resp

    # 运行 adb 命令，使用 adb_bin
    def cmd(self, cmd: str, decode: bool = False) -> Union[bytes, str]:
        """ run adb command with adb_bin """
        # 将命令拆分成列表
        cmd = [self.adb_bin, '-s', self.device_id] + cmd.split(' ')
        # 调用 run_cmd 函数执行命令，并返回结果
        return run_cmd(cmd, decode)

    # 运行 adb shell 命令，使用 adb_bin
    def cmd_shell(self, cmd: str, decode: bool = False) -> Union[bytes, str]:
        """ run adb shell command with adb_bin """
        # 将命令拆分成列表
        cmd = [self.adb_bin, '-s', self.device_id, 'shell'] + cmd.split(' ')
        # 调用 run_cmd 函数执行命令，并返回结果
        return run_cmd(cmd, decode)

    # 将文件推送到设备，使用 adb_bin
    def cmd_push(self, filepath: str, target: str) -> None:
        """ push file into device with adb_bin """
        # 构建推送文件的命令
        cmd = [self.adb_bin, '-s', self.device_id, 'push', filepath, target]
        # 执行推送文件的命令
        run_cmd(cmd)

    # 运行进程，使用 adb_bin
    def process(self, path: str, args: list[str] = [], stderr: int = subprocess.DEVNULL) -> subprocess.Popen:
        # 记录调试信息，输出运行的进程和参数
        logger.debug(f'run process: {path}, args: {args}')
        # 构建运行进程的命令
        cmd = [self.adb_bin, '-s', self.device_id, 'shell', path] + args
        # 启动进程并返回进程对象
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr)

    # 将文件推送到设备
    def push(self, target_path: str, target: bytes) -> None:
        """ push file into device """
        # 使用会话对象将文件推送到设备
        self.session().push(target_path, target)
    # 定义一个方法，用于运行 adb 命令并返回 socket 对象
    def stream(self, cmd: str) -> Socket:
        """ run adb command, return socket """
        # 调用 session 方法创建会话，并发送指定的 adb 命令，返回 socket 对象
        return self.session().request(cmd, True).sock

    # 定义一个方法，用于运行 adb shell 命令并返回 socket 对象
    def stream_shell(self, cmd: str) -> Socket:
        """ run adb shell command, return socket """
        # 调用 stream 方法发送带有 'shell:' 前缀的 adb 命令，返回 socket 对象
        return self.stream('shell:' + cmd)

    # 定义一个方法，用于获取 Android 版本信息
    def android_version(self) -> str:
        """ get android_version """
        # 调用 cmd_shell 方法发送获取 Android 版本信息的 adb shell 命令，返回结果
        return self.cmd_shell('getprop ro.build.version.release', True)
```