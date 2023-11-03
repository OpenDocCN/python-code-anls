# ArknightMower源码解析 10

# `arknights_mower/utils/typealias.py`

这段代码定义了一个Image类和一个Recognizer类。

Image类表示一个二进制图像，使用int8类型表示像素的灰度值，有两个整数类型的参数分别表示图片的维度和灰度值。

Recognizer类有两个类型参数，一个是表示图像范围内像素数量的最大值，另一个是表示坐标偏移量。

这段代码的主要作用是定义一个简单的图像处理库，可以用于图像处理和分析任务。


```py
from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Image
Image = NDArray[np.int8]
Pixel = Tuple[int, int, int]

GrayImage = NDArray[np.int8]
GrayPixel = int

# Recognizer
Range = Tuple[int, int]
Coordinate = Tuple[int, int]
```

这段代码定义了一个命名范围（Scope），包含两个坐标（Coordinate）。接着定义了一个包含范围的切片（Slice），以及一个包含四个坐标的矩形（Rectangle）。然后定义了一个可以存储多个范围的并集类型（Location），可以使用一个范围、一个矩形或一个范围来指定。

接下来定义了一个哈希表类型（Hash）类型的列表（List）类型，用于存储键值对（Key-Value Pair）。再定义了一个分数类型（Score）类型，包含四个浮点数（float）。

接着定义了一个操作计划类型（OpePlan）类型，用于表示如何根据给定的参数找到匹配项。最后定义了一个基于范围的操作计划（BaseConstruct Plan）类型，用于表示如何根据给定的参数找到最匹配的元素。最后，定义了一个基建设置计划（BasePlan）类型，用于存储另一个字典类型（Dict）类型的基础构建计划。


```py
Scope = Tuple[Coordinate, Coordinate]
Slice = Tuple[Range, Range]
Rectangle = Tuple[Coordinate, Coordinate, Coordinate, Coordinate]
Location = Union[Rectangle, Scope, Coordinate]

# Matcher
Hash = List[int]
Score = Tuple[float, float, float, float]

# Operation Plan
OpePlan = Tuple[str, int]

# BaseConstruct Plan
BasePlan = Dict[str, List[str]]

```

这段代码定义了一个名为ParamArgs的参数列表类型。这个列表类型的参数可以用来传递给函数或方法，并且在函数或方法内部被用作参数。

具体来说，这段代码定义了一个名为ParamArgs的列表类型，这个列表类型的每个元素都是str类型的字符串。这个列表类型的参数可以用来传递给函数或方法，在函数或方法内部，这个列表类型的元素被用作参数。例如，如果一个函数需要接收一个字符串和一个整数作为参数，可以传递一个名为ParamArgs的列表类型的参数，这个列表类型包含一个字符串类型和一个整数类型。函数在内部可以访问这个列表类型的元素，并根据需要进行相应的操作。


```py
# Parameter
ParamArgs = List[str]

```

# `arknights_mower/utils/update.py`

这段代码是一个批处理脚本，名为“upgrade.bat”。它实现了两个主要功能：1）用新的程序替换掉旧的程序；2）在替换掉旧的程序后，等待5秒钟并显示更新完成的过程。

具体来说，这段代码首先导入了logging、os和zipfile模块，这些模块在程序运行时用于将日志信息输出到控制台、操作文件系统和将文件压缩打包。

接下来，定义了一个名为“__write_restart_cmd”的函数，该函数生成一个批处理脚本，将新的程序替换掉旧的程序，并在替换成功后启动新的程序。

函数的核心部分是以下几行：


b.write(TempList)        # 将临时列表存储到文件中
b.close()             # 关闭文件
os.system('start upgrade.bat')  # 启动升级程序



if not exist旧_program.bat exit <CODESEEK_LINEN>
timeout /t 5 /nobreak
if exist old_program.bat del old_program.bat
copy old_program.bat new_program.bat
echo 更新完成，正在启动...
start old_program.bat


这段代码的第一个参数是“new_program.bat”和“old_program.bat”两个文件名，它们分别是新旧程序的名称。函数使用os.system函数启动升级程序，并且在成功替换程序后，会等待5秒钟并显示更新完成的过程。最后，使用start函数启动升级程序，该函数的参数是“升级.bat”，这里应该是一个批处理文件名，具体是哪个程序需要根据实际情况而定。


```py
import logging
import os
import zipfile
import requests
from .. import __version__


# 编写bat脚本，删除旧程序，运行新程序
def __write_restart_cmd(new_name, old_name):
    b = open("upgrade.bat", 'w')
    TempList = "@echo off\n"
    TempList += "if not exist " + new_name + " exit \n"  # 判断是否有新版本的程序，没有就退出更新。
    TempList += "echo 正在更新至最新版本...\n"
    TempList += "timeout /t 5 /nobreak\n"  # 等待5秒
    TempList += "if exist " + old_name + ' del "' + old_name.replace("/", "\\\\") + '"\n'  # 删除旧程序
    TempList += 'copy  "' + new_name.replace("/", "\\\\") + '" "' + old_name.replace("/", "\\\\") + '"\n'  # 复制新版本程序
    TempList += "echo 更新完成，正在启动...\n"
    TempList += "timeout /t 3 /nobreak\n"
    TempList += 'start  ' + old_name + ' \n'  # "start 1.bat\n"
    TempList += "exit"
    b.write(TempList)
    b.close()
    # subprocess.Popen("upgrade.bat") #不显示cmd窗口
    os.system('start upgrade.bat')  # 显示cmd窗口
    os._exit(0)


```

这段代码是一个 Python 函数，名为 `compare_version()`，用于比较两个版本的当前项目版本号，并返回更新版本号（若需要更新）或者 None。

首先，函数获取最新的版本号并将其存储在变量 `newest_version` 中。然后，它遍历当前版本号的两部分（通过 `str(__version__).split('.')` 获取），并将它们存储在一个新的列表中。接着，它将新版本号剩余部分补充为零，以使列表长度相等，这样两个列表就可以进行比较了。

接着，函数对列表 `v1` 和 `v2` 进行排序，并对它们进行比较。如果两个列表相等，函数返回 `None`，表示不需要更新。否则，函数返回最新的版本号（若需要更新）。


```py
def compere_version():
    """
        与github上最新版比较
        :return res: str | None, 若需要更新 返回版本号, 否则返回None
    """
    newest_version = __get_newest_version()

    v1 = [str(x) for x in str(__version__).split('.')]
    v2 = [str(x) for x in str(newest_version).split('.')]

    # 如果2个版本号位数不一致，后面使用0补齐，使2个list长度一致，便于后面做对比
    if len(v1) > len(v2):
        v2 += [str(0) for x in range(len(v1) - len(v2))]
    elif len(v1) < len(v2):
        v1 += [str(0) for x in range(len(v2) - len(v1))]
    list_sort = sorted([v1, v2])
    if list_sort[0] == list_sort[1]:
        return None
    elif list_sort[0] == v1:
        return newest_version
    else:
        return None


```

这段代码是一个 Python 脚本，主要功能是更新并下载最新的 "arknights-mower" 库。

具体来说，代码分为以下几个部分：

1. update_version() 函数：
  - 首先检查是否有可执行文件 "upgrade.bat"，若有，则删除该文件。
  - 如果不是可执行文件，则执行以下操作：
    - 使用 requests 库从 GitHub 获取最新的 "arknights-mower" 库版本。
    - 将获取到的版本信息存储在变量 "tag_name"。

2. __get_newest_version() 函数：
  - 使用 requests 库从 GitHub 获取最新的 "arknights-mower" 库版本。
  - 返回获取到的版本信息中的 "tag_name"。

3. download_version() 函数：
  - 如果 "./tmp" 目录不存在，则创建该目录。
  - 使用 requests 库下载 "arknights-mower" 库的最新版本。
  - 将下载的文件名存储在变量 "version"。
  - 解压下载的 zip 文件。
  - 根据需要，将 "arknights-mower-3.0.4.zip" 文件中的 "arknights-mower" 目录复制到 "./tmp" 目录中。
  - 删除下载的 "arknights-mower.zip" 文件。
  - 使用 requests 库下载最新的 "arknights-mower" 库版本。
  - 将下载的版本信息存储在变量 "tag_name"。


```py
def update_version():
    if os.path.isfile("upgrade.bat"):
        os.remove("upgrade.bat")
    __write_restart_cmd("tmp/mower.exe", "./mower.exe")


def __get_newest_version():
    response = requests.get("https://api.github.com/repos/ArkMowers/arknights-mower/releases/latest")
    return response.json()["tag_name"]


def download_version(version):
    if not os.path.isdir("./tmp"):
        os.makedirs("./tmp")
    r = requests.get(f"https://github.com/ArkMowers/arknights-mower/releases/download/{version}/mower.zip",stream=True)
    # r = requests.get(
    #     f"https://github.com/ArkMowers/arknights-mower/releases/download/{version}/arknights-mower-3.0.4.zip",
    #     stream=True)
    total = int(r.headers.get('content-length', 0))
    index = 0
    with open('./tmp/mower.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=10485760):
            if chunk:
                f.write(chunk)
                index += len(chunk)
                print(f"更新进度：{'%.2f%%' % (index*100 / total)}({index}/{total})")
    zip_file = zipfile.ZipFile("./tmp/mower.zip")
    zip_list = zip_file.namelist()

    for f in zip_list:
        zip_file.extract(f, './tmp/')
    zip_file.close()
    os.remove("./tmp/mower.zip")


```

这段代码是一个Python程序，名为“main”。程序的主要作用是在新程序启动时，删除旧程序制造的脚本。

具体来说，这段代码首先检查“升级.bat”文件是否存在。如果文件存在，程序会使用os.remove()函数删除该文件。然后，会使用__write_restart_cmd()函数在当前目录下创建并运行一个名为“newVersion.exe”的新程序。在新程序运行时，会自动调用该函数，并将“newVersion.exe”和“Version.exe”作为参数传递给它，这样新程序就可以卸载旧程序了。


```py
def main():
    # 新程序启动时，删除旧程序制造的脚本
    if os.path.isfile("upgrade.bat"):
        os.remove("upgrade.bat")
    __write_restart_cmd("newVersion.exe", "Version.exe")


if __name__ == '__main__':
    compere_version()

```

# `arknights_mower/utils/yaml.py`

这段代码使用了ruamel.yaml库中的YAML函数，创建了一个YAML对象。这个YAML对象(ruamel.yaml.YAML()或yaml)可以用来读取、修改和输出YAML文件和数据。具体来说，这个代码会在当前目录下创建一个名为test.yml的YAML文件，并向其中读取内容，然后将其转换为Python对象。最后，代码将test.yml文件的内容输出到控制台。


```py
import ruamel.yaml

yaml = ruamel.yaml.YAML()

```

# `arknights_mower/utils/__init__.py`

我需要更具体的上下文来回答你的问题。可以请你提供更多背景和信息，以便我能够帮助你更好地解释代码的作用。


```py

```

# `arknights_mower/utils/device/device.py`

`Main`类是应用的主程序，负责调用所有子程序的函数。主要用


```py
from __future__ import annotations

import time
from typing import Optional

from .. import config
from ..log import logger, save_screenshot
from .adb_client import ADBClient
from .minitouch import MiniTouch
from .scrcpy import Scrcpy


class Device(object):
    """ Android Device """

    class Control(object):
        """ Android Device Control """

        def __init__(self, device: Device, client: ADBClient = None, touch_device: str = None) -> None:
            self.device = device
            self.minitouch = None
            self.scrcpy = None

            if config.ADB_CONTROL_CLIENT == 'minitouch':
                self.minitouch = MiniTouch(client, touch_device)
            elif config.ADB_CONTROL_CLIENT == 'scrcpy':
                self.scrcpy = Scrcpy(client)
            else:
                # MiniTouch does not support Android 10+
                if int(client.android_version().split('.')[0]) < 10:
                    self.minitouch = MiniTouch(client, touch_device)
                else:
                    self.scrcpy = Scrcpy(client)

        def tap(self, point: tuple[int, int]) -> None:
            if self.minitouch:
                self.minitouch.tap([point], self.device.display_frames())
            elif self.scrcpy:
                self.scrcpy.tap(point[0], point[1])
            else:
                raise NotImplementedError

        def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int) -> None:
            if self.minitouch:
                self.minitouch.swipe(
                    [start, end], self.device.display_frames(), duration=duration)
            elif self.scrcpy:
                self.scrcpy.swipe(
                    start[0], start[1], end[0], end[1], duration / 1000)
            else:
                raise NotImplementedError

        def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int) -> None:
            if self.minitouch:
                self.minitouch.swipe(
                    points, self.device.display_frames(), duration=durations, up_wait=up_wait)
            elif self.scrcpy:
                total = len(durations)
                for idx, (S, E, D) in enumerate(zip(points[:-1], points[1:], durations)):
                    self.scrcpy.swipe(S[0], S[1], E[0], E[1], D / 1000,
                                      up_wait / 1000 if idx == total-1 else 0,
                                      fall=idx == 0, lift=idx == total-1)
            else:
                raise NotImplementedError

    def __init__(self, device_id: str = None, connect: str = None, touch_device: str = None) -> None:
        self.device_id = device_id
        self.connect = connect
        self.touch_device = touch_device
        self.client = None
        self.control = None
        self.start()

    def start(self) -> None:
        self.client = ADBClient(self.device_id, self.connect)
        self.control = Device.Control(self, self.client)

    def run(self, cmd: str) -> Optional[bytes]:
        return self.client.run(cmd)

    def launch(self) -> None:
        """ launch the application """
        tap = config.TAP_TO_LAUNCH["enable"]
        x = config.TAP_TO_LAUNCH["x"]
        y = config.TAP_TO_LAUNCH["y"]

        if tap:
            self.run(f'input tap {x} {y}')
        else:
            self.run(f'am start -n {config.APPNAME}/{config.APP_ACTIVITY_NAME}')

    def exit(self, app: str) -> None:
        """ exit the application """
        self.run(f'am force-stop {app}')

    def send_keyevent(self, keycode: int) -> None:
        """ send a key event """
        logger.debug(f'keyevent: {keycode}')
        command = f'input keyevent {keycode}'
        self.run(command)

    def send_text(self, text: str) -> None:
        """ send a text """
        logger.debug(f'text: {repr(text)}')
        text = text.replace('"', '\\"')
        command = f'input text "{text}"'
        self.run(command)

    def screencap(self, save: bool = False) -> bytes:
        """ get a screencap """
        command = 'screencap -p 2>/dev/null'
        screencap = self.run(command)
        if save:
            save_screenshot(screencap)
        return screencap

    def current_focus(self) -> str:
        """ detect current focus app """
        command = 'dumpsys window | grep mCurrentFocus'
        line = self.run(command).decode('utf8')
        return line.strip()[:-1].split(' ')[-1]

    def display_frames(self) -> tuple[int, int, int]:
        """ get display frames if in compatibility mode"""
        if not config.MNT_COMPATIBILITY_MODE:
            return None

        command = 'dumpsys window | grep DisplayFrames'
        line = self.run(command).decode('utf8')
        """ eg. DisplayFrames w=1920 h=1080 r=3 """
        res = line.strip().replace('=', ' ').split(' ')
        return int(res[2]), int(res[4]), int(res[6])

    def tap(self, point: tuple[int, int]) -> None:
        """ tap """
        logger.debug(f'tap: {point}')
        self.control.tap(point)

    def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int = 100) -> None:
        """ swipe """
        logger.debug(f'swipe: {start} -> {end}, duration={duration}')
        self.control.swipe(start, end, duration)

    def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int = 500) -> None:
        """ swipe_ext """
        logger.debug(
            f'swipe_ext: points={points}, durations={durations}, up_wait={up_wait}')
        self.control.swipe_ext(points, durations, up_wait)

    def check_current_focus(self):
        """ check if the application is in the foreground """
        if self.current_focus() != f"{config.APPNAME}/{config.APP_ACTIVITY_NAME}":
            self.launch()
            # wait for app to finish launching
            time.sleep(10)

```

# `arknights_mower/utils/device/utils.py`

这段代码是一个异步函数，用于下载一个文件到临时文件夹中，并返回该文件的文件名。

具体来说，它通过 `requests` 库发送一个 HTTP GET 请求获取一个给定 URL 的内容，然后将内容写入一个临时文件中，并将文件名保存为 `file_name`。最后，它返回 `file_name` 作为结果。

此函数的作用是用于在下载文件后，自动将其保存到系统的临时文件夹中，因此它需要运行在 `__system__` 函数中或者在脚本的开始时定义。


```py
from __future__ import annotations

import http
import socket
import tempfile

import requests

from ... import __system__
from ..log import logger


def download_file(target_url: str) -> str:
    """ download file to temp path, and return its file path for further usage """
    logger.debug(f'downloading: {target_url}')
    resp = requests.get(target_url, verify=False)
    with tempfile.NamedTemporaryFile('wb+', delete=False) as f:
        file_name = f.name
        f.write(resp.content)
    return file_name

```

这段代码定义了一个名为 `is_port_using` 的函数，用于检查端口是否被其他进程或客户端占用。函数接受两个参数 `host` 和 `port`，并返回一个布尔值。

函数首先创建一个 socket 对象并设置其套接字类型为 `AF_INET`(用于 IPv4) 和套接字操作类型为 `SOCK_STREAM`(用于连接套接字)，然后设置一个超时时间为 1 秒。

函数接下来尝试使用创建的 socket 对象连接到 `host` 对应的 IP 地址和 `port` 对应的端口。如果连接成功，函数将返回状态码 0，表示可以连接。如果连接失败，函数将返回一些错误信息并关闭 socket 对象。

总的来说，这段代码的作用是判断端口是否被占用，并尝试连接到该端口，如果连接成功则返回 0。


```py
# def is_port_using(host: str, port: int) -> bool:
#     """ if port is using by others, return True. else return False """
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.settimeout(1)

#     try:
#         result = s.connect_ex((host, port))
#         # if port is using, return code should be 0. (can be connected)
#         return result == 0
#     finally:
#         s.close()

```

# `arknights_mower/utils/device/__init__.py`

这段代码定义了两个类：KeyCode 和 Device。KeyCode 是一个枚举类型，定义了四种不同的按键类型，分别为：

python
enum KeyCode {
   K_ESCAPE = 0,
   K_SPACE = 1,
   K_POPMINUS = 2,
   K_PUMPMAX = 3,
   K_ZERO = 4,
   K_ONE = 5,
   K_睛 = 6,
   K_雪兼顾 = 7,
   K_骨 = 8,
   K_看 = 9,
   K_EXEC = 10,
   K_UNKNOWN = 11
}


Device 类是一个抽象类，没有具体的数据成员。它继承自 Device 的父类，并重写了两个子类：Device 和屏幕相关的类。


```py
from .adb_client.const import KeyCode
from .device import Device

```

# `arknights_mower/utils/device/adb_client/const.py`

在一些编程语言中，可以使用下列代码来获取当前时间的 Unix 时间戳（从 1970 年 1 月 1 日 00:00:00 UTC 到 2022 年 2 月 18 日 18:22:48 UTC 的总共 43 年 10 个月 19 天 9 小时 59 分钟 8 秒）。注意，这里的时间戳不包括闰年和时区。
python
import datetime

today = datetime.datetime.utcnow()
timestamp = today.timetuple().tm_iext()


获取当前时间的 Unix 时间戳（从 1970 年 1 月 1 日 00:00:00 UTC 到 2022 年 2 月 18 日 18:22:48 UTC 的总共 43 年 10 个月 19 天 9 小时 59 分钟 8 秒）。这里的时间戳不包括闰年和时区。
python
import datetime

today = datetime.datetime.utcnow()
timestamp = today.timetuple().tm_iext()

print("获取当前时间的 Unix 时间戳：", timestamp)



```py
class KeyCode:
    """ https://developer.android.com/reference/android/view/KeyEvent.html """

    KEYCODE_CALL = 5                 # 拨号键
    KEYCODE_ENDCALL = 6              # 挂机键
    KEYCODE_HOME = 3                 # Home 键
    KEYCODE_MENU = 82                # 菜单键
    KEYCODE_BACK = 4                 # 返回键
    KEYCODE_SEARCH = 84              # 搜索键
    KEYCODE_CAMERA = 27              # 拍照键
    KEYCODE_FOCUS = 80               # 对焦键
    KEYCODE_POWER = 26               # 电源键
    KEYCODE_NOTIFICATION = 83        # 通知键
    KEYCODE_MUTE = 91                # 话筒静音键
    KEYCODE_VOLUME_MUTE = 164        # 扬声器静音键
    KEYCODE_VOLUME_UP = 24           # 音量 + 键
    KEYCODE_VOLUME_DOWN = 25         # 音量 - 键
    KEYCODE_ENTER = 66               # 回车键
    KEYCODE_ESCAPE = 111             # ESC 键
    KEYCODE_DPAD_CENTER = 23         # 导航键 >> 确定键
    KEYCODE_DPAD_UP = 19             # 导航键 >> 向上
    KEYCODE_DPAD_DOWN = 20           # 导航键 >> 向下
    KEYCODE_DPAD_LEFT = 21           # 导航键 >> 向左
    KEYCODE_DPAD_RIGHT = 22          # 导航键 >> 向右
    KEYCODE_MOVE_HOME = 122          # 光标移动到开始键
    KEYCODE_MOVE_END = 123           # 光标移动到末尾键
    KEYCODE_PAGE_UP = 92             # 向上翻页键
    KEYCODE_PAGE_DOWN = 93           # 向下翻页键
    KEYCODE_DEL = 67                 # 退格键
    KEYCODE_FORWARD_DEL = 112        # 删除键
    KEYCODE_INSERT = 124             # 插入键
    KEYCODE_TAB = 61                 # Tab 键
    KEYCODE_NUM_LOCK = 143           # 小键盘锁
    KEYCODE_CAPS_LOCK = 115          # 大写锁定键
    KEYCODE_BREAK = 121              # Break / Pause 键
    KEYCODE_SCROLL_LOCK = 116        # 滚动锁定键
    KEYCODE_ZOOM_IN = 168            # 放大键
    KEYCODE_ZOOM_OUT = 169           # 缩小键
    KEYCODE_0 = 7                    # 0
    KEYCODE_1 = 8                    # 1
    KEYCODE_2 = 9                    # 2
    KEYCODE_3 = 10                   # 3
    KEYCODE_4 = 11                   # 4
    KEYCODE_5 = 12                   # 5
    KEYCODE_6 = 13                   # 6
    KEYCODE_7 = 14                   # 7
    KEYCODE_8 = 15                   # 8
    KEYCODE_9 = 16                   # 9
    KEYCODE_A = 29                   # A
    KEYCODE_B = 30                   # B
    KEYCODE_C = 31                   # C
    KEYCODE_D = 32                   # D
    KEYCODE_E = 33                   # E
    KEYCODE_F = 34                   # F
    KEYCODE_G = 35                   # G
    KEYCODE_H = 36                   # H
    KEYCODE_I = 37                   # I
    KEYCODE_J = 38                   # J
    KEYCODE_K = 39                   # K
    KEYCODE_L = 40                   # L
    KEYCODE_M = 41                   # M
    KEYCODE_N = 42                   # N
    KEYCODE_O = 43                   # O
    KEYCODE_P = 44                   # P
    KEYCODE_Q = 45                   # Q
    KEYCODE_R = 46                   # R
    KEYCODE_S = 47                   # S
    KEYCODE_T = 48                   # T
    KEYCODE_U = 49                   # U
    KEYCODE_V = 50                   # V
    KEYCODE_W = 51                   # W
    KEYCODE_X = 52                   # X
    KEYCODE_Y = 53                   # Y
    KEYCODE_Z = 54                   # Z
    KEYCODE_PLUS = 81                # +
    KEYCODE_MINUS = 69               # -
    KEYCODE_STAR = 17                # *
    KEYCODE_SLASH = 76               # /
    KEYCODE_EQUALS = 70              # =
    KEYCODE_AT = 77                  # @
    KEYCODE_POUND = 18               # #
    KEYCODE_APOSTROPHE = 75          # '
    KEYCODE_BACKSLASH = 73           # \
    KEYCODE_COMMA = 55               # ,
    KEYCODE_PERIOD = 56              # .
    KEYCODE_LEFT_BRACKET = 71        # [
    KEYCODE_RIGHT_BRACKET = 72       # ]
    KEYCODE_SEMICOLON = 74           # ;
    KEYCODE_GRAVE = 68               # `
    KEYCODE_SPACE = 62               # 空格键
    KEYCODE_MEDIA_PLAY = 126         # 多媒体键 >> 播放
    KEYCODE_MEDIA_STOP = 86          # 多媒体键 >> 停止
    KEYCODE_MEDIA_PAUSE = 127        # 多媒体键 >> 暂停
    KEYCODE_MEDIA_PLAY_PAUSE = 85    # 多媒体键 >> 播放 / 暂停
    KEYCODE_MEDIA_FAST_FORWARD = 90  # 多媒体键 >> 快进
    KEYCODE_MEDIA_REWIND = 89        # 多媒体键 >> 快退
    KEYCODE_MEDIA_NEXT = 87          # 多媒体键 >> 下一首
    KEYCODE_MEDIA_PREVIOUS = 88      # 多媒体键 >> 上一首
    KEYCODE_MEDIA_CLOSE = 128        # 多媒体键 >> 关闭
    KEYCODE_MEDIA_EJECT = 129        # 多媒体键 >> 弹出
    KEYCODE_MEDIA_RECORD = 130       # 多媒体键 >> 录音

```

# `arknights_mower/utils/device/adb_client/core.py`

This is a class that appears to be a part of the Android SDK for Python package, specifically the Android shell (adb_shell) and Android process (adb).

The class provides methods for running various commands and interacting with Android devices using the ADB tool. The methods include:

* `cmd_shell`: runs a shell command on the device. This can be used to run any command that can be passed as an argument.
* `push`: pushes the specified file to the device.
* `process`: runs a specified process on the device. This can take the filepath and a list of arguments as arguments.
* `cmd_push`: pushes the specified file to the device.
* `session`: opens a connection to the Android device. This can be used to later interact with the device.
* `request`: sends a request to the device. This can be used to retrieve information or perform actions.
* `sock`: returns a socket to the device's process.

The class also provides methods for getting the Android version and pushing a file's contents to the device.


```py
from __future__ import annotations

import socket
import subprocess
import time
from typing import Optional, Union

from ... import config
from ...log import logger
from .session import Session
from .socket import Socket
from .utils import adb_buildin, run_cmd


class Client(object):
    """ ADB Client """

    def __init__(self, device_id: str = None, connect: str = None, adb_bin: str = None) -> None:
        self.device_id = device_id
        self.connect = connect
        self.adb_bin = adb_bin
        self.error_limit = 3
        self.__init_adb()
        self.__init_device()

    def __init_adb(self) -> None:
        if self.adb_bin is not None:
            return
        for adb_bin in config.ADB_BINARY:
            logger.debug(f'try adb binary: {adb_bin}')
            if self.__check_adb(adb_bin):
                self.adb_bin = adb_bin
                return
        if config.ADB_BUILDIN is None:
            adb_buildin()
        if self.__check_adb(config.ADB_BUILDIN):
            self.adb_bin = config.ADB_BUILDIN
            return
        raise RuntimeError("Can't start adb server")

    def __init_device(self) -> None:
        # wait for the newly started ADB server to probe emulators
        time.sleep(1)
        if self.device_id is None or self.device_id not in config.ADB_DEVICE:
            self.device_id = self.__choose_devices()
        if self.device_id is None :
            if self.connect is None:
                if config.ADB_DEVICE[0] != '':
                    for connect in config.ADB_CONNECT:
                        Session().connect(connect)
            else:
                Session().connect(self.connect)
            self.device_id = self.__choose_devices()
        elif self.connect is None:
            Session().connect(self.device_id)

        # if self.device_id is None or self.device_id not in config.ADB_DEVICE:
        #     if self.connect is None or self.device_id not in config.ADB_CONNECT:
        #         for connect in config.ADB_CONNECT:
        #             Session().connect(connect)
        #     else:
        #         Session().connect(self.connect)
        #     self.device_id = self.__choose_devices()
        logger.info(self.__available_devices())
        if self.device_id not in self.__available_devices():
            logger.error('未检测到相应设备。请运行 `adb devices` 确认列表中列出了目标模拟器或设备。')
            raise RuntimeError('Device connection failure')

    def __choose_devices(self) -> Optional[str]:
        """ choose available devices """
        devices = self.__available_devices()
        for device in config.ADB_DEVICE:
            if device in devices:
                return device
        if len(devices) > 0 and config.ADB_DEVICE[0] == '':
            logger.debug(devices[0])
            return devices[0]


    def __available_devices(self) -> list[str]:
        """ return available devices """
        return [x[0] for x in Session().devices_list() if x[1] != 'offline']

    def __exec(self, cmd: str, adb_bin: str = None) -> None:
        """ exec command with adb_bin """
        logger.debug(f'client.__exec: {cmd}')
        if adb_bin is None:
            adb_bin = self.adb_bin
        subprocess.run([adb_bin, cmd], check=True)

    def __run(self, cmd: str, restart: bool = True) -> Optional[bytes]:
        """ run command with Session """
        error_limit = 3
        while True:
            try:
                return Session().run(cmd)
            except (socket.timeout, ConnectionRefusedError, RuntimeError):
                if restart and error_limit > 0:
                    error_limit -= 1
                    self.__exec('kill-server')
                    self.__exec('start-server')
                    time.sleep(10)
                    continue
                return

    def check_server_alive(self, restart: bool = True) -> bool:
        """ check adb server if it works """
        return self.__run('host:version', restart) is not None

    def __check_adb(self, adb_bin: str) -> bool:
        """ check adb_bin if it works """
        try:
            self.__exec('start-server', adb_bin)
            if self.check_server_alive(False):
                return True
            self.__exec('kill-server', adb_bin)
            self.__exec('start-server', adb_bin)
            time.sleep(10)
            if self.check_server_alive(False):
                return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False
        else:
            return False

    def session(self) -> Session:
        """ get a session between adb client and adb server """
        if not self.check_server_alive():
            raise RuntimeError('ADB server is not working')
        return Session().device(self.device_id)

    def run(self, cmd: str) -> Optional[bytes]:
        """ run adb exec command """
        logger.debug(f'command: {cmd}')
        error_limit = 3
        while True:
            try:
                resp = self.session().exec(cmd)
                break
            except (socket.timeout, ConnectionRefusedError, RuntimeError) as e:
                if error_limit > 0:
                    error_limit -= 1
                    self.__exec('kill-server')
                    self.__exec('start-server')
                    time.sleep(10)
                    self.__init_device()
                    continue
                raise e
        if len(resp) <= 256:
            logger.debug(f'response: {repr(resp)}')
        return resp

    def cmd(self, cmd: str, decode: bool = False) -> Union[bytes, str]:
        """ run adb command with adb_bin """
        cmd = [self.adb_bin, '-s', self.device_id] + cmd.split(' ')
        return run_cmd(cmd, decode)

    def cmd_shell(self, cmd: str, decode: bool = False) -> Union[bytes, str]:
        """ run adb shell command with adb_bin """
        cmd = [self.adb_bin, '-s', self.device_id, 'shell'] + cmd.split(' ')
        return run_cmd(cmd, decode)

    def cmd_push(self, filepath: str, target: str) -> None:
        """ push file into device with adb_bin """
        cmd = [self.adb_bin, '-s', self.device_id, 'push', filepath, target]
        run_cmd(cmd)

    def process(self, path: str, args: list[str] = [], stderr: int = subprocess.DEVNULL) -> subprocess.Popen:
        logger.debug(f'run process: {path}, args: {args}')
        cmd = [self.adb_bin, '-s', self.device_id, 'shell', path] + args
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr)

    def push(self, target_path: str, target: bytes) -> None:
        """ push file into device """
        self.session().push(target_path, target)

    def stream(self, cmd: str) -> Socket:
        """ run adb command, return socket """
        return self.session().request(cmd, True).sock

    def stream_shell(self, cmd: str) -> Socket:
        """ run adb shell command, return socket """
        return self.stream('shell:' + cmd)

    def android_version(self) -> str:
        """ get android_version """
        return self.cmd_shell('getprop ro.build.version.release', True)

```

# `arknights_mower/utils/device/adb_client/session.py`



This is a class called `ADBDevice` that appears to be a part of the Android Debug Bridge (ADB) protocol. It has methods for disconnecting a device, getting a list of devices that the ADB server knows, and pushing data to a device.

The `connect` method takes a string representing the device to connect to and returns a tuple of the connected device and a status code (0 for success, 1 for failure). The `disconnect` method disconnects from the specified device and logs a message if an error occurs.

The `devices_list` method returns a list of devices that the ADB server knows.

The `push` method takes a string representing the target file path and a byte array or a `bytes` object containing the data to push, and an optional integer `mode` (0o100755) that specifies the mode for the push operation. It sends the data to the device and waits for a response, returning the result. If the push fails, it raises a `RuntimeError`.

Note that the `connect`, `disconnect`, and `push` methods are marked as "private" and have a docstring that does not provide any information about their parameters.


```py
from __future__ import annotations

import socket
import struct
import time

from ... import config
from ...log import logger
from .socket import Socket


class Session(object):
    """ Session between ADB client and ADB server """

    def __init__(self, server: tuple[str, int] = None, timeout: int = None) -> None:
        if server is None:
            server = (config.ADB_SERVER_IP, config.ADB_SERVER_PORT)
        if timeout is None:
            timeout = config.ADB_SERVER_TIMEOUT
        self.server = server
        self.timeout = timeout
        self.device_id = None
        self.sock = Socket(self.server, self.timeout)

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass

    def request(self, cmd: str, reconnect: bool = False) -> Session:
        """ make a service request to ADB server, consult ADB sources for available services """
        cmdbytes = cmd.encode()
        data = b'%04X%b' % (len(cmdbytes), cmdbytes)
        while self.timeout <= 60:
            try:
                self.sock.send(data).check_okay()
                return self
            except socket.timeout:
                logger.warning(f'socket.timeout: {self.timeout}s, +5s')
                self.timeout += 5
                self.sock = Socket(self.server, self.timeout)
                if reconnect:
                    self.device(self.device_id)
        raise socket.timeout(f'server: {self.server}')

    def response(self, recv_all: bool = False) -> bytes:
        """ receive response """
        if recv_all:
            return self.sock.recv_all()
        else:
            return self.sock.recv_response()

    def exec(self, cmd: str) -> bytes:
        """ exec: cmd """
        if len(cmd) == 0:
            raise ValueError('no command specified for exec')
        return self.request('exec:' + cmd, True).response(True)

    def shell(self, cmd: str) -> bytes:
        """ shell: cmd """
        if len(cmd) == 0:
            raise ValueError('no command specified for shell')
        return self.request('shell:' + cmd, True).response(True)

    def host(self, cmd: str) -> bytes:
        """ host: cmd """
        if len(cmd) == 0:
            raise ValueError('no command specified for host')
        return self.request('host:' + cmd, True).response()

    def run(self, cmd: str, recv_all: bool = False) -> bytes:
        """ run command """
        if len(cmd) == 0:
            raise ValueError('no command specified')
        return self.request(cmd, True).response(recv_all)

    def device(self, device_id: str = None) -> Session:
        """ switch to a device """
        self.device_id = device_id
        if device_id is None:
            return self.request('host:transport-any')
        else:
            return self.request('host:transport:' + device_id)

    def connect(self, device: str, throw_error: bool = False) -> None:
        """ connect device [ip:port] """
        resp = self.request(f'host:connect:{device}').response()
        logger.debug(f'adb connect {device}: {repr(resp)}')
        if throw_error and (b'unable' in resp or b'cannot' in resp):
            raise RuntimeError(repr(resp))

    def disconnect(self, device: str, throw_error: bool = False) -> None:
        """ disconnect device [ip:port] """
        resp = self.request(f'host:disconnect:{device}').response()
        logger.debug(f'adb disconnect {device}: {repr(resp)}')
        if throw_error and (b'unable' in resp or b'cannot' in resp):
            raise RuntimeError(repr(resp))

    def devices_list(self) -> list[tuple[str, str]]:
        """ returns list of devices that the adb server knows """
        resp = self.request('host:devices').response().decode(errors='ignore')
        devices = [tuple(line.split('\t')) for line in resp.splitlines()]
        logger.debug(devices)
        return devices

    def push(self, target_path: str, target: bytes, mode=0o100755, mtime: int = None):
        """ push data to device """
        self.request('sync:', True)
        request = b'%s,%d' % (target_path.encode(), mode)
        self.sock.send(b'SEND' + struct.pack('<I', len(request)) + request)
        buf = bytearray(65536+8)
        buf[0:4] = b'DATA'
        idx = 0
        while idx < len(target):
            content = target[idx:idx+65536]
            content_len = len(content)
            idx += content_len
            buf[4:8] = struct.pack('<I', content_len)
            buf[8:8+content_len] = content
            self.sock.sendall(bytes(buf[0:8+content_len]))
        if mtime is None:
            mtime = int(time.time())
        self.sock.send(b'DONE' + struct.pack('<I', mtime))
        response = self.sock.recv_exactly(8)
        if response[:4] != b'OKAY':
            raise RuntimeError('push failed')

```

# `arknights_mower/utils/device/adb_client/socket.py`

以上代码是一个Python的网络库中的`Socket`类，它的`recv`方法接受一个字节数组（`Buffer`类）和一个接收缓冲区（`MemoryView`类）作为参数。该方法返回一个字节数组，其中包含服务器发送给客户端的数据。

`recv`方法接受一个固定长度（`len`）的字节数组（`Buffer`类）和一个接收缓冲区（`MemoryView`类）作为参数。该方法返回一个字节数组，其中包含服务器发送给客户端的数据。

`send`方法发送数据到服务器。

`sendall`方法发送所有数据到服务器。

`recv_into`方法将接收缓冲区中的数据发送到服务器。

`ConnectionError`类表示`Socket`类中一个已知的错误，当尝试创建一个`Socket`实例时，可能会发生网络连接问题。

上述代码中，`recv_exactly`方法用于接收一个指定长度的数据。

python
class Socket:
   def __init__(self, server_address, port=22):
       self.server_address = server_address
       self.port = port
       self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       self.sock.connect((self.server_address, port))
       self.sock.send(b'GET宿主机'))
       self.sock.recv(1024)
       self.address = self.sock.getsockname()
       self.handle = socket.accept((self.server_address, 0), None)
       self.get_peer = self.sock.getpeername()
       self.select_timeout = 100
       self.send_timeout = 100

   def send(self, data: bytes) -> Socket:
       """ send data to server """
       self.sock.send(data)
       return self

   def sendall(self, data: bytes) -> Socket:
       """ send data to server """
       self.sock.sendall(data)
       return self

   def recv(self, length: int) -> bytes:
       return self.sock.recv(length)

   def recv_into(self, buffer, nbytes: int) -> None:
       self.sock.recv_into(buffer, nbytes)

   def get_peer(self) -> bytes:
       return self.sock.snd_sockname()

   def get_port(self) -> int:
       return self.sock.getsockname()[1]

   def check_okay(self) -> None:
       """ check if first 4 bytes is "OKAY" """
       result = self.recv_exactly(4)
       if result != b'OKAY':
           raise ConnectionError(self.recv_response())

   def connect(self, server_address, port=22) -> None:
       """ connect to server """
       self.server_address = server_address
       self.port = port
       self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       self.sock.connect((self.server_address, port))
       self.address = self.sock.getsockname()
       self.handle = socket.accept((self.server_address, 0), None)
       self.get_peer = self.sock.getpeername()
       self.select_timeout = 100
       self.send_timeout = 100


这段代码中，`Socket`类表示网络连接中的服务器端。它接受一个`server_address`（服务器IP地址）和一个`port`（服务器端口号），用于建立与客户端的连接。

`Socket`类中，`connect`方法用于将服务器端地址和端口设置为客户端连接的地址和端口，然后建立一个`Socket`实例并返回。

`get_port`方法用于获取服务器端口号。

`send`和`sendall`方法用于向客户端发送数据。

`recv`、`recv_into`和`get_peer`方法用于接收数据。

`


```py
from __future__ import annotations

import socket

from ...log import logger


class Socket(object):
    """ Connect ADB server with socket """

    def __init__(self, server: tuple[str, int], timeout: int) -> None:
        logger.debug(f'server: {server}, timeout: {timeout}')
        try:
            self.sock = None
            self.sock = socket.create_connection(server, timeout=timeout)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except ConnectionRefusedError as e:
            logger.error(f'ConnectionRefusedError: {server}')
            raise e

    def __enter__(self) -> Socket:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """ close socket """
        self.sock and self.sock.close()
        self.sock = None

    def recv_all(self, chunklen: int = 65536) -> bytes:
        data = []
        buf = bytearray(chunklen)
        view = memoryview(buf)
        pos = 0
        while True:
            if pos >= chunklen:
                data.append(buf)
                buf = bytearray(chunklen)
                view = memoryview(buf)
                pos = 0
            rcvlen = self.sock.recv_into(view)
            if rcvlen == 0:
                break
            view = view[rcvlen:]
            pos += rcvlen
        data.append(buf[:pos])
        return b''.join(data)

    def recv_exactly(self, len: int) -> bytes:
        buf = bytearray(len)
        view = memoryview(buf)
        pos = 0
        while pos < len:
            rcvlen = self.sock.recv_into(view)
            if rcvlen == 0:
                break
            view = view[rcvlen:]
            pos += rcvlen
        if pos != len:
            raise EOFError('recv_exactly %d bytes failed' % len)
        return bytes(buf)

    def recv_response(self) -> bytes:
        """ read a chunk of length indicated by 4 hex digits """
        len = int(self.recv_exactly(4), 16)
        if len == 0:
            return b''
        return self.recv_exactly(len)

    def check_okay(self) -> None:
        """ check if first 4 bytes is "OKAY" """
        result = self.recv_exactly(4)
        if result != b'OKAY':
            raise ConnectionError(self.recv_response())

    def recv(self, len: int) -> bytes:
        return self.sock.recv(len)

    def send(self, data: bytes) -> Socket:
        """ send data to server """
        self.sock.send(data)
        return self

    def sendall(self, data: bytes) -> Socket:
        """ send data to server """
        self.sock.sendall(data)
        return self

    def recv_into(self, buffer, nbytes: int) -> None:
        self.sock.recv_into(buffer, nbytes)

```

# `arknights_mower/utils/device/adb_client/utils.py`

这段代码是一个Python语法的引入，用于支持未来版本的Python2.x版本。这个函数将来的引入将包括一个名为“annotations”的类型声明。

更具体地说，这个代码下载了用于构建Android设备的目标和工具，并将其存储在本地。其中包括Android Debug Bridge(ADB)工具，它允许用户在电脑上运行Android应用程序和设备。

以下是代码中的一些函数和类：

- `from __future__ import annotations`：用于支持Python 2.x版本中的`annotations`类型声明。
- `from shutil import download_file`：用于下载文件。
- `import subprocess`：用于导入`subprocess`模块。
- `import config`：用于导入配置文件。
- `from ...log import logger`：用于导入`logger`类。
- `from ...utils import download_file`：用于导入`download_file`函数。
- `ADB_BUILDIN_URL`：定义了ADB Build-in文件的下载链接。
- `ADB_BUILDIN_FILELIST`：定义了一个Map，用于将ADB Build-in文件映射到操作系统。
- `__system__`：定义了一个内部函数，用于获取操作系统。
- `logger`：定义了一个内部函数，用于输出调试信息。
- `download_file`：定义了一个内部函数，用于下载文件。


```py
from __future__ import annotations

import shutil
import subprocess
from typing import Union

from .... import __system__
from ... import config
from ...log import logger
from ..utils import download_file

ADB_BUILDIN_URL = 'https://oss.nano.ac/arknights_mower/adb-binaries'
ADB_BUILDIN_FILELIST = {
    'linux': ['adb'],
    'windows': ['adb.exe', 'AdbWinApi.dll', 'AdbWinUsbApi.dll'],
    'darwin': ['adb'],
}


```

这段代码是一个Python函数，名为`adb_buildin()`，它用于下载Android Debug Bridge（ADB）的二进制文件。

函数的作用是下载ADB的二进制文件到指定的ADB_BUILDIN文件夹中。首先，函数会检查当前操作系统是否在ADB_BUILDIN_FILELIST中定义过，如果不是，则函数会抛出一个`NotImplementedError`，指出未知系统。

接着，函数遍历ADB_BUILDIN_FILELIST中定义过的操作系统，并下载相应的文件。下载的文件被存储到指定的目标路径中，这个路径可能会被`mkdir`函数创建的文件夹中。

最后，函数会将下载的文件复制到ADB_BUILDIN文件夹中，并设置ADB_BUILDIN文件夹的权限为744。


```py
def adb_buildin() -> None:
    """ download adb_bin """
    folder = config.init_adb_buildin()
    folder.mkdir(exist_ok=True, parents=True)
    if __system__ not in ADB_BUILDIN_FILELIST.keys():
        raise NotImplementedError(f'Unknown system: {__system__}')
    for file in ADB_BUILDIN_FILELIST[__system__]:
        target_path = folder / file
        if not target_path.exists():
            url = f'{ADB_BUILDIN_URL}/{__system__}/{file}'
            logger.debug(f'adb_buildin: {url}')
            tmp_path = download_file(url)
            shutil.copy(tmp_path, str(target_path))
    config.ADB_BUILDIN = folder / ADB_BUILDIN_FILELIST[__system__][0]
    config.ADB_BUILDIN.chmod(0o744)


```

该函数`run_cmd`接受两个参数：一个包含多个字符串的列表`cmd`，和一个布尔参数`decode`，表示是否解码输出。函数的作用是运行命令并返回输出。

函数内部首先尝试使用`subprocess.check_output`函数输出命令，并将输出存储在`r`变量中。如果使用`subprocess.CalledProcessError`异常，则将错误信息和输出存储在`e.output`变量中，并抛出该异常。

如果`decode`参数为真，函数将尝试使用`r.decode`方法解码`r`的输出，并将其存储在`r_decoded`变量中。最后，函数返回`r`或`r_decoded`中的一个值。


```py
def run_cmd(cmd: list[str], decode: bool = False) -> Union[bytes, str]:
    logger.debug(f"run command: {cmd}")
    try:
        r = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.debug(e.output)
        raise e
    if decode:
        return r.decode('utf8')
    return r

```

# `arknights_mower/utils/device/adb_client/__init__.py`

这段代码定义了一个名为 "Client" 的类，继承自 "core" 模块中的 "Client" 类。这个 "Client" 类的作用是在 Android 设备上执行各种自动化任务，比如使用 Android 自带的 Android 自动化库来模拟用户操作，比如打开应用程序、获取屏幕上的元素、编写自动化测试脚本等等。

由于这段代码并没有提供完整的代码，因此我无法提供更多有关这个代码的更具体信息。


```py
from .core import Client as ADBClient

```

# `arknights_mower/utils/device/minitouch/command.py`

This is a Python class that defines a `CommandBuilder` class for building Minitouch commands. The `CommandBuilder` class has several methods for appending content to the command, including a `wait` method for adding a delay, a `commit` method for adding the `/` character to the command to indicate a new command, and a `publish` method to apply the current commands to the device. The `CommandBuilder` class also has methods for appending content to the command using the `append` method, which adds a new line of content to the end of the existing command, and the `up` and `down` methods, which add a `<` followed by the `<contact_id>` and `<content>` parameters to the command.

The `CommandBuilder` class inherits from the `Object` class and has a `__init__` method that initializes the `content` attribute to an empty string and the `delay` attribute to a default value of 0.05 seconds. The `__commit__` method is overridden to add a `<` character to the command and `<content>` parameter, and the `__publish__` method is overridden to apply the current commands to the device.

The `CommandBuilder` class can be实例iated to create a `CommandBuilder` object, which can then be used to build a `Command` object by calling the `CommandBuilder` object's `append`, `commit`, and `publish` methods. The `Command` object can then be passed to the `send` method of the `Session` class to apply the command to the device.


```py
from __future__ import annotations

import time

from ...log import logger
from .session import Session

DEFAULT_DELAY = 0.05


class CommandBuilder(object):
    """ Build command str for minitouch """

    def __init__(self) -> None:
        self.content = ''
        self.delay = 0

    def append(self, new_content: str) -> None:
        self.content += new_content + '\n'

    def commit(self) -> None:
        """ add minitouch command: 'c\n' """
        self.append('c')

    def wait(self, ms: int) -> None:
        """ add minitouch command: 'w <ms>\n' """
        self.append(f'w {ms}')
        self.delay += ms

    def up(self, contact_id: int) -> None:
        """ add minitouch command: 'u <contact_id>\n' """
        self.append(f'u {contact_id}')

    def down(self, contact_id: int, x: int, y: int, pressure: int) -> None:
        """ add minitouch command: 'd <contact_id> <x> <y> <pressure>\n' """
        self.append(f'd {contact_id} {x} {y} {pressure}')

    def move(self, contact_id: int, x: int, y: int, pressure: int) -> None:
        """ add minitouch command: 'm <contact_id> <x> <y> <pressure>\n' """
        self.append(f'm {contact_id} {x} {y} {pressure}')

    def publish(self, session: Session):
        """ apply current commands to device """
        self.commit()
        logger.debug('send operation: %s' % self.content.replace('\n', '\\n'))
        session.send(self.content)
        time.sleep(self.delay / 1000 + DEFAULT_DELAY)
        self.reset()

    def reset(self):
        """ clear current commands """
        self.content = ''

```

# `arknights_mower/utils/device/minitouch/core.py`

这段代码是一个Python程序，它从第三方包中导入了一些在未来可能被定义的函数类型。它还引入了os、time、random、typing、config、log_sync、logger、adb_client、download_file、utils和命令模块的一些成员。

具体来说，这段代码的作用是：

1. 从config模块中导入了一个名为ADBClient的类，以及一个名为Session的类。
2. 从typing模块中导入了一个名为Union的联合类型变量。
3. 导入了一个名为download_file的函数模块。
4. 导入了命令模块中的CommandBuilder类。
5. 导入了命令模块中的Session类。
6. 从_.__future__ import annotations，这表示这段代码使用了Python 3未来的类型注释。
7. 在导入了一些第三方包后，定义了一些变量，包括MNT_PREBUILT_URL，用于下载预构建文件。


```py
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

# MNT_PREBUILT_URL = 'https://github.com/williamfzc/stf-binaries/raw/master/node_modules/minitouch-prebuilt/prebuilt'
```



This function appears to be a implementation of swiping between points in a display, with pressure and duration. It takes in a list of points, displays the points in a specific format, and allows the user to specify certain details such as the pressure on each point, the duration of the pressure, and whether to fall or lift the points.

The function first converts the input points into a list, and then loops through each point, calculating the offset between the current point and the previous point based on the specified parameters. This is then used to create a new list of points that the user can swipe between.

The function then specifies the display frames, the pressure on each point, the duration of the pressure, and whether to fall or lift the points. This is then used to update the display, playing the new points and updating the display frames.

Note that this implementation does not handle the case where the touch point is too far away or the user does not specify a valid display to use.


```py
MNT_PREBUILT_URL = 'https://oss.nano.ac/arknights_mower/minitouch'
MNT_PATH = '/data/local/tmp/minitouch'


class Client(object):
    """ Use minitouch to control Android devices easily """

    def __init__(self, client: ADBClient, touch_device: str = config.MNT_TOUCH_DEVICE) -> None:
        self.client = client
        self.touch_device = touch_device
        self.process = None
        self.start()

    def start(self) -> None:
        self.__install()
        self.__server()

    def __del__(self) -> None:
        self.stop()

    def stop(self) -> None:
        self.__server_stop()

    def __install(self) -> None:
        """ install minitouch for android devices """
        self.abi = self.__get_abi()
        if self.__is_mnt_existed():
            logger.debug(
                f'minitouch already existed in {self.client.device_id}')
        else:
            self.__download_mnt()

    def __get_abi(self) -> str:
        """ query device ABI """
        abi = self.client.cmd_shell('getprop ro.product.cpu.abi', True).strip()
        logger.debug(f'device_abi: {abi}')
        return abi

    def __is_mnt_existed(self) -> bool:
        """ check if minitouch is existed in the device """
        file_list = self.client.cmd_shell('ls /data/local/tmp', True)
        return 'minitouch' in file_list

    def __download_mnt(self) -> None:
        """ download minitouch """
        url = f'{MNT_PREBUILT_URL}/{self.abi}/bin/minitouch'
        logger.info(f'minitouch url: {url}')
        mnt_path = download_file(url)

        # push and grant
        self.client.cmd_push(mnt_path, MNT_PATH)
        self.client.cmd_shell(f'chmod 777 {MNT_PATH}')
        logger.info('minitouch already installed in {MNT_PATH}')

        # remove temp
        os.remove(mnt_path)

    def __server(self) -> None:
        """ execute minitouch with adb shell """
        # self.port = self.__get_port()
        self.port = config.MNT_PORT
        self.__forward_port()
        self.process = None
        r, self.stderr = os.pipe()
        log_sync('minitouch', r).start()
        self.__start_mnt()

        # make sure minitouch is up
        time.sleep(1)
        if not self.check_mnt_alive(False):
            raise RuntimeError('minitouch did not work. see https://github.com/Konano/arknights-mower/issues/82')

    def __server_stop(self) -> None:
        """ stop minitouch """
        self.process and self.process.kill()

    # def __get_port(cls) -> int:
    #     """ get a random port from port set """
    #     while True:
    #         port = random.choice(list(range(20000, 21000)))
    #         if is_port_using(DEFAULT_HOST, port):
    #             return port

    def __forward_port(self) -> None:
        """ allow pc access minitouch with port """
        output = self.client.cmd(
            f'forward tcp:{self.port} localabstract:minitouch')
        logger.debug(f'output: {output}')

    def __start_mnt(self) -> None:
        """ fork a process to start minitouch on android """
        if self.touch_device is None:
            self.process = self.client.process('/data/local/tmp/minitouch', [], self.stderr)
        else:
            self.process = self.client.process('/data/local/tmp/minitouch', ['-d', self.touch_device], self.stderr)

    def check_mnt_alive(self, restart: bool = True) -> bool:
        """ check if minitouch process alive """
        if self.process and self.process.poll() is None:
            return True
        elif restart:
            self.__server_stop()
            self.__forward_port()
            self.__start_mnt()
            time.sleep(1)
            if not (self.process and self.process.poll() is None):
                raise RuntimeError('minitouch did not work. see https://github.com/Konano/arknights-mower/issues/82')
            return True
        return False

    def check_adb_alive(self) -> bool:
        """ check if adb server alive """
        return self.client.check_server_alive()

    def convert_coordinate(self, point: tuple[int, int], display_frames: tuple[int, int, int], max_x: int, max_y: int) -> tuple[int, int]:
        """
        check compatibility mode and convert coordinate
        see details: https://github.com/Konano/arknights-mower/issues/85
        """
        if not config.MNT_COMPATIBILITY_MODE:
            return point
        x, y = point
        w, h, r = display_frames
        if r == 1:
            return [(h - y) * max_x // h, x * max_y // w]
        if r == 3:
            return [y * max_x // h, (w - x) * max_y // w]
        logger.debug(f'warning: unexpected rotation parameter: display_frames({w}, {h}, {r})')
        return point

    def tap(self, points: list[tuple[int, int]], display_frames: tuple[int, int, int], pressure: int = 100, duration: int = None, lift: bool = True) -> None:
        """
        tap on screen with pressure and duration

        :param points: list[int], look like [(x1, y1), (x2, y2), ...]
        :param display_frames: tuple[int, int, int], which means [weight, high, rotation] by "adb shell dumpsys window | grep DisplayFrames"
        :param pressure: default to 100
        :param duration: in milliseconds
        :param lift: if True, "lift" the touch point
        """
        self.check_adb_alive()
        self.check_mnt_alive()

        builder = CommandBuilder()
        points = [list(map(int, point)) for point in points]
        with Session(self.port) as conn:
            for id, point in enumerate(points):
                x, y = self.convert_coordinate(point, display_frames, int(conn.max_x), int(conn.max_y))
                builder.down(id, x, y, pressure)
            builder.commit()

            if duration:
                builder.wait(duration)
                builder.commit()

            if lift:
                for id in range(len(points)):
                    builder.up(id)

            builder.publish(conn)

    def __swipe(self, points: list[tuple[int, int]], display_frames: tuple[int, int, int], pressure: int = 100, duration: Union[list[int], int] = None, up_wait: int = 0, fall: bool = True, lift: bool = True) -> None:
        """
        swipe between points one by one, with pressure and duration

        :param points: list, look like [(x1, y1), (x2, y2), ...]
        :param display_frames: tuple[int, int, int], which means [weight, high, rotation] by "adb shell dumpsys window | grep DisplayFrames"
        :param pressure: default to 100
        :param duration: in milliseconds
        :param up_wait: in milliseconds
        :param fall: if True, "fall" the first touch point
        :param lift: if True, "lift" the last touch point
        """
        self.check_adb_alive()
        self.check_mnt_alive()

        points = [list(map(int, point)) for point in points]
        if not isinstance(duration, list):
            duration = [duration] * (len(points) - 1)
        assert len(duration) + 1 == len(points)

        builder = CommandBuilder()
        with Session(self.port) as conn:
            if fall:
                x, y = self.convert_coordinate(points[0], display_frames, int(conn.max_x), int(conn.max_y))
                builder.down(0, x, y, pressure)
                builder.publish(conn)

            for idx, point in enumerate(points[1:]):
                x, y = self.convert_coordinate(point, display_frames, int(conn.max_x), int(conn.max_y))
                builder.move(0, x, y, pressure)
                if duration[idx-1]:
                    builder.wait(duration[idx-1])
                builder.commit()
            builder.publish(conn)

            if lift:
                builder.up(0)
                if up_wait:
                    builder.wait(up_wait)
                builder.publish(conn)

    def swipe(self, points: list[tuple[int, int]], display_frames: tuple[int, int, int], pressure: int = 100, duration: Union[list[int], int] = None, up_wait: int = 0, part: int = 10, fall: bool = True, lift: bool = True) -> None:
        """
        swipe between points one by one, with pressure and duration
        it will split distance between points into pieces

        :param points: list, look like [(x1, y1), (x2, y2), ...]
        :param display_frames: tuple[int, int, int], which means [weight, high, rotation] by "adb shell dumpsys window | grep DisplayFrames"
        :param pressure: default to 100
        :param duration: in milliseconds
        :param up_wait: in milliseconds
        :param part: default to 10
        :param fall: if True, "fall" the first touch point
        :param lift: if True, "lift" the last touch point
        """
        points = [list(map(int, point)) for point in points]
        if not isinstance(duration, list):
            duration = [duration] * (len(points) - 1)
        assert len(duration) + 1 == len(points)
        
        new_points = [points[0]]
        new_duration = []
        for id in range(1, len(points)):
            pre_point = points[id-1]
            cur_point = points[id]
            offset = (
                (cur_point[0] - pre_point[0]) // part,
                (cur_point[1] - pre_point[1]) // part,
            )
            new_points += [
                (pre_point[0] + i * offset[0], pre_point[1] + i * offset[1])
                for i in range(1, part+1)
            ]
            if duration[id-1] is None:
                new_duration += [None] * part
            else:
                new_duration += [duration[id-1] // part] * part
        self.__swipe(new_points, display_frames, pressure, new_duration, up_wait, fall, lift)

```