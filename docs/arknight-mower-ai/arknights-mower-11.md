# ArknightMower源码解析 11

# `/opt/arknights-mower/arknights_mower/utils/device/minitouch/session.py`

This is a simple implementation of a minimap protocol that establishes a connection to a remote device. The `Minitouch` class is a simple class that wraps the connection and provides a `send` method to send data and a `close` method to close the connection.

The `Minitouch` class uses the `socket` module provided by Python to establish the connection. The connection is established on a specified port by calling the `socket.socket` method with the protocol code `AF_INET` and the socket type `SOCK_STREAM`.

The `Minitouch` class has a `__enter__` method that returns an instance of the class, a `__exit__` method that does nothing, and a `__del__` method that also does nothing.

The `send` method sends data to the remote device by calling the `sendall` method on the socket object, and returns the bytes returned by the `sock.recv` method.

Note that this implementation is just a simple example and does not handle errors, security, or other important considerations that are typically important in a production-level implementation.


```
from __future__ import annotations

import socket

from ...log import logger

DEFAULT_HOST = '127.0.0.1'


class Session(object):
    """ manage socket connections between PC and Android """

    def __init__(self, port: int, buf_size: int = 0) -> None:
        self.port = port
        self.buf_size = buf_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((DEFAULT_HOST, port))
        socket_out = self.sock.makefile()

        # v <version>
        # protocol version, usually it is 1. needn't use this
        socket_out.readline()

        # ^ <max-contacts> <max-x> <max-y> <max-pressure>
        _, max_contacts, max_x, max_y, max_pressure, *_ = (
            socket_out.readline().strip().split(' '))
        self.max_contacts = max_contacts
        self.max_x = max_x
        self.max_y = max_y
        self.max_pressure = max_pressure

        # $ <pid>
        _, pid = socket_out.readline().strip().split(' ')
        self.pid = pid

        logger.debug(
            f'minitouch running on port: {self.port}, pid: {self.pid}')
        logger.debug(
            f'max_contact: {max_contacts}; max_x: {max_x}; max_y: {max_y}; max_pressure: {max_pressure}')

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        pass

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """ cancel connection """
        self.sock and self.sock.close()
        self.sock = None

    def send(self, content: str) -> bytes:
        content = content.encode('utf8')
        self.sock.sendall(content)
        return self.sock.recv(self.buf_size)

```

# `/opt/arknights-mower/arknights_mower/utils/device/minitouch/__init__.py`

这段代码定义了一个名为 "Client" 的类，继承自 "MiniTouch" 类。这个类的具体作用在代码中没有明确说明，但可以推测它是一个用于与某个系统进行交互的客户端，用于在 MiniTouch 类中实现一些核心功能。


```
from .core import Client as MiniTouch

```

# `/opt/arknights-mower/arknights_mower/utils/device/scrcpy/const.py`

这段代码定义了一个模块中的所有常量，包括项目中共用到的常量。

该模块定义了三种动作类型：ACTION_DOWN、ACTION_UP和ACTION_MOVE。

该模块还定义了三种键盘编码：KEYCODE_UNKNOWN、KEYCODE_SOFT_LEFT和KEYCODE_SOFT_RIGHT。

KEYCODE_UNKNOWN表示未知的键盘编码。

KEYCODE_SOFT_LEFT表示软退行键。

KEYCODE_SOFT_RIGHT表示软回车键。

该模块还定义了四种键方向：KEYCODE_HOME、KEYCODE_BACK和KEYCODE_UP。

KEYCODE_HOME表示回车键。

KEYCODE_BACK表示退行键。

该模块中的常量可以被用来在程序中读取和设置键盘输入。


```
"""
This module includes all consts used in this project
"""

# Action
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2

# KeyCode
KEYCODE_UNKNOWN = 0
KEYCODE_SOFT_LEFT = 1
KEYCODE_SOFT_RIGHT = 2
KEYCODE_HOME = 3
KEYCODE_BACK = 4
```

这是一段使用Keyboard布局（Keymap）的代码，它定义了一系列预定义的键盘事件（KeyCODE）及其对应的ASCII码。这些预定义的键盘事件可以用于在不同应用程序之间传递用户输入信息。

KEYCODE_CALL：一个字符，用于表示一个特殊的ASCII码，用于调用一个带有参数的函数。这个字符通常在键盘上标有“Ctrl+”或者“Cmd+”字样。在大多数Linux和类Unix系统上，这个字符是`Cmd-C`。

KEYCODE_ENDCALL：与KEYCODE_CALL相对应的字符，用于表示一个特殊的ASCII码，用于调用一个带有参数的函数。这个字符通常在键盘上标有“Ctrl+”或者“Cmd+”字样。在大多数Linux和类Unix系统上，这个字符是`Cmd-A`。

KEYCODE_0：一个字符，代表0，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_1：一个字符，代表1，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_2：一个字符，代表2，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_3：一个字符，代表3，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_4：一个字符，代表4，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_5：一个字符，代表5，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_6：一个字符，代表6，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_7：一个字符，代表7，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_8：一个字符，代表8，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_9：一个字符，代表9，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_STAR：一个字符，代表*，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_POUND：一个字符，代表#，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。

KEYCODE_DPAD_UP：一个字符，代表^，用于表示一个特殊的ASCII码，用于在特定应用程序之间传递用户输入信息。


```
KEYCODE_CALL = 5
KEYCODE_ENDCALL = 6
KEYCODE_0 = 7
KEYCODE_1 = 8
KEYCODE_2 = 9
KEYCODE_3 = 10
KEYCODE_4 = 11
KEYCODE_5 = 12
KEYCODE_6 = 13
KEYCODE_7 = 14
KEYCODE_8 = 15
KEYCODE_9 = 16
KEYCODE_STAR = 17
KEYCODE_POUND = 18
KEYCODE_DPAD_UP = 19
```

这段代码定义了六个 constants，它们是用于 Android 设备上的快捷键。这些快捷键的编码值如下：

- KEYCODE_DPAD_DOWN：按下 physical 键（如主页键和音量上/下键）并拖动手指，以最大化应用程序屏幕。
- KEYCODE_DPAD_LEFT：按下 physical 键（如主页键和音量上/下键）并拖动手指，以最小化应用程序屏幕。
- KEYCODE_DPAD_RIGHT：按下 physical 键（如主页键和音量上/下键）并拖动手指，以旋转应用程序屏幕。
- KEYCODE_DPAD_CENTER：按下 physical 键（如主页键和音量上/下键）并拖动手指，以将应用程序屏幕居中对齐。

- KEYCODE_VOLUME_UP：按下 physical 键（如主页键和音量上/下键）并向上拖动手指，以调整音量。
- KEYCODE_VOLUME_DOWN：按下 physical 键（如主页键和音量上/下键）并向下拖动手指，以降低音量。
- KEYCODE_POWER：按下 physical 键（如主页键和音量上/下键）并向上拖动手指，以开启或关闭设备的电源。
- KEYCODE_CAMERA：按下 physical 键（如主页键和音量上/下键）并向上拖动手指，以打开或关闭设备的相机。
- KEYCODE_CLEAR：按下 physical 键（如主页键和音量上/下键）并向上拖动手指，以清除屏幕。

这些快捷键代码用于在 Android 应用程序中实现一些常见的功能。


```
KEYCODE_DPAD_DOWN = 20
KEYCODE_DPAD_LEFT = 21
KEYCODE_DPAD_RIGHT = 22
KEYCODE_DPAD_CENTER = 23
KEYCODE_VOLUME_UP = 24
KEYCODE_VOLUME_DOWN = 25
KEYCODE_POWER = 26
KEYCODE_CAMERA = 27
KEYCODE_CLEAR = 28
KEYCODE_A = 29
KEYCODE_B = 30
KEYCODE_C = 31
KEYCODE_D = 32
KEYCODE_E = 33
KEYCODE_F = 34
```

以上代码定义了20个不同的 `KEYCODE_` 常量，每个常量都有一个唯一的数字，用于表示特定的快捷键。这些常量将作为 Python 程序中的键，用于启动特定的功能或操作。


```
KEYCODE_G = 35
KEYCODE_H = 36
KEYCODE_I = 37
KEYCODE_J = 38
KEYCODE_K = 39
KEYCODE_L = 40
KEYCODE_M = 41
KEYCODE_N = 42
KEYCODE_O = 43
KEYCODE_P = 44
KEYCODE_Q = 45
KEYCODE_R = 46
KEYCODE_S = 47
KEYCODE_T = 48
KEYCODE_U = 49
```

以上代码定义了16个自定义键盘键（KEYCODE_V through KEYCODE_X），每个键的编号都是从50开始递增的。这些键被用于在应用程序中实现键盘输入。例如，你可以将KEYCODE_V用于在屏幕上输出"V"符号。


```
KEYCODE_V = 50
KEYCODE_W = 51
KEYCODE_X = 52
KEYCODE_Y = 53
KEYCODE_Z = 54
KEYCODE_COMMA = 55
KEYCODE_PERIOD = 56
KEYCODE_ALT_LEFT = 57
KEYCODE_ALT_RIGHT = 58
KEYCODE_SHIFT_LEFT = 59
KEYCODE_SHIFT_RIGHT = 60
KEYCODE_TAB = 61
KEYCODE_SPACE = 62
KEYCODE_SYM = 63
KEYCODE_EXPLORER = 64
```

这是一段定义了若干个 ASCII 码的代码，这些码被用于在文本编辑器中执行一些基本的文本操作。具体来说，这些码对应了以下操作：

- KEYCODE_ENVELOPE：用于打开或关闭一个应用程序的电子邮件客户端。
- KEYCODE_ENTER：用于在文本编辑器中执行命令键(通常是回车键)来进入编辑模式。
- KEYCODE_DEL：用于删除文本编辑器中的一个字符或字符串。
- KEYCODE_GRAVE：用于将文本编辑器中的一个字符或字符串加引号。
- KEYCODE_MINUS：用于删除文本编辑器中的一个字符或字符串，并将它的值取反。
- KEYCODE_EQUALS：用于在文本编辑器中执行等号操作。
- KEYCODE_LEFT_BRACKET：用于在文本编辑器中执行左括号操作。
- KEYCODE_RIGHT_BRACKET：用于在文本编辑器中执行右括号操作。
- KEYCODE_BACKSLASH：用于在文本编辑器中执行反斜杠操作。
- KEYCODE_SEMICOLON：用于在文本编辑器中执行分号操作。
- KEYCODE_APOSTROPHE：用于在文本编辑器中执行大写字母操作。
- KEYCODE_SLASH：用于在文本编辑器中执行斜杠操作。
- KEYCODE_AT：用于在文本编辑器中执行圆括号操作。
- KEYCODE_NUM：用于在文本编辑器中执行数字操作。
- KEYCODE_HEADSETHOOK：用于在文本编辑器中执行操作系统命令，通常是用于显示文件或目录的属性。


```
KEYCODE_ENVELOPE = 65
KEYCODE_ENTER = 66
KEYCODE_DEL = 67
KEYCODE_GRAVE = 68
KEYCODE_MINUS = 69
KEYCODE_EQUALS = 70
KEYCODE_LEFT_BRACKET = 71
KEYCODE_RIGHT_BRACKET = 72
KEYCODE_BACKSLASH = 73
KEYCODE_SEMICOLON = 74
KEYCODE_APOSTROPHE = 75
KEYCODE_SLASH = 76
KEYCODE_AT = 77
KEYCODE_NUM = 78
KEYCODE_HEADSETHOOK = 79
```

这段代码定义了八个不同的KEYCODE常量，用于识别用户与设备之间的交互操作。每个KEYCODE都有一个对应的数字，这些数字用于识别不同的用户交互操作。例如，KEYCODE_PLUS 对应于按下 Plus 按钮（通常是回退键）所触发的操作。


```
KEYCODE_PLUS = 81
KEYCODE_MENU = 82
KEYCODE_NOTIFICATION = 83
KEYCODE_SEARCH = 84
KEYCODE_MEDIA_PLAY_PAUSE = 85
KEYCODE_MEDIA_STOP = 86
KEYCODE_MEDIA_NEXT = 87
KEYCODE_MEDIA_PREVIOUS = 88
KEYCODE_MEDIA_REWIND = 89
KEYCODE_MEDIA_FAST_FORWARD = 90
KEYCODE_MUTE = 91
KEYCODE_PAGE_UP = 92
KEYCODE_PAGE_DOWN = 93
KEYCODE_BUTTON_A = 96
KEYCODE_BUTTON_B = 97
```

这是一段定义了在不同设备上预定义的常量（KeyCode）的代码。每个常量都有一个对应的数字，称为“KeyCode”，表示特定的按键功能。

以下是每个常量的作用：

- KEYCODE_BUTTON_C = 98：这是一个组合按键，代表“Ctrl”（通常是C）和“SPACE”按键。
- KEYCODE_BUTTON_X = 99：这是一个组合按键，代表“Ctrl”和“Pause/Break”按键（通常是P）。
- KEYCODE_BUTTON_Y = 100：这是一个组合按键，代表“Y”和“Enter”按键。
- KEYCODE_BUTTON_Z = 101：这是一个组合按键，代表“Z”和“Enter”按键。
- KEYCODE_BUTTON_L1 = 102：这是一个单独的按键，没有组合键。可能是“A”或“1”。
- KEYCODE_BUTTON_R1 = 103：这是一个单独的按键，没有组合键。可能是“D”或“1”。
- KEYCODE_BUTTON_L2 = 104：这是一个单独的按键，没有组合键。可能是“E”或“1”。
- KEYCODE_BUTTON_R2 = 105：这是一个单独的按键，没有组合键。可能是“F”或“1”。
- KEYCODE_BUTTON_THUMBL = 106：这是一个组合按键，代表“V”和“Grant”按键。
- KEYCODE_BUTTON_THUMBR = 107：这是一个组合按键，代表“V”和“Break”按键。
- KEYCODE_BUTTON_START = 108：这是一个组合按键，代表“Begin珠江 Ble医恩滚恩布”按键。
- KEYCODE_BUTTON_SELECT = 109：这是一个组合按键，代表“S”和“Delete”按键。
- KEYCODE_ESCAPE = 111：这是Escape键，通常用于中英文的Delete键。
- KEYCODE_FORWARD_DEL = 112：这是浏览器中前进删除键。


```
KEYCODE_BUTTON_C = 98
KEYCODE_BUTTON_X = 99
KEYCODE_BUTTON_Y = 100
KEYCODE_BUTTON_Z = 101
KEYCODE_BUTTON_L1 = 102
KEYCODE_BUTTON_R1 = 103
KEYCODE_BUTTON_L2 = 104
KEYCODE_BUTTON_R2 = 105
KEYCODE_BUTTON_THUMBL = 106
KEYCODE_BUTTON_THUMBR = 107
KEYCODE_BUTTON_START = 108
KEYCODE_BUTTON_SELECT = 109
KEYCODE_BUTTON_MODE = 110
KEYCODE_ESCAPE = 111
KEYCODE_FORWARD_DEL = 112
```

这是一段定义了多个键盘键的常量，包括：KEYCODE_CTRL_LEFT（左CTRL键）、KEYCODE_CTRL_RIGHT（右CTRL键）、KEYCODE_CAPS_LOCK（CAPS锁定键）、KEYCODE_SCROLL_LOCK（SCROLL锁定键）、KEYCODE_META_LEFT（左META键）、KEYCODE_META_RIGHT（右META键）、KEYCODE_FUNCTION（FUNCTION键）、KEYCODE_SYSRQ（SYSRQ键）、KEYCODE_BREAK（BREAK键）、KEYCODE_MOVE_HOME（HOME键）、KEYCODE_MOVE_END（END键）、KEYCODE_INSERT（INSERT键）、KEYCODE_FORWARD（FORWARD键）、KEYCODE_MEDIA_PLAY（PLAY键）、KEYCODE_MEDIA_PAUSE（PAUSE键）。这些常量用于识别和处理键盘输入。


```
KEYCODE_CTRL_LEFT = 113
KEYCODE_CTRL_RIGHT = 114
KEYCODE_CAPS_LOCK = 115
KEYCODE_SCROLL_LOCK = 116
KEYCODE_META_LEFT = 117
KEYCODE_META_RIGHT = 118
KEYCODE_FUNCTION = 119
KEYCODE_SYSRQ = 120
KEYCODE_BREAK = 121
KEYCODE_MOVE_HOME = 122
KEYCODE_MOVE_END = 123
KEYCODE_INSERT = 124
KEYCODE_FORWARD = 125
KEYCODE_MEDIA_PLAY = 126
KEYCODE_MEDIA_PAUSE = 127
```

以上代码定义了12个自定义按键，包括媒体键(KEYCODE_MEDIA_CLOSE、KEYCODE_MEDIA_EJECT、KEYCODE_MEDIA_RECORD等)以及F1至F12键(KEYCODE_F1至KEYCODE_F12)。这些自定义按键可以在应用程序中用来执行各种操作。


```
KEYCODE_MEDIA_CLOSE = 128
KEYCODE_MEDIA_EJECT = 129
KEYCODE_MEDIA_RECORD = 130
KEYCODE_F1 = 131
KEYCODE_F2 = 132
KEYCODE_F3 = 133
KEYCODE_F4 = 134
KEYCODE_F5 = 135
KEYCODE_F6 = 136
KEYCODE_F7 = 137
KEYCODE_F8 = 138
KEYCODE_F9 = 139
KEYCODE_F10 = 140
KEYCODE_F11 = 141
KEYCODE_F12 = 142
```

这段代码定义了几个常量，它们似乎用于某种编程语言中的锁相关操作。KEYCODE_NUM_LOCK，KEYCODE_NUMPAD_0，KEYCODE_NUMPAD_1，KEYCODE_NUMPAD_2，KEYCODE_NUMPAD_3，KEYCODE_NUMPAD_4，KEYCODE_NUMPAD_5，KEYCODE_NUMPAD_6，KEYCODE_NUMPAD_7，KEYCODE_NUMPAD_8，KEYCODE_NUMPAD_9，KEYCODE_NUMPAD_DIVIDE，KEYCODE_NUMPAD_MULTIPLY和KEYCODE_NUMPAD_SUBTRACT似乎都是与数字锁或密码相关操作有关的常量。


```
KEYCODE_NUM_LOCK = 143
KEYCODE_NUMPAD_0 = 144
KEYCODE_NUMPAD_1 = 145
KEYCODE_NUMPAD_2 = 146
KEYCODE_NUMPAD_3 = 147
KEYCODE_NUMPAD_4 = 148
KEYCODE_NUMPAD_5 = 149
KEYCODE_NUMPAD_6 = 150
KEYCODE_NUMPAD_7 = 151
KEYCODE_NUMPAD_8 = 152
KEYCODE_NUMPAD_9 = 153
KEYCODE_NUMPAD_DIVIDE = 154
KEYCODE_NUMPAD_MULTIPLY = 155
KEYCODE_NUMPAD_SUBTRACT = 156
KEYCODE_NUMPAD_ADD = 157
```

这是一段Python代码，它定义了numpad键的预定义功能。numpy键是一个用于NumPy数组操作的键盘快捷键，具有各种各样的功能。

具体来说，这段代码定义了以下11个NumPy键：

- KEYCODE_NUMPAD_DOT：用于NumPy中的.（点）键。
- KEYCODE_NUMPAD_COMMA：用于NumPy中的逗号（,）键。
- KEYCODE_NUMPAD_ENTER：用于NumPy中的Enter键。
- KEYCODE_NUMPAD_EQUALS：用于NumPy中的等号（=）键。
- KEYCODE_NUMPAD_LEFT_PAREN：用于NumPy中的左括号（<）键。
- KEYCODE_NUMPAD_RIGHT_PAREN：用于NumPy中的右括号（>）键。
- KEYCODE_VOLUME_MUTE：用于NumPy中的静音（MUTE）键。
- KEYCODE_INFO：用于NumPy中的信息（INFO）键。
- KEYCODE_CHANNEL_UP：用于NumPy中的向上箭头（↑）键。
- KEYCODE_CHANNEL_DOWN：用于NumPy中的向下箭头（↓）键。
- KEYCODE_ZOOM_IN：用于NumPy中的放大（IN）键。
- KEYCODE_ZOOM_OUT：用于NumPy中的缩小（OUT）键。
- KEYCODE_TV：用于NumPy中的 Television（TV）键。
- KEYCODE_WINDOW：用于NumPy中的窗口（WINDOW）键。
- KEYCODE_GUIDE：用于NumPy中的指南（GUIDE）键。


```
KEYCODE_NUMPAD_DOT = 158
KEYCODE_NUMPAD_COMMA = 159
KEYCODE_NUMPAD_ENTER = 160
KEYCODE_NUMPAD_EQUALS = 161
KEYCODE_NUMPAD_LEFT_PAREN = 162
KEYCODE_NUMPAD_RIGHT_PAREN = 163
KEYCODE_VOLUME_MUTE = 164
KEYCODE_INFO = 165
KEYCODE_CHANNEL_UP = 166
KEYCODE_CHANNEL_DOWN = 167
KEYCODE_ZOOM_IN = 168
KEYCODE_ZOOM_OUT = 169
KEYCODE_TV = 170
KEYCODE_WINDOW = 171
KEYCODE_GUIDE = 172
```

这是一段定义了多个常量的Python代码。这些常量用数字表示，包括KEYCODE_DVR、KEYCODE_BOOKMARK、KEYCODE_CAPTIONS、KEYCODE_SETTINGS、KEYCODE_TV_POWER、KEYCODE_TV_INPUT、KEYCODE_STB_POWER、KEYCODE_STB_INPUT、KEYCODE_AVR_POWER和KEYCODE_AVR_INPUT。它们用于表示不同的电视设置，包括节目表、频道、开关机、亮度等。


```
KEYCODE_DVR = 173
KEYCODE_BOOKMARK = 174
KEYCODE_CAPTIONS = 175
KEYCODE_SETTINGS = 176
KEYCODE_TV_POWER = 177
KEYCODE_TV_INPUT = 178
KEYCODE_STB_POWER = 179
KEYCODE_STB_INPUT = 180
KEYCODE_AVR_POWER = 181
KEYCODE_AVR_INPUT = 182
KEYCODE_PROG_RED = 183
KEYCODE_PROG_GREEN = 184
KEYCODE_PROG_YELLOW = 185
KEYCODE_PROG_BLUE = 186
KEYCODE_APP_SWITCH = 187
```

以上代码是一个嵌套循环，用于给16个按钮绑定不同的键盘码。具体来说，外层循环遍历16个键盘码，内层循环从键盘码中获取对应的按钮编号，然后将按钮编号存储到一个名为KEYCODE_BUTTON_的变量中。


```
KEYCODE_BUTTON_1 = 188
KEYCODE_BUTTON_2 = 189
KEYCODE_BUTTON_3 = 190
KEYCODE_BUTTON_4 = 191
KEYCODE_BUTTON_5 = 192
KEYCODE_BUTTON_6 = 193
KEYCODE_BUTTON_7 = 194
KEYCODE_BUTTON_8 = 195
KEYCODE_BUTTON_9 = 196
KEYCODE_BUTTON_10 = 197
KEYCODE_BUTTON_11 = 198
KEYCODE_BUTTON_12 = 199
KEYCODE_BUTTON_13 = 200
KEYCODE_BUTTON_14 = 201
KEYCODE_BUTTON_15 = 202
```

以上代码定义了216个自定义键的代码，用于应用程序中的UI元素，如按钮、开关、模式等。这些代码按照字母顺序排列，以便于程序中代码的编写和维护。


```
KEYCODE_BUTTON_16 = 203
KEYCODE_LANGUAGE_SWITCH = 204
KEYCODE_MANNER_MODE = 205
KEYCODE_3D_MODE = 206
KEYCODE_CONTACTS = 207
KEYCODE_CALENDAR = 208
KEYCODE_MUSIC = 209
KEYCODE_CALCULATOR = 210
KEYCODE_ZENKAKU_HANKAKU = 211
KEYCODE_EISU = 212
KEYCODE_MUHENKAN = 213
KEYCODE_HENKAN = 214
KEYCODE_KATAKANA_HIRAGANA = 215
KEYCODE_YEN = 216
KEYCODE_RO = 217
```

这是一段定义了232个按键代码的Python代码。这些按键代码用于控制多媒体应用程序中的功能。例如，可以使用这些按键来打开应用程序中的菜单、播放音量、静音电视等等。每个按键代码都由一个数字和一个大写或小写字母组成。这些数字表示了日本字典表中的一个键，例如，数字218对应的是“コード(建設)”键。


```
KEYCODE_KANA = 218
KEYCODE_ASSIST = 219
KEYCODE_BRIGHTNESS_DOWN = 220
KEYCODE_BRIGHTNESS_UP = 221
KEYCODE_MEDIA_AUDIO_TRACK = 222
KEYCODE_SLEEP = 223
KEYCODE_WAKEUP = 224
KEYCODE_PAIRING = 225
KEYCODE_MEDIA_TOP_MENU = 226
KEYCODE_11 = 227
KEYCODE_12 = 228
KEYCODE_LAST_CHANNEL = 229
KEYCODE_TV_DATA_SERVICE = 230
KEYCODE_VOICE_ASSIST = 231
KEYCODE_TV_RADIO_SERVICE = 232
```

这是一段Android中值为数组KEYCODE_TV_TELETEXT, KEYCODE_TV_NUMBER_ENTRY, KEYCODE_TV_TERRESTRIAL_ANALOG, KEYCODE_TV_TERRESTRIAL_DIGITAL, KEYCODE_TV_SATELLITE, KEYCODE_TV_SATELLITE_BS, KEYCODE_TV_SATELLITE_CS, KEYCODE_TV_SATELLITE_SERVICE, KEYCODE_TV_NETWORK, KEYCODE_TV_ANTENNA_CABLE, KEYCODE_TV_INPUT_HDMI_1, KEYCODE_TV_INPUT_HDMI_2, KEYCODE_TV_INPUT_HDMI_3, KEYCODE_TV_INPUT_HDMI_4, KEYCODE_TV_INPUT_COMPOSITE_1的编号定义。

它定义了一个名为KEYCODE_TV_TELETEXT的常量，一个名为KEYCODE_TV_NUMBER_ENTRY的常量，四个名为KEYCODE_TV_TERRESTRIAL_ANALOG, KEYCODE_TV_TERRESTRIAL_DIGITAL, KEYCODE_TV_SATELLITE和KEYCODE_TV_SATELLITE_BS的常量。还包括KEYCODE_TV_SATELLITE_CS和KEYCODE_TV_SATELLITE_SERVICE常量。同时定义了KEYCODE_TV_NETWORK和KEYCODE_TV_ANTENNA_CABLE常量。最后还包括了四个名为KEYCODE_TV_INPUT_HDMI_1, KEYCODE_TV_INPUT_HDMI_2, KEYCODE_TV_INPUT_HDMI_3, and KEYCODE_TV_INPUT_HDMI_4的常量和一个名为KEYCODE_TV_INPUT_COMPOSITE_1的常量。


```
KEYCODE_TV_TELETEXT = 233
KEYCODE_TV_NUMBER_ENTRY = 234
KEYCODE_TV_TERRESTRIAL_ANALOG = 235
KEYCODE_TV_TERRESTRIAL_DIGITAL = 236
KEYCODE_TV_SATELLITE = 237
KEYCODE_TV_SATELLITE_BS = 238
KEYCODE_TV_SATELLITE_CS = 239
KEYCODE_TV_SATELLITE_SERVICE = 240
KEYCODE_TV_NETWORK = 241
KEYCODE_TV_ANTENNA_CABLE = 242
KEYCODE_TV_INPUT_HDMI_1 = 243
KEYCODE_TV_INPUT_HDMI_2 = 244
KEYCODE_TV_INPUT_HDMI_3 = 245
KEYCODE_TV_INPUT_HDMI_4 = 246
KEYCODE_TV_INPUT_COMPOSITE_1 = 247
```

这是一段Android中的常量，用于定义了电视机输入组件的KeyCode值。其中包括：

KEYCODE_TV_INPUT_COMPOSITE_2 = 248
KEYCODE_TV_INPUT_COMPONENT_1 = 249
KEYCODE_TV_INPUT_COMPONENT_2 = 250
KEYCODE_TV_INPUT_VGA_1 = 251
KEYCODE_TV_AUDIO_DESCRIPTION = 252
KEYCODE_TV_AUDIO_DESCRIPTION_MIX_UP = 253
KEYCODE_TV_AUDIO_DESCRIPTION_MIX_DOWN = 254
KEYCODE_TV_ZOOM_MODE = 255
KEYCODE_TV_CONTENTS_MENU = 256
KEYCODE_TV_MEDIA_CONTEXT_MENU = 257
KEYCODE_TV_TIMER_PROGRAMMING = 258
KEYCODE_HELP = 259
KEYCODE_NAVIGATE_PREVIOUS = 260
KEYCODE_NAVIGATE_NEXT = 261
KEYCODE_NAVIGATE_IN = 262

这些KeyCode值用于在应用程序中识别电视机输入组件，如音量调节、频道设置、输入源选择等。


```
KEYCODE_TV_INPUT_COMPOSITE_2 = 248
KEYCODE_TV_INPUT_COMPONENT_1 = 249
KEYCODE_TV_INPUT_COMPONENT_2 = 250
KEYCODE_TV_INPUT_VGA_1 = 251
KEYCODE_TV_AUDIO_DESCRIPTION = 252
KEYCODE_TV_AUDIO_DESCRIPTION_MIX_UP = 253
KEYCODE_TV_AUDIO_DESCRIPTION_MIX_DOWN = 254
KEYCODE_TV_ZOOM_MODE = 255
KEYCODE_TV_CONTENTS_MENU = 256
KEYCODE_TV_MEDIA_CONTEXT_MENU = 257
KEYCODE_TV_TIMER_PROGRAMMING = 258
KEYCODE_HELP = 259
KEYCODE_NAVIGATE_PREVIOUS = 260
KEYCODE_NAVIGATE_NEXT = 261
KEYCODE_NAVIGATE_IN = 262
```

这是一段Android中的键盘布局(KeyboardLayout)属性代码，其中定义了不同的主题键(KEYCODE)以及对应的功能。

例如，KEYCODE_NAVIGATE_OUT = 263表示“侧边栏”主题键，用于在应用程序的侧边栏中切换各种工具栏和界面。

KEYCODE_STEM_PRIMARY = 264表示“通知”主题键，用于在应用程序中打开通知栏。

KEYCODE_STEM_1 = 265表示“天气”主题键，用于在应用程序中打开天气信息。

KEYCODE_STEM_2 = 266表示“日期”主题键，用于在应用程序中打开日历应用程序。

KEYCODE_STEM_3 = 267表示“ Sandbox ”主题键，用于在应用程序中的沙盒中运行指定的应用程序。

KEYCODE_DPAD_UP_LEFT = 268表示“大标题栏”主题键，用于在应用程序中的大标题栏中切换各种工具栏和界面。

KEYCODE_DPAD_DOWN_LEFT = 269表示“状态栏”主题键，用于在应用程序中的状态栏中切换各种状态，如电池、Wi-Fi和蓝牙等。

KEYCODE_DPAD_UP_RIGHT = 270表示“操作中心”主题键，用于在应用程序中的操作中心中切换各种工具栏和界面。

KEYCODE_DPAD_DOWN_RIGHT = 271表示“通知栏”主题键，用于在应用程序中的通知栏中切换各种通知。

KEYCODE_MEDIA_SKIP_FORWARD = 272表示“异步播放”主题键，用于在应用程序中异步播放媒体内容。

KEYCODE_MEDIA_SKIP_BACKWARD = 273表示“设置”主题键，用于在应用程序中的设置菜单中切换各种设置。

KEYCODE_MEDIA_STEP_FORWARD = 274表示“播放/暂停”主题键，用于在媒体播放器中切换播放或暂停。

KEYCODE_MEDIA_STEP_BACKWARD = 275表示“上一首/下一首”主题键，用于在媒体播放器中切换上一首或下一首歌曲。

KEYCODE_SOFT_SLEEP = 276表示“省电锁屏”主题键，用于在应用程序中的省电锁屏中切换锁屏模式。

KEYCODE_CUT = 277表示“剪切”主题键，用于在应用程序中的剪切板中保存和复制文本、图像和其他内容。


```
KEYCODE_NAVIGATE_OUT = 263
KEYCODE_STEM_PRIMARY = 264
KEYCODE_STEM_1 = 265
KEYCODE_STEM_2 = 266
KEYCODE_STEM_3 = 267
KEYCODE_DPAD_UP_LEFT = 268
KEYCODE_DPAD_DOWN_LEFT = 269
KEYCODE_DPAD_UP_RIGHT = 270
KEYCODE_DPAD_DOWN_RIGHT = 271
KEYCODE_MEDIA_SKIP_FORWARD = 272
KEYCODE_MEDIA_SKIP_BACKWARD = 273
KEYCODE_MEDIA_STEP_FORWARD = 274
KEYCODE_MEDIA_STEP_BACKWARD = 275
KEYCODE_SOFT_SLEEP = 276
KEYCODE_CUT = 277
```

这段代码定义了一系列事件常量，包括KEYCODE_COPY、KEYCODE_PASTE、KEYCODE_SYSTEM_NAVIGATION_UP、KEYCODE_SYSTEM_NAVIGATION_DOWN、KEYCODE_SYSTEM_NAVIGATION_LEFT、KEYCODE_SYSTEM_NAVIGATION_RIGHT和KEYCODE_KEYCODE_ALL_APPS。它们用于定义应用程序在不同主题设置中，如何响应用户执行的操作。

具体来说，KEYCODE_COPY事件处理程序会将一个或多个应用程序的启动或恢复的关键码复制粘贴到剪贴板。KEYCODE_PASTE事件处理程序将在应用程序启动或恢复时将剪贴板的内容 pastebin 成当前应用程序。KEYCODE_SYSTEM_NAVIGATION_UP、KEYCODE_SYSTEM_NAVIGATION_DOWN、KEYCODE_SYSTEM_NAVIGATION_LEFT 和 KEYCODE_SYSTEM_NAVIGATION_RIGHT 事件处理程序分别用于当用户在设备的方向键上向上、向下、向左或向右移动时，如何改变应用程序的导航菜单。KEYCODE_KEYCODE_ALL_APPS 事件处理程序用于在应用程序启动时自动运行所有应用程序。KEYCODE_KEYCODE_REFRESH 事件处理程序用于在应用程序发生崩溃后重新启动应用程序。KEYCODE_KEYCODE_THUMBS_UP 和 KEYCODE_KEYCODE_THUMBS_DOWN 事件处理程序用于在设备的箭头键上向上和向下移动时，如何更改应用程序的设置。


```
KEYCODE_COPY = 278
KEYCODE_PASTE = 279
KEYCODE_SYSTEM_NAVIGATION_UP = 280
KEYCODE_SYSTEM_NAVIGATION_DOWN = 281
KEYCODE_SYSTEM_NAVIGATION_LEFT = 282
KEYCODE_SYSTEM_NAVIGATION_RIGHT = 283
KEYCODE_KEYCODE_ALL_APPS = 284
KEYCODE_KEYCODE_REFRESH = 285
KEYCODE_KEYCODE_THUMBS_UP = 286
KEYCODE_KEYCODE_THUMBS_DOWN = 287

# Event
EVENT_INIT = "init"
EVENT_FRAME = "frame"

```

这段代码定义了 Android 的原生组件中可以使用的键、文本和触摸事件的类型标识。其中，KEYCODE 枚举类型定义了 Android 键的 ideation 键，比如按下设置 home 键时的空前键(padding key)、 back 键和通过侧边栏开关机键等。TEXT 枚举类型定义了 Android 文本框的输入和输出事件。TOUCH_EVENT 和 SCROLL_EVENT 枚举类型定义了 Android 触摸事件的类型，包括按下、拖放和滑动等。BACK_OR_SCREEN_ON 和 EXPAND_PANELS 枚举类型定义了通知栏和扩展屏幕模式。CLIPBOARD 和 SET_CLIPBOARD 枚举类型定义了如何从主屏幕的剪贴板中复制或粘贴内容。ROTATE_DEVICE 枚举类型定义了如何旋转设备。最后，COPY_KEY_NONE 枚举类型定义了是否可以从通知栏中复制键。


```
# Type
TYPE_INJECT_KEYCODE = 0
TYPE_INJECT_TEXT = 1
TYPE_INJECT_TOUCH_EVENT = 2
TYPE_INJECT_SCROLL_EVENT = 3
TYPE_BACK_OR_SCREEN_ON = 4
TYPE_EXPAND_NOTIFICATION_PANEL = 5
TYPE_EXPAND_SETTINGS_PANEL = 6
TYPE_COLLAPSE_PANELS = 7
TYPE_GET_CLIPBOARD = 8
TYPE_SET_CLIPBOARD = 9
TYPE_SET_SCREEN_POWER_MODE = 10
TYPE_ROTATE_DEVICE = 11

COPY_KEY_NONE = 0
```

这段代码定义了几个常量：

1. COPY_KEY_COPY：一个整数，表示从主屏幕复制到剪贴板的内容，如果没有设置该常量，它的值将未知。
2. COPY_KEY_CUT：一个整数，表示从剪贴板复制到主屏幕的内容，如果没有设置该常量，它的值将未知。
3. LOCK_SCREEN_ORIENTATION_UNLOCKED：一个整数，表示锁屏方向为未解锁时的主屏幕方向。它的值可以从-1到2之间的任何数字，具体取决于设备上的实际设置。
4. LOCK_SCREEN_ORIENTATION_INITIAL：一个整数，表示初始锁屏方向。它的值可以从-2到3之间的任何数字，具体取决于设备上的实际设置。
5. LOCK_SCREEN_ORIENTATION_0：一个整数，表示锁定屏幕竖屏方向为0。
6. LOCK_SCREEN_ORIENTATION_1：一个整数，表示锁定屏幕横屏方向为1。
7. LOCK_SCREEN_ORIENTATION_2：一个整数，表示锁定屏幕逆时针方向为2。
8. LOCK_SCREEN_ORIENTATION_3：一个整数，表示锁定屏幕顺时针方向为3。
9. POWER_MODE_OFF：一个整数，表示设备的电源管理模式为关闭。它的值可以从0到2之间的任何数字，具体取决于设备上的实际设置。
10. POWER_MODE_NORMAL：一个整数，表示设备的电源管理模式为正常。它的值从2开始，具体取决于设备上的实际设置。


```
COPY_KEY_COPY = 1
COPY_KEY_CUT = 2

# Lock screen orientation
LOCK_SCREEN_ORIENTATION_UNLOCKED = -1
LOCK_SCREEN_ORIENTATION_INITIAL = -2
LOCK_SCREEN_ORIENTATION_0 = 0
LOCK_SCREEN_ORIENTATION_1 = 1
LOCK_SCREEN_ORIENTATION_2 = 2
LOCK_SCREEN_ORIENTATION_3 = 3

# Screen power mode
POWER_MODE_OFF = 0
POWER_MODE_NORMAL = 2

```

# `/opt/arknights-mower/arknights_mower/utils/device/scrcpy/control.py`

这段代码定义了一个名为 `inject` 的函数，用于将控制代码(也称为事件)注入到远程主机的控制进程中。

函数接受一个整数参数 `control_type`，表示要发送的control类型，该值应在 `const` 模块中定义。函数内部使用 `functools` 模块的一个名为 `wraps` 的函数，它可以将一个函数包装成一个可以创建新函数的包装器。因此，`inject` 函数实际上将 `control_type` 变量与一个闭包(closure)结合，该闭包包含一个函数 `inner`，该函数接受 `control_type` 参数，并按照传递给它的参数类型将其包装起来。

`inner` 函数内部，使用 `struct.pack` 函数将 `control_type` 参数和其它参数打包成一个字节序列，并使用 `sys.core几点` 函数 `__len__` 获取该字节序列的长度。然后，使用 `with` 语句和一个文件对象(可能是一个控制进程的套接字)进行 `文件操作`，向远程主机的控制进程中发送字节序列。

最后，`wrapper` 函数将 `inner` 函数包装起来，这样就可以通过调用 `inject` 函数来注入控制代码了。


```
import functools
import socket
import struct
from time import sleep

from . import const


def inject(control_type: int):
    """
    Inject control code, with this inject, we will be able to do unit test
    Args:
        control_type: event to send, TYPE_*
    """

    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            package = struct.pack(">B", control_type) + f(*args, **kwargs)
            if args[0].parent.control_socket is not None:
                with args[0].parent.control_socket_lock:
                    args[0].parent.control_socket.send(package)
            return package

        return inner

    return wrapper


```



This is a simple Python class that simulates a touch interaction with a touch screen. The touch interaction includes some basic functionality such as a延迟， a hold time, and a maximum button press duration.

The `touch` method takes four arguments: the x, y, and z coordinates of the touch position, the action to perform (DOWN, LEFT, RIGHT, UP), and a delay in seconds after each action. This method performs a touch gesture on the touch screen, and then performs an action defined by the `ACTION_UP` constant.

The `hold_time` argument is the maximum time the touch screen can be held open, and if the user touches the screen for this duration, the action will be performed again.

The `move_steps_delay` argument is the time of a delay between each step of the touch gesture.

Overall, this class provides a basic touch interaction functionality, but can be expanded and customized further to meet the specific needs of a touch screen.


```
class ControlSender:
    def __init__(self, parent):
        self.parent = parent

    @inject(const.TYPE_INJECT_KEYCODE)
    def keycode(
        self, keycode: int, action: int = const.ACTION_DOWN, repeat: int = 0
    ) -> bytes:
        """
        Send keycode to device
        Args:
            keycode: const.KEYCODE_*
            action: ACTION_DOWN | ACTION_UP
            repeat: repeat count
        """
        return struct.pack(">Biii", action, keycode, repeat, 0)

    @inject(const.TYPE_INJECT_TEXT)
    def text(self, text: str) -> bytes:
        """
        Send text to device
        Args:
            text: text to send
        """

        buffer = text.encode("utf-8")
        return struct.pack(">i", len(buffer)) + buffer

    @inject(const.TYPE_INJECT_TOUCH_EVENT)
    def touch(
        self, x: int, y: int, action: int = const.ACTION_DOWN, touch_id: int = -1
    ) -> bytes:
        """
        Touch screen
        Args:
            x: horizontal position
            y: vertical position
            action: ACTION_DOWN | ACTION_UP | ACTION_MOVE
            touch_id: Default using virtual id -1, you can specify it to emulate multi finger touch
        """
        x, y = max(x, 0), max(y, 0)
        return struct.pack(
            ">BqiiHHHi",
            action,
            touch_id,
            int(x),
            int(y),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            0xFFFF,
            1,
        )

    @inject(const.TYPE_INJECT_SCROLL_EVENT)
    def scroll(self, x: int, y: int, h: int, v: int) -> bytes:
        """
        Scroll screen
        Args:
            x: horizontal position
            y: vertical position
            h: horizontal movement
            v: vertical movement
        """

        x, y = max(x, 0), max(y, 0)
        return struct.pack(
            ">iiHHii",
            int(x),
            int(y),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            int(h),
            int(v),
        )

    @inject(const.TYPE_BACK_OR_SCREEN_ON)
    def back_or_turn_screen_on(self, action: int = const.ACTION_DOWN) -> bytes:
        """
        If the screen is off, it is turned on only on ACTION_DOWN
        Args:
            action: ACTION_DOWN | ACTION_UP
        """
        return struct.pack(">B", action)

    @inject(const.TYPE_EXPAND_NOTIFICATION_PANEL)
    def expand_notification_panel(self) -> bytes:
        """
        Expand notification panel
        """
        return b""

    @inject(const.TYPE_EXPAND_SETTINGS_PANEL)
    def expand_settings_panel(self) -> bytes:
        """
        Expand settings panel
        """
        return b""

    @inject(const.TYPE_COLLAPSE_PANELS)
    def collapse_panels(self) -> bytes:
        """
        Collapse all panels
        """
        return b""

    def get_clipboard(self, copy_key=const.COPY_KEY_NONE) -> str:
        """
        Get clipboard
        """
        # Since this function need socket response, we can't auto inject it any more
        s: socket.socket = self.parent.control_socket

        with self.parent.control_socket_lock:
            # Flush socket
            s.setblocking(False)
            while True:
                try:
                    s.recv(1024)
                except BlockingIOError:
                    break
            s.setblocking(True)

            # Read package
            package = struct.pack(">BB", const.TYPE_GET_CLIPBOARD, copy_key)
            s.send(package)
            (code,) = struct.unpack(">B", s.recv(1))
            assert code == 0
            (length,) = struct.unpack(">i", s.recv(4))

            return s.recv(length).decode("utf-8")

    @inject(const.TYPE_SET_CLIPBOARD)
    def set_clipboard(self, text: str, paste: bool = False) -> bytes:
        """
        Set clipboard
        Args:
            text: the string you want to set
            paste: paste now
        """
        buffer = text.encode("utf-8")
        return struct.pack(">?i", paste, len(buffer)) + buffer

    @inject(const.TYPE_SET_SCREEN_POWER_MODE)
    def set_screen_power_mode(self, mode: int = const.POWER_MODE_NORMAL) -> bytes:
        """
        Set screen power mode
        Args:
            mode: POWER_MODE_OFF | POWER_MODE_NORMAL
        """
        return struct.pack(">b", mode)

    @inject(const.TYPE_ROTATE_DEVICE)
    def rotate_device(self) -> bytes:
        """
        Rotate device
        """
        return b""

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        move_step_length: int = 5,
        move_steps_delay: float = 0.005,
    ) -> None:
        """
        Swipe on screen
        Args:
            start_x: start horizontal position
            start_y: start vertical position
            end_x: start horizontal position
            end_y: end vertical position
            move_step_length: length per step
            move_steps_delay: sleep seconds after each step
        :return:
        """

        self.touch(start_x, start_y, const.ACTION_DOWN)
        next_x = start_x
        next_y = start_y

        if end_x > self.parent.resolution[0]:
            end_x = self.parent.resolution[0]

        if end_y > self.parent.resolution[1]:
            end_y = self.parent.resolution[1]

        decrease_x = True if start_x > end_x else False
        decrease_y = True if start_y > end_y else False
        while True:
            if decrease_x:
                next_x -= move_step_length
                if next_x < end_x:
                    next_x = end_x
            else:
                next_x += move_step_length
                if next_x > end_x:
                    next_x = end_x

            if decrease_y:
                next_y -= move_step_length
                if next_y < end_y:
                    next_y = end_y
            else:
                next_y += move_step_length
                if next_y > end_y:
                    next_y = end_y

            self.touch(next_x, next_y, const.ACTION_MOVE)

            if next_x == end_x and next_y == end_y:
                self.touch(next_x, next_y, const.ACTION_UP)
                break
            sleep(move_steps_delay)

    def tap(self, x, y, hold_time: float = 0.07) -> None:
        """
        Tap on screen
        Args:
            x: horizontal position
            y: vertical position
            hold_time: hold time
        """
        self.touch(x, y, const.ACTION_DOWN)
        sleep(hold_time)
        self.touch(x, y, const.ACTION_UP)

```

# `/opt/arknights-mower/arknights_mower/utils/device/scrcpy/core.py`

这段代码是一个用于创建一个名为"my_program"的程序的Python装饰函数。具体来说，它从未来的函数中导入了一个名为"annotations"的模块，然后使用该模块中的函数来装饰函数代码。

从代码中可以看出，该程序的主要作用是创建一个名为"my_program"的程序，并运行该程序。具体来说，它将当前程序的根目录（即"__rootdir__"）设置为当前目录，然后使用一个名为"my_program"的函数来替换原来程序的名称。接着，它将使用该函数创建一个ADB客户端对象，并在客户端对象上发送一些我们希望监听的网络流量。最后，程序将一直运行，直到被强制停止。


```
from __future__ import annotations

import functools
import socket
import struct
import threading
import time
import traceback
from typing import Optional, Tuple

import numpy as np

from .... import __rootdir__
from ...log import logger
from ..adb_client import ADBClient
```



This is a class that manages multiple Android devices. It has a function called `start()` which starts the ADB server and a function called `stop()` which stops the ADB server. It also has a function called `adb_alive()` which checks if the ADB server is still running.

The class also has a function called `tap(x:int, y:int)` which simulates a tap gesture, and a function called `swipe(x0:int, y0:int, x1:int, y1:int, move_duraion:float=1, hold_before_release:float=0, fall:bool=True, lift:bool=True)` which simulates a swipe gesture.

The `tap()` and `swipe()` functions are not安全的 to use in production and should not be called by other code in派的层。 The `adb_alive()` function should also not be used in production, as it can potentially cause the device to hang or freeze.


```
from ..adb_client.socket import Socket
from . import const
from .control import ControlSender

SCR_PATH = '/data/local/tmp/minitouch'


class Client:
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
        self.client = client
        self.last_frame: Optional[np.ndarray] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.device_name: Optional[str] = None
        self.control = ControlSender(self)

        # Params
        self.flip = flip
        self.max_width = max_width
        self.bitrate = bitrate
        self.max_fps = max_fps
        self.block_frame = block_frame
        self.stay_awake = stay_awake
        self.lock_screen_orientation = lock_screen_orientation
        self.connection_timeout = connection_timeout
        self.displayid = displayid

        # Need to destroy
        self.__server_stream: Optional[Socket] = None
        self.__video_socket: Optional[Socket] = None
        self.control_socket: Optional[Socket] = None
        self.control_socket_lock = threading.Lock()

        self.start()

    def __del__(self) -> None:
        self.stop()

    def __start_server(self) -> None:
        """
        Start server and get the connection
        """
        cmdline = f'CLASSPATH={SCR_PATH} app_process /data/local/tmp com.genymobile.scrcpy.Server 1.21 log_level=verbose control=true tunnel_forward=true'
        if self.displayid is not None:
            cmdline += f' display_id={self.displayid}'
        self.__server_stream: Socket = self.client.stream_shell(cmdline)
        # Wait for server to start
        response = self.__server_stream.recv(100)
        logger.debug(response)
        if b'[server]' not in response:
            raise ConnectionError(
                'Failed to start scrcpy-server: ' + response.decode('utf-8', 'ignore'))

    def __deploy_server(self) -> None:
        """
        Deploy server to android device
        """
        server_file_path = __rootdir__ / 'vendor' / \
            'scrcpy-server-novideo' / 'scrcpy-server-novideo.jar'
        server_buf = server_file_path.read_bytes()
        self.client.push(SCR_PATH, server_buf)
        self.__start_server()

    def __init_server_connection(self) -> None:
        """
        Connect to android server, there will be two sockets, video and control socket.
        This method will set: video_socket, control_socket, resolution variables
        """
        try:
            self.__video_socket = self.client.stream('localabstract:scrcpy')
        except socket.timeout:
            raise ConnectionError('Failed to connect scrcpy-server')

        dummy_byte = self.__video_socket.recv(1)
        if not len(dummy_byte) or dummy_byte != b'\x00':
            raise ConnectionError('Did not receive Dummy Byte!')

        try:
            self.control_socket = self.client.stream('localabstract:scrcpy')
        except socket.timeout:
            raise ConnectionError('Failed to connect scrcpy-server')

        self.device_name = self.__video_socket.recv(64).decode('utf-8')
        self.device_name = self.device_name.rstrip('\x00')
        if not len(self.device_name):
            raise ConnectionError('Did not receive Device Name!')

        res = self.__video_socket.recv(4)
        self.resolution = struct.unpack('>HH', res)
        # self.__video_socket.setblocking(False)

    def start(self) -> None:
        """
        Start listening video stream
        """
        try_count = 0
        while try_count < 3:
            try:
                self.__deploy_server()
                time.sleep(0.5)
                self.__init_server_connection()
                break
            except ConnectionError:
                logger.debug(traceback.format_exc())
                logger.warning('Failed to connect scrcpy-server.')
                self.stop()
                logger.warning('Try again in 10 seconds...')
                time.sleep(10)
                try_count += 1
        else:
            raise RuntimeError('Failed to connect scrcpy-server.')

    def stop(self) -> None:
        """
        Stop listening (both threaded and blocked)
        """
        if self.__server_stream is not None:
            self.__server_stream.close()
            self.__server_stream = None
        if self.control_socket is not None:
            self.control_socket.close()
            self.control_socket = None
        if self.__video_socket is not None:
            self.__video_socket.close()
            self.__video_socket = None

    def check_adb_alive(self) -> bool:
        """ check if adb server alive """
        return self.client.check_server_alive()

    def stable(f):
        @functools.wraps(f)
        def inner(self: Client, *args, **kwargs):
            try_count = 0
            while try_count < 3:
                try:
                    f(self, *args, **kwargs)
                    break
                except (ConnectionResetError, BrokenPipeError):
                    self.stop()
                    time.sleep(1)
                    self.check_adb_alive()
                    self.start()
                    try_count += 1
            else:
                raise RuntimeError('Failed to start scrcpy-server.')
        return inner

    @stable
    def tap(self, x: int, y: int) -> None:
        self.control.tap(x, y)

    @stable
    def swipe(self, x0, y0, x1, y1, move_duraion: float = 1, hold_before_release: float = 0, fall: bool = True, lift: bool = True):
        frame_time = 1 / 60

        start_time = time.perf_counter()
        end_time = start_time + move_duraion
        fall and self.control.touch(x0, y0, const.ACTION_DOWN)
        t1 = time.perf_counter()
        step_time = t1 - start_time
        if step_time < frame_time:
            time.sleep(frame_time - step_time)
        while True:
            t0 = time.perf_counter()
            if t0 > end_time:
                break
            time_progress = (t0 - start_time) / move_duraion
            path_progress = time_progress
            self.control.touch(int(x0 + (x1 - x0) * path_progress),
                               int(y0 + (y1 - y0) * path_progress), const.ACTION_MOVE)
            t1 = time.perf_counter()
            step_time = t1 - t0
            if step_time < frame_time:
                time.sleep(frame_time - step_time)
        self.control.touch(x1, y1, const.ACTION_MOVE)
        if hold_before_release > 0:
            time.sleep(hold_before_release)
        lift and self.control.touch(x1, y1, const.ACTION_UP)

```

# `/opt/arknights-mower/arknights_mower/utils/device/scrcpy/__init__.py`

这段代码是导入了一个名为"Client"的类，该类属于一个名为"core"的模块。这个类的定义应该在代码的其他部分给出，但目前我没有获取到更多的上下文，所以无法提供确切的解释。请注意，这段代码没有进行任何实际的操作，因此它不会产生任何输出。


```
from .core import Client as Scrcpy

```

## New
- 新功能 (#1) (Thanks @UserName)

## Bug Fixes
- 修复了点 Bug (#1) (Thanks @UserName)

## Documentation
- 完善 README.md 的说明及文档结构 (#1) (Thanks @UserName)

## Dependency Updates
- 依赖更新 (#1) (Thanks @UserName)


# `/opt/arknights-mower/packaging/image.py`

这段代码是一个Python文件，其中定义了一些图像预处理的基本接口。具体来说，这段代码声明了一个名为“image_preprocess”的函数组，该函数组可以作为其他函数或模块的子模块。

这段代码的作用是提供一个通用的接口，让开发人员可以使用同一个函数组来处理他们的图像数据。这个接口包含了一些通用的图像处理函数，如图像尺寸调整、图像增强、图像分割等。使用这个接口，开发人员可以更轻松地构建出符合他们特定需求的图像预处理过程。


```
#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains some common interfaces for image preprocess.
```

这段代码描述了一个图像布局的介绍，其中包括通道（CHW）、图像高度（H）、宽度（W）以及图像打开时默认的布局（HWC）。接着，它解释了CHW布局的具体含义，并表示PaddlePaddle只支持CHW布局。同时，指出OPENCV使用BGR颜色格式，而PIL使用RGB颜色格式。这两个格式都可以用于训练，但在培训和推理期间应该保持一致。


```
Many users are confused about the image layout. We introduce
the image layout as follows.

- CHW Layout

  - The abbreviations: C=channel, H=Height, W=Width
  - The default layout of image opened by cv2 or PIL is HWC.
    PaddlePaddle only supports the CHW layout. And CHW is simply
    a transpose of HWC. It must transpose the input image.

- Color format: RGB or BGR

  OpenCV use BGR color format. PIL use RGB color format. Both
  formats can be used for training. Noted that, the format should
  be keep consistent between the training and inference period.
```

这段代码是一个用于打印特定数字的Python函数。具体来说，这个函数将打印数字1到给定的整数（通过调用 six.print_function() 实现），并在数字1前添加了一个参数字符'F'，使得这个函数在Python 3中更易于阅读。


```
"""

from __future__ import print_function

import six
import numpy as np
# FIXME(minqiyang): this is an ugly fix for the numpy bug reported here
# https://github.com/numpy/numpy/issues/12497
# if six.PY3:
#     import subprocess
#     import sys
#     import os
#     interpreter = sys.executable
#     # Note(zhouwei): if use Python/C 'PyRun_SimpleString', 'sys.executable'
#     # will be the C++ execubable on Windows
```

这段代码的作用是检查在 Windows 系统下是否可以通过系统命令行找到 Python 解释器。如果找不到，则执行一些操作以安装 Python 和相应的库。

具体来说，代码首先检查当前操作系统是否为 Windows。如果是，代码会执行以下操作：

1. 在 Windows 系统下，通过系统命令行找到 Python 解释器。可以使用以下命令找到所有系统中默认安装的 Python 解释器：


python -m console where "python.exe"


2. 如果找到 Python 解释器，则执行以下操作以安装相应的 Python 库：


python -m pip install --upgrade pip


3. 如果尚未安装 pip，则执行以下操作以安装它：


python -m pip install --install-extension pip


4. 如果安装 pip 后仍然无法找到 Python 解释器，则执行以下操作以安装它：


python -m pip install --upgrade --no-cache-dir python



这些操作会在后台执行，并替换掉代码中的 `import_cv2_proc` 函数。函数使用 `subprocess.Popen` 创建一个进程来执行以下命令：


import cv2



cv2 = None



cv2 = cv2.CaptureC基准库


如果上述步骤成功安装了 Python 和相应的库，则退出 `cv2.CaptureC基准库`。否则，如果 retcode 不等于 0，则说明安装失败，将返回一个 None 值。


```
#     if sys.platform == 'win32' and 'python.exe' not in interpreter:
#         interpreter = sys.exec_prefix + os.sep + 'python.exe'
#     import_cv2_proc = subprocess.Popen([interpreter, "-c", "import cv2"],
#                                        stdout=subprocess.PIPE,
#                                        stderr=subprocess.PIPE)
#     out, err = import_cv2_proc.communicate()
#     retcode = import_cv2_proc.poll()
#     if retcode != 0:
#         cv2 = None
#     else:
#         try:
#             import cv2
#         except ImportError:
#             cv2 = None
# else:
```

这段代码是一个带有 try-except 语句的函数，用于在 Python 脚本中检查是否安装了 OpenCV（cv2）库。如果没有安装，则代码会尝试导入并使用 cv2 库。如果已安装，则该函数会返回一个布尔值。

具体来说，这段代码以下几个步骤：

1. 导入 cv2 库：如果 cv2 库已安装，则直接导入；否则，会尝试导入，如果失败，则会输出一个警告信息。
2. 导入 os 库：该库用于在 Linux、macOS 和 Windows 操作系统中进行文件操作。
3. 导入 tarfile 库：该库用于处理 tar 档案。
4. 导入 six.moves.cPickle 库：该库用于将 Python 中的小型对象序列化为字节流，以便在其他程序中使用。
5. 定义一个名为 __all__ 的列表，用于将以上所有库的导入延迟到运行时。
6. 在函数内部，使用 _check_cv2() 函数检查 cv2 库是否已安装。如果没有安装，则会输出一个警告信息并返回 False；否则，返回 True。
7. 在函数外部，通过调用 __all__ 列表中的所有库，来检测是否安装了 OpenCV 库。


```
try:
    import cv2
except ImportError:
    cv2 = None
import os
import tarfile
import six.moves.cPickle as pickle

__all__ = []


def _check_cv2():
    if cv2 is None:
        import sys
        sys.stderr.write(
            '''Warning with paddle image module: opencv-python should be imported,
         or paddle image module could NOT work; please install opencv-python first.'''
        )
        return False
    else:
        return True


```

This function appears to extract the label and data information from a batch file and store it in a dictionary for later processing. It takes as input the path of the batch file and a dataset name.

It creates a directory called `batch_%d` inside the root directory of the `data_file` if it doesn't exist already. Then it creates a text file called `%s_%s.txt` inside the `batch_%d` directory.

It reads the batch file and extracts the label and data from it. The data is a list of files that correspond to the batch. For each file, it reads the contents of the file and stores it in the `data` list.

When it has read all the files in the batch, it checks if there are any remaining files and extracts them if they are. Then it sets the `data` list to empty and resets the `file_id` to 0.

It then creates a dictionary with the label and data, and writes it to the `meta_file` in the root directory of the `data_file`.

Finally, it returns the `meta_file` so that it can be used later.


```
def batch_images_from_tar(data_file,
                          dataset_name,
                          img2label,
                          num_per_batch=1024):
    """
    Read images from tar file and batch them into batch file.

    :param data_file: path of image tar file
    :type data_file: string
    :param dataset_name: 'train','test' or 'valid'
    :type dataset_name: string
    :param img2label: a dic with image file name as key
                    and image's label as value
    :type img2label: dic
    :param num_per_batch: image number per batch file
    :type num_per_batch: int
    :return: path of list file containing paths of batch file
    :rtype: string
    """
    batch_dir = data_file + "_batch"
    out_path = "%s/%s_%s" % (batch_dir, dataset_name, os.getpid())
    meta_file = "%s/%s_%s.txt" % (batch_dir, dataset_name, os.getpid())

    if os.path.exists(out_path):
        return meta_file
    else:
        os.makedirs(out_path)

    tf = tarfile.open(data_file)
    mems = tf.getmembers()
    data = []
    labels = []
    file_id = 0
    for mem in mems:
        if mem.name in img2label:
            data.append(tf.extractfile(mem).read())
            labels.append(img2label[mem.name])
            if len(data) == num_per_batch:
                output = {}
                output['label'] = labels
                output['data'] = data
                pickle.dump(output,
                            open('%s/batch_%d' % (out_path, file_id), 'wb'),
                            protocol=2)
                file_id += 1
                data = []
                labels = []
    if len(data) > 0:
        output = {}
        output['label'] = labels
        output['data'] = data
        pickle.dump(output,
                    open('%s/batch_%d' % (out_path, file_id), 'wb'),
                    protocol=2)

    with open(meta_file, 'a') as meta:
        for file in os.listdir(out_path):
            meta.write(os.path.abspath("%s/%s" % (out_path, file)) + "\n")
    return meta_file


```

这段代码定义了一个名为 `load_image_bytes` 的函数，它接受一个字节数组（bytes）作为输入参数，并可以选择是否加载一个颜色图像。函数内部使用了一个名为 `_check_cv2` 的函数，但这个函数没有定义具体的逻辑，因此无法确定它是否会成功检查所需的库是否已经安装。

函数的作用是加载一个字节数组中的图像，并返回一个图像对象。它通过调用 `cv2.imdecode` 函数来加载图像，这个函数接受一个字节数组，并将其转换为灰度图像。如果参数 `is_color` 为真，函数将加载一个颜色图像；否则，它将加载一个灰度图像。


```
def load_image_bytes(bytes, is_color=True):
    """
    Load an color or gray image from bytes array.

    Example usage:

    .. code-block:: python

        with open('cat.jpg') as f:
            im = load_image_bytes(f.read())

    :param bytes: the input image bytes array.
    :type bytes: str
    :param is_color: If set is_color True, it will load and
                     return a color image. Otherwise, it will
                     load and return a gray image.
    :type is_color: bool
    """
    assert _check_cv2() is True

    flag = 1 if is_color else 0
    file_bytes = np.asarray(bytearray(bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, flag)
    return img


```

这段代码定义了一个名为 `load_image` 的函数，用于从指定文件路径加载颜色或灰度图像。函数有两个参数，第一个参数是一个文件路径，第二个参数 `is_color` 是一个布尔值，表示是否加载颜色图像。

函数内部首先调用 `_check_cv2()` 函数来检查输入是否支持 OpenCV 3，如果不是，则使用 OpenCV 2 的函数。然后，根据 `is_color` 参数的值，选择正确的函数来加载图像，1 表示加载颜色图像，0 表示加载灰度图像。

最后，函数使用 `cv2.imread()` 函数来加载图像，并返回图像对象。


```
def load_image(file, is_color=True):
    """
    Load an color or gray image from the file path.

    Example usage:

    .. code-block:: python

        im = load_image('cat.jpg')

    :param file: the input image path.
    :type file: string
    :param is_color: If set is_color True, it will load and
                     return a color image. Otherwise, it will
                     load and return a gray image.
    :type is_color: bool
    """
    assert _check_cv2() is True

    # cv2.IMAGE_COLOR for OpenCV3
    # cv2.CV_LOAD_IMAGE_COLOR for older OpenCV Version
    # cv2.IMAGE_GRAYSCALE for OpenCV3
    # cv2.CV_LOAD_IMAGE_GRAYSCALE for older OpenCV Version
    # Here, use constant 1 and 0
    # 1: COLOR, 0: GRAYSCALE
    flag = 1 if is_color else 0
    im = cv2.imread(file, flag)
    return im


```

这段代码定义了一个名为 `resize_short` 的函数，它接受一个输入图像 `im` 和一个尺寸 `size` 作为参数。函数的作用是调整图像的大小，使得输入图像中的短边尺寸 `h` 等于尺寸 `size` 的一倍，同时保持图像的宽高比不变。

具体实现过程如下：

1. 如果输入图像 `im` 的宽高比（即 `h` 与 `w` 的比值）大于尺寸 `size` 的一倍，那么将尺寸 `size` 设为 `w` 除以 `h`，并执行以下操作：
  
  h_new = size * h // w
  
  如果 `h` 小于 `size`，则交换 `h_new` 和 `w_new` 的位置，并执行以下操作：
  
  w_new = size * w // h
  
2. 使用 `cv2.resize` 函数对调整后的图像进行大小调整，并返回调整后的图像。注意，函数使用了参数 `(w_new, h_new)`，表示调整后的图像的大小为 `(w_new, h_new)`。


```
def resize_short(im, size):
    """
    Resize an image so that the length of shorter edge is size.

    Example usage:

    .. code-block:: python

        im = load_image('cat.jpg')
        im = resize_short(im, 256)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the shorter edge size of image after resizing.
    :type size: int
    """
    assert _check_cv2() is True

    h, w = im.shape[:2]
    h_new, w_new = size, size
    if h > w:
        h_new = size * h // w
    else:
        w_new = size * w // h
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_CUBIC)
    return im


```

这段代码定义了一个名为 `to_chw` 的函数，它接受一个输入图像 `im` 和一个指定的订单 `order`。函数的作用是将输入图像 `im` 根据指定的 `order` 进行通道（color）重排，将图像从 HWC 格式转换为 CHW 格式。

具体来说，函数首先检查输入图像 `im` 的形状是否与指定的 `order` 相同，如果是，则执行通道重排操作。重排后的结果直接返回。函数的一个简单示例如下：
python
from cv2 import load_image, resize
from PIL import Image

im = load_image('cat.jpg')
im = resize(im, (256, 256))
im = to_chw((im))

这段代码首先读取 `cat.jpg` 图片，然后将其尺寸调整为 256x256，最后将图像转换为 CHW 格式并保存。


```
def to_chw(im, order=(2, 0, 1)):
    """
    Transpose the input image order. The image layout is HWC format
    opened by cv2 or PIL. Transpose the input image to CHW layout
    according the order (2,0,1).

    Example usage:

    .. code-block:: python

        im = load_image('cat.jpg')
        im = resize_short(im, 256)
        im = to_chw(im)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param order: the transposed order.
    :type order: tuple|list
    """
    assert len(im.shape) == len(order)
    im = im.transpose(order)
    return im


```

这段代码定义了一个名为 `center_crop` 的函数，它接受一个输入图像 `im`，以及一个大小 `size` 和一个布尔值 `is_color` 作为参数。函数的作用是在保持图像的上下文不变的情况下，将输入图像的左下角和右上角的区域提取出来，如果输入图像是有颜色的，则只提取像素值不为 0 的区域。最终返回提取的图像区域。


```
def center_crop(im, size, is_color=True):
    """
    Crop the center of image with size.

    Example usage:

    .. code-block:: python

        im = center_crop(im, 224)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the cropping size.
    :type size: int
    :param is_color: whether the image is color or not.
    :type is_color: bool
    """
    h, w = im.shape[:2]
    h_start = (h - size) // 2
    w_start = (w - size) // 2
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    return im


```

这段代码定义了一个名为 `random_crop` 的函数，它接受一个输入图像 `im`，一个指定的大小 `size`，和一个布尔选项 `is_color`，表示是否截取彩色图像。函数的作用是随机地从输入图像中裁剪出一个指定大小的子图像，并将裁剪出来的图像转换为灰度图像，如果 `is_color` 为 `True`，则裁剪操作仅限于颜色空间，否则只裁剪非颜色部分。最终，函数返回裁剪后的图像。


```
def random_crop(im, size, is_color=True):
    """
    Randomly crop input image with size.

    Example usage:

    .. code-block:: python

        im = random_crop(im, 224)

    :param im: the input image with HWC layout.
    :type im: ndarray
    :param size: the cropping size.
    :type size: int
    :param is_color: whether the image is color or not.
    :type is_color: bool
    """
    h, w = im.shape[:2]
    h_start = np.random.randint(0, h - size + 1)
    w_start = np.random.randint(0, w - size + 1)
    h_end, w_end = h_start + size, w_start + size
    if is_color:
        im = im[h_start:h_end, w_start:w_end, :]
    else:
        im = im[h_start:h_end, w_start:w_end]
    return im


```

这段代码定义了一个名为 `left_right_flip` 的函数，它接受一个图像（`im`）和一个布尔参数 `is_color`，用于指定输入图像是否为颜色图像。如果 `is_color` 为 `True`，则函数将在图像的垂直方向上对图像进行翻转，否则仅在水平方向上进行翻转。函数返回翻转后的图像。

在函数内部，首先检查输入图像 `im` 是否有 `H` 形状（即有两维）。如果是，则创建一个 `numpy.roll` 函数，将图像的第一维从左到右平移，逆时针方向翻转，然后将结果设置为 `im`。如果不是，则创建一个将图像的第一维从左到右平移，逆时针方向翻转的图像，然后将其设置为 `im`。

需要注意的是，`left_right_flip` 函数在创建翻转后的图像时，仅在水平和垂直方向上进行翻转，而不会对图像的深度方向进行操作。


```
def left_right_flip(im, is_color=True):
    """
    Flip an image along the horizontal direction.
    Return the flipped image.

    Example usage:

    .. code-block:: python

        im = left_right_flip(im)

    :param im: input image with HWC layout or HW layout for gray image
    :type im: ndarray
    :param is_color: whether input image is color or not
    :type is_color: bool
    """
    if len(im.shape) == 3 and is_color:
        return im[:, ::-1, :]
    else:
        return im[:, ::-1]


```

This function is a data argumentation for training, and it applies simple transformations to the input image such as resizing, cropping, and flipping. The input image is expected to be in the form of a multi-channel (HWC) tensor, and it should have the same shape as the one provided in the example usage.

The `resize_short` function resizes the image to a smaller edge with the specified size. If `is_train` is `True`, it crops the image to a smaller bounding box and applies a random horizontal rotation by a small angle (default is 0.1).

The `random_crop` function crops the image to a smaller bounding box. If `is_train` is `True`, it also applies a random horizontal rotation by a small angle (default is 0.1).

The `left_right_flip` function flips the image horizontally. If `is_train` is `True`, it also crops the image to a smaller bounding box and applies a random horizontal rotation by a small angle (default is 0.1).

The `to_chw` function converts the image to the required format for a channel-wise average (HWC) tensor.

If `mean` is not `None`, it is converted to a NumPy array and divided by the batch size to obtain the mean values per channel. The mean values are then divided by the scale of the input image to obtain the scaling factors for each channel.


```
def simple_transform(im,
                     resize_size,
                     crop_size,
                     is_train,
                     is_color=True,
                     mean=None):
    """
    Simply data argumentation for training. These operations include
    resizing, croping and flipping.

    Example usage:

    .. code-block:: python

        im = simple_transform(im, 256, 224, True)

    :param im: The input image with HWC layout.
    :type im: ndarray
    :param resize_size: The shorter edge length of the resized image.
    :type resize_size: int
    :param crop_size: The cropping size.
    :type crop_size: int
    :param is_train: Whether it is training or not.
    :type is_train: bool
    :param is_color: whether the image is color or not.
    :type is_color: bool
    :param mean: the mean values, which can be element-wise mean values or
                 mean values per channel.
    :type mean: numpy array | list
    """
    im = resize_short(im, resize_size)
    if is_train:
        im = random_crop(im, crop_size, is_color=is_color)
        if np.random.randint(2) == 0:
            im = left_right_flip(im, is_color)
    else:
        im = center_crop(im, crop_size, is_color=is_color)
    if len(im.shape) == 3:
        im = to_chw(im)

    im = im.astype('float32')
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)
        # mean value, may be one value per channel
        if mean.ndim == 1 and is_color:
            mean = mean[:, np.newaxis, np.newaxis]
        elif mean.ndim == 1:
            mean = mean
        else:
            # elementwise mean
            assert len(mean.shape) == len(im)
        im -= mean

    return im


```

这段代码定义了一个名为 `load_and_transform` 的函数，它接受一个输入文件名、一个输出图像的大小、一个图像裁剪的大小、一个是否进行训练的参数、一个是否对颜色通道进行平均的参数，以及一个输入图像是否包含颜色信息。

函数首先使用 `load_image` 函数从指定文件名中读取输入图像，这个函数需要使用一个 `filename` 参数。接着，函数调用 `simple_transform` 函数对输入图像进行变换，这个函数需要传入一个输入图像、一个输出图像的大小、一个图像裁剪的大小、一个是否进行训练的参数、一个是否对颜色通道进行平均的参数，以及一个输入图像是否包含颜色信息的参数。函数的这些参数使用了星号 `*`，表示这些参数是输入参数，而 `**` 则表示这些参数是输出参数。

函数的最后一层返回语句返回对 `simple_transform` 函数的输出图像，这个输出图像的大小是 `(filename, resize_size, crop_size, is_train, is_color, mean.shape[1])`，其中 `mean` 参数的含义会在后面详细解释。


```
def load_and_transform(filename,
                       resize_size,
                       crop_size,
                       is_train,
                       is_color=True,
                       mean=None):
    """
    Load image from the input file `filename` and transform image for
    data argumentation. Please refer to the `simple_transform` interface
    for the transform operations.

    Example usage:

    .. code-block:: python

        im = load_and_transform('cat.jpg', 256, 224, True)

    :param filename: The file name of input image.
    :type filename: string
    :param resize_size: The shorter edge length of the resized image.
    :type resize_size: int
    :param crop_size: The cropping size.
    :type crop_size: int
    :param is_train: Whether it is training or not.
    :type is_train: bool
    :param is_color: whether the image is color or not.
    :type is_color: bool
    :param mean: the mean values, which can be element-wise mean values or
                 mean values per channel.
    :type mean: numpy array | list
    """
    im = load_image(filename, is_color)
    im = simple_transform(im, resize_size, crop_size, is_train, is_color, mean)
    return im

```