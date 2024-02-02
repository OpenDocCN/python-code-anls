# `arknights-mower\arknights_mower\utils\device\adb_client\const.py`

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
    # 定义键盘按键代码，对应的键盘按键为数字 7
    KEYCODE_7 = 14                   # 7
    # 定义键盘按键代码，对应的键盘按键为数字 8
    KEYCODE_8 = 15                   # 8
    # 定义键盘按键代码，对应的键盘按键为数字 9
    KEYCODE_9 = 16                   # 9
    # 定义键盘按键代码，对应的键盘按键为字母 A
    KEYCODE_A = 29                   # A
    # 定义键盘按键代码，对应的键盘按键为字母 B
    KEYCODE_B = 30                   # B
    # 定义键盘按键代码，对应的键盘按键为字母 C
    KEYCODE_C = 31                   # C
    # 定义键盘按键代码，对应的键盘按键为字母 D
    KEYCODE_D = 32                   # D
    # 定义键盘按键代码，对应的键盘按键为字母 E
    KEYCODE_E = 33                   # E
    # 定义键盘按键代码，对应的键盘按键为字母 F
    KEYCODE_F = 34                   # F
    # 定义键盘按键代码，对应的键盘按键为字母 G
    KEYCODE_G = 35                   # G
    # 定义键盘按键代码，对应的键盘按键为字母 H
    KEYCODE_H = 36                   # H
    # 定义键盘按键代码，对应的键盘按键为字母 I
    KEYCODE_I = 37                   # I
    # 定义键盘按键代码，对应的键盘按键为字母 J
    KEYCODE_J = 38                   # J
    # 定义键盘按键代码，对应的键盘按键为字母 K
    KEYCODE_K = 39                   # K
    # 定义键盘按键代码，对应的键盘按键为字母 L
    KEYCODE_L = 40                   # L
    # 定义键盘按键代码，对应的键盘按键为字母 M
    KEYCODE_M = 41                   # M
    # 定义键盘按键代码，对应的键盘按键为字母 N
    KEYCODE_N = 42                   # N
    # 定义键盘按键代码，对应的键盘按键为字母 O
    KEYCODE_O = 43                   # O
    # 定义键盘按键代码，对应的键盘按键为字母 P
    KEYCODE_P = 44                   # P
    # 定义键盘按键代码，对应的键盘按键为字母 Q
    KEYCODE_Q = 45                   # Q
    # 定义键盘按键代码，对应的键盘按键为字母 R
    KEYCODE_R = 46                   # R
    # 定义键盘按键代码，对应的键盘按键为字母 S
    KEYCODE_S = 47                   # S
    # 定义键盘按键代码，对应的键盘按键为字母 T
    KEYCODE_T = 48                   # T
    # 定义键盘按键代码，对应的键盘按键为字母 U
    KEYCODE_U = 49                   # U
    # 定义键盘按键代码，对应的键盘按键为字母 V
    KEYCODE_V = 50                   # V
    # 定义键盘按键代码，对应的键盘按键为字母 W
    KEYCODE_W = 51                   # W
    # 定义键盘按键代码，对应的键盘按键为字母 X
    KEYCODE_X = 52                   # X
    # 定义键盘按键代码，对应的键盘按键为字母 Y
    KEYCODE_Y = 53                   # Y
    # 定义键盘按键代码，对应的键盘按键为字母 Z
    KEYCODE_Z = 54                   # Z
    # 定义键盘按键代码，对应的键盘按键为加号
    KEYCODE_PLUS = 81                # +
    # 定义键盘按键代码，对应的键盘按键为减号
    KEYCODE_MINUS = 69               # -
    # 定义键盘按键代码，对应的键盘按键为星号
    KEYCODE_STAR = 17                # *
    # 定义键盘按键代码，对应的键盘按键为斜杠
    KEYCODE_SLASH = 76               # /
    # 定义键盘按键代码，对应的键盘按键为等号
    KEYCODE_EQUALS = 70              # =
    # 定义键盘按键代码，对应的键盘按键为@
    KEYCODE_AT = 77                  # @
    # 定义键盘按键代码，对应的键盘按键为井号
    KEYCODE_POUND = 18               # #
    # 定义键盘按键代码，对应的键盘按键为撇号
    KEYCODE_APOSTROPHE = 75          # '
    # 定义键盘按键代码，对应的键盘按键为反斜杠
    KEYCODE_BACKSLASH = 73           # \
    # 定义键盘按键代码，对应的键盘按键为逗号
    KEYCODE_COMMA = 55               # ,
    # 定义键盘按键代码，对应的键盘按键为句号
    KEYCODE_PERIOD = 56              # .
    # 定义键盘按键代码，对应的键盘按键为左方括号
    KEYCODE_LEFT_BRACKET = 71        # [
    # 定义键盘按键代码，对应的键盘按键为右方括号
    KEYCODE_RIGHT_BRACKET = 72       # ]
    # 定义键盘按键代码，对应的键盘按键为分号
    KEYCODE_SEMICOLON = 74           # ;
    # 定义键盘按键代码，对应的键盘按键为重音符号
    KEYCODE_GRAVE = 68               # `
    # 定义键盘按键代码，对应的键盘按键为空格键
    KEYCODE_SPACE = 62               # 空格键
    # 定义键盘按键代码，对应的键盘按键为多媒体键 >> 播放
    KEYCODE_MEDIA_PLAY = 126         # 多媒体键 >> 播放
    # 定义键盘按键代码，对应的键盘按键为多媒体键 >> 停止
    KEYCODE_MEDIA_STOP = 86          # 多媒体键 >> 停止
    # 定义键盘按键代码，对应的键盘按键为多媒体键 >> 暂停
    KEYCODE_MEDIA_PAUSE = 127        # 多媒体键 >> 暂停
    # 定义多媒体键码，表示多媒体键的功能
    KEYCODE_MEDIA_PLAY_PAUSE = 85    # 多媒体键 >> 播放 / 暂停
    KEYCODE_MEDIA_FAST_FORWARD = 90  # 多媒体键 >> 快进
    KEYCODE_MEDIA_REWIND = 89        # 多媒体键 >> 快退
    KEYCODE_MEDIA_NEXT = 87          # 多媒体键 >> 下一首
    KEYCODE_MEDIA_PREVIOUS = 88      # 多媒体键 >> 上一首
    KEYCODE_MEDIA_CLOSE = 128        # 多媒体键 >> 关闭
    KEYCODE_MEDIA_EJECT = 129        # 多媒体键 >> 弹出
    KEYCODE_MEDIA_RECORD = 130       # 多媒体键 >> 录音
```