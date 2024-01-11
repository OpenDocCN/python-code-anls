# `yolov5-DNF\getkeys.py`

```
# 导入 win32api 库并重命名为 wapi
import win32api as wapi

# 创建键盘按键到数字的映射字典
dict = {"A": 0, "S": 1,"D": 2, "F": 3, "G": 4, "H": 5, "Q": 6, "W": 7, "E": 8, "R": 9, "T": 10, "Y": 11, "up": 12,
        "down": 13, "left": 14, "right":15, "ctrl": 16, "alt": 17, "Z":18, "X":19, "C": 20, "esc": 21, "f2": 22,
        "space": 23, "num0": 24, "left_up": 25, "left down": 26, "right_up": 27, "right_down": 28, "left_A": 29,
        "left_S": 30, "left_D": 31, "left_F": 32, "left_G": 33, "left_H": 34,"left_Q": 35, "left_W": 36, "left_E": 37,
        "left_R": 38, "left_T": 39, "left_Y": 40, "up_A": 41,"up_S": 42, "up_D": 43, "up_F": 44, "up_G": 45,
        "up_H": 46,"up_Q": 47, "up_W": 48, "up_E": 49, "up_R": 50, "up_T": 51, "up_Y": 52,"down_A": 53,
        "down_S": 54, "down_D": 55, "down_F": 56, "down_G": 57, "down_H": 58,"down_Q": 59, "down_W": 60, "down_E": 61,
        "down_R": 62, "down_T": 63, "down_Y": 64, "right_A": 65, "right_S": 66, "right_D": 67, "right_F": 68, "right_G": 69,
        "right_H": 70,"right_Q": 71, "right_W": 72, "right_E": 73, "right_R": 74, "right_T": 75, "right_Y": 76, "left_z": 77,
        "left_x": 78, "left_c": 79, "up_z": 80,"up_x": 81, "up_c": 82, "down_z": 83, "down_x": 84, "down_c": 85, "right_z": 86,
        "right_x": 87, "right_c": 88, "left_ctrl": 89, "up_ctrl": 90, "down_ctrl": 91, "right_ctrl": 92, "P": 100}

# 创建键盘按键列表
keyList = []
for char in "ASDFGHQWERTYZXCP":
    keyList.append(char)

# 检查键盘按键状态
def key_check():
    keys = []
    # 遍历键盘按键列表
    for key in keyList:
        # 检查键盘按键状态，如果按下则添加到 keys 列表中
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    # 检查特殊按键状态，如果按下则添加到 keys 列表中
    if wapi.GetAsyncKeyState(37):
        keys.append("left")
    if wapi.GetAsyncKeyState(39):
        keys.append("right")
    if wapi.GetAsyncKeyState(38):
        keys.append("up")
    if wapi.GetAsyncKeyState(40):
        keys.append("down")
    if wapi.GetAsyncKeyState(17):
        keys.append("ctrl")
    if wapi.GetAsyncKeyState(18):
        keys.append("alt")
    if wapi.GetAsyncKeyState(27):
        keys.append("esc")
    # 如果按下 F2 键，则将 "f2" 添加到 keys 列表中
    if wapi.GetAsyncKeyState(113):
        keys.append("f2")
    # 如果按下空格键，则将 "space" 添加到 keys 列表中
    if wapi.GetAsyncKeyState(32):
        keys.append("space")
    # 如果按下数字键盘上的 0 键，则将 "num0" 添加到 keys 列表中
    if wapi.GetAsyncKeyState(96):
        keys.append("num0")
    # 返回按下的键组成的列表
    return keys
# 根据给定的按键列表获取对应的数值
def get_key(keys):
    # 如果按键列表长度为1，直接返回对应数值
    if len(keys) == 1:
        output = dict[keys[0]]
    # 如果按键列表长度为2
    elif len(keys) == 2:
        # 遍历按键列表
        for k in keys:
            # 如果按键是"left"、"up"、"down"或"right"
            if k == "left" or k == "up" or k == "down" or k == "right":
                # 移除按键列表中的当前按键
                keys.pop(keys.index(k))
                # 生成新的按键名
                key_name = k + "_" + keys[0]
                # 如果新的按键名在字典中存在
                if key_name in dict.keys():
                    output = dict[key_name]
                else:
                    output = dict[keys[0]]
            else:
                output = dict[keys[0]]
    # 如果按键列表长度大于2，直接返回第一个按键对应的数值
    elif len(keys) > 2:
        output = dict[keys[0]]
    # 如果按键列表为空，返回93表示不做任何动作
    else:
        output = 93   # 不做任何动作
    return output

# 如果当前脚本为主程序
if __name__ == '__main__':
    # 创建一个空字典
    undict = {}
    # 遍历原始字典的键值对，将值作为新字典的键，原始键作为新字典的值
    for key, val in dict.items():
        undict[val] = key
    # 打印新字典
    print(undict)
```