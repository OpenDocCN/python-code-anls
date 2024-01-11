# `yolov5-DNF\test\direction_move.py`

```
# 导入时间和随机模块
import time
import random
# 从 directkeys 模块中导入 PressKey 和 ReleaseKey 函数
from directkeys import PressKey, ReleaseKey

# 定义方向键码的字典
direct_dic = {"UP": 0xC8, "DOWN": 0xD0, "LEFT": 0xCB, "RIGHT": 0xCD}

# 定义移动函数，接受方向、材料标志、动作缓存、按键按下延迟和释放延迟作为参数
def move(direct, material=False, action_cache=None, press_delay=0.1,
         release_delay=0.1):
    # 打印动作缓存
    print(action_cache)
    # 如果方向是向右
    if direct == "RIGHT":
        # 如果动作缓存不为空
        if action_cache is not None:
            # 如果动作缓存不是向右
            if action_cache != "RIGHT":
                # 如果动作缓存不是左、右、上、下中的任何一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放动作缓存对应的两个方向键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放动作缓存对应的方向键
                    ReleaseKey(direct_dic[action_cache])
                # 按下向右键
                PressKey(direct_dic["RIGHT"])
                # 如果不是材料模式，延迟按下和释放
                if not material:
                    time.sleep(press_delay)
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    PressKey(direct_dic["RIGHT"])
                # 更新动作缓存为向右
                action_cache = "RIGHT"
                # 打印向右移动
                print("向右移动")
            else:
                # 打印向右移动
                print("向右移动")
        else:
            # 按下向右键
            PressKey(direct_dic["RIGHT"])
            # 如果不是材料模式，延迟按下和释放
            if not material:
                time.sleep(press_delay)
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                PressKey(direct_dic["RIGHT"])
            # 更新动作缓存为向右
            action_cache = "RIGHT"
            # 打印向右移动
            print("向右移动")
    # 返回更新后的动作缓存
    return action_cache
    # 如果方向是向左
    elif direct == "LEFT":
        # 如果有上一个动作
        if action_cache is not None:
            # 如果上一个动作不是向左
            if action_cache != "LEFT":
                # 如果上一个动作不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放上一个动作的按键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放上一个动作的按键
                    ReleaseKey(direct_dic[action_cache])
                # 按下向左的按键
                PressKey(direct_dic["LEFT"])
                # 如果没有材料
                if not material:
                    # 等待按下延迟时间
                    time.sleep(press_delay)
                    # 释放向左的按键
                    ReleaseKey(direct_dic["LEFT"])
                    # 等待释放延迟时间
                    time.sleep(release_delay)
                    # 再次按下向左的按键
                    PressKey(direct_dic["LEFT"])
                # 缓存当前动作为向左
                action_cache = "LEFT"
                # 打印向左移动
                print("向左移动")
            else:
                # 打印向左移动
                print("向左移动")
        else:
            # 按下向左的按键
            PressKey(direct_dic["LEFT"])
            # 如果没有材料
            if not material:
                # 等待按下延迟时间
                time.sleep(press_delay)
                # 释放向左的按键
                ReleaseKey(direct_dic["LEFT"])
                # 等待释放延迟时间
                time.sleep(release_delay)
                # 再次按下向左的按键
                PressKey(direct_dic["LEFT"])
            # 缓存当前动作为向左
            action_cache = "LEFT"
            # 打印向左移动
            print("向左移动")
        # 返回当前动作
        return action_cache
    # 如果方向是向上
    elif direct == "UP":
        # 如果动作缓存不为空
        if action_cache != None:
            # 如果动作缓存不是向上
            if action_cache != "UP":
                # 如果动作缓存不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放动作缓存中第一个方向键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    # 释放动作缓存中第二个方向键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放动作缓存中的方向键
                    ReleaseKey(direct_dic[action_cache])
                # 按下向上方向键
                PressKey(direct_dic["UP"])
                # 更新动作缓存为向上
                action_cache = "UP"
                # 打印向上移动
                print("向上移动")
            else:
                # 打印向上移动
                print("向上移动")
        else:
            # 按下向上方向键
            PressKey(direct_dic["UP"])
            # 更新动作缓存为向上
            action_cache = "UP"
            # 打印向上移动
            print("向上移动")
        # 返回更新后的动作缓存
        return action_cache
    # 如果方向是向下
    elif direct == "DOWN":
        # 如果动作缓存不为空
        if action_cache != None:
            # 如果动作缓存不是向下
            if action_cache != "DOWN":
                # 如果动作缓存不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放动作缓存中的第一个方向键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    # 释放动作缓存中的第二个方向键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放动作缓存中的方向键
                    ReleaseKey(direct_dic[action_cache])
                # 按下向下方向键
                PressKey(direct_dic["DOWN"])
                # 更新动作缓存为向下
                action_cache = "DOWN"
                # 打印向下移动
                print("向下移动")
            else:
                # 打印向下移动
                print("向下移动")
        else:
            # 按下向下方向键
            PressKey(direct_dic["DOWN"])
            # 更新动作缓存为向下
            action_cache = "DOWN"
            # 打印向下移动
            print("向下移动")
        # 返回动作缓存
        return action_cache
    # 如果方向为右上
    elif direct == "RIGHT_UP":
        # 如果动作缓存不为空
        if action_cache != None:
            # 如果动作缓存不是右上
            if action_cache != "RIGHT_UP":
                # 如果动作缓存不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放动作缓存中的左右键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放动作缓存中的键
                    ReleaseKey(direct_dic[action_cache])
                # 如果没有材料
                if not material:
                    # 按下右键
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                    # 释放右键
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    # 再次按下右键
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                # 如果有材料
                if material:
                    # 按下右键
                    PressKey(direct_dic["RIGHT"])
                # 按下上键
                PressKey(direct_dic["UP"])
                # 更新动作缓存为右上
                action_cache = "RIGHT_UP"
                # 打印右上移动
                print("右上移动")
            else:
                # 打印右上移动
                print("右上移动")
        else:
            # 如果没有材料
            if not material:
                # 按下右键
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
                # 释放右键
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                # 再次按下右键
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
            # 如果有材料
            if material:
                # 按下右键
                PressKey(direct_dic["RIGHT"])
            # 按下上键
            PressKey(direct_dic["UP"])
            # 更新动作缓存为右上
            action_cache = "RIGHT_UP"
            # 打印右上移动
            print("右上移动")
        # 返回动作缓存
        return action_cache
    # 如果方向为右下
    elif direct == "RIGHT_DOWN":
        # 如果动作缓存不为空
        if action_cache != None:
            # 如果动作缓存不是右下
            if action_cache != "RIGHT_DOWN":
                # 如果动作缓存不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放动作缓存中的两个方向键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放动作缓存中的一个方向键
                    ReleaseKey(direct_dic[action_cache])
                # 如果没有材料
                if not material:
                    # 按下右键
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                    # 释放右键
                    ReleaseKey(direct_dic["RIGHT"])
                    time.sleep(release_delay)
                    # 再次按下右键
                    PressKey(direct_dic["RIGHT"])
                    time.sleep(press_delay)
                # 如果有材料
                if material:
                    # 按下右键
                    PressKey(direct_dic["RIGHT"])
                # 按下下键
                PressKey(direct_dic["DOWN"])
                # 更新动作缓存为右下
                action_cache = "RIGHT_DOWN"
                # 打印右上移动
                print("右上移动")
            else:
                # 打印右上移动
                print("右上移动")
        else:
            # 如果没有材料
            if not material:
                # 按下右键
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
                # 释放右键
                ReleaseKey(direct_dic["RIGHT"])
                time.sleep(release_delay)
                # 再次按下右键
                PressKey(direct_dic["RIGHT"])
                time.sleep(press_delay)
            # 如果有材料
            if material:
                # 按下右键
                PressKey(direct_dic["RIGHT"])
            # 按下下键
            PressKey(direct_dic["DOWN"])
            # 更新动作缓存为右下
            action_cache = "RIGHT_DOWN"
            # 打印右上移动
            print("右上移动")
        # 返回动作缓存
        return action_cache
    # 如果方向是左上
    elif direct == "LEFT_UP":
        # 如果有缓存的动作
        if action_cache != None:
            # 如果缓存的动作不是左上
            if action_cache != "LEFT_UP":
                # 如果缓存的动作不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放缓存动作对应的按键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放缓存动作对应的按键
                    ReleaseKey(direct_dic[action_cache])
                # 如果没有材料
                if not material:
                    # 按下左键
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                    # 释放左键
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    # 按下左键
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                # 如果有材料
                if material:
                    # 按下左键
                    PressKey(direct_dic["LEFT"])
                # 按下上键
                PressKey(direct_dic["UP"])
                # 更新缓存的动作为左上
                action_cache = "LEFT_UP"
                # 打印左上移动
                print("左上移动")
            else:
                # 打印左上移动
                print("左上移动")
        else:
            # 如果没有材料
            if not material:
                # 按下左键
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
                # 释放左键
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                # 按下左键
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
            # 如果有材料
            if material:
                # 按下左键
                PressKey(direct_dic["LEFT"])
            # 按下上键
            PressKey(direct_dic["UP"])
            # 更新缓存的动作为左上
            action_cache = "LEFT_UP"
            # 打印左上移动
            print("左上移动")
        # 返回缓存的动作
        return action_cache
    # 如果方向是左下
    elif direct == "LEFT_DOWN":
        # 如果有缓存的动作
        if action_cache != None:
            # 如果缓存的动作不是左下
            if action_cache != "LEFT_DOWN":
                # 如果缓存的动作不是左、右、上、下中的一个
                if action_cache not in ["LEFT", "RIGHT", "UP", "DOWN"]:
                    # 释放缓存动作对应的按键
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[0]])
                    ReleaseKey(direct_dic[action_cache.strip().split("_")[1]])
                else:
                    # 释放缓存动作对应的按键
                    ReleaseKey(direct_dic[action_cache])
                # 如果没有材料
                if not material:
                    # 按下左键
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                    # 释放左键
                    ReleaseKey(direct_dic["LEFT"])
                    time.sleep(release_delay)
                    # 按下左键
                    PressKey(direct_dic["LEFT"])
                    time.sleep(press_delay)
                # 如果有材料
                if material:
                    # 按下左键
                    PressKey(direct_dic["LEFT"])
                # 按下下键
                PressKey(direct_dic["DOWN"])
                # 更新缓存的动作为左下
                action_cache = "LEFT_DOWN"
                # 打印左下移动
                print("左下移动")
            else:
                # 打印左下移动
                print("左下移动")
        else:
            # 如果没有材料
            if not material:
                # 按下左键
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
                # 释放左键
                ReleaseKey(direct_dic["LEFT"])
                time.sleep(release_delay)
                # 按下左键
                PressKey(direct_dic["LEFT"])
                time.sleep(press_delay)
            # 如果有材料
            if material:
                # 按下左键
                PressKey(direct_dic["LEFT"])
            # 按下下键
            PressKey(direct_dic["DOWN"])
            # 更新缓存的动作为左下
            action_cache = "LEFT_DOWN"
            # 打印左下移动
            print("左下移动")
        # 返回缓存的动作
        return action_cache
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 初始化动作缓存为 None
    action_cache = None
    # 记录当前时间
    t1 = time.time()
    # 循环执行以下操作
    # if  int(time.time() - t1) % 2 == 0:
    #     action_cache = move("LEFT_DOWN", material=False, action_cache=action_cache, press_delay=0.1, release_delay=0.1)
    # else:
    # 执行移动动作，向右上方移动，使用材料，传入动作缓存，按下和释放延迟均为 0.1 秒
    action_cache = move("RIGHT_UP", material=True, action_cache=action_cache, press_delay=0.1, release_delay=0.1)
```