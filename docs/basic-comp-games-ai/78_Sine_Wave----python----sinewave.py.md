# `basic-computer-games\78_Sine_Wave\python\sinewave.py`

```

# 导入 math 和 time 模块
import math
import time

# 定义主函数
def main() -> None:
    # 定义常量
    STRINGS = ("Creative", "Computing")  # 要显示的文本
    MAX_LINES = 160  # 最大行数
    STEP_SIZE = 0.25  # 每行增加的弧度数。控制水平打印移动的速度。
    CENTER = 26  # 控制“中间”字符串的左边缘
    DELAY = 0.05  # 每行之间等待的秒数

    # 显示“介绍”文本
    print("\n                        Sine Wave")
    print("         Creative Computing  Morristown, New Jersey")
    print("\n\n\n\n")

    string_index = 0
    radians: float = 0
    width = CENTER - 1

    # "Start long loop"
    # 开始循环
    for _line_num in range(MAX_LINES):

        # 获取要在此行显示的字符串
        curr_string = STRINGS[string_index]

        # 计算要打印文本的位置
        sine = math.sin(radians)
        padding = int(CENTER + width * sine)
        print(curr_string.rjust(padding + len(curr_string)))

        # 增加弧度并增加我们的元组索引
        radians += STEP_SIZE
        string_index += 1
        if string_index >= len(STRINGS):
            string_index = 0

        # 确保文本不会飞得太快...
        time.sleep(DELAY)

# 如果运行的是主程序，则调用主函数
if __name__ == "__main__":
    main()

```