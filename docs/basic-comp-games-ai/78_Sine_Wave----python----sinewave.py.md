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
    # "REMarkable program by David Ahl"

    # 初始化变量
    string_index = 0
    radians: float = 0
    width = CENTER - 1

    # "Start long loop"
    # 循环 MAX_LINES 次，控制显示的行数
    for _line_num in range(MAX_LINES):

        # 获取要在此行显示的字符串
        curr_string = STRINGS[string_index]

        # 计算文本的打印位置
        sine = math.sin(radians)  # 计算正弦值
        padding = int(CENTER + width * sine)  # 根据正弦值计算打印位置
        print(curr_string.rjust(padding + len(curr_string)))  # 在计算出的位置右对齐打印字符串

        # 增加弧度值并递增字符串索引
        radians += STEP_SIZE  # 增加弧度值
        string_index += 1  # 递增字符串索引
        if string_index >= len(STRINGS):  # 如果字符串索引超出范围
            string_index = 0  # 重置字符串索引为0

        # 确保文本不会飞速滚动...
        time.sleep(DELAY)  # 延迟一段时间
# 如果当前脚本被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()

########################################################
#
# 移植说明
#
#   原始的 BASIC 版本在代码主体中硬编码了两个单词，然后使用一个标志位
#   （在 0 和 1 之间切换）和 IF 语句来确定下一个要显示的单词。
#
#   在这里，单词已经移动到了一个 Python 元组中，可以在不假设其长度的情况下进行迭代。
#   因此，STRINGS 元组可以被修改，以便程序打印出任意数量的文本行的任何序列。
#
#   由于运行 Python 的现代计算机会比运行 BASIC 的 70 年代计算机更快地将内容打印到屏幕上，
#   因此在这个版本中引入了延迟组件，以使输出更具历史准确性。
#
#
# 修改的想法
#
#   请求用户输入所需的行数（也许有一个“无限”选项）和/或步长。
#
#   让用户输入要显示的文本字符串，而不是在常量中预定义它。
#   根据最长字符串的长度计算一个适当的 CENTER。
#
#   尝试更改 STRINGS，使其只包含单个字符串，就像这样：
#
#       STRINGS = ('Howdy!')
#
#   会发生什么？为什么？你会如何修复它？
#
########################################################
```