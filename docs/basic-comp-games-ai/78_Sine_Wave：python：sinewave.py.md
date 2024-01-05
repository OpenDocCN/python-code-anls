# `78_Sine_Wave\python\sinewave.py`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从给定的文件名读取二进制数据，并将其封装成字节流对象
    使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里的内容创建一个 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回包含文件名到数据的字典
# 导入 math 和 time 模块
import math
import time


def main() -> None:
    # 常量
    STRINGS = ("Creative", "Computing")  # 要显示的文本
    MAX_LINES = 160  # 最大行数
    STEP_SIZE = 0.25  # 每行增加的弧度数。控制水平打印移动的速度。
    CENTER = 26  # 控制“中间”字符串的左边缘
    DELAY = 0.05  # 每行之间等待的秒数

    # 显示“介绍”文本
    print("\n                        Sine Wave")
    print("         Creative Computing  Morristown, New Jersey")
    print("\n\n\n\n")  # 打印四个换行符，用于分隔输出

    string_index = 0  # 初始化字符串索引为0
    radians: float = 0  # 初始化弧度为0
    width = CENTER - 1  # 计算宽度为CENTER减1

    # "Start long loop"  # 开始长循环的注释
    for _line_num in range(MAX_LINES):  # 遍历MAX_LINES次，循环变量不需要使用，用下划线表示

        # Get string to display on this line  # 获取要在此行显示的字符串的注释
        curr_string = STRINGS[string_index]  # 获取当前索引对应的字符串

        # Calculate how far over to print the text  # 计算要打印文本的偏移量的注释
        sine = math.sin(radians)  # 计算弧度的正弦值
        padding = int(CENTER + width * sine)  # 计算填充量
        print(curr_string.rjust(padding + len(curr_string)))  # 打印右对齐的字符串

        # Increase radians and increment our tuple index  # 增加弧度并增加我们的元组索引的注释
        radians += STEP_SIZE  # 增加步长
        string_index += 1  # 增加字符串索引，用于切换显示下一个字符串
        if string_index >= len(STRINGS):  # 如果字符串索引超过了字符串列表的长度
            string_index = 0  # 将字符串索引重置为0，循环显示字符串列表中的内容

        # Make sure the text doesn't fly by too fast...
        time.sleep(DELAY)  # 通过延迟一定时间来控制文本显示的速度


if __name__ == "__main__":
    main()

########################################################
#
# Porting Notes
#
#   The original BASIC version hardcoded two words in
#   the body of the code and then used a sentinel flag
#   (flipping between 0 and 1) with IF statements to
#   determine the word to display next.
#
```
#   Here, the words have been moved to a Python tuple,
#   which is iterated over without any assumptions about
#   how long it is.  The STRINGS tuple can therefore be
#   modified to have to program print out any sequence
#   of any number of lines of text.
# 将单词移动到 Python 元组中，无需假设其长度，因此可以修改 STRINGS 元组，使程序打印出任意数量的文本行序列。

#   Since a modern computer running Python will print
#   to the screen much more quickly than a '70s-era
#   computer running BASIC would, a delay component
#   has been introduced in this version to make the
#   output more historically accurate.
# 由于运行 Python 的现代计算机会比运行 BASIC 的 70 年代计算机更快地将内容打印到屏幕上，因此在这个版本中引入了延迟组件，使输出更具历史准确性。

# Ideas for Modifications
# 修改的想法

#   Ask the user for desired number of lines (perhaps
#   with an "infinite" option) and/or step size.
#   请求用户输入所需的行数（可能包括“无限”选项）和/或步长。

#   Let the user input the text strings to display,
#   rather than having it pre-defined in a constant.
#   允许用户输入要显示的文本字符串，而不是在常量中预先定义。
# 根据最长字符串的长度计算一个合适的中心位置
#
# 尝试改变STINGS，使其只包含一个字符串，就像这样：
#
#       STRINGS = ('Howdy!')
#
# 发生了什么？为什么？你会如何修复它？
#
########################################################
```