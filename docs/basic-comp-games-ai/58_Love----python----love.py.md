# `basic-computer-games\58_Love\python\love.py`

```
# 程序的介绍和版权信息
"""
LOVE

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"This program is designed to reproduce Robert Indiana's great art
 work 'Love' with a message of your choice up to 60 characters long.

"The [DATA variable is] an alternating count of the number
 of characters and blanks which form the design.  These data give
 the correct proportions for a standard 10 character per inch
 Teletype or line printer.

"The LOVE program was created by David Ahl."


Python port by Jeff Jetton, 2019
"""

# 图像数据，每个顶层元素是一行，每行元素包含交替的字符和空白的长度
DATA = [
    [
        60,
    ],
    [1, 12, 26, 9, 12],
    [3, 8, 24, 17, 8],
    ...
    # 省略部分数据
    ...
    [11, 8, 13, 27, 1],
    [
        60,
    ],
]

# 假设第一个元素的总长度是每行使用的长度
ROW_LEN = sum(DATA[0])

# 主函数
def main() -> None:
    # 显示介绍文本
    print("\n                  Love")
    print("Creative Computing  Morristown, New Jersey")
    # 打印空行
    print("\n\n")
    # 打印关于美国艺术家Robert Indiana的致敬词
    print("A tribute to the great American artist, Robert Indiana.")
    # 打印关于他作品的描述
    print("His great work will be reproduced with a message of")
    print("your choice up to 60 characters.  If you can't think of")
    print("a message, simple type the word 'love'\n")  # (sic)
    
    # 从用户获取消息
    message = input("Your message, please? ")
    # 如果用户没有输入消息，则默认为"LOVE"
    if message == "":
        message = "LOVE"
    
    # 重复消息直到至少有一行的长度
    while len(message) < ROW_LEN:
        message += message
    
    # 显示图像
    print("\n" * 9)
    # 遍历数据中的每一行
    for row in DATA:
        # 初始化打印消息标志和位置
        print_message = True
        position = 0
        line_text = ""
        # 遍历每一行中的长度
        for length in row:
            # 如果需要打印消息
            if print_message:
                # 从消息中获取对应长度的文本
                text = message[position : (position + length)]
                print_message = False
            else:
                # 否则填充空格
                text = " " * length
                print_message = True
            # 将文本添加到行文本中
            line_text += text
            position += length
        # 打印行文本
        print(line_text)
    
    print()
# 如果当前脚本被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()


######################################################################
#
# 移植说明
#
#   逻辑上与原始版本并没有太大不同。图像最初被编码为一系列BASIC的“DATA”行。在这里，
#   我们将其转换为更符合Python风格的嵌套列表结构。其他更改包括减少一些垂直间距
#   （因为我们可能会在屏幕上显示，而不是程序最初编写的拖拉机进纸打印机上），
#   并在没有输入时将消息默认为LOVE。
#
#   该程序使用简单的游程编码版本，将60 x 36的图像（2,160个字符）压缩为仅252个DATA值。
#   这大约是8.5比1的数据压缩比，相当不错！
#
#
# 修改的想法
#
#   处理用户输入的消息以删除空格并转换为大写。
#
#   以类似的方式编码其他图像，并让用户选择他们想要用来显示消息的图像。
#
#   为了帮助上述步骤，创建一个程序，读取任何类似字符/空格艺术的文本文件，并生成
#   初始化正确嵌套值的Python代码。
#
#   例如，如果输入文件是：
#
#     *****
#     *  **
#     **  *
#
#   您的程序将输出：
#
#    ((5, ), (1, 1, 2), (2, 1, 1))
#
######################################################################
```