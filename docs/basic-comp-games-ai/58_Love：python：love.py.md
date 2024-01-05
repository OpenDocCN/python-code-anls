# `58_Love\python\love.py`

```
# 这是一个多行注释，用三个双引号包裹起来
# 该程序是为了复制罗伯特·印第安纳（Robert Indiana）的伟大艺术作品“爱”的设计，并且可以添加最多60个字符的消息。
# DATA变量是一个交替计数，用于表示设计中字符和空格的数量。这些数据提供了标准每英寸10个字符的Teletype或线打印机的正确比例。
# LOVE程序由大卫·阿尔（David Ahl）创建。
# Python版本由Jeff Jetton于2019年制作。
# Image data. Each top-level element is a row. Each row element
# contains alternating character and blank run lengths.
# 图像数据。每个顶层元素是一行。每行元素包含交替的字符和空白运行长度。
DATA = [
    [
        60,  # 第一行只有一个长度为60的字符
    ],
    [1, 12, 26, 9, 12],  # 第二行包含交替的1和12个空白、26和9个字符、12个空白
    [3, 8, 24, 17, 8],   # 第三行包含交替的3和8个空白、24和17个字符、8个空白
    [4, 6, 23, 21, 6],    # 第四行包含交替的4和6个空白、23和21个字符、6个空白
    [4, 6, 22, 12, 5, 6, 5],  # 第五行包含交替的4和6个空白、22和12个字符、5和6个空白、5个字符
    [4, 6, 21, 11, 8, 6, 4],  # 第六行包含交替的4和6个空白、21和11个字符、8和6个空白、4个字符
    [4, 6, 21, 10, 10, 5, 4],  # 第七行包含交替的4和6个空白、21和10个字符、10和5个空白、4个字符
    [4, 6, 21, 9, 11, 5, 4],   # 第八行包含交替的4和6个空白、21和9个字符、11和5个空白、4个字符
    [4, 6, 21, 8, 11, 6, 4],   # 第九行包含交替的4和6个空白、21和8个字符、11和6个空白、4个字符
    [4, 6, 21, 7, 11, 7, 4],   # 第十行包含交替的4和6个空白、21和7个字符、11和7个空白、4个字符
    [4, 6, 21, 6, 11, 8, 4],   # 第十一行包含交替的4和6个空白、21和6个字符、11和8个空白、4个字符
    [4, 6, 19, 1, 1, 5, 11, 9, 4],  # 第十二行包含交替的4和6个空白、19和1个字符、1和5个空白、11和9个字符、4个空白
    [4, 6, 19, 1, 1, 5, 10, 10, 4],  # 第十三行包含交替的4和6个空白、19和1个字符、1和5个空白、10和10个字符、4个空白
    [4, 6, 18, 2, 1, 6, 8, 11, 4],   # 第十四行包含交替的4和6个空白、18和2个字符、1和6个空白、8和11个字符、4个空白
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，'r'表示以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 创建一个名为DATA的二维列表，包含5个子列表，每个子列表包含不同数量的整数
DATA = [
    [11, 8, 13, 27, 1],
    [
        60,
    ],
]

# 假设第一个子列表的总长度是每行使用的行长度
ROW_LEN = sum(DATA[0])

# 定义一个名为main的函数，不返回任何内容
def main() -> None:
    # 显示介绍文本
    print("\n                  Love")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    print("A tribute to the great American artist, Robert Indiana.")
    print("His great work will be reproduced with a message of")
    print("your choice up to 60 characters.  If you can't think of")
    # 打印一条消息，提示用户输入“love”这个单词
    print("a message, simple type the word 'love'\n")  # (sic)

    # 从用户获取消息
    message = input("Your message, please? ")
    if message == "":
        message = "LOVE"

    # 重复消息直到至少有一行的长度
    while len(message) < ROW_LEN:
        message += message

    # 显示图像
    print("\n" * 9)
    for row in DATA:
        print_message = True
        position = 0
        line_text = ""
        for length in row:
            if print_message:
                # 从消息中获取指定位置和长度的文本
                text = message[position : (position + length)]
                print_message = False  # 设置一个布尔变量，用于控制是否打印消息
            else:
                text = " " * length  # 根据长度创建一个空格字符串
                print_message = True  # 设置布尔变量，用于控制是否打印消息
            line_text += text  # 将文本添加到行文本中
            position += length  # 更新位置变量
        print(line_text)  # 打印行文本

    print()  # 打印空行


if __name__ == "__main__":
    main()  # 调用主函数


######################################################################
#
# Porting Notes
#
#   Not too different from the original, logic-wise. The image was
```
#   originally encoded as a series of BASIC "DATA" lines. Here,
#   we've converted it to a more Pythonic nested list structure.
#   Other changes include reducing some of the vertical spacing
#   (since we'll probably be showing this on a screen rather than
#   the sort of tractor-feed printer the program was written for)
#   and having the message default to LOVE when no input is given.
#
#   This program uses a simple version of run-length encoding to
#   compress a 60 x 36 image (2,160 characters) into just 252 DATA
#   values.  That's about an 8.5-to-1 data compression ratio,
#   which is pretty good!
#
#
# Ideas for Modifications
#
#   Process the user's message input to remove spaces and change
#   to uppercase.
#
#   Encode other images in a similar fashion and let the user choose
#   which one they'd like to use to display their message.
# 为了帮助上面的步骤，创建一个程序，它可以读取任何类型的字符/空格艺术的文本文件，并生成初始化正确嵌套值列表的Python代码。
# 例如，如果输入文件是：
#     *****
#     *  **
#     **  *
# 你的程序将输出：
#    ((5, ), (1, 1, 2), (2, 1, 1))
# ######################################################################
```