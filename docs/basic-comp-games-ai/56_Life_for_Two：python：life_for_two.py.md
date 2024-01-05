# `56_Life_for_Two\python\life_for_two.py`

```
'''
LIFE FOR TWO

Competitive Game of Life (two or more players).

Ported by Sajid Sarker (2022).
'''
# Global Variable Initialisation
# Initialise the board
gn = [[0 for i in range(6)] for j in range(6)]  # 6x6 grid for the game board
gx = [0 for x in range(3)]  # List to store x-coordinates of players
gy = [0 for x in range(3)]  # List to store y-coordinates of players
gk = [0, 3, 102, 103, 120, 130, 121, 112, 111, 12, 21, 30, 1020, 1030, 1011, 1021, 1003, 1002, 1012]  # List of predefined patterns for the game
ga = [0, -1, 0, 1, 0, 0, -1, 0, 1, -1, -1, 1, -1, -1, 1, 1, 1]  # List of predefined movements for the game
m2 = 0  # Variable to store a specific value
m3 = 0  # Variable to store a specific value
# Helper Functions
# 定义一个函数，根据输入的数字返回相应数量的空格字符串
def tab(number) -> str:
    t = ""
    while len(t) < number:
        t += " "
    return t

# 定义一个函数，用于显示游戏的标题
def display_header() -> None:
    print("{}LIFE2".format(tab(33)))
    print("{}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n".format(tab(15)))
    print("{}U.B. LIFE GAME".format(tab(10)))


# Board Functions
# 定义一个函数，用于初始化游戏棋盘
def setup_board() -> None:
    # 玩家添加符号来初始化棋盘
    for b in range(1, 3):
        p1 = 3 if b != 2 else 30
        print("\nPLAYER {} - 3 LIVE PIECES.".format(b))
        for _ in range(1, 4):
            # 循环3次，用于玩家进行3次操作
            query_player(b)
            # 调用query_player函数，让玩家输入操作
            gn[gx[b]][gy[b]] = p1
            # 在游戏棋盘上根据玩家的操作修改相应位置的值


def modify_board() -> None:
    # Players take turns to add symbols and modify the board
    # 玩家轮流添加符号并修改游戏棋盘
    for b in range(1, 3):
        print("PLAYER {} ".format(b))
        # 打印当前玩家的编号
        query_player(b)
        # 调用query_player函数，让玩家输入操作
        if b == 99:
            break
    if b <= 2:
        gn[gx[1]][gy[1]] = 100
        gn[gx[2]][gy[2]] = 1000
        # 如果玩家编号小于等于2，根据玩家的操作修改游戏棋盘


def simulate_board() -> None:
    # Simulate the board for one step
    # 模拟游戏棋盘进行一步操作
    for j in range(1, 6):
        # 循环5次，用于模拟游戏棋盘的操作
            if k != 6:
                print(" " + str(j) + " ", end="")
            else:
                print(" 0 ", end="")
        else:
            if gn[j][k] == 0:
                print(" . ", end="")
            elif gn[j][k] == 1:
                print(" X ", end="")
            else:
                print(" O ", end="")
    print("")
```

在这段代码中，我们需要为两个函数中的每个语句添加注释。

对于第一个函数中的代码：
```
        for k in range(1, 6):
            if gn[j][k] > 99:
                b = 1 if gn[j][k] <= 999 else 10
                for o1 in range(1, 16, 2):
                    gn[j + ga[o1] - 1][k + ga[o1 + 1] - 1] += b
                    # gn[j+ga[o1]][k+ga[o1+1]-1] = gn[j+ga[o1]][k+ga[o1+1]]+b
```
这段代码是一个嵌套的循环，它对数组中的元素进行操作。需要解释每个循环的作用以及条件语句的含义。

对于第二个函数中的代码：
```
def display_board() -> None:
    # Draws the board with all symbols
    m2, m3 = 0, 0
    for j in range(7):
        print("")
        for k in range(7):
            if j == 0 or j == 6:
                if k != 6:
                    print(" " + str(k) + " ", end="")
                else:
                    print(" 0 ", end="")
            elif k == 0 or k == 6:
                if k != 6:
                    print(" " + str(j) + " ", end="")
                else:
                    print(" 0 ", end="")
            else:
                if gn[j][k] == 0:
                    print(" . ", end="")
                elif gn[j][k] == 1:
                    print(" X ", end="")
                else:
                    print(" O ", end="")
    print("")
```
这段代码是一个用于绘制游戏板的函数。需要解释每个循环和条件语句的作用，以及打印出的符号代表的含义。
                if j != 6:  # 如果 j 不等于 6
                    print(" " + str(j) + " ", end="")  # 打印 j 的值，并在末尾不换行
                else:  # 否则
                    print(" 0\n")  # 打印 0 并换行
            else:  # 否则
                if gn[j][k] < 3:  # 如果 gn[j][k] 小于 3
                    gn[j][k] = 0  # 将 gn[j][k] 设为 0
                    print("   ", end="")  # 打印三个空格，并在末尾不换行
                else:  # 否则
                    for o1 in range(1, 19):  # 遍历范围为 1 到 18
                        if gn[j][k] == gk[o1]:  # 如果 gn[j][k] 等于 gk[o1]
                            break  # 跳出循环
                    if o1 <= 18:  # 如果 o1 小于等于 18
                        if o1 > 9:  # 如果 o1 大于 9
                            gn[j][k] = 1000  # 将 gn[j][k] 设为 1000
                            m3 += 1  # m3 加一
                            print(" # ", end="")  # 打印" # "，并在末尾不换行
                        else:  # 否则
                            gn[j][k] = 100  # 将 gn[j][k] 设为 100
                            m2 += 1  # m2 加一
                            print(" * ", end="")  # 打印一个星号并以空格结尾
                    else:
                        gn[j][k] = 0  # 将 gn[j][k] 的值设为 0
                        print("   ", end="")  # 打印三个空格并以空格结尾


# Player Functions
def query_player(b) -> None:
    # 查询玩家放置符号的坐标
    while True:
        print("X,Y\nXXXXXX\n$$$$$$\n&&&&&&")
        a_ = input("??")  # 获取用户输入的值
        b_ = input("???")  # 获取用户输入的值
        x_ = [int(num) for num in a_.split() if num.isdigit()]  # 将用户输入的值按空格分割并转换为整数
        y_ = [int(num) for num in b_.split() if num.isdigit()]  # 将用户输入的值按空格分割并转换为整数
        x_ = [0] if len(x_) == 0 else x_  # 如果 x_ 的长度为 0，则将其设为 [0]
        y_ = [0] if len(y_) == 0 else y_  # 如果 y_ 的长度为 0，则将其设为 [0]
        gx[b] = y_[0]  # 将 gx[b] 的值设为 y_ 的第一个元素
        gy[b] = x_[0]  # 将 gy[b] 的值设为 x_ 的第一个元素
        if gx[b] in range(1, 6)\  # 如果 gx[b] 在范围 [1, 6) 内
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
    if m2 == 0:
        # 如果 m2 等于 0，则打印出“PLAYER 2 IS THE WINNER”，然后返回
        print("\nPLAYER 2 IS THE WINNER\n")
        return


# Program Flow
def main() -> None:
    # 显示程序的标题
    display_header()
    # 设置游戏板
    setup_board()
    # 显示游戏板
    display_board()
    # 循环执行以下步骤
    while True:
        # 打印换行符
        print("\n")
        # 模拟游戏板的变化
        simulate_board()
        # 显示游戏板
        display_board()
        # 检查是否有获胜者
        check_winner(m2, m3)
        # 修改游戏板
        modify_board()


if __name__ == "__main__":
    # 如果程序作为主程序运行，则执行 main() 函数
    main()
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```