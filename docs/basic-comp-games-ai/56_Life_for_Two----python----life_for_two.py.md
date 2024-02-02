# `basic-computer-games\56_Life_for_Two\python\life_for_two.py`

```py
'''
LIFE FOR TWO

Competitive Game of Life (two or more players).

Ported by Sajid Sarker (2022).
'''
# Global Variable Initialisation
# 初始化游戏板
gn = [[0 for i in range(6)] for j in range(6)]  # 创建一个6x6的游戏板
gx = [0 for x in range(3)]  # 用于存储玩家的x坐标
gy = [0 for x in range(3)]  # 用于存储玩家的y坐标
gk = [0, 3, 102, 103, 120, 130, 121, 112, 111, 12, 21, 30, 1020, 1030, 1011, 1021, 1003, 1002, 1012]  # 存储特定位置的值
ga = [0, -1, 0, 1, 0, 0, -1, 0, 1, -1, -1, 1, -1, -1, 1, 1, 1]  # 存储相邻位置的偏移量
m2 = 0  # 初始化m2变量
m3 = 0  # 初始化m3变量


# Helper Functions
# 辅助函数
def tab(number) -> str:
    t = ""
    while len(t) < number:
        t += " "
    return t


def display_header() -> None:
    # 显示游戏标题
    print("{}LIFE2".format(tab(33)))
    print("{}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n".format(tab(15)))
    print("{}U.B. LIFE GAME".format(tab(10)))


# Board Functions
# 游戏板相关函数
def setup_board() -> None:
    # 玩家添加符号以初始化游戏板
    for b in range(1, 3):
        p1 = 3 if b != 2 else 30
        print("\nPLAYER {} - 3 LIVE PIECES.".format(b))
        for _ in range(1, 4):
            query_player(b)
            gn[gx[b]][gy[b]] = p1


def modify_board() -> None:
    # 玩家轮流添加符号并修改游戏板
    for b in range(1, 3):
        print("PLAYER {} ".format(b))
        query_player(b)
        if b == 99:
            break
    if b <= 2:
        gn[gx[1]][gy[1]] = 100
        gn[gx[2]][gy[2]] = 1000


def simulate_board() -> None:
    # 模拟游戏板的一步
    for j in range(1, 6):
        for k in range(1, 6):
            if gn[j][k] > 99:
                b = 1 if gn[j][k] <= 999 else 10
                for o1 in range(1, 16, 2):
                    gn[j + ga[o1] - 1][k + ga[o1 + 1] - 1] += b
                    # gn[j+ga[o1]][k+ga[o1+1]-1] = gn[j+ga[o1]][k+ga[o1+1]]+b


def display_board() -> None:
    # 绘制带有所有符号的游戏板
    m2, m3 = 0, 0
    # 循环7次，控制行数
    for j in range(7):
        # 打印空行
        print("")
        # 循环7次，控制列数
        for k in range(7):
            # 如果是第一行或最后一行
            if j == 0 or j == 6:
                # 如果不是最后一列
                if k != 6:
                    # 打印空格和列数
                    print(" " + str(k) + " ", end="")
                else:
                    # 打印0和换行
                    print(" 0 ", end="")
            # 如果是第一列或最后一列
            elif k == 0 or k == 6:
                # 如果不是最后一行
                if j != 6:
                    # 打印空格和行数
                    print(" " + str(j) + " ", end="")
                else:
                    # 打印0和换行
                    print(" 0\n")
            else:
                # 如果gn[j][k]小于3
                if gn[j][k] < 3:
                    # 将gn[j][k]设为0，打印3个空格
                    gn[j][k] = 0
                    print("   ", end="")
                else:
                    # 遍历gk列表
                    for o1 in range(1, 19):
                        # 如果gn[j][k]等于gk[o1]
                        if gn[j][k] == gk[o1]:
                            # 退出循环
                            break
                    # 如果o1小于等于18
                    if o1 <= 18:
                        # 如果o1大于9
                        if o1 > 9:
                            # 将gn[j][k]设为1000，m3加1，打印#和空格
                            gn[j][k] = 1000
                            m3 += 1
                            print(" # ", end="")
                        else:
                            # 将gn[j][k]设为100，m2加1，打印*和空格
                            gn[j][k] = 100
                            m2 += 1
                            print(" * ", end="")
                    else:
                        # 将gn[j][k]设为0，打印3个空格
                        gn[j][k] = 0
                        print("   ", end="")
# Player Functions
# 查询玩家放置符号的坐标
def query_player(b) -> None:
    # 循环直到玩家输入合法坐标
    while True:
        print("X,Y\nXXXXXX\n$$$$$$\n&&&&&&")
        a_ = input("??")
        b_ = input("???")
        x_ = [int(num) for num in a_.split() if num.isdigit()]
        y_ = [int(num) for num in b_.split() if num.isdigit()]
        x_ = [0] if len(x_) == 0 else x_
        y_ = [0] if len(y_) == 0 else y_
        gx[b] = y_[0]
        gy[b] = x_[0]
        # 检查坐标是否合法
        if gx[b] in range(1, 6)\
                and gy[b] in range(1, 6)\
                and gn[gx[b]][gy[b]] == 0:
            break
        print("ILLEGAL COORDS. RETYPE")
    # 如果不是玩家1，检查坐标是否与玩家1重复
    if b != 1:
        if gx[1] == gx[2] and gy[1] == gy[2]:
            print("SAME COORD. SET TO 0")
            gn[gx[b] + 1][gy[b] + 1] = 0
            b = 99


# Game Functions
# 检查游戏是否结束
def check_winner(m2, m3) -> None:
    if m2 == 0 and m3 == 0:
        print("\nA DRAW\n")
        return
    if m3 == 0:
        print("\nPLAYER 1 IS THE WINNER\n")
        return
    if m2 == 0:
        print("\nPLAYER 2 IS THE WINNER\n")
        return


# Program Flow
# 主程序
def main() -> None:
    display_header()
    setup_board()
    display_board()
    while True:
        print("\n")
        simulate_board()
        display_board()
        check_winner(m2, m3)
        modify_board()


if __name__ == "__main__":
    main()
```