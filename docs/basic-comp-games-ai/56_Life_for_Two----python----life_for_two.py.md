# `basic-computer-games\56_Life_for_Two\python\life_for_two.py`

```

'''
LIFE FOR TWO

Competitive Game of Life (two or more players).

Ported by Sajid Sarker (2022).
'''
# Global Variable Initialisation
# 初始化游戏板
gn = [[0 for i in range(6)] for j in range(6)]  # 初始化游戏板的二维数组
gx = [0 for x in range(3)]  # 初始化玩家X坐标数组
gy = [0 for x in range(3)]  # 初始化玩家Y坐标数组
gk = [0, 3, 102, 103, 120, 130, 121, 112, 111, 12, 21, 30, 1020, 1030, 1011, 1021, 1003, 1002, 1012]  # 初始化游戏板的符号
ga = [0, -1, 0, 1, 0, 0, -1, 0, 1, -1, -1, 1, -1, -1, 1, 1, 1]  # 初始化游戏板的移动规则
m2 = 0  # 初始化玩家2的符号数量
m3 = 0  # 初始化玩家3的符号数量


# Helper Functions
# 辅助函数
def tab(number) -> str:
    t = ""
    while len(t) < number:
        t += " "
    return t


def display_header() -> None:
    print("{}LIFE2".format(tab(33)))  # 打印游戏标题
    print("{}CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n".format(tab(15)))  # 打印创意计算的信息
    print("{}U.B. LIFE GAME".format(tab(10))  # 打印游戏名称


# Board Functions
# 游戏板相关函数
def setup_board() -> None:
    # 初始化游戏板，玩家添加符号
    for b in range(1, 3):
        p1 = 3 if b != 2 else 30
        print("\nPLAYER {} - 3 LIVE PIECES.".format(b))  # 打印玩家信息
        for _ in range(1, 4):
            query_player(b)  # 查询玩家符号放置位置
            gn[gx[b]][gy[b]] = p1  # 在游戏板上放置符号


def modify_board() -> None:
    # 玩家轮流添加符号和修改游戏板
    for b in range(1, 3):
        print("PLAYER {} ".format(b))  # 打印当前玩家信息
        query_player(b)  # 查询玩家符号放置位置
        if b == 99:
            break
    if b <= 2:
        gn[gx[1]][gy[1]] = 100  # 在游戏板上放置符号
        gn[gx[2]][gy[2]] = 1000  # 在游戏板上放置符号


def simulate_board() -> None:
    # 模拟游戏板的一步
    for j in range(1, 6):
        for k in range(1, 6):
            if gn[j][k] > 99:
                b = 1 if gn[j][k] <= 999 else 10
                for o1 in range(1, 16, 2):
                    gn[j + ga[o1] - 1][k + ga[o1 + 1] - 1] += b  # 根据规则修改游戏板


def display_board() -> None:
    # 绘制游戏板及所有符号
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
                if j != 6:
                    print(" " + str(j) + " ", end="")
                else:
                    print(" 0\n")
            else:
                if gn[j][k] < 3:
                    gn[j][k] = 0
                    print("   ", end="")
                else:
                    for o1 in range(1, 19):
                        if gn[j][k] == gk[o1]:
                            break
                    if o1 <= 18:
                        if o1 > 9:
                            gn[j][k] = 1000
                            m3 += 1
                            print(" # ", end="")
                        else:
                            gn[j][k] = 100
                            m2 += 1
                            print(" * ", end="")
                    else:
                        gn[j][k] = 0
                        print("   ", end="")


# Player Functions
# 玩家相关函数
def query_player(b) -> None:
    # 查询玩家符号放置位置
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
        if gx[b] in range(1, 6) and gy[b] in range(1, 6) and gn[gx[b]][gy[b]] == 0:
            break
        print("ILLEGAL COORDS. RETYPE")
    if b != 1:
        if gx[1] == gx[2] and gy[1] == gy[2]:
            print("SAME COORD. SET TO 0")
            gn[gx[b] + 1][gy[b] + 1] = 0
            b = 99


# Game Functions
# 游戏相关函数
def check_winner(m2, m3) -> None:
    # 检查游戏是否结束
    if m2 == 0 and m3 == 0:
        print("\nA DRAW\n")  # 打印平局信息
        return
    if m3 == 0:
        print("\nPLAYER 1 IS THE WINNER\n")  # 打印玩家1获胜信息
        return
    if m2 == 0:
        print("\nPLAYER 2 IS THE WINNER\n")  # 打印玩家2获胜信息
        return


# Program Flow
# 程序流程
def main() -> None:
    display_header()  # 显示游戏标题
    setup_board()  # 初始化游戏板
    display_board()  # 显示游戏板
    while True:
        print("\n")
        simulate_board()  # 模拟游戏板
        display_board()  # 显示游戏板
        check_winner(m2, m3)  # 检查游戏是否结束
        modify_board()  # 修改游戏板


if __name__ == "__main__":
    main()

```