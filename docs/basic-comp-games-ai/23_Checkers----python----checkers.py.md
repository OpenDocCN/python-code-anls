# `basic-computer-games\23_Checkers\python\checkers.py`

```

# 导入必要的模块
from typing import Iterator, NamedTuple, Optional, Tuple

# 定义常量
PAGE_WIDTH = 64
HUMAN_PLAYER = 1
COMPUTER_PLAYER = -1
HUMAN_PIECE = 1
HUMAN_KING = 2
COMPUTER_PIECE = -1
COMPUTER_KING = -2
EMPTY_SPACE = 0
TOP_ROW = 7
BOTTOM_ROW = 0

# 定义一个名为MoveRecord的命名元组，用于记录移动的质量和坐标
class MoveRecord(NamedTuple):
    quality: int
    start_x: int
    start_y: int
    dest_x: int
    dest_y: int

# 打印居中的消息
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

# 获取用户输入的坐标
def get_coordinates(prompt: str) -> Tuple[int, int]:
    err_msg = "ENTER COORDINATES in X,Y FORMAT"
    while True:
        print(prompt)
        response = input()
        if "," not in response:
            print(err_msg)
            continue
        try:
            x, y = (int(c) for c in response.split(","))
        except ValueError:
            print(err_msg)
            continue
        return x, y

# 检查坐标是否合法
def is_legal_board_coordinate(x: int, y: int) -> bool:
    return (0 <= x <= 7) and (0 <= y <= 7)

# 打印游戏说明
def print_instructions() -> None:
    print("THIS IS THE GAME OF CHECKERS.  THE COMPUTER IS X,")
    # 其他说明略

# 打印人类玩家获胜的消息
def print_human_won() -> None:
    print("\nYOU WIN.")

# 打印计算机玩家获胜的消息
def print_computer_won() -> None:
    print("\nI WIN.")

# 进行游戏
def play_game() -> None:
    board = Board()
    # 游戏逻辑略

# 主函数
def main() -> None:
    print_header("CHECKERS")
    print_instructions()
    play_game()

# 如果是直接运行该脚本，则执行主函数
if __name__ == "__main__":
    main()

```