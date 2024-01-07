# `basic-computer-games\65_Nim\python\Traditional_NIM.py`

```

# 导入 random 模块和 typing 模块中的 Tuple 类型
import random
from typing import Tuple

# 定义 NIM 类
class NIM:
    # 初始化方法，初始化游戏的初始状态
    def __init__(self) -> None:
        self.piles = {1: 7, 2: 5, 3: 3, 4: 1}

    # 移除棋子的方法，根据玩家输入的命令来移除对应数量的棋子
    def remove_pegs(self, command) -> None:
        try:
            # 尝试解析玩家输入的命令
            pile, num = command.split(",")
            num = int(num)
            pile = int(pile)
        except Exception as e:
            # 处理异常情况
            if "not enough values" in str(e):
                print(
                    '\nNot a valid command. Your command should be in the form of "1,3", Try Again\n'
                )
            else:
                print("\nError, Try again\n")
            return None
        # 根据命令的合法性来移除对应数量的棋子
        if self._command_integrity(num, pile):
            self.piles[pile] -= num
        else:
            print("\nInvalid value of either Peg or Pile\n")

    # 获取 AI 的移动，随机选择一个非空的堆，并随机移除其中的一些棋子
    def get_ai_move(self) -> Tuple[int, int]:
        possible_pile = []
        for k, v in self.piles.items():
            if v != 0:
                possible_pile.append(k)
        pile = random.choice(possible_pile)
        num = random.randint(1, self.piles[pile])
        return pile, num

    # 检查命令的合法性
    def _command_integrity(self, num, pile) -> bool:
        return pile <= 4 and pile >= 1 and num <= self.piles[pile]

    # 打印当前棋盘状态
    def print_pegs(self) -> None:
        for pile, peg in self.piles.items():
            print("Pile {} : {}".format(pile, "O " * peg)

    # 提供游戏帮助信息
    def help(self) -> None:
        print("-" * 10)
        print('\nThe Game is player with a number of Piles of Objects("O" == one peg)')
        # 更多帮助信息...

    # 检查是否有玩家获胜
    def check_for_win(self) -> bool:
        sum = 0
        for v in self.piles.values():
            sum += v
        return sum == 0

# 游戏主函数
def main() -> None:
    # 初始化游戏
    game = NIM()
    print("Hello, This is a game of NIM")
    help = input("Do You Need Instruction (YES or NO): ")
    if help.lower() == "yes":
        game.help()
    # 游戏循环
    input("\nPress Enter to start the Game:\n")
    end = False
    while True:
        game.print_pegs()
        # 玩家移动
        command = input("\nYOUR MOVE - Number of PILE, Number of Object? ")
        game.remove_pegs(command)
        end = game.check_for_win()
        if end:
            print("\nPlayer Wins the Game, Congratulations!!")
            input("\nPress any key to exit")
            break
        # 电脑移动
        ai_command = game.get_ai_move()
        print(
            "\nA.I MOVE - A.I Removed {} pegs from Pile {}".format(
                ai_command[1], ai_command[0]
            )
        )
        game.remove_pegs(str(ai_command[0]) + "," + str(ai_command[1]))
        end = game.check_for_win()
        if end:
            print("\nComputer Wins the Game, Better Luck Next Time\n")
            input("Press any key to exit")
            break

# 程序入口
if __name__ == "__main__":
    main()

```