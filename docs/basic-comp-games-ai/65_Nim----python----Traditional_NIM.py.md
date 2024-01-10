# `basic-computer-games\65_Nim\python\Traditional_NIM.py`

```
# 导入 random 模块和 typing 模块中的 Tuple 类型
import random
from typing import Tuple

# 定义 NIM 类
class NIM:
    # 初始化函数，创建初始的堆
    def __init__(self) -> None:
        self.piles = {1: 7, 2: 5, 3: 3, 4: 1}

    # 移除指定数量的小木棍
    def remove_pegs(self, command) -> None:
        try:
            # 尝试解析命令，获取堆和数量
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

        # 检查命令的合法性，如果合法则移除小木棍
        if self._command_integrity(num, pile):
            self.piles[pile] -= num
        else:
            print("\nInvalid value of either Peg or Pile\n")

    # 获取 AI 的移动，随机选择堆和数量
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

    # 打印当前的小木棍情况
    def print_pegs(self) -> None:
        for pile, peg in self.piles.items():
            print("Pile {} : {}".format(pile, "O " * peg))
    # 定义一个帮助函数，打印游戏规则和说明
    def help(self) -> None:
        # 打印分隔线
        print("-" * 10)
        # 打印游戏规则说明
        print('\nThe Game is player with a number of Piles of Objects("O" == one peg)')
        # 打印文件堆的排列方式
        print("\nThe Piles are arranged as given below(Tradional NIM)\n")
        # 调用print_pegs函数打印文件堆
        self.print_pegs()
        # 打印游戏操作说明
        print(
            '\nAny Number of of Objects are removed one pile by "YOU" and the machine alternatively'
        )
        print("\nOn your turn, you may take all the objects that remain in any pile")
        print("but you must take ATLEAST one object")
        print("\nAnd you may take objects from only one pile on a single turn.")
        print("\nThe winner is defined as the one that picks the last remaning object")
        # 打印分隔线
        print("-" * 10)

    # 检查是否有玩家获胜
    def check_for_win(self) -> bool:
        # 初始化总数为0
        sum = 0
        # 遍历文件堆中的值，累加到总数中
        for v in self.piles.values():
            sum += v
        # 返回总数是否为0，即是否有玩家获胜
        return sum == 0
def main() -> None:
    # 游戏初始化
    game = NIM()

    print("Hello, This is a game of NIM")
    # 询问玩家是否需要游戏说明
    help = input("Do You Need Instruction (YES or NO): ")

    if help.lower() == "yes":
        game.help()

    # 开始游戏循环
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


if __name__ == "__main__":
    main()
```