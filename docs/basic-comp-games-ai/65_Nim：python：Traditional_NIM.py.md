# `d:/src/tocomm/basic-computer-games\65_Nim\python\Traditional_NIM.py`

```
import random  # 导入 random 模块，用于生成随机数
from typing import Tuple  # 导入 typing 模块中的 Tuple 类型，用于指定函数返回类型为元组


class NIM:
    def __init__(self) -> None:  # 初始化函数，初始化游戏的初始状态
        self.piles = {1: 7, 2: 5, 3: 3, 4: 1}  # 初始化游戏的初始状态，用字典表示每个堆中的初始数量

    def remove_pegs(self, command) -> None:  # 定义移除棋子的方法，接受一个命令参数，无返回值
        try:  # 尝试执行以下代码块
            pile, num = command.split(",")  # 将命令按逗号分割，得到堆和数量
            num = int(num)  # 将数量转换为整数类型
            pile = int(pile)  # 将堆转换为整数类型

        except Exception as e:  # 捕获异常并将异常对象赋值给变量 e
            if "not enough values" in str(e):  # 如果异常信息中包含 "not enough values"
                print('\nNot a valid command. Your command should be in the form of "1,3", Try Again\n')  # 打印错误提示信息
                )

            else:
                print("\nError, Try again\n")
            return None
```
这部分代码是一个条件语句，如果条件成立则执行其中的代码块，否则执行另一个代码块。如果条件不成立，打印错误信息并返回None。

```
        if self._command_integrity(num, pile):
            self.piles[pile] -= num
        else:
            print("\nInvalid value of either Peg or Pile\n")
```
这部分代码是另一个条件语句，根据条件的成立与否执行不同的代码块。如果条件成立，执行第一个代码块，否则执行第二个代码块。如果条件不成立，打印错误信息。

```
    def get_ai_move(self) -> Tuple[int, int]:
        possible_pile = []
        for k, v in self.piles.items():
            if v != 0:
                possible_pile.append(k)

        pile = random.choice(possible_pile)

        num = random.randint(1, self.piles[pile])
```
这部分代码定义了一个名为get_ai_move的方法，返回一个元组。首先创建一个空列表possible_pile，然后遍历self.piles字典，将值不为0的键添加到possible_pile中。接着从possible_pile中随机选择一个元素作为pile，然后从1到self.piles[pile]之间随机选择一个整数作为num。
        return pile, num
        # 返回两个值，pile 和 num

    def _command_integrity(self, num, pile) -> bool:
        # 检查输入的数字是否符合游戏规则，返回布尔值
        return pile <= 4 and pile >= 1 and num <= self.piles[pile]

    def print_pegs(self) -> None:
        # 打印每个堆的数量
        for pile, peg in self.piles.items():
            print("Pile {} : {}".format(pile, "O " * peg))

    def help(self) -> None:
        # 打印游戏规则和提示
        print("-" * 10)
        print('\nThe Game is player with a number of Piles of Objects("O" == one peg)')
        print("\nThe Piles are arranged as given below(Tradional NIM)\n")
        self.print_pegs()
        print(
            '\nAny Number of of Objects are removed one pile by "YOU" and the machine alternatively'
        )
        print("\nOn your turn, you may take all the objects that remain in any pile")
        print("but you must take ATLEAST one object")
        # 打印游戏规则提示
        print("\nAnd you may take objects from only one pile on a single turn.")
        # 打印游戏胜利条件提示
        print("\nThe winner is defined as the one that picks the last remaning object")
        # 打印分隔线
        print("-" * 10)

    def check_for_win(self) -> bool:
        # 初始化总数
        sum = 0
        # 遍历每个堆的数量，累加到总数
        for v in self.piles.values():
            sum += v
        # 返回总数是否为0，即是否胜利
        return sum == 0


def main() -> None:
    # 初始化游戏
    game = NIM()

    # 打印游戏欢迎语
    print("Hello, This is a game of NIM")
    # 询问是否需要游戏说明
    help = input("Do You Need Instruction (YES or NO): ")

    # 如果需要游戏说明
    if help.lower() == "yes":
        game.help()  # 调用游戏对象的帮助函数，显示游戏帮助信息

    # Start game loop
    input("\nPress Enter to start the Game:\n")  # 提示用户按下回车键开始游戏
    end = False  # 初始化游戏结束标志为False
    while True:  # 进入游戏循环
        game.print_pegs()  # 调用游戏对象的打印函数，显示当前游戏状态

        # Players Move
        command = input("\nYOUR MOVE - Number of PILE, Number of Object? ")  # 提示玩家输入移动指令
        game.remove_pegs(command)  # 调用游戏对象的移除棋子函数，执行玩家的移动
        end = game.check_for_win()  # 调用游戏对象的检查胜利函数，检查是否玩家获胜
        if end:  # 如果游戏结束标志为True
            print("\nPlayer Wins the Game, Congratulations!!")  # 显示玩家获胜信息
            input("\nPress any key to exit")  # 提示用户按下任意键退出游戏
            break  # 退出游戏循环

        # Computers Move
        ai_command = game.get_ai_move()  # 调用游戏对象的获取AI移动函数，获取计算机的移动指令
        print(  # 打印计算机的移动指令
            "\nA.I MOVE - A.I Removed {} pegs from Pile {}".format(
                ai_command[1], ai_command[0]
            )
        )
        # 从游戏中移除AI命令中指定的数量的棋子
        game.remove_pegs(str(ai_command[0]) + "," + str(ai_command[1]))
        # 检查游戏是否结束
        end = game.check_for_win()
        # 如果游戏结束，打印电脑赢得游戏的消息，并等待用户输入任意键退出游戏
        if end:
            print("\nComputer Wins the Game, Better Luck Next Time\n")
            input("Press any key to exit")
            break


if __name__ == "__main__":
    main()
```