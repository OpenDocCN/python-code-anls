# `60_Mastermind\python\mastermind.py`

```
import random  # 导入 random 模块，用于生成随机数
import sys  # 导入 sys 模块，用于与 Python 解释器交互
from typing import List, Union, Tuple  # 从 typing 模块中导入 List、Union、Tuple 类型

# 为游戏定义一些不应该被修改的参数
def setup_game() -> Tuple[int, int, int, int]:  # 定义 setup_game 函数，返回类型为元组
    print("""
                                  MASTERMIND
                   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY



    """)  # 打印游戏标题

    # 获取用户输入的游戏条件
    num_colors: int = len(COLOR_LETTERS) + 1  # 计算颜色数量
    while num_colors > len(COLOR_LETTERS):  # 当颜色数量大于颜色字母列表长度时
        num_colors = int(input("Number of colors (max 8): "))  # 获取颜色数量输入
    num_positions = int(input("Number of positions: "))  # 获取位置数量输入
    num_rounds = int(input("Number of rounds: "))  # 获取回合数量输入
    possibilities = num_colors**num_positions  # 计算可能的组合数量

    print(f"Number of possibilities {possibilities}")  # 打印可能的组合数量
    print("Color\tLetter")  # 打印表头
    print("=====\t======")  # 打印分隔线
    for element in range(0, num_colors):  # 遍历颜色列表
        print(f"{COLORS[element]}\t{COLORS[element][0]}")  # 打印颜色和对应的首字母
    return num_colors, num_positions, num_rounds, possibilities  # 返回游戏设置的参数


# Global variables
COLORS = ["BLACK", "WHITE", "RED", "GREEN", "ORANGE", "YELLOW", "PURPLE", "TAN"]  # 颜色列表
COLOR_LETTERS = "BWRGOYPT"  # 颜色对应的首字母
NUM_COLORS, NUM_POSITIONS, NUM_ROUNDS, POSSIBILITIES = setup_game()  # 调用设置游戏参数的函数
human_score = 0  # 玩家得分
computer_score = 0  # 电脑得分


def main() -> None:
    current_round = 1  # 当前回合数
    while current_round <= NUM_ROUNDS:  # 当前回合小于等于总回合数时执行循环
        print(f"Round number {current_round}")  # 打印当前回合数
        human_turn()  # 执行玩家回合
        computer_turn()  # 执行计算机回合
        current_round += 1  # 当前回合数加一
    print_score(is_final_score=True)  # 打印最终得分
    sys.exit()  # 退出程序


def human_turn() -> None:  # 定义玩家回合函数
    global human_score  # 声明全局变量 human_score
    num_moves = 1  # 初始化移动次数为1
    guesses: List[List[Union[str, int]]] = []  # 初始化猜测列表
    print("Guess my combination ...")  # 打印提示信息
    secret_combination = int(POSSIBILITIES * random.random())  # 生成随机的秘密组合
    answer = possibility_to_color_code(secret_combination)  # 将秘密组合转换为颜色代码
    while True:  # 无限循环
        print(f"Move # {num_moves} Guess : ")  # 打印移动次数和提示信息
        user_command = input("Guess ")  # 获取用户输入的猜测
        if user_command == "BOARD":  # 如果用户输入为"BOARD"
            print_board(guesses)  # 打印猜测结果
        elif user_command == "QUIT":  # 如果用户输入QUIT，则打印答案并退出程序
            print(f"QUITTER! MY COMBINATION WAS: {answer}")
            print("GOOD BYE")
            quit()
        elif len(user_command) != NUM_POSITIONS:  # 如果用户输入的数字位数不等于指定的位数，打印错误信息
            print("BAD NUMBER OF POSITIONS")
        else:
            invalid_letters = get_invalid_letters(user_command)  # 获取用户输入中无效的字母
            if invalid_letters > "":  # 如果存在无效的字母，打印无效猜测的信息
                print(f"INVALID GUESS: {invalid_letters}")
            else:
                guess_results = compare_two_positions(user_command, answer)  # 比较用户猜测和答案的结果
                if guess_results[1] == NUM_POSITIONS:  # 如果猜测全部正确，打印猜测次数并返回
                    print(f"You guessed it in {num_moves} moves!")
                    human_score = human_score + num_moves
                    print_score()
                    return  # 从用户回合返回，获胜
                else:
                    print(  # 打印比较结果
                    "You have {} blacks and {} whites".format(
                        guess_results[1], guess_results[2]
                    )
                )
                # 将猜测结果添加到猜测列表中
                guesses.append(guess_results)
                # 增加猜测次数
                num_moves += 1

    # 如果猜测次数超过10次，则打印消息并结束游戏
    if num_moves > 10:  # RAN OUT OF MOVES
        print("YOU RAN OUT OF MOVES! THAT'S ALL YOU GET!")
        print(f"THE ACTUAL COMBINATION WAS: {answer}")
        # 增加人类玩家的分数
        human_score = human_score + num_moves
        # 打印分数
        print_score()
        return  # from human turn, defeated


# 定义计算机玩家的回合函数
def computer_turn() -> None:
    # 声明全局变量
    global computer_score
    # 进入无限循环
    while True:
        # 创建包含所有可能性的列表
        all_possibilities = [1] * POSSIBILITIES
        # 初始化猜测次数
        num_moves = 1
        print("NOW I GUESS. THINK OF A COMBINATION.")  # 打印提示信息，提示用户电脑将要猜测
        input("HIT RETURN WHEN READY: ")  # 等待用户按下回车键，表示准备好了

        while True:  # 进入无限循环
            possible_guess = find_first_solution_of(all_possibilities)  # 找到可能的猜测组合
            if possible_guess < 0:  # 如果没有剩余的解决方案
                print("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.")  # 打印提示信息，表示用户提供了不一致的信息
                print("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.")  # 提示用户再试一次，并且请更加小心
                break  # 跳出内部循环，重新开始电脑的回合

            computer_guess = possibility_to_color_code(possible_guess)  # 将可能的猜测组合转换为颜色代码
            print(f"My guess is: {computer_guess}")  # 打印电脑的猜测
            blacks_str, whites_str = input(
                "ENTER BLACKS, WHITES (e.g. 1,2): "
            ).split(",")  # 获取用户输入的黑色和白色猜中数量
            blacks = int(blacks_str)  # 将输入的黑色猜中数量转换为整数
            whites = int(whites_str)  # 将输入的白色猜中数量转换为整数
            if blacks == NUM_POSITIONS:  # 如果黑色猜中数量等于位置数量，表示猜中了
                print(f"I GOT IT IN {num_moves} MOVES")  # 打印电脑猜中的信息和猜测次数
                computer_score = computer_score + num_moves  # 更新电脑的得分
                print_score()  # 打印得分信息
                return  # from computer turn
                # 返回，结束计算机的回合

            # computer guessed wrong, deduce which solutions to eliminate.
            # 计算机猜错了，推断要排除哪些解决方案。
            for i in range(0, POSSIBILITIES):
                if all_possibilities[i] == 0:  # already ruled out
                    continue
                # 如果已经排除，则继续下一个可能性
                possible_answer = possibility_to_color_code(i)
                # 将可能性转换为颜色代码
                comparison = compare_two_positions(
                    possible_answer, computer_guess
                )
                # 比较两个位置的颜色代码
                if (blacks != comparison[1]) or (whites != comparison[2]):
                    all_possibilities[i] = 0
                # 如果黑色和白色的数量不匹配，则排除该可能性

            if num_moves == 10:
                print("I USED UP ALL MY MOVES!")
                print("I GUESS MY CPU IS JUST HAVING AN OFF DAY.")
                computer_score = computer_score + num_moves
                print_score()
                return  # from computer turn, defeated.
                # 如果移动次数达到10次，则打印消息并结束计算机的回合，宣布失败
            num_moves += 1
            # 移动次数加1
# 定义一个函数，接受一个整数列表作为参数，返回一个整数
def find_first_solution_of(all_possibilities: List[int]) -> int:
    """Scan through all_possibilities for first remaining non-zero marker,
    starting from some random position and wrapping around if needed.
    If not found return -1."""
    # 从某个随机位置开始扫描 all_possibilities 列表，如果需要则循环
    start = int(POSSIBILITIES * random.random())
    # 遍历 all_possibilities 列表
    for i in range(0, POSSIBILITIES):
        # 计算当前位置的解决方案
        solution = (i + start) % POSSIBILITIES
        # 如果找到非零解决方案，返回该位置
        if all_possibilities[solution]:
            return solution
    # 如果未找到非零解决方案，返回 -1
    return -1


# 470
def get_invalid_letters(user_command) -> str:
    """Makes sure player input consists of valid colors for selected game configuration."""
    # 获取有效颜色列表
    valid_colors = COLOR_LETTERS[:NUM_COLORS]
    # 初始化无效字母字符串
    invalid_letters = ""
    # 遍历用户输入的命令
    for letter in user_command:
        if letter not in valid_colors:
            # 如果字母不在有效颜色列表中，则将其添加到无效字母列表中
            invalid_letters = invalid_letters + letter
    return invalid_letters
# 返回无效字母列表


# 2000
def print_board(guesses) -> None:
    """Print previous guesses within the round."""
    # 打印游戏板
    print("Board")
    # 打印移动、猜测和黑白匹配情况
    print("Move\tGuess\tBlack White")
    # 遍历猜测列表并打印每个猜测的移动次数、猜测内容和黑白匹配情况
    for idx, guess in enumerate(guesses):
        print(f"{idx + 1}\t{guess[0]}\t{guess[1]}     {guess[2]}")


def possibility_to_color_code(possibility: int) -> str:
    """Accepts a (decimal) number representing one permutation in the realm of
    possible secret codes and returns the color code mapped to that permutation.
    This algorithm is essentially converting a decimal  number to a number with
    a base of #num_colors, where each color code letter represents a digit in
    that #num_colors base."""
# 将可能性转换为颜色代码
    color_code: str = ""  # 初始化一个空字符串用于存储颜色编码
    pos: int = NUM_COLORS ** NUM_POSITIONS  # 计算可能性的总数，初始值为颜色数量的位置数量次方
    remainder = possibility  # 将可能性赋值给remainder变量
    for _ in range(NUM_POSITIONS - 1, 0, -1):  # 从倒数第二位开始循环到第一位
        pos = pos // NUM_COLORS  # 更新pos值为上一位的可能性总数
        color_code += COLOR_LETTERS[remainder // pos]  # 将当前位的颜色编码添加到color_code中
        remainder = remainder % pos  # 更新remainder为当前位的余数
    color_code += COLOR_LETTERS[remainder]  # 将最后一位的颜色编码添加到color_code中
    return color_code  # 返回最终的颜色编码


# 4500
def compare_two_positions(guess: str, answer: str) -> List[Union[str, int]]:
    """Returns blacks (correct color and position) and whites (correct color
    only) for candidate position (guess) versus reference position (answer)."""
    increment = 0  # 初始化增量为0
    blacks = 0  # 初始化黑色匹配数量为0
    whites = 0  # 初始化白色匹配数量为0
    initial_guess = guess  # 将猜测的初始值保存下来
    for pos in range(0, NUM_POSITIONS):  # 遍历每个位置
        if guess[pos] != answer[pos]:  # 如果猜测的位置上的颜色与答案的位置上的颜色不相同
            for pos2 in range(0, NUM_POSITIONS):  # 遍历所有可能的位置
                if not (guess[pos] != answer[pos2] or guess[pos2] == answer[pos2]):  # 如果猜测的颜色与答案的其他位置上的颜色不相同，但是颜色是正确的
                    whites = whites + 1  # 白色提示加一
                    answer = answer[:pos2] + chr(increment) + answer[pos2 + 1:]  # 更新答案中的颜色
                    guess = guess[:pos] + chr(increment + 1) + guess[pos + 1:]  # 更新猜测中的颜色
                    increment = increment + 2  # 增加增量
        else:  # 如果猜测的颜色与答案的位置上的颜色相同
            blacks = blacks + 1  # 黑色提示加一
            # THIS IS DEVIOUSLY CLEVER
            guess = guess[:pos] + chr(increment + 1) + guess[pos + 1:]  # 更新猜测中的颜色
            answer = answer[:pos] + chr(increment) + answer[pos + 1:]  # 更新答案中的颜色
            increment = increment + 2  # 增加增量
    return [initial_guess, blacks, whites]  # 返回初始猜测、黑色提示和白色提示的列表


# 5000 + logic from 1160
def print_score(is_final_score: bool = False) -> None:  # 打印分数，is_final_score参数表示是否是最终分数
    """Print score after each turn ends, including final score at end of game."""  # 打印每轮结束后的得分，包括游戏结束时的最终得分
    if is_final_score:  # 如果是最终得分
        print("GAME OVER")  # 打印游戏结束
        print("FINAL SCORE:")  # 打印最终得分
    else:  # 否则
        print("SCORE:")  # 打印得分
    print(f"     COMPUTER {computer_score}")  # 打印计算机得分
    print(f"     HUMAN    {human_score}")  # 打印玩家得分


if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用主函数
```