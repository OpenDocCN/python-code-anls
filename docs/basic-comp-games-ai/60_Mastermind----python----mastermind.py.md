# `basic-computer-games\60_Mastermind\python\mastermind.py`

```py
# 导入 random 模块
import random
# 导入 sys 模块
import sys
# 导入 List、Union、Tuple 类型提示
from typing import List, Union, Tuple

# 定义一些游戏参数，不应该被修改
def setup_game() -> Tuple[int, int, int, int]:
    # 打印游戏标题
    print("""
                                  MASTERMIND
                   CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY



    """)
    # 获取用户输入的游戏条件
    num_colors: int = len(COLOR_LETTERS) + 1
    while num_colors > len(COLOR_LETTERS):
        num_colors = int(input("Number of colors (max 8): "))  # C9 in BASIC
    num_positions = int(input("Number of positions: "))  # P9 in BASIC
    num_rounds = int(input("Number of rounds: "))  # R9 in BASIC
    possibilities = num_colors**num_positions

    print(f"Number of possibilities {possibilities}")
    print("Color\tLetter")
    print("=====\t======")
    for element in range(0, num_colors):
        print(f"{COLORS[element]}\t{COLORS[element][0]}")
    return num_colors, num_positions, num_rounds, possibilities

# 全局变量
COLORS = ["BLACK", "WHITE", "RED", "GREEN", "ORANGE", "YELLOW", "PURPLE", "TAN"]
COLOR_LETTERS = "BWRGOYPT"
NUM_COLORS, NUM_POSITIONS, NUM_ROUNDS, POSSIBILITIES = setup_game()
human_score = 0
computer_score = 0

# 主函数
def main() -> None:
    current_round = 1
    while current_round <= NUM_ROUNDS:
        print(f"Round number {current_round}")
        human_turn()
        computer_turn()
        current_round += 1
    print_score(is_final_score=True)
    sys.exit()

# 玩家回合
def human_turn() -> None:
    global human_score
    num_moves = 1
    guesses: List[List[Union[str, int]]] = []
    print("Guess my combination ...")
    secret_combination = int(POSSIBILITIES * random.random())
    answer = possibility_to_color_code(secret_combination)
    # 无限循环，直到条件被打破
    while True:
        # 打印当前移动次数和提示用户输入猜测
        print(f"Move # {num_moves} Guess : ")
        user_command = input("Guess ")
        # 如果用户输入为"BOARD"，则打印当前猜测情况
        if user_command == "BOARD":
            print_board(guesses)  # 2000
        # 如果用户输入为"QUIT"，则打印正确答案并退出游戏
        elif user_command == "QUIT":  # 2500
            print(f"QUITTER! MY COMBINATION WAS: {answer}")
            print("GOOD BYE")
            quit()
        # 如果用户输入的长度不等于指定的位置数，则打印错误信息
        elif len(user_command) != NUM_POSITIONS:  # 410
            print("BAD NUMBER OF POSITIONS")
        else:
            # 获取无效字母并打印
            invalid_letters = get_invalid_letters(user_command)
            if invalid_letters > "":
                print(f"INVALID GUESS: {invalid_letters}")
            else:
                # 比较用户猜测和正确答案，并根据结果进行相应操作
                guess_results = compare_two_positions(user_command, answer)
                if guess_results[1] == NUM_POSITIONS:  # correct guess
                    # 如果猜测全部正确，则打印信息并返回
                    print(f"You guessed it in {num_moves} moves!")
                    human_score = human_score + num_moves
                    print_score()
                    return  # from human turn, triumphant
                else:
                    # 如果猜测不全正确，则打印提示信息，并更新猜测次数和结果
                    print(
                        "You have {} blacks and {} whites".format(
                            guess_results[1], guess_results[2]
                        )
                    )
                    guesses.append(guess_results)
                    num_moves += 1

        # 如果猜测次数超过10次，则打印信息并返回
        if num_moves > 10:  # RAN OUT OF MOVES
            print("YOU RAN OUT OF MOVES! THAT'S ALL YOU GET!")
            print(f"THE ACTUAL COMBINATION WAS: {answer}")
            human_score = human_score + num_moves
            print_score()
            return  # from human turn, defeated
# 定义一个函数，表示计算机的回合，不返回任何结果
def computer_turn() -> None:
    # 声明引用全局变量 computer_score
    global computer_score
    # 无限循环，直到猜中为止
    while True:
        # 创建一个长度为 POSSIBILITIES 的列表，每个元素都是 1
        all_possibilities = [1] * POSSIBILITIES
        # 猜测次数初始化为 1
        num_moves = 1
        # 打印提示信息
        print("NOW I GUESS. THINK OF A COMBINATION.")
        # 等待用户准备好后继续
        input("HIT RETURN WHEN READY: ")
        # 再次无限循环
        while True:
            # 找到第一个可能的解
            possible_guess = find_first_solution_of(all_possibilities)
            # 如果没有解了
            if possible_guess < 0:  # no solutions left :(
                # 打印提示信息
                print("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.")
                print("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.")
                # 退出内部循环，重新开始计算机的回合
                break

            # 将可能的解转换成颜色代码
            computer_guess = possibility_to_color_code(possible_guess)
            # 打印计算机的猜测
            print(f"My guess is: {computer_guess}")
            # 获取用户输入的黑白猜中情况
            blacks_str, whites_str = input(
                "ENTER BLACKS, WHITES (e.g. 1,2): "
            ).split(",")
            blacks = int(blacks_str)
            whites = int(whites_str)
            # 如果全部猜中
            if blacks == NUM_POSITIONS:  # Correct guess
                # 打印猜中信息
                print(f"I GOT IT IN {num_moves} MOVES")
                # 更新计算机得分
                computer_score = computer_score + num_moves
                # 打印得分
                print_score()
                # 从计算机回合返回
                return

            # 计算机猜错了，推断哪些解需要排除
            for i in range(0, POSSIBILITIES):
                # 如果已经排除了，继续下一个
                if all_possibilities[i] == 0:  # already ruled out
                    continue
                # 将可能的答案转换成颜色代码
                possible_answer = possibility_to_color_code(i)
                # 比较计算机猜测和可能的答案，得到黑白猜中情况
                comparison = compare_two_positions(
                    possible_answer, computer_guess
                )
                # 如果猜中情况不符合用户输入的黑白猜中情况
                if (blacks != comparison[1]) or (whites != comparison[2]):
                    # 将该解排除
                    all_possibilities[i] = 0

            # 如果猜测次数达到 10 次
            if num_moves == 10:
                # 打印提示信息
                print("I USED UP ALL MY MOVES!")
                print("I GUESS MY CPU IS JUST HAVING AN OFF DAY.")
                # 更新计算机得分
                computer_score = computer_score + num_moves
                # 打印得分
                print_score()
                # 从计算机回合返回，失败
                return
            # 猜测次数加一
            num_moves += 1
# 找到所有可能性中第一个非零标记的位置，从某个随机位置开始扫描，如果需要则循环。如果找不到则返回-1。
def find_first_solution_of(all_possibilities: List[int]) -> int:
    start = int(POSSIBILITIES * random.random())  # 从所有可能性中随机选择一个起始位置
    for i in range(0, POSSIBILITIES):  # 遍历所有可能性
        solution = (i + start) % POSSIBILITIES  # 计算当前位置
        if all_possibilities[solution]:  # 如果当前位置的标记非零
            return solution  # 返回当前位置
    return -1  # 如果找不到非零标记则返回-1


# 470
def get_invalid_letters(user_command) -> str:
    """确保玩家输入符合所选游戏配置的有效颜色。"""
    valid_colors = COLOR_LETTERS[:NUM_COLORS]  # 获取有效颜色列表
    invalid_letters = ""  # 初始化无效字母字符串
    for letter in user_command:  # 遍历玩家输入的每个字母
        if letter not in valid_colors:  # 如果字母不在有效颜色列表中
            invalid_letters = invalid_letters + letter  # 将该字母添加到无效字母字符串中
    return invalid_letters  # 返回无效字母字符串


# 2000
def print_board(guesses) -> None:
    """打印本轮猜测的结果。"""
    print("Board")  # 打印标题
    print("Move\tGuess\tBlack White")  # 打印表头
    for idx, guess in enumerate(guesses):  # 遍历猜测结果列表
        print(f"{idx + 1}\t{guess[0]}\t{guess[1]}     {guess[2]}")  # 打印每一轮的猜测结果


def possibility_to_color_code(possibility: int) -> str:
    """接受一个（十进制）数字，表示可能秘密代码中的一个排列，返回映射到该排列的颜色代码。
    该算法本质上是将一个十进制数转换为一个以#num_colors为基数的数，其中每个颜色代码字母表示该#num_colors基数中的一个数字。"""
    color_code: str = ""  # 初始化颜色代码字符串
    pos: int = NUM_COLORS ** NUM_POSITIONS  # 从总可能性开始
    remainder = possibility  # 初始化余数为可能性
    for _ in range(NUM_POSITIONS - 1, 0, -1):  # 处理除最后一位之外的所有位
        pos = pos // NUM_COLORS  # 计算当前位的基数
        color_code += COLOR_LETTERS[remainder // pos]  # 将当前位的颜色代码添加到颜色代码字符串中
        remainder = remainder % pos  # 更新余数
    color_code += COLOR_LETTERS[remainder]  # 最后一位是剩下的余数
    return color_code  # 返回颜色代码


# 4500
# 比较两个位置的猜测和答案，返回黑色（正确的颜色和位置）和白色（正确的颜色但错误的位置）的数量
def compare_two_positions(guess: str, answer: str) -> List[Union[str, int]]:
    increment = 0  # 初始化增量
    blacks = 0  # 初始化黑色数量
    whites = 0  # 初始化白色数量
    initial_guess = guess  # 保存初始猜测
    for pos in range(0, NUM_POSITIONS):  # 遍历位置
        if guess[pos] != answer[pos]:  # 如果猜测和答案在当前位置不相等
            for pos2 in range(0, NUM_POSITIONS):  # 再次遍历位置
                if not (guess[pos] != answer[pos2] or guess[pos2] == answer[pos2]):  # 如果颜色正确但位置不正确
                    whites = whites + 1  # 白色数量加一
                    answer = answer[:pos2] + chr(increment) + answer[pos2 + 1:]  # 修改答案的字符
                    guess = guess[:pos] + chr(increment + 1) + guess[pos + 1:]  # 修改猜测的字符
                    increment = increment + 2  # 增量加二
        else:  # 如果颜色和位置都正确
            blacks = blacks + 1  # 黑色数量加一
            # 这是一个巧妙的操作
            guess = guess[:pos] + chr(increment + 1) + guess[pos + 1:]  # 修改猜测的字符
            answer = answer[:pos] + chr(increment) + answer[pos + 1:]  # 修改答案的字符
            increment = increment + 2  # 增量加二
    return [initial_guess, blacks, whites]  # 返回结果列表


# 打印每轮结束后的得分，包括游戏结束时的最终得分
def print_score(is_final_score: bool = False) -> None:
    if is_final_score:  # 如果是最终得分
        print("GAME OVER")  # 打印游戏结束
        print("FINAL SCORE:")  # 打印最终得分
    else:  # 如果不是最终得分
        print("SCORE:")  # 打印得分
    print(f"     COMPUTER {computer_score}")  # 打印计算机得分
    print(f"     HUMAN    {human_score}")  # 打印玩家得分


if __name__ == "__main__":
    main()  # 调用主函数
```