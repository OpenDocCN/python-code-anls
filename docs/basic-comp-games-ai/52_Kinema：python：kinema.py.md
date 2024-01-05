# `d:/src/tocomm/basic-computer-games\52_Kinema\python\kinema.py`

```
"""
KINEMA

A kinematics physics quiz.

Ported by Dave LeCompte
"""

import random  # 导入 random 模块，用于生成随机数

# We approximate gravity from 9.8 meters/second squared to 10, which
# is only off by about 2%. 10 is also a lot easier for people to use
# for mental math.

g = 10  # 设置重力加速度 g 为 10 米/秒^2

# We only expect the student to get within this percentage of the
# correct answer. This isn't rocket science.

EXPECTED_ACCURACY_PERCENT = 15  # 设置预期精度为 15%
# 定义一个名为 do_quiz 的函数，返回类型为 None
def do_quiz() -> None:
    # 打印空行
    print()
    # 打印空行
    print()
    # 初始化正确回答问题的数量为 0
    num_questions_correct = 0

    # 随机选择初始速度
    v0 = random.randint(5, 40)
    # 打印初始速度信息
    print(f"A BALL IS THROWN UPWARDS AT {v0} METERS PER SECOND.")
    # 打印空行
    print()

    # 计算预期的最大高度
    answer = v0**2 / (2 * g)
    # 调用 ask_player 函数询问问题，并将回答是否正确的结果加到 num_questions_correct 中
    num_questions_correct += ask_player("HOW HIGH WILL IT GO (IN METERS)?", answer)

    # 计算预期的返回时间
    answer = 2 * v0 / g
    # 调用 ask_player 函数询问问题，并将回答是否正确的结果加到 num_questions_correct 中
    num_questions_correct += ask_player(
        "HOW LONG UNTIL IT RETURNS (IN SECONDS)?", answer
    )
    t = 1 + random.randint(0, 2 * v0) // g  # 计算 t 的值，使用了随机数和给定的变量 v0 和 g
    answer = v0 - g * t  # 计算答案，根据给定的公式计算
    num_questions_correct += ask_player(  # 调用 ask_player 函数，传入问题和答案，更新正确问题数量
        f"WHAT WILL ITS VELOCITY BE AFTER {t} SECONDS?", answer
    )

    print()
    print(f"{num_questions_correct} right out of 3.")  # 打印出正确问题数量
    if num_questions_correct >= 2:  # 如果正确问题数量大于等于2
        print("  NOT BAD.")  # 打印提示信息


def ask_player(question: str, answer) -> int:  # 定义 ask_player 函数，接受问题和答案，返回整数
    print(question)  # 打印问题
    player_answer = float(input())  # 获取玩家输入的答案并转换为浮点数

    accuracy_frac = EXPECTED_ACCURACY_PERCENT / 100.0  # 计算精度分数
    if abs((player_answer - answer) / answer) < accuracy_frac:  # 判断玩家答案的精度是否符合要求
        print("CLOSE ENOUGH.")  # 打印提示信息
        score = 1  # 设置得分为1
    else:
        # 如果答案不正确，则打印提示信息
        print("NOT EVEN CLOSE....")
        # 分数归零
        score = 0
    # 打印正确答案
    print(f"CORRECT ANSWER IS {answer}")
    # 打印空行
    print()
    # 返回分数
    return score


def main() -> None:
    # 打印标题
    print(" " * 33 + "KINEMA")
    # 打印副标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # 循环进行测验
    while True:
        do_quiz()


if __name__ == "__main__":
    # 调用主函数
    main()
```