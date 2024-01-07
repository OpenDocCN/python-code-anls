# `basic-computer-games\17_Bullfight\python\bullfight.py`

```

# 导入所需的模块
import math
import random
from typing import Dict, List, Literal, Tuple, Union

# 打印指定数量的空行
def print_n_newlines(n: int) -> None:
    for _ in range(n):
        print()

# 确定玩家击杀情况
def determine_player_kills(
    bull_quality: int,
    player_type: Literal["TOREAD", "PICADO"],
    plural_form: Literal["ORES", "RES"],
    job_qualities: List[str],
) -> float:
    # 根据公牛质量和玩家类型计算公牛表现
    bull_performance = 3 / bull_quality * random.random()
    # 根据公牛表现确定工作质量因素
    if bull_performance < 0.37:
        job_quality_factor = 0.5
    elif bull_performance < 0.5:
        job_quality_factor = 0.4
    elif bull_performance < 0.63:
        job_quality_factor = 0.3
    elif bull_performance < 0.87:
        job_quality_factor = 0.2
    else:
        job_quality_factor = 0.1
    # 根据工作质量因素计算工作质量
    job_quality = math.floor(10 * job_quality_factor + 0.2)  # higher is better
    # 打印玩家类型、工作质量和击杀情况
    print(f"THE {player_type}{plural_form} DID A {job_qualities[job_quality]} JOB.")
    if job_quality >= 4:
        if job_quality == 5:
            player_was_killed = random.choice([True, False])
            if player_was_killed:
                print(f"ONE OF THE {player_type}{plural_form} WAS KILLED.")
            elif player_was_killed:
                print(f"NO {player_type}{plural_form} WERE KILLED.")
        else:
            if player_type != "TOREAD":
                killed_horses = random.randint(1, 2)
                print(
                    f"{killed_horses} OF THE HORSES OF THE {player_type}{plural_form} KILLED."
                )
            killed_players = random.randint(1, 2)
            print(f"{killed_players} OF THE {player_type}{plural_form} KILLED.")
    print()
    return job_quality_factor

# 计算最终得分
def calculate_final_score(
    move_risk_sum: float, job_quality_by_round: Dict[int, float], bull_quality: int
) -> float:
    # 计算得分
    quality = (
        4.5
        + move_risk_sum / 6
        - (job_quality_by_round[1] + job_quality_by_round[2]) * 2.5
        + 4 * job_quality_by_round[4]
        + 2 * job_quality_by_round[5]
        - (job_quality_by_round[3] ** 2) / 120
        - bull_quality
    ) * random.random()
    if quality < 2.4:
        return 0
    elif quality < 4.9:
        return 1
    elif quality < 7.4:
        return 2
    else:
        return 3

# 打印游戏标题
def print_header() -> None:
    print(" " * 34 + "BULL")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print_n_newlines(2)

# 打印游戏说明
def print_instructions() -> None:
    # 打印游戏说明
    print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.")
    # ... (其他说明内容)

# 打印游戏介绍
def print_intro() -> None:
    print_header()
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    if want_instructions != "NO":
        print_instructions()
    print_n_newlines(2)

# 询问布尔类型的问题
def ask_bool(prompt: str) -> bool:
    while True:
        answer = input(prompt).lower()
        if answer == "yes":
            return True
        elif answer == "no":
            return False
        else:
            print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.")

# 询问整数类型的问题
def ask_int() -> int:
    while True:
        foo = float(input())
        if foo != float(int(abs(foo))):  # we actually want an integer
            print("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER")
        elif foo < 3:
            break
    return int(foo)

# 判断公牛是否击中
def did_bull_hit(
    bull_quality: int,
    cape_move: int,
    job_quality_by_round: Dict[int, float],
    move_risk_sum: float,
) -> Tuple[bool, float]:
    # ... (判断公牛是否击中的逻辑)

# 处理公牛击杀尝试
def handle_bullkill_attempt(
    kill_method: int,
    job_quality_by_round: Dict[int, float],
    bull_quality: int,
    gore: int,
) -> int:
    # ... (处理公牛击杀尝试的逻辑)

# 打印最终消息
def final_message(
    job_quality_by_round: Dict[int, float], bull_quality: int, move_risk_sum: float
) -> None:
    # ... (打印最终消息的逻辑)

# 主函数
def main() -> None:
    # ... (主函数的逻辑)

# 如果是主程序，则执行主函数
if __name__ == "__main__":
    main()

```