# `basic-computer-games\17_Bullfight\python\bullfight.py`

```
# 导入 math、random、Dict、List、Literal、Tuple、Union 模块
import math
import random
from typing import Dict, List, Literal, Tuple, Union

# 定义一个函数，打印指定数量的空行
def print_n_newlines(n: int) -> None:
    for _ in range(n):
        print()

# 定义一个函数，确定玩家的击杀情况
def determine_player_kills(
    bull_quality: int,  # 公牛质量
    player_type: Literal["TOREAD", "PICADO"],  # 玩家类型
    plural_form: Literal["ORES", "RES"],  # 复数形式
    job_qualities: List[str],  # 工作质量列表
) -> float:
    # 根据公牛表现计算工作质量因子
    bull_performance = 3 / bull_quality * random.random()
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
    # 根据工作质量因子计算工作质量
    job_quality = math.floor(10 * job_quality_factor + 0.2)  # 越高越好
    # 打印玩家类型和工作质量
    print(f"THE {player_type}{plural_form} DID A {job_qualities[job_quality]} JOB.")
    # 根据工作质量判断击杀情况
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

# 定义一个函数，计算最终得分
def calculate_final_score(
    move_risk_sum: float,  # 移动风险总和
    job_quality_by_round: Dict[int, float],  # 每轮的工作质量
    bull_quality: int  # 公牛质量
) -> float:
    # 计算质量得分，根据一系列因素计算得出
    quality = (
        4.5  # 基础得分
        + move_risk_sum / 6  # 根据移动风险总和计算得分
        - (job_quality_by_round[1] + job_quality_by_round[2]) * 2.5  # 根据不同轮次的工作质量计算得分
        + 4 * job_quality_by_round[4]  # 根据第四轮工作质量计算得分
        + 2 * job_quality_by_round[5]  # 根据第五轮工作质量计算得分
        - (job_quality_by_round[3] ** 2) / 120  # 根据第三轮工作质量计算得分
        - bull_quality  # 根据牛市质量计算得分
    ) * random.random()  # 乘以随机数，引入一定的随机性
    
    # 根据质量得分返回不同的等级
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
    # 打印游戏标题
    print(" " * 34 + "BULL")
    # 打印游戏信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印两行空行
    print_n_newlines(2)


# 打印游戏说明
def print_instructions() -> None:
    # 打印游戏说明
    print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.")
    print("HERE IS YOUR BIG CHANCE TO KILL A BULL.")
    print()
    print("ON EACH PASS OF THE BULL, YOU MAY TRY")
    print("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)")
    print("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE")
    print("2 - ORDINARY SWIRL OF THE CAPE.")
    print()
    print("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL")
    print("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).")
    print("BUT IF I WERE YOU,")
    print("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.")
    print()
    print("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE")
    print("(POSTHUMOUSLY IF NECESSARY).")
    print("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.")
    print()
    print("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,")
    print("THE BETTER YOUR CHANCES ARE.")


# 打印游戏介绍
def print_intro() -> None:
    # 打印游戏标题
    print_header()
    # 询问是否需要游戏说明
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    # 如果需要游戏说明，则打印游戏说明
    if want_instructions != "NO":
        print_instructions()
    # 打印两行空行
    print_n_newlines(2)


# 询问布尔类型的问题
def ask_bool(prompt: str) -> bool:
    # 循环直到得到正确的回答
    while True:
        # 获取用户输入的答案并转换为小写
        answer = input(prompt).lower()
        # 如果答案是"yes"，返回True
        if answer == "yes":
            return True
        # 如果答案是"no"，返回False
        elif answer == "no":
            return False
        # 如果答案不是"yes"或"no"，提示用户输入正确的答案
        else:
            print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.")


# 询问整数类型的问题
def ask_int() -> int:
    # 循环直到得到正确的整数
    while True:
        # 获取用户输入的数字
        foo = float(input())
        # 如果输入的不是整数，提示用户输入正确的数字
        if foo != float(int(abs(foo))):  # we actually want an integer
            print("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER")
        # 如果输入的数字小于3，跳出循环
        elif foo < 3:
            break
    # 返回输入的整数
    return int(foo)


# 判断公牛是否被击中
def did_bull_hit(
    bull_quality: int,
    cape_move: int,
    job_quality_by_round: Dict[int, float],
    move_risk_sum: float,
) -> Tuple[bool, float]:
    # 如果 cape_move 等于 0，则 move_risk 等于 3；如果 cape_move 等于 1，则 move_risk 等于 2；否则 move_risk 等于 0.5
    if cape_move == 0:
        move_risk: Union[int, float] = 3
    elif cape_move == 1:
        move_risk = 2
    else:
        move_risk = 0.5
    # 将 move_risk 加到 move_risk_sum 上
    move_risk_sum += move_risk
    # 计算 bull_strength，6 减去 bull_quality
    bull_strength = 6 - bull_quality
    # 计算 bull_hit_factor，根据公式计算得出
    bull_hit_factor = (
        (bull_strength + move_risk / 10)
        * random.random()
        / (
            (
                job_quality_by_round[1]
                + job_quality_by_round[2]
                + job_quality_by_round[3] / 10
            )
            * 5
        )
    )
    # 判断 bull_hit_factor 是否大于等于 0.51，得到 bull_hit 的布尔值
    bull_hit = bull_hit_factor >= 0.51
    # 返回 bull_hit 和 move_risk_sum
    return bull_hit, move_risk_sum
# 处理对公牛的杀戮尝试，返回结果
def handle_bullkill_attempt(
    kill_method: int,  # 杀戮方法
    job_quality_by_round: Dict[int, float],  # 每轮工作质量的字典
    bull_quality: int,  # 公牛质量
    gore: int,  # 血腥程度
) -> int:  # 返回整数
    # 如果杀戮方法不是4或5，输出信息并设置血腥程度为2
    if kill_method not in [4, 5]:
        print("YOU PANICKED.  THE BULL GORED YOU.")
        gore = 2
    else:
        # 计算公牛力量
        bull_strength = 6 - bull_quality
        # 计算杀死公牛的概率
        kill_probability = (
            bull_strength
            * 10
            * random.random()
            / (
                (job_quality_by_round[1] + job_quality_by_round[2])
                * 5
                * job_quality_by_round[3]
            )
        )
        # 根据杀戮方法和概率设置血腥程度
        if kill_method == 4:
            if kill_probability > 0.8:
                gore = 1
        else:
            if kill_probability > 0.2:
                gore = 1
        # 如果血腥程度为0，输出信息并返回结果
        if gore == 0:
            print("YOU KILLED THE BULL!")
            job_quality_by_round[5] = 2
            return gore
    return gore


# 输出最终信息
def final_message(
    job_quality_by_round: Dict[int, float],  # 每轮工作质量的字典
    bull_quality: int,  # 公牛质量
    move_risk_sum: float,  # 移动风险总和
) -> None:  # 返回空值
    print_n_newlines(3)  # 输出3个换行符
    # 根据工作质量判断输出不同的信息
    if job_quality_by_round[4] == 0:
        print("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW")
        print("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--")
        print("UNLESS THE BULL DOES FIRST.")
    else:
        if job_quality_by_round[4] == 2:
            print("THE CROWD CHEERS WILDLY!")
        elif job_quality_by_round[5] == 2:
            print("THE CROWD CHEERS!")
            print()
        print("THE CROWD AWARDS YOU")
        # 计算最终得分
        score = calculate_final_score(move_risk_sum, job_quality_by_round, bull_quality)
        # 根据得分输出不同的信息
        if score == 0:
            print("NOTHING AT ALL.")
        elif score == 1:
            print("ONE EAR OF THE BULL.")
        elif score == 2:
            print("BOTH EARS OF THE BULL!")
            print("OLE!")
        else:
            print("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!")
        print()
        print("ADIOS")
        print_n_newlines(3)  # 输出3个换行符


# 主函数
def main() -> None:  # 返回空值
    print_intro()  # 输出介绍信息
    # 设置移动风险总和为1
    move_risk_sum: float = 1
    # 创建一个字典，表示每轮比赛的工作质量，初始值为{4: 1, 5: 1}
    job_quality_by_round: Dict[int, float] = {4: 1, 5: 1}
    # 定义工作质量的等级列表
    job_quality = ["", "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"]
    # 生成一个1到5之间的随机整数，表示斗牛的力量
    bull_quality = random.randint(1, 5)
    # 打印斗牛的力量等级
    print(f"YOU HAVE DRAWN A {job_quality[bull_quality]} BULL.")
    # 如果斗牛的力量大于4，则打印“YOU'RE LUCKY.”
    if bull_quality > 4:
        print("YOU'RE LUCKY.")
    # 如果斗牛的力量小于2，则打印“GOOD LUCK.  YOU'LL NEED IT.”
    elif bull_quality < 2:
        print("GOOD LUCK.  YOU'LL NEED IT.")
        # 打印空行
        print()
    # 打印空行
    print()
    
    # 第一轮：运行Picadores
    # 玩家类型为“PICADO”
    player_type: Literal["TOREAD", "PICADO"] = "PICADO"
    # 复数形式为“RES”
    plural_form: Literal["ORES", "RES"] = "RES"
    # 确定玩家击杀数，更新工作质量因子
    job_quality_factor = determine_player_kills(bull_quality, player_type, plural_form, job_quality)
    # 将工作质量因子存入第一轮的工作质量字典中
    job_quality_by_round[1] = job_quality_factor
    
    # 第二轮：运行Toreadores
    # 玩家类型为“TOREAD”
    player_type = "TOREAD"
    # 复数形式为“ORES”
    plural_form = "ORES"
    # 确定玩家击杀数，更新工作质量因子
    determine_player_kills(bull_quality, player_type, plural_form, job_quality)
    # 将工作质量因子存入第二轮的工作质量字典中
    job_quality_by_round[2] = job_quality_factor
    # 打印两个空行
    print_n_newlines(2)
    
    # 第三轮
    # 将第三轮的工作质量因子设为0
    job_quality_by_round[3] = 0
    # 输出最终消息，包括工作质量字典、斗牛力量和移动风险总和
    final_message(job_quality_by_round, bull_quality, move_risk_sum)
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```