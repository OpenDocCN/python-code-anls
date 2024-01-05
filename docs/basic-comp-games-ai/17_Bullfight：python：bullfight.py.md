# `d:/src/tocomm/basic-computer-games\17_Bullfight\python\bullfight.py`

```
    job_quality_factor = 0.7
    else:
        job_quality_factor = 1.0
    if player_type == "TOREAD":
        player_factor = 1.2
    else:
        player_factor = 1.0
    total_quality = math.prod([job_qualities.count(q) for q in job_qualities]) * player_factor * job_quality_factor
    if plural_form == "ORES":
        return total_quality * 1.5
    else:
        return total_quality
        job_quality_factor = 0.4  # 设置工作质量因子的初始值为0.4
    elif bull_performance < 0.63:  # 如果bull_performance小于0.63
        job_quality_factor = 0.3  # 设置工作质量因子为0.3
    elif bull_performance < 0.87:  # 如果bull_performance小于0.87
        job_quality_factor = 0.2  # 设置工作质量因子为0.2
    else:  # 如果bull_performance大于等于0.87
        job_quality_factor = 0.1  # 设置工作质量因子为0.1
    job_quality = math.floor(10 * job_quality_factor + 0.2)  # 计算工作质量，取整数部分，加0.2，越高越好
    print(f"THE {player_type}{plural_form} DID A {job_qualities[job_quality]} JOB.")  # 打印工作质量的描述
    if job_quality >= 4:  # 如果工作质量大于等于4
        if job_quality == 5:  # 如果工作质量等于5
            player_was_killed = random.choice([True, False])  # 随机选择玩家是否被杀死
            if player_was_killed:  # 如果玩家被杀死
                print(f"ONE OF THE {player_type}{plural_form} WAS KILLED.")  # 打印玩家被杀死的信息
            elif player_was_killed:  # 如果玩家没有被杀死
                print(f"NO {player_type}{plural_form} WERE KILLED.")  # 打印没有玩家被杀死的信息
        else:  # 如果工作质量不等于5
            if player_type != "TOREAD":  # 如果玩家类型不是"TOREAD"
                killed_horses = random.randint(1, 2)  # 随机生成1到2之间的整数，表示被杀死的马匹数量
                print(  # 打印
def calculate_final_score(
    move_risk_sum: float, job_quality_by_round: Dict[int, float], bull_quality: int
) -> float:
    # 定义一个函数，计算最终得分，接受三个参数：move_risk_sum（移动风险总和）、job_quality_by_round（每轮工作质量的字典）、bull_quality（公牛质量）
    quality = (
        4.5
        + move_risk_sum / 6
        - (job_quality_by_round[1] + job_quality_by_round[2]) * 2.5
        + 4 * job_quality_by_round[4]
        + 2 * job_quality_by_round[5]
        - (job_quality_by_round[3] ** 2) / 120
        - bull_quality
    ) * random.random()
    # 计算得分并返回
    if quality < 2.4:  # 如果质量小于2.4
        return 0  # 返回0
    elif quality < 4.9:  # 否则如果质量小于4.9
        return 1  # 返回1
    elif quality < 7.4:  # 否则如果质量小于7.4
        return 2  # 返回2
    else:  # 否则
        return 3  # 返回3


def print_header() -> None:  # 定义一个打印标题的函数，不返回任何值
    print(" " * 34 + "BULL")  # 打印空格后跟"BULL"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印15个空格后跟"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print_n_newlines(2)  # 调用print_n_newlines函数打印2个空行


def print_instructions() -> None:  # 定义一个打印指令的函数，不返回任何值
    print("HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS.")  # 打印"HELLO, ALL YOU BLOODLOVERS AND AFICIONADOS."
    print("HERE IS YOUR BIG CHANCE TO KILL A BULL.")  # 打印"HERE IS YOUR BIG CHANCE TO KILL A BULL."
    # 打印每次斗牛的选项
    print("ON EACH PASS OF THE BULL, YOU MAY TRY")
    print("0 - VERONICA (DANGEROUS INSIDE MOVE OF THE CAPE)")
    print("1 - LESS DANGEROUS OUTSIDE MOVE OF THE CAPE")
    print("2 - ORDINARY SWIRL OF THE CAPE.")
    print()
    # 打印另外的选项
    print("INSTEAD OF THE ABOVE, YOU MAY TRY TO KILL THE BULL")
    print("ON ANY TURN: 4 (OVER THE HORNS), 5 (IN THE CHEST).")
    print("BUT IF I WERE YOU,")
    print("I WOULDN'T TRY IT BEFORE THE SEVENTH PASS.")
    print()
    # 打印奖励相关信息
    print("THE CROWD WILL DETERMINE WHAT AWARD YOU DESERVE")
    print("(POSTHUMOUSLY IF NECESSARY).")
    print("THE BRAVER YOU ARE, THE BETTER THE AWARD YOU RECEIVE.")
    print()
    print("THE BETTER THE JOB THE PICADORES AND TOREADORES DO,")
    print("THE BETTER YOUR CHANCES ARE.")


def print_intro() -> None:
    # 调用打印标题的函数
    print_header()
    # 询问用户是否需要说明
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ")
    # 如果用户不需要说明，则打印说明
    if want_instructions != "NO":
        print_instructions()
    # 打印两个换行符
    print_n_newlines(2)


def ask_bool(prompt: str) -> bool:
    # 循环直到用户输入正确的答案
    while True:
        # 获取用户输入并转换为小写
        answer = input(prompt).lower()
        # 如果用户输入yes，则返回True
        if answer == "yes":
            return True
        # 如果用户输入no，则返回False
        elif answer == "no":
            return False
        # 如果用户输入不是yes或no，则提示用户重新输入
        else:
            print("INCORRECT ANSWER - - PLEASE TYPE 'YES' OR 'NO'.")


def ask_int() -> int:
    # 循环直到用户输入正确的整数
    while True:
        # 获取用户输入并转换为浮点数
        foo = float(input())
```
在这些代码中，注释解释了每个函数和代码块的作用，以及它们的输入和输出。这有助于其他程序员理解代码的功能和逻辑。
        if foo != float(int(abs(foo))):  # we actually want an integer
            # 如果 foo 不等于其绝对值的整数形式，打印错误信息
            print("DON'T PANIC, YOU IDIOT!  PUT DOWN A CORRECT NUMBER")
        elif foo < 3:
            # 如果 foo 小于 3，跳出循环
            break
    return int(foo)
    # 返回 foo 的整数形式


def did_bull_hit(
    bull_quality: int,
    cape_move: int,
    job_quality_by_round: Dict[int, float],
    move_risk_sum: float,
) -> Tuple[bool, float]:
    # The bull quality is a grade: The lower the grade, the better the bull
    # 公牛质量是一个等级：等级越低，公牛越好
    if cape_move == 0:
        move_risk: Union[int, float] = 3
    elif cape_move == 1:
        move_risk = 2
    else:
        move_risk = 0.5
    # 根据披风移动情况确定移动风险值
    move_risk_sum += move_risk  # 将 move_risk 加到 move_risk_sum 上
    bull_strength = 6 - bull_quality  # 计算 bull_strength 的值
    bull_hit_factor = (  # 计算 bull_hit_factor 的值，用于判断是否命中
        (bull_strength + move_risk / 10)  # 计算命中因子的一部分
        * random.random()  # 乘以一个随机数
        / (  # 除以以下表达式的结果
            (
                job_quality_by_round[1]  # 第一轮的工作质量
                + job_quality_by_round[2]  # 第二轮的工作质量
                + job_quality_by_round[3] / 10  # 第三轮的工作质量的一部分
            )
            * 5  # 乘以 5
        )
    )
    bull_hit = bull_hit_factor >= 0.51  # 判断是否命中
    return bull_hit, move_risk_sum  # 返回是否命中以及 move_risk_sum 的值


def handle_bullkill_attempt(
    kill_method: int,  # 处理 bullkill 尝试的方法
    job_quality_by_round: Dict[int, float],  # 定义一个名为job_quality_by_round的字典，键为整数，值为浮点数
    bull_quality: int,  # 定义一个名为bull_quality的整数变量
    gore: int,  # 定义一个名为gore的整数变量
) -> int:  # 定义函数返回类型为整数
    if kill_method not in [4, 5]:  # 如果kill_method不在列表[4, 5]中
        print("YOU PANICKED.  THE BULL GORED YOU.")  # 打印提示信息
        gore = 2  # 将gore赋值为2
    else:  # 否则
        bull_strength = 6 - bull_quality  # 计算bull_strength的值
        kill_probability = (  # 计算kill_probability的值
            bull_strength
            * 10
            * random.random()
            / (
                (job_quality_by_round[1] + job_quality_by_round[2])  # 访问job_quality_by_round字典的值
                * 5
                * job_quality_by_round[3]  # 访问job_quality_by_round字典的值
            )
        )
        if kill_method == 4:  # 如果kill_method等于4
            if kill_probability > 0.8:  # 如果杀死概率大于0.8
                gore = 1  # 则设置gore为1
        else:  # 否则
            if kill_probability > 0.2:  # 如果杀死概率大于0.2
                gore = 1  # 则设置gore为1
        if gore == 0:  # 如果gore为0
            print("YOU KILLED THE BULL!")  # 打印"YOU KILLED THE BULL!"
            job_quality_by_round[5] = 2  # 设置job_quality_by_round的第5个元素为2
            return gore  # 返回gore
    return gore  # 返回gore


def final_message(
    job_quality_by_round: Dict[int, float], bull_quality: int, move_risk_sum: float
) -> None:
    print_n_newlines(3)  # 打印3个空行
    if job_quality_by_round[4] == 0:  # 如果job_quality_by_round的第4个元素为0
        print("THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW")  # 打印"THE CROWD BOOS FOR TEN MINUTES.  IF YOU EVER DARE TO SHOW"
        print("YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--")  # 打印"YOUR FACE IN A RING AGAIN, THEY SWEAR THEY WILL KILL YOU--"
        print("UNLESS THE BULL DOES FIRST.")  # 打印"UNLESS THE BULL DOES FIRST."
    else:  # 如果不是特殊情况，执行以下代码
        if job_quality_by_round[4] == 2:  # 如果第四轮的工作质量为2
            print("THE CROWD CHEERS WILDLY!")  # 打印“观众疯狂欢呼！”
        elif job_quality_by_round[5] == 2:  # 如果第五轮的工作质量为2
            print("THE CROWD CHEERS!")  # 打印“观众欢呼！”
            print()  # 打印空行
        print("THE CROWD AWARDS YOU")  # 打印“观众奖励你”
        score = calculate_final_score(move_risk_sum, job_quality_by_round, bull_quality)  # 计算最终得分
        if score == 0:  # 如果得分为0
            print("NOTHING AT ALL.")  # 打印“什么都没有。”
        elif score == 1:  # 如果得分为1
            print("ONE EAR OF THE BULL.")  # 打印“公牛的一只耳朵。”
        elif score == 2:  # 如果得分为2
            print("BOTH EARS OF THE BULL!")  # 打印“公牛的两只耳朵！”
            print("OLE!")  # 打印“OLE！”
        else:  # 如果得分不是0、1、2
            print("OLE!  YOU ARE 'MUY HOMBRE'!! OLE!  OLE!")  # 打印“OLE！你是‘MUY HOMBRE’！OLE！OLE！”
        print()  # 打印空行
        print("ADIOS")  # 打印“再见”
        print_n_newlines(3)  # 打印3个空行
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 移动风险总数初始化为1
    move_risk_sum: float = 1
    # 每轮的工作质量，初始值为4和5的工作质量都为1
    job_quality_by_round: Dict[int, float] = {4: 1, 5: 1}
    # 工作质量的描述，分别对应不同的工作质量等级
    job_quality = ["", "SUPERB", "GOOD", "FAIR", "POOR", "AWFUL"]
    # 随机生成斗牛的质量，数值越小，斗牛的力量越大
    bull_quality = random.randint(
        1, 5
    )  # 数值越小，斗牛的力量越大
    # 打印斗牛的质量描述
    print(f"YOU HAVE DRAWN A {job_quality[bull_quality]} BULL.")
    # 根据斗牛的质量进行不同的输出
    if bull_quality > 4:
        print("YOU'RE LUCKY.")
    elif bull_quality < 2:
        print("GOOD LUCK.  YOU'LL NEED IT.")
        print()
    print()

    # Round 1: Run Picadores
    # 玩家类型初始化为"PICADO"
    player_type: Literal["TOREAD", "PICADO"] = "PICADO"
    plural_form: Literal["ORES", "RES"] = "RES"  # 定义变量plural_form，类型为Literal["ORES", "RES"]，初始值为"RES"
    job_quality_factor = determine_player_kills(  # 调用determine_player_kills函数，计算job_quality_factor
        bull_quality, player_type, plural_form, job_quality
    )
    job_quality_by_round[1] = job_quality_factor  # 将计算得到的job_quality_factor赋值给job_quality_by_round字典的第一个元素

    # Round 2: Run Toreadores
    player_type = "TOREAD"  # 修改player_type的值为"TOREAD"
    plural_form = "ORES"  # 修改plural_form的值为"ORES"
    determine_player_kills(bull_quality, player_type, plural_form, job_quality)  # 再次调用determine_player_kills函数
    job_quality_by_round[2] = job_quality_factor  # 将计算得到的job_quality_factor赋值给job_quality_by_round字典的第二个元素
    print_n_newlines(2)  # 打印两行空行

    # Round 3
    job_quality_by_round[3] = 0  # 将job_quality_by_round字典的第三个元素的值设为0
    while True:  # 进入无限循环
        job_quality_by_round[3] = job_quality_by_round[3] + 1  # 第三轮的job_quality_by_round值加1
        print(f"PASS NUMBER {job_quality_by_round[3]}")  # 打印当前轮数
        if job_quality_by_round[3] >= 3:  # 如果当前轮数大于等于3
            run_from_ring = ask_bool("HERE COMES THE BULL.  TRY FOR A KILL? ")  # 询问是否要尝试躲避公牛
# 如果不是从斗牛场逃跑，则打印提示信息
if not run_from_ring:
    print("CAPE MOVE? ", end="")
else:
    # 如果是从斗牛场逃跑，则打印提示信息
    print("THE BULL IS CHARGING AT YOU!  YOU ARE THE MATADOR--")
    # 询问是否想要杀死公牛
    run_from_ring = ask_bool("DO YOU WANT TO KILL THE BULL? ")
    # 如果不想从斗牛场逃跑，则打印提示信息
    if not run_from_ring:
        print("WHAT MOVE DO YOU MAKE WITH THE CAPE? ", end="")
# 初始化 gore 变量
gore = 0
# 如果不想从斗牛场逃跑
if not run_from_ring:
    # 询问使用斗篷的移动方式
    cape_move = ask_int()
    # 判断公牛是否击中，计算移动的风险总和
    bull_hit, move_risk_sum = did_bull_hit(
        bull_quality, cape_move, job_quality_by_round, move_risk_sum
    )
    # 如果公牛击中，则 gore 等于 1
    if bull_hit:
        gore = 1
    else:
        # 如果公牛没有击中，则继续循环
        continue
else:
    # 如果从斗牛场逃跑，则打印提示信息
    print()
    print("IT IS THE MOMENT OF TRUTH.")
            # 打印空行
            print()
            # 获取用户输入的杀死公牛的方法，并转换为整数
            kill_method = int(input("HOW DO YOU TRY TO KILL THE BULL? "))
            # 调用 handle_bullkill_attempt 函数处理杀牛的尝试，更新 gore 变量
            gore = handle_bullkill_attempt(
                kill_method, job_quality_by_round, bull_quality, gore
            )
            # 如果 gore 变量为 0，则跳出循环
            if gore == 0:
                break
        # 如果 gore 变量大于 0
        if gore > 0:
            # 如果 gore 变量等于 1，打印信息表示公牛刺伤了玩家
            if gore == 1:
                print("THE BULL HAS GORED YOU!")
            # 初始化 death 变量为 False
            death = False
            # 进入循环，直到玩家死亡
            while True:
                # 如果随机生成的数字为 1，表示玩家死亡
                if random.randint(1, 2) == 1:
                    print("YOU ARE DEAD.")
                    # 更新 job_quality_by_round 列表的第五个元素为 1.5
                    job_quality_by_round[4] = 1.5
                    # 将 death 变量设置为 True，表示玩家已死亡
                    death = True
                    # 跳出循环
                    break
                # 如果随机生成的数字不为 1，表示玩家仍然存活
                else:
                    print("YOU ARE STILL ALIVE.")
                    # 打印空行
                    print()
                    print("DO YOU RUN FROM THE RING? ", end="")
                    # 询问用户是否从斗牛场逃跑，并将用户输入的布尔值存储在变量run_from_ring中
                    run_from_ring = ask_bool("DO YOU RUN FROM THE RING? ")
                    # 如果用户选择不逃跑
                    if not run_from_ring:
                        # 打印提示信息
                        print("YOU ARE BRAVE.  STUPID, BUT BRAVE.")
                        # 如果随机数为1，设置第4轮的工作质量为2，设置死亡标志为True，并跳出循环
                        if random.randint(1, 2) == 1:
                            job_quality_by_round[4] = 2
                            death = True
                            break
                        # 如果随机数不为1
                        else:
                            # 打印提示信息
                            print("YOU ARE GORED AGAIN!")
                    # 如果用户选择逃跑
                    else:
                        # 打印提示信息
                        print("COWARD")
                        # 设置第4轮的工作质量为0，设置死亡标志为True，并跳出循环
                        job_quality_by_round[4] = 0
                        death = True
                        break

            # 如果死亡标志为True，跳出循环
            if death:
                break

    # 调用final_message函数，传入参数job_quality_by_round, bull_quality, move_risk_sum
    final_message(job_quality_by_round, bull_quality, move_risk_sum)
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以使代码更具可重用性和模块化。
```